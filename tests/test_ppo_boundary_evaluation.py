import gzip
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from math import radians, sqrt
from pathlib import Path

import gymnasium as gym
import pytest
from scipy.stats import beta

from active_eval_gym import ppo_boundary_evaluation
from active_eval_gym.config import load_ppo_boundary_study
from active_eval_gym.envs.factory import make_environment
from active_eval_gym.evaluate import make_artifact_environment
from active_eval_gym.metrics import RECOVERY_METRIC_VERSION, compute_saved_metrics
from active_eval_gym.policies.artifacts import load_policy_artifact
from active_eval_gym.ppo_boundary import prepare_study
from active_eval_gym.ppo_boundary_analysis import (
    analyze_campaign,
    clopper_pearson,
    coverage_summary,
)
from active_eval_gym.ppo_boundary_evaluation import (
    collect_campaign,
    plan_jobs,
)
from active_eval_gym.rollout import collect_episode
from active_eval_gym.serialization import (
    EPISODE_SHARD_LAYOUT,
    read_episode_shard,
    read_saved_episode,
    read_shard_header,
    write_episode,
    write_episode_shard,
)
from tests.helpers import (
    ROOT,
    SMALL_STUDY_GRID,
    rollout_provenance,
    tiny_parent_config,
)


class ConstantPolicy:
    def act(self, observation, *, deterministic: bool = True) -> int:
        return 0


@pytest.fixture
def study(tmp_path: Path) -> dict[str, Path]:
    config, root = tiny_parent_config(tmp_path, *SMALL_STUDY_GRID)
    study_dir = tmp_path / "study"
    prepare_study(config, artifact_root=root, output_dir=study_dir, allow_dirty=True)
    return {"config": config, "root": root, "study": study_dir}


def _collect(study: dict[str, Path], campaign: str = "discovery", **options):
    return collect_campaign(
        study["config"],
        study_dir=study["study"],
        campaign=campaign,
        artifact_root=study["root"],
        allow_dirty=True,
        **options,
    )


def _analyze(study: dict[str, Path], campaign: str = "discovery", **options):
    return analyze_campaign(
        study["config"], study_dir=study["study"], campaign=campaign, **options
    )


def _shards(study: dict[str, Path], campaign: str = "discovery") -> dict[str, bytes]:
    root = study["study"] / campaign / "shards"
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*.gz")}


def _episode(seed: int = 3):
    env = make_environment("CartPole-v1")
    policy_artifact, nominal, resolved = rollout_provenance(env, "CartPole-v1")
    try:
        return collect_episode(
            env,
            ConstantPolicy(),
            policy_artifact=policy_artifact,
            nominal_environment=nominal,
            resolved_environment=resolved,
            episode_seed=seed,
        )
    finally:
        env.close()


def _recompress_with(path: Path, old: bytes, new: bytes) -> None:
    content = gzip.decompress(path.read_bytes())
    assert old in content
    path.write_bytes(gzip.compress(content.replace(old, new, 1), mtime=0))


def _forged_shard(fields: dict, episodes: list) -> bytes:
    rows = [{"type": "shard-header", "layout": EPISODE_SHARD_LAYOUT, **fields}]
    content = json.dumps(rows[0]).encode() + b"\n"
    for episode in episodes:
        trajectory = (
            b"\n".join(
                json.dumps(row).encode()
                for row in (episode.reset, *episode.transitions)
            )
            + b"\n"
        )
        frame = {
            "type": "episode",
            "metadata": episode.metadata,
            "trajectory_bytes": len(trajectory),
            "trajectory_sha256": hashlib.sha256(trajectory).hexdigest(),
        }
        content += json.dumps(frame).encode() + b"\n" + trajectory
    footer = {"type": "shard-footer", "episode_count": len(episodes)}
    return content + json.dumps(footer).encode() + b"\n"


def test_shard_round_trips_the_existing_episode_format_exactly(tmp_path: Path) -> None:
    episodes = [_episode(3), _episode(4)]
    expected = []
    for index, episode in enumerate(episodes):
        directory = tmp_path / f"episode-{index}"
        write_episode(directory, episode)
        expected.append(
            (
                read_saved_episode(directory),
                (directory / "trajectory.jsonl").read_bytes(),
            )
        )
    path = tmp_path / "shard.jsonl.gz"

    count = write_episode_shard(path, {"job": "x"}, iter(episodes), compression_level=6)

    saved = list(read_episode_shard(path))
    assert count == 2
    assert read_shard_header(path)["job"] == "x"
    content = gzip.decompress(path.read_bytes())
    for (original, trajectory), loaded in zip(expected, saved, strict=True):
        assert loaded == original
        assert trajectory in content
        assert loaded.trajectory_sha256 == hashlib.sha256(trajectory).hexdigest()


def test_shard_metrics_equal_directory_metrics(tmp_path: Path) -> None:
    episode = _episode()
    write_episode(tmp_path / "episode", episode)
    path = tmp_path / "shard.jsonl.gz"
    write_episode_shard(path, {}, [episode], compression_level=6)

    assert compute_saved_metrics(
        next(read_episode_shard(path)), metric_version=RECOVERY_METRIC_VERSION
    ) == compute_saved_metrics(
        read_saved_episode(tmp_path / "episode"),
        metric_version=RECOVERY_METRIC_VERSION,
    )


def test_shard_bytes_are_reproducible(tmp_path: Path) -> None:
    for name in ("a", "b"):
        write_episode_shard(
            tmp_path / name, {"job": 1}, [_episode()], compression_level=6
        )

    assert (tmp_path / "a").read_bytes() == (tmp_path / "b").read_bytes()


def test_shard_refuses_to_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "shard.jsonl.gz"
    write_episode_shard(path, {}, [_episode()], compression_level=6)
    before = path.read_bytes()

    with pytest.raises(FileExistsError):
        write_episode_shard(path, {}, [_episode(9)], compression_level=6)

    assert path.read_bytes() == before


def test_failed_shard_never_appears_under_its_final_name(tmp_path: Path) -> None:
    def episodes():
        yield _episode()
        raise RuntimeError("worker died")

    path = tmp_path / "shard.jsonl.gz"
    with pytest.raises(RuntimeError):
        write_episode_shard(path, {}, episodes(), compression_level=6)

    assert list(tmp_path.iterdir()) == []


def test_tampered_trajectory_fails_its_digest(tmp_path: Path) -> None:
    path = tmp_path / "shard.jsonl.gz"
    write_episode_shard(path, {}, [_episode()], compression_level=6)
    _recompress_with(path, b'"action":0', b'"action":1')

    with pytest.raises(ValueError, match="digest mismatch"):
        list(read_episode_shard(path))


def test_truncated_shard_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "shard.jsonl.gz"
    write_episode_shard(path, {}, [_episode()], compression_level=6)
    path.write_bytes(path.read_bytes()[:-40])

    with pytest.raises(ValueError, match="truncated"):
        list(read_episode_shard(path))


def test_shard_without_its_footer_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "shard.jsonl.gz"
    write_episode_shard(path, {}, [_episode()], compression_level=6)
    content = gzip.decompress(path.read_bytes())
    path.write_bytes(gzip.compress(content[: content.rindex(b'{"episode_count"')]))

    with pytest.raises(ValueError, match="no footer"):
        list(read_episode_shard(path))


def test_footer_must_match_the_episodes_present(tmp_path: Path) -> None:
    path = tmp_path / "shard.jsonl.gz"
    write_episode_shard(path, {}, [_episode()], compression_level=6)
    _recompress_with(path, b'"episode_count":1', b'"episode_count":2')

    with pytest.raises(ValueError, match="does not match its footer"):
        list(read_episode_shard(path))


def test_shard_appearing_during_collection_is_not_replaced(tmp_path: Path) -> None:
    path = tmp_path / "shard.jsonl.gz"

    def episodes():
        yield _episode()
        path.write_bytes(b"another writer")

    with pytest.raises(FileExistsError):
        write_episode_shard(path, {}, episodes(), compression_level=6)

    assert path.read_bytes() == b"another writer"
    assert not path.with_name(f"{path.name}.partial").exists()


def test_campaign_collects_every_planned_episode_with_paired_seeds(study) -> None:
    result = _collect(study)

    assert result == {
        "campaign": "discovery",
        "planned_jobs": 4,
        "already_complete_jobs": 0,
        "failed_jobs": [],
        "complete": True,
    }
    seeds = [10**12, 10**12 + 1, 10**12 + 2]
    paths = sorted((study["study"] / "discovery/shards").rglob("*.gz"))
    assert len(paths) == 4
    for path in paths:
        header = read_shard_header(path)
        assert header["episode_seeds"] == seeds
        assert header["deterministic"] is True
        assert [e.metadata["episode_seed"] for e in read_episode_shard(path)] == seeds
    collection = json.loads((study["study"] / "discovery/collection.json").read_text())
    assert collection["episode_count"] == 12


def test_reused_environment_matches_a_fresh_environment_per_episode(study) -> None:
    _collect(study)
    policy, metadata = load_policy_artifact(study["root"] / "tiny_parent")
    spec = load_ppo_boundary_study(study["config"])
    job = plan_jobs(
        spec, "discovery", study_dir=study["study"], artifact_root=study["root"]
    )[0]
    saved = list(read_episode_shard(Path(job.shard_path)))
    perturbation = ppo_boundary_evaluation.PerturbationSpec(
        job.header["perturbation"]["name"], job.header["perturbation"]["parameters"]
    )

    for index, seed in enumerate(job.header["episode_seeds"]):
        env, resolved = make_artifact_environment(metadata, perturbation)
        try:
            episode = collect_episode(
                env,
                policy,
                policy_artifact=metadata,
                nominal_environment=metadata.design_spec.environment,
                resolved_environment=resolved,
                episode_seed=seed,
            )
        finally:
            env.close()
        directory = study["study"] / f"fresh-{index}"
        digest = write_episode(directory, episode)
        assert digest == saved[index].trajectory_sha256


def test_parallel_workers_write_identical_shards(study, tmp_path: Path) -> None:
    _collect(study, workers=1)
    serial = _shards(study)
    other = dict(study, study=tmp_path / "other")
    prepare_study(
        study["config"],
        artifact_root=study["root"],
        output_dir=other["study"],
        allow_dirty=True,
    )

    assert _collect(other, workers=2)["complete"] is True

    assert _shards(other) == serial


POOL_DRIVER = """
import sys
from pathlib import Path
from active_eval_gym.ppo_boundary_evaluation import run_jobs
from tests.helpers import record_pid_and_wait
run_jobs(sys.argv[2:], Path(sys.argv[1]), 1, 2, run=record_pid_and_wait)
"""


def _running(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _wait_until(condition, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return condition()


def test_pool_workers_stop_when_their_parent_is_killed(tmp_path: Path) -> None:
    jobs = [str(tmp_path / f"job-{index}") for index in range(4)]
    announced = lambda: [Path(job) for job in jobs if Path(job).exists()]  # noqa: E731
    driver = subprocess.Popen(
        [sys.executable, "-c", POOL_DRIVER, str(tmp_path), *jobs], cwd=ROOT
    )
    workers: list[int] = []
    try:
        assert _wait_until(lambda: len(announced()) == 2, 60)
        assert _wait_until(lambda: all(p.read_text() for p in announced()), 5)
        workers = [int(path.read_text()) for path in announced()]
        assert all(_running(pid) for pid in workers)

        driver.kill()
        driver.wait()

        assert _wait_until(lambda: not any(map(_running, workers)), 20)
        assert len(announced()) == 2
    finally:
        driver.kill()
        driver.wait()
        for pid in filter(_running, workers):
            os.kill(pid, signal.SIGKILL)


def test_interrupted_campaign_resumes_only_missing_jobs(
    study, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    real_run = ppo_boundary_evaluation.run_evaluation_job
    real_collect = ppo_boundary_evaluation.collect_episode

    def interrupted_collect(env, policy, *, episode_seed, **options):
        if len(calls) in (2, 3) and episode_seed == 10**12 + 1:
            raise RuntimeError("machine lost")
        return real_collect(env, policy, episode_seed=episode_seed, **options)

    def tracked_run(job):
        calls.append(job.job_id)
        return real_run(job)

    monkeypatch.setattr(ppo_boundary_evaluation, "collect_episode", interrupted_collect)
    monkeypatch.setattr(ppo_boundary_evaluation, "run_evaluation_job", tracked_run)
    first = _collect(study)

    assert first["complete"] is False
    assert len(first["failed_jobs"]) == 2
    assert len(_shards(study)) == 2
    assert not (study["study"] / "discovery/collection.json").exists()
    failures = sorted((study["study"] / "discovery/failures").iterdir())
    assert len(failures) == 2
    assert json.loads(failures[0].read_text())["error_type"] == "RuntimeError"
    with pytest.raises(ValueError, match="incomplete"):
        _analyze(study)

    interrupted = _shards(study)
    (
        study["study"] / "discovery/shards/tiny_parent/stale.jsonl.gz.partial"
    ).write_bytes(b"partial")
    calls.clear()
    monkeypatch.setattr(ppo_boundary_evaluation, "collect_episode", real_collect)
    second = _collect(study)

    assert second["complete"] is True
    assert second["already_complete_jobs"] == 2
    assert sorted(calls) == sorted(first["failed_jobs"])
    final = _shards(study)
    assert len(final) == 4
    assert all(final[name] == content for name, content in interrupted.items())

    clean = dict(study, study=tmp_path / "clean")
    prepare_study(
        study["config"],
        artifact_root=study["root"],
        output_dir=clean["study"],
        allow_dirty=True,
    )
    _collect(clean)
    assert _shards(clean) == final


def test_completed_campaign_reruns_nothing(
    study, monkeypatch: pytest.MonkeyPatch
) -> None:
    _collect(study)
    monkeypatch.setattr(
        ppo_boundary_evaluation,
        "run_evaluation_job",
        lambda job: pytest.fail("A complete job was rerun."),
    )

    assert _collect(study)["already_complete_jobs"] == 4


def test_confirmatory_collection_is_refused_before_the_lock(study) -> None:
    with pytest.raises(ValueError, match="S7 lock"):
        _collect(study, campaign="confirmatory")

    assert not (study["study"] / "reference_raw").exists()


def test_collection_refuses_uncommitted_source_by_default(
    study, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ppo_boundary_evaluation,
        "source_state",
        lambda: {"git_commit": "a", "source_dirty": True, "changed_paths": ["x"]},
    )

    with pytest.raises(ValueError, match="committed source"):
        collect_campaign(
            study["config"],
            study_dir=study["study"],
            campaign="discovery",
            artifact_root=study["root"],
        )


def test_config_changed_after_prepare_is_refused(study) -> None:
    text = study["config"].read_text()
    study["config"].write_text(
        text.replace("mixture_lambda = 0.5", "mixture_lambda = 0.4")
    )

    with pytest.raises(ValueError, match="protocol recorded"):
        _collect(study)


def test_resume_refuses_a_changed_plan(study, monkeypatch: pytest.MonkeyPatch) -> None:
    _collect(study)
    monkeypatch.setattr(
        ppo_boundary_evaluation,
        "source_state",
        lambda: {"git_commit": "other", "source_dirty": False, "changed_paths": []},
    )

    with pytest.raises(ValueError, match="recorded plan"):
        _collect(study)


def test_existing_shard_for_another_job_is_rejected(study) -> None:
    _collect(study)
    spec = load_ppo_boundary_study(study["config"])
    jobs = plan_jobs(
        spec, "discovery", study_dir=study["study"], artifact_root=study["root"]
    )
    first, second = Path(jobs[0].shard_path), Path(jobs[1].shard_path)
    second.write_bytes(first.read_bytes())

    with pytest.raises(ValueError, match="does not match job"):
        _collect(study)


def test_recovery_recomputed_from_stored_states_matches_the_metric(study) -> None:
    _collect(study)
    summary = _analyze(study)

    for cell in summary["cells"]:
        path = (
            study["study"]
            / "discovery/shards"
            / cell["checkpoint_id"]
            / f"{cell['condition_id']}.jsonl.gz"
        )
        recovered = survived = 0
        for episode in read_episode_shard(path):
            states = [episode.reset["environment_state"]]
            states += [row["environment_state"] for row in episode.transitions]
            tail = [state["pole_angle"] for state in states][-100:]
            final = episode.transitions[-1]
            alive = final["truncated"] and not final["terminated"]
            rms = sqrt(sum(value * value for value in tail) / len(tail))
            survived += alive
            recovered += alive and rms <= radians(5.0)
        assert cell["recovery_count"] == recovered
        assert cell["survival_count"] == survived
        assert cell["episode_count"] == 3


def test_analysis_needs_no_environment_or_policy(
    study, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _collect(study)
    first = _analyze(study, output_dir=tmp_path / "first")

    def refuse(*arguments, **options):
        raise AssertionError("Analysis must not touch environments or policies.")

    for module, name in (
        (gym, "make"),
        (ppo_boundary_evaluation, "load_policy_artifact"),
        (ppo_boundary_evaluation, "make_artifact_environment"),
        (ppo_boundary_evaluation, "collect_episode"),
    ):
        monkeypatch.setattr(module, name, refuse)

    second = _analyze(study, output_dir=tmp_path / "second")

    assert first == second
    for name in ("summary.json", "episode-metrics.jsonl.gz"):
        assert (tmp_path / "first" / name).read_bytes() == (
            tmp_path / "second" / name
        ).read_bytes()


def test_parallel_analysis_matches_serial(study, tmp_path: Path) -> None:
    _collect(study)

    serial = _analyze(study, output_dir=tmp_path / "serial")
    parallel = _analyze(study, output_dir=tmp_path / "parallel", workers=2)

    assert parallel == serial


def test_analysis_refuses_to_overwrite(study) -> None:
    _collect(study)
    _analyze(study)
    before = (study["study"] / "metrics_v1/discovery/summary.json").read_bytes()

    with pytest.raises(FileExistsError):
        _analyze(study)

    assert (study["study"] / "metrics_v1/discovery/summary.json").read_bytes() == before


def test_analysis_rejects_a_tampered_shard(study) -> None:
    _collect(study)
    path = next((study["study"] / "discovery/shards").rglob("*.gz"))
    _recompress_with(path, b'"action":', b'"action": ')

    with pytest.raises(ValueError, match="digest mismatch"):
        _analyze(study)


def test_analysis_rejects_a_shard_missing_a_planned_episode(study) -> None:
    _collect(study)
    path = next((study["study"] / "discovery/shards").rglob("*.gz"))
    header = read_shard_header(path)
    fields = {
        name: value for name, value in header.items() if name not in ("type", "layout")
    }
    records = []
    for episode in read_episode_shard(path):
        records.append(episode)
    path.unlink()
    content = gzip.compress(_forged_shard(fields, records[:-1]), mtime=0)
    path.write_bytes(content)

    with pytest.raises(ValueError, match="planned episodes"):
        _analyze(study)


def test_analysis_rejects_stochastic_episodes(study) -> None:
    _collect(study)
    path = next((study["study"] / "discovery/shards").rglob("*.gz"))
    first, rest = gzip.decompress(path.read_bytes()).split(b"\n", 1)
    forged = rest.replace(b'"deterministic":true', b'"deterministic":false')
    path.write_bytes(gzip.compress(first + b"\n" + forged, mtime=0))

    with pytest.raises(ValueError, match="Stochastic"):
        _analyze(study)


def test_analysis_rejects_another_checkpoints_episode(study) -> None:
    _collect(study)
    path = next((study["study"] / "discovery/shards").rglob("*.gz"))
    digest = read_shard_header(path)["model_sha256"].encode()
    first, rest = gzip.decompress(path.read_bytes()).split(b"\n", 1)
    path.write_bytes(
        gzip.compress(first + b"\n" + rest.replace(digest, b"0" * 64), mtime=0)
    )

    with pytest.raises(ValueError, match="another checkpoint"):
        _analyze(study)


def test_cells_use_the_campaign_family_confidence_level(study) -> None:
    _collect(study)
    summary = _analyze(study)
    pair_alpha = summary["simultaneous_family"]["pair_alpha"]

    assert pair_alpha == 0.05 / 4
    for cell in summary["cells"]:
        assert cell["recovery_simultaneous"] == clopper_pearson(
            cell["recovery_count"], cell["episode_count"], pair_alpha
        )


def test_analysis_records_one_row_per_episode(study) -> None:
    _collect(study)
    _analyze(study)

    path = study["study"] / "metrics_v1/discovery/episode-metrics.jsonl.gz"
    rows = [
        json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()
    ]

    assert len(rows) == 12
    assert {row["episode_seed"] for row in rows} == {10**12 + i for i in range(3)}
    assert all(row["metrics"]["metric_version"] == "episode-summary-v4" for row in rows)


def test_coverage_is_only_reported_on_the_reference_support(
    study, monkeypatch: pytest.MonkeyPatch
) -> None:
    _collect(study)
    assert _analyze(study)["coverage"] is None

    monkeypatch.setattr(ppo_boundary_evaluation, "tier_children", lambda *_: ())
    _collect(study, campaign="pilot")
    summary = _analyze(study, campaign="pilot")

    assert summary["support"] == "reference"
    assert summary["simultaneous_family"]["checkpoint_condition_pairs"] == 6
    assert summary["simultaneous_family"]["pair_alpha"] == 0.05 / 6
    coverage = summary["coverage"]["tiny_parent"]
    assert set(coverage["thresholds"]) == {"0.9", "0.5", "0.75", "0.95"}


def test_coverage_summary_matches_hand_computation() -> None:
    def cell(name, recovery, survival, lower, upper):
        return {
            "condition_id": name,
            "recovery_rate": recovery,
            "survival_rate": survival,
            "recovery_simultaneous": {"lower": lower, "upper": upper},
        }

    cells = [
        cell("a", 1.0, 1.0, 0.95, 1.0),
        cell("b", 0.9, 1.0, 0.70, 0.99),
        cell("c", 0.5, 0.8, 0.20, 0.80),
        cell("d", 0.0, 0.0, 0.00, 0.10),
    ]

    result = coverage_summary(cells, dict.fromkeys("abcd", 0.25), (0.9,))

    assert result["mean_recovery_probability"] == pytest.approx(0.6)
    assert result["mean_survival_probability"] == pytest.approx(0.7)
    assert result["thresholds"]["0.9"] == {
        "recovery_plug_in": 0.5,
        "recovery_simultaneous_lower": 0.25,
        "recovery_simultaneous_upper": 0.5,
        "survival_plug_in": 0.5,
    }


def test_coverage_summary_requires_every_condition_once() -> None:
    with pytest.raises(ValueError, match="exactly once"):
        coverage_summary([], {"a": 1.0}, (0.9,))


def test_clopper_pearson_matches_the_beta_definition() -> None:
    pair_alpha = 0.05 / 8463

    interval = clopper_pearson(100, 128, pair_alpha)

    assert interval["lower"] == pytest.approx(
        beta.ppf(pair_alpha / 2, 100, 29), rel=1e-9
    )
    assert interval["upper"] == pytest.approx(
        beta.ppf(1 - pair_alpha / 2, 101, 28), rel=1e-9
    )


def test_perfect_cell_matches_the_plans_simultaneous_limit() -> None:
    assert clopper_pearson(128, 128, 0.05 / 8463) == pytest.approx(
        {"lower": 0.905, "upper": 1.0}, abs=0.0005
    )
    assert clopper_pearson(0, 128, 0.05 / 8463)["lower"] == 0.0
