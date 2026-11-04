import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO

from active_eval_gym import ppo_boundary_training
from active_eval_gym.config import load_ppo_boundary_study
from active_eval_gym.envs.factory import make_environment
from active_eval_gym.policies.artifacts import (
    EVALUATION_SEAL_FILENAME,
    FREEZE_FILENAME,
    MANIFEST_FILENAME,
    TRAINING_SUMMARY_FILENAME,
    load_evaluation_ready_artifact,
    load_policy_artifact,
    verify_evaluation_seal,
    write_freeze_result,
)
from active_eval_gym.ppo_boundary import (
    child_artifact_dir,
    episode_seed,
    job_accounting,
    prepare_study,
    support_conditions,
    tier_children,
    training_policy_seed,
)
from active_eval_gym.ppo_boundary_analysis import analyze_campaign
from active_eval_gym.ppo_boundary_evaluation import (
    campaign_checkpoints,
    collect_campaign,
    plan_jobs,
)
from active_eval_gym.ppo_boundary_targets import build_distributions
from active_eval_gym.ppo_boundary_training import (
    OptimizerStepRecorder,
    TransitionCapture,
    _verify_first_update,
    load_distributions,
    make_training_environment,
    migrate_parent,
    plan_training,
    realized_sampling,
    resolved_settings,
    run_training_job,
    state_sha256,
    train_campaign,
)
from active_eval_gym.serialization import read_transition_shard
from tests.helpers import SMALL_STUDY_GRID, tiny_parent_config

SMALL_BUDGETS = (
    ("additional_transitions = 8192", "additional_transitions = 512"),
    ("additional_transitions = 102400", "additional_transitions = 1024"),
    (
        "snapshot_transitions = [25600, 51200, 76800]",
        "snapshot_transitions = [256, 768]",
    ),
    ("episodes_per_condition = 8", "episodes_per_condition = 1"),
)
SOURCE = {"git_commit": "a", "source_dirty": False, "changed_paths": []}


@pytest.fixture(scope="module")
def base_study(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root_dir = tmp_path_factory.mktemp("training")
    config, root = tiny_parent_config(root_dir, *SMALL_STUDY_GRID, *SMALL_BUDGETS)
    study = root_dir / "study"
    prepare_study(config, artifact_root=root, output_dir=study, allow_dirty=True)
    collect_campaign(
        config,
        study_dir=study,
        campaign="discovery",
        artifact_root=root,
        allow_dirty=True,
    )
    analyze_campaign(config, study_dir=study, campaign="discovery")
    build_distributions(config, study_dir=study, allow_dirty=True)
    return {"config": config, "root": root, "study": study}


@pytest.fixture
def study(base_study: dict[str, Path], tmp_path: Path) -> dict[str, Path]:
    copy = tmp_path / "study"
    shutil.copytree(base_study["study"], copy)
    return {**base_study, "study": copy}


def _spec(study: dict[str, Path]):
    return load_ppo_boundary_study(study["config"])


def _jobs(study: dict[str, Path], tier: str = "smoke"):
    return plan_training(
        _spec(study),
        tier,
        config_path=study["config"],
        study_dir=study["study"],
        artifact_root=study["root"],
        development_record=True,
        source=SOURCE,
    )


def _train(study: dict[str, Path], tier: str = "smoke", **options):
    return train_campaign(
        study["config"],
        study_dir=study["study"],
        tier=tier,
        artifact_root=study["root"],
        allow_dirty=True,
        **options,
    )


def _summary(directory: Path) -> dict:
    return json.loads((directory / TRAINING_SUMMARY_FILENAME).read_text())


def _child(study: dict[str, Path], job) -> Path:
    return child_artifact_dir(study["study"], job.tier, job.job_id)


def _transitions(study: dict[str, Path], job) -> dict[str, np.ndarray]:
    path = study["study"] / "training_raw" / job.tier / job.job_id / "transitions.npz"
    return read_transition_shard(path)[1]


def _weights(directory: Path) -> str:
    return state_sha256(
        PPO.load(directory / "model.zip", device="cpu").policy.state_dict()
    )


def test_first_update_uses_the_declared_values_in_the_optimizer(study) -> None:
    job = _jobs(study)[0]
    run_training_job(job)

    summary = _summary(_child(study, job))
    assert summary["migration"]["restored_optimizer_learning_rate"] != 0.0003
    assert summary["first_update"]["first_optimizer_step_learning_rate"] == 0.0003
    assert summary["first_update"]["first_update"]["learning_rate"] == 0.0003
    assert summary["first_update"]["first_update"]["clip_range"] == 0.2
    assert summary["first_update"]["last_update"]["clip_range"] == 0.2


def test_model_records_constant_schedules_after_training(study) -> None:
    job = _jobs(study)[0]
    run_training_job(job)

    model = PPO.load(_child(study, job) / "model.zip", device="cpu")
    assert model.lr_schedule(0.0) == model.lr_schedule(1.0) == 0.0003
    assert model.clip_range(0.0) == model.clip_range(1.0) == 0.2


def test_new_transition_count_is_exact_and_fully_retained(study) -> None:
    job = _jobs(study)[0]
    run_training_job(job)

    summary = _summary(_child(study, job))
    budget = summary["budget_accounting"]
    assert budget["num_timesteps_after"] - budget["num_timesteps_before"] == 512
    assert budget["rollouts"] == 2
    assert budget["n_updates_after"] - budget["n_updates_before"] == 40
    assert _transitions(study, job)["state"].shape == (512, 4)
    assert summary["first_update"]["optimizer_steps"] == 40


def test_same_seed_twice_gives_identical_weights_and_data(
    study, base_study, tmp_path: Path
) -> None:
    other_dir = tmp_path / "other"
    shutil.copytree(base_study["study"], other_dir)
    other = {**base_study, "study": other_dir}
    first, second = _jobs(study)[1], _jobs(other)[1]
    run_training_job(first)
    run_training_job(second)

    one, two = _summary(_child(study, first)), _summary(_child(other, second))
    assert one["final_policy_sha256"] == two["final_policy_sha256"]
    assert one["final_optimizer_sha256"] == two["final_optimizer_sha256"]
    assert (
        one["transitions_file"]["content_sha256"]
        == two["transitions_file"]["content_sha256"]
    )
    assert _weights(_child(study, first)) == _weights(_child(other, second))


def test_a_job_does_not_inherit_random_state_from_an_earlier_job(
    study, base_study, tmp_path: Path
) -> None:
    other_dir = tmp_path / "other"
    shutil.copytree(base_study["study"], other_dir)
    other = {**base_study, "study": other_dir}
    jobs = _jobs(study)
    run_training_job(jobs[0])
    run_training_job(jobs[1])
    alone = _jobs(other)[1]
    run_training_job(alone)

    assert (
        _summary(_child(study, jobs[1]))["final_policy_sha256"]
        == _summary(_child(other, alone))["final_policy_sha256"]
    )


def test_chunked_and_single_call_training_are_identical(
    study, base_study, tmp_path: Path
) -> None:
    other_dir = tmp_path / "other"
    shutil.copytree(base_study["study"], other_dir)
    other = {**base_study, "study": other_dir}
    chunked, single = _jobs(study, "pilot")[0], _jobs(other, "pilot")[0]
    run_training_job(chunked)
    run_training_job(single, snapshot_transitions=())

    one, two = _summary(_child(study, chunked)), _summary(_child(other, single))
    assert one["final_policy_sha256"] == two["final_policy_sha256"]
    assert one["final_optimizer_sha256"] == two["final_optimizer_sha256"]
    assert (
        one["transitions_file"]["content_sha256"]
        == two["transitions_file"]["content_sha256"]
    )
    assert not (_child(other, single) / "snapshots").exists()


def test_snapshots_are_saved_after_their_rollout_is_optimized(study) -> None:
    job = _jobs(study, "pilot")[0]
    run_training_job(job)

    child = _child(study, job)
    hashes = _transitions(study, job)["rollout_policy_sha256"].tolist()
    # The behavior policy of rollout k is the policy after k optimized rollouts.
    assert _weights(child / "snapshots/t000256") == hashes[1]
    assert _weights(child / "snapshots/t000768") == hashes[3]
    assert _weights(child) not in hashes


def test_snapshots_are_sealed_diagnostics_with_their_own_identity(study) -> None:
    job = _jobs(study, "pilot")[0]
    run_training_job(job)

    directory = _child(study, job) / "snapshots/t000256"
    seal = verify_evaluation_seal(directory)
    assert seal["checkpoint_role"] == "diagnostic-snapshot"
    assert seal["additional_transitions"] == 256
    assert seal["policy_id"] == f"{job.job_id}-t000256"
    _, metadata = load_evaluation_ready_artifact(directory)
    assert metadata.design_spec.training_steps == 256
    assert _summary(directory)["checkpoint_selection"] == "final-fixed-budget"


def test_each_rollout_records_its_behavior_policy(study) -> None:
    job = _jobs(study, "pilot")[0]
    run_training_job(job)

    summary = _summary(_child(study, job))
    table = _transitions(study, job)
    hashes = table["rollout_policy_sha256"].tolist()
    assert hashes[0] == summary["migration"]["policy_sha256"]
    assert len(set(hashes)) == 4
    assert np.diff(table["rollout_n_updates"]).tolist() == [20, 20, 20]
    assert np.diff(table["rollout_num_timesteps"]).tolist() == [256, 256, 256]
    assert table["rollout"].tolist() == np.repeat(np.arange(4), 256).tolist()


def test_retained_actions_replay_to_the_retained_states(study) -> None:
    job = _jobs(study)[1]
    run_training_job(job)

    table = _transitions(study, job)
    conditions = support_conditions(_spec(study), "discovery")
    keys = np.column_stack((table["worker"], table["episode_index"]))
    replayed = 0
    for key in np.unique(keys, axis=0):
        rows = np.flatnonzero((keys == key).all(axis=1))
        first = rows[0]
        condition = conditions[table["condition_index"][first]]
        env = make_environment("CartPole-v1", condition.perturbation)
        observation, _ = env.reset(seed=int(table["reset_seed"][first]))
        np.testing.assert_array_equal(observation, table["observation"][first])
        np.testing.assert_array_equal(env.unwrapped.state, table["state"][first])
        for row in rows:
            _, reward, terminated, truncated, _ = env.step(int(table["action"][row]))
            np.testing.assert_array_equal(env.unwrapped.state, table["next_state"][row])
            assert reward == table["environment_reward"][row]
            assert terminated == table["terminated"][row]
            assert truncated == table["truncated"][row]
            replayed += 1
        env.close()

    assert replayed == 512


def test_every_reset_follows_the_declared_seed_stream(study) -> None:
    job = _jobs(study)[0]
    run_training_job(job)

    table = _transitions(study, job)
    expected = [
        episode_seed(
            role=2,
            replicate_id=job.replicate_id,
            worker_id=int(worker),
            episode_index=int(episode),
        )
        for worker, episode in zip(
            table["reset_worker"], table["reset_episode_index"], strict=True
        )
    ]
    assert table["reset_reset_seed"].tolist() == expected
    for worker in range(8):
        indices = table["reset_episode_index"][table["reset_worker"] == worker]
        assert indices.tolist() == list(range(indices.size))


def test_arms_of_a_replicate_share_their_seeds_and_differ_only_by_arm(study) -> None:
    jobs = _jobs(study)
    for job in jobs:
        run_training_job(job)

    summaries = [_summary(_child(study, job)) for job in jobs]
    assert {summary["seeds"]["policy_seed"] for summary in summaries} == {
        training_policy_seed(_spec(study), replicate_id=900)
    }
    assert {summary["migration"]["policy_sha256"] for summary in summaries} == {
        summaries[0]["migration"]["policy_sha256"]
    }
    tables = [_transitions(study, job) for job in jobs]
    for table in tables[1:]:
        assert table["reset_seed"][:8].tolist() == tables[0]["reset_seed"][:8].tolist()


def test_each_arm_trains_on_its_own_distribution(study) -> None:
    path = study["study"] / "distributions/distributions.json"
    record = json.loads(path.read_text())
    for index, arm in enumerate(("background", "acceptance_boundary", "intermediate")):
        weights = [0.0] * 4
        weights[index] = 1.0
        record["arms"][arm]["probabilities"] = weights
    path.write_text(json.dumps(record))
    _train(study)

    for job in _jobs(study):
        expected = _spec(study).curriculum.arms.index(job.arm)
        table = _transitions(study, job)
        assert set(table["condition_index"].tolist()) == {expected}
        assert set(table["reset_condition_index"].tolist()) == {expected}


def test_replicates_apply_their_own_continuation_seed(study) -> None:
    first, second = _jobs(study, "pilot")[:4:3]
    assert first.arm == second.arm
    assert first.replicate_id != second.replicate_id
    run_training_job(first)
    run_training_job(second)

    for job in (first, second):
        model = PPO.load(_child(study, job) / "model.zip", device="cpu")
        assert model.seed == 300000 + job.replicate_id
    one, two = _transitions(study, first), _transitions(study, second)
    assert one["action"].tolist() != two["action"].tolist()
    assert one["reset_seed"].tolist() != two["reset_seed"].tolist()


def test_realized_shares_count_episodes_transitions_and_unfinished_runs() -> None:
    tables = {
        "condition_index": np.array([0, 0, 1, 1, 1]),
        "terminated": np.array([False, True, False, False, False]),
        "truncated": np.array([False, False, False, False, False]),
        "worker": np.array([0, 0, 0, 0, 0]),
        "episode_index": np.array([0, 0, 1, 1, 1]),
        "reset_condition_index": np.array([0, 1]),
        "reset_worker": np.array([0, 0]),
        "reset_episode_index": np.array([0, 1]),
    }

    result = realized_sampling(tables, ("a", "b", "c"))

    assert result["episodes_started"].tolist() == [1, 1, 0]
    assert result["episodes_completed"].tolist() == [1, 0, 0]
    assert result["transitions"].tolist() == [2, 3, 0]
    assert result["episode_shares"].tolist() == [0.5, 0.5, 0.0]
    assert result["transition_shares"].tolist() == [0.4, 0.6, 0.0]
    assert result["unfinished_episodes"] == [
        {"worker": 0, "episode_index": 1, "condition_id": "b", "transitions": 3}
    ]


def test_an_episode_reset_at_the_budget_end_is_retained_with_no_transitions() -> None:
    tables = {
        "condition_index": np.array([0]),
        "terminated": np.array([True]),
        "truncated": np.array([False]),
        "worker": np.array([0]),
        "episode_index": np.array([0]),
        "reset_condition_index": np.array([0, 1]),
        "reset_worker": np.array([0, 0]),
        "reset_episode_index": np.array([0, 1]),
    }

    assert realized_sampling(tables, ("a", "b"))["unfinished_episodes"] == [
        {"worker": 0, "episode_index": 1, "condition_id": "b", "transitions": 0}
    ]


def test_child_training_record_is_not_described_as_nominal(study) -> None:
    job = _jobs(study)[2]
    run_training_job(job)

    summary = _summary(_child(study, job))
    _, metadata = load_evaluation_ready_artifact(_child(study, job))
    assert summary["training_kind"] == "curriculum-continuation"
    assert summary["training_task"]["theta_threshold_deg"] == 90
    assert summary["training_task"]["training_reward"] == "gymnasium-survival-plus-one"
    assert metadata.policy_id == job.job_id != _spec(study).parent_policy_id
    assert metadata.design_spec.environment.spec_id == (
        _spec(study).parent_design.environment.spec_id
    )
    assert metadata.design_spec.hyperparameters["continues_model_sha256"] == (
        _spec(study).parent_model_sha256
    )


def test_parent_files_are_unchanged_by_training(study) -> None:
    directory = study["root"] / _spec(study).parent_policy_id
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    _train(study)

    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before


def test_a_regressing_child_seals_loads_and_is_evaluated(study) -> None:
    _train(study)
    children = [_child(study, job) for job in _jobs(study)]
    for directory in children:
        assert not (directory / FREEZE_FILENAME).exists()
        with pytest.raises(FileNotFoundError):
            load_policy_artifact(directory, require_frozen=True)
    write_freeze_result(children[0], passed=False, gate={}, results={})

    with pytest.raises(FileNotFoundError):
        load_policy_artifact(children[0], require_frozen=True)
    policy, _ = load_evaluation_ready_artifact(children[0])
    assert policy.act(np.zeros(4, dtype=np.float32)) in (0, 1)
    result = collect_campaign(
        study["config"],
        study_dir=study["study"],
        campaign="smoke",
        artifact_root=study["root"],
        allow_dirty=True,
    )
    assert result["complete"]


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("model.zip", "model file"),
        (MANIFEST_FILENAME, "manifest"),
        (TRAINING_SUMMARY_FILENAME, "training summary"),
    ],
)
def test_tampering_with_a_sealed_file_is_refused(study, target, message) -> None:
    job = _jobs(study)[0]
    run_training_job(job)
    directory = _child(study, job)
    path = directory / target
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match=message):
        load_evaluation_ready_artifact(directory)


def test_a_seal_without_a_completed_budget_is_refused(study) -> None:
    job = _jobs(study)[0]
    run_training_job(job)
    path = _child(study, job) / EVALUATION_SEAL_FILENAME
    seal = json.loads(path.read_text())
    path.write_text(json.dumps({**seal, "completed_fixed_budget": False}))

    with pytest.raises(ValueError, match="no completed budget"):
        verify_evaluation_seal(_child(study, job))


def test_direct_attachment_fails_and_migration_changes_no_behavior(study) -> None:
    spec = _spec(study)
    parent_dir = study["root"] / spec.parent_policy_id
    _, parent = load_policy_artifact(parent_dir)
    probabilities = np.full(4, 0.25)
    environment = make_training_environment(
        spec, parent, probabilities, replicate_id=900
    )
    try:
        with pytest.raises(ValueError, match="[Oo]bservation"):
            PPO.load(parent_dir / "model.zip", env=environment, device="cpu")
        model, record = migrate_parent(
            parent_dir / "model.zip", environment, spec.continuation, 300900
        )
        original = PPO.load(parent_dir / "model.zip", device="cpu")
        assert state_sha256(model.policy.state_dict()) == state_sha256(
            original.policy.state_dict()
        )
        assert state_sha256(model.policy.optimizer.state_dict()) == state_sha256(
            original.policy.optimizer.state_dict()
        )
        assert record["policy_sha256"] == state_sha256(original.policy.state_dict())
        assert model.observation_space == environment.observation_space
        assert model.action_space == original.action_space
    finally:
        environment.close()


@pytest.mark.parametrize("name", ["policy_sha256", "logits"])
def test_a_migration_that_changes_the_policy_is_refused(
    study, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    spec = _spec(study)
    parent_dir = study["root"] / spec.parent_policy_id
    _, parent = load_policy_artifact(parent_dir)
    real = ppo_boundary_training._fingerprint
    calls = []

    def altered(model):
        result = real(model)
        calls.append(model)
        if len(calls) == 2:
            result[name] = "x" if name == "policy_sha256" else result[name] + 1.0
        return result

    monkeypatch.setattr(ppo_boundary_training, "_fingerprint", altered)
    environment = make_training_environment(
        spec, parent, np.full(4, 0.25), replicate_id=900
    )
    try:
        with pytest.raises(RuntimeError, match=f"changed {name}"):
            migrate_parent(
                parent_dir / "model.zip", environment, spec.continuation, 300900
            )
    finally:
        environment.close()


def _update_checks(learning_rate: float, clip_range: float):
    recorder = OptimizerStepRecorder.__new__(OptimizerStepRecorder)
    recorder.first_learning_rate = learning_rate
    recorder.learning_rates = {learning_rate}
    recorder.steps = 20
    capture = TransitionCapture(1)
    capture.rollouts = [{}]
    capture.updates = [
        {"n_updates": 20, "learning_rate": learning_rate, "clip_range": clip_range}
    ]
    return recorder, capture


@pytest.mark.parametrize(
    ("learning_rate", "clip_range", "message"),
    [(-9.6e-07, 0.2, "First optimizer step"), (0.0003, 0.1, "clip range")],
)
def test_wrong_first_update_values_fail_the_job(
    study, learning_rate, clip_range, message
) -> None:
    recorder, capture = _update_checks(learning_rate, clip_range)

    with pytest.raises(RuntimeError, match=message):
        _verify_first_update(recorder, capture, _spec(study).continuation)


def test_a_missing_update_record_fails_the_job(study) -> None:
    recorder, capture = _update_checks(0.0003, 0.2)
    capture.rollouts = [{}, {}]

    with pytest.raises(RuntimeError, match="one recorded update"):
        _verify_first_update(recorder, capture, _spec(study).continuation)


def test_a_short_training_run_is_never_sealed(
    study, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _jobs(study)[0]
    real = PPO.learn

    def short(self, total_timesteps, **kwargs):
        return real(self, total_timesteps - 256, **kwargs)

    monkeypatch.setattr(ppo_boundary_training.PPO, "learn", short)

    with pytest.raises(RuntimeError, match="new transitions"):
        run_training_job(job)
    assert not _child(study, job).exists()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("n_epochs", 5, "undeclared n_epochs"),
        ("gamma", 0.5, "undeclared gamma"),
        ("n_envs", 4, "undeclared n_envs"),
        ("lr_schedule", lambda progress: 0.001 * progress, "Learning-rate"),
        ("clip_range", lambda progress: 0.2 * progress, "Clip-range"),
    ],
)
def test_undeclared_model_settings_are_refused(study, name, value, message) -> None:
    spec = _spec(study)
    parent_dir = study["root"] / spec.parent_policy_id
    _, parent = load_policy_artifact(parent_dir)
    environment = make_training_environment(
        spec, parent, np.full(4, 0.25), replicate_id=900
    )
    try:
        model, _ = migrate_parent(
            parent_dir / "model.zip", environment, spec.continuation, 300900
        )
        resolved_settings(model, spec.continuation)
        setattr(model, name, value)
        with pytest.raises(RuntimeError, match=message):
            resolved_settings(model, spec.continuation)
    finally:
        environment.close()


def test_a_varying_optimizer_rate_fails_the_job(study) -> None:
    recorder, capture = _update_checks(0.0003, 0.2)
    recorder.learning_rates = {0.0003, 0.0001}

    with pytest.raises(RuntimeError, match="varied"):
        _verify_first_update(recorder, capture, _spec(study).continuation)


def test_a_failed_job_is_recorded_unsealed_then_resumes_identically(
    study, base_study, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other_dir = tmp_path / "other"
    shutil.copytree(base_study["study"], other_dir)
    other = {**base_study, "study": other_dir}
    _train(other)

    def interrupted(*_args, **_kwargs):
        raise OSError("disk went away")

    monkeypatch.setattr(ppo_boundary_training, "write_transition_shard", interrupted)
    first = _train(study)
    monkeypatch.undo()

    assert not first["complete"]
    assert len(first["failed_jobs"]) == 3
    job = _jobs(study)[0]
    assert not _child(study, job).exists()
    failures = study["study"] / "policies/smoke/failures"
    assert len(list(failures.iterdir())) == 3
    assert not (study["study"] / "policies/smoke/training.json").exists()

    resumed = _train(study)

    assert resumed["complete"]
    assert resumed["already_complete_jobs"] == 0
    abandoned = study["study"] / "training_raw/smoke/abandoned"
    assert sorted(path.name for path in abandoned.iterdir()) == sorted(
        f"{job.job_id}.attempt-1" for job in _jobs(study)
    )
    for ours, theirs in zip(_jobs(study), _jobs(other), strict=True):
        assert (
            _summary(_child(study, ours))["final_policy_sha256"]
            == _summary(_child(other, theirs))["final_policy_sha256"]
        )


def test_resume_runs_only_missing_children(
    study, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = _jobs(study)
    run_training_job(jobs[1])
    ran = []
    real = ppo_boundary_training.run_training_job

    def tracked(job):
        ran.append(job.job_id)
        return real(job)

    monkeypatch.setattr(ppo_boundary_training, "run_training_job", tracked)
    result = _train(study)

    assert result["already_complete_jobs"] == 1
    assert ran == [jobs[0].job_id, jobs[2].job_id]
    assert _train(study)["already_complete_jobs"] == 3
    assert len(ran) == 2


def test_a_crash_between_publication_steps_is_shelved_and_rerun(
    study, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _jobs(study)[0]
    real_rename = ppo_boundary_training.os.rename
    calls = []

    def crash_on_second(source, target):
        calls.append(source)
        if len(calls) == 2:
            raise OSError("killed")
        real_rename(source, target)

    monkeypatch.setattr(ppo_boundary_training.os, "rename", crash_on_second)
    with pytest.raises(OSError, match="killed"):
        run_training_job(job)
    monkeypatch.undo()
    raw = study["study"] / "training_raw/smoke" / job.job_id
    assert raw.exists()
    assert not _child(study, job).exists()

    run_training_job(job)

    shelf = study["study"] / f"training_raw/smoke/abandoned/{job.job_id}.attempt-1"
    assert (shelf / "published-raw/transitions.npz").is_file()
    assert (shelf / "work/policy/evaluation-ready.json").is_file()
    verify_evaluation_seal(_child(study, job))


def test_a_changed_parent_stops_publication(
    study, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _jobs(study)[0]
    parent_zip = Path(job.parent_dir) / "model.zip"
    original = parent_zip.read_bytes()
    real = ppo_boundary_training.write_transition_shard

    def write_and_alter_parent(*args, **kwargs):
        parent_zip.write_bytes(original + b"x")
        return real(*args, **kwargs)

    monkeypatch.setattr(
        ppo_boundary_training, "write_transition_shard", write_and_alter_parent
    )
    try:
        with pytest.raises(RuntimeError, match="changed the parent"):
            run_training_job(job)
    finally:
        parent_zip.write_bytes(original)

    assert not _child(study, job).exists()


def test_a_published_child_is_never_retrained(study) -> None:
    job = _jobs(study)[0]
    run_training_job(job)

    with pytest.raises(FileExistsError, match="retrain"):
        run_training_job(job)
    assert (study["study"] / "training_raw/smoke" / job.job_id).is_dir()


def test_a_child_whose_model_changed_is_refused_on_resume(study) -> None:
    job = _jobs(study)[0]
    run_training_job(job)
    path = _child(study, job) / "model.zip"
    path.write_bytes(path.read_bytes() + b"x")

    with pytest.raises(ValueError, match="model file"):
        _train(study)


def test_a_child_whose_raw_data_changed_is_refused_on_resume(study) -> None:
    job = _jobs(study)[0]
    run_training_job(job)
    path = study["study"] / "training_raw/smoke" / job.job_id / "transitions.npz"
    path.write_bytes(path.read_bytes() + b"x")

    with pytest.raises(ValueError, match="does not match sealed child"):
        _train(study)


def test_parallel_training_matches_serial_training(
    study, base_study, tmp_path: Path
) -> None:
    other_dir = tmp_path / "other"
    shutil.copytree(base_study["study"], other_dir)
    other = {**base_study, "study": other_dir}
    assert _train(study)["complete"]
    assert _train(other, workers=2)["complete"]

    for ours, theirs in zip(_jobs(study), _jobs(other), strict=True):
        one, two = _summary(_child(study, ours)), _summary(_child(other, theirs))
        assert one["final_policy_sha256"] == two["final_policy_sha256"]
        assert (
            one["transitions_file"]["content_sha256"]
            == two["transitions_file"]["content_sha256"]
        )


def test_confirmatory_training_is_refused_before_the_lock(study) -> None:
    with pytest.raises(ValueError, match="S7 lock"):
        _train(study, tier="confirmatory")

    assert not (study["study"] / "policies").exists()


def test_training_refuses_uncommitted_source_by_default(
    study, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ppo_boundary_training,
        "source_state",
        lambda: {"git_commit": "a", "source_dirty": True, "changed_paths": ["x"]},
    )

    with pytest.raises(ValueError, match="committed source"):
        train_campaign(
            study["config"],
            study_dir=study["study"],
            tier="smoke",
            artifact_root=study["root"],
        )


def test_resume_on_changed_source_is_refused(
    study, monkeypatch: pytest.MonkeyPatch
) -> None:
    _train(study)
    monkeypatch.setattr(
        ppo_boundary_training,
        "source_state",
        lambda: {"git_commit": "other", "source_dirty": True, "changed_paths": []},
    )

    with pytest.raises(ValueError, match="recorded plan"):
        _train(study)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda record: record.update(study_id="other"), "another study"),
        (lambda record: record.update(target_rule_version="x"), "undeclared target"),
        (
            lambda record: record["selection"].update(protocol_config_sha256="x"),
            "study protocol",
        ),
        (
            lambda record: record["selection"].update(parent_model_sha256="x"),
            "another parent",
        ),
        (lambda record: record["condition_ids"].reverse(), "discovery conditions"),
        (lambda record: record["arms"].pop("intermediate"), "declared arms"),
    ],
)
def test_distributions_from_another_protocol_are_refused(study, edit, message) -> None:
    path = study["study"] / "distributions/distributions.json"
    record = json.loads(path.read_text())
    edit(record)
    path.write_text(json.dumps(record))

    with pytest.raises(ValueError, match=message):
        _train(study)


def test_development_distributions_are_refused_without_allow_dirty(study) -> None:
    path = study["study"] / "distributions/distributions.json"
    record = json.loads(path.read_text())
    path.write_text(json.dumps({**record, "development_record": True}))
    provenance = json.loads((study["study"] / "protocol/provenance.json").read_text())

    with pytest.raises(ValueError, match="development record"):
        load_distributions(_spec(study), provenance, study["study"])
    load_distributions(_spec(study), provenance, study["study"], allow_dirty=True)


def test_missing_distributions_are_refused(study) -> None:
    (study["study"] / "distributions/distributions.json").unlink()

    with pytest.raises(FileNotFoundError, match="frozen distributions"):
        _train(study)


def test_tier_evaluation_refuses_until_every_child_is_sealed(study) -> None:
    run_training_job(_jobs(study)[0])

    with pytest.raises(ValueError, match="every child sealed"):
        collect_campaign(
            study["config"],
            study_dir=study["study"],
            campaign="smoke",
            artifact_root=study["root"],
            allow_dirty=True,
        )
    assert not (study["study"] / "smoke").exists()


def test_tier_evaluation_covers_the_parent_and_all_children(study) -> None:
    _train(study)
    spec = _spec(study)
    jobs = plan_jobs(
        spec,
        "smoke",
        study_dir=study["study"],
        artifact_root=study["root"],
        allow_dirty=True,
    )
    expected = job_accounting(spec)["tiers"]["smoke"]["primary_evaluation"]
    assert len(jobs) == expected["jobs"]
    assert jobs[0].header["checkpoint_id"] == spec.parent_policy_id
    assert not jobs[0].sealed
    assert {job.sealed for job in jobs[expected["conditions"] :]} == {True}
    assert len({job.header["checkpoint_id"] for job in jobs}) == 4
    assert len({tuple(job.header["episode_seeds"]) for job in jobs}) == 1

    collected = collect_campaign(
        study["config"],
        study_dir=study["study"],
        campaign="smoke",
        artifact_root=study["root"],
        allow_dirty=True,
    )
    assert collected["complete"]
    summary = analyze_campaign(
        study["config"], study_dir=study["study"], campaign="smoke"
    )
    assert len(summary["cells"]) == expected["jobs"]
    assert {cell["checkpoint_id"] for cell in summary["cells"]} == {
        spec.parent_policy_id,
        *(child["policy_id"] for child in tier_children(spec, "smoke")),
    }


def test_evaluation_changes_no_child_weights(study) -> None:
    _train(study)
    children = [_child(study, job) for job in _jobs(study)]
    before = [(path / "model.zip").read_bytes() for path in children]
    collect_campaign(
        study["config"],
        study_dir=study["study"],
        campaign="smoke",
        artifact_root=study["root"],
        allow_dirty=True,
    )

    assert [(path / "model.zip").read_bytes() for path in children] == before


def test_development_children_are_refused_for_real_campaigns(
    study, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ppo_boundary_training,
        "source_state",
        lambda: {"git_commit": "a", "source_dirty": True, "changed_paths": ["x"]},
    )
    _train(study)
    arguments = {"study_dir": study["study"], "artifact_root": study["root"]}

    with pytest.raises(ValueError, match="development record"):
        campaign_checkpoints(_spec(study), "smoke", **arguments)
    allowed = campaign_checkpoints(_spec(study), "smoke", allow_dirty=True, **arguments)
    assert len(allowed) == 4


def test_a_child_of_another_parent_is_refused(study) -> None:
    _train(study)
    spec = _spec(study)
    job = _jobs(study)[0]
    path = _child(study, job) / EVALUATION_SEAL_FILENAME
    seal = json.loads(path.read_text())
    path.write_text(json.dumps({**seal, "parent_model_sha256": "other"}))

    with pytest.raises(ValueError, match="another parent"):
        campaign_checkpoints(
            spec,
            "smoke",
            study_dir=study["study"],
            artifact_root=study["root"],
            allow_dirty=True,
        )


def test_child_ids_are_fresh_and_ordered(study) -> None:
    spec = _spec(study)
    identifiers = [child["policy_id"] for child in tier_children(spec, "pilot")]

    assert len(identifiers) == len(set(identifiers)) == 9
    assert spec.parent_policy_id not in identifiers
    assert identifiers[0].endswith("pilot-r000-background")
    assert identifiers[-1].endswith("pilot-r002-intermediate")
