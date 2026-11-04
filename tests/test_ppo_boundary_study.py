import json
import subprocess
import tomllib
from pathlib import Path

import numpy as np
import pytest

from active_eval_gym import ppo_boundary
from active_eval_gym.config import QualityGate, load_ppo_boundary_study
from active_eval_gym.envs.perturbations import CARTPOLE_RECOVERY_ANGLE_LENGTH
from active_eval_gym.policies.artifacts import (
    train_learned_artifact,
    write_freeze_result,
)
from active_eval_gym.ppo_boundary import (
    condition_seed_sequence,
    episode_seed,
    evaluation_episode_seeds,
    job_accounting,
    prepare_study,
    reference_weights,
    seed_plan,
    support_conditions,
    training_policy_seed,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/cartpole_ppo_boundary_v1.toml"
PARENT_DESIGN = ROOT / "configs/policies/cartpole_ppo_nominal_v1.toml"
NOMINAL = ROOT / "configs/envs/cartpole_v1_nominal.toml"


@pytest.fixture
def spec():
    return load_ppo_boundary_study(CONFIG)


def _edited_config(tmp_path: Path, *replacements: tuple[str, str]) -> Path:
    text = CONFIG.read_text().replace(
        'design_config = "../policies/cartpole_ppo_nominal_v1.toml"',
        f'design_config = "{PARENT_DESIGN}"',
    )
    for old, new in replacements:
        assert old in text
        text = text.replace(old, new)
    path = tmp_path / "study.toml"
    path.write_text(text)
    return path


@pytest.fixture
def tiny_parent(tmp_path: Path) -> tuple[Path, Path]:
    """Build a real, frozen, short-trained parent and a config declaring it."""

    design = tmp_path / "tiny.toml"
    design.write_text(
        PARENT_DESIGN.read_text()
        .replace('"cartpole_ppo_nominal_v1"', '"tiny_parent"')
        .replace("training_steps = 100000", "training_steps = 256")
        .replace('"../envs/cartpole_v1_nominal.toml"', f'"{NOMINAL}"')
    )
    root = tmp_path / "policies"
    root.mkdir()
    metadata = train_learned_artifact(design, root / "tiny_parent")
    write_freeze_result(
        root / "tiny_parent", passed=True, gate=QualityGate(), results={}
    )
    config = _edited_config(
        tmp_path,
        ('policy_id = "cartpole_ppo_nominal_v1"', 'policy_id = "tiny_parent"'),
        (f'"{PARENT_DESIGN}"', f'"{design}"'),
        (
            "37c9bac10919b2d6bdc833fd5a3a920460d4b80f09392ef7fb758e4777a856e8",
            metadata.model_sha256,
        ),
    )
    return config, root


def _clean_source(monkeypatch: pytest.MonkeyPatch, *, dirty: bool = False) -> None:
    monkeypatch.setattr(
        ppo_boundary,
        "source_state",
        lambda: {
            "git_commit": "abc",
            "source_dirty": dirty,
            "changed_paths": ["src/x.py"] if dirty else [],
        },
    )


def test_supports_have_the_declared_size_and_order(spec) -> None:
    discovery = support_conditions(spec, "discovery")
    reference = support_conditions(spec, "reference")
    smoke = support_conditions(spec, "smoke")

    assert (len(discovery), len(reference), len(smoke)) == (143, 273, 11)
    assert discovery[0].condition_id == "theta-m50_length-p0p02"
    assert reference[-1].condition_id == "theta-p50_length-p3"
    assert {c.condition_id for c in discovery} < {c.condition_id for c in reference}
    assert {"theta-m20_length-p0p5", "theta-p20_length-p0p5"} <= {
        c.condition_id for c in smoke
    }


def test_every_condition_uses_the_recovery_perturbation_at_90_degrees(spec) -> None:
    for name in spec.supports:
        for condition in support_conditions(spec, name):
            assert condition.perturbation.name == CARTPOLE_RECOVERY_ANGLE_LENGTH
            parameters = condition.perturbation.parameters
            assert set(parameters) == {
                "initial_theta_deg",
                "length",
                "theta_threshold_deg",
            }
            assert parameters["theta_threshold_deg"] == 90
            assert -50 <= parameters["initial_theta_deg"] <= 50
            assert 0.02 <= parameters["length"] <= 3.0


def test_reference_weights_are_equal_and_normalized(spec) -> None:
    weights = reference_weights(spec)

    assert weights.shape == (273,)
    assert np.all(weights == 1 / 273)
    assert np.isclose(weights.sum(), 1.0)


def test_decisions_are_recorded_in_the_config(spec) -> None:
    assert spec.storage.evaluation_layout == "checkpoint-condition-shard-v1"
    assert spec.storage.compression == "gzip"
    assert spec.storage.episode_digest == "sha256-uncompressed-trajectory"
    assert spec.curriculum.thin_band_min_conditions == 5
    assert spec.tier("pilot").snapshot_evaluation == "if-s6-headroom"
    assert spec.tier("confirmatory").snapshot_evaluation == "none"
    assert spec.tier("smoke").snapshot_evaluation == "none"


def test_continuation_settings_match_the_plan_and_parent(spec) -> None:
    assert spec.continuation.learning_rate == 0.0003
    assert spec.continuation.clip_range == 0.2
    for name, value in spec.continuation.hyperparameters.items():
        assert spec.parent_design.hyperparameters[name] == value
    assert spec.tier("confirmatory").additional_transitions == 400 * 8 * 32


def test_episode_seed_matches_the_declared_formula() -> None:
    assert (
        episode_seed(
            role=2, outer_round=3, replicate_id=45, worker_id=6, episode_index=789
        )
        == 2 * 10**12 + 3 * 10**10 + 45 * 10**7 + 6 * 10**5 + 789
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"role": 0, "episode_index": 0},
        {"role": 10, "episode_index": 0},
        {"role": 1, "episode_index": 100_000},
        {"role": 1, "episode_index": -1},
        {"role": 1, "episode_index": 0, "outer_round": 100},
        {"role": 1, "episode_index": 0, "replicate_id": 1000},
        {"role": 1, "episode_index": 0, "worker_id": 100},
        {"role": 1, "episode_index": True},
        {"role": 1, "episode_index": 1.0},
    ],
)
def test_episode_seed_rejects_values_that_could_overlap(arguments) -> None:
    with pytest.raises(ValueError):
        episode_seed(**arguments)


def test_extreme_seed_indices_stay_inside_their_role() -> None:
    largest = episode_seed(
        role=1, outer_round=99, replicate_id=999, worker_id=99, episode_index=99_999
    )

    assert largest < episode_seed(role=2, episode_index=0)


def test_all_declared_seed_sets_are_disjoint(spec) -> None:
    roles = spec.seeds.roles
    groups = {
        "discovery": set(evaluation_episode_seeds(roles["discovery"], 32)),
        **{
            tier.name: set(
                evaluation_episode_seeds(
                    roles[tier.seed_role], tier.episodes_per_condition
                )
            )
            for tier in spec.tiers
        },
        "training": {
            episode_seed(
                role=roles["training"],
                replicate_id=replicate,
                worker_id=worker,
                episode_index=index,
            )
            for tier in spec.tiers
            for replicate in tier.replicate_ids
            for worker in range(8)
            for index in range(64)
        },
    }
    combined = [seed for group in groups.values() for seed in group]

    assert len(set(combined)) == len(combined)
    assert min(combined) > 49


def test_training_seeds_pair_arms_and_differ_between_replicates(spec) -> None:
    seeds = [
        training_policy_seed(spec, replicate_id=replicate)
        for tier in spec.tiers
        for replicate in tier.replicate_ids
    ]

    assert training_policy_seed(spec, replicate_id=3, outer_round=2) == 302003
    assert len(set(seeds)) == len(seeds)


def test_condition_streams_are_reproducible_and_separate(spec) -> None:
    def draw(**arguments) -> np.ndarray:
        stream = condition_seed_sequence(spec, **arguments)
        return np.random.default_rng(stream).integers(0, 2**32, 8)

    first = draw(replicate_id=100, worker_id=0)

    assert np.array_equal(first, draw(replicate_id=100, worker_id=0))
    assert not np.array_equal(first, draw(replicate_id=100, worker_id=1))
    assert not np.array_equal(first, draw(replicate_id=101, worker_id=0))
    assert condition_seed_sequence(spec, replicate_id=1, worker_id=2).entropy == [
        6180,
        0,
        1,
        2,
        10,
    ]


def test_job_accounting_matches_the_plan(spec) -> None:
    plan = job_accounting(spec)
    confirmatory = plan["tiers"]["confirmatory"]
    pilot = plan["tiers"]["pilot"]
    smoke = plan["tiers"]["smoke"]

    assert plan["discovery"]["episodes"] == 4_576
    assert smoke["primary_evaluation"]["episodes"] == 352
    assert smoke["training_transitions"] == 24_576
    assert pilot["children"] == 9
    assert pilot["training_transitions"] == 921_600
    assert pilot["primary_evaluation"]["episodes"] == 174_720
    assert pilot["snapshot_evaluation"]["checkpoints"] == 27
    assert pilot["snapshot_evaluation"]["episodes"] == 471_744
    assert confirmatory["children"] == 30
    assert confirmatory["training_transitions"] == 3_072_000
    assert confirmatory["primary_evaluation"]["checkpoints"] == 31
    assert confirmatory["primary_evaluation"]["jobs"] == 8_463
    assert confirmatory["primary_evaluation"]["episodes"] == 1_083_264
    assert confirmatory["primary_evaluation"]["max_transitions"] == 541_632_000
    assert confirmatory["snapshot_evaluation"] == {"status": "none"}
    family = plan["confidence_family"]
    assert family["checkpoint_condition_pairs"] == 31 * 273
    assert family["pair_alpha"] == 0.05 / 8_463


def test_seed_plan_lists_concrete_seeds(spec) -> None:
    plan = seed_plan(spec)

    assert plan["evaluation_episode_seeds"]["confirmatory"] == {
        "role": 3,
        "count": 128,
        "first": 3 * 10**12,
        "last": 3 * 10**12 + 127,
    }
    assert plan["training_policy_seeds"]["pilot"] == {
        "0": 300000,
        "1": 300001,
        "2": 300002,
    }
    json.dumps(plan)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("mixture_lambda = 0.5", "mixture_lambda = 1.5", "mixture_lambda"),
        ("n_epochs = 20", "n_epochs = 10", "preserve parent"),
        ("gamma = 0.98", "gamma = 0.99", "preserve parent"),
        ("additional_transitions = 8192", "additional_transitions = 8200", "multiples"),
        ("[25600, 51200, 76800]", "[25600, 25600]", "snapshot"),
        ("[25600, 51200, 76800]", "[25600, 102400]", "snapshot"),
        ("[25600, 51200, 76800]", "[25601]", "snapshot"),
        (
            "initial_theta_deg = [-40, 0, 40]",
            "initial_theta_deg = [0, -40]",
            "increasing",
        ),
        ("length = [0.02, 0.50, 3.00]", "length = [0.0, 0.50, 3.00]", "half-lengths"),
        ("primary_tau = 0.90", "primary_tau = 1.0", "thresholds"),
        ("recovery_tail_steps = 100", "recovery_tail_steps = 50", "metric"),
        ("recovery_rms_angle_deg = 5.0", "recovery_rms_angle_deg = 4.0", "metric"),
        ('metric_version = "episode-summary-v4"', 'metric_version = "x"', "metric"),
        ("theta_threshold_deg = 90", "theta_threshold_deg = 45", "cutoff"),
        ("smoke = 5", "smoke = 4", "distinct"),
        ("replicate_ids = [900]", "replicate_ids = [0]", "unique"),
        ('seed_role = "pilot"', 'seed_role = "training"', "seed role"),
        ("outer_rounds = 1", "outer_rounds = 3", "outer round"),
        ('"intermediate"]', '"high_cost"]', "arms"),
        ('snapshot_evaluation = "none"', 'snapshot_evaluation = "always"', "snapshot"),
    ],
)
def test_loader_rejects_protocol_changes(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_ppo_boundary_study(_edited_config(tmp_path, (old, new)))


def test_prepare_writes_the_complete_protocol(
    tiny_parent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, root = tiny_parent
    _clean_source(monkeypatch)
    output = tmp_path / "study"

    result = prepare_study(config, artifact_root=root, output_dir=output)

    protocol = output / "protocol"
    assert sorted(path.name for path in protocol.iterdir()) == [
        "conditions.json",
        "job-accounting.json",
        "provenance.json",
        "resolved-config.json",
        "seed-plan.json",
        "study-config.toml",
    ]
    assert result["development_record"] is False
    assert tomllib.loads((protocol / "study-config.toml").read_text())
    conditions = json.loads((protocol / "conditions.json").read_text())
    assert conditions["supports"]["discovery"]["condition_count"] == 143
    assert conditions["supports"]["reference"]["weight"]["denominator"] == 273
    resolved = json.loads((protocol / "resolved-config.json").read_text())
    assert resolved["parent_policy_id"] == "tiny_parent"
    provenance = json.loads((protocol / "provenance.json").read_text())
    assert provenance["parent"]["policy_id"] == "tiny_parent"
    assert provenance["source"]["git_commit"] == "abc"
    assert "src/active_eval_gym/ppo_boundary.py" in provenance["source_sha256"]
    assert "uv.lock" in provenance["source_sha256"]
    assert provenance["runtime"]["packages"]["scipy"]
    assert provenance["runtime"]["declared_torch_threads"] == 1


def test_prepare_refuses_to_overwrite_a_protocol(
    tiny_parent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, root = tiny_parent
    _clean_source(monkeypatch)
    output = tmp_path / "study"
    prepare_study(config, artifact_root=root, output_dir=output)
    before = (output / "protocol/provenance.json").read_bytes()

    with pytest.raises(FileExistsError):
        prepare_study(config, artifact_root=root, output_dir=output)

    assert (output / "protocol/provenance.json").read_bytes() == before


def test_prepare_marks_or_refuses_uncommitted_source(
    tiny_parent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, root = tiny_parent
    _clean_source(monkeypatch, dirty=True)

    with pytest.raises(ValueError, match="src/x.py"):
        prepare_study(config, artifact_root=root, output_dir=tmp_path / "refused")
    assert not (tmp_path / "refused").exists()

    result = prepare_study(
        config, artifact_root=root, output_dir=tmp_path / "dev", allow_dirty=True
    )
    provenance = json.loads((tmp_path / "dev/protocol/provenance.json").read_text())
    assert result["development_record"] is True
    assert provenance["development_record"] is True


def test_prepare_rejects_a_substituted_parent(
    tiny_parent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, root = tiny_parent
    _clean_source(monkeypatch)
    text = config.read_text()
    digest = tomllib.loads(text)["parent"]["model_sha256"]
    config.write_text(text.replace(digest, "0" * 64))

    with pytest.raises(ValueError, match="Do not substitute"):
        prepare_study(config, artifact_root=root, output_dir=tmp_path / "study")
    assert not (tmp_path / "study").exists()


def test_prepare_rejects_a_tampered_parent(
    tiny_parent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, root = tiny_parent
    _clean_source(monkeypatch)
    model = root / "tiny_parent/model.zip"
    model.write_bytes(model.read_bytes() + b"x")

    with pytest.raises(ValueError, match="digest mismatch"):
        prepare_study(config, artifact_root=root, output_dir=tmp_path / "study")


def test_source_state_flags_only_paths_that_can_change_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def git(*arguments: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *arguments],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    monkeypatch.setattr(ppo_boundary, "REPOSITORY_ROOT", tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.py").write_text("x = 1\n")
    git("init")
    git("add", ".")
    git("commit", "-m", "base")

    (tmp_path / ".vscode").mkdir()
    (tmp_path / ".vscode/settings.json").write_text("{}")
    (tmp_path / "notes.md").write_text("notes")
    assert ppo_boundary.source_state()["source_dirty"] is False

    (tmp_path / "src/a.py").write_text("x = 2\n")
    (tmp_path / "src/b.py").write_text("y = 1\n")
    state = ppo_boundary.source_state()
    assert state["source_dirty"] is True
    assert state["changed_paths"] == ["src/a.py", "src/b.py"]
