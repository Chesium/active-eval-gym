"""Duration balancing, raw provenance, and reproducible adaptive PPO collection."""

import json
import shutil
from dataclasses import replace

import numpy as np
import pytest

from active_eval_gym.config import load_ppo_boundary_study
from active_eval_gym.ppo_boundary import child_artifact_dir, prepare_study
from active_eval_gym.ppo_boundary_analysis import analyze_campaign
from active_eval_gym.ppo_boundary_evaluation import collect_campaign
from active_eval_gym.ppo_boundary_targets import build_distributions
from active_eval_gym.ppo_boundary_training import (
    load_distributions,
    plan_training,
    run_training_job,
)
from active_eval_gym.ppo_transition_sampling import (
    InverseLengthSampler,
    calibrate_lengths,
    load_calibration,
)
from active_eval_gym.serialization import read_transition_shard
from tests.helpers import FRONTIER_RULE_EDIT, ROOT, SMALL_STUDY_GRID, tiny_parent_config
from tests.test_ppo_boundary_training import SMALL_BUDGETS

V4 = ROOT / "configs/experiments/cartpole_ppo_boundary_v4.toml"
SETTINGS = load_ppo_boundary_study(V4).sampling


def test_v4_changes_only_sampling_and_study_identity() -> None:
    v3 = load_ppo_boundary_study(
        ROOT / "configs/experiments/cartpole_ppo_boundary_v3.toml"
    )
    v4 = load_ppo_boundary_study(V4)
    assert replace(v4, study_id=v3.study_id, sampling=None) == v3


def test_inverse_lengths_balance_expected_transition_counts() -> None:
    lengths = np.array([500.0, 42.0, 130.0])
    target = np.array([0.2, 0.3, 0.5])
    sampler = InverseLengthSampler(target, lengths, SETTINGS)
    exposure = sampler.probabilities * lengths
    np.testing.assert_allclose(exposure / exposure.sum(), target)
    rng = np.random.default_rng(6180)
    selected = np.searchsorted(sampler.cdf, rng.random(100_000), side="right")
    observed = np.bincount(selected, minlength=3) * lengths
    np.testing.assert_allclose(observed / observed.sum(), target, atol=0.01)


def test_estimates_update_without_changing_reset_law_until_refresh() -> None:
    sampler = InverseLengthSampler(np.array([0.5, 0.5]), np.array([500, 50]), SETTINGS)
    old = sampler.probabilities.copy()
    sampler.observe_completion(0, 100)
    np.testing.assert_array_equal(sampler.probabilities, old)
    assert sampler.length_estimates[0] == 460
    sampler.refresh()
    assert sampler.probabilities[0] > old[0]
    assert sampler.revision == 1
    assert sampler.completed_episodes.tolist() == [1, 0]


def test_frontier_retarget_preserves_lengths_and_stages_next_reset_law() -> None:
    sampler = InverseLengthSampler(np.array([0.5, 0.5]), np.array([500, 50]), SETTINGS)
    sampler.observe_completion(0, 100)
    old_probabilities = sampler.probabilities.copy()
    sampler.set_target_shares(np.array([0.2, 0.8]))
    np.testing.assert_array_equal(sampler.length_estimates, [460, 50])
    np.testing.assert_array_equal(sampler.completed_episodes, [1, 0])
    np.testing.assert_array_equal(sampler.probabilities, old_probabilities)
    assert sampler.revision == 0
    sampler.refresh()
    exposure = sampler.probabilities * sampler.length_estimates
    np.testing.assert_allclose(exposure / exposure.sum(), [0.2, 0.8])
    assert sampler.revision == 1


@pytest.mark.parametrize("target", [[1], [-0.1, 1.1], [0.2, 0.2], [float("nan"), 0]])
def test_invalid_frontier_retarget_is_atomic(target) -> None:
    sampler = InverseLengthSampler(np.array([0.5, 0.5]), np.array([500, 50]), SETTINGS)
    with pytest.raises(ValueError, match="replacement target"):
        sampler.set_target_shares(np.array(target))
    np.testing.assert_array_equal(sampler.target_shares, [0.5, 0.5])


@pytest.mark.parametrize("length", [0, 501])
def test_invalid_completed_lengths_are_refused(length: int) -> None:
    sampler = InverseLengthSampler(np.array([1.0]), np.array([100.0]), SETTINGS)
    with pytest.raises(ValueError, match="complete, uncensored"):
        sampler.observe_completion(0, length)


@pytest.fixture(scope="module")
def calibrated(tmp_path_factory):
    root = tmp_path_factory.mktemp("transition-sampling")
    config, artifacts = tiny_parent_config(
        root, *SMALL_STUDY_GRID, *SMALL_BUDGETS, FRONTIER_RULE_EDIT
    )
    sampling_text = V4.read_text().split("[sampling]\n", 1)[1].split("[seeds]", 1)[0]
    config.write_text(
        config.read_text()
        + "\n[sampling]\n"
        + sampling_text.replace("= 8\n", "= 2\n").replace(
            "refresh_rollouts = 10", "refresh_rollouts = 1"
        )
    )
    study = root / "study"
    prepare_study(config, artifact_root=artifacts, output_dir=study, allow_dirty=True)
    collect_campaign(
        config,
        study_dir=study,
        campaign="discovery",
        artifact_root=artifacts,
        allow_dirty=True,
    )
    analyze_campaign(config, study_dir=study, campaign="discovery")
    build_distributions(config, study_dir=study, allow_dirty=True)
    calibrate_lengths(
        config, study_dir=study, artifact_root=artifacts, allow_dirty=True
    )
    return config, artifacts, study


def test_calibration_is_stochastic_disjoint_and_has_no_optimizer_updates(calibrated):
    config, _, study = calibrated
    spec = load_ppo_boundary_study(config)
    provenance = json.loads((study / "protocol/provenance.json").read_text())
    result, _ = load_calibration(spec, provenance, study)
    assert result["optimizer_updates"] == 0
    assert result["episodes"] == 8
    for shard in result["shards"]:
        assert all(seed // 10**12 == 6 for seed in shard["episode_seeds"])
        assert shard["action_seed"] >= 730000
    plan = json.loads((study / "sampling-calibration/plan.json").read_text())
    assert plan["deterministic"] is False


def test_calibration_resume_preserves_raw_and_summary(calibrated):
    config, artifacts, study = calibrated
    path = study / "sampling-calibration/summary.json"
    before = path.read_bytes()
    calibrate_lengths(
        config, study_dir=study, artifact_root=artifacts, allow_dirty=True
    )
    assert path.read_bytes() == before


def test_training_refuses_tampered_calibration_raw(calibrated, tmp_path):
    config, _, base = calibrated
    study = tmp_path / "study"
    shutil.copytree(base, study)
    shard = next((study / "sampling-calibration/shards").iterdir())
    shard.write_bytes(shard.read_bytes() + b"tampered")
    spec = load_ppo_boundary_study(config)
    provenance = json.loads((study / "protocol/provenance.json").read_text())
    with pytest.raises(ValueError, match="Calibration raw digest mismatch"):
        load_calibration(spec, provenance, study)


def test_training_refuses_episode_weights_in_transition_study(calibrated, tmp_path):
    config, _, base = calibrated
    study = tmp_path / "study"
    shutil.copytree(base, study)
    path = study / "distributions/distributions.json"
    record = json.loads(path.read_text())
    record["sampling_unit"] = "episode"
    path.write_text(json.dumps(record))
    spec = load_ppo_boundary_study(config)
    provenance = json.loads((study / "protocol/provenance.json").read_text())
    with pytest.raises(ValueError, match="sampling unit"):
        load_distributions(spec, provenance, study, allow_dirty=True)


def test_adaptive_training_is_reproducible_across_snapshot_chunking(
    calibrated, tmp_path
):
    config, artifacts, base = calibrated
    spec = load_ppo_boundary_study(config)
    results = []
    for name, snapshots in (("chunked", None), ("single", ())):
        study = tmp_path / name
        shutil.copytree(base, study)
        job = plan_training(
            spec,
            "pilot",
            config_path=config,
            study_dir=study,
            artifact_root=artifacts,
            development_record=True,
            source={"test": True},
        )[0]
        run_training_job(job, snapshot_transitions=snapshots)
        child = child_artifact_dir(study, "pilot", job.job_id)
        summary = json.loads((child / "training-summary.json").read_text())
        _, raw = read_transition_shard(study / summary["transitions_file"]["path"])
        np.testing.assert_array_equal(
            raw["sampling_transition_count"], [0, 256, 512, 768]
        )
        np.testing.assert_allclose(raw["sampling_probabilities"].sum(axis=1), 1)
        expected = raw["sampling_probabilities"][
            raw["reset_sampling_revision"], raw["reset_condition_index"]
        ]
        np.testing.assert_array_equal(raw["reset_selection_probability"], expected)
        assert summary["sampling"]["revision"] == 3
        assert summary["additional_transitions"] == 1024
        assert summary["budget_accounting"]["rollouts"] == 4
        np.testing.assert_array_equal(
            summary["sampling"]["completed_episodes"],
            summary["realized_sampling"]["episodes_completed"],
        )
        results.append(summary)
    assert results[0]["final_policy_sha256"] == results[1]["final_policy_sha256"]
    assert results[0]["final_optimizer_sha256"] == results[1]["final_optimizer_sha256"]
    assert (
        results[0]["transitions_file"]["content_sha256"]
        == results[1]["transitions_file"]["content_sha256"]
    )
