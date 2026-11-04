"""Scientific guardrails for discovery-driven multiround training."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from active_eval_gym.config import load_ppo_boundary_study
from active_eval_gym.policies.artifacts import load_policy_artifact
from active_eval_gym.ppo_adaptive_curriculum import (
    AdaptiveCapture,
    assert_rng_unchanged,
    evaluation_jobs,
    load_adaptive_study,
    rng_state,
    select_frontier,
)
from active_eval_gym.ppo_boundary import support_conditions
from active_eval_gym.ppo_boundary_training import (
    make_training_environment,
    migrate_parent,
    state_sha256,
)
from active_eval_gym.ppo_transition_sampling import InverseLengthSampler
from tests.helpers import FRONTIER_RULE_EDIT, ROOT, tiny_parent_config
from tests.test_ppo_transition_sampling import calibrated as calibrated

CONFIG = ROOT / "configs/experiments/cartpole_ppo_boundary_v5_adaptive.toml"


def discovery(base, counts):
    return {
        "support": "discovery",
        "cells": [
            {"condition_id": c.condition_id, "episode_count": 32, "recovery_count": n}
            for c, n in zip(support_conditions(base, "discovery"), counts, strict=True)
        ],
    }


def test_declared_design_and_historical_comparison():
    settings, base = load_adaptive_study(CONFIG)
    assert settings.rounds == 10
    assert settings.transitions_per_round == 25600
    assert settings.total_transitions == 256000
    assert settings.replicate_ids == (0, 1, 2)
    assert settings.empty_frontier_action == "stop-seed"
    assert base.continuation.learning_rate == 3e-5
    assert base.curriculum.mixture_lambda == 0.5
    assert base.sampling is not None


def test_empty_acceptance_stops_instead_of_uniform_fallback():
    _, base = load_adaptive_study(CONFIG)
    target = select_frontier(base, discovery(base, [0] * 143))
    assert target["stop_reason"] == "no-accepted-conditions"
    assert not target["frontier_mask"].any()


def test_frontier_recomputed_from_current_discovery_only():
    _, base = load_adaptive_study(CONFIG)
    counts = np.zeros(143, int)
    counts[6 * 13 + 7] = 32
    old = select_frontier(base, discovery(base, counts))
    counts[6 * 13 + 8] = 32
    new = select_frontier(base, discovery(base, counts))
    assert old["stop_reason"] is None
    assert old["frontier_mask"].sum() == 4
    assert new["frontier_mask"].sum() == 6
    assert not np.array_equal(
        old["target_transition_shares"], new["target_transition_shares"]
    )
    wrong = discovery(base, counts)
    wrong["support"] = "reference"
    with pytest.raises(ValueError, match="discovery support"):
        select_frontier(base, wrong)


@pytest.fixture(scope="module")
def parent_artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("adaptive-parent")
    config, artifacts = tiny_parent_config(root, FRONTIER_RULE_EDIT)
    return artifacts / "tiny_parent", load_ppo_boundary_study(config)


def test_targeting_reset_seeds_fresh_reference_seeds_fixed(tmp_path, parent_artifact):
    settings, base = load_adaptive_study(CONFIG)
    parent, tiny_base = parent_artifact
    base = replace(base, parent_policy_id=tiny_base.parent_policy_id)
    _, first = evaluation_jobs(
        settings, base, tmp_path, parent, support="discovery", round_index=0
    )
    _, second = evaluation_jobs(
        settings, base, tmp_path, parent, support="discovery", round_index=1
    )
    _, reference = evaluation_jobs(
        settings, base, tmp_path, parent, support="reference", round_index=1
    )
    assert set(first[0].header["episode_seeds"]).isdisjoint(
        second[0].header["episode_seeds"]
    )
    assert reference[0].header["episode_seeds"] == list(
        range(4_000_000_000_000, 4_000_000_000_064)
    )


def test_rng_guard_checks_numpy_state_and_torch():
    before = rng_state()
    assert_rng_unchanged(before)
    state = np.random.get_state()
    changed = state[1].copy()
    changed[len(changed) // 2] ^= 1
    np.random.set_state((state[0], changed, *state[2:]))
    with pytest.raises(RuntimeError, match="training RNG"):
        assert_rng_unchanged(before)
    np.random.set_state(state)
    before = rng_state()
    torch.rand(1)
    with pytest.raises(RuntimeError, match="training RNG"):
        assert_rng_unchanged(before)


def train_two_rounds(base, parent, *, replace_target):
    _, metadata = load_policy_artifact(parent, require_frozen=True)
    q = np.full(143, 1 / 143)
    settings = replace(base.sampling, refresh_rollouts=1)
    sampler = InverseLengthSampler(q, np.full(143, 100.0), settings)
    env = make_training_environment(base, metadata, q, replicate_id=0, sampling=sampler)
    try:
        model, _ = migrate_parent(parent / "model.zip", env, base.continuation, 300000)
        capture = AdaptiveCapture(512, sampler)
        model.learn(256, callback=capture, reset_num_timesteps=False)
        prior_state = model._last_obs.copy()
        prior_episode_indices = [w.get_wrapper_attr("_next_episode") for w in env.envs]
        prior_optimizer = state_sha256(model.policy.optimizer.state_dict())
        if replace_target:
            target = np.full(143, 0.5 / 143)
            target[0] += 0.5
            sampler.set_target_shares(target)
        else:
            sampler.set_target_shares(q)
        # Staging a target must not reset the environment or optimizer.
        np.testing.assert_array_equal(model._last_obs, prior_state)
        assert [
            w.get_wrapper_attr("_next_episode") for w in env.envs
        ] == prior_episode_indices
        assert state_sha256(model.policy.optimizer.state_dict()) == prior_optimizer
        model.learn(256, callback=capture, reset_num_timesteps=False)
        return capture.tables(), state_sha256(model.policy.state_dict())
    finally:
        env.close()


def test_retarget_schedule_keeps_old_episode_reset_law(parent_artifact):
    _, base = load_adaptive_study(CONFIG)
    torch.set_num_threads(1)
    parent, _ = parent_artifact
    raw, _ = train_two_rounds(base, parent, replace_target=True)
    assert raw["sampling_transition_count"].tolist() == [0, 256]
    assert raw["sampling_target_transition_shares"][1, 0] > 0.5
    np.testing.assert_array_equal(
        raw["reset_selection_probability"],
        raw["sampling_probabilities"][
            raw["reset_sampling_revision"], raw["reset_condition_index"]
        ],
    )
    # The recorded old resets remain under revision0, including ongoing episodes.
    assert np.any(raw["reset_sampling_revision"] == 0)
    assert len(raw["state"]) == 512


def test_live_seed_discovery_preserves_rng_and_replays_completed_rounds(
    calibrated, tmp_path
):
    import json

    from active_eval_gym.ppo_adaptive_curriculum import (
        prepare_adaptive,
        read,
        train_adaptive_seed,
    )
    from active_eval_gym.serialization import write_json_new

    base_config, artifacts, baseline = calibrated
    base = load_ppo_boundary_study(base_config)
    config = tmp_path / "adaptive.toml"
    config.write_text(
        'schema_version = 1\nstudy_id = "tiny-adaptive"\n'
        f'base_config = "{base_config}"\n'
        f'baseline_study = "{baseline}"\n'
        f'historical_summary = "{baseline / "unused.json"}"\n'
        "rounds = 2\ntransitions_per_round = 256\nreplicate_ids = [0]\n"
        'empty_frontier_action = "stop-seed"\n'
        "discovery_workers_per_seed = 2\nreference_workers = 2\n"
    )
    study = tmp_path / "adaptive"
    prepare_adaptive(config, study, artifacts)
    # Synthetic discovery makes the tiny test enter training independently of
    # this deliberately weak parent's performance. Subsequent discovery is real.
    conditions = support_conditions(base, "discovery")
    initial = {
        "support": "discovery",
        "cells": [
            {
                "condition_id": condition.condition_id,
                "episode_count": base.curriculum.discovery_episodes_per_condition,
                "recovery_count": base.curriculum.discovery_episodes_per_condition
                if i == 0
                else 0,
            }
            for i, condition in enumerate(conditions)
        ],
    }
    (study / "initial-discovery").mkdir()
    (study / "training").mkdir()
    write_json_new(study / "initial-discovery/summary.json", initial)
    result = train_adaptive_seed(str(config), str(study), 0)
    assert result["completed_transitions"] in (256, 512)
    assert result["evaluation_rng_checks_passed"]
    assert result["completed_transitions"] + result["unrun_transitions"] == 512
    saved = (study / "training/r000.json").read_bytes()
    replayed = train_adaptive_seed(str(config), str(study), 0)
    assert replayed == result
    assert (study / "training/r000.json").read_bytes() == saved
    for record in result["rounds"]:
        summary = read(study / record["checkpoint"] / "training-summary.json")
        assert summary["completed_rounds"] == record["round_index"] + 1
        assert summary["target_sha256"]
        assert summary["additional_transitions"] == record["transitions"]
    # The only evaluation during seed training must use discovery role1,
    # with fresh round1 nuisance resets. No reference directory may exist.
    plans = list((study / "evaluations/discovery").glob("*/evaluation-plan.json"))
    assert len(plans) == 1
    plan = json.loads(plans[0].read_text())
    assert plan["jobs"][0]["header"]["seed_role"] == 1
    assert plan["jobs"][0]["header"]["seed_round"] == 1
    assert not (study / "evaluations/reference").exists()
