from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from active_eval_gym.config import load_cartpole_symmetry_suite, load_sweep_suite
from active_eval_gym.envs.factory import make_environment
from active_eval_gym.envs.perturbations import (
    CARTPOLE_FIXED_INITIAL_STATE,
    PerturbationSpec,
)
from active_eval_gym.plotting import _plot_cartpole, _plot_cartpole_one_dimensional
from active_eval_gym.policies.sb3 import (
    AntisymmetrizedCartPolePPOPolicy,
    SB3Policy,
    derive_antisymmetrized_cartpole_ppo,
)
from active_eval_gym.symmetry import _symmetry_summary
from tests.helpers import rollout_provenance

ROOT = Path(__file__).resolve().parents[1]


class LinearDiagnosticPolicy:
    def __init__(self, bias: float) -> None:
        self.bias = bias

    def cartpole_actor_critic(
        self, observations: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        states = np.asarray(observations, dtype=np.float32)
        margin = self.bias + states @ np.array([1.0, 2.0, 3.0, 4.0])
        logits = np.column_stack((-0.5 * margin, 0.5 * margin))
        probability_right = 1.0 / (1.0 + np.exp(-margin))
        probabilities = np.column_stack((1.0 - probability_right, probability_right))
        values = np.sum(states * states, axis=1)
        return logits, probabilities, values


def test_symmetry_suite_config_is_explicit() -> None:
    suite = load_cartpole_symmetry_suite(
        ROOT / "configs/eval/che49_cartpole_symmetry_v1.toml"
    )

    assert suite.policy_id == "cartpole_ppo_nominal_v1"
    assert suite.derived_policy_id.endswith("antisymmetrized_v1")
    assert suite.angle_magnitudes_deg == (2.0, 4.0, 6.0, 8.0)
    assert suite.signed_angles_deg[0] == -8.0
    assert suite.signed_angles_deg[-1] == 8.0
    assert len(suite.seeds) == 20


def test_three_policy_sweep_declares_hash_bound_derivation() -> None:
    for filename in (
        "che49_cartpole_angle_length_v2.toml",
        "che49_cartpole_pole_angle_noise_v2.toml",
        "che49_cartpole_action_delay_v2.toml",
        "che49_cartpole_action_dropout_v2.toml",
    ):
        suite = load_sweep_suite(ROOT / "configs/eval" / filename)

        assert len(suite.policy_ids) == 3
        derived = suite.derived_policies[suite.policy_ids[-1]]
        assert derived["kind"] == "antisymmetrized-cartpole-ppo-v1"
        assert derived["source_policy_id"] == "cartpole_ppo_nominal_v1"
        assert len(derived["source_model_sha256"]) == 64


def test_fixed_initial_state_perturbation_preserves_cartpole_symmetry() -> None:
    state = np.array([0.2, -0.3, 0.1, -0.4])
    environments = []
    observations = []
    try:
        for initial_state, action in ((state, 1), (-state, 0)):
            env = make_environment(
                "CartPole-v1",
                PerturbationSpec(
                    CARTPOLE_FIXED_INITIAL_STATE,
                    {"initial_state": initial_state.tolist(), "length": 0.65},
                ),
            )
            environments.append(env)
            observation, _ = env.reset(seed=7)
            np.testing.assert_allclose(observation, initial_state)
            next_observation, *_ = env.step(action)
            observations.append(next_observation)
            assert env.unwrapped.length == 0.65
        np.testing.assert_allclose(observations[0], -observations[1], atol=1e-7)
    finally:
        for environment in environments:
            environment.close()


def test_antisymmetrized_policy_removes_even_logit_component() -> None:
    policy = AntisymmetrizedCartPolePPOPolicy(LinearDiagnosticPolicy(bias=5.0))
    state = np.array([0.1, -0.2, 0.03, 0.4], dtype=np.float32)

    action = policy.act(state)
    reflected_action = policy.act(-state)

    assert action.action == 1 - reflected_action.action
    assert (
        action.diagnostics["antisymmetrized_logit_margin"]
        == -reflected_action.diagnostics["antisymmetrized_logit_margin"]
    )
    assert action.diagnostics["action_1_probability"] == pytest.approx(
        1.0 - reflected_action.diagnostics["action_1_probability"]
    )


def test_symmetry_summary_separates_actor_and_value_checks() -> None:
    states = np.array(
        [[0.1, 0.2, -0.03, 0.4], [-0.5, 0.4, 0.02, -0.1]],
        dtype=np.float32,
    )
    symmetric = _symmetry_summary(LinearDiagnosticPolicy(bias=0.0), states)
    biased = _symmetry_summary(LinearDiagnosticPolicy(bias=2.0), states)

    assert symmetric["mean_absolute_probability_error"] < 1e-7
    assert symmetric["mean_absolute_logit_margin_error"] < 1e-7
    assert symmetric["mean_absolute_value_error"] == 0.0
    assert biased["mean_absolute_probability_error"] > 0.1
    assert biased["mean_absolute_logit_margin_error"] == 4.0


def test_derived_policy_metadata_preserves_source_hash() -> None:
    env = make_environment("CartPole-v1")
    try:
        metadata, _, _ = rollout_provenance(env, "CartPole-v1")
    finally:
        env.close()
    source_design = replace(
        metadata.design_spec,
        policy_id="source-ppo",
        design_id="source-ppo-design",
        algorithm="PPO",
    )
    source_metadata = replace(
        metadata,
        policy_id="source-ppo",
        design_spec=source_design,
        model_sha256="a" * 64,
    )

    _, derived = derive_antisymmetrized_cartpole_ppo(
        SB3Policy(object()),
        source_metadata,
        derived_policy_id="derived-ppo",
    )

    assert derived.policy_id == "derived-ppo"
    assert derived.model_sha256 == source_metadata.model_sha256
    assert derived.design_spec.hyperparameters["weights_changed"] is False


def test_cartpole_plot_supports_three_policies(tmp_path: Path) -> None:
    policies = ["lqr", "ppo", "ppo_antisymmetrized"]
    aggregate = {
        "success_rate": 1.0,
        "episode_length": {"mean": 500.0, "standard_deviation": 0.0},
        "environment_metrics": {
            name: {"mean": 0.1, "standard_deviation": 0.0}
            for name in (
                "rms_pole_angle_radians",
                "max_abs_pole_angle_radians",
                "rms_cart_position",
                "action_switch_rate",
            )
        },
    }
    summary = {
        "policy_ids": policies,
        "conditions": [
            {
                "parameters": {"delta_theta_deg": 0, "length": 0.5},
                "policies": {policy: aggregate for policy in policies},
            }
        ],
    }

    paths = _plot_cartpole(summary, tmp_path)

    assert [path.name for path in paths] == [
        "che49_cartpole_success_surface_three_policy.png",
        "che49_cartpole_nominal_slices_three_policy.png",
    ]
    assert all(path.is_file() for path in paths)


def test_one_dimensional_plot_versions_three_policy_filename(tmp_path: Path) -> None:
    policies = ["lqr", "ppo", "ppo_antisymmetrized"]
    aggregate = {
        "success_rate": 1.0,
        "episode_length": {"mean": 500.0, "standard_deviation": 0.0},
        "environment_metrics": {
            name: {"mean": 0.1, "standard_deviation": 0.0}
            for name in (
                "rms_pole_angle_radians",
                "max_abs_pole_angle_radians",
                "rms_cart_position",
                "action_switch_rate",
            )
        },
    }
    summary = {
        "policy_ids": policies,
        "conditions": [
            {
                "parameters": {"delay_steps": delay},
                "policies": {policy: aggregate for policy in policies},
            }
            for delay in range(5)
        ],
    }

    paths = _plot_cartpole_one_dimensional(
        summary, tmp_path, "cartpole-action-delay-v1"
    )

    assert [path.name for path in paths] == [
        "che49_cartpole_action_delay_sweep_three_policy.png"
    ]
