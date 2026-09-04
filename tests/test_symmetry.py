from pathlib import Path

import numpy as np
import pytest

from active_eval_gym.config import load_cartpole_symmetry_suite
from active_eval_gym.envs.factory import make_environment
from active_eval_gym.envs.perturbations import (
    CARTPOLE_FIXED_INITIAL_STATE,
    PerturbationSpec,
)
from active_eval_gym.policies.sb3 import AntisymmetrizedCartPolePPOPolicy
from active_eval_gym.symmetry import _symmetry_summary

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
