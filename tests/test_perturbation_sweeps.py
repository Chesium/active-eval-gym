import hashlib
import json
from math import pi, radians, sqrt
from pathlib import Path

import numpy as np
import pytest

from active_eval_gym.config import load_policy_design, load_sweep_suite
from active_eval_gym.envs.factory import make_environment
from active_eval_gym.envs.perturbations import (
    CARTPOLE_ANGLE_LENGTH,
    MINIGRID_START_POSE,
    PENDULUM_LENGTH,
    PerturbationSpec,
)
from active_eval_gym.envs.specs import (
    IDENTITY_OBSERVATION,
    capture_resolved_environment,
)
from active_eval_gym.metrics import compute_saved_metrics
from active_eval_gym.serialization import read_saved_episode
from active_eval_gym.sweeps import analyze_sweep, expand_sweep

ROOT = Path(__file__).resolve().parents[1]


def test_cartpole_seeded_angle_offset_preserves_other_state() -> None:
    direct = make_environment("CartPole-v1")
    spec = PerturbationSpec(
        CARTPOLE_ANGLE_LENGTH, {"delta_theta_deg": 8, "length": 0.65}
    )
    perturbed = make_environment("CartPole-v1", spec)
    try:
        direct_observation, _ = direct.reset(seed=17)
        observation, _ = perturbed.reset(seed=17)
        np.testing.assert_allclose(
            observation[[0, 1, 3]], direct_observation[[0, 1, 3]]
        )
        assert observation[2] == pytest.approx(
            direct_observation[2] + radians(8), abs=1e-7
        )
        np.testing.assert_allclose(observation, perturbed.unwrapped.state)
        assert perturbed.unwrapped.length == 0.65
        assert perturbed.unwrapped.polemass_length == pytest.approx(0.065)
    finally:
        direct.close()
        perturbed.close()


def test_resolved_specs_capture_changed_primitives_and_reset_semantics() -> None:
    design = load_policy_design(
        ROOT / "configs/policies/cartpole_lqr_nominal_quantized_v1.toml"
    )
    spec = PerturbationSpec(
        CARTPOLE_ANGLE_LENGTH, {"delta_theta_deg": -6, "length": 0.425}
    )
    env = make_environment("CartPole-v1", spec)
    try:
        resolved = capture_resolved_environment(
            env, design.environment, IDENTITY_OBSERVATION, spec
        )
    finally:
        env.close()
    assert resolved.schema_version == 2
    assert resolved.parameters["length"] == 0.425
    assert resolved.derived_parameters["polemass_length"] == pytest.approx(0.0425)
    assert resolved.initial_state_distribution["kind"] == "seeded_nominal_plus_offset"


@pytest.mark.parametrize(
    ("env_id", "spec"),
    [
        (
            "Pendulum-v1",
            PerturbationSpec(
                CARTPOLE_ANGLE_LENGTH,
                {"delta_theta_deg": 0, "length": 0.5},
            ),
        ),
        (
            "Pendulum-v1",
            PerturbationSpec(PENDULUM_LENGTH, {"l": 1.0, "dt": 0.1}),
        ),
        (
            "MiniGrid-Empty-8x8-v0",
            PerturbationSpec(
                MINIGRID_START_POSE,
                {"agent_start_pos": [6, 6], "agent_start_dir": 0},
            ),
        ),
    ],
)
def test_perturbations_reject_wrong_or_unapproved_parameters(
    env_id: str, spec: PerturbationSpec
) -> None:
    with pytest.raises(ValueError):
        make_environment(env_id, spec)


def test_required_sweep_expansion_counts_and_minigrid_cells() -> None:
    cases = [
        (
            "che49_cartpole_angle_length_v1.toml",
            "cartpole_lqr_nominal_quantized_v1.toml",
            45,
        ),
        (
            "che49_pendulum_length_v1.toml",
            "pendulum_sac_nominal_v1.toml",
            5,
        ),
        (
            "che49_minigrid_start_pose_v1.toml",
            "minigrid_empty8x8_ppo_partial_image_v1.toml",
            140,
        ),
    ]
    for suite_name, policy_name, expected in cases:
        suite = load_sweep_suite(ROOT / "configs/eval" / suite_name)
        nominal = load_policy_design(
            ROOT / "configs/policies" / policy_name
        ).environment
        conditions = expand_sweep(suite, nominal)
        assert len(conditions) == expected
        assert len({condition.condition_id for condition in conditions}) == expected

    positions = {
        tuple(condition.perturbation.parameters["agent_start_pos"])
        for condition in conditions
    }
    assert len(positions) == 35
    assert (6, 6) not in positions


def test_saved_cartpole_metrics_and_digest_validation(tmp_path: Path) -> None:
    episode_dir = _saved_episode(
        tmp_path,
        "CartPole-v1",
        reset_state={
            "cart_position": 0.0,
            "cart_velocity": 0.0,
            "pole_angle": 0.0,
            "pole_angular_velocity": 0.0,
        },
        transitions=[
            _transition(
                0,
                1.0,
                {
                    "cart_position": 1.0,
                    "cart_velocity": 0.0,
                    "pole_angle": 0.1,
                    "pole_angular_velocity": 0.0,
                },
            ),
            _transition(
                1,
                1.0,
                {
                    "cart_position": 0.0,
                    "cart_velocity": 0.0,
                    "pole_angle": -0.1,
                    "pole_angular_velocity": 0.0,
                },
                truncated=True,
            ),
        ],
    )
    metrics = compute_saved_metrics(read_saved_episode(episode_dir))
    assert metrics.task_success
    assert metrics.environment_metrics["action_switch_rate"] == 1.0
    assert metrics.environment_metrics["rms_cart_position"] == pytest.approx(
        sqrt(1 / 3)
    )
    assert metrics.environment_metrics["max_abs_pole_angle_radians"] == 0.1

    (episode_dir / "trajectory.jsonl").write_text("tampered\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        read_saved_episode(episode_dir)


def test_pendulum_and_minigrid_metric_formulas(tmp_path: Path) -> None:
    pendulum_dir = _saved_episode(
        tmp_path / "pendulum",
        "Pendulum-v1",
        reset_state={"angle": pi, "angular_velocity": 0.0},
        transitions=[
            _transition(
                [3.0],
                -1.0,
                {"angle": -pi, "angular_velocity": 2.0},
                truncated=True,
            )
        ],
        parameters={"max_torque": 2.0},
    )
    pendulum = compute_saved_metrics(read_saved_episode(pendulum_dir))
    assert pendulum.environment_metrics["rms_angle_error_radians"] == pytest.approx(pi)
    assert pendulum.environment_metrics["rms_torque"] == 2.0

    minigrid_dir = _saved_episode(
        tmp_path / "minigrid",
        "MiniGrid-Empty-8x8-v0",
        reset_state={"agent_position": [1, 1], "agent_direction": 0},
        transitions=[
            _transition(
                0, 0.0, {"agent_position": [1, 1], "agent_direction": 3}
            ),
            _transition(
                2,
                1.0,
                {"agent_position": [6, 6], "agent_direction": 3},
                terminated=True,
            ),
        ],
        parameters={"width": 8, "height": 8},
    )
    minigrid = compute_saved_metrics(read_saved_episode(minigrid_dir))
    assert minigrid.environment_metrics["turn_count"] == 1
    assert minigrid.environment_metrics["path_efficiency"] == 1.0
    assert minigrid.environment_metrics["action_counts"]["forward"] == 1


def test_analysis_is_offline_versioned_and_non_overwriting(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation"
    condition_id = "theta-p0_length-p0p5"
    policy_id = "policy"
    _saved_episode(
        evaluation / "episodes" / policy_id / condition_id / "seed-000",
        "CartPole-v1",
        reset_state={
            "cart_position": 0.0,
            "cart_velocity": 0.0,
            "pole_angle": 0.0,
            "pole_angular_velocity": 0.0,
        },
        transitions=[
            _transition(
                0,
                1.0,
                {
                    "cart_position": 0.0,
                    "cart_velocity": 0.0,
                    "pole_angle": 0.0,
                    "pole_angular_velocity": 0.0,
                },
                truncated=True,
            )
        ],
    )
    suite = {
        "suite": {
            "suite_id": "tiny",
            "environment_id": "CartPole-v1",
            "policy_ids": [policy_id],
            "seeds": [0],
        },
        "conditions": [
            {
                "condition_id": condition_id,
                "perturbation": {
                    "parameters": {"delta_theta_deg": 0, "length": 0.5}
                },
            }
        ],
    }
    evaluation.mkdir(parents=True, exist_ok=True)
    (evaluation / "suite.json").write_text(json.dumps(suite))
    summary = analyze_sweep(evaluation)
    assert summary["episode_count"] == 1
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        analyze_sweep(evaluation)


def _saved_episode(
    root: Path,
    env_id: str,
    *,
    reset_state: dict,
    transitions: list[dict],
    parameters: dict | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 3,
        "resolved_environment": {
            "environment_id": env_id,
            "parameters": parameters or {},
        },
    }
    reset = {
        "type": "reset",
        "observation": [],
        "environment_state": reset_state,
        "info": {},
    }
    trajectory = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in [reset, *transitions]
    ).encode()
    digest = hashlib.sha256(trajectory).hexdigest()
    (root / "metadata.json").write_text(json.dumps(metadata))
    (root / "trajectory.jsonl").write_bytes(trajectory)
    (root / "trajectory.sha256").write_text(digest + "\n")
    return root


def _transition(
    action: int | list[float],
    reward: float,
    state: dict,
    *,
    terminated: bool = False,
    truncated: bool = False,
) -> dict:
    return {
        "type": "transition",
        "action": action,
        "policy_diagnostics": {},
        "reward": reward,
        "observation": [],
        "environment_state": state,
        "terminated": terminated,
        "truncated": truncated,
        "info": {},
    }
