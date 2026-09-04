import hashlib
import json
from math import pi, radians, sqrt
from pathlib import Path

import numpy as np
import pytest

from active_eval_gym.config import load_policy_design, load_sweep_suite
from active_eval_gym.envs.factory import make_environment
from active_eval_gym.envs.perturbations import (
    ACTION_DROPOUT_STREAM_ID,
    CARTPOLE_ACTION_DELAY,
    CARTPOLE_ACTION_DROPOUT,
    CARTPOLE_ANGLE_LENGTH,
    CARTPOLE_FORCE_MAGNITUDE,
    CARTPOLE_MASS,
    CARTPOLE_POLE_ANGLE_NOISE,
    MINIGRID_START_POSE,
    PENDULUM_LENGTH,
    PERTURBATION_INFO_KEY,
    PerturbationSpec,
)
from active_eval_gym.envs.specs import (
    IDENTITY_OBSERVATION,
    capture_resolved_environment,
)
from active_eval_gym.metrics import INTERVENTION_METRIC_VERSION, compute_saved_metrics
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


def test_cartpole_mass_and_force_resolve_only_declared_primitives() -> None:
    design = load_policy_design(
        ROOT / "configs/policies/cartpole_lqr_nominal_quantized_v1.toml"
    )
    mass_spec = PerturbationSpec(CARTPOLE_MASS, {"masspole": 0.15})
    mass_env = make_environment("CartPole-v1", mass_spec)
    force_spec = PerturbationSpec(CARTPOLE_FORCE_MAGNITUDE, {"force_mag": 5})
    force_env = make_environment("CartPole-v1", force_spec)
    try:
        mass = capture_resolved_environment(
            mass_env, design.environment, IDENTITY_OBSERVATION, mass_spec
        )
        force = capture_resolved_environment(
            force_env, design.environment, IDENTITY_OBSERVATION, force_spec
        )
    finally:
        mass_env.close()
        force_env.close()
    assert mass.parameters["masspole"] == 0.15
    assert mass.derived_parameters["total_mass"] == pytest.approx(1.15)
    assert mass.derived_parameters["polemass_length"] == pytest.approx(0.075)
    assert mass.parameters["length"] == 0.5
    assert force.parameters["force_mag"] == 5
    assert force.parameters["masspole"] == 0.1


def test_pole_angle_noise_is_seeded_observation_only() -> None:
    spec = PerturbationSpec(
        CARTPOLE_POLE_ANGLE_NOISE, {"pole_angle_noise_std_deg": 2}
    )
    direct = make_environment("CartPole-v1")
    first = make_environment("CartPole-v1", spec)
    second = make_environment("CartPole-v1", spec)
    try:
        direct_observation, _ = direct.reset(seed=23)
        first_observation, first_info = first.reset(seed=23)
        second_observation, second_info = second.reset(seed=23)
        np.testing.assert_allclose(first.unwrapped.state, direct.unwrapped.state)
        np.testing.assert_array_equal(first_observation, second_observation)
        np.testing.assert_allclose(
            first_observation[[0, 1, 3]], direct_observation[[0, 1, 3]]
        )
        assert first_observation[2] != direct_observation[2]
        assert first.observation_space.contains(first_observation)
        assert (
            first_info[PERTURBATION_INFO_KEY]["pole_angle_noise_radians"]
            == second_info[PERTURBATION_INFO_KEY]["pole_angle_noise_radians"]
        )
    finally:
        direct.close()
        first.close()
        second.close()


def test_action_delay_and_dropout_first_action_semantics() -> None:
    delay = make_environment(
        "CartPole-v1", PerturbationSpec(CARTPOLE_ACTION_DELAY, {"delay_steps": 1})
    )
    dropout = make_environment(
        "CartPole-v1",
        PerturbationSpec(CARTPOLE_ACTION_DROPOUT, {"dropout_probability": 1}),
    )
    requests = [0, 1, 1, 0]
    try:
        _, delay_reset_info = delay.reset(seed=5)
        delayed = [
            delay.step(action)[-1][PERTURBATION_INFO_KEY]["environment_action"]
            for action in requests
        ]
        _, dropout_reset_info = dropout.reset(seed=5)
        dropped = [
            dropout.step(action)[-1][PERTURBATION_INFO_KEY]
            for action in requests
        ]
    finally:
        delay.close()
        dropout.close()
    assert delay_reset_info[PERTURBATION_INFO_KEY] == {
        "kind": "action_delay",
        "delay_steps": 1,
        "first_request_passthrough": True,
    }
    assert dropout_reset_info[PERTURBATION_INFO_KEY] == {
        "kind": "action_dropout",
        "dropout_probability": 1.0,
        "first_request_passthrough": True,
        "rng": "numpy.default_rng",
        "stream_id": ACTION_DROPOUT_STREAM_ID,
    }
    assert delayed == [0, 0, 1, 1]
    assert [item["environment_action"] for item in dropped] == [0, 0, 0, 0]
    assert [item["dropout_event"] for item in dropped] == [False, True, True, True]


def test_dropout_random_stream_replays_independently_of_actions() -> None:
    spec = PerturbationSpec(
        CARTPOLE_ACTION_DROPOUT, {"dropout_probability": 0.2}
    )
    first = make_environment("CartPole-v1", spec)
    second = make_environment("CartPole-v1", spec)
    try:
        first.reset(seed=91)
        second.reset(seed=91)
        first_diagnostics = [
            first.step(action)[-1][PERTURBATION_INFO_KEY]
            for action in [0, 1, 0, 1]
        ]
        second_diagnostics = [
            second.step(action)[-1][PERTURBATION_INFO_KEY]
            for action in [1, 1, 1, 0]
        ]
    finally:
        first.close()
        second.close()
    assert [item["random_draw"] for item in first_diagnostics] == [
        item["random_draw"] for item in second_diagnostics
    ]
    assert [item["dropout_event"] for item in first_diagnostics] == [
        item["dropout_event"] for item in second_diagnostics
    ]


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


@pytest.mark.parametrize(
    ("suite_name", "parameter"),
    [
        ("che49_cartpole_mass_v1.toml", "masspole"),
        ("che49_cartpole_pole_angle_noise_v1.toml", "pole_angle_noise_std_deg"),
        ("che49_cartpole_force_magnitude_v1.toml", "force_mag"),
        ("che49_cartpole_action_delay_v1.toml", "delay_steps"),
        ("che49_cartpole_action_dropout_v1.toml", "dropout_probability"),
    ],
)
def test_secondary_suite_expansion(suite_name: str, parameter: str) -> None:
    suite = load_sweep_suite(ROOT / "configs/eval" / suite_name)
    nominal = load_policy_design(
        ROOT / "configs/policies/cartpole_lqr_nominal_quantized_v1.toml"
    ).environment
    conditions = expand_sweep(suite, nominal)
    assert suite.metric_version == INTERVENTION_METRIC_VERSION
    assert len(conditions) == 5
    assert all(
        parameter in condition.perturbation.parameters for condition in conditions
    )


@pytest.mark.parametrize(
    "spec",
    [
        PerturbationSpec(CARTPOLE_MASS, {"masspole": 0.1}),
        PerturbationSpec(
            CARTPOLE_POLE_ANGLE_NOISE, {"pole_angle_noise_std_deg": 0}
        ),
        PerturbationSpec(CARTPOLE_FORCE_MAGNITUDE, {"force_mag": 10}),
        PerturbationSpec(CARTPOLE_ACTION_DELAY, {"delay_steps": 0}),
        PerturbationSpec(
            CARTPOLE_ACTION_DROPOUT, {"dropout_probability": 0}
        ),
    ],
)
def test_nominal_secondary_conditions_match_no_op_behavior(
    spec: PerturbationSpec,
) -> None:
    direct = make_environment("CartPole-v1")
    perturbed = make_environment("CartPole-v1", spec)
    try:
        direct_trace = _cartpole_trace(direct, seed=37)
        perturbed_trace = _cartpole_trace(perturbed, seed=37)
    finally:
        direct.close()
        perturbed.close()
    assert perturbed_trace == direct_trace


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


def test_v3_metrics_distinguish_requested_and_environment_actions(
    tmp_path: Path,
) -> None:
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
                    "cart_position": 0.0,
                    "cart_velocity": 0.0,
                    "pole_angle": 0.0,
                    "pole_angular_velocity": 0.0,
                },
                environment_action=0,
            ),
            _transition(
                1,
                1.0,
                {
                    "cart_position": 0.0,
                    "cart_velocity": 0.0,
                    "pole_angle": 0.0,
                    "pole_angular_velocity": 0.0,
                },
                environment_action=0,
                perturbation_diagnostics={"kind": "action_delay"},
            ),
            _transition(
                0,
                1.0,
                {
                    "cart_position": 0.0,
                    "cart_velocity": 0.0,
                    "pole_angle": 0.0,
                    "pole_angular_velocity": 0.0,
                },
                environment_action=1,
                truncated=True,
            ),
        ],
        schema_version=4,
    )
    metrics = compute_saved_metrics(
        read_saved_episode(episode_dir), metric_version=INTERVENTION_METRIC_VERSION
    )
    assert metrics.schema_version == 3
    assert metrics.environment_metrics["action_switch_rate"] == 0.5
    assert metrics.environment_metrics["requested_action_switch_rate"] == 1.0
    assert metrics.environment_metrics[
        "requested_applied_action_mismatch_rate"
    ] == pytest.approx(2 / 3)


def test_v3_noise_and_dropout_metrics_use_recorded_diagnostics(
    tmp_path: Path,
) -> None:
    state = {
        "cart_position": 0.0,
        "cart_velocity": 0.0,
        "pole_angle": 0.0,
        "pole_angular_velocity": 0.0,
    }
    noise_dir = _saved_episode(
        tmp_path / "noise",
        "CartPole-v1",
        reset_state=state,
        reset_diagnostics={"pole_angle_noise_radians": 1.0},
        transitions=[
            _transition(
                0,
                1.0,
                state,
                perturbation_diagnostics={
                    "kind": "pole_angle_observation_noise",
                    "pole_angle_noise_radians": -1.0,
                },
                truncated=True,
            )
        ],
        schema_version=4,
    )
    noise = compute_saved_metrics(
        read_saved_episode(noise_dir),
        metric_version=INTERVENTION_METRIC_VERSION,
    )
    assert noise.environment_metrics[
        "rms_pole_angle_observation_error_radians"
    ] == 1.0

    dropout_dir = _saved_episode(
        tmp_path / "dropout",
        "CartPole-v1",
        reset_state=state,
        transitions=[
            _transition(
                0,
                1.0,
                state,
                perturbation_diagnostics={
                    "kind": "action_dropout",
                    "dropout_event": False,
                    "random_draw": None,
                },
            ),
            _transition(
                1,
                1.0,
                state,
                perturbation_diagnostics={
                    "kind": "action_dropout",
                    "dropout_event": True,
                    "random_draw": 0.1,
                },
            ),
            _transition(
                0,
                1.0,
                state,
                perturbation_diagnostics={
                    "kind": "action_dropout",
                    "dropout_event": False,
                    "random_draw": 0.9,
                },
                truncated=True,
            ),
        ],
        schema_version=4,
    )
    dropout = compute_saved_metrics(
        read_saved_episode(dropout_dir),
        metric_version=INTERVENTION_METRIC_VERSION,
    )
    assert dropout.environment_metrics["realized_dropout_rate"] == 0.5


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
    reset_diagnostics: dict | None = None,
    schema_version: int = 3,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": schema_version,
        "resolved_environment": {
            "environment_id": env_id,
            "parameters": parameters or {},
        },
    }
    reset = {
        "type": "reset",
        "observation": [],
        "environment_state": reset_state,
        "perturbation_diagnostics": reset_diagnostics or {},
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
    environment_action: int | None = None,
    perturbation_diagnostics: dict | None = None,
    terminated: bool = False,
    truncated: bool = False,
) -> dict:
    return {
        "type": "transition",
        "action": action,
        "environment_action": (
            action if environment_action is None else environment_action
        ),
        "policy_diagnostics": {},
        "perturbation_diagnostics": perturbation_diagnostics or {},
        "reward": reward,
        "observation": [],
        "environment_state": state,
        "terminated": terminated,
        "truncated": truncated,
        "info": {},
    }


def _cartpole_trace(env, *, seed: int) -> list[dict]:
    observation, _ = env.reset(seed=seed)
    trace = [
        {
            "observation": np.asarray(observation).tolist(),
            "state": np.asarray(env.unwrapped.state).tolist(),
        }
    ]
    for action in [0, 1, 1, 0, 1, 0, 0, 1]:
        observation, reward, terminated, truncated, _ = env.step(action)
        trace.append(
            {
                "observation": np.asarray(observation).tolist(),
                "state": np.asarray(env.unwrapped.state).tolist(),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            }
        )
        if terminated or truncated:
            break
    return trace
