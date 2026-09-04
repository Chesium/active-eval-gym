import json
from math import pi, radians
from pathlib import Path

import numpy as np
import pytest

from active_eval_gym.boundary import plan_boundary_stage
from active_eval_gym.config import load_policy_design, load_sweep_suite
from active_eval_gym.envs.factory import make_environment
from active_eval_gym.envs.perturbations import (
    CARTPOLE_RECOVERY_ANGLE_LENGTH,
    PerturbationSpec,
)
from active_eval_gym.envs.specs import (
    IDENTITY_OBSERVATION,
    capture_resolved_environment,
)
from active_eval_gym.metrics import RECOVERY_METRIC_VERSION, compute_saved_metrics
from active_eval_gym.serialization import SavedEpisode
from active_eval_gym.sweeps import expand_sweep

ROOT = Path(__file__).resolve().parents[1]


def test_recovery_perturbation_replaces_only_angle_and_updates_spec() -> None:
    direct = make_environment("CartPole-v1")
    spec = PerturbationSpec(
        CARTPOLE_RECOVERY_ANGLE_LENGTH,
        {"initial_theta_deg": 30, "length": 1.25, "theta_threshold_deg": 90},
    )
    recovery = make_environment("CartPole-v1", spec)
    design = load_policy_design(
        ROOT / "configs/policies/cartpole_lqr_nominal_quantized_v1.toml"
    )
    try:
        nominal, _ = direct.reset(seed=17)
        observation, _ = recovery.reset(seed=17)
        resolved = capture_resolved_environment(
            recovery, design.environment, IDENTITY_OBSERVATION, spec
        )
        np.testing.assert_allclose(observation[[0, 1, 3]], nominal[[0, 1, 3]])
        assert observation[2] == pytest.approx(radians(30))
        assert recovery.observation_space.contains(observation)
        assert recovery.observation_space.high[2] == pytest.approx(pi)
        assert resolved.parameters["length"] == 1.25
        assert resolved.parameters["theta_threshold_radians"] == pytest.approx(pi / 2)
        assert resolved.derived_parameters["polemass_length"] == pytest.approx(0.125)
        distribution = resolved.initial_state_distribution
        assert distribution["kind"] == "seeded_nominal_with_replaced_component"
    finally:
        direct.close()
        recovery.close()


@pytest.mark.parametrize("angle", [-90, 90, float("inf")])
def test_recovery_perturbation_rejects_invalid_angles(angle: float) -> None:
    spec = PerturbationSpec(
        CARTPOLE_RECOVERY_ANGLE_LENGTH,
        {"initial_theta_deg": angle, "length": 0.5, "theta_threshold_deg": 90},
    )
    with pytest.raises(ValueError):
        make_environment("CartPole-v1", spec)


def test_pilot_config_expands_to_143_conditions() -> None:
    suite = load_sweep_suite(
        ROOT / "configs/eval/cartpole_failure_boundary_pilot_v1.toml"
    )
    nominal = load_policy_design(
        ROOT / "configs/policies/cartpole_lqr_nominal_quantized_v1.toml"
    ).environment
    conditions = expand_sweep(suite, nominal)
    assert len(conditions) == 143
    assert len({item.condition_id for item in conditions}) == 143


def test_recovery_metrics_distinguish_survival_recovery_and_failure_causes() -> None:
    stable = _saved_episode(
        [_transition(0.04, truncated=index == 499) for index in range(500)]
    )
    metrics = compute_saved_metrics(stable, metric_version=RECOVERY_METRIC_VERSION)
    assert metrics.task_success is True
    assert metrics.recovery_success is True
    assert metrics.failure_cause == "none"
    tail_rms = metrics.environment_metrics["tail_100_rms_pole_angle_radians"]
    assert tail_rms == pytest.approx(0.04)

    oscillating = _saved_episode(
        [_transition(0.2, truncated=index == 499) for index in range(500)]
    )
    metrics = compute_saved_metrics(
        oscillating, metric_version=RECOVERY_METRIC_VERSION
    )
    assert metrics.task_success is True
    assert metrics.recovery_success is False

    for position, angle, expected in (
        (2.5, 0.0, "cart_limit"),
        (0.0, 1.7, "angle_limit"),
        (2.5, 1.7, "both"),
        (0.0, 0.0, "unknown"),
    ):
        failed = _saved_episode(
            [_transition(angle, position=position, terminated=True)]
        )
        metrics = compute_saved_metrics(failed, metric_version=RECOVERY_METRIC_VERSION)
        assert metrics.failure_cause == expected
        assert metrics.recovery_success is False


def test_boundary_selector_runs_two_rounds_and_writes_final_suite(
    tmp_path: Path,
) -> None:
    pilot_suite = {
        "schema_version": 1,
        "suite_id": "tiny-pilot",
        "environment_id": "CartPole-v1",
        "perturbation_name": CARTPOLE_RECOVERY_ANGLE_LENGTH,
        "metric_version": RECOVERY_METRIC_VERSION,
        "seeds": [0],
        "policy_ids": ["policy"],
        "derived_policies": {},
        "grid": {
            "initial_theta_deg": [-1, 1],
            "length": [0.25, 1.0],
            "theta_threshold_deg": [90],
        },
        "boundary_study": {
            "kind": "adaptive-boundary-v1",
            "study_id": "tiny",
            "refinement_rounds": 2,
            "classification_threshold": 0.5,
            "recovery_tail_steps": 100,
            "recovery_rms_angle_deg": 5.0,
            "max_final_conditions": 50,
            "final_seeds": [0, 1],
        },
    }
    pilot = tmp_path / "pilot"
    _evaluation(pilot, pilot_suite)

    round_one_config = tmp_path / "round-one.json"
    first = plan_boundary_stage(
        pilot, [pilot], stage="refinement-1", output=round_one_config
    )
    assert first["condition_count"] == 5
    round_one = tmp_path / "round-one"
    _evaluation(round_one, json.loads(round_one_config.read_text()))

    round_two_config = tmp_path / "round-two.json"
    second = plan_boundary_stage(
        pilot,
        [pilot, round_one],
        stage="refinement-2",
        output=round_two_config,
    )
    assert second["condition_count"] > 0
    round_two = tmp_path / "round-two"
    _evaluation(round_two, json.loads(round_two_config.read_text()))

    final_config = tmp_path / "final.json"
    final = plan_boundary_stage(
        pilot,
        [pilot, round_one, round_two],
        stage="final",
        output=final_config,
    )
    loaded = load_sweep_suite(final_config)
    assert final["condition_count"] <= 50
    assert loaded.seeds == (0, 1)
    assert loaded.grid == {}
    assert len(loaded.conditions) == final["condition_count"]
    assert loaded.boundary_study["source_summary_sha256"]


def _transition(
    angle: float,
    *,
    position: float = 0.0,
    terminated: bool = False,
    truncated: bool = False,
) -> dict:
    return {
        "type": "transition",
        "action": 0,
        "environment_action": 0,
        "reward": 1.0,
        "environment_state": {
            "cart_position": position,
            "cart_velocity": 0.0,
            "pole_angle": angle,
            "pole_angular_velocity": 0.0,
        },
        "terminated": terminated,
        "truncated": truncated,
        "perturbation_diagnostics": {},
    }


def _saved_episode(transitions: list[dict]) -> SavedEpisode:
    return SavedEpisode(
        metadata={
            "resolved_environment": {
                "environment_id": "CartPole-v1",
                "parameters": {
                    "x_threshold": 2.4,
                    "theta_threshold_radians": pi / 2,
                },
            }
        },
        reset={
            "environment_state": {
                "cart_position": 0.0,
                "cart_velocity": 0.0,
                "pole_angle": 0.0,
                "pole_angular_velocity": 0.0,
            },
            "perturbation_diagnostics": {},
        },
        transitions=tuple(transitions),
        trajectory_sha256="test",
    )


def _evaluation(path: Path, suite: dict) -> None:
    path.mkdir()
    (path / "suite.json").write_text(json.dumps({"suite": suite}))
    conditions = []
    if "grid" in suite:
        parameters = [
            {
                "initial_theta_deg": angle,
                "length": length,
                "theta_threshold_deg": suite["grid"]["theta_threshold_deg"][0],
            }
            for angle in suite["grid"]["initial_theta_deg"]
            for length in suite["grid"]["length"]
        ]
    else:
        parameters = [item["parameters"] for item in suite["conditions"]]
    for item in parameters:
        rate = 0.0 if item["initial_theta_deg"] < 0 else 1.0
        conditions.append(
            {
                "parameters": item,
                "policies": {
                    "policy": {"success_rate": rate, "recovery_rate": rate}
                },
            }
        )
    summary = {
        "seeds": suite["seeds"],
        "conditions": conditions,
    }
    analysis = path / "analysis" / suite["metric_version"]
    analysis.mkdir(parents=True)
    (analysis / "summary.json").write_text(json.dumps(summary))
