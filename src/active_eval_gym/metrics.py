"""Derived metrics computed without mutating raw episode records."""

from dataclasses import dataclass
from math import atan2, cos, fsum, radians, sin, sqrt
from typing import Any

from active_eval_gym.models import EpisodeRecord

METRIC_VERSION = "episode-summary-v1"
SWEEP_METRIC_VERSION = "episode-summary-v2"
INTERVENTION_METRIC_VERSION = "episode-summary-v3"
RECOVERY_METRIC_VERSION = "episode-summary-v4"
RECOVERY_TAIL_STEPS = 100
RECOVERY_RMS_ANGLE_RADIANS = radians(5.0)


@dataclass(frozen=True)
class EpisodeMetrics:
    """The first version of environment-agnostic episode metrics."""

    schema_version: int
    metric_version: str
    source_trajectory_sha256: str
    episode_return: float
    episode_length: int
    terminated: bool
    truncated: bool
    end_reason: str
    task_success: bool | None


@dataclass(frozen=True)
class SweepEpisodeMetrics:
    """Versioned common and environment-specific perturbation metrics."""

    schema_version: int
    metric_version: str
    source_trajectory_sha256: str
    episode_return: float
    episode_length: int
    terminated: bool
    truncated: bool
    end_reason: str
    task_success: bool | None
    environment_metrics: dict[str, Any]


@dataclass(frozen=True)
class RecoveryEpisodeMetrics(SweepEpisodeMetrics):
    """Recovery-study metrics with explicit co-primary outcomes."""

    recovery_success: bool
    failure_cause: str


def compute_metrics(
    episode: EpisodeRecord, *, source_trajectory_sha256: str
) -> EpisodeMetrics:
    """Compute summary metrics solely from a completed raw trajectory."""

    if not episode.transitions:
        raise ValueError("Cannot compute metrics for an empty trajectory.")
    final = episode.transitions[-1]
    if not (final.terminated or final.truncated):
        raise ValueError("Cannot compute metrics for an incomplete trajectory.")

    if final.terminated and final.truncated:
        end_reason = "terminated_and_truncated"
    elif final.terminated:
        end_reason = "terminated"
    else:
        end_reason = "truncated"

    env_id = episode.metadata.resolved_environment.environment_id
    if env_id == "CartPole-v1":
        task_success = final.truncated and not final.terminated
    elif env_id == "MiniGrid-Empty-8x8-v0":
        task_success = final.terminated
    else:
        task_success = None

    return EpisodeMetrics(
        schema_version=1,
        metric_version=METRIC_VERSION,
        source_trajectory_sha256=source_trajectory_sha256,
        episode_return=fsum(step.reward for step in episode.transitions),
        episode_length=len(episode.transitions),
        terminated=final.terminated,
        truncated=final.truncated,
        end_reason=end_reason,
        task_success=task_success,
    )


def compute_saved_metrics(
    episode: Any, *, metric_version: str = SWEEP_METRIC_VERSION
) -> SweepEpisodeMetrics:
    """Compute selected metrics solely from a hash-verified saved raw episode."""

    if metric_version not in (
        SWEEP_METRIC_VERSION,
        INTERVENTION_METRIC_VERSION,
        RECOVERY_METRIC_VERSION,
    ):
        raise ValueError(f"Unsupported sweep metric version {metric_version!r}.")

    transitions = episode.transitions
    if not transitions:
        raise ValueError("Cannot compute metrics for an empty trajectory.")
    final = transitions[-1]
    terminated = bool(final["terminated"])
    truncated = bool(final["truncated"])
    if not (terminated or truncated):
        raise ValueError("Cannot compute metrics for an incomplete trajectory.")
    if terminated and truncated:
        end_reason = "terminated_and_truncated"
    elif terminated:
        end_reason = "terminated"
    else:
        end_reason = "truncated"

    env_id = episode.metadata["resolved_environment"]["environment_id"]
    if env_id == "CartPole-v1":
        task_success: bool | None = truncated and not terminated
        environment_metrics = _cartpole_metrics(episode, metric_version)
    elif env_id == "Pendulum-v1":
        task_success = None
        environment_metrics = _pendulum_metrics(episode)
    elif env_id == "MiniGrid-Empty-8x8-v0":
        task_success = terminated
        environment_metrics = _minigrid_metrics(episode, task_success)
    else:
        raise ValueError(f"Unsupported metrics environment {env_id!r}.")

    common = dict(
        schema_version={
            SWEEP_METRIC_VERSION: 2,
            INTERVENTION_METRIC_VERSION: 3,
            RECOVERY_METRIC_VERSION: 4,
        }[metric_version],
        metric_version=metric_version,
        source_trajectory_sha256=episode.trajectory_sha256,
        episode_return=fsum(float(row["reward"]) for row in transitions),
        episode_length=len(transitions),
        terminated=terminated,
        truncated=truncated,
        end_reason=end_reason,
        task_success=task_success,
        environment_metrics=environment_metrics,
    )
    if metric_version == RECOVERY_METRIC_VERSION:
        if env_id != "CartPole-v1":
            raise ValueError("Recovery metrics require CartPole-v1.")
        return RecoveryEpisodeMetrics(
            **common,
            recovery_success=bool(
                task_success
                and environment_metrics["tail_100_rms_pole_angle_radians"]
                <= RECOVERY_RMS_ANGLE_RADIANS
            ),
            failure_cause=_cartpole_failure_cause(episode, task_success),
        )
    return SweepEpisodeMetrics(**common)


def _cartpole_metrics(episode: Any, metric_version: str) -> dict[str, Any]:
    states = [episode.reset["environment_state"]]
    states.extend(row["environment_state"] for row in episode.transitions)
    angles = [float(state["pole_angle"]) for state in states]
    positions = [float(state["cart_position"]) for state in states]
    requested_actions = [int(row["action"]) for row in episode.transitions]
    result: dict[str, Any] = {
        "rms_pole_angle_radians": _rms(angles),
        "max_abs_pole_angle_radians": max(abs(value) for value in angles),
        "rms_cart_position": _rms(positions),
    }
    if metric_version == RECOVERY_METRIC_VERSION:
        tail_angles = angles[-RECOVERY_TAIL_STEPS:]
        result["tail_100_rms_pole_angle_radians"] = _rms(tail_angles)
    if metric_version == SWEEP_METRIC_VERSION:
        result["action_switch_rate"] = _action_switch_rate(requested_actions)
        return result

    environment_actions = [
        int(row.get("environment_action", row["action"]))
        for row in episode.transitions
    ]
    result.update(
        {
            "action_switch_rate": _action_switch_rate(environment_actions),
            "requested_action_switch_rate": _action_switch_rate(
                requested_actions
            ),
            "requested_applied_action_mismatch_rate": fsum(
                requested != applied
                for requested, applied in zip(
                    requested_actions, environment_actions, strict=True
                )
            )
            / len(requested_actions),
            "rms_pole_angle_observation_error_radians": _rms(
                _observation_noise_trace(episode)
            ),
            "realized_dropout_rate": _realized_dropout_rate(episode),
        }
    )
    return result


def _cartpole_failure_cause(episode: Any, task_success: bool | None) -> str:
    if task_success:
        return "none"
    final = episode.transitions[-1]
    if not bool(final["terminated"]):
        return "unknown"
    state = final["environment_state"]
    parameters = episode.metadata["resolved_environment"]["parameters"]
    cart_failed = abs(float(state["cart_position"])) > float(
        parameters["x_threshold"]
    )
    angle_failed = abs(float(state["pole_angle"])) > float(
        parameters["theta_threshold_radians"]
    )
    if cart_failed and angle_failed:
        return "both"
    if cart_failed:
        return "cart_limit"
    if angle_failed:
        return "angle_limit"
    return "unknown"


def _pendulum_metrics(episode: Any) -> dict[str, float]:
    states = [episode.reset["environment_state"]]
    states.extend(row["environment_state"] for row in episode.transitions)
    angles = [
        atan2(sin(float(state["angle"])), cos(float(state["angle"])))
        for state in states
    ]
    velocities = [float(state["angular_velocity"]) for state in states]
    maximum = float(
        episode.metadata["resolved_environment"]["parameters"]["max_torque"]
    )
    torques = [
        max(-maximum, min(maximum, _scalar_action(row["action"])))
        for row in episode.transitions
    ]
    return {
        "rms_angle_error_radians": _rms(angles),
        "rms_angular_velocity": _rms(velocities),
        "rms_torque": _rms(torques),
        "mean_absolute_torque": fsum(abs(value) for value in torques) / len(torques),
    }


def _minigrid_metrics(episode: Any, success: bool) -> dict[str, Any]:
    action_names = ("left", "right", "forward", "pickup", "drop", "toggle", "done")
    counts = dict.fromkeys(action_names, 0)
    for row in episode.transitions:
        action = int(row["action"])
        if action not in range(len(action_names)):
            raise ValueError(f"Unknown MiniGrid action {action}.")
        counts[action_names[action]] += 1
    states = [episode.reset["environment_state"]]
    states.extend(row["environment_state"] for row in episode.transitions)
    positions = [state["agent_position"] for state in states]
    traveled = fsum(
        abs(int(first[0]) - int(second[0])) + abs(int(first[1]) - int(second[1]))
        for first, second in zip(positions, positions[1:], strict=False)
    )
    parameters = episode.metadata["resolved_environment"]["parameters"]
    goal = (int(parameters["width"]) - 2, int(parameters["height"]) - 2)
    start = positions[0]
    shortest = abs(int(start[0]) - goal[0]) + abs(int(start[1]) - goal[1])
    efficiency = shortest / traveled if success and traveled > 0 else None
    return {
        "path_efficiency": efficiency,
        "turn_count": counts["left"] + counts["right"],
        "action_counts": counts,
    }


def _rms(values: list[float]) -> float:
    return sqrt(fsum(value * value for value in values) / len(values))


def _action_switch_rate(actions: list[int]) -> float:
    if len(actions) < 2:
        return 0.0
    switches = sum(
        first != second for first, second in zip(actions, actions[1:], strict=False)
    )
    return switches / (len(actions) - 1)


def _observation_noise_trace(episode: Any) -> list[float]:
    diagnostics = [episode.reset.get("perturbation_diagnostics", {})]
    diagnostics.extend(
        row.get("perturbation_diagnostics", {}) for row in episode.transitions
    )
    return [
        float(item.get("pole_angle_noise_radians", 0.0)) for item in diagnostics
    ]


def _realized_dropout_rate(episode: Any) -> float | None:
    events = [
        bool(row.get("perturbation_diagnostics", {}).get("dropout_event"))
        for row in episode.transitions
        if row.get("perturbation_diagnostics", {}).get("kind") == "action_dropout"
        and row.get("perturbation_diagnostics", {}).get("random_draw") is not None
    ]
    return None if not events else fsum(events) / len(events)


def _scalar_action(action: Any) -> float:
    if isinstance(action, list) and len(action) == 1:
        return float(action[0])
    return float(action)
