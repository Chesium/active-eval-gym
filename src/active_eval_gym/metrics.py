"""Derived metrics computed without mutating raw episode records."""

from dataclasses import dataclass
from math import atan2, cos, fsum, sin, sqrt
from typing import Any

from active_eval_gym.models import EpisodeRecord

METRIC_VERSION = "episode-summary-v1"
SWEEP_METRIC_VERSION = "episode-summary-v2"


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


def compute_saved_metrics(episode: Any) -> SweepEpisodeMetrics:
    """Compute v2 metrics solely from a hash-verified saved raw episode."""

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
        environment_metrics = _cartpole_metrics(episode)
    elif env_id == "Pendulum-v1":
        task_success = None
        environment_metrics = _pendulum_metrics(episode)
    elif env_id == "MiniGrid-Empty-8x8-v0":
        task_success = terminated
        environment_metrics = _minigrid_metrics(episode, task_success)
    else:
        raise ValueError(f"Unsupported metrics environment {env_id!r}.")

    return SweepEpisodeMetrics(
        schema_version=2,
        metric_version=SWEEP_METRIC_VERSION,
        source_trajectory_sha256=episode.trajectory_sha256,
        episode_return=fsum(float(row["reward"]) for row in transitions),
        episode_length=len(transitions),
        terminated=terminated,
        truncated=truncated,
        end_reason=end_reason,
        task_success=task_success,
        environment_metrics=environment_metrics,
    )


def _cartpole_metrics(episode: Any) -> dict[str, float]:
    states = [episode.reset["environment_state"]]
    states.extend(row["environment_state"] for row in episode.transitions)
    angles = [float(state["pole_angle"]) for state in states]
    positions = [float(state["cart_position"]) for state in states]
    actions = [int(row["action"]) for row in episode.transitions]
    switches = sum(
        first != second for first, second in zip(actions, actions[1:], strict=False)
    )
    return {
        "rms_pole_angle_radians": _rms(angles),
        "max_abs_pole_angle_radians": max(abs(value) for value in angles),
        "rms_cart_position": _rms(positions),
        "action_switch_rate": (
            0.0 if len(actions) < 2 else switches / (len(actions) - 1)
        ),
    }


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


def _scalar_action(action: Any) -> float:
    if isinstance(action, list) and len(action) == 1:
        return float(action[0])
    return float(action)
