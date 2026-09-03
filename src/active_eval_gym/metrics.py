"""Derived metrics computed without mutating raw episode records."""

from dataclasses import dataclass
from math import fsum

from active_eval_gym.models import EpisodeRecord

METRIC_VERSION = "episode-summary-v1"


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
