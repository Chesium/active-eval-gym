"""Typed records shared by rollout, metrics, and serialization."""

from dataclasses import dataclass
from typing import Any

from active_eval_gym.envs.perturbations import PerturbationSpec


@dataclass(frozen=True)
class PolicyMetadata:
    """Identity and provenance for a fixed policy."""

    policy_id: str
    checkpoint: str | None = None
    training_seed: int | None = None
    action_seed: int | None = None


@dataclass(frozen=True)
class EpisodeMetadata:
    """Provenance required to reproduce one episode."""

    schema_version: int
    environment_id: str
    package_versions: dict[str, str]
    policy: PolicyMetadata
    episode_seed: int
    perturbation: PerturbationSpec
    initial_state: dict[str, Any]


@dataclass(frozen=True)
class ResetRecord:
    """Observation and environment information returned by reset."""

    observation: Any
    info: dict[str, Any]


@dataclass(frozen=True)
class TransitionRecord:
    """One action and the resulting environment transition."""

    action: Any
    reward: float
    observation: Any
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass(frozen=True)
class EpisodeRecord:
    """Raw trajectory plus the provenance needed to interpret it."""

    metadata: EpisodeMetadata
    reset: ResetRecord
    transitions: tuple[TransitionRecord, ...]
