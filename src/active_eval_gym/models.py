"""Typed records shared by policy construction, rollout, and serialization."""

from dataclasses import dataclass, field
from typing import Any

from active_eval_gym.envs.perturbations import PerturbationSpec


@dataclass(frozen=True)
class NominalEnvSpec:
    """The declared reference environment for an evaluation."""

    schema_version: int
    spec_id: str
    environment_id: str
    parameters: dict[str, Any]
    initial_state_distribution: dict[str, Any]
    max_episode_steps: int


@dataclass(frozen=True)
class ResolvedEnvSpec:
    """The actual environment configuration used for a rollout."""

    schema_version: int
    nominal_spec_id: str
    environment_id: str
    parameters: dict[str, Any]
    derived_parameters: dict[str, Any]
    initial_state_distribution: dict[str, Any]
    observation_adapter: str
    max_episode_steps: int


@dataclass(frozen=True)
class PolicyDesignSpec:
    """The complete assumptions and procedure used to produce a policy."""

    schema_version: int
    design_id: str
    policy_id: str
    policy_type: str
    algorithm: str
    algorithm_library: str
    environment: NominalEnvSpec
    environment_package_versions: dict[str, str]
    observation_adapter: str
    training_seed: int | None
    training_steps: int | None
    device: str | None
    hyperparameters: dict[str, Any]


@dataclass(frozen=True)
class PolicyArtifactMetadata:
    """Identity, provenance, and integrity data for one frozen policy."""

    schema_version: int
    policy_id: str
    design_spec: PolicyDesignSpec
    artifact_format: str
    model_filename: str
    model_sha256: str
    package_versions: dict[str, str]
    source_version: dict[str, Any]


@dataclass(frozen=True)
class PolicyAction:
    """An environment action with optional policy-side diagnostics."""

    action: Any
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeMetadata:
    """Provenance required to reproduce and interpret one episode."""

    schema_version: int
    policy_design: PolicyDesignSpec
    policy_artifact: PolicyArtifactMetadata
    nominal_environment: NominalEnvSpec
    perturbation: PerturbationSpec
    resolved_environment: ResolvedEnvSpec
    episode_seed: int
    deterministic: bool
    package_versions: dict[str, str]
    initial_state: dict[str, Any]


@dataclass(frozen=True)
class ResetRecord:
    """Observation and environment information returned by reset."""

    observation: Any
    environment_state: dict[str, Any]
    perturbation_diagnostics: dict[str, Any]
    info: dict[str, Any]


@dataclass(frozen=True)
class TransitionRecord:
    """One policy decision and the resulting environment transition."""

    action: Any
    environment_action: Any
    policy_diagnostics: dict[str, Any]
    perturbation_diagnostics: dict[str, Any]
    reward: float
    observation: Any
    environment_state: dict[str, Any]
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass(frozen=True)
class EpisodeRecord:
    """Raw trajectory plus the provenance needed to interpret it."""

    metadata: EpisodeMetadata
    reset: ResetRecord
    transitions: tuple[TransitionRecord, ...]
