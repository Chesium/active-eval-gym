"""Small TOML configuration loader for policy and evaluation specifications."""

import tomllib
from dataclasses import dataclass
from importlib.metadata import version
from math import isclose, radians
from pathlib import Path
from typing import Any

from active_eval_gym.envs.perturbations import CARTPOLE_RECOVERY_ANGLE_LENGTH
from active_eval_gym.envs.specs import package_versions
from active_eval_gym.metrics import (
    RECOVERY_METRIC_VERSION,
    RECOVERY_RMS_ANGLE_RADIANS,
    RECOVERY_TAIL_STEPS,
)
from active_eval_gym.models import NominalEnvSpec, PolicyDesignSpec

PPO_BOUNDARY_ARMS = ("background", "acceptance_boundary", "intermediate")
PPO_BOUNDARY_SUPPORTS = ("discovery", "reference", "smoke")
PPO_BOUNDARY_SEED_ROLES = ("discovery", "training", "confirmatory", "pilot", "smoke")
PPO_BOUNDARY_TIERS = ("smoke", "pilot", "confirmatory")
# How hard each rollout is optimized may change; the task and batch geometry may not.
PPO_BOUNDARY_TUNABLE_SETTINGS = ("n_epochs",)
PPO_BOUNDARY_SNAPSHOT_MODES = ("none", "if-s6-headroom")
PPO_BOUNDARY_PROBABILITY_BAND_RULE = "ppo-boundary-targets-v1"
PPO_BOUNDARY_FRONTIER_RULE = "ppo-boundary-targets-v2"


@dataclass(frozen=True)
class QualityGate:
    """Nominal behavior required before an artifact can be frozen."""

    minimum_success_rate: float | None = None
    minimum_mean_episode_length: float | None = None
    maximum_mean_episode_length: float | None = None
    minimum_mean_return: float | None = None


@dataclass(frozen=True)
class NominalSuiteSpec:
    """A fixed multi-policy nominal evaluation protocol."""

    schema_version: int
    suite_id: str
    seeds: tuple[int, ...]
    policy_ids: tuple[str, ...]
    gates: dict[str, QualityGate]


@dataclass(frozen=True)
class SweepSuiteSpec:
    """A deterministic grid of perturbations for frozen policies."""

    schema_version: int
    suite_id: str
    environment_id: str
    perturbation_name: str
    metric_version: str
    seeds: tuple[int, ...]
    policy_ids: tuple[str, ...]
    grid: dict[str, Any]
    conditions: tuple[dict[str, Any], ...]
    derived_policies: dict[str, dict[str, Any]]
    boundary_study: dict[str, Any]


@dataclass(frozen=True)
class CartPoleSymmetrySuiteSpec:
    """Checkpoint probes and paired rollouts for a CartPole symmetry study."""

    schema_version: int
    suite_id: str
    environment_id: str
    policy_id: str
    control_policy_id: str
    derived_policy_id: str
    seeds: tuple[int, ...]
    angle_magnitudes_deg: tuple[float, ...]
    signed_angles_deg: tuple[float, ...]
    lengths: tuple[float, ...]
    observed_stride: int
    plot_grid_points: int
    plot_angular_velocity_limit: float
    plot_cart_velocities: tuple[float, ...]


@dataclass(frozen=True)
class RecoverySupport:
    """A declared angle-by-half-length grid plus any explicit extra points."""

    initial_theta_deg: tuple[float, ...]
    length: tuple[float, ...]
    extra_points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class AcceptanceSpec:
    """Recovery acceptance thresholds over frozen-policy evaluations."""

    primary_tau: float
    secondary_taus: tuple[float, ...]
    recovery_tail_steps: int
    recovery_rms_angle_deg: float


@dataclass(frozen=True)
class CurriculumSpec:
    """Target-distribution rules shared by every comparison arm."""

    arms: tuple[str, ...]
    discovery_episodes_per_condition: int
    target_rule: str
    mixture_lambda: float
    boundary_half_width: float | None
    pseudocount_success: float
    pseudocount_total: float
    thin_band_min_conditions: int


@dataclass(frozen=True)
class ContinuationSpec:
    """Settings for additional PPO training from the parent checkpoint."""

    learning_rate: float
    clip_range: float
    device: str
    checkpoint_selection: str
    vec_env_backend: str
    torch_threads: int
    hyperparameters: dict[str, Any]


@dataclass(frozen=True)
class TransitionSamplingSpec:
    """Training-only inverse-duration correction for desired transition shares."""

    mode: str
    calibration_episodes_per_condition: int
    calibration_seed_role: int
    calibration_action_seed: int
    length_ema_alpha: float
    refresh_rollouts: int
    minimum_length: float


@dataclass(frozen=True)
class StudySeedSpec:
    """Named seed roots and episode-reset roles."""

    master_seed: int
    training_policy_seed_base: int
    condition_stream_tag: int
    roles: dict[str, int]


@dataclass(frozen=True)
class StudyTierSpec:
    """One execution tier with its budgets and evaluation design."""

    name: str
    replicate_ids: tuple[int, ...]
    additional_transitions: int
    snapshot_transitions: tuple[int, ...]
    evaluation_support: str
    episodes_per_condition: int
    seed_role: str
    snapshot_evaluation: str
    arms: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ConfidenceSpec:
    """The simultaneous confidence family for the confirmatory campaign."""

    family_alpha: float
    interval: str


@dataclass(frozen=True)
class EvaluationStorageSpec:
    """Lossless raw-evaluation layout declared before collection."""

    evaluation_layout: str
    compression: str
    compression_level: int
    episode_digest: str


@dataclass(frozen=True)
class PPOBoundaryStudySpec:
    """The complete declared protocol for the CartPole PPO boundary study."""

    schema_version: int
    study_id: str
    environment_id: str
    perturbation_name: str
    metric_version: str
    theta_threshold_deg: float
    outer_rounds: int
    training_reward: str
    parent_policy_id: str
    parent_model_sha256: str
    parent_design: PolicyDesignSpec
    supports: dict[str, RecoverySupport]
    acceptance: AcceptanceSpec
    curriculum: CurriculumSpec
    continuation: ContinuationSpec
    seeds: StudySeedSpec
    tiers: tuple[StudyTierSpec, ...]
    confidence: ConfidenceSpec
    storage: EvaluationStorageSpec
    sampling: TransitionSamplingSpec | None = None

    def tier(self, name: str) -> StudyTierSpec:
        """Return one execution tier by name."""

        for tier in self.tiers:
            if tier.name == name:
                return tier
        raise KeyError(f"Unknown study tier {name!r}.")


def load_nominal_env_spec(path: Path) -> NominalEnvSpec:
    """Load one nominal environment specification."""

    data = _read_toml(path)
    return NominalEnvSpec(
        schema_version=_required_int(data, "schema_version"),
        spec_id=_required_str(data, "spec_id"),
        environment_id=_required_str(data, "environment_id"),
        parameters=dict(_required_table(data, "parameters")),
        initial_state_distribution=dict(
            _required_table(data, "initial_state_distribution")
        ),
        max_episode_steps=_required_int(data, "max_episode_steps"),
    )


def load_policy_design(path: Path) -> PolicyDesignSpec:
    """Load and fully resolve a policy design specification."""

    data = _read_toml(path)
    nominal_path = path.parent / _required_str(data, "nominal_environment")
    nominal = load_nominal_env_spec(nominal_path.resolve())
    algorithm_library = _required_str(data, "algorithm_library")
    versions = package_versions(nominal.environment_id)
    if algorithm_library == "stable-baselines3":
        versions["stable-baselines3"] = version("stable-baselines3")
    elif algorithm_library == "python-control":
        versions["control"] = version("control")
    else:
        raise ValueError(f"Unsupported algorithm library {algorithm_library!r}.")
    return PolicyDesignSpec(
        schema_version=_required_int(data, "schema_version"),
        design_id=_required_str(data, "design_id"),
        policy_id=_required_str(data, "policy_id"),
        policy_type=_required_str(data, "policy_type"),
        algorithm=_required_str(data, "algorithm"),
        algorithm_library=algorithm_library,
        environment=nominal,
        environment_package_versions=versions,
        observation_adapter=_required_str(data, "observation_adapter"),
        training_seed=_optional_int(data.get("training_seed"), "training_seed"),
        training_steps=_optional_int(data.get("training_steps"), "training_steps"),
        device=_optional_str(data.get("device"), "device"),
        hyperparameters=dict(_required_table(data, "hyperparameters")),
    )


def load_nominal_suite(path: Path) -> NominalSuiteSpec:
    """Load the seed set, artifact order, and freeze gates."""

    data = _read_toml(path)
    seeds = tuple(
        _required_int_value(value, "seeds") for value in data.get("seeds", [])
    )
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty list of unique integers.")
    policy_ids = tuple(data.get("policy_ids", []))
    if not policy_ids or any(not isinstance(value, str) for value in policy_ids):
        raise ValueError("policy_ids must be a non-empty list of strings.")
    gate_data = _required_table(data, "gates")
    gates: dict[str, QualityGate] = {}
    for policy_id in policy_ids:
        raw = gate_data.get(policy_id)
        if not isinstance(raw, dict):
            raise ValueError(f"Missing quality gate for policy {policy_id!r}.")
        gates[policy_id] = QualityGate(
            minimum_success_rate=_optional_number(raw.get("minimum_success_rate")),
            minimum_mean_episode_length=_optional_number(
                raw.get("minimum_mean_episode_length")
            ),
            maximum_mean_episode_length=_optional_number(
                raw.get("maximum_mean_episode_length")
            ),
            minimum_mean_return=_optional_number(raw.get("minimum_mean_return")),
        )
    return NominalSuiteSpec(
        schema_version=_required_int(data, "schema_version"),
        suite_id=_required_str(data, "suite_id"),
        seeds=seeds,
        policy_ids=policy_ids,
        gates=gates,
    )


def load_sweep_suite(path: Path) -> SweepSuiteSpec:
    """Load a tracked perturbation sampling strategy."""

    data = _read_json(path) if path.suffix == ".json" else _read_toml(path)
    seeds = tuple(
        _required_int_value(value, "seeds") for value in data.get("seeds", [])
    )
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty list of unique integers.")
    policy_ids = tuple(data.get("policy_ids", []))
    if not policy_ids or any(not isinstance(value, str) for value in policy_ids):
        raise ValueError("policy_ids must be a non-empty list of strings.")
    raw_derived = data.get("derived_policies", {})
    if not isinstance(raw_derived, dict) or any(
        not isinstance(policy_id, str) or not isinstance(spec, dict)
        for policy_id, spec in raw_derived.items()
    ):
        raise ValueError("derived_policies must be a table of policy tables.")
    if any(policy_id not in policy_ids for policy_id in raw_derived):
        raise ValueError("Every derived policy must appear in policy_ids.")
    raw_grid = data.get("grid")
    raw_conditions = data.get("conditions")
    if (raw_grid is None) == (raw_conditions is None):
        raise ValueError("A sweep requires exactly one of grid or conditions.")
    if raw_grid is not None and not isinstance(raw_grid, dict):
        raise ValueError("grid must be a table.")
    conditions: tuple[dict[str, Any], ...] = ()
    if raw_conditions is not None:
        if not isinstance(raw_conditions, list) or not raw_conditions:
            raise ValueError("conditions must be a non-empty array of tables.")
        normalized: list[dict[str, Any]] = []
        identifiers: set[str] = set()
        for raw in raw_conditions:
            if not isinstance(raw, dict):
                raise ValueError("Every condition must be a table.")
            condition_id = _required_str(raw, "condition_id")
            if condition_id in identifiers:
                raise ValueError(f"Duplicate condition_id {condition_id!r}.")
            identifiers.add(condition_id)
            normalized.append(
                {
                    "condition_id": condition_id,
                    "parameters": dict(_required_table(raw, "parameters")),
                }
            )
        conditions = tuple(normalized)
    boundary_study = data.get("boundary_study", {})
    if not isinstance(boundary_study, dict):
        raise ValueError("boundary_study must be a table.")
    return SweepSuiteSpec(
        schema_version=_required_int(data, "schema_version"),
        suite_id=_required_str(data, "suite_id"),
        environment_id=_required_str(data, "environment_id"),
        perturbation_name=_required_str(data, "perturbation_name"),
        metric_version=_optional_str(data.get("metric_version"), "metric_version")
        or "episode-summary-v2",
        seeds=seeds,
        policy_ids=policy_ids,
        grid={} if raw_grid is None else dict(raw_grid),
        conditions=conditions,
        derived_policies={
            policy_id: dict(spec) for policy_id, spec in raw_derived.items()
        },
        boundary_study=dict(boundary_study),
    )


def load_cartpole_symmetry_suite(path: Path) -> CartPoleSymmetrySuiteSpec:
    """Load and validate the tracked CartPole symmetry protocol."""

    data = _read_toml(path)
    seeds = tuple(
        _required_int_value(value, "seeds") for value in data.get("seeds", [])
    )
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty list of unique integers.")
    angles = _number_tuple(data.get("angle_magnitudes_deg"), "angle_magnitudes_deg")
    signed_angles = _number_tuple(data.get("signed_angles_deg"), "signed_angles_deg")
    lengths = _number_tuple(data.get("lengths"), "lengths")
    cart_velocities = _number_tuple(
        data.get("plot_cart_velocities"), "plot_cart_velocities"
    )
    if any(value <= 0.0 for value in angles):
        raise ValueError("angle_magnitudes_deg values must be positive.")
    if any(value <= 0.0 for value in lengths):
        raise ValueError("lengths values must be positive.")
    observed_stride = _required_int(data, "observed_stride")
    plot_grid_points = _required_int(data, "plot_grid_points")
    angular_velocity_limit = _required_number(data, "plot_angular_velocity_limit")
    if observed_stride <= 0 or plot_grid_points < 3 or angular_velocity_limit <= 0:
        raise ValueError("Symmetry sampling counts and limits must be positive.")
    if len(cart_velocities) != 3:
        raise ValueError("plot_cart_velocities must contain exactly three values.")
    return CartPoleSymmetrySuiteSpec(
        schema_version=_required_int(data, "schema_version"),
        suite_id=_required_str(data, "suite_id"),
        environment_id=_required_str(data, "environment_id"),
        policy_id=_required_str(data, "policy_id"),
        control_policy_id=_required_str(data, "control_policy_id"),
        derived_policy_id=_required_str(data, "derived_policy_id"),
        seeds=seeds,
        angle_magnitudes_deg=angles,
        signed_angles_deg=signed_angles,
        lengths=lengths,
        observed_stride=observed_stride,
        plot_grid_points=plot_grid_points,
        plot_angular_velocity_limit=angular_velocity_limit,
        plot_cart_velocities=cart_velocities,
    )


def load_ppo_boundary_study(path: Path) -> PPOBoundaryStudySpec:
    """Load and validate the CartPole PPO boundary-study protocol."""

    data = _read_toml(path)
    parent = _required_table(data, "parent")
    design = load_policy_design(
        (path.parent / _required_str(parent, "design_config")).resolve()
    )
    supports = _required_table(data, "supports")
    spec = PPOBoundaryStudySpec(
        schema_version=_required_int(data, "schema_version"),
        study_id=_required_str(data, "study_id"),
        environment_id=_required_str(data, "environment_id"),
        perturbation_name=_required_str(data, "perturbation_name"),
        metric_version=_required_str(data, "metric_version"),
        theta_threshold_deg=_required_number(data, "theta_threshold_deg"),
        outer_rounds=_required_int(data, "outer_rounds"),
        training_reward=_required_str(data, "training_reward"),
        parent_policy_id=_required_str(parent, "policy_id"),
        parent_model_sha256=_required_str(parent, "model_sha256"),
        parent_design=design,
        supports={
            name: _recovery_support(_required_table(supports, name), name)
            for name in PPO_BOUNDARY_SUPPORTS
        },
        acceptance=_acceptance_spec(_required_table(data, "acceptance")),
        curriculum=_curriculum_spec(_required_table(data, "curriculum")),
        continuation=_continuation_spec(_required_table(data, "continuation")),
        seeds=_study_seed_spec(_required_table(data, "seeds")),
        tiers=tuple(_study_tier_spec(item) for item in _table_list(data, "tiers")),
        confidence=_confidence_spec(_required_table(data, "confidence")),
        storage=_storage_spec(_required_table(data, "storage")),
        sampling=_transition_sampling_spec(data.get("sampling")),
    )
    _validate_ppo_boundary_study(spec)
    return spec


def _recovery_support(data: dict[str, Any], name: str) -> RecoverySupport:
    points = data.get("extra_points", [])
    if not isinstance(points, list):
        raise ValueError(f"{name} extra_points must be a list.")
    extra_points = []
    for point in points:
        pair = _number_tuple(point, f"{name} extra_points")
        if len(pair) != 2:
            raise ValueError(f"{name} extra_points must be [angle, length] pairs.")
        extra_points.append((pair[0], pair[1]))
    return RecoverySupport(
        initial_theta_deg=_number_tuple(
            data.get("initial_theta_deg"), f"{name} initial_theta_deg"
        ),
        length=_number_tuple(data.get("length"), f"{name} length"),
        extra_points=tuple(extra_points),
    )


def _acceptance_spec(data: dict[str, Any]) -> AcceptanceSpec:
    return AcceptanceSpec(
        primary_tau=_required_number(data, "primary_tau"),
        secondary_taus=_number_tuple(data.get("secondary_taus"), "secondary_taus"),
        recovery_tail_steps=_required_int(data, "recovery_tail_steps"),
        recovery_rms_angle_deg=_required_number(data, "recovery_rms_angle_deg"),
    )


def _curriculum_spec(data: dict[str, Any]) -> CurriculumSpec:
    return CurriculumSpec(
        arms=_string_tuple(data.get("arms"), "arms"),
        discovery_episodes_per_condition=_required_int(
            data, "discovery_episodes_per_condition"
        ),
        target_rule=data.get("target_rule", PPO_BOUNDARY_PROBABILITY_BAND_RULE),
        mixture_lambda=_required_number(data, "mixture_lambda"),
        boundary_half_width=_optional_number(data.get("boundary_half_width")),
        pseudocount_success=_required_number(data, "pseudocount_success"),
        pseudocount_total=_required_number(data, "pseudocount_total"),
        thin_band_min_conditions=_required_int(data, "thin_band_min_conditions"),
    )


def _continuation_spec(data: dict[str, Any]) -> ContinuationSpec:
    return ContinuationSpec(
        learning_rate=_required_number(data, "learning_rate"),
        clip_range=_required_number(data, "clip_range"),
        device=_required_str(data, "device"),
        checkpoint_selection=_required_str(data, "checkpoint_selection"),
        vec_env_backend=_required_str(data, "vec_env_backend"),
        torch_threads=_required_int(data, "torch_threads"),
        hyperparameters=dict(_required_table(data, "hyperparameters")),
    )


def _transition_sampling_spec(
    data: dict[str, Any] | None,
) -> TransitionSamplingSpec | None:
    if data is None:
        return None
    return TransitionSamplingSpec(
        mode=_required_str(data, "mode"),
        calibration_episodes_per_condition=_required_int(
            data, "calibration_episodes_per_condition"
        ),
        calibration_seed_role=_required_int(data, "calibration_seed_role"),
        calibration_action_seed=_required_int(data, "calibration_action_seed"),
        length_ema_alpha=_required_number(data, "length_ema_alpha"),
        refresh_rollouts=_required_int(data, "refresh_rollouts"),
        minimum_length=_required_number(data, "minimum_length"),
    )


def _study_seed_spec(data: dict[str, Any]) -> StudySeedSpec:
    roles = _required_table(data, "roles")
    return StudySeedSpec(
        master_seed=_required_int(data, "master_seed"),
        training_policy_seed_base=_required_int(data, "training_policy_seed_base"),
        condition_stream_tag=_required_int(data, "condition_stream_tag"),
        roles={name: _required_int(roles, name) for name in roles},
    )


def _study_tier_spec(data: dict[str, Any]) -> StudyTierSpec:
    return StudyTierSpec(
        name=_required_str(data, "name"),
        replicate_ids=_int_tuple(data.get("replicate_ids"), "replicate_ids"),
        additional_transitions=_required_int(data, "additional_transitions"),
        snapshot_transitions=_int_tuple(
            data.get("snapshot_transitions"),
            "snapshot_transitions",
            allow_empty=True,
        ),
        evaluation_support=_required_str(data, "evaluation_support"),
        episodes_per_condition=_required_int(data, "episodes_per_condition"),
        seed_role=_required_str(data, "seed_role"),
        snapshot_evaluation=_required_str(data, "snapshot_evaluation"),
        arms=_string_tuple(data["arms"], "tier arms") if "arms" in data else None,
    )


def _confidence_spec(data: dict[str, Any]) -> ConfidenceSpec:
    return ConfidenceSpec(
        family_alpha=_required_number(data, "family_alpha"),
        interval=_required_str(data, "interval"),
    )


def _storage_spec(data: dict[str, Any]) -> EvaluationStorageSpec:
    return EvaluationStorageSpec(
        evaluation_layout=_required_str(data, "evaluation_layout"),
        compression=_required_str(data, "compression"),
        compression_level=_required_int(data, "compression_level"),
        episode_digest=_required_str(data, "episode_digest"),
    )


def _validate_ppo_boundary_study(spec: PPOBoundaryStudySpec) -> None:
    if spec.schema_version != 1:
        raise ValueError(f"Unsupported study schema {spec.schema_version!r}.")
    if spec.perturbation_name != CARTPOLE_RECOVERY_ANGLE_LENGTH:
        raise ValueError(f"Study requires {CARTPOLE_RECOVERY_ANGLE_LENGTH!r}.")
    if spec.environment_id != spec.parent_design.environment.environment_id:
        raise ValueError("Study environment does not match the parent design.")
    if spec.parent_design.policy_id != spec.parent_policy_id:
        raise ValueError("Parent design does not match parent policy_id.")
    if spec.outer_rounds != 1:
        raise ValueError("The first-phase protocol uses exactly one outer round.")
    _validate_recovery_metric(spec)
    _validate_supports(spec)
    _validate_curriculum(spec.curriculum)
    if (
        spec.curriculum.target_rule == PPO_BOUNDARY_FRONTIER_RULE
        and spec.supports["discovery"].extra_points
    ):
        raise ValueError("The frontier rule requires a complete discovery grid.")
    _validate_continuation(spec)
    _validate_seed_roles(spec.seeds)
    if spec.sampling is not None:
        sampling = spec.sampling
        if sampling.mode != "inverse-episode-length-v1":
            raise ValueError("Unknown transition sampling mode.")
        if sampling.calibration_episodes_per_condition <= 0:
            raise ValueError("Calibration episodes must be positive.")
        if not 1 <= sampling.calibration_seed_role <= 9 or (
            sampling.calibration_seed_role in spec.seeds.roles.values()
        ):
            raise ValueError("Calibration seed role must be disjoint from study roles.")
        if not 0 <= sampling.calibration_action_seed < 2**32 - 100_000:
            raise ValueError("Invalid calibration action seed.")
        if not 0 < sampling.length_ema_alpha <= 1:
            raise ValueError("Length EMA alpha must be in (0, 1].")
        if sampling.refresh_rollouts <= 0 or not 1 <= sampling.minimum_length <= 500:
            raise ValueError("Invalid sampling refresh or minimum length.")
        if (
            len(spec.supports["discovery"].initial_theta_deg)
            * len(spec.supports["discovery"].length)
            * sampling.calibration_episodes_per_condition
            >= 100_000
        ):
            raise ValueError("Calibration reset seed namespace would overflow.")
    _validate_tiers(spec)
    if not 0.0 < spec.confidence.family_alpha < 1.0:
        raise ValueError("family_alpha must lie strictly between 0 and 1.")


def _validate_recovery_metric(spec: PPOBoundaryStudySpec) -> None:
    acceptance = spec.acceptance
    if spec.metric_version != RECOVERY_METRIC_VERSION:
        raise ValueError(f"Study requires metric {RECOVERY_METRIC_VERSION!r}.")
    if acceptance.recovery_tail_steps != RECOVERY_TAIL_STEPS or not isclose(
        radians(acceptance.recovery_rms_angle_deg), RECOVERY_RMS_ANGLE_RADIANS
    ):
        raise ValueError("Acceptance definition differs from the recovery metric.")
    for tau in (acceptance.primary_tau, *acceptance.secondary_taus):
        if not 0.0 < tau < 1.0:
            raise ValueError("Acceptance thresholds must lie strictly in (0, 1).")


def _validate_supports(spec: PPOBoundaryStudySpec) -> None:
    for name, support in spec.supports.items():
        for values, label in (
            (support.initial_theta_deg, "initial_theta_deg"),
            (support.length, "length"),
        ):
            if list(values) != sorted(set(values)):
                raise ValueError(f"{name} {label} must be strictly increasing.")
        angles = (*support.initial_theta_deg, *(p[0] for p in support.extra_points))
        lengths = (*support.length, *(p[1] for p in support.extra_points))
        if any(abs(angle) >= spec.theta_threshold_deg for angle in angles):
            raise ValueError(f"{name} angles must lie inside the pole cutoff.")
        if any(length <= 0.0 for length in lengths):
            raise ValueError(f"{name} half-lengths must be positive.")
    discovery = spec.supports["discovery"]
    reference = spec.supports["reference"]
    if not set(discovery.initial_theta_deg) <= set(reference.initial_theta_deg):
        raise ValueError("Reference angles must include the discovery angles.")
    if discovery.length != reference.length:
        raise ValueError("Reference and discovery half-lengths must be identical.")


def _validate_curriculum(curriculum: CurriculumSpec) -> None:
    if curriculum.arms != PPO_BOUNDARY_ARMS:
        raise ValueError(f"Study arms must be {PPO_BOUNDARY_ARMS}.")
    if curriculum.discovery_episodes_per_condition <= 0:
        raise ValueError("discovery_episodes_per_condition must be positive.")
    if not 0.0 <= curriculum.mixture_lambda <= 1.0:
        raise ValueError("mixture_lambda must lie in [0, 1].")
    if curriculum.target_rule == PPO_BOUNDARY_PROBABILITY_BAND_RULE:
        if curriculum.boundary_half_width is None:
            raise ValueError("The probability-band rule needs boundary_half_width.")
        if curriculum.boundary_half_width <= 0.0:
            raise ValueError("boundary_half_width must be positive.")
    elif curriculum.target_rule == PPO_BOUNDARY_FRONTIER_RULE:
        if curriculum.boundary_half_width is not None:
            raise ValueError("The frontier rule does not use boundary_half_width.")
    else:
        raise ValueError(f"Unknown target_rule {curriculum.target_rule!r}.")
    if not 0.0 < curriculum.pseudocount_success < curriculum.pseudocount_total:
        raise ValueError("Target pseudocounts must keep the score inside (0, 1).")
    if curriculum.thin_band_min_conditions <= 0:
        raise ValueError("thin_band_min_conditions must be positive.")


def _validate_continuation(spec: PPOBoundaryStudySpec) -> None:
    continuation = spec.continuation
    if continuation.learning_rate <= 0.0 or continuation.clip_range <= 0.0:
        raise ValueError("Continuation learning_rate and clip_range must be positive.")
    if continuation.device != "cpu":
        raise ValueError("Study continuation requires device='cpu'.")
    if continuation.torch_threads <= 0:
        raise ValueError("torch_threads must be positive.")
    parent = spec.parent_design.hyperparameters
    changed = sorted(
        name
        for name, value in continuation.hyperparameters.items()
        if parent.get(name) != value and name not in PPO_BOUNDARY_TUNABLE_SETTINGS
    )
    if changed:
        raise ValueError(f"Continuation must preserve parent settings: {changed}.")
    if "n_envs" not in continuation.hyperparameters or (
        "n_steps" not in continuation.hyperparameters
    ):
        raise ValueError("Continuation must declare n_envs and n_steps.")


def _validate_seed_roles(seeds: StudySeedSpec) -> None:
    if set(seeds.roles) != set(PPO_BOUNDARY_SEED_ROLES):
        raise ValueError(f"Seed roles must be {PPO_BOUNDARY_SEED_ROLES}.")
    values = list(seeds.roles.values())
    if len(set(values)) != len(values) or any(not 1 <= value <= 9 for value in values):
        raise ValueError("Seed roles must be distinct integers from 1 to 9.")


def _validate_tiers(spec: PPOBoundaryStudySpec) -> None:
    names = tuple(tier.name for tier in spec.tiers)
    if names != PPO_BOUNDARY_TIERS:
        raise ValueError(f"Study tiers must be {PPO_BOUNDARY_TIERS}.")
    hyperparameters = spec.continuation.hyperparameters
    rollout = int(hyperparameters["n_envs"]) * int(hyperparameters["n_steps"])
    replicates = [item for tier in spec.tiers for item in tier.replicate_ids]
    if len(set(replicates)) != len(replicates):
        raise ValueError("Replicate IDs must be unique across tiers.")
    for tier in spec.tiers:
        if tier.arms is not None and (
            not tier.arms
            or len(set(tier.arms)) != len(tier.arms)
            or not set(tier.arms) <= set(spec.curriculum.arms)
        ):
            raise ValueError("Tier arms must be a nonempty unique curriculum subset.")
        if tier.evaluation_support not in spec.supports:
            raise ValueError(f"Tier {tier.name!r} has an unknown support.")
        if tier.seed_role not in spec.seeds.roles or tier.seed_role == "training":
            raise ValueError(f"Tier {tier.name!r} has an invalid seed role.")
        if tier.snapshot_evaluation not in PPO_BOUNDARY_SNAPSHOT_MODES:
            raise ValueError(f"Tier {tier.name!r} has an invalid snapshot mode.")
        if tier.episodes_per_condition <= 0:
            raise ValueError("episodes_per_condition must be positive.")
        budget = tier.additional_transitions
        if budget <= 0 or budget % rollout:
            raise ValueError(f"additional_transitions must be multiples of {rollout}.")
        snapshots = tier.snapshot_transitions
        if list(snapshots) != sorted(set(snapshots)) or any(
            value <= 0 or value >= budget or value % rollout for value in snapshots
        ):
            raise ValueError(f"Tier {tier.name!r} has invalid snapshot transitions.")
    if spec.tier("confirmatory").snapshot_evaluation != "none":
        raise ValueError("Confirmatory snapshots would change the confidence family.")


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as file:
            return tomllib.load(file)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Configuration does not exist: {path}.") from error


def _read_json(path: Path) -> dict[str, Any]:
    import json

    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Configuration does not exist: {path}.") from error
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be an object: {path}.")
    return value


def _required_str(data: dict[str, Any], name: str) -> str:
    return _optional_str(data.get(name), name, required=True)


def _optional_str(value: Any, name: str, *, required: bool = False) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _required_int(data: dict[str, Any], name: str) -> int:
    return _required_int_value(data.get(name), name)


def _required_number(data: dict[str, Any], name: str) -> float:
    value = data.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric.")
    return float(value)


def _number_tuple(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list of numbers.")
    result = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError(f"{name} must be a non-empty list of numbers.")
        result.append(float(item))
    return tuple(result)


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list of strings.")
    return tuple(_required_str({name: item}, name) for item in value)


def _int_tuple(value: Any, name: str, *, allow_empty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{name} must be a list of integers.")
    return tuple(_required_int_value(item, name) for item in value)


def _table_list(data: dict[str, Any], name: str) -> list[dict[str, Any]]:
    value = data.get(name)
    if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
        raise ValueError(f"{name} must be a TOML array of tables.")
    return value


def _required_int_value(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    return value


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _required_int_value(value, name)


def _required_table(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a TOML table.")
    return value


def _optional_number(value: Any) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("Quality gate values must be numeric.")
    return float(value)
