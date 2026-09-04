"""Small TOML configuration loader for policy and evaluation specifications."""

import tomllib
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from active_eval_gym.envs.specs import package_versions
from active_eval_gym.models import NominalEnvSpec, PolicyDesignSpec


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
