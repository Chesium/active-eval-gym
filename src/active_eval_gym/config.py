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

    data = _read_toml(path)
    seeds = tuple(
        _required_int_value(value, "seeds") for value in data.get("seeds", [])
    )
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty list of unique integers.")
    policy_ids = tuple(data.get("policy_ids", []))
    if not policy_ids or any(not isinstance(value, str) for value in policy_ids):
        raise ValueError("policy_ids must be a non-empty list of strings.")
    return SweepSuiteSpec(
        schema_version=_required_int(data, "schema_version"),
        suite_id=_required_str(data, "suite_id"),
        environment_id=_required_str(data, "environment_id"),
        perturbation_name=_required_str(data, "perturbation_name"),
        metric_version=_optional_str(data.get("metric_version"), "metric_version")
        or "episode-summary-v2",
        seeds=seeds,
        policy_ids=policy_ids,
        grid=dict(_required_table(data, "grid")),
    )


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as file:
            return tomllib.load(file)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Configuration does not exist: {path}.") from error


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
