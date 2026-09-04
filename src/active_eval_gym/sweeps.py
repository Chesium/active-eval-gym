"""Deterministic perturbation sweep collection and offline analysis."""

import json
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from active_eval_gym.config import SweepSuiteSpec
from active_eval_gym.envs.perturbations import (
    CARTPOLE_ACTION_DELAY,
    CARTPOLE_ACTION_DROPOUT,
    CARTPOLE_ANGLE_LENGTH,
    CARTPOLE_FORCE_MAGNITUDE,
    CARTPOLE_MASS,
    CARTPOLE_POLE_ANGLE_NOISE,
    CARTPOLE_RECOVERY_ANGLE_LENGTH,
    MINIGRID_START_POSE,
    PENDULUM_LENGTH,
    PerturbationSpec,
)
from active_eval_gym.evaluate import make_artifact_environment
from active_eval_gym.metrics import SWEEP_METRIC_VERSION, compute_saved_metrics
from active_eval_gym.models import NominalEnvSpec
from active_eval_gym.policies.artifacts import load_policy_artifact
from active_eval_gym.policies.sb3 import (
    SB3Policy,
    derive_antisymmetrized_cartpole_ppo,
)
from active_eval_gym.rollout import collect_episode
from active_eval_gym.serialization import (
    read_saved_episode,
    to_jsonable,
    write_episode,
    write_json_new,
)


@dataclass(frozen=True)
class SweepCondition:
    """One explicit point in a perturbation sampling grid."""

    condition_id: str
    perturbation: PerturbationSpec


def expand_sweep(
    suite: SweepSuiteSpec, nominal: NominalEnvSpec
) -> tuple[SweepCondition, ...]:
    """Expand a tracked grid into a deterministic ordered condition list."""

    if suite.environment_id != nominal.environment_id:
        raise ValueError(
            f"Suite {suite.suite_id!r} requires {suite.environment_id}, "
            f"not {nominal.environment_id}."
        )
    if suite.conditions:
        return tuple(
            SweepCondition(
                condition_id=item["condition_id"],
                perturbation=PerturbationSpec(
                    suite.perturbation_name, item["parameters"]
                ),
            )
            for item in suite.conditions
        )
    if suite.perturbation_name == CARTPOLE_ANGLE_LENGTH:
        _require_grid_keys(suite, {"delta_theta_deg", "length"})
        return tuple(
            SweepCondition(
                condition_id=(
                    f"theta-{_slug_number(angle)}_length-{_slug_number(length)}"
                ),
                perturbation=PerturbationSpec(
                    CARTPOLE_ANGLE_LENGTH,
                    {"delta_theta_deg": angle, "length": length},
                ),
            )
            for angle in _number_list(suite.grid["delta_theta_deg"], "delta_theta_deg")
            for length in _number_list(suite.grid["length"], "length")
        )
    if suite.perturbation_name == CARTPOLE_RECOVERY_ANGLE_LENGTH:
        _require_grid_keys(
            suite, {"initial_theta_deg", "length", "theta_threshold_deg"}
        )
        thresholds = _number_list(
            suite.grid["theta_threshold_deg"], "theta_threshold_deg"
        )
        if len(thresholds) != 1:
            raise ValueError("Recovery boundary suites require one fixed threshold.")
        threshold = thresholds[0]
        return tuple(
            SweepCondition(
                condition_id=(
                    f"theta-{_slug_number(angle)}_length-{_slug_number(length)}"
                ),
                perturbation=PerturbationSpec(
                    CARTPOLE_RECOVERY_ANGLE_LENGTH,
                    {
                        "initial_theta_deg": angle,
                        "length": length,
                        "theta_threshold_deg": threshold,
                    },
                ),
            )
            for angle in _number_list(
                suite.grid["initial_theta_deg"], "initial_theta_deg"
            )
            for length in _number_list(suite.grid["length"], "length")
        )
    scalar_cartpole_sweeps = {
        CARTPOLE_MASS: ("masspole", "mass"),
        CARTPOLE_POLE_ANGLE_NOISE: (
            "pole_angle_noise_std_deg",
            "noise-deg",
        ),
        CARTPOLE_FORCE_MAGNITUDE: ("force_mag", "force"),
        CARTPOLE_ACTION_DELAY: ("delay_steps", "delay"),
        CARTPOLE_ACTION_DROPOUT: ("dropout_probability", "dropout"),
    }
    if suite.perturbation_name in scalar_cartpole_sweeps:
        parameter, label = scalar_cartpole_sweeps[suite.perturbation_name]
        _require_grid_keys(suite, {parameter})
        values = (
            _integer_list(suite.grid[parameter], parameter)
            if parameter == "delay_steps"
            else _number_list(suite.grid[parameter], parameter)
        )
        return tuple(
            SweepCondition(
                condition_id=f"{label}-{_slug_number(value)}",
                perturbation=PerturbationSpec(
                    suite.perturbation_name, {parameter: value}
                ),
            )
            for value in values
        )
    if suite.perturbation_name == PENDULUM_LENGTH:
        _require_grid_keys(suite, {"l"})
        return tuple(
            SweepCondition(
                condition_id=f"length-{_slug_number(length)}",
                perturbation=PerturbationSpec(PENDULUM_LENGTH, {"l": length}),
            )
            for length in _number_list(suite.grid["l"], "l")
        )
    if suite.perturbation_name == MINIGRID_START_POSE:
        _require_grid_keys(suite, {"agent_start_pos", "agent_start_dir"})
        if suite.grid["agent_start_pos"] != "all_valid_non_wall_non_goal":
            raise ValueError(
                "MiniGrid agent_start_pos must be 'all_valid_non_wall_non_goal'."
            )
        directions = _integer_list(suite.grid["agent_start_dir"], "agent_start_dir")
        if any(direction not in range(4) for direction in directions):
            raise ValueError("MiniGrid start directions must be in 0..3.")
        width = int(nominal.parameters["width"])
        height = int(nominal.parameters["height"])
        goal = (width - 2, height - 2)
        positions = [
            (x, y)
            for x in range(1, width - 1)
            for y in range(1, height - 1)
            if (x, y) != goal
        ]
        return tuple(
            SweepCondition(
                condition_id=f"x-{x:02d}_y-{y:02d}_dir-{direction}",
                perturbation=PerturbationSpec(
                    MINIGRID_START_POSE,
                    {"agent_start_pos": [x, y], "agent_start_dir": direction},
                ),
            )
            for x, y in positions
            for direction in directions
        )
    raise ValueError(
        f"Suite {suite.suite_id!r} uses unsupported perturbation "
        f"{suite.perturbation_name!r}."
    )


def collect_sweep(
    suite: SweepSuiteSpec, *, artifact_root: Path, output_dir: Path
) -> dict[str, Any]:
    """Collect raw trajectories for every frozen policy/condition/seed."""

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite sweep directory: {output_dir}.")
    prepared: list[tuple[str, Any, Any]] = []
    nominal = None
    for policy_id in suite.policy_ids:
        if policy_id in suite.derived_policies:
            policy, metadata = _load_derived_sweep_policy(
                policy_id,
                suite.derived_policies[policy_id],
                artifact_root=artifact_root,
            )
        else:
            policy, metadata = load_policy_artifact(
                artifact_root / policy_id, require_frozen=True
            )
        if metadata.policy_id != policy_id:
            raise ValueError(
                f"Expected artifact {policy_id!r}, loaded {metadata.policy_id!r}."
            )
        policy_nominal = metadata.design_spec.environment
        if policy_nominal.environment_id != suite.environment_id:
            raise ValueError(
                f"Policy {policy_id!r} requires {policy_nominal.environment_id}, "
                f"not {suite.environment_id}."
            )
        if nominal is None:
            nominal = policy_nominal
        elif policy_nominal != nominal:
            raise ValueError("Paired policies do not share the same nominal spec.")
        prepared.append((policy_id, policy, metadata))
    if nominal is None:
        raise ValueError("Sweep has no policies.")
    conditions = expand_sweep(suite, nominal)

    output_dir.mkdir(parents=True)
    suite_record = {
        "schema_version": 1,
        "suite": suite,
        "nominal_environment": nominal,
        "conditions": [
            {
                "condition_id": condition.condition_id,
                "perturbation": condition.perturbation,
            }
            for condition in conditions
        ],
        "artifacts": {
            policy_id: {
                "model_sha256": metadata.model_sha256,
                "design_id": metadata.design_spec.design_id,
            }
            for policy_id, _, metadata in prepared
        },
    }
    write_json_new(output_dir / "suite.json", suite_record)

    episode_count = 0
    for condition in conditions:
        for seed in suite.seeds:
            paired: list[tuple[str, Any, Any]] = []
            for policy_id, policy, metadata in prepared:
                env, resolved = make_artifact_environment(
                    metadata, condition.perturbation
                )
                try:
                    episode = collect_episode(
                        env,
                        policy,
                        policy_artifact=metadata,
                        nominal_environment=nominal,
                        resolved_environment=resolved,
                        episode_seed=seed,
                        deterministic=True,
                    )
                finally:
                    env.close()
                paired.append((policy_id, resolved, episode))
            _validate_paired_environments(paired, condition.condition_id, seed)
            for policy_id, _, episode in paired:
                episode_dir = (
                    output_dir
                    / "episodes"
                    / policy_id
                    / condition.condition_id
                    / f"seed-{seed:03d}"
                )
                write_episode(episode_dir, episode)
                episode_count += 1

    result = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "condition_count": len(conditions),
        "episode_count": episode_count,
    }
    write_json_new(output_dir / "collection.json", result)
    return result


def analyze_sweep(evaluation_dir: Path) -> dict[str, Any]:
    """Compute versioned metrics and aggregates from saved raw trajectories."""

    suite_record = json.loads((evaluation_dir / "suite.json").read_text())
    suite_data = suite_record["suite"]
    metric_version = suite_data.get("metric_version", SWEEP_METRIC_VERSION)
    analysis_dir = evaluation_dir / "analysis" / metric_version
    if analysis_dir.exists():
        raise FileExistsError(f"Refusing to overwrite metric analysis: {analysis_dir}.")

    records: list[dict[str, Any]] = []
    for condition in suite_record["conditions"]:
        condition_id = condition["condition_id"]
        for policy_id in suite_data["policy_ids"]:
            for seed in suite_data["seeds"]:
                episode_dir = (
                    evaluation_dir
                    / "episodes"
                    / policy_id
                    / condition_id
                    / f"seed-{seed:03d}"
                )
                episode = read_saved_episode(episode_dir)
                metrics = compute_saved_metrics(episode, metric_version=metric_version)
                records.append(
                    {
                        "condition_id": condition_id,
                        "parameters": condition["perturbation"]["parameters"],
                        "policy_id": policy_id,
                        "seed": seed,
                        "metrics": to_jsonable(metrics),
                    }
                )

    analysis_dir.mkdir(parents=True)
    for record in records:
        path = (
            analysis_dir
            / "episodes"
            / record["policy_id"]
            / record["condition_id"]
            / f"seed-{record['seed']:03d}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_new(path, record["metrics"])

    conditions = []
    for condition in suite_record["conditions"]:
        condition_id = condition["condition_id"]
        by_policy = {
            policy_id: _aggregate(
                [
                    record["metrics"]
                    for record in records
                    if record["condition_id"] == condition_id
                    and record["policy_id"] == policy_id
                ]
            )
            for policy_id in suite_data["policy_ids"]
        }
        item = {
            "condition_id": condition_id,
            "parameters": condition["perturbation"]["parameters"],
            "policies": by_policy,
        }
        if len(suite_data["policy_ids"]) == 2:
            item["paired_difference"] = _paired_difference(
                records, condition_id, suite_data["policy_ids"]
            )
        elif len(suite_data["policy_ids"]) > 2:
            baseline = suite_data["policy_ids"][0]
            item["paired_differences"] = {
                f"{policy_id}_minus_{baseline}": _paired_difference(
                    records, condition_id, [baseline, policy_id]
                )
                for policy_id in suite_data["policy_ids"][1:]
            }
        conditions.append(item)

    summary = {
        "schema_version": 1,
        "metric_version": metric_version,
        "perturbation_name": suite_data.get("perturbation_name", "unknown"),
        "suite_id": suite_data["suite_id"],
        "environment_id": suite_data["environment_id"],
        "policy_ids": suite_data["policy_ids"],
        "seeds": suite_data["seeds"],
        "episode_count": len(records),
        "conditions": conditions,
    }
    write_json_new(analysis_dir / "summary.json", summary)
    return summary


def _validate_paired_environments(
    paired: list[tuple[str, Any, Any]], condition_id: str, seed: int
) -> None:
    if len(paired) < 2:
        return
    expected_resolved = to_jsonable(paired[0][1])
    expected_initial = to_jsonable(paired[0][2].metadata.initial_state)
    expected_randomness = _randomness_trace(paired[0][2])
    for policy_id, resolved, episode in paired[1:]:
        if to_jsonable(resolved) != expected_resolved:
            raise RuntimeError(
                f"Resolved environment mismatch for {policy_id}, "
                f"{condition_id}, seed {seed}."
            )
        if to_jsonable(episode.metadata.initial_state) != expected_initial:
            raise RuntimeError(
                f"Initial-state mismatch for {policy_id}, {condition_id}, seed {seed}."
            )
        actual_randomness = _randomness_trace(episode)
        shared_length = min(len(expected_randomness), len(actual_randomness))
        if expected_randomness[:shared_length] != actual_randomness[:shared_length]:
            raise RuntimeError(
                f"Perturbation-randomness mismatch for {policy_id}, "
                f"{condition_id}, seed {seed}."
            )


def _load_derived_sweep_policy(
    policy_id: str,
    spec: dict[str, Any],
    *,
    artifact_root: Path,
) -> tuple[Any, Any]:
    expected_keys = {"kind", "source_policy_id", "source_model_sha256"}
    if set(spec) != expected_keys:
        raise ValueError(
            f"Derived policy {policy_id!r} requires keys {sorted(expected_keys)!r}."
        )
    if spec["kind"] != "antisymmetrized-cartpole-ppo-v1":
        raise ValueError(f"Unsupported derived policy kind {spec['kind']!r}.")
    source_policy_id = spec["source_policy_id"]
    if not isinstance(source_policy_id, str) or not source_policy_id:
        raise ValueError("Derived source_policy_id must be a non-empty string.")
    source_policy, source_metadata = load_policy_artifact(
        artifact_root / source_policy_id, require_frozen=True
    )
    if source_metadata.model_sha256 != spec["source_model_sha256"]:
        raise ValueError(
            f"Derived policy {policy_id!r} source model hash does not match."
        )
    if not isinstance(source_policy, SB3Policy):
        raise TypeError("Antisymmetrization requires an SB3 PPO source policy.")
    return derive_antisymmetrized_cartpole_ppo(
        source_policy,
        source_metadata,
        derived_policy_id=policy_id,
    )


def _randomness_trace(episode: Any) -> list[dict[str, Any]]:
    records = [episode.reset.perturbation_diagnostics]
    records.extend(step.perturbation_diagnostics for step in episode.transitions)
    keys = ("pole_angle_noise_radians", "random_draw", "dropout_event")
    return [
        {key: record[key] for key in keys if key in record}
        for record in records
        if any(key in record for key in keys)
    ]


def _aggregate(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        raise ValueError("Cannot aggregate an empty metric list.")
    successes = [
        value for metric in metrics if (value := metric["task_success"]) is not None
    ]
    result: dict[str, Any] = {
        "episode_count": len(metrics),
        "success_rate": None if not successes else fmean(successes),
    }
    if successes:
        result["success_count"] = sum(bool(value) for value in successes)
        result["success_rate_wilson_95"] = _wilson_interval(
            [bool(value) for value in successes]
        )
    if all("recovery_success" in metric for metric in metrics):
        recovery = [bool(metric["recovery_success"]) for metric in metrics]
        result["recovery_count"] = sum(recovery)
        result["recovery_rate"] = fmean(recovery)
        result["recovery_rate_wilson_95"] = _wilson_interval(recovery)
        causes = ("none", "angle_limit", "cart_limit", "both", "unknown")
        result["failure_cause_counts"] = {
            cause: sum(metric["failure_cause"] == cause for metric in metrics)
            for cause in causes
        }
    for name in ("episode_return", "episode_length"):
        result[name] = _summary([float(metric[name]) for metric in metrics])
    environment_names = sorted(
        {
            name
            for metric in metrics
            for name, value in metric["environment_metrics"].items()
            if not isinstance(value, dict)
        }
    )
    result["environment_metrics"] = {}
    for name in environment_names:
        values = [
            float(metric["environment_metrics"][name])
            for metric in metrics
            if metric["environment_metrics"].get(name) is not None
        ]
        result["environment_metrics"][name] = None if not values else _summary(values)
    action_names = sorted(
        {
            name
            for metric in metrics
            for name in metric["environment_metrics"].get("action_counts", {})
        }
    )
    if action_names:
        result["environment_metrics"]["action_counts"] = {
            name: _summary(
                [
                    float(metric["environment_metrics"]["action_counts"][name])
                    for metric in metrics
                ]
            )
            for name in action_names
        }
    return result


def _paired_difference(
    records: list[dict[str, Any]], condition_id: str, policy_ids: list[str]
) -> dict[str, Any]:
    first_id, second_id = policy_ids
    first = {
        record["seed"]: _flat_numeric_metrics(record["metrics"])
        for record in records
        if record["condition_id"] == condition_id and record["policy_id"] == first_id
    }
    second = {
        record["seed"]: _flat_numeric_metrics(record["metrics"])
        for record in records
        if record["condition_id"] == condition_id and record["policy_id"] == second_id
    }
    if first.keys() != second.keys():
        raise RuntimeError(f"Paired seeds differ for condition {condition_id}.")
    names = sorted(set.intersection(*(set(values) for values in first.values())))
    return {
        "order": f"{second_id} minus {first_id}",
        "metrics": {
            name: _summary(
                [second[seed][name] - first[seed][name] for seed in sorted(first)]
            )
            for name in names
        },
    }


def _flat_numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    result = {
        "episode_return": float(metrics["episode_return"]),
        "episode_length": float(metrics["episode_length"]),
    }
    if metrics["task_success"] is not None:
        result["task_success"] = float(metrics["task_success"])
    if "recovery_success" in metrics:
        result["recovery_success"] = float(metrics["recovery_success"])
    for name, value in metrics["environment_metrics"].items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[name] = float(value)
    return result


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": fmean(values),
        "standard_deviation": pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _wilson_interval(values: list[bool], z: float = 1.96) -> dict[str, float]:
    count = len(values)
    rate = sum(values) / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    radius = (
        z
        * sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count))
        / denominator
    )
    return {"lower": max(0.0, center - radius), "upper": min(1.0, center + radius)}


def _require_grid_keys(suite: SweepSuiteSpec, expected: set[str]) -> None:
    actual = set(suite.grid)
    if actual != expected:
        raise ValueError(
            f"Suite {suite.suite_id!r} requires grid keys {sorted(expected)!r}; "
            f"received {sorted(actual)!r}."
        )


def _number_list(value: Any, name: str) -> tuple[float | int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(item, (int, float)) or isinstance(item, bool)
            for item in value
        )
    ):
        raise ValueError(f"{name} must be a non-empty numeric list.")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} values must be unique.")
    return tuple(value)


def _integer_list(value: Any, name: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise ValueError(f"{name} must be a non-empty integer list.")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} values must be unique.")
    return tuple(value)


def _slug_number(value: float | int) -> str:
    number = float(value)
    sign = "m" if number < 0 else "p"
    body = format(abs(number), ".12g").replace(".", "p")
    return sign + body
