"""Nominal multi-seed evaluation and policy freeze gates."""

import json
from dataclasses import asdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from active_eval_gym.config import NominalSuiteSpec, QualityGate
from active_eval_gym.envs.factory import make_environment
from active_eval_gym.envs.perturbations import NO_OP, PerturbationSpec
from active_eval_gym.envs.specs import (
    apply_observation_adapter,
    capture_resolved_environment,
    package_versions,
)
from active_eval_gym.metrics import EpisodeMetrics, compute_metrics
from active_eval_gym.policies.artifacts import (
    FREEZE_FAILURE_FILENAME,
    FREEZE_FILENAME,
    load_policy_artifact,
    write_freeze_result,
)
from active_eval_gym.rollout import collect_episode
from active_eval_gym.serialization import to_jsonable, write_episode, write_metrics


def freeze_candidate(artifact_dir: Path, suite: NominalSuiteSpec) -> bool:
    """Run the fixed nominal gate and freeze a passing candidate."""

    existing_results = [
        name
        for name in (FREEZE_FILENAME, FREEZE_FAILURE_FILENAME)
        if (artifact_dir / name).exists()
    ]
    if existing_results:
        raise FileExistsError(
            f"Refusing to repeat freeze evaluation for {artifact_dir}: "
            f"{', '.join(existing_results)} already exists."
        )
    policy, metadata = load_policy_artifact(artifact_dir, require_frozen=False)
    try:
        gate = suite.gates[metadata.policy_id]
    except KeyError as error:
        raise ValueError(
            f"Suite {suite.suite_id!r} has no gate for {metadata.policy_id!r}."
        ) from error
    metrics = [
        _evaluate_once(policy, metadata, seed, persist_dir=None) for seed in suite.seeds
    ]
    summary = summarize_metrics(metrics)
    passed, failures = check_quality_gate(summary, gate)
    results = {**summary, "failures": failures, "seeds": list(suite.seeds)}
    write_freeze_result(
        artifact_dir,
        passed=passed,
        gate=asdict(gate),
        results=results,
    )
    return passed


def evaluate_nominal_suite(
    suite: NominalSuiteSpec,
    *,
    artifact_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate every frozen artifact and persist full episode bundles."""

    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite evaluation directory: {output_dir}."
        )

    # Verify the complete suite before writing anything. In particular, a missing
    # or unfrozen policy must not leave behind a directory that resembles a valid
    # partial evaluation.
    prepared: list[tuple[str, Any, Any]] = []
    for policy_id in suite.policy_ids:
        artifact_dir = artifact_root / policy_id
        policy, metadata = load_policy_artifact(artifact_dir, require_frozen=True)
        if metadata.policy_id != policy_id:
            raise ValueError(
                f"Expected artifact {policy_id!r}, loaded {metadata.policy_id!r}."
            )
        env, _ = make_artifact_environment(metadata)
        env.close()
        prepared.append((policy_id, policy, metadata))

    output_dir.mkdir(parents=True)
    _write_json_new(output_dir / "suite.json", suite)
    summaries: dict[str, Any] = {}
    for policy_id, policy, metadata in prepared:
        metrics = []
        for seed in suite.seeds:
            episode_dir = output_dir / policy_id / f"seed-{seed:03d}"
            metrics.append(_evaluate_once(policy, metadata, seed, episode_dir))
        summary = summarize_metrics(metrics)
        passed, failures = check_quality_gate(summary, suite.gates[policy_id])
        summaries[policy_id] = {
            **summary,
            "quality_gate_passed": passed,
            "quality_gate_failures": failures,
        }
    result = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "seeds": list(suite.seeds),
        "policies": summaries,
    }
    _write_json_new(output_dir / "summary.json", result)
    return result


def summarize_metrics(metrics: list[EpisodeMetrics]) -> dict[str, Any]:
    """Aggregate basic nominal behavior without discarding per-episode data."""

    if not metrics:
        raise ValueError("Cannot summarize an empty metric list.")
    returns = [metric.episode_return for metric in metrics]
    lengths = [metric.episode_length for metric in metrics]
    successes = [
        metric.task_success for metric in metrics if metric.task_success is not None
    ]
    return {
        "episode_count": len(metrics),
        "return": _summary_values(returns),
        "episode_length": _summary_values(lengths),
        "success_rate": None if not successes else fmean(successes),
    }


def check_quality_gate(
    summary: dict[str, Any], gate: QualityGate
) -> tuple[bool, list[str]]:
    """Evaluate one declared gate and return human-readable failures."""

    failures: list[str] = []
    success_rate = summary["success_rate"]
    mean_length = summary["episode_length"]["mean"]
    mean_return = summary["return"]["mean"]
    if gate.minimum_success_rate is not None:
        if success_rate is None or success_rate < gate.minimum_success_rate:
            failures.append(
                f"success_rate {success_rate!r} < {gate.minimum_success_rate}"
            )
    if (
        gate.minimum_mean_episode_length is not None
        and mean_length < gate.minimum_mean_episode_length
    ):
        failures.append(
            f"mean episode length {mean_length} < {gate.minimum_mean_episode_length}"
        )
    if (
        gate.maximum_mean_episode_length is not None
        and mean_length > gate.maximum_mean_episode_length
    ):
        failures.append(
            f"mean episode length {mean_length} > {gate.maximum_mean_episode_length}"
        )
    if gate.minimum_mean_return is not None and mean_return < gate.minimum_mean_return:
        failures.append(f"mean return {mean_return} < {gate.minimum_mean_return}")
    return not failures, failures


def make_artifact_environment(
    metadata: Any,
    perturbation: PerturbationSpec = NO_OP,
    *,
    render_mode: str | None = None,
):
    """Construct the exact nominal policy-facing environment."""

    design = metadata.design_spec
    nominal = design.environment
    installed_versions = package_versions(nominal.environment_id)
    for package, installed in installed_versions.items():
        recorded = design.environment_package_versions.get(package)
        if recorded != installed:
            raise ValueError(
                f"Policy {metadata.policy_id!r} requires {package}=={recorded}, "
                f"but evaluation has {installed}."
            )
    env = make_environment(
        nominal.environment_id, perturbation=perturbation, render_mode=render_mode
    )
    try:
        env = apply_observation_adapter(env, design.observation_adapter)
        resolved = capture_resolved_environment(
            env, nominal, design.observation_adapter, perturbation
        )
        return env, resolved
    except Exception:
        env.close()
        raise


def _evaluate_once(
    policy: Any,
    metadata: Any,
    seed: int,
    persist_dir: Path | None,
) -> EpisodeMetrics:
    env, resolved = make_artifact_environment(metadata)
    try:
        episode = collect_episode(
            env,
            policy,
            policy_artifact=metadata,
            nominal_environment=metadata.design_spec.environment,
            resolved_environment=resolved,
            episode_seed=seed,
            deterministic=True,
        )
    finally:
        env.close()
    if persist_dir is None:
        return compute_metrics(episode, source_trajectory_sha256="in-memory-validation")
    trajectory_hash = write_episode(persist_dir, episode)
    metrics = compute_metrics(episode, source_trajectory_sha256=trajectory_hash)
    write_metrics(persist_dir, metrics)
    return metrics


def _summary_values(values: list[float] | list[int]) -> dict[str, float]:
    return {
        "mean": fmean(values),
        "standard_deviation": pstdev(values),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _write_json_new(path: Path, value: Any) -> None:
    content = (
        json.dumps(to_jsonable(value), allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    try:
        with path.open("x", encoding="utf-8") as file:
            file.write(content)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite artifact: {path}.") from error
