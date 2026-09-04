"""Versioned CartPole policy-symmetry diagnostics and interventions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from math import radians
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from active_eval_gym.config import CartPoleSymmetrySuiteSpec
from active_eval_gym.envs.perturbations import (
    CARTPOLE_ANGLE_LENGTH,
    CARTPOLE_FIXED_INITIAL_STATE,
    PerturbationSpec,
)
from active_eval_gym.evaluate import make_artifact_environment
from active_eval_gym.metrics import (
    SWEEP_METRIC_VERSION,
    compute_metrics,
    compute_saved_metrics,
)
from active_eval_gym.models import EpisodeRecord, PolicyArtifactMetadata
from active_eval_gym.policies.artifacts import load_policy_artifact
from active_eval_gym.policies.sb3 import (
    AntisymmetrizedCartPolePPOPolicy,
    SB3Policy,
)
from active_eval_gym.rollout import collect_episode
from active_eval_gym.serialization import (
    read_saved_episode,
    to_jsonable,
    write_episode,
    write_json_new,
    write_metrics,
)

SYMMETRY_AUDIT_VERSION = "cartpole-policy-symmetry-audit-v1"
SYMMETRY_STUDY_VERSION = "cartpole-symmetry-study-v1"
SYMMETRIZATION = "antisymmetrized-binary-logit-margin-v1"
STATE_ORDER = (
    "cart_position",
    "cart_velocity",
    "pole_angle",
    "pole_angular_velocity",
)


def run_cartpole_symmetry_study(
    suite: CartPoleSymmetrySuiteSpec,
    *,
    artifact_root: Path,
    source_evaluation: Path,
    output_dir: Path,
    figure_path: Path,
) -> dict[str, Any]:
    """Run checkpoint probes, mirror pairs, and a derived-policy intervention."""

    if suite.environment_id != "CartPole-v1":
        raise ValueError("The symmetry study supports only CartPole-v1.")
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite symmetry study directory: {output_dir}."
        )
    if figure_path.exists():
        raise FileExistsError(f"Refusing to overwrite symmetry figure: {figure_path}.")

    source_policy, source_metadata = load_policy_artifact(
        artifact_root / suite.policy_id, require_frozen=True
    )
    control_policy, control_metadata = load_policy_artifact(
        artifact_root / suite.control_policy_id, require_frozen=True
    )
    if not isinstance(source_policy, SB3Policy):
        raise TypeError("The source symmetry policy must be a Stable-Baselines3 PPO.")
    for metadata in (source_metadata, control_metadata):
        if metadata.design_spec.environment.environment_id != suite.environment_id:
            raise ValueError(
                f"Policy {metadata.policy_id!r} does not use {suite.environment_id}."
            )
    if (
        source_metadata.design_spec.environment
        != control_metadata.design_spec.environment
    ):
        raise ValueError("Source and control policies must share one nominal spec.")

    source_data = _load_source_evaluation(suite, source_evaluation, source_metadata)
    base_states = _seeded_nominal_states(source_metadata, suite.seeds)
    derived_policy = AntisymmetrizedCartPolePPOPolicy(source_policy)
    derived_metadata = _derived_metadata(
        source_metadata, derived_policy_id=suite.derived_policy_id
    )

    audit = _checkpoint_audit(
        source_policy,
        resets=source_data["resets"],
        observed=source_data["observed"],
        observed_stride=suite.observed_stride,
        nominal=source_metadata.design_spec.environment,
    )

    output_dir.mkdir(parents=True)
    write_json_new(
        output_dir / "study.json",
        {
            "schema_version": 1,
            "study_version": SYMMETRY_STUDY_VERSION,
            "suite": suite,
            "source_evaluation": str(source_evaluation),
            "artifacts": {
                source_metadata.policy_id: _artifact_identity(source_metadata),
                control_metadata.policy_id: _artifact_identity(control_metadata),
            },
            "derived_policy": to_jsonable(derived_metadata),
        },
    )
    write_json_new(output_dir / "audit.json", audit)
    _plot_decision_boundary(source_policy, suite, figure_path)

    mirror = _collect_mirror_pairs(
        suite,
        policies=(
            (control_policy, control_metadata),
            (source_policy, source_metadata),
        ),
        base_states=base_states,
        output_dir=output_dir / "mirror-pairs",
    )
    causal = _collect_causal_intervention(
        suite,
        policy=derived_policy,
        metadata=derived_metadata,
        source_metrics=source_data["metrics"],
        source_initial_states=source_data["initial_states"],
        output_dir=output_dir / "causal-intervention",
    )
    result = {
        "schema_version": 1,
        "study_version": SYMMETRY_STUDY_VERSION,
        "suite_id": suite.suite_id,
        "audit": audit,
        "mirror_pairs": mirror,
        "causal_intervention": causal,
        "figure": str(figure_path),
    }
    write_json_new(output_dir / "summary.json", result)
    return result


def _load_source_evaluation(
    suite: CartPoleSymmetrySuiteSpec,
    evaluation_dir: Path,
    metadata: PolicyArtifactMetadata,
) -> dict[str, Any]:
    suite_path = evaluation_dir / "suite.json"
    try:
        source_suite = json.loads(suite_path.read_text())
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Source angle-length suite is missing: {suite_path}."
        ) from error
    artifact = source_suite.get("artifacts", {}).get(suite.policy_id, {})
    if artifact.get("model_sha256") != metadata.model_sha256:
        raise ValueError("Source evaluation does not match the frozen PPO model hash.")

    resets: list[Any] = []
    observed: list[Any] = []
    metrics: dict[tuple[str, int], Any] = {}
    initial_states: dict[tuple[str, int], Any] = {}
    for angle in suite.signed_angles_deg:
        for length in suite.lengths:
            condition_id = _angle_length_condition_id(angle, length)
            for seed in suite.seeds:
                episode_dir = (
                    evaluation_dir
                    / "episodes"
                    / suite.policy_id
                    / condition_id
                    / f"seed-{seed:03d}"
                )
                episode = read_saved_episode(episode_dir)
                recorded_hash = episode.metadata["policy_artifact"]["model_sha256"]
                if recorded_hash != metadata.model_sha256:
                    raise ValueError(f"Model hash mismatch in {episode_dir}.")
                resets.append(episode.reset["observation"])
                policy_inputs = [episode.reset["observation"]]
                policy_inputs.extend(
                    row["observation"] for row in episode.transitions[:-1]
                )
                observed.extend(policy_inputs[:: suite.observed_stride])
                key = (condition_id, seed)
                metrics[key] = compute_saved_metrics(
                    episode, metric_version=SWEEP_METRIC_VERSION
                )
                initial_states[key] = episode.metadata["initial_state"]
    return {
        "resets": np.asarray(resets, dtype=np.float32),
        "observed": np.asarray(observed, dtype=np.float32),
        "metrics": metrics,
        "initial_states": initial_states,
    }


def _checkpoint_audit(
    policy: SB3Policy,
    *,
    resets: np.ndarray,
    observed: np.ndarray,
    observed_stride: int,
    nominal: Any,
) -> dict[str, Any]:
    origin = np.zeros((1, 4), dtype=np.float32)
    logits, probabilities, values = policy.cartpole_actor_critic(origin)
    margin = float(logits[0, 1] - logits[0, 0])
    axis_limits = (
        float(nominal.parameters["x_threshold"]),
        3.0,
        float(nominal.parameters["theta_threshold_radians"]),
        3.0,
    )
    axes: dict[str, Any] = {}
    for index, (name, limit) in enumerate(zip(STATE_ORDER, axis_limits, strict=True)):
        values_axis = np.linspace(-limit, limit, 401, dtype=np.float32)
        states = np.zeros((len(values_axis), 4), dtype=np.float32)
        states[:, index] = values_axis
        axis_logits, axis_probabilities, _ = policy.cartpole_actor_critic(states)
        margins = axis_logits[:, 1] - axis_logits[:, 0]
        crossings = []
        for crossing in np.flatnonzero(margins[:-1] * margins[1:] <= 0.0):
            crossings.append(
                [float(values_axis[crossing]), float(values_axis[crossing + 1])]
            )
        axes[name] = {
            "range": [-limit, limit],
            "action_1_probability_at_negative_limit": float(axis_probabilities[0, 1]),
            "action_1_probability_at_zero": float(axis_probabilities[200, 1]),
            "action_1_probability_at_positive_limit": float(axis_probabilities[-1, 1]),
            "decision_boundary_intervals": crossings,
        }
    return {
        "schema_version": 1,
        "audit_version": SYMMETRY_AUDIT_VERSION,
        "reflection": {
            "state": "s -> -s",
            "action": "0 <-> 1",
            "probability_identity": "pi(1|s) = 1 - pi(1|-s)",
            "logit_margin_identity": "d(s) = -d(-s)",
            "value_identity": "V(s) = V(-s)",
        },
        "origin": {
            "logits": logits[0].tolist(),
            "action_probabilities": probabilities[0].tolist(),
            "logit_margin": margin,
            "value": float(values[0]),
        },
        "probe_sets": {
            "angle_length_resets": _symmetry_summary(policy, resets),
            "angle_length_policy_inputs": {
                **_symmetry_summary(policy, observed),
                "trajectory_stride": observed_stride,
            },
        },
        "axis_slices": axes,
    }


def _symmetry_summary(policy: SB3Policy, states: np.ndarray) -> dict[str, Any]:
    logits, probabilities, values = policy.cartpole_actor_critic(states)
    reflected_logits, reflected_probabilities, reflected_values = (
        policy.cartpole_actor_critic(-states)
    )
    margins = logits[:, 1] - logits[:, 0]
    reflected_margins = reflected_logits[:, 1] - reflected_logits[:, 0]
    probability_error = np.abs(
        probabilities[:, 1] + reflected_probabilities[:, 1] - 1.0
    )
    logit_error = np.abs(margins + reflected_margins)
    value_error = np.abs(values - reflected_values)
    decision_violations = (probabilities[:, 1] >= 0.5) == (
        reflected_probabilities[:, 1] >= 0.5
    )
    return {
        "state_count": len(states),
        "state_sha256": hashlib.sha256(states.tobytes(order="C")).hexdigest(),
        "mean_absolute_probability_error": float(np.mean(probability_error)),
        "median_absolute_probability_error": float(np.median(probability_error)),
        "p95_absolute_probability_error": float(np.quantile(probability_error, 0.95)),
        "maximum_absolute_probability_error": float(np.max(probability_error)),
        "mean_absolute_logit_margin_error": float(np.mean(logit_error)),
        "p95_absolute_logit_margin_error": float(np.quantile(logit_error, 0.95)),
        "deterministic_action_violation_rate": float(np.mean(decision_violations)),
        "mean_absolute_value_error": float(np.mean(value_error)),
        "p95_absolute_value_error": float(np.quantile(value_error, 0.95)),
    }


def _plot_decision_boundary(
    policy: SB3Policy,
    suite: CartPoleSymmetrySuiteSpec,
    output_path: Path,
) -> None:
    points = suite.plot_grid_points
    theta = np.linspace(-radians(12.0), radians(12.0), points, dtype=np.float32)
    omega_limit = suite.plot_angular_velocity_limit
    angular_velocity = np.linspace(-omega_limit, omega_limit, points, dtype=np.float32)
    theta_grid, omega_grid = np.meshgrid(theta, angular_velocity)
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), sharex=True, sharey=True)
    top_image = None
    bottom_image = None
    for column, cart_velocity in enumerate(suite.plot_cart_velocities):
        states = np.zeros((points * points, 4), dtype=np.float32)
        states[:, 1] = cart_velocity
        states[:, 2] = theta_grid.ravel()
        states[:, 3] = omega_grid.ravel()
        logits, probabilities, _ = policy.cartpole_actor_critic(states)
        reflected_logits, reflected_probabilities, _ = policy.cartpole_actor_critic(
            -states
        )
        margins = (logits[:, 1] - logits[:, 0]).reshape(points, points)
        reflected_margins = (reflected_logits[:, 1] - reflected_logits[:, 0]).reshape(
            points, points
        )
        p_right = probabilities[:, 1].reshape(points, points)
        symmetry_error = (
            probabilities[:, 1] + reflected_probabilities[:, 1] - 1.0
        ).reshape(points, points)

        top = axes[0, column]
        top_image = top.pcolormesh(
            np.degrees(theta_grid),
            omega_grid,
            p_right,
            shading="auto",
            vmin=0.0,
            vmax=1.0,
            cmap="coolwarm",
        )
        top.contour(
            np.degrees(theta_grid), omega_grid, margins, levels=[0.0], colors="black"
        )
        top.contour(
            np.degrees(theta_grid),
            omega_grid,
            reflected_margins,
            levels=[0.0],
            colors="black",
            linestyles="dashed",
        )
        top.set_title(f"cart velocity = {cart_velocity:g}")

        bottom = axes[1, column]
        bottom_image = bottom.pcolormesh(
            np.degrees(theta_grid),
            omega_grid,
            symmetry_error,
            shading="auto",
            vmin=-1.0,
            vmax=1.0,
            cmap="PiYG",
        )
        bottom.axhline(0.0, color="0.5", linewidth=0.5)
        bottom.axvline(0.0, color="0.5", linewidth=0.5)
        bottom.set_xlabel("Pole angle (degrees)")
    axes[0, 0].set_ylabel("Pole angular velocity (rad/s)")
    axes[1, 0].set_ylabel("Pole angular velocity (rad/s)")
    figure.text(0.015, 0.72, "P(action 1 | state)", rotation=90, va="center")
    figure.text(
        0.015,
        0.28,
        "Signed probability symmetry error",
        rotation=90,
        va="center",
    )
    if top_image is not None:
        figure.colorbar(top_image, ax=axes[0, :], shrink=0.85, pad=0.02)
    if bottom_image is not None:
        figure.colorbar(bottom_image, ax=axes[1, :], shrink=0.85, pad=0.02)
    figure.suptitle(
        "Frozen CartPole PPO decision boundary\n"
        "solid: observed boundary; dashed: boundary required by reflected states",
        fontsize=12,
    )
    figure.subplots_adjust(left=0.08, right=0.9, bottom=0.09, top=0.87, hspace=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _seeded_nominal_states(
    metadata: PolicyArtifactMetadata, seeds: tuple[int, ...]
) -> dict[int, np.ndarray]:
    env, _ = make_artifact_environment(metadata)
    result = {}
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            state = np.asarray(observation, dtype=np.float64)
            if state.shape != (4,):
                raise ValueError("CartPole symmetry requires four-value observations.")
            result[seed] = state.copy()
    finally:
        env.close()
    return result


def _collect_mirror_pairs(
    suite: CartPoleSymmetrySuiteSpec,
    *,
    policies: tuple[tuple[Any, PolicyArtifactMetadata], ...],
    base_states: dict[int, np.ndarray],
    output_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for policy, metadata in policies:
        for angle in suite.angle_magnitudes_deg:
            for length in suite.lengths:
                condition_id = _mirror_condition_id(angle, length)
                for seed in suite.seeds:
                    positive = base_states[seed].copy()
                    positive[2] += radians(angle)
                    branches = {}
                    for branch, initial_state in (
                        ("positive", positive),
                        ("reflected", -positive),
                    ):
                        perturbation = PerturbationSpec(
                            CARTPOLE_FIXED_INITIAL_STATE,
                            {
                                "initial_state": initial_state.tolist(),
                                "length": length,
                            },
                        )
                        env, resolved = make_artifact_environment(
                            metadata, perturbation
                        )
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
                        episode_dir = (
                            output_dir
                            / "episodes"
                            / metadata.policy_id
                            / condition_id
                            / f"seed-{seed:03d}"
                            / branch
                        )
                        trajectory_hash = write_episode(episode_dir, episode)
                        metrics = compute_metrics(
                            episode, source_trajectory_sha256=trajectory_hash
                        )
                        write_metrics(episode_dir, metrics)
                        branches[branch] = (episode, metrics)
                    rows.append(
                        _mirror_pair_record(
                            metadata.policy_id,
                            seed,
                            angle,
                            length,
                            branches["positive"],
                            branches["reflected"],
                        )
                    )
    summary = {
        "schema_version": 1,
        "pair_count": len(rows),
        "episode_count": 2 * len(rows),
        "policies": {
            metadata.policy_id: _aggregate_mirror_rows(
                [row for row in rows if row["policy_id"] == metadata.policy_id]
            )
            for _, metadata in policies
        },
        "pairs": rows,
    }
    write_json_new(output_dir / "summary.json", summary)
    return summary


def _mirror_pair_record(
    policy_id: str,
    seed: int,
    angle: float,
    length: float,
    positive: tuple[EpisodeRecord, Any],
    reflected: tuple[EpisodeRecord, Any],
) -> dict[str, Any]:
    positive_episode, positive_metrics = positive
    reflected_episode, reflected_metrics = reflected
    positive_actions = [int(step.action) for step in positive_episode.transitions]
    reflected_actions = [int(step.action) for step in reflected_episode.transitions]
    shared_length = min(len(positive_actions), len(reflected_actions))
    violations = [
        index
        for index in range(shared_length)
        if positive_actions[index] == reflected_actions[index]
    ]
    first_violation = None if not violations else violations[0]
    states_positive = [positive_episode.metadata.initial_state]
    states_positive.extend(
        step.environment_state for step in positive_episode.transitions
    )
    states_reflected = [reflected_episode.metadata.initial_state]
    states_reflected.extend(
        step.environment_state for step in reflected_episode.transitions
    )
    state_count = shared_length + 1 if first_violation is None else first_violation + 1
    residuals = []
    for first, second in zip(
        states_positive[:state_count], states_reflected[:state_count], strict=True
    ):
        residuals.extend(
            abs(float(first[name]) + float(second[name])) for name in STATE_ORDER
        )
    return {
        "policy_id": policy_id,
        "seed": seed,
        "angle_magnitude_deg": angle,
        "length": length,
        "positive": {
            "episode_length": positive_metrics.episode_length,
            "task_success": positive_metrics.task_success,
        },
        "reflected": {
            "episode_length": reflected_metrics.episode_length,
            "task_success": reflected_metrics.task_success,
        },
        "first_action_violation_step": first_violation,
        "fully_action_equivariant": (
            first_violation is None and len(positive_actions) == len(reflected_actions)
        ),
        "maximum_state_mirror_error_before_action_violation": max(residuals),
    }


def _aggregate_mirror_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    length_differences = [
        abs(row["positive"]["episode_length"] - row["reflected"]["episode_length"])
        for row in rows
    ]
    return {
        "pair_count": len(rows),
        "positive_success_rate": fmean(
            float(row["positive"]["task_success"]) for row in rows
        ),
        "reflected_success_rate": fmean(
            float(row["reflected"]["task_success"]) for row in rows
        ),
        "paired_success_agreement_rate": fmean(
            row["positive"]["task_success"] == row["reflected"]["task_success"]
            for row in rows
        ),
        "equal_episode_length_rate": fmean(
            difference == 0 for difference in length_differences
        ),
        "mean_absolute_episode_length_difference": fmean(length_differences),
        "fully_action_equivariant_rate": fmean(
            row["fully_action_equivariant"] for row in rows
        ),
        "first_action_equivariant_rate": fmean(
            row["first_action_violation_step"] != 0 for row in rows
        ),
        "maximum_pre_violation_state_mirror_error": max(
            row["maximum_state_mirror_error_before_action_violation"] for row in rows
        ),
    }


def _collect_causal_intervention(
    suite: CartPoleSymmetrySuiteSpec,
    *,
    policy: AntisymmetrizedCartPolePPOPolicy,
    metadata: PolicyArtifactMetadata,
    source_metrics: dict[tuple[str, int], Any],
    source_initial_states: dict[tuple[str, int], Any],
    output_dir: Path,
) -> dict[str, Any]:
    rows = []
    for angle in suite.signed_angles_deg:
        for length in suite.lengths:
            condition_id = _angle_length_condition_id(angle, length)
            for seed in suite.seeds:
                perturbation = PerturbationSpec(
                    CARTPOLE_ANGLE_LENGTH,
                    {"delta_theta_deg": angle, "length": length},
                )
                env, resolved = make_artifact_environment(metadata, perturbation)
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
                source_key = (condition_id, seed)
                if (
                    to_jsonable(episode.metadata.initial_state)
                    != source_initial_states[source_key]
                ):
                    raise RuntimeError(
                        f"Causal intervention initial-state mismatch for "
                        f"{condition_id}, seed {seed}."
                    )
                episode_dir = (
                    output_dir
                    / "episodes"
                    / metadata.policy_id
                    / condition_id
                    / f"seed-{seed:03d}"
                )
                trajectory_hash = write_episode(episode_dir, episode)
                metrics = compute_metrics(
                    episode, source_trajectory_sha256=trajectory_hash
                )
                write_metrics(episode_dir, metrics)
                source = source_metrics[source_key]
                rows.append(
                    {
                        "condition_id": condition_id,
                        "angle_deg": angle,
                        "length": length,
                        "seed": seed,
                        "source_episode_length": source.episode_length,
                        "source_success": source.task_success,
                        "symmetrized_episode_length": metrics.episode_length,
                        "symmetrized_success": metrics.task_success,
                    }
                )
    by_angle = {}
    for angle in suite.signed_angles_deg:
        selected = [row for row in rows if row["angle_deg"] == angle]
        by_angle[_slug_number(angle)] = {
            "angle_deg": angle,
            "episode_count": len(selected),
            "source_success_rate": fmean(row["source_success"] for row in selected),
            "symmetrized_success_rate": fmean(
                row["symmetrized_success"] for row in selected
            ),
            "source_mean_episode_length": fmean(
                row["source_episode_length"] for row in selected
            ),
            "symmetrized_mean_episode_length": fmean(
                row["symmetrized_episode_length"] for row in selected
            ),
        }
    directional = {}
    for angle in suite.angle_magnitudes_deg:
        negative = by_angle[_slug_number(-angle)]
        positive = by_angle[_slug_number(angle)]
        directional[_slug_number(angle)] = {
            "angle_magnitude_deg": angle,
            "source_positive_minus_negative_success_rate": (
                positive["source_success_rate"] - negative["source_success_rate"]
            ),
            "symmetrized_positive_minus_negative_success_rate": (
                positive["symmetrized_success_rate"]
                - negative["symmetrized_success_rate"]
            ),
        }
    summary = {
        "schema_version": 1,
        "intervention": SYMMETRIZATION,
        "source_policy_id": suite.policy_id,
        "derived_policy_id": suite.derived_policy_id,
        "episode_count": len(rows),
        "by_angle": by_angle,
        "directional_gaps": directional,
        "episodes": rows,
    }
    write_json_new(output_dir / "summary.json", summary)
    return summary


def _derived_metadata(
    source: PolicyArtifactMetadata, *, derived_policy_id: str
) -> PolicyArtifactMetadata:
    design = replace(
        source.design_spec,
        design_id=f"{source.design_spec.design_id}-antisymmetrized-v1",
        policy_id=derived_policy_id,
        policy_type="derived_fixed_policy",
        algorithm="PPO-antisymmetrized-binary-logit-margin",
        hyperparameters={
            "transformation": SYMMETRIZATION,
            "source_policy_id": source.policy_id,
            "source_model_sha256": source.model_sha256,
            "weights_changed": False,
        },
    )
    return replace(
        source,
        policy_id=derived_policy_id,
        design_spec=design,
        artifact_format="derived-from-frozen-sb3-v1",
        source_version={
            "kind": "deterministic_inference_transformation",
            "source_policy_id": source.policy_id,
            "source_model_sha256": source.model_sha256,
            "transformation": SYMMETRIZATION,
        },
    )


def _artifact_identity(metadata: PolicyArtifactMetadata) -> dict[str, str]:
    return {
        "policy_id": metadata.policy_id,
        "design_id": metadata.design_spec.design_id,
        "model_sha256": metadata.model_sha256,
    }


def _angle_length_condition_id(angle: float, length: float) -> str:
    return f"theta-{_slug_number(angle)}_length-{_slug_number(length)}"


def _mirror_condition_id(angle: float, length: float) -> str:
    return f"magnitude-{_slug_number(angle)}_length-{_slug_number(length)}"


def _slug_number(value: float) -> str:
    sign = "m" if value < 0 else "p"
    body = format(abs(float(value)), ".12g").replace(".", "p")
    return sign + body

