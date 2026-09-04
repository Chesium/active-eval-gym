"""Deterministic figures derived from saved sweep summaries."""

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def plot_sweep(evaluation_dir: Path, output_dir: Path) -> list[Path]:
    """Render the appropriate checked-in figure set for one sweep."""

    suite_record = json.loads((evaluation_dir / "suite.json").read_text())
    suite = suite_record["suite"]
    metric_version = suite.get("metric_version", "episode-summary-v2")
    summary_path = evaluation_dir / "analysis" / metric_version / "summary.json"
    summary = json.loads(summary_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    env_id = summary["environment_id"]
    if env_id == "CartPole-v1":
        if suite["perturbation_name"] == "cartpole-angle-length-v1":
            return _plot_cartpole(summary, output_dir)
        return _plot_cartpole_one_dimensional(
            summary, output_dir, suite["perturbation_name"]
        )
    if env_id == "Pendulum-v1":
        return _plot_pendulum(summary, output_dir)
    if env_id == "MiniGrid-Empty-8x8-v0":
        return _plot_minigrid(summary, output_dir)
    raise ValueError(f"Unsupported plotting environment {env_id!r}.")


def _plot_cartpole(summary: dict[str, Any], output_dir: Path) -> list[Path]:
    policies = summary["policy_ids"]
    angles = sorted(
        {
            item["parameters"]["delta_theta_deg"]
            for item in summary["conditions"]
        }
    )
    lengths = sorted({item["parameters"]["length"] for item in summary["conditions"]})
    lookup = _condition_lookup(summary)

    surfaces = []
    for policy in policies:
        surfaces.append(
            np.array(
                [
                    [
                        lookup[(angle, length)]["policies"][policy]["success_rate"]
                        for length in lengths
                    ]
                    for angle in angles
                ]
            )
        )
    difference = surfaces[1] - surfaces[0]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    titles = (_short_policy(policies[0]), _short_policy(policies[1]), "PPO - LQR")
    for index, (axis, values, title) in enumerate(
        zip(axes, (*surfaces, difference), titles, strict=True)
    ):
        limit = 1 if index < 2 else None
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            vmin=0 if index < 2 else -1,
            vmax=limit,
            cmap="viridis" if index < 2 else "coolwarm",
        )
        axis.set_xticks(range(len(lengths)), labels=lengths)
        axis.set_yticks(range(len(angles)), labels=angles)
        axis.set_xlabel("Pole length")
        axis.set_ylabel("Initial angle offset (degrees)")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, label="Success rate")
    surface_path = output_dir / "che49_cartpole_success_surface.png"
    _save_new(figure, surface_path)

    metric_specs = [
        ("success_rate", "Success rate"),
        ("episode_length", "Episode length"),
        ("rms_pole_angle_radians", "RMS pole angle (rad)"),
        ("max_abs_pole_angle_radians", "Max |pole angle| (rad)"),
        ("rms_cart_position", "RMS cart position"),
        ("action_switch_rate", "Action-switch rate"),
    ]
    figure, axes = plt.subplots(2, 6, figsize=(24, 8), constrained_layout=True)
    for row, (axis_values, fixed_name, fixed_value, x_label) in enumerate(
        (
            (angles, "length", 0.5, "Initial angle offset (degrees)"),
            (lengths, "delta_theta_deg", 0, "Pole length"),
        )
    ):
        for column, (metric, title) in enumerate(metric_specs):
            axis = axes[row, column]
            for policy in policies:
                values = []
                errors = []
                for value in axis_values:
                    key = (
                        (value, fixed_value)
                        if fixed_name == "length"
                        else (fixed_value, value)
                    )
                    aggregate = lookup[key]["policies"][policy]
                    if metric == "success_rate":
                        values.append(aggregate[metric])
                        errors.append(0.0)
                    elif metric == "episode_length":
                        values.append(aggregate[metric]["mean"])
                        errors.append(aggregate[metric]["standard_deviation"])
                    else:
                        item = aggregate["environment_metrics"][metric]
                        values.append(item["mean"])
                        errors.append(item["standard_deviation"])
                axis.errorbar(
                    axis_values,
                    values,
                    yerr=errors,
                    marker="o",
                    capsize=2,
                    label=_short_policy(policy),
                )
            axis.set_title(title)
            axis.set_xlabel(x_label)
            axis.grid(alpha=0.25)
            if row == 0 and column == 0:
                axis.legend()
    slice_path = output_dir / "che49_cartpole_nominal_slices.png"
    _save_new(figure, slice_path)
    return [surface_path, slice_path]


def _plot_pendulum(summary: dict[str, Any], output_dir: Path) -> list[Path]:
    policy = summary["policy_ids"][0]
    conditions = sorted(summary["conditions"], key=lambda item: item["parameters"]["l"])
    lengths = [item["parameters"]["l"] for item in conditions]
    specs = [
        ("episode_return", "Return", False),
        ("rms_angle_error_radians", "RMS angular error", True),
        ("rms_angular_velocity", "RMS angular velocity", True),
        ("rms_torque", "RMS torque", True),
        ("mean_absolute_torque", "Mean |torque|", True),
    ]
    figure, axes = plt.subplots(1, 5, figsize=(20, 4), constrained_layout=True)
    for axis, (metric, title, environment_metric) in zip(axes, specs, strict=True):
        aggregates = [item["policies"][policy] for item in conditions]
        items = [
            aggregate["environment_metrics"][metric]
            if environment_metric
            else aggregate[metric]
            for aggregate in aggregates
        ]
        means = [item["mean"] for item in items]
        errors = [item["standard_deviation"] for item in items]
        axis.errorbar(lengths, means, yerr=errors, marker="o", capsize=2)
        axis.set_title(title)
        axis.set_xlabel("Pendulum length")
        axis.grid(alpha=0.25)
    path = output_dir / "che49_pendulum_length_sweep.png"
    _save_new(figure, path)
    return [path]


def _plot_cartpole_one_dimensional(
    summary: dict[str, Any], output_dir: Path, perturbation_name: str
) -> list[Path]:
    specifications = {
        "cartpole-mass-v1": (
            "masspole",
            "Pole mass",
            "che49_cartpole_mass_sweep.png",
        ),
        "cartpole-pole-angle-noise-v1": (
            "pole_angle_noise_std_deg",
            "Pole-angle noise SD (degrees)",
            "che49_cartpole_pole_angle_noise_sweep.png",
        ),
        "cartpole-force-magnitude-v1": (
            "force_mag",
            "Force magnitude (N)",
            "che49_cartpole_force_magnitude_sweep.png",
        ),
        "cartpole-action-delay-v1": (
            "delay_steps",
            "Action delay (steps)",
            "che49_cartpole_action_delay_sweep.png",
        ),
        "cartpole-action-dropout-v1": (
            "dropout_probability",
            "Action-dropout probability",
            "che49_cartpole_action_dropout_sweep.png",
        ),
    }
    try:
        parameter, x_label, filename = specifications[perturbation_name]
    except KeyError as error:
        raise ValueError(
            f"Unsupported CartPole plot perturbation {perturbation_name!r}."
        ) from error
    policies = summary["policy_ids"]
    conditions = sorted(
        summary["conditions"], key=lambda item: item["parameters"][parameter]
    )
    x_values = [item["parameters"][parameter] for item in conditions]
    metric_specs = [
        ("success_rate", "Success rate"),
        ("episode_length", "Episode length"),
        ("rms_pole_angle_radians", "RMS pole angle (rad)"),
        ("max_abs_pole_angle_radians", "Max |pole angle| (rad)"),
        ("rms_cart_position", "RMS cart position"),
        ("action_switch_rate", "Applied-action switch rate"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, metric_specs, strict=True):
        for policy in policies:
            aggregates = [item["policies"][policy] for item in conditions]
            if metric == "success_rate":
                means = [item[metric] for item in aggregates]
                errors = [0.0] * len(means)
            elif metric == "episode_length":
                means = [item[metric]["mean"] for item in aggregates]
                errors = [item[metric]["standard_deviation"] for item in aggregates]
            else:
                values = [item["environment_metrics"][metric] for item in aggregates]
                means = [item["mean"] for item in values]
                errors = [item["standard_deviation"] for item in values]
            axis.errorbar(
                x_values,
                means,
                yerr=errors,
                marker="o",
                capsize=2,
                label=_short_policy(policy),
            )
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    path = output_dir / filename
    _save_new(figure, path)
    return [path]


def _plot_minigrid(summary: dict[str, Any], output_dir: Path) -> list[Path]:
    policy = summary["policy_ids"][0]
    lookup = {
        (
            item["parameters"]["agent_start_pos"][0],
            item["parameters"]["agent_start_pos"][1],
            item["parameters"]["agent_start_dir"],
        ): item["policies"][policy]
        for item in summary["conditions"]
    }
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for direction in range(4):
        success = np.full((6, 6), np.nan)
        length = np.full((6, 6), np.nan)
        for (x, y, item_direction), aggregate in lookup.items():
            if item_direction == direction:
                success[y - 1, x - 1] = aggregate["success_rate"]
                length[y - 1, x - 1] = aggregate["episode_length"]["mean"]
        for row, (values, label, cmap, vmax) in enumerate(
            (
                (success, "Success", "viridis", 1),
                (length, "Actions", "magma", np.nanmax(length)),
            )
        ):
            axis = axes[row, direction]
            image = axis.imshow(
                values, origin="upper", vmin=0, vmax=vmax, cmap=cmap
            )
            axis.set_xticks(range(6), labels=range(1, 7))
            axis.set_yticks(range(6), labels=range(1, 7))
            axis.set_xlabel("Start x")
            axis.set_ylabel("Start y")
            axis.set_title(f"Direction {direction}: {label}")
            figure.colorbar(image, ax=axis)
    path = output_dir / "che49_minigrid_start_pose_map.png"
    _save_new(figure, path)
    return [path]


def _condition_lookup(summary: dict[str, Any]) -> dict[tuple[float, float], Any]:
    return {
        (
            item["parameters"]["delta_theta_deg"],
            item["parameters"]["length"],
        ): item
        for item in summary["conditions"]
    }


def _short_policy(policy_id: str) -> str:
    if "lqr" in policy_id:
        return "Quantized LQR"
    if "ppo" in policy_id:
        return "PPO"
    return policy_id


def _save_new(figure: Any, path: Path) -> None:
    if path.exists():
        plt.close(figure)
        raise FileExistsError(f"Refusing to overwrite figure: {path}.")
    figure.savefig(path, dpi=160)
    plt.close(figure)
