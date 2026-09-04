"""Deterministic figures derived from saved sweep summaries."""

import json
from math import ceil
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402


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
        if suite["perturbation_name"] == "cartpole-recovery-angle-length-v1":
            return _plot_cartpole_boundary(summary, output_dir)
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
        {item["parameters"]["delta_theta_deg"] for item in summary["conditions"]}
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
    panels = [
        (values, _short_policy(policy), False)
        for policy, values in zip(policies, surfaces, strict=True)
    ]
    panels.extend(
        (
            surfaces[second] - surfaces[first],
            f"{_short_policy(policies[second])} - {_short_policy(policies[first])}",
            True,
        )
        for second in range(1, len(policies))
        for first in range(second)
    )
    columns = min(3, len(panels))
    rows = ceil(len(panels) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 4.5 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, (values, title, is_difference) in zip(axes.flat, panels, strict=False):
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            vmin=-1 if is_difference else 0,
            vmax=1,
            cmap="coolwarm" if is_difference else "viridis",
        )
        axis.set_xticks(range(len(lengths)), labels=lengths)
        axis.set_yticks(range(len(angles)), labels=angles)
        axis.set_xlabel("Pole length")
        axis.set_ylabel("Initial angle offset (degrees)")
        axis.set_title(title)
        figure.colorbar(
            image,
            ax=axis,
            label="Success-rate difference" if is_difference else "Success rate",
        )
    for axis in list(axes.flat)[len(panels) :]:
        axis.set_visible(False)
    surface_filename = (
        "che49_cartpole_success_surface.png"
        if len(policies) == 2
        else "che49_cartpole_success_surface_three_policy.png"
    )
    surface_path = output_dir / surface_filename
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
    slice_filename = (
        "che49_cartpole_nominal_slices.png"
        if len(policies) == 2
        else "che49_cartpole_nominal_slices_three_policy.png"
    )
    slice_path = output_dir / slice_filename
    _save_new(figure, slice_path)
    return [surface_path, slice_path]


_BOUNDARY_LEVELS = (0.0, 0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98, 1.0)
# One neutral for "no condition was sampled here", used by every boundary panel.
# It has to survive two collisions: a diverging map's pale centre (a measured
# zero difference) and a sequential map's near-white low end (a measured zero
# gap). A mid gray clears both, and no level of the discrete rate colormap is a
# neutral, so it never reads as a rate either.
_UNSAMPLED_COLOR = "#8c8c8c"
_BOUNDARY_CAUSES = ("none", "angle_limit", "cart_limit", "both", "unknown")
_BOUNDARY_CAUSE_COLORS = {
    "none": "#4daf4a",
    "angle_limit": "#e41a1c",
    "cart_limit": "#377eb8",
    "both": "#984ea3",
    # Not a gray: the unsampled neutral already owns that slot in the legend.
    "unknown": "#ff7f00",
}
_BOUNDARY_RATE_METRICS = (
    ("success_rate", "Survival rate"),
    ("recovery_rate", "Recovery rate"),
)


def _plot_cartpole_boundary(summary: dict[str, Any], output_dir: Path) -> list[Path]:
    """Render the adaptive failure-boundary lattice as index-space heat maps.

    The sampled conditions form a sparse subset of an angle-by-length lattice.
    Every panel is drawn in *mesh rank* space so that the irregular geometric
    length mesh does not hide the interior (strictly between 0 and 1) cells that
    carry the scientific content, and unsampled lattice slots stay visibly gray
    rather than being confused with a measured zero.
    """

    policies = summary["policy_ids"]
    angles = sorted(
        {item["parameters"]["initial_theta_deg"] for item in summary["conditions"]}
    )
    lengths = sorted({item["parameters"]["length"] for item in summary["conditions"]})
    lookup = _boundary_lookup(summary)

    paths = [
        _plot_boundary_rate(
            policies,
            lookup,
            angles,
            lengths,
            metric,
            label,
            output_dir / f"cartpole_boundary_{stem}_v2.png",
        )
        for metric, label, stem in (
            ("success_rate", "Survival rate", "survival"),
            ("recovery_rate", "Recovery rate", "recovery"),
        )
    ]
    paths.append(
        _plot_boundary_asymmetry(
            policies,
            lookup,
            angles,
            lengths,
            output_dir / "cartpole_boundary_signed_asymmetry_v2.png",
        )
    )
    paths.append(
        _plot_boundary_gap_and_cause(
            policies,
            lookup,
            angles,
            lengths,
            output_dir / "cartpole_boundary_recovery_gap_failure_cause_v2.png",
        )
    )
    paths.append(
        _plot_boundary_wilson(
            policies,
            lookup,
            angles,
            lengths,
            output_dir / "cartpole_boundary_wilson_uncertainty_v2.png",
        )
    )
    paths.append(
        _plot_boundary_physical_geometry(
            policies,
            lookup,
            angles,
            lengths,
            output_dir / "cartpole_boundary_physical_geometry_v2.png",
        )
    )
    return paths


def _boundary_lookup(summary: dict[str, Any]) -> dict[tuple[float, float], Any]:
    return {
        (
            float(item["parameters"]["initial_theta_deg"]),
            float(item["parameters"]["length"]),
        ): item
        for item in summary["conditions"]
    }


def _boundary_surface(
    lookup: dict[tuple[float, float], Any],
    angles: list[float],
    lengths: list[float],
    value: Any,
) -> np.ndarray:
    """Lay sampled conditions onto the (angle rank, length rank) lattice."""

    surface = np.full((len(angles), len(lengths)), np.nan)
    for row, angle in enumerate(angles):
        for column, length in enumerate(lengths):
            item = lookup.get((angle, length))
            if item is None:
                continue
            result = value(item)
            if result is not None:
                surface[row, column] = result
    return surface


def _nearest_filled(surface: np.ndarray) -> np.ndarray:
    """Fill lattice holes from the nearest sampled slot in index space.

    Only ever used as input to ``contour``; the displayed arrays keep their
    holes so the figures stay honest about where sampling actually happened.
    """

    filled = np.array(surface, dtype=float)
    sampled = np.argwhere(np.isfinite(surface))
    if len(sampled) == 0:
        return np.zeros_like(filled)
    for row, column in np.argwhere(~np.isfinite(surface)):
        offsets = sampled - np.array([row, column])
        nearest = sampled[int(np.argmin((offsets * offsets).sum(axis=1)))]
        filled[row, column] = surface[nearest[0], nearest[1]]
    return filled


def _boundary_colormap(name: str, count: int | None = None) -> Any:
    colormap = plt.get_cmap(name)
    if count is not None:
        colormap = colormap.resampled(count)
    return colormap.with_extremes(bad=_UNSAMPLED_COLOR)


def _boundary_norm() -> BoundaryNorm:
    return BoundaryNorm(list(_BOUNDARY_LEVELS), ncolors=len(_BOUNDARY_LEVELS) - 1)


def _tick_positions(count: int, maximum: int) -> list[int]:
    stride = max(1, ceil(count / maximum))
    return list(range(0, count, stride))


def _label_lattice_axes(
    axis: Any,
    angles: list[float],
    lengths: list[float],
    *,
    y_label: str = "Initial pole angle (degrees, mesh rank)",
) -> None:
    columns = _tick_positions(len(lengths), 12)
    axis.set_xticks(columns, labels=[f"{lengths[i]:.3g}" for i in columns])
    axis.tick_params(axis="x", labelrotation=90, labelsize=7)
    rows = _tick_positions(len(angles), 20)
    axis.set_yticks(rows, labels=[f"{angles[i]:g}" for i in rows])
    axis.tick_params(axis="y", labelsize=7)
    axis.set_xlabel("Pole half-length (mesh rank)")
    axis.set_ylabel(y_label)


def _draw_lattice(
    axis: Any,
    surface: np.ndarray,
    angles: list[float],
    lengths: list[float],
    *,
    colormap: Any,
    norm: Any = None,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str = "",
    contour: bool = False,
    y_label: str = "Initial pole angle (degrees, mesh rank)",
) -> Any:
    image = axis.imshow(
        surface,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=colormap,
        norm=norm,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
    )
    if contour and np.isfinite(surface).any():
        crossing = _nearest_filled(surface)
        if crossing.min() < 0.5 < crossing.max():
            axis.contour(crossing, levels=[0.5], colors="k", linewidths=1.2)
    _label_lattice_axes(axis, angles, lengths, y_label=y_label)
    axis.set_title(title, fontsize=10)
    return image


def _row_colorbar(figure: Any, image: Any, axes: list[Any], label: str, **kwargs: Any):
    return figure.colorbar(image, ax=axes, label=label, **kwargs)


def _hide_unused(axes: Any, used: int) -> None:
    for axis in list(axes)[used:]:
        axis.set_visible(False)


def _plot_boundary_rate(
    policies: list[str],
    lookup: dict[tuple[float, float], Any],
    angles: list[float],
    lengths: list[float],
    metric: str,
    label: str,
    path: Path,
) -> Path:
    surfaces = [
        _boundary_surface(
            lookup, angles, lengths, lambda item, p=policy: item["policies"][p][metric]
        )
        for policy in policies
    ]
    pairs = [
        (first, second) for second in range(1, len(policies)) for first in range(second)
    ]
    columns = max(len(policies), len(pairs), 1)
    rows = 2 if pairs else 1
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.6 * columns, 5.0 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    colormap = _boundary_colormap("RdYlBu", len(_BOUNDARY_LEVELS) - 1)
    norm = _boundary_norm()
    image = None
    for column, (policy, surface) in enumerate(zip(policies, surfaces, strict=True)):
        interior = int(np.sum((surface > 0.0) & (surface < 1.0)))
        image = _draw_lattice(
            axes[0, column],
            surface,
            angles,
            lengths,
            colormap=colormap,
            norm=norm,
            title=f"{_short_policy(policy)}\n{interior} interior cells",
            contour=True,
        )
    _hide_unused(axes[0], len(policies))
    _row_colorbar(
        figure,
        image,
        [axis for axis in axes[0] if axis.get_visible()],
        label,
        ticks=list(_BOUNDARY_LEVELS),
        spacing="uniform",
    )
    if pairs:
        difference_map = _boundary_colormap("coolwarm")
        for column, (first, second) in enumerate(pairs):
            image = _draw_lattice(
                axes[1, column],
                surfaces[second] - surfaces[first],
                angles,
                lengths,
                colormap=difference_map,
                vmin=-1,
                vmax=1,
                title=(
                    f"{_short_policy(policies[second])}"
                    f" - {_short_policy(policies[first])}"
                ),
            )
        _hide_unused(axes[1], len(pairs))
        _row_colorbar(
            figure,
            image,
            [axis for axis in axes[1] if axis.get_visible()],
            f"{label} difference",
        )
    figure.suptitle(
        f"{label} on the adaptive angle/length lattice"
        f" ({int(np.isfinite(surfaces[0]).sum())} of"
        f" {len(angles) * len(lengths)} slots sampled;"
        " gray = unsampled, black line = 0.5 contour interpolated across holes)"
    )
    _save_new(figure, path)
    return path


def _mirrored_angles(
    lookup: dict[tuple[float, float], Any],
    angles: list[float],
    lengths: list[float],
) -> list[float]:
    """Positive angles that have at least one exact negative counterpart."""

    return [
        angle
        for angle in angles
        if angle > 0
        and any(
            (angle, length) in lookup and (-angle, length) in lookup
            for length in lengths
        )
    ]


def _mirror_difference(
    lookup: dict[tuple[float, float], Any],
    positive_angles: list[float],
    lengths: list[float],
    policy: str,
    metric: str,
) -> np.ndarray:
    surface = np.full((len(positive_angles), len(lengths)), np.nan)
    for row, angle in enumerate(positive_angles):
        for column, length in enumerate(lengths):
            positive = lookup.get((angle, length))
            negative = lookup.get((-angle, length))
            if positive is None or negative is None:
                continue
            surface[row, column] = (
                positive["policies"][policy][metric]
                - negative["policies"][policy][metric]
            )
    return surface


def _plot_boundary_asymmetry(
    policies: list[str],
    lookup: dict[tuple[float, float], Any],
    angles: list[float],
    lengths: list[float],
    path: Path,
) -> Path:
    positive_angles = _mirrored_angles(lookup, angles, lengths)
    pair_count = sum(
        1
        for angle in positive_angles
        for length in lengths
        if (angle, length) in lookup and (-angle, length) in lookup
    )
    figure, axes = plt.subplots(
        len(_BOUNDARY_RATE_METRICS),
        max(len(policies), 1),
        figsize=(4.6 * max(len(policies), 1), 5.0 * len(_BOUNDARY_RATE_METRICS)),
        constrained_layout=True,
        squeeze=False,
    )
    colormap = _boundary_colormap("coolwarm")
    for row, (metric, label) in enumerate(_BOUNDARY_RATE_METRICS):
        image = None
        for column, policy in enumerate(policies):
            axis = axes[row, column]
            if not positive_angles:
                axis.set_axis_off()
                axis.text(0.5, 0.5, "no mirrored pairs", ha="center", va="center")
                continue
            surface = _mirror_difference(
                lookup, positive_angles, lengths, policy, metric
            )
            mean_absolute = (
                float(np.nanmean(np.abs(surface)))
                if np.isfinite(surface).any()
                else float("nan")
            )
            image = _draw_lattice(
                axis,
                surface,
                positive_angles,
                lengths,
                colormap=colormap,
                vmin=-1,
                vmax=1,
                title=(
                    f"{_short_policy(policy)}: {label}\n"
                    f"mean |asymmetry| = {mean_absolute * 100:.2f} pp"
                ),
                y_label="|Initial pole angle| (degrees, mesh rank)",
            )
        _hide_unused(axes[row], len(policies))
        if image is not None:
            _row_colorbar(
                figure,
                image,
                [axis for axis in axes[row] if axis.get_visible()],
                f"{label}(+theta) - {label.lower()}(-theta)",
            )
    figure.suptitle(
        "Mirror asymmetry across the sign of the initial angle"
        f" ({pair_count} exact +/- pairs; red = better on the positive side)"
    )
    _save_new(figure, path)
    return path


def _dominant_cause(aggregate: dict[str, Any]) -> str:
    counts = aggregate["failure_cause_counts"]
    failures = {name: count for name, count in counts.items() if name != "none"}
    if sum(failures.values()) == 0:
        return "none"
    return max(failures, key=lambda name: (failures[name], name))


def _plot_boundary_gap_and_cause(
    policies: list[str],
    lookup: dict[tuple[float, float], Any],
    angles: list[float],
    lengths: list[float],
    path: Path,
) -> Path:
    columns = max(len(policies), 1)
    figure, axes = plt.subplots(
        2,
        columns,
        figsize=(4.6 * columns, 10.0),
        constrained_layout=True,
        squeeze=False,
    )
    gap_map = _boundary_colormap("Purples")
    cause_map = ListedColormap(
        [_BOUNDARY_CAUSE_COLORS[name] for name in _BOUNDARY_CAUSES]
    ).with_extremes(bad=_UNSAMPLED_COLOR)
    cause_norm = BoundaryNorm(
        [index - 0.5 for index in range(len(_BOUNDARY_CAUSES) + 1)],
        ncolors=len(_BOUNDARY_CAUSES),
    )
    gap_image = None
    cause_image = None
    for column, policy in enumerate(policies):
        gap = _boundary_surface(
            lookup,
            angles,
            lengths,
            lambda item, p=policy: (
                item["policies"][p]["success_rate"]
                - item["policies"][p]["recovery_rate"]
            ),
        )
        gap_image = _draw_lattice(
            axes[0, column],
            gap,
            angles,
            lengths,
            colormap=gap_map,
            vmin=0,
            vmax=1,
            title=f"{_short_policy(policy)}: survival - recovery",
        )
        causes = _boundary_surface(
            lookup,
            angles,
            lengths,
            lambda item, p=policy: float(
                _BOUNDARY_CAUSES.index(_dominant_cause(item["policies"][p]))
            ),
        )
        cause_image = _draw_lattice(
            axes[1, column],
            causes,
            angles,
            lengths,
            colormap=cause_map,
            norm=cause_norm,
            title=f"{_short_policy(policy)}: dominant episode ending",
        )
    _hide_unused(axes[0], len(policies))
    _hide_unused(axes[1], len(policies))
    _row_colorbar(
        figure,
        gap_image,
        [axis for axis in axes[0] if axis.get_visible()],
        "Survival - recovery rate",
    )
    cause_bar = _row_colorbar(
        figure,
        cause_image,
        [axis for axis in axes[1] if axis.get_visible()],
        "Dominant episode ending",
        ticks=list(range(len(_BOUNDARY_CAUSES))),
    )
    cause_bar.ax.set_yticklabels(_BOUNDARY_CAUSES)
    handles = [
        Patch(facecolor=_BOUNDARY_CAUSE_COLORS[name], edgecolor="black", label=name)
        for name in _BOUNDARY_CAUSES
    ]
    handles.append(
        Patch(facecolor=_UNSAMPLED_COLOR, edgecolor="black", label="unsampled")
    )
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(handles),
        frameon=False,
    )
    figure.suptitle("Recovery gap and dominant episode ending on the mesh lattice")
    _save_new(figure, path)
    return path


def _straddles_half(interval: dict[str, float]) -> bool:
    """True when a Wilson 95% interval cannot resolve which side of 0.5 a cell is."""

    return bool(interval["lower"] < 0.5 < interval["upper"])


def _plot_boundary_wilson(
    policies: list[str],
    lookup: dict[tuple[float, float], Any],
    angles: list[float],
    lengths: list[float],
    path: Path,
) -> Path:
    columns = max(len(policies), 1)
    figure, axes = plt.subplots(
        len(_BOUNDARY_RATE_METRICS),
        columns,
        figsize=(4.6 * columns, 5.0 * len(_BOUNDARY_RATE_METRICS)),
        constrained_layout=True,
        squeeze=False,
    )
    colormap = _boundary_colormap("RdYlBu", len(_BOUNDARY_LEVELS) - 1)
    norm = _boundary_norm()
    for row, (metric, label) in enumerate(_BOUNDARY_RATE_METRICS):
        image = None
        for column, policy in enumerate(policies):
            surface = _boundary_surface(
                lookup,
                angles,
                lengths,
                lambda item, p=policy, m=metric: item["policies"][p][m],
            )
            straddle = _boundary_surface(
                lookup,
                angles,
                lengths,
                lambda item, p=policy, m=metric: float(
                    _straddles_half(item["policies"][p][f"{m}_wilson_95"])
                ),
            )
            count = int(np.nansum(straddle))
            axis = axes[row, column]
            image = _draw_lattice(
                axis,
                surface,
                angles,
                lengths,
                colormap=colormap,
                norm=norm,
                title=(
                    f"{_short_policy(policy)}: {label}\n"
                    f"{count} unresolved cell{'' if count == 1 else 's'}"
                    " (95% CI straddles 0.5)"
                ),
                contour=True,
            )
            for cell_row, cell_column in np.argwhere(straddle == 1.0):
                axis.add_patch(
                    Rectangle(
                        (cell_column - 0.5, cell_row - 0.5),
                        1,
                        1,
                        fill=False,
                        hatch="///",
                        edgecolor="black",
                        linewidth=1.0,
                    )
                )
        _hide_unused(axes[row], len(policies))
        _row_colorbar(
            figure,
            image,
            [axis for axis in axes[row] if axis.get_visible()],
            label,
            ticks=list(_BOUNDARY_LEVELS),
            spacing="uniform",
        )
    figure.suptitle(
        "Where should the next evaluations go? Hatched cells are the lattice slots"
        " whose 95% Wilson interval still straddles 0.5"
    )
    _save_new(figure, path)
    return path


def _cell_edges(values: list[float], *, geometric: bool) -> np.ndarray:
    work = (
        np.log(np.asarray(values, dtype=float))
        if geometric
        else np.asarray(values, dtype=float)
    )
    if len(work) == 1:
        edges = np.array([work[0] - 0.5, work[0] + 0.5])
    else:
        middles = (work[:-1] + work[1:]) / 2
        edges = np.concatenate(
            [
                [work[0] - (middles[0] - work[0])],
                middles,
                [work[-1] + (work[-1] - middles[-1])],
            ]
        )
    return np.exp(edges) if geometric else edges


def _plot_boundary_physical_geometry(
    policies: list[str],
    lookup: dict[tuple[float, float], Any],
    angles: list[float],
    lengths: list[float],
    path: Path,
) -> Path:
    """Companion figure keeping the true aspect ratio of the survivable region."""

    columns = max(len(policies), 1)
    figure, axes = plt.subplots(
        len(_BOUNDARY_RATE_METRICS),
        columns,
        figsize=(4.6 * columns, 4.6 * len(_BOUNDARY_RATE_METRICS)),
        constrained_layout=True,
        squeeze=False,
    )
    colormap = _boundary_colormap("RdYlBu", len(_BOUNDARY_LEVELS) - 1)
    norm = _boundary_norm()
    x_edges = _cell_edges(lengths, geometric=True)
    y_edges = _cell_edges(angles, geometric=False)
    for row, (metric, label) in enumerate(_BOUNDARY_RATE_METRICS):
        mesh = None
        for column, policy in enumerate(policies):
            surface = _boundary_surface(
                lookup,
                angles,
                lengths,
                lambda item, p=policy, m=metric: item["policies"][p][m],
            )
            axis = axes[row, column]
            mesh = axis.pcolormesh(
                x_edges,
                y_edges,
                np.ma.masked_invalid(surface),
                cmap=colormap,
                norm=norm,
                shading="flat",
            )
            axis.set_xscale("log")
            axis.set_xlabel("Pole half-length (physical, log scale)")
            axis.set_ylabel("Initial pole angle (degrees)")
            axis.set_title(f"{_short_policy(policy)}: {label}", fontsize=10)
        _hide_unused(axes[row], len(policies))
        _row_colorbar(
            figure,
            mesh,
            [axis for axis in axes[row] if axis.get_visible()],
            label,
            ticks=list(_BOUNDARY_LEVELS),
            spacing="uniform",
        )
    figure.suptitle(
        "Physical-geometry companion: real cell edges, geometric in length"
        " (gray = unsampled)"
    )
    _save_new(figure, path)
    return path


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
    if len(policies) > 2:
        stem = Path(filename).stem
        filename = f"{stem}_three_policy.png"
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
            image = axis.imshow(values, origin="upper", vmin=0, vmax=vmax, cmap=cmap)
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
    if "antisymmetrized" in policy_id:
        return "Antisymmetrized PPO"
    if "ppo" in policy_id:
        return "PPO"
    return policy_id


def _save_new(figure: Any, path: Path) -> None:
    if path.exists():
        plt.close(figure)
        raise FileExistsError(f"Refusing to overwrite figure: {path}.")
    figure.savefig(path, dpi=160)
    plt.close(figure)
