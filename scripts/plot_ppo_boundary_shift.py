"""Plot how acceptance moves during a PPO boundary-study tier, from saved results.

Reads saved evaluation/training summaries and frozen distributions only, so it
needs no environment or policy. Per-arm results are development data until a
confirmatory campaign exists.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.lines import Line2D

ARMS = ("background", "acceptance_boundary", "intermediate")
ARM_LABELS = {
    "background": "Background",
    "acceptance_boundary": "Acceptance boundary",
    "intermediate": "Intermediate",
}
ARM_COLORS = dict(zip(ARMS, ("#2a78d6", "#eb6834", "#1baf7a"), strict=True))
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE_RAMP = ("#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
ORANGE_RAMP = ("#fcfcfb", "#fbe0d3", "#f5b597", "#eb6834", "#b8461b", "#7c2d10")
# Direct labels sit where the three lines separate: (half-length rank, x, y, alignment).
DIRECT_LABELS = {
    "background": (4, -6, 9, "right"),
    "acceptance_boundary": (6, -9, 0, "right"),
    "intermediate": (6, 9, 0, "left"),
}
TAU = 0.9
PARENT_LABEL = "Parent"
FINAL = 102_400
SHORT_MAX = 0.3
NOMINAL_LENGTH = 0.5
BANDS = (
    ("Half-length 0.3 m or less", lambda length: length <= SHORT_MAX),
    ("Half-length 0.5 m", lambda length: length == NOMINAL_LENGTH),
    ("Half-length 0.75 m or more", lambda length: length > NOMINAL_LENGTH),
)


def load_cells(study: Path, tier: str, parent_id: str) -> dict:
    """Return recovery rates keyed by arm, replicate, and training budget."""

    rates: dict = defaultdict(dict)
    parent: dict = {}
    config = json.loads((study / "protocol/resolved-config.json").read_text())
    final_budget = next(
        item["additional_transitions"]
        for item in config["tiers"]
        if item["name"] == tier
    )
    for campaign in (tier, f"{tier}-snapshots"):
        path = study / "metrics_v1" / campaign / "summary.json"
        for cell in json.loads(path.read_text())["cells"]:
            point = (
                cell["parameters"]["initial_theta_deg"],
                cell["parameters"]["length"],
            )
            checkpoint = cell["checkpoint_id"]
            if checkpoint == parent_id:
                parent[point] = cell["recovery_rate"]
                continue
            _, _, tail = checkpoint.partition(f"-{tier}-r")
            replicate, _, rest = tail.partition("-")
            arm, _, snapshot = rest.partition("-t")
            budget = int(snapshot) if snapshot else final_budget
            rates[(arm, int(replicate), budget)][point] = cell["recovery_rate"]
    return {"parent": parent, "children": rates}


def present_arms(data: dict) -> tuple[str, ...]:
    names = {key[0] for key in data["children"]}
    if not names or not names <= set(ARMS):
        raise ValueError("Missing or unknown training arms.")
    return tuple(arm for arm in ARMS if arm in names)


def edges(values: list[float]) -> np.ndarray:
    """Return cell edges for ordered centers."""

    centers = np.array(values, dtype=float)
    middle = (centers[:-1] + centers[1:]) / 2
    first = centers[0] - (middle[0] - centers[0])
    last = centers[-1] + (centers[-1] - middle[-1])
    return np.concatenate(([first], middle, [last]))


def outline(axis, accepted: set, angles: list, lengths: list, step: float) -> None:
    """Trace the outer boundary of the parent's accepted cells."""

    half = step / 2
    rank = {length: index for index, length in enumerate(lengths)}
    segments = []
    for angle, length in accepted:
        y = rank[length]
        below = lengths[y - 1] if y else None
        above = lengths[y + 1] if y + 1 < len(lengths) else None
        left, right = angle - half, angle + half
        sides = (
            ((angle - step, length), ((left, y - 0.5), (left, y + 0.5))),
            ((angle + step, length), ((right, y - 0.5), (right, y + 0.5))),
            ((angle, below), ((left, y - 0.5), (right, y - 0.5))),
            ((angle, above), ((left, y + 0.5), (right, y + 0.5))),
        )
        segments += [line for other, line in sides if other not in accepted]
    # Draw the whole halo first, so it never cuts across the outline.
    for color, width, order in ((SURFACE, 3.0, 3), (INK, 1.2, 4)):
        axis.add_collection(
            LineCollection(
                segments, colors=color, linewidths=width, capstyle="round", zorder=order
            )
        )


def style_map(axis) -> None:
    axis.tick_params(length=0, labelsize=8, labelcolor=INK_SECONDARY)
    for side in axis.spines.values():
        side.set_color(AXIS)
        side.set_linewidth(0.6)
    axis.set_facecolor(SURFACE)


def training_windows(
    study: Path, tier: str, budgets: list[int], condition_ids: list[str]
) -> dict:
    """Difference cumulative counts to get transition shares between checkpoints.

    These shares describe the conditions actually visited during each interval,
    including unfinished episodes, under either fixed or adaptive reset laws.
    """

    windows = {}
    for child in sorted((study / "policies" / tier).iterdir()):
        final_path = child / "training-summary.json"
        if not final_path.is_file():
            continue
        paths = [final_path, *sorted(child.glob("snapshots/*/training-summary.json"))]
        summaries = {}
        for path in paths:
            summary = json.loads(path.read_text())
            budget = summary["additional_transitions"]
            if budget in summaries:
                raise ValueError(f"Duplicate training budget {budget}: {child}")
            summaries[budget] = summary
        previous = np.zeros(len(condition_ids), dtype=np.int64)
        start = 0
        for budget in budgets:
            if budget not in summaries:
                raise ValueError(f"Missing training summary at {budget}: {child}")
            summary = summaries[budget]
            realized = summary["realized_sampling"]
            if realized["condition_ids"] != condition_ids:
                raise ValueError(f"Training condition order differs: {child}")
            counts = np.asarray(realized["transitions"], dtype=np.int64)
            if counts.shape != previous.shape:
                raise ValueError(f"Training count shape differs: {child}")
            interval = counts - previous
            if np.any(interval < 0) or interval.sum() != budget - start:
                raise ValueError(f"Invalid training counts at {budget}: {child}")
            key = (summary["arm"], summary["replicate_id"], budget)
            if key in windows:
                raise ValueError(f"Duplicate training record: {key}")
            windows[key] = interval / (budget - start)
            previous, start = counts, budget
    return windows


def maps_figure(
    data: dict, distributions: dict, conditions: dict, windows: dict, path: Path
) -> None:
    """Align recovery with realized training exposure between checkpoints."""

    parent = data["parent"]
    arms = present_arms(data)
    angles = sorted({point[0] for point in parent})
    lengths = sorted({point[1] for point in parent})
    budgets = sorted({key[2] for key in data["children"]})
    transition_target = distributions.get("sampling_unit") == "transition"
    accepted = {point for point, rate in parent.items() if rate >= TAU}
    recovery_map = LinearSegmentedColormap.from_list("recovery", BLUE_RAMP)
    weight_map = LinearSegmentedColormap.from_list("weight", ORANGE_RAMP)
    q_angles = sorted({c["angle"] for c in conditions.values()})
    q_lengths = sorted({c["length"] for c in conditions.values()})
    if q_lengths != lengths:
        raise ValueError("Training and reference half-lengths differ.")
    if windows.keys() != data["children"].keys():
        raise ValueError("Training and evaluation arm/replicate/budget records differ.")
    share_max = max(
        max(max(distributions["arms"][arm]["probabilities"]) for arm in arms),
        max(float(shares.max()) for shares in windows.values()),
    )

    columns = 1 + len(budgets)
    figure, axes = plt.subplots(
        2 * len(arms),
        columns,
        figsize=(max(13.2, 2.64 * columns), max(9.0, 3.2 + 3.45 * len(arms))),
        squeeze=False,
        sharex=True,
        sharey=True,
        gridspec_kw={
            "wspace": 0.07,
            "hspace": 0.28,
        },
    )
    figure.patch.set_facecolor(SURFACE)
    # Shared axes: set ticks once, or later panels erase the labels.
    axes[0, 0].set_yticks(range(len(lengths)), [f"{value:g}" for value in lengths])
    axes[0, 0].set_xticks((-50, 0, 50), ("-50", "0", "+50"))
    weight_norm = PowerNorm(gamma=0.5, vmin=0, vmax=share_max)
    angle_edges = edges(angles)
    q_edges = edges(q_angles)
    length_edges = np.arange(len(lengths) + 1) - 0.5
    for index, arm in enumerate(arms):
        row = 2 * index
        recovery_axes, training_axes = axes[row], axes[row + 1]
        weights = np.zeros((len(lengths), len(q_angles)))
        for identifier, q in zip(
            distributions["condition_ids"],
            distributions["arms"][arm]["probabilities"],
            strict=True,
        ):
            condition = conditions[identifier]
            weights[
                lengths.index(condition["length"]), q_angles.index(condition["angle"])
            ] = q
        target = training_axes[0]
        shown_weights = target.pcolormesh(
            q_edges, length_edges, weights, cmap=weight_map, norm=weight_norm
        )
        target.set_title(
            "Fixed target t (transitions)"
            if transition_target
            else "Fixed selection q (episodes)",
            fontsize=9,
            color=INK,
            pad=6,
        )
        panels = [(recovery_axes[0], parent)]
        start = 0
        for column, budget in enumerate(budgets, start=1):
            replicates = [
                values
                for (name, _, spent), values in data["children"].items()
                if name == arm and spent == budget
            ]
            mean = {
                point: float(np.mean([values[point] for values in replicates]))
                for point in parent
            }
            panels.append((recovery_axes[column], mean))
            mean_shares = np.mean(
                [
                    shares
                    for (name, _, spent), shares in windows.items()
                    if name == arm and spent == budget
                ],
                axis=0,
            )
            weights = np.zeros((len(lengths), len(q_angles)))
            for identifier, share in zip(
                distributions["condition_ids"], mean_shares, strict=True
            ):
                condition = conditions[identifier]
                weights[
                    lengths.index(condition["length"]),
                    q_angles.index(condition["angle"]),
                ] = share
            training_axes[column].pcolormesh(
                q_edges, length_edges, weights, cmap=weight_map, norm=weight_norm
            )
            training_axes[column].set_title(
                f"Transitions: {start / 1000:g}\u2013{budget / 1000:g}k",
                fontsize=9,
                color=INK,
                pad=6,
            )
            start = budget
        for axis, values in panels:
            grid = np.array(
                [[values[(angle, length)] for angle in angles] for length in lengths]
            )
            shown = axis.pcolormesh(
                angle_edges, length_edges, grid, cmap=recovery_map, vmin=0, vmax=1
            )
        for axis in (*recovery_axes, *training_axes):
            outline(axis, accepted, angles, lengths, 5.0)
            style_map(axis)
        recovery_axes[0].set_ylabel(
            f"{ARM_LABELS[arm]}\nRecovery rate\nhalf-length (m)",
            fontsize=9.5,
            color=INK,
            labelpad=6,
        )
        training_axes[0].set_ylabel(
            "Training exposure\nhalf-length (m)",
            fontsize=9.5,
            color=INK,
            labelpad=6,
        )
    titles = [PARENT_LABEL] + [
        f"{budget:,} new transitions" if budget == max(budgets) else f"{budget:,}"
        for budget in budgets
    ]
    for row in range(0, 2 * len(arms), 2):
        for axis, title in zip(axes[row], titles, strict=True):
            axis.set_title(title, fontsize=9.5, color=INK, pad=7)
    for axis in axes[-1]:
        axis.set_xlabel("Initial angle (deg)", fontsize=8.5, color=INK_SECONDARY)
    figure.suptitle(
        "Recovery and training exposure along PPO continuation",
        x=0.065,
        y=0.985,
        ha="left",
        fontsize=15,
        color=INK,
    )
    selection_caption = (
        "Orange: fixed transition target t, then observed transition shares "
        "in each interval.\n"
        "t stays fixed; reset probabilities adapt to episode lengths. "
        if transition_target
        else "Orange: fixed episode selection q, then observed transition shares "
        "in each interval.\n"
        "q stays fixed; exposure can change as episode lengths change. "
    )
    figure.text(
        0.065,
        0.955,
        "Blue: mean recovery rate across 3 replicates, 64 episodes per condition. "
        + selection_caption
        + "Shares are averaged across replicates, not cumulative.\n"
        "Black: parent's accepted region (rate at least 0.9). "
        "Half-length is rank-spaced, so areas are not physical. "
        "Development data: does not rank the arms.",
        ha="left",
        va="top",
        fontsize=9,
        color=INK_SECONDARY,
    )
    figure.subplots_adjust(left=0.1, right=0.92, top=0.875, bottom=0.07)
    bar = figure.add_axes((0.94, 0.56, 0.011, 0.30))
    scale = figure.colorbar(shown, cax=bar)
    scale.set_label("Mean recovery rate", fontsize=8.5, color=INK_SECONDARY)
    scale.ax.tick_params(labelsize=8, labelcolor=INK_SECONDARY, length=0)
    scale.outline.set_visible(False)
    weights_bar = figure.add_axes((0.94, 0.10, 0.011, 0.30))
    weights_scale = figure.colorbar(
        shown_weights,
        cax=weights_bar,
        ticks=[tick for tick in (0, 0.01, 0.05, 0.1, 0.2, 0.3) if tick <= share_max],
    )
    weights_scale.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:.0%}")
    )
    weights_scale.set_label(
        "Condition share (square-root scale): target and observed transitions"
        if transition_target
        else "Condition share: q = episodes; intervals = transitions (sqrt scale)",
        fontsize=8.5,
        color=INK_SECONDARY,
    )
    weights_scale.ax.tick_params(labelsize=8, labelcolor=INK_SECONDARY, length=0)
    weights_scale.outline.set_visible(False)
    save(figure, path)


def comparison_figure(baseline: dict, corrected: dict, path: Path) -> None:
    """Compare identical-budget recovery maps under the two reset samplers."""

    parent = baseline["parent"]
    if corrected["parent"] != parent:
        raise ValueError(
            "Sampler comparison requires identical parent evaluation cells."
        )
    if corrected["children"].keys() != baseline["children"].keys():
        raise ValueError(
            "Sampler comparison requires matching arms/replicates/budgets."
        )
    angles = sorted({p[0] for p in parent})
    lengths = sorted({p[1] for p in parent})
    budgets = sorted({k[2] for k in baseline["children"]})
    accepted = {p for p, value in parent.items() if value >= TAU}
    figure, axes = plt.subplots(
        2 * len(ARMS),
        1 + len(budgets),
        figsize=(13.2, 13.6),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.07, "hspace": 0.28},
    )
    figure.patch.set_facecolor(SURFACE)
    axes[0, 0].set_yticks(range(len(lengths)), [f"{v:g}" for v in lengths])
    axes[0, 0].set_xticks((-50, 0, 50), ("-50", "0", "+50"))
    cmap = LinearSegmentedColormap.from_list("recovery", BLUE_RAMP)
    for index, arm in enumerate(ARMS):
        for offset, (label, data) in enumerate(
            (("Episode sampler (v3)", baseline), ("Duration corrected (v4)", corrected))
        ):
            row = 2 * index + offset
            for column, budget in enumerate([0, *budgets]):
                if budget == 0:
                    values = parent
                else:
                    replicates = [
                        v
                        for (a, _, b), v in data["children"].items()
                        if a == arm and b == budget
                    ]
                    values = {p: np.mean([r[p] for r in replicates]) for p in parent}
                grid = [[values[(a, length)] for a in angles] for length in lengths]
                axis = axes[row, column]
                shown = axis.pcolormesh(
                    edges(angles),
                    np.arange(len(lengths) + 1) - 0.5,
                    grid,
                    cmap=cmap,
                    vmin=0,
                    vmax=1,
                )
                outline(axis, accepted, angles, lengths, 5.0)
                style_map(axis)
                axis.set_title(
                    PARENT_LABEL if budget == 0 else f"{budget:,}",
                    fontsize=9.5,
                    color=INK,
                    pad=6,
                )
            axes[row, 0].set_ylabel(
                f"{ARM_LABELS[arm]}\n{label}\nhalf-length (m)",
                fontsize=9,
                color=INK,
                labelpad=6,
            )
    for axis in axes[-1]:
        axis.set_xlabel("Initial angle (deg)", fontsize=8.5, color=INK_SECONDARY)
    figure.suptitle(
        "Does balancing transition exposure change acceptance drift?",
        x=0.065,
        y=0.985,
        ha="left",
        fontsize=15,
        color=INK,
    )
    figure.text(
        0.065,
        0.955,
        "Mean recovery across 3 paired replicates, 64 episodes per condition. "
        "Columns show new training transitions; all runs start from the same parent.\n"
        "Black: parent's accepted region (rate at least 0.9). "
        "Half-length is rank-spaced. Development comparison; "
        "three replicates are not confirmatory evidence.",
        fontsize=9,
        color=INK_SECONDARY,
        ha="left",
        va="top",
    )
    figure.subplots_adjust(left=0.115, right=0.92, top=0.89, bottom=0.07)
    bar = figure.add_axes((0.94, 0.15, 0.011, 0.67))
    scale = figure.colorbar(shown, cax=bar)
    scale.set_label("Mean recovery rate", fontsize=9, color=INK_SECONDARY)
    scale.ax.tick_params(labelsize=8, length=0)
    scale.outline.set_visible(False)
    save(figure, path)


def replicate_maps_figure(data: dict, path: Path) -> None:
    """Expose between-seed differences in a single arm's transition maps."""

    arms = present_arms(data)
    if len(arms) != 1:
        raise ValueError("Replicate maps require exactly one arm.")
    arm = arms[0]
    parent = data["parent"]
    angles = sorted({p[0] for p in parent})
    lengths = sorted({p[1] for p in parent})
    replicates = sorted({k[1] for k in data["children"]})
    budgets = sorted({k[2] for k in data["children"]})
    accepted = {p for p, value in parent.items() if value >= TAU}
    figure, axes = plt.subplots(
        len(replicates),
        len(budgets) + 1,
        squeeze=False,
        sharex=True,
        sharey=True,
        figsize=(max(13.2, 2.64 * (len(budgets) + 1)), 3 + 2.4 * len(replicates)),
        gridspec_kw={"wspace": 0.07, "hspace": 0.15},
    )
    figure.patch.set_facecolor(SURFACE)
    axes[0, 0].set_yticks(range(len(lengths)), [f"{v:g}" for v in lengths])
    axes[0, 0].set_xticks((-50, 0, 50), ("-50", "0", "+50"))
    cmap = LinearSegmentedColormap.from_list("recovery", BLUE_RAMP)
    for row, replicate in enumerate(replicates):
        for column, budget in enumerate([0, *budgets]):
            values = (
                parent if budget == 0 else data["children"][(arm, replicate, budget)]
            )
            grid = [[values[(a, length)] for a in angles] for length in lengths]
            axis = axes[row, column]
            shown = axis.pcolormesh(
                edges(angles),
                np.arange(len(lengths) + 1) - 0.5,
                grid,
                cmap=cmap,
                vmin=0,
                vmax=1,
            )
            outline(axis, accepted, angles, lengths, 5.0)
            style_map(axis)
            if row == 0:
                axis.set_title(
                    PARENT_LABEL if budget == 0 else f"{budget:,}",
                    fontsize=10,
                    color=INK,
                    pad=7,
                )
        axes[row, 0].set_ylabel(f"Replicate {replicate}\nhalf-length (m)", fontsize=10)
    for axis in axes[-1]:
        axis.set_xlabel("Initial angle (deg)", fontsize=9, color=INK_SECONDARY)
    figure.suptitle(
        f"{ARM_LABELS[arm]}: coverage shifts in each training seed",
        x=0.06,
        y=0.98,
        ha="left",
        fontsize=15,
        color=INK,
    )
    figure.text(
        0.06,
        0.93,
        "64 deterministic evaluation episodes per condition. "
        "Columns show new training transitions. "
        "Black: parent's accepted region (rate at least 0.9).\n"
        "Half-length is rank-spaced. Every checkpoint and seed is retained; "
        "development diagnostic.",
        ha="left",
        va="top",
        fontsize=9,
        color=INK_SECONDARY,
    )
    figure.subplots_adjust(left=0.075, right=0.92, top=0.84, bottom=0.075)
    scale = figure.colorbar(shown, cax=figure.add_axes((0.94, 0.15, 0.009, 0.64)))
    scale.set_label("Recovery rate", fontsize=9)
    scale.ax.tick_params(length=0, labelsize=8)
    scale.outline.set_visible(False)
    save(figure, path)


def band_counts(children: dict, parent: dict, select) -> dict:
    """Return accepted-condition counts in a band for each arm and replicate."""

    baseline = sum(rate >= TAU for point, rate in parent.items() if select(point[1]))
    series: dict = defaultdict(lambda: defaultdict(dict))
    for (arm, replicate, budget), values in children.items():
        series[arm][replicate][budget] = sum(
            rate >= TAU for point, rate in values.items() if select(point[1])
        )
        series[arm][replicate][0] = baseline
    return series


def transition_shares(study: Path, tier: str, conditions: dict, order: list) -> dict:
    """Return each arm's realized transition share by half-length, per replicate."""

    shares: dict = defaultdict(list)
    for child in sorted((study / "policies" / tier).iterdir()):
        path = child / "training-summary.json"
        if not path.is_file():
            continue
        summary = json.loads(path.read_text())
        realized = summary["realized_sampling"]
        totals = dict.fromkeys(order, 0.0)
        for identifier, share in zip(
            realized["condition_ids"], realized["transition_shares"], strict=True
        ):
            totals[conditions[identifier]["length"]] += share
        shares[summary["arm"]].append([totals[length] for length in order])
    return shares


def style_axis(axis) -> None:
    axis.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(AXIS)
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.tick_params(length=0, labelsize=8.5, labelcolor=INK_SECONDARY)


def bands_figure(data: dict, shares: dict, lengths: list, path: Path) -> None:
    """Draw accepted counts by band, then where training transitions went."""

    arms = present_arms(data)
    figure = plt.figure(figsize=(12.4, 7.6))
    figure.patch.set_facecolor(SURFACE)
    layout = figure.add_gridspec(
        2,
        3,
        left=0.06,
        right=0.985,
        top=0.8,
        bottom=0.085,
        hspace=0.5,
        wspace=0.1,
        height_ratios=(1.15, 1),
    )
    band_axes = [figure.add_subplot(layout[0, index]) for index in range(3)]
    for axis in band_axes[1:]:
        axis.sharey(band_axes[0])
        axis.tick_params(labelleft=False)
    for axis, (title, select) in zip(band_axes, BANDS, strict=True):
        style_axis(axis)
        counts = band_counts(data["children"], data["parent"], select)
        for arm in arms:
            replicates = counts[arm]
            budgets = sorted(next(iter(replicates.values())))
            for values in replicates.values():
                axis.plot(
                    budgets,
                    [values[budget] for budget in budgets],
                    color=ARM_COLORS[arm],
                    linewidth=0.8,
                    alpha=0.35,
                )
            mean = [
                np.mean([values[budget] for values in replicates.values()])
                for budget in budgets
            ]
            axis.plot(
                budgets,
                mean,
                color=ARM_COLORS[arm],
                linewidth=2,
                marker="o",
                markersize=4.5,
                markeredgecolor=SURFACE,
                markeredgewidth=1,
            )
        axis.set_title(title, fontsize=10, color=INK, loc="left", pad=6)
        ticks = np.linspace(0, max(key[2] for key in data["children"]), 5)
        axis.set_xticks(ticks)
        axis.set_xticklabels(["0" if v == 0 else f"{v / 1000:g}k" for v in ticks])
        axis.set_xlabel("New training transitions", fontsize=9, color=INK_SECONDARY)
        axis.margins(x=0.03)
    band_axes[0].set_ylabel("Accepted conditions", fontsize=9, color=INK_SECONDARY)
    band_axes[0].set_ylim(bottom=0)

    axis = figure.add_subplot(layout[1, :])
    style_axis(axis)
    positions = np.arange(len(lengths))
    for arm in arms:
        mean = np.mean(shares[arm], axis=0)
        axis.plot(
            positions,
            mean * 100,
            color=ARM_COLORS[arm],
            linewidth=2,
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1,
        )
        index, x_offset, y_offset, alignment = DIRECT_LABELS[arm]
        axis.annotate(
            ARM_LABELS[arm],
            (positions[index], mean[index] * 100),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha=alignment,
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels([f"{value:g}" for value in lengths])
    axis.set_xlabel("Half-length (m), rank-spaced", fontsize=9, color=INK_SECONDARY)
    axis.set_ylabel("Training transitions (%)", fontsize=9, color=INK_SECONDARY)
    axis.set_ylim(bottom=0)
    axis.set_title(
        "Where training transitions went, by half-length",
        fontsize=10,
        color=INK,
        loc="left",
        pad=6,
    )
    handles = [
        Line2D(
            [],
            [],
            color=ARM_COLORS[arm],
            linewidth=2,
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            label=ARM_LABELS[arm],
        )
        for arm in arms
    ]
    figure.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.052, 0.875),
        ncol=len(arms),
        frameon=False,
        fontsize=9.5,
        labelcolor=INK,
        handlelength=1.8,
    )
    figure.suptitle(
        "Acceptance by pole length and training exposure",
        x=0.06,
        y=0.98,
        ha="left",
        fontsize=15,
        color=INK,
    )
    figure.text(
        0.06,
        0.93,
        "Accepted means a recovery rate of at least 0.9 at 64 episodes. "
        "Thick lines are arm means and thin lines are the 3 replicates; "
        "0 is the parent.\n"
        "Development data: does not rank the arms.",
        ha="left",
        va="top",
        fontsize=9,
        color=INK_SECONDARY,
    )
    save(figure, path)


def save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        figure.savefig(
            path.with_suffix(suffix), dpi=200, facecolor=figure.get_facecolor()
        )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--tier", default="pilot")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare-study", type=Path)
    args = parser.parse_args()
    output = args.output or args.study / "figures" / args.tier
    config = json.loads((args.study / "protocol/resolved-config.json").read_text())
    data = load_cells(args.study, args.tier, config["parent_policy_id"])
    distributions = json.loads(
        (args.study / "distributions/distributions.json").read_text()
    )
    record = json.loads((args.study / "protocol/conditions.json").read_text())
    conditions = {
        item["condition_id"]: {
            "angle": item["perturbation"]["parameters"]["initial_theta_deg"],
            "length": item["perturbation"]["parameters"]["length"],
        }
        for item in record["supports"]["discovery"]["conditions"]
    }
    lengths = sorted({point[1] for point in data["parent"]})
    shares = transition_shares(args.study, args.tier, conditions, lengths)
    budgets = sorted({key[2] for key in data["children"]})
    windows = training_windows(
        args.study, args.tier, budgets, distributions["condition_ids"]
    )
    maps_figure(
        data, distributions, conditions, windows, output / "coverage-shift-maps"
    )
    bands_figure(data, shares, lengths, output / "coverage-shift-bands")
    if len(present_arms(data)) == 1:
        replicate_maps_figure(data, output / "coverage-shift-replicates")
    if args.compare_study:
        baseline_config = json.loads(
            (args.compare_study / "protocol/resolved-config.json").read_text()
        )
        baseline = load_cells(
            args.compare_study, args.tier, baseline_config["parent_policy_id"]
        )
        comparison_figure(baseline, data, output / "coverage-shift-comparison")
    print(f"Wrote figures to {output}.")


if __name__ == "__main__":
    main()
