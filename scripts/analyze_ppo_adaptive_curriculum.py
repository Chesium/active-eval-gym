"""Artifact-only coverage, changing-frontier, and historical-comparison figures."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.lines import Line2D
from plot_ppo_boundary_shift import (
    BLUE_RAMP,
    ORANGE_RAMP,
    SURFACE,
    edges,
    outline,
    save,
    style_axis,
    style_map,
)

from active_eval_gym.ppo_adaptive_curriculum import read
from active_eval_gym.serialization import write_json_new


def cell_statistics(cells, parent):
    accepted = {c["condition_id"] for c in cells if c["recovery_rate"] >= 0.9}
    previous = {c["condition_id"] for c in parent if c["recovery_rate"] >= 0.9}
    nominal = next(
        (
            c["recovery_rate"]
            for c in cells
            if c["parameters"]["initial_theta_deg"] == 0
            and c["parameters"]["length"] == 0.5
        ),
        None,
    )
    return {
        "accepted": len(accepted),
        "coverage": len(accepted) / len(cells),
        "gained_parent": len(accepted - previous),
        "lost_parent": len(previous - accepted),
        "retained_parent": len(accepted & previous),
        "mean_recovery": float(np.mean([c["recovery_rate"] for c in cells])),
        "mean_survival": float(np.mean([c["survival_rate"] for c in cells])),
        "nominal_recovery": nominal,
    }


def rates(cells):
    return {
        (c["parameters"]["initial_theta_deg"], c["parameters"]["length"]): c[
            "recovery_rate"
        ]
        for c in cells
    }


def matrix(values, angles, lengths):
    return np.array(
        [[values.get((a, length), np.nan) for a in angles] for length in lengths]
    )


def add_map(axis, values, angles, lengths, *, cmap, norm=None):
    handle = axis.pcolormesh(
        edges(angles),
        np.arange(len(lengths) + 1) - 0.5,
        matrix(values, angles, lengths),
        cmap=cmap,
        norm=norm,
        vmin=0 if norm is None else None,
        vmax=1 if norm is None else None,
        rasterized=True,
    )
    style_map(axis)
    axis.set_xticks([-50, 0, 50], ["−50", "0", "+50"])
    axis.set_yticks(range(len(lengths)), [f"{v:g}" for v in lengths])
    return handle


def seed_maps(study, seed, protocol, parent_cells, output):
    base = protocol["base_spec"]
    lengths = base["supports"]["reference"]["length"]
    angles = base["supports"]["reference"]["initial_theta_deg"]
    discovery_angles = base["supports"]["discovery"]["initial_theta_deg"]
    discovery_lengths = base["supports"]["discovery"]["length"]
    points = [(a, length) for a in discovery_angles for length in discovery_lengths]
    blue = LinearSegmentedColormap.from_list("adaptive-blue", BLUE_RAMP)
    orange = LinearSegmentedColormap.from_list("adaptive-orange", ORANGE_RAMP)
    rounds = protocol["settings"]["rounds"]
    fig, axes = plt.subplots(
        3,
        rounds + 1,
        figsize=(2.65 * (rounds + 1), 10.5),
        squeeze=False,
        facecolor=SURFACE,
    )
    fig.subplots_adjust(
        left=0.055, right=0.955, top=0.79, bottom=0.10, hspace=0.34, wspace=0.12
    )
    fig.text(
        0.055, 0.95, f"Adaptive frontier: seed {seed['replicate_id']}", fontsize=20
    )
    fig.text(
        0.055,
        0.87,
        "Blue: reference recovery. Orange: desired / actual transition shares "
        "in that round.\n"
        "Black: accepted discovery region used to select that round’s frontier. "
        "Gray dashed: parent reference region.\n"
        "Frontier = unaccepted grid neighbours; 50% background + 50% frontier, "
        "with duration correction. Blank columns are unrun.",
        fontsize=11,
        color="#52514e",
    )
    parent_rates = rates(parent_cells)
    parent_set = {point for point, value in parent_rates.items() if value >= 0.9}
    blue_handle = add_map(axes[0, 0], parent_rates, angles, lengths, cmap=blue)
    outline(axes[0, 0], parent_set, angles, lengths, 5)
    axes[0, 0].set_title("Parent", fontsize=11)
    for row in (1, 2):
        axes[row, 0].axis("off")
    targets = [study / record["target_record"] for record in seed["rounds"]]
    largest_share = max(
        (max(read(p)["target_transition_shares"]) for p in targets), default=0.05
    )
    largest_share = max(
        largest_share,
        max(
            (max(r["interval_transition_shares"]) for r in seed["rounds"]), default=0.05
        ),
    )
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=largest_share)
    orange_handle = None
    for phase in range(rounds):
        column = phase + 1
        budget = (phase + 1) * protocol["settings"]["transitions_per_round"]
        axes[0, column].set_title(f"{budget:,}", fontsize=11)
        if phase >= len(seed["rounds"]):
            for row in range(3):
                axes[row, column].axis("off")
            axes[0, column].text(
                0.5,
                0.5,
                "Unrun",
                transform=axes[0, column].transAxes,
                ha="center",
                color="#898781",
            )
            continue
        record = seed["rounds"][phase]
        checkpoint_id = Path(record["checkpoint"]).name
        cells = read(
            study / "evaluations/reference" / checkpoint_id / "metrics_v1/summary.json"
        )["cells"]
        add_map(axes[0, column], rates(cells), angles, lengths, cmap=blue)
        # Reference parent remains a comparison; training outlines change below.
        outline(axes[0, column], parent_set, angles, lengths, 5)
        for line in axes[0, column].collections[1:]:
            line.set_color("#898781")
            line.set_linestyle("--")
        target = read(study / record["target_record"])
        accepted = {
            point
            for point, yes in zip(points, target["accepted_mask"], strict=True)
            if yes
        }
        for row, shares in (
            (1, target["target_transition_shares"]),
            (2, record["interval_transition_shares"]),
        ):
            values = dict(zip(points, shares, strict=True))
            orange_handle = add_map(
                axes[row, column],
                values,
                discovery_angles,
                discovery_lengths,
                cmap=orange,
                norm=norm,
            )
            outline(
                axes[row, column], accepted, discovery_angles, discovery_lengths, 10
            )
        for row in range(3):
            axes[row, column].set_yticklabels([])
    axes[0, 0].set_ylabel("Reference recovery\nPole half-length (m)")
    axes[1, 1].set_ylabel("Target transition shares\nPole half-length (m)")
    axes[2, 1].set_ylabel("Observed transition shares\nPole half-length (m)")
    for axis in axes[-1]:
        if axis.axison:
            axis.set_xlabel("Initial angle (deg)", fontsize=10)
    fig.colorbar(
        blue_handle,
        cax=fig.add_axes((0.967, 0.585, 0.008, 0.19)),
        label="Recovery rate",
    )
    if orange_handle is not None:
        bar = fig.colorbar(orange_handle, cax=fig.add_axes((0.967, 0.11, 0.008, 0.42)))
        bar.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda value, _: f"{100 * value:.0f}%")
        )
        bar.set_label("Transition share per condition (square-root scale)")
    save(fig, output)


def comparison_curve(study, rows, historical, parent_stats, output):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=SURFACE)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.72, bottom=0.19, wspace=0.25)
    fig.text(
        0.06, 0.94, "Refreshed frontier versus historical frozen frontier", fontsize=20
    )
    fig.text(
        0.06,
        0.82,
        "Solid: adaptive. Dashed: historical fixed-frontier run. Three original "
        "seeds; unchanged PPO and duration correction.\n"
        "Seeds stop when discovery has no accepted conditions. Lines end at "
        "actual evaluated checkpoints; no unrun budgets are filled.",
        fontsize=10,
        color="#52514e",
    )
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    for replicate, color in enumerate(colors):
        for records, linestyle in ((rows, "-"), (historical, "--")):
            selected = sorted(
                [
                    r
                    for r in records
                    if r["replicate_id"] == replicate and r["transitions"] <= 256000
                ],
                key=lambda r: r["transitions"],
            )
            if not selected:
                continue
            x = [0] + [r["transitions"] / 1000 for r in selected]
            for axis, key in zip(
                axes, ("accepted", "mean_recovery", "lost_parent"), strict=True
            ):
                axis.plot(
                    x,
                    [parent_stats[key]] + [r[key] for r in selected],
                    linestyle=linestyle,
                    marker="o",
                    markersize=3,
                    color=color,
                    alpha=0.9,
                )
    for axis, title in zip(
        axes,
        (
            "Accepted conditions / 273",
            "Mean recovery probability",
            "Lost parent-accepted conditions",
        ),
        strict=True,
    ):
        style_axis(axis)
        axis.set_title(title, loc="left", fontsize=12)
        axis.set_xlim(-5, 261)
        axis.set_xticks([0, 64, 128, 192, 256])
        axis.set_xlabel("New training transitions (thousands)")
    axes[0].legend(
        handles=[
            Line2D([0], [0], color=c, label=f"Seed {r}") for r, c in enumerate(colors)
        ],
        frameon=False,
    )
    save(fig, output)


def analyze(study: Path):
    if not (study / "completion.json").exists():
        raise ValueError("Adaptive study is not complete.")
    protocol = read(study / "protocol/adaptive-protocol.json")
    seeds = read(study / "training-complete.json")["seeds"]
    parent_id = protocol["base_spec"]["parent_policy_id"]
    parent_cells = read(
        study / "evaluations/reference" / parent_id / "metrics_v1/summary.json"
    )["cells"]
    parent_stats = cell_statistics(parent_cells, parent_cells)
    rows = []
    for seed in seeds:
        for record in seed["rounds"]:
            checkpoint_id = Path(record["checkpoint"]).name
            cells = read(
                study
                / "evaluations/reference"
                / checkpoint_id
                / "metrics_v1/summary.json"
            )["cells"]
            rows.append(
                {
                    "replicate_id": seed["replicate_id"],
                    "transitions": record["transitions"],
                    **cell_statistics(cells, parent_cells),
                }
            )
    historical = read(Path(protocol["settings"]["historical_summary"]))["rows"]
    result = {
        "study_id": protocol["settings"]["study_id"],
        "parent": parent_stats,
        "rows": rows,
        "seed_accounting": [
            {
                key: seed[key]
                for key in (
                    "replicate_id",
                    "completed_transitions",
                    "unrun_transitions",
                    "stop_reason",
                )
            }
            for seed in seeds
        ],
        "historical_rows": [r for r in historical if r["transitions"] <= 256000],
        "interpretation": (
            "Development comparison to historical controls; no fresh matched "
            "control, best-checkpoint selection, or confirmatory inference."
        ),
    }
    output = study / "analysis/summary.json"
    if output.exists():
        if read(output) != result:
            raise ValueError("Refusing to replace adaptive derived results.")
    else:
        output.parent.mkdir(exist_ok=True)
        write_json_new(output, result)
    for seed in seeds:
        seed_maps(
            study,
            seed,
            protocol,
            parent_cells,
            study / f"figures/pilot/coverage-shift-seed-{seed['replicate_id']:03d}",
        )
    comparison_curve(
        study,
        rows,
        historical,
        parent_stats,
        study / "figures/pilot/adaptive-vs-frozen-learning-curves",
    )
    print(json.dumps(result["seed_accounting"], indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        type=Path,
        default=Path("artifacts/experiments/cartpole-ppo-boundary-v5-adaptive"),
    )
    analyze(parser.parse_args().study)
