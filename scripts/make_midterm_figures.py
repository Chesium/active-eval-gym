"""Regenerate the midterm figure from retained artifacts, no experiments."""

import argparse
import json
import os
import re
import tempfile
from hashlib import sha256
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE_RUN = ROOT / "artifacts/certificate_generalization/20261021T171258986368Z"
CROSSED_REPORT = ROOT / "artifacts/linear-crossed-study-v1-report"
DEFAULT_OUTPUT = ROOT / "ese6180/Evaluation_Guided_Domain_Randomization/midterm_assets"
FAMILIES = {
    "same_shape": "Same shape",
    "general": "Random",
    "rotated_actuator": "Rotated",
    "mirrored": "Mirrored",
}
# Validated palette slots; each hue follows its rule across both panels.
RULES = {
    "erm_large_step": ("Max", "#2a78d6"),
    "erm_first": ("First tie", "#1baf7a"),
    "erm_small_step": ("Min", "#eb6834"),
}
INK, MUTED = "#333333", "#6b6b6b"
BLUE = "#2a78d6"
# Target colors stay apart from panel (b): this red clears the separation checks
# against its orange, which the palette's own red does not.
BOUNDARY, UNIFORM = "#b3261e", "#8c8c8c"
SHAPES = {"erm_large_step": "o", "erm_small_step": "s"}
# Panel (c) keeps its legend above the axes; all titles clear that height.
TITLE_PAD = 19
# The plotted window; the data file keeps every checkpoint through 256,000.
PPO_ROUNDS = 6
PPO_TRANSITIONS_PER_ROUND = 25_600
PPO_PARENT = "cartpole_ppo_nominal_v1"
START_COVERAGE = 53.9


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def refit_values(summary: dict) -> dict:
    """Mean final coverage under background, for panel (b)."""
    values = {}
    for family in FAMILIES:
        cells = summary["families"][family]["cells"]
        values[family] = {
            rule: {
                mode: cells[f"{rule}_{mode}_background"]["mean"]
                for mode in ("fixed", "refit")
            }
            for rule in RULES
        }
    return values


def target_values(summary: dict) -> dict:
    """Boundary minus background with fixed P, for panel (c)."""
    values = {}
    for family in FAMILIES:
        target = summary["families"][family]["target"]
        values[family] = {
            rule: {
                moment: {
                    "mean": target[f"{rule}_fixed_{moment}"]["mean"],
                    "ci95": target[f"{rule}_fixed_{moment}"]["mean_ci95"],
                }
                for moment in ("first", "last")
            }
            for rule in ("erm_large_step", "erm_small_step")
        }
    return values


def bound_panel(axis, sizes, low, median, high, bounds) -> None:
    axis.fill_between(sizes, low, high, color="#2a78d6", alpha=0.18, linewidth=0)
    axis.plot(
        sizes,
        bounds,
        "s--",
        color=MUTED,
        markersize=2.5,
        linewidth=0.8,
        label=r"Bound $\Delta_N$",
    )
    axis.plot(
        sizes,
        median,
        "o-",
        color="#2a78d6",
        markersize=2.5,
        linewidth=0.8,
        label="Largest gap",
    )
    axis.set(
        xscale="log",
        yscale="log",
        xlabel="Trajectories $N$",
        ylabel="Loss gap",
        ylim=(0.004, 1.3),
    )
    axis.set_title("(a) Generalization", pad=TITLE_PAD)
    axis.set_xticks(sizes[::2], [str(n) for n in sizes[::2]])
    axis.minorticks_off()
    axis.legend(loc="lower left", frameon=False, handlelength=1.8, borderaxespad=0.1)


def refit_panel(axis, values: dict) -> None:
    from matplotlib.lines import Line2D

    axis.axvline(START_COVERAGE, color=MUTED, linewidth=0.6, linestyle=":")
    axis.text(START_COVERAGE + 0.8, -0.4, "Start", color=MUTED, fontsize=6.5)
    for row, family in enumerate(FAMILIES):
        for offset, (rule, (_, color)) in zip(
            (0.26, 0, -0.26), RULES.items(), strict=True
        ):
            fixed = values[family][rule]["fixed"]
            refit = values[family][rule]["refit"]
            y = row - offset
            axis.plot([fixed, refit], [y, y], color=color, linewidth=0.8, zorder=2)
            axis.plot(
                fixed,
                y,
                "o",
                markersize=3.4,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.8,
                zorder=3,
            )
            axis.plot(refit, y, "o", markersize=3.4, color=color, zorder=3)
    axis.set(
        xlim=(50, 100),
        ylim=(len(FAMILIES) - 0.45, -0.55),
        xlabel="Final coverage (%)",
    )
    axis.set_title("(b) Tie-break and refitting", pad=TITLE_PAD)
    axis.set_yticks(range(len(FAMILIES)), list(FAMILIES.values()))
    handles = [
        Line2D([], [], color=color, marker="o", markersize=3, linewidth=0, label=name)
        for name, color in RULES.values()
    ]
    handles += [
        Line2D(
            [],
            [],
            color=MUTED,
            marker="o",
            markersize=3,
            linewidth=0,
            markerfacecolor="white",
            label="Fixed",
        ),
        Line2D(
            [], [], color=MUTED, marker="o", markersize=3, linewidth=0, label="Refit"
        ),
    ]
    axis.legend(
        handles=handles,
        loc="lower right",
        ncols=3,
        frameon=False,
        columnspacing=0.5,
        handletextpad=0.05,
        borderaxespad=0.1,
    )


def target_panel(axis, values: dict) -> None:
    """Every mark is boundary relative to uniform, so uniform is the zero line."""
    from matplotlib.lines import Line2D

    axis.axvline(0, color=UNIFORM, linewidth=1.2, zorder=1)
    levels = {
        ("erm_large_step", "first"): -0.3,
        ("erm_large_step", "last"): -0.12,
        ("erm_small_step", "first"): 0.12,
        ("erm_small_step", "last"): 0.3,
    }
    for row, family in enumerate(FAMILIES):
        for (rule, moment), offset in levels.items():
            point = values[family][rule][moment]
            y = row + offset
            axis.plot(point["ci95"], [y, y], color=BOUNDARY, linewidth=0.7, zorder=2)
            axis.plot(
                point["mean"],
                y,
                SHAPES[rule],
                markersize=3.2,
                markeredgecolor=BOUNDARY,
                markeredgewidth=0.8,
                markerfacecolor=BOUNDARY if moment == "last" else "white",
                zorder=3,
            )
    axis.set(xlabel="Boundary minus uniform (points)")
    axis.set_title("(c) Target and family", pad=TITLE_PAD)
    axis.tick_params(axis="y", length=0)

    def key(label, marker="o", color=INK, face=None):
        return Line2D(
            [],
            [],
            color=color,
            marker=marker,
            markersize=3,
            linewidth=0,
            markerfacecolor=face or color,
            label=label,
        )

    handles = [
        key("Boundary", color=BOUNDARY),
        Line2D([], [], color=UNIFORM, linewidth=1.2, label="Uniform"),
        key("Max"),
        key("Min", marker="s"),
        key("Round 1", face="white"),
        key("Round 3"),
    ]
    axis.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.99),
        ncols=3,
        frameon=False,
        columnspacing=0.7,
        handlelength=1.0,
        handletextpad=0.25,
        borderaxespad=0,
        labelspacing=0.15,
    )


def parse_condition(condition: str) -> tuple[int, float]:
    match = re.fullmatch(r"theta-([mp])(\d+)_length-p(\d+)(?:p(\d+))?", condition)
    if match is None:
        raise ValueError(f"Unrecognized condition: {condition}")
    sign, angle, whole, fraction = match.groups()
    return (-1 if sign == "m" else 1) * int(angle), float(f"{whole}.{fraction or 0}")


def edges(values: list[float]) -> np.ndarray:
    return np.arange(len(values) + 1) - 0.5


def angle_edges(angles: list[int]) -> np.ndarray:
    step = angles[1] - angles[0]
    return np.append(np.array(angles) - step / 2, angles[-1] + step / 2)


def condition_mask(conditions, angles, lengths) -> np.ndarray:
    mask = np.zeros((len(lengths), len(angles)), dtype=bool)
    for condition in conditions:
        angle, length = parse_condition(condition)
        mask[lengths.index(length), angles.index(angle)] = True
    return mask


def outline(axis, mask: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> None:
    """Trace cell boundaries so the outline stays on the grid."""
    padded = np.pad(mask, 1)
    for row, column in np.argwhere(padded[:-1, :] != padded[1:, :]):
        y = y_edges[row]
        axis.plot(x_edges[[column - 1, column]], [y, y], color=INK, linewidth=0.5)
    for row, column in np.argwhere(padded[:, :-1] != padded[:, 1:]):
        x = x_edges[column]
        axis.plot([x, x], y_edges[[row - 1, row]], color=INK, linewidth=0.5)


def ppo_counts(data: dict) -> dict:
    """Held-out accepted counts by seed and round, for the figure and provenance."""
    threshold = data["acceptance_threshold"]

    def accepted(checkpoint: str) -> set:
        record = data["reference"][checkpoint]
        total = record["episodes_per_condition"]
        return {
            c for c, n in record["recovery_counts"].items() if n / total >= threshold
        }

    parent = accepted(PPO_PARENT)
    counts = {"parent": len(parent), "seeds": {}}
    for seed in range(3):
        rows = []
        for round_index in range(1, 11):
            transitions = round_index * PPO_TRANSITIONS_PER_ROUND
            name = f"{data['study_id']}-r{seed:03d}-t{transitions:06d}"
            current = accepted(name)
            rows.append(
                {
                    "transitions": transitions,
                    "accepted": len(current),
                    "parent_lost": len(parent - current),
                    "nominal_recovery": data["reference"][name]["recovery_counts"].get(
                        "theta-p0_length-p0p5", 0
                    ),
                }
            )
        counts["seeds"][str(seed)] = rows
    return counts


def ppo_figure(output: Path, plt) -> None:
    """Interleave held-out accepted sets with the frontier that trains next."""
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    data = json.loads((output / "ppo_loop_data.json").read_text())
    threshold = data["acceptance_threshold"]
    reference = [parse_condition(c) for c in data["reference_conditions"]]
    angles = sorted({angle for angle, _ in reference})
    lengths = sorted({length for _, length in reference})
    target_angles = angles[::2]

    def accepted(checkpoint: str) -> np.ndarray:
        record = data["reference"][checkpoint]
        total = record["episodes_per_condition"]
        chosen = [
            c for c, n in record["recovery_counts"].items() if n / total >= threshold
        ]
        return condition_mask(chosen, angles, lengths)

    def checkpoint(seed: int, round_index: int) -> str:
        if round_index == 0:
            return PPO_PARENT
        transitions = round_index * PPO_TRANSITIONS_PER_ROUND
        return f"{data['study_id']}-r{seed:03d}-t{transitions:06d}"

    panels = [
        [
            (
                accepted(checkpoint(seed, k)),
                condition_mask(
                    data["targets"][f"{seed}/{k}"]["frontier"],
                    target_angles,
                    lengths,
                )
                if k < PPO_ROUNDS
                else None,
            )
            for k in range(PPO_ROUNDS + 1)
        ]
        for seed in range(3)
    ]
    occupied = [
        np.flatnonzero(mask.any(axis=1)).min()
        for row in panels
        for pair in row
        for mask in pair
        if mask is not None
    ]
    first_row = max(min(occupied) - 1, 0)
    parent = accepted(PPO_PARENT)
    x_edges, target_edges = angle_edges(angles), angle_edges(target_angles)
    y_edges = edges(lengths)
    accepted_map = ListedColormap(["#f4f4f2", BLUE])
    target_map = ListedColormap(["#f7ece9", BOUNDARY])
    columns = 2 * PPO_ROUNDS + 1
    fig, axes = plt.subplots(
        3, columns, figsize=(6.0, 1.16), sharex=True, sharey=True, layout="constrained"
    )
    fig.get_layout_engine().set(wspace=0, hspace=0, w_pad=0.004, h_pad=0.004)
    for seed, row in enumerate(panels):
        for k, (current, target) in enumerate(row):
            axis = axes[seed, 2 * k]
            axis.pcolormesh(
                x_edges,
                y_edges,
                current,
                cmap=accepted_map,
                vmin=0,
                vmax=1,
                rasterized=True,
            )
            outline(axis, parent, x_edges, y_edges)
            axis.text(
                0.04,
                0.95,
                str(int(current.sum())),
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=5.5,
                color=INK,
            )
            if seed == 0:
                label = round(k * PPO_TRANSITIONS_PER_ROUND / 1000)
                title = "Parent" if k == 0 else f"{label}k"
                axis.set_title(title, fontsize=6.5, pad=1.5)
            if k == 0:
                axis.set_ylabel(f"Seed {seed}", fontsize=5.5, labelpad=1)
            if target is not None:
                axis = axes[seed, 2 * k + 1]
                axis.pcolormesh(
                    target_edges,
                    y_edges,
                    target,
                    cmap=target_map,
                    vmin=0,
                    vmax=1,
                    rasterized=True,
                )
                if seed == 0:
                    axis.set_title(r"$\rightarrow$", fontsize=6.5, pad=1.5)
    for axis in axes.flat:
        axis.set_xlim(-55, 55)
        axis.set_ylim(first_row - 0.5, len(lengths) - 0.5)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.legend(
        handles=[
            Patch(color=BLUE, label="Accepted set"),
            Line2D([], [], color=INK, linewidth=0.6, label="Parent set"),
            Patch(color=BOUNDARY, label="Training frontier"),
            Patch(color="#f7ece9", label="Background"),
        ],
        loc="outside lower center",
        ncols=4,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.2,
        handletextpad=0.4,
        borderaxespad=0,
    )
    fig.savefig(
        output / "ppo_loop.pdf",
        dpi=600,
        metadata={
            "Title": "Acceptance sets and training frontiers",
            "Author": "Shimin Chen",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(output / "ppo_loop.png", dpi=200)
    plt.close(fig)


def make_figures(output: Path) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "active-eval-gym-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    certificate_metadata = json.loads((CERTIFICATE_RUN / "metadata.json").read_text())
    summary = json.loads((CROSSED_REPORT / "summary.json").read_text())
    with np.load(CERTIFICATE_RUN / "metrics_v1.npz") as data:
        sizes = data["sample_sizes"]
        gaps = data["grid_gap"]
        bounds = data["bound"]
        epsilon = data["epsilon"]
        candidate_count = len(data["candidates"])
    low, median, high = np.quantile(gaps, (0.1, 0.5, 0.9), axis=1)
    coverage, gaps_by_target = refit_values(summary), target_values(summary)

    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.5,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(6.0, 1.72), width_ratios=(0.9, 1.25, 1.05), layout="constrained"
    )
    fig.get_layout_engine().set(wspace=0.02)
    bound_panel(axes[0], sizes, low, median, high, bounds)
    refit_panel(axes[1], coverage)
    target_panel(axes[2], gaps_by_target)
    axes[2].sharey(axes[1])
    axes[2].tick_params(labelleft=False)

    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output / "linear_results.pdf",
        metadata={
            "Title": "Generalization and component effects",
            "Author": "Shimin Chen",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(output / "linear_results.png", dpi=200)
    plt.close(fig)
    ppo_figure(output, plt)
    ppo_data = json.loads((output / "ppo_loop_data.json").read_text())
    sources = (
        CERTIFICATE_RUN / "metadata.json",
        CERTIFICATE_RUN / "metrics_v1.npz",
        CROSSED_REPORT / "summary.json",
    )
    metadata = {
        "schema_version": 2,
        "script_sha256": digest(Path(__file__)),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "certificate": {
            "spec": certificate_metadata["spec"],
            "sample_sizes": sizes.tolist(),
            "replicates": gaps.shape[1],
            "grid_candidates": candidate_count,
            "delta_per_sample_size": certificate_metadata["delta_per_sample_size"],
            "epsilon": epsilon.tolist(),
            "epsilon_selection": "Bound constants only, before sampled losses",
            "grid_gap_median": median.tolist(),
            "grid_gap_10_percentile": low.tolist(),
            "grid_gap_90_percentile": high.tolist(),
            "full_class_bound": bounds.tolist(),
            "description": "Maximum observed loss gap over numerical grid; the "
            "theorem controls the full continuous class. Shading is dataset "
            "10th–90th percentile, not a confidence interval.",
        },
        "crossed_study": {
            "run_manifest_sha256": summary["manifest_sha256"],
            "study_spec": summary["study_spec"],
            "panel_b_mean_final_coverage_background": coverage,
            "panel_c_boundary_minus_background_fixed_p": gaps_by_target,
            "families_with_shared_certificate": {
                family: summary["families"][family]["shared_certificate"]
                for family in FAMILIES
            },
            "description": "Means over families with a shared certificate. Halted "
            "branches count at their current coverage. Panel (c) shows boundary minus "
            "uniform for fixed-P Max and Min arms; every arm is in the study summary.",
        },
        "ppo_loop": {
            "study_id": ppo_data["study_id"],
            "status": ppo_data["status"],
            "data_sha256": digest(output / "ppo_loop_data.json"),
            "study_summary_sha256": ppo_data["summary_sha256"],
            "plotted_window_transitions": PPO_ROUNDS * PPO_TRANSITIONS_PER_ROUND,
            "held_out_counts_all_rounds": ppo_counts(ppo_data),
            "description": "Blue sets are held-out reference evaluations; red "
            "frontiers are the discovery results that chose each round. All three "
            "seeds are shown, and every later round remains in this record.",
        },
    }
    (output / "figure_provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(output / "linear_results.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    make_figures(args.output_dir)


if __name__ == "__main__":
    main()
