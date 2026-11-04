"""Regenerate the midterm figure from retained artifacts, no experiments."""

import argparse
import json
import os
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
# Target colors stay apart from panel (b): this red clears the separation checks
# against its orange, which the palette's own red does not.
BOUNDARY, UNIFORM = "#b3261e", "#8c8c8c"
SHAPES = {"erm_large_step": "o", "erm_small_step": "s"}
# Panel (c) keeps its legend above the axes; all titles clear that height.
TITLE_PAD = 19
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
        1, 3, figsize=(6.0, 1.95), width_ratios=(0.9, 1.25, 1.05), layout="constrained"
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
