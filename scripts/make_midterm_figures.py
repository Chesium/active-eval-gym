"""Regenerate the midterm figure from retained artifacts, without experiments."""

import argparse
import json
import os
import tempfile
from hashlib import sha256
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE_RUN = ROOT / "artifacts/certificate_generalization/20261021T171258986368Z"
UPDATE_RUN = ROOT / "artifacts/linear-study-smoke-v2"
DEFAULT_OUTPUT = ROOT / "ese6180/Evaluation_Guided_Domain_Randomization/midterm_assets"
TARGETS = ("background", "boundary", "high_cost")
SELECTORS = ("known", "erm_small_step", "erm_large_step")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def make_figures(output: Path) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "active-eval-gym-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    certificate_metadata = json.loads((CERTIFICATE_RUN / "metadata.json").read_text())
    update_manifest = json.loads((UPDATE_RUN / "manifest.json").read_text())
    with np.load(CERTIFICATE_RUN / "metrics_v1.npz") as data:
        sizes = data["sample_sizes"]
        gaps = data["grid_gap"]
        bounds = data["bound"]
        epsilon = data["epsilon"]
        candidate_count = len(data["candidates"])
    low, median, high = np.quantile(gaps, (0.1, 0.5, 0.9), axis=1)
    records = update_manifest["records"]
    gains = {}
    for selector in SELECTORS:
        for target in TARGETS:
            selected = sorted(
                (
                    r
                    for r in records
                    if r["study"] == "one_step"
                    and r["sample_size"] == 32
                    and r["selector"] == selector
                    and r["target"] == target
                ),
                key=lambda r: r["replicate"],
            )
            expected = list(range(update_manifest["study_spec"]["replicates"]))
            if [r["replicate"] for r in selected] != expected:
                raise ValueError("Missing or duplicated predeclared replicate.")
            gains[selector, target] = np.array(
                [100 * r["summary"]["coverage_change"] for r in selected]
            )

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, (left, right) = plt.subplots(1, 2, figsize=(6, 1.94), layout="constrained")
    left.fill_between(sizes, low, high, color="#2563a6", alpha=0.18)
    left.plot(
        sizes, median, "o-", color="#2563a6", markersize=3, label="Numerical-grid gap"
    )
    left.plot(
        sizes,
        bounds,
        "s--",
        color="#555555",
        markersize=3,
        label=r"Full-class bound $\Delta_N$",
    )
    left.set(
        xscale="log",
        yscale="log",
        xlabel=r"Independent trajectories $N$",
        ylabel="Loss gap / bound",
        title="(a) Certificate generalization",
    )
    left.set_xticks(sizes, [str(n) for n in sizes])
    left.minorticks_off()
    left.grid(alpha=0.18)
    left.legend(loc="upper right", frameon=False, handlelength=1.5)

    colors = ("#2563a6", "#d97706", "#7c3aed")
    labels = ("Background", "Boundary", "High-cost")
    positions = np.arange(len(SELECTORS))
    width = 0.23
    for index, (target, color, label) in enumerate(
        zip(TARGETS, colors, labels, strict=True)
    ):
        x = positions + (index - 1) * width
        values = [gains[selector, target] for selector in SELECTORS]
        right.bar(x, [v.mean() for v in values], width, color=color, label=label)
        for position, observations in zip(x, values, strict=True):
            jitter = np.linspace(-0.035, 0.035, len(observations))
            right.scatter(
                position + jitter,
                observations,
                s=11,
                facecolors="white",
                edgecolors="#222222",
                linewidths=0.5,
                zorder=3,
            )
    right.set_xticks(
        positions, [r"Known $P_0$", "Min-capacity\nERM", "Max-capacity\nERM"]
    )
    right.set(
        ylabel="Coverage gain\n(percentage points)",
        ylim=(0, 10.2),
        title="(b) One controller update",
    )
    right.set_yticks((0, 4, 8))
    right.legend(
        loc="upper center",
        ncols=3,
        frameon=False,
        columnspacing=0.7,
        handlelength=0.9,
        handletextpad=0.3,
    )
    right.grid(axis="y", alpha=0.18)
    right.set_axisbelow(True)

    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output / "linear_results.pdf",
        metadata={
            "Title": "Linear certificate and target-map preliminary results",
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
        UPDATE_RUN / "manifest.json",
    )
    metadata = {
        "schema_version": 1,
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
        "updates": {
            "study_spec": update_manifest["study_spec"],
            "selection": "All one_step N=32 records for the three named selectors",
            "coverage_gains_percentage_points": {
                selector: {
                    target: gains[selector, target].tolist() for target in TARGETS
                }
                for selector in SELECTORS
            },
            "description": "Bars average both predeclared replicates; dots retain "
            "both observations. Initial reference coverage is 0.5390625.",
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
