"""Render the compact, paper-width recovery-rate panels used in the proposal.

Reads the stored `cartpole-failure-boundary-v1` evaluation summary and reuses the
repository's boundary-plot helpers, so the figure stays consistent with
`docs/findings.md`. Runs from anywhere: `python make_recovery_regions.py`.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(REPO / "src"))
from active_eval_gym import plotting as P

root = REPO / "artifacts/evaluations/cartpole-failure-boundary-v1/final"
summary = json.loads((root / "analysis/episode-summary-v4/summary.json").read_text())

policies = summary["policy_ids"]
angles = sorted({i["parameters"]["initial_theta_deg"] for i in summary["conditions"]})
lengths = sorted({i["parameters"]["length"] for i in summary["conditions"]})
lookup = P._boundary_lookup(summary)

surfaces = [
    P._boundary_surface(lookup, angles, lengths,
                        lambda item, p=pol: item["policies"][p]["recovery_rate"])
    for pol in policies
]

plt.rcParams.update({"font.size": 6, "axes.titlesize": 7, "axes.labelsize": 6})
fig, axes = plt.subplots(1, len(policies), figsize=(6.5, 1.45),
                         constrained_layout=True, squeeze=False)
cmap = P._boundary_colormap("RdYlBu", len(P._BOUNDARY_LEVELS) - 1)
norm = P._boundary_norm()

for col, (pol, surf) in enumerate(zip(policies, surfaces)):
    ax = axes[0, col]
    img = ax.imshow(surf, origin="lower", aspect="auto", interpolation="nearest",
                    cmap=cmap, norm=norm)
    filled = P._nearest_filled(surf)
    if filled.min() < 0.5 < filled.max():
        ax.contour(filled, levels=[0.5], colors="k", linewidths=0.7)
    rows = P._tick_positions(len(angles), 5)
    ax.set_yticks(rows, labels=[f"{angles[i]:g}" for i in rows])
    cols = P._tick_positions(len(lengths), 5)
    ax.set_xticks(cols, labels=[f"{lengths[i]:.2g}" for i in cols])
    ax.tick_params(labelsize=5, length=2, pad=1)
    ax.set_xlabel("pole half-length", labelpad=1)
    if col == 0:
        ax.set_ylabel("initial angle (deg)", labelpad=1)
    interior = int(np.sum((surf > 0.0) & (surf < 1.0)))
    ax.set_title(f"{P._short_policy(pol)} ({interior} interior)", fontsize=6.5, pad=2)

cb = fig.colorbar(img, ax=list(axes[0]), ticks=list(P._BOUNDARY_LEVELS),
                  spacing="uniform", fraction=0.025, pad=0.01)
cb.set_label("recovery rate", fontsize=6, labelpad=1)
cb.ax.tick_params(labelsize=5, length=2, pad=1)

out = Path(__file__).resolve().parent / "cartpole_recovery_regions.png"
fig.savefig(out, dpi=400)
print("wrote", out, plt.imread(out).shape)
