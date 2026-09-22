"""Reproduce the analytical figure in linear-control-and-retraining-explained.md.

Run with Python 3, NumPy, and Matplotlib; no RL environments or training needed.
Outputs an SVG beside this script. All conditions are deterministic.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def cost_matrix(g, q, horizon):
    p = np.zeros_like(q)
    power = np.eye(len(g))
    for _ in range(horizon):
        p += power.T @ q @ power
        power = g @ power
    return p


def main():
    a = b_plant = q = np.eye(2)
    k = np.diag([0.5, 0.5])
    k_new = np.diag([0.7, 0.3])
    horizon, threshold, radius = 2, 1.25, 1.2
    g, g_new = a - b_plant @ k, a - b_plant @ k_new
    p = cost_matrix(g, q, horizon)
    p_new = cost_matrix(g_new, q, horizon)
    drift = radius**2 * np.linalg.norm(p_new - p, 2)
    np.testing.assert_allclose(p, np.diag([1.25, 1.25]))
    np.testing.assert_allclose(p_new, np.diag([1.09, 1.49]))
    np.testing.assert_allclose(drift, 0.3456)

    starts = np.array([[1.0, 0.0], [0.0, 1.0], [1.04, 0.0]])
    for matrix, closed_loop in [(p, g), (p_new, g_new)]:
        for start in starts:
            state, rollout_cost = start.copy(), 0.0
            for _ in range(horizon):
                rollout_cost += state @ q @ state
                state = closed_loop @ state
            np.testing.assert_allclose(start @ matrix @ start, rollout_cost)

    axis = np.linspace(-radius, radius, 801)
    x, y = np.meshgrid(axis, axis)
    states = np.stack([x, y], axis=-1)
    old_cost = np.einsum("...i,ij,...j->...", states, p, states)
    new_cost = np.einsum("...i,ij,...j->...", states, p_new, states)
    domain = x*x + y*y <= radius**2
    old_ok = domain & (old_cost <= threshold)
    new_ok = domain & (new_cost <= threshold)
    protected = domain & (old_cost <= threshold - drift)
    assert np.all(new_ok[protected])
    loss_band = domain & (old_cost > threshold - drift) & old_ok
    assert np.all(loss_band[old_ok & ~new_ok])

    regions = np.zeros_like(x, dtype=int)
    regions[protected] = 1
    regions[new_ok & ~old_ok] = 2
    regions[old_ok & ~new_ok] = 3
    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(7.8, 7.2), layout="constrained")
    ax.contourf(x, y, regions, levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
                cmap=ListedColormap(["white", "#dbe1e8", "#68bf96", "#edb16b"]))
    angle = np.linspace(0, 2*np.pi, 1000)
    circle = np.array([np.cos(angle), np.sin(angle)])
    old_line, = ax.plot(*circle, color="#205c9b", lw=2, label="Old boundary")
    new_ellipse = np.sqrt(threshold / np.diag(p_new))[:, None] * circle
    new_line, = ax.plot(*new_ellipse, color="#783d93", lw=2, ls="--",
                        label="Updated boundary")
    domain_line, = ax.plot(*(radius * circle), color="#777777", lw=1.2, ls=":",
                           label="Evaluation domain")
    ax.axhline(0, color="#bbbbbb", lw=0.6, zorder=0)
    ax.axvline(0, color="#bbbbbb", lw=0.6, zorder=0)
    ax.set(xlim=(-1.28, 1.28), ylim=(-1.28, 1.28), xlabel="Initial state component 1",
           ylabel="Initial state component 2", aspect="equal",
           title="A stable controller update can gain and lose acceptable states")
    ax.legend(handles=[old_line, new_line, domain_line,
                       Patch(color="#dbe1e8", label="Guaranteed to remain acceptable"),
                       Patch(color="#68bf96", label="Newly acceptable"),
                       Patch(color="#edb16b", label="No longer acceptable")],
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
              frameon=False, fontsize=10)
    output = Path(__file__).with_suffix(".svg")
    fig.savefig(output, metadata={"Date": None})
    plt.close(fig)
    print(f"Wrote {output}")
    print(f"Uniform cost drift bound: {drift:.4f}")
    print(f"Protected radius: {np.sqrt((threshold-drift)/1.25):.6f}")
    print(f"Old and new geometric areas: {np.pi:.6f}, "
          f"{np.pi*threshold/np.sqrt(np.linalg.det(p_new)):.6f}")


if __name__ == "__main__":
    main()
