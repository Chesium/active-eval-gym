"""Matplotlib views and ipywidgets for certificate_generalization.ipynb."""

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display
from matplotlib.patches import Circle

from active_eval_gym.certificates import (
    KNOWN_CERTIFICATE,
    VERIFICATION_TOLERANCE,
    CandidateGrid,
    CertificateSpec,
    LossTable,
    Population,
    SweepResult,
    bound_terms,
    candidate_grid,
    certificate_matrix,
    choose_epsilon,
    empirical_losses,
    loss_table,
    make_population,
    normalized_residuals,
    run_sweep,
    sample_draws,
    save_experiment,
    verified_margins,
    wilson_interval,
)

BLUE, ORANGE, PURPLE, RED = "#2563a6", "#cc7b16", "#7851a9", "#c43d48"
COLORS = [BLUE, ORANGE, RED]
STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#fafbfe",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.16,
    "font.size": 10,
    "axes.titlesize": 11,
    "figure.dpi": 110,
}
DEFAULT_SPEC = CertificateSpec()


@dataclass(frozen=True)
class Lab:
    population: Population
    grid: CandidateGrid
    table: LossTable


@lru_cache(maxsize=8)
def prepare_lab(spec: CertificateSpec = DEFAULT_SPEC) -> Lab:
    population = make_population(spec)
    grid = candidate_grid(spec)
    return Lab(population, grid, loss_table(population, grid.matrices))


def setup_card(lab: Lab) -> HTML:
    p = lab.population
    margins = verified_margins(p.matrices, KNOWN_CERTIFICATE)
    stable_check = bool(
        np.all(margins[:2] >= p.spec.eta + p.spec.gamma + VERIFICATION_TOLERANCE)
    )
    return HTML(
        '<div style="border-left:5px solid #2563a6;padding:12px 18px;'
        'background:#f0f5fc;color:#172c45">'
        f"<b>{len(p.weights)} possible trajectories · {len(lab.grid.matrices):,} "
        f"candidate matrices · {p.spec.horizon} transitions per trajectory</b><br>"
        f"Frozen K = I · exact matrix dynamics · M = {p.dynamics_bound:.4f}<br>"
        f"Known P₀ = diag(0.6, 1.6): verified margins "
        f"{np.array2string(margins, precision=3)}. "
        f"Desired + fitting margin = {p.spec.eta + p.spec.gamma:.3f}.<br>"
        f"Stable-plant check: {'passed' if stable_check else 'unresolved'} "
        f"(eigenvalue tolerance {VERIFICATION_TOLERANCE:g}).<br>"
        "Population means sum over the entire declared distribution; "
        "no holdout estimate is substituted for truth.</div>"
    )


def _ellipse(ax, p, *, color=PURPLE, label="V(x) = 1", linewidth=2):
    values, vectors = np.linalg.eigh(p)
    theta = np.linspace(0, 2 * np.pi, 240)
    points = vectors @ (
        np.array([np.cos(theta), np.sin(theta)]) / np.sqrt(values[:, None])
    )
    ax.plot(*points, color=color, lw=linewidth, label=label)


def _phase_axes(ax):
    ax.axhline(0, color="#b1bac8", lw=0.8)
    ax.axvline(0, color="#b1bac8", lw=0.8)
    ax.set(
        xlabel="state x₁",
        ylabel="state x₂",
        aspect="equal",
        xlim=(-1.65, 1.65),
        ylim=(-1.65, 1.65),
    )


def plot_certificate(
    lab: Lab,
    *,
    eigenvalue_1=0.6,
    eigenvalue_2=1.6,
    angle_degrees=0.0,
    plant=0,
    direction=5,
    transition=None,
):
    population, spec = lab.population, lab.population.spec
    p = certificate_matrix(eigenvalue_1, eigenvalue_2, np.deg2rad(angle_degrees))
    own = loss_table(population, p)
    ids = np.flatnonzero(population.plant_ids == plant)
    chosen = ids[direction % spec.directions]
    path = population.states[chosen]
    transition = spec.horizon if transition is None else transition
    residual = normalized_residuals(p, population.states[[chosen]], spec.eta)[0, 0]
    margin = verified_margins(population.matrices[[plant]], p)[0]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(11.8, 8), layout="constrained")
        ax = axes[0, 0]
        _phase_axes(ax)
        for radius, label in [
            (1 / np.sqrt(spec.a), "outer radius 1/√a"),
            (1 / np.sqrt(spec.b), "inner radius 1/√b"),
        ]:
            ax.add_patch(
                Circle((0, 0), radius, fill=False, color="#99a5b5", ls=":", label=label)
            )
        _ellipse(ax, p)
        for index in ids[::8]:
            ax.plot(*population.states[index].T, color=BLUE, alpha=0.2, lw=1)
        ax.plot(
            *path[: transition + 1].T,
            "o-",
            ms=3,
            color=BLUE,
            label=f"selected trajectory, t = {transition}",
        )
        ax.scatter(*path[transition], s=65, color=ORANGE, zorder=6)
        bad = (residual > 0) & (np.arange(spec.horizon) < transition)
        ax.scatter(
            *path[:-1][bad].T, color=RED, s=35, zorder=5, label="violating transitions"
        )
        ax.set_title("Rotate / stretch the certificate; dynamics stay fixed")
        ax.legend(fontsize=8, loc="upper right")
        ax = axes[0, 1]
        degrees = np.rad2deg(population.angles[ids])
        worst = own.worst_residual[0, ids]
        ax.plot(degrees, worst, color=BLUE)
        ax.fill_between(degrees, 0, worst, where=worst > 0, color=RED, alpha=0.2)
        ax.axhline(0, color=RED, label="desired condition: s ≤ 0")
        ax.axhline(-spec.gamma, color=PURPLE, ls="--", label="zero loss: s ≤ −γ")
        ax.scatter(
            [degrees[direction % spec.directions]],
            [own.worst_residual[0, chosen]],
            color=ORANGE,
            s=60,
            zorder=4,
        )
        ax.set(
            xlabel="initial-state angle (degrees)",
            ylabel="worst residual s",
            title=f"G{plant + 1}: worst transition along each trajectory",
        )
        ax.legend(fontsize=8)
        ax = axes[1, 0]
        energy = np.einsum("ti,ij,tj->t", path, p, path)
        norm2 = np.sum(path**2, axis=-1)
        ax.semilogy(energy / energy[0], "o-", ms=3, color=PURPLE, label="V(xₜ) / V(x₀)")
        ax.semilogy(norm2 / norm2[0], "o-", ms=3, color=BLUE, label="‖xₜ‖² / ‖x₀‖²")
        ax.axvline(transition, color=ORANGE, ls=":", alpha=0.8)
        ax.set(
            xlabel="transition t",
            ylabel="relative energy / squared norm",
            title="Energy and Euclidean distance can behave differently",
        )
        ax.legend(fontsize=8)
        ax = axes[1, 1]
        s = own.worst_residual[0, chosen]
        domain = np.linspace(
            min(-2 * spec.gamma, s - spec.gamma / 3),
            max(spec.gamma, s + spec.gamma / 3),
            400,
        )
        ax.plot(domain, np.clip(1 + domain / spec.gamma, 0, 1), color=PURPLE, lw=2)
        ax.axvline(0, color=RED, alpha=0.5)
        ax.axvline(-spec.gamma, color=PURPLE, ls="--", alpha=0.5)
        ax.scatter([s], [own.loss[0, chosen]], s=85, color=ORANGE, zorder=4)
        ax.set(
            xlabel="trajectory's worst residual s",
            ylabel="trajectory loss ℓ",
            ylim=(-0.06, 1.06),
            title="Insufficient slack also incurs loss",
        )
        fig.suptitle(
            f"P = [[{p[0, 0]:.2f}, {p[0, 1]:.2f}], "
            f"[{p[1, 0]:.2f}, {p[1, 1]:.2f}]]   ·   "
            f"G{plant + 1} verified margin {margin:.3f}   ·   "
            f"population risk {own.population_risk[0]:.1%}",
            fontsize=12,
        )
    return fig


def geometry_controls(spec: CertificateSpec):
    controls = {
        "eigenvalue_1": widgets.FloatSlider(
            value=0.6, min=spec.a, max=spec.b, step=0.025, description="eigenvalue 1"
        ),
        "eigenvalue_2": widgets.FloatSlider(
            value=1.6, min=spec.a, max=spec.b, step=0.025, description="eigenvalue 2"
        ),
        "angle_degrees": widgets.FloatSlider(
            value=0, min=0, max=180, step=2, description="rotation °"
        ),
        "plant": widgets.Dropdown(
            options=[("G1", 0), ("G2", 1), ("G3: unstable stress case", 2)],
            value=0,
            description="plant",
        ),
        "direction": widgets.IntSlider(
            value=5, min=0, max=spec.directions - 1, description="reset index"
        ),
        "transition": widgets.IntSlider(
            value=spec.horizon, min=0, max=spec.horizon, description="transition t"
        ),
    }
    for control in controls.values():
        control.style.description_width = "initial"
        if hasattr(control, "continuous_update"):
            control.continuous_update = False

    def draw(**settings):
        # Previewing G3 explicitly activates the q=.1 stress law.
        q = 0.1 if settings["plant"] == 2 else spec.unsafe_probability
        fig = plot_certificate(
            prepare_lab(replace(spec, unsafe_probability=q)), **settings
        )
        plt.show()
        plt.close(fig)

    output = widgets.interactive_output(draw, controls)
    play = widgets.Play(
        value=spec.horizon, min=0, max=spec.horizon, interval=600, repeat=True
    )
    widgets.jslink((play, "value"), (controls["transition"], "value"))
    return widgets.VBox(
        [
            widgets.HTML(
                "<b>Move the sliders; the policy and trajectories stay frozen.</b>"
            ),
            widgets.HBox(list(controls.values())[:2]),
            widgets.HBox(list(controls.values())[2:4]),
            controls["direction"],
            widgets.HBox([play, controls["transition"]]),
            output,
        ]
    )


def plot_dataset(lab: Lab, *, n=16, seed=0, selection="fitted", delta=0.05):
    population, table = lab.population, lab.table
    draws = sample_draws(population, master_seed=6180, replicate=seed, size=n)
    empirical = empirical_losses(table, draws)
    fitted = int(empirical.argmin())
    worst = int(np.abs(table.population_loss - empirical).argmax())
    selected = fitted if selection == "fitted" else worst
    p = lab.grid.matrices[selected]
    _, bound = choose_epsilon(population, n, delta)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
        ax = axes[0]
        _phase_axes(ax)
        _ellipse(ax, p, label="selected P: V(x) = 1")
        counts = np.bincount(draws, minlength=len(population.weights))
        for plant in range(len(population.matrices)):
            ids = np.flatnonzero((population.plant_ids == plant) & (counts > 0))
            ax.scatter(
                *population.states[ids, 0].T,
                s=12 + 12 * counts[ids],
                alpha=0.6,
                color=COLORS[plant],
                label=f"sampled G{plant + 1} resets",
            )
        for index in draws[:8]:
            ax.plot(
                *population.states[index].T,
                color=COLORS[population.plant_ids[index]],
                alpha=0.35,
                lw=1,
            )
        ax.set_title("Only sampled trajectories enter fitting")
        ax.legend(fontsize=8, loc="upper right")
        ax = axes[1]
        axis = np.linspace(0, 1, 100)
        ax.fill_between(
            axis,
            np.maximum(0, axis - bound),
            np.minimum(1, axis + bound),
            color=BLUE,
            alpha=0.08,
            label=f"theorem band ±{bound:.3f}",
        )
        ax.plot(axis, axis, color="#939daa", lw=1)
        ax.scatter(empirical, table.population_loss, s=8, color=BLUE, alpha=0.15)
        ax.scatter(
            empirical[fitted],
            table.population_loss[fitted],
            marker="*",
            s=170,
            color=ORANGE,
            label="empirical minimizer",
            zorder=5,
        )
        ax.scatter(
            empirical[worst],
            table.population_loss[worst],
            marker="D",
            s=40,
            color=RED,
            label="largest absolute grid gap",
            zorder=4,
        )
        ax.set(
            xlabel="empirical loss on sampled trajectories",
            ylabel="exact population loss",
            title="Every dot is a candidate certificate",
            xlim=(-0.03, 1.03),
            ylim=(-0.03, 1.03),
        )
        ax.legend(fontsize=8, loc="lower right")
        ax = axes[2]
        for plant in range(len(population.matrices)):
            ids = np.flatnonzero(population.plant_ids == plant)
            angle = np.rad2deg(population.angles[ids])
            values = table.worst_residual[selected, ids]
            ax.plot(
                angle, values, color=COLORS[plant], label=f"G{plant + 1}: all resets"
            )
            used = counts[ids] > 0
            ax.scatter(angle[used], values[used], color=COLORS[plant], s=25)
        ax.axhline(0, color=RED, lw=1)
        ax.axhline(-population.spec.gamma, color=PURPLE, ls="--", lw=1)
        ax.set(
            xlabel="initial-state angle (degrees)",
            ylabel="worst residual s",
            title="Dots: observed resets; curves: full population",
        )
        ax.legend(fontsize=8)
        fig.suptitle(
            f"N = {n}, dataset seed = {seed}, selected = {selection} · "
            f"loss {empirical[selected]:.3f} sampled / "
            f"{table.population_loss[selected]:.3f} population · "
            f"true risk {table.population_risk[selected]:.1%}, "
            f"upper bound {min(1, empirical[selected] + bound):.1%}",
            fontsize=12,
        )
    return fig


def dataset_controls(spec: CertificateSpec):
    controls = {
        "n": widgets.SelectionSlider(
            options=[4, 8, 16, 32, 128, 512],
            value=16,
            description="trajectories N",
            continuous_update=False,
        ),
        "seed": widgets.IntSlider(
            value=0, min=0, max=499, description="dataset seed", continuous_update=False
        ),
        "selection": widgets.ToggleButtons(
            options=[("fitted P", "fitted"), ("largest grid gap", "worst")]
        ),
        "q": widgets.Dropdown(
            options=[
                ("stable family", 0.0),
                ("10% unstable plant", 0.1),
                ("25% unstable plant", 0.25),
            ],
            value=0.0,
        ),
    }

    def draw(q, **settings):
        fig = plot_dataset(prepare_lab(replace(spec, unsafe_probability=q)), **settings)
        plt.show()
        plt.close(fig)

    output = widgets.interactive_output(draw, controls)
    return widgets.VBox(
        [
            widgets.HBox([controls["n"], controls["seed"]]),
            widgets.HBox([controls["selection"], controls["q"]]),
            output,
        ]
    )


def _quantile_curve(ax, x, values, color, label):
    low, median, high = np.quantile(values, [0.1, 0.5, 0.9], axis=1)
    ax.fill_between(x, low, high, color=color, alpha=0.13)
    ax.plot(x, median, "o-", ms=4, color=color, label=label)


def plot_sweep(lab: Lab, result: SweepResult):
    n = result.sample_sizes
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
        ax = axes[0]
        _quantile_curve(ax, n, result.fitted_gap, BLUE, "fitted P: absolute gap")
        _quantile_curve(ax, n, result.grid_gap, PURPLE, "largest grid gap")
        ax.plot(n, result.bound, "s--", color=RED, label="theorem ΔN")
        ax.set(
            xlabel="independent training trajectories N",
            ylabel="generalization gap",
            title="Can any inspected candidate exploit the dataset?",
            xscale="log",
        )
        ax.legend(fontsize=8)
        ax = axes[1]
        _quantile_curve(ax, n, result.population_risk, BLUE, "true risk of fitted P")
        _quantile_curve(
            ax, n, result.risk_upper, RED, "training-based risk upper bound"
        )
        ax.axhline(
            lab.population.spec.unsafe_probability,
            color=ORANGE,
            ls=":",
            label="known irreducible risk floor",
        )
        ax.set(
            xlabel="independent training trajectories N",
            ylabel="trajectory violation probability",
            title="Does a useful risk certificate emerge?",
            xscale="log",
            ylim=(-0.02, 1.03),
        )
        ax.legend(fontsize=8)
        ax = axes[2]
        events = [
            result.fitted_gap > result.bound[:, None],
            result.grid_gap > result.bound[:, None],
            result.population_risk > result.risk_upper + 1e-12,
        ]
        labels = [
            "fitted gap exceeds ΔN",
            "grid gap exceeds ΔN",
            "risk exceeds upper bound",
        ]
        replicates = result.draws.shape[0]
        highest_interval = 0.0
        for offset, event, color, label in zip(
            [-0.12, 0, 0.12], events, [BLUE, PURPLE, ORANGE], labels, strict=True
        ):
            counts = event.sum(axis=1)
            rate = counts / replicates
            low, high = wilson_interval(counts, replicates)
            highest_interval = max(highest_interval, float(high.max()))
            ax.errorbar(
                n * np.exp(offset),
                rate,
                yerr=[rate - low, high - rate],
                fmt="o",
                ms=4,
                capsize=3,
                color=color,
                label=label,
            )
        ax.axhline(
            result.delta_per_size, color=RED, ls="--", label="allowed probability δ"
        )
        ax.set(
            xlabel="independent training trajectories N",
            ylabel="fraction of repeated datasets",
            title="Exceedance counts + pointwise 95% Wilson intervals",
            xscale="log",
            ylim=(
                -0.005,
                min(1.03, max(0.1, 2 * result.delta_per_size, 1.1 * highest_interval)),
            ),
        )
        ax.legend(fontsize=7, loc="upper right")
        fig.suptitle(
            f"{replicates} predeclared dataset seeds · "
            f"unstable probability {lab.population.spec.unsafe_probability:.0%} · "
            "shaded bands: 10th–90th percentiles across datasets",
            fontsize=12,
        )
    return fig


def sweep_summary(result: SweepResult) -> HTML:
    rows = "".join(
        f"<tr><td>{n:,}</td><td>{eps:.2g}</td><td>{bound:.3f}</td>"
        f"<td>{np.median(gap):.4f}</td><td>{np.max(grid_gap):.4f}</td>"
        f"<td>{int(np.sum(grid_gap > bound))}/{len(grid_gap)}</td></tr>"
        for n, eps, bound, gap, grid_gap in zip(
            result.sample_sizes,
            result.epsilon,
            result.bound,
            result.fitted_gap,
            result.grid_gap,
            strict=True,
        )
    )
    return HTML(
        "<table><thead><tr><th>N</th><th>theoretical ε</th><th>ΔN</th>"
        "<th>median fitted gap</th><th>largest observed grid gap</th>"
        "<th>grid exceedances</th></tr></thead><tbody>" + rows + "</tbody></table>"
    )


def sweep_controls(spec: CertificateSpec, root: Path, initial: SweepResult):
    repeats = widgets.Dropdown(
        options=[50, 100, 500], value=100, description="datasets"
    )
    q = widgets.Dropdown(
        options=[("stable family", 0.0), ("10% unstable", 0.1), ("25% unstable", 0.25)],
        value=0.0,
    )
    joint = widgets.Checkbox(
        value=False, description="joint guarantee over all five N values"
    )
    run = widgets.Button(description="Run sweep", button_style="primary", icon="play")
    export = widgets.Button(description="Export current run", icon="save")
    output, status = widgets.Output(), widgets.HTML()
    current = {"lab": prepare_lab(spec), "result": initial}

    def run_clicked(_):
        run.disabled = export.disabled = True
        status.value = "Computing all predeclared datasets…"
        try:
            lab = prepare_lab(replace(spec, unsafe_probability=q.value))
            result = run_sweep(
                lab.population,
                lab.grid,
                lab.table,
                replicates=repeats.value,
                joint_over_sizes=joint.value,
            )
            current.update(lab=lab, result=result)
            with output:
                output.clear_output(wait=True)
                fig = plot_sweep(lab, result)
                plt.show()
                plt.close(fig)
                display(sweep_summary(result))
            status.value = (
                f"Finished {repeats.value} datasets per N. "
                f"δ per N = {result.delta_per_size:.3f}. "
                "No seed was selected or omitted."
            )
        except Exception as error:
            status.value = f"Run failed: {error}"
            raise
        finally:
            run.disabled = export.disabled = False

    def export_clicked(_):
        lab, result = current["lab"], current["result"]
        import active_eval_gym.certificates as numerical_code

        sources = (
            Path(numerical_code.__file__),
            Path(__file__),
            root / "notebooks/certificate_generalization.ipynb",
        )
        destination = save_experiment(
            root / "artifacts/certificate_generalization",
            lab.population,
            lab.grid,
            lab.table,
            result,
            source_paths=sources,
        )
        status.value = f"Saved current computed run to {destination}"

    run.on_click(run_clicked)
    export.on_click(export_clicked)
    return widgets.VBox(
        [
            widgets.HTML(
                "<b>Settings take effect when you press Run sweep.</b> "
                "Export saves the last completed run; initially the default sweep."
            ),
            widgets.HBox([repeats, q]),
            joint,
            widgets.HBox([run, export]),
            status,
            output,
        ]
    )


def plot_bound(lab: Lab, *, n=512, epsilon=0.0002, joint=False):
    delta = 0.05 / 5 if joint else 0.05
    values = np.geomspace(1e-7, 0.05, 400)
    approximation, sampling = bound_terms(lab.population, n, values, delta)
    a, s = bound_terms(lab.population, n, epsilon, delta)
    best_eps, best_bound = choose_epsilon(lab.population, n, delta)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained")
        ax.loglog(
            values,
            approximation,
            color=ORANGE,
            label="two approximation errors: 2Lpar ε",
        )
        ax.loglog(values, sampling, color=BLUE, label="sampling + finite union bound")
        ax.loglog(
            values, approximation + sampling, color=PURPLE, lw=2.5, label="total ΔN"
        )
        ax.axvline(epsilon, color="#8896a8", ls=":")
        ax.scatter([epsilon], [a + s], color=RED, s=70, zorder=4, label="selected ε")
        ax.scatter(
            [best_eps],
            [best_bound],
            marker="*",
            color=ORANGE,
            s=160,
            zorder=5,
            label="chosen from constants before sampling",
        )
        ax.axhline(1, color=RED, lw=1, alpha=0.3)
        ax.set(
            xlabel="theoretical cover resolution ε (Frobenius distance)",
            ylabel="error allowance",
            ylim=(0.001, 10),
            title=f"N = {n}: ΔN = {float(a):.3f} + {float(s):.3f} "
            f"= {float(a + s):.3f} · δ per N = {delta:.3f}",
        )
        ax.legend(fontsize=9)
    return fig


def bound_controls(lab: Lab):
    controls = {
        "n": widgets.SelectionSlider(
            options=[32, 128, 512, 2048, 8192],
            value=512,
            description="N",
            continuous_update=False,
        ),
        "epsilon": widgets.FloatLogSlider(
            value=0.0002,
            base=10,
            min=-7,
            max=-1.4,
            step=0.1,
            description="ε",
            continuous_update=False,
            readout_format=".2g",
        ),
        "joint": widgets.Checkbox(
            value=False, description="joint over five sample sizes"
        ),
    }

    def draw(**settings):
        fig = plot_bound(lab, **settings)
        plt.show()
        plt.close(fig)

    output = widgets.interactive_output(draw, controls)
    return widgets.VBox(
        [widgets.HBox([controls["n"], controls["epsilon"]]), controls["joint"], output]
    )
