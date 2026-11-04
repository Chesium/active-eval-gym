"""Reproducible factorial experiments on target maps and quadratic certificates.

Run collection explicitly, then analyze its immutable artifacts independently.
Nothing here changes the original certificate notebook's experiment.
"""

import csv
import json
import platform
from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np

from active_eval_gym.certificates import (
    KNOWN_CERTIFICATE,
    STABLE_PLANTS,
    CertificateSpec,
    Population,
    candidate_grid,
    empirical_losses,
    loss_table,
    sample_draws,
    verified_margins,
)
from active_eval_gym.linear_updates import (
    ConditionGrid,
    LinearFamily,
    StepResult,
    UpdateSpec,
    backtrack,
    circle_grid,
    cost_drift_bound,
    cost_gradients,
    coverage_accounting,
    curvature_bound,
    rollout,
    step_capacity,
    target_law,
    trajectory_costs,
)

TARGETS = ("background", "boundary", "high_cost")
SELECTORS = ("known", "erm_first", "erm_small_step", "erm_large_step", "model_grid")
METRIC_VERSION = "linear-update-coverage-v1"
TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class StudySpec:
    preset: str = "full"
    master_seed: int = 6180
    replicates: int = 10
    rounds: int = 20
    sample_sizes: tuple[int, ...] = (8, 32, 128)
    repeated_sample_size: int = 32
    directions: int = 64
    reference_directions: int = 256
    band_width: float = 0.35
    mixture: float = 0.5
    landscape: bool = True

    @classmethod
    def from_preset(cls, preset: str) -> "StudySpec":
        if preset == "full":
            return cls()
        if preset == "smoke":
            return cls(
                preset="smoke",
                replicates=2,
                rounds=3,
                sample_sizes=(32,),
                landscape=False,
            )
        raise ValueError(f"Unknown preset: {preset}")


@dataclass(frozen=True)
class CertificateChoice:
    matrix: np.ndarray | None
    candidate_index: int | None
    status: str
    empirical_loss: float | None = None


@dataclass(frozen=True)
class BranchOutcome:
    record_id: str
    gain: np.ndarray
    status: str


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "infinity" if value > 0 else "-infinity" if value < 0 else None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(_jsonable(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def primary_fixture(spec: StudySpec):
    family = LinearFamily(
        STABLE_PLANTS + np.eye(2),
        np.repeat(np.eye(2)[None], 2, axis=0),
        np.eye(2),
        0.1 * np.eye(2),
    )
    return (
        family,
        circle_grid(2, spec.directions),
        circle_grid(2, spec.reference_directions),
        np.eye(2),
        UpdateSpec(),
    )


def _combine(discovery: ConditionGrid, reference: ConditionGrid) -> ConditionGrid:
    return ConditionGrid(
        np.concatenate((discovery.plant_ids, reference.plant_ids)),
        np.concatenate((discovery.resets, reference.resets)),
        np.concatenate((discovery.weights / 2, reference.weights / 2)),
    )


class ArtifactStore:
    """Deduplicate raw trajectories by fixture and gain; metric files are separate."""

    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=False)
        self.root = root
        (root / "raw").mkdir()
        (root / "metrics_v1").mkdir()
        self.raw_manifest: dict[str, str] = {}
        self.records: list[dict] = []
        self.fixtures: dict[str, dict] = {}

    def register(self, name, family, discovery, reference, gain, update):
        self.fixtures[name] = {
            "family": asdict(family),
            "discovery": asdict(discovery),
            "reference": asdict(reference),
            "initial_gain": gain,
            "update": asdict(update),
        }

    def evaluate(self, fixture: str, gain: np.ndarray):
        settings = self.fixtures[fixture]
        family = LinearFamily(**settings["family"])
        discovery = ConditionGrid(**settings["discovery"])
        reference = ConditionGrid(**settings["reference"])
        update = UpdateSpec(**settings["update"])
        grid = _combine(discovery, reference)
        identity = sha256(
            fixture.encode()
            + gain.shape.__repr__().encode()
            + gain.astype("<f8").tobytes()
        ).hexdigest()
        name = f"raw/{identity}.npz"
        if name not in self.raw_manifest:
            states, actions = rollout(family, grid, gain, update.horizon)
            np.savez_compressed(
                self.root / name, gain=gain, states=states, actions=actions
            )
            self.raw_manifest[name] = sha256(
                (self.root / name).read_bytes()
            ).hexdigest()
        else:
            with np.load(self.root / name) as raw:
                states, actions = raw["states"], raw["actions"]
        costs = trajectory_costs(family, states, actions)
        return name, states, actions, costs

    def record(self, metadata: dict, arrays: dict) -> str:
        identity = f"{len(self.records):06d}"
        metadata = {
            **metadata,
            "record_id": identity,
            "arrays": f"metrics_v1/{identity}.npz",
        }
        np.savez_compressed(self.root / metadata["arrays"], **arrays)
        metadata["arrays_sha256"] = sha256(
            (self.root / metadata["arrays"]).read_bytes()
        ).hexdigest()
        self.records.append(metadata)
        return identity


def _population(family, discovery, gain, states, update):
    spec = CertificateSpec(
        horizon=update.horizon, directions=len(discovery.weights) // len(family.a)
    )
    return Population(
        spec,
        family.closed_loops(gain),
        discovery.plant_ids,
        np.arctan2(discovery.resets[:, 1], discovery.resets[:, 0]),
        states[: len(discovery.weights)],
        discovery.weights,
    )


def fitting_indices(population, spec, replicate, round_index, size):
    # The existing sampler uses SeedSequence([master_seed, replicate_stream]).
    # Prefixes pair N and branches; distinct round streams provide fresh draws.
    stream = replicate + 100_000 * round_index
    return sample_draws(
        population, master_seed=spec.master_seed, replicate=stream, size=size
    )


def choose_certificate(
    family: LinearFamily,
    gain: np.ndarray,
    matrices: np.ndarray,
    empirical: np.ndarray,
    probe: np.ndarray,
    selector: str,
    update: UpdateSpec,
) -> CertificateChoice:
    loops = family.closed_loops(gain)
    if selector == "known":
        return CertificateChoice(KNOWN_CERTIFICATE.copy(), None, "selected")
    drift = matrices[:, None] - (
        loops.transpose(0, 2, 1)[None] @ matrices[:, None] @ loops[None]
    )
    margins = np.linalg.eigvalsh(drift)[..., 0].min(axis=1)
    verified = margins > update.verification_tolerance
    tied = np.flatnonzero(empirical <= empirical.min() + TIE_TOLERANCE)
    if selector == "erm_first":
        index = int(tied[0])
    elif selector in ("erm_small_step", "erm_large_step"):
        eligible = tied[verified[tied]]
        if not len(eligible):
            return CertificateChoice(None, None, "no_verified_erm")
        if np.linalg.norm(probe) <= update.gradient_tolerance:
            index = int(eligible[0])
        else:
            capacities = np.array(
                [
                    step_capacity(family, gain, matrices[i], probe, update)
                    for i in eligible
                ]
            )
            best = (
                capacities.min() if selector == "erm_small_step" else capacities.max()
            )
            index = int(
                eligible[
                    np.flatnonzero(
                        np.isclose(capacities, best, atol=TIE_TOLERANCE, rtol=0)
                    )[0]
                ]
            )
    elif selector == "model_grid":
        eligible = np.flatnonzero(verified)
        if not len(eligible):
            return CertificateChoice(None, None, "no_verified_candidate")
        scores = margins[eligible] / np.linalg.norm(
            matrices[eligible], ord=2, axis=(1, 2)
        )
        index = int(eligible[np.flatnonzero(scores >= scores.max() - TIE_TOLERANCE)[0]])
    else:
        raise ValueError(f"Unknown selector: {selector}")
    return CertificateChoice(
        matrices[index].copy(), index, "selected", float(empirical[index])
    )


def run_step(
    store: ArtifactStore,
    fixture: str,
    gain: np.ndarray,
    choice: CertificateChoice,
    target: str,
    study: StudySpec,
    metadata: dict,
    indices: np.ndarray,
    *,
    explicit_weights=None,
    halted=False,
) -> BranchOutcome:
    settings = store.fixtures[fixture]
    family = LinearFamily(**settings["family"])
    discovery, reference = (
        ConditionGrid(**settings[k]) for k in ("discovery", "reference")
    )
    update = UpdateSpec(**settings["update"])
    count = len(discovery.weights)
    old_raw, old_states, _, all_old = store.evaluate(fixture, gain)
    old, old_ref = all_old[:count], all_old[count:]
    _, gradients = cost_gradients(family, discovery, gain, update.horizon)
    _, ref_gradients = cost_gradients(family, reference, gain, update.horizon)
    law = target_law(
        old,
        discovery.weights,
        update.threshold,
        target,
        width=study.band_width,
        mixture=study.mixture,
    )
    weights = law.weights if explicit_weights is None else explicit_weights
    direction = -np.einsum("c,cij->ij", weights, gradients)
    # Bounds must be fixed before examining any candidate controller's costs.
    curvature = curvature_bound(family, reference, gain, direction, update)
    objective_raw: dict[bytes, str] = {}

    def evaluate(candidate):
        name, _, _, costs = store.evaluate(fixture, candidate)
        objective_raw[candidate.tobytes()] = name
        return costs[:count]

    if halted or choice.matrix is None:
        result = StepResult(
            gain.copy(), direction, np.array([]), "halted" if halted else choice.status
        )
    else:
        result = backtrack(
            family, discovery, gain, choice.matrix, weights, update, evaluate=evaluate
        )
    new_raw, _, _, all_new = store.evaluate(fixture, result.gain)
    new, new_ref = all_new[:count], all_new[count:]
    alignment = -np.einsum("cij,ij->c", ref_gradients, direction)
    drift = cost_drift_bound(family, reference, gain, result.gain - gain, update)
    accounting = coverage_accounting(
        old_ref,
        new_ref,
        reference.weights,
        update.threshold,
        alignment,
        curvature,
        result.step,
        drift,
    )
    if choice.matrix is None:
        p, margins, new_margins, capacity = (
            np.empty((0, 0)),
            np.array([]),
            np.array([]),
            0,
        )
        empirical_loss = None
    else:
        p = choice.matrix
        margins = verified_margins(family.closed_loops(gain), p)
        new_margins = verified_margins(family.closed_loops(result.gain), p)
        capacity = step_capacity(family, gain, p, direction, update)
        empirical_loss = None
        if len(indices):
            pop = _population(family, discovery, gain, old_states, update)
            empirical_loss = float(loss_table(pop, p).loss[0, indices].mean())
    trial_metadata = []
    for trial in result.trials:
        row = asdict(trial)
        candidate = gain + trial.step * direction
        row["raw"] = objective_raw.get(candidate.tobytes())
        trial_metadata.append(row)
    scalar = {k: v for k, v in accounting.items() if np.isscalar(v)}
    scalar.update(
        {
            "old_reference_cost": float(reference.weights @ old_ref),
            "new_reference_cost": float(reference.weights @ new_ref),
            "old_training_cost": float(weights @ old),
            "new_training_cost": float(weights @ new),
            "step": result.step,
            "step_capacity": capacity,
            "gain_change_norm": float(np.linalg.norm(result.gain - gain, 2)),
            "min_margin": float(margins.min()) if len(margins) else None,
            "new_min_margin": float(new_margins.min()) if len(new_margins) else None,
            "empirical_certificate_loss": empirical_loss,
            "trials": len(result.trials),
            "radius_rejections": sum(not t.radius_ok for t in result.trials),
            "margin_rejections": sum(not t.margin_ok for t in result.trials),
            "armijo_rejections": sum(t.armijo_ok is False for t in result.trials),
            "conservative_margin_rejections": sum(
                t.radius_ok
                and not t.margin_ok
                and bool(np.all(t.new_margins >= update.rho * margins))
                for t in result.trials
            ),
        }
    )
    arrays = {
        "gain": gain,
        "new_gain": result.gain,
        "p": p,
        "fitting_indices": indices,
        "q": weights,
        "target_selected": law.selected,
        "gradients": gradients,
        "reference_gradients": ref_gradients,
        "direction": direction,
        "curvature": curvature,
        "alignment": alignment,
        "drift": drift,
        "old_costs": old,
        "new_costs": new,
        "old_reference_costs": old_ref,
        "new_reference_costs": new_ref,
        "margins": margins,
        "new_margins": new_margins,
        "gradient_gram": gradients.reshape(count, -1) @ gradients.reshape(count, -1).T,
        **{k: v for k, v in accounting.items() if isinstance(v, np.ndarray)},
    }
    record_id = store.record(
        {
            **metadata,
            "fixture": fixture,
            "target": target,
            "status": result.status,
            "selection_status": choice.status,
            "candidate_index": choice.candidate_index,
            "target_fallback": law.fallback,
            "old_raw": old_raw,
            "new_raw": new_raw,
            "trial_history": trial_metadata,
            "summary": scalar,
            "fitting_role": "fit"
            if metadata.get("mode") != "fixed" or metadata.get("round", 0) == 0
            else "diagnostic_only",
        },
        arrays,
    )
    return BranchOutcome(record_id, result.gain.copy(), result.status)


def run_study(output: Path, spec: StudySpec, *, progress=print) -> Path:
    store = ArtifactStore(output)
    family, discovery, reference, initial, update = primary_fixture(spec)
    store.register("primary", family, discovery, reference, initial, update)
    cert_spec = CertificateSpec()
    grid = candidate_grid(cert_spec)
    np.savez_compressed(
        output / "candidate_grid.npz",
        matrices=grid.matrices,
        covering_radius=grid.covering_radius,
    )
    old_raw, states, _, _ = store.evaluate("primary", initial)
    population = _population(family, discovery, initial, states, update)
    table = loss_table(population, grid.matrices)
    _, grads = cost_gradients(family, discovery, initial, update.horizon)
    probe = -np.einsum("c,cij->ij", discovery.weights, grads)
    initial_choices = {}
    progress("One-step target × certificate comparisons")
    for replicate in range(spec.replicates):
        draws = fitting_indices(
            population,
            spec,
            replicate,
            0,
            max(*spec.sample_sizes, spec.repeated_sample_size),
        )
        for size in sorted(set((*spec.sample_sizes, spec.repeated_sample_size))):
            empirical = empirical_losses(table, draws[:size])
            for selector in SELECTORS:
                choice = choose_certificate(
                    family, initial, grid.matrices, empirical, probe, selector, update
                )
                if size == spec.repeated_sample_size:
                    initial_choices[replicate, selector] = choice
                if size not in spec.sample_sizes:
                    continue
                for target in TARGETS:
                    run_step(
                        store,
                        "primary",
                        initial,
                        choice,
                        target,
                        spec,
                        {
                            "study": "one_step",
                            "mode": "initial",
                            "selector": selector,
                            "replicate": replicate,
                            "sample_size": size,
                            "round": 0,
                        },
                        draws[:size],
                    )

    progress("Repeated rounds: fixed initial P versus refitting P")
    for replicate in range(spec.replicates):
        progress(f"  paired replicate {replicate + 1}/{spec.replicates}")
        for selector in SELECTORS:
            modes = ("fixed",) if selector == "known" else ("fixed", "refit")
            for target in TARGETS:
                for mode in modes:
                    gain, choice = initial.copy(), initial_choices[replicate, selector]
                    halted = False
                    for round_index in range(spec.rounds):
                        _, states, _, _ = store.evaluate("primary", gain)
                        pop = _population(family, discovery, gain, states, update)
                        indices = fitting_indices(
                            pop, spec, replicate, round_index, spec.repeated_sample_size
                        )
                        if mode == "refit" and round_index > 0 and not halted:
                            current_table = loss_table(pop, grid.matrices)
                            empirical = empirical_losses(current_table, indices)
                            _, gradients = cost_gradients(
                                family, discovery, gain, update.horizon
                            )
                            current_probe = -np.einsum(
                                "c,cij->ij", discovery.weights, gradients
                            )
                            choice = choose_certificate(
                                family,
                                gain,
                                grid.matrices,
                                empirical,
                                current_probe,
                                selector,
                                update,
                            )
                        outcome = run_step(
                            store,
                            "primary",
                            gain,
                            choice,
                            target,
                            spec,
                            {
                                "study": "repeated",
                                "mode": mode,
                                "selector": selector,
                                "replicate": replicate,
                                "sample_size": spec.repeated_sample_size,
                                "round": round_index,
                            },
                            indices,
                            halted=halted,
                        )
                        gain = outcome.gain
                        halted |= outcome.status in (
                            "verification_failed",
                            "no_verified_erm",
                            "no_verified_candidate",
                            "halted",
                        )

    if spec.landscape:
        progress("Exhaustive certificate landscape")
        for index, p in enumerate(grid.matrices):
            choice = CertificateChoice(p, index, "selected")
            for target in TARGETS:
                run_step(
                    store,
                    "primary",
                    initial,
                    choice,
                    target,
                    spec,
                    {
                        "study": "landscape",
                        "mode": "grid",
                        "selector": "grid",
                        "replicate": 0,
                        "sample_size": 0,
                        "round": 0,
                    },
                    np.array([], dtype=int),
                )
    progress("Scale invariance and scalar fixtures")
    for scale in (0.9, 1.0, 1.1):
        for target in TARGETS:
            run_step(
                store,
                "primary",
                initial,
                CertificateChoice(scale * KNOWN_CERTIFICATE, None, "selected"),
                target,
                spec,
                {
                    "study": "scale",
                    "mode": "fixed",
                    "selector": str(scale),
                    "replicate": 0,
                    "sample_size": 0,
                    "round": 0,
                },
                np.array([], dtype=int),
            )
    _scalar_studies(store, spec)

    source_root = Path(__file__).resolve().parents[2]
    paths = [
        Path(__file__),
        Path(__file__).with_name("linear_updates.py"),
        Path(__file__).with_name("certificates.py"),
        source_root / "scripts/linear_update_study.py",
    ]
    _write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "metric_version": METRIC_VERSION,
            "environment_id": "finite-linear-update-fixture-v1",
            "environment_package": "active-eval-gym",
            "environment_package_version": version("active-eval-gym"),
            "study_spec": asdict(spec),
            "certificate_spec": asdict(cert_spec),
            "tie_tolerance": TIE_TOLERANCE,
            "candidate_grid_covering_radius": grid.covering_radius,
            "policy_id": "shared-linear-feedback-per-record",
            "policy_checkpoint": "gain arrays in raw trajectories and metric records",
            "training_seed": spec.master_seed,
            "action_seed": None,
            "episode_seed": (
                "SeedSequence([master_seed, replicate + 100000 * round]); indices saved"
            ),
            "perturbation_spec": (
                "finite plant IDs × declared resets; fixed within episode"
            ),
            "fixtures": store.fixtures,
            "initial_raw": old_raw,
            "raw_sha256": store.raw_manifest,
            "records": store.records,
            "source_sha256": {
                str(p.relative_to(source_root)): sha256(p.read_bytes()).hexdigest()
                for p in paths
            },
            "versions": {"numpy": np.__version__, "python": platform.python_version()},
            "bound_arithmetic": (
                "float64 evaluations, not interval-arithmetic certificates"
            ),
            "completion": "complete",
        },
    )
    progress(
        f"Saved {len(store.records)} records and {len(store.raw_manifest)} raw gains"
    )
    return output


def _scalar_studies(store, spec):
    counter = LinearFamily(
        np.array([[[0.0]], [[1.0]]]),
        np.ones((2, 1, 1)),
        np.ones((1, 1)),
        np.zeros((1, 1)),
    )
    conditions = ConditionGrid(np.array([0, 1]), np.ones((2, 1)), np.full(2, 0.5))
    update = UpdateSpec(horizon=2, threshold=1.25, initial_step=0.4, armijo=0.5)
    gain = np.array([[0.5]])
    store.register("counterexample", counter, conditions, conditions, gain, update)
    run_step(
        store,
        "counterexample",
        gain,
        CertificateChoice(np.ones((1, 1)), None, "selected"),
        "boundary",
        spec,
        {
            "study": "scalar",
            "mode": "fixed",
            "selector": "counterexample",
            "replicate": 0,
            "sample_size": 0,
            "round": 0,
        },
        np.array([], dtype=int),
        explicit_weights=np.array([0.25, 0.75]),
    )
    aligned = LinearFamily(
        np.array([[[0.8]]]), np.ones((1, 1, 1)), np.ones((1, 1)), np.zeros((1, 1))
    )
    resets = np.array([[0.25], [0.5], [0.75], [1.0]])
    conditions = ConditionGrid(np.zeros(4, dtype=int), resets, np.full(4, 0.25))
    gain, update = np.array([[0.2]]), UpdateSpec(horizon=2, threshold=1.2)
    store.register("aligned", aligned, conditions, conditions, gain, update)
    for target in TARGETS:
        run_step(
            store,
            "aligned",
            gain,
            CertificateChoice(np.ones((1, 1)), None, "selected"),
            target,
            spec,
            {
                "study": "scalar",
                "mode": "fixed",
                "selector": "aligned",
                "replicate": 0,
                "sample_size": 0,
                "round": 0,
            },
            np.array([], dtype=int),
        )


def _load_fixture(manifest, name):
    settings = manifest["fixtures"][name]
    family = LinearFamily(**settings["family"])
    discovery = ConditionGrid(**settings["discovery"])
    reference = ConditionGrid(**settings["reference"])
    update = UpdateSpec(**settings["update"])
    return family, discovery, reference, update


def analyze_study(run: Path, output: Path) -> Path:
    """Rebuild metrics from retained trajectories and frozen update decisions."""
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["completion"] != "complete" or manifest["schema_version"] != 1:
        raise ValueError("Require a complete supported study.")
    output.mkdir(parents=True, exist_ok=False)
    for name, digest in manifest["raw_sha256"].items():
        if sha256((run / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Raw trajectory hash mismatch: {name}")
    rows = []
    for record in manifest["records"]:
        family, discovery, reference, update = _load_fixture(
            manifest, record["fixture"]
        )
        if (
            sha256((run / record["arrays"]).read_bytes()).hexdigest()
            != record["arrays_sha256"]
        ):
            raise ValueError(f"Metric input hash mismatch: {record['record_id']}")
        with np.load(run / record["arrays"]) as saved:
            arrays = dict(saved)
        with np.load(run / record["old_raw"]) as raw:
            old_states, old_actions = raw["states"], raw["actions"]
        with np.load(run / record["new_raw"]) as raw:
            new_states, new_actions = raw["states"], raw["actions"]
        n = len(discovery.weights)
        old = trajectory_costs(family, old_states, old_actions)
        new = trajectory_costs(family, new_states, new_actions)
        # Differentiate stored states, without collecting new rollouts.
        _, gradients = cost_gradients(
            family, reference, arrays["gain"], update.horizon, states=old_states[n:]
        )
        alignment = -np.einsum("cij,ij->c", gradients, arrays["direction"])
        curvature = curvature_bound(
            family, reference, arrays["gain"], arrays["direction"], update
        )
        drift = cost_drift_bound(
            family,
            reference,
            arrays["gain"],
            arrays["new_gain"] - arrays["gain"],
            update,
        )
        accounting = coverage_accounting(
            old[n:],
            new[n:],
            reference.weights,
            update.threshold,
            alignment,
            curvature,
            record["summary"]["step"],
            drift,
        )
        actual = accounting["coverage_change"]
        if actual + 1e-10 < max(
            accounting["directional_lower"], accounting["band_lower"]
        ):
            raise AssertionError(
                f"Coverage bound failed in record {record['record_id']}"
            )
        if np.any(new[n:] > accounting["upper_cost"] + 1e-8):
            raise AssertionError(
                f"Directional cost envelope failed: {record['record_id']}"
            )
        row = {
            k: record.get(k)
            for k in (
                "record_id",
                "study",
                "fixture",
                "mode",
                "selector",
                "target",
                "replicate",
                "sample_size",
                "round",
                "status",
            )
        }
        row.update(record["summary"])
        row.update({k: v for k, v in accounting.items() if np.isscalar(v)})
        row["old_training_cost"] = float(arrays["q"] @ old[:n])
        row["new_training_cost"] = float(arrays["q"] @ new[:n])
        row["old_reference_cost"] = float(reference.weights @ old[n:])
        row["new_reference_cost"] = float(reference.weights @ new[n:])
        if arrays["p"].size:
            margins = verified_margins(family.closed_loops(arrays["gain"]), arrays["p"])
            new_margins = verified_margins(
                family.closed_loops(arrays["new_gain"]), arrays["p"]
            )
            row["min_margin"] = float(margins.min())
            row["new_min_margin"] = float(new_margins.min())
            row["step_capacity"] = step_capacity(
                family, arrays["gain"], arrays["p"], arrays["direction"], update
            )
            if len(arrays["fitting_indices"]):
                pop = _population(family, discovery, arrays["gain"], old_states, update)
                row["empirical_certificate_loss"] = float(
                    loss_table(pop, arrays["p"])
                    .loss[0, arrays["fitting_indices"]]
                    .mean()
                )
        rows.append(row)
    _write_csv(output / "metrics_v1.csv", rows)
    paired = _paired_rows(rows)
    _write_csv(output / "paired_differences_v1.csv", paired)
    _plots(run, output, manifest, rows)
    _report(output, manifest, rows, paired)
    _write_json(
        output / "analysis.json",
        {
            "metric_version": METRIC_VERSION,
            "source_manifest_sha256": sha256(
                (run / "manifest.json").read_bytes()
            ).hexdigest(),
            "records": len(rows),
            "raw_rollouts_collected": 0,
        },
    )
    return output


def _write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _paired_rows(rows):
    lookup = {
        (
            r["study"],
            r["mode"],
            r["selector"],
            r["replicate"],
            r["sample_size"],
            r["round"],
            r["target"],
        ): r
        for r in rows
        if r["study"] in ("one_step", "repeated")
    }
    differences = []
    for key, row in lookup.items():
        if row["target"] != "background":
            base = lookup[(*key[:-1], "background")]
            comparison = "target_minus_background"
        elif row["mode"] == "refit":
            base = lookup[(key[0], "fixed", *key[2:])]
            comparison = "refit_minus_fixed"
        else:
            continue
        differences.append(
            {
                **{
                    k: row[k]
                    for k in (
                        "study",
                        "mode",
                        "selector",
                        "target",
                        "replicate",
                        "sample_size",
                        "round",
                    )
                },
                "comparison": comparison,
                "coverage_difference": row["new_coverage"] - base["new_coverage"],
                "reference_cost_difference": row["new_reference_cost"]
                - base["new_reference_cost"],
                "step_difference": row["step"] - base["step"],
            }
        )
    # Also compare refitting with fixed P within each targeted curriculum.
    for key, row in lookup.items():
        if row["mode"] != "refit" or row["target"] == "background":
            continue
        base = lookup[(key[0], "fixed", *key[2:])]
        differences.append(
            {
                **{
                    k: row[k]
                    for k in (
                        "study",
                        "mode",
                        "selector",
                        "target",
                        "replicate",
                        "sample_size",
                        "round",
                    )
                },
                "comparison": "refit_minus_fixed",
                "coverage_difference": row["new_coverage"] - base["new_coverage"],
                "reference_cost_difference": row["new_reference_cost"]
                - base["new_reference_cost"],
                "step_difference": row["step"] - base["step"],
            }
        )
    return differences


def _plots(run, output, manifest, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(zip(TARGETS, ("#2563a6", "#d97706", "#7c3aed"), strict=True))

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / name, dpi=140)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for target in TARGETS:
        selected = [
            r for r in rows if r["study"] == "one_step" and r["target"] == target
        ]
        axes[0].scatter(
            [r["step"] for r in selected],
            [r["coverage_change"] for r in selected],
            s=14,
            alpha=0.5,
            label=target,
            color=colors[target],
        )
        axes[1].scatter(
            [r["min_margin"] for r in selected],
            [r["step"] for r in selected],
            s=14,
            alpha=0.5,
            color=colors[target],
        )
        axes[2].scatter(
            [r["new_training_cost"] - r["old_training_cost"] for r in selected],
            [r["coverage_change"] for r in selected],
            s=14,
            alpha=0.5,
            color=colors[target],
        )
    for ax, x, y in zip(
        axes,
        ("Accepted step", "Old verified margin", "Training cost change"),
        ("Coverage change", "Accepted step", "Coverage change"),
        strict=True,
    ):
        ax.set(xlabel=x, ylabel=y)
    axes[0].legend()
    save(fig, "one_step.png")

    fields = (
        "new_coverage",
        "new_reference_cost",
        "new_min_margin",
        "gained_mass",
        "lost_mass",
        "step",
    )
    repeated = [r for r in rows if r["study"] == "repeated"]
    for selector in SELECTORS:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for target in TARGETS:
            for mode in ("fixed", "refit"):
                subset = [
                    r
                    for r in repeated
                    if (r["selector"], r["target"], r["mode"])
                    == (selector, target, mode)
                ]
                if not subset:
                    continue
                for ax, field in zip(axes.flat, fields, strict=True):
                    values = []
                    for replicate in range(manifest["study_spec"]["replicates"]):
                        branch = sorted(
                            (r for r in subset if r["replicate"] == replicate),
                            key=lambda r: r["round"],
                        )
                        ys = [np.nan if r[field] is None else r[field] for r in branch]
                        values.append(ys)
                        ax.plot(range(len(ys)), ys, color=colors[target], alpha=0.08)
                    mean = np.mean(np.asarray(values), axis=0)
                    ax.plot(
                        range(len(mean)),
                        mean,
                        color=colors[target],
                        linestyle="-" if mode == "fixed" else "--",
                        alpha=0.6,
                        label=f"{target}, {mode}",
                    )
                    ax.set(xlabel="Round", ylabel=field.replace("_", " "))
                    ax.set_xticks(range(manifest["study_spec"]["rounds"]))
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"{selector}: all replicate traces and means; dashed = refit")
        save(fig, f"repeated_{selector}.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, mode in zip(axes, ("fixed", "refit"), strict=True):
        for status in (
            "accepted",
            "zero_gradient",
            "search_exhausted",
            "verification_failed",
            "no_verified_erm",
            "no_verified_candidate",
            "halted",
        ):
            counts = [
                sum(
                    r["status"] == status
                    for r in repeated
                    if r["mode"] == mode and r["round"] == t
                )
                for t in range(manifest["study_spec"]["rounds"])
            ]
            ax.plot(range(len(counts)), counts, label=status)
        ax.set(title=mode, xlabel="Round", ylabel="Number of branch records")
        ax.set_xticks(range(manifest["study_spec"]["rounds"]))
    axes[1].legend(fontsize=7)
    save(fig, "repeated_statuses.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    subset = [r for r in rows if r["study"] in ("one_step", "repeated", "scalar")]
    for ax, field in zip(axes, ("directional_lower", "band_lower"), strict=True):
        ax.scatter(
            [r[field] for r in subset],
            [r["coverage_change"] for r in subset],
            s=8,
            alpha=0.3,
        )
        ax.plot([-1, 1], [-1, 1], color="gray", linestyle="--")
        ax.set(xlabel=field.replace("_", " "), ylabel="Actual coverage change")
    save(fig, "bounds.png")

    # Display replicate 0 by predeclared convention, never by performance.
    examples = [
        r
        for r in manifest["records"]
        if r["study"] == "one_step"
        and r["selector"] == "known"
        and r["replicate"] == 0
        and r["sample_size"] == 32
    ]
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for column, record in enumerate(examples):
        with np.load(run / record["arrays"]) as data:
            a = dict(data)
        axes[0, column].plot(a["q"], label="training weight")
        axes[0, column].set(title=record["target"], xlabel="Discovery condition")
        axes[1, column].plot(a["old_reference_costs"], label="old")
        axes[1, column].plot(a["new_reference_costs"], label="new")
        for mask, color, label in (
            (a["gained"], "green", "gained"),
            (a["lost"], "red", "lost"),
        ):
            if np.any(mask):
                axes[1, column].scatter(
                    np.flatnonzero(mask),
                    a["new_reference_costs"][mask],
                    s=12,
                    color=color,
                    label=label,
                )
        axes[1, column].axhline(3.5, color="gray", linestyle="--")
        axes[1, column].legend()
        actual = a["new_reference_costs"] - a["old_reference_costs"]
        predicted = -record["summary"]["step"] * a["alignment"]
        axes[2, column].scatter(predicted, actual, c=a["alignment"], s=10)
        axes[2, column].set(
            xlabel="First-order cost change", ylabel="Actual cost change"
        )
    save(fig, "target_mechanism.png")
    if examples:
        with np.load(run / examples[0]["arrays"]) as a:
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.colorbar(ax.imshow(a["gradient_gram"], cmap="coolwarm"), ax=ax)
            ax.set(
                title="Condition gradient inner products",
                xlabel="Condition",
                ylabel="Condition",
            )
            save(fig, "gradient_conflict.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    theta = np.linspace(0, 2 * np.pi, 256)
    unit = np.array([np.cos(theta), np.sin(theta)])
    certificates = [
        r
        for r in manifest["records"]
        if r["study"] == "one_step"
        and r["target"] == "background"
        and r["replicate"] == 0
        and r["sample_size"] == 32
    ]
    for record in certificates:
        with np.load(run / record["arrays"]) as a:
            p = a["p"]
            if p.size:
                eigenvalues, vectors = np.linalg.eigh(p)
                ellipse = vectors @ (unit / np.sqrt(eigenvalues[:, None]))
                axes[0].plot(*ellipse, label=record["selector"])
                axes[1].scatter(
                    record["summary"]["min_margin"],
                    record["summary"]["step_capacity"],
                    label=record["selector"],
                )
    axes[0].set_aspect("equal")
    axes[0].set_title("Selected V_P = 1 ellipses, replicate 0")
    axes[0].legend(fontsize=7)
    axes[1].set(xlabel="Verified margin", ylabel="Certified step capacity")
    save(fig, "certificates.png")
    landscape = [r for r in rows if r["study"] == "landscape"]
    if landscape:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for target in TARGETS:
            subset = [r for r in landscape if r["target"] == target]
            axes[0].scatter(
                [r["min_margin"] for r in subset],
                [r["step"] for r in subset],
                s=6,
                alpha=0.4,
                label=target,
            )
            axes[1].scatter(
                [r["step_capacity"] for r in subset],
                [r["coverage_change"] for r in subset],
                s=6,
                alpha=0.4,
            )
        axes[0].set(xlabel="Old verified margin", ylabel="Accepted step")
        axes[1].set(xlabel="Certified step capacity", ylabel="Coverage change")
        axes[0].legend()
        save(fig, "landscape.png")


def _report(output, manifest, rows, paired):
    statuses = defaultdict(int)
    for row in rows:
        statuses[row["status"]] += 1
    lines = [
        "# Linear target-map and certificate study",
        "",
        f"Preset: `{manifest['study_spec']['preset']}`. Records: {len(rows)}.",
        "",
        "Coverage is exact for the declared finite reference law, up to float64 error. "
        "It is not continuous-domain volume. All replicate IDs are retained.",
        "",
        "Analytical bounds are evaluated in float64, without interval arithmetic. "
        "First-order predictions are diagnostics; the curvature envelopes and "
        "Lemma 15 bands provide the reported analytical bounds.",
        "",
        "## Outcomes",
        "",
        "| Status | Records |",
        "|---|---:|",
        *[f"| {name} | {count} |" for name, count in sorted(statuses.items())],
        "",
        "## Paired comparisons",
        "",
        "Means below summarize paired differences; they are not confidence intervals. "
        "The CSV retains each replicate, sample size, and round.",
        "",
        "| Comparison | Selector | Target | Mode | N | Mean coverage difference |",
        "|---|---|---|---|---:|---:|",
    ]
    groups = defaultdict(list)
    for row in paired:
        if (
            row["study"] == "one_step"
            or row["round"] == manifest["study_spec"]["rounds"] - 1
        ):
            key = (
                row["study"] + ": " + row["comparison"],
                row["selector"],
                row["target"],
                row["mode"],
                row["sample_size"],
            )
            groups[key].append(row["coverage_difference"])
    for key, values in sorted(groups.items()):
        lines.append("| " + " | ".join(map(str, key)) + f" | {np.mean(values):+.6f} |")
    lines += [
        "",
        "## Figures",
        "",
        *[f"![{p.stem}]({p.name})" for p in sorted(output.glob("*.png"))],
        "",
        "## Interpretation and limits",
        "",
        "- Target distributions are refreshed between rounds and fixed within "
        "line search. Training objectives differ between rounds and do not define "
        "one common descending objective.",
        "- Certificate fitting is exact over the declared grid, with the specified "
        "tie tolerance. Continuous-class optimality is not generally established.",
        "- Small empirical loss does not replace all-state matrix verification. "
        "Known and model-grid certificates are model-based baselines.",
        "- A failed sufficient margin guard does not establish instability. "
        "Direct trial margins are diagnostics, not alternative acceptance tests.",
        "- Fixed certificates and certificates refitted between rounds are "
        "separate arms. "
        "Every certificate is frozen during its corresponding controller update.",
        "- No expansion, convergence, target superiority, or generalization across "
        "unmodeled plants is assumed. Large curvature bounds can be uninformative.",
        "- Scalar examples explain mechanisms; they are not universal "
        "curriculum evidence.",
        "- Raw data hashes are checked during analysis. Analysis collects no rollouts, "
        "fits no certificates, and performs no controller updates.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
