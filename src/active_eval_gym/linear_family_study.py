"""Declared comparison of update rules across plant families.

Collection writes raw trajectories and step logs; analysis reads only those files.
The design was fixed in docs/linear-family-study.md before any run.
"""

import csv
import json
import platform
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version
from math import comb
from pathlib import Path

import numpy as np

from active_eval_gym import linear_study as ls
from active_eval_gym.certificates import (
    CertificateSpec,
    candidate_grid,
    empirical_losses,
    loss_table,
    verified_margins,
)
from active_eval_gym.linear_updates import (
    ConditionGrid,
    LinearFamily,
    UpdateSpec,
    backtrack,
    capacity_step,
    circle_grid,
    cost_gradients,
    policy_actions,
    rollout,
    step_capacity,
    target_law,
    trajectory_costs,
)

FAMILY_TYPES = ("same_shape", "general", "rotated_actuator", "mirrored")
RULES = ("backtracking", "capacity", "guard_only")
TARGETS = ls.TARGETS
ARMS = (
    ("known", "fixed"),
    ("erm_first", "fixed"),
    ("erm_first", "refit"),
    ("erm_small_step", "fixed"),
    ("erm_small_step", "refit"),
    ("erm_large_step", "fixed"),
    ("erm_large_step", "refit"),
    ("model_grid", "fixed"),
    ("model_grid", "refit"),
)
METRIC_VERSION = "linear-family-coverage-v1"
REFERENCE_ID = "reference"
INITIAL_GAIN = np.eye(2)
STATE_COST, INPUT_COST = np.eye(2), 0.1 * np.eye(2)


@dataclass(frozen=True)
class FamilyStudySpec:
    preset: str = "full"
    master_seed: int = 6180
    families_per_type: int = 400
    rounds: int = 3
    directions: int = 64
    reference_directions: int = 256
    mixture: float = 0.5
    initial_coverage: float = 0.539
    relative_band_width: float = 0.1
    reference_replicates: int = 10
    fitting_size: int = 32
    step_shifts: int = 16
    tie_limit: int | None = None
    capacity_slack: float = 1e-9
    capacity_shrink: float = 0.9
    capacity_trials: int = 200
    bootstrap_resamples: int = 10_000
    meaningful_gap: float = 1.0

    @classmethod
    def from_preset(cls, preset: str) -> "FamilyStudySpec":
        if preset == "full":
            return cls()
        if preset == "smoke":
            return cls(
                preset="smoke",
                families_per_type=2,
                directions=8,
                reference_directions=16,
                reference_replicates=1,
                step_shifts=2,
                tie_limit=2,
                bootstrap_resamples=200,
            )
        raise ValueError(f"Unknown preset: {preset}")


def _triangular(rng: np.random.Generator) -> np.ndarray:
    return np.array(
        [[rng.uniform(0.5, 0.9), rng.uniform(0.3, 0.7)], [0, rng.uniform(0.4, 0.7)]]
    )


def _stable(rng: np.random.Generator) -> np.ndarray:
    g = rng.normal(size=(2, 2))
    return g * rng.uniform(0.6, 0.9) / np.abs(np.linalg.eigvals(g)).max()


def sample_plants(
    spec: FamilyStudySpec, family_type: int | str, index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return closed loops and actuators; each family has its own seed stream."""
    stream = FAMILY_TYPES.index(family_type)
    rng = np.random.default_rng(
        np.random.SeedSequence([spec.master_seed, stream, index])
    )
    actuators = np.stack((np.eye(2), np.eye(2)))
    if family_type == "same_shape":
        return np.stack((_triangular(rng), _triangular(rng))), actuators
    if family_type == "mirrored":
        first = _triangular(rng)
        noise = rng.normal(scale=0.08, size=(2, 2))
        return np.stack((first, -first + noise)), actuators
    loops = np.stack((_stable(rng), _stable(rng)))
    if family_type == "rotated_actuator":
        angle = rng.uniform(np.pi / 4, 3 * np.pi / 4)
        cos, sin = np.cos(angle), np.sin(angle)
        actuators[1] = [[cos, -sin], [sin, cos]]
    return loops, actuators


class RawLog:
    """Keep one reference-law trajectory set per distinct gain within a family."""

    def __init__(self, family: LinearFamily, reference: ConditionGrid, horizon: int):
        self.family, self.reference, self.horizon = family, reference, horizon
        self.gains: list[np.ndarray] = []
        self.states: list[np.ndarray] = []
        self.index: dict[bytes, int] = {}

    def add(self, gain: np.ndarray) -> int:
        key = gain.astype("<f8").tobytes()
        if key not in self.index:
            self.index[key] = len(self.gains)
            self.gains.append(gain.copy())
            self.states.append(
                rollout(self.family, self.reference, gain, self.horizon)[0]
            )
        return self.index[key]


@dataclass
class Context:
    family_id: str
    family: LinearFamily
    train: ConditionGrid
    reference: ConditionGrid
    update: UpdateSpec
    width: float
    spec: FamilyStudySpec
    log: RawLog


def _make_context(family_id, family, train, reference, update, width, spec) -> Context:
    log = RawLog(family, reference, update.horizon)
    log.add(INITIAL_GAIN)
    return Context(family_id, family, train, reference, update, width, spec, log)


def _take_step(ctx: Context, rule, gain, p, weights, initial_step):
    if rule == "backtracking":
        update = ctx.update
        if initial_step is not None:
            update = replace(update, initial_step=initial_step)
        return backtrack(ctx.family, ctx.train, gain, p, weights, update), None
    if rule not in RULES:
        raise ValueError(f"Unknown update rule: {rule}")
    cap = ctx.update.initial_step if rule == "capacity" else None
    result = capacity_step(
        ctx.family,
        ctx.train,
        gain,
        p,
        weights,
        ctx.update,
        cap=cap,
        slack=ctx.spec.capacity_slack,
        shrink=ctx.spec.capacity_shrink,
        max_trials=ctx.spec.capacity_trials,
    )
    return result, cap


def run_arm(ctx: Context, pick, target, rule, *, initial_step=None) -> list[dict]:
    """Run declared rounds; a failed selection or verification halts the branch."""
    gain, rows, halted = INITIAL_GAIN.copy(), [], False
    for round_index in range(ctx.spec.rounds):
        old_index = ctx.log.add(gain)
        choice = None if halted else pick(gain, round_index)
        if choice is None or choice.matrix is None:
            halted = True
            status = "halted" if choice is None else choice.status
            rows.append(
                {
                    "round": round_index,
                    "old_gain": old_index,
                    "new_gain": old_index,
                    "status": status,
                    "step": 0.0,
                }
            )
            continue
        p = choice.matrix
        states, actions = rollout(ctx.family, ctx.train, gain, ctx.update.horizon)
        law = target_law(
            trajectory_costs(ctx.family, states, actions),
            ctx.train.weights,
            ctx.update.threshold,
            target,
            width=ctx.width,
            mixture=ctx.spec.mixture,
        )
        result, cap = _take_step(ctx, rule, gain, p, law.weights, initial_step)
        halted = result.status == "verification_failed"
        norm = float(np.linalg.norm(result.direction, 2))
        loops = ctx.family.closed_loops
        rows.append(
            {
                "round": round_index,
                "old_gain": old_index,
                "new_gain": ctx.log.add(result.gain),
                "status": result.status,
                "step": result.step,
                "cap": cap,
                "capacity": step_capacity(
                    ctx.family, gain, p, result.direction, ctx.update
                ),
                "radius_limit": ctx.update.kappa / norm if norm > 0 else None,
                "trials": len(result.trials),
                "radius_rejections": sum(not t.radius_ok for t in result.trials),
                "margin_rejections": sum(not t.margin_ok for t in result.trials),
                "armijo_rejections": sum(t.armijo_ok is False for t in result.trials),
                "candidate_index": choice.candidate_index,
                "certificate": p,
                "direction": result.direction,
                "target_fallback": law.fallback,
                "band_size": int(law.selected.sum()),
                "min_margin": float(verified_margins(loops(gain), p).min()),
                "new_min_margin": float(verified_margins(loops(result.gain), p).min()),
            }
        )
        gain = result.gain
    return rows


@lru_cache(maxsize=1)
def _candidates() -> np.ndarray:
    return candidate_grid(CertificateSpec()).matrices


def _save_family(root: Path, ctx: Context, metadata: dict, records: list) -> dict:
    raw = root / "raw" / f"{ctx.family_id}.npz"
    np.savez_compressed(
        raw,
        a=ctx.family.a,
        b=ctx.family.b,
        q=ctx.family.q,
        r=ctx.family.r,
        threshold=ctx.update.threshold,
        band_width=ctx.width,
        plant_ids=ctx.reference.plant_ids,
        resets=ctx.reference.resets,
        weights=ctx.reference.weights,
        gains=np.stack(ctx.log.gains),
        states=np.stack(ctx.log.states),
    )
    runs = root / "runs" / f"{ctx.family_id}.json"
    ls._write_json(runs, {**metadata, "records": records})
    return {
        **metadata,
        "raw": str(raw.relative_to(root)),
        "raw_sha256": sha256(raw.read_bytes()).hexdigest(),
        "runs": str(runs.relative_to(root)),
        "runs_sha256": sha256(runs.read_bytes()).hexdigest(),
    }


def collect_ensemble_family(task: tuple) -> dict:
    spec, root, family_type, index = task
    loops, actuators = sample_plants(spec, family_type, index)
    family = LinearFamily(
        loops + actuators @ INITIAL_GAIN, actuators, STATE_COST, INPUT_COST
    )
    train = circle_grid(2, spec.directions)
    reference = circle_grid(2, spec.reference_directions)
    update = UpdateSpec()
    states, actions = rollout(family, reference, INITIAL_GAIN, update.horizon)
    costs = trajectory_costs(family, states, actions)
    update = replace(update, threshold=float(np.quantile(costs, spec.initial_coverage)))
    ctx = _make_context(
        f"{family_type}-{index:04d}",
        family,
        train,
        reference,
        update,
        spec.relative_band_width * update.threshold,
        spec,
    )
    matrices = _candidates()
    # The model-grid rule ignores empirical loss and the probe direction.
    choice = ls.choose_certificate(
        family,
        INITIAL_GAIN,
        matrices,
        np.zeros(len(matrices)),
        np.zeros((2, 2)),
        "model_grid",
        update,
    )
    records = []
    if choice.matrix is not None:
        for rule in RULES:
            for target in TARGETS:
                rows = run_arm(ctx, _always(choice), target, rule)
                records.append({"rule": rule, "target": target, "rounds": rows})
    status = "selected" if choice.matrix is not None else "no_shared_certificate"
    metadata = {
        "family_id": ctx.family_id,
        "family_type": family_type,
        "index": index,
        "status": status,
    }
    return _save_family(root, ctx, metadata, records)


def _always(choice):
    return lambda *_: choice


def _selector_pick(ctx: Context, study, selector, mode, replicate):
    chosen = []

    def pick(gain, round_index):
        if chosen and mode == "fixed":
            return chosen[0]
        horizon = ctx.update.horizon
        states, _ = rollout(ctx.family, ctx.train, gain, horizon)
        population = ls._population(ctx.family, ctx.train, gain, states, ctx.update)
        draws = ls.fitting_indices(
            population, study, replicate, round_index, ctx.spec.fitting_size
        )
        _, gradients = cost_gradients(ctx.family, ctx.train, gain, horizon)
        choice = ls.choose_certificate(
            ctx.family,
            gain,
            _candidates(),
            empirical_losses(loss_table(population, _candidates()), draws),
            -np.einsum("c,cij->ij", ctx.train.weights, gradients),
            selector,
            ctx.update,
        )
        chosen[:] = [choice]
        return choice

    return pick


def collect_reference(spec: FamilyStudySpec, root: Path, *, progress=print) -> dict:
    study = ls.StudySpec(
        directions=spec.directions,
        reference_directions=spec.reference_directions,
        master_seed=spec.master_seed,
    )
    family, train, reference, _, update = ls.primary_fixture(study)
    ctx = _make_context(
        REFERENCE_ID, family, train, reference, update, study.band_width, spec
    )
    records = []

    def add(part, pick, target, rule, *, initial_step=None, **labels):
        rows = run_arm(ctx, pick, target, rule, initial_step=initial_step)
        records.append(
            {
                "part": part,
                "rule": rule,
                "target": target,
                "initial_step": initial_step,
                **labels,
                "rounds": rows,
            }
        )

    progress("Reference: certificate rules under each update rule")
    for replicate in range(spec.reference_replicates):
        for selector, mode in ARMS:
            for target in TARGETS:
                for rule in RULES:
                    pick = _selector_pick(ctx, study, selector, mode, replicate)
                    add(
                        "rules",
                        pick,
                        target,
                        rule,
                        selector=selector,
                        mode=mode,
                        replicate=replicate,
                    )
    progress("Reference: shifted step grids")
    for shift in range(spec.step_shifts):
        initial_step = 2.0 ** (-shift / spec.step_shifts)
        for selector, mode in ARMS:
            for target in TARGETS:
                pick = _selector_pick(ctx, study, selector, mode, 0)
                add(
                    "shift",
                    pick,
                    target,
                    "backtracking",
                    initial_step=initial_step,
                    selector=selector,
                    mode=mode,
                    replicate=0,
                )
    progress("Reference: every verified tied certificate")
    states, _ = rollout(family, train, INITIAL_GAIN, update.horizon)
    table = loss_table(
        ls._population(family, train, INITIAL_GAIN, states, update), _candidates()
    )
    tied = np.flatnonzero(
        table.population_loss <= table.population_loss.min() + ls.TIE_TOLERANCE
    )
    loops = family.closed_loops(INITIAL_GAIN)
    verified = [
        int(i)
        for i in tied
        if verified_margins(loops, _candidates()[i]).min()
        > update.verification_tolerance
    ]
    for index in verified[: spec.tie_limit]:
        choice = ls.CertificateChoice(_candidates()[index].copy(), index, "selected")
        for target in TARGETS:
            for rule in ("backtracking", "capacity"):
                add(
                    "ties",
                    _always(choice),
                    target,
                    rule,
                    selector="tied",
                    mode="fixed",
                    candidate=index,
                )
    metadata = {
        "family_id": REFERENCE_ID,
        "family_type": REFERENCE_ID,
        "index": 0,
        "status": "selected",
        "tied_candidates": len(tied),
        "verified_tied_candidates": len(verified),
    }
    return _save_family(root, ctx, metadata, records)


def run_family_study(
    output: Path, spec: FamilyStudySpec, *, jobs: int = 1, progress=print
) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    (output / "runs").mkdir()
    (output / "spec.json").write_text(json.dumps(asdict(spec), indent=2) + "\n")
    families = [collect_reference(spec, output, progress=progress)]
    tasks = [
        (spec, output, family_type, index)
        for family_type in FAMILY_TYPES
        for index in range(spec.families_per_type)
    ]
    progress(f"Collecting {len(tasks)} families with {jobs} workers")
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as pool:
            families += list(pool.map(collect_ensemble_family, tasks, chunksize=8))
    else:
        families += [collect_ensemble_family(task) for task in tasks]
    source_root = Path(__file__).resolve().parents[2]
    paths = [
        Path(__file__),
        Path(__file__).with_name("linear_updates.py"),
        Path(__file__).with_name("linear_study.py"),
        Path(__file__).with_name("certificates.py"),
        source_root / "docs/linear-family-study.md",
        source_root / "scripts/linear_family_study.py",
    ]
    ls._write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "metric_version": METRIC_VERSION,
            "environment_id": "finite-linear-update-fixture-v1",
            "environment_package": "active-eval-gym",
            "environment_package_version": version("active-eval-gym"),
            "study_spec": asdict(spec),
            "certificate_spec": asdict(CertificateSpec()),
            "policy_id": "shared-linear-feedback-per-record",
            "policy_checkpoint": "gain arrays in raw files",
            "training_seed": spec.master_seed,
            "action_seed": None,
            "episode_seed": "SeedSequence([master_seed, type_index, family_index])",
            "perturbation_spec": "sampled plant pairs x unit-circle resets",
            "families": families,
            "source_sha256": {
                str(p.relative_to(source_root)): sha256(p.read_bytes()).hexdigest()
                for p in paths
            },
            "versions": {"numpy": np.__version__, "python": platform.python_version()},
            "completion": "complete",
        },
    )
    progress(f"Saved {len(families)} families")
    return output


def _load_family(root: Path, entry: dict) -> tuple[dict, dict]:
    for name in ("raw", "runs"):
        if (
            sha256((root / entry[name]).read_bytes()).hexdigest()
            != entry[f"{name}_sha256"]
        ):
            raise ValueError(f"Hash mismatch: {entry[name]}")
    with np.load(root / entry["raw"]) as loaded:
        raw = dict(loaded)
    return raw, json.loads((root / entry["runs"]).read_text())


def _gain_outcomes(raw: dict) -> tuple[LinearFamily, np.ndarray, np.ndarray]:
    """Costs and acceptance per gain, from saved states and the policy u = -K x."""
    family = LinearFamily(raw["a"], raw["b"], raw["q"], raw["r"])
    costs = np.stack(
        [
            trajectory_costs(family, states, policy_actions(gain, states))
            for gain, states in zip(raw["gains"], raw["states"], strict=True)
        ]
    )
    return family, costs, costs <= float(raw["threshold"])


def _cosine(u: np.ndarray, v: np.ndarray) -> float | None:
    scale = float(np.linalg.norm(u) * np.linalg.norm(v))
    return float(u @ v / scale) if scale > 0 else None


def _conflict(family, train, threshold, width, mixture) -> dict:
    costs, gradients = cost_gradients(family, train, INITIAL_GAIN, 10)
    flat = gradients.reshape(len(gradients), -1)
    base = train.weights
    law = target_law(costs, base, threshold, "boundary", width=width, mixture=mixture)
    background = -(base @ flat)
    mixed = -(law.weights @ flat)
    band = -((law.weights - (1 - mixture) * base) / mixture @ flat)
    norms = np.linalg.norm(flat, axis=1)
    active = norms > 1e-12
    cosines = flat[active] @ flat[active].T / np.outer(norms[active], norms[active])
    pairs = cosines[np.triu_indices(int(active.sum()), 1)]
    return {
        "band_size": int(law.selected.sum()),
        "band_fallback": law.fallback,
        "cos_mix": _cosine(background, mixed),
        "cos_band": _cosine(background, band),
        "conflicting_pairs": float((pairs < 0).mean()),
        "hurt_conditions": float((flat @ background > 0).mean()),
    }


def _binding(row: dict, rule: str) -> str:
    """What limited an accepted step; only the capacity rules have a sharp limit."""
    if row["status"] != "accepted":
        return row["status"]
    if rule == "backtracking":
        return "grid"
    if row["trials"] > 1:
        return "armijo"
    if row["cap"] is not None and row["step"] >= row["cap"]:
        return "s0_cap"
    if row["radius_limit"] is not None and row["capacity"] >= row["radius_limit"] * (
        1 - 1e-12
    ):
        return "radius"
    return "margin"


def _arm_outcome(rounds: list, coverage: np.ndarray, accepted: np.ndarray, weights):
    covered = [float(coverage[r["new_gain"]]) for r in rounds]
    lost = [
        float(weights @ (accepted[r["old_gain"]] & ~accepted[r["new_gain"]]))
        for r in rounds
    ]
    return covered, lost


def analyze_family(root, entry, spec) -> dict:
    raw, runs = _load_family(root, entry)
    family, _, accepted = _gain_outcomes(raw)
    weights = raw["weights"]
    coverage = accepted @ weights
    train = circle_grid(2, spec["directions"])
    row = {
        "family_id": entry["family_id"],
        "family_type": entry["family_type"],
        "index": entry["index"],
        "threshold": float(raw["threshold"]),
        "initial_coverage": float(coverage[0]),
        "shared_certificate": entry["status"] == "selected",
        **_conflict(
            family,
            train,
            float(raw["threshold"]),
            float(raw["band_width"]),
            spec["mixture"],
        ),
    }
    for record in runs["records"]:
        if entry["family_id"] == REFERENCE_ID and not (
            record["part"] == "rules"
            and record["selector"] == "model_grid"
            and record["mode"] == "fixed"
            and record["replicate"] == 0
        ):
            continue
        covered, lost = _arm_outcome(record["rounds"], coverage, accepted, weights)
        key = f"{record['rule']}_{record['target']}"
        row[f"{key}_first"], row[f"{key}_last"] = covered[0], covered[-1]
        row[f"{key}_lost"] = max(lost)
        row[f"{key}_bindings"] = "|".join(
            _binding(r, record["rule"]) for r in record["rounds"]
        )
    return row


def _sign_test(gaps: np.ndarray) -> float | None:
    positive, total = int((gaps > 0).sum()), int((gaps != 0).sum())
    if not total:
        return None
    tail = sum(comb(total, i) for i in range(min(positive, total - positive) + 1))
    return min(1.0, 2 * tail / 2**total)


def _gap_summary(gaps, resamples: int, seed: int, meaningful: float) -> dict:
    if not len(gaps):
        return {"families": 0}
    rng = np.random.default_rng(seed)
    means = gaps[rng.integers(len(gaps), size=(resamples, len(gaps)))].mean(axis=1)
    return {
        "families": len(gaps),
        "mean": float(gaps.mean()),
        "mean_ci95": np.quantile(means, (0.025, 0.975)).tolist(),
        "median": float(np.median(gaps)),
        "p10_p90": np.quantile(gaps, (0.1, 0.9)).tolist(),
        "meaningful_share": float((np.abs(gaps) >= meaningful).mean()),
        "positive_share": float((gaps > 0).mean()),
        "negative_share": float((gaps < 0).mean()),
        "zero_share": float((gaps == 0).mean()),
        "sign_test_p": _sign_test(gaps),
    }


def _ranks(values: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(_ranks(x), _ranks(y))[0, 1])


def _type_summary(rows, reference, spec) -> dict:
    summary = {
        "families": len(rows),
        "shared_certificate": sum(r["shared_certificate"] for r in rows),
    }
    for name in ("cos_mix", "cos_band", "conflicting_pairs", "hurt_conditions"):
        values = np.array([r[name] for r in rows if r[name] is not None], dtype=float)
        summary[name] = {
            "p10_p50_p90": np.quantile(values, (0.1, 0.5, 0.9)).tolist(),
            "at_or_below_reference": float((values <= reference[name]).mean()),
        }
    summary["empty_band"] = sum(r["band_fallback"] for r in rows)
    usable = [r for r in rows if r["shared_certificate"]]
    if not usable:
        return summary
    args = (spec["bootstrap_resamples"], spec["master_seed"], spec["meaningful_gap"])

    def points(key, subset):
        return np.array([100 * r[key] for r in subset], dtype=float)

    def gap(rule, target, moment, subset):
        return points(f"{rule}_{target}_{moment}", subset) - points(
            f"{rule}_background_{moment}", subset
        )

    summary["rules"] = {}
    for rule in RULES:
        saturated = [
            r
            for r in usable
            if r[f"{rule}_background_last"] == 1 and r[f"{rule}_boundary_last"] == 1
        ]
        open_rows = [r for r in usable if r not in saturated]
        result = {
            "saturated": len(saturated),
            "boundary_last": _gap_summary(gap(rule, "boundary", "last", usable), *args),
            "boundary_first": _gap_summary(
                gap(rule, "boundary", "first", usable), *args
            ),
            "high_cost_last": _gap_summary(
                gap(rule, "high_cost", "last", usable), *args
            ),
            "boundary_last_unsaturated": _gap_summary(
                gap(rule, "boundary", "last", open_rows), *args
            ),
            "background_last_mean": float(
                points(f"{rule}_background_last", usable).mean()
            ),
            "boundary_last_mean": float(points(f"{rule}_boundary_last", usable).mean()),
        }
        for target in ("background", "boundary"):
            labels = [
                label
                for r in usable
                for label in r[f"{rule}_{target}_bindings"].split("|")
            ]
            names, counts = np.unique(labels, return_counts=True)
            result[f"{target}_binding_share"] = {
                str(n): float(c / len(labels))
                for n, c in zip(names, counts, strict=True)
            }
            lost = points(f"{rule}_{target}_lost", usable)
            result[f"{target}_families_losing_mass"] = float((lost > 0).mean())
        with_band = [r for r in open_rows if r["cos_band"] is not None]
        result["spearman_conflict_gap"] = _spearman(
            np.array([1 - r["cos_band"] for r in with_band]),
            np.abs(gap(rule, "boundary", "last", with_band)),
        )
        summary["rules"][rule] = result
    return summary


def _reference_summary(root, entry) -> dict:
    raw, runs = _load_family(root, entry)
    _, _, accepted = _gain_outcomes(raw)
    coverage = accepted @ raw["weights"]
    final = [
        (record, 100 * float(coverage[record["rounds"][-1]["new_gain"]]))
        for record in runs["records"]
    ]
    limits: dict[str, int] = {}
    for record, _ in final:
        for row in record["rounds"]:
            label = f"{record['rule']}:{_binding(row, record['rule'])}"
            limits[label] = limits.get(label, 0) + 1

    def group(part):
        table: dict[tuple, list] = {}
        for record, value in final:
            if record["part"] == part:
                key = (
                    record.get("selector"),
                    record.get("mode"),
                    record["rule"],
                    record["target"],
                )
                table.setdefault(key, []).append((record, value))
        return table

    rules = {
        "/".join(key): {
            "values": [round(v, 4) for _, v in rows],
            "distinct": sorted({round(v, 4) for _, v in rows}),
            "steps": sorted(
                {tuple(r["step"] for r in rec["rounds"]) for rec, _ in rows}
            ),
        }
        for key, rows in group("rules").items()
    }
    shift = {}
    for key, rows in group("shift").items():
        values = np.array([v for _, v in rows])
        shift["/".join(key)] = {
            "values": values.tolist(),
            "min": float(values.min()),
            "median": float(np.median(values)),
            "max": float(values.max()),
        }
    ties = {}
    for key, rows in group("ties").items():
        values = np.array([v for _, v in rows])
        distinct, counts = np.unique(values.round(4), return_counts=True)
        ties["/".join(key)] = {
            "candidates": [rec["candidate"] for rec, _ in rows],
            "values": values.tolist(),
            "min": float(values.min()),
            "median": float(np.median(values)),
            "max": float(values.max()),
            "outcomes": dict(zip(distinct.tolist(), counts.tolist(), strict=True)),
        }
    return {
        "tied_candidates": entry["tied_candidates"],
        "verified_tied_candidates": entry["verified_tied_candidates"],
        "initial_coverage": 100 * float(coverage[0]),
        "update_limits": limits,
        "rules": rules,
        "shift": shift,
        "ties": ties,
    }


def analyze_family_study(run: Path, output: Path) -> Path:
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("completion") != "complete":
        raise ValueError("Analyze only a completed run.")
    output.mkdir(parents=True, exist_ok=False)
    spec = manifest["study_spec"]
    entries = manifest["families"]
    rows = [analyze_family(run, entry, spec) for entry in entries]
    reference = next(r for r in rows if r["family_id"] == REFERENCE_ID)
    ensembles = {
        name: _type_summary(
            [r for r in rows if r["family_type"] == name], reference, spec
        )
        for name in FAMILY_TYPES
    }
    entry = next(e for e in entries if e["family_id"] == REFERENCE_ID)
    summary = {
        "schema_version": 1,
        "metric_version": METRIC_VERSION,
        "run": str(run),
        "manifest_sha256": sha256((run / "manifest.json").read_bytes()).hexdigest(),
        "study_spec": spec,
        "reference_conflict": {
            k: reference[k]
            for k in (
                "band_size",
                "cos_mix",
                "cos_band",
                "conflicting_pairs",
                "hurt_conditions",
            )
        },
        "reference_outcomes": {
            k: v for k, v in reference.items() if k.endswith(("_first", "_last"))
        },
        "reference": _reference_summary(run, entry),
        "families": ensembles,
    }
    ls._write_json(output / "summary.json", summary)
    fields = sorted({key for row in rows for key in row})
    with (output / "family_metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    return output
