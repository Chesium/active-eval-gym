"""Crossed study of certificate rules, refitting, targets, and plant families.

Collection writes raw trajectories and step logs; analysis reads only those files.
The design was fixed in docs/linear-crossed-study.md before any run.
"""

import csv
import json
import platform
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np

from active_eval_gym import linear_family_study as lfs
from active_eval_gym import linear_study as ls
from active_eval_gym.certificates import (
    CertificateSpec,
    empirical_losses,
    loss_table,
    verified_margins,
)
from active_eval_gym.linear_family_study import (
    FAMILY_TYPES,
    INITIAL_GAIN,
    INPUT_COST,
    STATE_COST,
    FamilyStudySpec,
    sample_plants,
)
from active_eval_gym.linear_updates import (
    LinearFamily,
    UpdateSpec,
    circle_grid,
    cost_gradients,
    rollout,
    trajectory_costs,
)

SELECTORS = ("erm_first", "erm_small_step", "erm_large_step", "model_grid")
MODES = ("fixed", "refit")
TARGETS = ("background", "boundary")
RULE = "capacity"
METRIC_VERSION = "linear-crossed-coverage-v1"


def crossed_spec(preset: str) -> FamilyStudySpec:
    """Same seeds and settings as the family study, so its families recur here."""
    if preset == "full":
        return replace(FamilyStudySpec(), preset="crossed", families_per_type=200)
    if preset == "smoke":
        return replace(FamilyStudySpec.from_preset("smoke"), preset="crossed-smoke")
    raise ValueError(f"Unknown preset: {preset}")


def fitting_stream(family_type: str, index: int) -> int:
    return FAMILY_TYPES.index(family_type) * 10_000 + index


class Selection:
    """One family's fitting data; loss tables are cached by gain across arms."""

    def __init__(self, ctx, study, stream: int):
        self.ctx, self.study, self.stream = ctx, study, stream
        self.cache: dict[bytes, tuple] = {}

    def inputs(self, gain: np.ndarray) -> tuple:
        key = gain.astype("<f8").tobytes()
        if key not in self.cache:
            ctx = self.ctx
            states, _ = rollout(ctx.family, ctx.train, gain, ctx.update.horizon)
            population = ls._population(ctx.family, ctx.train, gain, states, ctx.update)
            _, gradients = cost_gradients(
                ctx.family, ctx.train, gain, ctx.update.horizon
            )
            self.cache[key] = (
                population,
                loss_table(population, lfs._candidates()),
                -np.einsum("c,cij->ij", ctx.train.weights, gradients),
            )
        return self.cache[key]

    def choose(self, gain, selector, round_index):
        population, table, probe = self.inputs(gain)
        draws = ls.fitting_indices(
            population,
            self.study,
            self.stream,
            round_index,
            self.ctx.spec.fitting_size,
        )
        return ls.choose_certificate(
            self.ctx.family,
            gain,
            lfs._candidates(),
            empirical_losses(table, draws),
            probe,
            selector,
            self.ctx.update,
        )

    def pick(self, selector: str, mode: str):
        kept = []

        def choose(gain, round_index):
            if kept and mode == "fixed":
                return kept[0]
            kept[:] = [self.choose(gain, selector, round_index)]
            return kept[0]

        return choose

    def ties_at_start(self) -> dict:
        """How much choice the data left at the first selection."""
        population, table, _ = self.inputs(INITIAL_GAIN)
        ctx = self.ctx
        draws = ls.fitting_indices(
            population, self.study, self.stream, 0, ctx.spec.fitting_size
        )
        empirical = empirical_losses(table, draws)
        tied = np.flatnonzero(empirical <= empirical.min() + ls.TIE_TOLERANCE)
        loops = ctx.family.closed_loops(INITIAL_GAIN)
        verified = [
            int(i)
            for i in tied
            if verified_margins(loops, lfs._candidates()[i]).min()
            > ctx.update.verification_tolerance
        ]
        chosen = {
            selector: self.choose(INITIAL_GAIN, selector, 0).candidate_index
            for selector in SELECTORS
        }
        return {
            "tied": len(tied),
            "verified_tied": len(verified),
            "minimum_empirical_loss": float(empirical.min()),
            "chosen": chosen,
        }


def collect_family(task: tuple) -> dict:
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
    ctx = lfs._make_context(
        f"{family_type}-{index:04d}",
        family,
        train,
        reference,
        update,
        spec.relative_band_width * update.threshold,
        spec,
    )
    study = ls.StudySpec(
        directions=spec.directions,
        reference_directions=spec.reference_directions,
        master_seed=spec.master_seed,
    )
    selection = Selection(ctx, study, fitting_stream(family_type, index))
    baseline = selection.choose(INITIAL_GAIN, "model_grid", 0)
    metadata = {
        "family_id": ctx.family_id,
        "family_type": family_type,
        "index": index,
        "status": "selected"
        if baseline.matrix is not None
        else "no_shared_certificate",
        "ties": selection.ties_at_start(),
    }
    records = []
    if baseline.matrix is not None:
        for selector in SELECTORS:
            for mode in MODES:
                for target in TARGETS:
                    pick = selection.pick(selector, mode)
                    records.append(
                        {
                            "selector": selector,
                            "mode": mode,
                            "target": target,
                            "rounds": lfs.run_arm(ctx, pick, target, RULE),
                        }
                    )
    return lfs._save_family(root, ctx, metadata, records)


def run_crossed_study(
    output: Path, spec: FamilyStudySpec, *, jobs: int = 1, progress=print
) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    (output / "runs").mkdir()
    tasks = [
        (spec, output, family_type, index)
        for family_type in FAMILY_TYPES
        for index in range(spec.families_per_type)
    ]
    progress(f"Collecting {len(tasks)} families with {jobs} workers")
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as pool:
            families = list(pool.map(collect_family, tasks, chunksize=4))
    else:
        families = [collect_family(task) for task in tasks]
    source_root = Path(__file__).resolve().parents[2]
    paths = [
        Path(__file__),
        Path(__file__).with_name("linear_family_study.py"),
        Path(__file__).with_name("linear_updates.py"),
        Path(__file__).with_name("linear_study.py"),
        Path(__file__).with_name("certificates.py"),
        source_root / "docs/linear-crossed-study.md",
        source_root / "scripts/linear_crossed_study.py",
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
            "episode_seed": "family stream plus 100000 * round; see fitting_stream",
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


def analyze_family(root: Path, entry: dict) -> dict:
    raw, runs = lfs._load_family(root, entry)
    _, _, accepted = lfs._gain_outcomes(raw)
    coverage = accepted @ raw["weights"]
    row = {
        "family_id": entry["family_id"],
        "family_type": entry["family_type"],
        "index": entry["index"],
        "shared_certificate": entry["status"] == "selected",
        "initial_coverage": float(coverage[0]),
        "tied": runs["ties"]["tied"],
        "verified_tied": runs["ties"]["verified_tied"],
        "capacity_rules_differ": runs["ties"]["chosen"]["erm_small_step"]
        != runs["ties"]["chosen"]["erm_large_step"],
    }
    for record in runs["records"]:
        rounds = record["rounds"]
        key = f"{record['selector']}_{record['mode']}_{record['target']}"
        row[f"{key}_first"] = float(coverage[rounds[0]["new_gain"]])
        row[f"{key}_last"] = float(coverage[rounds[-1]["new_gain"]])
        row[f"{key}_halted"] = any(
            r["status"] not in ("accepted", "zero_gradient") for r in rounds
        )
    return row


def _points(rows, key) -> np.ndarray:
    return np.array([100 * r[key] for r in rows], dtype=float)


def _effect(rows, plus: str, minus: str, args) -> dict:
    return lfs._gap_summary(_points(rows, plus) - _points(rows, minus), *args)


def _type_summary(rows: list, spec: dict) -> dict:
    usable = [r for r in rows if r["shared_certificate"]]
    args = (spec["bootstrap_resamples"], spec["master_seed"], spec["meaningful_gap"])
    summary = {
        "families": len(rows),
        "shared_certificate": len(usable),
        "median_tied": float(np.median([r["tied"] for r in usable])),
        "median_verified_tied": float(np.median([r["verified_tied"] for r in usable])),
        "no_verified_tie_share": float(
            np.mean([r["verified_tied"] == 0 for r in usable])
        ),
        "capacity_rules_differ_share": float(
            np.mean([r["capacity_rules_differ"] for r in usable])
        ),
    }
    cells, target, refit = {}, {}, {}
    for selector in SELECTORS:
        for mode in MODES:
            for name in TARGETS:
                stem = f"{selector}_{mode}_{name}"
                values = _points(usable, f"{stem}_last")
                cells[stem] = {
                    **lfs._gap_summary(values, *args),
                    "halted_share": float(
                        np.mean([r[f"{stem}_halted"] for r in usable])
                    ),
                    "full_coverage_share": float((values == 100).mean()),
                }
            for moment in ("first", "last"):
                target[f"{selector}_{mode}_{moment}"] = _effect(
                    usable,
                    f"{selector}_{mode}_boundary_{moment}",
                    f"{selector}_{mode}_background_{moment}",
                    args,
                )
        for name in TARGETS:
            refit[f"{selector}_{name}"] = _effect(
                usable,
                f"{selector}_refit_{name}_last",
                f"{selector}_fixed_{name}_last",
                args,
            )
    ties = {}
    for mode in MODES:
        for name in TARGETS:
            large = f"erm_large_step_{mode}_{name}_last"
            ties[f"max_minus_min_{mode}_{name}"] = _effect(
                usable, large, f"erm_small_step_{mode}_{name}_last", args
            )
            ties[f"max_minus_first_{mode}_{name}"] = _effect(
                usable, large, f"erm_first_{mode}_{name}_last", args
            )
    return {**summary, "cells": cells, "target": target, "refit": refit, "tie": ties}


def check_against_family_study(rows: list, family_run: Path) -> int:
    """model_grid with fixed P repeats the family study, so it must match exactly."""
    manifest = json.loads((family_run / "manifest.json").read_text())
    entries = {e["family_id"]: e for e in manifest["families"]}
    checked = 0
    for row in rows:
        if not row["shared_certificate"]:
            continue
        entry = entries[row["family_id"]]
        raw, runs = lfs._load_family(family_run, entry)
        _, _, accepted = lfs._gain_outcomes(raw)
        coverage = accepted @ raw["weights"]
        for record in runs["records"]:
            if record["rule"] != RULE or record["target"] not in TARGETS:
                continue
            key = f"model_grid_fixed_{record['target']}_last"
            expected = float(coverage[record["rounds"][-1]["new_gain"]])
            if row[key] != expected:
                raise ValueError(f"Family study mismatch: {row['family_id']}")
            checked += 1
    return checked


def analyze_crossed_study(run: Path, family_run: Path, output: Path) -> Path:
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("completion") != "complete":
        raise ValueError("Analyze only a completed run.")
    output.mkdir(parents=True, exist_ok=False)
    spec = manifest["study_spec"]
    rows = [analyze_family(run, entry) for entry in manifest["families"]]
    checked = check_against_family_study(rows, family_run)
    ls._write_json(
        output / "summary.json",
        {
            "schema_version": 1,
            "metric_version": METRIC_VERSION,
            "run": str(run),
            "manifest_sha256": sha256((run / "manifest.json").read_bytes()).hexdigest(),
            "family_study_cells_matched": checked,
            "study_spec": spec,
            "families": {
                name: _type_summary([r for r in rows if r["family_type"] == name], spec)
                for name in FAMILY_TYPES
            },
        },
    )
    fields = sorted({key for row in rows for key in row})
    with (output / "family_metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    return output
