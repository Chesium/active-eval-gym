"""Recompute the update diagnostics quoted in the midterm report.

These are deterministic linear rollouts: no training, no new sampling, and no
writes to the retained study. Every recorded arm is reproduced first, and the
script stops if one differs. Only then does it vary the backtracking factor.
"""

import argparse
import json
from dataclasses import fields, replace
from hashlib import sha256
from pathlib import Path

import numpy as np

from active_eval_gym import linear_study as ls
from active_eval_gym.certificates import (
    KNOWN_CERTIFICATE,
    CertificateSpec,
    candidate_grid,
    empirical_losses,
    loss_table,
    verified_margins,
)
from active_eval_gym.linear_updates import (
    backtrack,
    cost_drift_bound,
    cost_gradients,
    rollout,
    target_law,
    trajectory_costs,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/linear-study-smoke-v2"
DEFAULT_OUTPUT = ROOT / "ese6180/Evaluation_Guided_Domain_Randomization/midterm_assets"
ARMS = (
    ("known", "fixed"),
    ("erm_first", "fixed"),
    ("erm_first", "refit"),
    ("erm_small_step", "fixed"),
    ("erm_small_step", "refit"),
    ("erm_large_step", "fixed"),
    ("erm_large_step", "refit"),
)
# Search caps keep the smallest trial step near 1e-9 at every factor.
SEARCH_CAPS = {0.5: 30, 0.7: 60, 0.8: 90, 0.9: 180, 0.95: 360}


class Study:
    def __init__(self, manifest: dict):
        names = {field.name for field in fields(ls.StudySpec)}
        self.spec = ls.StudySpec(
            **{
                key: tuple(value) if isinstance(value, list) else value
                for key, value in manifest["study_spec"].items()
                if key in names
            }
        )
        fixture = ls.primary_fixture(self.spec)
        self.family, self.discovery, self.reference, self.initial, self.update = fixture
        self.grid = candidate_grid(CertificateSpec())

    def costs(self, gain, conditions):
        states, actions = rollout(self.family, conditions, gain, self.update.horizon)
        return trajectory_costs(self.family, states, actions)

    def coverage(self, gain) -> float:
        accepted = self.costs(gain, self.reference) <= self.update.threshold
        return float(self.reference.weights @ accepted)

    def empirical(self, gain, replicate, round_index):
        states, _ = rollout(self.family, self.discovery, gain, self.update.horizon)
        population = ls._population(
            self.family, self.discovery, gain, states, self.update
        )
        draws = ls.fitting_indices(population, self.spec, replicate, round_index, 32)
        return loss_table(population, self.grid.matrices), draws

    def direction(self, gain, weights):
        _, gradients = cost_gradients(
            self.family, self.discovery, gain, self.update.horizon
        )
        return -np.einsum("c,cij->ij", weights, gradients)

    def choose(self, gain, selector, replicate, round_index, update):
        if selector == "known":
            return KNOWN_CERTIFICATE.copy()
        table, draws = self.empirical(gain, replicate, round_index)
        probe = self.direction(gain, self.discovery.weights)
        return ls.choose_certificate(
            self.family,
            gain,
            self.grid.matrices,
            empirical_losses(table, draws),
            probe,
            selector,
            update,
        ).matrix

    def run_arm(self, selector, mode, target, replicate, update=None):
        update = update or self.update
        gain, certificate, history = self.initial.copy(), None, []
        for round_index in range(self.spec.rounds):
            if certificate is None or mode == "refit":
                certificate = self.choose(
                    gain, selector, replicate, round_index, update
                )
            law = target_law(
                self.costs(gain, self.discovery),
                self.discovery.weights,
                update.threshold,
                target,
                width=self.spec.band_width,
                mixture=self.spec.mixture,
            )
            result = backtrack(
                self.family, self.discovery, gain, certificate, law.weights, update
            )
            gain = result.gain
            history.append({"step": result.step, "coverage": self.coverage(gain)})
        return history


def check_reproduction(study: Study, records: list) -> int:
    checked = 0
    for selector, mode in ARMS:
        for target in ls.TARGETS:
            for replicate in range(study.spec.replicates):
                recorded = sorted(
                    (
                        r
                        for r in records
                        if r["study"] == "repeated"
                        and r["selector"] == selector
                        and r["mode"] == mode
                        and r["target"] == target
                        and r["replicate"] == replicate
                    ),
                    key=lambda r: r["round"],
                )
                history = study.run_arm(selector, mode, target, replicate)
                for saved, recomputed in zip(recorded, history, strict=True):
                    same = (
                        saved["summary"]["step"] == recomputed["step"]
                        and abs(
                            saved["summary"]["new_coverage"] - recomputed["coverage"]
                        )
                        < 1e-12
                    )
                    if not same:
                        raise ValueError(f"Not reproduced: {selector}/{mode}/{target}")
                    checked += 1
    return checked


def recorded_summary(records: list) -> dict:
    primary = [r for r in records if r["fixture"] == "primary"]
    pairs: dict = {}
    for r in primary:
        key = (
            r["study"],
            r["selector"],
            r["mode"],
            r["target"],
            r["sample_size"],
            r["round"],
        )
        pairs.setdefault(key, {})[r["replicate"]] = r["summary"]["new_coverage"]
    paired = [v for v in pairs.values() if len(v) == 2]
    diagnostics = [r for r in primary if r["fitting_role"] == "diagnostic_only"]
    summaries = [r["summary"] for r in primary]
    return {
        "primary_updates": len(primary),
        "updates_with_lost_mass": sum(s["lost_mass"] > 0 for s in summaries),
        "armijo_rejections": sum(s["armijo_rejections"] for s in summaries),
        "trials": sum(s["trials"] for s in summaries),
        "loss_band_covers_all_acceptable": sum(
            abs(s["band_lower"] + s["old_coverage"]) < 1e-12 for s in summaries
        ),
        "smallest_new_margin": min(s["new_min_margin"] for s in summaries),
        "paired_comparisons": len(paired),
        "paired_comparisons_differing": sum(abs(v[0] - v[1]) > 1e-12 for v in paired),
        "fixed_certificate_checks_on_later_controllers": len(diagnostics),
        "later_checks_with_positive_empirical_loss": sum(
            (r["summary"]["empirical_certificate_loss"] or 0) > 0 for r in diagnostics
        ),
        "largest_empirical_loss": max(
            s["empirical_certificate_loss"] or 0 for s in summaries
        ),
    }


def step_limits(records: list) -> dict:
    limits = {}
    for target in ("background", "boundary"):
        chosen = sorted(
            (
                r
                for r in records
                if r["study"] == "repeated"
                and r["selector"] == "erm_large_step"
                and r["mode"] == "refit"
                and r["target"] == target
                and r["replicate"] == 0
            ),
            key=lambda r: r["round"],
        )
        limits[target] = [
            {"step_limit": r["summary"]["step_capacity"], "step": r["summary"]["step"]}
            for r in chosen
        ]
    return limits


def backtracking_sweep(study: Study) -> dict:
    results = {}
    for factor, cap in SEARCH_CAPS.items():
        update = replace(study.update, shrink=factor, max_backtracks=cap)
        rows = []
        for selector, mode in ARMS:
            for replicate in range(study.spec.replicates):
                final = {
                    target: study.run_arm(selector, mode, target, replicate, update)[
                        -1
                    ]["coverage"]
                    for target in ("background", "boundary")
                }
                rows.append(
                    {
                        "selector": selector,
                        "mode": mode,
                        "replicate": replicate,
                        "background_percent": 100 * final["background"],
                        "boundary_percent": 100 * final["boundary"],
                        "difference_points": 100
                        * (final["boundary"] - final["background"]),
                    }
                )
        results[str(factor)] = rows
    return results


def initial_diagnostics(study: Study) -> dict:
    family, discovery, gain = study.family, study.discovery, study.initial
    update = study.update
    costs = study.costs(gain, discovery)
    laws = {
        target: target_law(
            costs,
            discovery.weights,
            update.threshold,
            target,
            width=study.spec.band_width,
            mixture=study.spec.mixture,
        )
        for target in ("background", "boundary")
    }
    directions = {t: study.direction(gain, law.weights) for t, law in laws.items()}
    background, boundary = directions["background"], directions["boundary"]
    cosine = np.sum(background * boundary) / (
        np.linalg.norm(background) * np.linalg.norm(boundary)
    )
    radius_only = {}
    for target, direction in directions.items():
        step = update.initial_step
        while np.linalg.norm(step * direction, 2) > update.kappa:
            step *= update.shrink
        radius_only[target] = {
            "step": step,
            "coverage": study.coverage(gain + step * direction),
            "smallest_known_margin": float(
                verified_margins(
                    family.closed_loops(gain + step * direction), KNOWN_CERTIFICATE
                ).min()
            ),
        }
    result = backtrack(
        family, discovery, gain, KNOWN_CERTIFICATE, laws["background"].weights, update
    )
    change = result.gain - gain
    old, new = (
        study.costs(gain, study.reference),
        study.costs(result.gain, study.reference),
    )
    drift = cost_drift_bound(family, study.reference, gain, change, update)
    ties = {}
    for replicate in range(study.spec.replicates):
        table, draws = study.empirical(gain, replicate, 0)
        empirical = empirical_losses(table, draws)
        ties[replicate] = {
            "smallest_empirical_loss": float(empirical.min()),
            "tied_candidates": int(
                np.sum(empirical <= empirical.min() + ls.TIE_TOLERANCE)
            ),
        }
    return {
        "direction_cosine": float(cosine),
        "band_conditions": int(laws["boundary"].selected.sum()),
        "training_conditions": len(costs),
        "known_certificate_smallest_margin": float(
            verified_margins(family.closed_loops(gain), KNOWN_CERTIFICATE).min()
        ),
        "guarded_first_step": result.step,
        "radius_limited_step": radius_only,
        "candidates": len(study.grid.matrices),
        "ties": ties,
        "first_update_cost_drift": {
            "bound_by_plant": sorted({round(float(v), 2) for v in drift}),
            "largest_actual_change": float(np.abs(new - old).max()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest_path = RUN / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records = manifest["records"]
    study = Study(manifest)
    payload = {
        "schema_version": 1,
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "reproduced_round_records": check_reproduction(study, records),
        "recorded": recorded_summary(records),
        "max_capacity_refit_step_limits": step_limits(records),
        "initial_gain": initial_diagnostics(study),
        "search_caps": {str(k): v for k, v in SEARCH_CAPS.items()},
        "boundary_vs_background_by_backtracking_factor": backtracking_sweep(study),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "update_diagnostics.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
