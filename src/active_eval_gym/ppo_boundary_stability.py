"""Arm-blind scoring of continuation settings from saved campaign summaries.

Settings are chosen before the locked comparison, so everything here pools the
arms. No output may show which curriculum did better: a choice made on that
basis would leak the question the study exists to answer.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from active_eval_gym.serialization import write_json_new

NOMINAL_ANGLE_DEG = 0.0
NOMINAL_LENGTH = 0.5
NOMINAL_RECOVERY_FLOOR = 0.90
COLLAPSE_COVERAGE_FRACTION = 0.5
MAX_COLLAPSED_FINALS = 1
MAX_COLLAPSED_CHECKPOINTS = 3
MIN_RECOVERY_GAIN = 0.02


@dataclass(frozen=True)
class CheckpointScore:
    """The pooled quantities one checkpoint contributes to a candidate."""

    coverage: float
    mean_recovery: float
    nominal_recovery: float
    accepted: frozenset[str]


def _checkpoint_scores(summary: dict[str, Any], tau: float) -> dict[str, Any]:
    cells: dict[str, list[dict[str, Any]]] = {}
    for cell in summary["cells"]:
        cells.setdefault(cell["checkpoint_id"], []).append(cell)
    return {
        checkpoint: CheckpointScore(
            coverage=summary["coverage"][checkpoint]["thresholds"][str(tau)][
                "recovery_plug_in"
            ],
            mean_recovery=summary["coverage"][checkpoint]["mean_recovery_probability"],
            nominal_recovery=_nominal_recovery(rows),
            accepted=frozenset(
                row["condition_id"] for row in rows if row["recovery_rate"] >= tau
            ),
        )
        for checkpoint, rows in cells.items()
    }


def _nominal_recovery(cells: list[dict[str, Any]]) -> float:
    matches = [
        cell["recovery_rate"]
        for cell in cells
        if cell["parameters"]["initial_theta_deg"] == NOMINAL_ANGLE_DEG
        and cell["parameters"]["length"] == NOMINAL_LENGTH
    ]
    if len(matches) != 1:
        raise ValueError("Stability scoring requires exactly one nominal cell.")
    return matches[0]


def score_candidate(
    pilot: dict[str, Any],
    snapshots: dict[str, Any],
    *,
    parent_policy_id: str,
    tau: float,
) -> dict[str, Any]:
    """Return one candidate's pooled collapse counts, gains, and gate results."""

    finals = _checkpoint_scores(pilot, tau)
    parent = finals.pop(parent_policy_id)
    trained = {**finals, **_checkpoint_scores(snapshots, tau)}
    if len(trained) != len(finals) + len(snapshots["coverage"]):
        raise ValueError("A checkpoint appears in both campaigns.")
    floor = COLLAPSE_COVERAGE_FRACTION * parent.coverage

    def collapsed(score: CheckpointScore) -> bool:
        return score.nominal_recovery < NOMINAL_RECOVERY_FLOOR or score.coverage < floor

    collapsed_finals = sum(map(collapsed, finals.values()))
    collapsed_checkpoints = sum(map(collapsed, trained.values()))
    final_recovery = sum(s.mean_recovery for s in finals.values()) / len(finals)
    gain = final_recovery - parent.mean_recovery
    return {
        "final_checkpoints": len(finals),
        "all_checkpoints": len(trained),
        "collapsed_finals": collapsed_finals,
        "collapsed_checkpoints": collapsed_checkpoints,
        "pooled_final_mean_recovery": final_recovery,
        "pooled_final_coverage": sum(s.coverage for s in finals.values()) / len(finals),
        "parent_mean_recovery": parent.mean_recovery,
        "parent_coverage": parent.coverage,
        "recovery_gain": gain,
        "pooled_parent_accepted_retained": sum(
            len(parent.accepted & s.accepted) for s in finals.values()
        )
        / (len(finals) * len(parent.accepted)),
        "stability_gate": (
            collapsed_finals <= MAX_COLLAPSED_FINALS
            and collapsed_checkpoints <= MAX_COLLAPSED_CHECKPOINTS
        ),
        "learning_gate": gain >= MIN_RECOVERY_GAIN,
    }


def select_candidate(scores: dict[str, dict[str, Any]]) -> str | None:
    """Return the best candidate passing both gates, or None if none passes."""

    passing = {
        name: score["pooled_final_mean_recovery"]
        for name, score in scores.items()
        if score["stability_gate"] and score["learning_gate"]
    }
    return max(passing, key=passing.get) if passing else None


def score_studies(candidates: dict[str, str], output: Path) -> dict[str, Any]:
    """Score analyzed studies from saved summaries and record the selection."""

    scores = {}
    for name, study in candidates.items():
        root = Path(study)
        config = json.loads((root / "protocol/resolved-config.json").read_text())
        metrics = root / "metrics_v1"
        scores[name] = score_candidate(
            json.loads((metrics / "pilot/summary.json").read_text()),
            json.loads((metrics / "pilot-snapshots/summary.json").read_text()),
            parent_policy_id=config["parent_policy_id"],
            tau=config["acceptance"]["primary_tau"],
        )
    report = {
        "schema_version": 1,
        "pooled_across_arms": True,
        "scores": scores,
        "selected": select_candidate(scores),
    }
    write_json_new(output, report)
    return report
