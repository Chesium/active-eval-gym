import json
from pathlib import Path

import pytest

from active_eval_gym.ppo_boundary_stability import (
    score_candidate,
    score_studies,
    select_candidate,
)

PARENT = "parent"
TAU = 0.9
CONDITIONS = (("nominal", 0.0, 0.5), ("a", 10.0, 0.5), ("b", 20.0, 1.0))


def _summary(checkpoints: dict[str, dict]) -> dict:
    cells = []
    coverage = {}
    for name, values in checkpoints.items():
        rates = values["rates"]
        coverage[name] = {
            "mean_recovery_probability": values["mean"],
            "thresholds": {str(TAU): {"recovery_plug_in": values["coverage"]}},
        }
        cells += [
            {
                "checkpoint_id": name,
                "condition_id": condition,
                "parameters": {"initial_theta_deg": angle, "length": length},
                "recovery_rate": rate,
            }
            for (condition, angle, length), rate in zip(CONDITIONS, rates, strict=True)
        ]
    return {"cells": cells, "coverage": coverage}


def _point(nominal=1.0, coverage=0.2, mean=0.2) -> dict:
    return {"rates": (nominal, 1.0, 0.0), "coverage": coverage, "mean": mean}


def _candidate(finals: list[dict], snapshots: list[dict]):
    pilot = {PARENT: _point(coverage=0.1, mean=0.12)}
    pilot |= {f"final-{index}": point for index, point in enumerate(finals)}
    later = {f"snapshot-{index}": point for index, point in enumerate(snapshots)}
    return _summary(pilot), _summary(later)


def _score(finals: list[dict], snapshots: list[dict]) -> dict:
    pilot, later = _candidate(finals, snapshots)
    return score_candidate(pilot, later, parent_policy_id=PARENT, tau=TAU)


def test_a_stable_learning_candidate_passes_both_gates() -> None:
    score = _score([_point()] * 9, [_point()] * 27)

    assert score["final_checkpoints"] == 9
    assert score["all_checkpoints"] == 36
    assert score["collapsed_checkpoints"] == 0
    assert score["recovery_gain"] == pytest.approx(0.08)
    assert score["stability_gate"] and score["learning_gate"]


@pytest.mark.parametrize(
    "bad", [_point(nominal=0.875), _point(coverage=0.049)], ids=["nominal", "coverage"]
)
def test_either_collapse_signal_counts(bad: dict) -> None:
    assert _score([bad] + [_point()] * 8, [_point()] * 27)["collapsed_finals"] == 1


def test_thresholds_themselves_are_not_collapse() -> None:
    edge = _point(nominal=0.9, coverage=0.05)

    assert _score([edge] * 9, [edge] * 27)["collapsed_checkpoints"] == 0


def test_two_collapsed_finals_fail_stability() -> None:
    bad = _point(nominal=0.0)

    assert _score([bad] + [_point()] * 8, [_point()] * 27)["stability_gate"]
    assert not _score([bad] * 2 + [_point()] * 7, [_point()] * 27)["stability_gate"]


def test_four_collapsed_checkpoints_fail_stability() -> None:
    bad = _point(nominal=0.0)
    good = [_point()]

    assert _score(good * 9, [bad] * 3 + good * 24)["stability_gate"]
    assert not _score(good * 9, [bad] * 4 + good * 23)["stability_gate"]


def test_a_frozen_policy_fails_the_learning_gate() -> None:
    frozen = _point(mean=0.12)
    edge = _point(mean=0.14 + 1e-9)

    assert not _score([frozen] * 9, [frozen] * 27)["learning_gate"]
    assert _score([edge] * 9, [edge] * 27)["learning_gate"]


def test_retention_of_parent_accepted_conditions_is_pooled() -> None:
    lost = {"rates": (0.0, 0.0, 1.0), "coverage": 0.2, "mean": 0.2}

    score = _score([_point()] * 8 + [lost], [_point()] * 27)

    assert score["pooled_parent_accepted_retained"] == pytest.approx(8 / 9)


def test_a_checkpoint_in_both_campaigns_is_refused() -> None:
    pilot, _ = _candidate([_point()], [])
    repeated = _summary({"final-0": _point()})

    with pytest.raises(ValueError, match="both campaigns"):
        score_candidate(pilot, repeated, parent_policy_id=PARENT, tau=TAU)


def test_selection_needs_both_gates_and_prefers_recovery() -> None:
    def score(recovery, stable=True, learning=True):
        return {
            "pooled_final_mean_recovery": recovery,
            "stability_gate": stable,
            "learning_gate": learning,
        }

    assert (
        select_candidate(
            {
                "unstable": score(0.9, stable=False),
                "frozen": score(0.8, learning=False),
                "lower": score(0.2),
                "higher": score(0.3),
            }
        )
        == "higher"
    )
    assert select_candidate({"unstable": score(0.9, stable=False)}) is None


def test_report_is_pooled_and_never_names_a_checkpoint(tmp_path: Path) -> None:
    study = tmp_path / "study"
    (study / "protocol").mkdir(parents=True)
    (study / "protocol/resolved-config.json").write_text(
        json.dumps({"parent_policy_id": PARENT, "acceptance": {"primary_tau": TAU}})
    )
    pilot, later = _candidate([_point()] * 9, [_point()] * 27)
    for name, summary in (("pilot", pilot), ("pilot-snapshots", later)):
        (study / "metrics_v1" / name).mkdir(parents=True)
        (study / "metrics_v1" / name / "summary.json").write_text(json.dumps(summary))
    output = tmp_path / "report.json"

    report = score_studies({"only": str(study)}, output)

    assert report["selected"] == "only"
    assert report["pooled_across_arms"] is True
    assert "final-" not in output.read_text()
    assert "snapshot-" not in output.read_text()
    with pytest.raises(FileExistsError):
        score_studies({"only": str(study)}, output)
