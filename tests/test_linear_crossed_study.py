"""Pairing, failure handling, and artifact-only analysis of the crossed study."""

import json

import numpy as np
import pytest

from active_eval_gym import linear_crossed_study as lcs
from active_eval_gym import linear_family_study as lfs
from active_eval_gym.linear_crossed_study import (
    MODES,
    SELECTORS,
    TARGETS,
    analyze_crossed_study,
    crossed_spec,
    fitting_stream,
    run_crossed_study,
)
from active_eval_gym.linear_family_study import FAMILY_TYPES, run_family_study

SPEC = crossed_spec("smoke")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("crossed")
    crossed = run_crossed_study(root / "crossed", SPEC, progress=lambda _: None)
    family = run_family_study(
        root / "family",
        lfs.FamilyStudySpec.from_preset("smoke"),
        progress=lambda _: None,
    )
    return crossed, family


def entries(root):
    return json.loads((root / "manifest.json").read_text())["families"]


def test_full_preset_reuses_the_family_study_settings():
    full, family = crossed_spec("full"), lfs.FamilyStudySpec()
    assert full.families_per_type == 200
    assert full.master_seed == family.master_seed
    assert full.rounds == family.rounds
    assert lfs.sample_plants(full, "general", 7)[0].tolist() == (
        lfs.sample_plants(family, "general", 7)[0].tolist()
    )


def test_fitting_streams_are_distinct_and_leave_room_for_round_offsets():
    streams = {
        fitting_stream(name, index) for name in FAMILY_TYPES for index in range(400)
    }
    assert len(streams) == 4 * 400
    assert max(streams) < 100_000


def test_every_family_runs_every_combination(runs):
    crossed, _ = runs
    for entry in entries(crossed):
        records = json.loads((crossed / entry["runs"]).read_text())["records"]
        if entry["status"] != "selected":
            assert records == []
            continue
        expected = {(s, m, t) for s in SELECTORS for m in MODES for t in TARGETS}
        assert {(r["selector"], r["mode"], r["target"]) for r in records} == expected
        assert len(records) == len(expected)
        for record in records:
            rows = record["rounds"]
            for before, after in zip(rows, rows[1:], strict=False):
                assert after["old_gain"] == before["new_gain"]


def test_arms_with_one_rule_share_the_first_selection(runs):
    crossed, _ = runs
    for entry in entries(crossed):
        records = json.loads((crossed / entry["runs"]).read_text())["records"]
        first = {}
        for record in records:
            row = record["rounds"][0]
            first.setdefault(record["selector"], set()).add(row.get("candidate_index"))
        assert all(len(choices) == 1 for choices in first.values())


def test_fixed_arms_keep_one_certificate(runs):
    crossed, _ = runs
    for entry in entries(crossed):
        records = json.loads((crossed / entry["runs"]).read_text())["records"]
        for record in records:
            if record["mode"] == "fixed":
                # Halted rounds choose nothing, so they carry no certificate.
                chosen = {
                    r["candidate_index"]
                    for r in record["rounds"]
                    if "candidate_index" in r
                }
                assert len(chosen) <= 1


def test_a_failed_rule_stays_in_the_means_at_its_current_gain(tmp_path, monkeypatch):
    original = lfs.ls.choose_certificate

    def fail_ties(family, gain, matrices, empirical, probe, selector, update):
        if selector == "erm_large_step":
            return lfs.ls.CertificateChoice(None, None, "no_verified_erm")
        return original(family, gain, matrices, empirical, probe, selector, update)

    monkeypatch.setattr(lfs.ls, "choose_certificate", fail_ties)
    (tmp_path / "raw").mkdir()
    (tmp_path / "runs").mkdir()
    entry = next(
        result
        for index in range(20)
        if (result := lcs.collect_family((SPEC, tmp_path, "same_shape", index)))[
            "status"
        ]
        == "selected"
    )
    row = lcs.analyze_family(tmp_path, entry)
    assert row["erm_large_step_fixed_boundary_halted"] is True
    assert row["erm_large_step_fixed_boundary_last"] == row["initial_coverage"]


def test_analysis_matches_the_family_study_and_uses_only_artifacts(
    runs, tmp_path, monkeypatch
):
    crossed, family = runs

    def forbidden(*args, **kwargs):
        raise AssertionError("Analysis must not run updates.")

    monkeypatch.setattr(lfs, "run_arm", forbidden)
    report = analyze_crossed_study(crossed, family, tmp_path / "report")
    summary = json.loads((report / "summary.json").read_text())
    assert summary["family_study_cells_matched"] > 0
    assert set(summary["families"]) == set(FAMILY_TYPES)
    with pytest.raises(FileExistsError):
        analyze_crossed_study(crossed, family, tmp_path / "report")


def test_a_family_study_mismatch_stops_analysis(runs):
    crossed, family = runs
    rows = [lcs.analyze_family(crossed, e) for e in entries(crossed)]
    row = next(r for r in rows if r["shared_certificate"])
    row["model_grid_fixed_boundary_last"] += 0.5
    with pytest.raises(ValueError, match="Family study mismatch"):
        lcs.check_against_family_study(rows, family)


def test_parallel_collection_matches_serial_collection(runs, tmp_path):
    crossed, _ = runs
    root = run_crossed_study(tmp_path / "jobs", SPEC, jobs=2, progress=lambda _: None)
    assert [e["raw_sha256"] for e in entries(crossed)] == [
        e["raw_sha256"] for e in entries(root)
    ]
    assert [e["runs_sha256"] for e in entries(crossed)] == [
        e["runs_sha256"] for e in entries(root)
    ]


def test_effects_are_paired_differences():
    rows = [{"a": 0.5, "b": 0.25}, {"a": 0.5, "b": 0.75}]
    summary = lcs._effect(rows, "a", "b", (100, 0, 1.0))
    assert summary["mean"] == pytest.approx(0.0)
    assert summary["positive_share"] == 0.5
    assert np.isclose(summary["negative_share"], 0.5)
