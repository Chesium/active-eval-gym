"""Artifact replay and branch pairing for the script-based analytical study."""

import json
from collections import defaultdict
from dataclasses import replace

import numpy as np
import pytest

from active_eval_gym import linear_study, linear_updates
from active_eval_gym.certificates import candidate_grid
from active_eval_gym.linear_study import (
    CertificateChoice,
    StudySpec,
    analyze_study,
    run_study,
)
from active_eval_gym.linear_updates import margin_consumption


def test_saved_study_pairs_branches_and_analyzes_without_execution(
    tmp_path, monkeypatch
):
    spec = replace(
        StudySpec.from_preset("smoke"), directions=8, reference_directions=16, rounds=2
    )
    original = linear_study.choose_certificate

    def force_failure(family, gain, matrices, empirical, probe, selector, update):
        if selector == "erm_first":
            return CertificateChoice(np.eye(2), 0, "selected")
        return original(family, gain, matrices, empirical, probe, selector, update)

    monkeypatch.setattr(linear_study, "choose_certificate", force_failure)
    root = run_study(tmp_path / "run", spec, progress=lambda _: None)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["completion"] == "complete"
    assert manifest["study_spec"]["master_seed"] == 6180
    paired = defaultdict(list)
    fixed_p = {}
    records = manifest["records"]
    for record in records:
        with np.load(root / record["arrays"]) as loaded:
            arrays = dict(loaded)
        if record["study"] in ("one_step", "repeated"):
            paired[record["replicate"], record["round"], record["sample_size"]].append(
                arrays["fitting_indices"]
            )
        if record["study"] != "repeated":
            continue
        branch = (record["selector"], record["target"], record["replicate"])
        if record["selector"] == "erm_first":
            expected = "verification_failed" if record["round"] == 0 else "halted"
            assert record["status"] == expected
            np.testing.assert_array_equal(arrays["gain"], arrays["new_gain"])
        if record["mode"] == "fixed":
            if record["round"] == 0:
                fixed_p[branch] = arrays["p"]
            np.testing.assert_array_equal(arrays["p"], fixed_p[branch])
        if record["status"] == "accepted":
            family, _, _, update = linear_study._load_fixture(manifest, "primary")
            assert np.all(
                arrays["new_margins"] >= update.rho * arrays["margins"] - 1e-12
            )
            for trial in record["trial_history"]:
                consumption = margin_consumption(
                    family,
                    arrays["gain"],
                    arrays["p"],
                    trial["step"] * arrays["direction"],
                )
                np.testing.assert_allclose(consumption, trial["consumption"])
        if record["mode"] == "refit" and record["status"] != "halted":
            assert record["fitting_role"] == "fit"
    for draws in paired.values():
        for other in draws[1:]:
            np.testing.assert_array_equal(draws[0], other)
    assert not np.array_equal(paired[0, 0, 32][0], paired[0, 1, 32][0])
    scales = defaultdict(list)
    for record in records:
        if record["study"] == "scale":
            scales[record["target"]].append(record["summary"]["step"])
    for steps in scales.values():
        np.testing.assert_allclose(steps, steps[0])

    def forbidden(*args, **kwargs):
        raise AssertionError("Analysis must not run fitting, rollouts, or updates")

    for module, names in (
        (
            linear_study,
            ("rollout", "backtrack", "choose_certificate", "fitting_indices"),
        ),
        (linear_updates, ("rollout", "backtrack")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    report = analyze_study(root, tmp_path / "report")
    assert (report / "report.md").exists()
    assert (report / "paired_differences_v1.csv").exists()
    assert len(list(report.glob("*.png"))) >= 6
    info = json.loads((report / "analysis.json").read_text())
    assert info["records"] == len(records)
    assert info["raw_rollouts_collected"] == 0
    with pytest.raises(FileExistsError):
        analyze_study(root, report)
    with pytest.raises(FileExistsError):
        run_study(root, spec, progress=lambda _: None)
    # Detect raw corruption before trusting any reconstructed result.
    raw = root / next(iter(manifest["raw_sha256"]))
    raw.write_bytes(raw.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        analyze_study(root, tmp_path / "corrupt_report")


def test_presets_are_the_declared_experiment():
    full, smoke = StudySpec.from_preset("full"), StudySpec.from_preset("smoke")
    assert full.replicates == 10 and full.rounds == 20
    assert full.sample_sizes == (8, 32, 128) and full.landscape
    assert smoke.replicates == 2 and smoke.rounds == 3
    assert smoke.sample_sizes == (32,) and not smoke.landscape
    assert full.directions == smoke.directions == 64
    assert full.reference_directions == smoke.reference_directions == 256
    with pytest.raises(ValueError):
        StudySpec.from_preset("unknown")


def test_landscape_keeps_failed_candidates_and_all_target_maps(tmp_path, monkeypatch):
    def small_grid(spec):
        return candidate_grid(spec, eigenvalue_count=2, angle_count=2)

    monkeypatch.setattr(linear_study, "candidate_grid", small_grid)
    spec = replace(
        StudySpec.from_preset("full"),
        replicates=1,
        rounds=1,
        sample_sizes=(8,),
        directions=4,
        reference_directions=8,
    )
    run = run_study(tmp_path / "landscape", spec, progress=lambda _: None)
    manifest = json.loads((run / "manifest.json").read_text())
    landscape = [r for r in manifest["records"] if r["study"] == "landscape"]
    grid = small_grid(linear_study.CertificateSpec())
    assert len(landscape) == 3 * len(grid.matrices)
    assert any(r["status"] == "verification_failed" for r in landscape)
    assert {r["candidate_index"] for r in landscape} == set(range(len(grid.matrices)))
    assert all(
        r["summary"]["step"] == 0
        for r in landscape
        if r["status"] == "verification_failed"
    )
