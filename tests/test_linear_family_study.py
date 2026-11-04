"""Sampling, the capacity step, and artifact-only analysis of the family study."""

import json
from dataclasses import replace

import numpy as np
import pytest

from active_eval_gym import linear_family_study as lfs
from active_eval_gym.certificates import KNOWN_CERTIFICATE
from active_eval_gym.linear_family_study import (
    FAMILY_TYPES,
    REFERENCE_ID,
    RULES,
    FamilyStudySpec,
    analyze_family_study,
    run_family_study,
    sample_plants,
)
from active_eval_gym.linear_updates import (
    ConditionGrid,
    LinearFamily,
    UpdateSpec,
    backtrack,
    capacity_step,
    margin_consumption,
    policy_actions,
    rollout,
    step_capacity,
    trajectory_costs,
)

SPEC = FamilyStudySpec.from_preset("smoke")


def scalar_fixture():
    family = LinearFamily(
        np.array([[[0.8]]]), np.ones((1, 1, 1)), np.ones((1, 1)), np.zeros((1, 1))
    )
    resets = np.array([[0.25], [0.5], [0.75], [1.0]])
    grid = ConditionGrid(np.zeros(4, dtype=int), resets, np.full(4, 0.25))
    return family, grid, np.array([[0.2]]), UpdateSpec(horizon=2, threshold=1.2)


@pytest.mark.parametrize("family_type", FAMILY_TYPES)
def test_each_family_has_its_own_reproducible_stream(family_type):
    first, actuators = sample_plants(SPEC, family_type, 3)
    again, _ = sample_plants(SPEC, family_type, 3)
    other, _ = sample_plants(SPEC, family_type, 4)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
    assert first.shape == (2, 2, 2) and actuators.shape == (2, 2, 2)


def test_sampling_rules_match_the_declared_families():
    loops, _ = sample_plants(SPEC, "same_shape", 0)
    assert np.all(loops[:, 1, 0] == 0)
    assert np.all(loops[:, 0, 0] >= 0.5) and np.all(loops[:, 0, 0] <= 0.9)
    loops, _ = sample_plants(SPEC, "general", 0)
    radii = np.abs(np.linalg.eigvals(loops)).max(axis=1)
    assert np.all(radii >= 0.6 - 1e-12) and np.all(radii <= 0.9 + 1e-12)
    _, actuators = sample_plants(SPEC, "rotated_actuator", 0)
    assert np.array_equal(actuators[0], np.eye(2))
    assert np.allclose(actuators[1] @ actuators[1].T, np.eye(2))
    assert not np.allclose(actuators[1], np.eye(2))
    loops, _ = sample_plants(SPEC, "mirrored", 0)
    assert np.abs(loops[0] + loops[1]).max() < 0.5


def test_capacity_step_stays_inside_the_guard_and_beats_the_grid():
    family, grid, gain, spec = scalar_fixture()
    p, weights = np.eye(1), grid.weights
    result = capacity_step(family, grid, gain, p, weights, spec)
    grid_result = backtrack(family, grid, gain, p, weights, spec)
    assert result.status == "accepted"
    assert result.step >= grid_result.step
    assert np.linalg.norm(result.gain - gain, 2) <= spec.kappa
    delta = result.gain - gain
    margins = 1 - (family.a[0, 0, 0] - gain[0, 0]) ** 2
    assert np.all(
        margin_consumption(family, gain, p, delta) <= (1 - spec.rho) * margins
    )
    assert result.step == pytest.approx(
        step_capacity(family, gain, p, result.direction, spec), rel=1e-6
    )


def test_cap_limits_the_first_trial():
    family, grid, gain, spec = scalar_fixture()
    result = capacity_step(family, grid, gain, np.eye(1), grid.weights, spec, cap=0.01)
    assert result.step == 0.01


def test_parallel_directions_give_the_same_guard_limited_update():
    family, grid, gain, spec = scalar_fixture()
    p = np.eye(1)
    spread = capacity_step(family, grid, gain, p, grid.weights, spec)
    focused = capacity_step(
        family, grid, gain, p, np.array([0.05, 0.05, 0.05, 0.85]), spec
    )
    assert focused.step != spread.step
    assert focused.gain == pytest.approx(spread.gain, rel=1e-8)


def test_failed_verification_and_zero_gradient_are_recorded():
    family, grid, gain, spec = scalar_fixture()
    weights = grid.weights
    unstable = LinearFamily(family.a + 1, family.b, family.q, family.r)
    result = capacity_step(unstable, grid, gain, np.eye(1), weights, spec)
    assert result.status == "verification_failed" and result.step == 0
    settled = capacity_step(family, grid, family.a[0], np.eye(1), weights, spec)
    assert settled.status == "zero_gradient"


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("family") / "run"
    return run_family_study(root, SPEC, progress=lambda _: None)


def test_run_saves_every_declared_family_and_analysis_uses_only_artifacts(
    study, tmp_path, monkeypatch
):
    manifest = json.loads((study / "manifest.json").read_text())
    assert manifest["completion"] == "complete"
    assert len(manifest["families"]) == 1 + 2 * len(FAMILY_TYPES)

    def forbidden(*args, **kwargs):
        raise AssertionError("Analysis must not run updates.")

    monkeypatch.setattr(lfs, "backtrack", forbidden)
    monkeypatch.setattr(lfs, "capacity_step", forbidden)
    report = analyze_family_study(study, tmp_path / "report")
    summary = json.loads((report / "summary.json").read_text())
    assert set(summary["families"]) == set(FAMILY_TYPES)
    assert summary["reference"]["rules"]
    assert (report / "family_metrics.csv").exists()
    with pytest.raises(FileExistsError):
        analyze_family_study(study, tmp_path / "report")


def test_saved_states_reproduce_a_fresh_rollout_of_every_gain(study):
    manifest = json.loads((study / "manifest.json").read_text())
    entry = next(e for e in manifest["families"] if e["family_id"] != REFERENCE_ID)
    with np.load(study / entry["raw"]) as raw:
        family = LinearFamily(raw["a"], raw["b"], raw["q"], raw["r"])
        grid = ConditionGrid(raw["plant_ids"], raw["resets"], raw["weights"])
        for gain, states in zip(raw["gains"], raw["states"], strict=True):
            fresh, actions = rollout(family, grid, gain, states.shape[1] - 1)
            assert np.array_equal(states, fresh)
            saved_actions = policy_actions(gain, states)
            assert np.array_equal(
                trajectory_costs(family, states, saved_actions),
                trajectory_costs(family, fresh, actions),
            )


def test_analysis_rejects_a_changed_file_and_an_incomplete_run(study, tmp_path):
    manifest = json.loads((study / "manifest.json").read_text())
    entry = manifest["families"][1]
    copy = tmp_path / "copy"
    (copy / "raw").mkdir(parents=True)
    (copy / "runs").mkdir()
    (copy / "raw" / f"{entry['family_id']}.npz").write_bytes(b"changed")
    (copy / "runs" / f"{entry['family_id']}.json").write_bytes(
        (study / entry["runs"]).read_bytes()
    )
    with pytest.raises(ValueError, match="Hash mismatch"):
        lfs._load_family(copy, entry)
    (copy / "manifest.json").write_text(json.dumps({"completion": "partial"}))
    with pytest.raises(ValueError, match="completed run"):
        analyze_family_study(copy, tmp_path / "report")


def test_a_family_without_a_certificate_stays_in_the_study(tmp_path, monkeypatch):
    def no_certificate(*args, **kwargs):
        return lfs.ls.CertificateChoice(None, None, "no_verified_candidate")

    (tmp_path / "raw").mkdir()
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(lfs.ls, "choose_certificate", no_certificate)
    entry = lfs.collect_ensemble_family((SPEC, tmp_path, "general", 0))
    assert entry["status"] == "no_shared_certificate"
    row = lfs.analyze_family(tmp_path, entry, replace(SPEC).__dict__)
    assert row["shared_certificate"] is False
    assert row["cos_band"] is not None


def test_parallel_collection_matches_serial_collection(study, tmp_path):
    root = run_family_study(tmp_path / "jobs", SPEC, jobs=2, progress=lambda _: None)
    serial = json.loads((study / "manifest.json").read_text())["families"]
    parallel = json.loads((root / "manifest.json").read_text())["families"]
    assert [e["raw_sha256"] for e in serial] == [e["raw_sha256"] for e in parallel]
    assert [e["runs_sha256"] for e in serial] == [e["runs_sha256"] for e in parallel]


def reference_context():
    spec, study = FamilyStudySpec(), lfs.ls.StudySpec()
    family, train, reference, _, update = lfs.ls.primary_fixture(study)
    return lfs._make_context(
        "reference", family, train, reference, update, study.band_width, spec
    ), study


def known_choice():
    return lfs.ls.CertificateChoice(KNOWN_CERTIFICATE.copy(), None, "selected")


def test_rounds_continue_from_the_previous_gain(study):
    manifest = json.loads((study / "manifest.json").read_text())
    updates = 0
    for entry in manifest["families"]:
        runs = json.loads((study / entry["runs"]).read_text())
        for record in runs["records"]:
            rounds = record["rounds"]
            assert [r["round"] for r in rounds] == list(range(SPEC.rounds))
            for before, after in zip(rounds, rounds[1:], strict=False):
                assert after["old_gain"] == before["new_gain"]
            updates += sum(r["step"] > 0 for r in rounds)
    assert updates > 0


@pytest.mark.parametrize(
    ("target", "steps", "coverage"),
    [
        ("background", (1 / 512, 1 / 512, 1 / 256), 0.83984375),
        ("boundary", None, 0.85546875),
        ("high_cost", None, 0.8359375),
    ],
)
def test_backtracking_reproduces_the_original_study(target, steps, coverage):
    ctx, _ = reference_context()
    rows = lfs.run_arm(ctx, lfs._always(known_choice()), target, "backtracking")
    if steps is not None:
        assert tuple(r["step"] for r in rows) == steps
    states = ctx.log.states[rows[-1]["new_gain"]]
    gain = ctx.log.gains[rows[-1]["new_gain"]]
    costs = trajectory_costs(ctx.family, states, policy_actions(gain, states))
    accepted = costs <= ctx.update.threshold
    assert ctx.reference.weights @ accepted == pytest.approx(coverage, abs=1e-12)


def test_capacity_rule_moves_at_least_as_far_as_the_grid_on_each_first_round():
    ctx, _ = reference_context()
    pick = lfs._always(known_choice())
    grid = lfs.run_arm(ctx, pick, "background", "backtracking")[0]
    exact = lfs.run_arm(ctx, pick, "background", "capacity")[0]
    assert exact["step"] >= grid["step"]
    assert exact["step"] == pytest.approx(exact["capacity"], rel=1e-6)


def test_rules_and_sign_test_are_declared():
    assert RULES == ("backtracking", "capacity", "guard_only")
    assert lfs._sign_test(np.array([1.0, 2.0, 3.0, 0.0])) == 0.25
    assert lfs._sign_test(np.zeros(3)) is None
    row = {"status": "search_exhausted"}
    assert lfs._binding(row, "capacity") == "search_exhausted"
    assert lfs._binding({"status": "accepted"}, "backtracking") == "grid"
