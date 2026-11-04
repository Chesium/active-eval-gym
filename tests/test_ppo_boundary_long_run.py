"""Single-arm planning and faithful analysis of an extended fixed-budget run."""

import json
import runpy
from dataclasses import replace

import numpy as np
import pytest

from active_eval_gym.config import load_ppo_boundary_study
from active_eval_gym.ppo_boundary import job_accounting, tier_children
from tests.helpers import ROOT

LONG = ROOT / "configs/experiments/cartpole_ppo_boundary_v4_long.toml"
V4 = ROOT / "configs/experiments/cartpole_ppo_boundary_v4.toml"
FOCUS = ROOT / "configs/experiments/cartpole_ppo_boundary_v4_focus.toml"


def test_focus_changes_only_checkpoint_density_and_budget():
    old = load_ppo_boundary_study(LONG)
    focus = load_ppo_boundary_study(FOCUS)
    assert replace(focus, study_id=old.study_id, tiers=old.tiers) == old
    tier = focus.tier("pilot")
    assert tier.additional_transitions == 256000
    assert tier.snapshot_transitions == tuple(range(102400, 256000, 25600))
    assert tier.replicate_ids == (0, 1, 2)
    assert tier.arms == ("acceptance_boundary",)
    planned = job_accounting(focus)["tiers"]["pilot"]
    assert planned["children"] == 3
    assert planned["snapshot_evaluation"]["jobs"] == 18 * 273


def test_long_design_changes_only_identity_and_execution_budget():
    old = load_ppo_boundary_study(V4)
    new = load_ppo_boundary_study(LONG)
    assert replace(new, study_id=old.study_id, tiers=old.tiers) == old
    assert new.tier("pilot").replicate_ids == (0, 1, 2)
    assert new.tier("pilot").additional_transitions == 1_024_000
    assert new.tier("pilot").snapshot_transitions[-2:] == (768_000, 896_000)


def test_only_explicit_boundary_jobs_are_planned():
    spec = load_ppo_boundary_study(LONG)
    jobs = tier_children(spec, "pilot")
    assert len(jobs) == 3
    assert {j["arm"] for j in jobs} == {"acceptance_boundary"}
    accounting = job_accounting(spec)["tiers"]["pilot"]
    assert accounting["training_transitions"] == 3_072_000
    assert accounting["primary_evaluation"]["jobs"] == 4 * 273
    assert accounting["snapshot_evaluation"]["jobs"] == 15 * 273


@pytest.mark.parametrize(
    "arms", ["[]", '["unknown"]', '["acceptance_boundary", "acceptance_boundary"]']
)
def test_invalid_tier_arm_subsets_are_refused(tmp_path, arms):
    design = ROOT / "configs/policies/cartpole_ppo_nominal_v1.toml"
    text = (
        LONG.read_text()
        .replace(
            'design_config = "../policies/cartpole_ppo_nominal_v1.toml"',
            f'design_config = "{design}"',
        )
        .replace('arms = ["acceptance_boundary"]', f"arms = {arms}")
    )
    path = tmp_path / "invalid.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match="arms"):
        load_ppo_boundary_study(path)


@pytest.fixture(scope="module")
def analysis():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / "scripts"))
        return runpy.run_path(str(ROOT / "scripts/analyze_ppo_boundary_long_run.py"))


def test_prefix_verification_checks_reset_and_schedule_columns(analysis):
    old = {
        "state": np.arange(12).reshape(3, 4),
        "sampling_probabilities": np.array([[0.2, 0.8]]),
    }
    extended = {
        "state": np.arange(24).reshape(6, 4),
        "sampling_probabilities": np.array([[0.2, 0.8], [0.3, 0.7]]),
    }
    analysis["verify_prefix_arrays"](old, extended)
    extended["sampling_probabilities"][0, 0] = 0.3
    with pytest.raises(AssertionError, match="Prefix column sampling_probabilities"):
        analysis["verify_prefix_arrays"](old, extended)


def test_equal_coverage_can_hide_large_set_turnover(analysis):
    def cells(rates):
        return {
            str(index): {
                "recovery_rate": rate,
                "survival_rate": rate,
                "parameters": {
                    "initial_theta_deg": 0 if index == 0 else index * 10,
                    "length": 0.5,
                },
            }
            for index, rate in enumerate(rates)
        }

    parent = cells([1, 1, 0, 0])
    current = cells([1, 0, 1, 0])
    result = analysis["checkpoint_statistics"](parent, current, parent, parent)
    assert result["accepted"] == 2
    assert result["coverage"] == 0.5
    assert result["gained_parent"] == result["lost_parent"] == 1
    assert result["set_turnover_previous"] == 2
    assert result["jaccard_previous"] == pytest.approx(1 / 3)


def test_plot_loader_distinguishes_million_step_final_from_old_endpoint(tmp_path):
    module = runpy.run_path(str(ROOT / "scripts/plot_ppo_boundary_shift.py"))
    (tmp_path / "protocol").mkdir()
    (tmp_path / "protocol/resolved-config.json").write_text(
        json.dumps({"tiers": [{"name": "pilot", "additional_transitions": 1_024_000}]})
    )

    def cell(checkpoint, rate):
        return {
            "parameters": {"initial_theta_deg": 0, "length": 0.5},
            "checkpoint_id": checkpoint,
            "recovery_rate": rate,
        }

    for campaign, cells in [
        (
            "pilot",
            [cell("parent", 1), cell("study-pilot-r000-acceptance_boundary", 0.5)],
        ),
        ("pilot-snapshots", [cell("study-pilot-r000-acceptance_boundary-t102400", 1)]),
    ]:
        directory = tmp_path / "metrics_v1" / campaign
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(json.dumps({"cells": cells}))
    result = module["load_cells"](tmp_path, "pilot", "parent")
    assert result["children"][("acceptance_boundary", 0, 102400)][(0, 0.5)] == 1
    assert result["children"][("acceptance_boundary", 0, 1024000)][(0, 0.5)] == 0.5
    assert module["present_arms"](result) == ("acceptance_boundary",)
