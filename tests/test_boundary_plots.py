"""Tests for the index-space failure-boundary figures."""

import json
from pathlib import Path

import numpy as np
import pytest

from active_eval_gym.plotting import (
    _boundary_lookup,
    _boundary_surface,
    _cell_edges,
    _mirror_difference,
    _mirrored_angles,
    _nearest_filled,
    _straddles_half,
    plot_sweep,
)

EXPECTED_FIGURES = (
    "cartpole_boundary_survival_v2.png",
    "cartpole_boundary_recovery_v2.png",
    "cartpole_boundary_signed_asymmetry_v2.png",
    "cartpole_boundary_recovery_gap_failure_cause_v2.png",
    "cartpole_boundary_wilson_uncertainty_v2.png",
    "cartpole_boundary_physical_geometry_v2.png",
)


def test_lattice_keeps_unsampled_slots_as_holes() -> None:
    summary = _synthetic_summary()
    angles, lengths = _axes(summary)
    lookup = _boundary_lookup(summary)

    surface = _boundary_surface(
        lookup,
        angles,
        lengths,
        lambda item: item["policies"]["policy-b"]["success_rate"],
    )

    assert surface.shape == (len(angles), len(lengths))
    assert np.isfinite(surface).sum() == len(summary["conditions"])
    assert surface.size > np.isfinite(surface).sum()
    # (-10 degrees, 1.0) is deliberately absent from the synthetic sweep.
    assert np.isnan(surface[angles.index(-10.0), lengths.index(1.0)])
    assert surface[angles.index(10.0), lengths.index(1.0)] == 1.0


def test_nearest_fill_does_not_leak_into_the_displayed_array() -> None:
    surface = np.array([[0.0, np.nan, 1.0], [np.nan, np.nan, np.nan]])

    filled = _nearest_filled(surface)

    assert not np.isnan(filled).any()
    # Holes take the value of the nearest sampled slot in index space.
    assert filled[0, 1] == 0.0
    assert filled[1, 0] == 0.0
    assert filled[1, 2] == 1.0
    # The source array is untouched, so the imshow layer keeps its holes.
    assert np.isnan(surface[0, 1])
    assert np.isnan(surface[1]).all()


def test_nearest_fill_of_an_empty_lattice_is_zero() -> None:
    filled = _nearest_filled(np.full((2, 2), np.nan))

    assert (filled == 0.0).all()


def test_mirrored_pair_selection_uses_exact_counterparts() -> None:
    summary = _synthetic_summary()
    angles, lengths = _axes(summary)
    lookup = _boundary_lookup(summary)

    positive_angles = _mirrored_angles(lookup, angles, lengths)

    assert positive_angles == [5.0, 10.0]

    antisymmetric = _mirror_difference(
        lookup, positive_angles, lengths, "policy-b", "success_rate"
    )
    symmetric = _mirror_difference(
        lookup, positive_angles, lengths, "policy-a", "success_rate"
    )

    assert antisymmetric.shape == (2, len(lengths))
    # Only the four exact +/- pairs are populated: (5, 0.5), (5, 1.0),
    # (10, 0.25) and (10, 0.5). The other slots lack a counterpart.
    assert int(np.isfinite(antisymmetric).sum()) == 4
    assert np.isnan(antisymmetric[positive_angles.index(10.0), lengths.index(1.0)])
    assert np.isnan(antisymmetric[positive_angles.index(5.0), lengths.index(0.25)])
    assert np.nanmin(antisymmetric) == pytest.approx(1.0)
    assert np.nanmax(antisymmetric) == pytest.approx(1.0)
    assert np.nansum(np.abs(symmetric)) == pytest.approx(0.0)


def test_wilson_straddle_predicate() -> None:
    assert _straddles_half({"lower": 0.36, "upper": 0.64})
    assert not _straddles_half({"lower": 0.5, "upper": 0.9})
    assert not _straddles_half({"lower": 0.1, "upper": 0.5})
    assert not _straddles_half({"lower": 0.0, "upper": 0.07})


def test_cell_edges_bracket_the_sample_points() -> None:
    arithmetic = _cell_edges([-10.0, -5.0, 0.0], geometric=False)
    geometric = _cell_edges([0.25, 0.5, 1.0], geometric=True)

    assert arithmetic == pytest.approx([-12.5, -7.5, -2.5, 2.5])
    assert geometric == pytest.approx(
        [0.25 / np.sqrt(2), 0.25 * np.sqrt(2), 0.5 * np.sqrt(2), 1.0 * np.sqrt(2)]
    )
    assert _cell_edges([0.5], geometric=False) == pytest.approx([0.0, 1.0])


def test_plot_sweep_writes_the_v2_boundary_figures(tmp_path: Path) -> None:
    evaluation = _write_evaluation(tmp_path / "evaluation")
    output = tmp_path / "figures"

    paths = plot_sweep(evaluation, output)

    assert [path.name for path in paths] == list(EXPECTED_FIGURES)
    for path in paths:
        assert path.parent == output
        assert path.stat().st_size > 0
    assert sorted(item.name for item in output.glob("*.png")) == sorted(
        EXPECTED_FIGURES
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        plot_sweep(evaluation, output)


def _axes(summary: dict) -> tuple[list[float], list[float]]:
    angles = sorted(
        {item["parameters"]["initial_theta_deg"] for item in summary["conditions"]}
    )
    lengths = sorted({item["parameters"]["length"] for item in summary["conditions"]})
    return angles, lengths


def _write_evaluation(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "suite.json").write_text(
        json.dumps(
            {
                "suite": {
                    "suite_id": "tiny-boundary",
                    "environment_id": "CartPole-v1",
                    "perturbation_name": "cartpole-recovery-angle-length-v1",
                    "metric_version": "episode-summary-v4",
                }
            }
        )
    )
    analysis = root / "analysis" / "episode-summary-v4"
    analysis.mkdir(parents=True)
    (analysis / "summary.json").write_text(json.dumps(_synthetic_summary()))
    return root


def _synthetic_summary() -> dict:
    # A 5 x 3 lattice with two holes, so the tests exercise sparse sampling.
    slots = [
        (angle, length)
        for angle in (-10.0, -5.0, 0.0, 5.0, 10.0)
        for length in (0.25, 0.5, 1.0)
        if (angle, length) not in {(-10.0, 1.0), (-5.0, 0.25)}
    ]
    return {
        "environment_id": "CartPole-v1",
        "metric_version": "episode-summary-v4",
        "perturbation_name": "cartpole-recovery-angle-length-v1",
        "policy_ids": ["policy-a", "policy-b"],
        "seeds": list(range(10)),
        "conditions": [
            {
                "condition_id": f"theta-{angle}_length-{length}",
                "parameters": {
                    "initial_theta_deg": angle,
                    "length": length,
                    "theta_threshold_deg": 90.0,
                },
                "policies": {
                    "policy-a": _aggregate(0.5, 0.5),
                    # policy-b survives only at non-negative angles, which makes
                    # the +/- mirror difference exactly 1.0 where a pair exists.
                    "policy-b": _aggregate(1.0 if angle >= 0 else 0.0, 0.0),
                },
            }
            for angle, length in slots
        ],
    }


def _aggregate(success_rate: float, recovery_rate: float) -> dict:
    successes = int(round(success_rate * 10))
    return {
        "episode_count": 10,
        "success_count": successes,
        "success_rate": success_rate,
        "success_rate_wilson_95": {
            "lower": max(0.0, success_rate - 0.2),
            "upper": min(1.0, success_rate + 0.2),
        },
        "recovery_count": int(round(recovery_rate * 10)),
        "recovery_rate": recovery_rate,
        "recovery_rate_wilson_95": {
            "lower": max(0.0, recovery_rate - 0.2),
            "upper": min(1.0, recovery_rate + 0.2),
        },
        "failure_cause_counts": {
            "none": successes,
            "angle_limit": 10 - successes,
            "cart_limit": 0,
            "both": 0,
            "unknown": 0,
        },
        "episode_length": {
            "mean": 100.0 * success_rate,
            "standard_deviation": 0.0,
            "minimum": 0.0,
            "maximum": 100.0,
        },
        "episode_return": {
            "mean": 100.0 * success_rate,
            "standard_deviation": 0.0,
            "minimum": 0.0,
            "maximum": 100.0,
        },
        "environment_metrics": {},
    }
