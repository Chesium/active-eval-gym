"""Check that training exposure uses intervals rather than cumulative shares."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

training_windows = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/plot_ppo_boundary_shift.py")
)["training_windows"]


def write_summary(path: Path, budget: int, counts: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "arm": "background",
                "replicate_id": 0,
                "additional_transitions": budget,
                "realized_sampling": {
                    "condition_ids": ["short", "long"],
                    "transitions": counts,
                },
            }
        )
    )


def test_exposure_uses_counts_between_checkpoints(tmp_path: Path) -> None:
    child = tmp_path / "policies/pilot/child"
    write_summary(child / "snapshots/t000010/training-summary.json", 10, [8, 2])
    write_summary(child / "training-summary.json", 20, [9, 11])

    windows = training_windows(tmp_path, "pilot", [10, 20], ["short", "long"])

    np.testing.assert_allclose(windows[("background", 0, 10)], [0.8, 0.2])
    np.testing.assert_allclose(windows[("background", 0, 20)], [0.1, 0.9])


@pytest.mark.parametrize("counts", [[7, 13], [10, 9]])
def test_inconsistent_counts_are_refused(tmp_path: Path, counts: list[int]) -> None:
    child = tmp_path / "policies/pilot/child"
    write_summary(child / "snapshots/t000010/training-summary.json", 10, [8, 2])
    write_summary(child / "training-summary.json", 20, counts)

    with pytest.raises(ValueError, match="Invalid training counts"):
        training_windows(tmp_path, "pilot", [10, 20], ["short", "long"])


def test_missing_snapshot_is_not_replaced_with_cumulative_exposure(tmp_path: Path):
    child = tmp_path / "policies/pilot/child"
    write_summary(child / "training-summary.json", 20, [9, 11])

    with pytest.raises(ValueError, match="Missing training summary at 10"):
        training_windows(tmp_path, "pilot", [10, 20], ["short", "long"])
