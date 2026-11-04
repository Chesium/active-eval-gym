"""Print a compact record of the adaptive PPO study for the midterm figure.

Run on the machine holding the study; it reads saved files and writes only to stdout.
Recovery comes from held-out reference evaluations. Frontiers come from the separate
discovery evaluations that selected them. Every checkpoint is kept, so the figure's
window is a plotting choice rather than a data selection.
"""

import json
import sys
from hashlib import sha256
from pathlib import Path

STUDY = "cartpole-ppo-boundary-v5-adaptive"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main(root: Path) -> None:
    reference = {}
    for path in sorted(
        (root / "evaluations/reference").glob("*/metrics_v1/summary.json")
    ):
        cells = json.loads(path.read_text())["cells"]
        by_checkpoint: dict[str, dict] = {}
        for cell in cells:
            counts = by_checkpoint.setdefault(cell["checkpoint_id"], {})
            if cell["recovery_count"]:
                counts[cell["condition_id"]] = cell["recovery_count"]
        for checkpoint, counts in by_checkpoint.items():
            reference[checkpoint] = {
                "episodes_per_condition": cells[0]["episode_count"],
                "recovery_counts": counts,
                "source_sha256": digest(path),
            }
    targets = {}
    for path in sorted((root / "targets").glob("r*/round-*.json")):
        data = json.loads(path.read_text())
        key = f"{data['replicate_id']}/{data['round_index']}"
        ids = data["condition_ids"]
        targets[key] = {
            "training_start": data["training_start"],
            "training_stop": data["training_stop"],
            "stop_reason": data["stop_reason"],
            "accepted": [
                c for c, m in zip(ids, data["accepted_mask"], strict=True) if m
            ],
            "frontier": [
                c for c, m in zip(ids, data["frontier_mask"], strict=True) if m
            ],
            "source_sha256": digest(path),
        }
    conditions = json.loads(
        next(
            iter(sorted(root.glob("evaluations/reference/*/metrics_v1/summary.json")))
        ).read_text()
    )["cells"]
    json.dump(
        {
            "schema_version": 1,
            "study_id": STUDY,
            "status": "development pilot on uncommitted source; no matched control",
            "acceptance_threshold": 0.9,
            "reference_conditions": sorted({c["condition_id"] for c in conditions}),
            "summary_sha256": digest(root / "analysis/summary.json"),
            "integrity": json.loads(
                (root / "analysis/integrity-checks.json").read_text()
            ),
            "reference": reference,
            "targets": targets,
        },
        sys.stdout,
        indent=1,
        sort_keys=True,
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
