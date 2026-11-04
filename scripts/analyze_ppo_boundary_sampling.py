"""Audit fixed PPO selection versus realized exposure using saved data only.

Writes a descriptive JSON diagnostic; does not train or evaluate policies.
Frontier sides refer to the frozen parent's discovery grid, not a refreshed set.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from active_eval_gym.ppo_boundary_targets import construct_arms
from active_eval_gym.ppo_transition_sampling import InverseLengthSampler
from active_eval_gym.serialization import read_transition_shard


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def diagnose(study: Path) -> dict:
    distributions = read_json(study / "distributions/distributions.json")
    config = read_json(study / "protocol/resolved-config.json")
    tier = next(t for t in config["tiers"] if t["name"] == "pilot")
    expected_arms = tier.get("arms") or config["curriculum"]["arms"]
    identifiers = distributions["condition_ids"]
    record = read_json(study / "protocol/conditions.json")
    conditions = {
        c["condition_id"]: (
            c["perturbation"]["parameters"]["initial_theta_deg"],
            c["perturbation"]["parameters"]["length"],
        )
        for c in record["supports"]["discovery"]["conditions"]
    }
    points = [conditions[i] for i in identifiers]
    angles = sorted({p[0] for p in points})
    lengths = sorted({p[1] for p in points})
    assert points == [(a, length) for a in angles for length in lengths]
    selection = distributions["selection"]
    episodes = selection["episodes_per_condition"]
    tau = selection["primary_tau"]
    counts = np.array(distributions["recovery_counts"])
    accepted = counts / episodes >= tau
    arms = construct_arms(
        counts,
        episodes,
        tau,
        SimpleNamespace(**selection["curriculum"]),
        grid_shape=(len(angles), len(lengths)),
    )
    for arm, built in arms.items():
        np.testing.assert_array_equal(
            built.probabilities, distributions["arms"][arm]["probabilities"]
        )
        np.testing.assert_array_equal(
            built.target_weights, distributions["arms"][arm]["target_weights"]
        )
    frontier = arms["acceptance_boundary"].target_weights > 0
    accepted_points = {p for p, yes in zip(points, accepted, strict=True) if yes}
    groups = {name: np.zeros(len(points), bool) for name in ("lower", "upper", "side")}
    independently_selected = np.zeros(len(points), bool)
    for index, (angle, length) in enumerate(points):
        if accepted[index]:
            continue
        a, length_rank = angles.index(angle), lengths.index(length)
        below = length_rank > 0 and (angle, lengths[length_rank - 1]) in accepted_points
        above = (
            length_rank + 1 < len(lengths)
            and (angle, lengths[length_rank + 1]) in accepted_points
        )
        side = any(
            0 <= other < len(angles) and (angles[other], length) in accepted_points
            for other in (a - 1, a + 1)
        )
        independently_selected[index] = below or above or side
        if independently_selected[index]:
            # Assign corners to their vertical side before remaining angle sides.
            groups["lower" if above else "upper" if below else "side"][index] = True
    np.testing.assert_array_equal(frontier, independently_selected)
    groups |= {
        "parent_accepted": accepted,
        "short_le_0.3": np.array([p[1] <= 0.3 for p in points]),
        "long_gt_0.5": np.array([p[1] > 0.5 for p in points]),
    }
    discovery = {
        c["condition_id"]: c
        for c in read_json(study / "metrics_v1/discovery/summary.json")["cells"]
    }
    rows = []
    distribution_hashes = set()
    transition_target = distributions.get("sampling_unit") == "transition"
    calibration = None
    if transition_target:
        calibration = read_json(study / "sampling-calibration/summary.json")
    for path in sorted((study / "policies/pilot").glob("*/training-summary.json")):
        summary = read_json(path)
        raw_path = study / summary["transitions_file"]["path"]
        header, raw = read_transition_shard(raw_path)
        assert header["condition_ids"] == identifiers
        assert header["arm"] == summary["arm"]
        assert header["replicate_id"] == summary["replicate_id"]
        condition = raw["condition_index"]
        assert len(condition) == summary["additional_transitions"]
        checkpoints = [
            *(
                read_json(p)
                for p in path.parent.glob("snapshots/*/training-summary.json")
            ),
            summary,
        ]
        checkpoints.sort(key=lambda s: s["additional_transitions"])
        intervals = []
        start = 0
        previous_resets = np.zeros(len(points), dtype=np.int64)
        for checkpoint in checkpoints:
            stop = checkpoint["additional_transitions"]
            distribution_hashes.add(checkpoint["distribution"]["sha256"])
            realized = checkpoint["realized_sampling"]
            assert realized["condition_ids"] == identifiers
            np.testing.assert_array_equal(
                np.bincount(condition[:stop], minlength=len(points)),
                realized["transitions"],
            )
            resets = np.array(realized["episodes_started"])
            interval_resets = resets - previous_resets
            assert np.all(interval_resets >= 0)
            interval_counts = np.bincount(condition[start:stop], minlength=len(points))
            intervals.append(
                {
                    "start": start,
                    "stop": stop,
                    "transition_shares": {
                        k: float(interval_counts[mask].sum() / (stop - start))
                        for k, mask in groups.items()
                    },
                }
            )
            start, previous_resets = stop, resets
        np.testing.assert_array_equal(
            np.bincount(raw["reset_condition_index"], minlength=len(points)),
            previous_resets,
        )
        _, first, inverse, duration = np.unique(
            np.column_stack((raw["worker"], raw["episode_index"])),
            axis=0,
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )
        np.testing.assert_array_equal(condition, condition[first][inverse])
        finished = raw["terminated"] | raw["truncated"]
        complete = np.zeros(len(first), bool)
        complete[inverse[finished]] = True
        survived = np.zeros(len(first), bool)
        survived[inverse[raw["truncated"]]] = True
        completed_by_condition = np.bincount(
            condition[first][complete], minlength=len(points)
        )
        np.testing.assert_array_equal(
            completed_by_condition, summary["realized_sampling"]["episodes_completed"]
        )
        if transition_target:
            from active_eval_gym.config import TransitionSamplingSpec

            settings = TransitionSamplingSpec(**summary["sampling"]["settings"])
            replay = InverseLengthSampler(
                np.array(distributions["arms"][summary["arm"]]["probabilities"]),
                np.array(calibration["initial_mean_lengths"]),
                settings,
            )
            terminal_rows = np.flatnonzero(finished)
            cursor = 0
            for revision, boundary in enumerate(raw["sampling_transition_count"]):
                while cursor < len(terminal_rows) and terminal_rows[cursor] < boundary:
                    terminal = terminal_rows[cursor]
                    replay.observe_completion(
                        int(condition[terminal]), int(duration[inverse[terminal]])
                    )
                    cursor += 1
                if revision:
                    replay.refresh()
                np.testing.assert_array_equal(
                    replay.probabilities, raw["sampling_probabilities"][revision]
                )
                np.testing.assert_array_equal(
                    replay.length_estimates, raw["sampling_length_estimates"][revision]
                )
                np.testing.assert_array_equal(
                    replay.completed_episodes,
                    raw["sampling_completed_episodes"][revision],
                )
            np.testing.assert_array_equal(
                raw["reset_selection_probability"],
                raw["sampling_probabilities"][
                    raw["reset_sampling_revision"], raw["reset_condition_index"]
                ],
            )
        rows.append(
            {
                "arm": summary["arm"],
                "replicate_id": summary["replicate_id"],
                "intervals": intervals,
                "groups": {
                    k: {
                        "episodes_started": int(previous_resets[mask].sum()),
                        "transitions": int(mask[condition].sum()),
                        "completed_episode_lengths": duration[
                            mask[condition[first]] & complete
                        ].tolist(),
                        "completed_survivals": int(
                            (mask[condition[first]] & complete & survived).sum()
                        ),
                    }
                    for k, mask in groups.items()
                },
                "episodes_started": int(previous_resets.sum()),
                "transitions": len(condition),
                "transition_target_total_variation": float(
                    0.5
                    * np.abs(
                        np.array(summary["realized_sampling"]["transition_shares"])
                        - np.array(
                            distributions["arms"][summary["arm"]]["probabilities"]
                        )
                    ).sum()
                ),
            }
        )
    assert rows
    assert {(r["arm"], r["replicate_id"]) for r in rows} == {
        (arm, replicate) for arm in expected_arms for replicate in tier["replicate_ids"]
    }
    assert len(distribution_hashes) == 1
    assert distribution_hashes == {
        hashlib.sha256(
            (study / "distributions/distributions.json").read_bytes()
        ).hexdigest()
    }
    aggregate = {}
    for arm, built in arms.items():
        if arm not in expected_arms:
            continue
        members = [r for r in rows if r["arm"] == arm]
        assert members
        stats = {}
        for name, mask in groups.items():
            pooled = [
                n
                for r in members
                for n in r["groups"][name]["completed_episode_lengths"]
            ]
            stats[name] = {
                "target_transition_share" if transition_target else "q": float(
                    built.probabilities[mask].sum()
                ),
                "episode_share": sum(
                    r["groups"][name]["episodes_started"] for r in members
                )
                / sum(r["episodes_started"] for r in members),
                "transition_share": sum(
                    r["groups"][name]["transitions"] for r in members
                )
                / sum(r["transitions"] for r in members),
                "completed_episodes": len(pooled),
                "mean_completed_length": float(np.mean(pooled)) if pooled else None,
                "completed_survival_rate": sum(
                    r["groups"][name]["completed_survivals"] for r in members
                )
                / len(pooled)
                if pooled
                else None,
            }
        intervals = defaultdict(list)
        for row in members:
            for interval in row["intervals"]:
                intervals[(interval["start"], interval["stop"])].append(interval)
        aggregate[arm] = {
            "groups": stats,
            "mean_transition_target_total_variation": float(
                np.mean([r["transition_target_total_variation"] for r in members])
            ),
            "intervals": [
                {
                    "start": start,
                    "stop": stop,
                    "transition_shares": {
                        name: float(
                            np.mean([v["transition_shares"][name] for v in values])
                        )
                        for name in groups
                    },
                }
                for (start, stop), values in sorted(intervals.items())
            ],
        }
    return {
        "study": str(study),
        "interpretation": "Descriptive pilot diagnostic; not confirmatory evidence.",
        "sampling_unit": "transition" if transition_target else "episode",
        "checks": {
            "serialized_q_reconstructed_exactly": True,
            "frontier_reconstructed_independently": True,
            "raw_content_digests_verified": len(rows),
            "cumulative_counts_match_raw_at_every_checkpoint": True,
            "fixed_distribution_hash": next(iter(distribution_hashes)),
            "adaptive_schedule_replayed_exactly": transition_target,
        },
        "frontier_groups": {
            name: [
                {
                    "condition_id": identifiers[i],
                    "parent_recoveries": int(counts[i]),
                    "parent_survivals": discovery[identifiers[i]]["survival_count"],
                    "parent_mean_episode_length": discovery[identifiers[i]][
                        "aggregate"
                    ]["episode_length"]["mean"],
                }
                for i in np.flatnonzero(groups[name])
            ]
            for name in ("lower", "upper", "side")
        },
        "arms": aggregate,
        "replicates": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.study / "sampling-diagnosis/summary.json"
    result = diagnose(args.study)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote sampling diagnostic to {output}.")


if __name__ == "__main__":
    main()
