"""Verify an exact v4 training prefix and describe the longer frozen checkpoints."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_ppo_boundary_shift import (
    BANDS,
    INK,
    INK_SECONDARY,
    SURFACE,
    save,
    style_axis,
)

from active_eval_gym.policies.artifacts import verify_evaluation_seal
from active_eval_gym.serialization import read_transition_shard, write_json_new

ARM = "acceptance_boundary"
PREFIX = 102_400


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def verify_prefix_arrays(old: dict, extended: dict) -> None:
    """Compare every original column, including reset and schedule rows."""

    if old.keys() != extended.keys():
        raise ValueError("Training prefix columns differ.")
    for name, values in old.items():
        np.testing.assert_array_equal(
            values, extended[name][: len(values)], err_msg=f"Prefix column {name}"
        )


def check_prefix(study: Path, baseline: Path) -> dict:
    config = read(study / "protocol/resolved-config.json")
    tier = next(t for t in config["tiers"] if t["name"] == "pilot")
    if tier["arms"] != [ARM] or tier["additional_transitions"] != 1_024_000:
        raise ValueError(
            "Expected the predeclared single-arm million-transition study."
        )
    calibration = read(study / "sampling-calibration/summary.json")
    old_calibration = read(baseline / "sampling-calibration/summary.json")
    for key in ("initial_mean_lengths", "condition_ids", "settings"):
        if calibration[key] != old_calibration[key]:
            raise ValueError(f"Calibration differs from v4: {key}.")
    records = []
    for replicate in tier["replicate_ids"]:
        old_id = f"{baseline.name}-pilot-r{replicate:03d}-{ARM}"
        child_id = f"{config['study_id']}-pilot-r{replicate:03d}-{ARM}"
        old_directory = baseline / "policies/pilot" / old_id
        directory = study / "policies/pilot" / child_id
        verify_evaluation_seal(old_directory)
        verify_evaluation_seal(directory / "snapshots/t102400")
        old_summary = read(old_directory / "training-summary.json")
        summary = read(directory / "training-summary.json")
        snapshot = read(directory / "snapshots/t102400/training-summary.json")
        for key in ("final_policy_sha256", "final_optimizer_sha256"):
            if old_summary[key] != snapshot[key]:
                raise ValueError(f"Replicate {replicate}: {key} differs at 102,400.")
        for key in (
            "target_transition_shares",
            "current_reset_probabilities",
            "current_length_estimates",
            "completed_episodes",
            "revision",
        ):
            if old_summary["sampling"][key] != snapshot["sampling"][key]:
                raise ValueError(f"Replicate {replicate}: sampler {key} differs.")
        old_header, old_raw = read_transition_shard(
            baseline / old_summary["transitions_file"]["path"]
        )
        _, raw = read_transition_shard(study / summary["transitions_file"]["path"])
        verify_prefix_arrays(old_raw, raw)
        records.append(
            {
                "replicate_id": replicate,
                "policy_sha256": snapshot["final_policy_sha256"],
                "optimizer_sha256": snapshot["final_optimizer_sha256"],
                "verified_prefix_columns": len(old_raw),
                "original_raw_content_sha256": old_header["content_sha256"],
            }
        )
    return {
        "prefix_transitions": PREFIX,
        "calibration_reproduced_exactly": True,
        "policy_optimizer_and_all_raw_columns_equal": True,
        "replicates": records,
    }


def check_replay(study: Path, baseline: Path) -> dict:
    """Verify a denser replay against the original run through both endpoints."""

    config = read(study / "protocol/resolved-config.json")
    tier = next(t for t in config["tiers"] if t["name"] == "pilot")
    if tier["additional_transitions"] != 256_000 or tier["arms"] != [ARM]:
        raise ValueError("Expected the predeclared 256,000-transition focused replay.")
    new_calibration = read(study / "sampling-calibration/summary.json")
    old_calibration = read(baseline / "sampling-calibration/summary.json")
    for key in ("initial_mean_lengths", "condition_ids", "settings"):
        if new_calibration[key] != old_calibration[key]:
            raise ValueError(f"Calibration replay differs: {key}.")
    records = []
    for replicate in tier["replicate_ids"]:
        child = (
            study
            / "policies/pilot"
            / f"{config['study_id']}-pilot-r{replicate:03d}-{ARM}"
        )
        original = (
            baseline
            / "policies/pilot"
            / f"{baseline.name}-pilot-r{replicate:03d}-{ARM}"
        )
        summary = read(child / "training-summary.json")
        old_summary = read(original / "training-summary.json")
        for budget in (102_400, 256_000):
            endpoint = child if budget == 256_000 else child / "snapshots/t102400"
            old_endpoint = original / "snapshots" / f"t{budget:06d}"
            verify_evaluation_seal(endpoint)
            verify_evaluation_seal(old_endpoint)
            current = read(endpoint / "training-summary.json")
            old = read(old_endpoint / "training-summary.json")
            for key in ("final_policy_sha256", "final_optimizer_sha256"):
                if current[key] != old[key]:
                    raise ValueError(
                        f"Replay differs at {budget}, replicate {replicate}: {key}"
                    )
            for key in (
                "current_length_estimates",
                "current_reset_probabilities",
                "completed_episodes",
                "revision",
            ):
                if current["sampling"][key] != old["sampling"][key]:
                    raise ValueError(f"Replay sampler differs at {budget}: {key}")
        _, replay_raw = read_transition_shard(
            study / summary["transitions_file"]["path"]
        )
        _, original_raw = read_transition_shard(
            baseline / old_summary["transitions_file"]["path"]
        )
        verify_prefix_arrays(replay_raw, original_raw)
        records.append({"replicate_id": replicate, "verified_columns": len(replay_raw)})
    return {
        "replayed_transitions": 256_000,
        "verified_endpoints": [102_400, 256_000],
        "all_raw_prefix_columns_equal": True,
        "policy_optimizer_and_controller_match": True,
        "replicates": records,
    }


def checkpoint_statistics(
    parent: dict, current: dict, reference: dict, previous: dict
) -> dict:
    if any(values.keys() != current.keys() for values in (parent, reference, previous)):
        raise ValueError("Checkpoint reference cells differ.")
    accepted = {k for k, c in current.items() if c["recovery_rate"] >= 0.9}
    parent_set = {k for k, c in parent.items() if c["recovery_rate"] >= 0.9}
    reference_set = {k for k, c in reference.items() if c["recovery_rate"] >= 0.9}
    previous_set = {k for k, c in previous.items() if c["recovery_rate"] >= 0.9}
    union = accepted | previous_set
    nominal = next(
        c
        for c in current.values()
        if c["parameters"]["initial_theta_deg"] == 0
        and c["parameters"]["length"] == 0.5
    )
    return {
        "accepted": len(accepted),
        "coverage": len(accepted) / len(current),
        "gained_parent": len(accepted - parent_set),
        "lost_parent": len(parent_set - accepted),
        "retained_parent": len(accepted & parent_set),
        "gained_102400": len(accepted - reference_set),
        "lost_102400": len(reference_set - accepted),
        "gained_previous": len(accepted - previous_set),
        "lost_previous": len(previous_set - accepted),
        "set_turnover_previous": len(accepted ^ previous_set),
        "jaccard_previous": len(accepted & previous_set) / len(union) if union else 1.0,
        "mean_recovery": float(np.mean([c["recovery_rate"] for c in current.values()])),
        "mean_survival": float(np.mean([c["survival_rate"] for c in current.values()])),
        "nominal_recovery": nominal["recovery_rate"],
        "accepted_bands": [
            sum(select(current[k]["parameters"]["length"]) for k in accepted)
            for _, select in BANDS
        ],
    }


def summarize(study: Path) -> dict:
    config = read(study / "protocol/resolved-config.json")
    tier = next(t for t in config["tiers"] if t["name"] == "pilot")
    evaluations = defaultdict(dict)
    for campaign in ("pilot", "pilot-snapshots"):
        for cell in read(study / "metrics_v1" / campaign / "summary.json")["cells"]:
            evaluations[cell["checkpoint_id"]][cell["condition_id"]] = cell
    parent = evaluations[config["parent_policy_id"]]
    budgets = [*tier["snapshot_transitions"], tier["additional_transitions"]]
    rows = []
    for replicate in tier["replicate_ids"]:
        child_id = f"{config['study_id']}-pilot-r{replicate:03d}-{ARM}"
        reference = evaluations[f"{child_id}-t102400"]
        previous = parent
        for budget in budgets:
            checkpoint = (
                child_id if budget == budgets[-1] else f"{child_id}-t{budget:06d}"
            )
            current = evaluations[checkpoint]
            rows.append(
                {
                    "replicate_id": replicate,
                    "transitions": budget,
                    **checkpoint_statistics(parent, current, reference, previous),
                }
            )
            previous = current
    metrics = (
        "accepted",
        "coverage",
        "gained_parent",
        "lost_parent",
        "retained_parent",
        "gained_102400",
        "lost_102400",
        "mean_recovery",
        "mean_survival",
        "set_turnover_previous",
        "jaccard_previous",
    )
    aggregate = [
        {
            "transitions": budget,
            **{
                key: float(
                    np.mean([r[key] for r in rows if r["transitions"] == budget])
                )
                for key in metrics
            },
        }
        for budget in budgets
    ]
    late = []
    for replicate in tier["replicate_ids"]:
        selected = [
            r
            for r in rows
            if r["replicate_id"] == replicate and r["transitions"] >= budgets[-3]
        ]
        late.append(
            {
                "replicate_id": replicate,
                "budgets": [r["transitions"] for r in selected],
                "accepted_counts": [r["accepted"] for r in selected],
                "accepted_range": int(np.ptp([r["accepted"] for r in selected])),
                "mean_recovery": [r["mean_recovery"] for r in selected],
                "mean_recovery_range": float(
                    np.ptp([r["mean_recovery"] for r in selected])
                ),
                "successive_set_turnover": [
                    r["set_turnover_previous"] for r in selected[1:]
                ],
                "successive_jaccard": [r["jaccard_previous"] for r in selected[1:]],
            }
        )
    return {
        "study_id": config["study_id"],
        "rows": rows,
        "aggregate": aggregate,
        "parent": checkpoint_statistics(parent, parent, parent, parent),
        "late_window": late,
        "interpretation": (
            "Finite-horizon development run; no best-checkpoint selection "
            "or asymptotic convergence claim."
        ),
    }


def learning_figure(result: dict, study: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 8.5))
    figure.patch.set_facecolor(SURFACE)
    colors = ("#2a78d6", "#eb6834", "#1baf7a")
    for axis in axes.flat:
        style_axis(axis)
        axis.set_xticks(
            np.linspace(0, max(r["transitions"] for r in result["rows"]) / 1000, 5)
        )
        axis.set_xlabel("New training transitions (thousands)", fontsize=9)
    rows = result["rows"]
    replicas = sorted({r["replicate_id"] for r in rows})
    for replicate, color in zip(replicas, colors, strict=True):
        selected = [r for r in rows if r["replicate_id"] == replicate]
        x = [0, *[r["transitions"] / 1000 for r in selected]]
        for axis, key in ((axes[0, 0], "accepted"), (axes[0, 1], "mean_recovery")):
            axis.plot(
                x,
                [result["parent"][key], *[r[key] for r in selected]],
                color=color,
                linewidth=1.4,
                marker="o",
                markersize=3,
                label=f"Replicate {replicate}",
            )
        axes[1, 2].plot(
            x[1:],
            [r["jaccard_previous"] for r in selected],
            color=color,
            linewidth=1.4,
            marker="o",
            markersize=3,
        )
    aggregate = result["aggregate"]
    x = [0, *[r["transitions"] / 1000 for r in aggregate]]
    for axis, key in ((axes[0, 0], "accepted"), (axes[0, 1], "mean_recovery")):
        axis.plot(
            x,
            [result["parent"][key], *[r[key] for r in aggregate]],
            color=INK,
            linewidth=2.2,
            label="Mean",
        )
        axis.axhline(
            result["parent"][key], color="#99958a", linewidth=0.8, linestyle=":"
        )
        axis.legend(frameon=False, fontsize=8)
    for key, label, color in (
        ("gained_parent", "Gained", "#1baf7a"),
        ("lost_parent", "Lost", "#c74737"),
    ):
        axes[0, 2].plot(
            x,
            [0, *[r[key] for r in aggregate]],
            color=color,
            linewidth=2,
            marker="o",
            markersize=3,
            label=label,
        )
    axes[0, 2].legend(frameon=False, fontsize=8)
    for index, ((label, _), color) in enumerate(zip(BANDS, colors, strict=True)):
        values = [result["parent"]["accepted_bands"][index]]
        for budget in [r["transitions"] for r in aggregate]:
            values.append(
                np.mean(
                    [
                        r["accepted_bands"][index]
                        for r in rows
                        if r["transitions"] == budget
                    ]
                )
            )
        axes[1, 0].plot(
            x, values, color=color, linewidth=2, marker="o", markersize=3, label=label
        )
    axes[1, 0].legend(frameon=False, fontsize=8)
    diagnostic = read(study / "sampling-diagnosis/summary.json")
    intervals = diagnostic["arms"][ARM]["intervals"]
    for key, label, color in zip(
        ("lower", "upper", "side"), ("Lower", "Upper", "Sides"), colors, strict=True
    ):
        axes[1, 1].plot(
            [(r["start"] + r["stop"]) / 2000 for r in intervals],
            [r["transition_shares"][key] * 100 for r in intervals],
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=3,
            label=label,
        )
    target = diagnostic["arms"][ARM]["groups"]["lower"]["target_transition_share"]
    axes[1, 1].axhline(target * 100, color=INK, linestyle=":", linewidth=1)
    axes[1, 1].set_xlabel("Training interval midpoint (thousands)", fontsize=9)
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 2].set_ylim(0, 1.05)
    late_rows = [
        r for r in rows if r["transitions"] >= result["late_window"][0]["budgets"][0]
    ]
    if late_rows and all(r["accepted"] == 0 for r in late_rows):
        axes[1, 2].text(
            0.3,
            0.82,
            "Later sets are empty (similarity = 1)",
            transform=axes[1, 2].transAxes,
            fontsize=8,
            color=INK_SECONDARY,
        )
    titles = [
        "Accepted conditions (out of 273)",
        "Mean recovery probability",
        "Gains and losses relative to parent",
        "Acceptance by pole half-length",
        "Frontier transition shares (%)",
        "Set similarity to preceding checkpoint",
    ]
    for axis, title in zip(axes.flat, titles, strict=True):
        axis.set_title(title, loc="left", fontsize=10, color=INK)
    figure.suptitle(
        "Duration-corrected acceptance-boundary training to "
        f"{max(r['transitions'] for r in rows):,} transitions",
        x=0.06,
        y=0.98,
        ha="left",
        fontsize=15,
        color=INK,
    )
    figure.text(
        0.06,
        0.93,
        "Three fixed pilot seeds; 64 deterministic evaluation episodes per condition. "
        "Every checkpoint is retained. Late similarity measures set turnover, "
        "not proof of convergence.",
        fontsize=9,
        color=INK_SECONDARY,
        ha="left",
        va="top",
    )
    figure.subplots_adjust(
        left=0.06, right=0.985, top=0.84, bottom=0.08, hspace=0.43, wspace=0.24
    )
    save(figure, study / "figures/pilot/long-run-learning-curves")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--check-prefix", action="store_true")
    parser.add_argument("--check-replay", action="store_true")
    args = parser.parse_args()
    output = args.study / "long-run"
    output.mkdir(exist_ok=True)
    if args.check_replay:
        result = check_replay(args.study, args.baseline)
        write_json_new(output / "replay-verification.json", result)
        print("All three focused replays match the original through 256,000 exactly.")
    elif args.check_prefix:
        result = check_prefix(args.study, args.baseline)
        write_json_new(output / "prefix-verification.json", result)
        print("All three 102,400-transition prefixes match v4 exactly.")
    else:
        read(
            output
            / (
                "replay-verification.json"
                if (output / "replay-verification.json").exists()
                else "prefix-verification.json"
            )
        )
        result = summarize(args.study)
        learning_figure(result, args.study)
        write_json_new(output / "summary.json", result)
        for row in result["aggregate"]:
            print(row)


if __name__ == "__main__":
    main()
