"""Derived metrics for saved PPO boundary-study campaigns.

Everything here reads immutable shards. It never constructs an environment or
loads a policy, so metrics can be recomputed without rerunning an experiment.
"""

import gzip
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from active_eval_gym.config import PPOBoundaryStudySpec
from active_eval_gym.metrics import compute_saved_metrics
from active_eval_gym.ppo_boundary import (
    load_protocol,
    reference_weights,
    support_conditions,
)
from active_eval_gym.ppo_boundary_evaluation import (
    CAMPAIGN_DIRECTORIES,
    COLLECTION_FILENAME,
    PLAN_FILENAME,
)
from active_eval_gym.serialization import (
    read_episode_shard,
    read_shard_header,
    to_jsonable,
    write_json_new,
)
from active_eval_gym.sweeps import _aggregate

METRICS_DIRECTORY = "metrics_v1"
EPISODE_METRICS_FILENAME = "episode-metrics.jsonl.gz"
SUMMARY_FILENAME = "summary.json"
REFERENCE_SUPPORT = "reference"


def clopper_pearson(successes: int, trials: int, pair_alpha: float) -> dict[str, float]:
    """Return the exact binomial interval at confidence `1 - pair_alpha`."""

    interval = binomtest(successes, trials).proportion_ci(
        confidence_level=1.0 - pair_alpha, method="exact"
    )
    return {"lower": float(interval.low), "upper": float(interval.high)}


def coverage_summary(
    cells: list[dict[str, Any]],
    weights: dict[str, float],
    taus: tuple[float, ...],
) -> dict[str, Any]:
    """Summarize one checkpoint's cells under fixed reference weights."""

    if {cell["condition_id"] for cell in cells} != set(weights):
        raise ValueError("Coverage requires every weighted condition exactly once.")
    weight = np.array([weights[cell["condition_id"]] for cell in cells])
    recovery = np.array([cell["recovery_rate"] for cell in cells])
    survival = np.array([cell["survival_rate"] for cell in cells])
    lower = np.array([cell["recovery_simultaneous"]["lower"] for cell in cells])
    upper = np.array([cell["recovery_simultaneous"]["upper"] for cell in cells])
    by_threshold = {
        str(tau): {
            "recovery_plug_in": float(weight[recovery >= tau].sum()),
            "recovery_simultaneous_lower": float(weight[lower >= tau].sum()),
            "recovery_simultaneous_upper": float(weight[upper >= tau].sum()),
            "survival_plug_in": float(weight[survival >= tau].sum()),
        }
        for tau in taus
    }
    return {
        "mean_recovery_probability": float(weight @ recovery),
        "mean_survival_probability": float(weight @ survival),
        "thresholds": by_threshold,
    }


def analyze_campaign(
    config_path: Path,
    *,
    study_dir: Path,
    campaign: str,
    output_dir: Path | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Recompute a completed campaign's metrics from its saved shards alone."""

    spec, provenance = load_protocol(config_path, study_dir)
    directory = study_dir / CAMPAIGN_DIRECTORIES[campaign]
    if not (directory / COLLECTION_FILENAME).is_file():
        raise ValueError(f"Campaign {campaign!r} is incomplete: {directory}.")
    plan = json.loads((directory / PLAN_FILENAME).read_text())
    if plan["protocol_config_sha256"] != provenance["config_sha256"]:
        raise ValueError("Campaign plan does not match the study protocol.")
    output = output_dir or study_dir / METRICS_DIRECTORY / campaign
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite metric analysis: {output}.")

    pair_alpha = spec.confidence.family_alpha / len(plan["jobs"])
    tasks = [
        (str(directory / job["shard"]), job, spec.metric_version, pair_alpha)
        for job in plan["jobs"]
    ]
    results = _map_jobs(tasks, workers)
    cells = [cell for cell, _ in results]

    output.mkdir(parents=True)
    with (output / EPISODE_METRICS_FILENAME).open("xb") as file:
        for _, episodes in results:
            file.write(episodes)
    summary = {
        "schema_version": 1,
        "study_id": spec.study_id,
        "campaign": campaign,
        "support": plan["support"],
        "metric_version": spec.metric_version,
        "evaluation_source": plan["source"],
        "simultaneous_family": {
            "interval": spec.confidence.interval,
            "family_alpha": spec.confidence.family_alpha,
            "checkpoint_condition_pairs": len(plan["jobs"]),
            "pair_alpha": pair_alpha,
            "outcome": "recovery",
        },
        "coverage": _campaign_coverage(spec, plan, cells),
        "cells": cells,
    }
    write_json_new(output / SUMMARY_FILENAME, summary)
    return summary


def _map_jobs(tasks: list[tuple[Any, ...]], workers: int) -> list[Any]:
    if workers == 1:
        return [_analyze_job(task) for task in tasks]
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        return list(executor.map(_analyze_job, tasks))


def _analyze_job(task: tuple[Any, ...]) -> tuple[dict[str, Any], bytes]:
    path, job, metric_version, pair_alpha = task
    shard = Path(path)
    header = job["header"]
    recorded = read_shard_header(shard)
    if any(recorded.get(name) != value for name, value in header.items()):
        raise ValueError(f"Shard does not match planned job {job['job_id']}: {path}.")

    seeds: list[int] = []
    metrics: list[dict[str, Any]] = []
    for episode in read_episode_shard(shard):
        if episode.metadata["deterministic"] is not True:
            raise ValueError(f"Stochastic evaluation episode in {path}.")
        if (
            episode.metadata["policy_artifact"]["model_sha256"]
            != header["model_sha256"]
        ):
            raise ValueError(f"Shard {path} contains another checkpoint's episode.")
        seeds.append(episode.metadata["episode_seed"])
        metrics.append(
            to_jsonable(compute_saved_metrics(episode, metric_version=metric_version))
        )
    if seeds != header["episode_seeds"]:
        raise ValueError(f"Shard {path} does not contain its planned episodes.")

    aggregate = _aggregate(metrics)
    recovery_count = aggregate["recovery_count"]
    cell = {
        "checkpoint_id": header["checkpoint_id"],
        "condition_id": header["condition_id"],
        "parameters": header["perturbation"]["parameters"],
        "episode_count": len(metrics),
        "recovery_count": recovery_count,
        "recovery_rate": aggregate["recovery_rate"],
        "recovery_wilson_95": aggregate["recovery_rate_wilson_95"],
        "recovery_simultaneous": clopper_pearson(
            recovery_count, len(metrics), pair_alpha
        ),
        "survival_count": aggregate["success_count"],
        "survival_rate": aggregate["success_rate"],
        "survival_wilson_95": aggregate["success_rate_wilson_95"],
        "shard_sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
        "aggregate": aggregate,
    }
    rows = (
        {
            "checkpoint_id": header["checkpoint_id"],
            "condition_id": header["condition_id"],
            "episode_seed": seed,
            "metrics": metric,
        }
        for seed, metric in zip(seeds, metrics, strict=True)
    )
    lines = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    return cell, gzip.compress(lines, mtime=0)


def _campaign_coverage(
    spec: PPOBoundaryStudySpec, plan: dict[str, Any], cells: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Coverage is declared only over the fixed reference weights."""

    if plan["support"] != REFERENCE_SUPPORT:
        return None
    conditions = support_conditions(spec, REFERENCE_SUPPORT)
    weights = dict(
        zip(
            (condition.condition_id for condition in conditions),
            reference_weights(spec).tolist(),
            strict=True,
        )
    )
    taus = (spec.acceptance.primary_tau, *spec.acceptance.secondary_taus)
    checkpoints = dict.fromkeys(cell["checkpoint_id"] for cell in cells)
    return {
        checkpoint: coverage_summary(
            [cell for cell in cells if cell["checkpoint_id"] == checkpoint],
            weights,
            taus,
        )
        for checkpoint in checkpoints
    }
