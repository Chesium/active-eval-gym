"""Explicit adaptive discovery/retarget/train rounds, separate from reference data.

Each seed keeps one live model, optimizer, training environment, RNG stream, and
length controller. Only a sealed checkpoint is evaluated in spawned processes.
Interrupted seeds replay deterministically from the parent and verify completed
rounds before continuing; reference data are collected only after training ends.
"""

import hashlib
import json
import multiprocessing
import random
import tarfile
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from active_eval_gym.config import PPOBoundaryStudySpec, load_ppo_boundary_study
from active_eval_gym.policies.artifacts import (
    load_policy_artifact,
    verify_evaluation_seal,
    write_continued_artifact,
    write_evaluation_seal,
)
from active_eval_gym.ppo_boundary import (
    RECORDED_PACKAGES,
    REPOSITORY_ROOT,
    evaluation_episode_seeds,
    source_hashes,
    source_state,
    support_conditions,
    training_policy_seed,
    verify_parent,
)
from active_eval_gym.ppo_boundary_analysis import _map_jobs
from active_eval_gym.ppo_boundary_evaluation import (
    EvaluationJob,
    _shard_complete,
    run_jobs,
    write_or_match_json,
)
from active_eval_gym.ppo_boundary_targets import construct_arms, frontier_mask
from active_eval_gym.ppo_boundary_training import (
    OptimizerStepRecorder,
    TransitionCapture,
    make_training_environment,
    migrate_parent,
    resolved_settings,
    state_sha256,
)
from active_eval_gym.ppo_transition_sampling import (
    InverseLengthSampler,
    load_calibration,
)
from active_eval_gym.serialization import (
    read_transition_shard,
    to_jsonable,
    write_json_new,
    write_transition_shard,
)


@dataclass(frozen=True)
class AdaptiveStudySpec:
    study_id: str
    base_config: Path
    baseline_study: Path
    historical_summary: Path
    rounds: int
    transitions_per_round: int
    replicate_ids: tuple[int, ...]
    empty_frontier_action: str
    discovery_workers_per_seed: int
    reference_workers: int

    @property
    def total_transitions(self) -> int:
        return self.rounds * self.transitions_per_round


def load_adaptive_study(path: Path) -> tuple[AdaptiveStudySpec, PPOBoundaryStudySpec]:
    data = tomllib.loads(path.read_text())
    if data.pop("schema_version", None) != 1:
        raise ValueError("Unsupported adaptive study schema.")
    expected = set(AdaptiveStudySpec.__dataclass_fields__)
    if set(data) != expected:
        raise ValueError(
            f"Adaptive configuration fields differ: {set(data) ^ expected}"
        )
    for key in ("base_config", "baseline_study", "historical_summary"):
        data[key] = (path.resolve().parent / data[key]).resolve()
    data["replicate_ids"] = tuple(data["replicate_ids"])
    settings = AdaptiveStudySpec(**data)
    base = load_ppo_boundary_study(settings.base_config)
    if not settings.study_id or Path(settings.study_id).name != settings.study_id:
        raise ValueError("Adaptive study_id must be a simple directory name.")
    if not 1 <= settings.rounds < 100:
        raise ValueError("Adaptive rounds must be in [1, 99].")
    rollout_size = (
        base.continuation.hyperparameters["n_envs"]
        * base.continuation.hyperparameters["n_steps"]
    )
    if base.sampling is None or settings.transitions_per_round <= 0:
        raise ValueError(
            "Adaptive rounds require duration correction and a positive budget."
        )
    if settings.transitions_per_round % (rollout_size * base.sampling.refresh_rollouts):
        raise ValueError(
            "Rounds must align with scheduled duration-controller refreshes."
        )
    if settings.empty_frontier_action != "stop-seed":
        raise ValueError("This study declares stop-seed for an empty frontier.")
    if (
        not settings.replicate_ids
        or len(set(settings.replicate_ids)) != len(settings.replicate_ids)
        or any(type(r) is not int or not 0 <= r < 1000 for r in settings.replicate_ids)
    ):
        raise ValueError("Invalid adaptive replicate IDs.")
    if settings.discovery_workers_per_seed < 2 or settings.reference_workers < 2:
        raise ValueError("Adaptive evaluations require separate worker processes.")
    if base.curriculum.target_rule != "ppo-boundary-targets-v2":
        raise ValueError(
            "Adaptive curriculum requires the existing grid frontier rule."
        )
    return settings, base


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tobytes(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
    }


def assert_rng_unchanged(before: dict) -> None:
    after = rng_state()
    if state_sha256(before) != state_sha256(after):
        raise RuntimeError("Frozen evaluation changed the live training RNG.")


class AdaptiveCapture(TransitionCapture):
    """Add the changing transition targets to the existing reset-law schedule."""

    def _on_rollout_start(self) -> None:
        count = len(self.sampling_history)
        super()._on_rollout_start()
        if len(self.sampling_history) != count:
            self.sampling_history[-1]["target"] = self.sampling.target_shares.copy()

    def tables(self) -> dict[str, np.ndarray]:
        return {
            **super().tables(),
            "sampling_target_transition_shares": np.array(
                [row["target"] for row in self.sampling_history]
            ),
        }


def evaluation_jobs(settings, base, study, checkpoint, *, support, round_index):
    if checkpoint.name == base.parent_policy_id:
        _, metadata = load_policy_artifact(checkpoint, require_frozen=True)
        sealed = False
    else:
        seal = verify_evaluation_seal(checkpoint)
        metadata = None
        sealed = True
    checkpoint_id = metadata.policy_id if metadata else seal["policy_id"]
    model_hash = metadata.model_sha256 if metadata else seal["model_sha256"]
    role = base.seeds.roles["discovery" if support == "discovery" else "pilot"]
    episodes = (
        base.curriculum.discovery_episodes_per_condition
        if support == "discovery"
        else base.tier("pilot").episodes_per_condition
    )
    # New nuisance draws at each targeting round; the reference list stays paired
    # with the historical campaign and is never exposed to the targeting code.
    seed_round = round_index if support == "discovery" else 0
    seeds = list(evaluation_episode_seeds(role, episodes, outer_round=seed_round))
    directory = study / "evaluations" / support / checkpoint_id
    jobs = [
        EvaluationJob(
            job_id=f"{checkpoint_id}/{condition.condition_id}",
            artifact_dir=str(checkpoint),
            shard_path=str(directory / "shards" / f"{condition.condition_id}.jsonl.gz"),
            compression_level=base.storage.compression_level,
            header={
                "study_id": settings.study_id,
                "campaign": f"adaptive-{support}",
                "checkpoint_id": checkpoint_id,
                "model_sha256": model_hash,
                "condition_id": condition.condition_id,
                "perturbation": to_jsonable(condition.perturbation),
                "seed_role": role,
                "seed_round": seed_round,
                "episode_seeds": seeds,
                "deterministic": True,
            },
            sealed=sealed,
        )
        for condition in support_conditions(base, support)
    ]
    return directory, jobs


def collect_and_analyze(
    settings, base, study, checkpoint, *, support, round_index, workers
):
    directory, jobs = evaluation_jobs(
        settings, base, study, checkpoint, support=support, round_index=round_index
    )
    write_or_match_json(
        directory / "evaluation-plan.json",
        {
            "study_id": settings.study_id,
            "support": support,
            "round_index": round_index,
            "metric_version": base.metric_version,
            "jobs": [to_jsonable(job) for job in jobs],
        },
    )
    pending = [job for job in jobs if not _shard_complete(job)]
    failures = run_jobs(pending, directory, base.continuation.torch_threads, workers)
    if failures:
        raise RuntimeError(f"Incomplete adaptive {support} evaluation: {failures}")
    # Reference intervals are descriptive diagnostics, not a confirmatory lock.
    summary_path = directory / "metrics_v1/summary.json"
    if not summary_path.exists():
        family_pairs = len(support_conditions(base, support)) * (
            1 + len(settings.replicate_ids) * settings.rounds
        )
        tasks = [
            (
                job.shard_path,
                {"job_id": job.job_id, "header": job.header},
                base.metric_version,
                base.confidence.family_alpha / family_pairs,
            )
            for job in jobs
        ]
        results = _map_jobs(tasks, workers)
        cells = [cell for cell, _ in results]
        summary_path.parent.mkdir(parents=True)
        with (summary_path.parent / "episode-metrics.jsonl.gz").open("xb") as file:
            for _, rows in results:
                file.write(rows)
        write_json_new(
            summary_path,
            {
                "study_id": settings.study_id,
                "support": support,
                "round_index": round_index,
                "metric_version": base.metric_version,
                "family_pairs": family_pairs,
                "cells": cells,
            },
        )
    else:
        # Detect modified raw data even when derived metrics already exist.
        summary = read(summary_path)
        if len(summary["cells"]) != len(jobs):
            raise ValueError("Incomplete saved adaptive metrics.")
        for job, cell in zip(jobs, summary["cells"], strict=True):
            if cell["condition_id"] != job.header["condition_id"] or cell[
                "shard_sha256"
            ] != sha256(Path(job.shard_path)):
                raise ValueError("Saved adaptive metric/raw identity differs.")
    return read(summary_path), summary_path


def select_frontier(base, discovery):
    expected = tuple(c.condition_id for c in support_conditions(base, "discovery"))
    if (
        discovery["support"] != "discovery"
        or tuple(c["condition_id"] for c in discovery["cells"]) != expected
    ):
        raise ValueError(
            "Adaptive targets require the complete ordered discovery support."
        )
    episodes = base.curriculum.discovery_episodes_per_condition
    if any(c["episode_count"] != episodes for c in discovery["cells"]):
        raise ValueError("Adaptive targeting episode count differs.")
    counts = np.array([c["recovery_count"] for c in discovery["cells"]])
    shape = (
        len(base.supports["discovery"].initial_theta_deg),
        len(base.supports["discovery"].length),
    )
    accepted = counts / episodes >= base.acceptance.primary_tau
    mask = frontier_mask(accepted, shape)
    arm = construct_arms(
        counts, episodes, base.acceptance.primary_tau, base.curriculum, grid_shape=shape
    )["acceptance_boundary"]
    reason = None
    if not mask.any():
        reason = (
            "no-accepted-conditions"
            if not accepted.any()
            else "all-conditions-accepted"
        )
    return {
        "condition_ids": expected,
        "recovery_counts": counts,
        "accepted_mask": accepted,
        "frontier_mask": mask,
        "target_transition_shares": arm.probabilities,
        "stop_reason": reason,
    }


def prepare_adaptive(config: Path, study: Path, artifact_root: Path) -> dict:
    settings, base = load_adaptive_study(config)
    verify_parent(base, artifact_root)
    baseline = settings.baseline_study
    old_provenance = read(baseline / "protocol/provenance.json")
    calibration, calibration_hash = load_calibration(base, old_provenance, baseline)
    if calibration["parent_model_sha256"] != base.parent_model_sha256:
        raise ValueError("Calibration parent differs.")
    hashes = source_hashes()
    protocol = {
        "schema_version": 1,
        "development_record": True,
        "settings": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(settings).items()
        },
        "base_spec": base,
        "config_sha256": sha256(config),
        "base_config_sha256": sha256(settings.base_config),
        "source": source_state(),
        "source_sha256": hashes,
        "runtime_packages": {name: version(name) for name in RECORDED_PACKAGES},
        "initial_calibration": {
            "summary_path": str(baseline / "sampling-calibration/summary.json"),
            "sha256": calibration_hash,
        },
        "parent_artifact_dir": str((artifact_root / base.parent_policy_id).resolve()),
        "training_state": (
            "one live model, optimizer, environment, RNG and EMA across rounds"
        ),
        "reference_access": (
            "only after all seed training jobs finish; never used for targeting"
        ),
        "comparison": "historical fixed-frontier long run; no new matched control",
    }
    write_or_match_json(study / "protocol/adaptive-protocol.json", protocol)
    archive = study / "protocol/source.tar.gz"
    if not archive.exists():
        with tarfile.open(archive, "w:gz") as file:
            for name in sorted(hashes):
                file.add(REPOSITORY_ROOT / name, arcname=name)
            for path in sorted((REPOSITORY_ROOT / "tests").glob("test_ppo*py")):
                file.add(path, arcname=str(path.relative_to(REPOSITORY_ROOT)))
        write_json_new(
            study / "protocol/source-archive.json",
            {"archive_sha256": sha256(archive), "source_sha256": hashes},
        )
    return to_jsonable(protocol)


def save_checkpoint(
    settings,
    base,
    study,
    replicate,
    round_index,
    model,
    capture,
    sampling,
    target_path,
    initial_timesteps,
    settings_record,
):
    budget = round_index * settings.transitions_per_round
    policy_id = f"{settings.study_id}-r{replicate:03d}-t{budget:06d}"
    destination = study / "policies" / policy_id
    tables = capture.tables()
    if destination.exists():
        verify_evaluation_seal(destination)
        old = read(destination / "training-summary.json")
        if old["final_policy_sha256"] != state_sha256(model.policy.state_dict()) or old[
            "final_optimizer_sha256"
        ] != state_sha256(model.policy.optimizer.state_dict()):
            raise ValueError("Replayed adaptive checkpoint differs.")
        if old["sampling"] != to_jsonable(sampler_state(sampling)):
            raise ValueError("Replayed adaptive duration state differs.")
        _, saved = read_transition_shard(destination / "training-transitions.npz")
        if saved.keys() != tables.keys():
            raise ValueError("Adaptive replay raw columns differ.")
        for key in tables:
            np.testing.assert_array_equal(
                saved[key], tables[key], err_msg=f"Adaptive replay: {key}"
            )
        return destination
    work = study / ".work" / policy_id
    if work.exists():
        abandoned = study / "abandoned" / f"{policy_id}-{time.time_ns()}"
        abandoned.parent.mkdir(exist_ok=True)
        work.rename(abandoned)
    work.mkdir(parents=True)
    model.save(work / "model.zip")
    raw_digests = write_transition_shard(
        work / "training-transitions.npz",
        {
            "study_id": settings.study_id,
            "policy_id": policy_id,
            "replicate_id": replicate,
            "completed_rounds": round_index,
            "condition_ids": support_conditions_ids(base),
            "training_reward": base.training_reward,
            "policy_seed": training_policy_seed(base, replicate_id=replicate),
            "round_transition_ranges": [
                [
                    r * settings.transitions_per_round,
                    (r + 1) * settings.transitions_per_round,
                ]
                for r in range(round_index)
            ],
        },
        tables,
    )
    design = replace(
        base.parent_design,
        design_id=f"{settings.study_id}-adaptive-design",
        policy_id=policy_id,
        training_seed=training_policy_seed(base, replicate_id=replicate),
        training_steps=budget,
        hyperparameters={
            **base.continuation.hyperparameters,
            "learning_rate": base.continuation.learning_rate,
            "clip_range": base.continuation.clip_range,
            "continues_policy_id": base.parent_policy_id,
            "continues_model_sha256": base.parent_model_sha256,
        },
    )
    write_continued_artifact(
        design,
        work / "model.zip",
        {
            "training_kind": "adaptive-frontier-continuation-v1",
            "study_id": settings.study_id,
            "replicate_id": replicate,
            "completed_rounds": round_index,
            "additional_transitions": budget,
            "parent_model_sha256": base.parent_model_sha256,
            "target_record": str(target_path.relative_to(study)),
            "target_sha256": sha256(target_path),
            "training_task": {
                "environment_id": base.environment_id,
                "perturbation_name": base.perturbation_name,
                "training_reward": base.training_reward,
                "theta_threshold_deg": base.theta_threshold_deg,
            },
            "resolved_ppo": settings_record,
            "sampling": sampler_state(sampling),
            "final_policy_sha256": state_sha256(model.policy.state_dict()),
            "final_optimizer_sha256": state_sha256(model.policy.optimizer.state_dict()),
            "raw": raw_digests,
            "initial_num_timesteps": initial_timesteps,
            "num_timesteps": model.num_timesteps,
            "n_updates": model._n_updates,
            "policy_seed": training_policy_seed(base, replicate_id=replicate),
            "training_episode_role": base.seeds.roles["training"],
            "development_record": True,
        },
    )
    write_evaluation_seal(
        work,
        {
            "study_id": settings.study_id,
            "completed_rounds": round_index,
            "additional_transitions": budget,
            "replicate_id": replicate,
            "development_record": True,
            "parent_model_sha256": base.parent_model_sha256,
        },
    )
    destination.parent.mkdir(exist_ok=True)
    work.rename(destination)
    return destination


def support_conditions_ids(base):
    return [c.condition_id for c in support_conditions(base, "discovery")]


def sampler_state(sampling):
    return {
        "target_transition_shares": sampling.target_shares,
        "current_reset_probabilities": sampling.probabilities,
        "current_length_estimates": sampling.length_estimates,
        "completed_episodes": sampling.completed_episodes,
        "revision": sampling.revision,
    }


def train_adaptive_seed(config_path: str, study_path: str, replicate: int) -> dict:
    config, study = Path(config_path), Path(study_path)
    settings, base = load_adaptive_study(config)
    protocol = read(study / "protocol/adaptive-protocol.json")
    if source_hashes() != protocol["source_sha256"]:
        raise ValueError("Source changed after adaptive launch.")
    torch.set_num_threads(base.continuation.torch_threads)
    initial = read(study / "initial-discovery/summary.json")
    target = select_frontier(base, initial)
    calibration = read(Path(protocol["initial_calibration"]["summary_path"]))
    sampling = InverseLengthSampler(
        target["target_transition_shares"],
        np.array(calibration["initial_mean_lengths"]),
        base.sampling,
    )
    parent_dir = Path(protocol["parent_artifact_dir"])
    _, parent = load_policy_artifact(parent_dir, require_frozen=True)
    environment = make_training_environment(
        base, parent, sampling.probabilities, replicate_id=replicate, sampling=sampling
    )
    records = []
    checkpoint_paths = []
    stop_reason = None
    try:
        model, migration = migrate_parent(
            parent_dir / "model.zip",
            environment,
            base.continuation,
            training_policy_seed(base, replicate_id=replicate),
        )
        ppo_settings = resolved_settings(model, base.continuation)
        initial_timesteps = model.num_timesteps
        capture = AdaptiveCapture(settings.total_transitions, sampling)
        recorder = OptimizerStepRecorder(model.policy.optimizer)
        discovery_path = study / "initial-discovery/summary.json"
        for phase in range(settings.rounds):
            target_path = (
                study / "targets" / f"r{replicate:03d}" / f"round-{phase:02d}.json"
            )
            write_or_match_json(
                target_path,
                {
                    **target,
                    "replicate_id": replicate,
                    "round_index": phase,
                    "training_start": phase * settings.transitions_per_round,
                    "training_stop": (phase + 1) * settings.transitions_per_round,
                    "selection_support": "discovery",
                    "selection_summary": str(discovery_path.relative_to(study)),
                    "selection_summary_sha256": sha256(discovery_path),
                    "empty_frontier_action": settings.empty_frontier_action,
                    "sampling_before": sampler_state(sampling),
                },
            )
            if target["stop_reason"] is not None:
                stop_reason = target["stop_reason"]
                break
            sampling.set_target_shares(np.array(target["target_transition_shares"]))
            start = capture.transition_count
            started = time.perf_counter()
            model.learn(
                settings.transitions_per_round,
                callback=capture,
                reset_num_timesteps=False,
            )
            if (
                capture.transition_count != (phase + 1) * settings.transitions_per_round
                or model.num_timesteps - initial_timesteps != capture.transition_count
            ):
                raise RuntimeError("Adaptive training budget accounting differs.")
            checkpoint = save_checkpoint(
                settings,
                base,
                study,
                replicate,
                phase + 1,
                model,
                capture,
                sampling,
                target_path,
                initial_timesteps,
                {"migration": migration, **ppo_settings},
            )
            checkpoint_paths.append(str(checkpoint.relative_to(study)))
            interval_condition = capture._columns["condition_index"][
                start : capture.transition_count
            ]
            counts = np.bincount(
                interval_condition, minlength=len(sampling.target_shares)
            )
            record = {
                "replicate_id": replicate,
                "round_index": phase,
                "transitions": capture.transition_count,
                "checkpoint": checkpoint_paths[-1],
                "target_record": str(target_path.relative_to(study)),
                "interval_transition_counts": counts,
                "interval_transition_shares": counts / counts.sum(),
                "target_transition_shares": sampling.target_shares.copy(),
                "training_elapsed_seconds": time.perf_counter() - started,
            }
            records.append(to_jsonable(record))
            print(
                f"Seed {replicate}: round {phase + 1}/{settings.rounds} "
                f"trained and sealed at {capture.transition_count:,} transitions.",
                flush=True,
            )
            if phase + 1 < settings.rounds:
                before_rng = rng_state()
                before_weights = state_sha256(model.policy.state_dict())
                discovery, discovery_path = collect_and_analyze(
                    settings,
                    base,
                    study,
                    checkpoint,
                    support="discovery",
                    round_index=phase + 1,
                    workers=settings.discovery_workers_per_seed,
                )
                assert_rng_unchanged(before_rng)
                if state_sha256(model.policy.state_dict()) != before_weights:
                    raise RuntimeError("Discovery evaluation changed training weights.")
                target = select_frontier(base, discovery)
        recorder.close()
        if records and (
            recorder.first_learning_rate != base.continuation.learning_rate
            or recorder.learning_rates != {base.continuation.learning_rate}
        ):
            raise RuntimeError("Adaptive optimizer learning rate changed.")
        result = {
            "study_id": settings.study_id,
            "replicate_id": replicate,
            "planned_transitions": settings.total_transitions,
            "completed_transitions": capture.transition_count,
            "unrun_transitions": settings.total_transitions - capture.transition_count,
            "stop_reason": stop_reason,
            "rounds": records,
            "checkpoints": checkpoint_paths,
            "optimizer_steps": recorder.steps,
            "migration": migration,
            "training_state_continues_across_rounds": True,
            "evaluation_rng_checks_passed": True,
        }
        # Elapsed times can differ on replay; immutable scientific records must match.
        result_path = study / "training" / f"r{replicate:03d}.json"
        if result_path.exists():
            old = read(result_path)
            for record in old["rounds"]:
                record.pop("training_elapsed_seconds", None)
            comparable = to_jsonable(result)
            for record in comparable["rounds"]:
                record.pop("training_elapsed_seconds", None)
            if old != comparable:
                raise ValueError("Adaptive seed replay result differs.")
            return read(result_path)
        write_json_new(result_path, result)
        return to_jsonable(result)
    finally:
        environment.close()


def run_adaptive(config: Path, study: Path, artifact_root: Path) -> dict:
    started = time.perf_counter()
    settings, base = load_adaptive_study(config)
    prepare_adaptive(config, study, artifact_root)
    parent = artifact_root / base.parent_policy_id
    initial, _ = collect_and_analyze(
        settings,
        base,
        study,
        parent,
        support="discovery",
        round_index=0,
        workers=settings.reference_workers,
    )
    write_or_match_json(study / "initial-discovery/summary.json", initial)
    (study / "training").mkdir(exist_ok=True)
    with ProcessPoolExecutor(
        max_workers=len(settings.replicate_ids),
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        futures = [
            pool.submit(
                train_adaptive_seed,
                str(config.resolve()),
                str(study.resolve()),
                replicate,
            )
            for replicate in settings.replicate_ids
        ]
        results = [future.result() for future in futures]
    # Write this barrier before any reference evaluation is even planned.
    write_or_match_json(study / "training-complete.json", {"seeds": results})
    checkpoints = [
        parent,
        *[study / name for result in results for name in result["checkpoints"]],
    ]
    for index, checkpoint in enumerate(checkpoints):
        collect_and_analyze(
            settings,
            base,
            study,
            checkpoint,
            support="reference",
            round_index=0,
            workers=settings.reference_workers,
        )
        print(
            f"Reference checkpoint {index + 1}/{len(checkpoints)} complete.", flush=True
        )
    result = {
        "study_id": settings.study_id,
        "elapsed_seconds": time.perf_counter() - started,
        "completed_seed_transitions": {
            str(r["replicate_id"]): r["completed_transitions"] for r in results
        },
        "unrun_seed_transitions": {
            str(r["replicate_id"]): r["unrun_transitions"] for r in results
        },
        "reference_checkpoints": len(checkpoints),
    }
    if not (study / "completion.json").exists():
        write_json_new(study / "completion.json", result)
    return result
