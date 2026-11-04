"""Frozen-policy evaluation campaigns for the CartPole PPO boundary study.

A job is one checkpoint at one condition. Jobs write immutable shards, so a
campaign can resume by running only the jobs whose shards do not yet exist.
This module records raw trajectories only; metrics belong to the analysis module.
"""

import json
import multiprocessing
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from active_eval_gym.config import PPOBoundaryStudySpec
from active_eval_gym.envs.perturbations import PerturbationSpec
from active_eval_gym.evaluate import make_artifact_environment
from active_eval_gym.policies.artifacts import (
    load_evaluation_ready_artifact,
    load_policy_artifact,
    verify_evaluation_seal,
)
from active_eval_gym.ppo_boundary import (
    child_artifact_dir,
    evaluation_episode_seeds,
    load_protocol,
    source_hashes,
    source_state,
    support_conditions,
    tier_children,
    verify_parent,
)
from active_eval_gym.rollout import collect_episode
from active_eval_gym.serialization import (
    EPISODE_SHARD_LAYOUT,
    read_shard_header,
    to_jsonable,
    write_episode_shard,
    write_json_new,
)

DISCOVERY = "discovery"
CONFIRMATORY = "confirmatory"
CAMPAIGN_DIRECTORIES = {
    DISCOVERY: "discovery",
    "smoke": "smoke",
    "pilot": "pilot",
    CONFIRMATORY: "reference_raw",
}
PLAN_FILENAME = "evaluation-plan.json"
COLLECTION_FILENAME = "collection.json"
SHARD_DIRECTORY = "shards"
FAILURE_DIRECTORY = "failures"

_LOADED_CHECKPOINTS: dict[str, tuple[Any, Any]] = {}


@dataclass(frozen=True)
class EvaluationJob:
    """One frozen checkpoint evaluated on the paired resets at one condition."""

    job_id: str
    artifact_dir: str
    shard_path: str
    compression_level: int
    header: dict[str, Any]
    sealed: bool = False


@dataclass(frozen=True)
class EvaluationCheckpoint:
    """One checkpoint a campaign evaluates, and how it must be verified."""

    checkpoint_id: str
    artifact_dir: Path
    model_sha256: str
    sealed: bool


def campaign_design(spec: PPOBoundaryStudySpec, campaign: str) -> dict[str, Any]:
    """Return the support and paired reset seeds a campaign must use."""

    if campaign == DISCOVERY:
        role = spec.seeds.roles["discovery"]
        support = "discovery"
        episodes = spec.curriculum.discovery_episodes_per_condition
    else:
        tier = spec.tier(campaign)
        role = spec.seeds.roles[tier.seed_role]
        support = tier.evaluation_support
        episodes = tier.episodes_per_condition
    return {
        "support": support,
        "seed_role": role,
        "episode_seeds": evaluation_episode_seeds(role, episodes),
    }


def campaign_checkpoints(
    spec: PPOBoundaryStudySpec,
    campaign: str,
    *,
    study_dir: Path,
    artifact_root: Path,
    allow_dirty: bool = False,
) -> tuple[EvaluationCheckpoint, ...]:
    """Return the parent, then every planned child of a tier campaign.

    Every child must be sealed first, so no child can depend on this campaign's
    outcomes. A regressing child is sealed like any other and stays in the list.
    """

    parent = EvaluationCheckpoint(
        spec.parent_policy_id,
        artifact_root / spec.parent_policy_id,
        spec.parent_model_sha256,
        sealed=False,
    )
    if campaign == DISCOVERY:
        return (parent,)
    children = []
    for child in tier_children(spec, campaign):
        directory = child_artifact_dir(study_dir, campaign, child["policy_id"])
        if not directory.is_dir():
            raise ValueError(
                f"Campaign {campaign!r} requires every child sealed first; "
                f"{child['policy_id']} is missing."
            )
        seal = verify_evaluation_seal(directory)
        if seal["development_record"] and not allow_dirty:
            raise ValueError(
                f"Child {child['policy_id']} is a development record. "
                "Pass allow_dirty only for temporary development runs."
            )
        if seal["parent_model_sha256"] != spec.parent_model_sha256:
            raise ValueError(f"Child {child['policy_id']} has another parent.")
        children.append(
            EvaluationCheckpoint(
                child["policy_id"], directory, seal["model_sha256"], sealed=True
            )
        )
    return (parent, *children)


def plan_jobs(
    spec: PPOBoundaryStudySpec,
    campaign: str,
    *,
    study_dir: Path,
    artifact_root: Path,
    allow_dirty: bool = False,
) -> tuple[EvaluationJob, ...]:
    """Expand a campaign into its exact, deterministically ordered jobs."""

    if spec.storage.evaluation_layout != EPISODE_SHARD_LAYOUT:
        raise ValueError(f"Study requires layout {EPISODE_SHARD_LAYOUT!r}.")
    if spec.storage.compression != "gzip":
        raise ValueError("The shard layout requires gzip compression.")
    design = campaign_design(spec, campaign)
    root = study_dir / CAMPAIGN_DIRECTORIES[campaign] / SHARD_DIRECTORY
    checkpoints = campaign_checkpoints(
        spec,
        campaign,
        study_dir=study_dir,
        artifact_root=artifact_root,
        allow_dirty=allow_dirty,
    )
    conditions = support_conditions(spec, design["support"])
    return tuple(
        EvaluationJob(
            job_id=f"{checkpoint.checkpoint_id}/{condition.condition_id}",
            artifact_dir=str(checkpoint.artifact_dir),
            shard_path=str(
                root / checkpoint.checkpoint_id / f"{condition.condition_id}.jsonl.gz"
            ),
            compression_level=spec.storage.compression_level,
            header={
                "study_id": spec.study_id,
                "campaign": campaign,
                "checkpoint_id": checkpoint.checkpoint_id,
                "model_sha256": checkpoint.model_sha256,
                "condition_id": condition.condition_id,
                "perturbation": to_jsonable(condition.perturbation),
                "seed_role": design["seed_role"],
                "episode_seeds": list(design["episode_seeds"]),
                "deterministic": True,
            },
            sealed=checkpoint.sealed,
        )
        for checkpoint in checkpoints
        for condition in conditions
    )


def run_evaluation_job(job: EvaluationJob) -> int:
    """Collect one job's episodes and publish its shard atomically."""

    policy, metadata = _loaded_checkpoint(job.artifact_dir, sealed=job.sealed)
    if metadata.model_sha256 != job.header["model_sha256"]:
        raise ValueError(f"Job {job.job_id} loaded an unexpected checkpoint.")
    perturbation = PerturbationSpec(
        job.header["perturbation"]["name"], job.header["perturbation"]["parameters"]
    )
    env, resolved = make_artifact_environment(metadata, perturbation)
    try:
        return write_episode_shard(
            Path(job.shard_path),
            job.header,
            (
                collect_episode(
                    env,
                    policy,
                    policy_artifact=metadata,
                    nominal_environment=metadata.design_spec.environment,
                    resolved_environment=resolved,
                    episode_seed=seed,
                    deterministic=True,
                )
                for seed in job.header["episode_seeds"]
            ),
            compression_level=job.compression_level,
        )
    finally:
        env.close()


def collect_campaign(
    config_path: Path,
    *,
    study_dir: Path,
    campaign: str,
    artifact_root: Path,
    workers: int = 1,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Run every job without a shard, recording failures instead of hiding them."""

    if campaign == CONFIRMATORY:
        raise ValueError("Confirmatory collection requires the signed S7 lock.")
    spec, provenance = load_protocol(config_path, study_dir)
    source = source_state()
    if not allow_dirty and (provenance["development_record"] or source["source_dirty"]):
        raise ValueError(
            "Study data require a committed source and non-development protocol. "
            "Pass allow_dirty only for temporary development runs."
        )
    verify_parent(spec, artifact_root)
    jobs = plan_jobs(
        spec,
        campaign,
        study_dir=study_dir,
        artifact_root=artifact_root,
        allow_dirty=allow_dirty,
    )
    directory = study_dir / CAMPAIGN_DIRECTORIES[campaign]
    _write_or_match_plan(directory, spec, campaign, jobs, provenance, source)

    pending = [job for job in jobs if not _shard_complete(job)]
    failed = run_jobs(pending, directory, spec.continuation.torch_threads, workers)
    complete = not failed
    if complete and not (directory / COLLECTION_FILENAME).exists():
        write_json_new(
            directory / COLLECTION_FILENAME,
            {
                "schema_version": 1,
                "study_id": spec.study_id,
                "campaign": campaign,
                "job_count": len(jobs),
                "episode_count": sum(len(job.header["episode_seeds"]) for job in jobs),
            },
        )
    return {
        "campaign": campaign,
        "planned_jobs": len(jobs),
        "already_complete_jobs": len(jobs) - len(pending),
        "failed_jobs": failed,
        "complete": complete,
    }


def write_or_match_json(path: Path, value: Any) -> None:
    """Record a plan once, then refuse any resume that would differ from it."""

    plan = to_jsonable(value)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_new(path, plan)
    elif json.loads(path.read_text()) != plan:
        raise ValueError(f"Resume does not match the recorded plan: {path}.")


def _write_or_match_plan(
    directory: Path,
    spec: PPOBoundaryStudySpec,
    campaign: str,
    jobs: tuple[EvaluationJob, ...],
    provenance: dict[str, Any],
    source: dict[str, Any],
) -> None:
    design = campaign_design(spec, campaign)
    plan = to_jsonable(
        {
            "schema_version": 1,
            "study_id": spec.study_id,
            "campaign": campaign,
            "support": design["support"],
            "seed_role": design["seed_role"],
            "episode_seeds": design["episode_seeds"],
            "metric_version": spec.metric_version,
            "storage": spec.storage,
            "protocol_config_sha256": provenance["config_sha256"],
            "source": source,
            "source_sha256": source_hashes(),
            "jobs": [
                {
                    "job_id": job.job_id,
                    "shard": str(Path(job.shard_path).relative_to(directory)),
                    "header": job.header,
                }
                for job in jobs
            ],
        }
    )
    write_or_match_json(directory / PLAN_FILENAME, plan)


def _shard_complete(job: EvaluationJob) -> bool:
    path = Path(job.shard_path)
    if not path.exists():
        return False
    header = read_shard_header(path)
    if any(header.get(name) != value for name, value in job.header.items()):
        raise ValueError(f"Existing shard does not match job {job.job_id}: {path}.")
    return True


def run_jobs(
    jobs: list[Any],
    directory: Path,
    threads: int,
    workers: int,
    run: Callable[[Any], Any] | None = None,
) -> list[str]:
    """Run jobs serially or in processes, recording each failure by job ID."""

    run = run or run_evaluation_job
    failed = []
    if workers == 1:
        _initialize_worker(threads)
        for job in jobs:
            try:
                run(job)
            except Exception as error:
                failed.append(_record_failure(directory, job, error))
        return failed
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(threads,),
    ) as executor:
        futures = [(job, executor.submit(run, job)) for job in jobs]
        for job, future in futures:
            error = future.exception()
            if error is not None:
                failed.append(_record_failure(directory, job, error))
    return failed


def _initialize_worker(threads: int) -> None:
    import torch

    torch.set_num_threads(threads)


def _loaded_checkpoint(artifact_dir: str, *, sealed: bool) -> tuple[Any, Any]:
    if artifact_dir not in _LOADED_CHECKPOINTS:
        directory = Path(artifact_dir)
        _LOADED_CHECKPOINTS[artifact_dir] = (
            load_evaluation_ready_artifact(directory)
            if sealed
            else load_policy_artifact(directory, require_frozen=True)
        )
    return _LOADED_CHECKPOINTS[artifact_dir]


def _record_failure(directory: Path, job: EvaluationJob, error: BaseException) -> str:
    root = directory / FAILURE_DIRECTORY
    root.mkdir(exist_ok=True)
    stem = job.job_id.replace("/", "__")
    attempt = sum(1 for _ in root.glob(f"{stem}.attempt-*.json")) + 1
    write_json_new(
        root / f"{stem}.attempt-{attempt}.json",
        {
            "job_id": job.job_id,
            "attempt": attempt,
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exception(error),
        },
    )
    return job.job_id
