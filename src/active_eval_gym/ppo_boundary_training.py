"""Continue the frozen parent under each arm's fixed condition distribution.

One job trains one child for a fixed budget. It writes to a private work
directory and publishes only after its checks pass, so a sealed child means a
finished budget. Nothing here reads evaluation results beyond the frozen
distributions, and nothing selects a checkpoint by performance.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from active_eval_gym.config import ContinuationSpec, PPOBoundaryStudySpec
from active_eval_gym.envs.condition_sampler import (
    CONDITION_INFO_KEY,
    SAMPLER_VERSION,
    CartPoleRecoveryConditionSampler,
)
from active_eval_gym.evaluate import make_artifact_environment
from active_eval_gym.models import PolicyArtifactMetadata, PolicyDesignSpec
from active_eval_gym.policies.artifacts import (
    TRAINING_SUMMARY_FILENAME,
    load_policy_artifact,
    verify_evaluation_seal,
    write_continued_artifact,
    write_evaluation_seal,
)
from active_eval_gym.policies.sb3 import SB3Policy
from active_eval_gym.ppo_boundary import (
    POLICIES_DIRECTORY,
    RECORDED_PACKAGES,
    SNAPSHOT_DIRECTORY,
    SNAPSHOT_ROLE,
    child_artifact_dir,
    condition_seed_sequence,
    episode_seed,
    load_protocol,
    snapshot_label,
    source_hashes,
    source_state,
    support_conditions,
    tier_children,
    training_policy_seed,
    verify_parent,
)
from active_eval_gym.ppo_boundary_evaluation import (
    CONFIRMATORY,
    run_jobs,
    write_or_match_json,
)
from active_eval_gym.ppo_boundary_targets import (
    DISTRIBUTIONS_DIRECTORY,
    DISTRIBUTIONS_FILENAME,
)
from active_eval_gym.serialization import (
    write_json_new,
    write_transition_shard,
)

TRAINING_RAW_DIRECTORY = "training_raw"
WORK_DIRECTORY = ".work"
ABANDONED_DIRECTORY = "abandoned"
POLICY_WORK_DIRECTORY = "policy"
RAW_WORK_DIRECTORY = "raw"
TRANSITIONS_FILENAME = "transitions.npz"
TRAINING_PLAN_FILENAME = "training-plan.json"
TRAINING_RECORD_FILENAME = "training.json"
MODEL_FILENAME = "model.zip"
MIGRATION_ID = "cartpole-recovery-observation-box-v1"
MIGRATION_BATCH_SIZE = 4096
MIGRATION_BATCH_SEED = 61800
MIGRATION_BATCH_BOUND = (2.4, 3.0, 1.6, 5.0)
FINAL_ROLE = "final"
OUTER_ROUND = 0
DECLARED_MODEL_SETTINGS = (
    "n_envs",
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "ent_coef",
)


@dataclass(frozen=True)
class TrainingJob:
    """One child: paths and identifiers only, so it survives process pickling."""

    job_id: str
    tier: str
    arm: str
    replicate_id: int
    config_path: str
    study_dir: str
    parent_dir: str
    development_record: bool
    source: dict[str, Any]


class OptimizerStepRecorder:
    """Read the learning rate at each real optimizer step, changing nothing."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.first_learning_rate: float | None = None
        self.learning_rates: set[float] = set()
        self.steps = 0
        self._handle = optimizer.register_step_pre_hook(self._record)

    def _record(self, optimizer: Any, _args: Any, _kwargs: Any) -> None:
        learning_rate = float(optimizer.param_groups[0]["lr"])
        if self.first_learning_rate is None:
            self.first_learning_rate = learning_rate
        self.learning_rates.add(learning_rate)
        self.steps += 1

    def close(self) -> None:
        self._handle.remove()


class TransitionCapture(BaseCallback):
    """Retain every training transition, reset, and rollout's behavior policy.

    Rows are read from the sampler's records, because the state before and after
    an automatic reset cannot be recovered from the vector environment's output.
    The same instance spans repeated `learn` calls. It draws no random numbers.
    """

    def __init__(self, capacity: int) -> None:
        super().__init__(verbose=0)
        self._count = 0
        self._columns: dict[str, np.ndarray] = {
            "state": np.empty((capacity, 4), np.float64),
            "next_state": np.empty((capacity, 4), np.float64),
            "observation": np.empty((capacity, 4), np.float32),
            "next_observation": np.empty((capacity, 4), np.float32),
            "action": np.empty(capacity, np.uint8),
            "environment_reward": np.empty(capacity, np.float64),
            "training_reward": np.empty(capacity, np.float64),
            "terminated": np.empty(capacity, np.bool_),
            "truncated": np.empty(capacity, np.bool_),
            "worker": np.empty(capacity, np.uint8),
            "episode_index": np.empty(capacity, np.uint32),
            "condition_index": np.empty(capacity, np.uint32),
            "reset_seed": np.empty(capacity, np.int64),
            "rollout": np.empty(capacity, np.uint32),
        }
        self._resets: dict[tuple[int, int], dict[str, Any]] = {}
        self.rollouts: list[dict[str, Any]] = []
        self.updates: list[dict[str, float | int]] = []

    @property
    def transition_count(self) -> int:
        return self._count

    def _on_training_start(self) -> None:
        self._record_resets()

    def _on_rollout_start(self) -> None:
        self._record_update()
        self.rollouts.append(
            {
                "rollout": len(self.rollouts),
                "n_updates": self.model._n_updates,
                "num_timesteps": self.model.num_timesteps,
                "policy_sha256": state_sha256(self.model.policy.state_dict()),
            }
        )

    def _on_step(self) -> bool:
        actions = np.asarray(self.locals["actions"]).reshape(-1)
        rewards = self.locals["rewards"]
        columns = self._columns
        for worker, info in enumerate(self.locals["infos"]):
            record = info[CONDITION_INFO_KEY]
            row = self._count
            for name in (
                "state",
                "next_state",
                "observation",
                "next_observation",
                "environment_reward",
                "terminated",
                "truncated",
                "episode_index",
                "condition_index",
                "reset_seed",
            ):
                columns[name][row] = record[name]
            columns["action"][row] = actions[worker]
            columns["training_reward"][row] = rewards[worker]
            columns["worker"][row] = worker
            columns["rollout"][row] = len(self.rollouts) - 1
            self._count += 1
        if np.any(self.locals["dones"]):
            self._record_resets()
        return True

    def _on_training_end(self) -> None:
        self._record_update()

    def _record_resets(self) -> None:
        for worker, info in enumerate(self.training_env.reset_infos):
            record = info[CONDITION_INFO_KEY]
            self._resets.setdefault((worker, record["episode_index"]), record)

    def _record_update(self) -> None:
        logged = self.logger.name_to_value
        if "train/clip_range" in logged and (
            not self.updates or self.updates[-1]["n_updates"] != self.model._n_updates
        ):
            self.updates.append(
                {
                    "n_updates": self.model._n_updates,
                    "learning_rate": float(logged["train/learning_rate"]),
                    "clip_range": float(logged["train/clip_range"]),
                }
            )

    def tables(self) -> dict[str, np.ndarray]:
        """Return everything captured so far as columns."""

        columns = {name: rows[: self._count] for name, rows in self._columns.items()}
        resets = list(self._resets.items())
        rollouts = self.rollouts
        return {
            **columns,
            "reset_worker": np.array([key[0] for key, _ in resets], np.uint8),
            "reset_episode_index": np.array([key[1] for key, _ in resets], np.uint32),
            "reset_condition_index": np.array(
                [row["condition_index"] for _, row in resets], np.uint32
            ),
            "reset_reset_seed": np.array(
                [row["reset_seed"] for _, row in resets], np.int64
            ),
            "reset_state": np.array(
                [row["state"] for _, row in resets], np.float64
            ).reshape(-1, 4),
            "rollout_n_updates": np.array(
                [row["n_updates"] for row in rollouts], np.int64
            ),
            "rollout_num_timesteps": np.array(
                [row["num_timesteps"] for row in rollouts], np.int64
            ),
            "rollout_policy_sha256": np.array(
                [row["policy_sha256"] for row in rollouts], "U64"
            ),
        }


def state_sha256(state: Any) -> str:
    """Hash names, shapes, and values, unlike a model zip's embedded metadata."""

    digest = hashlib.sha256()
    _update_digest(digest, state)
    return digest.hexdigest()


def realized_sampling(
    tables: dict[str, np.ndarray], condition_ids: tuple[str, ...]
) -> dict[str, Any]:
    """Return realized episode and transition shares for each condition.

    `q` is an episode distribution; longer episodes contribute more transitions,
    so the two shares differ. A budget can end mid-episode, and those episodes
    are listed as unfinished rather than counted as completed outcomes.
    """

    size = len(condition_ids)
    condition = tables["condition_index"]
    finished = tables["terminated"] | tables["truncated"]
    started = np.bincount(tables["reset_condition_index"], minlength=size)
    transitions = np.bincount(condition, minlength=size)
    completed = np.bincount(condition[finished], minlength=size)
    closed = set(
        zip(
            tables["worker"][finished].tolist(),
            tables["episode_index"][finished].tolist(),
            strict=True,
        )
    )
    keys, counts = np.unique(
        np.column_stack((tables["worker"], tables["episode_index"])),
        axis=0,
        return_counts=True,
    )
    steps = {
        (int(worker), int(episode)): int(count)
        for (worker, episode), count in zip(keys, counts, strict=True)
    }
    unfinished = [
        {
            "worker": worker,
            "episode_index": episode,
            "condition_id": condition_ids[index],
            "transitions": steps.get((worker, episode), 0),
        }
        for worker, episode, index in zip(
            tables["reset_worker"].tolist(),
            tables["reset_episode_index"].tolist(),
            tables["reset_condition_index"].tolist(),
            strict=True,
        )
        if (worker, episode) not in closed
    ]
    return {
        "condition_ids": condition_ids,
        "episodes_started": started,
        "episodes_completed": completed,
        "transitions": transitions,
        "episode_shares": started / max(int(started.sum()), 1),
        "transition_shares": transitions / max(int(transitions.sum()), 1),
        "unfinished_episodes": unfinished,
    }


def load_distributions(
    spec: PPOBoundaryStudySpec,
    provenance: dict[str, Any],
    study_dir: Path,
    *,
    allow_dirty: bool = False,
) -> tuple[dict[str, Any], str]:
    """Load the frozen distributions, refusing any another protocol produced."""

    path = study_dir / DISTRIBUTIONS_DIRECTORY / DISTRIBUTIONS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Training requires frozen distributions: {path}.")
    content = path.read_bytes()
    record = json.loads(content)
    selection = record["selection"]
    if record["study_id"] != spec.study_id:
        raise ValueError("Distributions belong to another study.")
    if record["target_rule_version"] != spec.curriculum.target_rule:
        raise ValueError("Distributions use an undeclared target rule.")
    if selection["protocol_config_sha256"] != provenance["config_sha256"]:
        raise ValueError("Distributions do not match the study protocol.")
    if selection["parent_model_sha256"] != spec.parent_model_sha256:
        raise ValueError("Distributions were selected for another parent.")
    if record["development_record"] and not allow_dirty:
        raise ValueError(
            "Distributions are a development record. "
            "Pass allow_dirty only for temporary development runs."
        )
    expected = [c.condition_id for c in support_conditions(spec, "discovery")]
    if record["condition_ids"] != expected:
        raise ValueError("Distributions do not match the discovery conditions.")
    if set(record["arms"]) != set(spec.curriculum.arms):
        raise ValueError("Distributions do not match the declared arms.")
    return record, hashlib.sha256(content).hexdigest()


def make_training_environment(
    spec: PPOBoundaryStudySpec,
    parent: PolicyArtifactMetadata,
    probabilities: np.ndarray,
    *,
    replicate_id: int,
) -> DummyVecEnv:
    """Build the workers, with each sampler inside `Monitor`."""

    conditions = support_conditions(spec, "discovery")
    backend = spec.continuation.vec_env_backend
    if backend != "DummyVecEnv":
        raise ValueError(f"Unsupported vector-environment backend {backend!r}.")

    def worker(worker_id: int):
        def build() -> Monitor:
            env, _ = make_artifact_environment(parent)
            generator = np.random.default_rng(
                condition_seed_sequence(
                    spec, replicate_id=replicate_id, worker_id=worker_id
                )
            )

            def reset_seed(index: int) -> int:
                return episode_seed(
                    role=spec.seeds.roles["training"],
                    outer_round=OUTER_ROUND,
                    replicate_id=replicate_id,
                    worker_id=worker_id,
                    episode_index=index,
                )

            return Monitor(
                CartPoleRecoveryConditionSampler(
                    env,
                    conditions,
                    probabilities,
                    condition_rng=generator,
                    reset_seed=reset_seed,
                    worker_id=worker_id,
                )
            )

        return build

    count = int(spec.continuation.hyperparameters["n_envs"])
    return DummyVecEnv([worker(index) for index in range(count)])


def migrate_parent(
    parent_zip: Path,
    environment: DummyVecEnv,
    continuation: ContinuationSpec,
    seed: int,
) -> tuple[PPO, dict[str, Any]]:
    """Attach the parent to the recovery environment, proving it did not change.

    Only observation-space metadata and the schedules differ. The reference is
    loaded first because every load reseeds the global random generators.
    """

    reference = _fingerprint(PPO.load(parent_zip, device=continuation.device))
    model = PPO.load(
        parent_zip,
        env=environment,
        device=continuation.device,
        custom_objects={"observation_space": environment.observation_space},
        learning_rate=continuation.learning_rate,
        clip_range=continuation.clip_range,
        seed=seed,
        force_reset=True,
    )
    migrated = _fingerprint(model)
    for name in ("policy_sha256", "optimizer_sha256"):
        if migrated[name] != reference[name]:
            raise RuntimeError(f"Checkpoint migration changed {name}.")
    for name in ("logits", "values", "actions"):
        if not np.array_equal(migrated[name], reference[name]):
            raise RuntimeError(f"Checkpoint migration changed {name}.")
    if model.get_vec_normalize_env() is not None:
        raise RuntimeError("Checkpoint migration must not add normalization.")
    return model, {
        "migration_id": MIGRATION_ID,
        "batch_size": MIGRATION_BATCH_SIZE,
        "policy_sha256": migrated["policy_sha256"],
        "optimizer_sha256": migrated["optimizer_sha256"],
        "restored_optimizer_learning_rate": float(
            model.policy.optimizer.param_groups[0]["lr"]
        ),
    }


def resolved_settings(model: PPO, continuation: ContinuationSpec) -> dict[str, Any]:
    """Require the declared settings and return every value that was inherited."""

    for name in DECLARED_MODEL_SETTINGS:
        if getattr(model, name) != continuation.hyperparameters[name]:
            raise RuntimeError(f"Continuation model has an undeclared {name}.")
    learning_rates = {model.lr_schedule(1.0), model.lr_schedule(0.0)}
    clip_ranges = {model.clip_range(1.0), model.clip_range(0.0)}
    if learning_rates != {continuation.learning_rate}:
        raise RuntimeError(f"Learning-rate schedule is not constant: {learning_rates}.")
    if clip_ranges != {continuation.clip_range}:
        raise RuntimeError(f"Clip-range schedule is not constant: {clip_ranges}.")
    return {
        **{name: getattr(model, name) for name in DECLARED_MODEL_SETTINGS},
        "learning_rate": continuation.learning_rate,
        "clip_range": continuation.clip_range,
        "clip_range_vf": model.clip_range_vf,
        "vf_coef": model.vf_coef,
        "max_grad_norm": model.max_grad_norm,
        "normalize_advantage": model.normalize_advantage,
        "target_kl": model.target_kl,
        "use_sde": model.use_sde,
        "policy_class": type(model.policy).__name__,
        "device": str(model.device),
    }


def plan_training(
    spec: PPOBoundaryStudySpec,
    tier: str,
    *,
    config_path: Path,
    study_dir: Path,
    artifact_root: Path,
    development_record: bool,
    source: dict[str, Any],
) -> tuple[TrainingJob, ...]:
    """Expand a tier into one job per replicate and arm."""

    return tuple(
        TrainingJob(
            job_id=child["policy_id"],
            tier=tier,
            arm=child["arm"],
            replicate_id=child["replicate_id"],
            config_path=str(config_path),
            study_dir=str(study_dir),
            parent_dir=str(artifact_root / spec.parent_policy_id),
            development_record=development_record,
            source=source,
        )
        for child in tier_children(spec, tier)
    )


def train_campaign(
    config_path: Path,
    *,
    study_dir: Path,
    tier: str,
    artifact_root: Path,
    workers: int = 1,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Run every unsealed child of a tier, recording failures without hiding them."""

    if tier == CONFIRMATORY:
        raise ValueError("Confirmatory training requires the signed S7 lock.")
    spec, provenance = load_protocol(config_path, study_dir)
    source = source_state()
    development = bool(provenance["development_record"] or source["source_dirty"])
    if development and not allow_dirty:
        raise ValueError(
            "Study data require a committed source and non-development protocol. "
            "Pass allow_dirty only for temporary development runs."
        )
    verify_parent(spec, artifact_root)
    distributions, distributions_sha256 = load_distributions(
        spec, provenance, study_dir, allow_dirty=allow_dirty
    )
    jobs = plan_training(
        spec,
        tier,
        config_path=config_path,
        study_dir=study_dir,
        artifact_root=artifact_root,
        development_record=development or distributions["development_record"],
        source=source,
    )
    tier_spec = spec.tier(tier)
    directory = study_dir / POLICIES_DIRECTORY / tier
    write_or_match_json(
        directory / TRAINING_PLAN_FILENAME,
        {
            "schema_version": 1,
            "study_id": spec.study_id,
            "tier": tier,
            "protocol_config_sha256": provenance["config_sha256"],
            "parent_model_sha256": spec.parent_model_sha256,
            "distributions_sha256": distributions_sha256,
            "additional_transitions": tier_spec.additional_transitions,
            "snapshot_transitions": tier_spec.snapshot_transitions,
            "continuation": spec.continuation,
            "source": source,
            "source_sha256": source_hashes(),
            "jobs": [
                {
                    "job_id": job.job_id,
                    "arm": job.arm,
                    "replicate_id": job.replicate_id,
                    "policy_seed": training_policy_seed(
                        spec, replicate_id=job.replicate_id
                    ),
                }
                for job in jobs
            ],
        },
    )
    pending = [job for job in jobs if not _job_complete(job)]
    failed = run_jobs(
        pending,
        directory,
        spec.continuation.torch_threads,
        workers,
        run=run_training_job,
    )
    complete = not failed
    if complete and not (directory / TRAINING_RECORD_FILENAME).exists():
        write_json_new(
            directory / TRAINING_RECORD_FILENAME,
            {
                "schema_version": 1,
                "study_id": spec.study_id,
                "tier": tier,
                "child_count": len(jobs),
                "additional_transitions_per_child": tier_spec.additional_transitions,
            },
        )
    return {
        "tier": tier,
        "planned_jobs": len(jobs),
        "already_complete_jobs": len(jobs) - len(pending),
        "failed_jobs": failed,
        "complete": complete,
    }


def run_training_job(
    job: TrainingJob, *, snapshot_transitions: tuple[int, ...] | None = None
) -> str:
    """Train one child from the parent and publish it with its raw transitions.

    A rerun starts from the parent with the same seeds, so an interrupted job
    needs no partial-model resume. `snapshot_transitions` exists so tests can
    compare chunked training against a single `learn` call.
    """

    study_dir = Path(job.study_dir)
    spec, provenance = load_protocol(Path(job.config_path), study_dir)
    distributions, distributions_sha256 = load_distributions(
        spec, provenance, study_dir, allow_dirty=job.development_record
    )
    tier = spec.tier(job.tier)
    budget = tier.additional_transitions
    snapshots = (
        tier.snapshot_transitions
        if snapshot_transitions is None
        else snapshot_transitions
    )
    seed = training_policy_seed(spec, replicate_id=job.replicate_id)
    arm = distributions["arms"][job.arm]
    probabilities = np.array(arm["probabilities"])

    parent_dir = Path(job.parent_dir)
    parent_zip = parent_dir / MODEL_FILENAME
    _, parent = load_policy_artifact(parent_dir, require_frozen=True)
    if parent.model_sha256 != spec.parent_model_sha256:
        raise ValueError("Training may only continue the declared parent.")

    paths = _job_paths(study_dir, job)
    if paths["policy"].exists():
        raise FileExistsError(f"Refusing to retrain child: {paths['policy']}.")
    _shelve_incomplete(paths, job)
    paths["policy_work"].mkdir(parents=True)
    paths["raw_work"].mkdir(parents=True)

    context = {
        "job": job,
        "spec": spec,
        "parent": parent,
        "seed": seed,
        "budget": budget,
        "distributions": distributions,
        "distributions_sha256": distributions_sha256,
    }
    environment = make_training_environment(
        spec, parent, probabilities, replicate_id=job.replicate_id
    )
    started = time.perf_counter()
    try:
        model, migration = migrate_parent(
            parent_zip, environment, spec.continuation, seed
        )
        settings = resolved_settings(model, spec.continuation)
        context |= {"migration": migration, "settings": settings}
        initial_timesteps = model.num_timesteps
        initial_updates = model._n_updates
        capture = TransitionCapture(budget)
        recorder = OptimizerStepRecorder(model.policy.optimizer)
        for boundary in (*snapshots, budget):
            done = model.num_timesteps - initial_timesteps
            model.learn(boundary - done, callback=capture, reset_num_timesteps=False)
            actual = model.num_timesteps - initial_timesteps
            if actual != boundary or capture.transition_count != boundary:
                raise RuntimeError(
                    f"Expected {boundary} new transitions; the model counted "
                    f"{actual} and capture retained {capture.transition_count}."
                )
            if boundary != budget:
                _write_snapshot(paths, context, model, capture, boundary)
        recorder.close()
        first_update = _verify_first_update(recorder, capture, spec.continuation)
        elapsed = time.perf_counter() - started
        tables = capture.tables()
        digests = write_transition_shard(
            paths["raw_work"] / TRANSITIONS_FILENAME,
            {
                "study_id": spec.study_id,
                "policy_id": job.job_id,
                "tier": job.tier,
                "arm": job.arm,
                "replicate_id": job.replicate_id,
                "outer_round": OUTER_ROUND,
                "phase": OUTER_ROUND,
                "policy_seed": seed,
                "condition_ids": distributions["condition_ids"],
                "training_reward": spec.training_reward,
            },
            tables,
        )
        _write_checkpoint(
            paths["policy_work"],
            context,
            model,
            policy_id=job.job_id,
            role=FINAL_ROLE,
            transitions=budget,
            tables=tables,
            extra={
                "transitions_file": {
                    "path": str(
                        Path(TRAINING_RAW_DIRECTORY)
                        / job.tier
                        / job.job_id
                        / TRANSITIONS_FILENAME
                    ),
                    **digests,
                },
                "first_update": first_update,
                "budget_accounting": {
                    "num_timesteps_before": initial_timesteps,
                    "num_timesteps_after": model.num_timesteps,
                    "n_updates_before": initial_updates,
                    "n_updates_after": model._n_updates,
                    "rollouts": len(capture.rollouts),
                },
                "runtime": {
                    "elapsed_seconds": elapsed,
                    "transitions_per_second": budget / elapsed,
                    "torch_threads": torch.get_num_threads(),
                    "packages": {name: version(name) for name in RECORDED_PACKAGES},
                },
            },
        )
    finally:
        environment.close()

    if parent.model_sha256 != _sha256(parent_zip):
        raise RuntimeError("Training changed the parent checkpoint.")
    _publish(paths)
    return job.job_id


def _write_snapshot(
    paths: dict[str, Path],
    context: dict[str, Any],
    model: PPO,
    capture: TransitionCapture,
    boundary: int,
) -> None:
    label = snapshot_label(boundary)
    _write_checkpoint(
        paths["policy_work"] / SNAPSHOT_DIRECTORY / label,
        context,
        model,
        policy_id=f"{context['job'].job_id}-{label}",
        role=SNAPSHOT_ROLE,
        transitions=boundary,
        tables=capture.tables(),
        extra={},
    )


def _write_checkpoint(
    directory: Path,
    context: dict[str, Any],
    model: PPO,
    *,
    policy_id: str,
    role: str,
    transitions: int,
    tables: dict[str, np.ndarray],
    extra: dict[str, Any],
) -> None:
    """Save a checkpoint taken after its rollout's optimization, then seal it."""

    job: TrainingJob = context["job"]
    spec: PPOBoundaryStudySpec = context["spec"]
    parent: PolicyArtifactMetadata = context["parent"]
    distributions = context["distributions"]
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / MODEL_FILENAME
    model.save(model_path)
    design = _child_design(spec, parent, policy_id, context["seed"], transitions)
    summary = {
        "training_kind": "curriculum-continuation",
        "study_id": spec.study_id,
        "tier": job.tier,
        "arm": job.arm,
        "replicate_id": job.replicate_id,
        "outer_round": OUTER_ROUND,
        "checkpoint_role": role,
        "checkpoint_selection": spec.continuation.checkpoint_selection,
        "additional_transitions": transitions,
        "fixed_budget": context["budget"],
        "parent": {
            "policy_id": parent.policy_id,
            "model_sha256": parent.model_sha256,
        },
        "training_task": {
            "environment_id": spec.environment_id,
            "perturbation_name": spec.perturbation_name,
            "theta_threshold_deg": spec.theta_threshold_deg,
            "condition_sampler": SAMPLER_VERSION,
            "training_reward": spec.training_reward,
            "nominal_spec_id": parent.design_spec.environment.spec_id,
        },
        "distribution": {
            "sha256": context["distributions_sha256"],
            "target_rule": distributions["target_rule_version"],
            "fallback": distributions["arms"][job.arm]["fallback"],
        },
        "seeds": {
            "policy_seed": context["seed"],
            "episode_seed_role": spec.seeds.roles["training"],
            "condition_stream_tag": spec.seeds.condition_stream_tag,
        },
        "resolved_ppo": context["settings"],
        "migration": context["migration"],
        "final_policy_sha256": state_sha256(model.policy.state_dict()),
        "final_optimizer_sha256": state_sha256(model.policy.optimizer.state_dict()),
        "realized_sampling": realized_sampling(
            tables, tuple(distributions["condition_ids"])
        ),
        "development_record": job.development_record,
        "source": job.source,
        **extra,
    }
    write_continued_artifact(design, model_path, summary)
    write_evaluation_seal(
        directory,
        {
            "study_id": spec.study_id,
            "tier": job.tier,
            "arm": job.arm,
            "replicate_id": job.replicate_id,
            "outer_round": OUTER_ROUND,
            "checkpoint_role": role,
            "additional_transitions": transitions,
            "parent_model_sha256": parent.model_sha256,
            "development_record": job.development_record,
        },
    )


def _child_design(
    spec: PPOBoundaryStudySpec,
    parent: PolicyArtifactMetadata,
    policy_id: str,
    seed: int,
    transitions: int,
) -> PolicyDesignSpec:
    """Keep the parent's nominal base; `training_task` holds the real task."""

    continuation = spec.continuation
    return replace(
        parent.design_spec,
        design_id=f"{spec.study_id}-continuation-design",
        policy_id=policy_id,
        training_seed=seed,
        training_steps=transitions,
        hyperparameters={
            **continuation.hyperparameters,
            "learning_rate": continuation.learning_rate,
            "clip_range": continuation.clip_range,
            "continues_policy_id": parent.policy_id,
            "continues_model_sha256": parent.model_sha256,
        },
    )


def _verify_first_update(
    recorder: OptimizerStepRecorder,
    capture: TransitionCapture,
    continuation: ContinuationSpec,
) -> dict[str, Any]:
    if recorder.first_learning_rate != continuation.learning_rate:
        raise RuntimeError(
            f"First optimizer step used LR {recorder.first_learning_rate}, "
            f"not {continuation.learning_rate}."
        )
    if recorder.learning_rates != {continuation.learning_rate}:
        raise RuntimeError(f"Optimizer LR varied: {recorder.learning_rates}.")
    first = capture.updates[0]
    if first["clip_range"] != continuation.clip_range:
        raise RuntimeError(
            f"First update used clip range {first['clip_range']}, "
            f"not {continuation.clip_range}."
        )
    if len(capture.updates) != len(capture.rollouts):
        raise RuntimeError("Every collected rollout must produce one recorded update.")
    return {
        "first_optimizer_step_learning_rate": recorder.first_learning_rate,
        "optimizer_steps": recorder.steps,
        "first_update": first,
        "last_update": capture.updates[-1],
    }


def _job_paths(study_dir: Path, job: TrainingJob) -> dict[str, Path]:
    raw_root = study_dir / TRAINING_RAW_DIRECTORY / job.tier
    work = raw_root / WORK_DIRECTORY / job.job_id
    return {
        "work": work,
        "policy_work": work / POLICY_WORK_DIRECTORY,
        "raw_work": work / RAW_WORK_DIRECTORY,
        "raw": raw_root / job.job_id,
        "policy": child_artifact_dir(study_dir, job.tier, job.job_id),
        "abandoned": raw_root / ABANDONED_DIRECTORY,
    }


def _job_complete(job: TrainingJob) -> bool:
    paths = _job_paths(Path(job.study_dir), job)
    if not paths["policy"].exists():
        return False
    verify_evaluation_seal(paths["policy"])
    summary = json.loads((paths["policy"] / TRAINING_SUMMARY_FILENAME).read_text())
    recorded = summary["transitions_file"]["file_sha256"]
    actual = _sha256(paths["raw"] / TRANSITIONS_FILENAME)
    if actual != recorded:
        raise ValueError(f"Training shard does not match sealed child {job.job_id}.")
    return True


def _shelve_incomplete(paths: dict[str, Path], job: TrainingJob) -> None:
    """Keep an interrupted attempt's files aside; never reuse or delete them."""

    leftovers = {"work": paths["work"], "published-raw": paths["raw"]}
    present = {name: path for name, path in leftovers.items() if path.exists()}
    if not present:
        return
    paths["abandoned"].mkdir(parents=True, exist_ok=True)
    attempt = sum(1 for _ in paths["abandoned"].glob(f"{job.job_id}.attempt-*")) + 1
    shelf = paths["abandoned"] / f"{job.job_id}.attempt-{attempt}"
    shelf.mkdir()
    for name, path in present.items():
        os.rename(path, shelf / name)


def _publish(paths: dict[str, Path]) -> None:
    """Publish raw data first: the sealed policy directory marks completion."""

    paths["raw"].parent.mkdir(parents=True, exist_ok=True)
    paths["policy"].parent.mkdir(parents=True, exist_ok=True)
    os.rename(paths["raw_work"], paths["raw"])
    os.rename(paths["policy_work"], paths["policy"])
    paths["work"].rmdir()


def _fingerprint(model: PPO) -> dict[str, Any]:
    batch = _migration_batch()
    logits, _, values = SB3Policy(model).cartpole_actor_critic(batch)
    actions, _ = model.predict(batch, deterministic=True)
    return {
        "policy_sha256": state_sha256(model.policy.state_dict()),
        "optimizer_sha256": state_sha256(model.policy.optimizer.state_dict()),
        "logits": logits,
        "values": values,
        "actions": actions,
    }


def _migration_batch() -> np.ndarray:
    bound = np.array(MIGRATION_BATCH_BOUND)
    generator = np.random.default_rng(MIGRATION_BATCH_SEED)
    return generator.uniform(
        -bound, bound, size=(MIGRATION_BATCH_SIZE, bound.size)
    ).astype(np.float32)


def _update_digest(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        digest.update(f"tensor|{array.dtype.str}|{array.shape}|".encode())
        digest.update(array.tobytes())
    elif isinstance(value, dict):
        digest.update(f"dict|{len(value)}|".encode())
        for key in sorted(value, key=str):
            digest.update(f"{key}=".encode())
            _update_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(f"list|{len(value)}|".encode())
        for item in value:
            _update_digest(digest, item)
    else:
        digest.update(f"{value!r}|".encode())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
