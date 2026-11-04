"""Training-only duration correction and frozen-parent stochastic calibration."""

import hashlib
import json
from pathlib import Path

import numpy as np

from active_eval_gym.config import TransitionSamplingSpec

CALIBRATION_DIRECTORY = "sampling-calibration"
CALIBRATION_SUMMARY = "summary.json"


class InverseLengthSampler:
    """Share length estimates across DummyVecEnv workers, without drawing RNGs.

    Estimates update on completed episodes. The categorical law changes only at
    declared rollout boundaries; ongoing episodes retain their sampled plant.
    """

    def __init__(
        self,
        target_shares: np.ndarray,
        initial_lengths: np.ndarray,
        settings: TransitionSamplingSpec,
    ) -> None:
        target = np.asarray(target_shares, dtype=np.float64)
        lengths = np.asarray(initial_lengths, dtype=np.float64)
        if (
            target.ndim != 1
            or lengths.shape != target.shape
            or not np.all(np.isfinite(target))
            or not np.all(np.isfinite(lengths))
            or np.any(target < 0)
            or not np.isclose(target.sum(), 1.0, rtol=0, atol=1e-12)
            or np.any(lengths < 1)
            or np.any(lengths > 500)
        ):
            raise ValueError("Invalid target shares or initial episode lengths.")
        self.settings = settings
        self.target_shares = target.copy()
        self.length_estimates = lengths.copy()
        self.completed_episodes = np.zeros(len(target), dtype=np.int64)
        self.revision = 0
        self._set_probabilities()

    def observe_completion(self, condition_index: int, length: int) -> None:
        if not 1 <= length <= 500:
            raise ValueError("Only complete, uncensored episode lengths are usable.")
        alpha = self.settings.length_ema_alpha
        self.length_estimates[condition_index] = (1 - alpha) * self.length_estimates[
            condition_index
        ] + alpha * length
        self.completed_episodes[condition_index] += 1

    def refresh(self) -> None:
        self._set_probabilities()
        self.revision += 1

    def set_target_shares(self, target_shares: np.ndarray) -> None:
        """Stage a new frontier target for the next scheduled refresh.

        Keep duration estimates and completed-episode counts. Existing episodes
        retain their condition; their recorded reset probabilities stay valid.
        """

        target = np.asarray(target_shares, dtype=np.float64)
        if (
            target.shape != self.target_shares.shape
            or not np.all(np.isfinite(target))
            or np.any(target < 0)
            or not np.isclose(target.sum(), 1.0, rtol=0, atol=1e-12)
        ):
            raise ValueError("Invalid replacement target shares.")
        self.target_shares = target.copy()

    def _set_probabilities(self) -> None:
        weights = self.target_shares / np.maximum(
            self.length_estimates, self.settings.minimum_length
        )
        self.probabilities = weights / weights.sum()
        self.cdf = np.cumsum(self.probabilities)
        self.cdf[self.cdf >= self.cdf[-1]] = 1.0


def calibrate_lengths(
    config_path: Path,
    *,
    study_dir: Path,
    artifact_root: Path,
    allow_dirty: bool = False,
) -> dict:
    """Collect fresh stochastic episodes under the frozen parent, retaining raw.

    Each condition has its own action seed so skipping a completed shard on
    resume cannot change subsequent conditions' action streams.
    """

    import torch
    from stable_baselines3.common.utils import set_random_seed

    from active_eval_gym.evaluate import make_artifact_environment
    from active_eval_gym.policies.artifacts import load_policy_artifact
    from active_eval_gym.ppo_boundary import (
        episode_seed,
        load_protocol,
        source_hashes,
        source_state,
        support_conditions,
        verify_parent,
    )
    from active_eval_gym.ppo_boundary_evaluation import write_or_match_json
    from active_eval_gym.rollout import collect_episode
    from active_eval_gym.serialization import (
        read_episode_shard,
        read_shard_header,
        to_jsonable,
    )
    from active_eval_gym.serialization import (
        write_episode_shard as write_shard,
    )

    spec, provenance = load_protocol(config_path, study_dir)
    settings = spec.sampling
    if settings is None:
        raise ValueError("Calibration requires a transition sampling specification.")
    source = source_state()
    if not allow_dirty and (source["source_dirty"] or provenance["development_record"]):
        raise ValueError("Development calibration requires allow_dirty.")
    verify_parent(spec, artifact_root)
    conditions = support_conditions(spec, "discovery")
    directory = study_dir / CALIBRATION_DIRECTORY
    plan = {
        "study_id": spec.study_id,
        "protocol_config_sha256": provenance["config_sha256"],
        "parent_model_sha256": spec.parent_model_sha256,
        "settings": settings,
        "condition_ids": [c.condition_id for c in conditions],
        "source": source,
        "source_sha256": source_hashes(),
        "deterministic": False,
        "optimizer_updates": 0,
    }
    write_or_match_json(directory / "plan.json", plan)
    torch.set_num_threads(spec.continuation.torch_threads)
    policy, metadata = load_policy_artifact(
        artifact_root / spec.parent_policy_id, require_frozen=True
    )
    initial_lengths = []
    total_transitions = 0
    shards = []
    for index, condition in enumerate(conditions):
        action_seed = settings.calibration_action_seed + index
        seeds = [
            episode_seed(
                role=settings.calibration_seed_role,
                episode_index=index * settings.calibration_episodes_per_condition + e,
            )
            for e in range(settings.calibration_episodes_per_condition)
        ]
        header = {
            "study_id": spec.study_id,
            "campaign": "sampling-calibration",
            "checkpoint_id": spec.parent_policy_id,
            "model_sha256": spec.parent_model_sha256,
            "condition_id": condition.condition_id,
            "perturbation": to_jsonable(condition.perturbation),
            "episode_seeds": seeds,
            "action_seed": action_seed,
            "deterministic": False,
        }
        path = directory / "shards" / f"{condition.condition_id}.jsonl.gz"
        if path.exists():
            saved = read_shard_header(path)
            if any(saved.get(k) != v for k, v in header.items()):
                raise ValueError(f"Calibration shard identity differs: {path}.")
        else:
            set_random_seed(action_seed)
            env, resolved = make_artifact_environment(metadata, condition.perturbation)
            try:
                write_shard(
                    path,
                    header,
                    (
                        collect_episode(
                            env,
                            policy,
                            policy_artifact=metadata,
                            nominal_environment=metadata.design_spec.environment,
                            resolved_environment=resolved,
                            episode_seed=seed,
                            deterministic=False,
                        )
                        for seed in seeds
                    ),
                    compression_level=spec.storage.compression_level,
                )
            finally:
                env.close()
        episodes = list(read_episode_shard(path))
        if len(episodes) != settings.calibration_episodes_per_condition:
            raise ValueError(f"Incomplete calibration condition: {path}.")
        lengths = [len(e.transitions) for e in episodes]
        initial_lengths.append(float(np.mean(lengths)))
        total_transitions += sum(lengths)
        shards.append(
            {
                "condition_id": condition.condition_id,
                "path": str(path.relative_to(study_dir)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "episode_lengths": lengths,
                "action_seed": action_seed,
                "episode_seeds": seeds,
            }
        )
        if (index + 1) % 20 == 0 or index + 1 == len(conditions):
            print(f"Calibration: {index + 1}/{len(conditions)} conditions.", flush=True)
    verify_parent(spec, artifact_root)
    result = {
        "study_id": spec.study_id,
        "protocol_config_sha256": provenance["config_sha256"],
        "parent_model_sha256": spec.parent_model_sha256,
        "settings": settings,
        "condition_ids": [c.condition_id for c in conditions],
        "initial_mean_lengths": initial_lengths,
        "episodes": len(conditions) * settings.calibration_episodes_per_condition,
        "transitions": total_transitions,
        "optimizer_updates": 0,
        "budget_accounting": "calibration-only; excluded from continuation transitions",
        "shards": shards,
    }
    write_or_match_json(directory / CALIBRATION_SUMMARY, result)
    return result


def load_calibration(spec, provenance: dict, study_dir: Path) -> tuple[dict, str]:
    """Bind training to the prepared calibration and its raw shard digests."""

    from active_eval_gym.ppo_boundary import support_conditions
    from active_eval_gym.serialization import to_jsonable

    path = study_dir / CALIBRATION_DIRECTORY / CALIBRATION_SUMMARY
    if not path.is_file():
        raise FileNotFoundError(f"Training requires completed calibration: {path}.")
    content = path.read_bytes()
    record = json.loads(content)
    expected = {
        "study_id": spec.study_id,
        "protocol_config_sha256": provenance["config_sha256"],
        "parent_model_sha256": spec.parent_model_sha256,
        "settings": to_jsonable(spec.sampling),
        "condition_ids": [
            c.condition_id for c in support_conditions(spec, "discovery")
        ],
        "optimizer_updates": 0,
    }
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError("Calibration does not match the declared study.")
    if [s["condition_id"] for s in record["shards"]] != expected["condition_ids"]:
        raise ValueError("Calibration shard conditions differ.")
    for shard in record["shards"]:
        raw = study_dir / shard["path"]
        if hashlib.sha256(raw.read_bytes()).hexdigest() != shard["sha256"]:
            raise ValueError(f"Calibration raw digest mismatch: {raw}.")
    return record, hashlib.sha256(content).hexdigest()
