"""Protocol layer for the CartPole PPO boundary-targeting study.

This module fixes what a study run may do: its conditions, seed namespaces, and
planned job counts. It records them immutably before any study data exist.
Collection, training, and analysis stay in later, separate commands.
"""

import hashlib
import json
import os
import platform
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from active_eval_gym.config import (
    PPOBoundaryStudySpec,
    SweepSuiteSpec,
    load_ppo_boundary_study,
)
from active_eval_gym.policies.artifacts import (
    FREEZE_FILENAME,
    MANIFEST_FILENAME,
    TRAINING_SUMMARY_FILENAME,
    load_policy_artifact,
)
from active_eval_gym.serialization import write_json_new
from active_eval_gym.sweeps import SweepCondition, expand_sweep

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = ("src", "configs", "scripts", "pyproject.toml", "uv.lock")
SOURCE_SUFFIXES = {".py", ".toml", ".json", ".lock"}
RECORDED_PACKAGES = ("gymnasium", "numpy", "scipy", "stable-baselines3", "torch")
PROTOCOL_SCHEMA_VERSION = 1
POLICIES_DIRECTORY = "policies"

MAX_ROLE = 10
MAX_OUTER_ROUND = 100
MAX_REPLICATE_ID = 1000
MAX_WORKER_ID = 100
MAX_EPISODE_INDEX = 100_000


def episode_seed(
    *,
    role: int,
    episode_index: int,
    outer_round: int = 0,
    replicate_id: int = 0,
    worker_id: int = 0,
) -> int:
    """Return one episode-reset seed, failing before namespaces can overlap."""

    _require_index(role, "role", MAX_ROLE)
    if role == 0:
        raise ValueError("role must be positive.")
    _require_index(outer_round, "outer_round", MAX_OUTER_ROUND)
    _require_index(replicate_id, "replicate_id", MAX_REPLICATE_ID)
    _require_index(worker_id, "worker_id", MAX_WORKER_ID)
    _require_index(episode_index, "episode_index", MAX_EPISODE_INDEX)
    return (
        role * 10**12
        + outer_round * 10**10
        + replicate_id * 10**7
        + worker_id * 10**5
        + episode_index
    )


def evaluation_episode_seeds(
    role: int, count: int, *, outer_round: int = 0
) -> tuple[int, ...]:
    """Return the paired reset list shared by every condition and checkpoint."""

    return tuple(
        episode_seed(role=role, episode_index=index, outer_round=outer_round)
        for index in range(count)
    )


def training_policy_seed(
    spec: PPOBoundaryStudySpec, *, replicate_id: int, outer_round: int = 0
) -> int:
    """Return the SB3 seed, shared across arms within one replicate."""

    _require_index(replicate_id, "replicate_id", MAX_REPLICATE_ID)
    _require_index(outer_round, "outer_round", MAX_OUTER_ROUND)
    return spec.seeds.training_policy_seed_base + 1000 * outer_round + replicate_id


def condition_seed_sequence(
    spec: PPOBoundaryStudySpec,
    *,
    replicate_id: int,
    worker_id: int,
    outer_round: int = 0,
) -> np.random.SeedSequence:
    """Return a worker's condition stream; arm is omitted to pair its draws."""

    _require_index(outer_round, "outer_round", MAX_OUTER_ROUND)
    _require_index(replicate_id, "replicate_id", MAX_REPLICATE_ID)
    _require_index(worker_id, "worker_id", MAX_WORKER_ID)
    return np.random.SeedSequence(
        [
            spec.seeds.master_seed,
            outer_round,
            replicate_id,
            worker_id,
            spec.seeds.condition_stream_tag,
        ]
    )


def tier_children(
    spec: PPOBoundaryStudySpec, tier_name: str
) -> tuple[dict[str, Any], ...]:
    """Return one tier's planned children in a fixed replicate and arm order."""

    tier = spec.tier(tier_name)
    return tuple(
        {
            "policy_id": f"{spec.study_id}-{tier.name}-r{replicate:03d}-{arm}",
            "tier": tier.name,
            "arm": arm,
            "replicate_id": replicate,
        }
        for replicate in tier.replicate_ids
        for arm in spec.curriculum.arms
    )


def child_artifact_dir(study_dir: Path, tier_name: str, policy_id: str) -> Path:
    """Return where a child's sealed artifact is published."""

    return study_dir / POLICIES_DIRECTORY / tier_name / policy_id


def support_conditions(
    spec: PPOBoundaryStudySpec, name: str
) -> tuple[SweepCondition, ...]:
    """Expand one declared support in its fixed condition order."""

    support = spec.supports[name]
    conditions = expand_sweep(
        _recovery_suite(spec, support.initial_theta_deg, support.length),
        spec.parent_design.environment,
    )
    for angle, length in support.extra_points:
        conditions += expand_sweep(
            _recovery_suite(spec, (angle,), (length,)),
            spec.parent_design.environment,
        )
    identifiers = [condition.condition_id for condition in conditions]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Support {name!r} repeats a condition.")
    return conditions


def reference_weights(spec: PPOBoundaryStudySpec) -> np.ndarray:
    """Return the equal finite-grid coverage weights over reference conditions."""

    count = len(support_conditions(spec, "reference"))
    return np.full(count, 1.0 / count)


def job_accounting(spec: PPOBoundaryStudySpec) -> dict[str, Any]:
    """Return exact planned jobs and worst-case volume for every tier."""

    steps = spec.parent_design.environment.max_episode_steps
    arms = len(spec.curriculum.arms)
    tiers: dict[str, Any] = {}
    for tier in spec.tiers:
        conditions = len(support_conditions(spec, tier.evaluation_support))
        children = len(tier.replicate_ids) * arms
        snapshots: dict[str, Any] = {"status": tier.snapshot_evaluation}
        if tier.snapshot_evaluation != "none":
            snapshots |= _evaluation_block(
                children * len(tier.snapshot_transitions),
                conditions,
                tier.episodes_per_condition,
                steps,
            )
        tiers[tier.name] = {
            "replicates": len(tier.replicate_ids),
            "children": children,
            "training_transitions": children * tier.additional_transitions,
            # One extra checkpoint: the parent is evaluated alongside its children.
            "primary_evaluation": _evaluation_block(
                children + 1, conditions, tier.episodes_per_condition, steps
            ),
            "snapshot_evaluation": snapshots,
        }
    family = tiers["confirmatory"]["primary_evaluation"]["jobs"]
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "discovery": _evaluation_block(
            1,
            len(support_conditions(spec, "discovery")),
            spec.curriculum.discovery_episodes_per_condition,
            steps,
        ),
        "tiers": tiers,
        "confidence_family": {
            "interval": spec.confidence.interval,
            "family_alpha": spec.confidence.family_alpha,
            "checkpoint_condition_pairs": family,
            "pair_alpha": spec.confidence.family_alpha / family,
        },
    }


def seed_plan(spec: PPOBoundaryStudySpec) -> dict[str, Any]:
    """Return the declared seed namespaces and every tier's concrete seeds."""

    roles = spec.seeds.roles
    discovery = evaluation_episode_seeds(
        roles["discovery"], spec.curriculum.discovery_episodes_per_condition
    )
    evaluation = {"discovery": _seed_range(roles["discovery"], discovery)}
    training: dict[str, dict[str, int]] = {}
    for tier in spec.tiers:
        role = roles[tier.seed_role]
        seeds = evaluation_episode_seeds(role, tier.episodes_per_condition)
        evaluation[tier.name] = _seed_range(role, seeds)
        training[tier.name] = {
            str(replicate): training_policy_seed(spec, replicate_id=replicate)
            for replicate in tier.replicate_ids
        }
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "master_seed": spec.seeds.master_seed,
        "roles": roles,
        "episode_seed_limits": {
            "outer_round": MAX_OUTER_ROUND,
            "replicate_id": MAX_REPLICATE_ID,
            "worker_id": MAX_WORKER_ID,
            "episode_index": MAX_EPISODE_INDEX,
        },
        "evaluation_episode_seeds": evaluation,
        "training_policy_seeds": training,
        "condition_stream_entropy": [
            "master_seed",
            "outer_round",
            "replicate_id",
            "worker_id",
            spec.seeds.condition_stream_tag,
        ],
    }


def source_state() -> dict[str, Any]:
    """Return the commit and whether any study-affecting source differs from it."""

    try:
        commit = _git("rev-parse", "HEAD").strip()
        changes = _git("status", "--porcelain", "--", *SOURCE_PATHS).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "source_dirty": True, "changed_paths": []}
    return {
        "git_commit": commit,
        "source_dirty": bool(changes),
        "changed_paths": sorted(line[3:] for line in changes),
    }


def source_hashes() -> dict[str, str]:
    """Return the digest of every file that can change a study result."""

    hashes = {}
    for entry in SOURCE_PATHS:
        root = REPOSITORY_ROOT / entry
        paths = sorted(root.rglob("*")) if root.is_dir() else [root]
        for path in paths:
            if (
                path.is_file()
                and path.suffix in SOURCE_SUFFIXES
                and "__pycache__" not in path.parts
            ):
                hashes[str(path.relative_to(REPOSITORY_ROOT))] = _sha256(path)
    return hashes


def verify_parent(spec: PPOBoundaryStudySpec, artifact_root: Path) -> dict[str, Any]:
    """Verify the declared parent, refusing to continue from any other model."""

    directory = artifact_root / spec.parent_policy_id
    _, metadata = load_policy_artifact(directory, require_frozen=True)
    if metadata.model_sha256 != spec.parent_model_sha256:
        raise ValueError(
            f"Parent model digest {metadata.model_sha256} differs from the declared "
            f"{spec.parent_model_sha256}. Do not substitute another checkpoint."
        )
    if metadata.design_spec.hyperparameters != spec.parent_design.hyperparameters:
        raise ValueError("Parent artifact and parent design hyperparameters differ.")
    return {
        "policy_id": metadata.policy_id,
        "model_sha256": metadata.model_sha256,
        "file_sha256": {
            name: _sha256(directory / name)
            for name in (MANIFEST_FILENAME, FREEZE_FILENAME, TRAINING_SUMMARY_FILENAME)
        },
        "source_version": metadata.source_version,
        "package_versions": metadata.package_versions,
        "package_version_differences": _version_differences(metadata.package_versions),
    }


def load_protocol(
    config_path: Path, study_dir: Path
) -> tuple[PPOBoundaryStudySpec, dict[str, Any]]:
    """Load a prepared study, refusing a config its protocol did not record."""

    provenance_path = study_dir / "protocol" / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"Study has no prepared protocol: {study_dir}.")
    provenance = json.loads(provenance_path.read_text())
    actual = _sha256(config_path)
    if actual != provenance["config_sha256"]:
        raise ValueError(
            f"Config {config_path} has digest {actual}, but the protocol recorded "
            f"{provenance['config_sha256']}."
        )
    return load_ppo_boundary_study(config_path), provenance


def prepare_study(
    config_path: Path,
    *,
    artifact_root: Path,
    output_dir: Path,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Write the immutable protocol record, without collecting any study data."""

    protocol_dir = output_dir / "protocol"
    if protocol_dir.exists():
        raise FileExistsError(f"Refusing to overwrite protocol: {protocol_dir}.")
    spec = load_ppo_boundary_study(config_path)
    source = source_state()
    if source["source_dirty"] and not allow_dirty:
        raise ValueError(
            "Study source differs from its commit: "
            f"{source['changed_paths']}. Commit it, or pass allow_dirty for a "
            "development record that must not back confirmatory results."
        )
    records = {
        "resolved-config.json": spec,
        "conditions.json": _conditions_record(spec),
        "seed-plan.json": seed_plan(spec),
        "job-accounting.json": job_accounting(spec),
        "provenance.json": {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "study_id": spec.study_id,
            "development_record": source["source_dirty"],
            "source": source,
            "source_sha256": source_hashes(),
            "config_sha256": _sha256(config_path),
            "parent": verify_parent(spec, artifact_root),
            "runtime": _runtime_record(spec, output_dir),
        },
    }
    protocol_dir.mkdir(parents=True)
    shutil.copyfile(config_path, protocol_dir / "study-config.toml")
    for name, record in records.items():
        write_json_new(protocol_dir / name, record)
    return {
        "protocol_dir": str(protocol_dir),
        "development_record": source["source_dirty"],
        "config_sha256": records["provenance.json"]["config_sha256"],
    }


def _recovery_suite(
    spec: PPOBoundaryStudySpec,
    angles: tuple[float, ...],
    lengths: tuple[float, ...],
) -> SweepSuiteSpec:
    return SweepSuiteSpec(
        schema_version=1,
        suite_id=spec.study_id,
        environment_id=spec.environment_id,
        perturbation_name=spec.perturbation_name,
        metric_version=spec.metric_version,
        seeds=(),
        policy_ids=(),
        grid={
            "initial_theta_deg": list(angles),
            "length": list(lengths),
            "theta_threshold_deg": [spec.theta_threshold_deg],
        },
        conditions=(),
        derived_policies={},
        boundary_study={},
    )


def _evaluation_block(
    checkpoints: int, conditions: int, episodes: int, max_steps: int
) -> dict[str, int]:
    total = checkpoints * conditions * episodes
    return {
        "checkpoints": checkpoints,
        "conditions": conditions,
        "episodes_per_condition": episodes,
        "jobs": checkpoints * conditions,
        "episodes": total,
        "max_transitions": total * max_steps,
    }


def _seed_range(role: int, seeds: tuple[int, ...]) -> dict[str, int]:
    return {"role": role, "count": len(seeds), "first": seeds[0], "last": seeds[-1]}


def _conditions_record(spec: PPOBoundaryStudySpec) -> dict[str, Any]:
    supports: dict[str, Any] = {}
    for name in spec.supports:
        conditions = support_conditions(spec, name)
        supports[name] = {
            "condition_count": len(conditions),
            "conditions": conditions,
        }
    supports["reference"]["weight"] = {
        "kind": "equal-finite-grid",
        "numerator": 1,
        "denominator": supports["reference"]["condition_count"],
    }
    return {"schema_version": PROTOCOL_SCHEMA_VERSION, "supports": supports}


def _runtime_record(spec: PPOBoundaryStudySpec, output_dir: Path) -> dict[str, Any]:
    import torch

    existing = next(
        parent for parent in (output_dir, *output_dir.parents) if parent.exists()
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
        "disk_free_bytes": shutil.disk_usage(existing).free,
        "packages": {name: version(name) for name in RECORDED_PACKAGES},
        "device": spec.continuation.device,
        "declared_torch_threads": spec.continuation.torch_threads,
        "process_torch_threads": torch.get_num_threads(),
        "vec_env_backend": spec.continuation.vec_env_backend,
    }


def _version_differences(recorded: dict[str, str]) -> dict[str, dict[str, str]]:
    return {
        name: {"parent": value, "current": version(name)}
        for name, value in recorded.items()
        if version(name) != value
    }


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_index(value: int, name: str, limit: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    if not 0 <= value < limit:
        raise ValueError(f"{name} must lie in [0, {limit}), not {value}.")
