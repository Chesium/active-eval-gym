"""Build, train, freeze, verify, and load immutable policy artifacts."""

import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from active_eval_gym.config import load_policy_design
from active_eval_gym.models import (
    NominalEnvSpec,
    PolicyArtifactMetadata,
    PolicyDesignSpec,
)
from active_eval_gym.policies.lqr import QuantizedLQRPolicy, design_cartpole_lqr
from active_eval_gym.policies.sb3 import load_sb3_policy, train_sb3_policy
from active_eval_gym.serialization import to_jsonable

MANIFEST_FILENAME = "manifest.json"
TRAINING_SUMMARY_FILENAME = "training-summary.json"
FREEZE_FILENAME = "freeze.json"
FREEZE_FAILURE_FILENAME = "freeze-failure.json"


def build_lqr_artifact(config_path: Path, output_dir: Path) -> PolicyArtifactMetadata:
    """Build a candidate LQR artifact from a tracked design spec."""

    design = load_policy_design(config_path)
    if design.policy_type != "quantized_lqr" or design.algorithm != "DLQR":
        raise ValueError("build-policy requires a quantized_lqr/DLQR design.")
    _prepare_new_directory(output_dir)
    model, linearization_error = design_cartpole_lqr(design)
    model_path = output_dir / "model.json"
    _write_json_new(model_path, model)
    metadata = _artifact_metadata(design, model_path, "active-eval-lqr-json-v1")
    _write_json_new(output_dir / MANIFEST_FILENAME, metadata)
    _write_json_new(
        output_dir / TRAINING_SUMMARY_FILENAME,
        {
            "schema_version": 1,
            "policy_id": design.policy_id,
            "build_kind": "analytic_discrete_lqr",
            "finite_difference_max_abs_error": linearization_error,
            "checkpoint_selection": "built_once_from_nominal_design",
        },
    )
    return metadata


def train_learned_artifact(
    config_path: Path, output_dir: Path
) -> PolicyArtifactMetadata:
    """Train one configured SB3 model and save it as a candidate artifact."""

    design = load_policy_design(config_path)
    if design.policy_type != "sb3":
        raise ValueError("train-policy requires a policy_type='sb3' design.")
    _prepare_new_directory(output_dir)
    model_path = output_dir / "model.zip"
    summary = train_sb3_policy(design, model_path)
    if not model_path.is_file():
        raise RuntimeError(f"Stable-Baselines3 did not create {model_path}.")
    metadata = _artifact_metadata(design, model_path, "stable-baselines3-zip-v1")
    _write_json_new(output_dir / MANIFEST_FILENAME, metadata)
    _write_json_new(
        output_dir / TRAINING_SUMMARY_FILENAME,
        {"schema_version": 1, "policy_id": design.policy_id, **summary},
    )
    return metadata


def load_policy_artifact(
    artifact_dir: Path, *, require_frozen: bool = True
) -> tuple[Any, PolicyArtifactMetadata]:
    """Verify artifact integrity and load an inference-only policy."""

    manifest_path = artifact_dir / MANIFEST_FILENAME
    manifest_bytes = _read_required_bytes(manifest_path)
    metadata = _metadata_from_dict(json.loads(manifest_bytes))
    if Path(metadata.model_filename).name != metadata.model_filename:
        raise ValueError("Artifact model_filename must be a basename.")
    model_path = artifact_dir / metadata.model_filename
    actual_hash = _sha256(model_path)
    if actual_hash != metadata.model_sha256:
        raise ValueError(
            f"Policy model digest mismatch: expected {metadata.model_sha256}, "
            f"received {actual_hash}."
        )
    if require_frozen:
        _verify_freeze(artifact_dir, metadata, manifest_bytes)

    if metadata.artifact_format == "active-eval-lqr-json-v1":
        model = json.loads(_read_required_bytes(model_path))
        gain = np.asarray(model["k"], dtype=np.float64)
        force = float(metadata.design_spec.environment.parameters["force_mag"])
        policy = QuantizedLQRPolicy(gain, force_magnitude=force)
    elif metadata.artifact_format == "stable-baselines3-zip-v1":
        policy = load_sb3_policy(metadata.design_spec.algorithm, model_path)
    else:
        raise ValueError(f"Unknown artifact format {metadata.artifact_format!r}.")
    return policy, metadata


def write_freeze_result(
    artifact_dir: Path,
    *,
    passed: bool,
    gate: Any,
    results: dict[str, Any],
) -> Path:
    """Write a hash-bound freeze marker or a failure report."""

    manifest_path = artifact_dir / MANIFEST_FILENAME
    manifest_bytes = _read_required_bytes(manifest_path)
    metadata = _metadata_from_dict(json.loads(manifest_bytes))
    payload = {
        "schema_version": 1,
        "policy_id": metadata.policy_id,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "model_sha256": metadata.model_sha256,
        "quality_gate": gate,
        "nominal_results": results,
        "passed": passed,
    }
    filename = FREEZE_FILENAME if passed else FREEZE_FAILURE_FILENAME
    path = artifact_dir / filename
    _write_json_new(path, payload)
    return path


def _artifact_metadata(
    design: PolicyDesignSpec, model_path: Path, artifact_format: str
) -> PolicyArtifactMetadata:
    packages = dict(design.environment_package_versions)
    packages["numpy"] = version("numpy")
    if design.algorithm_library == "stable-baselines3":
        packages["torch"] = version("torch")
    return PolicyArtifactMetadata(
        schema_version=1,
        policy_id=design.policy_id,
        design_spec=design,
        artifact_format=artifact_format,
        model_filename=model_path.name,
        model_sha256=_sha256(model_path),
        package_versions=packages,
        source_version=_source_version(),
    )


def _verify_freeze(
    artifact_dir: Path,
    metadata: PolicyArtifactMetadata,
    manifest_bytes: bytes,
) -> None:
    freeze = json.loads(_read_required_bytes(artifact_dir / FREEZE_FILENAME))
    expected_manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if not freeze.get("passed"):
        raise ValueError(f"Policy artifact {metadata.policy_id!r} is not frozen.")
    if freeze.get("policy_id") != metadata.policy_id:
        raise ValueError("Freeze record policy ID does not match its manifest.")
    if freeze.get("manifest_sha256") != expected_manifest_hash:
        raise ValueError("Freeze record does not match the current manifest.")
    if freeze.get("model_sha256") != metadata.model_sha256:
        raise ValueError("Freeze record does not match the current model.")


def _metadata_from_dict(data: dict[str, Any]) -> PolicyArtifactMetadata:
    design_data = data["design_spec"]
    env_data = design_data["environment"]
    nominal = NominalEnvSpec(**env_data)
    design = PolicyDesignSpec(**{**design_data, "environment": nominal})
    return PolicyArtifactMetadata(**{**data, "design_spec": design})


def _prepare_new_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite policy artifact directory: {path}."
        )
    path.mkdir(parents=True)


def _write_json_new(path: Path, value: Any) -> None:
    content = (
        json.dumps(to_jsonable(value), allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    try:
        with path.open("x", encoding="utf-8") as file:
            file.write(content)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite artifact: {path}.") from error


def _read_required_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Required policy artifact file is missing: {path}."
        ) from error


def _sha256(path: Path) -> str:
    return hashlib.sha256(_read_required_bytes(path)).hexdigest()


def _source_version() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = (
            subprocess.run(
                ["git", "diff", "--quiet"],
                check=False,
            ).returncode
            != 0
        )
        untracked = bool(
            subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"git_commit": commit, "git_dirty": dirty or untracked}
    except (OSError, subprocess.CalledProcessError):
        return {"package_version": version("active-eval-gym"), "git_dirty": None}
