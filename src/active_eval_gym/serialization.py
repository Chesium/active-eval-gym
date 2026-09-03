"""Deterministic, human-readable episode artifact serialization."""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from active_eval_gym.metrics import EpisodeMetrics
from active_eval_gym.models import EpisodeRecord

METADATA_FILENAME = "metadata.json"
TRAJECTORY_FILENAME = "trajectory.jsonl"
METRICS_FILENAME = "metrics.json"
ARTIFACT_FILENAMES = (METADATA_FILENAME, TRAJECTORY_FILENAME, METRICS_FILENAME)


def ensure_output_available(output_dir: Path) -> None:
    """Fail before collection if any artifact target already exists."""

    existing = [name for name in ARTIFACT_FILENAMES if (output_dir / name).exists()]
    if existing:
        names = ", ".join(existing)
        raise FileExistsError(
            f"Refusing to overwrite existing artifacts in {output_dir}: {names}."
        )


def write_episode(output_dir: Path, episode: EpisodeRecord) -> str:
    """Write provenance and raw trajectory, returning the trajectory digest."""

    ensure_output_available(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_bytes = _pretty_json_bytes(episode.metadata)
    trajectory_bytes = _trajectory_bytes(episode)
    trajectory_hash = hashlib.sha256(trajectory_bytes).hexdigest()

    _write_new(output_dir / METADATA_FILENAME, metadata_bytes)
    _write_new(output_dir / TRAJECTORY_FILENAME, trajectory_bytes)
    return trajectory_hash


def write_metrics(output_dir: Path, metrics: EpisodeMetrics) -> None:
    """Write derived metrics separately from the raw trajectory."""

    _write_new(output_dir / METRICS_FILENAME, _pretty_json_bytes(metrics))


def to_jsonable(value: Any, *, path: str = "$") -> Any:
    """Convert supported scientific Python values to strict JSON values."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_jsonable(
                getattr(value, field.name), path=f"{path}.{field.name}"
            )
            for field in fields(value)
        }
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist(), path=path)
    if isinstance(value, np.generic):
        return to_jsonable(value.item(), path=path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path}: non-finite floats are not valid artifacts.")
        return value
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path}: mapping key {key!r} is not a string.")
            converted[key] = to_jsonable(item, path=f"{path}.{key}")
        return converted
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            to_jsonable(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path}: unsupported value of type {type(value).__name__}.")


def _trajectory_bytes(episode: EpisodeRecord) -> bytes:
    rows: list[dict[str, Any]] = [
        {
            "type": "reset",
            "observation": episode.reset.observation,
            "info": episode.reset.info,
        }
    ]
    rows.extend(
        {
            "type": "transition",
            "action": transition.action,
            "reward": transition.reward,
            "observation": transition.observation,
            "terminated": transition.terminated,
            "truncated": transition.truncated,
            "info": transition.info,
        }
        for transition in episode.transitions
    )
    lines = [
        json.dumps(
            to_jsonable(row, path=f"trajectory[{index}]"),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        for index, row in enumerate(rows)
    ]
    return ("\n".join(lines) + "\n").encode()


def _pretty_json_bytes(value: Any) -> bytes:
    text = json.dumps(to_jsonable(value), allow_nan=False, indent=2, sort_keys=True)
    return (text + "\n").encode()


def _write_new(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as file:
            file.write(content)
    except FileExistsError as error:
        message = f"Refusing to overwrite existing artifact: {path}."
        raise FileExistsError(message) from error
