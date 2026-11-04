"""Deterministic, human-readable episode artifact serialization."""

import gzip
import hashlib
import json
import math
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from active_eval_gym.metrics import EpisodeMetrics
from active_eval_gym.models import EpisodeRecord

METADATA_FILENAME = "metadata.json"
TRAJECTORY_FILENAME = "trajectory.jsonl"
METRICS_FILENAME = "metrics.json"
TRAJECTORY_DIGEST_FILENAME = "trajectory.sha256"
ARTIFACT_FILENAMES = (
    METADATA_FILENAME,
    TRAJECTORY_FILENAME,
    TRAJECTORY_DIGEST_FILENAME,
    METRICS_FILENAME,
)
EPISODE_SHARD_LAYOUT = "checkpoint-condition-shard-v1"
SHARD_HEADER_ROW = "shard-header"
SHARD_EPISODE_ROW = "episode"
SHARD_FOOTER_ROW = "shard-footer"


@dataclass(frozen=True)
class SavedEpisode:
    """A hash-verified episode loaded without constructing an environment."""

    metadata: dict[str, Any]
    reset: dict[str, Any]
    transitions: tuple[dict[str, Any], ...]
    trajectory_sha256: str


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
    _write_new(
        output_dir / TRAJECTORY_DIGEST_FILENAME,
        (trajectory_hash + "\n").encode(),
    )
    return trajectory_hash


def write_metrics(output_dir: Path, metrics: EpisodeMetrics) -> None:
    """Write derived metrics separately from the raw trajectory."""

    _write_new(output_dir / METRICS_FILENAME, _pretty_json_bytes(metrics))


def read_saved_episode(output_dir: Path) -> SavedEpisode:
    """Load a supported raw episode and verify its recorded trajectory digest."""

    metadata = json.loads((output_dir / METADATA_FILENAME).read_text())
    if metadata.get("schema_version") not in (3, 4):
        raise ValueError("Sweep analysis requires episode schema version 3 or 4.")
    trajectory_bytes = (output_dir / TRAJECTORY_FILENAME).read_bytes()
    actual_hash = hashlib.sha256(trajectory_bytes).hexdigest()
    expected_hash = (output_dir / TRAJECTORY_DIGEST_FILENAME).read_text().strip()
    if actual_hash != expected_hash:
        raise ValueError(
            f"Trajectory digest mismatch in {output_dir}: "
            f"expected {expected_hash}, received {actual_hash}."
        )
    return _saved_episode(metadata, trajectory_bytes, actual_hash, str(output_dir))


def write_episode_shard(
    path: Path,
    header: Mapping[str, Any],
    episodes: Iterable[EpisodeRecord],
    *,
    compression_level: int,
) -> int:
    """Atomically write episodes to one reproducible, immutable gzip shard.

    Each episode keeps the exact bytes `write_episode` would digest. The shard
    appears under its final name only once every planned episode is on disk.
    """

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    count = 0
    try:
        with partial.open("wb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=compression_level,
                fileobj=raw,
                mtime=0,
            ) as stream:
                stream.write(
                    _row_bytes(
                        {
                            "type": SHARD_HEADER_ROW,
                            "layout": EPISODE_SHARD_LAYOUT,
                            **header,
                        }
                    )
                )
                for episode in episodes:
                    trajectory = _trajectory_bytes(episode)
                    stream.write(
                        _row_bytes(
                            {
                                "type": SHARD_EPISODE_ROW,
                                "metadata": episode.metadata,
                                "trajectory_bytes": len(trajectory),
                                "trajectory_sha256": hashlib.sha256(
                                    trajectory
                                ).hexdigest(),
                            }
                        )
                    )
                    stream.write(trajectory)
                    count += 1
                stream.write(
                    _row_bytes({"type": SHARD_FOOTER_ROW, "episode_count": count})
                )
            raw.flush()
            os.fsync(raw.fileno())
        try:
            os.link(partial, path)
        except FileExistsError as error:
            message = f"Refusing to overwrite existing artifact: {path}."
            raise FileExistsError(message) from error
    finally:
        partial.unlink(missing_ok=True)
    return count


def read_shard_header(path: Path) -> dict[str, Any]:
    """Return a shard's identity without reading its episodes."""

    with gzip.open(path, "rb") as stream:
        header = json.loads(stream.readline())
    if header.get("type") != SHARD_HEADER_ROW:
        raise ValueError(f"Shard {path} does not begin with a header.")
    if header.get("layout") != EPISODE_SHARD_LAYOUT:
        raise ValueError(f"Shard {path} has unsupported layout {header.get('layout')}.")
    return header


def read_episode_shard(path: Path) -> Iterator[SavedEpisode]:
    """Yield hash-verified episodes, failing unless the shard is complete."""

    count = 0
    try:
        with gzip.open(path, "rb") as stream:
            stream.readline()
            while True:
                line = stream.readline()
                if not line:
                    raise ValueError(f"Shard {path} has no footer.")
                row = json.loads(line)
                if row.get("type") == SHARD_FOOTER_ROW:
                    break
                if row.get("type") != SHARD_EPISODE_ROW:
                    raise ValueError(f"Shard {path} contains an invalid row type.")
                trajectory = stream.read(row["trajectory_bytes"])
                actual_hash = hashlib.sha256(trajectory).hexdigest()
                if actual_hash != row["trajectory_sha256"]:
                    raise ValueError(
                        f"Trajectory digest mismatch in {path}: expected "
                        f"{row['trajectory_sha256']}, received {actual_hash}."
                    )
                yield _saved_episode(
                    row["metadata"], trajectory, actual_hash, str(path)
                )
                count += 1
            if row["episode_count"] != count or stream.read(1):
                raise ValueError(f"Shard {path} does not match its footer.")
    except EOFError as error:
        raise ValueError(f"Shard {path} is truncated.") from error


def write_json_new(path: Path, value: Any) -> None:
    """Write strict, deterministic JSON without replacing an existing artifact."""

    _write_new(path, _pretty_json_bytes(value))


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


def _saved_episode(
    metadata: dict[str, Any], trajectory_bytes: bytes, digest: str, source: str
) -> SavedEpisode:
    rows = [json.loads(line) for line in trajectory_bytes.splitlines()]
    if not rows or rows[0].get("type") != "reset":
        raise ValueError(f"Trajectory in {source} does not begin with reset.")
    transitions = tuple(rows[1:])
    if any(row.get("type") != "transition" for row in transitions):
        raise ValueError(f"Trajectory in {source} contains an invalid row type.")
    return SavedEpisode(
        metadata=metadata,
        reset=rows[0],
        transitions=transitions,
        trajectory_sha256=digest,
    )


def _trajectory_bytes(episode: EpisodeRecord) -> bytes:
    rows: list[dict[str, Any]] = [
        {
            "type": "reset",
            "observation": episode.reset.observation,
            "environment_state": episode.reset.environment_state,
            "perturbation_diagnostics": episode.reset.perturbation_diagnostics,
            "info": episode.reset.info,
        }
    ]
    rows.extend(
        {
            "type": "transition",
            "action": transition.action,
            "environment_action": transition.environment_action,
            "policy_diagnostics": transition.policy_diagnostics,
            "perturbation_diagnostics": transition.perturbation_diagnostics,
            "reward": transition.reward,
            "observation": transition.observation,
            "environment_state": transition.environment_state,
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


def _row_bytes(value: Any) -> bytes:
    text = json.dumps(
        to_jsonable(value), allow_nan=False, separators=(",", ":"), sort_keys=True
    )
    return (text + "\n").encode()


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
