import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from active_eval_gym.envs.factory import make_environment
from active_eval_gym.metrics import METRIC_VERSION, compute_metrics
from active_eval_gym.policies.base import ConstantPolicy, zero_action
from active_eval_gym.rollout import collect_episode
from active_eval_gym.serialization import (
    METADATA_FILENAME,
    METRICS_FILENAME,
    TRAJECTORY_FILENAME,
    read_transition_shard,
    to_jsonable,
    transition_content_sha256,
    write_episode,
    write_metrics,
    write_transition_shard,
)
from tests.helpers import rollout_provenance


def test_identical_inputs_produce_byte_identical_artifacts(tmp_path: Path) -> None:
    first = _collect("CartPole-v1", seed=37)
    second = _collect("CartPole-v1", seed=37)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first_hash = write_episode(first_dir, first)
    write_metrics(
        first_dir,
        compute_metrics(first, source_trajectory_sha256=first_hash),
    )
    second_hash = write_episode(second_dir, second)
    write_metrics(
        second_dir,
        compute_metrics(second, source_trajectory_sha256=second_hash),
    )

    assert first_hash == second_hash
    for filename in (METADATA_FILENAME, TRAJECTORY_FILENAME, METRICS_FILENAME):
        assert (first_dir / filename).read_bytes() == (
            second_dir / filename
        ).read_bytes()


def test_different_episode_seeds_change_cartpole_initial_state() -> None:
    first = _collect("CartPole-v1", seed=1)
    second = _collect("CartPole-v1", seed=2)

    assert first.metadata.initial_state != second.metadata.initial_state


@pytest.mark.parametrize(
    "env_id", ["CartPole-v1", "Pendulum-v1", "MiniGrid-Empty-8x8-v0"]
)
def test_artifacts_round_trip_supported_observations(
    env_id: str, tmp_path: Path
) -> None:
    episode = _collect(env_id, seed=9)
    output = tmp_path / env_id
    trajectory_hash = write_episode(output, episode)
    metrics = compute_metrics(episode, source_trajectory_sha256=trajectory_hash)
    write_metrics(output, metrics)

    metadata = json.loads((output / METADATA_FILENAME).read_text())
    rows = [
        json.loads(line)
        for line in (output / TRAJECTORY_FILENAME).read_text().splitlines()
    ]
    saved_metrics = json.loads((output / METRICS_FILENAME).read_text())

    assert metadata["resolved_environment"]["environment_id"] == env_id
    assert metadata["schema_version"] == 4
    assert rows[0]["type"] == "reset"
    assert "environment_state" in rows[0]
    assert "environment_state" in rows[1]
    assert "perturbation_diagnostics" in rows[0]
    assert "perturbation_diagnostics" in rows[1]
    assert rows[1]["environment_action"] == rows[1]["action"]
    assert len(rows) == len(episode.transitions) + 1
    assert "episode_return" not in metadata
    assert saved_metrics["metric_version"] == METRIC_VERSION
    assert (
        saved_metrics["source_trajectory_sha256"]
        == hashlib.sha256((output / TRAJECTORY_FILENAME).read_bytes()).hexdigest()
    )

    raw_before_recomputation = (output / TRAJECTORY_FILENAME).read_bytes()
    recomputed = compute_metrics(episode, source_trajectory_sha256=trajectory_hash)
    assert recomputed == metrics
    assert (output / TRAJECTORY_FILENAME).read_bytes() == raw_before_recomputation


def test_serialization_refuses_to_overwrite(tmp_path: Path) -> None:
    episode = _collect("CartPole-v1", seed=5)
    output = tmp_path / "episode"
    write_episode(output, episode)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_episode(output, episode)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_strict_json_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match=r"\$\.reward"):
        to_jsonable({"reward": value})


def test_strict_json_reports_unsupported_value_path() -> None:
    with pytest.raises(TypeError, match=r"\$\.observation"):
        to_jsonable({"observation": object()})


def test_strict_json_converts_numpy_values() -> None:
    value = {
        "array": np.array([[1, 2], [3, 4]], dtype=np.int64),
        "scalar": np.float32(1.5),
    }

    assert to_jsonable(value) == {
        "array": [[1, 2], [3, 4]],
        "scalar": 1.5,
    }


def _collect(env_id: str, *, seed: int):
    env = make_environment(env_id)
    try:
        artifact, nominal, resolved = rollout_provenance(env, env_id)
        return collect_episode(
            env,
            ConstantPolicy(zero_action(env.action_space)),
            policy_artifact=artifact,
            nominal_environment=nominal,
            resolved_environment=resolved,
            episode_seed=seed,
        )
    finally:
        env.close()


def _columns() -> dict[str, np.ndarray]:
    return {
        "state": np.arange(8.0).reshape(2, 4),
        "action": np.array([0, 1], dtype=np.uint8),
        "rollout_policy_sha256": np.array(["a" * 64, "b" * 64], "U64"),
    }


def test_transition_shard_round_trips_columns_and_header(tmp_path: Path) -> None:
    path = tmp_path / "transitions.npz"
    digests = write_transition_shard(path, {"policy_id": "child"}, _columns())

    header, arrays = read_transition_shard(path)

    assert header["policy_id"] == "child"
    assert header["layout"] == "training-transition-shard-v1"
    assert header["content_sha256"] == digests["content_sha256"]
    assert digests["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert set(arrays) == set(_columns())
    for name, column in _columns().items():
        np.testing.assert_array_equal(arrays[name], column)
        assert arrays[name].dtype == column.dtype


def test_transition_shard_refuses_to_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "transitions.npz"
    write_transition_shard(path, {}, _columns())
    original = path.read_bytes()

    with pytest.raises(FileExistsError):
        write_transition_shard(path, {}, _columns())
    assert path.read_bytes() == original


def test_transition_shard_detects_changed_columns(tmp_path: Path) -> None:
    good = tmp_path / "good.npz"
    write_transition_shard(good, {}, _columns())
    with np.load(good) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["action"] = np.array([1, 1], dtype=np.uint8)
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, **arrays)

    with pytest.raises(ValueError, match="digest mismatch"):
        read_transition_shard(bad)


def test_transition_content_digest_covers_type_shape_and_name() -> None:
    base = np.zeros(4, dtype=np.float32)
    digest = transition_content_sha256({"a": base})

    assert digest != transition_content_sha256({"b": base})
    assert digest != transition_content_sha256({"a": base.reshape(2, 2)})
    assert digest != transition_content_sha256({"a": base.astype(np.int32)})
    assert digest == transition_content_sha256({"a": base.copy()})


def test_transition_shard_reserves_its_header_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reserved"):
        write_transition_shard(tmp_path / "x.npz", {}, {"__header__": np.zeros(1)})
