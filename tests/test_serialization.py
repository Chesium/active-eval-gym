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
    to_jsonable,
    write_episode,
    write_metrics,
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
    assert metadata["schema_version"] == 3
    assert rows[0]["type"] == "reset"
    assert "environment_state" in rows[0]
    assert "environment_state" in rows[1]
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
