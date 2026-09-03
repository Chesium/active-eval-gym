from importlib import import_module
from typing import Any

import gymnasium as gym
import pytest

from active_eval_gym.envs.factory import SUPPORTED_ENVIRONMENTS, make_environment
from active_eval_gym.envs.perturbations import PerturbationSpec
from active_eval_gym.policies.base import ConstantPolicy, zero_action
from active_eval_gym.rollout import collect_episode
from active_eval_gym.serialization import to_jsonable
from tests.helpers import rollout_provenance


@pytest.mark.parametrize(
    ("env_id", "space_type"),
    [
        ("CartPole-v1", gym.spaces.Discrete),
        ("Pendulum-v1", gym.spaces.Box),
        ("MiniGrid-Empty-8x8-v0", gym.spaces.Discrete),
    ],
)
def test_factory_creates_supported_environments(
    env_id: str, space_type: type[gym.Space]
) -> None:
    env = make_environment(env_id)
    try:
        observation, info = env.reset(seed=7)
        assert env.spec is not None
        assert env.spec.id == env_id
        assert isinstance(env.action_space, space_type)
        assert env.observation_space.contains(observation)
        assert isinstance(info, dict)
    finally:
        env.close()


def test_factory_rejects_unsupported_environment() -> None:
    with pytest.raises(ValueError, match="Unsupported environment"):
        make_environment("Acrobot-v1")


def test_factory_rejects_unsupported_perturbation() -> None:
    with pytest.raises(ValueError, match="Unsupported perturbation"):
        make_environment("CartPole-v1", PerturbationSpec("gravity"))


def test_no_op_rejects_parameters() -> None:
    with pytest.raises(ValueError, match="does not accept parameters"):
        make_environment("CartPole-v1", PerturbationSpec("none", {"unexpected": 1}))


@pytest.mark.parametrize("env_id", SUPPORTED_ENVIRONMENTS)
def test_constant_policy_completes_episode(env_id: str) -> None:
    env = make_environment(env_id)
    try:
        artifact, nominal, resolved = rollout_provenance(env, env_id)
        episode = collect_episode(
            env,
            ConstantPolicy(zero_action(env.action_space)),
            policy_artifact=artifact,
            nominal_environment=nominal,
            resolved_environment=resolved,
            episode_seed=123,
        )
    finally:
        env.close()

    assert episode.transitions
    assert episode.transitions[-1].terminated or episode.transitions[-1].truncated
    assert episode.metadata.resolved_environment.environment_id == env_id
    assert episode.metadata.perturbation.name == "none"


@pytest.mark.parametrize("env_id", SUPPORTED_ENVIRONMENTS)
def test_no_op_matches_direct_environment(env_id: str) -> None:
    if env_id.startswith("MiniGrid-"):
        import_module("minigrid")
    direct = gym.make(env_id)
    wrapped = make_environment(env_id)
    try:
        direct_trace = _raw_trace(direct, seed=41)
        wrapped_trace = _raw_trace(wrapped, seed=41)
    finally:
        direct.close()
        wrapped.close()

    assert to_jsonable(wrapped_trace) == to_jsonable(direct_trace)


def _raw_trace(env: gym.Env, *, seed: int) -> list[dict[str, Any]]:
    observation, info = env.reset(seed=seed)
    trace = [{"observation": observation, "info": info}]
    action = zero_action(env.action_space)
    while True:
        observation, reward, terminated, truncated, info = env.step(action)
        trace.append(
            {
                "action": action,
                "observation": observation,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
            }
        )
        if terminated or truncated:
            return trace
