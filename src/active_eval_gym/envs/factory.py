"""Construction of the deliberately small environment set."""

from importlib import import_module
from typing import Any

import gymnasium as gym
import numpy as np

from active_eval_gym.envs.perturbations import (
    NO_OP,
    PerturbationSpec,
    apply_perturbation,
)

SUPPORTED_ENVIRONMENTS = (
    "CartPole-v1",
    "Pendulum-v1",
    "MiniGrid-Empty-8x8-v0",
)


def make_environment(
    env_id: str,
    perturbation: PerturbationSpec = NO_OP,
    *,
    render_mode: str | None = None,
) -> gym.Env:
    """Create a supported environment and apply its explicit perturbation."""

    if env_id not in SUPPORTED_ENVIRONMENTS:
        supported = ", ".join(SUPPORTED_ENVIRONMENTS)
        raise ValueError(f"Unsupported environment {env_id!r}. Supported: {supported}.")

    if env_id.startswith("MiniGrid-"):
        import_module("minigrid")

    env = gym.make(env_id, render_mode=render_mode)
    try:
        return apply_perturbation(env, perturbation)
    except Exception:
        env.close()
        raise


def capture_environment_state(env: gym.Env, env_id: str) -> dict[str, Any]:
    """Read the current interpretable state of a supported environment."""

    base_env = env.unwrapped
    if env_id == "CartPole-v1":
        state = _required_state(base_env, env_id, expected_size=4)
        return {
            "cart_position": float(state[0]),
            "cart_velocity": float(state[1]),
            "pole_angle": float(state[2]),
            "pole_angular_velocity": float(state[3]),
        }
    if env_id == "Pendulum-v1":
        state = _required_state(base_env, env_id, expected_size=2)
        return {
            "angle": float(state[0]),
            "angular_velocity": float(state[1]),
        }
    if env_id == "MiniGrid-Empty-8x8-v0":
        position = getattr(base_env, "agent_pos", None)
        direction = getattr(base_env, "agent_dir", None)
        if position is None or direction is None:
            raise RuntimeError(f"{env_id} did not expose its agent initial state.")
        return {
            "agent_position": [int(position[0]), int(position[1])],
            "agent_direction": int(direction),
        }
    raise ValueError(f"Cannot capture state for unsupported environment {env_id!r}.")


# Compatibility name for callers that only capture state immediately after reset.
capture_initial_state = capture_environment_state


def _required_state(env: Any, env_id: str, *, expected_size: int) -> np.ndarray:
    state = getattr(env, "state", None)
    if state is None:
        raise RuntimeError(f"{env_id} did not expose its initial state after reset.")
    array = np.asarray(state)
    if array.size != expected_size:
        raise RuntimeError(
            f"{env_id} exposed {array.size} state values; expected {expected_size}."
        )
    return array
