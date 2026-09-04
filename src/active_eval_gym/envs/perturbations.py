"""Structured, allowlisted environment perturbations."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import radians
from types import MappingProxyType
from typing import Any

import gymnasium as gym
import numpy as np

CARTPOLE_ANGLE_LENGTH = "cartpole-angle-length-v1"
PENDULUM_LENGTH = "pendulum-length-v1"
MINIGRID_START_POSE = "minigrid-start-pose-v1"


@dataclass(frozen=True)
class PerturbationSpec:
    """A serializable description of an evaluation perturbation."""

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Perturbation name must not be empty.")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


NO_OP = PerturbationSpec(name="none")


class NoOpPerturbation(gym.Wrapper):
    """A transparent wrapper that records the nominal perturbation."""

    def __init__(self, env: gym.Env, spec: PerturbationSpec = NO_OP) -> None:
        super().__init__(env)
        self.perturbation_spec = spec


class CartPoleAngleLengthPerturbation(gym.Wrapper):
    """Apply a plant-length change and a seeded-reset pole-angle offset."""

    def __init__(self, env: gym.Env, spec: PerturbationSpec) -> None:
        super().__init__(env)
        _require_environment(env, "CartPole-v1", spec.name)
        _require_parameters(spec, {"delta_theta_deg", "length"})
        delta = _number(spec.parameters["delta_theta_deg"], "delta_theta_deg")
        length = _positive_number(spec.parameters["length"], "length")
        base = env.unwrapped
        base.length = length
        base.polemass_length = base.masspole * base.length
        self._delta_theta = radians(delta)
        self.perturbation_spec = spec

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        _, info = self.env.reset(**kwargs)
        base = self.env.unwrapped
        state = np.asarray(base.state, dtype=float).copy()
        state[2] += self._delta_theta
        base.state = state
        return np.asarray(state, dtype=np.float32), info


class PendulumLengthPerturbation(gym.Wrapper):
    """Apply a Pendulum primitive length change."""

    def __init__(self, env: gym.Env, spec: PerturbationSpec) -> None:
        super().__init__(env)
        _require_environment(env, "Pendulum-v1", spec.name)
        _require_parameters(spec, {"l"})
        env.unwrapped.l = _positive_number(spec.parameters["l"], "l")
        self.perturbation_spec = spec


class MiniGridStartPosePerturbation(gym.Wrapper):
    """Set a fixed valid start cell and orientation before each reset."""

    def __init__(self, env: gym.Env, spec: PerturbationSpec) -> None:
        super().__init__(env)
        _require_environment(env, "MiniGrid-Empty-8x8-v0", spec.name)
        _require_parameters(spec, {"agent_start_pos", "agent_start_dir"})
        position = spec.parameters["agent_start_pos"]
        if (
            not isinstance(position, (list, tuple))
            or len(position) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in position
            )
        ):
            raise ValueError("agent_start_pos must contain exactly two integers.")
        direction = spec.parameters["agent_start_dir"]
        if (
            not isinstance(direction, int)
            or isinstance(direction, bool)
            or direction not in range(4)
        ):
            raise ValueError("agent_start_dir must be one of 0, 1, 2, or 3.")
        base = env.unwrapped
        x, y = position
        if not (1 <= x < base.width - 1 and 1 <= y < base.height - 1):
            raise ValueError("agent_start_pos must be a non-wall interior cell.")
        if (x, y) == (base.width - 2, base.height - 2):
            raise ValueError("agent_start_pos must not be the goal cell.")
        base.agent_start_pos = (x, y)
        base.agent_start_dir = direction
        self.perturbation_spec = spec


def apply_perturbation(env: gym.Env, spec: PerturbationSpec = NO_OP) -> gym.Env:
    """Apply one supported perturbation to an environment."""

    if spec.name == NO_OP.name:
        if spec.parameters:
            raise ValueError("The 'none' perturbation does not accept parameters.")
        return NoOpPerturbation(env, spec)
    if spec.name == CARTPOLE_ANGLE_LENGTH:
        return CartPoleAngleLengthPerturbation(env, spec)
    if spec.name == PENDULUM_LENGTH:
        return PendulumLengthPerturbation(env, spec)
    if spec.name == MINIGRID_START_POSE:
        return MiniGridStartPosePerturbation(env, spec)
    supported = ", ".join(
        (NO_OP.name, CARTPOLE_ANGLE_LENGTH, PENDULUM_LENGTH, MINIGRID_START_POSE)
    )
    raise ValueError(
        f"Unsupported perturbation {spec.name!r}. Supported perturbations: {supported}."
    )


def resolved_initial_state_distribution(
    nominal: Mapping[str, Any], spec: PerturbationSpec
) -> dict[str, Any]:
    """Describe the reset distribution after applying a perturbation."""

    if spec.name == CARTPOLE_ANGLE_LENGTH:
        return {
            "kind": "seeded_nominal_plus_offset",
            "nominal": dict(nominal),
            "offset": {
                "cart_position": 0.0,
                "cart_velocity": 0.0,
                "pole_angle_radians": radians(
                    float(spec.parameters["delta_theta_deg"])
                ),
                "pole_angular_velocity": 0.0,
            },
        }
    if spec.name == MINIGRID_START_POSE:
        return {
            "kind": "fixed",
            "agent_position": list(spec.parameters["agent_start_pos"]),
            "agent_direction": int(spec.parameters["agent_start_dir"]),
        }
    return dict(nominal)


def expected_parameters(
    nominal: Mapping[str, Any], spec: PerturbationSpec
) -> dict[str, Any]:
    """Resolve requested primitive values for validation."""

    expected = dict(nominal)
    if spec.name == CARTPOLE_ANGLE_LENGTH:
        expected["length"] = float(spec.parameters["length"])
    elif spec.name == PENDULUM_LENGTH:
        expected["l"] = float(spec.parameters["l"])
    elif spec.name == MINIGRID_START_POSE:
        expected["agent_start_pos"] = list(spec.parameters["agent_start_pos"])
        expected["agent_start_dir"] = int(spec.parameters["agent_start_dir"])
    return expected


def _require_environment(env: gym.Env, expected: str, name: str) -> None:
    actual = None if env.spec is None else env.spec.id
    if actual != expected:
        raise ValueError(f"Perturbation {name!r} requires {expected}, not {actual}.")


def _require_parameters(spec: PerturbationSpec, expected: set[str]) -> None:
    actual = set(spec.parameters)
    if actual != expected:
        raise ValueError(
            f"Perturbation {spec.name!r} requires parameters {sorted(expected)!r}; "
            f"received {sorted(actual)!r}."
        )


def _number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric.")
    return float(value)


def _positive_number(value: Any, name: str) -> float:
    result = _number(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result
