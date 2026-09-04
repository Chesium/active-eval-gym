"""Structured, allowlisted environment perturbations."""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import radians
from types import MappingProxyType
from typing import Any

import gymnasium as gym
import numpy as np

CARTPOLE_ANGLE_LENGTH = "cartpole-angle-length-v1"
CARTPOLE_MASS = "cartpole-mass-v1"
CARTPOLE_POLE_ANGLE_NOISE = "cartpole-pole-angle-noise-v1"
CARTPOLE_FORCE_MAGNITUDE = "cartpole-force-magnitude-v1"
CARTPOLE_ACTION_DELAY = "cartpole-action-delay-v1"
CARTPOLE_ACTION_DROPOUT = "cartpole-action-dropout-v1"
CARTPOLE_FIXED_INITIAL_STATE = "cartpole-fixed-initial-state-v1"
PENDULUM_LENGTH = "pendulum-length-v1"
MINIGRID_START_POSE = "minigrid-start-pose-v1"

PERTURBATION_INFO_KEY = "active_eval_gym.perturbation"
OBSERVATION_NOISE_STREAM_ID = 1_001
ACTION_DROPOUT_STREAM_ID = 1_002


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
        _require_cartpole(env, spec.name)
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


class CartPoleFixedInitialStatePerturbation(gym.Wrapper):
    """Set an exact CartPole state and pole length for mirror-paired rollouts."""

    def __init__(self, env: gym.Env, spec: PerturbationSpec) -> None:
        super().__init__(env)
        _require_cartpole(env, spec.name)
        _require_parameters(spec, {"initial_state", "length"})
        state = np.asarray(spec.parameters["initial_state"], dtype=float)
        if state.shape != (4,) or not np.all(np.isfinite(state)):
            raise ValueError("initial_state must contain four finite numbers.")
        length = _positive_number(spec.parameters["length"], "length")
        base = env.unwrapped
        base.length = length
        base.polemass_length = base.masspole * base.length
        self._initial_state = state.copy()
        self.perturbation_spec = spec

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        _, info = self.env.reset(**kwargs)
        state = self._initial_state.copy()
        self.env.unwrapped.state = state
        return np.asarray(state, dtype=np.float32), info


class CartPoleMassPerturbation(gym.Wrapper):
    """Change pole mass and refresh every dependent dynamics quantity."""

    def __init__(self, env: gym.Env, spec: PerturbationSpec) -> None:
        super().__init__(env)
        _require_cartpole(env, spec.name)
        _require_parameters(spec, {"masspole"})
        base = env.unwrapped
        base.masspole = _positive_number(spec.parameters["masspole"], "masspole")
        base.total_mass = base.masspole + base.masscart
        base.polemass_length = base.masspole * base.length
        self.perturbation_spec = spec


class CartPoleForceMagnitudePerturbation(gym.Wrapper):
    """Change the magnitude associated with CartPole's two discrete actions."""

    def __init__(self, env: gym.Env, spec: PerturbationSpec) -> None:
        super().__init__(env)
        _require_cartpole(env, spec.name)
        _require_parameters(spec, {"force_mag"})
        env.unwrapped.force_mag = _positive_number(
            spec.parameters["force_mag"], "force_mag"
        )
        self.perturbation_spec = spec


class CartPolePoleAngleNoisePerturbation(gym.Wrapper):
    """Corrupt only the policy-observed pole angle with seeded Gaussian noise."""

    def __init__(self, env: gym.Env, spec: PerturbationSpec) -> None:
        super().__init__(env)
        _require_cartpole(env, spec.name)
        _require_parameters(spec, {"pole_angle_noise_std_deg"})
        standard_deviation = _nonnegative_number(
            spec.parameters["pole_angle_noise_std_deg"],
            "pole_angle_noise_std_deg",
        )
        self._standard_deviation = radians(standard_deviation)
        self._rng: np.random.Generator | None = None
        low = np.asarray(env.observation_space.low, dtype=np.float32).copy()
        high = np.asarray(env.observation_space.high, dtype=np.float32).copy()
        low[2] = -np.inf
        high[2] = np.inf
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )
        self.perturbation_spec = spec

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        seed = _required_reset_seed(kwargs)
        self._rng = _perturbation_rng(seed, OBSERVATION_NOISE_STREAM_ID)
        observation, info = self.env.reset(**kwargs)
        return self._corrupt(observation, info)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation, info = self._corrupt(observation, info)
        return observation, reward, terminated, truncated, info

    def _corrupt(
        self, observation: Any, info: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self._rng is None:
            raise RuntimeError("Observation-noise environment must be reset first.")
        noise = float(self._rng.normal(0.0, self._standard_deviation))
        corrupted = np.asarray(observation).copy()
        corrupted[2] += noise
        diagnostics = {
            "kind": "pole_angle_observation_noise",
            "pole_angle_noise_radians": noise,
            "rng": "numpy.default_rng",
            "stream_id": OBSERVATION_NOISE_STREAM_ID,
        }
        return corrupted, _with_diagnostics(info, diagnostics)


class CartPoleActionDelayPerturbation(gym.Wrapper):
    """Delay requests while allowing the first requested action to pass."""

    def __init__(self, env: gym.Env, spec: PerturbationSpec) -> None:
        super().__init__(env)
        _require_cartpole(env, spec.name)
        _require_parameters(spec, {"delay_steps"})
        self._delay_steps = _nonnegative_int(
            spec.parameters["delay_steps"], "delay_steps"
        )
        self._requests: deque[int] = deque()
        self.perturbation_spec = spec

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self._requests.clear()
        observation, info = self.env.reset(**kwargs)
        diagnostics = {
            "kind": "action_delay",
            "delay_steps": self._delay_steps,
            "first_request_passthrough": True,
        }
        return observation, _with_diagnostics(info, diagnostics)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        requested = _cartpole_action(action)
        self._requests.append(requested)
        request_index = len(self._requests) - 1
        source_index = max(0, request_index - self._delay_steps)
        environment_action = self._requests[source_index]
        observation, reward, terminated, truncated, info = self.env.step(
            environment_action
        )
        diagnostics = {
            "kind": "action_delay",
            "requested_action": requested,
            "environment_action": environment_action,
            "delay_steps": self._delay_steps,
            "source_step": source_index,
        }
        return (
            observation,
            reward,
            terminated,
            truncated,
            _with_diagnostics(info, diagnostics),
        )


class CartPoleActionDropoutPerturbation(gym.Wrapper):
    """Drop requests using seeded Bernoulli draws and hold the last action."""

    def __init__(self, env: gym.Env, spec: PerturbationSpec) -> None:
        super().__init__(env)
        _require_cartpole(env, spec.name)
        _require_parameters(spec, {"dropout_probability"})
        self._probability = _probability(
            spec.parameters["dropout_probability"], "dropout_probability"
        )
        self._rng: np.random.Generator | None = None
        self._last_action: int | None = None
        self.perturbation_spec = spec

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        seed = _required_reset_seed(kwargs)
        self._rng = _perturbation_rng(seed, ACTION_DROPOUT_STREAM_ID)
        self._last_action = None
        observation, info = self.env.reset(**kwargs)
        diagnostics = {
            "kind": "action_dropout",
            "dropout_probability": self._probability,
            "first_request_passthrough": True,
            "rng": "numpy.default_rng",
            "stream_id": ACTION_DROPOUT_STREAM_ID,
        }
        return observation, _with_diagnostics(info, diagnostics)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self._rng is None:
            raise RuntimeError("Action-dropout environment must be reset first.")
        requested = _cartpole_action(action)
        random_draw = None if self._last_action is None else float(self._rng.random())
        dropped = random_draw is not None and random_draw < self._probability
        environment_action = (
            self._last_action
            if dropped and self._last_action is not None
            else requested
        )
        self._last_action = environment_action
        observation, reward, terminated, truncated, info = self.env.step(
            environment_action
        )
        diagnostics = {
            "kind": "action_dropout",
            "requested_action": requested,
            "environment_action": environment_action,
            "dropout_probability": self._probability,
            "dropout_event": dropped,
            "random_draw": random_draw,
            "rng": "numpy.default_rng",
            "stream_id": ACTION_DROPOUT_STREAM_ID,
        }
        return (
            observation,
            reward,
            terminated,
            truncated,
            _with_diagnostics(info, diagnostics),
        )


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
    constructors = {
        CARTPOLE_ANGLE_LENGTH: CartPoleAngleLengthPerturbation,
        CARTPOLE_MASS: CartPoleMassPerturbation,
        CARTPOLE_POLE_ANGLE_NOISE: CartPolePoleAngleNoisePerturbation,
        CARTPOLE_FORCE_MAGNITUDE: CartPoleForceMagnitudePerturbation,
        CARTPOLE_ACTION_DELAY: CartPoleActionDelayPerturbation,
        CARTPOLE_ACTION_DROPOUT: CartPoleActionDropoutPerturbation,
        CARTPOLE_FIXED_INITIAL_STATE: CartPoleFixedInitialStatePerturbation,
        PENDULUM_LENGTH: PendulumLengthPerturbation,
        MINIGRID_START_POSE: MiniGridStartPosePerturbation,
    }
    try:
        constructor = constructors[spec.name]
    except KeyError as error:
        supported = ", ".join((NO_OP.name, *constructors))
        raise ValueError(
            f"Unsupported perturbation {spec.name!r}. "
            f"Supported perturbations: {supported}."
        ) from error
    return constructor(env, spec)


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
    if spec.name == CARTPOLE_FIXED_INITIAL_STATE:
        state = [float(value) for value in spec.parameters["initial_state"]]
        return {
            "kind": "fixed",
            "state_order": [
                "cart_position",
                "cart_velocity",
                "pole_angle",
                "pole_angular_velocity",
            ],
            "state": state,
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
    elif spec.name == CARTPOLE_FIXED_INITIAL_STATE:
        expected["length"] = float(spec.parameters["length"])
    elif spec.name == CARTPOLE_MASS:
        expected["masspole"] = float(spec.parameters["masspole"])
    elif spec.name == CARTPOLE_FORCE_MAGNITUDE:
        expected["force_mag"] = float(spec.parameters["force_mag"])
    elif spec.name == PENDULUM_LENGTH:
        expected["l"] = float(spec.parameters["l"])
    elif spec.name == MINIGRID_START_POSE:
        expected["agent_start_pos"] = list(spec.parameters["agent_start_pos"])
        expected["agent_start_dir"] = int(spec.parameters["agent_start_dir"])
    return expected


def _with_diagnostics(
    info: Mapping[str, Any], diagnostics: dict[str, Any]
) -> dict[str, Any]:
    result = dict(info)
    if PERTURBATION_INFO_KEY in result:
        raise RuntimeError(
            f"Environment info already contains {PERTURBATION_INFO_KEY!r}."
        )
    result[PERTURBATION_INFO_KEY] = diagnostics
    return result


def _perturbation_rng(seed: int, stream_id: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, stream_id]))


def _required_reset_seed(kwargs: Mapping[str, Any]) -> int:
    seed = kwargs.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("Stochastic perturbations require an integer reset seed.")
    return seed


def _require_cartpole(env: gym.Env, name: str) -> None:
    _require_environment(env, "CartPole-v1", name)


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


def _nonnegative_number(value: Any, name: str) -> float:
    result = _number(value, name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return value


def _probability(value: Any, name: str) -> float:
    result = _number(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return result


def _cartpole_action(action: Any) -> int:
    if not isinstance(action, (int, np.integer)) or int(action) not in (0, 1):
        raise ValueError("CartPole action must be 0 or 1.")
    return int(action)
