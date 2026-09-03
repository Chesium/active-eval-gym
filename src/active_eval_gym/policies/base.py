"""The common policy interface used by every rollout."""

from copy import deepcopy
from typing import Any, Protocol

import gymnasium as gym
import numpy as np


class Policy(Protocol):
    """A fixed policy that maps an observation to an action."""

    def act(
        self,
        observation: Any,
        *,
        deterministic: bool = True,
    ) -> Any:
        """Select an action without changing policy weights."""
        ...


class ConstantPolicy:
    """A deterministic policy that returns the same action at every step."""

    def __init__(self, action: Any) -> None:
        self._action = deepcopy(action)

    def act(
        self,
        observation: Any,
        *,
        deterministic: bool = True,
    ) -> Any:
        del observation, deterministic
        return deepcopy(self._action)


def zero_action(action_space: gym.Space) -> Any:
    """Construct a zero-valued action for the supported action-space families."""

    if isinstance(action_space, gym.spaces.Discrete):
        action = int(action_space.start)
    elif isinstance(action_space, gym.spaces.Box):
        action = np.zeros(action_space.shape, dtype=action_space.dtype)
    else:
        raise TypeError(
            "The constant-zero policy supports only Discrete and Box action spaces; "
            f"received {type(action_space).__name__}."
        )

    if not action_space.contains(action):
        raise ValueError(f"Action space {action_space!r} does not contain zero.")
    return action
