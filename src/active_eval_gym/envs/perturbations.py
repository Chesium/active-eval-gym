"""Structured environment perturbations."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import gymnasium as gym


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


def apply_perturbation(env: gym.Env, spec: PerturbationSpec = NO_OP) -> gym.Env:
    """Apply a supported perturbation to an environment."""

    if spec.name != NO_OP.name:
        raise ValueError(
            f"Unsupported perturbation {spec.name!r}. Supported perturbations: 'none'."
        )
    if spec.parameters:
        raise ValueError("The 'none' perturbation does not accept parameters.")
    return NoOpPerturbation(env, spec)
