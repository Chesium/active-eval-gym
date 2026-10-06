"""Known discrete-time linear plant families x+ = A(theta) x + B(theta) u."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from active_eval_gym.policies.lqr import cartpole_discrete_matrices

EXPLICIT = "explicit-v1"
CARTPOLE_LENGTH_LINEARIZATION = "cartpole-length-linearization-v1"


@dataclass(frozen=True, eq=False)
class LinearPlant:
    """One plant of a family, with read-only known matrices."""

    plant_id: str
    parameters: Mapping[str, Any]
    a: np.ndarray
    b: np.ndarray

    def __post_init__(self) -> None:
        if not self.plant_id:
            raise ValueError("plant_id must not be empty.")
        a = _read_only_matrix(self.a, f"{self.plant_id} A")
        b = _read_only_matrix(self.b, f"{self.plant_id} B")
        if a.shape[0] != a.shape[1]:
            raise ValueError(f"{self.plant_id} A must be square, got {a.shape}.")
        if b.shape[0] != a.shape[0]:
            raise ValueError(
                f"{self.plant_id} B has {b.shape[0]} rows; A has {a.shape[0]}."
            )
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    @property
    def state_dimension(self) -> int:
        return self.a.shape[0]

    @property
    def input_dimension(self) -> int:
        return self.b.shape[1]


@dataclass(frozen=True, eq=False)
class LinearFamily:
    """A fixed, finite plant set sharing state and input dimensions."""

    schema_version: int
    family_id: str
    kind: str
    plants: tuple[LinearPlant, ...]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.family_id:
            raise ValueError("family_id must not be empty.")
        plants = tuple(self.plants)
        if not plants:
            raise ValueError("A linear family requires at least one plant.")
        identifiers = [plant.plant_id for plant in plants]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"Duplicate plant_id in family {self.family_id!r}.")
        shapes = {(plant.state_dimension, plant.input_dimension) for plant in plants}
        if len(shapes) != 1:
            raise ValueError(
                f"Plants in family {self.family_id!r} have mixed dimensions {shapes}."
            )
        object.__setattr__(self, "plants", plants)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def plant_ids(self) -> tuple[str, ...]:
        return tuple(plant.plant_id for plant in self.plants)

    @property
    def state_dimension(self) -> int:
        return self.plants[0].state_dimension

    @property
    def input_dimension(self) -> int:
        return self.plants[0].input_dimension

    def plant(self, plant_id: str) -> LinearPlant:
        for plant in self.plants:
            if plant.plant_id == plant_id:
                return plant
        raise KeyError(f"Family {self.family_id!r} has no plant {plant_id!r}.")


def cartpole_length_family(
    family_id: str,
    nominal_parameters: Mapping[str, Any],
    lengths: Iterable[float],
    *,
    schema_version: int = 1,
    nominal_spec_id: str | None = None,
) -> LinearFamily:
    """Linearize Gym CartPole at upright for each pole half-length.

    All other physical parameters stay at their nominal values. Both A and B
    are affine in 1 / length, so a fixed quadratic certificate that verifies at
    two lengths also verifies at every length between them.
    """

    plants = []
    for length in lengths:
        length = float(length)
        if not length > 0.0:
            raise ValueError(f"Pole half-length must be positive, got {length}.")
        a, b = cartpole_discrete_matrices({**nominal_parameters, "length": length})
        plants.append(
            LinearPlant(
                plant_id=f"length-{_slug(length)}",
                parameters={"length": length},
                a=a,
                b=b,
            )
        )
    return LinearFamily(
        schema_version=schema_version,
        family_id=family_id,
        kind=CARTPOLE_LENGTH_LINEARIZATION,
        plants=tuple(plants),
        provenance={
            "nominal_spec_id": nominal_spec_id,
            "nominal_parameters": dict(nominal_parameters),
            "linearization": "upright-equilibrium",
            "discretization": "gym_explicit_euler",
            "state_order": [
                "cart_position",
                "cart_velocity",
                "pole_angle",
                "pole_angular_velocity",
            ],
            "input": "force_newtons",
        },
    )


def _read_only_matrix(value: Any, name: str) -> np.ndarray:
    matrix = np.array(value, dtype=np.float64)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError(f"{name} must be a non-empty 2-D matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite.")
    matrix.setflags(write=False)
    return matrix


def _slug(value: float) -> str:
    sign = "m" if value < 0 else "p"
    return sign + format(abs(value), ".12g").replace(".", "p")
