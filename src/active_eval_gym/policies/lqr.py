"""Nominal discrete-time LQR design and quantized CartPole policy."""

from dataclasses import dataclass
from typing import Any

import control
import gymnasium as gym
import numpy as np

from active_eval_gym.models import PolicyAction, PolicyDesignSpec


@dataclass(frozen=True)
class LQRModel:
    """All numerical values needed to audit and load a built LQR policy."""

    state_order: tuple[str, ...]
    input_name: str
    continuous_a: list[list[float]]
    continuous_b: list[list[float]]
    discrete_a: list[list[float]]
    discrete_b: list[list[float]]
    q: list[list[float]]
    r: list[list[float]]
    k: list[list[float]]
    riccati_solution: list[list[float]]
    closed_loop_eigenvalues: list[dict[str, float]]


class QuantizedLQRPolicy:
    """Frozen state-feedback gain mapped onto CartPole's two actions."""

    def __init__(self, gain: np.ndarray, *, force_magnitude: float) -> None:
        gain = np.asarray(gain, dtype=np.float64)
        if gain.shape != (1, 4):
            raise ValueError(f"Expected LQR gain shape (1, 4), received {gain.shape}.")
        self._gain = gain.copy()
        self._force_magnitude = float(force_magnitude)

    def act(
        self,
        observation: Any,
        *,
        deterministic: bool = True,
    ) -> PolicyAction:
        del deterministic
        state = np.asarray(observation, dtype=np.float64)
        if state.shape != (4,):
            raise ValueError(
                f"Expected CartPole observation shape (4,), got {state.shape}."
            )
        desired_force = float(-(self._gain @ state)[0])
        action = 1 if desired_force >= 0.0 else 0
        applied_force = self._force_magnitude if action == 1 else -self._force_magnitude
        return PolicyAction(
            action=action,
            diagnostics={
                "desired_force": desired_force,
                "applied_force": applied_force,
            },
        )


def design_cartpole_lqr(spec: PolicyDesignSpec) -> tuple[LQRModel, float]:
    """Build the nominal gain and report finite-difference matrix error."""

    if spec.environment.environment_id != "CartPole-v1":
        raise ValueError("Quantized LQR supports only CartPole-v1.")
    parameters = spec.environment.parameters
    continuous_a, continuous_b = cartpole_continuous_matrices(parameters)
    tau = float(parameters["tau"])
    discrete_a = np.eye(4) + tau * continuous_a
    discrete_b = tau * continuous_b
    q = np.diag(np.asarray(spec.hyperparameters["q_diagonal"], dtype=np.float64))
    r = np.asarray([[spec.hyperparameters["r"]]], dtype=np.float64)
    if q.shape != (4, 4):
        raise ValueError("q_diagonal must contain four values.")
    gain, riccati, eigenvalues = control.dlqr(discrete_a, discrete_b, q, r)
    finite_a, finite_b = finite_difference_cartpole_matrices(parameters)
    error = float(
        max(
            np.max(np.abs(discrete_a - finite_a)),
            np.max(np.abs(discrete_b - finite_b)),
        )
    )
    model = LQRModel(
        state_order=(
            "cart_position",
            "cart_velocity",
            "pole_angle",
            "pole_angular_velocity",
        ),
        input_name="desired_force_newtons",
        continuous_a=continuous_a.tolist(),
        continuous_b=continuous_b.tolist(),
        discrete_a=discrete_a.tolist(),
        discrete_b=discrete_b.tolist(),
        q=q.tolist(),
        r=r.tolist(),
        k=np.asarray(gain).tolist(),
        riccati_solution=np.asarray(riccati).tolist(),
        closed_loop_eigenvalues=[
            {"real": float(value.real), "imag": float(value.imag)}
            for value in np.asarray(eigenvalues)
        ],
    )
    return model, error


def cartpole_continuous_matrices(
    parameters: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Linearize Gym's nonlinear acceleration equations at upright zero."""

    gravity = float(parameters["gravity"])
    masscart = float(parameters["masscart"])
    masspole = float(parameters["masspole"])
    length = float(parameters["length"])
    total_mass = masscart + masspole
    denominator = length * (4.0 / 3.0 - masspole / total_mass)
    thetaacc_theta = gravity / denominator
    thetaacc_force = -1.0 / (total_mass * denominator)
    xacc_theta = -(masspole * length / total_mass) * thetaacc_theta
    xacc_force = 1.0 / total_mass - (masspole * length / total_mass) * thetaacc_force
    a = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, xacc_theta, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, thetaacc_theta, 0.0],
        ]
    )
    b = np.array([[0.0], [xacc_force], [0.0], [thetaacc_force]])
    return a, b


def finite_difference_cartpole_matrices(
    parameters: dict[str, Any], *, epsilon: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Numerically linearize the exact nominal explicit-Euler transition."""

    env = gym.make("CartPole-v1")
    base = env.unwrapped
    try:
        for name in ("gravity", "masscart", "masspole", "length", "tau"):
            installed = getattr(base, name)
            if installed != parameters[name]:
                raise ValueError(
                    f"Installed CartPole {name}={installed!r} does not match "
                    f"the nominal design value {parameters[name]!r}."
                )
        if base.kinematics_integrator != parameters["kinematics_integrator"]:
            raise ValueError("Installed CartPole integrator is not the nominal one.")

        base.reset(seed=0)
        origin = np.zeros(4)
        basis = np.eye(4)
        original_force = float(base.force_mag)
        base.force_mag = 0.0
        a = np.column_stack(
            [
                (
                    _step_installed_cartpole(base, origin + epsilon * basis[index], 1)
                    - _step_installed_cartpole(
                        base, origin - epsilon * basis[index], 1
                    )
                )
                / (2.0 * epsilon)
                for index in range(4)
            ]
        )
        base.force_mag = epsilon
        b = (
            _step_installed_cartpole(base, origin, 1)
            - _step_installed_cartpole(base, origin, 0)
        ).reshape(4, 1) / (2.0 * epsilon)
        base.force_mag = original_force
        return a, b
    finally:
        env.close()


def _step_installed_cartpole(base: Any, state: np.ndarray, action: int) -> np.ndarray:
    base.state = tuple(float(value) for value in state)
    base.steps_beyond_terminated = None
    base.step(action)
    return np.asarray(base.state, dtype=np.float64)
