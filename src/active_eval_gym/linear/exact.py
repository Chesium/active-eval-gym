"""Exact closed-loop quantities for a known linear family under u = -K x.

Notation follows ese6180/proposal/discussions/proposal-design-notes.md:
G_K = A - B K is the closed loop, P is a certificate matrix, and P_H is the
finite-horizon cost matrix. Section numbers below refer to those notes.
Nothing here samples, trains, or stores data; every value is a deterministic
function of the matrices it receives.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, log, sqrt

import numpy as np

from active_eval_gym.linear.family import LinearFamily, LinearPlant


@dataclass(frozen=True, eq=False)
class QuadraticCost:
    """Stage cost x'Qx + u'Ru summed over steps 0 through horizon - 1."""

    q: np.ndarray
    r: np.ndarray
    horizon: int

    def __post_init__(self) -> None:
        q = _symmetric_psd(self.q, "Q")
        r = _symmetric_psd(self.r, "R")
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int):
            raise ValueError("horizon must be an integer.")
        if self.horizon < 1:
            raise ValueError("horizon must be at least one step.")
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "r", r)

    def stage_weight(self, gain: np.ndarray) -> np.ndarray:
        """W_K = Q + K'RK, the state weight once u = -Kx is substituted."""

        gain = np.asarray(gain, dtype=np.float64)
        return self.q + gain.T @ self.r @ gain


@dataclass(frozen=True, eq=False)
class WeightedCondition:
    """One (plant, initial state) atom of a finite training distribution q."""

    plant_id: str
    initial_state: np.ndarray
    weight: float

    def __post_init__(self) -> None:
        state = np.array(self.initial_state, dtype=np.float64)
        if state.ndim != 1 or not np.all(np.isfinite(state)):
            raise ValueError("initial_state must be a finite vector.")
        state.setflags(write=False)
        object.__setattr__(self, "initial_state", state)
        object.__setattr__(self, "weight", float(self.weight))


@dataclass(frozen=True)
class PreservationCheck:
    """Section 4 test for one retained plant and one proposed gain step."""

    plant_id: str
    old_margin: float
    perturbation_bound: float
    allowed_bound: float
    passed: bool


def closed_loop(plant: LinearPlant, gain: np.ndarray) -> np.ndarray:
    """G_K(theta) = A(theta) - B(theta) K."""

    gain = _gain(gain, plant)
    return plant.a - plant.b @ gain


def closed_loop_states(
    closed_loop_matrix: np.ndarray, initial_state: np.ndarray, steps: int
) -> np.ndarray:
    """Return x_0, ..., x_steps for x_{t+1} = G x_t as rows."""

    g = np.asarray(closed_loop_matrix, dtype=np.float64)
    state = np.asarray(initial_state, dtype=np.float64)
    if state.shape != (g.shape[0],):
        raise ValueError(f"Expected initial state shape {(g.shape[0],)}.")
    states = np.empty((steps + 1, g.shape[0]))
    states[0] = state
    for index in range(steps):
        states[index + 1] = g @ states[index]
    return states


def cost_matrix(
    plant: LinearPlant, gain: np.ndarray, cost: QuadraticCost
) -> np.ndarray:
    """P_H(K, theta) = sum_{t<H} (G^t)' W_K G^t, so J_H(theta, x) = x' P_H x."""

    g = closed_loop(plant, gain)
    weight = cost.stage_weight(gain)
    matrix = np.zeros_like(weight)
    for _ in range(cost.horizon):
        matrix = weight + g.T @ matrix @ g
    return _symmetrize(matrix)


def trajectory_cost(
    plant: LinearPlant,
    gain: np.ndarray,
    initial_state: np.ndarray,
    cost: QuadraticCost,
) -> float:
    """J_{H,K}(theta, x) for one initial state."""

    state = np.asarray(initial_state, dtype=np.float64)
    return float(state @ cost_matrix(plant, gain, cost) @ state)


def training_objective(
    family: LinearFamily,
    gain: np.ndarray,
    conditions: Sequence[WeightedCondition],
    cost: QuadraticCost,
) -> float:
    """F_q(K) = E_q J_{H,K} for a finite probability distribution q."""

    grouped = _second_moments(family, conditions)
    return fsum(
        float(np.sum(cost_matrix(family.plant(plant_id), gain, cost) * moment))
        for plant_id, moment in grouped.items()
    )


def training_objective_gradient(
    family: LinearFamily,
    gain: np.ndarray,
    conditions: Sequence[WeightedCondition],
    cost: QuadraticCost,
) -> np.ndarray:
    """Exact gradient of F_q with respect to K, holding q fixed.

    With cost-to-go matrices P_{(k)} for k remaining steps and second moments
    Sigma_t = G^t X_0 (G^t)', the gradient is
    2 sum_t [(R + B'P_{(H-1-t)}B) K - B'P_{(H-1-t)}A] Sigma_t.
    """

    gain = _gain(gain, family.plants[0])
    gradient = np.zeros_like(gain)
    weight = cost.stage_weight(gain)
    for plant_id, moment in _second_moments(family, conditions).items():
        plant = family.plant(plant_id)
        g = closed_loop(plant, gain)
        moments = [moment]
        for _ in range(cost.horizon - 1):
            moments.append(g @ moments[-1] @ g.T)
        cost_to_go = np.zeros_like(weight)
        for step_moment in reversed(moments):
            effective = (cost.r + plant.b.T @ cost_to_go @ plant.b) @ gain
            effective -= plant.b.T @ cost_to_go @ plant.a
            gradient += 2.0 * effective @ step_moment
            cost_to_go = weight + g.T @ cost_to_go @ g
    return gradient


def decrease_margin(certificate: np.ndarray, closed_loop_matrix: np.ndarray) -> float:
    """eta = lambda_min(P - G'PG); positive means V_P decreases at every state."""

    p = _certificate(certificate)
    g = np.asarray(closed_loop_matrix, dtype=np.float64)
    return float(np.linalg.eigvalsh(_symmetrize(p - g.T @ p @ g))[0])


def preservation_bound(
    certificate: np.ndarray,
    plant: LinearPlant,
    gain: np.ndarray,
    gain_step: np.ndarray,
) -> float:
    """Section 4 c_theta(dK) = 2||G'P|| d + ||P|| d^2 with d = ||B dK||."""

    p = _certificate(certificate)
    g = closed_loop(plant, gain)
    d = _spectral_norm(plant.b @ _gain(gain_step, plant))
    return 2.0 * _spectral_norm(g.T @ p) * d + _spectral_norm(p) * d * d


def preservation_checks(
    certificate: np.ndarray,
    family: LinearFamily,
    gain: np.ndarray,
    gain_step: np.ndarray,
    *,
    retained_fraction: float,
) -> tuple[PreservationCheck, ...]:
    """Apply the Section 4 sufficient test to every plant in the family.

    A plant passes when c_theta <= (1 - rho) eta_theta, which guarantees a new
    decrease margin of at least rho * eta_theta for the same certificate. The
    test presupposes a verified old certificate: verify it first with a
    declared numerical tolerance.
    """

    if not 0.0 < retained_fraction < 1.0:
        raise ValueError("retained_fraction must lie strictly between 0 and 1.")
    checks = []
    for plant in family.plants:
        margin = decrease_margin(certificate, closed_loop(plant, gain))
        if margin <= 0.0:
            raise ValueError(
                f"Certificate does not decrease on {plant.plant_id!r} "
                f"(margin {margin:.3e}); the preservation test does not apply."
            )
        bound = preservation_bound(certificate, plant, gain, gain_step)
        allowed = (1.0 - retained_fraction) * margin
        checks.append(
            PreservationCheck(
                plant_id=plant.plant_id,
                old_margin=margin,
                perturbation_bound=bound,
                allowed_bound=allowed,
                passed=bound <= allowed,
            )
        )
    return tuple(checks)


def cost_drift_bound(
    plant: LinearPlant,
    gain: np.ndarray,
    new_gain: np.ndarray,
    cost: QuadraticCost,
) -> float:
    """Section 5 bound L_H(theta) ||dK|| on ||P_H(K') - P_H(K)|| (spectral).

    M is max(||G||, ||G'||). The bound may overflow to inf, which is still a
    valid (vacuous) bound.
    """

    gain = _gain(gain, plant)
    new_gain = _gain(new_gain, plant)
    step = _spectral_norm(new_gain - gain)
    if step == 0.0:
        return 0.0
    m = max(
        _spectral_norm(closed_loop(plant, gain)),
        _spectral_norm(closed_loop(plant, new_gain)),
    )
    steps = np.arange(cost.horizon, dtype=np.float64)
    with np.errstate(over="ignore"):
        dynamics_sum = float(np.sum(steps[1:] * m ** (2.0 * steps[1:] - 1.0)))
        input_sum = float(np.sum(m ** (2.0 * steps)))
    dynamics_coefficient = (
        2.0 * _spectral_norm(cost.stage_weight(gain)) * _spectral_norm(plant.b)
    )
    input_coefficient = _spectral_norm(cost.r) * (
        _spectral_norm(new_gain) + _spectral_norm(gain)
    )
    lipschitz = _scaled(dynamics_coefficient, dynamics_sum) + _scaled(
        input_coefficient, input_sum
    )
    return lipschitz * step


def decrease_residual(
    certificate: np.ndarray,
    state: np.ndarray,
    next_state: np.ndarray,
    *,
    decrease: float,
) -> float:
    """Section 2 r_P(x, x+) = (V_P(x+) - V_P(x)) / ||x||^2 + eta, for x != 0."""

    p = _certificate(certificate)
    x = np.asarray(state, dtype=np.float64)
    x_next = np.asarray(next_state, dtype=np.float64)
    squared_norm = float(x @ x)
    if squared_norm == 0.0:
        raise ValueError("The normalized residual is undefined at the zero state.")
    return float(x_next @ p @ x_next - x @ p @ x) / squared_norm + decrease


def trajectory_certificate_loss(
    certificate: np.ndarray,
    states: np.ndarray,
    *,
    decrease: float,
    learning_margin: float,
) -> float:
    """Section 2 clipped-ramp loss of one trajectory x_0, ..., x_H.

    The loss is 0 when every residual is at most -gamma, 1 when any residual is
    nonnegative, and it upper-bounds the violation indicator. Exactly-zero
    states are skipped; any near-zero cutoff is a protocol decision left to the
    caller.
    """

    if decrease <= 0.0 or learning_margin <= 0.0:
        raise ValueError("decrease and learning_margin must be positive.")
    states = np.asarray(states, dtype=np.float64)
    residuals = [
        decrease_residual(certificate, current, following, decrease=decrease)
        for current, following in zip(states[:-1], states[1:], strict=True)
        if np.any(current != 0.0)
    ]
    if not residuals:
        return 0.0
    return min(1.0, max(0.0, 1.0 + max(residuals) / learning_margin))


def certificate_loss_lipschitz(
    closed_loop_norm_bound: float, learning_margin: float
) -> float:
    """Section 2 constant L = (M^2 + 1) / gamma for the loss in Frobenius P."""

    if closed_loop_norm_bound < 0.0 or learning_margin <= 0.0:
        raise ValueError("Need M >= 0 and a positive learning margin.")
    return (closed_loop_norm_bound**2 + 1.0) / learning_margin


def certificate_generalization_slack(
    *,
    state_dimension: int,
    certificate_upper: float,
    trajectories: int,
    lipschitz: float,
    net_radius: float,
    delta: float,
) -> float:
    """Section 2 uniform gap between population and empirical certificate loss.

    Returns 2 L eps + sqrt((d log(1 + 2 b sqrt(n) / eps) + log(2 / delta)) / 2N)
    with d = n(n + 1) / 2. eps must be fixed before the data are drawn.
    """

    if state_dimension < 1 or trajectories < 1:
        raise ValueError("state_dimension and trajectories must be positive.")
    if certificate_upper <= 0.0 or lipschitz < 0.0 or net_radius <= 0.0:
        raise ValueError("Need b > 0, L >= 0, and eps > 0.")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between 0 and 1.")
    n = state_dimension
    d = n * (n + 1) // 2
    complexity = d * log(1.0 + 2.0 * certificate_upper * sqrt(n) / net_radius)
    return 2.0 * lipschitz * net_radius + sqrt(
        (complexity + log(2.0 / delta)) / (2.0 * trajectories)
    )


def _second_moments(
    family: LinearFamily, conditions: Sequence[WeightedCondition]
) -> dict[str, np.ndarray]:
    if not conditions:
        raise ValueError("A training distribution needs at least one condition.")
    weights = [condition.weight for condition in conditions]
    if any(not np.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("Condition weights must be finite and nonnegative.")
    if abs(fsum(weights) - 1.0) > 1e-9:
        raise ValueError(f"Condition weights must sum to one, got {fsum(weights)}.")
    grouped: dict[str, np.ndarray] = {}
    for condition in conditions:
        plant = family.plant(condition.plant_id)
        state = condition.initial_state
        if state.shape != (plant.state_dimension,):
            raise ValueError(
                f"Initial state for {plant.plant_id!r} must have shape "
                f"{(plant.state_dimension,)}."
            )
        moment = condition.weight * np.outer(state, state)
        grouped[plant.plant_id] = grouped.get(plant.plant_id, 0.0) + moment
    return grouped


def _gain(gain: np.ndarray, plant: LinearPlant) -> np.ndarray:
    matrix = np.asarray(gain, dtype=np.float64)
    expected = (plant.input_dimension, plant.state_dimension)
    if matrix.shape != expected:
        raise ValueError(f"Expected gain shape {expected}, received {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Gain must be finite.")
    return matrix


def _certificate(certificate: np.ndarray) -> np.ndarray:
    p = np.asarray(certificate, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != p.shape[1] or not np.all(np.isfinite(p)):
        raise ValueError("Certificate must be a finite square matrix.")
    scale = max(1.0, float(np.max(np.abs(p))))
    if not np.allclose(p, p.T, rtol=0.0, atol=1e-10 * scale):
        raise ValueError("Certificate must be symmetric.")
    return _symmetrize(p)


def _symmetric_psd(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.array(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    if not np.all(np.isfinite(matrix)) or not np.array_equal(matrix, matrix.T):
        raise ValueError(f"{name} must be finite and symmetric.")
    scale = max(1.0, float(np.max(np.abs(matrix))))
    if np.linalg.eigvalsh(matrix)[0] < -1e-12 * scale:
        raise ValueError(f"{name} must be positive semidefinite.")
    matrix.setflags(write=False)
    return matrix


def _scaled(coefficient: float, total: float) -> float:
    # A zero coefficient removes its term even when the geometric sum overflowed.
    return 0.0 if coefficient == 0.0 else coefficient * total


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _spectral_norm(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(matrix, 2))
