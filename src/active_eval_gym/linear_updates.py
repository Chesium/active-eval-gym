"""Exact linear-control updates and analytical, float64 sensitivity bounds.

This analytical fixture is independent of Gym and of policy training. Bounds are
evaluations of exact-arithmetic inequalities, not interval-arithmetic proofs.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from active_eval_gym.certificates import VERIFICATION_TOLERANCE, verified_margins


@dataclass(frozen=True)
class LinearFamily:
    a: np.ndarray
    b: np.ndarray
    q: np.ndarray
    r: np.ndarray

    def __post_init__(self) -> None:
        a, b, q, r = (
            np.asarray(v, dtype=float) for v in (self.a, self.b, self.q, self.r)
        )
        if a.ndim != 3 or a.shape[1] != a.shape[2] or not len(a):
            raise ValueError("A must have shape (plants, n, n).")
        if b.ndim != 3 or b.shape[:2] != a.shape[:2]:
            raise ValueError("B must have shape (plants, n, m).")
        if q.shape != a.shape[1:] or r.shape != (b.shape[2], b.shape[2]):
            raise ValueError("Cost weights must match state and action dimensions.")
        for name, value in (("a", a), ("b", b), ("q", q), ("r", r)):
            if not np.all(np.isfinite(value)):
                raise ValueError("All matrices must be finite.")
            object.__setattr__(self, name, value.copy())
        for weight in (q, r):
            if not np.allclose(weight, weight.T, atol=1e-14, rtol=0):
                raise ValueError("Cost weights must be symmetric.")
            if np.linalg.eigvalsh(weight).min() < -1e-14:
                raise ValueError("Cost weights must be positive semidefinite.")

    def closed_loops(self, gain: np.ndarray) -> np.ndarray:
        if gain.shape != (self.b.shape[2], self.a.shape[1]):
            raise ValueError("Gain has incompatible dimensions.")
        return self.a - self.b @ gain


@dataclass(frozen=True)
class ConditionGrid:
    plant_ids: np.ndarray
    resets: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        ids = np.asarray(self.plant_ids, dtype=int)
        resets = np.asarray(self.resets, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if resets.ndim != 2 or ids.shape != (len(resets),):
            raise ValueError("One plant ID is required per reset.")
        if weights.shape != ids.shape or not len(ids) or np.any(ids < 0):
            raise ValueError("Invalid condition IDs or weights.")
        if (
            not np.all(np.isfinite(resets))
            or not np.all(np.isfinite(weights))
            or np.any(weights <= 0)
            or not np.isclose(weights.sum(), 1)
        ):
            raise ValueError("Require finite resets and positive probability weights.")
        for name, value in (
            ("plant_ids", ids),
            ("resets", resets),
            ("weights", weights),
        ):
            object.__setattr__(self, name, value.copy())


@dataclass(frozen=True)
class UpdateSpec:
    horizon: int = 10
    threshold: float = 3.5
    rho: float = 0.5
    kappa: float = 0.25
    initial_step: float = 1.0
    shrink: float = 0.5
    armijo: float = 0.1
    max_backtracks: int = 30
    verification_tolerance: float = VERIFICATION_TOLERANCE
    gradient_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.max_backtracks < 0:
            raise ValueError("Require a positive horizon and nonnegative search cap.")
        if not 0 < self.rho < 1 or not 0 < self.shrink < 1:
            raise ValueError("Retained fraction and shrink factor must lie in (0, 1).")
        if not 0 < self.armijo < 1:
            raise ValueError("Armijo coefficient must lie in (0, 1).")
        positive = (self.kappa, self.initial_step, self.threshold)
        if any(not np.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("Require finite positive radius, step, and threshold.")
        if self.verification_tolerance < 0 or self.gradient_tolerance < 0:
            raise ValueError("Numerical tolerances must be nonnegative.")


@dataclass(frozen=True)
class TargetLaw:
    weights: np.ndarray
    selected: np.ndarray
    fallback: bool


@dataclass(frozen=True)
class Trial:
    index: int
    step: float
    radius_ok: bool
    margin_ok: bool
    armijo_ok: bool | None
    consumption: np.ndarray
    new_margins: np.ndarray
    objective: float | None


@dataclass
class StepResult:
    gain: np.ndarray
    direction: np.ndarray
    margins: np.ndarray
    status: str
    step: float = 0.0
    trials: list[Trial] = field(default_factory=list)


def circle_grid(plants: int, directions: int) -> ConditionGrid:
    angles = np.arange(directions) * 2 * np.pi / directions
    return ConditionGrid(
        np.repeat(np.arange(plants), directions),
        np.tile(np.column_stack((np.cos(angles), np.sin(angles))), (plants, 1)),
        np.full(plants * directions, 1 / (plants * directions)),
    )


def policy_actions(gain: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Inputs of u = -K x along saved trajectories, one definition for all callers."""
    return -np.einsum("ij,ctj->cti", gain, states[:, :-1])


def rollout(
    family: LinearFamily, grid: ConditionGrid, gain: np.ndarray, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    if horizon < 1:
        raise ValueError("Horizon must be positive.")
    loops = family.closed_loops(gain)[grid.plant_ids]
    states = np.empty((len(grid.weights), horizon + 1, gain.shape[1]))
    states[:, 0] = grid.resets
    for t in range(horizon):
        states[:, t + 1] = np.einsum("cij,cj->ci", loops, states[:, t])
    return states, policy_actions(gain, states)


def trajectory_costs(
    family: LinearFamily, states: np.ndarray, actions: np.ndarray
) -> np.ndarray:
    return np.einsum("cti,ij,ctj->c", states[:, :-1], family.q, states[:, :-1]) + (
        np.einsum("cti,ij,ctj->c", actions, family.r, actions)
    )


def cost_matrices(family: LinearFamily, gain: np.ndarray, horizon: int) -> np.ndarray:
    loops = family.closed_loops(gain)
    result = np.zeros_like(loops)
    power = np.broadcast_to(np.eye(gain.shape[1]), loops.shape).copy()
    weight = family.q + gain.T @ family.r @ gain
    for _ in range(horizon):
        result += power.transpose(0, 2, 1) @ weight @ power
        power = loops @ power
    return result


def cost_gradients(
    family: LinearFamily,
    grid: ConditionGrid,
    gain: np.ndarray,
    horizon: int,
    *,
    states: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate each cost through the closed-loop states, with q fixed."""
    m, n = gain.shape
    x = grid.resets.copy()
    sensitivity = np.zeros((len(x), n, m * n))
    gradients = np.zeros((len(x), m * n))
    costs = np.zeros(len(x))
    loops = family.closed_loops(gain)[grid.plant_ids]
    inputs = family.b[grid.plant_ids]
    weight = family.q + gain.T @ family.r @ gain
    if states is not None and states.shape != (len(x), horizon + 1, n):
        raise ValueError("Stored state trajectories have incompatible dimensions.")
    for t in range(horizon):
        if states is not None:
            x = states[:, t]
        costs += np.einsum("ci,ij,cj->c", x, weight, x)
        wx = x @ weight.T
        gradients += 2 * np.einsum("cip,ci->cp", sensitivity, wx)
        gradients += 2 * np.einsum("ci,cj->cij", x @ gain.T @ family.r.T, x).reshape(
            len(x), -1
        )
        direct = np.einsum("cai,cj->caij", inputs, x).reshape(len(x), n, m * n)
        sensitivity = np.einsum("cij,cjp->cip", loops, sensitivity) - direct
        if states is None:
            x = np.einsum("cij,cj->ci", loops, x)
    return costs, gradients.reshape(len(x), m, n)


def directional_derivatives(
    family: LinearFamily,
    grid: ConditionGrid,
    gain: np.ndarray,
    direction: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent first/second directional recurrences for diagnostics."""
    x = grid.resets.copy()
    dx, ddx = np.zeros_like(x), np.zeros_like(x)
    loops = family.closed_loops(gain)[grid.plant_ids]
    perturbation = -family.b[grid.plant_ids] @ direction
    first, second = np.zeros(len(x)), np.zeros(len(x))
    for _ in range(horizon):
        u = -x @ gain.T
        du = -dx @ gain.T - x @ direction.T
        ddu = -ddx @ gain.T - 2 * dx @ direction.T
        first += 2 * (
            np.einsum("ci,ij,cj->c", x, family.q, dx)
            + np.einsum("ci,ij,cj->c", u, family.r, du)
        )
        second += 2 * (
            np.einsum("ci,ij,cj->c", dx, family.q, dx)
            + np.einsum("ci,ij,cj->c", x, family.q, ddx)
            + np.einsum("ci,ij,cj->c", du, family.r, du)
            + np.einsum("ci,ij,cj->c", u, family.r, ddu)
        )
        ddx = np.einsum("cij,cj->ci", loops, ddx) + 2 * np.einsum(
            "cij,cj->ci", perturbation, dx
        )
        dx = np.einsum("cij,cj->ci", loops, dx) + np.einsum(
            "cij,cj->ci", perturbation, x
        )
        x = np.einsum("cij,cj->ci", loops, x)
    return first, second


def target_law(
    costs: np.ndarray,
    background: np.ndarray,
    threshold: float,
    name: str,
    *,
    width: float = 0.35,
    mixture: float = 0.5,
) -> TargetLaw:
    if (
        costs.shape != background.shape
        or not np.all(np.isfinite(costs))
        or np.any(background <= 0)
        or not np.isclose(background.sum(), 1)
    ):
        raise ValueError("Require finite costs and a full-support background law.")
    if width <= 0 or not 0 <= mixture < 1:
        raise ValueError("Invalid band width or mixture weight.")
    selected = np.ones(len(costs), dtype=bool)
    if name == "background":
        return TargetLaw(background.copy(), selected, False)
    if name == "boundary":
        selected = np.abs(costs - threshold) <= width
    elif name == "high_cost":
        order = np.lexsort((np.arange(len(costs)), -costs))
        selected = np.zeros(len(costs), dtype=bool)
        selected[order[: max(1, int(np.ceil(len(costs) / 4)))]] = True
    else:
        raise ValueError(f"Unknown target map: {name}")
    fallback = not np.any(selected)
    target = background.copy() if fallback else background * selected
    target /= target.sum()
    return TargetLaw((1 - mixture) * background + mixture * target, selected, fallback)


def margin_consumption(
    family: LinearFamily, gain: np.ndarray, p: np.ndarray, delta: np.ndarray
) -> np.ndarray:
    d = np.linalg.norm(family.b @ delta, ord=2, axis=(1, 2))
    gp = family.closed_loops(gain).transpose(0, 2, 1) @ p
    return 2 * np.linalg.norm(gp, ord=2, axis=(1, 2)) * d + np.linalg.norm(p, 2) * d**2


def _check_certificate(p: np.ndarray, n: int) -> None:
    if (
        p.shape != (n, n)
        or not np.all(np.isfinite(p))
        or not np.allclose(p, p.T, atol=1e-14, rtol=0)
        or np.linalg.eigvalsh(p).min() <= 0
    ):
        raise ValueError("Certificate must be symmetric positive definite.")


def step_capacity(
    family: LinearFamily,
    gain: np.ndarray,
    p: np.ndarray,
    direction: np.ndarray,
    spec: UpdateSpec,
) -> float:
    """Largest nonnegative step allowed by radius and consumption, not Armijo."""
    _check_certificate(p, gain.shape[1])
    margins = verified_margins(family.closed_loops(gain), p)
    if np.any(margins <= spec.verification_tolerance):
        return 0.0
    norm = np.linalg.norm(direction, 2)
    if norm <= spec.gradient_tolerance:
        return float("inf")
    d = np.linalg.norm(family.b @ direction, ord=2, axis=(1, 2))
    gp = np.linalg.norm(
        family.closed_loops(gain).transpose(0, 2, 1) @ p, ord=2, axis=(1, 2)
    )
    budget = (1 - spec.rho) * margins
    capacity = np.full_like(d, np.inf)
    nonzero = d > 0
    # Rationalized positive root avoids cancellation near a small budget.
    capacity[nonzero] = budget[nonzero] / (
        d[nonzero]
        * (
            np.sqrt(gp[nonzero] ** 2 + np.linalg.norm(p, 2) * budget[nonzero])
            + gp[nonzero]
        )
    )
    return float(min(spec.kappa / norm, capacity.min()))


def backtrack(
    family: LinearFamily,
    grid: ConditionGrid,
    gain: np.ndarray,
    p: np.ndarray,
    weights: np.ndarray,
    spec: UpdateSpec,
    *,
    evaluate=None,
) -> StepResult:
    """Evaluate callbacks may retain raw trial rollouts; they never select q."""
    _check_certificate(p, gain.shape[1])
    if (
        weights.shape != grid.weights.shape
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1)
    ):
        raise ValueError("Training weights must be a probability law on the grid.")
    costs, gradients = cost_gradients(family, grid, gain, spec.horizon)
    gradient = np.einsum("c,cij->ij", weights, gradients)
    direction = -gradient
    margins = verified_margins(family.closed_loops(gain), p)
    result = StepResult(gain.copy(), direction, margins, "verification_failed")
    if np.any(margins <= spec.verification_tolerance):
        return result
    norm_squared = float(np.sum(gradient**2))
    if np.sqrt(norm_squared) <= spec.gradient_tolerance:
        result.status = "zero_gradient"
        return result
    objective = float(weights @ costs)
    for index in range(spec.max_backtracks + 1):
        step = spec.initial_step * spec.shrink**index
        delta = step * direction
        consumption = margin_consumption(family, gain, p, delta)
        new_margins = verified_margins(family.closed_loops(gain + delta), p)
        radius_ok = bool(np.linalg.norm(delta, 2) <= spec.kappa)
        margin_ok = bool(np.all(consumption <= (1 - spec.rho) * margins))
        new_objective, armijo_ok = None, None
        if radius_ok and margin_ok:
            if evaluate is None:
                states, actions = rollout(family, grid, gain + delta, spec.horizon)
                new_costs = trajectory_costs(family, states, actions)
            else:
                new_costs = evaluate(gain + delta)
            new_objective = float(weights @ new_costs)
            armijo_ok = bool(
                new_objective <= objective - spec.armijo * step * norm_squared
            )
        result.trials.append(
            Trial(
                index,
                step,
                radius_ok,
                margin_ok,
                armijo_ok,
                consumption,
                new_margins,
                new_objective,
            )
        )
        if armijo_ok:
            result.gain = gain + delta
            result.step = step
            result.status = "accepted"
            return result
    result.status = "search_exhausted"
    return result


def capacity_step(
    family: LinearFamily,
    grid: ConditionGrid,
    gain: np.ndarray,
    p: np.ndarray,
    weights: np.ndarray,
    spec: UpdateSpec,
    *,
    cap: float | None = None,
    slack: float = 1e-9,
    shrink: float = 0.9,
    max_trials: int = 200,
) -> StepResult:
    """Try the largest step the guard allows, shrinking only if Armijo rejects it.

    `cap` limits the first trial; None leaves only the guard's limit. `slack` keeps
    that trial strictly inside the guard despite rounding. Trials run through
    `backtrack`, so the guard and Armijo tests exist in one place.
    """
    _, gradients = cost_gradients(family, grid, gain, spec.horizon)
    direction = -np.einsum("c,cij->ij", weights, gradients)
    step = (1 - slack) * step_capacity(family, gain, p, direction, spec)
    if cap is not None:
        step = min(cap, step)
    # A zero gradient or failed verification has no usable step; backtrack records it.
    first = float(step) if np.isfinite(step) and step > 0 else spec.initial_step
    trials = replace(
        spec, initial_step=first, shrink=shrink, max_backtracks=max_trials - 1
    )
    return backtrack(family, grid, gain, p, weights, trials)


def curvature_bound(
    family: LinearFamily,
    grid: ConditionGrid,
    gain: np.ndarray,
    direction: np.ndarray,
    spec: UpdateSpec,
) -> np.ndarray:
    """Uniform directional |J''| bound for ||s D||_2 <= kappa.

    X,U,V bound ||x||, ||dx/ds||, ||d²x/ds²||. Their recurrences hold at
    every point in the segment, not just at the old controller. W0,W1,W2
    bound the cost weight and its first two directional derivatives.
    """
    ids = grid.plant_ids
    bound_g = np.maximum(
        1,
        np.linalg.norm(family.closed_loops(gain), ord=2, axis=(1, 2))
        + np.linalg.norm(family.b, ord=2, axis=(1, 2)) * spec.kappa,
    )
    bd = np.linalg.norm(family.b @ direction, ord=2, axis=(1, 2))[ids]
    k_bound = np.linalg.norm(gain, 2) + spec.kappa
    d_norm = np.linalg.norm(direction, 2)
    r_norm = np.linalg.norm(family.r, 2)
    w0 = np.linalg.norm(family.q, 2) + r_norm * k_bound**2
    w1, w2 = 2 * r_norm * k_bound * d_norm, 2 * r_norm * d_norm**2
    x = np.linalg.norm(grid.resets, axis=1)
    u, v, result = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    for _ in range(spec.horizon):
        result += 2 * w0 * (u**2 + x * v) + 4 * w1 * x * u + w2 * x**2
        v = bound_g[ids] * v + 2 * bd * u
        u = bound_g[ids] * u + bd * x
        x = bound_g[ids] * x
    return result


def cost_drift_bound(
    family: LinearFamily,
    grid: ConditionGrid,
    gain: np.ndarray,
    delta: np.ndarray,
    spec: UpdateSpec,
) -> np.ndarray:
    """Lemma 15 with r = max ||x0||, uniformly over the condition domain."""
    loops = family.closed_loops(gain)
    b_norm = np.linalg.norm(family.b, ord=2, axis=(1, 2))
    m = np.maximum(1, np.linalg.norm(loops, ord=2, axis=(1, 2)) + b_norm * spec.kappa)
    first, second = np.zeros(len(m)), np.zeros(len(m))
    for t in range(spec.horizon):
        if t:
            first += t * m ** (2 * t - 1)
        second += m ** (2 * t)
    w = family.q + gain.T @ family.r @ gain
    lipschitz = 2 * np.linalg.norm(w, 2) * b_norm * first + (
        np.linalg.norm(family.r, 2)
        * (2 * np.linalg.norm(gain, 2) + spec.kappa)
        * second
    )
    radius_squared = np.max(np.sum(grid.resets**2, axis=1))
    return radius_squared * lipschitz[grid.plant_ids] * np.linalg.norm(delta, 2)


def coverage_accounting(
    old: np.ndarray,
    new: np.ndarray,
    weights: np.ndarray,
    threshold: float,
    alignment: np.ndarray,
    curvature: np.ndarray,
    step: float,
    drift: np.ndarray,
) -> dict:
    old_ok, new_ok = old <= threshold, new <= threshold
    upper = old - step * alignment + 0.5 * curvature * step**2
    guaranteed = ~old_ok & (upper <= threshold)
    possible = old_ok & (upper > threshold)
    band = old_ok & (old > threshold - drift)
    gained, lost = ~old_ok & new_ok, old_ok & ~new_ok
    return {
        "old_coverage": float(weights @ old_ok),
        "new_coverage": float(weights @ new_ok),
        "gained_mass": float(weights @ gained),
        "lost_mass": float(weights @ lost),
        "coverage_change": float(weights @ (new_ok.astype(float) - old_ok)),
        "directional_lower": float(weights @ guaranteed - weights @ possible),
        "band_lower": -float(weights @ band),
        "upper_cost": upper,
        "guaranteed_gains": guaranteed,
        "possible_losses": possible,
        "loss_band": band,
        "gained": gained,
        "lost": lost,
    }
