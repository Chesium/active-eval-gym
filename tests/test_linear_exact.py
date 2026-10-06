from math import log, sqrt
from pathlib import Path

import numpy as np
import pytest

from active_eval_gym.config import load_linear_family
from active_eval_gym.linear.exact import (
    QuadraticCost,
    WeightedCondition,
    certificate_generalization_slack,
    certificate_loss_lipschitz,
    closed_loop,
    closed_loop_states,
    cost_drift_bound,
    cost_matrix,
    decrease_margin,
    decrease_residual,
    preservation_bound,
    preservation_checks,
    training_objective,
    training_objective_gradient,
    trajectory_certificate_loss,
    trajectory_cost,
)
from active_eval_gym.linear.family import EXPLICIT, LinearFamily, LinearPlant

ROOT = Path(__file__).resolve().parents[1]
SCALAR_FAMILY = ROOT / "configs/linear/scalar_two_plant_v1.toml"
RANDOM_CASES = 200


def test_scalar_counterexample_from_the_design_notes() -> None:
    family = load_linear_family(SCALAR_FAMILY)
    cost = QuadraticCost(q=[[1.0]], r=[[0.0]], horizon=2)
    gain, start = np.array([[0.5]]), np.array([1.0])
    upper_boundary = [WeightedCondition("theta-p1", start, 1.0)]

    gradient = training_objective_gradient(family, gain, upper_boundary, cost)
    new_gain = gain - 0.2 * gradient

    np.testing.assert_allclose(gradient, [[-1.0]])
    np.testing.assert_allclose(new_gain, [[0.7]])
    assert training_objective(family, gain, upper_boundary, cost) == pytest.approx(1.25)
    assert training_objective(family, new_gain, upper_boundary, cost) == pytest.approx(
        1.09
    )
    old_costs = [trajectory_cost(p, gain, start, cost) for p in family.plants]
    new_costs = [trajectory_cost(p, new_gain, start, cost) for p in family.plants]
    assert old_costs == pytest.approx([1.25, 1.25])
    assert new_costs == pytest.approx([1.49, 1.09])
    threshold = 1.25
    assert np.mean([value <= threshold for value in old_costs]) == 1.0
    assert np.mean([value <= threshold for value in new_costs]) == 0.5

    certificate = np.array([[1.0]])
    checks = preservation_checks(
        certificate, family, gain, new_gain - gain, retained_fraction=0.5
    )
    assert [check.old_margin for check in checks] == pytest.approx([0.75, 0.75])
    assert [check.perturbation_bound for check in checks] == pytest.approx([0.24, 0.24])
    assert [check.allowed_bound for check in checks] == pytest.approx([0.375, 0.375])
    assert all(check.passed for check in checks)
    new_margins = [
        decrease_margin(certificate, closed_loop(plant, new_gain))
        for plant in family.plants
    ]
    assert new_margins == pytest.approx([0.51, 0.91])


def test_fixed_plant_ellipse_example_cost_matrices_and_drift() -> None:
    plant = LinearPlant("identity", {}, np.eye(2), np.eye(2))
    cost = QuadraticCost(q=np.eye(2), r=np.zeros((2, 2)), horizon=2)
    gain, new_gain = np.diag([0.5, 0.5]), np.diag([0.7, 0.3])

    old = cost_matrix(plant, gain, cost)
    new = cost_matrix(plant, new_gain, cost)
    drift = np.linalg.norm(new - old, 2)

    np.testing.assert_allclose(old, np.diag([1.25, 1.25]))
    np.testing.assert_allclose(new, np.diag([1.09, 1.49]))
    assert 1.2**2 * drift == pytest.approx(0.3456)
    assert cost_drift_bound(plant, gain, new_gain, cost) == pytest.approx(0.28)


def test_cost_matrix_matches_the_rollout_sum() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        n, m = int(rng.integers(1, 5)), int(rng.integers(1, 3))
        plant = _random_plant(rng, n, m)
        gain = 0.5 * rng.standard_normal((m, n))
        cost = QuadraticCost(_psd(rng, n), _psd(rng, m), int(rng.integers(1, 8)))
        start = rng.standard_normal(n)

        states = closed_loop_states(closed_loop(plant, gain), start, cost.horizon)
        expected = sum(
            x @ cost.q @ x + (gain @ x) @ cost.r @ (gain @ x) for x in states[:-1]
        )
        assert trajectory_cost(plant, gain, start, cost) == pytest.approx(
            expected, rel=1e-10, abs=1e-12
        )


def test_objective_gradient_matches_central_differences() -> None:
    rng = np.random.default_rng(1)
    n, m = 3, 2
    family = LinearFamily(
        1,
        "random",
        EXPLICIT,
        (_random_plant(rng, n, m, "first"), _random_plant(rng, n, m, "second")),
        {},
    )
    cost = QuadraticCost(_psd(rng, n), _psd(rng, m), 6)
    gain = 0.3 * rng.standard_normal((m, n))
    conditions = [
        WeightedCondition(plant_id, rng.standard_normal(n), weight)
        for plant_id, weight in [
            ("first", 0.1),
            ("first", 0.2),
            ("second", 0.3),
            ("second", 0.4),
        ]
    ]

    gradient = training_objective_gradient(family, gain, conditions, cost)
    numeric = np.zeros_like(gain)
    step = 1e-6
    for index in np.ndindex(gain.shape):
        offset = np.zeros_like(gain)
        offset[index] = step
        numeric[index] = (
            training_objective(family, gain + offset, conditions, cost)
            - training_objective(family, gain - offset, conditions, cost)
        ) / (2.0 * step)
    np.testing.assert_allclose(
        gradient, numeric, rtol=1e-6, atol=1e-6 * np.max(np.abs(numeric))
    )


def test_preservation_bound_lower_bounds_every_new_margin() -> None:
    rng = np.random.default_rng(2)
    for _ in range(RANDOM_CASES):
        n, m = int(rng.integers(1, 5)), int(rng.integers(1, 3))
        plant = _random_plant(rng, n, m)
        gain = 0.5 * rng.standard_normal((m, n))
        step = 0.1 * rng.standard_normal((m, n))
        certificate = _psd(rng, n) + 0.1 * np.eye(n)

        old = decrease_margin(certificate, closed_loop(plant, gain))
        new = decrease_margin(certificate, closed_loop(plant, gain + step))
        bound = preservation_bound(certificate, plant, gain, step)
        assert new >= old - bound - 1e-9 * (1.0 + abs(old) + bound)


def test_cost_drift_bound_dominates_the_exact_drift() -> None:
    rng = np.random.default_rng(3)
    for _ in range(RANDOM_CASES):
        n, m = int(rng.integers(1, 5)), int(rng.integers(1, 3))
        plant = _random_plant(rng, n, m)
        gain = 0.5 * rng.standard_normal((m, n))
        new_gain = gain + 0.1 * rng.standard_normal((m, n))
        cost = QuadraticCost(_psd(rng, n), _psd(rng, m), int(rng.integers(1, 8)))

        exact = np.linalg.norm(
            cost_matrix(plant, new_gain, cost) - cost_matrix(plant, gain, cost), 2
        )
        bound = cost_drift_bound(plant, gain, new_gain, cost)
        assert bound >= exact - 1e-9 * (1.0 + exact)


def test_cost_drift_bound_overflows_to_a_vacuous_bound_not_nan() -> None:
    plant = LinearPlant("expanding", {}, [[8.0]], [[1.0]])
    cost = QuadraticCost(q=[[1.0]], r=[[0.0]], horizon=500)

    assert cost_drift_bound(plant, [[0.0]], [[0.1]], cost) == np.inf
    assert cost_drift_bound(plant, [[0.0]], [[0.0]], cost) == 0.0


def test_certificate_loss_is_lipschitz_in_the_certificate() -> None:
    rng = np.random.default_rng(4)
    decrease, learning_margin = 0.1, 0.05
    for _ in range(RANDOM_CASES):
        n = int(rng.integers(1, 5))
        g = rng.standard_normal((n, n)) / sqrt(n)
        states = closed_loop_states(g, rng.standard_normal(n), int(rng.integers(1, 6)))
        certificate = _psd(rng, n) + 0.1 * np.eye(n)
        perturbation = _symmetric(0.01 * rng.standard_normal((n, n)))
        distance = np.linalg.norm(perturbation, "fro")
        m = np.linalg.norm(g, 2)

        residual_gap = abs(
            decrease_residual(certificate, states[0], states[1], decrease=decrease)
            - decrease_residual(
                certificate + perturbation, states[0], states[1], decrease=decrease
            )
        )
        loss_gap = abs(
            trajectory_certificate_loss(
                certificate,
                states,
                decrease=decrease,
                learning_margin=learning_margin,
            )
            - trajectory_certificate_loss(
                certificate + perturbation,
                states,
                decrease=decrease,
                learning_margin=learning_margin,
            )
        )
        assert residual_gap <= (m**2 + 1.0) * distance + 1e-12
        assert (
            loss_gap
            <= certificate_loss_lipschitz(m, learning_margin) * distance + 1e-12
        )


def test_trajectory_certificate_loss_is_a_clipped_ramp() -> None:
    certificate = [[1.0]]
    states = closed_loop_states([[0.5]], [1.0], 3)

    def loss(decrease: float) -> float:
        return trajectory_certificate_loss(
            certificate, states, decrease=decrease, learning_margin=0.1
        )

    # Every normalized residual equals decrease - 0.75 for G = 0.5 and P = 1.
    assert loss(0.5) == 0.0
    assert loss(0.7) == pytest.approx(0.5)
    assert loss(0.8) == 1.0
    assert (
        trajectory_certificate_loss(
            certificate, np.zeros((4, 1)), decrease=0.5, learning_margin=0.1
        )
        == 0.0
    )
    with pytest.raises(ValueError, match="zero state"):
        decrease_residual(certificate, [0.0], [0.0], decrease=0.5)


def test_generalization_slack_matches_the_design_note_formula() -> None:
    arguments = {
        "state_dimension": 1,
        "certificate_upper": 1.0,
        "lipschitz": 1.0,
        "net_radius": 0.01,
        "delta": 0.1,
    }
    slack = certificate_generalization_slack(trajectories=100, **arguments)

    assert slack == pytest.approx(
        0.02 + sqrt((log(1.0 + 2.0 / 0.01) + log(20.0)) / 200.0)
    )
    assert certificate_generalization_slack(trajectories=1000, **arguments) < slack


def test_preservation_checks_require_a_decreasing_certificate() -> None:
    family = load_linear_family(SCALAR_FAMILY)
    certificate = np.array([[1.0]])

    with pytest.raises(ValueError, match="does not apply"):
        preservation_checks(
            certificate, family, [[2.5]], [[0.0]], retained_fraction=0.5
        )
    for fraction in (0.0, 1.0):
        with pytest.raises(ValueError, match="retained_fraction"):
            preservation_checks(
                certificate, family, [[0.5]], [[0.0]], retained_fraction=fraction
            )


def test_inputs_are_validated() -> None:
    family = load_linear_family(SCALAR_FAMILY)
    cost = QuadraticCost(q=[[1.0]], r=[[0.0]], horizon=2)

    with pytest.raises(ValueError, match="positive semidefinite"):
        QuadraticCost(q=[[-1.0]], r=[[0.0]], horizon=2)
    with pytest.raises(ValueError, match="symmetric"):
        QuadraticCost(q=[[1.0, 1.0], [0.0, 1.0]], r=[[0.0]], horizon=2)
    with pytest.raises(ValueError, match="horizon"):
        QuadraticCost(q=[[1.0]], r=[[0.0]], horizon=0)
    with pytest.raises(ValueError, match="sum to one"):
        training_objective(
            family, [[0.5]], [WeightedCondition("theta-p0", [1.0], 0.5)], cost
        )
    with pytest.raises(KeyError):
        training_objective(
            family, [[0.5]], [WeightedCondition("theta-p9", [1.0], 1.0)], cost
        )
    with pytest.raises(ValueError, match="gain shape"):
        closed_loop(family.plants[0], [[0.5, 0.5]])
    with pytest.raises(ValueError, match="symmetric"):
        decrease_margin([[1.0, 1.0], [0.0, 1.0]], np.eye(2))


def _random_plant(
    rng: np.random.Generator, n: int, m: int, plant_id: str = "random"
) -> LinearPlant:
    return LinearPlant(
        plant_id,
        {},
        rng.standard_normal((n, n)) / sqrt(n),
        rng.standard_normal((n, m)) / sqrt(n),
    )


def _psd(rng: np.random.Generator, n: int) -> np.ndarray:
    factor = rng.standard_normal((n, n))
    return _symmetric(factor @ factor.T / n)


def _symmetric(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)
