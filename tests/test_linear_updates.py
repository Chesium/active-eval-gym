"""Independent checks of the linear-update mathematics and retained decisions."""

from dataclasses import replace

import numpy as np
import pytest

from active_eval_gym.certificates import KNOWN_CERTIFICATE, verified_margins
from active_eval_gym.linear_study import (
    CertificateChoice,
    StudySpec,
    choose_certificate,
    primary_fixture,
)
from active_eval_gym.linear_updates import (
    ConditionGrid,
    LinearFamily,
    UpdateSpec,
    backtrack,
    cost_drift_bound,
    cost_gradients,
    cost_matrices,
    coverage_accounting,
    curvature_bound,
    directional_derivatives,
    margin_consumption,
    rollout,
    step_capacity,
    target_law,
    trajectory_costs,
)


@pytest.fixture
def fixture():
    return primary_fixture(StudySpec(directions=8, reference_directions=16))


def costs(family, grid, gain, horizon):
    return trajectory_costs(family, *rollout(family, grid, gain, horizon))


def test_costs_and_full_parameter_gradient(fixture):
    family, grid, _, gain, spec = fixture
    values, gradients = cost_gradients(family, grid, gain, spec.horizon)
    np.testing.assert_allclose(values, costs(family, grid, gain, spec.horizon))
    matrices = cost_matrices(family, gain, spec.horizon)
    quadratic = np.einsum(
        "ci,cij,cj->c", grid.resets, matrices[grid.plant_ids], grid.resets
    )
    np.testing.assert_allclose(values, quadratic)
    for i in range(2):
        for j in range(2):
            perturbation = np.zeros_like(gain)
            perturbation[i, j] = 1e-6
            difference = (
                costs(family, grid, gain + perturbation, spec.horizon)
                - costs(family, grid, gain - perturbation, spec.horizon)
            ) / 2e-6
            np.testing.assert_allclose(
                gradients[:, i, j], difference, atol=2e-8, rtol=2e-8
            )
    stored, _ = rollout(family, grid, gain, spec.horizon)
    _, replayed = cost_gradients(family, grid, gain, spec.horizon, states=stored)
    np.testing.assert_array_equal(gradients, replayed)


def test_directional_second_derivative_and_uniform_envelope(fixture):
    family, grid, _, gain, spec = fixture
    direction = np.array([[0.1, -0.2], [0.15, 0.05]])
    first, second = directional_derivatives(family, grid, gain, direction, spec.horizon)
    old, gradients = cost_gradients(family, grid, gain, spec.horizon)
    np.testing.assert_allclose(first, np.einsum("cij,ij->c", gradients, direction))
    epsilon = 1e-4
    numerical = (
        costs(family, grid, gain + epsilon * direction, spec.horizon)
        - 2 * old
        + costs(family, grid, gain - epsilon * direction, spec.horizon)
    ) / epsilon**2
    np.testing.assert_allclose(second, numerical, atol=2e-5, rtol=2e-5)
    bound = curvature_bound(family, grid, gain, direction, spec)
    for step in np.linspace(0, spec.kappa / np.linalg.norm(direction, 2), 11):
        _, curvature = directional_derivatives(
            family, grid, gain + step * direction, direction, spec.horizon
        )
        assert np.all(np.abs(curvature) <= bound + 1e-9)
        actual = costs(family, grid, gain + step * direction, spec.horizon)
        assert np.all(
            np.abs(actual - old - step * first) <= 0.5 * bound * step**2 + 1e-9
        )


def test_backtracking_invariants_and_coverage_bounds(fixture):
    family, grid, reference, gain, spec = fixture
    result = backtrack(family, grid, gain, KNOWN_CERTIFICATE, grid.weights, spec)
    assert result.status == "accepted"
    delta = result.gain - gain
    consumption = margin_consumption(family, gain, KNOWN_CERTIFICATE, delta)
    margins = verified_margins(family.closed_loops(result.gain), KNOWN_CERTIFICATE)
    assert np.linalg.norm(delta, 2) <= spec.kappa
    assert np.all(consumption <= (1 - spec.rho) * result.margins)
    assert np.all(margins >= spec.rho * result.margins - 1e-12)
    old, gradients = cost_gradients(family, reference, gain, spec.horizon)
    new = costs(family, reference, result.gain, spec.horizon)
    alignment = -np.einsum("cij,ij->c", gradients, result.direction)
    curvature = curvature_bound(family, reference, gain, result.direction, spec)
    drift = cost_drift_bound(family, reference, gain, delta, spec)
    assert np.all(np.abs(new - old) <= drift + 1e-10)
    accounting = coverage_accounting(
        old,
        new,
        reference.weights,
        spec.threshold,
        alignment,
        curvature,
        result.step,
        drift,
    )
    assert np.all(~accounting["guaranteed_gains"] | accounting["gained"])
    assert np.all(~accounting["lost"] | accounting["possible_losses"])
    assert np.all(~accounting["lost"] | accounting["loss_band"])
    assert accounting["coverage_change"] >= accounting["band_lower"] - 1e-12
    assert accounting["coverage_change"] >= accounting["directional_lower"] - 1e-12
    gradient = np.einsum(
        "c,cij->ij", grid.weights, cost_gradients(family, grid, gain, spec.horizon)[1]
    )
    decrease = spec.armijo * result.step * np.sum(gradient**2)
    assert grid.weights @ costs(family, grid, result.gain, spec.horizon) <= (
        grid.weights @ costs(family, grid, gain, spec.horizon) - decrease
    )


def test_p_controls_step_not_direction_and_scaling_cancels(fixture):
    family, grid, _, gain, spec = fixture
    results = [
        backtrack(family, grid, gain, scale * KNOWN_CERTIFICATE, grid.weights, spec)
        for scale in (0.9, 1.0, 1.1)
    ]
    for result in results:
        np.testing.assert_array_equal(result.direction, results[0].direction)
        np.testing.assert_allclose(result.gain, results[0].gain, atol=1e-14)
    direction = results[0].direction
    capacities = [
        step_capacity(family, gain, scale * KNOWN_CERTIFICATE, direction, spec)
        for scale in (0.9, 1.0, 1.1)
    ]
    np.testing.assert_allclose(capacities, capacities[0])
    for p in (KNOWN_CERTIFICATE, np.diag([1.0, 1.5])):
        result = backtrack(family, grid, gain, p, grid.weights, spec)
        np.testing.assert_array_equal(result.direction, direction)
        capacity = step_capacity(family, gain, p, direction, spec)
        assert np.all(
            margin_consumption(family, gain, p, capacity * direction)
            <= (1 - spec.rho) * result.margins + 1e-12
        )
        larger = 1.001 * capacity * direction
        assert np.linalg.norm(larger, 2) > spec.kappa or np.any(
            margin_consumption(family, gain, p, larger)
            > (1 - spec.rho) * result.margins
        )


def test_explicit_edge_statuses_and_target_rules(fixture):
    family, grid, _, gain, spec = fixture
    costs_at_old = costs(family, grid, gain, spec.horizon)
    fallback = target_law(costs_at_old, grid.weights, 1000, "boundary")
    assert fallback.fallback
    np.testing.assert_allclose(fallback.weights, grid.weights)
    high = target_law(np.ones(8), np.full(8, 1 / 8), 1, "high_cost")
    np.testing.assert_array_equal(np.flatnonzero(high.selected), [0, 1])
    invalid = backtrack(family, grid, gain, np.eye(2), grid.weights, spec)
    assert invalid.status == "verification_failed"
    assert not invalid.trials
    exhausted = backtrack(
        family,
        grid,
        gain,
        KNOWN_CERTIFICATE,
        grid.weights,
        replace(spec, max_backtracks=0),
    )
    assert exhausted.status == "search_exhausted"
    np.testing.assert_array_equal(exhausted.gain, gain)
    zero_family = LinearFamily(family.a, family.b, np.zeros((2, 2)), np.zeros((2, 2)))
    zero = backtrack(zero_family, grid, gain, KNOWN_CERTIFICATE, grid.weights, spec)
    assert zero.status == "zero_gradient"
    assert not zero.trials
    with pytest.raises(ValueError, match="positive definite"):
        backtrack(family, grid, gain, -np.eye(2), grid.weights, spec)


def test_scalar_report_counterexample_and_positive_alignment():
    family = LinearFamily(
        np.array([[[0.0]], [[1.0]]]),
        np.ones((2, 1, 1)),
        np.ones((1, 1)),
        np.zeros((1, 1)),
    )
    grid = ConditionGrid(np.array([0, 1]), np.ones((2, 1)), np.full(2, 0.5))
    gain = np.array([[0.5]])
    spec = UpdateSpec(horizon=2, threshold=1.25, initial_step=0.4, armijo=0.5)
    weights = np.array([0.25, 0.75])
    result = backtrack(family, grid, gain, np.ones((1, 1)), weights, spec)
    np.testing.assert_allclose(result.gain, [[0.7]])
    before, after = costs(family, grid, gain, 2), costs(family, grid, result.gain, 2)
    assert weights @ before == pytest.approx(1.25)
    assert weights @ after == pytest.approx(1.19)
    assert grid.weights @ (after <= 1.25) == pytest.approx(0.5)
    np.testing.assert_allclose(result.trials[0].consumption, 0.24)
    assert result.trials[0].new_margins.min() == pytest.approx(0.51)
    aligned = LinearFamily(
        np.array([[[0.8]]]), np.ones((1, 1, 1)), np.ones((1, 1)), np.zeros((1, 1))
    )
    resets = np.array([[0.25], [0.5], [0.75], [1.0]])
    grid = ConditionGrid(np.zeros(4, dtype=int), resets, np.full(4, 0.25))
    gain = np.array([[0.2]])
    old = costs(aligned, grid, gain, 2)
    law = target_law(old, grid.weights, 1.2, "boundary")
    result = backtrack(
        aligned,
        grid,
        gain,
        np.ones((1, 1)),
        law.weights,
        UpdateSpec(horizon=2, threshold=1.2),
    )
    assert grid.weights @ (costs(aligned, grid, result.gain, 2) <= 1.2) == 1.0


def test_selectors_report_unverified_erm_and_keep_order(fixture):
    family, _, _, gain, spec = fixture
    matrices = np.array([np.eye(2), KNOWN_CERTIFICATE])
    probe = np.eye(2)
    first = choose_certificate(
        family, gain, matrices, np.zeros(2), probe, "erm_first", spec
    )
    assert first.candidate_index == 0
    assert isinstance(first, CertificateChoice)
    for name in ("erm_small_step", "erm_large_step"):
        failed = choose_certificate(
            family, gain, matrices, np.array([0.0, 1.0]), probe, name, spec
        )
        assert failed.status == "no_verified_erm"
        assert failed.matrix is None
        tied = choose_certificate(
            family, gain, matrices, np.zeros(2), np.zeros_like(gain), name, spec
        )
        assert tied.candidate_index == 1


def test_nonsquare_gain_and_one_step_horizon():
    family = LinearFamily(
        np.array([[[0.7, 0.1], [0.0, 0.5]]]),
        np.array([[[0.1], [0.2]]]),
        np.eye(2),
        np.array([[0.3]]),
    )
    grid = ConditionGrid(
        np.array([0, 0]), np.array([[1.0, 0.0], [0.4, 0.7]]), np.array([0.5, 0.5])
    )
    gain, direction = np.array([[0.2, 0.1]]), np.array([[0.1, -0.2]])
    for horizon in (1, 4):
        old, gradients = cost_gradients(family, grid, gain, horizon)
        first = np.einsum("cij,ij->c", gradients, direction)
        epsilon = 1e-6
        numerical = (
            costs(family, grid, gain + epsilon * direction, horizon)
            - costs(family, grid, gain - epsilon * direction, horizon)
        ) / (2 * epsilon)
        np.testing.assert_allclose(first, numerical, atol=1e-9)
        spec = UpdateSpec(horizon=horizon)
        bound = curvature_bound(family, grid, gain, direction, spec)
        step = 0.5
        actual = costs(family, grid, gain + step * direction, horizon)
        assert np.all(actual <= old + step * first + bound * step**2 / 2 + 1e-12)
