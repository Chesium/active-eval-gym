"""Independent checks of the notebook's population and theorem calculations."""

import json

import numpy as np
import pytest

from active_eval_gym.certificates import (
    KNOWN_CERTIFICATE,
    CertificateSpec,
    bound_terms,
    candidate_grid,
    certificate_matrix,
    empirical_losses,
    loss_table,
    make_population,
    normalized_residuals,
    run_sweep,
    sample_draws,
    save_experiment,
    verified_margins,
    wilson_interval,
)


def test_population_matches_direct_frozen_feedback_rollout():
    pop = make_population(CertificateSpec(horizon=7, unsafe_probability=0.1))
    np.testing.assert_allclose(pop.weights.sum(), 1)
    for index in [0, 5, 70, 130]:
        g = pop.matrices[pop.plant_ids[index]]
        # Realize G as A-BK, with K=B=I, and roll out using actions.
        a, b, k = g + np.eye(2), np.eye(2), np.eye(2)
        x = pop.states[index, 0].copy()
        for t in range(pop.spec.horizon):
            action = -k @ x
            x = a @ x + b @ action
            np.testing.assert_allclose(x, pop.states[index, t + 1], atol=1e-14)


def test_vectorized_loss_matches_scalar_energy_differences():
    spec = CertificateSpec(horizon=5, directions=12, unsafe_probability=0.1)
    pop = make_population(spec)
    p = certificate_matrix(0.55, 1.8, 0.7)
    table = loss_table(pop, p)
    losses, violations = [], []
    for path in pop.states:
        worst = max(
            (y @ p @ y - x @ p @ x) / (x @ x) + spec.eta
            for x, y in zip(path[:-1], path[1:], strict=True)
        )
        losses.append(min(1, max(0, 1 + worst / spec.gamma)))
        violations.append(worst > 0)
    np.testing.assert_allclose(table.loss[0], losses, atol=1e-13)
    np.testing.assert_array_equal(table.violation[0], violations)
    assert table.population_loss[0] == pytest.approx(np.dot(losses, pop.weights))
    assert table.population_risk[0] == pytest.approx(np.dot(violations, pop.weights))
    # A normalized residual must be independent of nonzero trajectory scale.
    scaled = normalized_residuals(p, 1e-5 * pop.states, spec.eta)
    np.testing.assert_allclose(scaled, normalized_residuals(p, pop.states, spec.eta))
    zeros = normalized_residuals(p, np.zeros((1, 4, 2)), spec.eta)
    assert np.all(np.isneginf(zeros))


@pytest.mark.parametrize("q", [0, 0.1, 0.25])
def test_known_certificate_and_unavoidable_risk_floor(q):
    pop = make_population(CertificateSpec(unsafe_probability=q))
    margins = verified_margins(pop.matrices[:2], KNOWN_CERTIFICATE)
    assert np.all(margins > pop.spec.eta + pop.spec.gamma)
    known = loss_table(pop, KNOWN_CERTIFICATE)
    assert known.population_loss[0] == pytest.approx(q)
    assert known.population_risk[0] == pytest.approx(q)
    grid = candidate_grid(pop.spec, eigenvalue_count=4, angle_count=5)
    table = loss_table(pop, grid.matrices)
    assert np.all(table.population_risk >= q - 1e-14)
    if q:
        assert np.all(table.loss[:, pop.plant_ids == 2] == 1)


def test_grid_is_admissible_and_cover_radius_bounds_unseen_matrices():
    spec = CertificateSpec()
    grid = candidate_grid(spec, eigenvalue_count=5, angle_count=8)
    eigenvalues = np.linalg.eigvalsh(grid.matrices)
    assert eigenvalues.min() >= spec.a - 1e-14
    assert eigenvalues.max() <= spec.b + 1e-14
    random = np.random.default_rng(13)
    for _ in range(100):
        p = certificate_matrix(
            *random.uniform(spec.a, spec.b, 2), random.uniform(0, np.pi)
        )
        distance = np.linalg.norm(grid.matrices - p, axis=(1, 2)).min()
        assert distance <= grid.covering_radius


def test_bound_constants_match_independent_scalar_calculation():
    pop = make_population(CertificateSpec())
    approximation, sampling = bound_terms(pop, 512, 0.0002, 0.05)
    m = max(np.sqrt(np.linalg.eigvalsh(g.T @ g).max()) for g in pop.matrices)
    expected = 2 * ((m**2 + 1) / 0.1) * 0.0002
    expected += np.sqrt((3 * np.log(1 + 4 * np.sqrt(2) / 0.0002) + np.log(40)) / 1024)
    assert approximation + sampling == pytest.approx(expected)


def test_zero_and_all_exceedances_have_valid_plotting_intervals():
    low, high = wilson_interval(np.array([0, 50, 100]), 100)
    observed = np.array([0, 0.5, 1])
    assert np.all(low <= observed)
    assert np.all(high >= observed)
    assert high[0] == pytest.approx(0.0369934982)
    assert low[-1] == pytest.approx(1 - high[0])


def test_sweep_replays_draws_and_fits_only_sampled_trajectories(tmp_path):
    pop = make_population(CertificateSpec(directions=8, unsafe_probability=0.1))
    grid = candidate_grid(pop.spec, eigenvalue_count=4, angle_count=5)
    table = loss_table(pop, grid.matrices)
    result = run_sweep(
        pop,
        grid,
        table,
        sample_sizes=(4, 16),
        replicates=5,
        master_seed=17,
        joint_over_sizes=True,
    )
    assert result.delta_per_size == 0.025
    for replicate in range(5):
        np.testing.assert_array_equal(
            result.draws[replicate],
            sample_draws(pop, master_seed=17, replicate=replicate, size=16),
        )
        for j, n in enumerate(result.sample_sizes):
            # Direct sample averaging, independent of the implementation's counts.
            empirical = table.loss[:, result.draws[replicate, :n]].mean(axis=1)
            np.testing.assert_allclose(
                empirical_losses(table, result.draws[replicate, :n]),
                empirical,
                atol=1e-14,
            )
            fitted = result.fitted_indices[j, replicate]
            assert empirical[fitted] == pytest.approx(empirical.min())
            assert result.empirical_loss[j, replicate] == pytest.approx(empirical.min())
            assert result.grid_gap[j, replicate] == pytest.approx(
                np.max(np.abs(empirical - table.population_loss))
            )
            # P0 attains the universal lower bound contributed by unstable draws.
            unsafe_fraction = np.mean(pop.plant_ids[result.draws[replicate, :n]] == 2)
            assert empirical.min() == pytest.approx(unsafe_fraction)
    saved = save_experiment(tmp_path, pop, grid, table, result)
    metadata = json.loads((saved / "metadata.json").read_text())
    raw = np.load(saved / "raw_trajectories.npz")
    np.testing.assert_array_equal(raw["sampled_trajectory_indices"], result.draws)
    np.testing.assert_array_equal(raw["states"], pop.states)
    assert metadata["metric_version"] == "normalized-quadratic-ramp-v1"
    assert metadata["training_seed"] is None
    assert save_experiment(tmp_path, pop, grid, table, result) != saved
