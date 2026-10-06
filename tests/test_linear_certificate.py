from pathlib import Path

import numpy as np
import pytest

from active_eval_gym.config import load_linear_family, load_policy_design
from active_eval_gym.linear.certificate import (
    max_margin_common_certificate,
    verify_certificate,
)
from active_eval_gym.linear.exact import closed_loop
from active_eval_gym.linear.family import (
    EXPLICIT,
    LinearFamily,
    LinearPlant,
    cartpole_length_family,
)
from active_eval_gym.policies.lqr import design_cartpole_lqr

ROOT = Path(__file__).resolve().parents[1]
CARTPOLE_FAMILY = ROOT / "configs/linear/cartpole_length_linearization_v1.toml"
SCALAR_FAMILY = ROOT / "configs/linear/scalar_two_plant_v1.toml"
LQR_CONFIG = ROOT / "configs/policies/cartpole_lqr_nominal_quantized_v1.toml"


def test_scalar_family_search_proposes_and_verification_confirms() -> None:
    family = load_linear_family(SCALAR_FAMILY)
    gain = np.array([[0.5]])

    search = max_margin_common_certificate(family, gain, lower=0.5, upper=1.0)
    verification = verify_certificate(search.matrix, family, gain, tolerance=1e-9)

    assert search.status == "optimal"
    assert search.plant_ids == ("theta-p0", "theta-p1")
    assert set(search.package_versions) == {"cvxpy", "clarabel"}
    assert search.solver_margin == pytest.approx(0.75, abs=1e-6)
    np.testing.assert_allclose(search.matrix, [[1.0]], atol=1e-6)
    assert verification.passed
    assert verification.plant_margins == pytest.approx(
        {"theta-p0": 0.75, "theta-p1": 0.75}, abs=1e-6
    )


def test_an_unstable_plant_leaves_no_common_certificate() -> None:
    family = LinearFamily(
        1,
        "one-unstable",
        EXPLICIT,
        (
            LinearPlant("stable", {}, [[0.0]], [[1.0]]),
            LinearPlant("unstable", {}, [[2.0]], [[1.0]]),
        ),
        {},
    )
    gain = np.array([[0.5]])

    search = max_margin_common_certificate(family, gain, lower=0.5, upper=1.0)
    verification = verify_certificate(search.matrix, family, gain, tolerance=0.0)

    # The best certificate is P = 0.5, where the unstable margin is -1.25 P.
    assert search.solver_margin == pytest.approx(-0.625, abs=1e-6)
    assert not verification.passed
    margins = verification.plant_margins
    assert margins["unstable"] < 0.0 < margins["stable"]


def test_nominal_lqr_gain_stabilizes_only_intermediate_linearized_lengths() -> None:
    family = load_linear_family(CARTPOLE_FAMILY)
    gain = _nominal_lqr_gain()

    stable = [
        plant.parameters["length"]
        for plant in family.plants
        if np.max(np.abs(np.linalg.eigvals(closed_loop(plant, gain)))) < 1.0
    ]

    assert stable == [0.075, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


def test_certificate_verified_at_two_lengths_covers_every_length_between() -> None:
    full = load_linear_family(CARTPOLE_FAMILY)
    nominal = full.provenance["nominal_parameters"]
    gain = _nominal_lqr_gain()
    endpoints = cartpole_length_family("endpoints", nominal, [0.075, 1.0])
    interior = cartpole_length_family(
        "interior",
        nominal,
        [
            plant.parameters["length"]
            for plant in full.plants
            if 0.075 < plant.parameters["length"] < 1.0
        ],
    )

    search = max_margin_common_certificate(endpoints, gain, lower=1e-4, upper=1.0)
    at_endpoints = verify_certificate(search.matrix, endpoints, gain, tolerance=1e-9)
    between = verify_certificate(search.matrix, interior, gain, tolerance=1e-9)

    assert at_endpoints.passed
    assert between.passed
    assert len(between.plant_margins) == 5
    assert min(between.plant_margins.values()) >= (
        min(at_endpoints.plant_margins.values()) - 1e-12
    )


def test_verification_rejects_an_indefinite_certificate() -> None:
    family = load_linear_family(SCALAR_FAMILY)

    verification = verify_certificate([[-1.0]], family, [[0.5]], tolerance=0.0)

    assert verification.eigenvalue_min == -1.0
    assert not verification.passed


def _nominal_lqr_gain() -> np.ndarray:
    model, _ = design_cartpole_lqr(load_policy_design(LQR_CONFIG))
    return np.asarray(model.k)
