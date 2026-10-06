from pathlib import Path

import numpy as np
import pytest

from active_eval_gym.config import (
    load_linear_family,
    load_nominal_env_spec,
    load_policy_design,
)
from active_eval_gym.linear.family import (
    CARTPOLE_LENGTH_LINEARIZATION,
    EXPLICIT,
    LinearFamily,
    LinearPlant,
    cartpole_length_family,
)
from active_eval_gym.policies.lqr import design_cartpole_lqr

ROOT = Path(__file__).resolve().parents[1]
CARTPOLE_FAMILY = ROOT / "configs/linear/cartpole_length_linearization_v1.toml"
SCALAR_FAMILY = ROOT / "configs/linear/scalar_two_plant_v1.toml"
NOMINAL_CARTPOLE = ROOT / "configs/envs/cartpole_v1_nominal.toml"
LQR_CONFIG = ROOT / "configs/policies/cartpole_lqr_nominal_quantized_v1.toml"


def test_cartpole_family_matches_the_lqr_design_model_at_nominal_length() -> None:
    family = load_linear_family(CARTPOLE_FAMILY)
    model, _ = design_cartpole_lqr(load_policy_design(LQR_CONFIG))

    assert family.kind == CARTPOLE_LENGTH_LINEARIZATION
    assert len(family.plants) == 13
    assert (family.state_dimension, family.input_dimension) == (4, 1)
    assert family.provenance["nominal_spec_id"] == "cartpole-v1-nominal-v1"
    nominal = family.plant("length-p0p5")
    assert nominal.parameters == {"length": 0.5}
    np.testing.assert_array_equal(nominal.a, np.asarray(model.discrete_a))
    np.testing.assert_array_equal(nominal.b, np.asarray(model.discrete_b))


def test_cartpole_matrices_are_affine_in_inverse_length() -> None:
    nominal = load_nominal_env_spec(NOMINAL_CARTPOLE).parameters
    short, long = 0.075, 1.0
    middle = 2.0 / (1.0 / short + 1.0 / long)
    plants = cartpole_length_family("affine", nominal, [short, middle, long]).plants

    for name in ("a", "b"):
        first, center, last = (getattr(plant, name) for plant in plants)
        np.testing.assert_allclose(center, 0.5 * (first + last), rtol=0, atol=1e-12)


def test_cartpole_family_requires_the_euler_integrator() -> None:
    nominal = dict(load_nominal_env_spec(NOMINAL_CARTPOLE).parameters)
    nominal["kinematics_integrator"] = "semi-implicit euler"

    with pytest.raises(ValueError, match="integrator"):
        cartpole_length_family("semi-implicit", nominal, [0.5])


def test_explicit_family_loads_read_only_matrices() -> None:
    family = load_linear_family(SCALAR_FAMILY)

    assert family.kind == EXPLICIT
    assert family.plant_ids == ("theta-p0", "theta-p1")
    assert family.plant("theta-p1").parameters == {"theta": 1.0}
    np.testing.assert_array_equal(family.plant("theta-p1").a, [[1.0]])
    with pytest.raises(ValueError):
        family.plant("theta-p1").a[0, 0] = 2.0
    with pytest.raises(KeyError):
        family.plant("theta-p2")


def test_family_rejects_inconsistent_plants() -> None:
    scalar = LinearPlant("scalar", {}, [[0.5]], [[1.0]])
    planar = LinearPlant("planar", {}, np.eye(2), np.eye(2))

    with pytest.raises(ValueError, match="Duplicate"):
        LinearFamily(1, "duplicate", EXPLICIT, (scalar, scalar), {})
    with pytest.raises(ValueError, match="mixed dimensions"):
        LinearFamily(1, "mixed", EXPLICIT, (scalar, planar), {})
    with pytest.raises(ValueError, match="rows"):
        LinearPlant("mismatch", {}, np.eye(2), [[1.0]])
    with pytest.raises(ValueError, match="finite"):
        LinearPlant("nan", {}, [[np.nan]], [[1.0]])


def test_linear_family_loader_rejects_unknown_kinds_and_missing_matrices(
    tmp_path: Path,
) -> None:
    unknown = tmp_path / "unknown.toml"
    unknown.write_text('schema_version = 1\nfamily_id = "x"\nkind = "unknown-v1"\n')
    missing = tmp_path / "missing.toml"
    missing.write_text(
        'schema_version = 1\nfamily_id = "x"\nkind = "explicit-v1"\n'
        '[[plants]]\nplant_id = "p"\na = [[1.0]]\n'
    )

    with pytest.raises(ValueError, match="Unsupported"):
        load_linear_family(unknown)
    with pytest.raises(ValueError, match="matrix b"):
        load_linear_family(missing)
