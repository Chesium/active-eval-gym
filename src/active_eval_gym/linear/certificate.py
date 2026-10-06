"""Common quadratic certificates: convex search, then independent verification.

The search solves an LMI and only proposes a matrix. A claim that V_P(x) = x'Px
decreases on a plant rests on verify_certificate, which recomputes every margin
with NumPy eigenvalues against a declared numerical tolerance.
"""

from dataclasses import dataclass
from importlib.metadata import version

import cvxpy as cp
import numpy as np

from active_eval_gym.linear.exact import closed_loop, decrease_margin
from active_eval_gym.linear.family import LinearFamily

SOLVER = "CLARABEL"


@dataclass(frozen=True, eq=False)
class CertificateSearch:
    """The solver's proposal; its margin is not a verification."""

    plant_ids: tuple[str, ...]
    lower: float
    upper: float
    solver: str
    package_versions: dict[str, str]
    status: str
    solver_margin: float | None
    matrix: np.ndarray | None


@dataclass(frozen=True)
class CertificateVerification:
    """Margins eta_theta = lambda_min(P - G'PG) recomputed for one gain."""

    tolerance: float
    eigenvalue_min: float
    eigenvalue_max: float
    plant_margins: dict[str, float]
    passed: bool


def max_margin_common_certificate(
    family: LinearFamily,
    gain: np.ndarray,
    *,
    lower: float,
    upper: float,
) -> CertificateSearch:
    """Maximize the smallest decrease margin over lower I <= P <= upper I.

    Solves max t subject to G_K(theta)'PG_K(theta) - P <= -t I on every plant.
    A nonpositive optimum means this family and gain admit no common quadratic
    certificate in the class; it does not mean any plant is unstable.
    """

    if not 0.0 < lower < upper:
        raise ValueError("Certificate bounds must satisfy 0 < lower < upper.")
    identity = np.eye(family.state_dimension)
    matrix = cp.Variable(identity.shape, symmetric=True)
    margin = cp.Variable()
    constraints = [matrix >> lower * identity, matrix << upper * identity]
    for plant in family.plants:
        g = closed_loop(plant, gain)
        constraints.append(g.T @ matrix @ g - matrix << -margin * identity)
    problem = cp.Problem(cp.Maximize(margin), constraints)
    problem.solve(solver=SOLVER)

    solved = problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    proposal = None
    if solved and matrix.value is not None:
        proposal = 0.5 * (matrix.value + matrix.value.T)
        proposal.setflags(write=False)
    return CertificateSearch(
        plant_ids=family.plant_ids,
        lower=float(lower),
        upper=float(upper),
        solver=SOLVER,
        package_versions={
            "cvxpy": version("cvxpy"),
            "clarabel": version("clarabel"),
        },
        status=str(problem.status),
        solver_margin=float(margin.value) if solved else None,
        matrix=proposal,
    )


def verify_certificate(
    certificate: np.ndarray,
    family: LinearFamily,
    gain: np.ndarray,
    *,
    tolerance: float,
) -> CertificateVerification:
    """Pass only if P > tolerance and every plant's margin exceeds tolerance."""

    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative.")
    margins = {
        plant.plant_id: decrease_margin(certificate, closed_loop(plant, gain))
        for plant in family.plants
    }
    matrix = np.asarray(certificate, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    passed = bool(
        eigenvalues[0] > tolerance
        and all(value > tolerance for value in margins.values())
    )
    return CertificateVerification(
        tolerance=float(tolerance),
        eigenvalue_min=float(eigenvalues[0]),
        eigenvalue_max=float(eigenvalues[-1]),
        plant_margins=margins,
        passed=passed,
    )
