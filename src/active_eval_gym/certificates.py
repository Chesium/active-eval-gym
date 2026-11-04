"""Finite-population experiment for the report's frozen-controller theorem.

This is an analytical research fixture, separate from the Gym rollout pipeline.
Only P is fitted. The known closed loops, sampling law, and gain never adapt.
"""

import json
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np

METRIC_VERSION = "normalized-quadratic-ramp-v1"
VERIFICATION_TOLERANCE = 1e-10
STABLE_PLANTS = np.array([[[0.8, 0.5], [0.0, 0.6]], [[0.7, 0.6], [0.0, 0.5]]])
KNOWN_CERTIFICATE = np.diag([0.6, 1.6])


@dataclass(frozen=True)
class CertificateSpec:
    a: float = 0.5
    b: float = 2.0
    eta: float = 0.02
    gamma: float = 0.1
    horizon: int = 10
    directions: int = 64
    unsafe_probability: float = 0.0

    def __post_init__(self) -> None:
        if not (0 < self.a < self.b < np.inf):
            raise ValueError("Require finite 0 < a < b.")
        if not (0 < self.eta < np.inf and 0 < self.gamma < np.inf):
            raise ValueError("eta and gamma must be finite and positive.")
        if not isinstance(self.horizon, int) or self.horizon < 1:
            raise ValueError("horizon must be a positive integer.")
        if not isinstance(self.directions, int) or self.directions < 4:
            raise ValueError("directions must be an integer >= 4.")
        if not 0 <= self.unsafe_probability < 1:
            raise ValueError("unsafe_probability must lie in [0, 1).")


@dataclass(frozen=True)
class Population:
    spec: CertificateSpec
    matrices: np.ndarray
    plant_ids: np.ndarray
    angles: np.ndarray
    states: np.ndarray
    weights: np.ndarray

    @property
    def dynamics_bound(self) -> float:
        return float(np.linalg.norm(self.matrices, ord=2, axis=(1, 2)).max())


@dataclass(frozen=True)
class CandidateGrid:
    matrices: np.ndarray
    covering_radius: float
    eigenvalue_count: int
    angle_count: int


@dataclass(frozen=True)
class LossTable:
    loss: np.ndarray
    violation: np.ndarray
    worst_residual: np.ndarray
    population_loss: np.ndarray
    population_risk: np.ndarray


@dataclass(frozen=True)
class SweepResult:
    sample_sizes: np.ndarray
    master_seed: int
    draws: np.ndarray
    delta_per_size: float
    epsilon: np.ndarray
    bound: np.ndarray
    fitted_indices: np.ndarray
    empirical_loss: np.ndarray
    population_loss: np.ndarray
    population_risk: np.ndarray
    grid_gap: np.ndarray

    @property
    def fitted_gap(self) -> np.ndarray:
        return np.abs(self.population_loss - self.empirical_loss)

    @property
    def risk_upper(self) -> np.ndarray:
        return np.minimum(1.0, self.empirical_loss + self.bound[:, None])


def make_population(spec: CertificateSpec) -> Population:
    """Enumerate the exact law on plants × equally spaced unit-circle resets."""
    matrices = STABLE_PLANTS.copy()
    probabilities = np.array([0.5, 0.5])
    if spec.unsafe_probability > 0:
        matrices = np.concatenate([matrices, [1.02 * np.eye(2)]])
        q = spec.unsafe_probability
        probabilities = np.array([(1 - q) / 2, (1 - q) / 2, q])
    angles = np.tile(
        np.arange(spec.directions) * 2 * np.pi / spec.directions, len(matrices)
    )
    plant_ids = np.repeat(np.arange(len(matrices)), spec.directions)
    states = np.empty((len(angles), spec.horizon + 1, 2))
    states[:, 0] = np.column_stack([np.cos(angles), np.sin(angles)])
    for t in range(spec.horizon):
        states[:, t + 1] = np.einsum("sij,sj->si", matrices[plant_ids], states[:, t])
    weights = np.repeat(probabilities / spec.directions, spec.directions)
    return Population(spec, matrices, plant_ids, angles, states, weights)


def certificate_matrix(low: float, high: float, angle: float) -> np.ndarray:
    """Rotate diag(low, high); angle is in radians."""
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    return rotation @ np.diag([low, high]) @ rotation.T


def candidate_grid(
    spec: CertificateSpec, *, eigenvalue_count: int = 9, angle_count: int = 24
) -> CandidateGrid:
    """A deterministic cover, with a conservative Frobenius covering radius.

    Eigenvalue rounding costs <= sqrt(2)*step/2. Rotation rounding costs
    <= sqrt(2)*(b-a)*sin(pi/(2*angle_count)). Centers all remain admissible.
    """
    if eigenvalue_count < 2 or angle_count < 1:
        raise ValueError("Need >= 2 eigenvalues and >= 1 angle.")
    values = np.linspace(spec.a, spec.b, eigenvalue_count)
    matrices = [
        certificate_matrix(lo, hi, theta)
        for lo in values
        for hi in values
        if lo <= hi
        for theta in np.arange(angle_count) * np.pi / angle_count
    ]
    if spec.a <= 0.6 and spec.b >= 1.6:
        matrices.append(KNOWN_CERTIFICATE.copy())
    matrices = np.asarray(matrices)
    # Remove redundant isotropic matrices without changing the cover or ordering.
    _, indices = np.unique(
        np.round(matrices.reshape(-1, 4), 12), axis=0, return_index=True
    )
    matrices = matrices[np.sort(indices)]
    radius = np.sqrt(2) * (
        (spec.b - spec.a) / (2 * (eigenvalue_count - 1))
        + (spec.b - spec.a) * np.sin(np.pi / (2 * angle_count))
    )
    return CandidateGrid(matrices, float(radius), eigenvalue_count, angle_count)


def normalized_residuals(
    matrices: np.ndarray, states: np.ndarray, eta: float
) -> np.ndarray:
    """Return (candidate, trajectory, transition); zero states get -infinity.

    The caller supplies homogeneous linear trajectories, so a zero state has
    a zero successor and its desired decrease condition holds automatically.
    """
    matrices = np.asarray(matrices, dtype=float)
    if matrices.ndim == 2:
        matrices = matrices[None]
    x, successor = states[:, :-1], states[:, 1:]
    denominator = np.sum(x * x, axis=-1)
    energy = np.einsum("sti,cij,stj->cst", x, matrices, x, optimize=False)
    next_energy = np.einsum(
        "sti,cij,stj->cst", successor, matrices, successor, optimize=False
    )
    residual = np.full_like(energy, -np.inf)
    np.divide(
        next_energy - energy,
        denominator[None],
        out=residual,
        where=denominator[None] > 0,
    )
    return residual + eta


def loss_table(population: Population, matrices: np.ndarray) -> LossTable:
    residual = normalized_residuals(matrices, population.states, population.spec.eta)
    worst = residual.max(axis=-1)
    loss = np.clip(1 + worst / population.spec.gamma, 0, 1)
    violation = worst > 0
    return LossTable(
        loss,
        violation,
        worst,
        np.einsum("cs,s->c", loss, population.weights, optimize=False),
        np.einsum("cs,s->c", violation, population.weights, optimize=False),
    )


def verified_margins(matrices: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.array([np.linalg.eigvalsh(p - g.T @ p @ g).min() for g in matrices])


def bound_terms(
    population: Population, n: int, epsilon: np.ndarray | float, delta: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    if n < 1 or not 0 < delta < 1 or np.any(np.asarray(epsilon) <= 0):
        raise ValueError("Require N >= 1, epsilon > 0, and delta in (0, 1).")
    lipschitz = (population.dynamics_bound**2 + 1) / population.spec.gamma
    approximation = 2 * lipschitz * np.asarray(epsilon)
    sampling = np.sqrt(
        (
            3 * np.log1p(2 * population.spec.b * np.sqrt(2) / np.asarray(epsilon))
            + np.log(2 / delta)
        )
        / (2 * n)
    )
    return approximation, sampling


def choose_epsilon(population: Population, n: int, delta: float) -> tuple[float, float]:
    """Choose resolution using theorem constants only, never sampled losses."""
    values = np.geomspace(1e-9, 1, 4096)
    approximation, sampling = bound_terms(population, n, values, delta)
    total = approximation + sampling
    index = int(total.argmin())
    return float(values[index]), float(total[index])


def sample_draws(
    population: Population, *, master_seed: int, replicate: int, size: int
) -> np.ndarray:
    """Named independent replicate streams; prefixes give nested sample sizes."""
    random = np.random.default_rng(np.random.SeedSequence([master_seed, replicate]))
    return random.choice(len(population.weights), size=size, p=population.weights)


def empirical_losses(table: LossTable, draws: np.ndarray) -> np.ndarray:
    counts = np.bincount(draws, minlength=table.loss.shape[1])
    return np.einsum("cs,s->c", table.loss, counts / len(draws), optimize=False)


def run_sweep(
    population: Population,
    grid: CandidateGrid,
    table: LossTable,
    *,
    sample_sizes: tuple[int, ...] = (32, 128, 512, 2048, 8192),
    replicates: int = 100,
    master_seed: int = 6180,
    delta: float = 0.05,
    joint_over_sizes: bool = False,
) -> SweepResult:
    if replicates < 1 or not sample_sizes or any(n < 1 for n in sample_sizes):
        raise ValueError("Need positive sample sizes and replicate count.")
    if table.loss.shape[0] != len(grid.matrices):
        raise ValueError("Loss table and grid disagree.")
    sizes = np.asarray(sample_sizes, dtype=int)
    draws = np.stack(
        [
            sample_draws(
                population, master_seed=master_seed, replicate=i, size=int(sizes.max())
            )
            for i in range(replicates)
        ]
    )
    per_size = delta / len(sizes) if joint_over_sizes else delta
    eps_bound = np.array([choose_epsilon(population, int(n), per_size) for n in sizes])
    shape = (len(sizes), replicates)
    fitted = np.empty(shape, dtype=int)
    empirical, population_loss, risk, gaps = [np.empty(shape) for _ in range(4)]
    for j, n in enumerate(sizes):
        counts = np.stack(
            [np.bincount(row[:n], minlength=len(population.weights)) for row in draws]
        )
        # These small weighted sums need no multithreaded BLAS in the kernel.
        empirical_all = np.einsum("cs,sr->cr", table.loss, counts.T / n, optimize=False)
        fitted[j] = empirical_all.argmin(axis=0)  # fixed first-index tie break
        empirical[j] = empirical_all[fitted[j], np.arange(replicates)]
        population_loss[j] = table.population_loss[fitted[j]]
        risk[j] = table.population_risk[fitted[j]]
        gaps[j] = np.max(np.abs(empirical_all - table.population_loss[:, None]), axis=0)
    return SweepResult(
        sizes,
        master_seed,
        draws,
        per_size,
        eps_bound[:, 0],
        eps_bound[:, 1],
        fitted,
        empirical,
        population_loss,
        risk,
        gaps,
    )


def wilson_interval(
    successes: np.ndarray, trials: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise 95% Wilson intervals for repeated-dataset exceedance counts."""
    p = np.asarray(successes) / trials
    z = 1.959963984540054
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    radius = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    # Roundoff at zero/all exceedances must not put the estimate outside its CI.
    return (
        np.maximum(0, np.minimum(p, center - radius)),
        np.minimum(1, np.maximum(p, center + radius)),
    )


def save_experiment(
    parent: Path,
    population: Population,
    grid: CandidateGrid,
    table: LossTable,
    result: SweepResult,
    *,
    source_paths: tuple[Path, ...] = (),
) -> Path:
    """New directory per export; raw states and sample indices precede metrics."""
    destination = parent / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    raw_path = destination / "raw_trajectories.npz"
    np.savez_compressed(
        raw_path,
        states=population.states,
        closed_loop_matrices=population.matrices,
        plant_ids=population.plant_ids,
        angles=population.angles,
        population_weights=population.weights,
        sampled_trajectory_indices=result.draws,
    )
    np.savez_compressed(
        destination / "metrics_v1.npz",
        candidates=grid.matrices,
        loss_table=table.loss,
        violation_table=table.violation,
        sample_sizes=result.sample_sizes,
        epsilon=result.epsilon,
        bound=result.bound,
        fitted_indices=result.fitted_indices,
        empirical_loss=result.empirical_loss,
        population_loss=result.population_loss,
        population_risk=result.population_risk,
        grid_gap=result.grid_gap,
    )
    metadata = {
        "schema_version": 1,
        "environment_id": "finite-linear-certificate-fixture-v1",
        "environment_package": "active-eval-gym",
        "environment_package_version": version("active-eval-gym"),
        "spec": asdict(population.spec),
        "policy_id": "frozen-identity-gain-v1",
        "policy_checkpoint": "analytic inline gain; no trained checkpoint",
        "gain_K": np.eye(2).tolist(),
        "plant_A": (population.matrices + np.eye(2)).tolist(),
        "plant_B": np.eye(2).tolist(),
        "training_seed": None,
        "action_seed": None,
        "perturbation_spec": {
            "kind": "finite-plant-and-unit-circle-reset-v1",
            "fixed_within_trajectory": True,
        },
        "sampling": "iid with replacement; sample sizes use nested prefixes",
        "master_seed": result.master_seed,
        "replicate_ids": list(range(len(result.draws))),
        "episode_seeds": "SeedSequence([master_seed, replicate_id]); indices retained",
        "delta_per_sample_size": result.delta_per_size,
        "metric_version": METRIC_VERSION,
        "numerics": {
            "arithmetic": "float64",
            "eigenvalue_verification_tolerance": VERIFICATION_TOLERANCE,
            "trajectory_violation_test": "strict worst residual > 0",
            "risk_bound_exceedance_tolerance": 1e-12,
        },
        "tie_break": "first grid index among empirical minimizers",
        "grid": {
            "eigenvalue_count": grid.eigenvalue_count,
            "angle_count": grid.angle_count,
            "frobenius_covering_radius_upper": grid.covering_radius,
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "raw_sha256": sha256(raw_path.read_bytes()).hexdigest(),
        "source_sha256": {
            str(p): sha256(p.read_bytes()).hexdigest() for p in source_paths
        },
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return destination
