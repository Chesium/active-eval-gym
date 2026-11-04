"""Fixed training distributions built from a frozen parent's discovery results.

Selection reads discovery counts only. The result is frozen before training, and
this module never reads pilot or reference evaluations.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from active_eval_gym.config import (
    PPO_BOUNDARY_FRONTIER_RULE,
    PPOBoundaryStudySpec,
)
from active_eval_gym.ppo_boundary import (
    load_protocol,
    source_state,
    support_conditions,
)
from active_eval_gym.ppo_boundary_analysis import METRICS_DIRECTORY, SUMMARY_FILENAME
from active_eval_gym.ppo_boundary_evaluation import CAMPAIGN_DIRECTORIES, DISCOVERY
from active_eval_gym.serialization import write_json_new

BACKGROUND = "background"
ACCEPTANCE_BOUNDARY = "acceptance_boundary"
INTERMEDIATE = "intermediate"
EMPTY_BAND_FALLBACK = "empty-band"
ZERO_SCORE_FALLBACK = "zero-score-normalizer"
DISTRIBUTIONS_DIRECTORY = "distributions"
DISTRIBUTIONS_FILENAME = "distributions.json"
CONTRAST_FILENAME = "contrast-report.json"
LATER_EVALUATION_CAMPAIGNS = ("pilot", "pilot-snapshots", "confirmatory")


@dataclass(frozen=True)
class ArmDistribution:
    """One arm's target weights and the mixture it will train on."""

    arm: str
    target_weights: np.ndarray
    probabilities: np.ndarray
    fallback: str | None


def smoothed_scores(counts: np.ndarray, episodes: int, curriculum: Any) -> np.ndarray:
    """Return the selection score, which is not a confidence bound."""

    return (counts + curriculum.pseudocount_success) / (
        episodes + curriculum.pseudocount_total
    )


def frontier_mask(accepted: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    """Mark unaccepted conditions beside an accepted one on the angle-length grid.

    Adjacency is by grid index, so a step is not a fixed physical distance.
    """

    grid = accepted.reshape(grid_shape)
    neighbour = np.zeros_like(grid)
    neighbour[1:, :] |= grid[:-1, :]
    neighbour[:-1, :] |= grid[1:, :]
    neighbour[:, 1:] |= grid[:, :-1]
    neighbour[:, :-1] |= grid[:, 1:]
    return (neighbour & ~grid).ravel()


def construct_arms(
    counts: np.ndarray,
    episodes: int,
    tau: float,
    curriculum: Any,
    *,
    grid_shape: tuple[int, int] | None = None,
) -> dict[str, ArmDistribution]:
    """Apply the declared target rules to recovery counts in condition order."""

    size = counts.size
    base = np.full(size, 1.0 / size)
    score = smoothed_scores(counts, episodes, curriculum)
    frontier_rule = curriculum.target_rule == PPO_BOUNDARY_FRONTIER_RULE

    if frontier_rule:
        if grid_shape is None:
            raise ValueError("The frontier rule requires grid_shape.")
        band = frontier_mask(counts / episodes >= tau, grid_shape)
    else:
        band = np.abs(score - tau) <= curriculum.boundary_half_width
    boundary_fallback = None if band.any() else EMPTY_BAND_FALLBACK
    boundary = base * band / (base * band).sum() if band.any() else base

    difficulty = base * score * (1.0 - score)
    if frontier_rule:
        difficulty = difficulty * ((counts > 0) & (counts < episodes))
    normalizer = difficulty.sum()
    usable = normalizer > np.finfo(float).tiny
    intermediate_fallback = None if usable else ZERO_SCORE_FALLBACK
    intermediate = difficulty / normalizer if usable else base

    targets = (
        (BACKGROUND, base, None),
        (ACCEPTANCE_BOUNDARY, boundary, boundary_fallback),
        (INTERMEDIATE, intermediate, intermediate_fallback),
    )
    mixture = curriculum.mixture_lambda
    return {
        arm: ArmDistribution(
            arm=arm,
            target_weights=target,
            probabilities=(1.0 - mixture) * base + mixture * target,
            fallback=fallback,
        )
        for arm, target, fallback in targets
    }


def contrast_report(
    arms: dict[str, ArmDistribution],
    *,
    angles: np.ndarray,
    counts: np.ndarray,
    episodes: int,
    tau: float,
    thin_band_min_conditions: int,
    condition_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Measure how far apart the arms really are, before any training budget."""

    boundary = arms[ACCEPTANCE_BOUNDARY]
    selected = boundary.fallback is None
    band = boundary.target_weights > 0.0 if selected else np.zeros(counts.size, bool)
    mixed = (counts > 0) & (counts < episodes)
    accepted = counts / episodes >= tau
    names = tuple(arms)
    return {
        "schema_version": 1,
        "condition_count": counts.size,
        "recovery_count_extremes": {
            "all_episodes_failed": int((counts == 0).sum()),
            "mixed": int(mixed.sum()),
            "all_episodes_recovered": int((counts == episodes).sum()),
            "accepted_at_primary_tau": int(accepted.sum()),
        },
        "boundary_band": {
            "size": int(band.sum()),
            "thin_band": int(band.sum()) < thin_band_min_conditions,
            "no_boundary_targeting_contrast": not selected,
            "conditions": [
                {
                    "condition_id": condition_ids[index],
                    "recovery_count": int(counts[index]),
                }
                for index in np.flatnonzero(band)
            ],
        },
        "arms": {
            name: {
                "fallback": arm.fallback,
                "effective_support": 1.0 / float((arm.probabilities**2).sum()),
                "largest_condition_probability": float(arm.probabilities.max()),
                "probability_on_mixed_conditions": float(
                    arm.probabilities[mixed].sum()
                ),
                "target_mass_on_mixed_conditions": float(
                    arm.target_weights[mixed].sum()
                ),
                "target_mass_on_accepted_conditions": float(
                    arm.target_weights[accepted].sum()
                ),
                "probability_by_angle_sign": _sign_balance(arm.probabilities, angles),
                "target_mass_by_angle_sign": _sign_balance(arm.target_weights, angles),
            }
            for name, arm in arms.items()
        },
        "pairwise_total_variation": {
            f"{first}__{second}": _total_variation(
                arms[first].probabilities, arms[second].probabilities
            )
            for index, first in enumerate(names)
            for second in names[index + 1 :]
        },
    }


def build_distributions(
    config_path: Path,
    *,
    study_dir: Path,
    output_dir: Path | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Freeze every arm's distribution from completed discovery metrics."""

    spec, provenance = load_protocol(config_path, study_dir)
    source = source_state()
    if not allow_dirty and (provenance["development_record"] or source["source_dirty"]):
        raise ValueError(
            "Distributions require a committed source and non-development "
            "protocol. Pass allow_dirty only for temporary development runs."
        )
    _require_no_later_evaluations(study_dir)
    output = output_dir or study_dir / DISTRIBUTIONS_DIRECTORY
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite distributions: {output}.")

    summary_path = study_dir / METRICS_DIRECTORY / DISCOVERY / SUMMARY_FILENAME
    if not summary_path.is_file():
        raise FileNotFoundError(f"Discovery has not been analyzed: {summary_path}.")
    summary = json.loads(summary_path.read_text())
    condition_ids, counts = _discovery_counts(spec, summary)
    episodes = spec.curriculum.discovery_episodes_per_condition
    tau = spec.acceptance.primary_tau
    support = spec.supports[DISCOVERY]
    arms = construct_arms(
        counts,
        episodes,
        tau,
        spec.curriculum,
        grid_shape=(len(support.initial_theta_deg), len(support.length)),
    )
    for arm in arms.values():
        if not np.isclose(arm.probabilities.sum(), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError(f"Arm {arm.arm!r} is not a probability distribution.")

    conditions = support_conditions(spec, "discovery")
    angles = np.array(
        [c.perturbation.parameters["initial_theta_deg"] for c in conditions]
    )
    distributions = {
        "schema_version": 1,
        "study_id": spec.study_id,
        "target_rule_version": spec.curriculum.target_rule,
        "outer_round": 0,
        "development_record": source["source_dirty"],
        "source": source,
        "selection": {
            "campaign": DISCOVERY,
            "discovery_summary_sha256": hashlib.sha256(
                summary_path.read_bytes()
            ).hexdigest(),
            "discovery_source": summary["evaluation_source"],
            "protocol_config_sha256": provenance["config_sha256"],
            "parent_model_sha256": spec.parent_model_sha256,
            "episodes_per_condition": episodes,
            "primary_tau": tau,
            "curriculum": spec.curriculum,
        },
        "condition_ids": condition_ids,
        "recovery_counts": counts,
        "smoothed_scores": smoothed_scores(counts, episodes, spec.curriculum),
        "arms": {
            name: {
                "fallback": arm.fallback,
                "target_weights": arm.target_weights,
                "probabilities": arm.probabilities,
            }
            for name, arm in arms.items()
        },
    }
    report = contrast_report(
        arms,
        angles=angles,
        counts=counts,
        episodes=episodes,
        tau=tau,
        thin_band_min_conditions=spec.curriculum.thin_band_min_conditions,
        condition_ids=condition_ids,
    )
    output.mkdir(parents=True)
    write_json_new(output / DISTRIBUTIONS_FILENAME, distributions)
    write_json_new(output / CONTRAST_FILENAME, report)
    return report


def _discovery_counts(
    spec: PPOBoundaryStudySpec, summary: dict[str, Any]
) -> tuple[tuple[str, ...], np.ndarray]:
    if summary["campaign"] != DISCOVERY or summary["support"] != DISCOVERY:
        raise ValueError("Targets must be selected from discovery results.")
    if summary["metric_version"] != spec.metric_version:
        raise ValueError("Discovery metrics use an undeclared metric version.")
    expected = tuple(
        condition.condition_id for condition in support_conditions(spec, DISCOVERY)
    )
    cells = summary["cells"]
    if tuple(cell["condition_id"] for cell in cells) != expected:
        raise ValueError("Discovery cells do not match the declared conditions.")
    episodes = spec.curriculum.discovery_episodes_per_condition
    for cell in cells:
        if cell["checkpoint_id"] != spec.parent_policy_id:
            raise ValueError("Discovery must evaluate only the declared parent.")
        if cell["episode_count"] != episodes:
            raise ValueError(f"Discovery cell {cell['condition_id']} is incomplete.")
    return expected, np.array([cell["recovery_count"] for cell in cells])


def _require_no_later_evaluations(study_dir: Path) -> None:
    for campaign in LATER_EVALUATION_CAMPAIGNS:
        paths = (
            study_dir / CAMPAIGN_DIRECTORIES[campaign],
            study_dir / METRICS_DIRECTORY / campaign,
        )
        if any(path.exists() for path in paths):
            raise ValueError(
                f"Target selection must precede {campaign} evaluation results."
            )


def _sign_balance(weights: np.ndarray, angles: np.ndarray) -> dict[str, float]:
    return {
        "negative": float(weights[angles < 0].sum()),
        "zero": float(weights[angles == 0].sum()),
        "positive": float(weights[angles > 0].sum()),
    }


def _total_variation(first: np.ndarray, second: np.ndarray) -> float:
    return 0.5 * float(np.abs(first - second).sum())
