"""Deterministic adaptive condition selection for CartPole boundary studies."""

import hashlib
import json
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

from active_eval_gym.serialization import write_json_new

SELECTOR_VERSION = "cartpole-boundary-selector-v1"
OUTCOMES = ("survival", "recovery")


@dataclass(frozen=True, order=True)
class Point:
    angle: float
    length: float


@dataclass(frozen=True, order=True)
class Cell:
    angle_low: float
    angle_high: float
    length_low: float
    length_high: float
    depth: int


@dataclass(frozen=True)
class MarkedCell:
    cell: Cell
    reasons: tuple[tuple[str, str], ...]
    mixed: bool
    closeness: float


def plan_boundary_stage(
    pilot_evaluation: Path,
    evaluations: list[Path],
    *,
    stage: str,
    output: Path,
) -> dict[str, Any]:
    """Write the next immutable refinement or final sweep suite."""

    expected_counts = {"refinement-1": 1, "refinement-2": 2, "final": 3}
    if stage not in expected_counts:
        raise ValueError(f"Unsupported boundary stage {stage!r}.")
    if len(evaluations) != expected_counts[stage]:
        raise ValueError(
            f"Stage {stage!r} requires {expected_counts[stage]} evaluations."
        )
    if evaluations[0].resolve() != pilot_evaluation.resolve():
        raise ValueError("The first evaluation must be the pilot evaluation.")

    pilot_record = _read_json(pilot_evaluation / "suite.json")
    pilot_suite = pilot_record["suite"]
    study = pilot_suite.get("boundary_study")
    if not isinstance(study, dict) or study.get("kind") != "adaptive-boundary-v1":
        raise ValueError("Pilot suite is not an adaptive boundary study.")
    expected_study_values = {
        "refinement_rounds": 2,
        "classification_threshold": 0.5,
        "recovery_tail_steps": 100,
        "recovery_rms_angle_deg": 5.0,
    }
    for name, expected in expected_study_values.items():
        if study.get(name) != expected:
            raise ValueError(
                f"Boundary study {name} must be {expected!r}, "
                f"received {study.get(name)!r}."
            )
    if pilot_suite.get("conditions"):
        raise ValueError("Pilot boundary suite must use a Cartesian grid.")
    grid = pilot_suite["grid"]
    angles = tuple(sorted(float(value) for value in grid["initial_theta_deg"]))
    lengths = tuple(sorted(float(value) for value in grid["length"]))
    if len(angles) < 2 or len(lengths) < 2:
        raise ValueError("Boundary pilot grid requires at least two values per axis.")

    summaries, source_hashes = _load_summaries(evaluations, pilot_suite)
    lookup = _merge_lookup(summaries)
    policies = tuple(pilot_suite["policy_ids"])
    threshold = float(grid["theta_threshold_deg"][0])
    if threshold != 90.0:
        raise ValueError("Recovery boundary study requires a 90-degree cutoff.")
    coarse = _coarse_cells(angles, lengths)
    marked_coarse = _marked_cells(coarse, lookup, policies)
    round_one_children = [
        child for marked in marked_coarse for child in _subdivide(marked.cell)
    ]

    if stage == "refinement-1":
        points = _new_subdivision_points(marked_coarse, set(lookup))
        seeds = list(pilot_suite["seeds"])
    else:
        marked_children = _marked_cells(round_one_children, lookup, policies)
        if stage == "refinement-2":
            points = _new_subdivision_points(marked_children, set(lookup))
            seeds = list(pilot_suite["seeds"])
        else:
            cap = _positive_int(
                study.get("max_final_conditions"), "max_final_conditions"
            )
            points = _final_points(
                angles,
                lengths,
                marked_coarse,
                marked_children,
                policies,
                cap,
            )
            seeds = _integer_list(study.get("final_seeds"), "final_seeds")

    if not points:
        raise ValueError(f"Boundary stage {stage!r} selected no conditions.")
    suite_id = f"{study['study_id']}-{stage}"
    result = {
        "schema_version": 1,
        "suite_id": suite_id,
        "environment_id": pilot_suite["environment_id"],
        "perturbation_name": pilot_suite["perturbation_name"],
        "metric_version": pilot_suite["metric_version"],
        "seeds": seeds,
        "policy_ids": list(policies),
        "derived_policies": pilot_suite.get("derived_policies", {}),
        "conditions": [
            {
                "condition_id": _condition_id(point),
                "parameters": {
                    "initial_theta_deg": point.angle,
                    "length": point.length,
                    "theta_threshold_deg": threshold,
                },
            }
            for point in sorted(points)
        ],
        "boundary_study": {
            **study,
            "selector_version": SELECTOR_VERSION,
            "stage": stage,
            "source_summary_sha256": source_hashes,
            "selected_condition_count": len(points),
        },
    }
    write_json_new(output, result)
    return {
        "suite_id": suite_id,
        "stage": stage,
        "condition_count": len(points),
        "output": str(output),
        "source_summary_sha256": source_hashes,
    }


def _load_summaries(
    evaluations: list[Path], pilot_suite: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    summaries = []
    hashes = []
    for evaluation in evaluations:
        suite_record = _read_json(evaluation / "suite.json")
        suite = suite_record["suite"]
        for name in (
            "environment_id",
            "perturbation_name",
            "metric_version",
            "policy_ids",
        ):
            if suite.get(name) != pilot_suite.get(name):
                raise ValueError(f"Boundary evaluation disagrees on {name}.")
        summary_path = (
            evaluation / "analysis" / suite["metric_version"] / "summary.json"
        )
        content = summary_path.read_bytes()
        summary = json.loads(content)
        if summary.get("seeds") != list(pilot_suite["seeds"]):
            raise ValueError("Refinement source must use the pilot seeds.")
        summaries.append(summary)
        hashes.append(hashlib.sha256(content).hexdigest())
    return summaries, hashes


def _merge_lookup(
    summaries: list[dict[str, Any]],
) -> dict[Point, dict[str, dict[str, float]]]:
    lookup: dict[Point, dict[str, dict[str, float]]] = {}
    for summary in summaries:
        for item in summary["conditions"]:
            parameters = item["parameters"]
            point = Point(
                float(parameters["initial_theta_deg"]),
                float(parameters["length"]),
            )
            values = {
                policy: {
                    "survival": float(aggregate["success_rate"]),
                    "recovery": float(aggregate["recovery_rate"]),
                }
                for policy, aggregate in item["policies"].items()
            }
            if point in lookup and lookup[point] != values:
                raise ValueError(f"Conflicting summaries for condition {point}.")
            lookup[point] = values
    return lookup


def _coarse_cells(angles: tuple[float, ...], lengths: tuple[float, ...]) -> list[Cell]:
    return [
        Cell(a0, a1, l0, l1, 0)
        for a0, a1 in zip(angles, angles[1:], strict=False)
        for l0, l1 in zip(lengths, lengths[1:], strict=False)
    ]


def _marked_cells(
    cells: list[Cell],
    lookup: dict[Point, dict[str, dict[str, float]]],
    policies: tuple[str, ...],
) -> list[MarkedCell]:
    result = []
    for cell in cells:
        corners = _corners(cell)
        missing = [point for point in corners if point not in lookup]
        if missing:
            raise ValueError(f"Boundary mesh is missing cell corners: {missing!r}.")
        reasons = []
        mixed = False
        closeness = 0.5
        for policy in policies:
            for outcome in OUTCOMES:
                rates = [lookup[point][policy][outcome] for point in corners]
                classifications = {rate >= 0.5 for rate in rates}
                has_mixed = any(0.0 < rate < 1.0 for rate in rates)
                if len(classifications) > 1 or has_mixed:
                    reasons.append((policy, outcome))
                    mixed = mixed or has_mixed
                    closeness = min(closeness, *(abs(rate - 0.5) for rate in rates))
        if reasons:
            result.append(MarkedCell(cell, tuple(reasons), mixed, closeness))
    return result


def _subdivide(cell: Cell) -> tuple[Cell, Cell, Cell, Cell]:
    angle_mid = (cell.angle_low + cell.angle_high) / 2.0
    length_mid = sqrt(cell.length_low * cell.length_high)
    depth = cell.depth + 1
    return (
        Cell(cell.angle_low, angle_mid, cell.length_low, length_mid, depth),
        Cell(cell.angle_low, angle_mid, length_mid, cell.length_high, depth),
        Cell(angle_mid, cell.angle_high, cell.length_low, length_mid, depth),
        Cell(angle_mid, cell.angle_high, length_mid, cell.length_high, depth),
    )


def _corners(cell: Cell) -> tuple[Point, Point, Point, Point]:
    return (
        Point(cell.angle_low, cell.length_low),
        Point(cell.angle_low, cell.length_high),
        Point(cell.angle_high, cell.length_low),
        Point(cell.angle_high, cell.length_high),
    )


def _subdivision_points(cell: Cell) -> set[Point]:
    angles = (
        cell.angle_low,
        (cell.angle_low + cell.angle_high) / 2.0,
        cell.angle_high,
    )
    lengths = (
        cell.length_low,
        sqrt(cell.length_low * cell.length_high),
        cell.length_high,
    )
    return {Point(angle, length) for angle in angles for length in lengths}


def _new_subdivision_points(
    marked: list[MarkedCell], known: set[Point]
) -> set[Point]:
    result: set[Point] = set()
    for item in marked:
        result.update(_subdivision_points(item.cell))
    return result - known


def _final_points(
    angles: tuple[float, ...],
    lengths: tuple[float, ...],
    first: list[MarkedCell],
    second: list[MarkedCell],
    policies: tuple[str, ...],
    cap: int,
) -> set[Point]:
    selected = {
        Point(angle, length)
        for angle in angles
        for length in lengths
        if angle in (angles[0], angles[-1]) or length in (lengths[0], lengths[-1])
    }
    selected.add(Point(0.0, 0.5))
    if len(selected) > cap:
        raise ValueError("Final condition cap is smaller than mandatory anchors.")
    reasons = tuple((policy, outcome) for policy in policies for outcome in OUTCOMES)
    for candidates in (first, second):
        ordered = _round_robin_cells(candidates, reasons)
        for marked in ordered:
            bundle = _subdivision_points(marked.cell)
            if len(selected | bundle) <= cap:
                selected.update(bundle)
    return selected


def _round_robin_cells(
    cells: list[MarkedCell], reasons: tuple[tuple[str, str], ...]
) -> list[MarkedCell]:
    def key(item: MarkedCell) -> tuple[Any, ...]:
        return (
            not item.mixed,
            item.closeness,
            item.cell.angle_low,
            item.cell.length_low,
            item.cell.angle_high,
            item.cell.length_high,
        )
    queues = {
        reason: sorted((cell for cell in cells if reason in cell.reasons), key=key)
        for reason in reasons
    }
    ordered: list[MarkedCell] = []
    seen: set[Cell] = set()
    while any(queues.values()):
        for reason in reasons:
            queue = queues[reason]
            while queue and queue[0].cell in seen:
                queue.pop(0)
            if queue:
                item = queue.pop(0)
                seen.add(item.cell)
                ordered.append(item)
    return ordered


def _condition_id(point: Point) -> str:
    return f"theta-{_slug(point.angle)}_length-{_slug(point.length)}"


def _slug(value: float) -> str:
    sign = "m" if value < 0 else "p"
    return sign + format(abs(value), ".12g").replace(".", "p")


def _integer_list(value: Any, name: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{name} must be a non-empty list of unique integers.")
    return list(value)


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Boundary input does not exist: {path}.") from error
    if not isinstance(value, dict):
        raise ValueError(f"Boundary input must be a JSON object: {path}.")
    return value
