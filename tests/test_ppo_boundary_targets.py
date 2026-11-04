import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from active_eval_gym import ppo_boundary_targets
from active_eval_gym.config import load_ppo_boundary_study
from active_eval_gym.ppo_boundary import prepare_study
from active_eval_gym.ppo_boundary_analysis import analyze_campaign
from active_eval_gym.ppo_boundary_evaluation import collect_campaign
from active_eval_gym.ppo_boundary_targets import (
    ACCEPTANCE_BOUNDARY,
    BACKGROUND,
    EMPTY_BAND_FALLBACK,
    INTERMEDIATE,
    ZERO_SCORE_FALLBACK,
    build_distributions,
    construct_arms,
    contrast_report,
    frontier_mask,
    smoothed_scores,
)
from tests.helpers import (
    FRONTIER_RULE_EDIT,
    LENGTHS,
    SMALL_STUDY_GRID,
    STUDY_CONFIG,
    STUDY_CONFIG_V2,
    tiny_parent_config,
)

CURRICULUM = load_ppo_boundary_study(STUDY_CONFIG).curriculum
CURRICULUM_V2 = load_ppo_boundary_study(STUDY_CONFIG_V2).curriculum
# Recovery counts from the 4 November 2026 discovery; rows are half-lengths and
# columns are angles from -50 to +50 degrees.
DISCOVERY_MAP = np.array(
    [[0] * 11] * 5
    + [
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 17, 3, 0, 0, 0, 0],
        [0, 0, 5, 32, 32, 32, 32, 32, 26, 0, 0],
        [0, 0, 0, 32, 32, 32, 32, 32, 0, 0, 0],
        [0, 0, 0, 0, 32, 32, 32, 32, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0],
    ]
    + [[0] * 11] * 2
).T.ravel()


def _arms(counts: list[int], episodes: int = 32, curriculum=CURRICULUM):
    return construct_arms(np.array(counts), episodes, 0.90, curriculum)


def _report(counts: list[int], angles: list[float], *, minimum: int = 5) -> dict:
    return contrast_report(
        _arms(counts),
        angles=np.array(angles),
        counts=np.array(counts),
        episodes=32,
        tau=0.90,
        thin_band_min_conditions=minimum,
        condition_ids=tuple(f"c{index}" for index in range(len(counts))),
    )


def test_smoothed_score_matches_the_declared_formula() -> None:
    scores = smoothed_scores(np.array([0, 16, 32]), 32, CURRICULUM)

    assert scores == pytest.approx([0.5 / 33, 16.5 / 33, 32.5 / 33])


def test_boundary_band_is_exactly_28_to_30_of_32() -> None:
    counts = list(range(33))

    weights = _arms(counts)[ACCEPTANCE_BOUNDARY].target_weights

    assert np.flatnonzero(weights).tolist() == [28, 29, 30]
    assert weights[[28, 29, 30]] == pytest.approx([1 / 3] * 3)


def test_background_is_the_uniform_base() -> None:
    arm = _arms([0, 16, 29, 32])[BACKGROUND]

    assert arm.target_weights == pytest.approx([0.25] * 4)
    assert arm.probabilities == pytest.approx([0.25] * 4)
    assert arm.fallback is None


def test_boundary_mixture_matches_hand_computation() -> None:
    arm = _arms([0, 16, 29, 32])[ACCEPTANCE_BOUNDARY]

    assert arm.target_weights == pytest.approx([0, 0, 1, 0])
    assert arm.probabilities == pytest.approx([0.125, 0.125, 0.625, 0.125])


def test_intermediate_mixture_matches_hand_computation() -> None:
    counts = [0, 16, 29, 32]
    p = np.array([0.5, 16.5, 29.5, 32.5]) / 33
    raw = p * (1 - p)

    arm = _arms(counts)[INTERMEDIATE]

    assert arm.target_weights == pytest.approx(raw / raw.sum())
    assert arm.probabilities == pytest.approx(0.125 + 0.5 * raw / raw.sum())
    assert arm.target_weights.argmax() == 1


def test_every_arm_is_normalized_with_full_support() -> None:
    for arm in _arms(list(range(33))).values():
        assert arm.probabilities.sum() == pytest.approx(1.0)
        assert np.all(arm.probabilities >= 0.5 / 33)


def test_empty_band_falls_back_to_background_and_says_so() -> None:
    arms = _arms([0, 16, 32, 32])

    boundary = arms[ACCEPTANCE_BOUNDARY]
    assert boundary.fallback == EMPTY_BAND_FALLBACK
    assert np.array_equal(boundary.probabilities, arms[BACKGROUND].probabilities)


def test_zero_intermediate_normalizer_falls_back() -> None:
    curriculum = replace(CURRICULUM, pseudocount_success=0.0, pseudocount_total=0.0)

    arm = _arms([0, 0], curriculum=curriculum)[INTERMEDIATE]

    assert arm.fallback == ZERO_SCORE_FALLBACK
    assert arm.target_weights == pytest.approx([0.5, 0.5])


def test_saturated_map_matches_the_plans_intermediate_warning() -> None:
    counts = np.zeros(143, dtype=int)
    counts[:5] = 16

    arm = construct_arms(counts, 32, 0.90, CURRICULUM)[INTERMEDIATE]

    assert arm.target_weights[:5].sum() == pytest.approx(0.378, abs=0.001)


def test_single_band_condition_receives_over_half_the_episodes() -> None:
    counts = np.zeros(143, dtype=int)
    counts[7] = 29

    arm = construct_arms(counts, 32, 0.90, CURRICULUM)[ACCEPTANCE_BOUNDARY]

    assert arm.probabilities.max() == pytest.approx(0.5 + 0.5 / 143)


def test_report_measures_contrast_by_hand() -> None:
    report = _report([0, 16, 29, 32], [-10, 0, 10, 10])

    assert report["recovery_count_extremes"] == {
        "all_episodes_failed": 1,
        "mixed": 2,
        "all_episodes_recovered": 1,
        "accepted_at_primary_tau": 2,
    }
    band = report["boundary_band"]
    assert band["size"] == 1
    assert band["thin_band"] is True
    assert band["conditions"] == [{"condition_id": "c2", "recovery_count": 29}]
    boundary = report["arms"][ACCEPTANCE_BOUNDARY]
    assert boundary["effective_support"] == pytest.approx(1 / (3 * 0.125**2 + 0.625**2))
    assert boundary["largest_condition_probability"] == pytest.approx(0.625)
    assert boundary["probability_on_mixed_conditions"] == pytest.approx(0.75)
    assert boundary["probability_by_angle_sign"] == pytest.approx(
        {"negative": 0.125, "zero": 0.125, "positive": 0.75}
    )
    assert boundary["target_mass_by_angle_sign"] == pytest.approx(
        {"negative": 0.0, "zero": 0.0, "positive": 1.0}
    )
    assert report["pairwise_total_variation"][
        "background__acceptance_boundary"
    ] == pytest.approx(0.375)


def test_thin_band_label_applies_below_the_declared_minimum() -> None:
    angles = [0.0] * 5
    assert _report([29] * 4 + [0], angles)["boundary_band"]["thin_band"] is True
    assert _report([29] * 5, angles)["boundary_band"]["thin_band"] is False


def test_empty_band_report_declares_no_contrast() -> None:
    band = _report([0, 16, 32], [-10, 0, 10])["boundary_band"]

    assert band["size"] == 0
    assert band["no_boundary_targeting_contrast"] is True
    assert band["conditions"] == []


def _total_variation(first: np.ndarray, second: np.ndarray) -> float:
    return 0.5 * float(np.abs(first - second).sum())


def test_frontier_marks_only_unaccepted_edge_neighbours() -> None:
    accepted = np.array(
        [
            [False, False, False],
            [False, True, False],
            [False, False, False],
        ]
    ).ravel()

    frontier = frontier_mask(accepted, (3, 3)).reshape(3, 3)

    assert frontier.tolist() == [
        [False, True, False],
        [True, False, True],
        [False, True, False],
    ]


def test_frontier_does_not_wrap_around_the_grid() -> None:
    accepted = np.array([[True, False, False], [False, False, False]]).ravel()

    frontier = frontier_mask(accepted, (2, 3)).reshape(2, 3)

    assert frontier.tolist() == [[False, True, False], [True, False, False]]


def test_frontier_rule_accepts_at_the_primary_threshold() -> None:
    counts = np.array([29, 28, 0])

    arm = construct_arms(counts, 32, 0.90, CURRICULUM_V2, grid_shape=(1, 3))[
        ACCEPTANCE_BOUNDARY
    ]

    assert arm.target_weights.tolist() == [0.0, 1.0, 0.0]


def test_frontier_rule_requires_a_grid() -> None:
    with pytest.raises(ValueError, match="grid_shape"):
        construct_arms(np.array([32, 0]), 32, 0.90, CURRICULUM_V2)


def test_v2_intermediate_uses_only_mixed_conditions() -> None:
    counts = np.array([0, 16, 29, 32])
    p = np.array([16.5, 29.5]) / 33
    raw = p * (1 - p)

    arm = construct_arms(counts, 32, 0.90, CURRICULUM_V2, grid_shape=(1, 4))[
        INTERMEDIATE
    ]

    assert arm.target_weights[[0, 3]].tolist() == [0.0, 0.0]
    assert arm.target_weights[[1, 2]] == pytest.approx(raw / raw.sum())
    assert arm.probabilities[[0, 3]] == pytest.approx([0.125, 0.125])


def test_v2_falls_back_when_no_condition_qualifies() -> None:
    arms = construct_arms(
        np.array([0, 0, 0, 0]), 32, 0.90, CURRICULUM_V2, grid_shape=(2, 2)
    )

    assert arms[ACCEPTANCE_BOUNDARY].fallback == EMPTY_BAND_FALLBACK
    assert arms[INTERMEDIATE].fallback == ZERO_SCORE_FALLBACK
    for name in (ACCEPTANCE_BOUNDARY, INTERMEDIATE):
        assert np.array_equal(arms[name].probabilities, arms[BACKGROUND].probabilities)


def test_v2_every_arm_keeps_full_support() -> None:
    arms = construct_arms(DISCOVERY_MAP, 32, 0.90, CURRICULUM_V2, grid_shape=(11, 13))

    for arm in arms.values():
        assert arm.probabilities.sum() == pytest.approx(1.0)
        assert np.all(arm.probabilities >= 0.5 / 143)


def test_v2_reproduces_the_contrast_costed_on_discovery() -> None:
    arms = construct_arms(DISCOVERY_MAP, 32, 0.90, CURRICULUM_V2, grid_shape=(11, 13))
    background = arms[BACKGROUND].probabilities
    boundary = arms[ACCEPTANCE_BOUNDARY]
    intermediate = arms[INTERMEDIATE]

    assert np.count_nonzero(boundary.target_weights) == 15
    assert np.count_nonzero(intermediate.target_weights) == 5
    assert boundary.target_weights[DISCOVERY_MAP >= 29].sum() == 0.0
    assert _total_variation(boundary.probabilities, background) == pytest.approx(
        0.448, abs=0.001
    )
    assert _total_variation(intermediate.probabilities, background) == pytest.approx(
        0.483, abs=0.001
    )
    assert _total_variation(
        boundary.probabilities, intermediate.probabilities
    ) == pytest.approx(0.367, abs=0.001)
    assert intermediate.probabilities.max() == pytest.approx(0.185, abs=0.001)


def test_v1_rule_is_unchanged_on_the_discovery_map() -> None:
    arms = construct_arms(DISCOVERY_MAP, 32, 0.90, CURRICULUM)

    assert arms[ACCEPTANCE_BOUNDARY].fallback == EMPTY_BAND_FALLBACK
    assert np.array_equal(
        arms[ACCEPTANCE_BOUNDARY].probabilities, arms[BACKGROUND].probabilities
    )
    assert _total_variation(
        arms[INTERMEDIATE].probabilities, arms[BACKGROUND].probabilities
    ) == pytest.approx(0.107, abs=0.001)


@pytest.fixture
def study(tmp_path: Path) -> dict[str, Path]:
    config, root = tiny_parent_config(tmp_path, *SMALL_STUDY_GRID)
    study_dir = tmp_path / "study"
    prepare_study(config, artifact_root=root, output_dir=study_dir, allow_dirty=True)
    collect_campaign(
        config,
        study_dir=study_dir,
        campaign="discovery",
        artifact_root=root,
        allow_dirty=True,
    )
    return {"config": config, "study": study_dir}


def _analyze(study: dict[str, Path]) -> None:
    analyze_campaign(study["config"], study_dir=study["study"], campaign="discovery")


def _build(study: dict[str, Path], **options) -> dict:
    return build_distributions(
        study["config"], study_dir=study["study"], allow_dirty=True, **options
    )


def test_distributions_serialize_complete_ordered_vectors(study) -> None:
    _analyze(study)

    report = _build(study)

    directory = study["study"] / "distributions"
    record = json.loads((directory / "distributions.json").read_text())
    assert record["condition_ids"] == [
        "theta-m10_length-p0p5",
        "theta-m10_length-p1",
        "theta-p10_length-p0p5",
        "theta-p10_length-p1",
    ]
    assert set(record["arms"]) == {BACKGROUND, ACCEPTANCE_BOUNDARY, INTERMEDIATE}
    for arm in record["arms"].values():
        assert len(arm["probabilities"]) == 4
        assert sum(arm["probabilities"]) == pytest.approx(1.0)
    selection = record["selection"]
    assert selection["campaign"] == "discovery"
    assert selection["episodes_per_condition"] == 3
    assert selection["curriculum"]["mixture_lambda"] == 0.5
    assert len(selection["discovery_summary_sha256"]) == 64
    assert json.loads((directory / "contrast-report.json").read_text()) == report


def test_rebuilding_from_the_same_discovery_is_byte_identical(
    study, tmp_path: Path
) -> None:
    _analyze(study)

    _build(study, output_dir=tmp_path / "a")
    _build(study, output_dir=tmp_path / "b")

    for name in ("distributions.json", "contrast-report.json"):
        assert (tmp_path / "a" / name).read_bytes() == (
            tmp_path / "b" / name
        ).read_bytes()


def test_distributions_refuse_to_overwrite(study) -> None:
    _analyze(study)
    _build(study)

    with pytest.raises(FileExistsError):
        _build(study)


def test_distributions_require_analyzed_discovery(study) -> None:
    with pytest.raises(FileNotFoundError, match="not been analyzed"):
        _build(study)


@pytest.mark.parametrize(
    "path",
    [
        "pilot",
        "reference_raw",
        "pilot_snapshots",
        "metrics_v1/pilot",
        "metrics_v1/pilot-snapshots",
        "metrics_v1/confirmatory",
    ],
)
def test_selection_refuses_to_run_after_later_evaluations(study, path: str) -> None:
    _analyze(study)
    (study["study"] / path).mkdir(parents=True)

    with pytest.raises(ValueError, match="must precede"):
        _build(study)

    assert not (study["study"] / "distributions").exists()


def test_distributions_refuse_uncommitted_source_by_default(
    study, monkeypatch: pytest.MonkeyPatch
) -> None:
    _analyze(study)
    monkeypatch.setattr(
        ppo_boundary_targets,
        "source_state",
        lambda: {"git_commit": "a", "source_dirty": True, "changed_paths": ["x"]},
    )

    with pytest.raises(ValueError, match="committed source"):
        build_distributions(study["config"], study_dir=study["study"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("campaign", "pilot", "discovery results"),
        ("support", "reference", "discovery results"),
        ("metric_version", "episode-summary-v3", "metric version"),
    ],
)
def test_distributions_reject_non_discovery_summaries(
    study, field: str, value: str, message: str
) -> None:
    _analyze(study)
    path = study["study"] / "metrics_v1/discovery/summary.json"
    summary = json.loads(path.read_text())
    summary[field] = value
    path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match=message):
        _build(study)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("checkpoint_id", "some_child", "only the declared parent"),
        ("episode_count", 2, "incomplete"),
        ("condition_id", "theta-p1_length-p1", "declared conditions"),
    ],
)
def test_distributions_reject_inconsistent_cells(
    study, field: str, value, message: str
) -> None:
    _analyze(study)
    path = study["study"] / "metrics_v1/discovery/summary.json"
    summary = json.loads(path.read_text())
    summary["cells"][1][field] = value
    path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match=message):
        _build(study)


def test_build_uses_the_v2_rule_and_records_it(tmp_path: Path) -> None:
    config, root = tiny_parent_config(tmp_path, FRONTIER_RULE_EDIT, *SMALL_STUDY_GRID)
    study_dir = tmp_path / "study"
    prepare_study(config, artifact_root=root, output_dir=study_dir, allow_dirty=True)
    collect_campaign(
        config,
        study_dir=study_dir,
        campaign="discovery",
        artifact_root=root,
        allow_dirty=True,
    )
    analyze_campaign(config, study_dir=study_dir, campaign="discovery")

    build_distributions(config, study_dir=study_dir, allow_dirty=True)

    record = json.loads((study_dir / "distributions/distributions.json").read_text())
    assert record["target_rule_version"] == "ppo-boundary-targets-v2"
    assert record["selection"]["curriculum"]["boundary_half_width"] is None


def test_build_uses_angle_by_length_axes_on_a_non_square_grid(tmp_path: Path) -> None:
    grid = [
        (old, "[0.5, 1.0, 1.5]" if old == LENGTHS else new)
        for old, new in SMALL_STUDY_GRID
    ]
    config, root = tiny_parent_config(tmp_path, FRONTIER_RULE_EDIT, *grid)
    study_dir = tmp_path / "study"
    prepare_study(config, artifact_root=root, output_dir=study_dir, allow_dirty=True)
    collect_campaign(
        config,
        study_dir=study_dir,
        campaign="discovery",
        artifact_root=root,
        allow_dirty=True,
    )
    analyze_campaign(config, study_dir=study_dir, campaign="discovery")
    path = study_dir / "metrics_v1/discovery/summary.json"
    summary = json.loads(path.read_text())
    for index, cell in enumerate(summary["cells"]):
        cell["recovery_count"] = 3 if index == 1 else 0
    path.write_text(json.dumps(summary))

    report = build_distributions(config, study_dir=study_dir, allow_dirty=True)

    assert [c["condition_id"] for c in report["boundary_band"]["conditions"]] == [
        "theta-m10_length-p0p5",
        "theta-m10_length-p1p5",
        "theta-p10_length-p1",
    ]
