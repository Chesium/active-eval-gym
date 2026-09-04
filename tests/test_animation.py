import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from active_eval_gym.animation import (
    _RECOVERY_BAND_FILL,
    COMPOSITE_PANEL_SIZE,
    MAX_COMPOSITE_CONDITIONS,
    POLICY_COLORS,
    EpisodeTrace,
    RecoveryBand,
    _composite_density_layers,
    _draw_recovery_wedge,
    _fonts,
    _policy_readout_label,
    _render_condition_frame,
    _scene_geometry,
    _trailing_rms_degrees,
    _visible_state,
    load_cartpole_sweep_animation,
)
from active_eval_gym.cli import main


def test_animate_sweep_writes_individual_composite_and_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    evaluation = _tiny_evaluation(tmp_path / "evaluation")
    output = tmp_path / "animations"

    result = main(
        [
            "animate-sweep",
            "--evaluation",
            str(evaluation),
            "--output",
            str(output),
            "--layout",
            "both",
            "--frame-stride",
            "2",
        ]
    )

    assert result == 0
    cli_result = json.loads(capsys.readouterr().out)
    manifest = json.loads((output / "animation-manifest.json").read_text())
    gifs = sorted(output.glob("*.gif"))
    assert len(gifs) == 3
    assert len(cli_result["animation_artifacts"]) == 4
    assert manifest["renderer_id"] == "cartpole-overlay-v1"
    assert manifest["frame_count"] == 3
    assert manifest["frame_duration_milliseconds"] == 40
    assert len(manifest["source_trajectories"]) == 8
    assert manifest["policy_colors"] == {
        "policy-lqr": "#0072B2",
        "policy-ppo": "#D62728",
    }
    for gif in gifs:
        with Image.open(gif) as image:
            assert image.format == "GIF"
            assert image.n_frames == 3
    for record in manifest["outputs"]:
        content = (output / record["filename"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == record["sha256"]

    with pytest.raises(SystemExit) as error:
        main(
            [
                "animate-sweep",
                "--evaluation",
                str(evaluation),
                "--output",
                str(output),
            ]
        )
    assert error.value.code == 2


def test_density_compositing_is_independent_of_policy_draw_order() -> None:
    base = Image.new("RGB", (3, 3), "white")
    first = Image.new("L", (3, 3), 0)
    second = Image.new("L", (3, 3), 0)
    first.putpixel((1, 1), 160)
    second.putpixel((1, 1), 80)
    blue = (0, 114, 178)
    red = (214, 39, 40)

    forward = _composite_density_layers(base, [(blue, first), (red, second)])
    reverse = _composite_density_layers(base, [(red, second), (blue, first)])

    np.testing.assert_array_equal(np.asarray(forward), np.asarray(reverse))
    center = np.asarray(forward)[1, 1]
    assert center[0] < 255
    assert center[2] < 255


def test_terminal_pose_is_shown_once_when_frame_stride_skips_it() -> None:
    trace = EpisodeTrace(
        policy_id="policy",
        seed=0,
        states=tuple(_state(float(step)) for step in range(4)),
        terminated=True,
        truncated=False,
        parameters={"length": 0.5, "tau": 0.02, "x_threshold": 2.4},
        max_episode_steps=3,
        trajectory_sha256="digest",
    )

    sampled = _visible_state(trace, step=4, previous_step=2)

    assert sampled is not None
    assert sampled.state == trace.states[-1]
    assert sampled.failed
    assert not sampled.active
    assert _visible_state(trace, step=6, previous_step=4) is None


def test_three_policy_boundary_sweep_loads_with_distinct_colors(
    tmp_path: Path,
) -> None:
    evaluation = _boundary_evaluation(tmp_path / "evaluation")

    data = load_cartpole_sweep_animation(evaluation)

    assert data.policy_ids == ("policy-lqr", "policy-ppo", "policy-ppo-anti")
    assert len(set(data.policy_colors.values())) == 3
    assert set(data.policy_colors.values()) <= set(POLICY_COLORS)
    assert data.recovery_band == RecoveryBand(rms_angle_degrees=5.0, tail_steps=4)


def test_three_density_layers_blend_independently_of_draw_order() -> None:
    base = Image.new("RGB", (3, 3), "white")
    masks = []
    for strength in (160, 80, 40):
        mask = Image.new("L", (3, 3), 0)
        mask.putpixel((1, 1), strength)
        masks.append(mask)
    layers = list(zip([_rgb(color) for color in POLICY_COLORS], masks, strict=True))

    forward = _composite_density_layers(base, layers)
    reverse = _composite_density_layers(base, list(reversed(layers)))

    np.testing.assert_array_equal(np.asarray(forward), np.asarray(reverse))
    assert tuple(np.asarray(forward)[1, 1]) != (255, 255, 255)


def test_condition_and_policy_filters_keep_suite_order(tmp_path: Path) -> None:
    evaluation = _boundary_evaluation(tmp_path / "evaluation")

    data = load_cartpole_sweep_animation(
        evaluation,
        condition_ids=["theta-p30_length-p0p2", "theta-m30_length-p0p5"],
        policy_ids=["policy-ppo-anti", "policy-ppo"],
    )

    assert [condition.condition_id for condition in data.conditions] == [
        "theta-m30_length-p0p5",
        "theta-p30_length-p0p2",
    ]
    assert data.policy_ids == ("policy-ppo", "policy-ppo-anti")
    assert all(
        trace.policy_id in data.policy_ids
        for condition in data.conditions
        for trace in condition.traces
    )


def test_unknown_condition_and_policy_identifiers_are_named(tmp_path: Path) -> None:
    evaluation = _boundary_evaluation(tmp_path / "evaluation")

    with pytest.raises(ValueError, match="theta-p99_length-p9"):
        load_cartpole_sweep_animation(
            evaluation, condition_ids=["theta-m30_length-p0p5", "theta-p99_length-p9"]
        )
    with pytest.raises(ValueError, match="policy-missing"):
        load_cartpole_sweep_animation(evaluation, policy_ids=["policy-missing"])


def test_scene_geometry_scales_each_pole_but_shares_the_position_axis(
    tmp_path: Path,
) -> None:
    data = load_cartpole_sweep_animation(_boundary_evaluation(tmp_path / "evaluation"))
    long_pole, short_pole = data.conditions
    assert long_pole.pole_half_length == 0.5
    assert short_pole.pole_half_length == 0.2

    size = (960, 540)
    first = _scene_geometry(data, size, pole_half_length=long_pole.pole_half_length)
    second = _scene_geometry(data, size, pole_half_length=short_pole.pole_half_length)

    assert first.scale == second.scale
    assert first.center_x == second.center_x
    assert first.pole_scale < second.pole_scale
    assert second.pole_scale_multiplier > first.pole_scale_multiplier
    assert first.pole_pixel_length == pytest.approx(second.pole_pixel_length)


def test_trailing_rms_uses_the_recovery_window_and_falls_back_without_a_prefix(
    tmp_path: Path,
) -> None:
    data = load_cartpole_sweep_animation(_boundary_evaluation(tmp_path / "evaluation"))
    trace = data.conditions[0].traces[0]
    angles = [state["pole_angle"] for state in trace.states]

    for step in (0, 2, len(angles) - 1):
        window = angles[max(0, step - 3) : step + 1]
        expected = math.degrees(
            math.sqrt(sum(angle**2 for angle in window) / len(window))
        )
        assert _trailing_rms_degrees(trace, step, 4) == pytest.approx(expected)

    without_prefix = EpisodeTrace(
        policy_id=trace.policy_id,
        seed=trace.seed,
        states=trace.states,
        terminated=trace.terminated,
        truncated=trace.truncated,
        parameters=trace.parameters,
        max_episode_steps=trace.max_episode_steps,
        trajectory_sha256=trace.trajectory_sha256,
    )
    assert without_prefix.squared_angle_prefix == ()
    assert _trailing_rms_degrees(without_prefix, 3, 4) == pytest.approx(
        _trailing_rms_degrees(trace, 3, 4)
    )


def test_recovery_readout_reports_degrees_only_for_boundary_studies() -> None:
    band = RecoveryBand(rms_angle_degrees=5.0, tail_steps=100)

    assert _policy_readout_label("cartpole_ppo_nominal_v1", 34, 50, 4.25, band) == (
        "PPO 34/50 4.2°"
    )
    assert _policy_readout_label("p", 0, 50, None, band) == "p 0/50 --"
    assert _policy_readout_label("p", 7, 50, 4.25, None) == "p 7/50"


def test_recovery_wedge_is_drawn_from_the_declared_band(tmp_path: Path) -> None:
    data = load_cartpole_sweep_animation(_boundary_evaluation(tmp_path / "evaluation"))
    size = (960, 540)
    geometry = _scene_geometry(data, size, pole_half_length=0.5)
    image = Image.new("RGB", size, "white")

    _draw_recovery_wedge(
        ImageDraw.Draw(image),
        geometry,
        RecoveryBand(rms_angle_degrees=5.0, tail_steps=4),
        _fonts(2, compact=True),
    )

    axle_y = geometry.baseline_y - geometry.axle_height * geometry.scale
    inside = image.getpixel((round(geometry.center_x), round(axle_y - 40)))
    outside_offset = round(geometry.pole_pixel_length * math.sin(math.radians(30)))
    outside = image.getpixel(
        (round(geometry.center_x) + outside_offset, round(axle_y - 40))
    )
    assert inside == _RECOVERY_BAND_FILL
    assert outside == (255, 255, 255)


def test_boundary_frames_show_the_band_and_plain_sweeps_do_not(
    tmp_path: Path,
) -> None:
    boundary = load_cartpole_sweep_animation(
        _boundary_evaluation(tmp_path / "boundary")
    )
    plain = load_cartpole_sweep_animation(_tiny_evaluation(tmp_path / "plain"))
    colors = dict(boundary.policy_colors)

    with_band = _render_condition_frame(
        boundary.conditions[0], boundary, 2, 1, COMPOSITE_PANEL_SIZE, colors
    )
    without_band = _render_condition_frame(
        plain.conditions[0],
        plain,
        2,
        1,
        COMPOSITE_PANEL_SIZE,
        dict(plain.policy_colors),
    )

    assert _has_band_pixels(with_band)
    assert not _has_band_pixels(without_band)


def test_composite_layout_accepts_more_than_six_conditions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert MAX_COMPOSITE_CONDITIONS >= 9
    evaluation = _many_condition_evaluation(tmp_path / "evaluation", count=7)
    output = tmp_path / "animations"

    assert (
        main(
            [
                "animate-sweep",
                "--evaluation",
                str(evaluation),
                "--output",
                str(output),
                "--layout",
                "composite",
                "--frame-stride",
                "4",
            ]
        )
        == 0
    )

    capsys.readouterr()
    manifest = json.loads((output / "animation-manifest.json").read_text())
    assert len(manifest["selected_condition_ids"]) == 7
    with Image.open(output / "cartpole-overlay-comparison.gif") as image:
        assert image.size == (3 * COMPOSITE_PANEL_SIZE[0], 3 * COMPOSITE_PANEL_SIZE[1])


def test_cli_condition_and_policy_filters_reach_the_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    evaluation = _boundary_evaluation(tmp_path / "evaluation")
    output = tmp_path / "animations"

    assert (
        main(
            [
                "animate-sweep",
                "--evaluation",
                str(evaluation),
                "--output",
                str(output),
                "--layout",
                "individual",
                "--frame-stride",
                "4",
                "--condition",
                "theta-p30_length-p0p2",
                "--policy",
                "policy-ppo,policy-ppo-anti",
            ]
        )
        == 0
    )

    capsys.readouterr()
    manifest = json.loads((output / "animation-manifest.json").read_text())
    assert manifest["selected_condition_ids"] == ["theta-p30_length-p0p2"]
    assert manifest["selected_policy_ids"] == ["policy-ppo", "policy-ppo-anti"]
    assert manifest["recovery_band"] == {
        "recovery_rms_angle_deg": 5.0,
        "recovery_tail_steps": 4,
        "source": "suite.boundary_study",
    }
    condition = manifest["conditions"][0]
    assert condition["pole_half_length"] == 0.2
    assert condition["pole_scale_multipliers"]["composite"] > 1
    assert len(manifest["source_trajectories"]) == 4
    assert all(
        record["trajectory_sha256"] for record in manifest["source_trajectories"]
    )
    assert sorted(path.name for path in output.glob("*.gif")) == [
        "cartpole-overlay-theta-p30_length-p0p2.gif"
    ]


def test_plain_sweeps_keep_a_null_recovery_band(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    evaluation = _tiny_evaluation(tmp_path / "evaluation")
    output = tmp_path / "animations"

    assert (
        main(
            [
                "animate-sweep",
                "--evaluation",
                str(evaluation),
                "--output",
                str(output),
                "--layout",
                "composite",
                "--frame-stride",
                "2",
            ]
        )
        == 0
    )

    capsys.readouterr()
    manifest = json.loads((output / "animation-manifest.json").read_text())
    assert manifest["recovery_band"] is None
    assert manifest["selected_policy_ids"] == ["policy-lqr", "policy-ppo"]


def _has_band_pixels(image: Image.Image) -> bool:
    pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
    distance = np.abs(pixels - np.asarray(_RECOVERY_BAND_FILL, dtype=np.int16)).sum(-1)
    return bool((distance < 24).sum() > 50)


def _rgb(value: str) -> tuple[int, int, int]:
    raw = value.removeprefix("#")
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)


def _boundary_evaluation(root: Path) -> Path:
    """A three-policy recovery-boundary sweep with two different pole lengths."""

    policies = ["policy-lqr", "policy-ppo", "policy-ppo-anti"]
    seeds = [0, 1]
    conditions = [
        ("theta-m30_length-p0p5", -30.0, 0.5),
        ("theta-p30_length-p0p2", 30.0, 0.2),
    ]
    max_episode_steps = 8
    suite = {
        "schema_version": 1,
        "suite": {
            "schema_version": 1,
            "suite_id": "tiny-cartpole-boundary",
            "environment_id": "CartPole-v1",
            "perturbation_name": "cartpole-recovery-angle-length-v1",
            "policy_ids": policies,
            "seeds": seeds,
            "grid": {"initial_theta_deg": [-30.0, 30.0], "length": [0.5, 0.2]},
            "boundary_study": {
                "kind": "adaptive-boundary-v1",
                "recovery_rms_angle_deg": 5.0,
                "recovery_tail_steps": 4,
            },
        },
        "nominal_environment": {
            "environment_id": "CartPole-v1",
            "max_episode_steps": max_episode_steps,
        },
        "conditions": [
            {
                "condition_id": condition_id,
                "perturbation": {
                    "name": "cartpole-recovery-angle-length-v1",
                    "parameters": {
                        "initial_theta_deg": theta,
                        "length": length,
                        "theta_threshold_deg": 90.0,
                    },
                },
            }
            for condition_id, theta, length in conditions
        ],
    }
    root.mkdir(parents=True)
    (root / "suite.json").write_text(json.dumps(suite))
    for condition_id, theta, length in conditions:
        for policy in policies:
            for seed in seeds:
                terminated = policy == "policy-ppo" and seed == 1
                step_count = 5 if terminated else max_episode_steps
                decay = 0.35 if "anti" in policy else 0.75
                states = [
                    _state(
                        0.3 * math.sin(0.4 * (step + seed)),
                        angle=math.radians(theta) * decay**step,
                    )
                    for step in range(step_count + 1)
                ]
                _write_saved_episode(
                    root / "episodes" / policy / condition_id / f"seed-{seed:03d}",
                    states,
                    terminated=terminated,
                    policy_id=policy,
                    seed=seed,
                    parameters={
                        "initial_theta_deg": theta,
                        "length": length,
                        "theta_threshold_deg": 90.0,
                    },
                    length=length,
                    max_episode_steps=max_episode_steps,
                )
    return root


def _many_condition_evaluation(root: Path, *, count: int) -> Path:
    """A single-policy sweep wide enough to exercise the composite grid cap."""

    policies = ["policy-lqr"]
    seeds = [0]
    thetas = [float(5 * index) for index in range(count)]
    suite = {
        "schema_version": 1,
        "suite": {
            "schema_version": 1,
            "suite_id": "tiny-cartpole-wide",
            "environment_id": "CartPole-v1",
            "perturbation_name": "cartpole-recovery-angle-length-v1",
            "policy_ids": policies,
            "seeds": seeds,
            "grid": {"initial_theta_deg": thetas},
        },
        "nominal_environment": {
            "environment_id": "CartPole-v1",
            "max_episode_steps": 4,
        },
        "conditions": [
            {
                "condition_id": f"theta-p{index}",
                "perturbation": {
                    "name": "cartpole-recovery-angle-length-v1",
                    "parameters": {"initial_theta_deg": theta, "length": 0.5},
                },
            }
            for index, theta in enumerate(thetas)
        ],
    }
    root.mkdir(parents=True)
    (root / "suite.json").write_text(json.dumps(suite))
    for index, theta in enumerate(thetas):
        states = [
            _state(0.05 * step, angle=math.radians(theta) * 0.8**step)
            for step in range(5)
        ]
        _write_saved_episode(
            root / "episodes" / policies[0] / f"theta-p{index}" / "seed-000",
            states,
            terminated=False,
            policy_id=policies[0],
            seed=0,
            parameters={"initial_theta_deg": theta, "length": 0.5},
        )
    return root


def _tiny_evaluation(root: Path) -> Path:
    policies = ["policy-lqr", "policy-ppo"]
    seeds = [0, 1]
    conditions = [
        ("dropout-p0", 0.0),
        ("dropout-p0p4", 0.4),
    ]
    suite = {
        "schema_version": 1,
        "suite": {
            "schema_version": 1,
            "suite_id": "tiny-cartpole-animation",
            "environment_id": "CartPole-v1",
            "perturbation_name": "cartpole-action-dropout-v1",
            "policy_ids": policies,
            "seeds": seeds,
            "grid": {"dropout_probability": [0.0, 0.4]},
        },
        "nominal_environment": {
            "environment_id": "CartPole-v1",
            "max_episode_steps": 4,
        },
        "conditions": [
            {
                "condition_id": condition_id,
                "perturbation": {
                    "name": "cartpole-action-dropout-v1",
                    "parameters": {"dropout_probability": probability},
                },
            }
            for condition_id, probability in conditions
        ],
    }
    root.mkdir(parents=True)
    (root / "suite.json").write_text(json.dumps(suite))
    for condition_id, probability in conditions:
        for policy in policies:
            for seed in seeds:
                terminated = probability > 0 and policy == "policy-ppo" and seed == 1
                step_count = 3 if terminated else 4
                states = [
                    _state(
                        0.08 * step * (1 if "lqr" in policy else -1),
                        angle=0.025 * (step + seed),
                    )
                    for step in range(step_count + 1)
                ]
                _write_saved_episode(
                    root
                    / "episodes"
                    / policy
                    / condition_id
                    / f"seed-{seed:03d}",
                    states,
                    terminated=terminated,
                    policy_id=policy,
                    seed=seed,
                    parameters={"dropout_probability": probability},
                )
    return root


def _write_saved_episode(
    root: Path,
    states: list[dict[str, float]],
    *,
    terminated: bool,
    policy_id: str | None = None,
    seed: int | None = None,
    parameters: dict[str, float] | None = None,
    length: float = 0.5,
    max_episode_steps: int = 4,
) -> None:
    root.mkdir(parents=True)
    metadata = {
        "schema_version": 4,
        "episode_seed": seed,
        "policy_artifact": {"policy_id": policy_id},
        "perturbation": {"parameters": dict(parameters or {})},
        "resolved_environment": {
            "environment_id": "CartPole-v1",
            "max_episode_steps": max_episode_steps,
            "parameters": {
                "length": length,
                "tau": 0.02,
                "x_threshold": 2.4,
            },
        },
    }
    rows = [
        {
            "type": "reset",
            "observation": [],
            "environment_state": states[0],
            "perturbation_diagnostics": {},
            "info": {},
        }
    ]
    for index, state in enumerate(states[1:], start=1):
        final = index == len(states) - 1
        rows.append(
            {
                "type": "transition",
                "action": 0,
                "environment_action": 0,
                "policy_diagnostics": {},
                "perturbation_diagnostics": {},
                "reward": 1.0,
                "observation": [],
                "environment_state": state,
                "terminated": final and terminated,
                "truncated": final and not terminated,
                "info": {},
            }
        )
    content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ).encode()
    digest = hashlib.sha256(content).hexdigest()
    (root / "metadata.json").write_text(json.dumps(metadata))
    (root / "trajectory.jsonl").write_bytes(content)
    (root / "trajectory.sha256").write_text(digest + "\n")


def _state(position: float, *, angle: float = 0.0) -> dict[str, float]:
    return {
        "cart_position": position,
        "cart_velocity": 0.0,
        "pole_angle": angle,
        "pole_angular_velocity": 0.0,
    }
