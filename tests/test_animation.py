import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from active_eval_gym.animation import (
    EpisodeTrace,
    _composite_density_layers,
    _visible_state,
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
                    dropout_probability=probability,
                )
    return root


def _write_saved_episode(
    root: Path,
    states: list[dict[str, float]],
    *,
    terminated: bool,
    policy_id: str | None = None,
    seed: int | None = None,
    dropout_probability: float | None = None,
) -> None:
    root.mkdir(parents=True)
    metadata = {
        "schema_version": 4,
        "episode_seed": seed,
        "policy_artifact": {"policy_id": policy_id},
        "perturbation": {
            "parameters": {"dropout_probability": dropout_probability}
        },
        "resolved_environment": {
            "environment_id": "CartPole-v1",
            "max_episode_steps": 4,
            "parameters": {
                "length": 0.5,
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
