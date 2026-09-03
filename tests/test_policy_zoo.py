import json
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import DQN, SAC

from active_eval_gym.config import (
    NominalSuiteSpec,
    QualityGate,
    load_nominal_suite,
    load_policy_design,
)
from active_eval_gym.envs.factory import make_environment
from active_eval_gym.envs.specs import (
    MINIGRID_PARTIAL_IMAGE_FLAT,
    apply_observation_adapter,
    capture_resolved_environment,
)
from active_eval_gym.evaluate import (
    evaluate_nominal_suite,
    freeze_candidate,
    make_artifact_environment,
)
from active_eval_gym.models import PolicyAction
from active_eval_gym.policies.artifacts import (
    build_lqr_artifact,
    load_policy_artifact,
)
from active_eval_gym.policies.lqr import (
    QuantizedLQRPolicy,
    design_cartpole_lqr,
)
from active_eval_gym.policies.sb3 import load_sb3_policy

ROOT = Path(__file__).resolve().parents[1]
LQR_CONFIG = ROOT / "configs/policies/cartpole_lqr_nominal_quantized_v1.toml"
SUITE_CONFIG = ROOT / "configs/eval/nominal.toml"


def test_lqr_design_matches_gym_transition_and_is_stable() -> None:
    design = load_policy_design(LQR_CONFIG)
    model, error = design_cartpole_lqr(design)

    assert error < 1e-8
    assert np.asarray(model.k).shape == (1, 4)
    eigenvalues = [
        complex(value["real"], value["imag"]) for value in model.closed_loop_eigenvalues
    ]
    assert max(abs(value) for value in eigenvalues) < 1.0


def test_quantized_lqr_returns_action_diagnostics() -> None:
    policy = QuantizedLQRPolicy(np.array([[0.0, 0.0, 1.0, 0.0]]), force_magnitude=10.0)

    positive = policy.act(np.array([0.0, 0.0, -0.2, 0.0]))
    negative = policy.act(np.array([0.0, 0.0, 0.2, 0.0]))

    assert positive == PolicyAction(
        action=1, diagnostics={"desired_force": 0.2, "applied_force": 10.0}
    )
    assert negative == PolicyAction(
        action=0, diagnostics={"desired_force": -0.2, "applied_force": -10.0}
    )


def test_lqr_artifact_build_freeze_load_and_tamper_detection(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "cartpole_lqr_nominal_quantized_v1"
    metadata = build_lqr_artifact(LQR_CONFIG, artifact_dir)

    incompatible_design = replace(
        metadata.design_spec,
        environment_package_versions={
            **metadata.design_spec.environment_package_versions,
            "gymnasium": "incompatible-version",
        },
    )
    with pytest.raises(ValueError, match="requires gymnasium==incompatible-version"):
        make_artifact_environment(replace(metadata, design_spec=incompatible_design))

    with pytest.raises(FileNotFoundError, match="freeze.json"):
        load_policy_artifact(artifact_dir)

    assert freeze_candidate(artifact_dir, load_nominal_suite(SUITE_CONFIG))
    with pytest.raises(FileExistsError, match="freeze.json already exists"):
        freeze_candidate(artifact_dir, load_nominal_suite(SUITE_CONFIG))
    policy, loaded = load_policy_artifact(artifact_dir)
    assert loaded == metadata
    assert policy.act(np.zeros(4)).action == 1

    evaluation_suite = NominalSuiteSpec(
        schema_version=1,
        suite_id="test-nominal",
        seeds=(1, 0),
        policy_ids=(metadata.policy_id,),
        gates={
            metadata.policy_id: QualityGate(
                minimum_success_rate=1.0,
                minimum_mean_episode_length=500.0,
            )
        },
    )
    evaluation_dir = tmp_path / "evaluation"
    summary = evaluate_nominal_suite(
        evaluation_suite,
        artifact_root=tmp_path,
        output_dir=evaluation_dir,
    )
    assert summary["seeds"] == [1, 0]
    assert summary["policies"][metadata.policy_id]["quality_gate_passed"]
    transition = json.loads(
        (evaluation_dir / metadata.policy_id / "seed-001" / "trajectory.jsonl")
        .read_text()
        .splitlines()[1]
    )
    assert "desired_force" in transition["policy_diagnostics"]

    incomplete_suite = NominalSuiteSpec(
        schema_version=1,
        suite_id="incomplete",
        seeds=(0,),
        policy_ids=(metadata.policy_id, "missing-policy"),
        gates={
            metadata.policy_id: QualityGate(),
            "missing-policy": QualityGate(),
        },
    )
    incomplete_output = tmp_path / "incomplete-evaluation"
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        evaluate_nominal_suite(
            incomplete_suite,
            artifact_root=tmp_path,
            output_dir=incomplete_output,
        )
    assert not incomplete_output.exists()

    model_path = artifact_dir / "model.json"
    model = json.loads(model_path.read_text())
    model["k"][0][0] += 1.0
    model_path.write_text(json.dumps(model))
    with pytest.raises(ValueError, match="digest mismatch"):
        load_policy_artifact(artifact_dir)


def test_minigrid_adapter_is_fixed_partial_image_vector() -> None:
    design = load_policy_design(
        ROOT / "configs/policies/minigrid_empty8x8_ppo_partial_image_v1.toml"
    )
    env = make_environment(design.environment.environment_id)
    try:
        env = apply_observation_adapter(env, MINIGRID_PARTIAL_IMAGE_FLAT)
        observation, _ = env.reset(seed=0)
        resolved = capture_resolved_environment(
            env, design.environment, design.observation_adapter
        )
        assert observation.shape == (147,)
        assert env.observation_space.shape == (147,)
        assert resolved.observation_adapter == MINIGRID_PARTIAL_IMAGE_FLAT
    finally:
        env.close()


def test_sb3_adapter_loads_checkpoint_for_deterministic_inference(
    tmp_path: Path,
) -> None:
    env = gym.make("CartPole-v1")
    try:
        model = DQN("MlpPolicy", env, seed=0, device="cpu", buffer_size=100)
        path = tmp_path / "tiny-dqn"
        model.save(path)
    finally:
        env.close()

    policy = load_sb3_policy("DQN", path.with_suffix(".zip"))
    action = policy.act(np.zeros(4, dtype=np.float32), deterministic=True)
    assert action in (0, 1)

    continuous_env = gym.make("Pendulum-v1")
    try:
        continuous_model = SAC(
            "MlpPolicy",
            continuous_env,
            seed=0,
            device="cpu",
            buffer_size=100,
        )
        continuous_path = tmp_path / "tiny-sac"
        continuous_model.save(continuous_path)
    finally:
        continuous_env.close()

    continuous_policy = load_sb3_policy("SAC", continuous_path.with_suffix(".zip"))
    observation = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    first = continuous_policy.act(observation, deterministic=True)
    second = continuous_policy.act(observation, deterministic=True)
    assert isinstance(first, np.ndarray)
    assert first.shape == (1,)
    np.testing.assert_array_equal(first, second)
