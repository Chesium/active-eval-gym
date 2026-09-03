"""Shared constructors for schema-v2 rollout tests."""

from pathlib import Path

import gymnasium as gym

from active_eval_gym.config import load_nominal_env_spec
from active_eval_gym.envs.specs import (
    IDENTITY_OBSERVATION,
    capture_resolved_environment,
    package_versions,
)
from active_eval_gym.models import PolicyArtifactMetadata, PolicyDesignSpec

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "envs"
CONFIG_NAMES = {
    "CartPole-v1": "cartpole_v1_nominal.toml",
    "Pendulum-v1": "pendulum_v1_nominal.toml",
    "MiniGrid-Empty-8x8-v0": "minigrid_empty8x8_v0_nominal.toml",
}


def rollout_provenance(env: gym.Env, env_id: str):
    nominal = load_nominal_env_spec(CONFIG_ROOT / CONFIG_NAMES[env_id])
    resolved = capture_resolved_environment(env, nominal, IDENTITY_OBSERVATION)
    design = PolicyDesignSpec(
        schema_version=1,
        design_id="test-constant-design",
        policy_id="test-constant",
        policy_type="builtin",
        algorithm="constant-zero",
        algorithm_library="active-eval-gym",
        environment=nominal,
        environment_package_versions=package_versions(env_id),
        observation_adapter=IDENTITY_OBSERVATION,
        training_seed=None,
        training_steps=None,
        device=None,
        hyperparameters={},
    )
    artifact = PolicyArtifactMetadata(
        schema_version=1,
        policy_id="test-constant",
        design_spec=design,
        artifact_format="builtin-v1",
        model_filename="builtin",
        model_sha256="builtin",
        package_versions={"active-eval-gym": "test"},
        source_version={"test": True},
    )
    return artifact, nominal, resolved
