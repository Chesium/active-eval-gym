"""Shared constructors for rollout-schema tests."""

from pathlib import Path

import gymnasium as gym

from active_eval_gym.config import QualityGate, load_nominal_env_spec
from active_eval_gym.envs.specs import (
    IDENTITY_OBSERVATION,
    capture_resolved_environment,
    package_versions,
)
from active_eval_gym.models import PolicyArtifactMetadata, PolicyDesignSpec
from active_eval_gym.policies.artifacts import (
    train_learned_artifact,
    write_freeze_result,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "envs"
STUDY_CONFIG = ROOT / "configs/experiments/cartpole_ppo_boundary_v1.toml"
PARENT_DESIGN = ROOT / "configs/policies/cartpole_ppo_nominal_v1.toml"
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


def edited_study_config(tmp_path: Path, *replacements: tuple[str, str]) -> Path:
    """Write the tracked study config with exact text replacements applied."""

    text = STUDY_CONFIG.read_text().replace(
        'design_config = "../policies/cartpole_ppo_nominal_v1.toml"',
        f'design_config = "{PARENT_DESIGN}"',
    )
    for old, new in replacements:
        assert old in text
        text = text.replace(old, new)
    path = tmp_path / "study.toml"
    path.write_text(text)
    return path


def tiny_parent_config(
    tmp_path: Path, *replacements: tuple[str, str]
) -> tuple[Path, Path]:
    """Build a real frozen short-trained parent and a config declaring it."""

    design = tmp_path / "tiny.toml"
    design.write_text(
        PARENT_DESIGN.read_text()
        .replace('"cartpole_ppo_nominal_v1"', '"tiny_parent"')
        .replace("training_steps = 100000", "training_steps = 256")
        .replace(
            '"../envs/cartpole_v1_nominal.toml"',
            f'"{CONFIG_ROOT / "cartpole_v1_nominal.toml"}"',
        )
    )
    root = tmp_path / "policies"
    root.mkdir()
    metadata = train_learned_artifact(design, root / "tiny_parent")
    write_freeze_result(
        root / "tiny_parent", passed=True, gate=QualityGate(), results={}
    )
    config = edited_study_config(
        tmp_path,
        ('policy_id = "cartpole_ppo_nominal_v1"', 'policy_id = "tiny_parent"'),
        (f'"{PARENT_DESIGN}"', f'"{design}"'),
        (
            "37c9bac10919b2d6bdc833fd5a3a920460d4b80f09392ef7fb758e4777a856e8",
            metadata.model_sha256,
        ),
        *replacements,
    )
    return config, root
