"""Nominal and resolved environment specifications."""

from importlib import import_module
from importlib.metadata import version
from typing import Any

import gymnasium as gym

from active_eval_gym.models import NominalEnvSpec, ResolvedEnvSpec

IDENTITY_OBSERVATION = "identity-v1"
MINIGRID_PARTIAL_IMAGE_FLAT = "minigrid-partial-image-flat-v1"


def package_versions(env_id: str) -> dict[str, str]:
    """Return exact environment package versions."""

    packages = {"gymnasium": version("gymnasium")}
    if env_id.startswith("MiniGrid-"):
        packages["minigrid"] = version("minigrid")
    return packages


def apply_observation_adapter(env: gym.Env, adapter: str) -> gym.Env:
    """Apply a named, fixed-shape policy observation adapter."""

    if adapter == IDENTITY_OBSERVATION:
        return env
    if adapter == MINIGRID_PARTIAL_IMAGE_FLAT:
        if not env.spec or not env.spec.id.startswith("MiniGrid-"):
            raise ValueError(f"Observation adapter {adapter!r} requires MiniGrid.")
        import_module("minigrid")
        from minigrid.wrappers import ImgObsWrapper

        return gym.wrappers.FlattenObservation(ImgObsWrapper(env))
    raise ValueError(f"Unknown observation adapter {adapter!r}.")


def capture_resolved_environment(
    env: gym.Env,
    nominal: NominalEnvSpec,
    observation_adapter: str,
) -> ResolvedEnvSpec:
    """Capture actual values from an environment and validate its identity."""

    if env.spec is None or env.spec.id != nominal.environment_id:
        actual = None if env.spec is None else env.spec.id
        raise ValueError(
            f"Expected environment {nominal.environment_id!r}, received {actual!r}."
        )
    parameters, derived, max_steps = _environment_values(env)
    for name, expected in nominal.parameters.items():
        if parameters.get(name) != expected:
            raise ValueError(
                f"Nominal parameter {name!r} expected {expected!r}, "
                f"resolved to {parameters.get(name)!r}."
            )
    if max_steps != nominal.max_episode_steps:
        raise ValueError(
            f"Expected max_episode_steps={nominal.max_episode_steps}, "
            f"resolved to {max_steps}."
        )
    return ResolvedEnvSpec(
        schema_version=1,
        nominal_spec_id=nominal.spec_id,
        environment_id=nominal.environment_id,
        parameters=parameters,
        derived_parameters=derived,
        initial_state_distribution=dict(nominal.initial_state_distribution),
        observation_adapter=observation_adapter,
        max_episode_steps=max_steps,
    )


def _environment_values(
    env: gym.Env,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    base = env.unwrapped
    env_id = env.spec.id if env.spec else ""
    if env_id == "CartPole-v1":
        parameters = {
            "gravity": float(base.gravity),
            "masscart": float(base.masscart),
            "masspole": float(base.masspole),
            "length": float(base.length),
            "force_mag": float(base.force_mag),
            "tau": float(base.tau),
            "kinematics_integrator": str(base.kinematics_integrator),
            "theta_threshold_radians": float(base.theta_threshold_radians),
            "x_threshold": float(base.x_threshold),
        }
        derived = {
            "total_mass": float(base.total_mass),
            "polemass_length": float(base.polemass_length),
        }
    elif env_id == "Pendulum-v1":
        parameters = {
            "g": float(base.g),
            "m": float(base.m),
            "l": float(base.l),
            "dt": float(base.dt),
            "max_speed": float(base.max_speed),
            "max_torque": float(base.max_torque),
        }
        derived = {}
    elif env_id == "MiniGrid-Empty-8x8-v0":
        parameters = {
            "width": int(base.width),
            "height": int(base.height),
            "agent_view_size": int(base.agent_view_size),
            "agent_start_pos": list(base.agent_start_pos),
            "agent_start_dir": int(base.agent_start_dir),
        }
        derived = {"max_steps": int(base.max_steps)}
    else:
        raise ValueError(f"Cannot resolve unsupported environment {env_id!r}.")

    if env_id.startswith("MiniGrid-"):
        max_steps = int(base.max_steps)
    else:
        if env.spec is None or env.spec.max_episode_steps is None:
            raise RuntimeError(f"{env_id} has no episode limit.")
        max_steps = int(env.spec.max_episode_steps)
    return parameters, derived, max_steps
