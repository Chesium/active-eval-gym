"""Fixed-policy episode collection."""

from copy import deepcopy

import gymnasium as gym

from active_eval_gym.envs.factory import capture_initial_state
from active_eval_gym.envs.perturbations import PerturbationSpec
from active_eval_gym.envs.specs import package_versions
from active_eval_gym.models import (
    EpisodeMetadata,
    EpisodeRecord,
    NominalEnvSpec,
    PolicyAction,
    PolicyArtifactMetadata,
    ResetRecord,
    ResolvedEnvSpec,
    TransitionRecord,
)
from active_eval_gym.policies.base import Policy

EPISODE_SCHEMA_VERSION = 2


def collect_episode(
    env: gym.Env,
    policy: Policy,
    *,
    policy_artifact: PolicyArtifactMetadata,
    nominal_environment: NominalEnvSpec,
    resolved_environment: ResolvedEnvSpec,
    episode_seed: int,
    deterministic: bool = True,
) -> EpisodeRecord:
    """Run a fixed policy until the environment terminates or truncates."""

    env_id = _environment_id(env)
    perturbation = _perturbation_spec(env)
    observation, reset_info = env.reset(seed=episode_seed)
    initial_observation = deepcopy(observation)
    initial_state = capture_initial_state(env, env_id)
    transitions: list[TransitionRecord] = []

    while True:
        decision = policy.act(observation, deterministic=deterministic)
        if isinstance(decision, PolicyAction):
            action = decision.action
            policy_diagnostics = deepcopy(decision.diagnostics)
        else:
            action = decision
            policy_diagnostics = {}
        observation, reward, terminated, truncated, info = env.step(action)
        transitions.append(
            TransitionRecord(
                action=deepcopy(action),
                policy_diagnostics=policy_diagnostics,
                reward=float(reward),
                observation=deepcopy(observation),
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=deepcopy(dict(info)),
            )
        )
        if terminated or truncated:
            break

    return EpisodeRecord(
        metadata=EpisodeMetadata(
            schema_version=EPISODE_SCHEMA_VERSION,
            policy_design=policy_artifact.design_spec,
            policy_artifact=policy_artifact,
            nominal_environment=nominal_environment,
            perturbation=perturbation,
            resolved_environment=resolved_environment,
            episode_seed=episode_seed,
            deterministic=deterministic,
            package_versions=package_versions(env_id),
            initial_state=initial_state,
        ),
        reset=ResetRecord(
            observation=initial_observation,
            info=deepcopy(dict(reset_info)),
        ),
        transitions=tuple(transitions),
    )


def _environment_id(env: gym.Env) -> str:
    if env.spec is None:
        raise RuntimeError(
            "Environment has no Gymnasium spec and cannot be identified."
        )
    return env.spec.id


def _perturbation_spec(env: gym.Env) -> PerturbationSpec:
    try:
        spec = env.get_wrapper_attr("perturbation_spec")
    except AttributeError as error:
        raise RuntimeError(
            "Environment was not constructed through make_environment()."
        ) from error
    if not isinstance(spec, PerturbationSpec):
        raise TypeError("Environment perturbation metadata is invalid.")
    return spec
