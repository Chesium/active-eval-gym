"""Stable-Baselines3 training and frozen inference adapters."""

from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from active_eval_gym.envs.factory import make_environment
from active_eval_gym.envs.specs import apply_observation_adapter
from active_eval_gym.models import PolicyDesignSpec

ALGORITHMS = {"DQN": DQN, "SAC": SAC, "PPO": PPO}
SCHEDULE_PARAMETERS = ("learning_rate", "clip_range")


def linear_schedule(initial_value: float):
    """Return the RL Zoo convention: value times progress remaining."""

    initial_value = float(initial_value)
    if initial_value <= 0.0:
        raise ValueError("Linear schedule initial_value must be positive.")

    def schedule(progress_remaining: float) -> float:
        return float(progress_remaining) * initial_value

    return schedule


class TrainingEpisodeRecorder(BaseCallback):
    """Collect Monitor episode summaries without selecting checkpoints."""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.returns: list[float] = []
        self.lengths: list[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            episode = info.get("episode")
            if episode is not None:
                self.returns.append(float(episode["r"]))
                self.lengths.append(int(episode["l"]))
        return True

    def summary(self) -> dict[str, Any]:
        if not self.returns:
            return {"episode_count": 0}
        return {
            "episode_count": len(self.returns),
            "return": _summary_values(self.returns),
            "episode_length": _summary_values(self.lengths),
        }


class SB3Policy:
    """Inference-only adapter around a loaded Stable-Baselines3 model."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def act(self, observation: Any, *, deterministic: bool = True) -> Any:
        action, _ = self._model.predict(observation, deterministic=deterministic)
        if isinstance(action, np.ndarray) and action.shape == ():
            return action.item()
        return action


def train_sb3_policy(spec: PolicyDesignSpec, model_path: Path) -> dict[str, Any]:
    """Train exactly one configured SB3 policy and save its final checkpoint."""

    if spec.algorithm not in ALGORITHMS:
        raise ValueError(f"Unsupported SB3 algorithm {spec.algorithm!r}.")
    if spec.training_seed is None or spec.training_steps is None:
        raise ValueError("SB3 designs require training_seed and training_steps.")
    if spec.device != "cpu":
        raise ValueError("CHE-48 policy designs require device='cpu'.")

    kwargs = _resolve_schedules(dict(spec.hyperparameters))
    n_envs = int(kwargs.pop("n_envs", 1))
    policy_name = str(kwargs.pop("policy"))
    if spec.algorithm == "PPO":
        factories = [
            _environment_factory(spec, spec.training_seed + rank)
            for rank in range(n_envs)
        ]
        environment: Any = DummyVecEnv(factories)
    else:
        if n_envs != 1:
            raise ValueError(f"{spec.algorithm} configuration requires n_envs=1.")
        environment = _make_policy_environment(spec)

    algorithm = ALGORITHMS[spec.algorithm]
    model = algorithm(
        policy_name,
        environment,
        seed=spec.training_seed,
        device=spec.device,
        verbose=0,
        **kwargs,
    )
    recorder = TrainingEpisodeRecorder()
    try:
        model.learn(total_timesteps=spec.training_steps, callback=recorder)
        model.save(model_path.with_suffix(""))
        actual_training_steps = int(model.num_timesteps)
    finally:
        environment.close()
    return {
        "algorithm": spec.algorithm,
        "training_seed": spec.training_seed,
        "training_steps": spec.training_steps,
        "requested_training_steps": spec.training_steps,
        "actual_training_steps": actual_training_steps,
        "n_envs": n_envs,
        "checkpoint_selection": "final",
        "training_episodes": recorder.summary(),
    }


def load_sb3_policy(algorithm_name: str, model_path: Path) -> SB3Policy:
    """Load a checkpoint without attaching a training environment."""

    try:
        algorithm = ALGORITHMS[algorithm_name]
    except KeyError as error:
        raise ValueError(f"Unsupported SB3 algorithm {algorithm_name!r}.") from error
    return SB3Policy(algorithm.load(model_path, device="cpu"))


def _environment_factory(spec: PolicyDesignSpec, initial_seed: int):
    def factory() -> gym.Env:
        env = _make_policy_environment(spec)
        env.reset(seed=initial_seed)
        return env

    return factory


def _make_policy_environment(spec: PolicyDesignSpec) -> gym.Env:
    env = make_environment(spec.environment.environment_id)
    try:
        adapted = apply_observation_adapter(env, spec.observation_adapter)
        return Monitor(adapted)
    except Exception:
        env.close()
        raise


def _summary_values(values: list[float] | list[int]) -> dict[str, float]:
    return {
        "mean": fmean(values),
        "standard_deviation": pstdev(values),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _resolve_schedules(hyperparameters: dict[str, Any]) -> dict[str, Any]:
    for name in SCHEDULE_PARAMETERS:
        value = hyperparameters.get(name)
        if not isinstance(value, dict):
            continue
        if set(value) != {"kind", "initial_value"} or value["kind"] != "linear":
            raise ValueError(
                f"{name} schedule must contain kind='linear' and initial_value."
            )
        hyperparameters[name] = linear_schedule(value["initial_value"])
    return hyperparameters
