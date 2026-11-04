"""Per-episode CartPole recovery conditions for curriculum training.

The sampler draws one condition at each reset and holds it for the whole episode.
It applies the same `CartPoleRecoveryTask` as the fixed evaluation wrapper, so a
training episode and an evaluation episode differ only in how they were chosen.
"""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np

from active_eval_gym.envs.perturbations import (
    CARTPOLE_RECOVERY_ANGLE_LENGTH,
    CartPoleRecoveryTask,
    _require_cartpole,
    install_recovery_observation_space,
)

if TYPE_CHECKING:
    from active_eval_gym.sweeps import SweepCondition

CONDITION_INFO_KEY = "active_eval_gym.condition"
SAMPLER_VERSION = "cartpole-recovery-condition-sampler-v1"
SAMPLER_PROBABILITY_TOLERANCE = 1e-12


class CartPoleRecoveryConditionSampler(gym.Wrapper):
    """Draw a recovery condition at reset and own the episode's reset seed.

    The vector environment passes SB3-derived seeds to its first reset and no seed
    afterwards, so any seed given here is ignored. Every reset instead uses the
    declared stream: `reset_seed(episode_index)` for the nuisance draws, and one
    uniform draw from `condition_rng` for the condition. Arms that share that
    generator therefore share those draws, whatever their probabilities.
    """

    def __init__(
        self,
        env: gym.Env,
        conditions: "Sequence[SweepCondition]",
        probabilities: np.ndarray,
        *,
        condition_rng: np.random.Generator,
        reset_seed: Callable[[int], int],
        worker_id: int,
    ) -> None:
        super().__init__(env)
        if not conditions:
            raise ValueError("The condition sampler requires at least one condition.")
        names = {condition.perturbation.name for condition in conditions}
        if names != {CARTPOLE_RECOVERY_ANGLE_LENGTH}:
            raise ValueError(f"The sampler cannot apply perturbations {sorted(names)}.")
        _require_cartpole(env, CARTPOLE_RECOVERY_ANGLE_LENGTH)
        self._conditions = tuple(conditions)
        self._tasks = tuple(
            CartPoleRecoveryTask.from_spec(condition.perturbation)
            for condition in conditions
        )
        thresholds = {task.theta_threshold for task in self._tasks}
        if len(thresholds) != 1:
            raise ValueError("Sampled conditions must share one failure cutoff.")
        self._cdf = _cumulative(probabilities, len(conditions))
        self._condition_rng = condition_rng
        self._reset_seed = reset_seed
        self._worker_id = worker_id
        self._next_episode = 0
        self._episode: dict[str, Any] | None = None
        self._observation: np.ndarray | None = None
        self.observation_space = install_recovery_observation_space(
            env, thresholds.pop()
        )
        self.perturbation_spec = conditions[0].perturbation

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if options:
            raise ValueError("Reset options would change the declared reset law.")
        episode_index = self._next_episode
        reset_seed = self._reset_seed(episode_index)
        draw = float(self._condition_rng.random())
        index = int(np.searchsorted(self._cdf, draw, side="right"))
        self._next_episode += 1

        condition = self._conditions[index]
        task = self._tasks[index]
        task.apply_plant(self.env)
        observation, info = task.reset(self.env, seed=reset_seed)
        self.perturbation_spec = condition.perturbation
        self._observation = observation
        self._episode = {
            "condition_index": index,
            "condition_id": condition.condition_id,
            "worker_id": self._worker_id,
            "episode_index": episode_index,
            "reset_seed": reset_seed,
        }
        return observation, {
            **info,
            CONDITION_INFO_KEY: {**self._episode, "state": self._state()},
        }

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self._episode is None:
            raise RuntimeError("The condition sampler must be reset before stepping.")
        state = self._state()
        observation, reward, terminated, truncated, info = self.env.step(action)
        record = {
            **self._episode,
            "state": state,
            "next_state": self._state(),
            "observation": self._observation,
            "next_observation": observation,
            "environment_reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        self._observation = observation
        return (
            observation,
            reward,
            terminated,
            truncated,
            {**info, CONDITION_INFO_KEY: record},
        )

    def _state(self) -> np.ndarray:
        return np.array(self.env.unwrapped.state, dtype=np.float64)


def _cumulative(probabilities: np.ndarray, count: int) -> np.ndarray:
    weights = np.asarray(probabilities, dtype=np.float64)
    if weights.shape != (count,):
        raise ValueError(f"Expected {count} condition probabilities.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Condition probabilities must be finite and nonnegative.")
    if abs(weights.sum() - 1.0) > SAMPLER_PROBABILITY_TOLERANCE:
        raise ValueError("Condition probabilities must sum to one.")
    cumulative = np.cumsum(weights)
    # Rounding must not leave a gap above the last reachable condition.
    cumulative[cumulative >= cumulative[-1]] = 1.0
    return cumulative
