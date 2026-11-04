from math import radians

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from active_eval_gym.config import load_ppo_boundary_study
from active_eval_gym.envs.condition_sampler import (
    CONDITION_INFO_KEY,
    CartPoleRecoveryConditionSampler,
)
from active_eval_gym.envs.factory import make_environment
from active_eval_gym.envs.perturbations import (
    CARTPOLE_RECOVERY_ANGLE_LENGTH,
    PerturbationSpec,
)
from active_eval_gym.ppo_boundary import (
    condition_seed_sequence,
    episode_seed,
    support_conditions,
)
from active_eval_gym.sweeps import SweepCondition
from tests.helpers import STUDY_CONFIG_V2

SPEC = load_ppo_boundary_study(STUDY_CONFIG_V2)
CONDITIONS = support_conditions(SPEC, "discovery")
UNIFORM = np.full(len(CONDITIONS), 1.0 / len(CONDITIONS))
NOMINAL_CONDITION = next(
    index
    for index, condition in enumerate(CONDITIONS)
    if condition.perturbation.parameters["initial_theta_deg"] == 0
    and condition.perturbation.parameters["length"] == 0.5
)


def _sampler(
    probabilities: np.ndarray = UNIFORM,
    *,
    conditions=CONDITIONS,
    worker_id: int = 0,
    replicate_id: int = 0,
    environment: gym.Env | None = None,
) -> CartPoleRecoveryConditionSampler:
    return CartPoleRecoveryConditionSampler(
        environment if environment is not None else make_environment("CartPole-v1"),
        conditions,
        probabilities,
        condition_rng=np.random.default_rng(
            condition_seed_sequence(
                SPEC, replicate_id=replicate_id, worker_id=worker_id
            )
        ),
        reset_seed=lambda index: episode_seed(
            role=SPEC.seeds.roles["training"],
            replicate_id=replicate_id,
            worker_id=worker_id,
            episode_index=index,
        ),
        worker_id=worker_id,
    )


def _point(probabilities_at: int) -> np.ndarray:
    weights = np.zeros(len(CONDITIONS))
    weights[probabilities_at] = 1.0
    return weights


def _record(info: dict) -> dict:
    return info[CONDITION_INFO_KEY]


def _action_sequence(seed: int, count: int) -> list[int]:
    return np.random.default_rng(seed).integers(0, 2, size=count).tolist()


def _run_fixed(condition, reset_seed: int, actions: list[int]) -> list[tuple]:
    env = make_environment("CartPole-v1", condition.perturbation)
    try:
        observation, _ = env.reset(seed=reset_seed)
        rows = [(observation.tobytes(), None, None, None)]
        for action in actions:
            observation, reward, terminated, truncated, _ = env.step(action)
            rows.append((observation.tobytes(), reward, terminated, truncated))
            if terminated or truncated:
                break
        return rows
    finally:
        env.close()


def test_sampled_episodes_match_the_fixed_wrapper_across_condition_changes() -> None:
    env = _sampler()
    actions = _action_sequence(3, 600)
    conditions_seen = set()
    try:
        for _ in range(60):
            observation, info = env.reset()
            start = _record(info)
            rows = [(observation.tobytes(), None, None, None)]
            for action in actions:
                observation, reward, terminated, truncated, _ = env.step(action)
                rows.append((observation.tobytes(), reward, terminated, truncated))
                if terminated or truncated:
                    break
            conditions_seen.add(start["condition_index"])
            condition = CONDITIONS[start["condition_index"]]
            assert rows == _run_fixed(condition, start["reset_seed"], actions)
    finally:
        env.close()

    assert len(conditions_seen) > 20


def test_full_true_state_matches_the_fixed_wrapper_within_an_episode() -> None:
    condition = CONDITIONS[NOMINAL_CONDITION]
    actions = _action_sequence(5, 40)
    sampled = _sampler(_point(NOMINAL_CONDITION))
    fixed = make_environment("CartPole-v1", condition.perturbation)
    try:
        _, info = sampled.reset()
        fixed.reset(seed=_record(info)["reset_seed"])
        for action in actions:
            _, _, done, cut, step_info = sampled.step(action)
            fixed.step(action)
            np.testing.assert_array_equal(
                _record(step_info)["next_state"], np.asarray(fixed.unwrapped.state)
            )
            if done or cut:
                break
    finally:
        sampled.close()
        fixed.close()


def test_reset_sets_exact_angle_length_cache_and_keeps_nuisance_draws() -> None:
    env = _sampler()
    reference = make_environment("CartPole-v1")
    try:
        for _ in range(40):
            _, info = env.reset()
            record = _record(info)
            base = env.unwrapped
            parameters = CONDITIONS[record["condition_index"]].perturbation.parameters
            assert base.state[2] == radians(parameters["initial_theta_deg"])
            assert base.length == parameters["length"]
            assert base.polemass_length == base.masspole * parameters["length"]
            assert base.theta_threshold_radians == radians(90)

            reference.reset(seed=record["reset_seed"])
            nominal = np.asarray(reference.unwrapped.state)
            np.testing.assert_array_equal(
                np.asarray(base.state)[[0, 1, 3]], nominal[[0, 1, 3]]
            )
            assert np.all(np.abs(np.asarray(base.state)[[0, 1, 3]]) <= 0.05)
    finally:
        env.close()
        reference.close()


def test_plant_stays_fixed_within_an_episode() -> None:
    env = _sampler()
    try:
        for _ in range(10):
            _, info = env.reset()
            length = env.unwrapped.length
            cache = env.unwrapped.polemass_length
            for action in _action_sequence(1, 200):
                *_, terminated, truncated, step_info = env.step(action)
                assert env.unwrapped.length == length
                assert env.unwrapped.polemass_length == cache
                assert (
                    _record(step_info)["episode_index"]
                    == (_record(info)["episode_index"])
                )
                if terminated or truncated:
                    break
    finally:
        env.close()


def test_observation_space_is_common_and_equals_the_fixed_recovery_box() -> None:
    env = _sampler()
    try:
        expected = None
        for condition in (CONDITIONS[0], CONDITIONS[len(CONDITIONS) // 2]):
            fixed = make_environment("CartPole-v1", condition.perturbation)
            box = fixed.observation_space
            fixed.close()
            assert expected is None or box == expected
            expected = box
        assert env.observation_space == expected
        for _ in range(5):
            env.reset()
            assert env.observation_space == expected
            assert env.unwrapped.observation_space == expected
    finally:
        env.close()


def test_observation_box_is_twice_the_recovery_cutoff_and_nothing_else() -> None:
    env = _sampler()
    nominal = make_environment("CartPole-v1")
    try:
        expected_high = nominal.observation_space.high.copy()
        expected_high[2] = np.float32(2.0 * radians(90))
        np.testing.assert_array_equal(env.observation_space.high, expected_high)
        np.testing.assert_array_equal(env.observation_space.low, -expected_high)
    finally:
        env.close()
        nominal.close()


def test_given_seeds_are_ignored_and_the_declared_stream_is_used() -> None:
    ignoring = _sampler(worker_id=3, replicate_id=7)
    plain = _sampler(worker_id=3, replicate_id=7)
    try:
        for index, seed in enumerate((123, None, 456, None)):
            first, info = ignoring.reset(seed=seed)
            second, plain_info = plain.reset()
            np.testing.assert_array_equal(first, second)
            record, plain_record = dict(_record(info)), dict(_record(plain_info))
            np.testing.assert_array_equal(
                record.pop("state"), plain_record.pop("state")
            )
            assert record == plain_record
            assert _record(info)["reset_seed"] == 2 * 10**12 + 7 * 10**7 + (
                3 * 10**5 + index
            )
            assert _record(info)["episode_index"] == index
            assert _record(info)["worker_id"] == 3
    finally:
        ignoring.close()
        plain.close()


def test_auto_resets_follow_the_stream_and_keep_terminal_handling() -> None:
    short = gym.make("CartPole-v1", max_episode_steps=5)
    env = DummyVecEnv(
        [lambda: Monitor(_sampler(_point(NOMINAL_CONDITION), environment=short))]
    )
    env.seed(999)
    try:
        env.reset()
        assert _record(env.reset_infos[0])["episode_index"] == 0
        for step in range(1, 12):
            _, rewards, dones, infos = env.step(np.array([step % 2]))
            record = _record(infos[0])
            assert record["episode_index"] == (step - 1) // 5
            assert rewards[0] == record["environment_reward"]
            assert dones[0] == (step % 5 == 0)
            if dones[0]:
                assert infos[0]["TimeLimit.truncated"] is True
                np.testing.assert_array_equal(
                    infos[0]["terminal_observation"], record["next_observation"]
                )
                assert infos[0]["episode"]["l"] == 5
                restarted = _record(env.reset_infos[0])
                assert restarted["episode_index"] == step // 5
                assert restarted["reset_seed"] == episode_seed(
                    role=2, episode_index=step // 5
                )
                assert restarted["state"][2] == 0.0
    finally:
        env.close()


def test_failure_is_termination_and_not_truncation() -> None:
    condition = SweepCondition(
        "falling",
        PerturbationSpec(
            CARTPOLE_RECOVERY_ANGLE_LENGTH,
            {"initial_theta_deg": 50, "length": 0.02, "theta_threshold_deg": 90},
        ),
    )
    env = DummyVecEnv([lambda: Monitor(_sampler(np.ones(1), conditions=(condition,)))])
    try:
        env.reset()
        for _ in range(500):
            _, _, dones, infos = env.step(np.array([0]))
            if dones[0]:
                assert infos[0]["TimeLimit.truncated"] is False
                assert _record(infos[0])["terminated"] is True
                return
        pytest.fail("The pole never fell.")
    finally:
        env.close()


def test_arms_share_condition_draws_and_reset_seeds() -> None:
    skewed = np.arange(1.0, len(CONDITIONS) + 1)
    skewed /= skewed.sum()
    first, second = _sampler(UNIFORM), _sampler(skewed)
    try:
        for _ in range(30):
            _, first_info = first.reset()
            _, second_info = second.reset()
            assert (
                _record(first_info)["reset_seed"]
                == (_record(second_info)["reset_seed"])
            )
            # Same draw through different CDFs: skew moves mass to later conditions.
            assert (
                _record(first_info)["condition_index"]
                <= (_record(second_info)["condition_index"])
            )
    finally:
        first.close()
        second.close()


def test_realized_frequencies_follow_the_declared_distribution() -> None:
    weights = np.zeros(len(CONDITIONS))
    weights[[0, 7, 100]] = [0.5, 0.3, 0.2]
    env = _sampler(weights)
    try:
        draws = [_record(env.reset()[1])["condition_index"] for _ in range(4000)]
    finally:
        env.close()

    values, counts = np.unique(draws, return_counts=True)
    assert values.tolist() == [0, 7, 100]
    assert counts / 4000 == pytest.approx([0.5, 0.3, 0.2], abs=0.03)


def test_worker_streams_differ_and_are_reproducible() -> None:
    def draws(worker: int) -> list[int]:
        env = _sampler(worker_id=worker)
        try:
            return [_record(env.reset()[1])["condition_index"] for _ in range(30)]
        finally:
            env.close()

    assert draws(1) == draws(1)
    assert draws(1) != draws(2)


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        (np.ones(2), "Expected"),
        (np.full(len(CONDITIONS), 2.0 / len(CONDITIONS)), "sum to one"),
        (np.r_[1.5, -0.5, np.zeros(len(CONDITIONS) - 2)], "nonnegative"),
        (np.r_[np.nan, np.zeros(len(CONDITIONS) - 1)], "finite"),
    ],
)
def test_invalid_distributions_are_refused(probabilities, message) -> None:
    env = make_environment("CartPole-v1")
    with pytest.raises(ValueError, match=message):
        _sampler(probabilities, environment=env)
    env.close()


def test_rounding_cannot_select_past_the_last_condition() -> None:
    class HighestDraw:
        def random(self) -> float:
            return 1.0 - 2.0**-53

    weights = np.array([0.5, 0.5 - 1e-13])
    env = CartPoleRecoveryConditionSampler(
        make_environment("CartPole-v1"),
        CONDITIONS[:2],
        weights,
        condition_rng=HighestDraw(),
        reset_seed=lambda index: index,
        worker_id=0,
    )
    try:
        assert _record(env.reset()[1])["condition_index"] == 1
    finally:
        env.close()


def test_mixed_cutoffs_and_other_perturbations_are_refused() -> None:
    cutoff = SweepCondition(
        "other-cutoff",
        PerturbationSpec(
            CARTPOLE_RECOVERY_ANGLE_LENGTH,
            {"initial_theta_deg": 0, "length": 0.5, "theta_threshold_deg": 45},
        ),
    )
    other = SweepCondition(
        "other", PerturbationSpec("cartpole-mass-v1", {"masspole": 0.2})
    )
    env = make_environment("CartPole-v1")
    with pytest.raises(ValueError, match="one failure cutoff"):
        _sampler(
            np.array([0.5, 0.5]), conditions=(CONDITIONS[0], cutoff), environment=env
        )
    with pytest.raises(ValueError, match="cannot apply"):
        _sampler(np.ones(1), conditions=(other,), environment=env)
    env.close()


def test_reset_options_and_early_steps_are_refused() -> None:
    env = _sampler()
    try:
        with pytest.raises(RuntimeError, match="must be reset"):
            env.step(0)
        with pytest.raises(ValueError, match="reset law"):
            env.reset(options={"low": -1.0, "high": 1.0})
    finally:
        env.close()


def test_episode_index_overflow_fails_before_a_namespace_can_overlap() -> None:
    env = _sampler()
    env._next_episode = 100_000
    try:
        with pytest.raises(ValueError, match="episode_index"):
            env.reset()
    finally:
        env.close()
