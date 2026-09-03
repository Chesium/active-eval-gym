"""Command-line interface for reproducible evaluation rollouts."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from active_eval_gym.envs.factory import SUPPORTED_ENVIRONMENTS, make_environment
from active_eval_gym.metrics import compute_metrics
from active_eval_gym.models import PolicyMetadata
from active_eval_gym.policies.base import ConstantPolicy, zero_action
from active_eval_gym.rollout import collect_episode
from active_eval_gym.serialization import (
    ensure_output_available,
    write_episode,
    write_metrics,
)

CONSTANT_POLICY_ID = "constant-zero-v1"


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""

    parser = argparse.ArgumentParser(prog="active-eval-gym")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rollout_parser = subparsers.add_parser(
        "rollout", help="collect and save one fixed-policy episode"
    )
    rollout_parser.add_argument(
        "--env", required=True, choices=SUPPORTED_ENVIRONMENTS, dest="env_id"
    )
    rollout_parser.add_argument("--seed", required=True, type=int)
    rollout_parser.add_argument("--output", required=True, type=Path)
    rollout_parser.set_defaults(handler=_run_rollout)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested command."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (FileExistsError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))


def _run_rollout(args: argparse.Namespace) -> int:
    ensure_output_available(args.output)
    env = make_environment(args.env_id)
    try:
        policy = ConstantPolicy(zero_action(env.action_space))
        episode = collect_episode(
            env,
            policy,
            policy_metadata=PolicyMetadata(policy_id=CONSTANT_POLICY_ID),
            episode_seed=args.seed,
        )
    finally:
        env.close()

    trajectory_hash = write_episode(args.output, episode)
    metrics = compute_metrics(episode, source_trajectory_sha256=trajectory_hash)
    write_metrics(args.output, metrics)
    print(
        json.dumps(
            {
                "environment_id": args.env_id,
                "episode_length": metrics.episode_length,
                "episode_return": metrics.episode_return,
                "output": str(args.output),
                "trajectory_sha256": trajectory_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
