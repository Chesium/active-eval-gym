"""Prepare the CartPole PPO boundary-targeting study."""

import argparse
from pathlib import Path

from active_eval_gym.ppo_boundary import prepare_study

DEFAULT_CONFIG = Path("configs/experiments/cartpole_ppo_boundary_v1.toml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="write the immutable protocol")
    prepare.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/policies")
    )
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument(
        "--allow-dirty",
        action="store_true",
        help="record uncommitted source as a development-only protocol",
    )
    args = parser.parse_args()
    result = prepare_study(
        args.config,
        artifact_root=args.artifact_root,
        output_dir=args.output,
        allow_dirty=args.allow_dirty,
    )
    print(f"Protocol: {result['protocol_dir']}")
    if result["development_record"]:
        print("Development record: not valid for confirmatory results.")


if __name__ == "__main__":
    main()
