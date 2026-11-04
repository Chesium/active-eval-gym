"""Prepare, train, evaluate, and analyze the CartPole PPO boundary study."""

import argparse
import json
from pathlib import Path

from active_eval_gym.ppo_boundary import prepare_study
from active_eval_gym.ppo_boundary_analysis import analyze_campaign
from active_eval_gym.ppo_boundary_evaluation import (
    CAMPAIGN_DIRECTORIES,
    collect_campaign,
)
from active_eval_gym.ppo_boundary_stability import score_studies
from active_eval_gym.ppo_boundary_targets import build_distributions
from active_eval_gym.ppo_boundary_training import train_campaign

DEFAULT_CONFIG = Path("configs/experiments/cartpole_ppo_boundary_v3.toml")
CAMPAIGNS = tuple(CAMPAIGN_DIRECTORIES)
TIERS = ("smoke", "pilot", "confirmatory")


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
    evaluate = commands.add_parser("evaluate", help="collect or resume a campaign")
    evaluate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    evaluate.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/policies")
    )
    evaluate.add_argument("--study", type=Path, required=True)
    evaluate.add_argument("--campaign", choices=CAMPAIGNS, required=True)
    evaluate.add_argument("--workers", type=int, default=1)
    evaluate.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow a temporary development run on uncommitted source",
    )
    analyze = commands.add_parser("analyze", help="analyze saved shards only")
    analyze.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    analyze.add_argument("--study", type=Path, required=True)
    analyze.add_argument("--campaign", choices=CAMPAIGNS, required=True)
    analyze.add_argument("--output", type=Path)
    analyze.add_argument("--workers", type=int, default=1)
    targets = commands.add_parser("targets", help="freeze training distributions")
    targets.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    targets.add_argument("--study", type=Path, required=True)
    targets.add_argument("--output", type=Path)
    targets.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow a temporary development record on uncommitted source",
    )
    train = commands.add_parser("train", help="train or resume a tier's children")
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train.add_argument("--artifact-root", type=Path, default=Path("artifacts/policies"))
    train.add_argument("--study", type=Path, required=True)
    train.add_argument("--tier", choices=TIERS, required=True)
    train.add_argument("--workers", type=int, default=1)
    train.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow a temporary development run on uncommitted source",
    )
    stability = commands.add_parser(
        "stability", help="score continuation candidates, pooled across arms"
    )
    stability.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="NAME=STUDY",
        help="a candidate's analyzed study; repeat for each candidate",
    )
    stability.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_study(
            args.config,
            artifact_root=args.artifact_root,
            output_dir=args.output,
            allow_dirty=args.allow_dirty,
        )
        print(f"Protocol: {result['protocol_dir']}")
        if result["development_record"]:
            print("Development record: not valid for confirmatory results.")
    elif args.command == "evaluate":
        result = collect_campaign(
            args.config,
            study_dir=args.study,
            campaign=args.campaign,
            artifact_root=args.artifact_root,
            workers=args.workers,
            allow_dirty=args.allow_dirty,
        )
        print(
            f"Campaign {result['campaign']}: {result['planned_jobs']} planned, "
            f"{result['already_complete_jobs']} already complete, "
            f"{len(result['failed_jobs'])} failed."
        )
        if not result["complete"]:
            raise SystemExit(f"Incomplete campaign: {result['failed_jobs']}")
    elif args.command == "train":
        result = train_campaign(
            args.config,
            study_dir=args.study,
            tier=args.tier,
            artifact_root=args.artifact_root,
            workers=args.workers,
            allow_dirty=args.allow_dirty,
        )
        print(
            f"Tier {result['tier']}: {result['planned_jobs']} planned, "
            f"{result['already_complete_jobs']} already complete, "
            f"{len(result['failed_jobs'])} failed."
        )
        if not result["complete"]:
            raise SystemExit(f"Incomplete tier: {result['failed_jobs']}")
    elif args.command == "stability":
        report = score_studies(
            dict(item.split("=", 1) for item in args.candidate), args.output
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "targets":
        report = build_distributions(
            args.config,
            study_dir=args.study,
            output_dir=args.output,
            allow_dirty=args.allow_dirty,
        )
        band = report["boundary_band"]
        print(f"Boundary band: {band['size']} conditions.")
    else:
        analyze_campaign(
            args.config,
            study_dir=args.study,
            campaign=args.campaign,
            output_dir=args.output,
            workers=args.workers,
        )
        print(f"Analyzed campaign {args.campaign}.")


if __name__ == "__main__":
    main()
