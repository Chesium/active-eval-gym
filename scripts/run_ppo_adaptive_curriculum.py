"""Run the explicitly configured adaptive-frontier development experiment."""

import argparse
import json
from pathlib import Path

from active_eval_gym.ppo_adaptive_curriculum import (
    load_adaptive_study,
    run_adaptive,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/cartpole_ppo_boundary_v5_adaptive.toml"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/policies")
    )
    args = parser.parse_args()
    settings, _ = load_adaptive_study(args.config)
    study = args.output or Path("artifacts/experiments") / settings.study_id
    print(
        f"Adaptive study: {settings.rounds} rounds × "
        f"{settings.transitions_per_round:,} transitions × "
        f"{len(settings.replicate_ids)} seeds.",
        flush=True,
    )
    print(
        json.dumps(run_adaptive(args.config, study, args.artifact_root), indent=2),
        flush=True,
    )
    # Metrics/figures read only the completed experiment's artifacts.
    from analyze_ppo_adaptive_curriculum import analyze

    analyze(study)


if __name__ == "__main__":
    main()
