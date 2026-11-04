"""Run or analyze the declared update-rule and plant-family study."""

import argparse
import os
from pathlib import Path

# One BLAS thread per worker; families run in separate processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from active_eval_gym.linear_family_study import (  # noqa: E402
    FamilyStudySpec,
    analyze_family_study,
    run_family_study,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="collect an immutable experiment")
    run.add_argument("--preset", choices=("smoke", "full"), default="smoke")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--jobs", type=int, default=1)
    analyze = commands.add_parser("analyze", help="analyze saved artifacts only")
    analyze.add_argument("--run", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        spec = FamilyStudySpec.from_preset(args.preset)
        run_family_study(args.output, spec, jobs=args.jobs)
    else:
        analyze_family_study(args.run, args.output)
        print(f"Summary: {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
