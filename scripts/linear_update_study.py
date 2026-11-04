"""Run or analyze the finite-linear target-map/certificate experiment."""

import argparse
from pathlib import Path

from active_eval_gym.linear_study import StudySpec, analyze_study, run_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="collect an immutable experiment")
    run.add_argument("--preset", choices=("smoke", "full"), default="smoke")
    run.add_argument("--output", type=Path, required=True)
    analyze = commands.add_parser("analyze", help="analyze saved artifacts only")
    analyze.add_argument("--run", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run_study(args.output, StudySpec.from_preset(args.preset))
    else:
        analyze_study(args.run, args.output)
        print(f"Report: {args.output / 'report.md'}")


if __name__ == "__main__":
    main()
