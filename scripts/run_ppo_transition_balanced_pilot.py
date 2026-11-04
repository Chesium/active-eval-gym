"""Run the versioned development pilot and retain stage timings and logs.

Resumes completed stages without overwriting raw data. Uses existing v3 pilot
results as the comparison and collects no confirmatory evaluation data.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--long", action="store_true", help="extend only the boundary arm to 1,024,000"
    )
    mode.add_argument(
        "--focus", action="store_true", help="reconstruct the 102,400–256,000 interval"
    )
    parser.add_argument(
        "--config",
        type=Path,
    )
    parser.add_argument(
        "--study",
        type=Path,
    )
    parser.add_argument(
        "--baseline",
        type=Path,
    )
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    version = "v4_focus" if args.focus else "v4_long" if args.long else "v4"
    study_id = (
        "cartpole-ppo-boundary-v4-focus"
        if args.focus
        else "cartpole-ppo-boundary-v4-long"
        if args.long
        else "cartpole-ppo-boundary-v4"
    )
    args.config = args.config or Path(
        f"configs/experiments/cartpole_ppo_boundary_{version}.toml"
    )
    args.study = args.study or Path("artifacts/experiments") / study_id
    args.baseline = args.baseline or Path("artifacts/experiments") / (
        "cartpole-ppo-boundary-v4-long"
        if args.focus
        else "cartpole-ppo-boundary-v4"
        if args.long
        else "cartpole-ppo-boundary-v3"
    )
    study = args.study
    study.mkdir(parents=True, exist_ok=True)
    logs = study / "logs"
    logs.mkdir(exist_ok=True)
    timing_path = study / "run-timing.json"
    timings = json.loads(timing_path.read_text()) if timing_path.exists() else []
    environment = dict(os.environ)
    environment.update(
        MPLBACKEND="Agg",
        MPLCONFIGDIR="/tmp/ppo-boundary-matplotlib",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    common = ["--config", str(args.config), "--study", str(study)]
    cli = [sys.executable, "scripts/cartpole_ppo_boundary_study.py"]

    def stage(name: str, command: list[str], complete: Path) -> None:
        if complete.is_file():
            print(f"Skip completed stage: {name}", flush=True)
            return
        started = time.monotonic()
        record = {
            "stage": name,
            "command": command,
            "started_utc": datetime.now(UTC).isoformat(),
            "status": "running",
        }
        timings.append(record)
        timing_path.write_text(json.dumps(timings, indent=2) + "\n")
        print(f"Starting {name}; log: {logs / (name + '.log')}", flush=True)
        with (logs / f"{name}.log").open("a") as output:
            process = subprocess.run(
                command,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record.update(
            elapsed_seconds=time.monotonic() - started,
            finished_utc=datetime.now(UTC).isoformat(),
            returncode=process.returncode,
            status="complete" if process.returncode == 0 else "failed",
        )
        timing_path.write_text(json.dumps(timings, indent=2) + "\n")
        print(
            f"Finished {name}: {record['elapsed_seconds']:.1f}s, "
            f"exit {process.returncode}",
            flush=True,
        )
        if process.returncode:
            raise SystemExit(f"Stage failed; inspect {logs / (name + '.log')}")
        if not complete.is_file():
            raise RuntimeError(
                f"Stage {name} returned without its completion artifact."
            )

    stage(
        "prepare",
        cli
        + [
            "prepare",
            "--config",
            str(args.config),
            "--output",
            str(study),
            "--allow-dirty",
        ],
        study / "protocol/provenance.json",
    )

    def evaluate(campaign: str) -> None:
        directory = "pilot_snapshots" if campaign == "pilot-snapshots" else campaign
        stage(
            f"{campaign}-evaluation",
            cli
            + [
                "evaluate",
                *common,
                "--campaign",
                campaign,
                "--workers",
                str(args.workers),
                "--allow-dirty",
            ],
            study / directory / "collection.json",
        )
        stage(
            f"{campaign}-metrics",
            cli
            + [
                "analyze",
                *common,
                "--campaign",
                campaign,
                "--workers",
                str(args.workers),
            ],
            study / "metrics_v1" / campaign / "summary.json",
        )

    evaluate("discovery")
    stage(
        "targets",
        cli + ["targets", *common, "--allow-dirty"],
        study / "distributions/distributions.json",
    )
    baseline = json.loads(
        (args.baseline / "distributions/distributions.json").read_text()
    )
    corrected = json.loads((study / "distributions/distributions.json").read_text())
    if baseline["condition_ids"] != corrected["condition_ids"]:
        raise ValueError("Baseline and corrected training conditions differ.")
    for arm in baseline["arms"]:
        if baseline["arms"][arm] != corrected["arms"][arm]:
            raise ValueError(f"Baseline and corrected target vectors differ: {arm}.")
    print(
        "All arm vectors equal the baseline exactly.",
        flush=True,
    )
    stage(
        "calibration",
        cli + ["calibrate", *common, "--allow-dirty"],
        study / "sampling-calibration/summary.json",
    )
    for tier in ("smoke", "pilot"):
        stage(
            f"{tier}-training",
            cli + ["train", *common, "--tier", tier, "--workers", "3", "--allow-dirty"],
            study / "policies" / tier / "training.json",
        )
        if tier == "pilot":
            if args.focus:
                stage(
                    "replay-verification",
                    [
                        sys.executable,
                        "scripts/analyze_ppo_boundary_long_run.py",
                        "--study",
                        str(study),
                        "--baseline",
                        str(args.baseline),
                        "--check-replay",
                    ],
                    study / "long-run/replay-verification.json",
                )
            if args.long:
                stage(
                    "prefix-verification",
                    [
                        sys.executable,
                        "scripts/analyze_ppo_boundary_long_run.py",
                        "--study",
                        str(study),
                        "--baseline",
                        str(args.baseline),
                        "--check-prefix",
                    ],
                    study / "long-run/prefix-verification.json",
                )
            stage(
                "exposure-audit",
                [
                    sys.executable,
                    "scripts/analyze_ppo_boundary_sampling.py",
                    "--study",
                    str(study),
                ],
                study / "sampling-diagnosis/summary.json",
            )
        evaluate(tier)
    evaluate("pilot-snapshots")
    stage(
        "figures",
        [
            sys.executable,
            "scripts/plot_ppo_boundary_shift.py",
            "--study",
            str(study),
            *(
                []
                if args.long or args.focus
                else ["--compare-study", str(args.baseline)]
            ),
        ],
        study
        / (
            "figures/pilot/coverage-shift-maps.pdf"
            if args.long or args.focus
            else "figures/pilot/coverage-shift-comparison.pdf"
        ),
    )
    if args.long or args.focus:
        stage(
            "long-run-outcomes",
            [
                sys.executable,
                "scripts/analyze_ppo_boundary_long_run.py",
                "--study",
                str(study),
                "--baseline",
                str(args.baseline),
            ],
            study / "long-run/summary.json",
        )
    print(f"Pilot and figures complete: {study}", flush=True)


if __name__ == "__main__":
    main()
