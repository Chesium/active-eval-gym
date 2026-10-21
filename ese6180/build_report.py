#!/usr/bin/env python3
"""Build the ESE6180 report with pdflatex and BibTeX (Python standard library only).

Usage from the repository root:
    python3 ese6180/build_report.py
    python3 ese6180/build_report.py --output-dir /tmp/ese6180-report

The script can run from any working directory. By default, the PDF and build
logs go in Evaluation_Guided_Domain_Randomization/build/ beside this script.
Requires pdflatex, bibtex, and the report's JMLR/PMLR LaTeX dependencies.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


REPORT_DIR = Path(__file__).resolve().parent / "Evaluation_Guided_Domain_Randomization"
REPORT_NAME = "main_report"


def run_step(
    command: list[str],
    *,
    description: str,
    cwd: Path,
    log_path: Path,
    env: dict[str, str],
) -> None:
    print(description, flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    if result.returncode:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
        raise RuntimeError(
            f"{description} failed (exit {result.returncode}).\n{tail}\n"
            f"Full output: {log_path}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORT_DIR / "build",
        help="Directory for the PDF and auxiliary files (default: report's build/).",
    )
    args = parser.parse_args()
    missing = [tool for tool in ("pdflatex", "bibtex") if shutil.which(tool) is None]
    if missing:
        parser.exit(1, f"Missing required executables on PATH: {', '.join(missing)}\n")

    try:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        # Run BibTeX inside the output directory: TeX Live disallows absolute
        # output filenames by default. Keep its default bibliography search too.
        env["BIBINPUTS"] = str(REPORT_DIR) + os.pathsep + env.get("BIBINPUTS", "")
        latex_command = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            f"-output-directory={output_dir}",
            f"{REPORT_NAME}.tex",
        ]
        for pass_number in range(1, 4):
            run_step(
                latex_command,
                description=f"pdflatex ({pass_number}/3)",
                cwd=REPORT_DIR,
                log_path=output_dir / f"pdflatex-{pass_number}.log",
                env=env,
            )
            if pass_number == 1:
                run_step(
                    ["bibtex", REPORT_NAME],
                    description="BibTeX",
                    cwd=output_dir,
                    log_path=output_dir / "bibtex.log",
                    env=env,
                )
        pdf_path = output_dir / f"{REPORT_NAME}.pdf"
        if not pdf_path.is_file():
            raise RuntimeError(f"Build completed without the expected PDF: {pdf_path}")
    except (OSError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1

    print(f"Built: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
