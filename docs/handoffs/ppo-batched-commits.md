# Batched PPO experiment commits

The user requested committing the completed work in focused batches. The code
is unchanged from the source that passed 390 tests before the adaptive pilot;
each staged batch passes `git diff --cached --check`.

Batches:

1. `9a3bc89` — duration-corrected transition sampling, calibration, continuation
   provenance, optional tier arm selection, v4 config, and sampling tests.
2. `4ef436b` — exposure audits, interval/per-seed coverage plots, long/focused
   replay configs and analysis, runner, and diagnostic/replay tests.
3. Adaptive discovery/retarget/train implementation, v5 config, artifact-only
   analysis/plots, and adaptive integration tests.
4. This note, experiment reports, main-plan results, and completed-task handoffs.

The final documentation batch includes this note; its own commit hash is found
in `git log`. All four batches are on the existing `ppo-boundary-s4-s5` branch.
No push was requested. Raw experiments and figures remain local under the
existing ignored `artifacts/` directory; their launch provenance is historical
and should not be rewritten to identify these later commits.

Unrelated `ese6180/discussions/a.md` is intentionally left untracked and untouched.
Earlier handoffs describe their historical run-time state (including uncommitted
source at launch); these batch commits occur after those completed experiments.

Usage guard: 17% five-hour /87% weekly at the fresh startup check on
2026-11-04 23:05:25 UTC, status ok. No subagents or background jobs were started.
After the documentation commit, verify `git status --short` contains only the
unrelated file and inspect the four commit summaries. No experiment rerun,
implementation change, merge, or push is part of this request.
