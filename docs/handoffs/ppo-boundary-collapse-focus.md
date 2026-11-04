# Focused coverage maps, 102,400–256,000 transitions

STATUS: COMPLETE. Resumed after the user explicitly requested continuation.
Fresh live quota on 2026-11-04 at 21:37 UTC was 99% five-hour / 100% weekly,
status ok. The formerly running local batch finished successfully at 21:37:23
UTC, exit 0. No background jobs or subagents remain for this task.

## Scope and results

User requested actual intermediate coverage maps inside 102,400–256,000.
Study: `artifacts/experiments/cartpole-ppo-boundary-v4-focus`.
Config: `configs/experiments/cartpole_ppo_boundary_v4_focus.toml`.
Acceptance-boundary only, all three existing seeds, unchanged parent, reward,
PPO settings, duration correction, and frozen target. Snapshots at 102400,
128000,153600,179200,204800,230400; final at256000. All 273 reference conditions
and the same 64 role-4 episodes per checkpoint are retained.

Accepted counts by replicate 0/1/2:
102400:43/44/35;128000:50/56/42;153600:47/59/0;179200 and every later
checkpoint:0/0/0. First observed complete collapse is bracketed by
(128000,153600] for replicate2 and (153600,179200] for replicates0/1. Nominal
recovery falls from64/64 to0/64 in the same respective sampled intervals.
All three have zero recovery and survival throughout the grid at179200.
Tiny isolated recovery at230400 does not restore accepted coverage.

The workflow took893.997 seconds (14.9 minutes), all stages exit0. Completed
6006 pilot/snapshot jobs,384384 episodes. Parent plus21 learned checkpoints.
Discovery/calibration/smoke are additional stages, not included in those counts.

## Verification

- All28 raw training/reset/sampler columns through256000 exactly replay the
  original long run; endpoint policy/optimizer/controller match102400/256000.
- All1638 shared endpoint evaluation cells exactly reproduce counts/rates and
  trajectory-derived aggregates. Evaluation seed ranges match the original.
- All21 learned checkpoint seals verify; launch source and archive digest match;
  nominal parent model unchanged (sha256
  `37c9bac10919b2d6bdc833fd5a3a920460d4b80f09392ef7fb758e4777a856e8`).
- Before launch,23 relevant tests and lint passed. No implementation changed
  after launch. Post-run changes are reports/handoff only; numerical output
  and provenance remain intact.
- Machine evidence: study `long-run/replay-verification.json`,
  `long-run/integrity-checks.json`, `long-run/summary.json`,
  `long-run/verify_results.py`, `sampling-diagnosis/summary.json`,
  `run-timing.json`. Launch source archived under `protocol/`.

## Deliverables

Under the focused study's `figures/pilot/`, visually inspected PNGs:
`coverage-shift-maps.png`, `coverage-shift-replicates.png`, and
`long-run-learning-curves.png`. Also `coverage-shift-bands.png`; all have PDFs.

Results/report: `ese6180/discussions/2026-11-04-cartpole-ppo-collapse-focus.md`.
Main plan updated Section13.9 with results and report pointer:
`ese6180/discussions/2026-11-04-post-midterm-cartpole-ppo-plan.md`.

User's boundary-update question is answered: existing campaigns have one frozen
parent frontier; duration correction adapts reset laws only. Black outlines
are the parent reference region on273 cells, distinct from the143-cell discovery
support used to construct training targets. The proposed train/evaluate/refresh
frontier/train loop is an adaptive curriculum, not implemented/run by this task.
Exposure tracking errors coincide with collapse but do not establish causality.
No new adaptive experiment, checkpoint selection, retuning, or confirmatory
lock was performed. Any next experiment needs a concrete user request.

## Working tree and continuation

All changes remain uncommitted. No commits, pushes, or cloud operations.
Original v3/v4/v4-long results preserved. Preserve unrelated user file
`ese6180/discussions/a.md` and preexisting changes to the main plan.
Implementation files include the focused config, `--focus` runner option,
replay checker, per-seed plotting, generic budget handling, and config tests.
Earlier completed work is documented in `ppo-transition-balanced-pilot.md` and
`ppo-boundary-long-run.md` in this handoff directory.

No required task work remains. Return links to mean and per-seed maps plus
learning curves and the collapse brackets. Follow any later user request with a
fresh live quota check before substantive sustained work. No delegation was used.
