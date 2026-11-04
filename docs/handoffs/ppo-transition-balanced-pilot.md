# Transition-balanced PPO pilot

Status: COMPLETE. All authorized implementation, runs, comparisons, and plots
finished successfully. No jobs remain active and no required work is pending.

User authorized implementation, runs, runtime estimates, and coverage-shift maps.
Worktree: `/home/chesium/active-eval-gym`. Preserve unrelated user edits, especially
`ese6180/discussions/a.md` and the pre-existing plan/plot/diagnosis changes.

Design: new `cartpole_ppo_boundary_v4.toml`, same parent/arms/seeds/budgets as v3,
interpreting frozen mixture vectors as target transition shares. Reset law is
`normalize(target / estimated_episode_length)`. Initialize with 8 stochastic
frozen-parent episodes per discovery condition, role 6 resets and action seeds
730000+condition index. Retain calibration raw trajectories separately. Update
length EMA (alpha .1) on completed episodes, refresh reset law every 10 rollouts.
Share estimates across the eight DummyVecEnv workers per child. No target refresh.

Implementation in progress: optional sampling config, inverse-length controller,
calibration command, sampler integration, raw schedule capture. Old-path tests
caught a misplaced argument; corrected before any actual experiment data.
Legacy tests passed. Full suite: 367 passed; final relevant rerun: 123 passed,
including two additional guard tests. Ruff and diff checks passed.

Implementation, plot labels/comparison maps, and pre-run protocol are complete.
Runner `scripts/run_ppo_transition_balanced_pilot.py` finished with exit 0 in
exec session 67746. Started 2026-11-04 20:07:45 UTC. Discovery reproduces all 143 v3 recovery
and survival counts; frozen target vectors equal v3. Calibration: 1,144 episodes,
157,019 transitions, 24 s, zero optimizer updates. Smoke training/evaluation/
analysis passed. All 9 pilot children and 27 snapshots are sealed (45.5 s train).
Exposure audit passes all raw/schedule checks. Boundary lower/upper/side shares:
18.38%, 17.53%, 17.23%, against 18.41% each. v3: 50.51%, 4.30%, 13.74%.

Primary evaluation/metrics: 477.7/39.9 s. Snapshot evaluation/metrics:
1,218.2/94.8 s, all 7,371 snapshot jobs and 471,744 episodes complete.
Total run: 1,955.9 s (32.6 minutes), about 7.8 GiB retained data. Matched outcome
tables are generated in `sampler-comparison/summary.json`, with reproduction
script alongside. All PNG figures were inspected; all have PDF counterparts.
Runner logs/timings are in the study directory; provenance includes a source
archive of launch code. Final source hashes and parent file still match.

Primary results: boundary v4 finals lose 3,4,1 parent-accepted cells versus
13,18,13 in v3; they gain 18,20,8 versus 27,26,28. Mean final accepted count
40.67 versus 40.33. Other corrected arms also retain more on average. These are
exploratory three-replicate results, not proof of uniform expansion. A compact
final map is generated with the saved artifact script
`sampler-comparison/final_maps.py` inside the v4 study; full trajectory maps are
complete alongside the recovery/exposure maps and accepted-count band plots.
No training parameters were retuned against outcomes. All 36 corrected
checkpoints recover 64/64 at nominal; none has fewer than 27 accepted cells.

Full report: `ese6180/discussions/2026-11-04-cartpole-ppo-transition-balanced-pilot.md`.
Main plan now links the complete results in Section 13.7. Source/config changes
and all earlier user edits remain uncommitted; no Git commit or push was made.
All results are development-only. Use the existing v3 data for the baseline;
do not overwrite them or launch confirmatory. No Git commit is needed.

Initial forecast communicated: implementation/testing 20–30 minutes, then
calibration/train/evaluation/plots 30–45 minutes, using 12 evaluation workers.
24 CPUs, about 130 GiB free at startup. Actual experimental run: 32.6 minutes.

Quota last check: 2026-11-04 20:42:59 UTC, five-hour 53%, weekly 91%, both ok.
Check `/home/chesium/.codex/skills/codex-usage-guard/scripts/check_usage.py` at
least every 5 minutes and before costly phases. No subagents are authorized.
