# Long duration-corrected acceptance-boundary run

Status: COMPLETE. All authorized runs, checkpoint evaluations, analyses, figures,
and report updates are finished. No owned jobs remain running.

User requested 1,024,000 transitions for the promising v4 boundary arm, examining
what it settles toward. Worktree `/home/chesium/active-eval-gym`; preserve all
prior user/agent changes. No delegation is authorized.

Plan: new `cartpole_ppo_boundary_v4_long.toml`, study
`artifacts/experiments/cartpole-ppo-boundary-v4-long`. Only acceptance_boundary,
same three pilot seeds, original parent, fixed parent frontier, reward, learning
rate, PPO settings, stochastic calibration, and adaptive duration correction.
Checkpoints at 102400, 256000, 512000, 768000, 896000, 1024000. Rerun from the
parent, verify first 102400 transitions/weights/optimizer/sampling prefix against
v4, preserving exact environment/RNG continuity instead of restarting from a zip.

Implemented optional tier arm subsets (default remains all arms), dynamic
plotting budgets/arms, long mode on the existing runner, exact prefix checking,
and outcome/late-window learning-curve analysis. Reproduce or resume with
`.venv/bin/python scripts/run_ppo_transition_balanced_pilot.py --long`.

All implementation is complete, 377 tests pass, and lint/diff checks pass.
The `--long` workflow finished successfully in exec session 7285, launched
around 20:58 UTC. Full workflow: 791.7 s (13.2 minutes); about 2.7 GiB data.
Its launch source archive/manifest are saved under the study's `protocol/`.
All three long policies are trained (137.9 s) and all 18 checkpoints sealed.
Prefix verification passed for all arrays, policy/optimizer hashes, controller
state, and calibration. Exposure audit passed. All 1,092 primary and 4,095
snapshot jobs / 331,968 evaluation episodes completed. All six PNG/PDF artifacts
are generated and inspected. No outcomes were used to retune or stop training.
The original parent and all training/evaluation/numerical-analysis functions
still match launch. Two post-run presentation-only revisions are explicitly
recorded in `protocol/plot-revision.json`; their AST changes are confined to
`maps_figure` and `learning_figure`.
Primary final evaluation/metrics complete (127.0 / 13.8 s): all three million-
transition finals have ZERO accepted conditions and nominal recovery 0/64.
Mean recovery r0/r1/r2 is 0.0005 / 0 / 0.0059. User has been informed; do not
retune or omit these failures. Snapshot evaluation establishes that every seed
has zero accepted conditions at 256k, 512k, 768k, 896k and 1024k. Nominal recovery
is 0/64 at all those checkpoints. The late sets remain empty, with Jaccard 1 by
the empty-set convention, rather than showing successful convergence.
Frontier exposure is still near-balanced at 18.18%/19.83%/19.01%; mean target TV
0.0685. Raw training episodes independently cease surviving after 204,800 in
the recorded blocks. Full report and main plan Section 13.8 are updated:
`ese6180/discussions/2026-11-04-cartpole-ppo-boundary-long-run.md`.
Prior v4 results remain unchanged. New runs use allow-dirty and source hashes/archive; no
confirmatory lock or Git commit is required. The report must distinguish late
plateaus from sustained set turnover/collapse and avoid claiming mathematical
convergence from a finite run. Keep every seed and fixed-final-budget result.

Communicated estimate: 20–35 minutes after setup checks, including about 3–6
minutes training; evaluation at six checkpoints dominates. 122 GiB free disk,
24 CPU cores, 30 GiB RAM. Use 12 eval workers, 3 training workers.

Quota startup: 2026-11-04 20:48:06 UTC, five-hour 51%, weekly 91%, status ok.
Last pre-launch check: 20:58:22 UTC, five-hour 42%, weekly 89%, status ok.
Last live check: 21:02:31 UTC, five-hour 38%, weekly 88%, status ok.
Final live check: 21:16:15 UTC, five-hour 25%, weekly 87%, status ok.
All results remain development-only, and source changes are
uncommitted. No background work besides the owned workflow is authorized.
Check `/home/chesium/.codex/skills/codex-usage-guard/scripts/check_usage.py` every
5 minutes and before costly phases, preserving early documentation/stop policy.
