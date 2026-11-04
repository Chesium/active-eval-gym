# Adaptive acceptance-frontier development pilot

Completed: refreshing preserves nonzero coverage through 256,000 transitions,
but all three seeds eventually lose the entire parent-accepted region. This is
coverage drift toward shorter poles, not sustained outward expansion.

## Decisions made before collection

The researcher requested evaluate → refresh boundary → retrain and chose:

- 10 rounds of 25,600 additional PPO transitions, up to 256,000 per seed.
- The original nominal parent and all three existing training seeds (0, 1, 2).
- Adaptive only, compared with the existing frozen-frontier long run; no new
  matched control or PPO stabilization changes.
- Stop a seed if discovery finds no accepted condition, recording the remaining
  transition budget as unrun. Do not fall back to uniform or hold a stale frontier.

If every discovery condition becomes accepted, the frontier is also empty; stop
with a distinct `all-conditions-accepted` reason. Neither stop is an infrastructure
failure. Retain the checkpoint and evaluate it on the reference grid. Unrun rounds
are missing by design; do not impute zero coverage or carry the last value forward.

Config: `configs/experiments/cartpole_ppo_boundary_v5_adaptive.toml`.
Study: `artifacts/experiments/cartpole-ppo-boundary-v5-adaptive`.
Command:

```sh
.venv/bin/python -u scripts/run_ppo_adaptive_curriculum.py
```

## Protocol

The versioned adaptive wrapper uses the scientific settings in the v4-long
configuration. The original single-round protocol remains guarded and unchanged.
Rewards, PPO learning rate (3e-5), clipping, epochs, environment counts, condition
supports, recovery metric, empirical acceptance threshold (0.9), mixture weight,
and duration-controller settings match v4-long.

Evaluate the frozen parent on the 143-condition discovery grid, 32 deterministic
nuisance episodes per condition, using role-1 round-0 seeds. Each seed starts from
that same accepted set. The frontier comprises the unaccepted four-neighbours of
accepted conditions on the angle/length grid. Desired transition mass is 50%
uniform background plus 50% uniform frontier. At each completed 25,600-transition
round except the final budget, seal the new policy, evaluate it on all discovery
conditions, and recompute acceptance/frontier/targets for that seed's next round.
Discovery reset seeds are fresh per round using the existing outer-round seed
namespace; the same round-specific nuisance list is paired across conditions
and seeds. No reference results enter targeting.

Each seed keeps one live model, optimizer, training environment, action RNG,
condition/reset streams, and duration controller across rounds. Training episodes
already in progress retain their conditions. Target replacement is staged at a
round boundary and applied at the next scheduled duration-law refresh. The
25,600-transition cadence aligns with its existing 2,560-transition refresh
schedule. EMA length estimates and completion counts are preserved. Initial
stochastic length calibration is reused from the matching frozen-parent v4-long
record with all raw shard hashes verified; no extra optimizer updates are used.

Discovery collection and analysis use separate spawned processes. Runtime checks
require unchanged live Python/NumPy/Torch RNG and policy weights across discovery
work. Evaluation raw trajectories, seeds, metric versions, model identities, and
per-episode metrics remain separate from training data. Each checkpoint includes
its raw training prefix, target source/hash, policy/optimizer hashes, duration
state, and an evaluation seal. Every target file binds the discovery metric
summary that selected it. Interrupted seed jobs replay from the original parent
and verify saved weights, optimizer, controller, and every raw column before
reusing a completed round.

Only after all seed training has completed or stopped, collect reference
measurements for the parent and every completed round. Use the existing 273-cell
grid, 64 role-4 resets per condition, deterministic actions, and all seeds. No
checkpoint selection or retuning is performed. Derived figures show recovery,
changing discovery acceptance outlines, desired transition shares, observed
interval exposure, and gains/losses relative to the parent.

## Comparison and resource forecast

Use the original v4-long data and its exactly verified dense replay v4-focus for
historical comparison, not an independent new control. The first round should
reproduce v4's 25,600-transition checkpoint for each seed; check policy, optimizer,
raw common columns, and controller state before interpreting later differences.
Later differences reflect the adaptive procedure, including fresh discovery
noise. This development study cannot establish a confirmatory causal ranking.

Maximum continuation training: 768,000 transitions. Discovery: parent plus nine
refreshes for each of three seeds, 4,004 jobs / 128,128 episodes. Reference:
parent plus 30 trained checkpoints, 8,463 jobs / 541,632 episodes. Stops reduce
these realized counts; report both planned and completed. Calibration is reused
and excluded from the additional transition budget.

At 500 steps per episode and the previous observed compression rate, raw
collection could approach 29 GB, plus training prefixes and metadata. A twofold
margin fits the 116 GiB free before launch. Initial runtime estimate after setup:
35–60 minutes, mostly reference evaluations; revise from actual throughput.

Implementation verification and observed results will be appended after the run.

## Launch checks and training completion

All 390 tests passed in 68.21 seconds; lint and whitespace checks passed. The
adaptive integration test performs real spawned discovery collection, verifies
training RNG isolation, and exactly replays completed rounds. Launch source is
archived with hashes under `protocol/`. This is uncommitted development source;
no confirmatory lock, commit, push, or cloud resources are involved.

The workflow launched on 4 November 2026 at approximately 16:55 EST (21:55 UTC).
All three seeds completed all ten rounds, 256,000 transitions each, without an
empty-frontier stop. Total continuation transitions: 768,000; unrun budget: zero.
All 30 learned checkpoint seals and raw training prefixes verify. For every
round, the target reconstructs from its bound discovery result, and the captured
reset probabilities equal the recorded target divided by online duration
estimates and normalized at each scheduled refresh. Parent/calibration hashes
and launch source/archive hashes verify.

The first 25,600 transitions match the original v4 run exactly for each seed:
all 28 original raw columns, weights, optimizer state, and duration-controller
state. Machine checks are saved in `analysis/first-round-verification.json` and
`analysis/training-integrity.json`; `verify_adaptive_results.py` retains the
post-launch artifact-only check code. Reference evaluation started only after
the immutable `training-complete.json` barrier. Final coverage results are reported below.

## Completed reference results

The workflow completed all 10 rounds for all three seeds, without an empty-set
stop, in 2,630.1 seconds (43.8 minutes), within the 35–60 minute estimate. Plotting
and final read-only checks followed. Storage is approximately 9.5 GiB. All 4,004
discovery jobs / 128,128 episodes and 8,463 reference jobs / 541,632 episodes are
retained: 12,467 jobs and 669,760 evaluation episodes in total. No collection
failures or retries occurred. No additional experiment or mitigation was run.

Accepted conditions out of 273, using the unchanged empirical recovery threshold:

| Additional transitions | Seed 0 | Seed 1 | Seed 2 | Parent conditions lost, seeds 0 / 1 / 2 |
| ---: | ---: | ---: | ---: | :--- |
| Parent | 28 | 28 | 28 | 0 / 0 / 0 |
| 25,600 | 30 | 27 | 29 | 0 / 1 / 0 |
| 51,200 | 29 | 31 | 29 | 0 / 2 / 1 |
| 76,800 | 39 | 38 | 36 | 1 / 1 / 1 |
| 102,400 | 51 | 54 | 45 | 1 / 4 / 1 |
| 128,000 | 54 | 50 | 56 | 5 / 9 / 3 |
| 153,600 | 58 | 70 | 56 | 15 / 6 / 6 |
| 179,200 | 60 | 72 | 65 | 25 / 6 / 11 |
| 204,800 | 68 | 80 | 71 | 28 / 13 / 19 |
| 230,400 | 59 | 75 | 82 | 28 / 28 / 28 |
| 256,000 | 61 | 68 | 58 | 28 / 28 / 28 |

At 102,400 transitions, mean accepted count is 50.0 versus 40.7 in the historical
fixed-frontier run, with parent losses 1/4/1 versus 3/4/1. At 256,000 the adaptive
counts are 61/68/58, mean 62.3 (22.83% of reference cells), versus 0/0/0 for the
historical frozen-frontier run. Mean recovery across all cells/seeds is 0.24931,
versus the parent's 0.12048 and the historical endpoint's zero. Those coverage
improvements do not imply retention.

Every adaptive final loses all 28 parent-accepted reference conditions. Recovery
throughout that old region is exactly 0/1,792 episodes for each seed, not merely
below the acceptance threshold. Nominal recovery falls from 64/64 to 0/64 between
179,200 and 204,800 in seed 0, and between 204,800 and 230,400 in seeds 1 and 2.
All three have lost the entire parent region by 230,400. The empty-set stop is
never triggered because new short-pole conditions remain accepted even while the
old region disappears.

Final accepted half-lengths concentrate at 0.04, 0.05, 0.075 and 0.10 m. Seed 1
also accepts four conditions at 0.20 m. None of the three finals accepts a
condition at half-length 0.30 m or greater. Thus the refreshed outline moves
downward with the accepted set instead of enforcing expansion around its union
with previous accepted regions. The rule was deliberately based on the current
set only; it contains no explicit retention constraint.

## Target movement and transition exposure

The target is recomputed correctly every round and is uniform across current
frontier cells before mixing with the 50% background. This does not guarantee
that realized transitions match it while the policy and episode lengths change.
Across seeds, desired mass on half-lengths at most 0.30 m grows from 43.59% in
round 1 to 71.74% in round 10. Actual mass grows from 45.00% to 89.62%.

The original 15 discovery-accepted cells receive 5.24% intended and 5.21% actual
mass in round 1. In round 10 they receive 9.78% intended but only 0.86% actual.
Mean per-seed total-variation distance between intended and actual cell exposure
rises from 0.255 in round 1 to 0.529 in round 10. These are interval quantities;
short-window sampling noise also contributes. The mismatch is visible in the
third row of each map compared with its second row.

Consequently, two observed effects accompany the drift: the selected frontier
moves toward shorter poles, and the online duration correction increasingly
misses its changing target. Keeping EMA estimates across rounds was a declared
implementation choice. Stale estimates for rarely sampled conditions during
rapid policy changes are a plausible mechanism, but these descriptive data do
not isolate whether the exposure mismatch causes, amplifies, or follows policy
forgetting. The experiment does not establish that frontier refresh alone solves
long-horizon stability or yields uniform outward expansion.

The post-run, artifact-only analysis is retained as `analyze_exposure.py` and
`analysis/exposure-and-retention.json`. This analysis did not alter training,
select a checkpoint, or trigger additional runs.

## Figures and reproducibility

- [Seed 0: coverage, refreshed targets, and observed exposure](../../artifacts/experiments/cartpole-ppo-boundary-v5-adaptive/figures/pilot/coverage-shift-seed-000.png).
- [Seed 1 maps](../../artifacts/experiments/cartpole-ppo-boundary-v5-adaptive/figures/pilot/coverage-shift-seed-001.png).
- [Seed 2 maps](../../artifacts/experiments/cartpole-ppo-boundary-v5-adaptive/figures/pilot/coverage-shift-seed-002.png).
- [Adaptive versus historical frozen-frontier curves](../../artifacts/experiments/cartpole-ppo-boundary-v5-adaptive/figures/pilot/adaptive-vs-frozen-learning-curves.png).

PDF versions are in the same directory. Black outlines in the orange rows show
the accepted discovery set used to choose that training round's frontier; they
now change across rounds. The gray reference outline stays fixed to the parent
for comparison. Each blue column evaluates the checkpoint after the displayed
round; its orange rows show the target chosen before that round and transitions
during it. Lengths are rank spaced, so displayed areas are not physical areas.
Each seed's exposure color scale covers both its desired and observed shares;
scales can differ across seed figures. All four PNGs were visually inspected.

Full post-run integrity checks passed: all 30 checkpoint seals, raw content/file
digests, target provenance and independent frontier reconstruction, reset-law
schedule replay, unchanged parent/calibration, and launch source/archive hashes.
The parent and all three first-round reference evaluations exactly reproduce
v4 in all 1,092 shared condition cells, including counts and trajectory-derived
aggregates. Evidence is in `analysis/integrity-checks.json` and the saved
verification script. No numerical source changed after launch. Existing v3/v4/
v4-long/v4-focus results and unrelated working-tree files are preserved.

Interpretation is limited to these three development seeds and this 256,000-step
horizon, compared with historical controls as requested. Larger coverage at the
endpoint is compatible with complete forgetting of the initial task. A future
retention or duration-tracking intervention requires its own declared comparison;
this run does not test one.
