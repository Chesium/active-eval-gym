# Focused replay of the acceptance-set collapse

The user requested maps inside the 102,400–256,000 interval after the long-run
diagnostic. No checkpoints were saved there. This development-only follow-up
reconstructs that training trajectory with additional saved checkpoints.

All three original seeds are retained. The nominal parent, frozen acceptance
frontier, target shares, duration correction, calibration, rewards, PPO settings,
and action/reset seeds remain unchanged. Train to 256,000 and save checkpoints
at 102,400, 128,000, 153,600, 179,200, 204,800, and 230,400 transitions. Before
evaluation, verify all reconstructed raw training/reset/sampler columns through
256,000 against the original long run, plus policy/optimizer/controller state
at 102,400 and 256,000. This verifies that extra saves did not change learning.

Evaluate every saved checkpoint and the final on all 273 reference conditions,
using the same 64 role-4 nuisance resets and deterministic actions. Retain raw
trajectories and all seeds, including failures. The purpose is to locate the
collapse interval, not select a best checkpoint or retune the learner. Maps are
computed from actual frozen checkpoint evaluations, not visual interpolation.

Config: `configs/experiments/cartpole_ppo_boundary_v4_focus.toml`.
Study: `artifacts/experiments/cartpole-ppo-boundary-v4-focus`.

```sh
.venv/bin/python scripts/run_ppo_transition_balanced_pilot.py --focus
```

Launch estimate after setup checks: 15–25 minutes, mainly evaluation. Use three
training workers and 12 evaluation/analysis workers. Results will be appended.

## Completed results

The workflow finished on 4 November 2026 at 21:37 UTC (16:37 EST), after
14.9 minutes. All 6,006 pilot/snapshot checkpoint-condition jobs completed,
retaining 384,384 evaluation episodes. This includes the parent and 21 learned
checkpoints; discovery, calibration, and smoke runs are separate.

Accepted reference conditions, out of 273, using the pre-existing empirical
recovery-rate threshold of at least 0.9:

| New training transitions | Replicate 0 | Replicate 1 | Replicate 2 | Mean recovery over conditions and seeds |
| ---: | ---: | ---: | ---: | ---: |
| Parent | 28 | 28 | 28 | 0.12048 |
| 102,400 | 43 | 44 | 35 | 0.16745 |
| 128,000 | 50 | 56 | 42 | 0.20259 |
| 153,600 | 47 | 59 | 0 | 0.14438 |
| 179,200 | 0 | 0 | 0 | 0 |
| 204,800 | 0 | 0 | 0 | 0 |
| 230,400 | 0 | 0 | 0 | 0.000095 |
| 256,000 | 0 | 0 | 0 | 0 |

Coverage continues growing through 128,000 in all three seeds. At that checkpoint,
replicates 0 and 1 each lose only one parent-accepted condition; replicate 2
retains all 28. At 153,600, replicates 0 and 1 retain all 28 parent conditions
and accept 47 and 59 reference conditions, while replicate 2 has already failed
throughout the reference grid. Mean maps hide this difference between seeds.

The first observed complete collapse is bracketed by (128,000, 153,600] for
replicate 2 and (153,600, 179,200] for replicates 0 and 1. These bounds describe
adjacent saved checkpoints, not an exact change point or proof of monotonic
behavior between them. Nominal recovery changes from 64/64 at each seed's last
noncollapsed checkpoint to 0/64 at its first collapsed checkpoint. At 179,200,
all three checkpoints have zero recovery and zero 500-step survival across all
273 conditions. Every later sampled checkpoint through 256,000 has an empty
accepted set. The small recovery at 230,400 is five successful episodes in one
replicate across 17,472 episodes, not restored coverage.

## Figures

All figures also have PDF versions in the same directory.

- [Mean recovery and interval training exposure](../../artifacts/experiments/cartpole-ppo-boundary-v4-focus/figures/pilot/coverage-shift-maps.png).
- [Recovery maps for every seed](../../artifacts/experiments/cartpole-ppo-boundary-v4-focus/figures/pilot/coverage-shift-replicates.png).
- [Coverage, retention, exposure, and turnover curves](../../artifacts/experiments/cartpole-ppo-boundary-v4-focus/figures/pilot/long-run-learning-curves.png).
- [Coverage shifts by pole-length band](../../artifacts/experiments/cartpole-ppo-boundary-v4-focus/figures/pilot/coverage-shift-bands.png).

## Verification and interpretation

Replay verification passes for all three seeds: all 28 raw training/reset/sampler
columns through 256,000 match the original long run. Weights, optimizer state,
and duration-controller state match at both 102,400 and 256,000. Calibration is
identical. All 1,638 endpoint reference evaluation cells reproduce the original
long run exactly in episode counts, recovery/survival counts and rates, and raw
trajectory-derived aggregates. All 21 learned checkpoint seals, launch source
hashes, the source archive digest, and the unchanged parent model verify. The
64 role-4 evaluation seeds match the original study. Relevant implementation
checks passed before launch (23 tests and lint); no training source was changed
after launch. The run uses archived uncommitted development source, not a
confirmatory lock.

Machine-readable evidence is retained under the study's `long-run/` directory:
`replay-verification.json`, `integrity-checks.json`, `summary.json`, and the
post-run read-only verification script `verify_results.py`. Runtime stages and
exit codes are in `run-timing.json`; all stages returned zero.

The target transition vector remains frozen to the parent's discovery frontier.
The duration controller updates reset probabilities to pursue that target; it
does not reevaluate the acceptance boundary. The black outline in these maps is
the parent's accepted region on the reference grid, repeated for comparison.
The orange panels show actual interval exposure, so their colors can change
while the black outline remains fixed. The training frontier comes from the
143-condition discovery grid, whereas the outline comes from the 273-condition
reference grid; they are related but not the same evaluation support.

Duration correction also has transient tracking error as policy behavior
changes: mean lower/upper/side frontier shares are 18.75%/19.28%/20.94% during
102,400–128,000, 13.34%/30.32%/18.47% during 128,000–153,600, and
9.92%/28.50%/16.53% during 153,600–179,200, against 18.41% each intended.
These exposure changes coincide with collapse, but this observation does not
establish whether they cause, amplify, or follow the learning instability.
The maps do not identify a causal mechanism for collapse.

The proposed train → discovery evaluation → refresh frontier/targets → train
loop is an adaptive curriculum experiment. It has not been implemented or run
in these campaigns. Its targeting evaluations should be separate from the held-out
reference evaluations used to measure coverage. This replay preserves the
original fixed-frontier experiment and does not select a checkpoint, retune PPO,
or claim successful convergence from similarity between empty sets.
