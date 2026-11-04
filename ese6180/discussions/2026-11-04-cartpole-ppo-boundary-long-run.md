# Longer duration-corrected acceptance-boundary training

This development extension was requested after the v4 pilot. The protocol is
written before long-run evaluations. It studies one selected arm; selection is
based on pilot findings, and this is not a confirmatory comparison of arms.

## Fixed design

- Study: `cartpole-ppo-boundary-v4-long`.
- Arm: acceptance_boundary only; all three original pilot replicates 0, 1, 2.
- Total additional training from the original nominal parent: 1,024,000
  transitions per replicate, exactly 4,000 PPO rollouts / 80,000 optimizer steps.
- Frozen targets, 50:50 background/frontier mixture, survival reward, LR 0.00003,
  20 epochs, optimizer initialization, seeds, calibration, and inverse-duration
  correction remain as in v4. No frontier refresh or reward adjustment.
- Checkpoints: 102,400; 256,000; 512,000; 768,000; 896,000; 1,024,000.
- Evaluate every checkpoint on all 273 reference conditions and the same 64
  role-4 nuisance resets as v4, with deterministic actions. Retain complete raw
  training and evaluation trajectories. Never select a checkpoint or seed based
  on observed performance.

Rerunning from the same parent avoids resetting worker/environment/action RNG
state at the old endpoint. Verify the first 102,400 transitions, sampler schedule,
policy weights, and optimizer state exactly against the original v4 runs. Its
frozen calibration lengths must also reproduce. The later checkpoints continue
the same in-memory learner, workers, and duration controller.

## What to examine

Report accepted counts, gains/losses relative to the parent and to the 102,400
checkpoint, mean recovery/survival, nominal recovery, and accepted cells by pole
length for every replicate and checkpoint. Show coverage-shift maps, training
exposure, and learning curves. Inspect the three late checkpoints (768,000,
896,000, 1,024,000) for ranges, successive set turnover/Jaccard similarity, and
mean-recovery changes. A finite run can show a plateau, continued movement, or
collapse; it cannot establish asymptotic convergence.

The fixed final budget remains primary even if an earlier snapshot is better.
Preserve regressing and failed outcomes. Evaluate only after all policies have
completed training, so outcomes cannot feed back into training design.

## Execution

Config: `configs/experiments/cartpole_ppo_boundary_v4_long.toml`.
Use the existing runner's explicit long mode:

```sh
.venv/bin/python scripts/run_ppo_transition_balanced_pilot.py --long
```

Run as development data with recorded uncommitted source, package provenance,
raw hashes, stage logs/timings, and a launch source archive. Prior v4 results stay
unchanged. Estimated runtime at startup: 20–35 minutes after setup checks, with
about 3–6 minutes training and most time spent evaluating the six checkpoints.
Worst-case evaluation storage is about 14 GiB plus training/calibration data;
available disk is about 122 GiB. Use 3 training workers and 12 evaluation workers.

## Results: collapse rather than a larger acceptance set

All three policies completed 1,024,000 transitions and 80,000 optimizer steps.
All 18 checkpoints were sealed. The 102,400-transition prefix reproduces v4
exactly: policy/optimizer hashes, all raw trajectory/reset/sampling columns,
duration-controller state, calibration means, and the 819 reference evaluation
cells across three seeds. The longer run is a verified extension of the earlier
training trajectory, rather than a restart with fresh environment/RNG state.

Accepted conditions at recovery threshold 0.9, out of 273 reference cells:

| New transitions | Replicate 0 | Replicate 1 | Replicate 2 | Mean recovery probability |
|---|---:|---:|---:|---:|
| Parent | 28 | 28 | 28 | 0.1205 |
| 102,400 | 43 | 44 | 35 | 0.167449 |
| 256,000 | 0 | 0 | 0 | 0.000000 |
| 512,000 | 0 | 0 | 0 | 0.000038 |
| 768,000 | 0 | 0 | 0 | 0.000534 |
| 896,000 | 0 | 0 | 0 | 0.000210 |
| 1,024,000 | 0 | 0 | 0 | 0.002118 |

Collapse occurs between the saved 102,400 and 256,000 checkpoints. Every later
checkpoint has an empty accepted set in all three seeds. Nominal recovery falls
from 64/64 at 102,400 to 0/64 at 256,000 and stays at 0/64 thereafter. By the final
budget every run has lost all 28 parent-accepted conditions and all conditions
accepted at its own 102,400 checkpoint.

Final outcomes are not only a threshold effect:

| Replicate | Mean recovery | Mean survival | Mean episode length | Nominal recovery |
|---|---:|---:|---:|---:|
| 0 | 0.000458 | 0.000515 | 70.0 | 0/64 |
| 1 | 0.000000 | 0.000000 | 61.9 | 0/64 |
| 2 | 0.005895 | 0.007326 | 97.1 | 0/64 |

The cart-position limit is the most common final failure category in all three
runs. In a post hoc check of the raw stochastic training trajectories, none of
the completed episodes ending in blocks from 204,800 transitions onward survives
to the 500-step limit. At the parent's accepted **discovery** conditions, completed
training episodes shorten from roughly 391–433 steps in the first block to
roughly 84–98 in the last. These training statistics use the sampler's observed
episode mix, not uniform reference weights, and remain distinct from frozen
evaluation probabilities.

## Late-window interpretation

At 768,000, 896,000, and 1,024,000 transitions, each seed has accepted counts
`[0, 0, 0]` and zero accepted-set turnover. The Jaccard similarity is defined as
1 when both sets are empty; its high late value describes a persistently failed
acceptance set, not successful learning. Mean recovery remains close to zero,
with late-window ranges 0.000458, 0.001374, and 0.005895 across seeds.

Thus the observed late regime is sustained failure. This is evidence about
closed-loop behavior over the sampled window, not a proof that neural weights
or the learning process have mathematically converged.

## Duration correction remained active

Over the full run, lower/upper/side frontier transition shares are
18.18% / 19.83% / 19.01%, against 18.41% each intended. Mean per-seed total
variation between target and realized transition shares is 0.0685. The last
three intervals also remain near-balanced (percentages):

| Interval | Lower | Upper | Sides |
|---|---:|---:|---:|
| 512,000–768,000 | 19.7 | 19.3 | 19.2 |
| 768,000–896,000 | 18.9 | 19.1 | 19.6 |
| 896,000–1,024,000 | 20.0 | 18.9 | 18.6 |

Raw replay reproduces every adaptive sampling schedule exactly. The exposure
correction therefore addressed the earlier large lower-edge imbalance, while
continued PPO still lost competence. The short-budget retention improvement
does not establish long-horizon stability under constant LR 0.00003 and 20
epochs. This run does not isolate whether the remaining instability comes from
the continuation schedule, repeated optimization, the survival reward, shared
policy tradeoffs, or their interaction. Stabilizing longer continuation is a
separate next experiment; no mitigation was tuned or run against these outcomes.

## Artifacts and validation

- [Coverage-shift maps and training exposure](../../artifacts/experiments/cartpole-ppo-boundary-v4-long/figures/pilot/coverage-shift-maps.png).
- [Learning curves, gains/losses, and late set similarity](../../artifacts/experiments/cartpole-ppo-boundary-v4-long/figures/pilot/long-run-learning-curves.png).
- [Acceptance by pole length](../../artifacts/experiments/cartpole-ppo-boundary-v4-long/figures/pilot/coverage-shift-bands.png).

All plots also have PDF counterparts. Numerical results and checks are in
`long-run/summary.json`, `long-run/prefix-verification.json`,
`long-run/integrity-checks.json`, and `sampling-diagnosis/summary.json` within the
study. The additional training diagnostic and its reproduction script are
`long-run/training-behavior.json` and `long-run/training_behavior.py`.

All 5,187 planned checkpoint-condition jobs / 331,968 evaluation episodes
completed with no collection failures. Poor policies were retained and evaluated.
Training took 137.9 seconds; the full workflow took 791.7 seconds (13.2 minutes),
less than the 20–35 minute estimate because failed policies have short episodes.
Data occupy about 2.7 GiB. All 377 tests passed before launch; lint and whitespace
checks passed. The original parent and all training/evaluation code still match
the launch records. Two presentation-only fixes after the run (map spacing and
an empty-set annotation) are documented in `protocol/plot-revision.json`; all
numerical analysis functions match the launch version. No confirmatory data were
collected, and earlier v4 artifacts remain unchanged.
