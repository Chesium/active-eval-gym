# Development pilot: inverse-episode-length correction

This protocol is written before v4 outcomes are collected. The researcher
authorized implementation and runs after the v3 sampling diagnosis. This is a
development comparison, using pilot seeds only; no confirmatory lock is created.

## Question and matched comparison

Does approximately balancing the **transition** exposure specified by each
frozen curriculum mixture reduce the short-pole shift observed in v3?

The baseline is the existing v3 uniform-episode pilot. v4 starts from the same
frozen nominal parent, uses the same three arms, training seeds 300000–300002,
8 workers, 32 steps per rollout, LR 0.00003, 20 epochs, +1 survival reward,
102,400 continuation transitions, and snapshots at 25,600/51,200/76,800.
All checkpoints are evaluated on all 273 reference conditions, with the same
64 role-4 resets and deterministic actions. No seed or checkpoint is selected
based on its outcome. The v3 data remain intact.

v4's serialized arm vectors must equal v3's numerically, but their meaning is
now desired transition shares `t(z)`, rather than reset probabilities. The
frontier remains the parent's fixed 15-cell discovery frontier. No new recovery
reward, accepted-region retention term, or frontier refresh is introduced.

## Reset controller

For every condition, initialize an episode-length estimate from 8 fresh
stochastic-action episodes of the frozen parent. Calibration uses role-6 reset
seeds, with episode index `condition_index * 8 + episode_index`, and a separate
action seed `730000 + condition_index`. Its raw trajectories, action/reset seeds,
parent identity, package provenance, lengths, and cost are retained. Calibration
performs zero optimizer updates and is outside continuation transition budgets.
All arms and replicates use the same calibration.

At reset, select the condition with:

```text
q_reset(z) = normalize(t(z) / max(estimated_length(z), 1))
```

One controller per child shares estimates across all eight DummyVecEnv workers.
After each completed episode of length `L`, update that condition:

```text
estimated_length <- 0.9 * estimated_length + 0.1 * L
```

Only complete episodes update the estimates. A true environment time-limit
truncation is a completed 500-step episode. A PPO rollout boundary, checkpoint,
or end-of-training partial episode is not an episode completion. Reset
probabilities refresh every 10 rollouts (2,560 transitions), using only preceding
training completions. Active episodes continue with their original condition.
The refreshed probabilities apply at subsequent natural resets.

This offers approximate long-run balance, not exact per-buffer quotas. The full
reset probability and length-estimate schedules are stored, along with each
reset's selected probability and schedule revision. Evaluation never updates the
controller, changes the policy, or supplies observations to selection.

## Verification and outcomes

Before pilot runs: legacy regression checks; synthetic renewal-count balance;
natural-completion-only updates; stochastic calibration provenance and resume;
identical adaptive weights/raw data with and without snapshot chunking; smoke
training and evaluation; verified frozen parent.

Report achieved transition shares versus intended shares, including total
variation and lower/upper/side frontier allocations. Validate sampling schedules
by replaying the EMA from raw completed training episodes. Report all three
replicates' accepted counts, gains/losses relative to the parent, mean recovery,
survival, nominal recovery, and trajectories across all checkpoints. Compare
with v3 using coverage-shift maps and gains/losses, without ranking three-replicate
arms as confirmatory evidence.

Increased exposure balance and reduced acceptance drift are separate outcomes.
Even if transition balancing succeeds, shared-policy tradeoffs, survival versus
recovery reward, and lack of an interior-retention objective can still produce
regressions. A negative or mixed result is retained without tuning this protocol
against its evaluation results.

## Execution and timing

Config: `configs/experiments/cartpole_ppo_boundary_v4.toml`.
Study: `artifacts/experiments/cartpole-ppo-boundary-v4`.
Use `--allow-dirty` to label the present uncommitted implementation as development
data and retain its source hashes. No commit or confirmatory run is needed.

Initial forecast: 20–30 minutes implementation/testing, then 30–45 minutes for
calibration, smoke, nine pilot children, evaluation of finals and 27 snapshots,
analysis and plots. Evaluate/analyze with 12 local processes; train with 3.
Machine at startup: 24 CPUs, 30 GiB RAM, about 130 GiB free disk. Record measured
stage times and revise the forecast from the smoke run.

All planned development runs, metrics, and figures are complete. Results follow.

### Completed sampling and training checks

Discovery reproduced v3's recovery and survival counts at every one of the 143
conditions, and all three serialized arm vectors equal v3 exactly. Calibration
collected 1,144 stochastic episodes / 157,019 transitions in 24.1 seconds, with
zero optimizer updates. Smoke collection and analysis passed. All nine pilot
children and 27 snapshots trained and sealed in 45.5 seconds.

Pooling the three boundary-arm children gives:

| Frontier region | Desired transition share | v3 actual | v4 actual |
|---|---:|---:|---:|
| Lower, 5 cells | 18.41% | 50.51% | 18.38% |
| Upper, 5 cells | 18.41% | 4.30% | 17.53% |
| Angle sides, 5 cells | 18.41% | 13.74% | 17.23% |

Mean total-variation distance between actual transition shares and the fixed arm
vector (computed per replicate before averaging) fell in every arm:

| Arm | v3 | v4 |
|---|---:|---:|
| Background | 0.520 | 0.159 |
| Acceptance boundary | 0.542 | 0.147 |
| Intermediate | 0.425 | 0.134 |

This establishes substantially better exposure balance, independent of the
pending recovery outcome. Finite budgets and changing lengths still prevent
exact per-condition allocation. All nine raw transition content/file digests,
all 36 checkpoint seals, every recorded reset seed, and all 27 post-update
snapshot hashes verify. Replaying the length EMA from raw completed episodes
reproduces every saved reset-probability schedule exactly. The source archive
matches launch hashes, all children start from identical policy weights, and
the parent file is unchanged.

### Primary final-budget outcomes

The primary campaign completed all 2,730 condition jobs and 174,720 episodes,
with no failed jobs. Collection took 477.7 seconds and analysis 39.9 seconds.
Every final still recovers 64/64 at nominal. The parent accepts 28/273 conditions.

| Arm | Replicate | v3 accepted | v4 accepted | v3 gained | v4 gained | v3 lost | v4 lost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Background | 0 | 33 | 43 | 26 | 29 | 21 | 14 |
| Background | 1 | 34 | 40 | 26 | 27 | 20 | 15 |
| Background | 2 | 43 | 50 | 29 | 31 | 14 | 9 |
| Acceptance boundary | 0 | 42 | 43 | 27 | 18 | 13 | 3 |
| Acceptance boundary | 1 | 36 | 44 | 26 | 20 | 18 | 4 |
| Acceptance boundary | 2 | 43 | 35 | 28 | 8 | 13 | 1 |
| Intermediate | 0 | 42 | 49 | 28 | 30 | 14 | 9 |
| Intermediate | 1 | 36 | 49 | 26 | 31 | 18 | 10 |
| Intermediate | 2 | 38 | 50 | 27 | 31 | 17 | 9 |

In the boundary arm, mean retention rises from 13.3 to 25.3 of the parent's
28 accepted cells. Mean gains fall from 27.0 to 15.3, and mean total accepted
counts remain close: 40.3 versus 40.7. Thus coverage alone would conceal the
main change: much less replacement of old competence. Background and
Intermediate retain more parent cells in every paired replicate as well.

Balancing does not produce expansion on every side. At the five upper-frontier
conditions, recovery remains 0/64 in every corrected boundary replicate, despite
their increased training exposure. Across the 15 side-frontier replicate-cells,
three become accepted in the corrected boundary arm, versus none in v3; 13/15
lower-frontier replicate-cells are accepted versus 15/15 in v3. Every corrected
final still loses at least one parent-accepted condition, so strict set inclusion
has not been achieved.

These paired development runs support exposure imbalance as a contributor to
forgetting. They also show that equalizing exposure alone does not solve upper
frontier expansion. No conclusion about the best curriculum arm is claimed
from three replicates.

### Complete checkpoint comparison

All 7,371 snapshot jobs and 471,744 episodes completed without failed jobs.
Collection took 1,218.2 seconds and metric recomputation 94.8 seconds. Together
with the primary campaign, this gives all 10,101 planned checkpoint-condition
cells and 646,464 evaluation episodes. All 36 corrected checkpoints recover
64/64 at nominal; the smallest accepted count at any checkpoint is 27/273.

Mean accepted counts across the three paired replicates are below. Each entry
is **v3 / v4**. The final column is the boundary arm's mean losses relative to
the parent's accepted set, also v3 / v4.

| New transitions | Background accepted | Boundary accepted | Intermediate accepted | Boundary parent-set losses |
|---|---:|---:|---:|---:|
| 25,600 | 33.0 / 29.3 | 35.7 / 28.7 | 33.7 / 29.7 | 3.7 / 0.3 |
| 51,200 | 41.7 / 33.3 | 38.7 / 30.7 | 39.3 / 32.7 | 6.7 / 1.7 |
| 76,800 | 44.7 / 43.0 | 42.0 / 32.7 | 42.0 / 42.0 | 9.0 / 1.0 |
| 102,400 | 36.7 / 44.3 | 40.3 / 40.7 | 38.7 / 49.3 | 14.7 / 2.7 |

The correction delays the early short-pole gains and preserves the old set much
better, especially in the boundary arm. It is not simply a uniformly larger
accepted set at each checkpoint. All snapshots remain diagnostic; the final
fixed budget is still the primary comparison.

The complete run took **1,955.9 seconds (32.6 minutes)** from prepare through
figures, within the 30–45 minute launch estimate. Data occupy approximately
7.8 GiB. All raw data and checkpoints are retained, including regressions.
The final source hashes and original parent file still match the launch records.
The full suite passed 367 tests before the last two guard additions; the final
123-test relevant rerun, including those guards, passed. Lint and whitespace
checks passed. No confirmatory data were collected, and v3 was not overwritten.

### Figures and machine-readable results

All figures below have both PNG and PDF versions:

- [Coverage shift: v3 versus v4 at every checkpoint](../../artifacts/experiments/cartpole-ppo-boundary-v4/figures/pilot/coverage-shift-comparison.png).
- [v4 recovery and realized training exposure](../../artifacts/experiments/cartpole-ppo-boundary-v4/figures/pilot/coverage-shift-maps.png).
- [v4 accepted counts by length band](../../artifacts/experiments/cartpole-ppo-boundary-v4/figures/pilot/coverage-shift-bands.png).
- [Compact final-budget comparison](../../artifacts/experiments/cartpole-ppo-boundary-v4/figures/pilot/final-coverage-comparison.png).

The study also holds `sampling-diagnosis/summary.json`,
`sampling-diagnosis/integrity-checks.json`, `sampler-comparison/summary.json`,
`run-timing.json`, stage logs, and the launch source archive/manifest under
`protocol/`. The comparison JSON verifies equal parent evaluation cells,
evaluation reset lists, and paired training seeds/budgets. To recompute its
descriptive outcome tables and compact map from saved data:

```sh
.venv/bin/python artifacts/experiments/cartpole-ppo-boundary-v4/sampler-comparison/reproduce.py
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/ppo-boundary-matplotlib .venv/bin/python \
  artifacts/experiments/cartpole-ppo-boundary-v4/sampler-comparison/final_maps.py
```

Reproduce or resume the complete development workflow from the repository root:

```sh
.venv/bin/python scripts/run_ppo_transition_balanced_pilot.py
```

The runner retains per-stage commands, UTC timestamps, elapsed time, and logs
inside the study directory. It skips completed immutable stages and stops if a
stage fails. Final figures include the v4 exposure maps and a v3/v4 comparison.
