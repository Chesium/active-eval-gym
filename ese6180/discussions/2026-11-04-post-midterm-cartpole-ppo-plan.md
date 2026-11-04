# Post-midterm plan: boundary-targeted PPO training on CartPole

Date: 4 November 2026.

Status: an implementation and experiment handoff, not a completed experiment.
The priority is a controlled CartPole/PPO comparison on **initial pole angle ×
pole half-length**, using this repository's recovery task. The linear analysis
continues as a smaller supporting track. No training or remote compute is
launched by this document. Section 13 orders the work into gated stages `S0`–`S9`.

Repository baseline inspected: `ceb5ccf3831366d866697f7e1a49ee19a271dc39`.
The executing agent should record its actual implementation commit and resolved
configuration before collecting confirmatory results.

## 1. Research question and order of work

> Starting from the same trained PPO checkpoint, does emphasizing conditions
> near its recovery-acceptance boundary produce more acceptable conditions than
> broad training with the same additional training budget?

The core sequence is:

```text
frozen starting PPO
    -> fresh discovery evaluations over angle × half-length
    -> construct background / boundary / intermediate-difficulty distributions
    -> clone the identical starting checkpoint for each comparison arm
    -> collect fresh PPO training trajectories under each fixed distribution
    -> freeze all final checkpoints, including regressing ones
    -> evaluate on an independent, fixed reference grid
    -> compare recovery coverage, gained conditions, and lost conditions
```

Run one outer training phase first. Only after this works should the agent add
several rounds with refreshed targeting. This isolates the curriculum effect
before introducing feedback between curriculum estimates and successive policies.

The original broad-training interpretation motivates the primary comparison.
The available parent was trained nominally: this first experiment compares broad
and targeted **additional** training from that parent. Testing a parent already
trained across the broad domain is a later, separately budgeted extension.
The existing nominal checkpoint also supplies the starting point for a later
nominal-seeded curriculum. Neither experiment requires a Lyapunov certificate for
the nonlinear, two-action CartPole policy. The current certificate `P` and its
verified margins remain part of the **linear theoretical track**.

Success means a reproducible answer, including an unfavorable or mixed result.
An increase in training return, an improved boundary estimate, and a larger
actual recovery-acceptance set are different outcomes.

## 2. Existing assets to reuse

Paths below are relative to the repository root.

| Asset | Use |
|---|---|
| `README.md`, “Adaptive CartPole failure-boundary study” | Original two-dimensional recovery experiment and commands |
| `docs/findings.md`, “CartPole recovery failure-boundary study” | Existing frozen-policy findings, including survival without recovery and signed asymmetry |
| `configs/eval/cartpole_failure_boundary_pilot_v1.toml` | Original physical domain, coarse grid, recovery metric, and source policy |
| `configs/policies/cartpole_ppo_nominal_v1.toml` | Starting PPO architecture and training hyperparameters |
| `configs/envs/cartpole_v1_nominal.toml` | Base physics, reset noise, horizon, and cart-position limit |
| `src/active_eval_gym/envs/perturbations.py` | Structured recovery perturbation and exact reset-angle semantics |
| `src/active_eval_gym/rollout.py`, `serialization.py`, `metrics.py` | Frozen evaluation, complete raw trajectories, integrity hashes, and recovery metrics |
| `src/active_eval_gym/sweeps.py` | Fixed-grid condition expansion and evaluation/analysis machinery |
| `src/active_eval_gym/policies/sb3.py`, `policies/artifacts.py` | PPO loading, original training, and model provenance |
| `src/active_eval_gym/linear_updates.py`, `linear_study.py` | Supporting linear mechanism experiments |

The transferable parent artifact is the complete directory
`artifacts/policies/cartpole_ppo_nominal_v1/`, containing:

```text
model.zip
manifest.json
freeze.json
training-summary.json
```

Expected parent **model** SHA-256:

```text
37c9bac10919b2d6bdc833fd5a3a920460d4b80f09392ef7fb758e4777a856e8
```

This directory is ignored by Git and must be transferred separately to the other
machine. The model is small (about 142 KB); the historical multi-gigabyte rollout
directories are not required. Verify both model and manifest/freeze bindings
with the existing artifact loader. If the parent is absent or has another hash,
report that discrepancy; do not silently substitute a newly trained checkpoint.
A new-parent experiment can be separately versioned.

Historical data motivate the design but do not supply independent test data for
the new comparison. In particular, historical pilot seeds 0–4 overlap the old
final seeds 0–49, even though the README describes the final evaluation as
independent. Use the disjoint seed namespaces below.

## 3. Lock the environment and outcome definitions

Use `CartPole-v1` with perturbation
`cartpole-recovery-angle-length-v1`, whose structured parameters are
`initial_theta_deg`, `length`, and `theta_threshold_deg`.

The condition is `z = (phi0, ell)`:

- `phi0`: exact initial pole angle in degrees, with both signs retained.
- `ell`: Gymnasium's `length`, the **pole half-length in metres**; the full pole
  length is `2 * ell`. Nominal half-length is 0.5 m.

Use the historical domain `[-50, +50]` degrees × `[0.02, 3.0]` metres.
During each episode, keep the sampled plant half-length fixed. At reset, set the
angle exactly; retain independent uniform `[-0.05, 0.05]` draws for cart
position, cart velocity, and pole angular velocity. Those nuisance resets supply
the randomness for condition-level recovery probabilities.

Retain gravity 9.8, cart mass 1.0, pole mass 0.1, force magnitude 10 N,
timestep 0.02 s, Euler integration, cart-position cutoff ±2.4 m, and 500-step
time limit. The recovery task uses a **90-degree pole-angle cutoff** in training
and evaluation. It is deliberately different from standard 12-degree CartPole.
Use no new observation noise, action delay, dropout, or antisymmetry intervention.
Observations stay the four state components; do not append the condition or length.

Use the recovery wrapper, not `cartpole-angle-length-v1`: that other wrapper
adds an angle offset to a random angle. Ensure every length change also updates
the cached `polemass_length`. Centralize these operations in a reusable wrapper
or condition-applicator, rather than mutating environment internals in scripts.

Reuse `episode-summary-v4`:

- **Survival:** reach the 500-step time limit, truncated and not terminated.
- **Recovery:** survival and RMS of the last 100 stored state angles at most
  5 degrees. Use the existing state-window convention in `metrics.py` exactly.
- Also retain final-window RMS, cart displacement, return, length, failure cause,
  and positive/negative-angle differences.

Training actions are sampled by PPO. Frozen evaluation uses
`deterministic=True`, matching the existing deployed-policy comparisons.
Do not mix stochastic evaluation probabilities into these maps.

### Reward decision

The primary study preserves the existing **+1 survival reward**, including the
first failure transition, identically across all curriculum arms. Its precise
question is whether changing the training distribution alone improves recovery
of the existing survival-reward-trained PPO. Record training reward separately
from recovery and survival metrics.

Survival can saturate while recovery remains poor, as existing findings show.
Therefore include the common-reward ablation in Section 11 if time permits;
interpret a primary negative result in light of the reward as well as targeting.
Do not change the reward only for the targeted arm.

## 4. Fixed grids, reference weights, and acceptance

**Discovery/training support** is the original 143-point grid:

```text
initial_theta_deg = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]
length = [0.02, 0.04, 0.05, 0.075, 0.10, 0.20, 0.30,
          0.50, 0.75, 1.00, 1.50, 2.00, 3.00]
theta_threshold_deg = 90
```

**Reference support** is a fixed 273-point grid: angles `-50, -45, ..., +50`
degrees, crossed with the same 13 half-lengths. Assign equal reference weight
`mu(z) = 1/273` to every point. This estimates **finite-grid coverage**; equal
weights on this nonuniform length grid are not uniform physical area.
Keep the entire domain and all its edges, including difficult conditions.
Do not turn the old adaptively selected 248-point mesh into a deployment measure.

For frozen policy `pi`, let `p_rec(pi, z)` be its recovery probability over the
declared nuisance resets. The main acceptance threshold is `tau = 0.90`:

```text
acceptable recovery conditions = {z: p_rec(pi, z) >= tau}
recovery coverage = sum_z mu(z) * 1[p_rec(pi, z) >= tau]
```

Use `tau = 0.50, 0.75, 0.95` as predeclared secondary thresholds, and plot the
coverage-versus-threshold curve. Report survival probabilities separately and
their corresponding coverage as a secondary outcome. Mean recovery probability
under `mu` is another useful, less threshold-sensitive outcome.

The old adaptive selector uses a 0.5 classification threshold. Its selection rule
must not be reused unchanged for a 0.9 acceptance boundary.

## 5. Three curriculum arms and the first outer phase

This section defines configuration v1. Discovery left its boundary arm empty, so
training uses the v2 target rule recorded under `S3` in Section 13.3. Everything
else in this section still applies.

Evaluate the shared frozen parent on all 143 discovery conditions using 32
fresh episodes per condition. Reuse this initial discovery record across all
arms and continuation replicates: the old policy is identical. Selection must
read discovery results, not final-reference results.
Consequently, replicate differences measure continuation variability conditional
on one shared discovery record. Repeating discovery estimation is a later
extension, rather than an unreported source of ten independent full pipelines.

Let `s(z)` be the recovery count from `n(z)=32` discovery episodes. For targeting
only, use the smoothed estimate `p_tilde(z) = (s(z)+0.5)/(n(z)+1)`. It is a
selection score, not a confidence bound. Let `q_base(z)=1/143` and keep mixture
weight `lambda=0.5` for all arms:

| Arm | Target weights `q_tar` | Motivation |
|---|---|---|
| `background` | `q_base` | Broad-training comparator |
| `acceptance_boundary` | Background restricted to `abs(p_tilde-tau) <= 0.05`, then normalized | Conditions close to meeting the declared reliability requirement |
| `intermediate` | Weights proportional to `q_base * p_tilde * (1-p_tilde)` | Mixed-success conditions, with the highest score near success probability 0.5 |

For each arm, set `q = (1-lambda)*q_base + lambda*q_tar`. If the boundary band
is empty, use `q_tar=q_base` and record the fallback. If the intermediate-score
normalizer is numerically zero, use the same fallback. Preserve fixed condition
ordering and serialize the complete probability vector before training.
When the boundary fallback makes its distribution identical to background, label
the phase as having no boundary-targeting contrast; a tie then does not test the
boundary hypothesis.

An acceptance boundary at 0.9 and an intermediate-difficulty target near 0.5
represent different hypotheses. Near-threshold conditions may need only modest
improvement; mixed-success conditions may provide more learning signal. Their
relative value, and transfer or regression elsewhere, are the experiment.

Freeze each distribution for one complete additional-training phase. Sample a
condition **at episode reset**, then keep it fixed within that episode. `q` is
an episode-condition distribution; longer episodes contribute more transitions
to PPO. Record both realized episode shares and transition shares by condition.
Do not replay old evaluation trajectories into PPO's on-policy rollout buffer.

Do not mirror or symmetrize the policy or target distribution in this primary
comparison. Existing sign asymmetry is a diagnostic, and may lead to legitimately
different training weights on positive and negative angles.

## 6. Checkpoint continuation and matched budgets

For each continuation replicate, load the **same complete parent checkpoint**
separately for every arm. Preserve actor/critic architecture, weights, and Adam
state. Use the same continuation seed across the three arms within a replicate,
with independent seeds between replicates. Training trajectories will diverge;
paired seeds do not make them identical.

The continuation settings are:

```text
algorithm = PPO, policy architecture = unchanged MlpPolicy
device = cpu
n_envs = 8, n_steps = 32, batch_size = 256, n_epochs = 20
gamma = 0.98, gae_lambda = 0.8, ent_coef = 0.0
learning_rate = constant 0.0003
clip_range = constant 0.2
additional_transitions = 102400  # exactly 400 rollouts of 8 * 32 transitions
checkpoint_selection = final fixed budget
```

Preserve other settings from the parent and record their resolved values.
The old checkpoint used linear schedules; blindly continuing their near-zero
endpoint would change the intended experiment. Explicitly replace LR and clip
schedules before continuation and verify the actual first-update values.
Use `learn(102400, reset_num_timesteps=False)` with the newly attached environment
reset; measure additional transitions as the change in `model.num_timesteps`.

Save intermediate checkpoints at 25,600, 51,200, and 76,800 new transitions as
diagnostics and the final checkpoint at 102,400. They are not candidates for
best-on-reference selection. The primary reference comparison uses only the
parent and final child checkpoints.
Save snapshots after optimizing the corresponding PPO rollout, including the
final rollout. A standard on-step callback can run before that optimization;
record the optimizer-update count and implement post-update snapshot timing.

### Observation-space migration

The nominal parent has standard angle observation bounds, whereas the recovery
wrapper changes those bounds. Directly attaching the recovery environment to
`PPO.load` fails SB3's space-equality check. Implement a versioned migration:

```python
model = PPO.load(
    parent_zip,
    env=recovery_vecenv,
    device="cpu",
    custom_objects={"observation_space": recovery_vecenv.observation_space},
    learning_rate=0.0003,
    clip_range=0.2,
    seed=continuation_seed,
    force_reset=True,
)
```

This changes metadata for the common four-dimensional observation Box; it must
not change observation values, action space, architecture, actor/critic weights,
or normalization. Verify equal logits and deterministic actions on a fixed batch
before any training. Verify optimizer state restoration separately. Never
overwrite the original checkpoint. All training conditions must expose the same
recovery Box, regardless of their particular reset angle or length.

### Evaluation-ready checkpoints and regression

Current artifact “freeze” means passing a nominal quality gate. That is not the
same as making a policy immutable for evaluation. Fine-tuned policies that
regress must still be measured.

Add a study-specific immutable `evaluation-ready.json` seal binding the child
model hash and manifest hash to a completed fixed training budget. The study
collector should verify this seal, load with the existing hash-verifying loader
using `require_frozen=False`, and perform frozen inference. Keep the existing
nominal-gate behavior unchanged for other workflows. Never fabricate a passed
nominal gate, discard a regressing checkpoint, or overwrite a parent artifact.

Keep the canonical nominal base specification compatible across parent/children;
record the actual 90-degree training task and its condition sampler explicitly
in separate training metadata. Each evaluation retains its resolved environment
and perturbation. Do not describe fine-tuning as nominal training.

## 7. Seed namespaces and independent evaluation

Use master seed 6180 for named condition-sampling RNG streams. Training policy
seeds are ordinary SB3-compatible integers:
`300000 + 1000*outer_round + replicate_id`.

Use a separate, nonoverlapping episode-reset namespace:

```python
episode_seed = (
    role * 10**12 + outer_round * 10**10
    + replicate_id * 10**7 + worker_id * 10**5 + episode_index
)
```

```text
role 1: discovery
role 2: training resets
role 3: confirmatory reference evaluation
role 4: pilot/development evaluation
role 5: smoke evaluation
```

Require nonnegative integer indices with `outer_round < 100`,
`replicate_id < 1000`, `worker_id < 100`, and `episode_index < 100000`;
fail before a namespace can overlap. These large
episode seeds go to Gymnasium/NumPy's reset seeding, not to SB3's global seed.
For discovery and reference evaluation, use replicate/worker zero and the same
episode-index list at each condition and checkpoint, providing paired nuisance
resets. Different roles remain disjoint from each other and historical seeds.

Training workers use dedicated condition-selection RNGs such as
`SeedSequence([6180, outer_round, replicate_id, worker_id, 10])`. Omit arm ID
from the root to pair underlying categorical draws; distributions still differ.
Policy-action RNGs are separate. Handle vector-environment auto-resets using
the declared per-worker episode counter and reset stream; do not reseed global
NumPy or Torch on each step. Test that every actual reset seed and condition ID
is retained and reproducible.

## 8. Execution tiers and compute forecast

Run these tiers in order; pilot/development results are labeled separately from
the confirmatory comparison.

| Tier | Continuation replicates | Training per arm | Evaluation |
|---|---|---|---|
| Smoke | ID 900 | 8,192 transitions | Fixed 11-point support below, 8 episodes per cell, role-5 seeds |
| Pilot | IDs 0, 1, 2 | 102,400 transitions | All 273 reference conditions, 64 episodes each, role-4 seeds |
| Confirmatory | IDs 100–109 | 102,400 transitions | All 273 reference conditions, 128 episodes each, role-3 seeds |

All tiers use all three curriculum arms. The initial discovery distribution uses
32 episodes per condition and may be reused for the identical parent. Lock the
protocol after implementation checks and pilot development, before viewing role-3
reference results. Any substantive pilot-driven change gets a new versioned
configuration and is applied to every arm. Never choose training seeds or
checkpoint budgets based on favorable final results.

The smoke evaluation support is `[-40, 0, +40]` degrees crossed with
`[0.02, 0.50, 3.00]` metres, plus `(-20, 0.50)` and `(+20, 0.50)`.
It tests nominal, extreme, and mirrored cases; it is not a coverage estimate for
the full reference grid.

The confirmatory study has 30 final children plus one common parent: 1,083,264
reference episodes, at most 541,632,000 reference transitions, and 3,072,000
additional training transitions. Reference evaluation and raw-data I/O may cost
much more than training. Pilot estimates of episode length, transitions/second,
and bytes/transition must be used to forecast runtime and storage. The historical
37,200-episode study used 5.7 GB; longer-surviving children may use more per
episode. Retain raw trajectories and allow at least twice the measured projected
storage requirement. A resource-limited pilot remains a pilot; do not silently
shrink or relabel the confirmatory design.

Use the remote machine primarily for CPU concurrency and storage. For this small
MLP, start with `DummyVecEnv` and eight environments, one Torch thread per job,
and parallel independent continuation jobs or evaluation shards. Benchmark
`SubprocVecEnv` if useful, then lock the backend before confirmation. Do not change
`n_envs` or PPO batch geometry between arms for speed. A GPU is not a prerequisite;
SB3 itself recommends CPU for this style of PPO workload.

## 9. Metrics, uncertainty, and interpretation

For every final policy, retain recovery/survival counts at every reference point.
Report the plug-in recovery coverage at tau=0.9, mean recovery probability,
survival outcomes, and predeclared secondary thresholds. Label thresholded sample
estimates as estimated acceptance, rather than exact knowledge of the true set.

Report both **absolute improvement over the parent** and **paired differences
between curriculum arms** at the same budget. For each replicate, retain gained
and lost conditions, so a net increase cannot hide regressions. Include nominal
recovery, failure causes, final-window angle RMS, and signed-angle asymmetry.
Use all ten confirmatory continuation seeds; show individual paired differences,
their mean/median and spread. These are ten continuations of one parent, not ten
independently pretrained parents. Initial-policy generality is a later extension.

Keep pointwise Wilson intervals for the recovery maps. Also compute a conservative
simultaneous acceptance summary for the fixed confirmatory campaign:

- Let `M = 31 * 273` be the number of checkpoint–condition pairs and global
  error budget `alpha=0.05`.
- For each pair use an exact Clopper–Pearson binomial interval with confidence
  level `1 - alpha/M`; SciPy's `binomtest(...).proportion_ci(method='exact')`
  provides it. Bonferroni gives simultaneous coverage despite pairing across
  policies or conditions. Episodes within each cell must use fresh reset draws.
- A point is definitely acceptable when its lower limit is at least tau;
  definitely unacceptable when its upper limit is below tau; otherwise unresolved.
- Lower/upper coverage bounds sum reference weights over definitely acceptable /
  possibly acceptable points. A confidently gained point has old upper limit below
  tau and new lower limit at least tau; reverse those conditions for a confident loss.

Do not increase sample counts by repeatedly inspecting pointwise intervals.
The primary count is fixed at 128. Any planned additional stage gets its own
predeclared sample count and error accounting. Simultaneous bounds may be wide;
report them alongside the raw rate maps and plug-in estimates. Unresolved old
points are not automatically “new successes.” Snapshot, reward-ablation, or
multi-round campaigns change `M` and must declare their own confidence family.
At this family size, even 128 recoveries in 128 trials gives a simultaneous
lower probability limit of about 0.905. Thus the simultaneous lower coverage
bound at secondary threshold 0.95 will be zero; its plug-in coverage is still
informative. Stronger precision there requires a separately declared larger
sample budget, not selectively adding episodes to promising cells.

For optional uncertainty summaries over training and reset sampling, resample
paired training replicates and reset-seed indices while preserving common-seed
pairing across conditions and policies. Do not treat all condition points as
independent training replicates.

Produce physical-coordinate recovery and survival maps, old/new difference maps,
gained/lost/unresolved masks, target-weight maps, coverage versus training budget,
and signed-angle diagnostics. Label length as half-length in metres. Sparse or
rank-spaced old mesh figures are not physical-area estimates; any interpolated
contour is a visualization, not a guarantee between sampled conditions.

Interpretation examples:

- Boundary beats background under the same reward/budget: evidence for targeting
  conditional on the parent, domain, metric, and PPO settings.
- Mean recovery improves but coverage does not: improvements may not cross tau.
- Coverage gains accompany losses elsewhere: report both, not just a final map.
- Survival improves but recovery does not: investigate reward/settling mismatch.
- Intermediate difficulty beats the acceptance boundary: the “nearly acceptable”
  and “learnable” hypotheses differ.
- All methods tie or decline: retain the result and inspect learning signal,
  sampled occupancy, forgetting, and uncertainty before claiming an explanation.

## 10. Implementation work required before running

There is **no existing curriculum fine-tuning command**. The current
`train_sb3_policy` constructs a fresh model, and `_make_policy_environment`
does not apply a curriculum by changing a nominal TOML. Implement explicitly:

1. A typed, versioned study configuration containing grids, fixed reference
   weights, rewards, acceptance thresholds, target rules, budgets, seed roles,
   and parent hash. Suggested new config:
   `configs/experiments/cartpole_ppo_boundary_v1.toml`.
2. A reset-level CartPole condition sampler using structured recovery specs.
   Maintain a common observation space, fixed within-episode plant, and recorded
   condition/reset metadata. Place it inside `Monitor`; preserve SB3's terminal
   observations and `TimeLimit.truncated` handling.
3. An explicit parent-checkpoint continuation path with the metadata migration,
   schedule overrides, final-budget selection, and immutable child artifacts.
4. A hash-verified study checkpoint seal and collector that evaluates every sealed
   child, whether or not it passes the old nominal quality gate. Reuse
   `collect_episode`, serialization, and `episode-summary-v4` calculations.
5. Study orchestration separating discovery, target construction, training,
   evaluation, and reanalysis. Suggested entry point:
   `scripts/cartpole_ppo_boundary_study.py` with `prepare`, `discover`, `train`,
   `evaluate`, and `analyze` subcommands. These names are proposed interfaces,
   not currently available commands.
6. Parallel evaluation shards with deterministic job IDs, exact planned job
   accounting, immutable outputs, and safe resume of missing jobs only.

Use the existing per-episode raw format for frozen evaluation. It is acceptable
to add lossless compression/sharding if necessary, provided digest verification,
existing metric semantics, replay, and raw/derived separation are preserved and
tested. Storage optimization must not replace raw trajectories with summaries.

For training, retain raw transitions in versioned compact shards, with state,
observation, action, environment/training rewards, termination/truncation flags,
worker, episode, condition, phase, reset seed, and PPO rollout/update identifier.
Record a behavior-policy parameter hash per collected PPO rollout and retain
parent/final/periodic checkpoints. A training episode may span PPO updates;
do not label its complete trajectory as generated by one frozen policy.
Retain unfinished episodes at fixed-budget boundaries as partial training data,
not completed recovery outcomes.

### Required checks

- Exact reset angle, correct half-length/cache, common observation bounds,
  unchanged nuisance reset distribution, fixed physics within episodes.
- Identical pre-training weights/logits/actions after checkpoint migration;
  restored optimizer; nonzero intended LR/clip and exact new-transition budgets.
- Correct target normalization, full-support mixture, empty-band fallback,
  deterministic tie ordering, and recorded realized sampling shares.
- Reproducible paired seed streams and disjoint roles; no reference-data reads
  during target construction or training.
- Recovery equals the existing metric recomputed from retained raw states;
  early failure never counts as recovery.
- Parent hash unchanged; fresh child IDs; regressing checkpoints remain evaluable;
  no training during evaluation; deterministic evaluation leaves weights unchanged.
- Saved-data analysis reproduces metrics without training or new rollouts;
  parallel resume neither duplicates jobs nor omits failed outcomes.

Run relevant existing tests in `tests/test_policy_zoo.py`,
`tests/test_environments_and_rollout.py`, `tests/test_perturbation_sweeps.py`,
`tests/test_boundary_study.py`, and `tests/test_serialization.py`, plus meaningful
new continuation/sampler/analysis tests. Record code or infrastructure failures;
do not substitute favorable seeds. Finish unaffected jobs and clearly label any
incomplete campaign.

## 11. Extensions after the controlled first phase

**Repeated curricula.** Use three outer rounds, each with 102,400 new transitions.
After each round, freeze the current policy and collect fresh discovery data for
that policy; refresh its target distribution for the next round. The background
arm remains background, with the same evaluation/training accounting. Never use
old-policy success outcomes to label the updated policy. Close/rebuild training
environments between phases so every new episode follows the new distribution;
retain policy/optimizer state and cumulative counters. Final comparisons use
equal total training budgets, and a separately declared reference confidence family.

**Shared recovery-oriented reward.** As a separately versioned factorial ablation,
repeat all three curricula from the original parent with the same dense reward:

```text
nonterminal training reward = 0.1 + exp(-(pole_angle / radians(5))^2)
                             - 0.01 * (cart_position / 2.4)^2
terminal-failure training reward = -1
time-limit truncation = ordinary nonterminal reward
```

Use post-transition true state, preserve the original environment reward in the
records, and evaluate using the original recovery metric. Keep all reward
parameters identical across curricula. This tests targeting conditional on a
better-aligned objective; it is not the same experiment as distribution changes
alone. Do not tune reward weights on reference results or continue these arms
from differently trained primary children.

**Nominal-seeded curriculum.** Start from the same nominal parent but initially
restrict training support to nearby conditions, then expand using a declared
performance rule. This is a separate curriculum arm/design, not an exact
Lyapunov-certified expansion of nonlinear CartPole.

**Linear theory supporting these experiments.** Reuse the existing Stage 1
generalization and Stage 2 guarded-update results. The Stage 3 question is what
additional alignment and threshold-crossing conditions make an update expand
acceptance coverage. Work in this order:

1. State the existing finite-grid envelope accounting as a proposition. For
   condition `z`, let `J_z(K)` be the finite-horizon cost, `h` its acceptance
   threshold, `D = -sum_z q(z) * grad J_z(K)` the proposed direction, `s` a
   candidate step, and `a_z = -<grad J_z(K), D>` its local improvement score.
   An analytical directional-curvature bound `C_z(D)` gives
   `U_z(s) = J_z(K) - s*a_z + C_z(D)*s*s/2`.
   Conditions with old cost above `h` and `U_z(s) <= h` are guaranteed gains;
   old accepted conditions with `U_z(s) > h` are possible losses. Their reference
   mass difference lower-bounds coverage change. This mechanism is already in
   the linear scripts; presenting it clearly is a starting point, not a new
   numerical finding.
2. Tighten curvature bounds on the actual admissible trial segment and identify
   assumptions under which this lower bound becomes positive **before** observing
   the updated costs. Study favorable boundary alignment, harmful gradient
   conflicts, and how much feasible step length is needed to cross a threshold.
   Compare target maps through these quantities; boundary membership alone does
   not determine alignment. A useful theorem should state when its hypotheses
   hold or can be checked, rather than assume the desired coverage increase.
3. Study certificates through their permitted steps. For fixed `K` and `q`,
   changing `P` changes the guard and admissible step, while the proposed `D`
   stays the same. Treat refitting as candidate generation and retain a useful
   old verified `P` among the candidates. Compare empirical fit, verified
   margins, step capacity, and eventual progress; empirical-loss improvement
   alone does not select the most useful certificate for an update.
4. Validate these claims on the existing linear fixtures before considering a
   continuous-condition boundary-gradient argument or certificate-based cost
   upper bounds. Keep the linear cost threshold and CartPole recovery probability
   distinct; they share a curriculum question but not a proved equivalence.

These results can explain the PPO hypotheses without asserting a PPO certificate.
Hinge fitting, longer-stride certificates, and general convergence theorems are
lower priority than this mechanism and the controlled PPO study.

## 12. Remote-machine handoff and deliverables

Transfer the repository (including this plan) and complete parent artifact.
On the execution machine:

```bash
uv sync --frozen --group dev
uv run active-eval-gym --help
uv run pytest -q tests/test_policy_zoo.py tests/test_boundary_study.py
```

Verify the checkpoint hash and pinned package versions before running. Log Python,
NumPy, Gymnasium, SB3, Torch, OS, hardware, device, thread settings, implementation
commit, and resolved configuration. The original artifact records Gymnasium 1.3.0,
SB3 2.9.0, NumPy 2.5.2, Torch 2.14.0; preserve the lockfile and report any migration.
Do not auto-provision paid compute as part of this note.

Suggested new study directory:

```text
artifacts/experiments/cartpole-ppo-boundary-v1/
  protocol/             # resolved immutable config, grids, seed plan, source hashes
  discovery/            # raw parent-policy episodes and derived counts
  distributions/        # fixed q vectors, target masks, selection provenance
  pilot/                # development results, separate from confirmation
  policies/             # parent references and immutable child artifacts/seals
  training_raw/         # transition shards and policy/update provenance
  reference_raw/        # full fixed-policy evaluation trajectories and digests
  metrics_v1/           # recomputable counts, intervals, coverage, gains/losses
  figures/
  report.md
```

The final report should state exactly which tiers ran, show all continuation
replicates, separate training/evaluation costs, and answer the boundary-targeting
question without selecting a favorable metric, threshold, seed, or checkpoint.
Keep unresolved conclusions visible. Provide an artifact-only analysis command
and a short reproduction command list using the interfaces actually implemented.

Copyable task for the executing agent:

> Implement and run the controlled first-phase CartPole/PPO study in
> `ese6180/discussions/2026-11-04-post-midterm-cartpole-ppo-plan.md`, following
> stages `S0`–`S9` and their exit gates in Section 13.
> Preserve the declared recovery task, both angle signs, parent checkpoint,
> comparator budgets, and independent reference evaluation. Implement missing
> continuation/sampler/sealing infrastructure, validate it, run smoke and pilot,
> then run the locked confirmatory tier as resources permit. Retain all raw data
> and failed/regressing outcomes. Produce a self-contained report and reproducible
> analysis. Do not replace the first-phase question with adaptive search, reward
> tuning, or a claim that the linear certificate already guarantees PPO behavior.

## 13. Staged execution plan

This section orders Sections 3–12 into stages with exit gates and stop rules.
It does not change the protocol. If a stage seems to conflict with an earlier
section, the earlier section holds until a versioned configuration changes it.

`S0`–`S9` are execution stages. They are distinct from the linear Stage 1–3 in
Section 11, from outer rounds, and from the tiers in Section 8, which map to
`S6` (smoke), `S7` (pilot), and `S8` (confirmatory).

| Stage | Purpose | Main output | Exit gate |
|---|---|---|---|
| `S0` | Bind machine and parent | Provenance record | Parent, lockfile, and tests verified |
| `S1` | Encode the protocol | Study config, `protocol/` | Grids and seeds tested; D1–D3 decided |
| `S2` | Build the evaluation path | `evaluate`, `analyze` | Metrics reproduce; resume is exact |
| `S3` | Discover and build targets | `discovery/`, `distributions/` | Arm contrast reported and reviewed |
| `S4` | Build the reset-level sampler | Condition sampler | Matches fixed recovery wrapper exactly |
| `S5` | Build continuation and sealing | `train`, child artifacts, seals | First update, budget, and repeatability verified |
| `S6` | Smoke tier | Full small run | Pipeline runs end to end; costs measured |
| `S7` | Pilot tier and protocol lock | `pilot/`, lock record | Forecast fits; researcher signs lock |
| `S8` | Confirmatory tier | `policies/`, `reference_raw/` | All planned jobs accounted for |
| `S9` | Analysis and report | `metrics_v1/`, `figures/`, `report.md` | Artifact-only reanalysis reproduces report |

```text
S0 -> S1 -> S2 -> S3 --+
       |               +--> S6 -> S7 -> [lock] -> S8 -> S9 -> extensions
       +--> S4 -> S5 --+
```

Discovery comes before the training infrastructure on purpose. It needs only
the frozen parent and the evaluation path, and it tests the largest scientific
risk: that the arms barely differ. `S4`–`S5` may proceed alongside `S2`–`S3`.
A stage's gate must pass before any later stage **collects study data**. Code
may be developed ahead of a gate.

### 13.1 Readiness measured on 4 November 2026

These results are from commit `4b35396` on a 24-CPU, 30 GiB machine with 153 GiB
(164 GB) free. They are resource and integrity checks only. The benchmark used seeds
outside every role, wrote only to temporary storage, computed no recovery
outcomes, and ran no training.

- **Parent:** `model.zip` matches the expected hash, `freeze.json` binds the
  model and manifest hashes, and `load_policy_artifact(require_frozen=True)`
  succeeds. The parent records 100,096 timesteps and 7,820 updates.
- **Migration:** direct attachment fails as Section 6 predicts. The migration
  in Section 6 loads, with identical weights, Adam state, and logits and
  deterministic actions on 4,096 fixed observations. The schedules report
  constant 0.0003 and 0.2. The parent file is unchanged.
- **Stale optimizer LR:** the restored Adam group holds **-9.6e-07**, because
  the parent's linear schedule overran its 100,000-step request. SB3 overwrites
  this at the start of `train()`, so `S5` must assert the value actually used
  by the first update; the load-time schedule alone does not prove it.
- **Reset seeds:** `DummyVecEnv.reset` passes SB3-derived seeds and auto-resets
  pass none. The sampler must therefore own its reset seed and ignore the seed
  it is given; otherwise the declared role-2 stream is silently not used.
- **Collector:** `collect_sweep` is serial, refuses an existing output directory,
  and requires a passed nominal gate, so it cannot be reused for sealed children.
  The new collector should wrap `collect_episode`, `make_artifact_environment`,
  and `write_episode` instead.
- **Evaluation cost:** one core sustains about 10,900 transitions/s for
  collection plus writing, excluding per-episode environment construction.
  Storage is about 482 bytes/transition and 3 files/episode. The 153.8 KB/episode
  matches the historical 5.7 GB over 37,200 episodes. Per-file gzip reached 5.6×.
- **Environment:** `.venv` was not lock-exact: it held 10 packages absent from
  `uv.lock`. A search of `src`, `scripts`, `tests`, `notebooks`, and `configs`
  found no use of `cvxpy`, `clarabel`, `joblib`, or `osqp`. `S0` removed all 10.
- **Dependency:** SciPy, needed for Clopper–Pearson intervals, was available only
  transitively through `control`. `S1` made it a direct dependency.

Forecast, using worst-case 500-step episodes and counting the parent among the
pilot's evaluated checkpoints:

| Stage | Training transitions | Evaluation episodes | Max evaluation transitions | Raw, max | At 5.6× | Core-hours |
|---|---|---|---|---|---|---|
| `S3` | 0 | 4,576 | 2.29 M | 1.1 GB | 0.2 GB | 0.06 |
| `S6` | 24,576 | 352 | 0.18 M | 0.08 GB | 0.02 GB | 0.01 |
| `S7` | 921,600 | 174,720 | 87.4 M | 42 GB | 7.5 GB | 2.2 |
| `S8` | 3,072,000 | 1,083,264 | 541.6 M | 261 GB | 47 GB | 13.8 |
| Total | 4,018,176 | 1,262,912 | 631.5 M | 304 GB | 54 GB | 16.1 |

Core-hours cover evaluation only; training rate is first measured in `S6`.
**Disk, not CPU, is the constraint.** Section 8's twofold margin needs about
608 GB raw, which does not fit in 164 GB. At 5.6× it needs about 109 GB, which
fits. The gzip ratio comes from five hand-picked conditions, so `S6` must
replace it with a measured value in the final format.

### 13.2 Decisions fixed before discovery is read

These were decided on 4 November 2026, before any discovery data existed, and
are recorded in `configs/experiments/cartpole_ppo_boundary_v1.toml`. Deciding
after seeing discovery would have made them data-dependent.

- **D1, evaluation storage.** Decided: one compressed shard per
  (checkpoint, condition) job, holding the existing JSON-lines trajectory rows,
  with each episode's digest computed on its uncompressed bytes as now. That
  gives 8,463 resumable confirmatory jobs instead of about 3.2 million files.
  Section 10 permits this if digests, metric semantics, and replay are preserved.
- **D2, thin boundary band.** With 32 episodes, Section 5's band contains only
  `s` in {28, 29, 30}. Section 5 handles an empty band but not a very small one,
  where a single condition could receive more than half of the episodes.
  Decided: keep the rule unchanged, and report a phase with fewer than 5 band
  conditions as "thin-band" so its result is read as a highly concentrated test.
- **D3, snapshot evaluation.** Section 9 asks for coverage versus training budget,
  but Section 6 limits the primary reference comparison to parent and final
  checkpoints, and no tier counts snapshot evaluations. Across confirmatory
  children they would add 3,144,960 episodes and change `M`. Decided:
  evaluate snapshots in the pilot only (27 checkpoints, 471,744 episodes, about
  20 GB at 5.6×), labelled as development data, if `S6` shows room. Otherwise
  drop the figure from v1 and say so.

### 13.3 Stages

**`S0`: bind the machine and parent. Passed 4 November 2026.**

- Work: run `uv sync --frozen --group dev`, rerun the Section 10 test suites, and
  record package versions, hardware, and thread settings.
- Gate: the lockfile applies cleanly and the tests pass at the recorded commit.
- If it fails: report the discrepancy; do not substitute a parent (Section 2).
- Result: the sync left `.venv` lock-exact, and all 114 tests passed at `4b35396`.
  Package versions match the parent's manifest with no differences. The manifest
  records the parent as built at `cfc3a51` from a dirty working tree, so its
  exact training source cannot be recovered from that commit alone.
  `prepare` now records runtime provenance in `protocol/provenance.json`.

**`S1`: encode the protocol. Gate passed 4 November 2026; committed protocol
record outstanding.**

- Work: add `configs/experiments/cartpole_ppo_boundary_v1.toml`, a typed loader,
  the seed functions with their range checks, and the `prepare` subcommand that
  writes `protocol/`. Add SciPy as a direct dependency at its locked version and
  report the `pyproject.toml` and `uv.lock` change.
- Gate: tests confirm 143, 273, and 11 grid points, reference weights of 1/273,
  and disjoint seed roles with out-of-range indices rejected. D1–D3 are recorded.
- Result: added the config, `load_ppo_boundary_study` in `config.py`,
  `ppo_boundary.py`, `scripts/cartpole_ppo_boundary_study.py`, and
  `tests/test_ppo_boundary_study.py`. The `uv.lock` change adds only the direct
  SciPy edge; no package version moved. All 161 tests pass, and deliberately
  broken seed limits, weights, accounting, and validation rules each fail a test.
  A development run on the real parent reproduced the Section 13.1 job counts.
- Remaining: `prepare` refuses uncommitted source, so the study's real
  `protocol/` waits for a commit of `S1`–`S2`. Then run
  `uv run python scripts/cartpole_ppo_boundary_study.py prepare --output
  artifacts/experiments/cartpole-ppo-boundary-v1`. Do this before `S3`.

**`S2`: build the evaluation path. Gate passed 4 November 2026.**

- Work: add the job collector, the D1 storage layer, the `evaluate` subcommand,
  and the `analyze` subcommand with cell counts, Wilson intervals,
  Clopper–Pearson intervals, and coverage. Load the parent through the frozen
  path; seal support for children arrives in `S5`.
- Gate, using a small parent evaluation in temporary storage: recovery recomputed
  from stored states equals `episode-summary-v4`; a shard round-trips byte-exact
  to the existing digest; an interrupted run resumes without duplicates or
  omissions; `analyze` runs without environments or policies.
- Result: added shard reading and writing to `serialization.py`, plus
  `ppo_boundary_evaluation.py`, `ppo_boundary_analysis.py`, and the `evaluate` and
  `analyze` subcommands. All 196 tests pass, and 17 deliberately broken safeguards
  each fail a test.
- Real-parent run, smoke seeds in temporary storage, integrity counters only: a
  `SIGKILL` left 3 of 11 shards and no completion record. Resuming ran only the
  8 missing jobs and produced shards byte-identical to an uninterrupted run.
  Reanalysis was byte-identical, and an independent recovery recomputation from
  stored states agreed on all 11 cells, with both outcomes present. Shards cost
  85 bytes/transition, a 5.8× ratio, consistent with the Section 13.1 forecast.
  `S6` still supplies the formal measurement.
- Layout: derived results go in `metrics_v1/<campaign>/` for every campaign, and
  smoke raw data goes in `smoke/`. Campaign directories otherwise follow Section 12.
- Scope limits: the command line exposes only `discovery`. Tier campaigns open
  once `S5` adds sealed children, which also extends job planning beyond the
  parent. Confirmatory collection is refused until the `S7` lock exists.
- Collection requires committed source and records its own source state, so
  later `S4`–`S5` code does not invalidate discovery. A resume on changed source
  is refused. Clopper–Pearson intervals cover recovery only, keeping `M` at the
  declared number of checkpoint–condition pairs.

**`S3`: run discovery and construct targets. Paused on the stop rule, then v2
adopted and gate passed 4 November 2026.**

- Work: run the parent on all 143 conditions with 32 role-1 episodes each, then
  serialize the three `q` vectors with their masks and provenance. Reference
  results must not exist yet.
- Report: the band size `k_B`; each arm's effective support, `1/sum_z q(z)^2`;
  the pairwise total-variation distances; and the sign balance of each target.
- Why this matters: the `p_tilde` floor gives a saturated cell about 1/17 of the score
  of a perfectly mixed cell. On a nearly saturated map, this pulls `intermediate`
  toward `background`: with 5 perfectly mixed cells among 143, they would hold
  only 38% of its target mass.
- Gate: the researcher reviews the contrast report before `S6`.
- If `k_B = 0`, or `intermediate` is close to `background`: pause. The first
  phase may not be able to test that hypothesis. Any response is a new versioned
  configuration applied to all arms before role-3 data exist, and the original
  discovery results stay on record.
- Result: discovery ran at clean commit `aafc379`, with 143 of 143 jobs, 4,576
  episodes, and no failures. Target rules were written and tested before the
  results were opened. The stop rule fired on both counts:
  - Recovery counts are almost binary: 123 conditions at 0/32, 15 at 32/32, and
    only 5 mixed (1, 3, 5, 17, and 26 of 32). None falls in {28, 29, 30}, so
    `k_B = 0` and `acceptance_boundary` is identical to `background`.
  - `intermediate` has total-variation distance 0.107 from `background`, and only
    14% of its episodes land on the 5 mixed conditions. The `p_tilde` floor gives
    the 138 saturated conditions 75% of its target mass.
  - The cause is structural: a deterministic policy with ±0.05 nuisance resets
    gives few intermediate probabilities on this grid. Eleven conditions survive
    at least 29/32 without recovering, including six on the acceptance frontier.
    The +1 reward is saturated there and may carry no signal for recovery.
- Alternatives costed on discovery counts only, before v2 was chosen:

  | Target rule | Conditions | Total-variation distance from background | Largest `q` | Target mass already accepted |
  |---|---|---|---|---|
  | Band ±0.05, as declared | 0 | 0.000 | 0.007 | Not applicable |
  | Band widened to ±0.10 | 16 | 0.444 | 0.035 | 94% |
  | Frontier: unaccepted 4-neighbours of accepted conditions | 15 | 0.448 | 0.037 | 0% |
  | Frontier: unaccepted 8-neighbours | 22 | 0.423 | 0.026 | 0% |
  | `intermediate` as declared | 143 | 0.107 | 0.049 | 8% |
  | `intermediate` restricted to mixed conditions | 5 | 0.483 | 0.185 | 0% |

  Widening the band further adds nothing up to ±0.20: it holds the same 16
  conditions, 15 already at 32/32.
- Decision: the researcher adopted `configs/experiments/cartpole_ppo_boundary_v2.toml`
  with rule `ppo-boundary-targets-v2`, applied to all arms.
  - `acceptance_boundary`: conditions with `s/32 < 0.90` that are 4-neighbours, by
    grid index, of a condition with `s/32 >= 0.90`, weighted equally.
  - `intermediate`: `p_tilde * (1-p_tilde)` weights, restricted to `0 < s < 32`.
  - Mixture weight, fallbacks, seeds, budgets, grids, and acceptance are unchanged.
    Tests require v1 and v2 to differ only in study ID and target rule. The v1
    config and its protocol and discovery records stay as they are.
- Meaning of the change: "near the boundary" now means adjacent in condition
  space, not close to probability 0.9. Eleven of the 15 frontier conditions are
  at 0/32, so some may not be nearly acceptable. One grid step is not a fixed
  physical distance, because the length grid is nonuniform.
- Preview in temporary storage, on the real parent: rerunning discovery under v2
  reproduced all 4,576 v1 trajectory digests and all recovery counts. The arms
  have no fallbacks and 15 frontier conditions, so the phase is not thin-band.
  Total-variation distances are 0.448 and 0.483 from `background` and 0.367
  between the targeted arms. `intermediate` has effective support 14.7 and puts
  18.5% of episodes on one condition (0 degrees, 0.3 m).
- Real v2 record: `prepare`, `evaluate`, `analyze`, and `targets` ran at clean
  commit `7e8c534` into `artifacts/experiments/cartpole-ppo-boundary-v2`. It is
  not a development record. Its 143 jobs and 4,576 episodes match every v1
  trajectory digest, and its contrast figures equal the preview above. The
  researcher reviewed those figures. No smoke, pilot, or reference data exist.
  The script now defaults to the v2 config. `S6` may start once `S4`–`S5` are
  complete.

**`S4`: build the reset-level condition sampler. Gate passed 5 November 2026;
commit outstanding.** The current recovery wrapper fixes length and angle at
construction, so per-episode conditions need new code.

- Work: move the angle, length, `polemass_length`, and cutoff logic into one
  shared function used by both the fixed wrapper and the sampler. Place the
  sampler inside `Monitor`, own the reset-seed stream, and expose condition ID,
  worker, episode index, and reset seed in `info`.
- Gate: for the same condition and reset seed, sampler and fixed-wrapper episodes
  are identical under a fixed action sequence. Length stays constant within an
  episode, the observation Box is common, auto-resets follow the declared stream,
  and every other Section 10 sampler check passes. Existing fixed-wrapper
  trajectories must not change.
- Result: added `CartPoleRecoveryTask` in `envs/perturbations.py`, now used by
  both the fixed wrapper and `CartPoleRecoveryConditionSampler` in
  `envs/condition_sampler.py`. The sampler uses one uniform draw per episode
  through the arm's inverse CDF, so arms share their underlying draws. Sixty
  sampled episodes, crossing many conditions in one environment, match fresh
  fixed-wrapper episodes exactly.
- Existing trajectories: replaying all 143 `S3` conditions after the refactor
  reproduced all 4,576 stored trajectory digests.

**`S5`: build continuation, capture, and sealing. Gate passed 5 November 2026;
commit outstanding.**

- Work: add the Section 6 migration, the `train` subcommand, raw transition
  shards, rollout-level policy hashes, child artifacts, `evaluation-ready.json`,
  and seal-verified loading. Prefer repeated
  `learn(chunk, reset_num_timesteps=False)` calls for snapshots: a save between
  calls is post-update by construction and needs no hook into PPO.
- Gate: the first update uses LR 0.0003 and clip 0.2, confirmed in the optimizer;
  the new-transition count is exact; the same seed twice gives identical final
  weights; chunked and single-call runs give identical weights, or the chunking
  is dropped; the parent hash is unchanged; a regressing child seals and loads.
- Result: added `ppo_boundary_training.py`, the `train` subcommand, transition
  shards in `serialization.py`, and seals in `policies/artifacts.py`. Evaluation
  now plans the parent plus every sealed child, and the command line opens
  `smoke` and `pilot`. A tier campaign refuses to start until all its children
  are sealed. Confirmatory training and collection stay refused until the `S7`
  lock. All 328 tests pass, and 65 deliberately broken safeguards each fail a test.
- Gaps those broken safeguards exposed, now closed: no test showed that a job
  trains on its own arm's distribution, or that replicates apply their own seed.
  Without `seed=`, SB3 reuses the parent's stored seed, so runs stay repeatable
  while every replicate shares one action stream. Seals also now hash the model
  file itself, not just the digest recorded in the manifest.
- Real-parent run, pilot budget and replicate 0, temporary storage, development
  source, no child evaluated:
  - The first optimizer step used LR 0.0003 and clip 0.2, replacing the restored
    -9.6e-07. Each arm gained exactly 102,400 transitions and 8,000 updates.
  - Two runs with the same seed, and chunked against single-call training, gave
    identical weights, optimizer state, and transition data. Chunking stays.
  - The parent hash is unchanged. Snapshots equal the next rollout's recorded
    behavior policy, which confirms that they are post-update.
  - Cost: about 7,500 transitions/s on one core, 92 bytes/transition, and 9.5 MB
    per child. `S6` supplies the formal measurement.
- Point for `S7` review: realized episode shares match `q` to within sampling
  noise (total-variation distance 0.13-0.19, against about 0.16 expected for
  background). Transition shares differ more (0.27-0.38), because surviving
  conditions run longer, as Section 5 expects. Arms also differ in episode count:
  1,008 for `acceptance_boundary`, 879 for `background`, and 458 for `intermediate`.
  PPO's updates therefore see less contrast than `q` suggests.
- Layout: children go in `policies/<tier>/<policy_id>/`, with snapshots under
  `snapshots/`, and transitions in `training_raw/<tier>/<policy_id>/`. An
  interrupted attempt moves to `training_raw/<tier>/abandoned/` and reruns from
  the parent with the same seeds. Publishing a sealed policy marks completion.
- Scope limits: real children have not been evaluated. Pilot snapshots are not
  yet in evaluation planning, pending the D3 decision in `S6`. `S6` needs these
  changes committed, because `train` and `evaluate` refuse uncommitted source.

**`S6`: smoke tier.** This is replicate 900 with all three arms, 8,192
transitions each, and evaluation on the 11-point support. It makes no scientific
claim.

- Work: run `prepare` through `analyze`, deliberately interrupting and resuming
  at least one training job and one evaluation job.
- Gate: job accounting is exact and every seal verifies. Measure training
  transitions/s, evaluation transitions/s, bytes/transition in the final format,
  and training-shard size. Re-forecast `S7` and `S8` using these values, and
  settle the D3 storage condition.

**`S7`: pilot tier and protocol lock.** This is replicates 0–2, with 102,400
transitions per arm, and evaluation at 64 role-4 episodes per reference condition.

- Review: realized episode and transition shares, regressions, failure causes,
  and whether the Section 10 checks hold at full budget.
- Reward check: decide here whether the Section 11 reward ablation moves ahead of
  the other extensions. Six frontier conditions already survive 32/32 without
  being accepted, so a saturated +1 reward may give them no signal for recovery.
- Change control: any substantive change gets a new versioned configuration, is
  applied to all arms, and restarts from the earliest affected stage.
- Forecast check: confirmatory storage with twofold margin must fit the disk
  actually available. If it does not, obtain storage or change the format. Do
  not shrink the design (Section 8).
- Gate: write a lock record containing the commit, resolved-config hash, parent
  hash, vector-environment backend, seed plan, and forecast. Pilot metrics are
  never pooled with confirmatory results. **The researcher signs the lock.**

**`S8`: confirmatory tier.** This is replicates 100–109 with all three arms.

- Order: train and seal all 30 children before the first role-3 episode, so no
  child can depend on reference outcomes. Then evaluate 31 checkpoints across
  273 conditions at 128 episodes each.
- While running, monitor job counts, failures, disk, and throughput only. Do not
  view partial maps or per-arm summaries.
- Failures: retry an infrastructure failure with the same declared seeds and
  record it. A poorly performing checkpoint is an outcome, not a failure.
- Gate: 30 sealed children, 8,463 evaluation jobs, and 1,083,264 episodes, all
  with verified digests. If any cannot be completed, finish unaffected jobs and
  label the campaign incomplete (Section 10).

**`S9`: analysis and report.**

- Work: run `analyze` from saved data only to produce `metrics_v1/`, every
  Section 9 figure, and `report.md`. Report all ten replicates, including gained,
  lost, and unresolved conditions, and apply the Section 9 interpretation rules
  without selecting a favorable metric, threshold, seed, or checkpoint. State
  that the targeted arms follow v2, what "near" means there, and that the reward
  may explain a null result.
- Gate: a clean artifact-only rerun reproduces all reported numbers with no
  training or new rollouts. New metrics get new versions rather than overwriting
  `metrics_v1/`.

### 13.4 After `S9`

Each Section 11 extension needs its own versioned configuration, confidence
family, and storage forecast, and none reuses confirmatory results as if it were
independent of them.

1. **Reward ablation first** if `S9` shows survival improving without recovery.
   This is the mismatch Section 3 predicts.
2. **Repeated curricula first** if one phase moves coverage but the band or
   target goes stale, since these test feedback from refreshed targeting.
3. **Nominal-seeded curriculum** after either of these.

The linear track in Section 11 runs alongside these stages. It neither blocks
nor is blocked by any gate, and its results do not count as evidence for a
PPO-stage gate.

## Primary references and interpretation

- [Stable-Baselines3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
  supports the existing-library implementation and CPU-first execution choice.
- [SB3 vector-environment documentation](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)
  explains auto-resets and terminal-observation handling.
- [Gymnasium CartPole documentation](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
  describes the base task; the repository's structured recovery wrapper supplies
  the explicitly modified task used here.
- [SciPy exact binomial intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html)
  documents the Clopper–Pearson interval implementation.
- [Reverse Curriculum Generation](https://proceedings.mlr.press/v78/florensa17a.html)
  motivates performance-adaptive starting-state curricula.
- [Prioritized Level Replay](https://proceedings.mlr.press/v139/jiang21b.html)
  motivates prioritizing estimated training potential of conditions, not replaying
  frozen-policy evaluation transitions into PPO.
- [No Regrets](https://arxiv.org/abs/2408.15099)
  motivates the mixed-success comparator. The simple p(1-p) arm above is our
  declared comparator, not a claim to reproduce that paper's full algorithm.
- [Learning Stability Certificates from Data](https://proceedings.mlr.press/v155/boffi21a.html)
  remains the theoretical anchor for the separate certificate-learning track.
  It does not establish coverage expansion for this PPO experiment.
