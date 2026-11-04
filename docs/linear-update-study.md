# Linear target maps, certificates, and controller updates

This analytical companion extends the certificate notebook's fixed matrices
into controller updates. It keeps training, frozen-policy rollout collection,
target-map construction, certificate fitting, and metric computation explicit.
The notebook and its support module retain their original behavior.

## Run and analyze

```bash
uv run python scripts/linear_update_study.py run --preset smoke \
  --output artifacts/linear-study-smoke
uv run python scripts/linear_update_study.py analyze \
  --run artifacts/linear-study-smoke --output artifacts/linear-study-smoke-report
```

`full` uses ten replicate IDs 0–9, sample sizes 8, 32, and 128 for one-step
comparisons, 32 fitting trajectories per repeated round, twenty rounds, and all
874 grid candidates in the initial certificate landscape. `smoke` uses replicate
IDs 0–1, sample size 32, three rounds, and no exhaustive landscape. Both presets
include scalar fixtures and scale-invariance checks. All declared replicates,
including failed selections and halted branches, remain in the results.

Run and report directories must be new. A completed run has a manifest with
`completion: complete`; interruption leaves an incomplete directory that analysis
will not accept. Use a new directory to rerun. Analysis has no training side
effects and does not require the original policy-update procedure to execute.

## Fixed choices and comparison arms

The primary family realizes the notebook's two matrices as `A = G + I`, `B = I`,
with initial `K = I`. The horizon is 10, `Q = I`, `R = 0.1 I`, and the acceptance
threshold is 3.5. Discovery uses 64 equally spaced unit-circle directions per
plant; reference evaluation uses 256 per plant. Each law gives equal mass to
plants and directions. The reference law remains fixed across all rounds.
Reference coverage is a finite-law probability, not area of a physical domain.

Training laws are background alone, a band `abs(J-h) <= 0.35`, and the highest-cost
quarter. Targeted laws mix half background with half normalized target weights.
Empty bands fall back to background. The highest-cost cutoff breaks exact ties
by condition ID. Only old-controller costs construct the targets; the weights
remain fixed during each line search and are refreshed between rounds.

Certificates use the original bounded class, fitting margins, and grid ordering.
The five selectors are:

1. Known notebook certificate `diag(0.6, 1.6)`.
2. First empirical minimizer, retaining grid order and the declared `1e-12` tie
   tolerance.
3. Verified empirical minimizer with the smallest certified step capacity along
   the background-gradient direction.
4. Verified empirical minimizer with the largest such capacity.
5. Model-based grid maximizer of `min(eta_theta) / ||P||_2`.

The background probe is common to all target maps at a given controller. Thus
certificate selection does not secretly optimize the subsequent target's measured
performance. Step capacity solves the radius and quadratic consumption inequalities
analytically, without Armijo or post-update performance. A zero probe uses candidate
order. The known and model-grid selectors are model-based baselines; selectors 3
and 4 also use model-based verification and capacity to resolve empirical ties.
They are not purely trajectory-only estimators.

Fixed branches keep the initial P; refit branches choose a new P from fresh
sampled trajectories before each later update. P is frozen during every update.
The known baseline has no duplicate refit branch. Failed selection or old
verification halts a branch at its current controller. Remaining rounds record
`halted`. Exhausted line searches and numerical zero gradients are no-ops; they
do not halt later rounds. No retained plants are removed.

Sampling reuses the certificate module's sampler with
`SeedSequence([6180, replicate + 100000 * round])`. Condition indices are paired
across targets, selectors, and fixed/refit modes. Their trajectories depend on each
branch's controller. Sample sizes use nested prefixes. Later fixed-P samples are
diagnostics, not certificate refits. The study does not assert a new simultaneous
statistical generalization guarantee across adaptive rounds.

## Update and bounds

The proposed direction is the exact negative gradient of the weighted
finite-horizon cost. Forward sensitivities include both changed actions and
changed future states. P affects the accepted step rather than this direction.
Trials use `s = 0.5**j`, for `j = 0,...,30`, with radius 0.25, retained fraction
0.5, and Armijo coefficient 0.1. Old verification uses tolerance `1e-10` and
gradients of Frobenius norm at most `1e-12` are numerical no-ops.

Each trial records radius, margin-consumption, and Armijo outcomes separately.
Direct new eigenvalue margins are diagnostics; they do not replace the sufficient
consumption guard. Rejected guard trials need no objective rollouts.

Gradient alignment is `a_z = -<grad J_z, D>`. The first-order prediction `-s a_z`
is a diagnostic. To bound the remainder throughout the allowed gain neighborhood,
let `X`, `U`, and `V` bound the state norm and its first two directional derivatives.
For each plant, use

```text
M = max(1, ||G_K||_2 + ||B||_2 kappa)
d = ||B D||_2
X_0 = ||x0||, U_0 = V_0 = 0
X_(t+1) = M X_t
U_(t+1) = M U_t + d X_t
V_(t+1) = M V_t + 2 d U_t
```

For `W = Q + (K+sD)^T R (K+sD)`, take

```text
W0 = ||Q||_2 + ||R||_2 (||K||_2 + kappa)^2
W1 = 2 ||R||_2 (||K||_2 + kappa) ||D||_2
W2 = 2 ||R||_2 ||D||_2^2
C_z = sum_t [2 W0 (U_t^2 + X_t V_t) + 4 W1 X_t U_t + W2 X_t^2]
```

Then `U_z(s) = J_z(K) - s a_z + C_z s^2 / 2` upper-bounds the new cost for
admissible segments. Guaranteed gains have `J_old > h` and `U_z <= h`; possible
losses have `J_old <= h` and `U_z > h`. Their mass difference lower-bounds coverage
change. The report also computes Lemma 15's cost-drift bound and Theorem 16's
threshold-loss-band bound independently.

All inequalities are evaluated in float64, not interval arithmetic. Tests compare
derivatives independently and sample whole admissible segments. These checks do
not replace a mathematical proof. Large bounds can be uninformative and are still
reported; they are not tuned after inspecting outcomes.

## Artifacts and report

- `manifest.json`: resolved fixtures, settings, branch metadata, trial history,
  policy/checkpoint descriptions, seeds, metric version, source and raw hashes.
- `candidate_grid.npz`: certificate candidates and covering-radius upper bound.
- `raw/`: deduplicated gains, full state trajectories, and applied actions for
  discovery and reference conditions. Fitting indices refer to these trajectories.
  Every objective-evaluated line-search trial references its raw snapshot.
- `metrics_v1/`: selected P, sampled indices, q, gradients, bounds, costs, and
  threshold classifications, separate from raw observations.
- Analysis: reconstructed per-record metrics, paired-difference CSV, figures,
  report, and source-manifest hash. It differentiates retained state trajectories
  to reconstruct alignment, without collecting new trajectories.

The figures show predeclared replicate 0 examples alongside aggregate plots and
all-replicate traces. Paired CSVs retain every replicate. Reported means are
descriptive, not confidence intervals. Gains and losses refer to conditions
crossing the fixed threshold, rather than every numerical increase/decrease in
cost. Training objectives change when q changes, so their values are not one
common objective decreasing across rounds.

Scale diagnostics use `0.9 P0`, `P0`, and `1.1 P0`: the consumption and margin both
scale by the same positive constant. Certificate fitting loss need not be invariant
to this rescaling, because eta and gamma remain fixed. The diagnostic isolates the
update guard rather than claiming these scaled matrices are tied empirical fits.

The scalar report counterexample uses its prescribed asymmetric weights, not the
symmetric boundary map. It demonstrates possible regression, not failure of every
boundary curriculum. The aligned scalar example illustrates a favorable direction.
Neither example establishes target superiority on the primary family.

No convergence, monotonic coverage, continuous-domain expansion, or nonlinear/RL
guarantee is asserted. Grid ERM need not solve continuous-class ERM. A rejected P
does not prove plant instability, and a rejected update may be stable but fail the
chosen sufficient test. A future CartPole/PPO experiment must separately address
nonlinear dynamics, discrete actions, sampled optimization, and evaluation noise.
