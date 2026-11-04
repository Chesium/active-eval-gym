# Linear update rules across plant families

This study follows `docs/linear-update-study.md`. That study's table came from one
pair of plants, one backtracking grid, and an arbitrary tie-break, and scratch checks
suggested that most of its gaps were step-grid artifacts. This study replaces those
single outcomes with declared distributions.

## Status of this design

This file was written before any run of this study. It is **not** a blind
preregistration: an exploratory scratch check had already used these four family
types, the exact-step rule, and the 1-point threshold, and had shown broadly what to
expect. To limit the damage, the sampling rules below are copied unchanged from that
check, nothing is tuned on retained results, and family seeds use a new stream
(`SeedSequence([master_seed, type_index, family_index])`), so the families differ from
the exploratory ones. Results are descriptive; no multiplicity correction is applied.

## Questions

1. On the original plants, how much of the coverage table depends on the step grid and
   on arbitrary tie-breaking, rather than on the certificate or target?
2. Does the original pair have unusually little gradient conflict?
3. When conflict is present, does boundary targeting change coverage, and is the
   change consistently positive?

## Update rules

All rules use the same exact gradient direction `D`, target laws, guard, and Armijo
test as the original study (`rho = 0.5`, `kappa = 0.25`, Armijo coefficient 0.1).

- `backtracking`: the original rule, `s = s_0 * 0.5^j` from `s_0 = 1`, first passing
  trial, at most 31 trials.
- `capacity`: first trial `min(1, (1 - 1e-9) * step capacity)`, shrinking by 0.9 only
  if Armijo rejects it, at most 200 trials.
- `guard_only`: the same without the cap at 1.

Step capacity is the largest step along `D` that passes the guard. Each record keeps
the step, capacity, radius limit, and rejection counts, so the limiting constraint can
be read back. `capacity` is a changed update rule, not a rerun of `backtracking`.

## Reference family

The original fixture: `G_1`, `G_2`, `A = G + I`, `B = I`, `K_0 = I`, `H = 10`, `Q = I`,
`R = 0.1 I`, `h = 3.5`, band half-width 0.35, mixture 0.5, 64 training directions and
256 reference directions per plant, three rounds. Three parts:

- `rules`: every certificate rule and fixed or refitted mode of the original study,
  all three targets, replicates 0-9, `N = 32`, under all three update rules.
- `shift`: replicate 0, the same arms under `backtracking` with
  `s_0 = 2^(-i/16)`, `i = 0, ..., 15`. These 16 values shift the step grid without
  changing its ratio.
- `ties`: every candidate with minimum population loss at `K_0` that is verified, as
  a fixed `P`, all three targets, under `backtracking` and `capacity`. No sampling is
  involved.

## Plant families

400 two-plant families of each type, all with two-dimensional states and inputs.
Each uses `A_i = G_i + B_i K_0` with `K_0 = I`, so `G_i` is the initial closed loop.

| Type | `G_i` | `B_i` |
|---|---|---|
| `same_shape` | upper triangular, independent `U[.5,.9]`, `U[.3,.7]`, `U[.4,.7]` | `I` |
| `general` | `rho g / spectral_radius(g)`, `g` standard normal, `rho ~ U[.6,.9]` | `I` |
| `rotated_actuator` | as `general` | `B_1 = I`, `B_2 = Rot(phi)`, `phi ~ U[pi/4, 3pi/4]` |
| `mirrored` | `G_1` as `same_shape`, `G_2 = -G_1 + E`, `E_ij ~ N(0, 0.08^2)` | `I` |

`Q`, `R`, `H`, rounds, condition grids, and mixture match the reference family. The
threshold `h` is the 53.9% quantile of reference-law costs at `K_0`, matching the
reference family's initial coverage, and the band half-width is `0.1 h`, matching its
0.35/3.5 ratio.

`P` is the model-grid rule at `K_0`: over the 874 candidates, the verified matrix
maximizing `min_theta eta_theta(P) / ||P||_2`. It ignores targets and is held fixed
for three rounds. Every target and update rule runs on that `P`.

No family is excluded or resampled. A family with no verified candidate is recorded as
`no_shared_certificate` and keeps its conflict measures, but has no updates.

## Outcomes

Coverage is the `mu`-mass of `J <= h` on the 512-condition reference law.

**Primary outcome:** boundary coverage minus background coverage, in percentage
points, after three rounds under `capacity`, per family, over all families with a
shared certificate. Families where both arms reach 100% count as a zero gap.

Secondary outcomes:

- the same gap after one round, under each update rule, and for `high_cost`;
- the same summaries restricted to unsaturated families;
- conflict at `K_0`: cosine of the background direction with the mixture and band-only
  directions, share of condition pairs with negative gradient inner product, and share
  of conditions whose cost rises along the background direction;
- the Spearman correlation of `1 - cos(band)` with the absolute gap;
- each family's limiting constraint, to check whether the `s_0` cap creates a target
  effect through the size of `D` rather than its direction;
- mass of conditions lost by each update.

A gap of at least 1 point in absolute value counts as meaningful. Per type, the
analysis reports mean and median gaps, a percentile bootstrap 95% interval for the mean
over families (10,000 resamples), the share of positive, negative, and zero gaps, and
an exact two-sided sign test on the nonzero gaps.

## Run and analyze

```bash
uv run python scripts/linear_family_study.py run --preset full --jobs 16 \
  --output artifacts/linear-family-study-v1
uv run python scripts/linear_family_study.py analyze \
  --run artifacts/linear-family-study-v1 \
  --output artifacts/linear-family-study-v1-report
```

Run and report directories must be new. `raw/` keeps the matrices, every accepted gain,
and its full reference-law state trajectories in float64. The training grid is a
subset of that law. Actions are `-K x` from the saved gain. Rejected trial gains are
not kept; their step and rejection counts are in `runs/`. Analysis reads only these
files, checks their hashes, and computes every metric from the saved states.

## Changes after the design was written

- A size check before the retained run showed that `run_arm` restarted every round from
  `K_0`, which no test had caught. It was fixed, with tests that require rounds to
  chain and require `backtracking` to reproduce the original study's values.
- The Spearman correlation uses unsaturated families, since a saturated pair has no
  gap to explain.
- A family with an empty band falls back to background under the existing target rule,
  so its band cosine is 1. These families stay in the study.

## Limits

These are designed test families, so "typical" means typical of these sampling rules.
`mirrored` is deliberately extreme. No result here extends to nonlinear plants, PPO,
or continuous initial-state domains.
