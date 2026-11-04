# Crossed comparison of certificate rules, refitting, targets, and families

`docs/linear-family-study.md` varied the family and target under one certificate rule,
and varied the certificate rule and refitting only on the original pair. Those factors
were not crossed, so it could not show whether their effects depend on the family.
This study crosses them.

## Status of this design

This file was written before any run of this study. It is not a blind preregistration:
the family study had already shown these family types, the exact step, and the effect of
refitting on the original pair. Nothing here is tuned on results from this run, and no
family or arm is dropped based on its outcome. Results are descriptive; no multiplicity
correction is applied.

## Factors

Every family runs every combination, so all comparisons are paired within a family.

- **Family type:** `same_shape`, `general`, `rotated_actuator`, and `mirrored`, using
  the sampling rules and seeds of the family study. Indices 0-199 of each type, 800
  families in all. This is half the earlier count, to limit disk use.
- **Tie-break:** among `N = 32` fitting trajectories, `erm_first` takes the first
  empirical minimizer in grid order; `erm_small_step` and `erm_large_step` take the
  verified minimizer with the smallest or largest step capacity along the background
  direction. `model_grid`, which ignores the data, is the baseline.
- **Refitting:** `fixed` keeps the choice made at `K_0`; `refit` repeats the rule on
  fresh draws each round.
- **Target:** `background` or `boundary`, with mixture 0.5 and band half-width `0.1 h`.

All arms use the `capacity` update of the family study, three rounds, `H = 10`, `Q = I`,
`R = 0.1 I`, `K_0 = I`, and `h` at the 53.9% quantile of initial reference costs.

Fitting draws use stream `type_index * 10000 + family_index`, offset by round as in the
original study. Within a family and round every arm sees the same draws, so
differences between arms are not sampling noise.

## Failures stay in the study

A rule can fail: `erm_first` may pick an unverified matrix, and the capacity rules may
find no verified tie. As in the original study, such a branch halts at its current gain
and keeps that coverage. These outcomes are part of what a rule does, so they count in
every mean, and each cell also reports its halted share.

A family with no verified grid candidate has no shared certificate for any rule. It is
recorded and left out of the means, as in the family study.

## Outcomes

Coverage is measured on the 512-condition reference law. Per type, over families with
a shared certificate:

- each cell's mean final coverage, with a percentile bootstrap 95% interval over
  families (10,000 resamples), its halted share, and its share at 100%;
- target effect: boundary minus background, for each rule and refitting mode, after one
  and three rounds;
- refit effect: refit minus fixed, for each rule and target, after three rounds;
- tie-break effect: `erm_large_step` minus `erm_small_step`, and `erm_large_step` minus
  `erm_first`, for each mode and target, after three rounds;
- at `K_0`: tied and verified tied candidates, and the share of families where the two
  capacity rules pick different matrices.

Each paired effect reports its mean, bootstrap interval, sign shares, and an exact sign
test. The original pair is not rerun; its values come from the family study.

## Regression check

`model_grid` with `fixed` uses the same families and update as the family study, so
its coverage must match that study exactly on every shared family. Analysis stops if
it does not.

## Run and analyze

```bash
uv run python scripts/linear_crossed_study.py run --preset full --jobs 16 \
  --output artifacts/linear-crossed-study-v1
uv run python scripts/linear_crossed_study.py analyze \
  --run artifacts/linear-crossed-study-v1 \
  --family-run artifacts/linear-family-study-v1 \
  --output artifacts/linear-crossed-study-v1-report
```

Run and report directories must be new. `raw/` keeps every accepted gain and its full
reference-law states in float64. Analysis reads only these files and checks their hashes.

## Limits

Random plants rarely have exact empirical ties, so tie-break rules may coincide there;
the tie counts show how often. Conclusions apply to these sampling rules and three
rounds. Nothing here extends to nonlinear plants or PPO.
