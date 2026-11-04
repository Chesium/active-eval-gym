# Why uniform frontier selection produces uneven PPO exposure

This is a descriptive follow-up to Section 13.6 of the post-midterm plan, using
the v3 pilot's existing discovery results, training summaries, and raw training
transitions. No policies were trained or evaluated for this investigation.

The observed transition imbalance is real. The acceptance-boundary arm samples
the parent's frontier uniformly **at episode reset**, but PPO receives far more
transitions from its lower edge. This is a plausible contributor to the drift,
not evidence that changing exposure would prevent it. Exposure also depends on
the policy's behavior, so it can be a consequence of regression.

## What the current distribution means

The v3 study uses `ppo-boundary-targets-v2`, inherited from v2. It does not use
the original probability band around 0.9, which was empty in discovery.

1. Evaluate the frozen parent at 143 discovery conditions: 11 angles crossed with
   13 half-lengths, with 32 episodes each.
2. Mark a discovery condition accepted when its unsmoothed recovery count divided
   by 32 is at least 0.90. There are 15 such conditions.
3. Mark **unaccepted** cells that are one grid index away from an accepted cell
   along angle or length. Diagonal neighbours are excluded. There are 15 frontier
   cells. Grid adjacency does not mean equal physical distance or arc length.
4. Give each frontier cell target weight `1/15`, then mix 50% target with 50%
   uniform background over all 143 cells:

   ```text
   q_boundary(z) = 0.5 / 143 + 0.5 * 1[z is on the frontier] / 15
   ```

   Every frontier cell therefore has probability 3.683%; every other cell has
   probability 0.350%. The frontier receives 55.245% of resets in total, including
   its share of background sampling.
5. At each environment reset, draw a condition from this fixed categorical
   distribution. Hold that condition for the entire episode. Reset again on
   termination or the 500-step time limit. PPO uses 256 transitions per rollout
   (`8 environments * 32 steps`), with 20 epochs per rollout.

The same `q` is used throughout all 102,400 additional transitions. Neither
snapshots nor newly accepted cells refresh it. It targets the parent's frontier,
not the evolving child's frontier. The accepted interior has no extra retention
weight: the 15 accepted discovery cells together get only 5.245% of resets in
the targeted arms, versus 10.490% in background.

Construction is in `src/active_eval_gym/ppo_boundary_targets.py`, principally
`frontier_mask` and `construct_arms`. Sampling is in
`src/active_eval_gym/envs/condition_sampler.py`, principally `reset`. The
episode/transition distinction was already specified in Section 5 of the plan
and `realized_sampling` in `ppo_boundary_training.py`; this is not a newly found
implementation deviation from the declared protocol.

## Equal episode probability, unequal transition exposure

On this particular discovery map, the 15 frontier cells can be partitioned into
five lower cells, five upper cells, and five remaining angle-side cells. A corner
with vertical adjacency is assigned to its vertical side first.

- Lower: angles -20, -10, 0, +10, +20 degrees, all at half-length 0.3 m.
- Upper: (-20, 1.0), (-10, 1.5), (0, 1.5), (+10, 2.0), (+20, 1.5).
- Sides: (-30, 0.5), (-30, 0.75), (+30, 0.5), (+30, 0.75), (+30, 1.0).

For the acceptance-boundary arm, pooling the three equal-budget training runs:

| Parent frontier part | Intended reset share | Observed reset share | Completed episodes | Mean completed length | Observed transition share |
|---|---:|---:|---:|---:|---:|
| Lower, 5 cells | 18.41% | 18.25% | 302 | 500.0 | 50.51% |
| Upper, 5 cells | 18.41% | 18.19% | 316 | 41.8 | 4.30% |
| Angle sides, 5 cells | 18.41% | 18.60% | 321 | 131.4 | 13.74% |

The reset counts support correct uniform frontier sampling. The lower group
receives 11.75 times the upper group's transitions. All 302 completed lower
episodes reach the time limit; none of the 316 upper episodes does. Completed
episode statistics exclude episodes unfinished at the end of training.
Transition shares include all recorded steps, including those partial episodes.

For a stationary policy, the expected fraction of transitions from a condition
is approximately:

```text
transition_share(z) = q(z) * E[episode_length | z]
                      / sum_u q(u) * E[episode_length | u]
```

This explains the length effect; it is not an exact stationary model of the
changing policy during training. Transition frequency also does not equal
gradient influence: advantages, clipping, and the visited states matter.

The asymmetry predates continuation. In the parent's deterministic discovery
evaluation, all five lower frontier cells survive 32/32 and run 500 steps.
None of the upper five survives; their mean lengths range from 36.7 to 68.9
steps, averaging 52.4 across those five cells. They are geometrically adjacent
to acceptance but differ greatly in survival difficulty. These deterministic
discovery lengths are separate from the stochastic training lengths above.

The lower group is already dominant in the first training interval:

| New training transitions | Lower frontier transition share | Upper frontier transition share | Angle sides transition share |
|---|---:|---:|---:|
| 0–25,600 | 50.97% | 5.30% | 15.67% |
| 25,600–51,200 | 47.10% | 4.33% | 13.26% |
| 51,200–76,800 | 51.61% | 4.05% | 17.33% |
| 76,800–102,400 | 52.38% | 3.51% | 8.72% |

These are interval shares averaged over the three replicates. The lower frontier
share stays around half; it does not steadily accumulate. Across all short-pole
conditions, further concentration is clearer:

| Arm | Fixed reset probability at half-length ≤0.3 m | Transition share, first interval | Transition share, final interval |
|---|---:|---:|---:|
| Background | 53.85% | 54.15% | 71.58% |
| Acceptance boundary | 43.59% | 65.30% | 72.18% |
| Intermediate | 55.22% | 64.69% | 82.84% |

Even background is uniform over **grid cells**, rather than over physical length:
7 of its 13 length values are at or below 0.3 m. It is not a control with equal
training probability above and below the parent's acceptance set.

## The Intermediate arm is a different selection hypothesis

The Intermediate arm is intentionally not uniform on the frontier. Its target
is proportional to `p_tilde * (1-p_tilde)` over cells with `0 < recoveries < 32`,
where `p_tilde = (recoveries + 0.5) / 33`, again mixed 50:50 with background.

Only five cells have mixed discovery outcomes:

| Angle, half-length | Recoveries / 32 | Final episode-selection probability q |
|---|---:|---:|
| -30°, 0.5 m | 5 | 10.50% |
| 0°, 0.2 m | 1 | 3.52% |
| 0°, 0.3 m | 17 | 18.55% |
| +10°, 0.3 m | 3 | 7.28% |
| +30°, 0.5 m | 26 | 11.91% |

All upper frontier cells have zero discovery recovery and receive only background
mass in this arm. Together they receive 1.75% of episode probability and 0.31%
of actual training transitions. The strong lower-edge emphasis in this row of
the figure is therefore partly designed into `q`, as well as length-weighted.

## What this establishes about the drift

The data establish an exposure mechanism: equal frontier reset probabilities
produce strongly unequal transition counts. The resulting short-pole emphasis
is consistent with the shared gains at 0.2–0.3 m and losses at longer poles.
It is stronger evidence than a visual impression alone, but the direction of
causation for the policy change remains unresolved.

- The length difference already exists for the parent, supporting an initial
  exposure imbalance rather than an imbalance entirely caused by later drift.
- As a policy loses long-pole competence, those episodes can shorten and their
  transition share can shrink further. The observed shift can thus reinforce
  itself without any change to `q`.
- All arms drift, including background. That is compatible with a shared
  episode-length effect and shared policy updates, but does not isolate either.
- The objective is still +1 survival, not recovery. Lower frontier episodes
  already survive for 500 steps even when recovery is poor. Their many steps
  do not by themselves imply a strong reward signal for recovery expansion.
- There is no constraint requiring the child's accepted set to contain the
  parent's. Uniform frontier selection alone cannot guarantee monotonic
  expansion or preserve the interior.

A useful next development comparison would separately test uniform **transition
exposure** on the fixed parent frontier against the present uniform-episode
sampler, keeping the policy, reward, budgets, and seeds matched. This could use
explicit stratum transition budgets, or reset probabilities inversely weighted
by independently estimated stochastic episode lengths. The latter only
approximately balances transitions and becomes stale as the policy changes.
Either intervention changes the training design and needs a new version.

Test extra sampling of previously accepted cells separately if the intended
outcome is retention plus outward expansion. Refreshing the frontier is another
separate change; it does not itself remove episode-length bias. None of these
changes has been implemented or shown to solve drift by this audit.

## Reproduce and checks

Run from the repository root:

```sh
MPLCONFIGDIR=/tmp/ppo-boundary-matplotlib .venv/bin/python \
  scripts/analyze_ppo_boundary_sampling.py \
  --study artifacts/experiments/cartpole-ppo-boundary-v3
```

The script writes `sampling-diagnosis/summary.json` inside the study, with group
definitions, per-replicate episode lengths, interval shares, and pooled results.
It reconstructs the saved target and mixture vectors exactly, independently
checks frontier adjacency, verifies all nine training shards' content digests,
matches raw transition counts to all 36 snapshot/final summaries, matches final
reset/completion counts, and verifies the single frozen distribution file hash.
