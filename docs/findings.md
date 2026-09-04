# CHE-48 and follow-up nominal findings

This file records the local acceptance run performed from the tracked CHE-48
designs on 3 September 2026. Results come from the independent seeds 0 through 19
and are not inferred from training rewards.

## Acceptance status

| Policy | Model SHA-256 | Mean return | Mean length | Success | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| `cartpole_lqr_nominal_quantized_v1` | `e3464a8489560dd88fcac5f9ae676983e99a11576b821ff4acb680e55819a2c1` | 500.000 | 500.0 | 100% | pass |
| `cartpole_dqn_nominal_v1` | `db50f5da54bed59a96e45e99af375901369f3daf70b524b74418f4f49dada84a` | 123.400 | 123.4 | 0% | **fail** |
| `cartpole_dqn_nominal_v2` | `a3da508261c696c334155d4cdce3bbda426e48d795852378283a511cc57ca13c` | 19.050 | 19.05 | 0% | **fail** |
| `cartpole_ppo_nominal_v1` | `37c9bac10919b2d6bdc833fd5a3a920460d4b80f09392ef7fb758e4777a856e8` | 500.000 | 500.0 | 100% | pass |
| `pendulum_sac_nominal_v1` | `c13cd159632f3c12e88c2c50c6344b3e4bc35601065e0c32b554b4f6d97c891c` | -129.249 | 200.0 | n/a | pass |
| `minigrid_empty8x8_ppo_partial_image_v1` | `018f0162bbf44835b5d2821702e59b9fb92146cbd909ad0f50e308559c2ccaea` | 0.958 | 12.0 | 100% | pass |

Stable-Baselines3 completed collection boundaries after reaching the requested
lower-bound budgets. The original saved models report 100,096 DQN steps, 50,000
SAC steps, and 500,736 MiniGrid PPO steps for requested budgets of 100,000,
50,000, and 500,000. CartPole PPO reports 100,096 actual steps for its 100,000-step
request.

The separately versioned DQN v2 requested the upstream RL Zoo budget of 50,000
steps and stopped at a collection boundary of 50,176. No other design field
changed. It also failed the original gate: episode lengths ranged from 14 to 25,
with mean 19.05 and no successes. Its final 20 recorded training episodes show a
collapse from longer episodes to lengths mostly between 12 and 20, so the poor
frozen-policy result is consistent with the end of training rather than an
evaluation serialization error.

The exact DQN v1 candidate did not meet either declared CartPole threshold: its
episode lengths ranged from 114 to 136, rather than reaching the 500-step time
limit. It remains preserved with `freeze-failure.json`; no `freeze.json` was
created. The original other three candidates are frozen.

Consequently, the required four-policy, 80-episode `nominal-v1` suite was not
written. This is an intentional integrity constraint: `evaluate-nominal`
preflights every artifact, rejects the unfrozen DQN, and leaves no partial output
directory. Changing the DQN budget or hyperparameters requires a new policy design
version rather than silently replacing this result. DQN v2 was that single
predeclared follow-up; after its failure, no further seed or hyperparameter search
was performed.

## Completed nominal-v2 suite

The separately predeclared CartPole PPO replacement passed with 20 of 20
time-limit successes and mean length 500. `nominal-v2` therefore contains frozen
CartPole LQR, CartPole PPO, Pendulum SAC, and MiniGrid PPO. Its 80 episode bundles
were written to `artifacts/evaluations/nominal-v2/`; each policy passed its gate:

- CartPole LQR: 100% success, mean return and length 500.
- CartPole PPO: 100% success, mean return and length 500.
- Pendulum SAC: mean return -129.249 and mean length 200.
- MiniGrid PPO: 100% success, mean return 0.9578125 and mean length 12.

## Seed-0 inspection

The render command was run in human mode for each frozen artifact using seed 0.
In this headless development session SDL used its dummy video driver; behavior was
also checked from the underlying states, actions, and termination flags.

- The quantized LQR alternated discrete forces to keep the pole near upright and
  the cart near center for all 500 steps. Desired force ranged from about -2.41 N
  to 2.18 N, while the applied action remained the environment's +/-10 N.
- CartPole PPO also completed all 500 steps. It used 250 actions in each direction;
  the cart stayed between approximately -0.115 and 1.021, and the pole angle stayed
  between approximately -0.166 and 0.138 radians.
- SAC started at an angle of about 0.861 rad, swung/stabilized the pendulum at
  approximately 0.003 rad by the final step, and returned -126.161 for seed 0.
- PPO turned south, moved from `(1, 1)` to `(1, 6)`, turned east, and moved to the
  goal at `(6, 6)`. It terminated in 12 actions with return 0.9578125.
- DQN v1 could not be loaded by the frozen-only render command. A read-only
  candidate rollout alternated both actions initially, drifted to the negative
  cart limit, and terminated at step 114 with cart position about -2.413. This
  agrees with its failed aggregate gate. DQN v2 is likewise intentionally
  unavailable to the frozen-only renderer.

The exact gate records and checkpoints live under the ignored local
`artifacts/policies/` tree. Because each manifest records a dirty source state,
these hashes identify this local acceptance run, not a portable checked-in binary
release.

# CHE-49 mandatory perturbation findings

The mandatory CHE-49 run collected 2,040 raw schema-v3 episodes from the frozen
`nominal-v2` artifacts. All trajectory digests verified before analysis. The
policy model hashes still match the CHE-48 table above, so none of the sweeps
changed or replaced a controller. These grid results are descriptive diagnostic
sampling, not estimates under a declared deployment distribution.

## CartPole angle × length surface

The quantized nominal-model LQR survived to the 500-step time limit on all
900 condition-seed pairs. PPO survived on 826 of 900 pairs (91.78%). Its
failures were highly asymmetric in the seeded angle offset:

| Angle offset | LQR success | PPO success | PPO mean length |
| ---: | ---: | ---: | ---: |
| -8 deg | 100% | 49% | 288.34 |
| -6 deg | 100% | 83% | 430.51 |
| -4 deg | 100% | 97% | 488.11 |
| -2 through +6 deg | 100% | 100% | 500.00 |
| +8 deg | 100% | 97% | 487.30 |

The weakest PPO cells were `-8 deg` at lengths `0.35` and `0.50`, both with
40% success. Increasing length to `0.65` raised success at `-8 deg` to 70%, so
the two perturbation axes interact rather than producing a single monotone
distance-from-nominal effect.

Even where both policies achieved nominal success, their trajectories differed.
At zero offset and length `0.50`, LQR versus PPO mean RMS pole angle was
`0.00493` versus `0.06585` radians, mean RMS cart position was `0.0717` versus
`0.6814`, and mean action-switch rate was `0.643` versus `0.907`. Terminal
success alone therefore hides substantial closed-loop behavior differences.

![CartPole success surfaces](figures/che49_cartpole_success_surface.png)

![CartPole nominal slices](figures/che49_cartpole_nominal_slices.png)

### Three-policy angle × length rerun

The separately versioned `che49-cartpole-angle-length-v2` suite reran the same
45 conditions and seeds 0 through 19 with quantized LQR, the frozen PPO, and
`cartpole_ppo_nominal_v1_antisymmetrized_v1`. It produced 2,700 new raw
schema-v4 episodes. All trajectory digests verified. For each of the two
unchanged policies, all 900 states, observations, actions, rewards, and episode
endings reproduced the v1 scientific trace exactly; the serialized bytes differ
because v1 predates the schema-v4 action and diagnostic fields.

The third policy survived to the time limit in all 900 episodes:

| Angle offset | LQR success | PPO success | Antisymmetrized PPO success | Antisymmetrized PPO mean length |
| ---: | ---: | ---: | ---: | ---: |
| -8 deg | 100% | 49% | 100% | 500.00 |
| -6 deg | 100% | 83% | 100% | 500.00 |
| -4 deg | 100% | 97% | 100% | 500.00 |
| -2 through +6 deg | 100% | 100% | 100% | 500.00 |
| +8 deg | 100% | 97% | 100% | 500.00 |

At the nominal zero-offset, length-0.50 cell, antisymmetrization also changed the
subterminal behavior substantially:

| Policy | RMS pole angle | RMS cart position | Action-switch rate |
| --- | ---: | ---: | ---: |
| Quantized LQR | 0.00493 | 0.07169 | 0.64339 |
| Frozen PPO | 0.06585 | 0.68143 | 0.90671 |
| Antisymmetrized PPO | 0.00938 | 0.06405 | 0.74128 |

Thus the intervention did more than move terminal failures outside the sampled
grid: under nominal conditions it moved the learned controller much closer to
LQR's pole- and cart-regulation regime, while retaining a distinct action-switch
profile. This remains a diagnostic policy intervention, not a replacement frozen
baseline or evidence about performance beyond the declared grid.

![Three-policy CartPole success surfaces](figures/che49_cartpole_success_surface_three_policy.png)

![Three-policy CartPole nominal slices](figures/che49_cartpole_nominal_slices_three_policy.png)

## Pendulum length sweep

The frozen SAC policy degraded smoothly as the pole length increased:

| Length | Mean return | RMS angular error | RMS torque |
| ---: | ---: | ---: | ---: |
| 0.70 | -94.04 | 0.487 | 0.557 |
| 0.85 | -109.23 | 0.544 | 0.556 |
| 1.00 | -129.25 | 0.603 | 0.579 |
| 1.15 | -163.31 | 0.695 | 0.658 |
| 1.30 | -212.41 | 0.804 | 0.790 |

The continuous trajectory metrics reveal degradation without inventing a binary
Pendulum success threshold: longer plants have worse return and angular tracking,
with torque usage rising most clearly beyond the nominal length.

![Pendulum length sweep](figures/che49_pendulum_length_sweep.png)

## MiniGrid start-pose map

The frozen partial-image PPO reached the goal from 76 of 140 start
position-orientation pairs (54.29%). Every direction failed from the 16-cell
interior block `x in 2..5, y in 2..5`, while every tested direction succeeded
from the remaining 19 boundary-ring cells. Failed runs exhausted all 256 actions.
Successful runs took 1–24 actions, but path efficiency ranged from `0.053` to
`1.0`, showing that goal completion alone also conceals inefficient navigation.
Initial direction barely changed aggregate successful-run length (means
12.37–12.47 actions).

![MiniGrid start-pose map](figures/che49_minigrid_start_pose_map.png)

## Active-evaluation questions

1. With only 20 additional CartPole evaluations, should an evaluator concentrate
   on PPO's asymmetric boundary near negative angle offsets, or sample policy
   disagreement against the uniformly successful LQR?
2. Should a useful early-warning metric target terminal failure, trajectory
   degradation such as RMS state error, or the disagreement between those
   signals?
3. For MiniGrid, can an active evaluator infer the sharp interior failure region
   efficiently from sparse start-pose queries while maintaining a fixed target
   distribution over cells and directions?

# CHE-49 secondary and stretch CartPole findings

The five independent follow-up suites added 1,000 raw schema-v4 episodes: five
conditions, 20 paired seeds, and two frozen policies per suite. Analysis used
`episode-summary-v3`; all trajectory hashes and paired-environment checks passed.
The nominal condition in every suite reproduced the no-op states, observations,
requested and delivered actions, rewards, and episode endings. Model hashes still
match the CHE-48 table above.

## Pole mass and force magnitude

Both controllers survived all 500 steps in every mass and force episode. Across
pole masses `0.05` to `0.15`, mean LQR RMS pole angle stayed between `0.00488`
and `0.00494` radians; PPO stayed between `0.06434` and `0.06764`. This sampled
mass range therefore did not separate the policies on success or materially on
state regulation.

Force magnitude also left success at 100%, but it changed how the fixed policies
used the binary actuator. From 5 N to 15 N, LQR mean action-switch rate fell from
`0.744` to `0.598`, while PPO rose from `0.832` to `0.911` after peaking at
`0.917` at 12.5 N. A terminal-only metric would miss that systematic change in
closed-loop behavior.

![CartPole pole-mass sweep](figures/che49_cartpole_mass_sweep.png)

![CartPole force-magnitude sweep](figures/che49_cartpole_force_magnitude_sweep.png)

## Pole-angle observation noise

The realized RMS observation errors were `0`, `0.00869`, `0.01737`, `0.03475`,
and approximately `0.06949` radians for the five requested standard deviations,
and paired LQR/PPO streams matched through the shorter trajectory. LQR retained
100% success at every level. PPO retained 100% through 2 degrees, then reached
95% success and mean length `479.25` at 4 degrees. At that endpoint mean true-
state RMS pole angle was `0.02862` for LQR and `0.07697` radians for PPO.

![CartPole pole-angle-noise sweep](figures/che49_cartpole_pole_angle_noise_sweep.png)

### Three-policy pole-angle-noise rerun

The separately versioned `che49-cartpole-pole-angle-noise-v2` suite added the
antisymmetrized PPO on the identical conditions and seeds. It produced 300 raw
schema-v4 episodes, and all trajectory digests verified. The 100 LQR and 100
frozen-PPO `trajectory.jsonl` files were byte-identical to v1, so the comparison
adds a policy without changing the earlier experiment.

| Noise standard deviation | LQR success | PPO success | Antisymmetrized PPO success | Antisymmetrized PPO RMS pole angle |
| ---: | ---: | ---: | ---: | ---: |
| 0 deg | 100% | 100% | 100% | 0.00938 |
| 0.5 deg | 100% | 100% | 100% | 0.01207 |
| 1 deg | 100% | 100% | 100% | 0.01829 |
| 2 deg | 100% | 100% | 100% | 0.02272 |
| 4 deg | 100% | 95% | 100% | 0.03912 |

At 4 degrees the intervention removed the one observed PPO failure and retained
mean length 500. Its mean true-state RMS pole angle was `0.03912` radians,
between LQR's `0.02862` and PPO's `0.07697`; its RMS cart position was `0.10827`,
compared with `0.11793` and `0.68821`. On this finite sample, enforcing actor
symmetry therefore improved robustness to symmetric observation noise as well as
to the signed initial-angle perturbation.

![Three-policy CartPole pole-angle-noise sweep](figures/che49_cartpole_pole_angle_noise_sweep_three_policy.png)

## Action delay and dropout

Delay produced the sharpest policy separation. LQR retained 100% success through
four delayed steps, although its mean RMS pole angle rose from `0.00493` to
`0.09322` radians. PPO fell to 95% success at one step and 0% at two through four
steps; its mean episode lengths at those failing conditions were `100.00`,
`44.95`, and `18.45`. Thus the continuous state metric warns of LQR degradation
even where its binary outcome remains unchanged, while PPO crosses a terminal
failure boundary between one and two delayed steps.

Dropout degraded both controllers monotonically in the sampled range. At
probabilities `0`, `0.05`, `0.10`, `0.20`, and `0.40`, LQR success was 100%,
100%, 95%, 90%, and 20%; PPO success was 100%, 90%, 65%, 40%, and 0%. At the
0.40 endpoint, mean episode length was `253.85` for LQR and `101.75` for PPO.
Realized dropout rates tracked the requested probabilities, while requested/
delivered mismatches were lower because dropping a request that already equals
the held action does not create a mismatch.

![CartPole action-delay sweep](figures/che49_cartpole_action_delay_sweep.png)

![CartPole action-dropout sweep](figures/che49_cartpole_action_dropout_sweep.png)

### Three-policy action-channel reruns

The versioned action-delay-v2 and action-dropout-v2 suites each added the same
hash-bound antisymmetrized PPO, again producing 300 raw schema-v4 episodes per
suite. All hashes verified, and all 400 unchanged-policy trajectory files were
byte-identical to their v1 counterparts.

| Delay | LQR success | PPO success | Antisymmetrized PPO success | PPO mean length | Antisymmetrized mean length |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 steps | 100% | 100% | 100% | 500.00 | 500.00 |
| 1 step | 100% | 95% | 100% | 485.65 | 500.00 |
| 2 steps | 100% | 0% | 35% | 100.00 | 414.85 |
| 3 steps | 100% | 0% | 0% | 44.95 | 204.90 |
| 4 steps | 100% | 0% | 0% | 18.45 | 115.95 |

Antisymmetrization shifted the sampled PPO delay boundary outward: it retained
100% success at one step and 35% at two steps, where the source PPO had no
successes. It did not recover LQR's robustness, but even at three and four steps
it extended mean survival substantially.

| Dropout probability | LQR success | PPO success | Antisymmetrized PPO success | Antisymmetrized mean length |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 100% | 100% | 100% | 500.00 |
| 0.05 | 100% | 90% | 100% | 500.00 |
| 0.10 | 95% | 65% | 95% | 494.85 |
| 0.20 | 90% | 40% | 45% | 411.80 |
| 0.40 | 20% | 0% | 5% | 137.50 |

For dropout, the intervention matched LQR's observed success through probability
`0.10`, then provided only a small advantage over PPO at `0.20` and `0.40` and
remained well below LQR. Its RMS pole angle nevertheless stayed below PPO's at
every nonzero level. These results suggest that repairing the actor's reflection
symmetry removes one source of brittleness but does not supply the feedback
memory or robustness margin needed for severe action-channel faults. RMS values
for failed conditions must be interpreted with episode length because they are
computed over unequal, failure-truncated horizons.

![Three-policy CartPole action-delay sweep](figures/che49_cartpole_action_delay_sweep_three_policy.png)

![Three-policy CartPole action-dropout sweep](figures/che49_cartpole_action_dropout_sweep_three_policy.png)

## Follow-up active-evaluation questions

1. Can an evaluator identify PPO's delay boundary near one to two steps with
   fewer samples than a uniform sweep while still detecting LQR's subterminal
   state degradation?
2. Should intervention severity be represented by requested delay/dropout
   parameters, realized random events, or requested/applied action mismatch when
   those measures disagree?
3. Given the flat success surfaces for mass and force, when should an evaluator
   stop probing an axis and redirect budget toward observation or action-channel
   perturbations?

# CHE-49 CartPole PPO symmetry study

The separately versioned `che49-cartpole-symmetry-v1` study audited the frozen
`cartpole_ppo_nominal_v1` checkpoint, collected exact mirror-paired closed-loop
trajectories, and evaluated an antisymmetrized inference intervention. The PPO
model hash remained
`37c9bac10919b2d6bdc833fd5a3a920460d4b80f09392ef7fb758e4777a856e8`;
the intervention did not retrain or change its weights. All 2,500 new raw
schema-v4 trajectory digests verified after collection.

For CartPole state
`s = [x, x_dot, theta, theta_dot]`, physical reflection maps `s` to `-s` and
swaps actions 0 and 1. Actor symmetry therefore requires
`pi(1 | s) = 1 - pi(1 | -s)`, or equivalently an odd binary logit margin
`d(s) = -d(-s)`. The critic should satisfy `V(s) = V(-s)`.

## Frozen-checkpoint audit and decision boundary

The PPO actor does not satisfy the action-distribution identity. At the origin it
assigned action probabilities `[0.9999904, 0.00000959]`; its action-1-minus-action-0
logit margin was `-11.5549`, whereas stochastic reflection symmetry requires both
probabilities to be `0.5` and the margin to be zero.

The audit evaluated the existing angle-length trajectory states and their exact
counterfactual mirrors without advancing an environment:

| Probe set | States | Mean absolute probability error | Deterministic action violation | Mean absolute value error |
| --- | ---: | ---: | ---: | ---: |
| Angle-length reset states | 900 | 0.94873 | 96.11% | 0.00866 |
| Saved policy-input states, stride 25 | 16,825 | 0.52575 | 52.22% | 0.42236 |

Probability error is `abs(pi(1 | s) + pi(1 | -s) - 1)` and is zero for a
symmetric actor. A deterministic violation means that the argmax action was the
same at `s` and `-s`, rather than complementary. The critic is also not exactly
symmetric, although its reset-state error is much smaller than the actor's
distribution error.

One-dimensional slices show a substantially displaced deterministic boundary.
With all other state components zero, the action changes only near cart position
`+0.70`, cart velocity `+0.70`, pole angle `+0.179` radians, or pole angular
velocity `+0.28` radians. An odd logit margin would instead pass through zero at
the origin. The two-dimensional slices below show the observed action boundary
as a solid curve and the boundary required by evaluating reflected states as a
dashed curve; their separation visualizes the nonzero even component of the
actor.

![CartPole PPO policy-symmetry audit](figures/che49_cartpole_ppo_symmetry_audit.png)

## Exact mirror-paired rollouts

The paired experiment used 20 seeds, angle magnitudes 2, 4, 6, and 8 degrees,
and all five pole lengths. For each seeded positive state `s`, a structured
fixed-state perturbation created a second episode at exactly `-s`. Each policy
therefore had 400 pairs and 800 raw episodes. The unmodified Gymnasium dynamics
receive complementary actions in a physically symmetric pair.

| Policy | First action equivariant | Fully action equivariant | Equal episode length | Positive success | Reflected success | Mean absolute length difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Quantized LQR | 100% | 100% | 100% | 100% | 100% | 0.00 |
| Frozen PPO | 1.25% | 0% | 81.25% | 99.25% | 81.25% | 73.64 |

LQR's paired states remained exact negatives up to floating-point comparison in
all 400 pairs, providing a positive control for both the environment symmetry and
the fixed-state protocol. PPO violated the required action swap on the first
decision in 395 of 400 pairs. Its paired success outcomes agreed on 82% of pairs,
with a pronounced 18-percentage-point success difference between the two exact
mirror branches. This establishes closed-loop symmetry breaking without relying
on the earlier seeded `+/-` angle-offset comparison, whose individual states were
not exact mirrors.

## Antisymmetrized-policy intervention

The diagnostic policy computes the frozen actor's binary margins at both `s` and
`-s` and acts from

```text
d_sym(s) = 0.5 * (d(s) - d(-s)).
```

This gives an odd margin by construction while retaining the frozen network as
its only learned component. It is recorded under the distinct derived policy ID
`cartpole_ppo_nominal_v1_antisymmetrized_v1`, with the source policy ID, model
hash, transformation version, and `weights_changed = false` in every episode's
metadata.

The intervention was evaluated on the same 900 signed-angle, pole-length, and
seed combinations as the original sweep. Initial states matched the saved source
episodes exactly:

| Seeded angle offset | Original PPO success | Antisymmetrized PPO success | Original mean length | Antisymmetrized mean length |
| ---: | ---: | ---: | ---: | ---: |
| -8 deg | 49% | 100% | 288.34 | 500.00 |
| -6 deg | 83% | 100% | 430.51 | 500.00 |
| -4 deg | 97% | 100% | 488.11 | 500.00 |
| -2 through +6 deg | 100% | 100% | 500.00 | 500.00 |
| +8 deg | 97% | 100% | 487.30 | 500.00 |

The original positive-minus-negative success gaps at magnitudes 4, 6, and 8
degrees were 3, 17, and 48 percentage points. All became zero after the
intervention. Together with the exact mirror experiment, this is strong causal
evidence that the frozen actor's asymmetric decision boundary drives the sampled
directional failures, rather than the asymmetry coming from CartPole dynamics or
the perturbation implementation.

The conclusion is limited to this frozen checkpoint and declared grid. A
deterministic binary controller cannot choose a self-reflecting action at the
exact origin because CartPole has no neutral action; this is why the categorical
distribution and logit margin are the primary symmetry objects. Finally, the
study establishes that the final trained checkpoint is asymmetric, but not when
that asymmetry arose. Fixed-interval training snapshots or predeclared replicate
training seeds are still required to distinguish optimization-induced symmetry
breaking from an asymmetric initialization.
