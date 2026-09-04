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
