# CHE-48 nominal findings

This file records the local acceptance run performed from the tracked CHE-48
designs on 3 September 2026. Results come from the independent seeds 0 through 19
and are not inferred from training rewards.

## Acceptance status

| Policy | Model SHA-256 | Mean return | Mean length | Success | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| `cartpole_lqr_nominal_quantized_v1` | `e3464a8489560dd88fcac5f9ae676983e99a11576b821ff4acb680e55819a2c1` | 500.000 | 500.0 | 100% | pass |
| `cartpole_dqn_nominal_v1` | `db50f5da54bed59a96e45e99af375901369f3daf70b524b74418f4f49dada84a` | 123.400 | 123.4 | 0% | **fail** |
| `pendulum_sac_nominal_v1` | `c13cd159632f3c12e88c2c50c6344b3e4bc35601065e0c32b554b4f6d97c891c` | -129.249 | 200.0 | n/a | pass |
| `minigrid_empty8x8_ppo_partial_image_v1` | `018f0162bbf44835b5d2821702e59b9fb92146cbd909ad0f50e308559c2ccaea` | 0.958 | 12.0 | 100% | pass |

Stable-Baselines3 completed collection boundaries after reaching the requested
lower-bound budgets. The saved models report 100,096 DQN steps, 50,000 SAC steps,
and 500,736 PPO steps for requested budgets of 100,000, 50,000, and 500,000.

The exact DQN candidate did not meet either declared CartPole threshold: its
episode lengths ranged from 114 to 136, rather than reaching the 500-step time
limit. It remains preserved with `freeze-failure.json`; no `freeze.json` was
created. The other three candidates are frozen.

Consequently, the required four-policy, 80-episode `nominal-v1` suite was not
written. This is an intentional integrity constraint: `evaluate-nominal`
preflights every artifact, rejects the unfrozen DQN, and leaves no partial output
directory. Changing the DQN budget or hyperparameters requires a new policy design
version rather than silently replacing this result.

## Seed-0 inspection

The render command was run in human mode for each frozen artifact using seed 0.
In this headless development session SDL used its dummy video driver; behavior was
also checked from the underlying states, actions, and termination flags.

- The quantized LQR alternated discrete forces to keep the pole near upright and
  the cart near center for all 500 steps. Desired force ranged from about -2.41 N
  to 2.18 N, while the applied action remained the environment's +/-10 N.
- SAC started at an angle of about 0.861 rad, swung/stabilized the pendulum at
  approximately 0.003 rad by the final step, and returned -126.161 for seed 0.
- PPO turned south, moved from `(1, 1)` to `(1, 6)`, turned east, and moved to the
  goal at `(6, 6)`. It terminated in 12 actions with return 0.9578125.
- DQN could not be loaded by the frozen-only render command. A read-only candidate
  rollout alternated both actions initially, drifted to the negative cart limit,
  and terminated at step 114 with cart position about -2.413. This agrees with its
  failed aggregate gate.

The exact gate records and checkpoints live under the ignored local
`artifacts/policies/` tree. Because each manifest records a dirty source state,
these hashes identify this local acceptance run, not a portable checked-in binary
release.
