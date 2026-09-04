# Frozen policies

CHE-48 treats policy production and environment resolution as independent data
flows:

```text
PolicyDesignSpec -> model checkpoint -> PolicyArtifactMetadata -> freeze record
NominalEnvSpec + PerturbationSpec -> ResolvedEnvSpec -> raw trajectory -> metrics
```

An evaluation command only loads a previously frozen checkpoint. It never calls
Stable-Baselines3 `learn()` and never recomputes an LQR gain. The freeze record is
bound to SHA-256 digests of both `manifest.json` and the model. Loading rejects a
missing freeze record, altered model, altered manifest, mismatched policy ID, or
unsupported artifact format.

## CartPole quantized LQR

The state order is

```text
x = [cart position, cart velocity, pole angle, pole angular velocity]
```

with the upright equilibrium at zero. The model uses Gymnasium's nominal values
`g=9.8`, cart mass `1.0`, pole mass `0.1`, half-length `0.5`, force magnitude
`10.0`, and integration interval `tau=0.02`. Linearizing Gymnasium's nonlinear
acceleration equations at the origin gives continuous matrices `A_c` and `B_c`.
The installed environment uses explicit Euler, so the discrete model is

```text
A_d = I + tau A_c
B_d = tau B_c
```

The tracked design sets `Q=diag(1, 0.1, 10, 0.1)` and `R=[[0.1]]`.
[`python-control.dlqr`](https://python-control.readthedocs.io/en/stable/generated/control.dlqr.html)
solves the discrete Riccati equation once at build time.
`model.json` records the continuous and discrete matrices, `Q`, `R`, gain `K`,
Riccati solution, and closed-loop eigenvalues. A central finite-difference
linearization of Gymnasium's exact transition independently checks the analytic
matrices.

At inference, the controller computes `u_desired = -Kx`, maps nonnegative values
to action `1` and negative values to action `0`, and records both
`desired_force` and the applied `+/-10 N` force with every raw transition.

## Learned policies

- `cartpole_dqn_nominal_v1` uses [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/)
  DQN with the tracked [RL Zoo CartPole parameters](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml),
  seed 0, a requested budget of 100,000 environment steps, and greedy
  deterministic inference.
- `cartpole_dqn_nominal_v2` is a separately versioned diagnostic design that
  changes only the requested budget to the upstream 50,000-step endpoint. It is
  not a replacement baseline because its unchanged nominal gate also failed.
- `cartpole_ppo_nominal_v1` is the learned CartPole replacement baseline. It uses
  the RL Zoo CartPole PPO settings: eight environments, 100,000 requested steps,
  32-step rollouts, batch size 256, 20 epochs, gamma 0.98, GAE lambda 0.8, and
  linear schedules from learning rate 0.001 and clip range 0.2. The schedule
  descriptions remain serializable in the manifest and are resolved to callables
  only inside the training command.
- `pendulum_sac_nominal_v1` uses Stable-Baselines3 SAC with seed 0, a requested
  budget of 50,000 steps,
  a `1e-3` learning rate, and all other important SB3 defaults made explicit in
  TOML. Evaluation uses the deterministic actor action.
- `minigrid_empty8x8_ppo_partial_image_v1` uses Stable-Baselines3 PPO with seed 0,
  500,000 total steps across eight deterministically seeded `DummyVecEnv`
  environments. Both training and evaluation apply `ImgObsWrapper` followed by
  `FlattenObservation`, preserving the agent's partial view as a fixed
  147-value vector. There is no recurrent state or reward shaping.

Every design records the nominal environment and reset distribution, observation
adapter, algorithm and library versions, seed, budget, device, and resolved
hyperparameters in its manifest. Learned artifacts store the final checkpoint;
there is no evaluation-informed checkpoint selection.

Stable-Baselines3 treats `total_timesteps` as a lower bound when an algorithm must
finish a rollout or collection interval. Training summaries therefore distinguish
the requested budget from the model's actual final `num_timesteps`; the design
spec continues to record the requested experimental budget. New training runs also
record aggregate completed-episode statistics for diagnosis only; those values do
not select or mutate the final checkpoint.

## Environment and episode records

`NominalEnvSpec` is the requested tracked configuration. At environment creation,
`ResolvedEnvSpec` captures the actual accessible primitive and derived parameters,
wrapper/observation contract, initial-state distribution, and time limit. A
mismatch causes construction to fail before rollout.

Episode metadata schema v3 embeds the same full provenance as schema v2 and adds
interpretable environment state to the reset and every transition. This matters
for MiniGrid, where the policy-facing partial image does not reveal global agent
position. Schema v4 preserves `action` as the policy request, adds
`environment_action` as the action delivered to Gym, and records reset/transition
perturbation diagnostics. It also records the NumPy version used by stochastic
perturbations. New raw bundles include `trajectory.sha256`; historical schemas
remain valid artifacts and are not migrated.

CHE-49 analysis reads and verifies those raw bundles without constructing an
environment or loading a policy. Its `episode-summary-v2` outputs live under a
metric-version directory, leaving raw trajectories and earlier metrics untouched.
The secondary mass and stretch suites use `episode-summary-v3`: CartPole state
metrics remain based on true simulator state, action-switch rate uses delivered
actions, and separate requested-switch, requested/applied mismatch, realized-
dropout, and pole-angle observation-error metrics expose interface interventions.

Pole-angle noise and action dropout use dedicated NumPy generators derived from
the episode seed and fixed perturbation-specific stream IDs. This keeps their
random streams reproducible and independent of policy implementation. Paired
validation requires equal resolved environments and initial true state, then
compares stochastic draws across LQR and PPO through the shorter trajectory.

## Quality gates and reproduction

Every immutable nominal protocol uses evaluation seeds 0 through 19:

- CartPole policies: at least 95% time-limit success and mean length at least 475
  steps.
- Pendulum SAC: mean return at least -250. Pendulum has no invented binary success
  field.
- MiniGrid PPO: 100% goal success and mean length at most 100 steps.

Success means truncation at 500 steps without termination for CartPole, and goal
termination for MiniGrid Empty. A failed gate preserves the candidate and writes
`freeze-failure.json`; it never silently changes the algorithm, budget, or design.
Any hyperparameter change requires a new design and policy version.

The original `nominal-v1` protocol retains the failed DQN baseline and is not
materialized as a partial result. The completed `nominal-v2` protocol replaces
DQN with frozen CartPole PPO, providing a same-task comparison between model-based
LQR and a learned policy while retaining Pendulum SAC and MiniGrid PPO.

Reproduce the environment and run the commands shown in the README with:

```bash
uv sync --locked --group dev
uv run pytest
uv run ruff check .
```

Generated policy directories contain `manifest.json`, `model.json` or
`model.zip`, `training-summary.json`, and either `freeze.json` or a failure
report. Generated checkpoints and evaluation bundles are intentionally ignored by
Git. Local artifact hashes and nominal acceptance outcomes are recorded in
[`findings.md`](findings.md).
