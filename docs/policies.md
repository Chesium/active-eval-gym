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
spec continues to record the requested experimental budget.

## Environment and episode records

`NominalEnvSpec` is the requested tracked configuration. At environment creation,
`ResolvedEnvSpec` captures the actual accessible primitive and derived parameters,
wrapper/observation contract, initial-state distribution, and time limit. A
mismatch causes construction to fail before rollout.

Episode metadata schema v2 embeds the full design and artifact metadata, nominal
environment, explicit no-op perturbation, resolved environment, evaluation seed,
deterministic flag, package versions, and realized initial state. Schema-v1 files
remain valid historical artifacts and are not migrated. Raw `trajectory.jsonl`
files remain separate from versioned `metrics.json`, whose source hash identifies
the exact trajectory used.

## Quality gates and reproduction

The immutable nominal suite uses evaluation seeds 0 through 19:

- CartPole LQR and DQN: at least 95% time-limit success and mean length at least
  475 steps.
- Pendulum SAC: mean return at least -250. Pendulum has no invented binary success
  field.
- MiniGrid PPO: 100% goal success and mean length at most 100 steps.

Success means truncation at 500 steps without termination for CartPole, and goal
termination for MiniGrid Empty. A failed gate preserves the candidate and writes
`freeze-failure.json`; it never silently changes the algorithm, budget, or design.
Any hyperparameter change requires a new design and policy version.

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
