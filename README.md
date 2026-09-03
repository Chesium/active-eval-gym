# Active Eval Gym

A small research sandbox for evaluating fixed control and reinforcement-learning
policies under explicit environment perturbations. The core data flow is:

```text
policy + environment + perturbation -> raw trajectory -> derived metrics
```

Training, rollout collection, perturbation generation, and metric computation
remain separate so that evaluation never silently changes a policy and new metrics
can be computed without rerunning an episode.

## Five-minute quickstart

The project uses Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked --group dev
uv run active-eval-gym rollout \
  --env CartPole-v1 \
  --seed 123 \
  --output artifacts/cartpole-seed-123
```

The command prints a short episode summary and creates:

```text
artifacts/cartpole-seed-123/
├── metadata.json
├── metrics.json
└── trajectory.jsonl
```

`metadata.json` records the environment and package versions, policy provenance,
episode seed, no-op perturbation, and accessible initial state.
`trajectory.jsonl` contains the reset observation followed by every raw transition.
`metrics.json` contains versioned derived metrics and the SHA-256 digest of the raw
trajectory it describes.

Artifacts are never overwritten. Use a new output directory for another run. To
check deterministic replay, write the same rollout to two directories and compare
their raw files:

```bash
uv run active-eval-gym rollout \
  --env CartPole-v1 --seed 123 --output artifacts/replay-a
uv run active-eval-gym rollout \
  --env CartPole-v1 --seed 123 --output artifacts/replay-b
cmp artifacts/replay-a/metadata.json artifacts/replay-b/metadata.json
cmp artifacts/replay-a/trajectory.jsonl artifacts/replay-b/trajectory.jsonl
```

## Frozen policy zoo

CHE-48 adds four reproducible nominal policy designs. Policy construction is a
separate command from freezing, and evaluation accepts only an intact frozen
artifact. Generated checkpoints and results live under the ignored `artifacts/`
directory; the code and TOML files under `configs/` are the tracked recipe.

Build the analytic CartPole policy, and train the three learned candidates:

```bash
uv run active-eval-gym build-policy \
  --config configs/policies/cartpole_lqr_nominal_quantized_v1.toml \
  --output artifacts/policies/cartpole_lqr_nominal_quantized_v1
uv run active-eval-gym train-policy \
  --config configs/policies/cartpole_dqn_nominal_v1.toml \
  --output artifacts/policies/cartpole_dqn_nominal_v1
uv run active-eval-gym train-policy \
  --config configs/policies/pendulum_sac_nominal_v1.toml \
  --output artifacts/policies/pendulum_sac_nominal_v1
uv run active-eval-gym train-policy \
  --config configs/policies/minigrid_empty8x8_ppo_partial_image_v1.toml \
  --output artifacts/policies/minigrid_empty8x8_ppo_partial_image_v1
```

Each command creates a candidate and refuses to overwrite an existing path.
Validate candidates against the predeclared seeds and quality gates:

```bash
for policy in \
  cartpole_lqr_nominal_quantized_v1 \
  cartpole_dqn_nominal_v1 \
  pendulum_sac_nominal_v1 \
  minigrid_empty8x8_ppo_partial_image_v1
do
  uv run active-eval-gym freeze-policy \
    --artifact "artifacts/policies/$policy" \
    --suite configs/eval/nominal.toml
done
```

A passing candidate receives a hash-bound `freeze.json`. A failed candidate
receives `freeze-failure.json` and remains unavailable to evaluation. Once all
required policies are frozen, collect the fixed 20-seed suite:

```bash
uv run active-eval-gym evaluate-nominal \
  --suite configs/eval/nominal.toml \
  --artifact-root artifacts/policies \
  --output artifacts/evaluations/nominal-v1
```

To inspect one frozen policy interactively:

```bash
uv run active-eval-gym render-policy \
  --artifact artifacts/policies/cartpole_lqr_nominal_quantized_v1 \
  --seed 0
```

See [the policy design notes](docs/policies.md) for equations, wrappers, artifact
contracts, and the fixed-policy invariant.

## Supported environments

- `CartPole-v1`: discrete action, interpretable four-value control state.
- `Pendulum-v1`: continuous action and continuous control performance.
- `MiniGrid-Empty-8x8-v0`: dictionary observation and discrete navigation action.

The original constant-zero policy remains useful for rollout smoke tests. The
policy zoo adds a quantized LQR and three learned policies, all evaluated through
the same raw-trajectory path. Non-nominal perturbation sweeps, failure-surface
plots, oracle LQR, and optional PID remain outside CHE-48.

## Development

```bash
uv run pytest
uv run ruff check .
```
