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

## Supported environments

- `CartPole-v1`: discrete action, interpretable four-value control state.
- `Pendulum-v1`: continuous action and continuous control performance.
- `MiniGrid-Empty-8x8-v0`: dictionary observation and discrete navigation action.

CHE-47 intentionally supplies only a constant-zero policy and a no-op
perturbation. Heuristic and learned policies, non-nominal perturbations, sweep
runners, and plots can be added later without changing the rollout or artifact
contracts.

## Development

```bash
uv run pytest
uv run ruff check .
```
