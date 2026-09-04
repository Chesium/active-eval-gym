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

CHE-48 and its follow-up add reproducible nominal policy designs. Policy
construction is a separate command from freezing, and evaluation accepts only an
intact frozen artifact. Generated checkpoints and results live under the ignored
`artifacts/` directory; the code and TOML files under `configs/` are the tracked
recipe.

Build the analytic CartPole policy and train the learned candidates:

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
uv run active-eval-gym train-policy \
  --config configs/policies/cartpole_ppo_nominal_v1.toml \
  --output artifacts/policies/cartpole_ppo_nominal_v1
```

Each command creates a candidate and refuses to overwrite an existing path.
Validate candidates against the predeclared seeds and quality gates:

```bash
for policy in \
  cartpole_lqr_nominal_quantized_v1 \
  cartpole_ppo_nominal_v1 \
  pendulum_sac_nominal_v1 \
  minigrid_empty8x8_ppo_partial_image_v1
do
  uv run active-eval-gym freeze-policy \
    --artifact "artifacts/policies/$policy" \
    --suite configs/eval/nominal_v2.toml
done
```

A passing candidate receives a hash-bound `freeze.json`. A failed candidate
receives `freeze-failure.json` and remains unavailable to evaluation. The original
DQN candidate failed, so `nominal-v1` remains a preserved, blocked protocol.
CartPole PPO was first validated through the dedicated, predeclared
`configs/eval/cartpole_ppo_v1_freeze.toml` gate; `nominal-v2` carries the identical
gate.

The original 100k-step DQN and the separately versioned 50k-step follow-up both
failed the nominal gate in the recorded local run. The v2 recipe and its dedicated
unchanged gate can be reproduced without altering the original suite:

```bash
uv run active-eval-gym train-policy \
  --config configs/policies/cartpole_dqn_nominal_v2.toml \
  --output artifacts/policies/cartpole_dqn_nominal_v2
uv run active-eval-gym freeze-policy \
  --artifact artifacts/policies/cartpole_dqn_nominal_v2 \
  --suite configs/eval/cartpole_dqn_v2_freeze.toml
```

See [the nominal findings](docs/findings.md) before treating either DQN candidate
as part of a frozen evaluation suite.

The passing `nominal-v2` suite compares LQR and PPO on CartPole and retains SAC
and PPO on the other tasks. Collect its 80 episodes with:

```bash
uv run active-eval-gym evaluate-nominal \
  --suite configs/eval/nominal_v2.toml \
  --artifact-root artifacts/policies \
  --output artifacts/evaluations/nominal-v2
```

To inspect one frozen policy interactively:

```bash
uv run active-eval-gym render-policy \
  --artifact artifacts/policies/cartpole_lqr_nominal_quantized_v1 \
  --seed 0
```

See [the policy design notes](docs/policies.md) for equations, wrappers, artifact
contracts, and the fixed-policy invariant.

## Mandatory CHE-49 perturbation sweeps

CHE-49 reuses the four frozen nominal-v2 artifacts without rebuilding or
training them. Its tracked sampling strategies cover the paired CartPole
angle-by-length grid, the Pendulum length sweep, and every valid MiniGrid start
pose. Collection writes raw schema-v3 trajectories first; analysis later reads
and hash-verifies those files without loading a policy or environment.

Run one collector per config, following this pattern:

```bash
uv run active-eval-gym collect-sweep \
  --suite configs/eval/che49_cartpole_angle_length_v1.toml \
  --artifact-root artifacts/policies \
  --output artifacts/evaluations/che-49/che49-cartpole-angle-length-v1
uv run active-eval-gym collect-sweep \
  --suite configs/eval/che49_pendulum_length_v1.toml \
  --artifact-root artifacts/policies \
  --output artifacts/evaluations/che-49/che49-pendulum-length-v1
uv run active-eval-gym collect-sweep \
  --suite configs/eval/che49_minigrid_start_pose_v1.toml \
  --artifact-root artifacts/policies \
  --output artifacts/evaluations/che-49/che49-minigrid-start-pose-v1
```

For each evaluation directory, derive episode-summary-v2 metrics and plots:

```bash
uv run active-eval-gym analyze-sweep --evaluation EVALUATION_DIRECTORY
uv run active-eval-gym plot-sweep \
  --evaluation EVALUATION_DIRECTORY \
  --output artifacts/figures/che-49
```

The full run contains 2,040 episode bundles: 1,800 paired CartPole episodes,
100 Pendulum episodes, and 140 MiniGrid episodes. Commands refuse to overwrite
collection, metric-version, or figure outputs. The checked-in figures are in
[`docs/figures`](docs/figures), and observed results are summarized in
[`docs/findings.md`](docs/findings.md).

## CHE-49 secondary and stretch CartPole sweeps

Five independent diagnostic suites reuse the same frozen CartPole LQR and PPO
artifacts. Each varies one axis over five conditions and paired seeds 0 through
19, for 200 episodes per suite and 1,000 episodes in total. The mass and force
suites perturb simulator dynamics; the noise, delay, and dropout suites exercise
the observation and action interfaces. They do not cross these axes with one
another or with the mandatory angle-by-length surface.

Collect each suite with:

```bash
for name in mass pole_angle_noise force_magnitude action_delay action_dropout; do
  uv run active-eval-gym collect-sweep \
    --suite "configs/eval/che49_cartpole_${name}_v1.toml" \
    --artifact-root artifacts/policies \
    --output "artifacts/evaluations/che-49/che49-cartpole-${name//_/-}-v1"
done
```

Then run `analyze-sweep` and `plot-sweep` on each evaluation directory as above.
These suites write raw schema-v4 episodes and `episode-summary-v3` analysis.
Schema v4 retains the policy-requested `action`, adds the delivered
`environment_action`, and records deterministic observation-noise or action-
intervention diagnostics. Their five checked-in dashboards and measured results
are included in [`docs/findings.md`](docs/findings.md).

### CartPole sweep animations

CartPole sweeps can also be visualized directly from their saved, hash-verified
raw trajectories. This does not load a policy, recreate an environment, or alter
the evaluation. The default creates one GIF per condition and a synchronized
comparison GIF:

```bash
uv run active-eval-gym animate-sweep \
  --evaluation artifacts/evaluations/che-49/che49-cartpole-action-dropout-v1 \
  --output artifacts/animations/che49-cartpole-action-dropout-v1 \
  --layout both \
  --frame-stride 5
```

Each condition overlays all paired seeds, with the first two policies rendered
as blue and red density layers. Overlap is purple and does not depend on
drawing order. A terminal failure is marked once and then removed; the live-run
counts remain visible. The composite uses one shared clock and physical scale so
that its condition panels stay synchronized and comparable. GIF timing is derived
from the recorded CartPole timestep and `--frame-stride`; the default is 10 FPS
for the nominal 0.02-second timestep.

An `animation-manifest.json` records the rendering settings, policy colors, input
trajectory hashes, and output hashes. Animation files are never overwritten.
Composite views are limited to six conditions for legibility; use
`--layout individual` for larger CartPole grids.

## CartPole policy-symmetry study

The tracked symmetry protocol directly audits the frozen CartPole PPO's
categorical action distribution, rather than inferring it from deterministic
rollout action counts. It evaluates the identities

```text
pi(action 1 | s) = 1 - pi(action 1 | -s)
logit_margin(s) = -logit_margin(-s)
V(s) = V(-s)
```

under the CartPole reflection that negates all four state components and swaps
actions 0 and 1. The same workflow collects exact mirror-paired trajectories for
PPO and LQR, then evaluates a separately identified diagnostic controller whose
binary PPO logit margin is antisymmetrized at inference. The source PPO checkpoint
and its weights are not modified.

Run the complete predeclared study against the existing angle-length evaluation:

```bash
uv run active-eval-gym evaluate-cartpole-symmetry \
  --suite configs/eval/che49_cartpole_symmetry_v1.toml \
  --artifact-root artifacts/policies \
  --source-evaluation \
    artifacts/evaluations/che-49/che49-cartpole-angle-length-v1 \
  --output artifacts/evaluations/che-49/che49-cartpole-symmetry-v1 \
  --figure docs/figures/che49_cartpole_ppo_symmetry_audit.png
```

The command refuses to overwrite either output. It writes a hash-bound static
audit, 1,600 raw mirror-pair episodes, 900 raw causal-intervention episodes,
per-episode metrics, aggregate summaries, and the decision-boundary figure. The
mirror collection uses an explicit fixed-initial-state perturbation: if one
branch begins at `s`, the other begins at exactly `-s`, with equal plant
parameters and complementary actions required by symmetry. Results from the
local acceptance run are recorded in [`docs/findings.md`](docs/findings.md).

The derived policy can also participate in the ordinary paired-sweep path. Its
v2 config binds the transformation to the frozen source model hash and adds it as
a third policy without creating or mutating a learned checkpoint:

```bash
uv run active-eval-gym collect-sweep \
  --suite configs/eval/che49_cartpole_angle_length_v2.toml \
  --artifact-root artifacts/policies \
  --output artifacts/evaluations/che-49/che49-cartpole-angle-length-v2
uv run active-eval-gym analyze-sweep \
  --evaluation artifacts/evaluations/che-49/che49-cartpole-angle-length-v2
uv run active-eval-gym plot-sweep \
  --evaluation artifacts/evaluations/che-49/che49-cartpole-angle-length-v2 \
  --output docs/figures
```

For three-policy angle-length suites, the success dashboard contains one surface
per policy and all three pairwise success-rate differences. The slice dashboard
overlays all three policies on the same axes. The original v1 suite and figures
remain unchanged.

The same derived policy is included in separately versioned reruns of the
pole-angle-noise, action-delay, and action-dropout sweeps. Each reuses the exact
v1 condition/seed grid, records 300 episodes, and preserves the two-policy v1
artifacts and figures:

```bash
for name in pole_angle_noise action_delay action_dropout; do
  evaluation="artifacts/evaluations/che-49/che49-cartpole-${name//_/-}-v2"
  uv run active-eval-gym collect-sweep \
    --suite "configs/eval/che49_cartpole_${name}_v2.toml" \
    --artifact-root artifacts/policies \
    --output "$evaluation"
  uv run active-eval-gym analyze-sweep --evaluation "$evaluation"
  uv run active-eval-gym plot-sweep \
    --evaluation "$evaluation" \
    --output docs/figures
done
```

The three-policy one-dimensional dashboards use the
`*_sweep_three_policy.png` suffix. Results and replay-integrity checks are
recorded in [`docs/findings.md`](docs/findings.md).

## Supported environments

- `CartPole-v1`: discrete action, interpretable four-value control state.
- `Pendulum-v1`: continuous action and continuous control performance.
- `MiniGrid-Empty-8x8-v0`: dictionary observation and discrete navigation action.

The original constant-zero policy remains useful for rollout smoke tests. The
policy zoo adds a quantized LQR and learned DQN, SAC, and PPO policies, all
evaluated through the same raw-trajectory path. CHE-49 covers the mandatory
angle, physical-length, and MiniGrid start-pose perturbations plus isolated
CartPole mass, force, observation-noise, action-delay, and action-dropout
diagnostics. Evaluation thresholds, simulator numerics, policy adaptation, and
crossed stretch perturbations remain out of scope.

## Development

```bash
uv run pytest
uv run ruff check .
```
