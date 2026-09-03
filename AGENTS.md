# AGENTS.md

## Purpose

This repository is a small research sandbox for studying the behavior and
evaluation of fixed control / reinforcement-learning policies under controlled
environment perturbations.

The immediate goal is to build intuition for:

    environment condition X
        -> policy pi
        -> closed-loop trajectory tau_pi(X)
        -> evaluation metric M(tau_pi(X))

The longer-term research context is **active policy evaluation**: an outer
evaluator may use previous evaluation results to choose which environment,
perturbation, or eventually metric to evaluate next.

This is therefore an **evaluation research repository**, not an RL benchmark
or an exercise in implementing as many algorithms as possible.

Current milestone: prepare a small reproducible policy/perturbation sandbox
for discussion on 7 September 2026.


## Research priorities

Optimize for, in order:

1. Correct and reproducible experiments.
2. Clear separation between policy training and policy evaluation.
3. Easy inspection of trajectories and failure behavior.
4. Simple, traceable experiment configuration.
5. Ease of adding perturbations and metrics.
6. Implementation elegance.
7. Training performance / algorithm breadth.

Do not introduce substantial framework complexity unless it directly supports
one of the first five priorities.


## Core conceptual separation

Keep these four concerns separate:

### 1. Training

Produces a frozen policy/checkpoint.

Training code may use environment rewards and training-specific wrappers.

Training must not silently occur during an evaluation run.

### 2. Rollout collection

Executes a fixed policy in a specified environment configuration and records
raw trajectories.

The rollout layer should not decide which experiments are scientifically
interesting.

### 3. Perturbation generation

Defines how the evaluation environment differs from nominal conditions.

Examples:

- initial state / initial condition;
- mass, pole length, gravity, or other dynamics parameters;
- observation corruption;
- action noise, delay, or dropout;
- navigation start state / obstacle configuration.

Perturbations should be explicit structured objects/configurations, not
scattered mutations of `env.unwrapped` throughout experiment scripts.

### 4. Evaluation / metrics

Computes metrics from raw rollout data.

A metric must not modify the policy or environment.

Raw trajectories must be retained independently of derived metrics so that
new metrics can be evaluated without rerunning the original experiment.


## Initial environment scope

Keep the first version intentionally small:

- CartPole: primary interpretable control sandbox.
- Pendulum: continuous-action / continuous-performance sandbox.
- MiniGrid: navigation analogue.

Do not add more environments until these three work through the common
evaluation path.


## Policy scope

Initially support two broad policy classes:

1. Transparent / hand-designed controllers.
2. Library-trained learned policies.

It is acceptable and encouraged to use established RL libraries to train
policies such as PPO.

Do not reimplement advanced RL algorithms unless the research question
specifically requires it.

The human researcher is currently learning RL; the purpose of using a learned
policy here is to obtain behavior to interrogate, not to hide RL complexity
behind unnecessary custom code.


## Policy interface

Prefer a minimal common abstraction similar to:

```python
class Policy(Protocol):
    def act(
        self,
        observation,
        *,
        deterministic: bool = True,
    ): ...
```


## Reproducibility guardrails

- Do not change policy weights during evaluation unless an experiment explicitly
  studies adaptation.
- Every evaluation must record the environment ID and package version, policy ID
  and checkpoint, training and action seeds, episode seed, perturbation spec, and
  metric version where applicable.
- Do not select, omit, or hide seeds based on observed performance.
- Preserve raw trajectories separately from derived metrics.
- New metric versions must not overwrite existing results.
- Keep training rewards distinct from evaluation metrics.
- Apply perturbations through reusable specs and wrappers rather than scattered
  mutations of environment internals.
- A nominal/no-op perturbation must reproduce the unmodified environment within
  the deterministic limits of the environment and policy.
