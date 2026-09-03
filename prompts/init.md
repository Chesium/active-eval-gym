Yes. For this week I would optimize for **evaluation intuition, not RL-algorithm breadth**. David’s todo is basically asking you to build a small “policy zoo + perturbation harness” so you can see, concretely, how a controller/policy behaves when the deployment distribution shifts.

A good repo should therefore be organized around the question:

$$
\text{policy} + \text{environment} + \text{perturbation}
\rightarrow
\text{trajectory}
\rightarrow
\text{evaluation}
$$

rather than around “implement lots of RL algorithms.”

## Repo shape I’d recommend

Something like this is enough:

```text
active-eval-gym/
├── README.md
├── pyproject.toml
├── uv.lock
├── configs/
│   ├── envs/
│   │   ├── cartpole.yaml
│   │   ├── pendulum.yaml
│   │   └── minigrid_empty.yaml
│   ├── policies/
│   │   ├── cartpole_heuristic.yaml
│   │   ├── ppo_cartpole.yaml
│   │   └── ppo_pendulum.yaml
│   └── eval/
│       ├── nominal.yaml
│       ├── init_state_sweep.yaml
│       └── dynamics_sweep.yaml
├── src/active_eval_gym/
│   ├── envs/
│   │   ├── factory.py
│   │   └── perturbations.py
│   ├── policies/
│   │   ├── base.py
│   │   ├── heuristic.py
│   │   └── sb3.py
│   ├── train.py
│   ├── rollout.py
│   ├── evaluate.py
│   ├── metrics.py
│   └── plotting.py
├── experiments/
│   ├── cartpole_init_sweep.py
│   ├── cartpole_dynamics_sweep.py
│   └── minigrid_navigation.py
├── notebooks/
│   └── exploration.ipynb
├── tests/
│   ├── test_perturbations.py
│   └── test_reproducibility.py
├── artifacts/
│   └── .gitkeep
└── docs/
    ├── environments.md
    ├── policies.md
    └── findings.md
```

The important separation is:

```text
training
≠
rollout collection
≠
perturbation generation
≠
evaluation metric
```

That will matter later if you start doing active evaluation. You want to be able to freeze a policy and ask:

> “What happens if the evaluator changes \(X\)?”

without accidentally retraining the policy.

Gymnasium is a good fit because the API is deliberately tiny—`reset()`, `step()`, observation/action spaces—and wrappers are intended for modular changes to observations, actions, rewards and environment behavior. ([Gymnasium][2])

## Start with only three environments

I would not try to cover the whole Classic Control suite this week. Gymnasium currently has CartPole, Pendulum, MountainCar/Continuous MountainCar and Acrobot as its classic-control set. ([Gymnasium][3])

Use these three:

**CartPole** should be your main sandbox. It has an extremely interpretable state, obvious failure boundary, short trajectories, and perturbations such as initial pole angle, cart position, gravity, pole length and mass all have intuitive physical meanings.

**Pendulum** gives you continuous action and a continuous notion of “badness,” rather than a mostly binary survival/failure outcome. This will become useful when thinking about metric design.

**MiniGrid** gives you a navigation analogue. MiniGrid is specifically designed as a simple, fast, configurable grid-world research environment, with discrete actions and tasks involving navigation/objects. ([MiniGrid][4])

I’d start MiniGrid with `MiniGrid-Empty-8x8-v0`, then perhaps `Dynamic-Obstacles` or `LavaCrossing`. The default observation is partially observable and dictionary-valued; `FullyObsWrapper`, `FlatObsWrapper`, and `ImgObsWrapper` are already available, so don't spend time writing your own observation encoding. ([MiniGrid][5])

## You do not need to “learn PPO” before using PPO

For this task I'd deliberately use **two policy classes**.

First, have at least one transparent controller. For CartPole, a simple hand-written linear/heuristic controller is valuable because you can reason directly about why it fails. You could later add an LQR-style controller if you want the control-theory connection.

Second, use a library-trained neural policy as a black box. For example, train PPO on CartPole and Pendulum using an existing RL library. Your goal this week is not to derive PPO; it's to obtain a policy

$$
\pi_\theta(a\mid s)
$$

whose behavior you can interrogate.

You should understand only this much initially:

$$
s_t \to \pi_\theta \to a_t
\to f(s_t,a_t)
\to s_{t+1}.
$$

Training adjusts \(\theta\) to increase expected accumulated reward. The exact policy-gradient estimator can wait for your RL/ESE courses.

This will actually give you a useful comparison:

```text
hand-designed controller
vs.
learned policy
```

under the **same evaluation perturbations**.

That is more scientifically useful than implementing Q-learning, DQN, PPO, SAC, A2C, DAgger, etc. all in one week.

## Make perturbations first-class objects

This is the part I would spend the most engineering effort on.

Don't write:

```python
env.unwrapped.masspole = 0.17
```

randomly throughout notebooks.

Define something like:

```python
PerturbationSpec(
    name="pole_length",
    value=0.75,
)
```

and make every rollout carry a full experiment record.

I would support four perturbation families initially:

$$
\text{initial condition}
$$

such as pole angle / cart velocity / MiniGrid start position;

$$
\text{dynamics parameters}
$$

such as mass, pole length or gravity;

$$
\text{observation corruption}
$$

such as Gaussian noise or masking one state variable;

and

$$
\text{actuation disturbance}
$$

such as action noise, delay or occasional dropped actions.

That gives you a nice progression from:

> “policy works nominally”

to:

> “where is its failure boundary?”

to:

> “which axes of distribution shift matter?”

MiniGrid already exposes useful wrappers such as stochastic actions and customizable field of view, which makes it convenient for this sort of robustness experiment. ([MiniGrid][5])

## Log trajectories, not just returns

This will matter enormously for the research direction you and David discussed.

For every episode, save something conceptually like:

```text
env_id
policy_id
policy_train_seed
episode_seed
perturbation_name
perturbation_value
initial_state

s_0, a_0, r_0
s_1, a_1, r_1
...
s_T

return
episode_length
terminated
truncated
failure_reason
```

For CartPole, later you can invent metrics such as:

$$
M_1 = \mathbf 1[\text{survived 500 steps}]
$$

$$
M_2 = T/500
$$

$$
M_3 =
1-\frac1T\sum_t \frac{|\theta_t|}{\theta_{\max}}
$$

$$
M_4 =
1-\max_t\frac{|\theta_t|}{\theta_{\max}}
$$

and ask whether these metrics tell different stories about the same policy.

That is already a toy version of the metric-design problem you've been discussing.

## A very useful first experiment

I'd make **CartPole initial-angle sweep** the first polished result you show David.

Train one nominal policy under the default environment.

Then freeze it.

Evaluate over

$$
\theta_0\in
\{-0.20,-0.18,\ldots,0.18,0.20\}
$$

with, say, 20–50 seeds per condition.

Plot:

$$
P(\text{success}\mid\theta_0)
$$

and perhaps mean episode length.

Then do a 2D sweep:

$$
(\theta_0,\;m_{\rm pole})
$$

or

$$
(\theta_0,\;\ell_{\rm pole}).
$$

Now you get a **failure surface** rather than a single average return.

Conceptually this gets you very close to active evaluation:

> If I only have 20 more evaluations, where on this surface should I sample?

Uniformly?

Near the estimated boundary?

Where uncertainty is high?

Where two policies disagree?

That is exactly the intuition David probably wants you to develop.

## How Failure Prediction fits

The Failure Prediction paper David mentioned assumes access to a black-box controller and aims to learn a failure predictor from observation histories, with statistical guarantees on false-positive/false-negative error rates using PAC-Bayes machinery. ([arXiv][6])

You don't need to implement the paper yet.

But your Gym repo can naturally generate the data structure it requires:

$$
\tau_{0:t}
\rightarrow
P(\text{eventual failure}).
$$

For CartPole you might later ask:

> Given the first 30 timesteps, can I predict whether the policy will fail before step 500?

That makes Failure Prediction much less abstract.

## How DAgger fits

I also wouldn't implement DAgger immediately.

The conceptual lesson matters more:

A policy trained only on states generated by an expert sees distribution

$$
d_{\pi^*}.
$$

Once the learned policy makes small errors, it visits its own distribution

$$
d_\pi,
$$

which can contain states absent from its training data.

DAgger repeatedly rolls out the learner, gets expert labels on the states **the learner actually visits**, aggregates those states, and retrains. That is the central solution to closed-loop distribution shift in the original work. ([arXiv][1])

Your perturbation experiments give you the same intuition from another direction:

```text
nominal training distribution
          ↓
policy acts
          ↓
perturbation / policy error
          ↓
new state distribution
          ↓
unexpected failure
```

So your repo becomes a useful concrete substrate for understanding the ESE 6180 theory rather than a disconnected software exercise.

## Where agents should help

I would use coding agents aggressively for the **mechanical infrastructure**, but protect the pieces from which you are supposed to acquire research intuition.

A good division is:

* Agent-owned: repo scaffolding, environment factory, SB3 integration, config parser, sweep runner, parallel execution, result serialization, plotting utilities, tests and README tables.
* Shared: perturbation wrapper design, trajectory schema, reproduction checks, specific experiment implementations.
* **You-owned:** choosing perturbation axes, reasoning about the dynamics, defining success/failure, interpreting failure surfaces, deciding which metrics are meaningful, and explaining why a proposed active evaluator should sample one region rather than another.

That separation is especially important because otherwise an agent can give you a beautiful PPO + Hydra + W&B framework while you finish the week with little stronger intuition than when you started.

I would also put a short `AGENTS.md` at the repo root with research guardrails such as:

```text
- Do not change policy weights during evaluation unless explicitly requested.
- Every evaluation run must record policy checkpoint, env version,
  seeds, perturbation config, and metric version.
- Do not select or hide seeds based on observed performance.
- Preserve raw trajectories separately from derived metrics.
- New metrics must not overwrite existing results.
- Training reward and evaluation metrics are separate concepts.
- Prefer simple reusable wrappers over environment-specific hacks.
- Tests should verify that nominal perturbation reproduces the
  unmodified environment.
```

That is actually a tiny version of the provenance discipline you may later want in an agentic evaluator.

## A realistic one-week target

I would consider the week successful if by Sunday you have:

* ✅ reproducible repo and environment abstraction;
* ✅ one heuristic CartPole controller;
* ✅ one learned CartPole/Pendulum policy;
* ✅ nominal evaluation with many seeds;
* ✅ initial-condition + one dynamics-parameter perturbation sweep;
* ✅ one failure-surface plot;
* ✅ MiniGrid environment loading/rollout, even if the policy is rudimentary;
* 🟨 a short `docs/findings.md` explaining **three surprising policy behaviors** you observed;
* 🟨 2–3 questions about what an active evaluator should sample next.

I would rather have that than five trained algorithms.

The most important shift in mindset is:

> **The policy is not this week's research object by itself. The mapping from environment conditions to policy behavior is.**

You are trying to acquire intuition for

$$
X
\longmapsto
\tau_\pi(X)
\longmapsto
M(\tau_\pi(X)),
$$

because later the outer evaluator will be making decisions about **which \(X\)** to test and perhaps **which \(M\)** to use.

If you want, the next useful step would be for me to turn this into a concrete **repo bootstrap spec / AGENTS.md + first 3 agent tasks**, so you can hand it directly to a coding agent and have the basic framework running quickly.

[1]: https://arxiv.org/abs/1011.0686?utm_source=chatgpt.com "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning"
[2]: https://gymnasium.farama.org/?utm_source=chatgpt.com "Gymnasium Documentation"
[3]: https://gymnasium.farama.org/v0.27.0/environments/classic_control/?utm_source=chatgpt.com "Classic Control - Gymnasium Documentation"
[4]: https://minigrid.farama.org/v2.5.0/?utm_source=chatgpt.com "MiniGrid Documentation"
[5]: https://minigrid.farama.org/api/wrappers/?utm_source=chatgpt.com "Action Bonus - MiniGrid Documentation"
[6]: https://arxiv.org/abs/2202.05894?utm_source=chatgpt.com "Failure Prediction with Statistical Guarantees for Vision-Based Robot Control"
