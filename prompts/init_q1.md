question:how should I deal with the interaction between parameterized policy (e.g. derive K for LQR assuming the pole length = 10cm), nominal environment config (maybe training data) (10cm, 0deg) and the perturbation (e.g. evaluate at angle 1.2deg, pole length 10.2 cm), what should be the correct orchestration and mental model?


---

The cleanest mental model is to treat these as **three different objects that should never be silently merged**:

$$
\boxed{\text{policy design/training specification}}
\qquad
\boxed{\text{nominal evaluation environment}}
\qquad
\boxed{\text{evaluation perturbation}}
$$

Then an experiment means:

$$
\text{freeze policy}
\rightarrow
\text{construct actual evaluation environment}
\rightarrow
\text{roll out policy}
\rightarrow
\text{measure behavior}.
$$

For your LQR example, that distinction is especially crisp.

## 1. Policy parameters belong to the policy's provenance

Suppose the CartPole dynamics depend on pole length \(\ell\), and you linearize about

$$
\ell_{\text{design}}=10\text{ cm},
\qquad
\theta_{\text{eq}}=0^\circ.
$$

From the corresponding linearized matrices

$$
A(0.10),B(0.10),
$$

you solve the Riccati equation and obtain

$$
K_{\text{10cm}}.
$$

Your controller is now

$$
u_t=-K_{\text{10cm}}x_t.
$$

At that point, **\(K_{\text{10cm}}\) is frozen policy state**.

Conceptually:

```text
Policy:
    type: LQR
    design_model:
        pole_length: 0.10 m
        equilibrium_angle: 0 deg
        ...
    Q: ...
    R: ...
    K: ...
```

The fact that \(K\) came from a 10 cm model is part of the policy's provenance, much like a neural policy's training dataset and random seed.

It should not be recomputed simply because the evaluation environment changes.

---

# 2. Nominal environment is your reference condition

Separately, define the benchmark's nominal environment:

$$
E_0:
\begin{cases}
\ell=10\text{ cm}\\
m_{\rm pole}=\cdots\\
g=9.81\\
\theta_0=0^\circ\\
\cdots
\end{cases}
$$

This answers:

> “What does an unperturbed deployment look like?”

For your first experiments, it's perfectly reasonable that

$$
\text{policy-design model}
=
\text{nominal environment}.
$$

But conceptually they are still different objects.

This distinction becomes important later. You could easily have:

$$
\ell_{\rm train}\sim U(9,11)\text{ cm}
$$

for a domain-randomized RL policy while defining the nominal benchmark as

$$
\ell_{\rm nominal}=10\text{ cm}.
$$

So don't make your software architecture assume they're necessarily identical.

---

# 3. Perturbation is an overlay on the evaluation environment

Now define:

```text
Perturbation:
    initial_angle: +1.2 deg
    pole_length: +0.2 cm
```

Apply that to \(E_0\):

$$
E_{\rm eval}
=
E_0\oplus\Delta.
$$

Hence the actual rollout has

$$
\ell_{\rm eval}=10.2\text{ cm},
\qquad
\theta_0=1.2^\circ.
$$

But the controller remains

$$
\boxed{u_t=-K_{\text{10cm}}x_t}.
$$

So the complete experiment is:

$$
\boxed{
K(10\text{ cm},0^\circ)
}
\quad\text{controlling}\quad
\boxed{
\text{actual plant }(10.2\text{ cm},\theta_0=1.2^\circ)
}.
$$

That is precisely what makes it a robustness/generalization experiment.

---

## 4. What if you recompute \(K\) at 10.2 cm?

Then you're asking a **different scientific question**.

### Experiment A — fixed controller robustness

$$
K=K(10\text{ cm})
$$

for every environment, while

$$
\ell_{\rm eval}
$$

changes.

Question:

> How robust is this fixed controller to plant mismatch?

### Experiment B — oracle/model-aware controller

For each evaluation environment:

$$
K=K(\ell_{\rm eval}).
$$

Question:

> How well does LQR work if the controller knows the true plant parameters and is redesigned accordingly?

These are both valid, but they are not interchangeable.

In fact, Experiment B can be an excellent **oracle baseline** for Experiment A.

For example:

$$
\text{fixed }K(10\text{ cm})
\quad\text{vs}\quad
\text{oracle }K(\ell_{\rm true}).
$$

Then degradation can be attributed partly to model mismatch.

---

# 5. The same abstraction applies to RL

Replace

$$
K
$$

with neural-network weights

$$
w.
$$

For example,

$$
w=
\operatorname{Train}
(
D_{\rm train},
\text{algorithm},
\text{seed},
\text{hyperparameters}
).
$$

Perhaps

$$
D_{\rm train}:
\ell=10\text{ cm},
\quad
\theta_0\sim[-2^\circ,2^\circ].
$$

Once training ends:

$$
\boxed{\pi_w\text{ is frozen}.}
$$

Then evaluate it at

$$
\ell=10.2\text{ cm},
\quad
\theta_0=1.2^\circ.
$$

So there is an almost exact analogy:

| LQR                        | Learned policy                 |
| -------------------------- | ------------------------------ |
| model used to derive \(K\) | training distribution          |
| \(Q,R\)                    | reward/loss + hyperparameters  |
| Riccati solution \(K\)     | trained weights \(w\)          |
| freeze \(K\)               | freeze checkpoint              |
| perturb real plant         | perturb evaluation environment |

This is a useful way to compare classical and learned controllers without special-casing them conceptually.

---

# 6. I would actually define four separate specifications in the repo

Something like:

```yaml
policy:
  id: cartpole-lqr-10cm
  type: lqr
  design:
    pole_length_m: 0.10
    equilibrium:
      angle_rad: 0.0
    Q: [...]
    R: [...]
  artifact:
    K: [...]

nominal_environment:
  id: cartpole-default
  pole_length_m: 0.10
  gravity: 9.81
  initial_state:
    angle_rad: 0.0
    angular_velocity: 0.0

perturbation:
  initial_state:
    angle_deg: 1.2
  dynamics:
    pole_length_m: 0.102

evaluation:
  seeds: [...]
  episodes_per_condition: 20
  metrics:
    - survival_time_v1
    - angle_rms_v1
```

The evaluation runner resolves these into:

```text
policy_id
    ↓
load frozen controller

nominal_environment
    +
perturbation
    ↓
resolved actual environment

seed
    ↓
initial realization

policy + actual environment
    ↓
trajectory

trajectory
    ↓
metrics
```

And I would store the **resolved environment** as well as the perturbation:

```text
requested perturbation:
    pole_length += 0.002

resolved environment:
    pole_length = 0.102
```

That helps enormously with provenance.

---

# 7. Initial-condition perturbation and model perturbation are subtly different

You mentioned both:

$$
\theta_0=1.2^\circ
$$

and

$$
\ell=10.2\text{ cm}.
$$

It's worth distinguishing them.

### Initial-condition shift

Dynamics remain the same:

$$
f_{\rm eval}=f_{\rm nominal},
$$

but

$$
x_0\neq x_{\rm nominal}.
$$

You're asking:

> What is the controller's region of attraction / recovery capability?

For LQR, this becomes especially meaningful because it was designed from a local linearization around

$$
x=0.
$$

As \(\theta_0\) increases, nonlinear effects eventually make that approximation poor.

### Dynamics/model shift

Now:

$$
f_{\rm eval}\neq f_{\rm design}.
$$

For example,

$$
\ell_{\rm eval}=10.2\text{ cm}
$$

changes the true plant dynamics.

You're asking:

> How robust is the controller to model mismatch?

### Combined shift

$$
x_0\neq x_{\rm nominal},
\qquad
f_{\rm eval}\neq f_{\rm nominal}.
$$

This is usually harder and can expose interactions.

That's why I'd initially do **one-dimensional sweeps separately**, then 2D surfaces.

---

# 8. There is another distinction: does the policy know the perturbed parameter?

This becomes very important later.

Suppose the policy is

$$
u=\pi(x)
$$

and you change pole length.

The controller doesn't explicitly know \(\ell\). It only experiences the consequences through state feedback.

That's a normal robustness experiment.

But suppose your policy is

$$
u=\pi(x,\ell).
$$

Then giving it the true

$$
\ell=10.2\text{ cm}
$$

lets it condition its behavior on the perturbation.

That is closer to a **gain-scheduled / parameter-conditioned policy**.

And if it estimates \(\ell\) online,

$$
\hat\ell_t=\text{Estimator}(x_{0:t},u_{0:t-1}),
$$

followed by

$$
u_t=\pi(x_t,\hat\ell_t),
$$

you now have **online adaptation/system identification**.

Those should be distinct policy classes:

```text
fixed policy
parameter-conditioned policy
adaptive policy
```

Again, don't let the evaluation harness quietly pass environment parameters to the policy unless that's explicitly part of the experiment.

---

# 9. The core causal model is useful

I'd mentally draw your experiment this way:

$$
\boxed{D_{\rm design}}
\rightarrow
\boxed{\pi}
$$

and separately:

$$
\boxed{E_0}
+
\boxed{\Delta}
\rightarrow
\boxed{E_{\rm eval}}.
$$

Then:

$$
\pi
+
E_{\rm eval}
\rightarrow
\tau
\rightarrow
M(\tau).
$$

Or more explicitly:

$$
\underbrace{
(\text{design model / training data})
}_{\text{past}}
\rightarrow
\underbrace{
\pi_{\rm frozen}
}_{\text{policy}}
$$

while

$$
\underbrace{
E_{\rm nominal}
}_{\text{reference}}
+
\underbrace{
\Delta
}_{\text{evaluator action}}
\rightarrow
\underbrace{
E_{\rm actual}
}_{\text{deployment}}
$$

and then

$$
(\pi_{\rm frozen},E_{\rm actual})
\rightarrow
\tau.
$$

This causal separation is exactly what you'll later need for **active evaluation**.

The outer evaluator is allowed to alter

$$
\Delta_k,
$$

not silently alter

$$
\pi.
$$

So later:

$$
D_k
\rightarrow
\mathcal A
\rightarrow
\Delta_{k+1}
\rightarrow
E_{k+1}
\rightarrow
\tau_{k+1}
\rightarrow
D_{k+1}.
$$

The object being interrogated remains fixed.

---

# 10. Training distributions deserve first-class treatment

For RL especially, I wouldn't only store “nominal training config.”

I'd distinguish:

$$
\boxed{D_{\rm train}}
\quad\text{from}\quad
\boxed{D_{\rm target}}
\quad\text{from}\quad
\boxed{D_{\rm eval/sample}}.
$$

For example:

$$
D_{\rm train}:
\ell=10\text{ cm},
\quad
\theta_0\sim U(-2^\circ,2^\circ).
$$

Your desired deployment population may be

$$
D_{\rm target}:
\ell\sim U(9.5,10.5)\text{ cm},
\quad
\theta_0\sim U(-5^\circ,5^\circ).
$$

But your active evaluator may deliberately sample

$$
D_{\rm eval}
$$

near the estimated failure boundary.

Those three distributions answer completely different questions.

This distinction will become essential once you return to SureSim/N-SCORE:

$$
\text{where the policy learned}
\neq
\text{where we ultimately care about performance}
\neq
\text{where the evaluator strategically collects data}.
$$

---

## My preferred vocabulary for the repo

I would use these names consistently:

**`design_spec`**
What assumptions/data produced the policy.

**`policy_artifact`**
The frozen \(K\), checkpoint, heuristic, etc.

**`nominal_env_spec`**
Reference evaluation condition.

**`perturbation_spec`**
Delta applied by the evaluator.

**`resolved_env_spec`**
Actual environment used for this rollout.

**`target_distribution`**
The population over which you eventually want to make a performance claim.

**`sampling_strategy`**
How evaluation conditions were actually chosen.

Then your example is simply:

$$
\begin{aligned}
\text{design\_spec}:&\quad \ell=10\text{ cm},\theta_{\rm eq}=0\\
\text{policy}:&\quad K_{10\text{cm}}\\
\text{nominal\_env}:&\quad \ell=10\text{ cm},\theta_0=0\\
\text{perturbation}:&\quad \Delta\ell=+0.2\text{ cm},
\quad\Delta\theta_0=+1.2^\circ\\
\text{resolved\_env}:&\quad \ell=10.2\text{ cm},\theta_0=1.2^\circ.
\end{aligned}
$$

And the central rule is:

> **Perturbations modify the world presented to a frozen policy; they do not retroactively modify the assumptions under which that policy was produced.**

If you deliberately redesign/retrain the controller after seeing the perturbation, that's not wrong—it just creates a **new policy artifact and a different experiment**. That one rule will keep most of the orchestration conceptually clean.

---

Yes. The current CHE-46/48/49 specs have the right research intent, but after CHE-47 I would make the remaining specs **much more explicit about orchestration** so an agent cannot accidentally turn a robustness experiment into a controller-redesign experiment. CHE-46 currently says “policy + environment + perturbation” and separates training/rollout/perturbation/metrics, but it does not yet distinguish the policy’s design assumptions from the nominal and resolved evaluation environments.  CHE-48 also leaves the actual policies intentionally vague, and CHE-49 leaves the perturbation axes vague.  

I would tighten them now, before handing CHE-48/49 to agents.

## 1. CHE-46: establish one canonical experiment vocabulary

I'd add an explicit orchestration contract around these objects:

| Object                   | Meaning                                                                     |              Can evaluation change it? |
| ------------------------ | --------------------------------------------------------------------------- | -------------------------------------: |
| `PolicyDesignSpec`       | Model assumptions or training distribution used to construct the controller |                                     No |
| `PolicyArtifact`         | Frozen \(K\), neural checkpoint, scripted policy, etc.                      |                                     No |
| `NominalEnvSpec`         | Reference evaluation environment                                            |                                     No |
| `PerturbationSpec`       | Explicit change requested by evaluator                                      |                                    Yes |
| `ResolvedEnvSpec`        | `NominalEnvSpec ⊕ PerturbationSpec` actually used                           |                            Result only |
| `TargetDistributionSpec` | Population over which we eventually care about performance                  |                                     No |
| `SamplingStrategy`       | Which conditions we actually choose to evaluate                             |                                    Yes |
| `MetricSpec`             | Versioned mapping trajectory → score(s)                                     | Fixed within a confirmatory experiment |

The central orchestration should be written almost as an invariant:

$$
\boxed{
\text{PolicyArtifact}
=
\operatorname{Build}(\text{PolicyDesignSpec})
}
$$

once, followed by

$$
\boxed{
E_{\rm resolved}
=
\operatorname{Resolve}(E_{\rm nominal},\Delta)
}
$$

and then

$$
\boxed{
\tau
=
\operatorname{Rollout}(\pi_{\rm frozen},E_{\rm resolved}).
}
$$

The key wording for agents should be:

> **Evaluation perturbations modify the environment presented to a frozen policy. They do not retroactively modify the policy's design/training assumptions. If the controller is recomputed or the neural policy retrained for a perturbed environment, that creates a new `PolicyArtifact` and a different experiment.**

That single paragraph will prevent a major class of mistakes.

I would also add a perturbation taxonomy:

$$
\text{initial condition},\;
\text{physical dynamics},\;
\text{actuation},\;
\text{observation},\;
\text{task geometry},\;
\text{simulation numerics},\;
\text{evaluation semantics}.
$$

For this week's robustness work, only the first 3–4 should normally vary. Things such as CartPole's `theta_threshold_radians`, `x_threshold`, MiniGrid `max_steps`, or changing the integration method should **not** be mixed into the physical robustness sweep because they alter failure semantics or simulator behavior rather than the physical/task condition.

The attached environment inventory makes this distinction useful: CartPole exposes physical parameters such as gravity, masses and length, but also derived values (`total_mass`, `polemass_length`), numerical parameters (`tau`, `kinematics_integrator`) and termination thresholds. 

One more invariant worth adding:

> Perturb primitive parameters only. Derived quantities must be recomputed by the environment resolver.

For CartPole:

$$
m_{\rm total}=m_{\rm cart}+m_{\rm pole},
\qquad
m\ell=m_{\rm pole}\ell.
$$

An agent should never set `masspole=0.12` while leaving `total_mass=1.1` and `polemass_length=0.05`.

---

# 2. CHE-48: make the policy zoo concrete

I would change CHE-48 from “transparent controller + learned policies” into this explicit initial matrix:

| Environment            | Required policy           | Why                                                                                                      |
| ---------------------- | ------------------------- | -------------------------------------------------------------------------------------------------------- |
| **CartPole**           | **nominal quantized LQR** | Most relevant classical-control baseline; makes model mismatch very interpretable                        |
| **CartPole**           | **DQN**                   | Natural next conceptual step from your Q-learning background; same discrete action space                 |
| **Pendulum**           | **SAC**                   | Standard continuous-action learned controller; gives a qualitatively different policy/evaluation problem |
| **MiniGrid Empty-8x8** | **PPO**                   | Robust practical baseline for navigation; no need for you to learn PPO derivation this week              |
| CartPole               | cascaded PD/PID           | **Stretch**, not blocker                                                                                 |

### I would prioritize LQR over PID

I would **not make PID mandatory**.

For an inverted pendulum, “a PID controller” is not as canonical/simple as it sounds. You usually end up with something like an inner pole-angle controller plus an outer cart-position loop, or essentially PD/state-feedback behavior. That adds design choices without buying you much for Monday.

LQR is much cleaner for your research purpose:

$$
(A_{\rm nominal},B_{\rm nominal},Q,R)
\rightarrow
K
\rightarrow
u=-Kx.
$$

And it connects directly to ESE 6180/control robustness.

So I would specify:

**Required:** LQR + DQN on CartPole.
**Optional stretch:** cascaded PD/PID if implementation is cheap.

---

## 3. Be extremely precise about “automatic LQR \(K\) calculation”

This is the likely agent trap.

Gym's actual CartPole defaults in your inventory are:

$$
g=9.8,\quad
m_c=1.0,\quad
m_p=0.1,\quad
\ell=0.5,\quad
F=10,\quad
\tau=0.02.
$$



So don't use the earlier hypothetical “10 cm” example as the repo default; use the environment's actual nominal `length=0.5` as defined by Gym.

The desired lifecycle is:

```text
PolicyDesignSpec
    nominal CartPole model
    Q, R
    equilibrium
        ↓
calculate A, B
        ↓
calculate K
        ↓
freeze PolicyArtifact
```

Then CHE-49 does:

```text
Frozen K
+
resolved environment with length = 0.575
        ↓
rollout
```

**It must not do:**

```text
length = 0.575
        ↓
recalculate K(0.575)
        ↓
rollout
```

unless you're explicitly producing an **oracle/model-aware LQR artifact** for comparison.

I would actually have CHE-48 name the policy something like:

`cartpole_lqr_nominal_quantized_v1`

because there's another subtle issue: `CartPole-v1` has `Discrete(2)` actions, while ordinary LQR outputs a continuous force. 

So define:

$$
u_{\rm desired}=-Kx
$$

but map that onto Gym actions as, approximately,

$$
a=
\begin{cases}
1,&u_{\rm desired}\ge0\\
0,&u_{\rm desired}<0.
\end{cases}
$$

Gym then applies \(\pm10\) N.

Log **both**:

```text
raw_control = -Kx
env_action = 0/1
```

and document that this is a **quantized LQR controller**, not ideal continuous-input LQR.

A true continuous-actuation CartPole can be a later experiment.

---

# 4. Learned-policy choices

### CartPole → DQN

I'd choose DQN rather than PPO first.

Conceptually it is:

$$
Q(s,a)
$$

just like Q-learning, but represented by a neural network. You don't need to understand replay-buffer/target-network details deeply this week, but it connects naturally to what you already know.

The design/training spec should explicitly record:

```text
environment dynamics = nominal CartPole
initial-state distribution = Gym nominal training reset distribution
algorithm = DQN
training seed
training steps
library + version
checkpoint
```

Evaluation uses deterministic greedy actions.

### Pendulum → SAC

Pendulum has continuous action

$$
u\in[-2,2]
$$

and nominal parameters

$$
g=10,\quad m=1,\quad \ell=1.
$$



SAC is a standard continuous-control choice and gives you useful contrast with DQN. Treat SAC as a **policy generator**, not a theory-learning task this week.

Evaluate using its deterministic action mode.

### MiniGrid Empty-8x8 → PPO

The nominal environment is 8×8, starts at `(1,1)` facing direction `0`, uses discrete 7-action control and a partial-view interface with `agent_view_size=7`. 

I'd use PPO with **one fixed observation wrapper/encoding chosen at training time** and record that wrapper in `PolicyDesignSpec`.

Most important:

> Do not change the observation dimensionality during CHE-49.

So I would **drop `agent_view_size` perturbation from this week's CHE-49**. Changing view size can change the policy input interface unless you deliberately build an observation adapter.

Start-position/orientation perturbations are cleaner.

---

# 5. CHE-49: specify an exact experiment matrix

I would make **CartPole the mandatory polished experiment**, Pendulum a smaller secondary experiment, and MiniGrid a navigation sanity experiment.

That is much better than asking an agent to “try perturbing mass etc.”

### CartPole nominal reference

Use the actual defaults from the attached inventory:

$$
g=9.8,\;
m_c=1.0,\;
m_p=0.1,\;
\ell=0.5,\;
F=10,\;
\tau=.02.
$$



Use a predeclared paired seed set, e.g.

$$
S=\{0,\ldots,19\}.
$$

Every policy sees **the same condition/seed pairs**.

### Required initial-condition axis

Rather than replacing the whole initial state, I'd define an **offset-on-seeded-reset** perturbation:

$$
x_0^{\rm eval}
=
x_0^{\rm nominal}(\text{seed})
+
\Delta x_0.
$$

For pole angle:

$$
\Delta\theta_0
\in
\{-8,-6,-4,-2,0,2,4,6,8\}^{\circ}.
$$

Why stop around ±8°? CartPole's termination threshold is about

$$
0.2094\text{ rad}\approx12^\circ,
$$

so this explores difficult but initially valid states rather than simply starting past the failure boundary. 

This gives:

$$
9\times20=180
$$

episodes per policy.

I'd optionally add one deterministic diagnostic slice with

$$
x=\dot x=\dot\theta=0,
$$

and only \(\theta_0\) varying. That's good for understanding dynamics, but the seeded sweep should be the main experiment.

---

## 6. Required CartPole dynamics axis: pole length

I'd make pole length the primary dynamics perturbation because it interacts directly with the LQR design mismatch you're trying to understand.

Use:

$$
\ell\in
\{0.35,\;0.425,\;0.50,\;0.575,\;0.65\}.
$$

Equivalently:

$$
\ell/\ell_0
\in
\{0.70,0.85,1.00,1.15,1.30\}.
$$

For each resolved environment:

$$
\text{polemass\_length}
\leftarrow
m_p\ell.
$$

The **LQR \(K\) remains the one designed for \(\ell=0.5\)**.

Then produce the main 2D surface:

$$
(\Delta\theta_0,\ell)
$$

with the 9×5 grid above and 20 paired seeds:

$$
9\times5\times20=900
$$

episodes per policy.

Cheap enough for CartPole and extremely interpretable.

This should probably be **the flagship plot for David**.

---

# 7. Add pole mass as the secondary 1-D dynamics test

David mentioned changing mass, and it's useful because it provides another kind of model mismatch.

I'd specify:

$$
m_p/m_{p,0}
\in
\{0.5,\;0.75,\;1.0,\;1.25,\;1.5\},
$$

so with nominal \(m_p=0.1\):

$$
m_p
\in
\{0.05,0.075,0.10,0.125,0.15\}.
$$

Whenever this changes, recompute both

$$
m_{\rm total}=m_c+m_p
$$

and

$$
m_p\ell.
$$

Again: never recompute LQR \(K\).

I would make this secondary sweep required **only if the angle × length experiment is already complete**.

---

# 8. Parameters I would explicitly forbid CHE-49 from perturbing this week

This should be stated because agents love trying everything available.

For CartPole, do **not** perturb:

* `theta_threshold_radians`
* `x_threshold`

because those alter the definition of terminal failure.

Do not initially perturb:

* `kinematics_integrator`
* `tau`

because those are simulator/numerical-model changes, not clean physical plant shifts.

`force_mag` is interesting, but classify it explicitly as an **actuation perturbation** and leave it as stretch work.

This taxonomy will later be scientifically useful:

$$
\text{plant shift}
\neq
\text{actuator shift}
\neq
\text{sensor shift}
\neq
\text{simulator shift}
\neq
\text{evaluation-definition shift}.
$$

---

# 9. Exact CartPole metrics

I'd freeze these before looking at results:

| Metric                              | Purpose                          |   |                        |
| ----------------------------------- | -------------------------------- | - | ---------------------- |
| termination / survives to TimeLimit | coarse task success              |   |                        |
| episode length                      | partial success                  |   |                        |
| RMS (                               | \theta                           | ) | balance quality        |
| max (                               | \theta                           | ) | worst excursion        |
| RMS (                               | x                                | ) | cart-position behavior |
| action-switch rate                  | useful actuator/chattering proxy |   |                        |

I would **not use “control effort” naïvely** for CartPole, because all actual actions apply fixed magnitude ±`force_mag`; cumulative \(|u|\) is therefore largely proportional to episode duration.

For LQR, however, save its pre-quantization

$$
u_{\rm desired}=-Kx
$$

as a diagnostic, but don't compare that directly against DQN as if it were actual actuator effort.

---

# 10. Pendulum CHE-49 should be smaller

Don't create another giant matrix.

Use the frozen SAC policy and test dynamics mismatch:

$$
\ell/\ell_0
\in
\{0.7,0.85,1.0,1.15,1.3\}.
$$

Nominal Pendulum has

$$
g=10,\quad m=1,\quad\ell=1,\quad
u_{\max}=2.
$$



Use paired seeds.

Metrics:

$$
\text{return},\quad
\text{RMS angular error},\quad
\text{RMS angular velocity},\quad
\text{torque usage}.
$$

If time remains, a second physical axis can be gravity:

$$
g/g_0\in\{0.8,1.0,1.2\}.
$$

I'd leave `max_torque` for a later **actuator perturbation** study and `dt` for a later simulator/numerics study.

---

# 11. MiniGrid CHE-49: only change initial condition this week

For `MiniGrid-Empty-8x8-v0`, nominal start is `(1,1)` with direction `0`. 

Rather than hardcoding five positions, I'd have the experiment code enumerate:

$$
\{\text{all valid non-wall, non-goal start cells}\}
\times
\{0,1,2,3\}.
$$

That gives you a clean navigation failure surface:

$$
(\text{start position},\text{start orientation})
\rightarrow
\text{success / steps}.
$$

Metrics:

$$
\text{success},\quad
\text{steps-to-goal},\quad
\text{path efficiency},\quad
\text{number of turns}.
$$

This is particularly interesting if PPO was trained only from the nominal fixed start—the failure surface then directly visualizes **training-distribution vs deployment-distribution generalization**.

Don't vary `max_steps`: that's an evaluation boundary.

Don't vary `agent_view_size` yet: that's an observation-model change and may change the neural input interface.

---

# 12. So I would change the three Linear issues conceptually like this

### CHE-46

Add:

> **Canonical vocabulary + orchestration contract**

and make

$$
\text{design/training}
\rightarrow
\text{frozen artifact}
$$

completely separate from

$$
\text{nominal env}
+
\text{perturbation}
\rightarrow
\text{resolved env}.
$$

Also explicitly distinguish `TargetDistributionSpec` from `SamplingStrategy`. This will matter later when you implement active evaluation.

### CHE-48

Make the required policy set:

```text
CartPole
  quantized LQR (auto-build K from nominal PolicyDesignSpec)
  DQN

Pendulum
  SAC

MiniGrid Empty-8x8
  PPO

Stretch
  CartPole cascaded PD/PID
```

Add the rule:

> Automatic LQR design occurs when building the policy artifact, never when resolving an evaluation perturbation.

And:

> Agents may not silently substitute RL algorithms if training is difficult; report the problem first.

### CHE-49

Replace the vague “angle / some dynamics parameter” wording with:

```text
Mandatory flagship:
  CartPole θ0 offset:
    [-8,-6,-4,-2,0,2,4,6,8] deg

  CartPole length:
    [0.35,0.425,0.50,0.575,0.65] m

  paired seeds:
    0..19

  main 2-D surface:
    θ0 offset × length

Secondary:
  masspole:
    [0.05,0.075,0.10,0.125,0.15]

Pendulum:
  length scale:
    [0.7,0.85,1.0,1.15,1.3]

MiniGrid:
  all valid start positions × four orientations
```

and add the derived-parameter/recompute rules.

---

## The research story this creates

The resulting repo will have a very clean conceptual stack:

$$
\boxed{
D_{\rm design}
\rightarrow
\pi_{\rm frozen}
}
$$

and

$$
\boxed{
E_0+\Delta
\rightarrow
E_{\rm resolved}
}
$$

then

$$
\boxed{
(\pi_{\rm frozen},E_{\rm resolved})
\rightarrow
\tau
\rightarrow
M(\tau).
}
$$

Your first comparison then becomes genuinely interesting:

$$
\text{LQR designed at nominal plant}
\quad\text{vs}\quad
\text{DQN trained at nominal plant}
$$

under the **same controlled model mismatch**.

The next research question almost writes itself:

> Given these experimentally observed failure surfaces, if the evaluator cannot exhaustively evaluate every \((\theta_0,\ell,m,\ldots)\), where should the outer evaluation policy sample next?

That's exactly where CHE-46 can later evolve from a static perturbation harness into an **active-evaluation feedback loop**.

So my main recommendation is: **make LQR mandatory, PID optional; use DQN/SAC/PPO as the first learned policies; make angle × pole-length the canonical CartPole experiment; and hard-code the policy-design/nominal/resolved distinction into CHE-46 before any more agents touch the repo.**

---

For this repo, I’d keep it very conventional:

* **LQR:** `python-control` + NumPy/SciPy
* **DQN / SAC / PPO:** **Stable-Baselines3**, which uses **PyTorch** underneath
* **Environments:** Gymnasium + MiniGrid
* **Experiment/evaluation layer:** your own lightweight code

You do **not** need to implement DQN/SAC/PPO directly in PyTorch for this week. Stable-Baselines3 already provides them, and its own current guidance matches the choices we made: DQN for discrete actions, SAC for continuous actions, and PPO as a good vectorizable/general-purpose option. ([Stable Baselines3 Docs][1])

## Recommended stack

### LQR — `python-control`

Use:

```python
import control as ct

K, S, E = ct.lqr(A, B, Q, R)
```

or, since Gym's CartPole actually evolves at a discrete timestep, potentially:

```python
K, S, E = ct.dlqr(A_d, B_d, Q, R)
```

after discretizing your linearized model.

`python-control` directly supports both `lqr()` and `dlqr()` and returns \(K\), the Riccati solution, and closed-loop eigenvalues. ([Python Control][2])

For this research sandbox I'd prefer this over writing your own Riccati solver. The part worth understanding yourself is:

$$
f(x,u)
\rightarrow
(A,B)
\rightarrow
(A_d,B_d)
\rightarrow
K,
$$

not numerical linear algebra implementation details.

I'd still have the agent/code document where \(A,B\) come from and test the resulting closed-loop eigenvalues.

---

### CartPole DQN — Stable-Baselines3

Something conceptually as simple as:

```python
from stable_baselines3 import DQN

model = DQN("MlpPolicy", env, ...)
model.learn(...)
model.save(...)
```

This is a good first learned policy for you because it preserves the connection to ordinary Q-learning:

$$
Q(s,a)
$$

becomes a neural approximation

$$
Q_\theta(s,a).
$$

CartPole's discrete `{left,right}` action space is exactly the setting for which SB3 recommends DQN-family algorithms. ([Stable Baselines3 Docs][1])

---

### Pendulum SAC — Stable-Baselines3

```python
from stable_baselines3 import SAC

model = SAC("MlpPolicy", env, ...)
```

SAC works naturally with Pendulum's continuous torque action. SB3's current guidance specifically lists SAC among its standard strong continuous-action algorithms. ([Stable Baselines3 Docs][1])

You don't need to understand SAC's entropy-regularized actor-critic derivation before using it.

For now, your conceptual API can simply be:

$$
s_t
\rightarrow
\pi_\theta(s_t)
\rightarrow
u_t\in[-2,2].
$$

Later, when your RL course reaches policy gradients / actor-critic methods, you can revisit what is inside the box.

---

### MiniGrid PPO — Stable-Baselines3

Also use SB3:

```python
from stable_baselines3 import PPO
```

The main complication isn't PPO itself. It's **observation representation**.

MiniGrid observations are richer than the plain 4-vector in CartPole, so choose one observation adapter during training and freeze that choice inside `PolicyDesignSpec`.

For the first week I would choose the simplest SB3-compatible representation rather than building a custom CNN/recurrent architecture.

For example:

```text
MiniGrid raw observation
      ↓
fixed wrapper / flattening
      ↓
fixed-size vector
      ↓
PPO MLP policy
```

Then CHE-49 varies starting positions and orientations without changing that observation interface.

---

# Do you need a discrete GPU?

**No. For these experiments, I would actively avoid making a GPU a requirement.**

A normal modern laptop CPU should be enough for:

* LQR: essentially instantaneous;
* CartPole DQN: small MLP + extremely cheap simulator;
* Pendulum SAC: still a tiny neural network and cheap environment;
* MiniGrid PPO: small environment and relatively small policy;
* thousands of evaluation rollouts: usually environment/CPU work anyway.

These are nothing like training vision transformers or large robotics policies.

The network might only have something like a couple of small fully connected layers. The amount of matrix multiplication per step is tiny.

In fact, for these small environments a discrete GPU can provide **little benefit or even worse wall-clock performance**, because you pay overhead moving small batches through the GPU while environment stepping remains on CPU.

The computational shape is roughly:

$$
\underbrace{\text{tiny simulation step}}_{\rm CPU}
+
\underbrace{\text{tiny MLP inference/training}}_{\rm CPU\ is\ fine}.
$$

A GPU becomes much more useful when you move toward:

* image-based observations;
* large CNNs/transformers;
* hundreds/thousands of parallel simulated environments;
* large replay batches;
* expensive robot simulators such as Isaac Sim;
* large-scale hyperparameter searches;
* foundation-model policies.

None of those are required for your Sep 7 sandbox.

---

## I'd actually standardize the repo around CPU first

For reproducibility, I'd configure the policy specs explicitly:

```yaml
algorithm: DQN
library: stable-baselines3
device: cpu
```

and similarly for SAC/PPO.

That gives you a very portable baseline:

> clone repo → `uv sync` → train/run on ordinary laptop.

Then if someone has CUDA, they can explicitly request

```yaml
device: cuda
```

later.

SB3 lets you specify the device, so this is easy.

And it avoids creating a hidden dependency where David clones the repo on some ordinary machine and training scripts unexpectedly demand CUDA.

---

# One thing I would *not* do: mix SB3 and raw PyTorch policy implementations

At this stage, avoid:

```text
DQN → custom PyTorch
SAC → another tutorial implementation
PPO → CleanRL
```

even though that can be educational.

You'd introduce algorithm-implementation differences into an evaluation project whose scientific object is supposed to be **policy behavior under perturbation**.

Better:

```text
                Stable-Baselines3
              /        |        \
           DQN        SAC       PPO
            |          |         |
        CartPole    Pendulum   MiniGrid
```

so training/checkpoint/evaluation APIs are as uniform as possible.

That reduces irrelevant variation.

Later, if you want to study algorithm choice itself, you can bring in CleanRL or your own PyTorch implementations.

---

## There is one alternative worth knowing: CleanRL

For **learning the algorithms**, CleanRL is often very useful because its implementations are intentionally compact, single-file and readable.

So I'd use the two libraries differently:

**Research infrastructure this week**

$$
\boxed{\text{Stable-Baselines3}}
$$

because you want reliable policy artifacts quickly.

**Learning later**

$$
\boxed{\text{CleanRL / textbook implementation}}
$$

when you want to actually trace how PPO/SAC/DQN work.

You don't want to conflate:

> “I need to understand SAC”

with

> “I need a frozen continuous-control policy by tonight.”

---

# A useful dependency layout

I'd expect approximately:

```toml
dependencies = [
    "gymnasium[classic-control]",
    "minigrid",
    "numpy",
    "scipy",
    "control",
    "stable-baselines3",
    "torch",
    "pandas",
    "matplotlib",
]
```

Strictly speaking, installing Stable-Baselines3 will bring PyTorch as a dependency, so you don't necessarily need to think of PyTorch as part of your direct application API.

Architecturally:

```text
your code
│
├── python-control
│     └── LQR
│
├── Stable-Baselines3
│     ├── DQN
│     ├── SAC
│     └── PPO
│           │
│           └── PyTorch
│
└── Gymnasium / MiniGrid
```

That is the stack I'd hand to the CHE-48 agent.

## One small modification I'd consider for CHE-48

For **MiniGrid PPO**, if it starts consuming disproportionate time because of observation wrappers or sparse-reward learning, don't let that block the CartPole work. Your research priority should remain:

$$
\boxed{\text{LQR vs DQN CartPole failure surfaces}}
$$

then Pendulum/SAC, then MiniGrid/PPO.

The CartPole comparison is much more valuable for your Monday discussion because you can reason about both the policy-design assumptions and the actual physical perturbations analytically.

So a plain CPU laptop + `python-control` + Stable-Baselines3/PyTorch is more than enough for the current milestone.

[1]: https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html?utm_source=chatgpt.com "Reinforcement Learning Tips and Tricks — Stable Baselines3 2.9.1a1 documentation"
[2]: https://python-control.readthedocs.io/en/latest/statesp.html?utm_source=chatgpt.com "State Space Analysis and Design — Python Control Systems Library 0.10.2-22-g75a658b documentation"
