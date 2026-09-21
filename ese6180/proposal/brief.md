# Anytime-Valid Active Certification of Controller Operating Regions

**Team member:** Shimin Chen

**Status:** Expanded working brief for the ESE6180 proposal. This is material to
compress into two pages, not the final submission. Repository links are retained
for traceability; the later LaTeX version must explain the setup without them.

**Agreed scope:** Active certification is the main study. The primary objective
is coverage of conditions certified acceptable at a fixed rollout budget.
CartPole recovery is the primary outcome. Linear-system cost sensitivity is a
separate, limited supporting result. Keep failure tolerance $\alpha$ and
certification error budget $\delta$ symbolic for now.
Keep both GP acquisition designs as candidates, and retain boundary-guided
retraining as an explicitly optional stretch.

## Abstract

Evaluating a controller only at nominal conditions can conceal failures under
changes in initial state or plant parameters. We propose to study how an
evaluator can adaptively allocate simulation rollouts to certify a large set of
acceptable operating conditions for a frozen controller. Each rollout produces
a binary recovery outcome; a condition is acceptable when its failure
probability meets a prescribed tolerance. A statistical certification procedure
will control the probability of any incorrect label across conditions and
evaluation times, while alternative sampling strategies determine where to
collect evidence. We will compare simple allocation rules, a bandit-based
strategy, and Gaussian-process-guided sampling on a reproducible CartPole
recovery task. A separate linear-system analysis will explain how closed-loop
cost sensitivity can justify generalization between nearby initial conditions.
The intended contribution is a control-focused synthesis, a transparent proof,
and an empirical account of when adaptive allocation improves certified coverage.

## Motivation and questions

A controller's usable operating region depends on more than nominal reward.
In this repository, frozen CartPole controllers can survive a complete episode
without recovering upright, and recovery changes sharply across initial pole
angle and pole length. These observations motivate evaluating a declared
trajectory outcome over a domain of operating conditions.

The existing [boundary experiments](../../docs/findings.md#cartpole-recovery-failure-boundary-study)
and [adaptive selector](../../src/active_eval_gym/boundary.py) provide
preliminary motivation and infrastructure. The proposed work adds sequential
statistical certification and controlled allocation comparisons. The existing
experiments have not established the proposed guarantees or efficiency gains.

The main question is:

> Under a fixed simulation budget, how much of a frozen controller's acceptable
> operating region can an adaptive evaluator certify while controlling the
> probability of ever issuing an incorrect label?

Supporting questions:

1. Does a spatial surrogate improve certified acceptable coverage beyond simply
   stopping work at resolved conditions? How do inappropriate spatial assumptions
   affect the gains?
2. In a linear closed-loop system, how do horizon and transient amplification
   determine continuous-cost sensitivity and the spatial reach of a justified
   evaluation?

Boundary reconstruction is secondary. Near-threshold points can be expensive to
certify; maximizing acceptable coverage need not favor the same samples as
accurately tracing a boundary.

## Formulation and feedback

Fix a controller $\pi$, horizon $H$, and finite candidate domain
$\mathcal Z=\{z_1,\ldots,z_m\}$. A closed-loop trajectory
$\tau_\pi(z,\xi)$ depends on the operating condition $z$ and episode
randomness $\xi\sim P_\xi(\cdot\mid z)$. Define the fixed binary metric

$$
Y_\pi(z,\xi)=\mathbf 1\{\text{trajectory fails the recovery requirement}\},
\qquad p_\pi(z)=\mathbb E_\xi[Y_\pi(z,\xi)].
$$

The target is

$$
S_\pi=\{z\in\mathcal Z:p_\pi(z)\le\alpha\}.
$$

The failure tolerance $\alpha$ concerns controller performance; the error
budget $\delta$ below concerns the evaluator's conclusions.

At round $t$, the evaluator selects $z_t$ using previous observations and
collects one rollout. Its statistical model requires

$$
\Pr(Y_t=1\mid\mathcal F_{t-1},z_t)=p_\pi(z_t),
$$

where $\mathcal F_{t-1}$ is the evaluator's history. Fresh independent
rollouts at selected conditions provide a simple sufficient implementation.
Time steps within a trajectory are not independent evaluation samples. The
policy, metric, and conditional randomness law remain fixed within a campaign.

The output comprises certified acceptable conditions $S_t$, certified
unacceptable conditions $F_t$, and unresolved conditions
$R_t=\mathcal Z\setminus(S_t\cup F_t)$. The intended guarantee is

$$
\Pr\!\left[
  \forall t\ge0:\quad S_t\subseteq S_\pi
  \ \text{and}\ F_t\subseteq\mathcal Z\setminus S_\pi
\right]\ge1-\delta.
$$

At budget $N$, the primary score is $C_N=|S_N|/m$. The secondary resolved
fraction is $(|S_N|+|F_N|)/m$. These describe a fixed, equally weighted grid,
not physical volume or deployment reliability. Fix the grid before comparative
campaigns; any different weighting needs its own declared measure.

The feedback loops are:

- **Within a rollout:** observation → frozen controller → action → next observation.
- **Across rollouts:** selected condition → trajectory/outcome → updated evidence
  and surrogate → next selected condition.

The L4DC connection is the evaluation of closed-loop behavior under plant and
initial-condition perturbations, together with analysis of how closed-loop
propagation changes metric sensitivity. The evaluator learns about a controller
while that controller remains fixed.

## Method and technical goals

### Sequential certification

Maintain a confidence sequence $[L_n(z),U_n(z)]$ from the first $n$ direct
rollout outcomes at each condition. Allocate error across conditions to obtain
coverage simultaneously over conditions and local sample counts. With
$n_t(z)$ samples, label a condition acceptable if $U_{n_t(z)}(z)\le\alpha$,
unacceptable if $L_{n_t(z)}(z)>\alpha$, and unresolved otherwise.

Derive an elementary time-uniform Hoeffding construction and use a Bernoulli
mixture confidence sequence for the main experiments, fixing its parameters in
advance. Running intersections retain previously justified bounds. The proof
will connect simultaneous coverage to sound labeling under adaptive allocation
and stopping. These are established statistical tools, not new concentration
results [5]. The [supporting notes](discussions/proposal-design-notes.md)
give the elementary construction and its assumptions.

All allocation comparisons use the same certifier, error budget, domain, and
rollout budget. Joint claims across multiple policies or metrics require error
allocation across that expanded family; a per-policy guarantee is labeled as such.

Certification covers directly tested grid points. GP predictions guide
acquisition but do not supply certificates. A misspecified GP can waste samples
without, by itself, invalidating the certification proof. Invalid assumptions
about the rollout distribution can invalidate it.

### Allocation

| Strategy | Role |
| --- | --- |
| Uniform allocation over the complete grid | Baseline without outcome-directed allocation |
| Round-robin among unresolved conditions | Isolate the benefit of stopping work on resolved conditions |
| Good-arm-identification allocation | Prioritize acceptable conditions without a spatial model |
| Bernoulli-GP-guided allocation | Test whether spatial predictions improve acceptable certification |

For the third strategy, use the sampling component of the modified MOSS-anytime
method in Cho et al. [6], with recovery rewards $1-Y_t$, retaining the common
certifier. This adaptation isolates allocation effects; the original
full-algorithm optimality theorem will not automatically be claimed for it.

Two GP acquisition designs remain candidates by choice:

- **Look-ahead certificate gain:** rank condition/batch-size pairs by the
  model-predicted probability of obtaining an acceptable certificate after
  that batch, divided by rollout cost. Hypothetical certificates use the
  same direct-data bound as actual certificates.
- **Simple surrogate ranking:** prioritize unresolved conditions predicted to
  be acceptable, with explicit exploration of uncertain or neglected conditions.
  This is easier to implement but does not directly estimate certification effort.

Choose between these before comparative experiments; implementing both is not
a core commitment. Use a Bernoulli observation model, informed by [3], rather
than importing Gaussian-observation guarantees for binary data.

The exact score, allowed batch sizes, surrogate settings, and exploration
schedule remain protocol decisions. Include a safeguard against permanently
neglecting conditions and a fallback when all predicted gains are zero.
Initialization and exploration count against the budget. Report computation
time separately. Boundary-focused straddle sampling is an optional diagnostic.

Independent per-condition certification does not share evidence spatially.
With identical per-condition sample streams and a fixed certifier, changing
their order does not reduce the local evidence each condition needs. The
hypothesis concerns **partial acceptable coverage at finite budgets**, not a
general reduction in total samples needed to resolve every point. Conditions
at the threshold may remain unresolved indefinitely.

### Limited control-theory result

Analyze a separate deterministic linear closed-loop system

$$
x_{k+1}=Gx_k,\qquad G=A-BK,
$$

with fixed $K$, state-cost matrix $Q\succeq0$, and finite-horizon cost

$$
J_H(x)=\sum_{k=0}^{H-1}x_k^\top Qx_k=x^\top P_Hx,\qquad
P_H=\sum_{k=0}^{H-1}(G^k)^\top QG^k.
$$

On $\|x\|_2,\|x'\|_2\le r$, derive

$$
|J_H(x)-J_H(x')|\le2r\|P_H\|_2\,\|x-x'\|_2.
$$

Thus $J_H(x)+2r\|P_H\|_2\|x-x'\|_2\le h$ suffices to establish
$J_H(x')\le h$. Study how the bound changes with horizon and transient
amplification, and compare bound-based coverage with the analytical cost
sublevel set. Known matrices provide a validation reference; simulation is not
needed to solve that known quadratic problem.

The finite-horizon identity does not require stability. A bound
$\|G^k\|_2\le c\rho^k$, $\rho<1$, gives an interpretable bound on
$\|P_H\|_2$. Stable eigenvalues alone do not imply Euclidean contraction or
exclude large transients.

This supporting result explains structural generalization for a continuous
cost. It does not prove smooth failure probabilities or continuum certificates
for CartPole. The quantized LQR and discrete-action PPO do not inherit the
linear example's assumptions. A probability-level extension needs additional
assumptions near the failure threshold and is outside the agreed core.

## Experiments and validation

**Primary task.** Reuse the explicitly modified CartPole recovery task: a
$90^\circ$ pole-angle termination cutoff, 500-step horizon, and recovery
defined as surviving with final-100-step RMS pole angle at most $5^\circ$.
Terminated trajectories are recovery failures. Retain cart-position termination.
Survival under these same modified dynamics is a secondary diagnostic.
Standard CartPole survival, if included, is a separate experiment.

The axes are initial pole angle and Gymnasium pole half-length. The recovery
wrapper fixes the angle exactly and retains seeded reset randomness in cart
position, cart velocity, and pole angular velocity. Document the installed
environment's reset law and package version: this randomness defines the
conditional probability being estimated. Replaying the same seed at the same
condition is not a fresh trial.

Start with the frozen nominal PPO checkpoint; repeat on quantized LQR if
resources permit. Core work requires no new policy training. Conclusions about
one checkpoint are not claims about all PPO training runs. Set grid resolution,
budgets, campaign repetitions, and the second-controller commitment after a
runtime assessment and before comparative experiments.

**Outputs.**

- Certified acceptable fraction versus rollout count, with variation across
  independent campaigns.
- Certified unacceptable and unresolved fractions, including spatial maps.
- Rollouts to reach predeclared acceptable-coverage targets, explicitly reporting
  targets not reached within budget.
- Sampling locations and counts that explain allocation behavior.
- Campaign-level frequency of any erroneous certificate where truth can be
  established; pointwise error fractions are not the simultaneous guarantee.

**Reference and statistical validation.** Use known-probability Bernoulli
surfaces, including near-threshold cases, to check the certifier and measure
false-certification behavior. These are statistical fixtures, not new RL
environments. The linear example separately validates the sensitivity result.
CartPole reference estimates require fresh, disjoint randomness and uncertainty
bounds. A finite Monte Carlo map is not exact ground truth; reference-ambiguous
points remain ambiguous in error assessments. Report reference cost separately
from each method's online budget.

Predetermine fresh random streams for each condition within a campaign.
Methods may share the same latent per-condition streams for paired comparison,
but each method sees only its own queried outcomes. Independent campaigns and
the reference study use separate streams. Fix hyperparameters and budgets using
development data before comparative campaigns. Retain raw trajectories,
versioned metrics, policy provenance, and the complete query history.

**Model sensitivity.** Compare predeclared surrogate settings, including an
overly smooth kernel. A narrow feature on a known-probability surface can test
whether acquisition neglects localized failures. This studies allocation
efficiency; it does not establish robustness to arbitrary simulator errors.

## Related work and contribution

**Level-set estimation.** Gotovos et al. [1] establish adaptive GP level-set
estimation with approximate classification guarantees under GP/noise
assumptions. Zanette et al. [2] directly study acquisition aimed at enlarging
the GP-estimated acceptable region, closely matching our objective. Their
exploration results under prior misspecification are distinct from finite-sample,
simultaneous certification under arbitrary misspecification. Letham et al. [3]
develop acquisition methods for Bernoulli level-set observations. These works
motivate the spatial proposer; their model-based labels differ from direct-data
certificates.

**Adaptive threshold decisions.** Locatelli et al. [4] study fixed-budget
thresholding bandits. Howard et al. [5] provide confidence-sequence tools for
time-uniform inference. Cho et al. [6] already combine adaptive sampling and
anytime-valid tests for good arm identification, closely matching the goal of
accumulating acceptable certificates. Our budget-truncated study retains a
fixed error guarantee and permits unresolved points, rather than requiring a
complete partition at the budget limit.

**Control evaluation.** Raghavan and Johansson [7] study adaptive region
classification through trajectory-level experimental design, where movement
and Markov-chain mixing constrain sampling. Our simulator permits independent
resets and keeps the inner controller fixed. Dietrich et al. [8] study
statistical certification of viable initial sets using importance sampling.
Their target is failure probability under a distribution over a candidate set;
ours is a collection of conditional probabilities indexed by operating
condition. Importance-weighted evaluation is outside the initial scope.

The contribution is a **control-focused synthesis and experimental study**,
with a transparent certification proof and limited sensitivity analysis.
Adaptive level-set estimation, acceptable-set expansion, and separating
acquisition from sequential testing are not claimed as new. A new optimality
theorem for the GP-guided allocation is not a promised deliverable.

## Deliverables and optional extension

1. A precise rollout-probability formulation and proof of simultaneous
   certification under the stated sampling assumptions.
2. A reproducible allocation comparison with honest budget accounting and
   unresolved outcomes.
3. A derivation and numerical illustration of linear cost sensitivity.
4. An analysis of when spatial modeling improves finite-budget acceptable
   coverage, and when it does not.

**Optional stretch: boundary-guided retraining.** If time permits, compare
boundary-selected and uniformly selected training conditions at equal training
budgets, freeze each updated policy, and evaluate with new data. Under a fixed
evaluation measure, report both gains and losses in acceptable coverage.
Additional boundary data does not imply monotonic growth of the true acceptable
set. Barrier construction and continual-learning theory are outside the core.

## Selected references

These eight references cover the immediate method, objective, and control
context. The [search notes](discussions/proposal-design-notes.md) record nearby
work considered but not included. [references.bib](references.bib) supplies
matching entries for later LaTeX conversion.

1. Alkis Gotovos, Nathalie Casati, Gregory Hitz, and Andreas Krause.
   **Active Learning for Level Set Estimation.** IJCAI, 2013.
   [Paper](https://people.csail.mit.edu/alkisg/files/gotovos13active.pdf).
2. Andrea Zanette, Junzi Zhang, and Mykel J. Kochenderfer.
   **Robust Super-Level Set Estimation using Gaussian Processes.**
   ECML PKDD 2018; proceedings published in 2019.
   [Published version](https://doi.org/10.1007/978-3-030-10928-8_17);
   [open preprint](https://arxiv.org/abs/1811.09977).
3. Benjamin Letham, Phillip Guan, Chase Tymms, Eytan Bakshy, and Michael
   Shvartsman. **Look-Ahead Acquisition Functions for Bernoulli Level Set
   Estimation.** AISTATS, 2022.
   [Paper](https://proceedings.mlr.press/v151/letham22a.html).
4. Andrea Locatelli, Maurilio Gutzeit, and Alexandra Carpentier.
   **An optimal algorithm for the Thresholding Bandit Problem.** ICML, 2016.
   [Paper](https://proceedings.mlr.press/v48/locatelli16.html).
5. Steven R. Howard, Aaditya Ramdas, Jon McAuliffe, and Jasjeet Sekhon.
   **Time-uniform, nonparametric, nonasymptotic confidence sequences.**
   The Annals of Statistics, 49(2):1055–1080, 2021.
   [Paper](https://arxiv.org/abs/1810.08240).
6. Brian M. Cho, Dominik Meier, Kyra Gan, and Nathan Kallus.
   **Reward Maximization for Pure Exploration: Minimax Optimal Good Arm
   Identification for Nonparametric Multi-Armed Bandits.** AISTATS, 2025.
   [Paper](https://proceedings.mlr.press/v258/cho25a.html).
7. Aneesh Raghavan and Karl Henrik Johansson.
   **Trajectory-Level Experimental Design for Fast Safety Parameter
   Estimation of Unknown Environments by Autonomous Systems.** L4DC, 2026.
   [Paper](https://proceedings.mlr.press/v331/raghavan26b.html).
8. Elizabeth Dietrich, Hanna Krasowski, Vegard Flovik, and Murat Arcak.
   **Importance Sampling for Statistical Certification of Viable Initial
   Sets.** arXiv:2604.02939, 2026, version 2.
   [Preprint](https://arxiv.org/abs/2604.02939v2).

## Open decisions and LaTeX conversion

- **Numerical protocol:** $\alpha,\delta$, grid, budgets, campaign count,
  reference precision, and coverage targets remain unspecified intentionally.
- **Acquisition:** choose between look-ahead certificate gain and simple
  surrogate ranking, then settle the formula and exploration schedule.
- **Experimental scope:** decide whether the LQR repetition fits the available
  time. PPO plus the linear supporting example form the minimum.
- **Optional stretch:** retain the retraining idea without making core completion
  depend on it.

The [course requirements](../resources/ESE6180-26Fall-Final-Project-Description.txt)
specify two pages maximum excluding references, in L4DC LaTeX, with title,
members, abstract, related work, formulation, and goals. For conversion, retain
a short abstract, the target/guarantee equations, one paragraph each on method
and control sensitivity, a compact experiment plan, and grouped related work.
Move proof and protocol detail into working notes. Replace repository links
with self-contained definitions and accurately attributed preliminary
observations; distinguish completed work from proposed work.
