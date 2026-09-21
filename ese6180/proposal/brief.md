# From Evaluation to Improvement: Boundary-Guided Training of Closed-Loop Controllers

**Team member:** Shimin Chen

**Status:** Expanded working brief for the ESE6180 proposal. This will be
compressed into a self-contained two-page L4DC LaTeX proposal. Repository links
currently provide traceability; they are not prerequisites for the eventual reader.

## Project progression and commitments

**Course foundation → Main research → Ambitious theoretical outcome**

| Stage | Purpose | Planned commitment |
| --- | --- | --- |
| **1. Course foundation** | Establish a dependable connection to learning, feedback, and control theory | A sound level-set evaluation/certification procedure, an accessible proof, and a linear closed-loop sensitivity result |
| **2. Main research** | Investigate boundary-guided retraining as the central research question | Controlled retraining experiments and theoretical exploration of expansion, regression, and training-condition selection |
| **3. Ambitious theoretical outcome** | Explain when boundary-guided updates can preserve or expand an acceptable region | Seek sufficient conditions or a restricted-system guarantee; a general monotonic-expansion theorem is not promised |

Boundary-guided retraining is a planned research component, not an optional
extra. The foundation provides a low-risk course-theory contribution even if
retraining yields a negative result or the strongest theorem remains open.
Completing the foundation alone would not complete the planned research study.

The progression is also methodological: **measure the region reliably →
intervene through training → investigate why the region changes**. Initial
research experiments should begin once the basic evaluator works, without
waiting for an extensive comparison of acquisition algorithms.

Keep failure tolerance $\alpha$ and statistical error budget $\delta$ symbolic.
Recovery remains the primary CartPole outcome. The continuous-cost linear
analysis is a tractable theoretical model, distinct from a failure-probability
guarantee for the nonlinear learned controller.

## Abstract

A controller's failures under changing initial conditions can reveal where
additional training may be useful. We propose to investigate whether training
near an estimated performance boundary expands a controller's acceptable
operating region while limiting regression in previously acceptable conditions.
The project progresses from a course foundation in level-set estimation,
sequential certification, and linear closed-loop sensitivity to a research study
of evaluation-guided retraining. We will freeze each policy for evaluation,
estimate its recovery probabilities, select training conditions, and evaluate
the updated policy using fresh data. Controlled CartPole experiments will compare
boundary-guided training with uniform and intermediate-difficulty curricula,
measuring gains and losses in acceptable coverage as well as evaluation and
training costs. Theoretical investigation will connect bounded changes in
controller performance to preservation of interior conditions and explore
conditions for net expansion. A general monotonic-improvement result is an
ambitious outcome rather than an assumption of the project.

## Motivation and research questions

The repository's [CartPole experiments](../../docs/findings.md#cartpole-recovery-failure-boundary-study)
show that survival and recovery can disagree, and that frozen controllers have
different recovery regions across initial angle and pole length. The existing
[perturbation specifications](../../src/active_eval_gym/envs/perturbations.py),
[raw rollout collector](../../src/active_eval_gym/rollout.py), and
[boundary selector](../../src/active_eval_gym/boundary.py) make those regions
inspectable. These are preliminary reported results; they do not demonstrate
benefits from retraining.

The central research question is:

> Can evaluation of a controller's performance boundary guide training that
> expands its acceptable operating region, while limiting degradation in
> previously acceptable conditions?

Supporting questions are:

1. Which information makes an operating condition useful for training:
   proximity to the acceptance threshold, intermediate difficulty, uncertainty
   in its estimated performance, or evidence that the policy can improve there?
2. Does boundary-guided training produce more net acceptable coverage than
   simple curricula at equal training cost, and after charging evaluation cost?
3. What assumptions on the dynamics, metric, controller class, and update rule
   permit preservation or expansion guarantees?

The intuition that boundary conditions are informative is a hypothesis about
training value. Informative evaluation conditions need not be informative
training conditions, and improving selected conditions can worsen others.

## Common formulation and feedback loops

For policy iteration $k$, freeze $\pi_k$. Let $z$ be an operating condition,
$H$ a fixed horizon, and $\xi\sim P_\xi(\cdot\mid z)$ the declared episode
randomness. For trajectory $\tau_{\pi_k}(z,\xi)$, define

$$
Y_k(z,\xi)=\mathbf 1\{\text{trajectory fails the recovery requirement}\},
\qquad p_k(z)=\mathbb E_\xi[Y_k(z,\xi)].
$$

Under a fixed tolerance $\alpha$, the true acceptable region is

$$
S_k=\{z:p_k(z)\le\alpha\}.
$$

Fix the evaluation domain and measure $\mu$ before comparing training methods.
Initially use an equally weighted finite grid $\mathcal Z$, so
$\mu(A)=|A|/|\mathcal Z|$. Define

$$
V_k=\mu(S_k),\qquad
g_k=\mu(S_{k+1}\setminus S_k),\qquad
\ell_k=\mu(S_k\setminus S_{k+1}).
$$

Then $V_{k+1}-V_k=g_k-\ell_k$. The main research concerns gains $g_k$,
regressions $\ell_k$, and net coverage change. Grid fractions are not physical
volume; extending to a continuous domain needs an explicit measure and
additional justification.

The project contains three nested feedback loops:

- **Control:** observation → action under the current controller → next observation.
- **Evaluation:** condition selection → frozen-policy rollout → updated evidence →
  next evaluation condition.
- **Improvement:** frozen policy $\pi_k$ → estimated performance region →
  training distribution $q_k$ → explicit retraining → frozen policy $\pi_{k+1}$.

The last loop can be summarized as

$$
\pi_k
\longrightarrow \widehat p_k,\ \widehat S_k
\longrightarrow q_k
\longrightarrow \operatorname{Train}(\pi_k,q_k)
\longrightarrow \pi_{k+1}.
$$

Training changes the target function from $p_k$ to $p_{k+1}$. Policies remain
fixed within an evaluation phase; updates occur only in a separate training
phase. This preserves the repository's separation of training, rollout
collection, perturbations, and metrics.

## Stage 1 — Course foundation

### Level-set estimation and sequential certification

Use level-set estimation to organize evaluation of a frozen controller [1, 2].
A surrogate can guide where to sample, while direct rollout evidence supports
certified acceptable, certified unacceptable, and unresolved labels.

For each selected condition, require fresh observations satisfying

$$
\Pr(Y_{k,t}=1\mid\mathcal F_{k,t-1},z_{k,t})=p_k(z_{k,t}),
$$

where the history includes earlier training and evaluation phases. One episode
supplies one binary outcome; its time steps are not independent samples.
Use simultaneous confidence sequences [3] and classify a condition only when
its upper failure-probability bound is at most $\alpha$, or its lower bound
exceeds $\alpha$. Otherwise abstain.

Write $\widehat S_{k,t}$ and $\widehat F_{k,t}$ for the certified sets.
For a joint claim across policy iterations, select budgets $\delta_k$ with
$\sum_{k\ge0}\delta_k\le\delta$ and establish conditional coverage within each
new evaluation phase. The intended conclusion is

$$
\Pr\!\left[
 \forall k,t:\quad
 \widehat S_{k,t}\subseteq S_k,\quad
 \widehat F_{k,t}\subseteq\mathcal Z\setminus S_k
\right]\ge1-\delta.
$$

Derive a simple Hoeffding/union-bound construction as the accessible proof.
A Bernoulli mixture confidence sequence is the intended practical alternative.
Use the same chosen certifier across comparisons. Fresh data are required for
the updated policy; old trajectories do not automatically certify its behavior.

Keep the evaluator study small: one transparent allocation baseline and one
LSE-guided evaluator. The previous look-ahead certificate-gain and simple GP
ranking designs remain candidates for the evaluation layer, not required
parallel implementations. Acquiring certificates and locating conditions useful
for retraining are separate objectives; a rule optimized for one must not be
assumed optimal for the other. Selection of the simple baseline and acquisition
rule remains a protocol decision.

GP predictions are not certificates, and the guarantee initially covers only
the finite grid. Certified acceptable coverage at a fixed evaluation budget is
a foundation diagnostic. The main research outcome is change in the controller's
region, assessed with a common reference protocol.

### Linear control result and a bridge to retraining

For fixed feedback $K$, analyze

$$
x_{j+1}=G_Kx_j,\qquad G_K=A-BK,
$$

with $Q\succeq0$ and

$$
J_{H,K}(x)=\sum_{j=0}^{H-1}x_j^\top Qx_j=x^\top P_H(K)x,
\qquad
P_H(K)=\sum_{j=0}^{H-1}(G_K^j)^\top QG_K^j.
$$

On $\|x\|_2,\|x'\|_2\le r$, derive

$$
|J_{H,K}(x)-J_{H,K}(x')|
\le2r\|P_H(K)\|_2\|x-x'\|_2.
$$

This gives a sufficient rule for generalizing a continuous-cost conclusion
between nearby initial states. Examine horizon and transient amplification;
stable eigenvalues alone do not imply Euclidean contraction.

A closely related controller-update bound supplies the bridge to the research:

$$
|J_{H,K'}(x)-J_{H,K}(x)|
\le r^2\|P_H(K')-P_H(K)\|_2=:b.
$$

Define $S_K^J=\{x:\|x\|_2\le r,\ J_{H,K}(x)\le h\}$. Then

$$
\{x:\|x\|_2\le r,\ J_{H,K}(x)\le h-b\}\subseteq S_{K'}^J,
$$

and losses are restricted to an old boundary band:

$$
S_K^J\setminus S_{K'}^J
\subseteq
\{x:\|x\|_2\le r,\ h-b<J_{H,K}(x)\le h\}.
$$

These identities and corollaries are elementary baseline results to derive
and illustrate, not novel claims. Known matrices provide analytical reference
sets. The finite-horizon statements do not require asymptotic stability.

This analysis is distinct from the quantized LQR and discrete-action PPO in
nonlinear CartPole. It does not automatically establish smooth failure
probabilities, infinite-horizon safety, or monotonic acceptable-region growth.

## Stage 2 — Main research: boundary-guided retraining

### What counts as a boundary?

| Object | Definition or diagnostic | Role to investigate |
| --- | --- | --- |
| Reliability boundary | $p_k(z)$ near the prescribed $\alpha$ | Target the acceptance requirement |
| Intermediate-difficulty region | Recovery success near $1/2$ | Seek conditions with potential training signal |
| Statistically unresolved region | Confidence interval straddles a threshold | Seek missing evaluation evidence |

These sets need not coincide. Mixed success can reflect reset randomness as
well as policy limitations; it is not proof that further training will help.
A threshold band $|p_k(z)-\alpha|\le\varepsilon$ is a band in probability
values, not necessarily a thin geometric band in state space.

### Controlled training intervention

Start from the same frozen nominal PPO checkpoint and branch into matched
additional-training runs. Compare three training-condition strategies:

1. **Uniform:** sample the declared operating domain without performance-based
   prioritization.
2. **Boundary-guided:** prioritize a band around the estimated reliability
   boundary.
3. **Intermediate difficulty:** use a Sampling-for-Learnability-inspired
   score $\widehat p_k(z)(1-\widehat p_k(z))$ [6].

Use the same evaluator for the two targeted curricula in the initial comparison
to isolate the training selection rule. Estimating the intermediate-difficulty
region must receive explicit support in the query plan; do not starve it with
an evaluator that only samples near $\alpha$. A full reproduction of SFL,
with its own data-collection procedure, is a later comparison if warranted.

A shared mixture makes the intervention explicit:

$$
q_k=(1-\lambda)q_{\rm base}+\lambda q_{{\rm target},k}.
$$

Here $q_{\rm base}$ provides common background coverage, potentially mixing
nominal and broadly sampled conditions. Hold it and $\lambda$ fixed across
methods. The uniform method uses a uniform target component. Mixture weights,
boundary-band width, treatment of uncertain points, and fallback behavior
when the selected band is empty remain to be specified. The mixture is a
retention mechanism to test, not a guarantee against forgetting.

Use a common recovery-oriented training objective across methods. The original
survival reward can reward trajectories that never recover upright. The exact
reward design is open and must be fixed before the comparative experiment;
keep training reward distinct from the unchanged evaluation criterion.

With PPO, selected conditions determine where to collect fresh on-policy
training rollouts. Replaying a condition does not mean inserting arbitrary old
evaluation trajectories into an on-policy update. Fix and record evaluation
action mode separately from training-time action sampling.

### Experimental progression

Begin with initial-angle variation at nominal pole length to isolate recovery
from different starts. The existing angle–length domain is the natural next
experiment, introducing plant variation without adding another environment.

First study a single update $\pi_0\to\pi_1$ across multiple additional-training
seeds. Starting from one shared checkpoint isolates the curriculum intervention;
broader claims about training require additional independently trained starting
checkpoints. Then study repeated evaluation–retraining rounds to examine whether
gains accumulate, plateau, or reverse. The minimum commitment for the multi-round
study is awaiting a scope decision.

Evaluate:

- Gains $g_k$, losses $\ell_k$, and net acceptable-coverage change under the
  same fixed measure.
- Nominal performance and performance in previously acceptable conditions.
- Maps and trajectories showing where improvements or regressions occur.
- Variation across training seeds and sensitivity to the training-condition rule.
- Evaluation cost, training transitions, and computation time.

First match training transitions and optimization settings to isolate training
effects. Then account for boundary-discovery cost in the total evaluation-plus-
training budget. Equal episode counts alone are insufficient because failures
shorten episodes. Keep the independent reference evaluation budget explicit.

### Research interpretation

The experimental study and theoretical exploration are required research
activities; favorable results are not assumed. A negative result should explain
whether the issue is boundary estimation, lack of useful training signal,
reward mismatch, or degradation elsewhere. Counterexamples and restricted
sufficient conditions are useful outcomes.

## Stage 3 — Ambitious theoretical outcome

The ambitious goal is to connect the selection of training conditions and the
actual controller update to preservation or expansion. The baseline bound
localizes possible loss; it does not show that a training algorithm obtains
enough gain to offset that loss.

Candidate theoretical steps are:

1. Bound performance drift in terms of update size, horizon, and closed-loop
   amplification, rather than only an after-the-fact matrix difference.
2. Construct examples where improving selected conditions degrades other
   conditions, explaining why unconditional monotonic expansion fails.
3. Identify assumptions on a restricted controller class and training update
   that guarantee improvement in some outside conditions while controlling
   loss of previously acceptable ones.
4. Derive a sufficient condition for net expansion, and investigate whether
   a boundary-guided update satisfies it.

The choice of continuous-cost expansion as the first target versus
failure-probability expansion as the primary target is awaiting confirmation.
The linear model offers a tractable route; a probability-level result also
needs assumptions about randomness and mass near the failure threshold.
Neither pointwise cost improvement nor empirical growth of certified sets
alone proves growth of the true failure-probability level set.

A theorem that assumes improvement everywhere would not explain the value of
boundary-guided training. The research challenge is to connect the proposed
selection/update mechanism to the required improvement and retention
conditions. No general monotonic-growth theorem for PPO is promised.

## Task definition, evidence, and reproducibility

The primary CartPole task retains the repository's modified $90^\circ$
pole-angle termination cutoff, cart-position termination, and 500-step horizon.
Recovery means surviving and achieving final-100-step RMS pole angle at most
$5^\circ$. Terminated episodes fail recovery. Survival under those same
dynamics is a secondary diagnostic; standard CartPole survival is a separate
task if included.

Fix the initial angle exactly and retain the declared reset randomness in cart
position, cart velocity, and pole angular velocity. Record the installed reset
law and package versions. Repeating the same seed at the same condition is not
a fresh trial. Retain raw trajectories, requested/applied actions where relevant,
policy/checkpoint hashes, training and episode seeds, metric versions,
training-distribution specifications, and query histories.

Use known-probability Bernoulli fixtures to check statistical validity and a
small analytical linear example to check the theory. These do not expand the
RL environment scope.

Updated policies receive fresh evaluation data. A common reference study
estimates each $S_k$ with uncertainty; a finite Monte Carlo map is not exact
ground truth. Ambiguous points cannot be counted automatically as correct or
as improvements. Separate certificate growth caused by more evaluation from
performance changes caused by retraining.

The previous final boundary study reused pilot seeds 0–4 within final seeds
0–49; it is not wholly held out. Existing Wilson intervals are pointwise
summaries. Preserve those artifacts as preliminary evidence and use a new
versioned protocol for the new claims. Details and proofs are in the
[supporting notes](discussions/proposal-design-notes.md).

## Related work and research positioning

**Evaluation foundation.** Gotovos et al. [1] develop adaptive GP level-set
estimation with model-dependent approximate classification guarantees.
Letham et al. [2] address Bernoulli observations. Howard et al. [3] provide
time-uniform confidence-sequence tools. They support the evaluation layer;
the proposal does not claim a new LSE or concentration framework.

**Training-condition selection.** Reverse Curriculum Generation [4] constructs
performance-adaptive curricula of starting states. Prioritized Level Replay [5]
prioritizes environment configurations using estimated learning potential.
Sampling for Learnability [6] directly targets mixed-success conditions and
also studies shifts in the preferred success rate. Boundary-guided training
and changing a sampling threshold are therefore not sufficient novelty claims.

**Control guarantees.** Berkenkamp et al. [7] connect model-based policy learning
and safe-region expansion through Lyapunov analysis and statistical dynamics
models. Their stability certificates differ from our finite-horizon recovery
probabilities and reset-based experiments. The comparison motivates stating
precisely which assumptions would support an expansion result.

The potential research contribution is a controlled investigation of
**evaluation-defined boundary selection, improvement toward a declared
acceptance requirement, and retention of prior competence**, together with
theoretical characterization in a tractable control setting. Establishing a
new method or theorem requires sharper positioning after the update mechanism
is fixed. A rigorous synthesis, counterexample, or negative experimental result
can still satisfy the research objectives.

## Selected references

The main bibliography now follows the foundation-to-retraining progression.
Earlier evaluator-allocation references remain linked in the
[supporting notes](discussions/proposal-design-notes.md), without expanding the
core citation list. Matching entries are in [references.bib](references.bib).

1. Alkis Gotovos, Nathalie Casati, Gregory Hitz, and Andreas Krause.
   **Active Learning for Level Set Estimation.** IJCAI, 2013.
   [Paper](https://people.csail.mit.edu/alkisg/files/gotovos13active.pdf).
2. Benjamin Letham, Phillip Guan, Chase Tymms, Eytan Bakshy, and Michael
   Shvartsman. **Look-Ahead Acquisition Functions for Bernoulli Level Set
   Estimation.** AISTATS, 2022.
   [Paper](https://proceedings.mlr.press/v151/letham22a.html).
3. Steven R. Howard, Aaditya Ramdas, Jon McAuliffe, and Jasjeet Sekhon.
   **Time-uniform, nonparametric, nonasymptotic confidence sequences.**
   The Annals of Statistics, 49(2):1055–1080, 2021.
   [Paper](https://arxiv.org/abs/1810.08240).
4. Carlos Florensa, David Held, Markus Wulfmeier, Michael Zhang, and Pieter
   Abbeel. **Reverse Curriculum Generation for Reinforcement Learning.**
   CoRL, 2017. [Paper](https://proceedings.mlr.press/v78/florensa17a.html).
5. Minqi Jiang, Edward Grefenstette, and Tim Rocktäschel.
   **Prioritized Level Replay.** ICML, 2021.
   [Paper](https://proceedings.mlr.press/v139/jiang21b.html).
6. Alex Rutherford, Michael Beukman, Timon Willi, Bruno Lacerda, Nick Hawes,
   and Jakob Foerster. **No Regrets: Investigating and Improving Regret
   Approximations for Curriculum Discovery.** NeurIPS, 2024.
   [Paper](https://arxiv.org/abs/2408.15099).
7. Felix Berkenkamp, Matteo Turchetta, Angela P. Schoellig, and Andreas Krause.
   **Safe Model-based Reinforcement Learning with Stability Guarantees.**
   NeurIPS, 2017. [Paper](https://arxiv.org/abs/1705.08551).

## Open decisions and two-page conversion

- **Scope decisions requested:** whether multiple retraining rounds are required
  and which level-set notion is the first ambitious theoretical target.
- **Protocol decisions:** $\alpha,\delta$, grid, evaluation and training budgets,
  training seeds, reward design, mixture weights, boundary-band width,
  uncertainty handling, and reference precision.
- **Evaluator design:** pick one simple baseline and one LSE-guided rule;
  do not let evaluator optimization displace the retraining investigation.
- **Extensions:** angle–length training and multiple starting checkpoints can
  follow the initial-angle experiment.

The [course requirements](../resources/ESE6180-26Fall-Final-Project-Description.txt)
require two pages maximum excluding references, in L4DC LaTeX. Retain the
three-stage progression prominently. Give the foundation one compact
method/theory paragraph, reserve the largest share for the research question
and retraining experiment, and describe the ambitious theorem as an objective
with explicit assumptions to investigate. Include title, team member, abstract,
related work, formulation, goals, and the feedback loops. Move proof details
to working notes, replace repository links with self-contained descriptions,
and distinguish preliminary results from planned work.
