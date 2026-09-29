# Evaluation-Guided Domain Randomization: Certificate Margins and Controller Updates

**Team member:** Shimin Chen

**Status:** v3 working proposal for feedback from Prof. Matni. This brief and the
[design notes](discussions/proposal-design-notes.md) supersede the v2 scope. The
[LaTeX submission draft](latex/proposal.tex) and its PDF now summarize this scope
in two pages, with references on a separate page. The [v3 revision record](discussions/v3-revisin-notes.md)
explains the changes. The paper choice below is a working recommendation, not
an endorsement already received from the professor.

## Project progression and commitments

**Course foundation → Main theoretical investigation → Ambitious extension**

| Stage | Required or optional outcome | Supporting evidence |
| --- | --- | --- |
| **1. Course foundation** | Specialize certificate learning to a frozen, discrete-time linear closed loop and a bounded quadratic certificate class; give an accessible generalization proof | Compare a candidate fitted from trajectories with exact matrix verification |
| **2. Main investigation** | Analyze one evaluation-guided update: prove a sufficient certificate-preservation condition and exhibit improvement/regression tradeoffs for a specified update rule | Small analytical examples and numerical checks of margin, update size, and conservatism |
| **3. Ambitious extension** | Seek cost-region expansion under an actual restricted update, or a justified local nonlinear extension; choose one after Stages 1–2 | CartPole illustrates behavior and the limits of transferring the linear result |

The central deliverable is a rigorous specialization and analysis. Stage 1 alone
is not the whole project: the certificate must be connected to a concrete update
in Stage 2. Expansion is optional; preservation and a precise counterexample to
unconditional improvement remain meaningful outcomes. One outer update is the
minimum study. Repeated PPO rounds, neural certificates, barrier synthesis, and
an acquisition-algorithm benchmark are not required deliverables.

## Abstract

Evaluation-guided domain randomization uses a controller's observed weaknesses
to select subsequent training conditions, but improving selected conditions can
degrade behavior elsewhere. We propose a theory-focused study in a small family
of discrete-time linear systems. First, we will specialize trajectory-based
certificate learning to quadratic Lyapunov candidates, distinguishing statistical
generalization from verified stability. We will then analyze how a verified
decrease margin constrains one evaluation-guided update of a shared feedback
gain. Finite-horizon cost bounds and explicit counterexamples will explain what
stability preservation does and does not imply about acceptable performance
coverage. Small analytical experiments will test the bounds and their
conservatism. Existing CartPole recovery experiments motivate the question and
provide a possible nonlinear illustration; they do not inherit the linear
guarantees. Expansion under a specified update is an extension rather than an
assumed property of boundary-guided training.

## Motivation and research questions

The evaluation-guided domain-randomization idea was inspired by David Snyder.
In office hours, Prof. Matni emphasized clear scope, a central
theoretical contribution, and relevant work on learning Lyapunov/barrier
functions. This revision follows that guidance without assuming which paper
he intended to recommend.

The repository's [preliminary CartPole findings](../../docs/findings.md#cartpole-recovery-failure-boundary-study)
show that survival and recovery can disagree and that fixed policies have
different recovery regions. They motivate evaluating competence across
conditions, but do not establish that retraining improves those regions.

> What can trajectory-based evaluation certify about a restricted closed-loop
> family, and which certificate margins are sufficient to preserve a guarantee
> during an evaluation-guided controller update?

1. **Certificate generalization:** what does satisfaction on independent sample
   trajectories imply about a new trajectory from the same distribution?
2. **Update preservation:** how do the decrease margin, plant matrices, and
   update size determine whether a verified certificate remains valid, and can
   a simple update rule enforce that condition?
3. **Limits of improvement:** when can targeted training improve its own
   objective while losing previously acceptable conditions? What additional
   structure would be needed for net expansion?

The project is specifically L4DC because feedback changes the trajectories on
which learned components are evaluated; statistical certificate learning and
control-theoretic robustness must be connected explicitly. Boundary proximity
is a hypothesis about training usefulness, not a guarantee of a useful gradient.

## Model, sets, and feedback loops

The primary analytical model is

$$
x_{t+1}=A(\theta)x_t+B(\theta)u_t,\qquad u_t=-Kx_t,
\qquad G_K(\theta)=A(\theta)-B(\theta)K.
$$

Use a small fixed finite plant set $\Theta$, full-state observations, and one
shared gain $K$. A plant stays fixed within each trajectory. Dynamics are
deterministic and continuous-action, without quantization, saturation, process
noise, or early termination. Matrices are known so exact verification is
possible; fitting certificates from transitions is a controlled learning
exercise, not system identification. Start with one plant, then a small family.
Joint stabilizability and existence of a common quadratic certificate are
separate assumptions; failure to find the latter does not establish instability.

| Object | Domain and meaning |
| --- | --- |
| $V_P(x)=x^\top Px$ | Candidate certificate on state space, with $0<aI\preceq P\preceq bI$ |
| $\mathcal E_c(P)=\{x:V_P(x)\le c\}$ | An invariant state ellipsoid only when the required decrease condition is verified |
| $S_K^J=\{(\theta,x):\|x\|\le r,\ J_{H,K}(\theta,x)\le h\}$ | Finite-horizon cost-acceptable set over plants and initial states |
| $S_k^p=\{z:p_k(z)\le\alpha\}$ | CartPole reliability set over plant/reset conditions $z=(\theta,s)$ |

For fixed $Q\succeq0,R\succeq0$, define

$$
J_{H,K}(\theta,x)=\sum_{t=0}^{H-1}(x_t^\top Qx_t+u_t^\top Ru_t)
=x^\top P_H(K,\theta)x.
$$

$P_H$ is a cost matrix, not automatically the certificate matrix $P$. In
CartPole, $x_0\sim\nu_s$ and remaining randomness have declared conditional
laws; $p_k(z)$ is the recovery-failure probability of frozen $\pi_k$.

For each set notion, fix its reference probability measure $\mu$ before
comparisons. The changing training distribution $q_k$ does not change $\mu$.
Coverage $C_k=\mu(S_k)$ satisfies

$$
C_{k+1}-C_k=\mu(S_{k+1}\setminus S_k)-\mu(S_k\setminus S_{k+1}).
$$

Declare a separate measure for each domain. A grid fraction is not physical
volume, and certificate-violation probability is not task-failure probability.

- **Control:** state → frozen policy action → next state.
- **Evaluation:** selected condition → trajectory → certificate/performance
  evidence → next condition.
- **Improvement:** evidence → training distribution → proposed gain update →
  margin check → accepted gain → fresh evaluation.

Stage 1 uses a distribution fixed in advance. Adaptive evaluation is an optional
allocation mechanism, not a substitute for the i.i.d. assumption in its proof.
Policies remain frozen within each collection phase; updates occur separately.

## Stage 1 — Specialize a certificate-learning result

The preferred anchor is Boffi et al., *Learning Stability Certificates from
Data* [1]. The deliverable is a discrete-time quadratic specialization of its
learning viewpoint, using Chapter 5's finite-cover/uniform-convergence tools.
This is a pedagogical adaptation, not a reproduction of its strongest rate or
an assertion that its continuous-time theorem applies unchanged.

Fit $P$ from independent trajectories of frozen $K$, with plants and initial
conditions drawn from a fixed distribution $D_0$. Restrict $aI\preceq P\preceq bI$
so margins cannot be enlarged arbitrarily by scaling. Study violations of

$$
V_P(x_{t+1})-V_P(x_t)\le-\eta\|x_t\|_2^2,\qquad \eta>0.
$$

The [design notes](discussions/proposal-design-notes.md) give a bounded trajectory
loss, finite-cover proof, and treatment of the equilibrium. One trajectory is
one independent sample; its time steps are dependent. Uniformity over the
certificate class is necessary because $P$ is fitted to the data. A pointwise
interval for a preselected $P$ does not provide that uniformity.

Separately verify, using known matrices,

$$
G_K(\theta)^\top P G_K(\theta)-P\preceq-\eta_\theta I,
\qquad \eta_\theta>0,
$$

on every plant for which stability is claimed. This all-state inequality makes
$V_P$ a Lyapunov function there and its sublevel sets invariant. The statistical
bound alone controls violation risk under the sampled distribution over the
declared horizon, not all-state or infinite-horizon stability. A learned
candidate may fail exact verification; report that outcome.

The direct course connections are Chapter 3, Theorem 3.10 and Example 3.12, and
Chapter 5's uniform convergence and trajectory-level sampling. Chapter 2
motivates the controller-update question. Chapter 4 becomes central only if
learning dynamics or an expert policy is deliberately added, which is outside
the minimum scope.

## Stage 2 — Analyze one restricted evaluation-guided update

Hold verified $P$ fixed and write $K'=K+\Delta K$. For one plant, let
$G=G_K(\theta)$ and $d_\theta=\|B(\theta)\Delta K\|_2$. Expanding the
quadratic form gives the sufficient preservation condition

$$
2\|G^\top P\|_2d_\theta+\|P\|_2d_\theta^2<\eta_\theta.
$$

Enforce it on a fixed retention set of plants, using a common $P$ and their
verified old margins. The retained decrease guarantees stability and invariance
of the old certificate ellipsoids on those plants. It does not guarantee
preservation of a cost threshold or of the CartPole recovery criterion.
For an unconstrained stable linear system the true region of attraction is
already global; enlarging a certificate ellipsoid alone would not establish
growth of that true region. Cost coverage supplies a separate meaningful target.

Make the update concrete: evaluate the old gain on a fixed cost grid, target
conditions near $J_{H,K}=h$, and mix with fixed background coverage,

$$
q=(1-\lambda)q_{\rm base}+\lambda q_{\rm target}.
$$

Propose one gradient step for $F_q(K)=\mathbb E_q[J_{H,K}(\theta,x)]$ and
backtrack until the preservation test passes with a declared positive residual
margin. Hold $q$ fixed throughout the step. If no useful step is accepted,
record that outcome; retaining $K$ is the fallback. Small steps preserve a
strictly positive old margin but need not produce useful progress. The guarantee
comes from the update restriction, not the sampling mixture.

The required analysis has three parts:

1. Prove margin preservation and connect it to this accepted update, including
   zero-gradient and rejected-step cases.
2. Derive an explicit finite-horizon cost-drift bound in $\|\Delta K\|$, accounting
   for both changed dynamics and the $K^\top RK$ input penalty. Bound possible
   regression near the old cost threshold.
3. Give a targeted-gradient counterexample showing that lower training cost,
   or retained stability, need not increase reference coverage.

The old bound $r^2\|P_H(K',\theta)-P_H(K,\theta)\|$ remains a baseline, not the
whole theory deliverable. The [explanation](discussions/linear-control-and-retraining-explained.md)
retains its derivation and examples. These elementary results are not claimed
as novel. Numerical work compares exact/bounded drift, checked margins, and
coverage gains/losses. Start with uniform versus cost-boundary targeting from
the same gain, with matched budgets and the same margin rule.

## Stage 3 — Optional expansion or nonlinear extension

The preferred first extension is net expansion of $S_K^J$ under a specified
update. It must establish enough gains to offset losses, rather than assume
improvement everywhere. Set inclusion and positive net coverage differ.
Certificate growth caused by less conservative inference is also not controller
improvement.

An alternative is a local nonlinear extension with an explicit remainder bound,
a preserved equilibrium, and a verified invariant neighborhood. Choose one
extension after the linear analysis. Failure-probability expansion and repeated
PPO retraining are later questions, not minimum outcomes.

## Supporting CartPole study and evaluation protocol

Keep the existing task: relaxed $90^\circ$ angle cutoff, retained cart
termination, horizon 500, and recovery defined by survival plus final-100-step
RMS angle at most $5^\circ$. Recovery remains the primary CartPole metric.

The repository's quantized LQR applies one of two fixed forces, including at
zero state. Its closed loop is not $x^+=(A-BK)x$. Discrete-action PPO has a
related mismatch with smooth linear-feedback assumptions. Local plant
linearizability therefore does not transfer the theorem to these policies or
justify large-angle recovery guarantees. Practical stability, safety, and
recovery would require separate analyses.

Use existing frozen-policy evidence or one matched PPO update only if it answers
a question raised by the theory. If retraining is included, compare uniform and
reliability-boundary targets first; intermediate difficulty $\widehat p(1-\widehat p)$
is an optional third comparator. It peaks at $1/2$, which need not equal $\alpha$.
Keep objective, mixture, optimizer, and budgets fixed, use fresh on-policy
training data, and evaluate frozen checkpoints on fresh reference episodes.
Fix any recovery-oriented training reward before comparing curricula.

For grid-based recovery claims, use fresh Bernoulli outcomes under a declared
conditional law and simultaneous confidence bounds [4]. An upper failure bound
at most $\alpha$ certifies acceptability; a lower bound above $\alpha$ certifies
unacceptability; otherwise abstain. Allocate error across conditions, methods,
and policy phases. Keep $\alpha,\delta$ symbolic until protocol design. If
certificate-learning and rollout claims share a joint probability statement,
allocate separate portions of $\delta$ to them too.

The elementary time-uniform proof and optional GP acquisition remain supporting
infrastructure in the design notes. GP predictions do not certify unobserved
conditions. Learning a quadratic certificate and fitting a performance surrogate
are different tasks. No GP implementation is needed to complete the theory.

## Reproducibility and interpretation

Retain raw trajectories independently of metrics. Record environment ID/package
versions, plant/reset specs, checkpoint hashes, training/action/episode seeds,
action rule, metric versions, sampling distributions, query histories, and
update rejections. Also retain linear matrices, fitted $P$, normalization,
old/new gains, and verification tolerances. Floating-point eigenvalue checks
need a declared numerical margin; the exact-arithmetic proposition supplies
the proof, while computations provide numerical evidence.

Never select seeds by performance. Report unresolved conditions, keep rewards
distinct from metrics, and count evaluation/training transitions as well as
episodes. Keep reference data separate from selection/training. The earlier
boundary study's pilot seeds 0–4 overlap final seeds 0–49; its map is not wholly
held out and its Wilson intervals are pointwise. Preserve those artifacts and
version any new protocol. This revision claims no new experiments or completed
paper reproduction.

## Selected literature and course fit

1. **Boffi et al., Learning Stability Certificates from Data.** CoRL 2020,
   PMLR publication 2021. Preferred foundation for certificate learning and
   trajectory generalization. [Publisher](https://proceedings.mlr.press/v155/boffi21a.html),
   [local PDF](literature/boffi21a-learning-stability-certificates.pdf).
2. **Zhang et al., Adversarially Robust Stability Certificates can be
   Sample-Efficient.** L4DC 2022. Optional robustness context involving
   incremental stability; its adversarial theorem is not a required reproduction.
   [Publisher](https://proceedings.mlr.press/v168/zhang22a.html),
   [local PDF](literature/zhang22a-adversarially-robust-stability-certificates.pdf).
3. **Berkenkamp et al., Safe Model-based Reinforcement Learning with Stability
   Guarantees.** NeurIPS 2017. Context for model uncertainty and safe-region
   expansion, with additional assumptions including a given Lyapunov candidate.
   [Paper](https://arxiv.org/abs/1705.08551), [local PDF](literature/1705.08551v3.pdf).
4. **Howard et al., Time-uniform, nonparametric, nonasymptotic confidence
   sequences.** Annals of Statistics 2021. Supporting rollout inference.
   [Paper](https://arxiv.org/abs/1810.08240), [local PDF](literature/1810.08240v9.pdf).

Fujinami et al.'s domain-randomized LQR work remains controller-learning context;
its average-objective convergence does not imply acceptable-region expansion.
Gotovos/Letham support optional level-set acquisition; Florensa/Jiang/Rutherford
provide curriculum precedents. Their algorithms are not all implementation
requirements. Entries are in [latex/references.bib](latex/references.bib); download
provenance is in [literature/certificate-learning-sources.md](literature/certificate-learning-sources.md).

The [syllabus](../resources/ESE6180-26Fall-Syllabus.pdf), page 2, calls for a
theory-focused project. The [project requirements](../resources/ESE6180-26Fall-Final-Project-Description.txt)
allow pedagogical simplifications of existing theory with numerical verification.
The proposed contribution is an adaptation and analysis of its limits;
methodological novelty is not presumed.

## Decisions for professor feedback and two-page conversion

- Is discrete-time quadratic certificate learning followed by one verified
  update an appropriate central theoretical contribution?
- Is Boffi et al. the right anchor, or is another Lyapunov/barrier result more
  suitable? These are candidate readings, not papers attributed to his advice.
- Is a finite family with a common quadratic certificate appropriately scoped,
  with cost expansion optional and CartPole supporting the analysis?

Numerical choices remain: family matrices, certificate bounds, decrease/learning
margins, sampling distribution, weights/horizon, retention plants, target band,
mixture weight, step size, and budgets. Settle assumptions before interpreting
numerical outcomes.

The two-page LaTeX now leads with the theory and makes Stages 1–2 the main
contribution. It includes author, abstract, formulation, feedback loops, related
work, and goals, with experiments supporting the analysis and proof details
retained in these notes. The [email draft](discussions/email-prof-matni-v3.md)
requests feedback on the foundation and scope, with an Overleaf-link placeholder.
