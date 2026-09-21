# Supporting notes: from evaluation to boundary-guided retraining

These notes support [brief.md](../brief.md). They retain derivations and source
selection details that do not belong in the two-page proposal.

## Current scope and progression

**Course foundation → Main research → Ambitious theoretical outcome**

- **Foundation:** LSE and sequential certification, plus an accessible linear
  control sensitivity result, provide a dependable course-theory component.
- **Main research:** boundary-guided retraining experiments and theoretical
  investigation are planned work, not an optional stretch.
- **Ambitious outcome:** investigate conditions for preserving or expanding
  acceptable regions. A general monotonic-improvement theorem is not promised.

Shimin Chen retains recovery as the primary CartPole metric and leaves
\(\alpha,\delta\) symbolic. The evaluation layer is reduced to one simple
baseline and one LSE-guided rule; the two earlier GP acquisition designs remain
candidates, not a requirement to implement both. Acquisition for certification
and selection for retraining have different objectives.

Two scope questions are pending: whether multiple retraining rounds are
required, and whether the first ambitious theorem targets continuous-cost
level sets or failure-probability level sets. Numerical protocol and training
reward design also remain open. The main brief is authoritative for the
current commitments.

## Elementary certification construction

This is an elementary derivation for the project, not a new theorem attributed
to the related-work papers. Let there be \(m\) fixed conditions, and suppose
each condition's potential observations form an i.i.d. Bernoulli sequence with
mean \(p(z)\). The adaptive evaluator reveals fresh observations from these
sequences, never future outcomes. Write

\[
\widehat p_n(z)=\frac1n\sum_{i=1}^nY_i(z),\qquad
r_n=\sqrt{\frac{\log(2mn(n+1)/\delta)}{2n}}.
\]

Let \(I_0(z)=[0,1]\) and

\[
I_n(z)=[\widehat p_n(z)-r_n,\widehat p_n(z)+r_n]\cap[0,1].
\]

Hoeffding's inequality gives

\[
\Pr\{p(z)\notin I_n(z)\}\le 2e^{-2nr_n^2}
=\frac{\delta}{mn(n+1)}.
\]

Summing over \(z\) and all positive \(n\), using
\(\sum_{n\ge1}1/[n(n+1)]=1\), yields simultaneous coverage of at least
\(1-\delta\). On that event, running intersections
\(C_n(z)=\bigcap_{j=0}^n I_j(z)\) also contain \(p(z)\).
Using their endpoints in the brief's labeling rule proves soundness at every
global evaluation round, including data-dependent stopping times.

Cross-condition independence is not required by the union bound, but adaptive
reuse of correlated, already exposed randomness needs care. The proposed
implementation uses fresh per-condition randomness to satisfy the conditional
sampling model. Pair methods through latent streams that remain hidden from
each other's evaluator histories.

An empty intersection signals a contradiction among the confidence sets:
abstain and record the event, rather than issuing both labels. If
\(\Delta_z=|p(z)-\alpha|>0\), the sufficient condition
\(2r_n<\Delta_z\) resolves that point on the coverage event. This illustrates
gap-dependent difficulty, not an efficiency theorem for an arbitrary proposer.
Points at the threshold need not resolve.

For experiments, compare the elementary construction during validation with a
Bernoulli mixture confidence sequence, then fix one common certifier for all
allocation comparisons. Howard et al. explain mixture constructions and
beta-binomial bounds (Section 4.1 and Appendix A.3 of the linked arXiv version).
Any tuning must be fixed in advance or covered by its own valid construction.
[Howard et al.](https://arxiv.org/pdf/1810.08240)

## Extension across retrained policies

The preceding construction is for one frozen policy. In iteration \(k\),
condition on all prior training/evaluation history and substitute \(\delta_k\)
for \(\delta\), using fresh rollout outcomes of the now-fixed \(\pi_k\).
Predetermine nonnegative budgets with \(\sum_k\delta_k\le\delta\).
Conditional phase-wise coverage, the tower property, and a union bound give
the joint guarantee across iterations in the brief.

For example, a fixed number of phases can receive equal budgets, or a countable
sequence can use a summable schedule. Select the schedule before drawing the
corresponding evaluation data. Include additional methods or metrics in the
error accounting if a single joint guarantee is claimed over them.

The adaptively trained policies need not have been fixed at the start of the
project. They must be frozen before each evaluation phase, and the fresh phase
data must satisfy the stated conditional sampling model. Old outcomes describe
old policies. Reusing an old surrogate to propose conditions is distinct from
using old evidence to certify a new policy.

## Candidate GP acquisition: evaluation-layer background

This candidate is retained from the evaluator-focused draft. It is not the
training-condition score, and its implementation is not a prerequisite for the
retraining study. Simple GP ranking remains the other candidate.

For unresolved \(z\), let \(\mathcal B\) be a finite, predeclared set of batch
sizes and let \(C^+_{z,b}\) be the actual certifier's hypothetical confidence set
after \(b\) additional binary observations. A candidate score is

\[
a_t(z,b)=\frac{
 \Pr_{\mathrm{GP}}\{C^+_{z,b}\ne\varnothing,\
                   \sup C^+_{z,b}\le\alpha\mid\mathcal F_t\}
}{b}.
\]

The probability integrates the GP's predictive distribution over the possible
batch outcomes; it is a planning probability, not a certificate. For a certifier
that intersects bounds after every new outcome, account for outcome ordering
as well as total failures in the batch, or explicitly use a different,
predeclared update schedule. A model can update predictions everywhere, but
only fresh direct observations change certificates at the queried condition.
The same fresh outcomes can update both the GP and the certifier: their
different inferential roles do not require separate surrogate-fitting and
certification samples within a frozen-policy evaluation phase. This does not
permit policy training on the same data and then certifying the updated policy
as if it had generated the old outcomes.

The score is a proposed adaptation inspired by acceptable-region acquisition
and Bernoulli look-ahead methods, not a verbatim implementation of either
paper. Small batches can yield zero predicted certificate gains everywhere;
use a declared fallback and a schedule that revisits neglected unresolved
conditions. Its benefit over simple allocation is an empirical question.
[Zanette et al.](https://arxiv.org/abs/1811.09977),
[Letham et al.](https://proceedings.mlr.press/v151/letham22a.html)

Before implementation, decide the batch set and safeguard, and whether this
look-ahead calculation is worth its cost relative to a simpler surrogate
ranking. Certificate gain is an evaluator diagnostic; it is not the main
research objective of improving the controller.

## Linear result: derivation and limits

For the symmetric matrix \(P_H\) in the brief,

\[
x^\top P_Hx-x'^\top P_Hx'
=(x-x')^\top P_Hx+x'^\top P_H(x-x').
\]

Cauchy–Schwarz gives
\[
|J_H(x)-J_H(x')|
\le\|P_H\|_2(\|x\|_2+\|x'\|_2)\|x-x'\|_2.
\]

If \(\|G^k\|_2\le c\rho^k\) with \(0\le\rho<1\), then

\[
\|P_H\|_2
\le\|Q\|_2c^2\sum_{k=0}^{H-1}\rho^{2k}
=\|Q\|_2c^2\frac{1-\rho^{2H}}{1-\rho^2}.
\]

This separates decay rate from the prefactor that can reflect transient
amplification. A two-dimensional analytical example suffices; it need not
become another environment in the repository.

The result concerns initial-state variation and continuous state cost. It does
not yet cover plant-parameter variation, random disturbances, or failure
probabilities. If \(P_H\) is positive definite and \(h>0\), the cost sublevel
set is an ellipsoid; with only positive semidefiniteness it need not be bounded.

## Controller-update preservation and a possible expansion argument

Use a distinct notation \(S_K^J\) for continuous-cost sets, to avoid conflating
them with the failure-probability sets \(S_k\). On \(\|x\|_2\le r\),

\[
|J_{H,K'}(x)-J_{H,K}(x)|
=|x^\top(P_H(K')-P_H(K))x|
\le r^2\|P_H(K')-P_H(K)\|_2=b.
\]

Consequently, points with \(J_{H,K}(x)\le h-b\) remain acceptable, and the
loss set lies in the band \(h-b<J_{H,K}(x)\le h\). This is a deterministic
baseline corollary, not a result about the effectiveness of boundary sampling.

One quantitative route is to bound the matrix difference in terms of the update.
If \(\|G_K\|_2,\|G_{K'}\|_2\le M\), with \(M\ge1\), the telescoping identity
gives, for \(j\ge1\),

\[
\|G_{K'}^j-G_K^j\|_2
\le jM^{j-1}\|B\|_2\|K'-K\|_2.
\]

It follows that

\[
b\le
2r^2\|Q\|_2\|B\|_2\|K'-K\|_2
\sum_{j=1}^{H-1}jM^{2j-1}.
\]

This elementary finite-horizon bound can be very conservative. Improving its
usefulness through closed-loop structure is a possible analysis direction.
Neither this bound nor a trust-region restriction establishes a favorable
training direction.

For a fixed measure on the state domain, suppose an actual update additionally
achieves \(J_{H,K'}(x)\le J_{H,K}(x)-a\), \(a>0\), on a set \(T\).
Then all points in
\(T\cap\{h<J_{H,K}(x)\le h+a\}\) become acceptable, and

\[
\begin{aligned}
\mu(S_{K'}^J)-\mu(S_K^J)
\ \ge\ &
\mu\!\left(T\cap\{h<J_{H,K}\le h+a\}\right)\\
&-\mu\!\left(\{h-b<J_{H,K}\le h\}\right).
\end{aligned}
\]

This is a conditional accounting argument, not an explanation of why a
learning algorithm satisfies its assumptions. The ambitious research step is
to connect boundary selection and an actual update rule to useful bounds on
\(a,b,T\). An assumption of improvement everywhere would bypass that question.

### An elementary regression example

A tied controller parameter can expose interference between training conditions.
Consider \(A=B=I_2\), \(K_\theta=\operatorname{diag}(\theta,-\theta)\),
\(Q=I_2\), horizon \(H=2\), and initial states \(e_1,e_2\). Their costs are

\[
J_\theta(e_1)=1+(1-\theta)^2,\qquad
J_\theta(e_2)=1+(1+\theta)^2.
\]

At \(\theta=0\), both meet the threshold \(h=2\). For \(0<\theta<1\),
the update improves \(e_1\) but makes \(e_2\) unacceptable. Uniform coverage of
these two conditions falls from one to one half. This is a deliberately
restricted linear example of harmful interference, not a claim about every
controller parameterization or about PPO. It shows why improving sampled
boundary conditions alone cannot prove monotonic expansion.

## Repository evidence and protocol corrections

- [Recovery findings](../../../docs/findings.md#cartpole-recovery-failure-boundary-study)
  distinguish survival and recovery and report results for frozen controllers.
  They are preliminary reported results, not new experiments run for this brief.
- The [pilot configuration](../../../configs/eval/cartpole_failure_boundary_pilot_v1.toml)
  has pilot seeds 0–4 and final seeds 0–49. The
  [collector](../../../src/active_eval_gym/sweeps.py) passes those episode seeds
  through to [reset](../../../src/active_eval_gym/rollout.py). Thus the final
  evaluation described as independent is not wholly held out.
- Existing Wilson intervals are pointwise summaries, not simultaneous
  confidence sequences. Keep the prior artifacts; use a separately versioned
  procedure and fresh data for the new claims.
- The [recovery perturbation](../../../src/active_eval_gym/envs/perturbations.py)
  replaces the reset angle exactly. It differs from the ordinary angle-offset
  perturbation, which adds an offset to a random angle.
- The [LQR policy](../../../src/active_eval_gym/policies/lqr.py) quantizes a
  continuous force command to two actions. Nominal linear-model stability
  does not establish the supporting theorem for its actual nonlinear rollouts.
- Grid cardinality, physical domain area, and mass under a deployment
  distribution are distinct. The current adaptively refined mesh cannot
  provide a physical-volume estimate by counting points.

For a Monte Carlo reference interval \([\ell_{\rm ref},u_{\rm ref}]\), an
acceptable certificate is demonstrably wrong only when
\(\ell_{\rm ref}>\alpha\); an unacceptable certificate is demonstrably wrong
when \(u_{\rm ref}\le\alpha\), subject to the reference's own coverage event.
Ambiguous cases must not be silently counted as correct. Known-probability
fixtures permit direct evaluation of the campaign-level error event.
Report uncertainty across repeated campaigns; observing no errors is not a
proof of the desired error bound.

## Research design decisions to keep explicit

- The three training curricula are uniform, reliability-boundary-guided, and
  intermediate-difficulty-guided. Match the background mixture and training
  objective to isolate the target-condition distribution.
- Use the same evidence-acquisition procedure for targeted curricula in the
  initial controlled comparison. Cover both threshold regions; an evaluator
  focused exclusively on one threshold could bias the comparison.
- Record uncertainty handling: membership in an estimated boundary band is
  not itself a certificate. Uncertainty, intermediate difficulty, and
  closeness to the acceptance threshold have different meanings.
- With PPO, sample conditions and collect fresh training trajectories.
  Condition replay is not unrestricted off-policy trajectory replay.
- Evaluate the same deployed action rule for every frozen checkpoint.
  Training-time stochastic action selection does not silently redefine the
  evaluation probability.
- Compare both equal-training-budget results and total evaluation-plus-training
  costs. Count environment transitions as well as episodes.
- Keep reference evaluation separate from training and curriculum selection.
  More evaluation can increase certified coverage without any policy improvement.
- Gains and losses between consecutive estimated regions require uncertainty
  accounting. In particular, an unresolved old condition cannot automatically
  be counted as a new success after training.

## Literature positioning and bibliography selection

The core bibliography now has seven papers organized around the revised
progression: Gotovos, Letham, and Howard for the evaluation foundation;
Florensa, Jiang, and Rutherford for curricula; and Berkenkamp for control
guarantees. This focused selection is not a claim of exhaustive literature
coverage or established novelty.

The closest research comparisons are:

- [Reverse Curriculum Generation (Florensa et al., 2017)](https://proceedings.mlr.press/v78/florensa17a.html):
  performance-adaptive initial-state curricula are already studied.
- [Prioritized Level Replay (Jiang et al., 2021)](https://proceedings.mlr.press/v139/jiang21b.html):
  environment configurations can be prioritized by estimated learning potential.
- [Sampling for Learnability (Rutherford et al., 2024)](https://arxiv.org/abs/2408.15099):
  mixed-success conditions are a direct curriculum target. Appendix I.3 also
  varies the preferred success probability; shifting a sampling threshold
  alone is not a novelty claim. The proposed comparison adapts its score to
  the repository's declared reset law and evaluation action mode.
- [Safe model-based RL (Berkenkamp et al., 2017)](https://arxiv.org/abs/1705.08551):
  policy improvement and safe-region expansion have control-theoretic precedents
  under dynamics-model and Lyapunov assumptions. The proposed finite-horizon
  recovery study does not inherit those guarantees.

The following sources remain relevant background without becoming required
algorithms or expanding the core reference list:

| Source | Role if the scope needs it |
| --- | --- |
| [Zanette et al., Robust Super-Level Set Estimation (ECML PKDD 2018; proceedings 2019)](https://doi.org/10.1007/978-3-030-10928-8_17) | Acquisition aimed at expanding the estimated acceptable region; relevant to the retained certificate-gain candidate |
| [Locatelli et al., Thresholding Bandit Problem (2016)](https://proceedings.mlr.press/v48/locatelli16.html) | Fixed-budget threshold classification |
| [Cho et al., Reward Maximization for Pure Exploration (2025)](https://proceedings.mlr.press/v258/cho25a.html) | Already combines adaptive sampling and anytime-valid tests; separating those layers is not new |
| [Raghavan and Johansson, Trajectory-Level Experimental Design (2026)](https://proceedings.mlr.press/v331/raghavan26b.html) | Region classification when movement and mixing constrain sampling |
| [Dietrich et al., Statistical Certification of Viable Initial Sets (2026)](https://arxiv.org/abs/2604.02939v2) | Importance-weighted certification of aggregate failure probability over a candidate set |
| [Mason et al., Nearly Optimal Algorithms for LSE (2022)](https://proceedings.mlr.press/v151/mason22a.html) | Allocation theory if pursued; linear dynamics do not automatically give a linear performance function |
| [Florensa et al., Automatic Goal Generation (2018)](https://proceedings.mlr.press/v80/florensa18a.html) | Goal curricula; less direct than initial-state curricula for the first experiment |

A stronger claim about a new training rule, retention mechanism, or theorem
needs a targeted follow-up search once that mechanism is specified. Barrier
certificates, distributionally robust LSE, and a broad continual-learning
survey are not required by the current design.
