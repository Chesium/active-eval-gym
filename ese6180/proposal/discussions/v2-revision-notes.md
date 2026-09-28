# September 22 proposal revision

The current submission draft is [proposal.tex](../latex/proposal.tex), based on
the [nine review comments](v1-comments/comments-2026-09-22.md). The expanded
[brief](../brief.md) and earlier design notes remain background documents; the
linear-family extension below supersedes their fixed-plant foundation for the
submission draft. The comment export is preserved unchanged.

## Scope selected with Shimin

Use a small **finite-horizon quadratic-cost example and direct proof** for the
linear foundation. The two suggested domain-randomized LQR papers provide
context; reproducing an infinite-horizon convergence result is not required.
The progression remains **Course foundation → Main research → Ambitious
theoretical outcome**. Boundary-guided retraining experiments and theoretical
investigation are main research, not an optional stretch.

## Review coverage

| Comment | Revision in the LaTeX |
| --- | --- |
| C001 | Title and abstract begin with controller learning across conditions, then introduce evaluation-guided training and the three stages. |
| C002 | Motivation and Problem Formulation are separate sections. |
| C003 | CartPole parameters and recovery definition appear in the Test problems paragraph. The preliminary figure is retained only as supporting material. |
| C004 | Notation follows the loop explanation; outer round, rollout time, plant parameters, initial-state distribution, and episode randomness have distinct roles. |
| C005 | Control, Evaluation, and Improvement are introduced in a contextual three-item list. |
| C006 | RQ1 concerns empirical coverage improvement; RQ2 concerns regression bounds and conditions for expansion. |
| C007 | Related work follows Control and synthesis, Evaluation, and Improvement. |
| C008 | The foundation now uses a parameterized linear family. Both suggested Fujinami papers are cited, with history-dependent controllers as context. |
| C009 | Three connected goal paragraphs explain the progression and purpose; detailed protocol choices stay in notes. |

## Linear foundation: intended interpretation

For a low-dimensional deterministic family, use
\(x_{t+1}=A(\theta)x_t+B(\theta)u_t\), \(u_t=-Kx_t\).
Write \(G_K(\theta)=A(\theta)-B(\theta)K\). For fixed
\(Q\succeq0\), \(R\succeq0\), define

\[
J_{H,K}(\theta,x)=\sum_{t=0}^{H-1}
 (x_t^\top Qx_t+u_t^\top Ru_t)
=x^\top P_H(K,\theta)x,
\]
\[
P_H(K,\theta)=\sum_{t=0}^{H-1}
 (G_K(\theta)^t)^\top(Q+K^\top RK)G_K(\theta)^t.
\]

A small numerical example can learn a shared static gain by minimizing the
expected finite-horizon cost under a fixed distribution of plant parameters
and initial states. This is controller learning for known analytical dynamics;
it does not claim system identification or a global optimization guarantee.
Exact matrices, distribution, optimizer, and horizon remain experiment-design
choices. This analytical example need not add a new Gym environment.

For any fixed plant parameter and \(\|x\|_2\le r\),

\[
|J_{H,K'}(\theta,x)-J_{H,K}(\theta,x)|
\le r^2\|P_H(K',\theta)-P_H(K,\theta)\|_2=:b(\theta).
\]

Thus \(J_{H,K}(\theta,x)\le h-b(\theta)\) implies
\(J_{H,K'}(\theta,x)\le h\). Any loss of acceptability lies in
\(h-b(\theta)<J_{H,K}(\theta,x)\le h\). A uniform statement uses a finite
upper bound on \(\sup_\theta b(\theta)\), for example over a finite grid.
This finite-horizon bound requires no asymptotic stability assumption, though
large amplification can make it uninformative. It does not prove that a
particular sampler or optimizer improves coverage.

The main research investigates update-size bounds and improvement/regression
tradeoffs, including counterexamples. The ambitious outcome additionally needs
conditions ensuring preservation or enough new acceptable mass to outweigh
losses. Cost sets and failure-probability sets are distinct. A probability result
would require a specified coupling of randomness and control of probability
mass near the failure threshold, or another explicit sufficient assumption.

## Evaluation interpretation and open protocol choices

- Keep failure tolerance \(\alpha\) and campaign error budget \(\delta\)
  symbolic until numerical protocol design.
- Fix the reference measure \(\mu\) and evaluation grid before comparisons;
  changing training distribution \(q_k\) does not change the scoring measure.
- For \(z=(\theta,s)\), \(s\) can fix an initial angle while specifying the
  distribution of other initial-state components. Do not count the same
  randomness twice. Certificates concern this conditional episode law.
- Use fresh outcomes after each policy is frozen. Allocate statistical error
  across conditions, methods, and rounds; adaptive proposals do not make GP
  posterior probabilities into direct rollout certificates.
- The evaluator objective remains certified acceptable coverage at fixed budget.
  The retraining objective is improved controller performance under common
  reference evaluation. Differences between certified sets alone do not identify
  the true gains and losses of acceptable regions. Report unresolved conditions
  and uncertainty when estimating those changes.
- Compare uniform, reliability-boundary, and intermediate-difficulty targets
  under the same background mixture and training budget. Their scores serve
  training selection, distinct from the evaluator's acquisition function.
- Exact acquisition rule, mixture weight, reward alignment with recovery,
  budgets, seeds, and family parameters remain to be fixed before experiments.
  Existing GP acquisition candidates remain candidates; neither is newly selected.
- Whether multiple outer rounds are required remains open; a single update is
  still a candidate minimum study. Continuous-cost and failure-probability level
  sets remain alternative targets for the ambitious theorem without a new ordering.

## References and figure

The local [bibliography](../latex/references.bib) adds
[Fujinami et al. (2025)](https://arxiv.org/abs/2503.24371) and
[Fujinami et al. (2026)](https://arxiv.org/abs/2609.16300). Their convergence
analysis concerns domain-randomized average LQR cost under heterogeneity
assumptions; it is not an expansion guarantee for adaptively selected level sets.
The 2026 curriculum expands controller memory rather than adapting the
environment sampling distribution. History-dependent controllers are not an
implementation requirement for this project.

The preliminary panels and their generator remain under `latex/figures/`.
If reused, label zero/one entries as empirical recovery rates and the black
contour as 0.5 recovery, not as the reliability boundary for arbitrary
\(\alpha\). Unsampled cells and mesh-rank coordinates also need explanation.
Omitting the figure makes space for a standalone problem formulation within
the two-page body; references occupy a separate page.
