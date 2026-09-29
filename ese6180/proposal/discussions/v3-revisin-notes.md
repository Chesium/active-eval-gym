# v3 revision: a central theoretical contribution

**Date:** 2026-09-28. The filename follows the requested `v3-revisin-notes.md`
spelling. This revision updates the working Markdown proposal and the LaTeX submission
after the review of the course resources and the office-hour guidance reported
by Shimin.

## Current documents and authority

- [brief.md](../brief.md): current research scope, stage commitments, and questions
  for Prof. Matni.
- [proposal-design-notes.md](proposal-design-notes.md): assumptions, derivations,
  proposed update mechanism, and retained empirical-evaluation infrastructure.
- [linear-control-and-retraining-explained.md](linear-control-and-retraining-explained.md):
  pedagogical explanation, retaining the cost examples and figure from v2.
- [proposal.tex](../latex/proposal.tex) and its PDF: synchronized to **v3**, with
  two body pages and references on page 3. The earlier Markdown-only revision
  was followed by this requested LaTeX conversion.
- [Email draft](email-prof-matni-v3.md): concise request for feedback and the
  recommended papers, with an Overleaf-link placeholder; not sent.
- [v2 notes](v2-revision-notes.md), original options, and prior discussions remain
  historical records. Their experiment-centered commitments are superseded here.

The bibliography gained two entries supporting v3. No experimental code,
checkpoints, raw trajectories, or evaluation metrics were changed.

## Why the scope changed

The reported office-hour guidance was to scope the project clearly and put
theory at the center, with experiments assisting it. The syllabus also calls
for a theory-focused project. The project description expressly permits
pedagogically accessible or simplified proofs of existing L4DC theory.

V2 had a valid but elementary quadratic-form inequality as its foundation,
PPO curriculum comparisons as the main study, and the substantive preservation
or expansion result as an ambitious outcome. The inequality did not itself
explain how controller updates or training-condition selection affect behavior.
V3 makes a certificate-learning specialization and a restricted-update analysis
the required contribution. Novel general-purpose RL theory is not promised.

## Scope decisions in this working revision

| Item | v2 emphasis | v3 decision |
| --- | --- | --- |
| Foundation | Cost-matrix difference plus sequential rollout certification | Discrete-time quadratic certificate learning with an explicit uniform generalization argument |
| Main theoretical claim | Broad question about conditions for regression/expansion | Verified Lyapunov decrease margin preserved through one specified gain update |
| Main intervention | PPO training under three curricula | Exact-gradient update on a small analytical family, restricted by a margin test |
| Performance analysis | Gains/losses as an experimental objective | Required cost-drift bound and an explicit counterexample to unconditional coverage improvement |
| Experiments | Main deliverable | Validate assumptions, bounds, and conservatism; CartPole is supporting evidence |
| Outer rounds | Minimum commitment unresolved | One update is sufficient for the minimum project |
| Expansion target | Cost and failure probability both unresolved | Cost-set expansion first if pursued; probability expansion deferred |
| Optional scope | General expansion theorem | One of cost expansion or a justified local nonlinear extension |
| Statistical evaluator | LSE and confidence sequences as course foundation | Retained supporting infrastructure; no GP implementation required |

These are concrete working choices for professor feedback, not a claim that he
has approved the paper or theorem selection. The broader evaluation-guided
domain-randomization motivation is retained, including its inspiration from
David Snyder. The notes do not attribute specific papers to his recommendation.

## Theoretical content now made explicit

### Stage 1: certificate learning and verification

The preferred source is Boffi et al., *Learning Stability Certificates from Data*.
Its main theoretical setting is continuous time and it already treats quadratic
classes. The proposed contribution is an accessible discrete-time specialization,
not novelty from choosing a quadratic function or a verbatim source theorem.

For a frozen gain and fixed independent trajectory distribution, fit
$V_P(x)=x^\top Px$ within $aI\preceq P\preceq bI$. The design notes define a
normalized decrease residual, a bounded margin loss, and a finite-cover uniform
bound. They handle zero states explicitly and use the number of trajectories,
not the number of time steps, as the independent sample count. The bound applies
to the fitted candidate and controls finite-horizon violation probability under
the same distribution; it does not claim the source paper's faster rate.

Known matrices provide a separate exact-arithmetic verification route through
$G_K(\theta)^\top PG_K(\theta)-P\preceq-\eta_\theta I$. A low empirical loss
does not establish this inequality. No global stability conclusion is inferred
from sampled satisfaction alone. Numerical verification needs a stated tolerance.

### Stage 2: a margin-preserving update and its limits

With fixed verified $P$, $K'=K+\Delta K$, and
$d_\theta=\|B(\theta)\Delta K\|_2$, the sufficient test is

$$
2\|G_K(\theta)^\top P\|_2d_\theta+\|P\|_2d_\theta^2<\eta_\theta.
$$

The notes derive the inequality directly and specify a finite-grid training
objective, frozen target distribution, exact gradient, backtracking, and an
Armijo descent test. The preservation statement assumes an old certificate
verified on every predeclared retention plant. A common certificate is an
explicit restriction; individual plant stability does not imply its existence.

The finite-horizon cost calculation remains and now accounts for the input
penalty $K^\top RK$. Stability preservation, interior cost preservation, set
inclusion, and net coverage growth are distinct. A two-plant scalar example
passes the stability-margin test and lowers targeted training cost while losing
half the equal-weight cost coverage. It is a counterexample to arbitrary
boundary targeting, not a claim against every symmetric sampling rule.

### Stage 3: one extension, not several additional projects

Prefer an expansion result for the cost set under a specified update. It must
derive useful gains from the mechanism, rather than assume improvement
everywhere. Alternatively, pursue a local nonlinear extension with a remainder
bound and an invariant neighborhood. Repeated rounds and recovery-probability
theorems are deferred.

## Course connections and transfer limits

- Chapter 2 motivates how policy changes alter closed-loop trajectories.
- Chapter 3, Theorem 3.10 and Example 3.12, supplies the discrete-time Lyapunov
  reasoning and quadratic matrix verification.
- Chapter 5 supplies uniform convergence and independent trajectory sampling.
  Its predictor analysis does not directly cover a controller changing the
  trajectories; the update argument is supplied separately.
- Chapter 4 is relevant to identification or imitation learning, neither of
  which is silently claimed by optimizing a gain with known matrices.

The repository's quantized LQR maps to two fixed forces and applies a nonzero
force at zero state. Its nonlinear closed loop is not the continuous-action
linear model. The current CartPole recovery criterion, local stability,
invariance, and barrier-based safety remain distinct. Large-angle recovery does
not follow from a local linearization. For unconstrained stable linear systems,
the true region of attraction is already global; certificate ellipsoid growth
alone would not establish growth of that true region.

## Literature added

Downloaded the publisher PDFs for:

1. [Boffi et al., Learning Stability Certificates from Data](../literature/boffi21a-learning-stability-certificates.pdf),
   CoRL 2020 / PMLR publication 2021: preferred foundation.
2. [Zhang et al., Adversarially Robust Stability Certificates can be Sample-Efficient](../literature/zhang22a-adversarially-robust-stability-certificates.pdf),
   L4DC 2022: optional robustness context.

The [source record](../literature/certificate-learning-sources.md) includes
publisher URLs, sizes, page counts, and SHA-256 checksums. The corresponding
entries were added to [references.bib](../latex/references.bib). The brief's
old link to a nonexistent `proposal/references.bib` now points to this actual
bibliography. Existing PDFs and older background references were preserved.

## Questions for the professor

1. Is the discrete-time quadratic certificate-learning specialization, followed
   by one restricted-update analysis, an appropriate theory-centered scope?
2. Is Boffi et al. the best starting result, or is another Lyapunov/barrier paper
   more suitable for the intended course connection?
3. Is a finite family with a common quadratic certificate an acceptable
   restriction, with expansion optional and CartPole as supporting evidence?

## Remaining work

- Obtain feedback on the anchor result and depth, then choose numerical family,
  normalization, margins, sample law, horizon, retention set, and step parameters.
- Work through the source result and compare its assumptions/rates with the
  simpler proof, rather than treating a citation as a proof of applicability.
- Implement and validate the selected analytical experiment with a versioned
  protocol; report failed certificate fits and rejected updates.
- Incorporate professor feedback into the current two-page LaTeX and working notes.
- If using CartPole, choose a narrow question supported by the linear analysis,
  specify fresh evaluation data, and avoid claims that transfer automatically.

This revision supplies mathematical derivations and a concrete study design;
it does not represent a completed experimental study or an email sent to the
professor.

## Revision checks

Checked the revised documents' local links and display-math delimiters, and
verified both downloaded PDFs' titles, page counts, and checksums. A one-off
numerical check reproduced the scalar counterexample and checked the certificate
drift, cost drift, and normalized-residual Lipschitz inequalities on 200 generated
matrix cases. Those checks catch arithmetic/transcription errors; they are not
a substitute for the stated proofs or a project experiment.

The subsequent LaTeX conversion was built with pdflatex and BibTeX. The output
has two body pages and one reference page, with no unresolved citations or
LaTeX layout warnings. Both body pages were visually inspected. A short AI-use
statement follows the syllabus disclosure guidance. The Overleaf upload archive
contains only the source, bibliography, and local L4DC class.
