# Supporting notes for the active-certification proposal

Prepared 20 September 2026 (America/New_York). These notes support
[brief.md](../brief.md); they are not additional text for the two-page proposal.

## Decisions from the discussion

Confirmed by Shimin Chen:

- Main objective: certified acceptable coverage at a fixed rollout budget.
- Primary outcome: recovery in the repository's modified CartPole task.
- Supporting theory: linear continuous-cost sensitivity as a separate result.
- Keep failure tolerance and certification error budget symbolic.
- Keep look-ahead certificate gain and simple GP ranking as acquisition candidates.
- Retain boundary-guided retraining as an explicitly optional stretch paragraph.

No numerical protocol has been selected. The detailed acquisition rule,
second-controller repetition, and implementation of the optional retraining
stretch remain open. Fix the comparative protocol before running experiments.

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

## Candidate GP acquisition: a design for review

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
different inferential roles do not require separate training and certification
samples within a campaign.

The score is a proposed adaptation inspired by acceptable-region acquisition
and Bernoulli look-ahead methods, not a verbatim implementation of either
paper. Small batches can yield zero predicted certificate gains everywhere;
use a declared fallback and a schedule that revisits neglected unresolved
conditions. Its benefit over simple allocation is an empirical question.
[Zanette et al.](https://arxiv.org/abs/1811.09977),
[Letham et al.](https://proceedings.mlr.press/v151/letham22a.html)

Before implementation, decide the batch set and safeguard, and whether this
look-ahead calculation is worth its cost relative to a simpler surrogate
ranking. That decision does not change the agreed statistical objective.

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

## Literature search scope and selection

The additional search covered GP level-set estimation, acceptable-region
acquisition, Bernoulli observations, thresholding/good-arm identification,
anytime-valid inference, and simulation-based control certification. Primary
papers and publisher/author pages support the selected bibliography. The
search checks the proposal's positioning; it does not establish exhaustive
coverage of all literature or prove novelty.

The most consequential additions to the earlier notes are:

- **Zanette et al. (ECML PKDD 2018; proceedings 2019):** acceptable-region expansion is already an
  acquisition objective; boundary sampling is not the only LSE perspective.
- **Cho et al. (2025):** pairing allocation with anytime-valid tests, including
  accumulated good-arm labels, is already directly studied. Inspect Algorithm 1
  for sampling and Algorithm 2/Theorem 3 for the complete procedure and error
  control. The common-certifier adaptation in the brief is not their full method.
- **Dietrich et al. (2026):** GP-assisted failure discovery with separate
  statistical control certification is also relevant prior work. The target
  probability and use of importance weighting distinguish it from this project.

The eight selected papers have distinct roles and are all cited in the brief.
For Zanette et al., the bibliography uses the publisher's 2019 publication
year and separately identifies the 2018 conference.
[Publisher record](https://doi.org/10.1007/978-3-030-10928-8_17)
The following nearby works were checked but need not expand its bibliography:

| Work | Reason to retain only as background |
| --- | --- |
| [Kano et al., Good arm identification via bandit feedback (2019)](https://link.springer.com/article/10.1007/s10994-019-05784-4) | Foundational GAI formulation; Cho supplies a closer anytime-valid method for the chosen comparison. Add Kano if its algorithm is implemented. |
| [Jourdan, Delahaye-Duriez, and Réda, An Anytime Algorithm for Good Arm Identification (2026)](https://www.jmlr.org/papers/v27/24-0680.html) | Relevant anytime sampling, but its central target is finding one good arm; accumulated acceptable coverage is better aligned with the selected Cho reference. |
| [Mason et al., Nearly Optimal Algorithms for Level Set Estimation (2022)](https://proceedings.mlr.press/v151/mason22a.html) | Relevant for a later allocation/sample-complexity theorem. Linear dynamics do not automatically make the performance function linear in the paper's features. |
| [Wagenmaker and Jamieson, Active Learning for Identification of Linear Dynamical Systems (2020)](https://arxiv.org/abs/2002.00495) | Course-listed context, but system identification is not the chosen target; Raghavan provides closer region-classification context. |
| [Lee et al., Active Learning for Control-Oriented Identification of Nonlinear Systems (2024)](https://arxiv.org/abs/2404.09030) | Concerns model identification and subsequent control synthesis, beyond the fixed-policy scope. |

TRUVAR, multiscale LSE, distributionally robust LSE, adaptive stress testing,
barrier certificates, and policy-improvement literature are not required by the
current method. Revisit the relevant subset only if cost heterogeneity,
continuous-domain guarantees, distribution shift, or retraining enters the
actual proposal. An expanded reading list is not a substitute for specifying
the core experiment.
