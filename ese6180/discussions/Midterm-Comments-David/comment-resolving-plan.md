# Midterm comment tracker

Tracks David's Overleaf review of
[`midterm_report.tex`](../../Evaluation_Guided_Domain_Randomization/midterm_report.tex).
The full comment text is in [`comments-2026-11-05.md`](comments-2026-11-05.md).

- **Codes:** `M01`-`M14` are this tracker's items. `C001`-`C014` are the Overleaf
  comment IDs. All 14 comments are covered below.
- **Line numbers:** item locations follow the current `midterm_report.tex`. The M02 and M03 edit
  tables use the line numbers from when each edit was made.
- **Owner:** Shimin, Agent, Discuss (Shimin and agent together), or Unassigned.
- **Status:** todo, discussing, drafted, or done.
- **Original notes:** these are kept verbatim from the first version of this plan.
  They are still work in progress and have not been answered or corrected.

## Index

| Code | Item | Comments | Lines | David's priority | Owner | Status |
| --- | --- | --- | --- | --- | --- | --- |
| [M01](#m01---abstract) | Abstract | C001, C002 | 22-28 | High (C001) | Shimin | √ |
| [M02](#m02---the-term-guarded) | The term "guarded" | C003, C011 | 63, 193 | | Agent | √ |
| [M03](#m03---r-positive-definite-or-semidefinite) | $R\succ0$ or $R\succeq0$ | C004 | 79 | | Discuss | √ |
| [M04](#m04---performance-metric-intuition) | Performance metric intuition | C005 | 78-85 | High | Shimin | √ |
| [M05](#m05---certificate-loss-intuition) | Certificate loss intuition | C006 | 88-114 | | Shimin, after discussion | √ |
| [M06](#m06---uniform-bound-idea) | Uniform bound idea | C007 | 111 | | Agent | √ |
| [M07](#m07---what-verifies-stability) | What verifies stability | C008 | 152 | | Discuss | √ |
| [M08](#m08---targeting-idea) | Targeting idea | C009 | 178 | | Agent | √ |
| [M09](#m09---one-guarded-update) | One guarded update | C010 | 190 | | Unassigned | √ |
| [M10](#m10---linear-study-figure) | Linear study figure | C012 | 210-220 | | Discuss | largely √ |
| [M11](#m11---ppo-experiment-contrast) | PPO experiment contrast | C013 | 246-262 | | Unassigned | ignored |
| [M12](#m12---references) | References | C014 | 292, plus citations | | Agent | √ |
| [M13](#m13---appendix) | Appendix (not in original plan) | C014 | new | | Unassigned | TODO |
| [M14](#m14---proposal-introduction-and-motivation) | Proposal introduction and motivation (not in original plan) | C014 | new | Optional | Unassigned | TODO |

## Constraints and dependencies

- **Page limit:** the course limits the midterm to four pages, excluding references, and
  `ese6180/build_report.py` fails above that. The main text ends on page 4 with about two lines
  free, so added text needs matching cuts. This affects M04-M10 and M14.
- **Due date:** the midterm report is due 2026-11-25. The course lists this date as
  tentative.
- **M01 depends on M06-M09:** David suggests drafting much of the abstract from the
  comments on the Section 2 paragraph titles (C007-C010).
- **M02 is applied:** M01 and M09 should keep its terms: "safe", "safety test" and "Armijo
  condition".

## Items

### M01 - Abstract

- **Comments:** C001 (high priority), C002
- **Where:** lines 22-28
- **Problem:** David asks for the abstract to be rewritten in plain language, not LLM
  output. David also advises against calling the proof "elementary", because that
  devalues it.
- **David's suggestion:** string together the necessary sentences from the comments on
  the bold paragraph titles of Section 2 to get a good draft of much of the abstract.
- **Original note:**

> abstract : todo, I will supply text

#### Applied abstract and title

The selected title, **Evaluation-Guided Domain Randomization: A Feedback Loop for
Controller and Certificate Learning**, is applied in the source and PDF metadata.
The selected 75-word balanced abstract from the [three shortened options](abstract-options.md)
is applied in the source. It introduces the feedback loop, states the certificate and
stability results, summarizes the preliminary targeting evidence, and identifies the
remaining coverage question. The proof is described through its result.

**Checked:** rebuilt the repository PDF with the selected abstract: four main-text
pages, references starting on page 5, no overfull boxes or undefined citations/references.

### M02 - The term "guarded"

- **Comments:** C003, C011
- **Where (before the fix):** comments at lines 63 and 193. "guarded" also appears at lines 23 and 46. The
  noun "guard" appears at lines 174, 182, 188, 191, 233, 239, 255, 274 and 285, and in the
  label `eq:mid_guard`. The research note sections and the README also use the word.
- **Problem:** David calls "guarded" an odd term and asks for the term Armijo uses,
  because it is technical, recognized and searchable. David gives "safe" as an
  alternative.
- **Original note:**

> term "guarded" for Armijo update: todo (agent)

#### Literature check

- **Armijo (1966):** Corollary 2 calls the method "The Modified Steepest Descent Algorithm".
  The paper never says "Armijo rule", "line search" or "backtracking". Those are later names,
  and they are the searchable ones.
- **Local sources:** "guard" and "guarded" appear in none of the papers in
  `proposal/literature`, the course chapters, Armijo or Fujinami.
- **Berkenkamp et al. (2017):** the abstract defines "safety, defined in terms of stability
  guarantees". This supports David's alternative, "safe".
- **Mohammadi et al. (arXiv:1912.11899):** an LQR paper that uses "a stepsize selected using a
  backtracking procedure that guarantees stability of the closed-loop system". That is close
  to this update's idea.
- **Boyd and Vandenberghe, Section 9.2:** "we first multiply t by β until x + t∆x ∈ dom f; then
  we start to check" the sufficient-decrease inequality. This update has a similar order: limit
  the step for stability, then apply Armijo.
- **Fujinami et al. (2025):** uses a fixed stepsize, so it offers no line-search term. It does
  say "jointly stabilizing", which may help M07.

#### Diagnosis

"Guarded" covered more than the Armijo step. In the report, "guard" named the certificate
stability test, and line 198 named the Armijo test separately. Armijo only checks descent of
$F_q$, so "Armijo update" alone would drop the stability half.

#### Applied fix

Use one name for each half of the update:

- **Stability half:** "safe", with "safety test" as the noun.
- **Descent half:** "Armijo condition" and "backtrack", replacing "Armijo-style test".
- **Whole update:** "safe Armijo update" in the paragraph title and "safe update" elsewhere.

"Safe" could be read as "keeps coverage", and it applies only to retained plants. The title
"Safe means stable" and the sentence "Safe descent need not preserve coverage" (added in M03)
address the first risk. The "on the retention set" wording addresses the second.

| Line | Before | After |
| --- | --- | --- |
| 23 | `a guarded` | `a safe` (M01 will rewrite this) |
| 46 | `\text{guarded update}` | `\text{safe update}` |
| 63 | `the guarded update` | `the safe update` |
| 174 | `\textbf{Certificate guard.}` | `\textbf{Safe means stable \citep{berkenkamp2017safe}.}` |
| 176 | `margin-consumption bound` | `margin-use bound` |
| 177 | `\label{eq:mid_guard}` | `\label{eq:mid_safety}` |
| 182 | `The guard requires` | `The safety test requires` |
| 188 | `The guard thus keeps` | `The test thus keeps` |
| 191 | `passes the guard.` | `passes the test.` |
| 193 | `\textbf{One guarded update.}` | `\textbf{One safe Armijo update.}` |
| 196-197 | `Otherwise let $\bar s$ be the smaller of $s_0$ and the step capacity, and return` | `Otherwise backtrack from $\bar s$, the smaller of $s_0$ and the step capacity: return` |
| 198 | `that passes the Armijo-style test` | `that meets the Armijo condition` |
| 199-200 | `If none passes, return $K$.` | `If none does, return $K$.` |
| 201 | `would instead create` | `would create` |
| 233 | `changes the guard, not $D$, so refitting acts only` | `changes the safety test, not $D$, so it acts only` |
| 239 | `The guard limits $\Delta K$, not $s$, so when the guard sets the step` | `The safety test limits $\Delta K$, not $s$, so when it sets the step` |
| 255 | `With no certificate or guard,` | `With no certificate,` |
| 274 | `sign-blind guard:` | `sign-blind safety test:` |
| 285 | `make the guard sign-aware` | `make that test sign-aware` |

Lines 176, 199-200 and 201 are not renames. They offset the longer wording so the page count
holds.

**Checked:** with all 19 edits applied, `midterm_report.tex` builds at four pages with no
overfull boxes or undefined citations. Pages 3 and 4 end on the same lines as before.

**Follow-ups:**

- Copy these edits to Overleaf.
- Update the same term in `Report_Sections/02_problem_formulation.tex` line 504,
  `Report_Sections/05_theoretical_results.tex` lines 518 and 647, and `README.md` lines 7 and
  79. This matters most if M13 reuses that text.
- Code and `docs/` also say "guard". They are not part of the report and can stay.

### M03 - R positive definite or semidefinite

- **Comment:** C004
- **Where:** line 79. Related uses of $R$:
  - `Report_Sections/02_problem_formulation.tex`, lines 309, 342 and 630;
  - `Report_Sections/05_theoretical_results.tex`, line 640.
- **Problem:** David says $R\succeq0$ is probably imprecise, because $R\succ0$ is almost
  always required for numerical reasons.
- **Original note:**

> R succ 0 or R succeq 0: todo (agent discuss)

#### Findings

- **Proofs:** none needs $R\succ0$. There is no $R^{-1}$, $\lambda_{\min}(R)$ or Riccati
  equation in the midterm or research note, and the bounds use only $\|R\|_2$. So $R\succeq0$ is
  not false, and every result also holds under $R\succ0$.
- **Experiments:** the linear studies already use $Q=I$ and $R=0.1I$. Only the scalar examples
  use $R=0$.
- **Code:** nothing inverts $R$. `LinearFamily` only checks that $R$ is positive semidefinite.
- **Literature:**
  - Fujinami et al. (2025) assumes $Q\succeq I$ and $R=I$, and generalizes through
    $B(\theta)\leftarrow B(\theta)R^{-1/2}$, which requires an invertible $R$.
  - Hazan and Singh, *Introduction to Online Control*, assumes $Q\succ0$ and $R\succ0$.
  - Mohammadi et al. states that $Q$ and $R$ are positive definite.
  - The standard reason is the $(R+B^\top SB)^{-1}$ term in the optimal LQR gain.
- **Degenerate cases at $R=0$:** the cost has no terminal term, so with $H=1$ it does not depend
  on $K$ and the gradient is zero. A plant with an unused actuator gives Hessian eigenvalues
  $(0, 0, 1.54, 1.64)$ at $R=0$, against $(0.11, 0.13, 1.63, 1.75)$ at $R=0.1I$.
- **Main problem:** the worked counterexample did not survive $R>0$ as written. At $K=0.5$ both
  costs rise above $h=1.25$, so even $R=0.01$ gives coverage $0\to1/2$, which is the opposite of
  the claim. Moving $h$ restores the drop, but the numbers change and uniform weights no longer
  give $D=0$.

#### Applied fix

The midterm now states $R\succ0$. Its worked example became one sentence in the "One safe Armijo
update" paragraph, so the midterm no longer shows $R=0$. The "band" wording was tightened in the
same change, because "band" named two different things.

| Line | Before | After |
| --- | --- | --- |
| 79 | `$R\succeq0$` | `$R\succ0$` |
| 167 | `to the band` | `to the \emph{boundary band}` |
| after 201 | none | `\emph{Safe descent need not preserve coverage:} in a two-plant scalar example with hand-set $q$, an accepted step lowers $F_q$ and keeps every verified margin positive, yet coverage falls from $1$ to $1/2$. Boundary targeting gives $D=0$ there, so this does not refute it.` |
| 246-252 | paragraph "Stability and descent do not imply coverage" | removed |
| 276-277 | `losses lie within ... so that band's $\mu$-mass bounds lost coverage` | `only conditions within ... can be lost, so their $\mu$-mass bounds lost coverage` |

**Example behind the sentence:** $A(\theta)=\theta-1/2$, $B=1$, $\theta\in\{0,1\}$, $x_0=1$, $Q=1$,
$R=0.1$, $H=2$, $h=1.25$, $K=0$, $P=1$, uniform evaluation weights, $q=(1/4,3/4)$, $s_0=0.4$,
$c_A=\rho=0.5$ and $\kappa=0.25$. The old gain is $K=0$, so $u=0$ and $R$ cannot change the
starting costs or gradient.

**Checked with `backtrack` in `linear_updates.py`, and exactly with fractions:** $\Delta K=0.2$
passes the first trial, $F_q$ falls from $1.25$ to $1.19476$, coverage falls from $1$ to $1/2$, the
smallest margin stays $0.51$, and $c_\theta=0.24\le0.375$. Uniform weights give $D=0$ at any $R$.
The first trial passes up to $R\approx0.21$; above that the search halves the step, and coverage
still falls at the values tested ($R=0.22$ and $R=1$). The full record and a runnable check are
in [the example log](../2026-11-04-safe-descent-coverage-example.md).

**Report check:** `midterm_report.tex` builds at four pages with no overfull boxes or undefined
citations, and about two lines are free on page 4.

**Follow-ups:**

- Copy these edits to Overleaf.
- `Report_Sections/02_problem_formulation.tex` lines 309, 342 and 630, including that example's
  values at lines 636-643, still use $R\succeq0$ and $R=0$. See M13.
- `Report_Sections/05_theoretical_results.tex` line 640 uses $R=0$. Using $A=0.6$, $K=0$, $R=0.1$
  keeps the coverage result (boundary $0.75\to1$, background unchanged). The quoted bound of $0.25$
  has not been recomputed.
- `linear_study.py` (`_scalar_studies`) and `tests/test_linear_updates.py` pin $R=0$. Add the
  $R=0.1$ cases beside them rather than overwriting the recorded fixtures.

### M04 - Performance metric intuition

- **Comment:** C005 (high priority)
- **Where:** "Performance" paragraph, lines 78-85. The comment is at line 81.
- **Problem:** David says the definition is technical and not intuitively clear, and asks
  what the idea is. David's answer: to target the collection of new data to cases where
  the policy has some ability, but does not have mastery.
- **Original note:**

> Not intuitive Performance metric definition: (todo, I will supply agent text)
> - J_H,K(z): H-step closed-loop cost(finite-horizon LQ cost) of condition z=<theta, x> \in mu - trajectory-wise
> - S^J_K: acceptance set - the region where the cost is less than a threshold h
> - C_J(K): the coverage of a certain policy gain K under mu

### M05 - Certificate loss intuition

- **Comment:** C006
- **Where:** "Certificate loss and risk" paragraph, lines 88-114. The comment is at line 91.
- **Problem:** David says the definition is too technical and lacks insight for the reader.
  David's understanding of the idea: we trade off some stability margin on easier cases in
  order to provably stabilize harder cases.
- **Original note:**

> Not intuitive Certificate Loss definition: (todo, I will supply text, pending agent discussion)
> - how to intuitively understand describe the role of eta and gamma

### M06 - Uniform bound idea

- **Comment:** C007
- **Where:** "What generalizes: a uniform bound" paragraph, starting at line 111. The
  diagram at lines 117-128 is the "nice visual equation" David mentions.
- **Problem:** David says the idea needs to be clear, and likely needs only 2-3 sentences
  before the math.
- **David's suggestion:** most of the first block of text can become "we follow the method
  of Boffi et al., which ultimately bounds sample error via bounding successive
  differences", followed by the visual equation. Pose this as the foundation of the work:
  following the ideas of Boffi.
- **Original note:**

> Uniform bound idea: todo (agent propose)

#### Applied fix

Two opening sentences attribute the certificate-learning foundation to Boffi et al.
and explain the sample-to-population gap through comparison with a nearby candidate.
The diagram follows immediately; the net size, dimension, Lipschitz constant, and
concentration argument now follow it. The bound, fitted-certificate implication,
violation-risk inequality, and fresh-data requirement remain. The temporary symbol
$M_K$ is expanded, and the closing comparison with Boffi et al.'s rates is removed.

The opening motivation is also shortened by one rendered line so the performance
equation stays on page 1 with each candidate abstract. No formatting commands changed.

**Checked:** the current source and all three abstract alternatives build at four
main-text pages with references starting on page 5, no overfull boxes, and no
undefined citations or references. See [the options and PDF previews](abstract-options.md).

### M07 - What verifies stability

- **Comment:** C008
- **Where:** "What verifies stability: a deterministic test" paragraph, starting at line 152.
  The certificate guard definition, now titled "Safe means stable", follows it at line 161.
- **Problem:** David identifies this as the beginning of the novel continuation: building
  this into a feedback loop using the domain randomization ideas of Fujinami et al.
- **Original note:**

> What verifies stability: a deterministic test: (agent discuss)
> - does the eta_theta(P)=lambda_min(P-GPG) directly leads to the Certificate Guard Definition? if so we should show that continuation

#### Answer

Yes for the margin half of the safety test. Let $\eta_\theta^+(P)$ be the same margin at
$K^+=K+\Delta K$. Expanding $(G+E)^\top P(G+E)$ gives
$\eta_\theta^+(P)\ge\eta_\theta(P)-c_\theta(\Delta K)$, so requiring
$c_\theta(\Delta K)\le(1-\rho)\eta_\theta(P)$ gives $\eta_\theta^+(P)\ge\rho\eta_\theta(P)>0$. $P$
is then still verified at $K^+$, which meets the next round's starting condition. This is the
feedback loop David points to in C008. Without a refit, the guaranteed margin shrinks like
$\rho^T$ over $T$ rounds, which is a natural place to bring in the Boffi refit step.

Three things do not follow from $\eta_\theta$:

- **$\kappa$:** it serves the cost-drift bound, not stability.
- **$\rho$:** this is a design choice.
- **The premise $\eta_\theta(P)>0$:** if a margin is negative, even $s=0$ fails the test and step
  capacity is undefined. The code returns `verification_failed`; the midterm had not defined
  "verified".

**Checked:** in the M03 example, $0.75-0.24=0.51$ is an equality on one plant, and the other plant
rises to $0.91$. On random stable two-plant families the inequality held in every sample. Along
$D$, the exact stable step was a median of about 7 times the $c_\theta$-based capacity (10th-90th
percentile 2.7-33). Those families are synthetic, not the study's, $\kappa$ was ignored, and the
script was a throwaway check that is not kept in the repo.

#### Applied fix

The paragraph order is now verifies, Safe, Targeting, One safe Armijo update, because Targeting
split the derivation and only the step-capacity sentence needs $D$.

| Location | Before | After |
| --- | --- | --- |
| "Safe means stable" | after "Targeting", starting at line 174 | before "Targeting", starting at line 163 |
| Safe, opening | `For new gain $K^+=K+\Delta K$, hold verified $P$ fixed.` | defines "verified", then derives $\eta_\theta^+\ge\eta_\theta-c_\theta$ |
| Safe, `eq:mid_safety` | display of $d_\theta$ and $c_\theta$ | inline; the label is removed and nothing referenced it |
| Safe, proof chain | `(G+E)^\top P(G+E)-P\preceq\ldots` | replaced by the inequality above and $\eta_\theta^+\ge\rho\eta_\theta>0$ |
| Safe, $\kappa$ | listed with the margin condition | listed second, with `which serves cost bounds, not stability` |
| Safe, closing | none | `So $P$ stays verified at $K^+$, and the next round can start from it.` |
| Safe | `for any target map` | `however $\Delta K$ was chosen`, because the target is not defined yet at that point |
| Targeting, end | step capacity defined in Safe | `$P$ limits the step, not $D$: the \emph{step capacity} of $P$ along $D$ is ...` |

The inequality has to stay inline: a display version overflows to five pages, even with
"Keeping $\eta$ ..." and "$P$ limits the step, not $D$" cut. It is a page-break effect, not a
length one. The inline version builds at four pages with no overfull boxes, and page 4 ends at the
same height as before this edit.

#### Rephrase

Both paragraphs now open with why they exist, and the second reads as a derivation from the margin:
start from $\eta_\theta$, bound what the step can spend, set the safety test, conclude that $P$
stays verified. They stay separate because David's C008 and the M02 title sit on them, and M01 builds
the abstract from these paragraphs' opening lines.

| Location | Before | After |
| --- | --- | --- |
| "What verifies stability", opening | `For a declared nonempty retention set ...` | `Low violation probability implies neither stability nor the desired decrease, so the update uses a deterministic test, not the bound.` |
| "What verifies stability" | "verified" defined in Safe | defined here, with the margin |
| "Safe means stable", opening | `Call $P$ \emph{verified} at $K$ ...` | `To keep $P$ verified when $K$ becomes $K^+=K+\Delta K$, we start from this margin and bound how much the step can lower it.` |
| "Safe means stable" | `The margin is a budget and $c_\theta$ bounds what the step spends.` | removed; $c_\theta$ now `bounds the terms in $E$` |
| "Safe means stable" | `however $\Delta K$ was chosen` | removed for the page limit |
| "Safe means stable", $\kappa$ | inside the test sentence | closing sentence: `The test also requires ... for cost bounds, not stability.` |

The main text ends on page 4 with no overfull boxes, at the same height as before this edit, but
that page is now full. A display equation, or the explicit expansion of $(G+E)^\top P(G+E)$, overflows
to five pages.

#### Open question

C009 says the method requires "only finite-horizon cost realizations", but $\eta_\theta$, $c_\theta$
and the exact gradient $D$ all use the known $A(\theta),B(\theta)$. Only fitting $P$ is data-driven.
Before that sentence goes into M01 or M08, either call it a goal or ask David whether he means a
later model-free version.

### M08 - Targeting idea

- **Comment:** C009
- **Where:** "Targeting" paragraph, starting at line 178.
- **Problem:** David identifies this as the big novel component: by iterating on Boffi for
  certificate learning and Fujinami for domain-randomized control, we seek to iteratively
  generate controllers and certificates while only requiring finite-horizon cost
  realizations.
- **Before drafting:** see the M07 open question on "only requiring finite-horizon cost
  realizations".
- **Original note:**

> Targeting Idea: todo (agent propose: emphasize the idea with passion)

### M09 - One guarded update

- **Comment:** C010
- **Where:** "One safe Armijo update" paragraph (previously "One guarded update"), starting at
  line 190.
- **Problem:** David sees a chance to clarify the procedure, and asks how a single iteration
  works.
- **History:** an `algorithm2e` block was added in `f4c39bd` and removed in `9587bb6`.
- **Original note:**

> One guarded update: todo (introduce back the algorithm block?)

### M10 - Linear study figure

- **Comment:** C012
- **Where:** Figure `fig:mid_results`, lines 210-220. Its discussion follows from line 222.
- **Problem:** David says the meaning of these results is not clear, and that they need more
  detailed discussion if they stay in. David asks what the plot shows and how it ties into
  the story.
- **Original note:**

> linear study graph elaboration: todo (pending discussion)

### M11 - PPO experiment contrast

- **Comment:** C013
- **Where:** "CartPole with PPO" paragraph at lines 246-252, and Figure `fig:ppo_loop` at lines
  254-262.
- **Problem:** David finds this result very interesting because robustness is clearly
  increasing, and asks whether it can be falsified. David suggests running PPO with uniform
  data allocation to see how policy robustness evolves over training iterations.
- **Current evidence:** the README says Figure 2's study
  (`cartpole-ppo-boundary-v5-adaptive`) lives on another machine, has no matched control, and
  is not confirmatory.
- **Original note:**

> PPO experiment contrast: todo - maybe change one row to a baseline (without boundary targeting the data allocation?)

### M12 - References

- **Comment:** C014 (first ask)
- **Where:** `\bibliography` at line 292, plus the citations in the table below.
- **Problem:** David finds it odd that this draft does not include the references from the
  proposal.
- **Gap at review:** Boffi, Florensa, Berkenkamp and Fujinami were already cited. Only
  `howard2021timeuniform` and `zhang2022adversarially` were missing. All six proposal
  citations now appear in the midterm.
- **Original note:**

> reference: redo by agents, incorporating the references in the proposal

#### Review

Each cited claim was read against its source, and each record against Crossref, PMLR, NeurIPS or
arXiv. Only `fujinami2025domain` had an outdated record. Fujinami quotes come from the arXiv
versions.

- **Boffi:** "continuous-time framework" and "Sections 3-4" are accurate. "Following Boffi" was
  not, because it introduced the three-arrow diagram. Boffi's result uses Srebro's fast rate and
  Rademacher complexity, and needs a feasible certificate (Lemma 4.1). The diagram is cover plus
  Hoeffding, which the research note attributes to course Theorem 5.8. M06's removal of the rate
  comparison had dropped this distinction.
- **Howard:** Figure 2 uses point estimates, so Howard is cited as a stated gap, not a method.
  Acceptance means at least 58/64, whose exact one-sided 95% lower limit is 0.823. Across Figure
  2's 5,187 checkpoint-condition pairs (273 conditions, 19 checkpoints, Bonferroni), even 64/64
  gives only 0.835, and a cell needs at least 110 episodes to clear 0.9. Howard's bounds are also
  uniform over time, not across conditions, and the post-midterm plan uses Clopper-Pearson with
  Bonferroni.
- **Fujinami, CDC 2025:** only uniform weights are analyzed. High-cost (entropic-risk) weights are
  one numerical test: "We lack theoretical convergence guarantees for this approach."
- **Fujinami, L4DC 2025 (new):** states the project's gap: "it remains poorly understood how to
  select the randomization distribution". Neither the proposal nor the midterm cited it.
- **Novelty:** "evaluation-guided targeting" overclaimed. Florensa selects by performance,
  Fujinami's entropic weights depend on current costs, and Akkaya's ADR gates range expansion on
  performance thresholds. ADR's "boundary" is the edge of the parameter range, not a cost level.
  C009 places the novelty in iterating Boffi and Fujinami. This was not an exhaustive search, so
  the narrower claim is unproven too.
- **C005:** "some ability but not mastery" was not in the text even though M04 is ticked. Florensa
  Section 4.1 ("sometimes, but not always") and Rutherford Section 4.1 ("can solve sometimes but
  have not yet mastered") support it, so it now sits in the Targeting paragraph.
- **PPO:** Schulman and Raffin were uncited. Stable-Baselines3 is confirmed for the parent config
  and for retraining, in `ppo_boundary_training.py` on chesiumx.

#### Applied fix

The related-work text follows the proposal and the course FAQ, which allows compressed related
work. It sits at the end of Section 1 because three added lines on page 1 push `eq:mid_cost` to
page 2 and lose about five lines there.

| Line | Before | After |
| --- | --- | --- |
| 110-114 | none | new "Related work" paragraph: Zhang, Fujinami L4DC, Akkaya, and the proposal's closing sentence |
| 118-121 | `Following \citet{boffi2021learning}, we bound the gap ...` | `Our foundation is the trajectory-margin framework ...`, with the rate and feasibility difference and `(course notes, Chapter~5)` |
| 174 | `What is new: evaluation-guided targeting.` | `What is new: boundary targeting under a certificate.` |
| 177-178 | `while requiring only finite-horizon rollouts` | `eventually from finite-horizon rollouts alone; the current update still uses known plants` |
| 179-180 | `Fujinami et al.\ weight plants uniformly or toward high cost` | `\citeauthor{fujinami2025domain} analyze uniform plant weights and test high-cost weights numerically` |
| 180-181 | `the acceptance boundary.` | `the acceptance boundary, where $K$ has some ability but not mastery \citep{florensa2017reverse,rutherford2024noregrets}.` |
| 194 | `\citep{armijo1966minimization}` | `\citep[Corollary~2]{armijo1966minimization}` |
| 211-215 | Section 3 opening | removed `Experiments check ...`, compacted the family list, and dropped `taking the first empirical minimizer in grid order` |
| 253-254 | `a frozen PPO policy:` | `a frozen Stable-Baselines3 PPO policy \citep{schulman2017proximal,raffin2021stable}:` |
| 258-259 | none | `Accepted sets are 64-episode point estimates, without the proposal's simultaneous bounds \citep{howard2021timeuniform}.` |

**`references.bib`:** `fujinami2025domain` moves from arXiv to CDC 2025, pp. 4174-4181, DOI
`10.1109/CDC57313.2025.11312642`. New entries are `fujinami2025sample`, `akkaya2019solving`,
`schulman2017proximal` and `raffin2021stable`. `gotovos2013active`, `letham2022lookahead`,
`jiang2021prioritized` and `fujinami2026history` stay uncited; `\nocite` is not used.

**Checked:** four main-text pages, references starting on page 5, 12 printed references, no
overfull boxes, undefined citations or BibTeX warnings. Page 4 is now full, so any added line
needs a matching cut. Figure 2's caption is unchanged.

**Follow-ups:**

- Copy these edits to Overleaf.
- Tell David why the opening differs from C007's "method of Boffi" wording.
- Ask David whether "finite-horizon rollouts" means a later model-free version. The M07 open
  question is handled for now by stating it as a goal.
- Replace "our research note" (lines 60-61) when the M13 Appendix exists.
- For the final report, consider Fujinami 2026 (its $a=\pm1.1$ example motivates the retention
  set), Kenanian et al. 2019 (common quadratic Lyapunov function from data) and Mohammadi et al.
  (stability-limited backtracking for LQR).

### M13 - Appendix

Added from C014. The original plan did not list this item.

- **Comment:** C014 (second ask)
- **Where:** new material after the main text.
- **Problem:** when the Appendix with proofs is added, David recommends including the
  original proposal there as well.
- **Requirement from M03:** the midterm states $R\succ0$, so the Appendix must prove the $R=0.1$
  example in [the example log](../2026-11-04-safe-descent-coverage-example.md), not the research note's current $R=0$ version.
- **Open questions:**
  - Which proposal copy is the original? `ese6180/proposal/latex/proposal.tex` and
    `Evaluation_Guided_Domain_Randomization/proposal.tex` differ.
  - Where does the Appendix go relative to the page limit and the build's page check?
- **Original note:** none.

### M14 - Proposal introduction and motivation

Added from C014. The original plan did not list this item.

- **Comment:** C014 (third ask)
- **Where:** to be decided.
- **Problem:** if time and space allow, David recommends working some of the proposal's
  introductory material and motivation into the report.
- **Original note:** none.
