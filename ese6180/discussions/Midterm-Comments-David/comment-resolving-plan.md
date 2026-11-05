# Midterm comment tracker

Tracks David's Overleaf review of
[`midterm_report.tex`](../../Evaluation_Guided_Domain_Randomization/midterm_report.tex).
The full comment text is in [`comments-2026-11-05.md`](comments-2026-11-05.md).

- **Codes:** `M01`-`M14` are this tracker's items. `C001`-`C014` are the Overleaf
  comment IDs. All 14 comments are covered below.
- **Line numbers:** these refer to `midterm_report.tex` as of the export.
- **Owner:** Shimin, Agent, Discuss (Shimin and agent together), or Unassigned.
- **Status:** todo, discussing, drafted, or done.
- **Original notes:** these are kept verbatim from the first version of this plan.
  They are still work in progress and have not been answered or corrected.

## Index

| Code | Item | Comments | Lines | David's priority | Owner | Status |
| --- | --- | --- | --- | --- | --- | --- |
| [M01](#m01---abstract) | Abstract | C001, C002 | 22-28 | High (C001) | Shimin | todo |
| [M02](#m02---the-term-guarded) | The term "guarded" | C003, C011 | 63, 193 | | Agent | done |
| [M03](#m03---r-positive-definite-or-semidefinite) | $R\succ0$ or $R\succeq0$ | C004 | 79 | | Discuss | todo |
| [M04](#m04---performance-metric-intuition) | Performance metric intuition | C005 | 78-85 | High | Shimin | todo |
| [M05](#m05---certificate-loss-intuition) | Certificate loss intuition | C006 | 88-114 | | Shimin, after discussion | todo |
| [M06](#m06---uniform-bound-idea) | Uniform bound idea | C007 | 117 | | Agent | todo |
| [M07](#m07---what-verifies-stability) | What verifies stability | C008 | 154 | | Discuss | todo |
| [M08](#m08---targeting-idea) | Targeting idea | C009 | 163 | | Agent | todo |
| [M09](#m09---one-guarded-update) | One guarded update | C010 | 193 | | Unassigned | todo |
| [M10](#m10---linear-study-figure) | Linear study figure | C012 | 210-220 | | Discuss | todo |
| [M11](#m11---ppo-experiment-contrast) | PPO experiment contrast | C013 | 254-270 | | Unassigned | todo |
| [M12](#m12---references) | References | C014 | 291-293 | | Agent | todo |
| [M13](#m13---appendix) | Appendix (not in original plan) | C014 | new | | Unassigned | todo |
| [M14](#m14---proposal-introduction-and-motivation) | Proposal introduction and motivation (not in original plan) | C014 | new | Optional | Unassigned | todo |

## Constraints and dependencies

- **Page limit:** the course limits the midterm to four pages, excluding references, and
  `ese6180/build_report.py` fails above that. The main text already ends on page 4, so
  added text needs matching cuts. This affects M04-M10 and M14.
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
"Safe means stable" and the existing paragraph "Stability and descent do not imply coverage"
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
- Page 4 stays full, so this leaves no room for M04-M10.
- Update the same term in `Report_Sections/02_problem_formulation.tex` line 504,
  `Report_Sections/05_theoretical_results.tex` lines 518 and 647, and `README.md` lines 7 and
  79. This matters most if M13 reuses that text.
- Code and `docs/` also say "guard". They are not part of the report and can stay.

### M03 - R positive definite or semidefinite

- **Comment:** C004
- **Where:** line 79. Related uses of $R$:
  - the counterexample with $R=0$ at line 247;
  - `Report_Sections/02_problem_formulation.tex`, lines 309, 342 and 630;
  - `Report_Sections/05_theoretical_results.tex`, line 640.
- **Problem:** David says $R\succeq0$ is probably imprecise, because $R\succ0$ is almost
  always required for numerical reasons.
- **Original note:**

> R succ 0 or R succeq 0: todo (agent discuss)

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
- **Where:** "What generalizes: a uniform bound" paragraph, starting at line 117. The
  diagram at lines 127-138 appears to be the "nice visual equation" David mentions.
- **Problem:** David says the idea needs to be clear, and likely needs only 2-3 sentences
  before the math.
- **David's suggestion:** most of the first block of text can become "we follow the method
  of Boffi et al., which ultimately bounds sample error via bounding successive
  differences", followed by the visual equation. Pose this as the foundation of the work:
  following the ideas of Boffi.
- **Original note:**

> Uniform bound idea: todo (agent propose)

### M07 - What verifies stability

- **Comment:** C008
- **Where:** "What verifies stability: a deterministic test" paragraph, starting at line 154.
  The certificate guard definition, now titled "Safe means stable", starts at line 174.
- **Problem:** David identifies this as the beginning of the novel continuation: building
  this into a feedback loop using the domain randomization ideas of Fujinami et al.
- **Original note:**

> What verifies stability: a deterministic test: (agent discuss)
> - does the eta_theta(P)=lambda_min(P-GPG) directly leads to the Certificate Guard Definition? if so we should show that continuation

### M08 - Targeting idea

- **Comment:** C009
- **Where:** "Targeting" paragraph, starting at line 163.
- **Problem:** David identifies this as the big novel component: by iterating on Boffi for
  certificate learning and Fujinami for domain-randomized control, we seek to iteratively
  generate controllers and certificates while only requiring finite-horizon cost
  realizations.
- **Original note:**

> Targeting Idea: todo (agent propose: emphasize the idea with passion)

### M09 - One guarded update

- **Comment:** C010
- **Where:** "One safe Armijo update" paragraph (previously "One guarded update"), starting at
  line 193.
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
- **Where:** "CartPole with PPO" paragraph at lines 254-260, and Figure `fig:ppo_loop` at lines
  262-270.
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
- **Where:** lines 291-293, plus each new citation in the text.
- **Problem:** David finds it odd that this draft does not include the references from the
  proposal.
- **Current gap:** the proposal cites `fujinami2025domain`, `howard2021timeuniform` and
  `zhang2022adversarially`; the midterm cites none of them. All three are already in
  `references.bib`, but only cited entries are printed. M07 and M08 also rely on Fujinami.
  M02 already added the fourth proposal citation, `berkenkamp2017safe`.
- **Original note:**

> reference: redo by agents, incorporating the references in the proposal

### M13 - Appendix

Added from C014. The original plan did not list this item.

- **Comment:** C014 (second ask)
- **Where:** new material after the main text.
- **Problem:** when the Appendix with proofs is added, David recommends including the
  original proposal there as well.
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
