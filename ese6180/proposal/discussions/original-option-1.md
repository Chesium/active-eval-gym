**1. Active evaluation with guarantees under adaptive sampling is my preferred starting point.**

A possible title is **“Anytime-Valid Active Evaluation of Closed-Loop Controllers.”**

The question would be:

> How can an evaluator concentrate simulation effort near a controller’s failure boundary while retaining valid guarantees about the conditions it labels acceptable?

Your repository is unusually well positioned for this. It already reports an adaptive recovery-boundary study, separate survival and recovery outcomes, independent final evaluations, and unresolved conditions. It also explicitly distinguishes results on the selected grid from performance under a deployment distribution. Those are useful foundations. I am treating these as **reported repository results**, not independently reproduced experiments. [Current findings](https://github.com/Chesium/active-eval-gym/blob/master/docs/findings.md)

The next contribution would be a principled rule for **where to sample next, when to stop, and which conclusions can be certified**.

**The relevant literature gives three different starting points.**

- **Gotovos et al., “Active Learning for Level Set Estimation,” IJCAI 2013.** This formalizes learning which inputs lie above or below a threshold, using Gaussian-process confidence bounds to direct sampling. It is a much closer mathematical match to failure-boundary mapping than ordinary Bayesian optimization. [Paper](https://people.csail.mit.edu/alkisg/files/gotovos13active.pdf)
- **Mason et al., “Nearly Optimal Algorithms for Level Set Estimation,” AISTATS 2022.** This connects threshold classification to experimental design and gives instance-dependent sample-complexity guarantees in linear/kernel settings, including an approximation-error formulation. This is the ambitious theoretical reference. [Paper](https://proceedings.mlr.press/v151/mason22a.html)
- **Howard et al., “Time-uniform, nonparametric, nonasymptotic confidence sequences.”** Confidence sequences remain valid across repeated inspection and data-dependent stopping, under their stated assumptions. They provide the statistical machinery for an evaluator that continually asks whether enough evidence has accumulated. [Paper](https://arxiv.org/abs/1810.08240)

A particularly relevant newer connection is **Raghavan and Johansson, L4DC 2026**, on trajectory-level experimental design for learning safety parameters. Their exploration policy adapts to classify regions, with analysis involving the induced Markov chain and mixing. However, its finite-state and transition assumptions are specific; its theorem cannot simply be imported into a continuous CartPole perturbation sweep. [Paper](https://proceedings.mlr.press/v331/raghavan26b.html)

Also distinguish this from **adaptive stress testing**: AST uses sequential search/RL to discover likely failure trajectories. Finding failures and estimating how often failure occurs under a specified distribution are different objectives. AST is a useful search baseline or extension, not automatically a calibrated reliability estimator. [Lee et al., JAIR 2020](https://arxiv.org/abs/1811.02188)

**A feasible theoretical core**

Freeze a controller and define

\[
p(z)=\Pr(\text{failure within }H\text{ steps}\mid z),
\]

where \(z\) specifies an operating condition, such as initial angle and pole length. Define the randomness explicitly: random initial velocities, observation noise, policy sampling, or disturbances.

For a fixed set of \(M\) controller–condition pairs, a simple starting bound is

\[
r_n=\sqrt{\frac{\log(2Mn(n+1)/\delta)}{2n}},
\qquad
[L_n,U_n]=[\hat p_n-r_n,\hat p_n+r_n]\cap[0,1].
\]

Under fresh independent Bernoulli trials for each pair, a Hoeffding bound and union bound give simultaneous coverage over all pairs and sample counts with probability at least \(1-\delta\). Adaptive allocation across pairs and stopping are then permitted.

This is an **elementary proposed specialization**, not a new theorem from the papers above. It is conservative but easy to prove and implement; tighter confidence sequences can be the comparison.

For a failure tolerance \(\varepsilon\):

- certify a condition when \(U_n\leq\varepsilon\);
- reject it when \(L_n>\varepsilon\);
- otherwise leave it unresolved.

An agent, Gaussian process, or numerical heuristic can suggest which unresolved condition to evaluate. A poor suggestion wastes budget but does not invalidate the statistical conclusion.

Two limitations are central to the formulation:

1. **Validity does not imply efficiency.** An arbitrary proposer can keep sampling uninformative points. Sample-complexity claims require a specified allocation rule and usually a nonzero gap from the threshold.
2. **A finite grid certificate covers that grid.** Extending it to every point between samples requires additional regularity.

**How to make the theory distinctly about control**

Add a small analytical system for which you can derive a sensitivity bound. For example, if the closed-loop transition is contractive in state with factor \(\rho<1\), and Lipschitz in an environment parameter \(z\), then trajectories under nearby parameters satisfy a recursion of the form

\[
\|x_{t+1}(z)-x_{t+1}(z')\|
\leq
\rho\|x_t(z)-x_t(z')\|+L_z\|z-z'\|.
\]

Unrolling it explains how stability controls the spatial resolution needed for evaluation. A continuous trajectory safety margin can inherit this regularity.

**Do not automatically apply that conclusion to binary failure probability.** Thresholding a smooth trajectory can produce a sharp boundary; translating margin bounds into probability bounds needs additional assumptions about randomness and mass near the threshold. This distinction could itself be an excellent part of the report.

**Experiments and figures**

Compare uniform allocation, your current boundary heuristic, GP-based selection, and uncertainty-directed allocation under identical rollout budgets. Add an agent proposer only after these work.

Measure:

- incorrectly certified conditions across repeated campaigns;
- fraction of the domain certified or unresolved;
- simulations required to reach a declared precision;
- boundary accuracy against a separately computed reference;
- survival versus recovery conclusions.

Animate the evolving acceptable/rejected/unresolved map and show certification progress against simulator calls.

Your current independent final evaluation is already a useful protection against discovery bias. The research question concerns **repeated certification during adaptation**, not a claim that your existing final protocol is invalid. Also, 50 successes out of 50 do not establish near-perfect reliability: the exact one-sided 95% upper bound on failure probability is about **5.8%**, even before accounting for multiple conditions.

**My assessment:** Very feasible, with substantial room for careful theory. The main risk is producing a statistics project with control examples; the trajectory-sensitivity component addresses that.

---