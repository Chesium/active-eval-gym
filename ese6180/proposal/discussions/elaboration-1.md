**Gotovos is a strong starting point for option 1—active evaluation of closed-loop controllers—but it sharpens what your project should claim.** The paper already addresses adaptive sampling with confidence bounds that hold across locations and iterations. Your opportunity is to specialize that framework to controller rollouts, clarify what “certified” means, and investigate how much sampling efficiency survives when the statistical model is imperfect.

Below, I distinguish the paper’s established results, extensions already covered by later literature, and my suggested project directions.

**Paper card — Gotovos et al. (2013), *Active Learning for Level Set Estimation***

Alkis Gotovos, Nathalie Casati, Gregory Hitz, Andreas Krause. IJCAI 2013. Algorithm 1 and Theorem 1 are the central parts for your project; these appear on PDF pages 3–4.

:codex-file-citation{path="C:/A/s5/ESE6180/papers/activeeval/gotovos13active.pdf" purpose="source"}

---

**1. Story / research questions**

Imagine that measuring a function is expensive, but you only need to know **where it exceeds a threshold**.

The paper’s motivating examples include locating excessive chlorophyll concentrations in a lake and identifying servers whose network latency satisfies a requirement. Exhaustively measuring every location wastes resources, especially far from the threshold.

Formally, given an unknown function \(f\), a finite candidate set \(D\), and threshold \(h\), identify

\[
H=\{x\in D:f(x)>h\},
\qquad
L=\{x\in D:f(x)\le h\}.
\]

You can sequentially query locations and receive noisy measurements.

The research questions are:

- **Allocation:** Which location should be measured next to resolve the level set efficiently?
- **Reliability:** Can the resulting classification be accurate everywhere, with high probability?
- **Sample complexity:** How many measurements suffice, and how does this depend on the function’s structure?
- **Extensions:** What changes when the threshold depends on an unknown maximum, or measurements arrive in batches?

The distinction from Bayesian optimization matters. Optimization seeks a good maximizer; level-set estimation seeks **the entire acceptable region**. For controller evaluation, finding one successful operating condition is usually insufficient—you want to characterize where the controller works.

**My reading of the central contribution:** it turns threshold-boundary exploration into an algorithm with an explicit classification guarantee and a bound on the required measurements.

---

**2. Method / statistical and optimization concepts**

**The model: Gaussian-process regression**

The observation model is

\[
y_t=f(x_t)+\eta_t,
\qquad
\eta_t\sim\mathcal N(0,\sigma^2),
\]

with independent Gaussian noise and a GP prior over \(f\).

The GP provides a posterior mean \(\mu_{t-1}(x)\) and standard deviation \(s_{t-1}(x)\). Its kernel allows a measurement at one location to inform neighboring locations.

This spatial sharing is the main source of potential efficiency: **a point can be classified without being directly measured.**

**The confidence construction: uncertainty that shrinks consistently**

At iteration \(t\), construct

\[
Q_t(x)=
\left[
\mu_{t-1}(x)-\sqrt{\beta_t}s_{t-1}(x),
\;
\mu_{t-1}(x)+\sqrt{\beta_t}s_{t-1}(x)
\right].
\]

Then intersect with earlier confidence intervals:

\[
C_t(x)=C_{t-1}(x)\cap Q_t(x).
\]

Write its endpoints as \(\ell_t(x)\) and \(u_t(x)\). Intersections make the retained intervals nested, which supports permanent classification decisions and the convergence argument.

**The classification rule: allow a tolerance near the threshold**

For an accuracy parameter \(\epsilon>0\):

\[
\begin{aligned}
\ell_t(x)+\epsilon>h
&\quad\Rightarrow\quad \text{classify above},\\
u_t(x)-\epsilon\le h
&\quad\Rightarrow\quad \text{classify below},\\
\text{otherwise}
&\quad\Rightarrow\quad \text{leave unresolved}.
\end{aligned}
\]

That \(\epsilon\) is consequential. The algorithm permits classification errors sufficiently close to the threshold, allowing it to finish without resolving arbitrarily tiny differences.

**The acquisition rule: query the most ambiguous unresolved point**

Define

\[
a_t(x)=\min\{u_t(x)-h,\;h-\ell_t(x)\}.
\]

The algorithm queries an unresolved point with maximum ambiguity.

For a single symmetric interval, this becomes

\[
a_t(x)=
\sqrt{\beta_t}s_{t-1}(x)
-
|\mu_{t-1}(x)-h|.
\]

It favors points that are both uncertain and close to the threshold. Pure maximum-variance sampling could spend measurements on uncertain regions that are nevertheless clearly acceptable or unacceptable.

The rule itself is a simple greedy selection. Mutual information enters the **analysis**, rather than being the acquisition objective directly.

**The guarantee: what Theorem 1 actually says**

The paper chooses

\[
\beta_t=
2\log\!\left(\frac{|D|\pi^2t^2}{6\delta}\right).
\]

Under its GP and observation assumptions, this supports confidence coverage simultaneously across the candidate set and iterations.

The returned partition satisfies, with probability at least \(1-\delta\),

\[
\max_{x\in D}\operatorname{loss}_h(x)\le\epsilon,
\]

where a misclassified point’s loss is its function-value distance from the threshold.

Consequently:

- Points sufficiently far above or below \(h\) receive the correct label.
- Wrong labels are permitted within the \(\epsilon\)-band around \(h\).
- \(\epsilon\) is **not** the fraction of incorrectly labeled points.
- It is also **not** a geometric distance between estimated and true boundaries.

The sample bound is expressed using \(\gamma_T\), the maximum information obtainable from \(T\) noisy measurements:

\[
\frac{T}{\beta_T\gamma_T}
\ge
\frac{C_1}{4\epsilon^2},
\qquad
C_1=\frac{8}{\log(1+\sigma^{-2})}.
\]

The proof connects four ideas:

\[
\text{simultaneous coverage}
\;\longrightarrow\;
\text{sound relaxed labels}
\;\longrightarrow\;
\text{information bounds uncertainty reduction}
\;\longrightarrow\;
\text{finite termination}.
\]

For ESE6180, this is a useful proof architecture to learn.

| Concept | Role in this paper |
|---|---|
| GP regression / kernels | Share information across locations |
| Concentration and union bounds | Account for multiple locations and repeated decisions |
| Active learning | Allocate measurements to unresolved classifications |
| Information gain | Bound cumulative uncertainty and sample complexity |
| Sequential experimental design | Choose each experiment using previous observations |
| Control theory | Not explicitly part of the core theorem; this is where your specialization can enter |

A qualification: **Theorem 1 is model-dependent.** Its probability statement uses the stipulated GP/noise model; it does not provide distribution-free coverage for every arbitrary fixed function.

**What the experiments establish**

The paper evaluates network-latency and lake-monitoring problems, primarily using classification F1 scores. Its active methods perform well against alternatives such as maximum-variance sampling in those experiments.

However, the experiments use practical parameter choices, including a constant confidence multiplier, rather than precisely implementing the theorem’s growing \(\beta_t\). Empirical F1 performance and the theoretical uniform guarantee should therefore be discussed separately.

---

**3. Limitations—and what later literature has already addressed**

Several natural “extensions” are already substantial research areas. I would use them as tools or comparison methods, rather than present them as untouched gaps.

| Limitation or question arising from Gotovos | Later literature | Consequence for your project |
|---|---|---|
| **Continuous Gaussian measurements do not directly model pass/fail rollouts.** | **Letham et al. (2022), [*Look-Ahead Acquisition Functions for Bernoulli Level Set Estimation*](https://proceedings.mlr.press/v151/letham22a.html)** develops GP-classification-based acquisition functions for binary observations, accounting for their expected effect on the level-set posterior. | This is the most direct next paper for your failure-probability setting. “Extend LSE to binary outcomes” is already addressed. |
| **Local ambiguity does not explicitly value uncertainty reduction elsewhere, or unequal evaluation costs.** | **Bogunovic et al. (2016), [TRUVAR](https://arxiv.org/abs/1610.07379)** selects measurements through truncated variance reduction and accommodates heterogeneous costs and noise. | Useful when rollout duration varies, or when deciding between many cheap tests and fewer expensive tests. Cost-aware sampling alone is not a new contribution. |
| **A fixed finite grid can be expensive and misses what happens between grid points.** | **Shekhar and Javidi (2019), [*Multiscale Gaussian Process Level Set Estimation*](https://proceedings.mlr.press/v89/shekhar19a.html)** uses hierarchical refinement to address continuous domains under GP regularity assumptions. | Adaptive refinement near a controller’s boundary has established foundations. Any continuum guarantee still needs regularity assumptions. |
| **A worst-case information-gain bound may poorly describe the difficulty of a particular boundary.** | **Mason et al. (2022), [*Nearly Optimal Algorithms for Level Set Estimation*](https://proceedings.mlr.press/v151/mason22a.html)** connects LSE to experimental design and derives instance-dependent bounds in linear/kernel settings. It also treats approximation by an RKHS function with misspecification. | Strong reading for sample-allocation theory. Its misspecification treatment is structured, not a guarantee under arbitrary model error. |
| **The theoretically prescribed confidence multiplier can be conservative.** | **Inatsu et al. (2024), [*Active Learning for Level Set Estimation Using Randomized Straddle Algorithms*](https://arxiv.org/abs/2408.03144)** randomizes the exploration parameter and establishes guarantees under its loss formulations. | A useful acquisition baseline. Its main expected-loss results and related high-probability variants are not interchangeable with simultaneous strict certification. |
| **Performance depends on uncertain environmental distributions.** | **Inatsu et al. (2021), [*Active Learning for Distributionally Robust Level-Set Estimation*](https://proceedings.mlr.press/v139/inatsu21a.html)** studies reliable regions under a family of environmental distributions. | Relevant to wind, disturbances, or plant uncertainty. Distributionally robust LSE already exists; a control-specific application must state what it adds. |

Two further limitations deserve particular attention.

**Model calibration is part of the guarantee.** A smooth GP can confidently smooth over a narrow failure region. Learning kernel hyperparameters from the evaluation data does not automatically preserve the original theorem. Later kernel and misspecification results help under their own assumptions; they do not remove the need to justify spatial generalization.

**Level-set accuracy is different from strict certification.** Gotovos deliberately permits errors close to the threshold. That is appropriate for approximate mapping, but may differ from what you want to tell a controller designer.

Also, three things are **already in Gotovos**:

- Adaptive sampling with confidence accounting across iterations.
- Thresholds defined relative to the unknown maximum.
- Batch selection with adjustments for delayed feedback.

They should not be proposed as missing features of this paper.

---

**4. Connection to option 1: active evaluation of closed-loop controllers**

The most natural mapping is:

| Gotovos formulation | Your controller-evaluation formulation |
|---|---|
| Location \(x\) | Operating condition \(z\): initial condition, plant parameter, disturbance setting |
| Unknown function \(f(x)\) | Failure probability \(p_\pi(z)\) of a fixed controller \(\pi\) |
| Threshold \(h\) | Maximum acceptable failure probability \(\alpha\) |
| Noisy measurement | One closed-loop rollout with a binary failure outcome |
| Below-threshold set | Conditions satisfying \(p_\pi(z)\le\alpha\) |
| Next measurement | Next condition at which to run an episode |

For a fixed controller and horizon \(H\),

\[
p_\pi(z)
=
\Pr\!\left(
\text{failure occurs within }H
\mid \pi,z
\right).
\]

The randomness must be specified—for example, disturbances, sensor noise, or randomized initial conditions within a declared condition.

**One rollout supplies one binary outcome.** Its time steps do not become independent failure-probability samples.

The resulting feedback structure is:

\[
\text{choose condition}
\;\rightarrow\;
\text{run closed-loop system}
\;\rightarrow\;
\text{observe outcome}
\;\rightarrow\;
\text{update evaluation and choose again}.
\]

There is both feedback inside the controller and feedback in experimental allocation. This gives the project a clear learning-from-feedback interpretation.

**The crucial difference: approximate maps versus certified acceptable sets**

Suppose your requirement is

\[
p_\pi(z)\le 0.05.
\]

An \(\epsilon=0.01\) level-set guarantee can permit a point with \(p_\pi(z)=0.052\) to receive an acceptable label: the error is within the allowed function-value tolerance.

That satisfies approximate LSE accuracy, but does not establish the stated \(5\%\) requirement.

For option 1, a more appropriate target could be

\[
\Pr\!\left(
\forall t,\;\forall z\in S_t:
p_\pi(z)\le\alpha
\right)\ge1-\delta,
\]

where \(S_t\) is the set certified acceptable by iteration \(t\).

Given valid simultaneous intervals \([L_t(z),U_t(z)]\), use:

\[
\begin{aligned}
U_t(z)\le\alpha
&\Rightarrow \text{certify acceptable},\\
L_t(z)>\alpha
&\Rightarrow \text{certify unacceptable},\\
\text{otherwise}
&\Rightarrow \text{remain unresolved}.
\end{aligned}
\]

**Abstention replaces the relaxed labels near the threshold.** You gain a stricter interpretation, but lose a general promise that every point will be classified in finite time. Points at or extremely close to the threshold can remain unresolved.

Here, “acceptable” concerns the specified finite-horizon failure probability. It does not establish infinite-horizon stability or safety for every possible disturbance.

---

**5. Possible extension / specialization: what I would actually propose**

**A. Most manageable: GP-guided sampling with separate statistical certification**

Use a GP or Bernoulli-LSE model to suggest informative test conditions. Use confidence sequences built from fresh rollout outcomes to decide which conditions can be certified.

Time-uniform confidence sequences are established methodology; [Howard et al., *Time-uniform, nonparametric, nonasymptotic confidence sequences*](https://arxiv.org/abs/1810.08240) provides the relevant foundation.

The architecture is:

| Component | Responsibility |
|---|---|
| Acquisition model | Suggest where another rollout would be useful |
| Statistical certification | Determine which conclusions are supported |
| Allocation safeguard | Ensure important conditions are not permanently neglected |

For a fixed finite condition set, suitable per-condition sequences and a union bound can give simultaneous coverage. Adaptive allocation is compatible with this construction when each selected condition produces a fresh outcome with the stipulated, stationary conditional distribution.

An agent or LLM could participate in proposing tests. **The guarantee would concern the certification procedure, rather than the agent’s reasoning.**

The research question becomes:

> How much evaluation cost can model-guided sampling save while preserving valid certification, particularly when the spatial model is wrong?

The important tradeoff is that independent per-condition certification does not automatically share statistical evidence across conditions. It sacrifices some of Gotovos’s efficiency to reduce reliance on the GP.

Soundness alone also does not prove efficiency: a poor proposer can waste its entire budget. A sampling policy or exploration safeguard is needed for termination or sample-complexity claims.

**B. Stronger ESE6180 specialization: derive spatial regularity from closed-loop dynamics**

This is the direction that would make the project more distinctly about control.

Let

\[
s_{k+1}=F_z(s_k)
\]

be the closed-loop dynamics. Suppose, on a relevant region,

\[
\|F_z(s)-F_{z'}(s')\|
\le
\rho\|s-s'\|+L_z\|z-z'\|,
\qquad \rho<1.
\]

Then

\[
\|s_k(z)-s_k(z')\|
\le
\rho^k\|s_0(z)-s_0(z')\|
+
L_z\frac{1-\rho^k}{1-\rho}\|z-z'\|.
\]

This bounds how trajectories change as the test condition changes. With a Lipschitz safety-margin function, it can support justified generalization between nearby conditions.

The conceptual connection is:

\[
\text{closed-loop regularity}
\;\rightarrow\;
\text{trajectory sensitivity}
\;\rightarrow\;
\text{safety-margin regularity}
\;\rightarrow\;
\text{fewer required evaluations}.
\]

There is a subtlety: **regularity of trajectories does not automatically imply regularity of failure probability.** Thresholding a continuous margin produces a discontinuous indicator. Probability-level bounds require additional assumptions, such as control over how much probability mass lies near zero margin.

A feasible specialization is therefore to analyze a continuous trajectory safety margin first, and make probability-level extensions optional.

**C. Further application: robustness to environmental uncertainty**

Instead of fixing one disturbance distribution, certify acceptable conditions against a stated family of distributions. The distributionally robust LSE literature provides a starting point.

This is appealing for simulator-to-deployment mismatch, but adds another layer of uncertainty. I would keep it as a later extension unless environmental uncertainty is the project’s central question.

---

**6. A concrete experimental study**

For a first implementation, I would freeze one or two controllers and use a two-dimensional condition map. This keeps the unknown function stationary and makes the results easy to visualize.

Compare:

| Method | Purpose |
|---|---|
| Uniform allocation + the same certification rule | Establish the baseline evaluation cost |
| Straddle-style allocation + certification | Test the value of local boundary-focused sampling |
| Bernoulli look-ahead allocation + certification | Test whether global posterior impact improves allocation |
| GP-only level-set labels | Measure the practical benefits and risks of relying on model-based classification |

The GP-only output should retain its own stated assumptions and accuracy criterion; comparing it with strict certification requires acknowledging that the guarantees differ.

Useful figures would show:

- Certified acceptable, certified unacceptable, and unresolved regions as the budget grows.
- Rollouts required to certify a specified fraction of the grid.
- Sampling locations relative to the true boundary.
- False-certification frequency across independently repeated campaigns.
- Sensitivity to a deliberately misspecified kernel or a narrow failure region.

Ground truth should come from a separately justified analytical model or sufficiently precise reference evaluation. A dense Monte Carlo map has its own uncertainty.

For a theory component, I would aim for **one clean certification proposition**, then optionally a dynamics-based regularity lemma. The experiment would measure efficiency and robustness beyond what those propositions guarantee.

**My recommendation:** use Gotovos as the conceptual and proof template; read **Letham next** for binary rollouts, then **Howard** for the strict sequential-certification layer. Read **Mason** when you want to strengthen the sample-allocation analysis.

A defensible course-project framing is:

> **Active certification of controller operating regions: model-guided experiment selection with time-uniform guarantees.**

Its contribution can be a careful synthesis and control specialization. Establishing that it is a new research method would require a more targeted novelty search once the exact algorithm and guarantee are fixed.

- Next paper card: Letham
- Walk through the Gotovos guarantee
- Turn this into a project formulation