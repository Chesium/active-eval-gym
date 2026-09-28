# Understanding the linear foundation and its bridge to retraining

This note explains the revised [proposal](../latex/proposal.tex), assuming basic linear algebra and feedback control. It follows the progression **Course foundation → Main research → Ambitious theoretical outcome**:

- **Course foundation:** learn a shared linear controller across a small family of plants, derive a finite-horizon performance bound, and build an evaluator with statistical certificates.
- **Main research:** use evaluation to adapt the training distribution, investigate gains and regressions analytically, and compare training-condition rules in CartPole experiments.
- **Ambitious theoretical outcome:** identify assumptions under which a specified controller update preserves acceptable conditions or increases their coverage.

The selected foundation is a **finite-horizon quadratic-cost example and direct sensitivity/preservation proof**. Reproducing an infinite-horizon domain-randomized LQR convergence theorem is not required. The [revision notes](v2-revision-notes.md) record this decision and the remaining choices. The older [brief](../brief.md) and [design notes](proposal-design-notes.md) provide background; their fixed-plant derivations need the extensions below to match the current proposal.

The textbook background is João P. Hespanha, *Linear Systems Theory*, second edition (2018), available as a [local PDF](../literature/Linear-Systems-2e.pdf). Reading pointers appear at the end. The calculations and examples below are explanations for this project, not claims of new textbook theorems or completed experiments.

## 1. Understand the three loops and their different variables

The **control loop** acts within an episode: observe the state, apply an action, and observe the next state. The **evaluation loop** selects conditions and collects trajectories of a frozen controller to estimate where it meets a requirement. The **improvement loop** uses that evidence to choose training conditions, learns an updated controller, and freezes it for a new evaluation.

| Symbol | Meaning |
| --- | --- |
| $t$, $H$ | Time within a trajectory and its finite horizon |
| $k$ | Outer evaluation–training round |
| $\theta$ | Plant parameters, such as pole length or entries of a dynamics matrix |
| $K$, $K'$ | Old and updated linear feedback gains |
| $\pi_k$ | Policy frozen for evaluation in outer round $k$ |
| $z=(\theta,s)$ | Evaluation condition: plant parameters and an initial-state distribution indexed by $s$ |
| $x_0\sim\nu_s$ | Initial state drawn according to the selected condition |
| $q_k$ | Training-condition distribution, which may change after evaluation |
| $\mu$ | Fixed reference measure used to assess coverage |

A deterministic initial state is a special case: $\nu_s$ puts all its mass at one vector $x$. In CartPole, $s$ could instead fix the pole angle while specifying randomness in other reset coordinates. Remaining episode randomness must also be declared. A coordinate should not simultaneously be treated as fixed and independently randomized.

The key separation is that **$q_k$ changes while $\mu$ stays fixed**. Training may emphasize difficult conditions, but improvement is judged on the same reference domain. Otherwise, a change in the distribution being scored could look like a change in controller performance.

## 2. Start with a family of plants and close the control loop

The revised linear example is

$$
x_{t+1}=A(\theta)x_t+B(\theta)u_t,\qquad u_t=-Kx_t.
$$

The state $x_t\in\mathbb R^n$ contains the variables needed to predict the next step; $u_t\in\mathbb R^m$ is the action. The matrices $A(\theta)$ and $B(\theta)$ describe one member of a known, low-dimensional family. For example, a scalar family could have $A(\theta)=\theta$ and $B(\theta)=1$.

Draw or choose $\theta$ once for an episode and keep it fixed throughout that trajectory. A **shared controller $K$** is used across plant parameters. Learning a different gain for every plant would be a different problem.

Substituting the feedback law gives

$$
x_{t+1}=\underbrace{[A(\theta)-B(\theta)K]}_{G_K(\theta)}x_t,
\qquad x_t=G_K(\theta)^t x_0.
$$

Thus $G_K(\theta)$ combines the plant and controller into one closed-loop matrix. Starting at $x_0=x$, the first states are $x,G_K(\theta)x,G_K(\theta)^2x$. We use $G^0=I$. A prime in $K'$ means an updated gain; transpose is written $\top$.

For these calculations, assume full-state feedback and deterministic dynamics without disturbances, saturation, action quantization, or early termination. The model is known for the analytical example: learning the controller does not mean identifying unknown dynamics. The restriction $\|x\|_2\le r$ introduced below concerns initial states, not all subsequent states.

Unlike the old fixed-plant example, this formulation can evaluate different plant parameters. However, each trajectory calculation still holds a particular $\theta$ fixed. A theorem comparing neighboring plant parameters would require a separate bound on how their matrices differ.

## 3. Turn a trajectory into a quadratic cost

For fixed symmetric weights $Q\succeq0$ and $R\succeq0$, define

$$
J_{H,K}(\theta,x)
=\sum_{t=0}^{H-1}(x_t^\top Qx_t+u_t^\top Ru_t).
$$

Lower cost is better. Positive semidefiniteness means that every stage penalty is nonnegative. For example, $Q=\operatorname{diag}(10,1)$ weights error in the first state coordinate ten times as much as in the second. $R$ weights control effort. Fix coordinate units and weights before interpreting cost or distances.

The state-only objective from the previous note is the special case $R=0$. Keeping $R$ explicit makes the connection to quadratic controller learning clear without deciding its numerical value for the project. There is no separate terminal penalty. At $H=1$, the controller affects the input penalty, but cannot affect the only scored state; with $R=0$, that cost is independent of $K$.

Under feedback, $u_t^\top Ru_t=x_t^\top K^\top RKx_t$. Define the stage-cost matrix

$$
W_K=Q+K^\top RK.
$$

Then

$$
\begin{aligned}
J_{H,K}(\theta,x)
&=\sum_{t=0}^{H-1}(G_K(\theta)^t x)^\top W_K(G_K(\theta)^t x)\\
&=x^\top P_H(K,\theta)x,\\
P_H(K,\theta)
&=\sum_{t=0}^{H-1}(G_K(\theta)^t)^\top W_KG_K(\theta)^t.
\end{aligned}
$$

**$W_K$ scores one step; $P_H(K,\theta)$ summarizes the whole trajectory through its initial state.** Every summand is positive semidefinite, so $P_H$ is too. An equivalent computation, for a fixed $K,\theta$, is

$$
P_0=0,\qquad P_{H+1}=W_K+G_K(\theta)^\top P_HG_K(\theta).
$$

For a scalar closed loop with $G_K(\theta)=0.5$, $Q=1$, $R=0$, and $H=3$,

$$
J_{3,K}(\theta,x)=x^2+(0.5x)^2+(0.25x)^2=1.3125x^2.
$$

Computing this cost evaluates a given controller. Choosing a good $K$ is an additional optimization problem. With nonzero $R$, an update changes **both** the dynamics matrix $G_K$ and stage-cost matrix $W_K$; an update-size bound must account for both.

## 4. Learn a controller, then evaluate its acceptable region

For a fixed training distribution $q$ on conditions $z=(\theta,s)$, one possible linear training objective is

$$
F_q(K)=\mathbb E_{(\theta,s)\sim q}\,
       \mathbb E_{x\sim\nu_s}[J_{H,K}(\theta,x)].
$$

A small analytical experiment can minimize a finite-sample approximation using gradients through the matrix formula. The foundation first keeps $q$ fixed. In the main research, evaluation informs a new $q_k$ before the next update. This defines a manageable controller-learning example, not a claim of global convergence or a requirement to reproduce an infinite-horizon algorithm.

Training minimizes an average cost, but the evaluation question can be whether individual conditions meet a threshold. On a plant domain $\Theta$ and initial-state domain $D=\{x:\|x\|_2\le r\}$, define

$$
S_K^J=\{(\theta,x)\in\Theta\times D:J_{H,K}(\theta,x)\le h\},\qquad h>0.
$$

The superscript $J$ distinguishes this deterministic cost set from the proposal's failure-probability set $S_k$. The cost set here uses realized initial states; when a condition specifies a nontrivial initial-state distribution, expected-cost and failure-probability criteria would be different choices.

For a **fixed plant parameter**, $x^\top P_H(K,\theta)x\le h$ describes an ellipsoid if $P_H$ is positive definite, intersected with $D$. If $P=\operatorname{diag}(p_1,p_2)$ with positive entries, the ellipse has semiaxis lengths $\sqrt{h/p_1}$ and $\sqrt{h/p_2}$. Larger cost in a direction allows less initial error. For a general positive-definite matrix, eigenvectors determine the axes. A semidefinite matrix can leave unpenalized directions; intersecting with $D$ still bounds the domain.

The full region over $(\theta,x)$ is a collection of these plant-specific slices, not necessarily a single ellipsoid. For the scalar example above with $h=1$, the acceptable interval at that plant is $|x|\le\sqrt{1/1.3125}\approx0.873$, intersected with $D$.

Membership is a finite-horizon performance statement. It does not by itself prove invariance, absence of other constraint violations, or asymptotic stability.

## 5. A useful warm-up: vary the initial state

Hold $K$ and $\theta$ fixed and abbreviate $P_H(K,\theta)$ to $P$. The Euclidean matrix norm obeys

$$
|a^\top Mb|\le\|a\|_2\|M\|_2\|b\|_2.
$$

Here $\|M\|_2$ is the largest singular value, describing the maximum stretch of a vector. For symmetric positive-semidefinite $P$, it is the largest eigenvalue.

Add and subtract $x'^\top Px$ to obtain

$$
\begin{aligned}
x^\top Px-x'^\top Px'
&=(x-x')^\top Px+x'^\top P(x-x'),\\
|J_{H,K}(\theta,x)-J_{H,K}(\theta,x')|
&\le\|P\|_2(\|x\|_2+\|x'\|_2)\|x-x'\|_2\\
&\le 2r\|P_H(K,\theta)\|_2\|x-x'\|_2.
\end{aligned}
$$

The last line assumes $x,x'\in D$. This Lipschitz bound says that cost changes by at most a constant times initial-state distance. If the old cost is $h-m$ and $L=2r\|P_H(K,\theta)\|_2$, then $L\|x'-x\|_2\le m$ guarantees acceptability at $x'$. For instance, margin $m=2$ and $L=4$ protect a radius of $0.5$ within $D$.

This is sufficient, not necessary. With known matrices, directly evaluating the new start is exact; the bound explains how performance margins can justify generalization. It holds at one plant parameter and does not establish continuity in $\theta$. The controller-update bound below is the result emphasized in the revised proposal.

Horizon and transient amplification affect both bounds because

$$
\|P_H(K,\theta)\|_2
\le\|W_K\|_2\sum_{t=0}^{H-1}\|G_K(\theta)^t\|_2^2.
$$

Even a stable matrix can temporarily amplify states. For example,

$$
G=\begin{pmatrix}0.5&4\\0&0.5\end{pmatrix},\qquad
x_0=\begin{pmatrix}0\\1\end{pmatrix}
\quad\Longrightarrow\quad
x_1=\begin{pmatrix}4\\0.5\end{pmatrix}.
$$

Both eigenvalues are $0.5$, yet the state norm grows from $1$ to about $4.03$. With $W_K=I$ and $H=2$, the cost is $17.25$. If one establishes $\|G_K(\theta)^t\|_2\le c\rho^t$ with $c\ge1$ and $0<\rho<1$, then

$$
\|P_H(K,\theta)\|_2
\le\|W_K\|_2c^2\frac{1-\rho^{2H}}{1-\rho^2}.
$$

The prefactor allows transient growth. These constants need not be uniform across plants. The finite-horizon results themselves require no stability assumption; unstable dynamics may simply make the bounds very large.

## 6. The foundation result: change the shared controller

Compare $K$ with $K'$ at the **same** plant parameter and initial state. Hold $A(\theta),B(\theta),Q,R,H,h,r$ fixed and set

$$
\Delta P(\theta)=P_H(K',\theta)-P_H(K,\theta).
$$

The quadratic representation immediately gives

$$
\begin{aligned}
|J_{H,K'}(\theta,x)-J_{H,K}(\theta,x)|
&=|x^\top\Delta P(\theta)x|\\
&\le\|x\|_2^2\|\Delta P(\theta)\|_2\\
&\le r^2\|\Delta P(\theta)\|_2=:b(\theta).
\end{aligned}
$$

**At each plant parameter, every start in $D$ experiences a cost change of at most $b(\theta)$.** This includes changes in the input penalty when $R\ne0$, because they are already part of $P_H$. The bound controls improvements and degradations; it does not predict their direction.

An old acceptable condition with enough margin is preserved:

$$
J_{H,K}(\theta,x)\le h-b(\theta)
\quad\Longrightarrow\quad
J_{H,K'}(\theta,x)\le h.
$$

Consequently,

$$
\{(\theta,x)\in\Theta\times D:J_{H,K}(\theta,x)\le h-b(\theta)\}
\subseteq S_{K'}^J,
$$

and losses can occur only in the band

$$
S_K^J\setminus S_{K'}^J
\subseteq B_{\rm loss}
:=\{(\theta,x)\in\Theta\times D:
 h-b(\theta)<J_{H,K}(\theta,x)\le h\}.
$$

For $h=10$ and $b(\theta)=2$, all starts at that plant with old cost at most $8$ remain acceptable. Starts between $8$ and $10$ require further checking. The band width is in cost units; it need not correspond to a geometrically thin set or small probability mass.

A single bound across plants is $\bar b=\sup_{\theta\in\Theta}b(\theta)$, **provided it is finite**. A finite grid permits taking a maximum. A continuous domain needs justified uniform bounds, for example bounded continuous matrices on a compact parameter domain with fixed finite gains and horizon. Pointwise finite-horizon calculations alone do not bound an arbitrary unbounded family.

If $b(\theta)>h$, the guaranteed set at that plant is empty; if $b(\theta)=h$, only zero-cost starts qualify. Even a correct bound can be uninformative. It can be computed after a proposed update, but does not by itself choose a useful update or prove that boundary-guided training works.

## 7. Keep the ellipse example as a fixed-plant illustration

The existing figure remains useful as one slice of the family, with $R=0$. Set

$$
A=B=I_2,\quad Q=I_2,\quad H=2,\quad h=1.25,\quad r=1.2,
\qquad
K_\kappa=\operatorname{diag}(0.5+\kappa,0.5-\kappa).
$$

Here $\kappa$ is a **controller parameter**, not the plant parameter $\theta$. Compare $\kappa=0$ with $\kappa'=0.2$. The two closed-loop matrices are

$$
G_K=\operatorname{diag}(0.5,0.5),\qquad
G_{K'}=\operatorname{diag}(0.3,0.7),
$$

so both controllers are stable. Their cost matrices are

$$
P_2(K)=\operatorname{diag}(1.25,1.25),\qquad
P_2(K')=\operatorname{diag}(1.09,1.49).
$$

The old acceptable region is the unit disk; the new ellipse extends farther along one axis and less far along the other.

| Initial state | Old cost | New cost | Outcome |
| --- | ---: | ---: | --- |
| $(1,0)^\top$ | 1.25 | 1.09 | More acceptable margin |
| $(0,1)^\top$ | 1.25 | 1.49 | Becomes unacceptable |
| $(1.04,0)^\top$ | 1.352 | 1.178944 | Becomes acceptable |

Here $\Delta P=\operatorname{diag}(-0.16,0.24)$ and $b=1.2^2(0.24)=0.3456$. Thus $J_{2,K}\le0.9044$ is protected, giving the disk of radius $\sqrt{0.9044/1.25}\approx0.8506$.

![Old and updated acceptable regions, gains and losses, and the disk guaranteed to remain acceptable.](figures/linear-control-retraining.svg)

The blue boundary is old; the dashed purple boundary is updated. Green regions are gains, orange regions are losses, and the gray disk is protected by the bound. White areas between the disk and actual losses show conservatism. The dotted outer circle is $D$.

Both ellipses fit inside $D$. Their geometric areas are approximately $3.1416$ and $3.0814$, so the update loses net area despite improving some boundary conditions. Under a different reference measure placing equal mass on just $(1,0)^\top$ and $(0,1)^\top$, coverage instead falls from $1$ to $1/2$. Always specify the reference measure.

This illustrates interference from shared controller parameters; it is not a PPO experiment or a parameter-randomization experiment. The existing [figure source](figures/linear-control-retraining.py) reproduces the displayed costs and plot.

## 8. A small example connects plant variation to training

Now vary the plant itself:

$$
x_{t+1}=\theta x_t+u_t,\qquad u_t=-Kx_t,
\qquad \theta\in[0,2],\quad x_0=1.
$$

Take $H=2$, $Q=1$, $R=0$, and $h=1.25$. Then

$$
J_{2,K}(\theta,1)=1+(\theta-K)^2,
\qquad
S_K^J=\{\theta\in[0,2]:|\theta-K|\le0.5\}.
$$

For this example the initial state is fixed, so the cost region is a set of plant parameters. Updating $K=0.5$ to $K'=0.7$ moves it from $[0,1]$ to $[0.2,1.2]$. Under the uniform reference probability measure on $[0,2]$, gains and losses each have mass $0.1$, and coverage remains $0.5$. Improving performance on some plants does not necessarily improve total acceptable coverage.

There is also an explicit training interpretation. With a training distribution $q$ over $\theta$,

$$
F_q(K)=1+\mathbb E_q[(\theta-K)^2],\qquad
\frac{dF_q}{dK}=2(K-\mathbb E_q[\theta]).
$$

A gradient step of size $\eta>0$ gives

$$
K'=K-2\eta(K-\mathbb E_q[\theta]).
$$

Suppose $q$ puts equal mass at $\theta=0.9$ and $1.1$, near the old upper cost boundary. From $K=0.5$, a step with $\eta=0.2$ gives $K'=0.7$. The training objective drops from $1.26$ to $1.10$, even though reference coverage does not increase. This connects a targeted distribution, an actual optimization step, and gains and losses across plants.

The choices above are illustrative, not a newly selected experiment protocol. This deterministic cost boundary is also different from a Bernoulli reliability boundary. The example makes the research question concrete: which targeting rules and update restrictions produce useful improvements under the fixed reference measure?

## 9. What additional theory would connect targeting to expansion?

The direct bound compares given controllers. A stronger analysis should connect performance change to the update mechanism and explain gains as well as losses.

### Bound drift using update size

At one plant parameter, abbreviate $G=G_K(\theta)$, $G'=G_{K'}(\theta)$, and $\Delta K=K'-K$. Assume $\|G\|_2,\|G'\|_2\le M$ with $M\ge1$. A telescoping identity gives, for $t\ge1$,

$$
G'^t-G^t=\sum_{i=0}^{t-1}G'^{\,t-1-i}(G'-G)G^i,
\qquad
\|G'^t-G^t\|_2\le tM^{t-1}\|B(\theta)\|_2\|\Delta K\|_2.
$$

When $R\ne0$, also account for $\Delta W=W_{K'}-W_K$:

$$
\Delta W=K'^\top R\Delta K+\Delta K^\top RK,
\qquad
\|\Delta W\|_2\le\|R\|_2(\|K'\|_2+\|K\|_2)\|\Delta K\|_2.
$$

For $U=G'^t$ and $V=G^t$, split the cost-summand difference into

$$
U^\top W_{K'}U-V^\top W_KV
=(U-V)^\top W_KU+V^\top W_K(U-V)+U^\top\Delta WU.
$$

Applying the norm bound and summing yields

$$
\begin{aligned}
b(\theta)\le r^2\Bigg[
&2\|W_K\|_2\|B(\theta)\|_2\|\Delta K\|_2
  \sum_{t=1}^{H-1}tM^{2t-1}\\
&+\|\Delta W\|_2\sum_{t=0}^{H-1}M^{2t}\Bigg].
\end{aligned}
$$

For $R=0$, $\Delta W=0$ and $W_K=Q$, recovering the earlier state-only formula. A uniform family bound requires uniform matrix bounds. The expression exposes dependence on update size, horizon, and amplification, but can be loose. Limiting update size restricts possible damage without guaranteeing a favorable direction.

### Show enough gains to outweigh losses

Suppose a specified update reduces cost by at least $a>0$ on a set $T\subseteq\Theta\times D$. Then every condition in

$$
E=T\cap\{(\theta,x):h<J_{H,K}(\theta,x)\le h+a\}
$$

becomes acceptable. With the loss band from Section 6 and a fixed reference probability measure $\mu$ on this domain,

$$
\mu(S_{K'}^J)-\mu(S_K^J)
\ge\mu(E)-\mu(B_{\rm loss}).
$$

Thus $\mu(E)>\mu(B_{\rm loss})$ suffices for **net coverage expansion**. It does not imply set inclusion: some old acceptable conditions can still be lost. The research challenge is to connect useful values of $a$, $b(\theta)$, and $T$ to an actual training distribution and update rule. Merely assuming those properties does not explain why boundary-guided training succeeds.

A stronger condition, $P_H(K',\theta)\preceq P_H(K,\theta)$ for every plant, guarantees that cost never increases, hence $S_K^J\subseteq S_{K'}^J$. It does not alone ensure strict expansion. Neither the ellipse example nor arbitrary boundary sampling satisfies this condition automatically.

These calculations explain why counterexamples and update analysis belong to the **main research**. A proved useful preservation or expansion condition for a restricted mechanism is the **ambitious outcome**.

## 10. Connect the linear example to the CartPole evaluator

The CartPole study uses

$$
p_k(z)=\Pr(\text{recovery fails under frozen }\pi_k\mid z),
\qquad S_k=\{z:p_k(z)\le\alpha\}.
$$

A condition includes plant parameters and an initial-state distribution. Recovery requires survival for $500$ steps with final-100-step RMS angle at most $5^\circ$, using the relaxed $90^\circ$ angle termination cutoff and retaining other termination rules. This differs from the linear example in several ways:

| Linear foundation | CartPole research |
| --- | --- |
| Known family $A(\theta),B(\theta)$ and continuous linear feedback | Nonlinear dynamics and discrete-action learned policies |
| Exact cost at each $(\theta,x)$ | Conditional failure probability at $z=(\theta,s)$ |
| Quadratic cost threshold $h$ | Recovery-failure tolerance $\alpha$ |
| Matrix calculations give exact regions | Fresh rollouts give estimates and certificates |
| Shared gain learned under a specified distribution | PPO branches trained under matched condition-selection rules |

In outer round $k$, the evaluator uses time-uniform bounds for each grid condition. An upper bound at most $\alpha$ certifies that condition acceptable. Fresh episode outcomes and statistical error allocation across conditions, methods, and rounds support a campaign-wide error budget $\delta$. The guarantee concerns the declared grid and conditional episode law; GP predictions between points do not become direct certificates.

The evaluator is assessed by **certified acceptable coverage at a fixed rollout budget**. Controller improvement is assessed under the fixed reference measure, with gains, regressions, and uncertainty reported. A larger certified set can reflect more evaluation data rather than a better policy, so differences between certificates alone do not identify true gains and losses.

For training, compare

$$
q_k=(1-\lambda)q_{\rm base}+\lambda q_{{\rm target},k}
$$

with uniform targets, targets near $\widehat p_k=\alpha$, and intermediate-difficulty targets scored by $\widehat p_k(1-\widehat p_k)$. The latter peaks at $1/2$, which need not be the acceptance threshold. An evaluation acquisition rule and a training-condition score serve different purposes: useful evidence for certification need not be useful training data.

PPO branches share the starting checkpoint, background mixture, objective, optimizer, and budgets. Collect fresh on-policy training trajectories, then freeze the updated policy and evaluate with fresh data. Evaluation trajectories of the old controller are not outcomes of the updated controller. A surrogate may carry information forward to propose conditions, but old outcomes cannot directly certify the new policy.

### Why a cost bound does not automatically bound failure probability

For a score-defined failure event $\{C>h\}$, suppose a specified common-randomness coupling gives the pathwise bound $|C'-C|\le\varepsilon$. Then

$$
|\Pr(C'>h)-\Pr(C>h)|\le\Pr(|C-h|\le\varepsilon).
$$

An outcome can change classification only near the old threshold. To make the probability change small, one needs control of the mass near that threshold. The deterministic linear proof does not establish these assumptions for CartPole recovery, which also includes termination. Likewise, a small neural-network parameter update does not automatically bound closed-loop trajectory change.

The extension to $A(\theta),B(\theta)$ addresses plant variation **within the linear family**. It does not justify a linear approximation across large CartPole angles or quantized actions. Continuous-cost and failure-probability level sets remain candidate targets for the ambitious theorem, without a newly selected priority.

### Why finite-horizon acceptability differs from stability

At a fixed plant, if $G_K(\theta)$ is Schur stable, the infinite sum

$$
P_\infty=\sum_{t=0}^{\infty}(G_K(\theta)^t)^\top W_KG_K(\theta)^t
$$

satisfies $G_K(\theta)^\top P_\infty G_K(\theta)-P_\infty=-W_K$. With $W_K\succ0$, this gives a positive-definite quadratic function decreasing along nonzero trajectories. In contrast, the finite sum obeys

$$
G_K(\theta)^\top P_HG_K(\theta)-P_H
=(G_K(\theta)^H)^\top W_KG_K(\theta)^H-W_K,
$$

which need not be negative semidefinite. Finite-horizon cost sublevel sets therefore do not automatically certify stability or invariance. The infinite-horizon connection is background, not an added project requirement.

## 11. Reading path, deliverables, and remaining choices

The textbook supplies the following background. Page references use printed pages; add 19 for this local PDF's viewer page number.

| Read | Printed pages / PDF pages | What it supplies |
| --- | --- | --- |
| §1.1.2 and §6.5 | 6 / 25; 69 / 88 | Discrete-time models and matrix-power trajectories |
| §8.2 and §8.4 | 88–91 / 107–110 | Norm inequalities and quadratic forms |
| §8.6 | 95–97 / 114–116 | Stability and the discrete-time Lyapunov equation |
| §10.1 and §10.4, as context | 120 / 139; 122–123 / 141–142 | Quadratic performance and feedback design |
| §7.2, optionally | 78–79 / 97–98 | Transient amplification in matrix powers |

The revised proposal's closest controller-learning references are:

- [Fujinami et al., *Policy Gradient for LQR with Domain Randomization* (2025)](https://arxiv.org/abs/2503.24371): analyzes policy-gradient optimization of domain-randomized LQR average cost under system-heterogeneity assumptions. It motivates learning one controller across a family. Its convergence guarantee is not automatically a guarantee for our finite-horizon objective, changing training distributions, or acceptable-region expansion.
- [Fujinami et al., *Policy Gradient over History-Dependent Policy Classes for LQR with Domain Randomization* (2026)](https://arxiv.org/abs/2609.16300): extends the analysis to controllers with memory and a curriculum that expands controller memory. This concerns controller expressiveness; our proposed adaptation changes the training-condition distribution. Implementing history-dependent controllers is not required.

For evaluation and improvement, the [proposal bibliography](../latex/references.bib) includes Gotovos and Letham on level-set estimation and Howard on confidence sequences. The local [Florensa paper](../literature/florensa17a.pdf) explains performance-guided start-state curricula, the [Rutherford paper](../literature/2408.15099v3.pdf) motivates the mixed-success score, and the [Berkenkamp paper](../literature/1705.08551v3.pdf) illustrates the additional structure needed for Lyapunov-based safe-region arguments. These are complementary connections, not interchangeable guarantees.

A concrete foundation deliverable is a small shared-controller learning example, the direct preservation proof, and a statistical evaluation procedure. The main research then studies targeted training through analytical tradeoffs and matched CartPole experiments. A positive expansion theorem is an ambitious outcome; a well-supported negative result still contributes to the main study.

The exact linear family, weights, horizon, optimizer, budgets, seeds, reward design, and mixture weight remain experiment-design choices. The two GP acquisition designs remain candidates, not a requirement to implement both. Failure tolerance $\alpha$ and error budget $\delta$ remain symbolic. Whether repeated outer rounds are mandatory, and which level-set notion the ambitious theorem targets, remain explicitly open. The illustrative numbers above do not settle those choices.
