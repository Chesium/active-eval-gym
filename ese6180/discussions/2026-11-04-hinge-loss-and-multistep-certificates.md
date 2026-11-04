# Uncapped hinge loss and longer-stride certificate learning

Date: 4 November 2026.

Status: discussion and proposed extensions. The report and numerical support
currently use the clipped, one-step loss. This note records alternatives and
their proof obligations; it does not change those definitions or implementations.
The derivations below concern the existing exact, finite, fixed-plant linear
setting. They are not claims about nonlinear CartPole or PPO.

Relevant repository sources:

- [Problem formulation](../Evaluation_Guided_Domain_Randomization/Report_Sections/02_problem_formulation.tex),
  especially `def:foundation_candidates`, `def:foundation_risk`,
  `def:foundation_fitting`, `thm:foundation_generalization`,
  `thm:foundation_verification`, and `lem:stage2_margin_drift`.
- [Current foundation proofs](../Evaluation_Guided_Domain_Randomization/Report_Sections/05_theoretical_results.tex).
- [Proposal design notes](../proposal/discussions/proposal-design-notes.md).
- [Certificate notebook](../../notebooks/certificate_generalization.ipynb)
  and [support module](../../src/active_eval_gym/certificates.py).
- [Linear controller-update study](../../docs/linear-update-study.md), including
  its fixed/refitted certificate comparisons and target-map experiments.

## 1. Two distinct changes

The current experiment fixes a gain $K$, draws $N$ independent trajectories with
$H$ transitions, and holds the plant parameter $\theta$ fixed within each
trajectory. Its closed loop and candidate class are

$$
x_{t+1}=G_K(\theta)x_t,\qquad G_K(\theta)=A(\theta)-B(\theta)K,
\qquad \mathcal P=\{P=P^\top:aI\preceq P\preceq bI\}.
$$

For nonzero $x$, the one-step residual and trajectory loss are

$$
r_P(x,x^+)=\frac{x^{+\top}Px^+-x^\top Px}{\|x\|^2}+\eta,
\qquad s_P(\tau)=\max_{t:x_t\ne0}r_P(x_t,x_{t+1}),
$$

$$
\ell_P^{\rm clip}(\tau)
=\min\{1,\max\{0,1+s_P(\tau)/\gamma\}\}.
$$

The proposed changes answer different questions:

| Change | Purpose | Main consequence |
| --- | --- | --- |
| Remove the upper loss cap | Penalize violation severity and obtain a convex fitting objective | The loss needs a uniform range bound in the concentration proof |
| Compare states several steps apart | Permit temporary growth while requiring net decrease over a block | The certificate tests a different decrease property; intermediate invariance needs reconsideration |

They can be studied separately or combined. Neither by itself proves that a
boundary-targeted training law improves acceptance coverage.

## 2. Replace the clipped ramp by an uncapped hinge

Define

$$
\ell_P^{\rm hinge}(\tau)=\max\{0,1+s_P(\tau)/\gamma\}.
$$

For an all-zero trajectory, keep loss zero and $s_P=-\infty$. The two losses
agree when $s_P\le0$. For $s_P>0$, the clipped loss is one, whereas the hinge
continues increasing. Thus the hinge distinguishes small and large violations.
Both assign zero loss exactly when every evaluated residual is at most
$-\gamma$.

### 2.1 The hinge is bounded on the declared experiment

It is unbounded as a function of an arbitrary scalar residual, but the actual
residuals have a uniform upper bound. Let

$$
M_K=\max_\theta\|G_K(\theta)\|_2.
$$

For every admissible $P$ and nonzero $x$,

$$
\begin{aligned}
r_P(x,G_K(\theta)x)
&\le\frac{b\|G_K(\theta)x\|^2-a\|x\|^2}{\|x\|^2}+\eta\\
&\le bM_K^2-a+\eta.
\end{aligned}
$$

A convenient bound, deliberately at least one, is

$$
0\le\ell_P^{\rm hinge}(\tau)\le B_\ell,
\qquad B_\ell=\max\{1,1+(bM_K^2-a+\eta)/\gamma\}.
$$

This uses the dynamics bound, bounded certificate class, and state
normalization together. A bound on $M_K$ alone would not bound the loss over an
unrestricted class of matrices $P$. No condition $M_K<1$ is needed. Taking the
maximum over a trajectory preserves the bound, so it does not grow with $H$.

### 2.2 Changes to the statistical argument

Indicator domination remains valid:

$$
\mathbf1\{s_P(\tau)>0\}\le\ell_P^{\rm clip}(\tau)
\le\ell_P^{\rm hinge}(\tau).
$$

The loss Lipschitz constant also remains

$$
L_{\rm par}=(M_K^2+1)/\gamma.
$$

The maximum and scalar hinge are Lipschitz operations, and the Frobenius cover
of $\mathcal P$ is unchanged. Hoeffding's range constant changes from one to
$B_\ell$. With $d=n(n+1)/2$, the corresponding two-sided uniform penalty is

$$
\Delta_N^{\rm hinge}(\varepsilon,\delta)
=2L_{\rm par}\varepsilon
+B_\ell\sqrt{
\frac{d\log(1+2b\sqrt n/\varepsilon)+\log(2/\delta)}{2N}}.
$$

Consequently, on the uniform event,

$$
e_{K,H}(\widehat P)
\le\min\{1,\widehat L^{\rm hinge}_{N,K}(\widehat P)
+\Delta_N^{\rm hinge}\}.
$$

The population near-optimality statement has the same form with hinge losses,
$2\Delta_N^{\rm hinge}$, and the optimization tolerance $\xi_{\rm opt}$.
That statement still requires the approximate empirical-minimization assumption.
The risk bound itself applies to any selected candidate in the class.

The rate in $N$ has the same order for fixed problem constants, but the explicit
constant can be much larger. In particular, small $\gamma$ can enlarge both
$L_{\rm par}$ and $B_\ell$. Dividing the hinge by $B_\ell$ puts it in $[0,1]$
but also introduces $B_\ell$ into indicator domination; rescaling alone does
not remove this statistical cost.

### 2.3 A concrete convex fitting algorithm

For fixed observations, each residual is affine in $P$. Its maximum over
transitions is convex, and taking the maximum with zero preserves convexity.
Therefore empirical hinge minimization over $\mathcal P$ is convex. The upper
clipping in the current objective generally destroys that convexity.

Introduce a scalar $t_i$ for each trajectory and solve the semidefinite program

$$
\begin{aligned}
\min_{P,t}\quad &\frac1N\sum_{i=1}^N t_i\\
\text{subject to}\quad
&aI\preceq P\preceq bI,\qquad t_i\ge0,\\
&t_i\ge1+\frac{1}{\gamma}
\left(\frac{x_{t+1,i}^\top Px_{t+1,i}-x_{t,i}^\top Px_{t,i}}
{\|x_{t,i}\|^2}+\eta\right)
\quad\text{for every nonzero sampled }x_{t,i}.
\end{aligned}
$$

All transition constraints are affine in $P,t_i$. An all-zero trajectory has
only $t_i\ge0$ and contributes zero at the optimum. Numerical solver tolerances
and deterministic tie-breaking need to be declared. Convexity provides a route
to globally solving this fitting problem; it does not guarantee a zero-loss
solution or successful all-state verification.

### 2.4 An alternative: fit with hinge, evaluate with the clipped loss

The current clipped-loss generalization theorem is uniform over all $P$ in the
class. It can therefore be applied to a candidate obtained by hinge fitting,
using its measured clipped empirical loss and the original clipped-loss
penalty. No extra holdout split is required for that existing uniform argument,
provided $K$, the class, and the sampling law satisfy its assumptions.

This separates the optimization objective from the statistical evaluation
surrogate. It avoids putting $B_\ell$ into the clipped-loss concentration term.
However, a hinge minimizer need not minimize clipped loss, so the current
clipped-loss ERM population comparison cannot be asserted without an additional
optimization-error argument. Zero hinge loss does imply zero clipped loss.

Thus there are two coherent choices:

1. Replace both fitting and generalization losses with the hinge, updating the
   range constant and theorem statements.
2. Use hinge fitting and retain clipped-loss risk evaluation, explicitly
   distinguishing the objectives and revising Definition 4 accordingly.

### 2.5 Consequences for the notebook and Stage 2

The notebook's stress fixture includes $G_3=1.02I$ with probability $0.10$.
Under clipping, every positive-definite candidate incurs loss exactly one on
that plant. The known certificate has zero loss on the other plants, yielding
the stated $0.10$ population-loss and violation-risk floors.

For the hinge, the unsafe-plant residual is

$$
r_P(x,G_3x)=(1.02^2-1)\frac{x^\top Px}{\|x\|^2}+\eta>0.
$$

Its hinge loss exceeds one and depends on $P$. The unavoidable violation-risk
floor remains $0.10$, but the hinge-loss floor is different. The known
certificate's empirical optimality must be re-established rather than inherited
from the clipped experiment. Different certificate shapes/scales can now trade
off stable-plant slack against unsafe-plant violation severity.

If only the fitting loss changes, Stage 2's existing matrix preservation and
finite-horizon cost arguments still apply to any resulting verified $P$.
The selected $P$ can nevertheless change the accepted controller step through
both its verified margins and margin-consumption bounds. Lower hinge loss does
not automatically mean greater update capacity or better cost coverage.

## 3. Replace one-step decrease by decrease over stride $k$

Use $k$ for the stride to avoid conflicting with the input dimension $m$.
Fix $1\le k\le H$ and a desired block-decrease coefficient $\eta^{(k)}>0$.
For a nonzero starting state, define

$$
r_P^{(k)}(x_t,x_{t+k})
=\frac{x_{t+k}^\top Px_{t+k}-x_t^\top Px_t}{\|x_t\|^2}
+\eta^{(k)},
$$

$$
s_P^{(k)}(\tau)
=\max_{\substack{0\le t\le H-k\\x_t\ne0}}
r_P^{(k)}(x_t,x_{t+k}).
$$

Use $s_P^{(k)}=-\infty$ and loss zero if there is no nonzero window start.
Either the clipped ramp or hinge can be applied to this maximum, with fitting
margin $\gamma^{(k)}>0$.

This definition uses every overlapping window of length $k$. Evaluating only
$t=0,k,2k,\ldots$ is another possible experiment, with a different loss/risk;
it should not be silently substituted. In either case the independent sample
count is $N$ trajectories, not the number of windows.

The physical simulator still advances one step at a time and applies feedback
at each step. Stride changes the certificate's comparison interval, not the
controller's action frequency.

### 3.1 Verification uses a matrix power

Because the plant is fixed within an episode,

$$
F_{K,k}(\theta):=G_K(\theta)^k,\qquad x_{t+k}=F_{K,k}(\theta)x_t.
$$

The verified block margin is

$$
\eta_\theta^{(k)}(P)
=\lambda_{\min}\left(P-F_{K,k}(\theta)^\top P F_{K,k}(\theta)\right).
$$

The desired block inequality holds at every state exactly when
$\eta_\theta^{(k)}(P)\ge\eta^{(k)}$. Strict positivity alone guarantees
some block decrease, which is weaker than meeting that prescribed target.

A positive block margin permits temporary increases of $V_P$ inside the block.
It is a different certificate criterion, not evidence that the original
one-step residual is nonpositive.

### 3.2 A stable family with no common one-step certificate

Consider the two exact closed-loop matrices

$$
G_1=\begin{pmatrix}0&2\\1/4&0\end{pmatrix},\qquad
G_2=\begin{pmatrix}0&1/4\\2&0\end{pmatrix}.
$$

They can be realized in the report's model with $K=I$, $B_1=B_2=I$, and
$A_i=G_i+I$. Their eigenvalues are $\pm1/\sqrt2$, and
$G_1^2=G_2^2=I/2$.

There is no common strictly decreasing one-step quadratic certificate, even
without spectral bounds on $P$. Write

$$
P=\begin{pmatrix}p_{11}&p_{12}\\p_{12}&p_{22}\end{pmatrix}\succ0.
$$

Since $G_1e_2=2e_1$, strict decrease requires $p_{22}>4p_{11}$.
Since $G_2e_1=2e_2$, it also requires $p_{11}>4p_{22}$. These inequalities
are incompatible.

At stride two, $P=I$ gives

$$
P-(G_i^2)^\top PG_i^2=\tfrac34I.
$$

It belongs to the current default class $\tfrac12I\preceq P\preceq2I$.
The desired coefficient and fitting margin must still satisfy
$\eta^{(2)}+\gamma^{(2)}\le3/4$ for zero hinge/clipped fitting loss with
this certificate.

The first four strides illustrate that feasibility is not monotone in small
integer strides:

| Stride | Common strictly decreasing quadratic certificate |
| --- | --- |
| 1 | None |
| 2 | $P=I$, margin $3/4$ |
| 3 | None |
| 4 | $P=I$, margin $15/16$ |

For stride three, $G_i^3=G_i/2$. The coordinate tests require both
$p_{22}>p_{11}$ and $p_{11}>p_{22}$. Larger stride does not therefore always
help at each successive integer. A certificate for stride $k$ does imply
strict decrease over positive integer multiples of $k$, by iterating that
block contraction.

### 3.3 What stability and invariance mean now

Let $\eta_*^{(k)}>0$ be the smallest verified block margin on the finite
retention family and set

$$
q_k=1-\eta_*^{(k)}/b\in[0,1),\qquad
C_k=\max_{\theta\in\Theta_{\rm ret}}\max_{0\le j<k}
\|G_K(\theta)^j\|_2^2.
$$

For $t=\ell k+j$, $0\le j<k$, block contraction and bounded intermediate
powers yield

$$
\|x_t\|^2\le C_k\frac ba q_k^\ell\|x_0\|^2.
$$

When $q_k=0$, the state reaches zero after a block; the initial block is
bounded separately. Thus verified block decrease gives exponential stability
for each fixed plant, including intermediate times. This conclusion requires
the all-state matrix test, not merely small sampled block-violation risk.

However, the original ellipsoid $\{x:x^\top Px\le c\}$ is guaranteed invariant
only at block times $0,k,2k,\ldots$. Intermediate states may leave it.
Claims of one-step invariance or constraint satisfaction need an additional
argument. In the example, $\|G_i\|_2=2$ despite two-step contraction.

For each individual plant, another way to see the connection to one-step
stability is to construct

$$
\widetilde P_\theta=\sum_{j=0}^{k-1}(G_K(\theta)^j)^\top P G_K(\theta)^j.
$$

Then

$$
G_K(\theta)^\top\widetilde P_\theta G_K(\theta)-\widetilde P_\theta
=(G_K(\theta)^k)^\top P G_K(\theta)^k-P\prec0.
$$

This new one-step certificate generally depends on the plant and need not
belong to the original bounded class. It does not contradict nonexistence of
a *shared* one-step $P$.

For a single Schur-stable matrix, an unrestricted one-step quadratic
certificate already exists. For any finite family of individually Schur-stable
matrices, all sufficiently large powers have norm less than one, so a common
block certificate $P=I$ eventually exists. If the class excludes $I$, a permitted
positive scalar multiple of $I$ gives strict decrease too. Meeting prescribed
numerical margins remains a separate requirement.

Fixed plants within blocks are essential. Testing only $G_\theta^k$ does not
test arbitrary mixed products of different plants. In the example,
$G_2G_1=\operatorname{diag}(1/16,4)$, so alternating the plants can be unstable.
A common one-step certificate would have ruled out this switching behavior;
the proposed block certificate does not.

### 3.4 Generalization constants and comparisons across strides

Define the exact uniform block bound

$$
M_{K,k}=\max_\theta\|G_K(\theta)^k\|_2\le M_K^k.
$$

The loss regularity constant is now

$$
L_{\rm par}^{(k)}=(M_{K,k}^2+1)/\gamma^{(k)}.
$$

For the hinge, use

$$
B_{\ell,k}=\max\{1,1+(bM_{K,k}^2-a+\eta^{(k)})/\gamma^{(k)}\}.
$$

The same finite-cover argument applies with these constants. In the example,
$M_{K,1}=2$ but $M_{K,2}=1/2$; replacing the latter by $M_K^2=4$ would be very
conservative. These constants are known-model bounds, not unqualified maxima
estimated from the sampled transitions.

The violation event, feasible window count, and desired decrease property all
change with $k$. A smaller reported risk at another stride is not a like-for-like
improvement in the original criterion. Total block coefficients should not be
interpreted as one-step rates. For explicit rate comparisons, a possible
alternative target is

$$
(G_K(\theta)^k)^\top P G_K(\theta)^k\preceq q^kP,
\qquad 0<q<1,
$$

which declares a common per-time multiplicative target at block endpoints.
It is a separate formulation from fixing $\eta^{(k)}$.

If a stride is selected after examining data, a theorem for one fixed stride
does not automatically cover the selection. For a finite predeclared set of
strides, one simple solution is a union bound with allocated failure budgets.
Fresh evaluation is another option. Dividing a residual by $k$ also requires
corresponding margin rescaling; it should not silently redefine the loss.

### 3.5 Adapting Stage 2's certificate guard

The existing one-step guard cannot simply be reused after switching to block
verification. Put $G=G_K(\theta)$, $E=-B(\theta)\Delta K$, and

$$
F=G^k,\qquad E_k=(G+E)^k-G^k.
$$

The same quadratic expansion at the block level gives

$$
c_{\theta,k}(\Delta K)
=2\|F^\top P\|_2\|E_k\|_2+\|P\|_2\|E_k\|_2^2.
$$

Requiring

$$
c_{\theta,k}(\Delta K)\le(1-\rho)\eta_\theta^{(k)}(P)
$$

preserves at least $\rho\eta_\theta^{(k)}(P)$ of the block margin. A
noncommutative telescoping identity supplies an explicit bound:

$$
E_k=\sum_{j=0}^{k-1}(G+E)^{k-1-j}E G^j.
$$

For a declared neighborhood $\|\Delta K\|_2\le\kappa$, take
$\overline M_\theta=\max\{1,\|G\|_2+\|B(\theta)\|_2\kappa\}$. Then

$$
\|E_k\|_2\le k\overline M_\theta^{k-1}\|B(\theta)\Delta K\|_2.
$$

This supplies a sufficient guard, possibly conservative. For fixed finite $k$,
its consumption tends to zero with $\Delta K$, supporting the same small-step
existence route with strict old block margins and smooth cost descent.

The finite-horizon LQ objective can still sum every original time step. Its
existing cost-sensitivity and threshold-band arguments concern that cost and
do not require redefining the horizon in blocks. The updated stability statement
must nevertheless replace one-step ellipsoid invariance by block-time
invariance plus an intermediate-state bound.

## 4. Effect on the project narrative

The central progression remains: fit a certificate for a frozen controller,
distinguish statistical risk from exact verification, and use a verified margin
to restrict one controller update. The two proposals sharpen different parts
of that story.

| Existing component | Hinge only | Longer stride |
| --- | --- | --- |
| Trajectory loss/risk definition | Loss changes; violation event stays the same | Residual and violation event change |
| Certificate fitting | Convex SDP becomes available | For fixed $k$, hinge fitting remains a convex SDP using $x_{t+k}$ |
| Uniform generalization proof | Change range constant, or retain clipped evaluation explicitly | Replace block dynamics constants and account for stride selection |
| Exact verification | Same one-step eigenvalue test | Test powers of each fixed-plant closed loop |
| Controller-update guard | Same formula, potentially different fitted $P$ | Bound perturbations of the matrix powers |
| Stability/invariance | Existing conclusion for a verified $P$ | Stability survives; one-step invariance requires revision |
| Cost and coverage | Same cost definitions and regression logic | Same if costs still use all original transitions |
| Boundary-targeting benefit | Still unproved | Still unproved |

The hinge strengthens the computational connection between Stage 1 and Stage 2:
it gives an explicit fitting algorithm whose output can be carried into the
update. Longer stride explores how the certificate's temporal criterion changes
feasibility and conservatism. Its success must not be described as an improvement
in the controller when $K$ has not changed.

Boffi et al. remain the reference for trajectory-based certificate learning,
fitting margins, and generalization. The exact normalized hinge, discrete block
residuals, and controller-update calculations here are proposed project
specializations, not reproductions of that paper's rates or an asserted novel
certificate framework. Existing source attribution is recorded in the linked
design notes and report.

Neither extension resolves the main Stage 3 question about how target maps
affect actual gains and losses of acceptable conditions. They also do not
transfer the linear conclusions to quantized CartPole/PPO without additional
dynamics, policy-update, and statistical arguments.

## 5. Numerical exploration and revision checklist

A compact experiment can isolate the two changes:

1. On the existing fixed-matrix fixture, compare grid clipped ERM and convex
   hinge fitting on the same sampled trajectories. Report both losses, violation
   risk, exact margins, and numerical optimization tolerances.
2. Carry each fitted certificate into the existing controller-update study.
   Hold target maps and proposed directions fixed for paired comparisons;
   measure accepted step sizes, margins, objective changes, and coverage.
3. Use the explicit two-plant family above with predeclared strides 1, 2, 3, 4.
   Separate sampled fitting from exact matrix verification. Plot energy and
   state norm at every original time step to expose intermediate growth.
4. If combining extensions, compare hinge fits at each stride with its own
   stated target and constants. Report block risk and original one-step risk
   as distinct quantities. Keep all failure cases and declared seeds.

The analytic matrix example proves the common-certificate feasibility claims;
a grid search or Monte Carlo run alone does not prove infeasibility over all
$P$. Conversely, a fitted low-loss candidate still needs exact verification
before being used by the certified update branch.

Before adopting either change in the report or code:

- Decide whether the hinge is both the fitting/evaluation loss or only the
  fitting objective, and revise Definition 4 and ERM claims consistently.
- Update proofs and constants wherever $0\le\ell\le1$ or clipping is used.
- Recompute notebook loss floors and check known-certificate optimality.
- Declare stride, window starts, block margins, and any selection correction.
- Revise stability, invariance, and Stage 2 guards for block certificates.
- Keep the physical horizon, LQ costs, reference measure, and acceptance
  threshold explicit so certificate changes are not confused with coverage gains.
- Version derived metrics by loss and stride configuration; retain original
  raw trajectories and previous results. The same raw trajectories can support
  these new metrics without rerunning the original frozen experiment.

The least disruptive next step is a separate numerical hinge-fitting arm and
the explicit stride example, with the current report treated as the baseline.
Whether to promote either alternative into the main formulation should follow
from its explanatory value and the cost of maintaining the revised guarantees.
