# Supporting notes: certificate learning and one evaluation-guided update

These notes support the [v3 brief](../brief.md). They contain proof sketches,
assumptions, and retained evaluation infrastructure. The [LaTeX](../latex/proposal.tex)
now summarizes v3 in two pages plus references. See the [revision record](v3-revisin-notes.md) for scope changes.
No paper reproduction or new experiment is claimed complete by these notes.

## Current scope and proof obligations

**Course foundation → Main theoretical investigation → Ambitious extension**

- **Required foundation:** adapt trajectory-based certificate learning to a
  frozen discrete-time linear family and a bounded quadratic class; explain
  the distinction between generalization and all-state verification.
- **Required main analysis:** connect a verified decrease margin to one actual
  gradient update with backtracking, derive a cost-regression bound, and show
  why targeted improvement does not imply coverage expansion.
- **Optional:** cost-set expansion under an actual rule, or a local nonlinear
  extension. Multiple PPO rounds, learned barriers, GP acquisition comparisons,
  and neural certificate classes are not minimum deliverables.

The default anchor is Boffi et al., *Learning Stability Certificates from Data*
([publisher](https://proceedings.mlr.press/v155/boffi21a.html),
[local PDF](../literature/boffi21a-learning-stability-certificates.pdf)), Sections
3–4. Its main analysis is continuous time and already discusses quadratic
classes. The discrete-time proof below is an elementary adaptation of the
learning viewpoint, not a new certificate-learning framework or a claim to its
faster rates. Chapter 3 supplies the Lyapunov argument; Chapter 5 supplies the
finite-cover uniform-convergence tools. Paper choice remains subject to feedback.

## 1. Assumptions and distinct guarantees

Let $\Theta$ be a fixed finite plant set and let
$x^+=G_K(\theta)x$, $G_K(\theta)=A(\theta)-B(\theta)K$. Each episode holds
$\theta$ fixed. States and actions are continuous, transitions are noiseless,
and there are no constraints or termination rules unless separately introduced.
Matrices are known for analytical verification, while certificate fitting uses
sampled transitions. This is not a system-identification claim.

Use one shared $K$ and initially one shared certificate

$$
\mathcal P=\{P=P^\top:aI\preceq P\preceq bI\},\qquad 0<a<b<\infty.
$$

A common quadratic certificate is a restriction, not a consequence of every
plant being individually stable or even of a shared stabilizing gain existing.
For the first numerical example choose a family admitting such a certificate.
For other examples, report infeasibility or failed verification rather than
changing the retained plants based on favorable results.

| Statement | What supports it | What it does not imply |
| --- | --- | --- |
| Small certificate-violation risk under $D_0$ over $H$ steps | Uniform statistical bound for the fitted $P$ | Every state is certified; infinite-horizon stability |
| Decrease at every state for a retained plant | Exact matrix inequality | CartPole recovery or a chosen cost threshold |
| Preservation of $V_P$ under an update | Old verified margin plus perturbation bound | Cost-region expansion |
| Preservation of an interior cost set | Bound on $P_H(K',\theta)-P_H(K,\theta)$ | Asymptotic stability |
| Grid condition has recovery failure probability at most $\alpha$ | Direct rollouts and valid upper confidence bound | A Lyapunov or barrier certificate |

The notation $P$ always means a certificate matrix; $P_H$ means the cost matrix.
A Lyapunov sublevel set lives in state space at a given plant, whereas $S_K^J$
and $S_k^p$ also index plants or reset laws. Do not identify their domains.

## 2. A concrete discrete-time certificate-generalization proof

This construction gives a modest, reviewable foundation with a complete proof
route. It deliberately uses a bounded loss and a finite cover instead of
claiming the sharper generalization rate in Boffi et al.

Freeze $K$ before sampling. Draw $N$ independent pairs $(\theta_i,x_{0,i})$
from a fixed distribution $D_0$ and observe $H$ transitions of each. Dependence
within a trajectory is permitted. The sample size is $N$, not $NH$. Let
$M=\max_{\theta\in\Theta}\|G_K(\theta)\|_2<\infty$. This is a known bound,
not a maximum estimated from the same sample without justification.

Fix a desired decrease coefficient $\eta>0$ and a learning margin $\gamma>0$.
For a nonzero visited state define

$$
r_P(x,x^+)=\frac{x^{+\top}Px^+-x^\top Px}{\|x\|_2^2}+\eta.
$$

This is a normalized constraint residual. Start with the desired decrease
$V_P(x^+)-V_P(x)\le-\eta\|x\|_2^2$, move everything to the left, and
divide by $\|x\|_2^2>0$. Consequently,

$$
r_P(x,x^+)\le0
\quad\Longleftrightarrow\quad
V_P(x^+)-V_P(x)\le-\eta\|x\|_2^2.
$$

Negative residual means slack, zero means the condition is met exactly, and
positive residual means violation. Normalization preserves this sign test and,
for linear dynamics and quadratic $V_P$, removes dependence on state magnitude:
$r_P(cx,cx^+)=r_P(x,x^+)$ for $c\ne0$. It also makes the stricter learning
condition $r_P\le-\gamma$ equivalent to
$V_P(x^+)-V_P(x)\le-(\eta+\gamma)\|x\|_2^2$. Without normalization, a
fixed absolute residual margin would demand a nonvanishing extra decrease near
zero; the original desired inequality itself already vanishes there. At $x=0$,
linear homogeneous dynamics give $x^+=0$ and the inequality holds automatically;
skip this state. A trajectory with no nonzero states has loss and violation
indicator zero. No measurement noise or numerical division claim is hidden here;
finite-precision handling near zero must be specified before implementation.

For all other trajectories $\tau$, set

$$
r_P^{\max}(\tau)=\max_{0\le t<H:x_t\ne0}r_P(x_t,x_{t+1}),\qquad
\ell_P(\tau)=\min\{1,\max\{0,1+r_P^{\max}(\tau)/\gamma\}\}.
$$

Then $\ell_P\in[0,1]$, it is zero if every residual is at most $-\gamma$,
and $\mathbf1\{r_P^{\max}>0\}\le\ell_P$. Equality at zero residual can incur
loss even though the non-strict desired inequality holds; this is conservative.
The empirical loss can be minimized over $\mathcal P$ without assuming a
zero-loss feasible solution exists.

To read the loss, write $s=r_P^{\max}(\tau)$ for the worst step's residual:

$$
\ell_P(\tau)=
\begin{cases}
0,&s\le-\gamma,\\
1+s/\gamma,&-\gamma<s<0,\\
1,&s\ge0.
\end{cases}
$$

Thus $\eta$ sets the desired decrease, while $\gamma$ asks for additional
slack when fitting the certificate. For example, with $\gamma=0.1$, worst
residuals $-0.2,-0.05,0.02$ give losses $0,0.5,1$. A positive loss need not
mean violation: a satisfied condition with insufficient slack also incurs
loss. Taking the maximum tests whether **any** visited step violates the
condition; an average residual could hide a violating step among negative ones.
The clipped ramp upper-bounds the violation indicator, remains in $[0,1]$ for
Hoeffding, and is $1/\gamma$-Lipschitz in $s$ for the finite-cover proof.
It is continuous but has corners, and clipping saturates the loss on violations;
the proof alone does not establish that a particular optimizer will fit it well.

**Sources and adaptation.** The exact normalized residual and clipped ramp
above are choices for this proposal, not definitions copied from either source.

| Ingredient | Reading and relationship to this construction |
| --- | --- |
| Quadratic one-step decrease | [Chapter 3](../../resources/ch3%20Dynamics,%20Stability,%20and%20Basic%20Lyapunov%20Theory-1.pdf), Theorem 3.10, Lemma 3.11, and Example 3.12 (PDF pp. 4–5), supply the underlying stability condition. Rearranging and normalizing it gives our $r_P$. |
| Certificate residual, negative fitting margin, and worst-step violation event | [Boffi et al.](https://proceedings.mlr.press/v155/boffi21a/boffi21a.pdf), Section 3.1, equations (3.1)–(3.3), and Section 4 (PDF pp. 3–5), supply this learning viewpoint. Their main residual is continuous-time; Section 4 uses a margin indicator, rather than our exact clipped ramp. |
| Surrogate loss | [Chapter 5](../../resources/ch5%20Empirical%20Risk%20Minimization%20for%20Nonlinear%20Predictors.pdf), Example 5.14 (PDF pp. 10–11), illustrates hinge-loss surrogates. Our loss is a hinge on the signed slack $-s/\gamma$, capped at one; it is a related construction, not that example's exact loss. |
| Generalization proof tools | Chapter 5, Lemma 5.6, Theorem 5.8, and Sections 5.5–5.6, supply Hoeffding, finite-class uniform convergence, covering numbers, and whole-trajectory sampling. The proof below combines those tools directly; it does not reproduce the paper's fast-rate theorem or Chapter 5's prediction-loss analysis. |

For any $P,P'\in\mathcal P$, the norm inequality gives

$$
|r_P(x,G_K(\theta)x)-r_{P'}(x,G_K(\theta)x)|
\le (M^2+1)\|P-P'\|_F.
$$

**Derivation and meaning.** Compare two certificate matrices on the same
transition: $K$, $\theta$, and $x\ne0$ stay fixed. Write $G=G_K(\theta)$
and $\Delta P=P-P'$. The common $\eta$ cancels, so

$$
r_P(x,Gx)-r_{P'}(x,Gx)
=\frac{(Gx)^\top\Delta P(Gx)-x^\top\Delta Px}{\|x\|_2^2}.
$$

For any matrix $C$ and vector $y$, Cauchy–Schwarz and the induced matrix
norm give

$$
|y^\top Cy|\le\|y\|_2\|Cy\|_2
\le\|C\|_2\|y\|_2^2
\le\|C\|_F\|y\|_2^2.
$$

Here $\|C\|_2$ is the largest singular value, while
$\|C\|_F=(\sum_{i,j}C_{ij}^2)^{1/2}$ is the square root of the sum of
squared singular values, so $\|C\|_2\le\|C\|_F$. Apply the triangle
inequality and the quadratic-form bound to the two terms separately:

$$
\begin{aligned}
|r_P(x,Gx)-r_{P'}(x,Gx)|
&\le\frac{\|\Delta P\|_F(\|Gx\|_2^2+\|x\|_2^2)}{\|x\|_2^2}\\
&\le(\|G\|_2^2+1)\|\Delta P\|_F\\
&\le(M^2+1)\|P-P'\|_F.
\end{aligned}
$$

The $M^2$ comes from the next-state energy and the $1$ from the current-state
energy. Normalization cancels state magnitude, leaving a bound uniform over
every nonzero state and every plant. No stability assumption or $M<1$ is
needed. This establishes Lipschitz continuity in the **certificate parameter
$P$**: nearby certificates assign nearby residuals to the same transition.
It does not compare two policies or two resulting trajectories. The bounds
$aI\preceq P\preceq bI$ are used elsewhere for positivity and a bounded
parameter class; this particular difference bound does not require them.

**Reading connection.** Chapter 3, Example 3.12, supplies the quadratic drift
identity. The norm estimate above is a direct linear-algebra derivation for
our normalized discrete-time residual. Chapter 5, Proposition 5.19 and
Section 5.6.2, equation (5.42), explain its learning-theory role: bound how
much a loss changes with its parameters, then transfer a finite parameter
cover to a loss cover. [Boffi et al.](https://proceedings.mlr.press/v155/boffi21a/boffi21a.pdf),
Section 4.1 and equation (4.3), likewise analyze Lipschitz parameterizations
and symmetric certificate matrices with a Frobenius-norm bound. They supply
the related framework; the explicit $(M^2+1)$ constant is derived here.

Taking a maximum preserves this Lipschitz constant; the clipped ramp multiplies
it by $1/\gamma$. Thus $\ell_P$ is $L=(M^2+1)/\gamma$-Lipschitz in $P$.
Indeed, $|\max_t u_t-\max_t v_t|\le\max_t|u_t-v_t|$, and clipping
$1+s/\gamma$ to $[0,1]$ cannot enlarge differences. Therefore, on the same
trajectory,

$$
|\ell_P(\tau)-\ell_{P'}(\tau)|
\le\frac{M^2+1}{\gamma}\|P-P'\|_F.
$$

This is what lets a representative matrix within distance $\varepsilon$
approximate the loss of any candidate to accuracy $L\varepsilon$.
For state vectors $x\in\mathbb R^n$, the symmetric parameter space has dimension
$d=n(n+1)/2$ and $\|P\|_F\le b\sqrt n$. An $\varepsilon$-net of $\mathcal P$
can be chosen with cardinality

$$
\mathcal N_\varepsilon\le(1+2b\sqrt n/\varepsilon)^d.
$$

**Why the dimension is $n(n+1)/2$.** Here $n$ counts state coordinates,
not sampled states or trajectories. The certificate matrix is $n\times n$
and symmetric. Its $n$ diagonal entries and $n(n-1)/2$ entries above the
diagonal determine every entry, giving

$$
d=n+\frac{n(n-1)}2=\frac{n(n+1)}2.
$$

The constraints $aI\preceq P\preceq bI$ restrict the allowed values, not this
ambient dimension. Since $a<b$, matrices near $((a+b)/2)I$ satisfy the
constraints, so the class has an interior in the symmetric matrix space.
For example, a state with two coordinates uses

$$
P=\begin{pmatrix}p_{11}&p_{12}\\p_{12}&p_{22}\end{pmatrix},
$$

which has three free entries. To make the Frobenius metric exactly Euclidean,
use the coordinate vector $(p_{11},\sqrt2p_{12},p_{22})$. In general, list
the diagonal entries and $\sqrt2$ times each entry above the diagonal in a
vector $v(P)\in\mathbb R^d$. Then

$$
\|v(P)-v(P')\|_2=\|P-P'\|_F,
$$

because each off-diagonal entry occurs twice in the Frobenius sum. Thus the
matrix class can be treated as a subset of ordinary $d$-dimensional Euclidean
space without changing distances.

**Why the radius is $b\sqrt n$.** Since $P$ is symmetric and
$aI\preceq P\preceq bI$, every eigenvalue satisfies
$a\le\lambda_i(P)\le b$. Orthogonal diagonalization preserves the
Frobenius norm, giving

$$
\|P\|_F^2=\sum_{i=1}^n\lambda_i(P)^2\le nb^2,
\qquad \|P\|_F\le b\sqrt n=:R.
$$

Consequently $v(\mathcal P)$ lies inside a radius-$R$ ball in $\mathbb R^d$.
This is an enclosing ball, not a claim that every matrix in that ball is
positive definite or belongs to $\mathcal P$.

**What an $\varepsilon$-net means.** For $\varepsilon>0$, choose finitely
many representatives $P^{(1)},\ldots,P^{(m)}\in\mathcal P$ such that

$$
\forall P\in\mathcal P,\quad
\min_{1\le j\le m}\|P-P^{(j)}\|_F\le\varepsilon.
$$

Every candidate is approximated by at least one representative. The covering
number is the smallest possible number of representatives; we only need an
upper bound on the size of one such net. Smaller $\varepsilon$ gives a more
accurate approximation, generally at the cost of more representatives.

**Where $(1+2R/\varepsilon)^d$ comes from.** Choose a maximal collection of
points in $v(\mathcal P)$ whose pairwise distances are greater than
$\varepsilon$: add a point whenever it is farther than $\varepsilon$ from
all existing points. Maximal means no more points can be added. This makes
the collection an $\varepsilon$-net; otherwise an uncovered point could be
added. A volume argument bounds its size and ensures this process terminates:

1. The radius-$\varepsilon/2$ balls around the chosen points are disjoint.
2. Each center has norm at most $R$, so all those balls lie inside the
   radius-$(R+\varepsilon/2)$ ball around the origin.
3. A ball of radius $r$ in $d$ dimensions has volume $c_dr^d$. If there are
   $m$ chosen points, disjointness and containment imply

$$
m\,c_d(\varepsilon/2)^d\le c_d(R+\varepsilon/2)^d,
\qquad
m\le\left(\frac{R+\varepsilon/2}{\varepsilon/2}\right)^d
=\left(1+\frac{2R}{\varepsilon}\right)^d.
$$

Substitute $R=b\sqrt n$ to obtain the stated bound. The factor $2$ comes
from packing balls of radius $\varepsilon/2$; the exponent $d$ comes from
volume scaling in the symmetric parameter space. Choosing centers directly
in $\mathcal P$ ensures each representative is an admissible candidate, even
though the packing balls themselves may extend outside the class.

**Why this helps generalization.** The previous Lipschitz bound makes each
representative approximate its nearby candidates' losses by at most
$L\varepsilon$, uniformly over trajectories. We can apply concentration and
a finite union bound to the representatives, then extend the result to every
$P$. The number of representatives enters through its logarithm:

$$
\log\mathcal N_\varepsilon
\le d\log\left(1+\frac{2b\sqrt n}{\varepsilon}\right).
$$

This is the origin of the dimension-and-resolution term in the statistical
bound below. The net is a proof device; fitting a certificate does not require
enumerating all of its representatives.

**Reading connection.** [Chapter 5](../../resources/ch5%20Empirical%20Risk%20Minimization%20for%20Nonlinear%20Predictors.pdf),
Definition 5.15 (PDF p. 11), defines nets and covering numbers. Proposition
5.18 (PDF p. 12) gives the unit-ball bound
$\log N(u;B_2^d(1),\|\cdot\|_2)\le d\log(1+2/u)$;
scaling to radius $R$ uses $u=\varepsilon/R$. Proposition 5.20
discusses bounded parameter sets. The packing argument here supplies the
bound while explicitly keeping centers inside our constrained matrix class.
Proposition 5.19 then explains the transfer from parameter approximation to
loss approximation. [Boffi et al.](https://proceedings.mlr.press/v155/boffi21a/boffi21a.pdf),
Section 4.1, equation (4.3) and the paragraph after it (PDF p. 5), use
Frobenius-bounded symmetric certificate matrices and count $k=d(d+1)/2$
parameters for a $d\times d$ matrix. Here the paper's $d$ denotes matrix size,
whereas our $d$ denotes parameter count. Their covering-number
analysis motivates this specialization; their sharper Theorem 4.3 uses
additional matrix structure and is not reproduced by our finite-net proof.

Fix $\varepsilon>0$ before drawing data. Hoeffding plus a union bound on the net
and two Lipschitz approximation errors gives, with probability at least
$1-\delta_{\rm cert}$, simultaneously for all $P\in\mathcal P$,

$$
\mathbb E_{D_0}\ell_P\le\frac1N\sum_{i=1}^N\ell_P(\tau_i)
+2L\varepsilon
+\sqrt{\frac{d\log(1+2b\sqrt n/\varepsilon)+\log(2/\delta_{\rm cert})}{2N}}.
$$

**Proof details.** For each net center the two-sided deviation probability at
radius $s$ is at most $2e^{-2Ns^2}$. Sum over the net and set
$s=\sqrt{\log(2\mathcal N_\varepsilon/\delta_{\rm cert})/(2N)}$.
Approximating an arbitrary $P$ by a center changes its population and empirical
losses by at most $L\varepsilon$ each. The cardinality bound yields the display.
The inequality therefore applies to a data-dependent fitted $\widehat P$;
its right-hand side also bounds violation probability and can be clipped at one.
No claim that this bound is numerically tight is made.

For $\varepsilon=N^{-1/2}$ the complexity term has the familiar order
$\sqrt{d\log N/N}$ with the displayed constants. This is not the fast rate
proved under the source paper's assumptions. Normalization and a bounded linear
family make this particular loss Lipschitz independently of $H$; it is still
only a statement about violation along the chosen finite horizon. The
trajectory-level maximum and its empirical behavior can depend strongly on $H$.
Do not claim that this special construction reproduces Chapter 5's general
stability-dependent horizon bounds or proves stability itself.

A change from $D_0$ to targeted $q$, or from $K$ to $K'$, changes the data law.
This bound does not automatically transfer. For a new frozen phase, use fresh
independent data and conditional error accounting, or prove a separate shift
bound. Neither adaptive point selection nor a pointwise held-out interval can
replace the stated uniform argument for a certificate fitted on those samples.

## 3. From a candidate to a verified certificate

For each predeclared retention plant, compute

$$
\eta_\theta=\lambda_{\min}\big(P-G_K(\theta)^\top P G_K(\theta)\big).
$$

If $\eta_\theta>0$, then $V_P(x)=x^\top Px$ decreases by at least
$\eta_\theta\|x\|^2$ for every state. Since $aI\preceq P\preceq bI$,

$$
V_P(x_t)\le(1-\eta_\theta/b)^tV_P(x_0),\qquad
\|x_t\|^2\le(b/a)(1-\eta_\theta/b)^t\|x_0\|^2.
$$

This establishes exponential stability and invariance of every
$\mathcal E_c(P)=\{x:V_P(x)\le c\}$ for this unconstrained linear model.
An inequality valid only on a neighborhood additionally needs a sublevel set
contained in that neighborhood. State/input constraints require containment
checks; none are implied by the unconstrained statement.

This verification uses known matrices and is logically separate from the
statistical theorem. A low empirical certificate loss can coexist with a failed
all-state inequality. Verification failure means the candidate does not support
this claim, not that a particular trajectory must fail or the plant is unstable.

## 4. Preserve the verified margin through an actual update

At one plant write $G=G_K(\theta)$, $E=-B(\theta)\Delta K$ and $G'=G+E$.
Keep $P$ fixed. If $G^\top PG-P\preceq-\eta_\theta I$, then

$$
G'^\top PG'-P=G^\top PG-P+G^\top PE+E^\top PG+E^\top PE.
$$

For every $x$, the added quadratic form is at most

$$
\big(2\|G^\top P\|_2\|E\|_2+\|P\|_2\|E\|_2^2\big)\|x\|_2^2.
$$

Consequently the same certificate remains valid whenever

$$
c_\theta(\Delta K):=2\|G^\top P\|_2d_\theta+\|P\|_2d_\theta^2
<\eta_\theta,\qquad d_\theta=\|B(\theta)\Delta K\|_2.
$$

For a finite retention family enforce this for every member using the shared
$P$. To retain a specified fraction $\rho\in(0,1)$ of each margin, require
$c_\theta\le(1-\rho)\eta_\theta$. Then the new decrease is at least
$\rho\eta_\theta$. The condition is sufficient, not necessary; direct new
matrix verification can show that rejected proposals were in fact stable.
Record this conservatism rather than silently changing the acceptance rule.

### A reviewable single-update mechanism

1. Freeze $K$ and verify the old common $P$ on the predeclared retention plants.
   If verification fails, the preservation theorem's prerequisite is absent;
   stop this certified update branch and report it.
2. Evaluate old finite-horizon costs on a declared condition grid. Give uniform
   weight to points in $|J_{H,K}-h|\le w$, with a predeclared uniform fallback if
   the band is empty. Mix with $q_{\rm base}$ using fixed $\lambda$.
3. Holding this $q$ fixed, form $D=-\nabla_K F_q(K)$, where
   $F_q(K)=\mathbb E_qJ_{H,K}$ is a finite weighted sum for the first example.
4. Try $\Delta K=\beta^j s_0D$ for $j=0,1,\ldots$, $0<\beta<1$. Accept the
   first step satisfying all retained-margin tests and, if $D\ne0$, an Armijo
   test $F_q(K+\Delta K)\le F_q(K)-c_A\beta^j s_0\|\nabla F_q(K)\|_F^2$
   with $c_A\in(0,1)$ fixed in advance.
5. Use a declared finite search cap in computations. If $D=0$ or no step passes,
   retain $K$ and report no update. Otherwise freeze $K'$ and evaluate afresh.

For exact gradients, finite matrices/horizon, and $D\ne0$, $F_q$ is smooth.
A sufficiently small step meets Armijo, and $c_\theta(sD)\to0$ as $s\to0$.
Strict old margins on a finite family therefore ensure a step exists in the
ideal uncapped search. This proves decrease of the fixed training objective
and retention of verified stability. It proves neither a useful progress rate
nor an increase in cost/reliability coverage. Sampled gradients would require
additional analysis and are not part of this first proposition.

Learning a new $P'$ after updating is a different operation. Comparing
$\{x:x^\top P'x\le c\}$ to $\{x:x^\top Px\le c\}$ without common normalization
and geometric checks can mistake certificate scaling for region expansion.

## 5. Finite-horizon cost regression is a separate calculation

Let $W_K=Q+K^\top RK$ and

$$
P_H(K,\theta)=\sum_{t=0}^{H-1}(G_K(\theta)^t)^\top W_KG_K(\theta)^t.
$$

For $\|x\|\le r$, the elementary baseline is

$$
|J_{H,K'}(\theta,x)-J_{H,K}(\theta,x)|
\le r^2\|P_H(K',\theta)-P_H(K,\theta)\|_2=:b_J(\theta).
$$

To relate it to the update, use $\|G\|,\|G'\|\le M$ with $M\ge1$ and

$$
G'^t-G^t=\sum_{i=0}^{t-1}G'^{t-1-i}(G'-G)G^i,
\quad \|G'^t-G^t\|\le tM^{t-1}\|B(\theta)\|\|\Delta K\|.
$$

The changed input penalty obeys
$\|W_{K'}-W_K\|\le\|R\|(\|K'\|+\|K\|)\|\Delta K\|$. Splitting each
quadratic summand and summing proves

$$
\|P_H(K',\theta)-P_H(K,\theta)\|\le L_H(\theta)\|\Delta K\|,
$$
$$
L_H(\theta)=2\|W_K\|\|B(\theta)\|\sum_{t=1}^{H-1}tM^{2t-1}
+\|R\|(\|K'\|+\|K\|)\sum_{t=0}^{H-1}M^{2t}.
$$

Use a predeclared bounded gain neighborhood to replace $M$ and $\|K'\|$ by
uniform constants when an a priori step-size rule is desired. All norms above
are spectral except explicitly labeled Frobenius norms. Finite horizon requires
no asymptotic stability, although amplification can make the bound vacuous.
Schur eigenvalues alone do not justify setting $M<1$ in Euclidean norm.

On $\Theta\times\{x:\|x\|\le r\}$, define $S_K^J=\{J_{H,K}\le h\}$ and
$B_{\rm loss}=\{h-b_J(\theta)<J_{H,K}\le h\}$. Then

$$
S_K^J\setminus S_{K'}^J\subseteq B_{\rm loss},\qquad
\mu(S_K^J\setminus S_{K'}^J)\le\mu(B_{\rm loss}).
$$

This is preservation of the interior cost set, not all old acceptable points.
If the actual update also reduces cost by at least $a_J>0$ on a set $T$, then

$$
\mu(S_{K'}^J)-\mu(S_K^J)
\ge\mu(T\cap\{h<J_{H,K}\le h+a_J\})-\mu(B_{\rm loss}).
$$

The expansion extension must derive useful $a_J,T$ from the chosen update.
Assuming their existence is only accounting, not a proof that targeting works.

### A stable targeted-gradient counterexample

Take scalar plants $\theta\in\{0,1\}$, $x^+=\theta x+u$, $u=-Kx$,
$x_0=1$, $Q=1,R=0,H=2,h=1.25$, and equal reference mass on the two plants.
Then $J_{2,K}(\theta,1)=1+(\theta-K)^2$. At $K=0.5$, both plants are
acceptable, both have closed-loop magnitude $0.5$, and $P=1$ has margin $0.75$.

A target distribution concentrated at the upper boundary plant $\theta=1$
has gradient $2(K-1)=-1$. A step of size $0.2$ gives $K'=0.7$. Training cost
falls from $1.25$ to $1.09$, but the other plant's cost rises to $1.49$; reference
coverage falls from $1$ to $1/2$. Both plants remain stable, with the smallest
new certificate margin $0.51$. The preservation bound gives
$2(0.5)(0.2)+(0.2)^2=0.24<0.75$, and passes even when retaining half the margin.
An Armijo coefficient $c_A=1/2$ also accepts this step.

This is a counterexample to arbitrary selection among boundary points, not to
every symmetric boundary-band rule: the symmetric rule in the algorithm above
would give zero gradient on this particular two-point example. It isolates the
fact that useful training descent plus preserved stability need not preserve
cost coverage. The explanatory note retains the ellipse example and a
continuous-plant example with equal gains/losses for complementary intuition.

## 6. Local nonlinear and CartPole limits

For a smooth closed-loop map with the same equilibrium write
$f_K(x)=G_Kx+e_K(x)$ and require an explicit remainder bound on a neighborhood.
If $\|e_K(x)\|\le c\|x\|^2$, the additional Lyapunov drift is at most
$2\|G_K^\top P\|c\|x\|^3+\|P\|c^2\|x\|^4$. To use it, choose an ellipsoid
inside a radius where this is smaller than the quadratic decrease margin and
verify invariance. Uniform plant/update claims require uniform remainder bounds.
Merely writing a linearization is insufficient.

The repository's two-force CartPole action rule does not preserve the zero
state and is not the smooth feedback used above. Practical stability about a
set, or a hybrid/quantized analysis, is a different task. Large-angle recovery
and final-window RMS performance also do not follow from local stability.

For a score-defined failure event $\{C>h\}$, a pathwise coupling satisfying
$|C'-C|\le\epsilon$ implies
$|\Pr(C'>h)-\Pr(C>h)|\le\Pr(|C-h|\le\epsilon)$. This illustrates the
additional threshold-mass assumption needed for a probability statement; the
linear cost proof supplies neither this coupling nor the CartPole termination
analysis. Lyapunov, barrier, finite-cost, and recovery guarantees stay distinct.

## Supporting rollout inference

The following construction is retained for empirical CartPole recovery claims.
It is supporting infrastructure, not the main Stage 1 learning-theory result.
Use distinct error allocations if these claims and certificate generalization
are combined in one campaign-wide guarantee. The retained acquisition design
is optional and does not change the fixed-distribution proof above.

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

The elementary construction is sufficient for the minimum illustration. A
Bernoulli mixture confidence sequence is an optional tighter alternative; fix
one common certifier before any allocation comparison. Howard et al. explain
mixture constructions and
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
 \Pr_{\mathrm{GP}}\{C^+_{z,b}\ne\varnothing,\quad
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

## Protocol if the optional CartPole retraining illustration is run

- Begin with uniform and reliability-boundary targets; intermediate difficulty
  is an optional third comparator. Match the background mixture and training
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


## Retained curriculum and evaluation literature

The following curriculum and evaluation references are retained from v2 as
background. They do not replace the certificate-learning anchor or add
implementation requirements:

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

These additional sources remain optional background, without becoming required
algorithms:

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
