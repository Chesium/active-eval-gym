# Safe descent need not preserve coverage: the R > 0 example

Date: 4 November 2026.

Status: this note records the example behind one sentence in the midterm report. The research
note and code fixtures still use the older $R=0$ version, so the example is not yet part of the
recorded studies. The results below concern this exact scalar linear setting only.

Relevant sources:

- [Midterm report](../Evaluation_Guided_Domain_Randomization/midterm_report.tex), the sentence
  "Safe descent need not preserve coverage" in the "One safe Armijo update" paragraph.
- [Research note](../Evaluation_Guided_Domain_Randomization/Report_Sections/02_problem_formulation.tex),
  `prop:stage2_counterexample`, which holds the older $R=0$ version.
- [Comment tracker](Midterm-Comments-David/comment-resolving-plan.md), items M03 and M13.

## Claim

An accepted update can lower the training objective and keep every retained plant certifiably
stable while coverage falls. This does not count against boundary targeting, because the example
uses a hand-set $q$ and boundary targeting gives $D=0$ there.

## Setup

| Quantity | Value |
| --- | --- |
| Plants | $A(\theta)=\theta-1/2$, $B=1$, $\theta\in\{0,1\}$, so $A\in\{-1/2,1/2\}$ |
| Conditions | $(\theta,x_0)$ with $x_0=1$ for both plants |
| Costs | $Q=1$, $R=0.1$, $H=2$, $h=1.25$ |
| Old gain and certificate | $K=0$, $P=1$ |
| Evaluation weights | $\mu=(1/2,1/2)$ |
| Training weights | $q=(1/4,3/4)$, set by hand |
| Update | $s_0=0.4$, $c_A=\rho=0.5$, $\kappa=0.25$ |

## Derivation

With $x_0=1$ and $H=2$, the closed loop $G=A(\theta)-K$ gives

$$J_{H,K}(\theta)=(1+RK^2)(1+G^2).$$

At $K=0$ the input is zero, so both costs equal $1.25$ for every $R$. The gradient there is
$\partial_KJ=-2A(\theta)$, which is $+1$ and $-1$ on the two plants. So
$\nabla_KF_q(0)=\tfrac14(1)+\tfrac34(-1)=-\tfrac12$ and $D=\tfrac12$, also for every $R$. The first
trial is $\Delta K=s_0D=0.2$.

## Results

Exact values at $R=1/10$:

| Quantity | Before ($K=0$) | After ($K=0.2$) |
| --- | --- | --- |
| Cost on $A=-1/2$ | $5/4$ | $37399/25000=1.49596$ |
| Cost on $A=1/2$ | $5/4$ | $27359/25000=1.09436$ |
| $F_q$ | $5/4$ | $29869/25000=1.19476$ |
| Coverage | $1$ | $1/2$ |
| Margin on $A=-1/2$ | $0.75$ | $0.51$ |
| Margin on $A=1/2$ | $0.75$ | $0.91$ |

The step is accepted:

- **Safety test:** $|\Delta K|=0.2\le\kappa$, and $c_\theta=6/25=0.24\le(1-\rho)(0.75)=0.375$ on both
  plants.
- **Armijo condition:** the bound is $5/4-c_As_0D^2=6/5$, and $1.19476\le1.2$.

Boundary targeting gives $D=0$ because both costs equal $h$. Both conditions are in the boundary
band, so $q$ is uniform and the two gradients cancel. This holds for every $R$.

## Sensitivity to R

The first trial passes when $1.19(1+0.04R)\le1.2$, which means $R\le25/119\approx0.21$. Above that
the search halves the step. Coverage still fell at the two larger values tested, $R=0.22$ and $R=1$.

## Why not the older version

The older example uses $A(\theta)=\theta$, $R=0$ and $K=0.5$. Its closed loops are identical, but
it fails under $R>0$ because the old input is nonzero:

- **Same $h$:** both old costs rise above $1.25$. Even $R=0.01$ gives coverage $0\to1/2$, the
  opposite of the claim.
- **Raised $h$:** the coverage drop returns, but the step and values change, and uniform weights
  no longer give $D=0$.

Setting the old gain to $K=0$ avoids both problems.

## Reproduce

Run from the repository root with `uv run python`. This uses the repository's update code.

```python
import numpy as np

from active_eval_gym.linear_updates import (
    ConditionGrid,
    LinearFamily,
    UpdateSpec,
    backtrack,
    rollout,
    target_law,
    trajectory_costs,
)

family = LinearFamily(
    np.array([[[-0.5]], [[0.5]]]),
    np.ones((2, 1, 1)),
    np.ones((1, 1)),
    np.array([[0.1]]),
)
grid = ConditionGrid(np.array([0, 1]), np.ones((2, 1)), np.full(2, 0.5))
gain, certificate = np.zeros((1, 1)), np.ones((1, 1))
spec = UpdateSpec(horizon=2, threshold=1.25, initial_step=0.4, armijo=0.5)
training = np.array([0.25, 0.75])


def costs(k):
    return trajectory_costs(family, *rollout(family, grid, k, spec.horizon))


before = costs(gain)
result = backtrack(family, grid, gain, certificate, training, spec)
after = costs(result.gain)
boundary = target_law(before, grid.weights, spec.threshold, "boundary")
targeted = backtrack(family, grid, gain, certificate, boundary.weights, spec)

assert result.status == "accepted"
np.testing.assert_allclose(result.gain, [[0.2]])
np.testing.assert_allclose(before, [1.25, 1.25])
np.testing.assert_allclose(after, [1.49596, 1.09436])
np.testing.assert_allclose(training @ after, 1.19476)
np.testing.assert_allclose(grid.weights @ (before <= spec.threshold), 1.0)
np.testing.assert_allclose(grid.weights @ (after <= spec.threshold), 0.5)
np.testing.assert_allclose(result.trials[0].new_margins.min(), 0.51)
np.testing.assert_allclose(result.trials[0].consumption, 0.24)
np.testing.assert_allclose(boundary.weights, [0.5, 0.5])
assert targeted.status != "accepted"
np.testing.assert_allclose(targeted.gain, gain)
print("example reproduced; boundary targeting status:", targeted.status)
```

## Follow-up

The Appendix (M13) should prove this version, because the midterm states $R\succ0$. Until then,
the research note's proposition and the fixtures in `linear_study.py` and
`tests/test_linear_updates.py` still use $R=0$.
