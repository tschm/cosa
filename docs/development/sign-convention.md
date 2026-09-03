# The sign convention

The record of the decision taken in [#9](https://github.com/tschm/cosa/issues/9). The
project plan defers this one on purpose -- *"The exact sign convention will be fixed in
the implementation and must be used consistently throughout the derivation and code"* --
so it is fixed here, once, and every consumer reads it from here rather than choosing
again.

It matters because four separate pieces of the algorithm consume it: the KKT assembly,
the multiplier computation and its sign tests, the conic KKT residuals and termination
criterion, and the conic working-set logic that decides when a cone leaves the working
set. If each picks its own sign, the multiplier tests agree with nothing and the
residuals silently disagree with the working-set decisions.

## The problem

`cosa.problem.socp.SOCP` is the general form, of which the plan's eq. (7) is one
instance:

```text
min  c.T @ z
s.t. A @ z <= b
     E @ z = d
     G @ z + h in K
```

`K = Q^(n_1) x ... x Q^(n_J)` is a Cartesian product of second-order cones. Eq. (7) is
this with `z = (x, t)`, `c = (-mu, lam)`, `G = [[0, 1], [L, 0]]`, `h = 0` and a
single factor `Q^(1 + k)`.

## The cone, written head first

```text
Q^n = {(s_0, s_1) in R x R^(n-1) : ||s_1||_2 <= s_0}
```

The head `s_0` is entry **0**, never the last entry. That ordering is part of the
convention, not a detail of one function: it is the layout of the conic slack
`G @ z + h`, of the dual variable `w`, and of every `(head, tail)` pair either is split
into. `SecondOrderCone.split` and `ConeProduct.split` are the only places that index
it.

In this representation the second-order cone is self-dual, `K* = K`. There is therefore
no separate dual-cone type: `w in K` is the dual feasibility condition, which is also
what the plan asserts (`w_soc in Q`).

## Multipliers and the Lagrangian

`y` for the inequalities, `nu` for the equalities, `w` for the cone:

```text
Lagr(z, y, nu, w) = c.T @ z + y.T @ (A @ z - b) + nu.T @ (E @ z - d) - w.T @ (G @ z + h)
```

The conic term is **subtracted**. That is the whole choice, and everything below follows
from it.

## The KKT conditions

```text
stationarity            c + A.T @ y + E.T @ nu - G.T @ w = 0
primal feasibility      A @ z <= b,  E @ z = d,  G @ z + h in K
dual feasibility        y >= 0,  w in K
complementarity         y_i * (a_i.T @ z - b_i) = 0
                        w.T @ (G @ z + h) = 0
```

Subtracting the conic term is what puts `w` in `K` rather than in `-K`. Adding it
instead would make dual feasibility read `-w in K`, which contradicts the plan's own
statement of the condition and would leave every consumer negating a vector before it
could ask whether it is in the cone.

## What it implies for eq. (7)

In eq. (7) the variable `t` appears only in the objective, with coefficient `lam`, and
in the head row of `G`. Its stationarity equation is therefore `lam - w_0 = 0`:

```text
w_0 = lam
```

The head of the cone multiplier *is* the risk-aversion parameter. This is the
convention's signature, and the cheapest way to tell whether a consumer has adopted it:
the opposite choice produces `w_0 = -lam`, which for `lam > 0` is not even in `Q`.

## Where this differs from the plan

The plan's stationarity display writes the conic term with a plus sign,
`-mu + A.T @ y + E.T @ nu + L.T @ w = 0`, while also asking for `w_soc in Q`. Those two
cannot both hold for the same `w`: the sign that leaves `w` in the cone is the minus
sign. The plan is explicit that the display is provisional and the convention is the
implementation's to fix, and of the two statements the one worth keeping is
`w_soc in Q`, because self-duality is what the conic complementarity relation and the
working-set logic are built on. So the sign on `L.T @ w` in that display flips, and
nothing else in the plan changes.

Where the plan and the code disagree about anything else, treat it as a bug in one of
them and say so on the issue.

## How to consume it

Two entry points, and no third:

- `SOCP.stationarity_residual(y, nu, w)` for the residual itself.
- `cosa.SIGN_CONVENTION` for the three signs, when a consumer assembles the terms into
  something else -- a KKT matrix, a multiplier solve, a warm-start check.

```python
import numpy as np

from cosa import SIGN_CONVENTION, MeanStdForm

# One asset, mu = 1, lam = 1/2, x <= 1. The bound binds, so x = t = 1, and the KKT
# conditions give y = 1/2 and w = (1/2, -1/2): the head of w is lam, and w is on the
# boundary of Q.
problem = MeanStdForm(
    mu=np.array([1.0]),
    lam=0.5,
    A=np.array([[1.0]]),
    b=np.array([1.0]),
    E=np.zeros((0, 1)),
    d=np.zeros(0),
    L=np.array([[1.0]]),
).to_socp()

z = np.array([1.0, 1.0])
y = np.array([0.5])
nu = np.zeros(0)
w = np.array([0.5, -0.5])

assert np.allclose(problem.stationarity_residual(y, nu, w), 0.0)
assert np.isclose(w @ problem.cone_slack(z), 0.0)

head, tail = problem.cone.split(w)[0]
assert head == 0.5
assert np.linalg.norm(tail) <= head

# The same three signs a consumer that assembles its own terms must use.
assert (SIGN_CONVENTION.inequality, SIGN_CONVENTION.equality, SIGN_CONVENTION.cone) == (1.0, 1.0, -1.0)
```

## How it is enforced

`tests/test_socp.py` carries two instances solved by hand -- the one above, and a
two-asset budget-constrained one with a nonzero `nu`. Their multipliers are facts about
this convention, not recordings of what the code produced, and three tests turn them
into a gate:

- `test_sign_convention_is_the_one_fixed_convention` pins the three signs, so a
  consumer that assembles its own terms cannot quietly assemble different ones.
- `test_the_cone_multiplier_head_is_lambda` pins `w_0 = lam`.
- `test_flipped_signs_are_not_stationary` negates each block of multipliers in turn and
  requires that stationarity break. The opposite convention does not also satisfy this
  one, which is what makes the choice checkable rather than merely written down.

A consumer that adopts the other sign fails at least one of the three. When a later
issue adds a residual set, a multiplier solve or a working-set rule, it asserts against
these same instances.
