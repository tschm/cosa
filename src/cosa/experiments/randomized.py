"""A seeded random problem generator: the instances §16.3 says must all be cross-checked.

§16.3 (``paper.tex:1124``) sets the standard that gives this module its reason to exist:
*"For **every randomly generated** test problem, compare COSA against a reference solver."*
Every randomly generated problem -- and the plan never says where those come from. The
structured families of :mod:`cosa.experiments.portfolio` are the opposite of randomly
generated: they are six fixed shapes, and no amount of reseeding turns a box-constrained
portfolio into a different *kind* of problem. So the requirement presupposes a generator
that randomizes the shape itself, and this is it.

**Three knobs, because three things can go wrong.** A seed here draws not just data but a
*specification* -- see :class:`RandomSpec` -- over the three axes the algorithm is
sensitive to along different failure modes:

* **dimension.** The number of assets and the number of extra inequality rows. Cheap to
  vary and the first thing an off-by-one in the row layout will trip over.
* **conditioning.** The covariance's condition number, drawn across six orders of
  magnitude, and its *rank*, drawn so that a rank-deficient covariance turns up regularly.
  Rank deficiency is what puts the apex within reach of a nonzero portfolio, so drawing it
  is how the apex branch gets exercised without anyone constructing an apex instance by
  hand.
* **active-set structure.** How many of the extra rows are *exactly tight* at the witness
  point. This is the axis a naive generator misses entirely: random rows with random
  right-hand sides are almost surely all slack or infeasible, so the interesting states --
  several constraints active at once, a working set that has to grow -- would simply never
  be generated.

**Feasible by construction, from a witness outward.** A witness ``x0`` is drawn on the
interior of the simplex *first*, and the constraint data is then built around it: the
tight rows get ``b_i = a_i.T @ x0`` exactly, the slack rows get something larger. So the
instance is feasible because a feasible point was chosen before the constraints were, not
because the draw happened to work out. Boundedness is free -- every instance keeps the
budget equality and the long-only bounds, so its feasible set is a subset of the simplex.

That is not a convenience. §16.3's check is only a check if it runs on every draw: a
generator that is infeasible one time in five turns a property test into a test of the
generator's luck, and the natural fix -- skipping the infeasible draws -- silently narrows
what is being tested.

**A failure names its own reproduction.** Every instance's name carries the whole
specification, and :meth:`RandomSpec.reproduce` prints the call that rebuilds it. Issue #32
asks for "reproducible seeds recorded with any failure", and a seed is only recorded if it
is in the failure message rather than in the mind of whoever ran the test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from cosa.active_set.working_set import ConstraintNames
from cosa.experiments.portfolio import LAMBDA, PortfolioInstance, synthetic_market
from cosa.problem.socp import ProblemError

if TYPE_CHECKING:
    from cosa import Vector

__all__ = [
    "MAX_ASSETS",
    "MAX_CONDITION",
    "MAX_SEED",
    "RandomSpec",
    "random_instance",
    "random_spec",
]

MAX_SEED: Final = 2**32 - 1
"""The largest seed the generator accepts, which is what a strategy should draw from."""

MAX_ASSETS: Final = 8
"""The default cap on the drawn asset count.

Small on purpose. These instances are solved inside property tests, several hundred times
a run, so each one has to cost milliseconds; and the failures this generator is hunting --
a row in the wrong place, a sign flipped, an apex mishandled -- show up at four assets as
readily as at four hundred. Size is what the ``large`` family of
:mod:`cosa.experiments.portfolio` and the scaling study of #28 are for.
"""

MAX_CONDITION: Final = 1.0e6
"""The worst covariance conditioning drawn, as a condition number.

Six orders of magnitude: enough that a reference solver starts losing digits and the
cross-check's tolerance is doing real work, and not so much that the instance stops having
a well-defined answer to agree about. The genuinely hostile end belongs to #33's
robustness families, which construct it deliberately rather than stumbling on it.
"""


@dataclass(frozen=True)
class RandomSpec:
    """What a seed drew: the shape of one random instance.

    Separated from the instance it describes so that the draw can be inspected, filtered
    or reported without building anything -- a property test that wants only
    rank-deficient instances can draw specs and discard, and a failure can print the spec
    even if construction is what failed.

    Attributes:
        seed: the seed that produced this specification, and that reproduces it.
        assets: the number of assets ``n``.
        rank: the covariance's rank, between 1 and ``assets``. Below ``assets`` the apex
            is reachable at a nonzero portfolio.
        condition: the covariance's condition number.
        rows: how many random inequality rows sit on top of the long-only bounds.
        tight: how many of those rows are exactly active at the witness.
        lam: the risk-aversion parameter.
    """

    seed: int
    assets: int
    rank: int
    condition: float
    rows: int
    tight: int
    lam: float

    @property
    def name(self) -> str:
        """A name carrying the whole specification, for a benchmark row or a failure."""
        return (
            f"random-n{self.assets}-rank{self.rank}-cond{self.condition:.0e}"
            f"-tight{self.tight}of{self.rows}-s{self.seed}"
        )

    @property
    def is_rank_deficient(self) -> bool:
        """Is the covariance singular, so that the apex is reachable at some ``x != 0``?"""
        return self.rank < self.assets

    def reproduce(self) -> str:
        """The call that rebuilds this instance, as a line to paste into a session.

        Returns:
            A one-line reproduction, which is what "reproducible seeds recorded with any
            failure" means in practice.
        """
        return f"cosa.experiments.randomized.random_instance({self.seed})"

    def __str__(self) -> str:
        """The name and the reproduction line together."""
        return f"{self.name} -- reproduce with: {self.reproduce()}"


def random_spec(seed: int, *, max_assets: int = MAX_ASSETS, lam: float = LAMBDA) -> RandomSpec:
    """Draw a specification from a seed.

    The draws are deliberately not independent of each other: ``tight`` is capped at
    ``assets - 1`` as well as at ``rows``, so the tight rows together with the budget
    equality cannot pin the witness down completely. A witness that is the unique point
    satisfying its own active constraints is a degenerate vertex, and while those are
    worth testing, they are #33's and #36's business to construct deliberately rather than
    something this generator should produce by accident and then be unable to distinguish
    from a bug.

    Args:
        seed: the seed. Any integer in ``[0, MAX_SEED]``.
        max_assets: the largest asset count to draw.
        lam: the risk-aversion parameter, which is not drawn -- it scales the objective
            rather than changing the shape, and the frontier study of #35 is what varies it.

    Returns:
        The specification.

    Raises:
        ProblemError: if the seed is out of range or ``max_assets`` is below 2.
    """
    if not 0 <= seed <= MAX_SEED:
        raise ProblemError("seed", f"expected a seed in [0, {MAX_SEED}], found {seed}")
    if max_assets < 2:
        raise ProblemError("max_assets", f"a random instance needs room for at least 2 assets, found {max_assets}")

    rng = np.random.default_rng(seed)
    assets = int(rng.integers(2, max_assets + 1))
    rows = int(rng.integers(0, assets + 1))
    return RandomSpec(
        seed=int(seed),
        assets=assets,
        rank=int(rng.integers(1, assets + 1)),
        condition=float(10.0 ** rng.uniform(0.0, np.log10(MAX_CONDITION))),
        rows=rows,
        tight=int(rng.integers(0, min(rows, assets - 1) + 1)),
        lam=float(lam),
    )


def _witness(spec: RandomSpec) -> Vector:
    """Draw the interior point the instance is built around.

    A Dirichlet draw with a concentration above one, so the point is in the *interior* of
    the simplex rather than near a face: a witness with a weight at zero would make a
    long-only bound accidentally active, which would put the drawn active-set structure
    and the actual one out of step.

    Args:
        spec: the specification, for its seed and asset count.

    Returns:
        A point on the interior of the simplex, ``(assets,)``.
    """
    return np.random.default_rng((spec.seed, 1)).dirichlet(np.full(spec.assets, 4.0))


def random_instance(
    seed: int,
    *,
    max_assets: int = MAX_ASSETS,
    lam: float = LAMBDA,
) -> PortfolioInstance:
    """Build one random instance from a seed: draw the spec, then build around a witness.

    The result is an ordinary :class:`cosa.experiments.portfolio.PortfolioInstance`, so a
    randomized instance and a structured one are interchangeable everywhere downstream --
    the cross-check, the working-set description, the benchmark row. The only difference is
    that this one's shape was drawn.

    Args:
        seed: the seed.
        max_assets: the largest asset count to draw.
        lam: the risk-aversion parameter.

    Returns:
        The instance, feasible at its :attr:`~PortfolioInstance.witness` and bounded.

    Raises:
        ProblemError: if the seed is out of range.
    """
    spec = random_spec(seed, max_assets=max_assets, lam=lam)
    point = _witness(spec)
    market = synthetic_market(
        spec.assets,
        seed=spec.seed,
        lam=spec.lam,
        rank=spec.rank,
        condition=spec.condition,
    )
    portfolio = market.with_equalities(np.ones((1, spec.assets)), np.ones(1)).with_inequalities(
        -np.eye(spec.assets), np.zeros(spec.assets)
    )

    names = [f"long-only bound on asset {i}" for i in range(spec.assets)]
    if spec.rows:
        rng = np.random.default_rng((spec.seed, 2))
        extra = rng.normal(size=(spec.rows, spec.assets))
        # The right-hand side is measured at the witness, which is what puts exactly
        # `tight` rows on their boundary there and leaves the rest strictly slack.
        exposure = extra @ point
        slack = np.concatenate([np.zeros(spec.tight), rng.uniform(0.05, 0.5, size=spec.rows - spec.tight)])
        portfolio = portfolio.with_inequalities(extra, exposure + slack)
        names.extend(f"random row {i} ({'tight' if i < spec.tight else 'slack'}) " for i in range(spec.rows))

    return PortfolioInstance(
        name=spec.name,
        portfolio=portfolio,
        problem=portfolio.to_socp(),
        names=ConstraintNames(
            inequalities=tuple(name.strip() for name in names),
            equalities=("fully invested",),
            cones=("risk",),
        ),
        witness=portfolio.socp_point(point),
    )
