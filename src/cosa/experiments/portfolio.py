"""The portfolio test problems of §10: the instances COSA is asked to solve.

§10 (``paper.tex:788``) names five families, and this module is all five plus the large
instance §12.1 asks for separately. They are *inputs*, not results -- the prototype of #20
has nothing to run without them, which is why the first two live in M5 rather than in the
M9 benchmark suite.

Every family is a variation on one theme: eq. (8), the basic long-only fully-invested
portfolio, with more linear structure bolted on. That is the whole point of the exercise.
The conic part never changes -- one cone, `Q^(1 + rank)` -- while the polyhedral part gets
progressively harder, so a failure can be attributed. The plan says what each family is
*for*:

* :func:`basic` -- eq. (8), "the simplest realistic test".
* :func:`box` -- "particularly useful for active-set testing because many portfolio
  bounds are expected to become active" (``paper.tex:811``). The family whose active set
  is large.
* :func:`sector` -- these "create nontrivial combinations of active constraints"
  (``paper.tex:817``). The family whose active set is *structured*: sector caps overlap
  the long-only bounds, so which combination binds is not obvious.
* :func:`factor_exposure` -- "a useful test of correlated active constraints"
  (``paper.tex:826``). Rows of ``F`` are dense and correlated, so the active rows are
  nearly dependent, which is where the rank detection of #25 will earn its keep.
* :func:`turnover` -- "particularly interesting for warm starts because successive
  portfolio rebalancing problems are naturally related" (``paper.tex:839``). The family
  #30 and #35 exist for.
* :func:`large` -- §12.1 (``paper.tex:905``) wants "individual large problems" as one of
  four comparison modes, and none of the structured families is large by construction.

**Feasibility is by construction, never by luck.** Each family is built around a point
known to satisfy it -- the equal-weight portfolio, or the previous holdings for turnover --
and the constraint data is centred on that point rather than drawn independently of it.
:func:`factor_exposure` is the clearest case: random factor bounds around a random ``F``
would be infeasible most of the time, so the bounds are placed around ``F @ x_equal``.
This matters because #19's and #31's "done when" is that a reference solver confirms every
family feasible and bounded, and a generator that is *usually* feasible turns that check
into a flaky test. Boundedness comes free everywhere: every family keeps the budget
equality and a bound below, so the feasible set is compact.

**Turnover is the family that needed the representation to be general.** It is the one
that cannot be written over ``x`` alone: §10.5 asks for auxiliary variables and linear
inequalities, so :func:`turnover` returns a problem whose variable vector is
``(x, t, delta)`` and uses :meth:`cosa.SOCP.augment`. That method exists for this family;
if it had not, this issue would have had to reopen the M1 representation.

**Seeded, and therefore reproducible.** Every generator takes a ``seed`` and puts it in
the instance's name, so a failing benchmark row names the instance that produced it. The
market data itself comes from :func:`synthetic_market`, which controls rank and condition
number exactly -- the two knobs the robustness families turn.

**The robustness families are the opposite kind of instance.** §12.4 (``paper.tex:936``)
asks for six adversarial families whose purpose is *"to identify failure modes **before**
optimizing performance"*, and they are here too, below the six above. The distinction is
worth keeping in mind while reading: everything above is built so that a failure is
*attributable* -- well conditioned, feasible by construction, one pathology at a time
absent. Everything below is built so that a failure *happens*. A generator that is careful
about conditioning and a generator whose entire point is a condition number of ``1e10``
belong to the same module because they are the same kind of object, but they are used for
opposite purposes.

Each robustness family ships with the measurement that shows it is pathological, through
:func:`diagnose` -- which is what §12.4's "runnable as a standalone diagnostic" has to mean
before the solver of #20 exists, and what #36's failure-mode study will extend. Two of the
six are load-bearing for other issues: :func:`nearly_redundant` and
:func:`degenerate_optimum` are what #25's rank detection has to survive, and
:func:`nearly_active_cone` is what #29's hysteresis has to stop oscillating on. Neither can
be shown to fix anything without them, which is why #43 schedules this ahead of M7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from cosa.active_set.working_set import ConstraintNames, WorkingSet
from cosa.problem.portfolio import MeanStdPortfolio
from cosa.problem.socp import SOCP, ProblemError, _vector

if TYPE_CHECKING:
    from cosa import Matrix, Vector

__all__ = [
    "BOX_WIDTH",
    "CONDITION",
    "ILL_CONDITION",
    "LAMBDA",
    "Diagnosis",
    "PortfolioInstance",
    "all_families",
    "all_robustness",
    "badly_scaled",
    "basic",
    "box",
    "degenerate_optimum",
    "diagnose",
    "factor_exposure",
    "highly_correlated",
    "ill_conditioned",
    "large",
    "many_active_bounds",
    "nearly_active_cone",
    "nearly_redundant",
    "sector",
    "synthetic_market",
    "turnover",
]

LAMBDA: Final = 2.0
"""The default risk aversion.

Chosen so that neither term dominates on the synthetic market of
:func:`synthetic_market`: with returns around 8% and volatilities around 20%, ``lam = 2``
puts ``mu.T @ x`` and ``lam * sigma(x)`` within a factor of a few of each other, so the
optimum is interior to neither the return-seeking nor the risk-averse extreme. A ``lam``
so large that the minimum-variance portfolio wins, or so small that the objective is
effectively linear, would make every family test the same thing.
"""

_FACTOR_STREAM: Final = 1
_TURNOVER_STREAM: Final = 2
"""Stream tags for the data a family draws on top of the market.

A tuple seed gives ``numpy.random.default_rng`` an independent stream, so the factor
matrix and the previous holdings are drawn from streams of their own rather than from the
market's. That is what keeps a family's market fixed when its own parameters change:
asking for three factors instead of two must not silently move ``mu`` and ``Sigma``, or
two rows of a comparison table stop being comparable.
"""

BOX_WIDTH: Final = 1.5
"""How tight the default box is, as a multiple of the equal weight ``1 / assets``.

Not a fixed weight, because a fixed one does not bind. Measured on this module's own
market: a cap of ``0.4`` leaves the optimum of :func:`box` *identical* to
:func:`basic`'s, active set and objective alike -- the long-only optimum already respects
it, so the family collapses into the one it is supposed to differ from. §10.2 wants the
family whose bounds become active, so the cap has to scale with the number of assets. At
``1.5 / assets`` it binds on every seed tried, and roughly a third of the bound rows are
active at the optimum against a fifth for :func:`basic`.
"""

CONDITION: Final = 1.0e2
"""The default condition number of the nonzero part of the covariance.

Deliberately mild. A well-conditioned covariance is what the correctness-oriented
families want, so that a failure is attributable to the active-set logic rather than to
arithmetic. The ill-conditioned end is a robustness family, and #33 owns it.
"""


def synthetic_market(
    assets: int,
    *,
    seed: int,
    lam: float = LAMBDA,
    rank: int | None = None,
    condition: float = CONDITION,
    volatility: float = 0.2,
) -> MeanStdPortfolio:
    """A synthetic market as an unconstrained :class:`cosa.MeanStdPortfolio`.

    The covariance is built from its own eigendecomposition rather than from a random
    ``B.T @ B``, which is what makes both knobs exact: a rank of ``k`` gives a covariance
    of rank exactly ``k``, and a condition number of ``c`` gives nonzero eigenvalues spread
    over exactly ``c``. Drawing ``B`` at random would leave both to chance, and the
    robustness families of #33 need to *set* them.

    The eigenvectors come from the QR factorization of a Gaussian matrix, so they are a
    genuinely random orthonormal basis and the covariance is not accidentally diagonal --
    a diagonal covariance would make the risk term separable and hide any error in the
    conic geometry.

    Args:
        assets: the number of assets ``n``.
        seed: the seed, which makes the market reproducible.
        lam: the risk-aversion parameter.
        rank: the rank of the covariance, or ``None`` for full rank. A rank below
            ``assets`` makes the apex reachable at a nonzero portfolio.
        condition: the ratio of the largest nonzero eigenvalue to the smallest.
        volatility: the target average asset volatility, which fixes the overall scale.

    Returns:
        The market as an unconstrained instance, ready for a family to add constraints to.

    Raises:
        ProblemError: if the size, rank, condition number or volatility is out of range.
    """
    if assets < 1:
        raise ProblemError("assets", f"a market needs at least one asset, found {assets}")
    width = assets if rank is None else rank
    if not 0 <= width <= assets:
        raise ProblemError("rank", f"expected a rank in [0, {assets}], found {width}")
    if condition < 1.0:
        raise ProblemError("condition", f"a condition number is at least 1, found {condition}")
    if volatility <= 0.0:
        raise ProblemError("volatility", f"a volatility is positive, found {volatility}")

    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(assets, assets)))
    spectrum = np.zeros(assets)
    if width:
        spectrum[:width] = np.logspace(0.0, -np.log10(condition), width) if width > 1 else np.ones(1)
    covariance = (basis * spectrum) @ basis.T
    covariance = (covariance + covariance.T) / 2.0
    trace = float(np.trace(covariance))
    if trace > 0.0:
        covariance *= assets * volatility**2 / trace
    return MeanStdPortfolio.unconstrained(
        mu=rng.normal(loc=0.08, scale=0.04, size=assets),
        Sigma=covariance,
        lam=lam,
    )


@dataclass(frozen=True, eq=False)
class PortfolioInstance:
    """One generated test problem, with everything a consumer needs to use it.

    The four fields answer four different questions, which is why they are all here rather
    than being recomputed by each caller:

    * :attr:`problem` is what a solver takes -- the authoritative SOCP, auxiliary
      variables included.
    * :attr:`portfolio` is what a *report* takes: the expected return and standard
      deviation of a solution are questions about ``x``, and only this knows the
      covariance.
    * :attr:`names` is what a working-set description takes, so Success Criterion 3's
      "interpreted in terms of the active portfolio constraints" is available on generated
      instances and not only on hand-built ones.
    * :attr:`name` is what a benchmark table takes, and it carries the seed, so a failing
      row names the instance that produced it.

    Attributes:
        name: a short identifier including the family, the size and the seed.
        portfolio: the mean-standard-deviation problem over ``x``.
        problem: the SOCP to solve. For every family but :func:`turnover` this is exactly
            ``portfolio.to_socp()``; turnover augments it.
        names: names for the rows of :attr:`problem`, in row order.
        witness: a point known to be feasible, in the coordinates of :attr:`problem`. The
            evidence behind "feasible by construction" -- and a warm start for #30 to
            begin from.
    """

    name: str
    portfolio: MeanStdPortfolio
    problem: SOCP
    names: ConstraintNames
    witness: Vector

    @property
    def num_assets(self) -> int:
        """The number of assets ``n``."""
        return self.portfolio.num_assets

    @property
    def num_auxiliary(self) -> int:
        """How many variables beyond ``(x, t)`` the problem carries -- nonzero only for turnover."""
        return self.problem.num_variables - self.num_assets - 1

    def working_set(self) -> WorkingSet:
        """The empty working set over this instance's shape, as Phase I would start it.

        Returns:
            A working set with nothing active, matching :attr:`problem`.
        """
        return WorkingSet.empty(self.problem)

    def __str__(self) -> str:
        """The instance's name and shape, for a benchmark row or a failure message."""
        return (
            f"{self.name}: {self.num_assets} assets, {self.problem.num_inequalities} inequalities, "
            f"{self.problem.num_equalities} equalities, cone {self.problem.cone.dim}"
        )


def _budget(portfolio: MeanStdPortfolio) -> MeanStdPortfolio:
    """Add eq. (8)'s budget equality ``1.T @ x = 1``.

    Args:
        portfolio: the market.

    Returns:
        The instance with the fully-invested constraint.
    """
    return portfolio.with_equalities(np.ones((1, portfolio.num_assets)), np.ones(1))


def _lower_bounds(assets: int, lower: Vector) -> tuple[Matrix, Vector]:
    """The rows of ``-x <= -l``, which is ``x >= l``.

    Args:
        assets: the number of assets.
        lower: the lower bounds.

    Returns:
        The rows and the right-hand side.
    """
    return -np.eye(assets), -lower


def _instance(
    family: str,
    assets: int,
    seed: int,
    portfolio: MeanStdPortfolio,
    names: ConstraintNames,
    witness: Vector,
    problem: SOCP | None = None,
) -> PortfolioInstance:
    """Package a family's pieces, naming the instance after its family, size and seed.

    Args:
        family: the family name.
        assets: the number of assets, for the name.
        seed: the seed, for the name.
        portfolio: the mean-standard-deviation problem.
        names: names for the problem's rows.
        witness: a feasible point in the problem's coordinates.
        problem: the SOCP, or ``None`` to take ``portfolio.to_socp()``.

    Returns:
        The instance.
    """
    socp = portfolio.to_socp() if problem is None else problem
    return PortfolioInstance(
        name=f"{family}-n{assets}-s{seed}",
        portfolio=portfolio,
        problem=socp,
        names=names,
        witness=_vector("witness", witness, size=socp.num_variables),
    )


def basic(
    assets: int = 8,
    *,
    seed: int = 0,
    lam: float = LAMBDA,
    rank: int | None = None,
    condition: float = CONDITION,
) -> PortfolioInstance:
    """Eq. (8): the long-only fully-invested portfolio, "the simplest realistic test".

        min  -mu.T @ x + lam * t
        s.t. 1.T @ x = 1,  x >= 0,  ||L @ x||_2 <= t

    The long-only bounds are written as ``-x <= 0`` so they are ordinary rows of ``A``,
    which is what puts them in reach of the working-set logic. The equal-weight portfolio
    is feasible, and the feasible set is the simplex, so the instance is bounded.

    Args:
        assets: the number of assets.
        seed: the seed for the market data.
        lam: the risk-aversion parameter.
        rank: the covariance rank, or ``None`` for full rank.
        condition: the covariance's condition number.

    Returns:
        The instance.
    """
    market = synthetic_market(assets, seed=seed, lam=lam, rank=rank, condition=condition)
    portfolio = _budget(market).with_inequalities(*_lower_bounds(assets, np.zeros(assets)))
    names = ConstraintNames(
        inequalities=tuple(f"long-only bound on asset {i}" for i in range(assets)),
        equalities=("fully invested",),
        cones=("risk",),
    )
    equal = np.full(assets, 1.0 / assets)
    return _instance("basic", assets, seed, portfolio, names, market.socp_point(equal))


def box(
    assets: int = 8,
    *,
    lower: float = 0.0,
    upper: float | None = None,
    seed: int = 0,
    lam: float = LAMBDA,
    rank: int | None = None,
    condition: float = CONDITION,
) -> PortfolioInstance:
    """Box constraints ``l <= x <= u`` on top of the budget equality.

    §10.2 (``paper.tex:805``) wants this family precisely because "many portfolio bounds
    are expected to become active": with ``n`` assets and an upper bound of ``u``, at
    least ``1/u`` positions must sit at their cap, so the active set is large by
    arithmetic rather than by luck. It is the family that stresses the add/drop rules
    hardest.

    The bounds are written as two blocks of ``A``, uppers first, so a row index maps to a
    bound without arithmetic.

    Args:
        assets: the number of assets.
        lower: the common lower bound.
        upper: the common upper bound, or ``None`` for :data:`BOX_WIDTH` times the equal
            weight -- which is the value that actually binds. See :data:`BOX_WIDTH`.
        seed: the seed for the market data.
        lam: the risk-aversion parameter.
        rank: the covariance rank, or ``None`` for full rank.
        condition: the covariance's condition number.

    Returns:
        The instance.

    Raises:
        ProblemError: if the box cannot hold a fully-invested portfolio, which needs
            ``lower <= upper`` and ``assets * lower <= 1 <= assets * upper``.
    """
    cap = BOX_WIDTH / assets if upper is None else upper
    if lower > cap:
        raise ProblemError("box", f"expected lower <= upper, found {lower} > {cap}")
    if not assets * lower <= 1.0 <= assets * cap:
        raise ProblemError(
            "box",
            f"a fully-invested portfolio needs {assets} * lower <= 1 <= {assets} * upper, "
            f"found bounds [{lower}, {cap}]",
        )
    market = synthetic_market(assets, seed=seed, lam=lam, rank=rank, condition=condition)
    portfolio = (
        _budget(market)
        .with_inequalities(np.eye(assets), np.full(assets, cap))
        .with_inequalities(*_lower_bounds(assets, np.full(assets, lower)))
    )
    names = ConstraintNames(
        inequalities=(
            *(f"upper bound on asset {i}" for i in range(assets)),
            *(f"lower bound on asset {i}" for i in range(assets)),
        ),
        equalities=("fully invested",),
        cones=("risk",),
    )
    equal = np.full(assets, 1.0 / assets)
    return _instance("box", assets, seed, portfolio, names, market.socp_point(equal))


def sector(
    assets: int = 9,
    *,
    sectors: int = 3,
    cap: float = 0.5,
    seed: int = 0,
    lam: float = LAMBDA,
    rank: int | None = None,
    condition: float = CONDITION,
) -> PortfolioInstance:
    """Sector caps ``A_sector @ x <= b_sector`` on top of eq. (8).

    Assets are dealt round-robin into sectors, so the sectors are interleaved rather than
    contiguous and a sector cap is not a bound on a contiguous slice -- which would make
    the combination with the long-only bounds trivial. §10.3 (``paper.tex:817``) wants
    "nontrivial combinations of active constraints", and interleaving is what makes the
    combination nontrivial: each sector row overlaps every part of the weight vector.

    Args:
        assets: the number of assets.
        sectors: how many sectors to deal them into.
        cap: the maximum weight per sector.
        seed: the seed for the market data.
        lam: the risk-aversion parameter.
        rank: the covariance rank, or ``None`` for full rank.
        condition: the covariance's condition number.

    Returns:
        The instance.

    Raises:
        ProblemError: if there are not enough sectors, or if the caps cannot together hold
            a fully-invested portfolio -- which needs ``sectors * cap >= 1``.
    """
    if not 1 <= sectors <= assets:
        raise ProblemError("sectors", f"expected a sector count in [1, {assets}], found {sectors}")
    if sectors * cap < 1.0:
        raise ProblemError(
            "cap",
            f"{sectors} sectors capped at {cap} can hold at most {sectors * cap}, which cannot be fully invested",
        )
    instance = basic(assets, seed=seed, lam=lam, rank=rank, condition=condition)
    membership = np.zeros((sectors, assets))
    membership[np.arange(assets) % sectors, np.arange(assets)] = 1.0
    portfolio = instance.portfolio.with_inequalities(membership, np.full(sectors, cap))
    names = ConstraintNames(
        inequalities=(*instance.names.inequalities, *(f"sector {k} cap" for k in range(sectors))),
        equalities=instance.names.equalities,
        cones=instance.names.cones,
    )
    return _instance("sector", assets, seed, portfolio, names, instance.witness)


def factor_exposure(
    assets: int = 8,
    *,
    factors: int = 2,
    width: float = 0.25,
    seed: int = 0,
    lam: float = LAMBDA,
    rank: int | None = None,
    condition: float = CONDITION,
) -> PortfolioInstance:
    """Factor bounds ``l_f <= F @ x <= u_f`` on top of eq. (8).

    ``F`` is dense and Gaussian, so its rows are correlated with each other and with the
    budget row -- which is the point. §10.4 (``paper.tex:826``) calls this "a useful test
    of correlated active constraints", and correlated active rows are exactly what makes a
    working-set matrix nearly rank-deficient, so this is the family that will find out
    whether #25's rank detection is needed and whether #12's singular-system stop fires
    in practice.

    The bounds are centred on ``F @ x_equal`` and half-width ``width``, which is what makes
    the family feasible by construction rather than by luck: independent bounds on a random
    ``F`` are infeasible far more often than not.

    Args:
        assets: the number of assets.
        factors: how many factor rows.
        width: the half-width of each factor band, in units of exposure.
        seed: the seed for the market data and for ``F``.
        lam: the risk-aversion parameter.
        rank: the covariance rank, or ``None`` for full rank.
        condition: the covariance's condition number.

    Returns:
        The instance.

    Raises:
        ProblemError: if the factor count or the band width is not positive.
    """
    if factors < 1:
        raise ProblemError("factors", f"expected at least one factor, found {factors}")
    if width <= 0.0:
        raise ProblemError("width", f"a factor band needs a positive half-width, found {width}")
    instance = basic(assets, seed=seed, lam=lam, rank=rank, condition=condition)
    exposures = np.random.default_rng((seed, _FACTOR_STREAM, factors)).normal(size=(factors, assets))
    centre = exposures @ np.full(assets, 1.0 / assets)
    portfolio = instance.portfolio.with_inequalities(
        np.vstack([exposures, -exposures]),
        np.concatenate([centre + width, -(centre - width)]),
    )
    names = ConstraintNames(
        inequalities=(
            *instance.names.inequalities,
            *(f"factor {j} upper" for j in range(factors)),
            *(f"factor {j} lower" for j in range(factors)),
        ),
        equalities=instance.names.equalities,
        cones=instance.names.cones,
    )
    return _instance("factor", assets, seed, portfolio, names, instance.witness)


def turnover(
    assets: int = 8,
    *,
    budget: float = 0.3,
    seed: int = 0,
    lam: float = LAMBDA,
    rank: int | None = None,
    condition: float = CONDITION,
) -> PortfolioInstance:
    """A turnover limit on top of eq. (8), through auxiliary variables.

    §10.5 (``paper.tex:835``) asks for turnover "using auxiliary variables and linear
    inequalities", and this is that: with ``delta`` in ``R^n`` and previous holdings
    ``x_old``,

        x - delta <= x_old,   -x - delta <= -x_old,   1.T @ delta <= budget,

    the first two force ``delta_i >= |x_i - x_old_i|`` and the third caps the total. The
    variable vector becomes ``(x, t, delta)``, which is why this family returns a problem
    that is *not* ``portfolio.to_socp()`` -- it is that problem put through
    :meth:`cosa.SOCP.augment`. No ``delta >= 0`` rows are needed: the first two blocks
    already imply it.

    ``x_old`` is drawn on the simplex, so ``x = x_old`` with ``delta = 0`` is feasible for
    any positive budget and the instance is feasible by construction. §10.5's own reason
    for caring is warm starts -- "successive portfolio rebalancing problems are naturally
    related" -- and #30 and #35 are where that gets tested.

    Args:
        assets: the number of assets.
        budget: the maximum total turnover.
        seed: the seed for the market data and the previous holdings.
        lam: the risk-aversion parameter.
        rank: the covariance rank, or ``None`` for full rank.
        condition: the covariance's condition number.

    Returns:
        The instance, whose problem carries ``assets`` auxiliary variables.

    Raises:
        ProblemError: if the turnover budget is not positive.
    """
    if budget <= 0.0:
        raise ProblemError("budget", f"a turnover budget is positive, found {budget}")
    instance = basic(assets, seed=seed, lam=lam, rank=rank, condition=condition)
    previous = np.random.default_rng((seed, _TURNOVER_STREAM, assets)).dirichlet(np.ones(assets))

    identity = np.eye(assets)
    zero = np.zeros((assets, 1))
    problem = instance.problem.augment(assets)
    problem = problem.add_inequalities(np.hstack([identity, zero, -identity]), previous)
    problem = problem.add_inequalities(np.hstack([-identity, zero, -identity]), -previous)
    problem = problem.add_inequalities(
        np.concatenate([np.zeros(assets + 1), np.ones(assets)]).reshape(1, -1),
        np.array([budget]),
    )
    names = ConstraintNames(
        inequalities=(
            *instance.names.inequalities,
            *(f"turnover up on asset {i}" for i in range(assets)),
            *(f"turnover down on asset {i}" for i in range(assets)),
            "turnover budget",
        ),
        equalities=instance.names.equalities,
        cones=instance.names.cones,
    )
    witness = np.concatenate([instance.portfolio.socp_point(previous), np.zeros(assets)])
    return _instance("turnover", assets, seed, instance.portfolio, names, witness, problem=problem)


def large(
    assets: int = 200,
    *,
    factors: int = 10,
    upper: float | None = None,
    seed: int = 0,
    lam: float = LAMBDA,
    condition: float = CONDITION,
) -> PortfolioInstance:
    """A large instance: many linear bounds, a small cone.

    §12.1 (``paper.tex:905``) wants "individual large problems" as one of its four
    comparison modes, and this is the shape a large portfolio actually has. The covariance
    is a factor model, so its rank is ``factors`` and the cone is ``Q^(1 + factors)`` --
    small and fixed -- while the box gives ``2 * assets`` linear rows. Large in the
    polyhedral dimension, tiny in the conic one.

    That asymmetry is the reason to expect anything of COSA here at all: an interior-point
    method pays for all ``2 * assets`` rows at every iteration, while an active-set method
    pays for the ones that bind. It is also, per §18 (``paper.tex:1310``), where the
    low-rank structure of a real covariance becomes exploitable.

    The box inherits :func:`box`'s default cap, which scales with the asset count, so the
    active set grows with the problem -- a large active set, not just a large problem.

    Args:
        assets: the number of assets.
        factors: the rank of the factor covariance.
        upper: the common upper bound on a weight, or ``None`` for :func:`box`'s default.
        seed: the seed for the market data.
        lam: the risk-aversion parameter.
        condition: the condition number of the factor covariance.

    Returns:
        The instance.

    Raises:
        ProblemError: if the factor count exceeds the asset count, or the box cannot hold
            a fully-invested portfolio.
    """
    if not 1 <= factors <= assets:
        raise ProblemError("factors", f"expected a factor count in [1, {assets}], found {factors}")
    instance = box(assets, upper=upper, seed=seed, lam=lam, rank=factors, condition=condition)
    return _instance("large", assets, seed, instance.portfolio, instance.names, instance.witness)


def all_families(*, seed: int = 0) -> tuple[PortfolioInstance, ...]:
    """One instance of every family, at sizes small enough to solve in a test.

    The list #19's and #31's "done when" is checked against, and the list #34's comparison
    study will iterate. :func:`large` is included at a reduced size: the point of the
    family is its shape, and its shape is the same at fifty assets as at five thousand.

    Args:
        seed: the seed handed to every family.

    Returns:
        The instances, in increasing order of polyhedral complexity.
    """
    return (
        basic(seed=seed),
        box(seed=seed),
        sector(seed=seed),
        factor_exposure(seed=seed),
        turnover(seed=seed),
        large(50, factors=5, seed=seed),
    )


# ======================================================================================
# §12.4's robustness families: instances built so that something breaks
# ======================================================================================


ILL_CONDITION: Final = 1.0e10
"""The default condition number for :func:`ill_conditioned`.

Ten orders of magnitude, which is past the point where a ``float64`` Cholesky of the
covariance loses every digit of the smallest eigenvalue: with ``eps`` around ``2.2e-16``,
a condition number of ``1e10`` leaves roughly six digits, and the *square* root taken to
form ``L`` leaves three. Chosen to be genuinely hostile while still having a well-defined
answer for a reference solver to agree about -- past ``1e14`` the instance stops having one,
and a test that cannot say what the right answer is cannot fail informatively.
"""


def nearly_redundant(
    assets: int = 8,
    *,
    gap: float = 1.0e-9,
    seed: int = 0,
    lam: float = LAMBDA,
) -> PortfolioInstance:
    """§12.4's first family: two constraint rows that differ by ``gap``.

    A box-constrained instance with one upper bound duplicated and perturbed. Both copies
    become active together at the optimum, so the working-set matrix acquires two rows that
    are identical to within ``gap`` -- and ``W_k`` is then numerically rank-deficient while
    being algebraically full rank, which is the worst of the two cases. An exactly
    duplicated row is caught by any rank test; a row that differs in the tenth digit is
    caught only by a test with the right threshold, and that threshold is #25's problem.

    This is the family that makes #25 demonstrable. Until it exists, "rank detection and
    dependent-constraint removal" has nothing to detect: the direction solve of #12 raises
    :class:`cosa.SingularKktError` on an *exactly* dependent set, and stays quiet while
    returning nonsense on a nearly dependent one.

    Args:
        assets: the number of assets.
        gap: how far the duplicate row is perturbed. Smaller is harder.
        seed: the seed for the market data.
        lam: the risk-aversion parameter.

    Returns:
        The instance. Its last inequality row is the near-duplicate of its first.

    Raises:
        ProblemError: if ``gap`` is negative.
    """
    if gap < 0.0:
        raise ProblemError("gap", f"a perturbation is non-negative, found {gap}")
    instance = box(assets, seed=seed, lam=lam)
    duplicate = instance.portfolio.A[0:1].copy()
    duplicate[0, 0] += gap
    portfolio = instance.portfolio.with_inequalities(duplicate, instance.portfolio.b[0:1])
    names = ConstraintNames(
        inequalities=(*instance.names.inequalities, "near-duplicate of upper bound on asset 0"),
        equalities=instance.names.equalities,
        cones=instance.names.cones,
    )
    return _instance("nearly-redundant", assets, seed, portfolio, names, instance.witness)


def highly_correlated(
    assets: int = 8,
    *,
    correlation: float = 1.0 - 1.0e-8,
    seed: int = 0,
    lam: float = LAMBDA,
) -> PortfolioInstance:
    """§12.4's second family: assets whose correlations are all within ``1 - eps`` of one.

    The covariance is ``(1 - rho) * I + rho * 1 1.T`` scaled to the usual volatility, so
    every pair of assets has correlation exactly ``rho``. As ``rho`` approaches one the
    matrix approaches rank one, and its smallest eigenvalue is ``1 - rho`` exactly -- so
    this family is a *controlled* approach to rank deficiency, unlike
    :func:`ill_conditioned`, which spreads the whole spectrum.

    It is the family that puts the apex within reach numerically rather than exactly. The
    factorization of #10 keeps every eigenvalue above its cut, so ``L`` has full rank and
    the tangent direction ``u`` exists -- but along the near-null directions ``||L @ x||``
    is tiny, so ``u`` is the normalization of a nearly-zero vector and its *direction* is
    determined by rounding. #17's guard does not fire, because the tail has not vanished;
    it has merely stopped meaning anything. That gap between "refused" and "meaningless" is
    what this family probes.

    Args:
        assets: the number of assets.
        correlation: the common pairwise correlation, in ``[0, 1)``.
        seed: the seed for the expected returns.
        lam: the risk-aversion parameter.

    Returns:
        The instance, long-only and fully invested.

    Raises:
        ProblemError: if the correlation is not in ``[0, 1)``. At exactly one the matrix is
            rank one and the family is no longer *nearly* singular -- use
            ``synthetic_market(rank=1)`` for that.
    """
    if not 0.0 <= correlation < 1.0:
        raise ProblemError("correlation", f"expected a correlation in [0, 1), found {correlation}")
    market = synthetic_market(assets, seed=seed, lam=lam)
    covariance = (1.0 - correlation) * np.eye(assets) + correlation * np.ones((assets, assets))
    covariance *= 0.2**2
    portfolio = MeanStdPortfolio(
        mu=market.mu,
        Sigma=covariance,
        lam=lam,
        A=-np.eye(assets),
        b=np.zeros(assets),
        E=np.ones((1, assets)),
        d=np.ones(1),
    )
    names = ConstraintNames(
        inequalities=tuple(f"long-only bound on asset {i}" for i in range(assets)),
        equalities=("fully invested",),
        cones=("risk",),
    )
    equal = np.full(assets, 1.0 / assets)
    return _instance("highly-correlated", assets, seed, portfolio, names, portfolio.socp_point(equal))


def ill_conditioned(
    assets: int = 8,
    *,
    condition: float = ILL_CONDITION,
    seed: int = 0,
    lam: float = LAMBDA,
) -> PortfolioInstance:
    """§12.4's third family: a covariance whose spectrum spans ``condition``.

    :func:`synthetic_market` already takes the knob, so this family is that knob turned to
    :data:`ILL_CONDITION` -- which is the point. The condition number is *exact* rather than
    approximate, so the family can be swept: #28's scaling work is measured by whether the
    conditioning of the assembled KKT matrix improves on these instances, and a sweep needs
    the input conditioning to be a number rather than an outcome.

    Different from :func:`highly_correlated` in a way that matters: there the whole
    pathology is one near-null direction, here it is spread across the spectrum, so a
    factorization that copes with a single tiny eigenvalue can still fail on this.

    Args:
        assets: the number of assets.
        condition: the covariance's condition number.
        seed: the seed for the market data.
        lam: the risk-aversion parameter.

    Returns:
        The instance, long-only and fully invested.
    """
    market = synthetic_market(assets, seed=seed, lam=lam, condition=condition)
    portfolio = _budget(market).with_inequalities(*_lower_bounds(assets, np.zeros(assets)))
    names = ConstraintNames(
        inequalities=tuple(f"long-only bound on asset {i}" for i in range(assets)),
        equalities=("fully invested",),
        cones=("risk",),
    )
    equal = np.full(assets, 1.0 / assets)
    return _instance("ill-conditioned", assets, seed, portfolio, names, market.socp_point(equal))


def nearly_active_cone(
    assets: int = 8,
    *,
    gap: float = 1.0e-9,
    seed: int = 0,
    lam: float = LAMBDA,
) -> PortfolioInstance:
    """§12.4's fourth family: an iterate whose conic slack is ``gap``, not zero.

    The one family whose pathology is in the *witness* rather than in the data, and that is
    not a shortcut -- it is where the pathology has to be. For eq. (7) with ``lam > 0`` the
    cone is *exactly* active at every optimum: any slack in it costs ``lam`` per unit, so
    the optimum always has ``t = sigma(x)``. A family whose optimum has a nearly-active cone
    therefore cannot be constructed. What §8.2 (``paper.tex:650``) is actually worried about
    is an *iterate* sitting inside the activation band, where "numerical tolerances can lead
    to oscillation between active and inactive states" -- so this family supplies exactly
    that iterate.

    The witness is the equal-weight portfolio with ``t`` raised by ``gap``, so its conic
    slack is ``gap`` and the cone is inside any activation tolerance coarser than that while
    being strictly interior to the cone. Handed to
    :func:`cosa.active_set.updates.activate_cones` it activates; handed to
    :func:`cosa.geometry.soc.is_interior` it is interior. Both are correct, and the
    disagreement is the hysteresis problem #29 exists to solve.

    Args:
        assets: the number of assets.
        gap: the conic slack at the witness. Smaller is deeper inside the band.
        seed: the seed for the market data.
        lam: the risk-aversion parameter.

    Returns:
        The instance, whose witness has a conic slack of exactly ``gap``.

    Raises:
        ProblemError: if ``gap`` is not positive. At zero the cone is exactly active, which
            is the ordinary case and needs no family.
    """
    if gap <= 0.0:
        raise ProblemError("gap", f"a nearly-active cone needs a positive slack, found {gap}")
    instance = basic(assets, seed=seed, lam=lam)
    witness = instance.witness.copy()
    witness[-1] += gap
    return _instance(
        "nearly-active-cone",
        assets,
        seed,
        instance.portfolio,
        instance.names,
        witness,
        problem=instance.problem,
    )


def degenerate_optimum(assets: int = 8, *, seed: int = 0, lam: float = LAMBDA) -> PortfolioInstance:
    """§12.4's fifth family: an optimum whose active constraints outnumber the variables.

    Built to be degenerate by arithmetic rather than by tuning. The box cap is set to
    exactly ``1 / assets``, so ``x_i <= 1/n`` for every ``i`` together with ``1.T @ x = 1``
    admits precisely one point -- the equal-weight portfolio -- and every one of the ``n``
    upper bounds is active there, alongside the budget equality. That is ``n + 1``
    active constraints on ``n + 1`` variables, of which only ``n`` are independent: the
    budget row is the sum of the upper-bound rows at the optimum.

    So the primal solution is unique and the *multipliers are not*, which is the textbook
    definition of primal degeneracy and the exact condition
    :class:`cosa.SingularKktError` was written for. This family is what demonstrates that
    #12's refusal to guess is a real behaviour rather than a defensive branch, and what
    #25 has to make survivable.

    Args:
        assets: the number of assets.
        seed: the seed for the market data. The optimum does not depend on it -- the
            feasible set is a single point -- which is itself a useful property: the
            degeneracy is not a lucky draw.
        lam: the risk-aversion parameter.

    Returns:
        The instance, whose feasible set in ``x`` is the single equal-weight point.
    """
    instance = box(assets, upper=1.0 / assets, seed=seed, lam=lam)
    return _instance(
        "degenerate-optimum",
        assets,
        seed,
        instance.portfolio,
        instance.names,
        instance.witness,
        problem=instance.problem,
    )


def many_active_bounds(
    assets: int = 20,
    *,
    slack: float = 1.1,
    seed: int = 0,
    lam: float = LAMBDA,
) -> PortfolioInstance:
    """§12.4's sixth family: most bounds active, without being degenerate.

    The box cap is ``slack / assets``, a hair above the ``1 / assets`` that would pin the
    portfolio completely. So the feasible set has interior, the optimum is not forced, and
    yet almost every upper bound binds -- with a cap of ``1.1 / assets`` at most
    ``assets / 1.1`` positions can be at it, so at least ninety per cent of the weight sits
    on bounds.

    The distinction from :func:`degenerate_optimum` is the whole point of having both. That
    family tests what happens when the active set is *dependent*; this one tests what
    happens when it is merely *large*. An active-set method can be perfectly correct on the
    second and fail on the first, and a single family conflating them would not say which.

    Args:
        assets: the number of assets. Larger is more punishing, and this family defaults
            larger than the others because its cost is the size of the active set.
        slack: how much room above ``1 / assets`` the cap leaves, as a multiple. Must
            exceed one.
        seed: the seed for the market data.
        lam: the risk-aversion parameter.

    Returns:
        The instance.

    Raises:
        ProblemError: if ``slack`` is not greater than one, which would make the instance
            degenerate or infeasible rather than merely tight.
    """
    if slack <= 1.0:
        raise ProblemError(
            "slack",
            f"expected slack > 1 so the box has interior, found {slack} -- at exactly 1 use degenerate_optimum",
        )
    instance = box(assets, upper=slack / assets, seed=seed, lam=lam)
    return _instance(
        "many-active-bounds",
        assets,
        seed,
        instance.portfolio,
        instance.names,
        instance.witness,
        problem=instance.problem,
    )


def all_robustness(*, seed: int = 0) -> tuple[PortfolioInstance, ...]:
    """One instance of each of §12.4's six families.

    The list #33's "done when" is checked against, and the one #36's failure-mode study
    will iterate. Sizes are small enough to solve inside a test, because a pathology does
    not need to be large to be a pathology -- :func:`many_active_bounds` is the exception
    and keeps its larger default, since its whole content is the size of the active set.

    Args:
        seed: the seed handed to every family.

    Returns:
        The six instances, in the order §12.4 lists them.
    """
    return (
        nearly_redundant(seed=seed),
        highly_correlated(seed=seed),
        ill_conditioned(seed=seed),
        nearly_active_cone(seed=seed),
        degenerate_optimum(seed=seed),
        many_active_bounds(seed=seed),
    )


@dataclass(frozen=True, eq=False)
class Diagnosis:
    """What makes one instance hard, measured rather than asserted.

    §12.4 asks that each robustness family be "runnable as a standalone diagnostic against
    the current solver". There is no solver yet -- that is #20 -- so what runs against the
    instance is the measurement itself: the numbers that say *which* pathology is present
    and how severe it is. Each robustness family's test then asserts its own number, so a
    family that stops being pathological (because a default moved, as the box cap once did)
    fails rather than quietly passing.

    Attributes:
        instance: the instance's name.
        status: what the reference solver said.
        objective: the optimal value, or a non-finite value if there is not one.
        covariance_condition: the ratio of largest to smallest nonzero eigenvalue of
            ``Sigma``, or infinity if it is singular.
        covariance_rank: the numerical rank of ``Sigma``.
        active_rows: how many inequality rows are active at the point examined.
        active_rank: the numerical rank of those rows stacked with ``E``. Below
            :attr:`active_rows` plus the equality count means the active set is dependent.
        independent_rows: how many rows the active set *would* need for
            :attr:`active_rank` to be full -- the active rows plus every equality.
        conic_slack: ``t - ||L @ x||`` at the point examined.
        risk: ``sigma(x)`` at the point examined. A tiny value means the point is near the
            apex, where the tangent direction is ill-determined.
    """

    instance: str
    status: str
    objective: float
    covariance_condition: float
    covariance_rank: int
    active_rows: int
    active_rank: int
    independent_rows: int
    conic_slack: float
    risk: float

    @property
    def is_primal_degenerate(self) -> bool:
        """Are the active constraints linearly dependent?

        The condition that makes the multipliers non-unique, and so the condition
        :class:`cosa.SingularKktError` reports and #25 has to remove.
        """
        return self.active_rank < self.independent_rows

    def __str__(self) -> str:
        """One line naming the instance and every measured pathology."""
        degenerate = " DEGENERATE" if self.is_primal_degenerate else ""
        return (
            f"{self.instance}: {self.status} obj={self.objective:.6g} "
            f"cond(Sigma)={self.covariance_condition:.2e} rank={self.covariance_rank} "
            f"active={self.active_rows}(rank {self.active_rank}/{self.independent_rows}) "
            f"conic slack={self.conic_slack:.2e} risk={self.risk:.2e}{degenerate}"
        )


def diagnose(
    instance: PortfolioInstance,
    at: Vector | None = None,
    *,
    tolerance: float = 1.0e-6,
) -> Diagnosis:
    """Measure what makes ``instance`` hard, at its optimum or at a given point.

    Args:
        instance: the instance to diagnose.
        at: the point to examine, or ``None`` for the reference solver's optimum. Pass
            :attr:`PortfolioInstance.witness` for a family whose pathology is in the
            iterate rather than in the data -- :func:`nearly_active_cone` is the one.
        tolerance: how close to its bound a row must be to count as active.

    Returns:
        The diagnosis.

    Raises:
        SolverUnavailableError: if ``at`` is ``None`` and no reference solver is available.
    """
    from cosa.experiments.reference import solve_reference

    problem = instance.problem
    if at is None:
        solution = solve_reference(problem)
        status, objective = solution.status, solution.objective
        point = instance.witness if solution.z is None else solution.z
    else:
        point = _vector("at", at, size=problem.num_variables)
        status, objective = "given", float(problem.c @ point)

    eigenvalues = np.linalg.eigvalsh(instance.portfolio.Sigma)
    cut = max(eigenvalues.max(initial=0.0), 0.0) * instance.num_assets * float(np.finfo(np.float64).eps)
    positive = eigenvalues[eigenvalues > cut]
    condition = float(positive.max() / positive.min()) if positive.size else math.inf

    active = np.abs(problem.b - problem.A @ point) <= tolerance * np.maximum(1.0, np.abs(problem.b))
    rows = np.vstack([problem.A[active], problem.E])
    slack = problem.cone_slack(point)
    return Diagnosis(
        instance=instance.name,
        status=status,
        objective=objective,
        covariance_condition=condition if positive.size == instance.num_assets else math.inf,
        covariance_rank=int(positive.size),
        active_rows=int(active.sum()),
        active_rank=int(np.linalg.matrix_rank(rows)) if rows.size else 0,
        independent_rows=int(active.sum()) + problem.num_equalities,
        conic_slack=float(slack[0] - np.linalg.norm(slack[1:])),
        risk=instance.portfolio.std(point[: instance.num_assets]),
    )


def badly_scaled(
    assets: int = 8,
    *,
    weight_unit: float = 1.0e-4,
    risk_unit: float = 1.0e6,
    seed: int = 0,
    lam: float = LAMBDA,
) -> PortfolioInstance:
    """An instance whose *units* disagree, which is what §13.3's five targets are about.

    Not one of §12.4's six families. It belongs to
    [#28](https://github.com/tschm/cosa/issues/28), because measuring that issue's work
    needs an instance where scaling has something to do -- and, as
    :func:`ill_conditioned` turns out to demonstrate, an ill-conditioned *covariance* is
    not such an instance. See :mod:`cosa.linear_algebra.scaling` for why.

    The pathology is a modelling one rather than a numerical one, and entirely realistic:
    weights expressed in basis points instead of fractions, and the risk variable in
    millions of currency instead of the same units as the returns. Neither is a mistake
    anyone would notice by reading the model, and together they spread the constraint
    matrix's entries over ten orders of magnitude. That is precisely the situation §13.3
    lists "portfolio variables", "linear constraints" and "SOC variables" as separate
    targets for.

    Args:
        assets: the number of assets.
        weight_unit: the unit the weights are expressed in. The default of ``1e-4`` is
            basis points, so a coefficient of ``1`` becomes ``1e4``.
        risk_unit: the unit the risk variable ``t`` is expressed in.
        seed: the seed for the market data.
        lam: the risk-aversion parameter.

    Returns:
        The instance -- the same problem as :func:`box` up to a change of units, so its
        optimum corresponds exactly and a solver that handles both must agree.

    Raises:
        ProblemError: if either unit is not positive.
    """
    if weight_unit <= 0.0 or risk_unit <= 0.0:
        raise ProblemError("units", f"units are positive, found {weight_unit} and {risk_unit}")
    instance = box(assets, seed=seed, lam=lam)
    problem = instance.problem
    units = np.append(np.full(assets, weight_unit), risk_unit)
    rescaled = SOCP(
        c=problem.c * units,
        A=problem.A * units,
        b=problem.b,
        E=problem.E * units,
        d=problem.d,
        G=problem.G * units,
        h=problem.h,
        cone=problem.cone,
    )
    return PortfolioInstance(
        name=f"badly-scaled-n{assets}-s{seed}",
        portfolio=instance.portfolio,
        problem=rescaled,
        names=instance.names,
        witness=instance.witness / units,
    )
