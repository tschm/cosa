"""The efficient frontier, solved twice: cold, and each point warm started from the last.

§11 (``paper.tex:842``) calls this "one of the principal numerical experiments", and #35 is
it. The sequence is

    min  -mu.T @ x + lam_k * sigma(x)      for lam_1, ..., lam_N,

which traces the risk-return frontier. §11's observation is that for consecutive ``lam_k``
and ``lam_(k+1)`` "the optimal portfolios and active constraints are expected to be
related", and the experiment is whether that expectation is worth anything.

**Only ``c`` changes along the sequence.** ``A``, ``E``, ``G`` and the cone product are
identical from point to point -- ``lam`` enters eq. (7) as the coefficient of ``t`` in the
objective and nowhere else. So a working set transfers exactly, in the sense
:meth:`cosa.solver.warm.WarmStart.fits` checks: the row indices still name the same rows.
That is the structural fact the whole hypothesis rests on, and it is worth stating because
it is *not* generic. A sequence that perturbed ``Sigma`` instead would change ``G`` and the
transfer would be a guess rather than a fact.

**The comparison is paired, and that is the only honest way to run it.** Each ``lam`` is
solved twice, cold and warm, against the same instance in the same process. Comparing a
cold sweep against a warm sweep in aggregate would confound the warm start with everything
else that differs between two runs; solving the same point both ways and differencing
leaves only the thing under test.

**Seven quantities, because §11 names seven.** :class:`Point` carries six of them per
frontier point and the seventh, "iterations saved by warm starts", is a function of two
solves rather than a property of one -- which is why
:func:`cosa.solver.instrumentation.iterations_saved` is a function and not a field. A
negative saving is reported rather than clamped: a warm start that costs *more* iterations
than a cold one is a finding, and this experiment exists partly to find out whether that
happens.

**What a warm start may not do is change the answer.** :meth:`Point.agrees` is checked at
every point, and a disagreement is a bug rather than a trade-off. The saving is in how the
answer was reached.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

import numpy as np

from cosa.experiments import portfolio as families
from cosa.linear_algebra.reuse import Reuse
from cosa.solver import cosa as solver
from cosa.solver.instrumentation import Metrics, iterations_saved
from cosa.solver.warm import from_solution

if TYPE_CHECKING:
    from cosa.experiments.portfolio import PortfolioInstance
    from cosa.problem.portfolio import MeanStdPortfolio
    from cosa.problem.socp import SOCP

__all__ = [
    "AGREEMENT",
    "LAMBDAS",
    "Frontier",
    "Point",
    "report",
    "risk_aversions",
    "sweep",
]

LAMBDAS: Final = 24
"""How many frontier points the default sweep traces.

Enough that consecutive problems are genuinely close -- the whole hypothesis is about
*nearby* problems, and a sweep of four would be testing something else -- and few enough
that the experiment runs inside a test suite.
"""

AGREEMENT: Final = 1e-7
"""How far a warm-started objective may differ from its cold counterpart.

Absolute, and tighter than §6's termination tolerance by an order of magnitude: the two
solves are of the *same problem*, so their objectives should agree to rather better than
either agrees with the true optimum. A gap at the termination tolerance would mean the warm
start had stopped somewhere else, which is exactly what this is watching for.
"""


def risk_aversions(low: float = 0.5, high: float = 8.0, count: int = LAMBDAS) -> tuple[float, ...]:
    """The ``lam`` sequence, geometrically spaced.

    Geometric rather than linear because ``lam`` is a *ratio* -- return traded against risk --
    so the interesting structure is spread evenly in its logarithm. A linear sweep of the
    same range spends most of its points where the frontier is nearly flat, which is where
    warm starting is easiest and so is the least informative place to measure it.

    Args:
        low: the smallest risk aversion.
        high: the largest.
        count: how many points.

    Returns:
        The sequence, ascending.
    """
    return tuple(float(value) for value in np.geomspace(low, high, count))


@dataclass(frozen=True, eq=False)
class Point:
    """One frontier point, solved both ways.

    Attributes:
        lam: the risk aversion.
        cold: the metrics of the cold solve.
        warm: the metrics of the warm-started solve.
        objective: the cold solve's objective, which is the reference value.
        gap: how far the warm solve's objective is from it.
        expected_return: ``mu.T @ x`` at the solution, §12.2's accuracy metric.
        deviation: ``sigma(x)`` there, the other one.
        status: the warm solve's status.
    """

    lam: float
    cold: Metrics
    warm: Metrics
    objective: float
    gap: float
    expected_return: float
    deviation: float
    status: str

    @property
    def saved(self) -> int:
        """§11's seventh quantity: ``cold.iterations - warm.iterations``."""
        return iterations_saved(self.cold, self.warm)

    @property
    def churn(self) -> int:
        """How many rows the warm solve had to add or drop.

        Zero means the working set carried in was already the right one, which is the case
        warm starting is *for*. A nonzero count means the belief was wrong and the loop spent
        iterations correcting it -- and correcting a belief is more expensive than acquiring
        one, which is the asymmetry that decides whether a sweep saves anything.
        """
        return self.warm.constraints_added + self.warm.constraints_removed

    @property
    def agrees(self) -> bool:
        """Did warm starting leave the answer alone?

        The precondition for every other number here meaning anything. A warm start that
        changed the answer would be a bug, not a speed-up.
        """
        return self.gap <= AGREEMENT

    def __str__(self) -> str:
        """One table row: six of §11's seven quantities, plus the seventh as ``saved``.

        Wall clock is deliberately absent, and it is the one quantity §11 names that is
        missing. This row is written into a *committed* artifact, and the point of committing
        one is that its diff shows when the numbers change -- a timing column churns on every
        run and drowns that signal. §12.3's benchmark table is where wall clock belongs, and
        it is honest there because nobody commits it expecting a clean diff.
        """
        return (
            f"lam={self.lam:7.3f}  {self.status:9s} "
            f"iters {self.cold.iterations:5d} -> {self.warm.iterations:5d} (saved {self.saved:+5d})  "
            f"+{self.warm.constraints_added:3d}/-{self.warm.constraints_removed:3d} rows  "
            f"fact {self.cold.factorizations:4d} -> {self.warm.factorizations:3d}  "
            f"residual {self.warm.kkt_residual:8.1e}  "
            f"return {self.expected_return:8.4f}  sd {self.deviation:7.4f}"
        )


@dataclass(frozen=True, eq=False)
class Frontier:
    """A whole sweep, and the totals §11 asks to compare.

    Attributes:
        points: one per risk aversion, in sweep order.
    """

    points: tuple[Point, ...]

    @property
    def cold_iterations(self) -> int:
        """Total active-set iterations solving every point from scratch."""
        return sum(point.cold.iterations for point in self.points)

    @property
    def warm_iterations(self) -> int:
        """Total active-set iterations solving every point from the last one's answer."""
        return sum(point.warm.iterations for point in self.points)

    @property
    def saved(self) -> int:
        """§11's seventh quantity over the whole sweep."""
        return self.cold_iterations - self.warm_iterations

    @property
    def share(self) -> float:
        """The saving as a fraction of the cold cost, which is the headline number."""
        return self.saved / self.cold_iterations if self.cold_iterations else 0.0

    @property
    def agrees(self) -> bool:
        """Did every point agree between the two routes?"""
        return all(point.agrees for point in self.points)

    @property
    def stable(self) -> tuple[Point, ...]:
        """The points where the warm start's believed working set turned out to be right."""
        return tuple(point for point in self.points if not point.churn)

    @property
    def changed(self) -> tuple[Point, ...]:
        """The points where it was wrong and the loop had to correct it."""
        return tuple(point for point in self.points if point.churn)

    def saving_on(self, points: Sequence[Point]) -> float:
        """The share of cold iterations saved on a subset, which is where the structure is.

        Args:
            points: :attr:`stable` or :attr:`changed`, ordinarily.

        Returns:
            The fraction saved, negative when warm starting cost more than it saved.
        """
        cold = sum(point.cold.iterations for point in points)
        return sum(point.saved for point in points) / cold if cold else 0.0

    @property
    def is_monotone(self) -> bool:
        """Does risk fall as ``lam`` rises, which is what a frontier means?

        Not a property of the solver but of the *answers*, and the cheapest available check
        that the sweep traced a frontier rather than a sequence of unrelated optima. A
        violation would mean one of the points is wrong however good its residuals look.
        """
        deviations = [point.deviation for point in self.points]
        return all(later <= earlier + 1e-9 for earlier, later in itertools.pairwise(deviations))

    def totals(self) -> Metrics:
        """The warm sweep's metrics summed, for a benchmark row.

        Returns:
            One :class:`cosa.solver.instrumentation.Metrics` whose counters are the sweep's
            totals. The residual is the *worst* rather than the last, because a sweep is only
            as good as its least converged point.
        """
        warm = [point.warm for point in self.points]
        return Metrics(
            iterations=sum(metrics.iterations for metrics in warm),
            constraints_added=sum(metrics.constraints_added for metrics in warm),
            constraints_removed=sum(metrics.constraints_removed for metrics in warm),
            cone_changes=sum(metrics.cone_changes for metrics in warm),
            factorizations=sum(metrics.factorizations for metrics in warm),
            kkt_solves=sum(metrics.kkt_solves for metrics in warm),
            runtime=sum(metrics.runtime for metrics in warm),
            factorization_time=sum(metrics.factorization_time for metrics in warm),
            kkt_residual=max((metrics.kkt_residual for metrics in warm), default=float("nan")),
            working_set_revisits=max((metrics.working_set_revisits for metrics in warm), default=0),
        )

    def __str__(self) -> str:
        """The sweep's headline: work saved, and whether the answers survived it."""
        agreement = "answers agree" if self.agrees else "ANSWERS DISAGREE"
        return (
            f"frontier: {len(self.points)} points, "
            f"{self.cold_iterations} cold iterations -> {self.warm_iterations} warm "
            f"({self.share:.0%} saved), {agreement}\n"
            f"  {len(self.stable)} point(s) with the working set unchanged: "
            f"{self.saving_on(self.stable):+.0%}\n"
            f"  {len(self.changed)} point(s) that had to correct it:        "
            f"{self.saving_on(self.changed):+.0%}"
        )


def _at(portfolio: MeanStdPortfolio, lam: float) -> SOCP:
    """The SOCP for one risk aversion.

    Args:
        portfolio: the instance's eq. (1) form.
        lam: the risk aversion to substitute.

    Returns:
        The SOCP, which differs from its neighbours in ``c`` and nothing else.
    """
    return replace(portfolio, lam=lam).to_socp()


def sweep(
    instance: PortfolioInstance | None = None,
    lams: Sequence[float] | None = None,
    *,
    assets: int = 10,
    seed: int = 0,
) -> Frontier:
    """Solve the frontier twice and measure the difference.

    Args:
        instance: the portfolio to trace, or ``None`` for a box-constrained one. Box rather
            than basic: bounds are what make the active set non-trivial, and a frontier whose
            working set is empty at every point would show warm starting saving nothing
            because there was nothing to carry.
        lams: the risk aversions, or ``None`` for :func:`risk_aversions`.
        assets: how many assets, when building the default instance.
        seed: the draw, when building the default instance.

    Returns:
        The sweep.
    """
    instance = instance if instance is not None else families.box(assets, seed=seed)
    sequence = tuple(lams) if lams is not None else risk_aversions()
    cache = Reuse()
    hint = None
    points = []
    for lam in sequence:
        problem = _at(instance.portfolio, lam)
        cold = solver.solve(problem)
        hot = solver.solve(problem, warm=hint)
        portfolio = replace(instance.portfolio, lam=lam)
        holdings = hot.z[: instance.num_assets]
        points.append(
            Point(
                lam=lam,
                cold=cold.metrics,
                warm=hot.metrics,
                objective=cold.objective(problem),
                gap=abs(hot.objective(problem) - cold.objective(problem)),
                expected_return=portfolio.expected_return(holdings),
                deviation=portfolio.std(holdings),
                status=hot.status,
            )
        )
        hint = from_solution(hot, cache=cache)
    return Frontier(points=tuple(points))


def report(
    instance: PortfolioInstance | None = None,
    lams: Sequence[float] | None = None,
    *,
    assets: int = 10,
    seed: int = 0,
) -> str:
    """The sweep as text, with §11's seven quantities per point and the totals beneath.

    Args:
        instance: the portfolio to trace, or ``None`` for the default.
        lams: the risk aversions, or ``None`` for :func:`risk_aversions`.
        assets: how many assets, when building the default instance.
        seed: the draw, when building the default instance.

    Returns:
        The report.
    """
    traced = sweep(instance, lams, assets=assets, seed=seed)
    lines = [str(traced), ""]
    lines += [str(point) for point in traced.points]
    totals = traced.totals()
    lines += [
        "",
        f"warm totals: {totals.iterations} iters, {totals.active_set_changes} active-set changes, "
        f"{totals.factorizations} factorizations, {totals.kkt_solves} KKT solves, "
        f"worst residual {totals.kkt_residual:.3g}",
        f"monotone frontier: {traced.is_monotone}",
    ]
    return "\n".join(lines)
