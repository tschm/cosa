"""The front door: solve a portfolio without knowing what an SOCP is.

Deliverable 5 (``paper.tex:1250``) asks for a "public portfolio optimization interface",
"distinct from the internal SOCP builder", and #37 is it. Everything else in this package is
addressed to someone implementing or studying a conic active-set method; this module is
addressed to someone with a covariance matrix and a deadline.

**One function and one result type.** :func:`solve_portfolio` takes eq. (1) as a person
would write it -- expected returns, a covariance, a risk aversion, and whatever linear
constraints the mandate imposes -- and returns holdings. The reduction through eq. (2),
eq. (7) and the general SOCP happens inside and is not the caller's problem. Nothing here
adds capability: it removes the requirement to know where ``t`` goes, why the cone's head
row is what it is, and which of the twelve modules to import.

**The result reports in the units the question was asked in.** A caller who supplied a
covariance wants a *standard deviation* back, not the value of ``c.T @ z`` over a lifted
variable that includes an auxiliary. :class:`Portfolio` carries both -- the holdings, the
expected return, the risk, and the utility -- alongside enough of the solver's own state
(:attr:`Portfolio.status`, :attr:`Portfolio.residuals`) to tell whether to believe them.

**Two things that are decisions rather than conveniences**, and both are here because the
study that preceded this had to make them and a caller should not have to.

* :func:`solve_portfolio` **does not equilibrate by default**, and arriving at that took
  changing the answer twice. #36's study found one family COSA could not solve as given --
  ``badly scaled``, whose constraint matrix spans fourteen orders of magnitude -- and §13.3's
  Ruiz equilibration rescued it, which made equilibration look like the obvious default for a
  front door. Building the front door then exposed why that family was failing, and it was
  not conditioning: :func:`cosa.solver.initialization.raise_free_heads` required a cone head
  row to select its variable with a coefficient of *exactly one*, which no rescaled instance
  satisfies, so the retraction was silently unavailable and an iterate on the cone's boundary
  could not move at all. With that restriction removed the family solves to ``1e-11`` with no
  equilibration -- and equilibrating it makes the residual *worse*, by five orders of
  magnitude, while costing between 30% and 80% more iterations on every other family. So the
  default is off, and ``scale=True`` is there for a caller whose units are genuinely
  pathological.
* it **refuses rather than returns** on a non-optimal solve, unless asked not to. The
  solver's own :class:`cosa.solver.cosa.Solution` reports a status and lets the caller
  decide, which is right for a study; a portfolio interface that silently returned holdings
  from a stalled solve would be handing someone a position to trade. :exc:`NotOptimalError`
  carries the status and the residuals so the caller can decide anyway.

**Warm starting is exposed, because it is the point.** :meth:`Portfolio.warm` hands back
what the next solve needs, so a frontier sweep is a loop over ``lam`` and nothing else. What
that saves, and the condition under which it saves anything, is #35 and
``docs/development/failure-modes.md``'s neighbour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from cosa.linear_algebra.scaling import Scaling, equilibrate, identity
from cosa.problem.portfolio import MeanStdPortfolio
from cosa.problem.socp import ProblemError
from cosa.solver import cosa as solver
from cosa.solver.warm import WarmStart

if TYPE_CHECKING:
    from cosa import Matrix, Vector
    from cosa.linear_algebra.reuse import Reuse
    from cosa.solver.instrumentation import Metrics
    from cosa.solver.termination import Residuals

__all__ = [
    "NotOptimalError",
    "Portfolio",
    "solve_portfolio",
]


class NotOptimalError(RuntimeError):
    """The solve did not reach a certified optimum.

    Raised rather than returned, because the alternative is handing someone a position to
    trade that the solver does not stand behind. The status and the residuals are attached
    so that a caller who wants to look anyway can, and
    :func:`solve_portfolio` takes ``strict=False`` for callers who would rather have the
    answer and judge it themselves.

    Attributes:
        result: the portfolio that was rejected, so that nothing is lost by raising.
    """

    def __init__(self, result: Portfolio) -> None:
        """Report the status and the worst residual.

        Args:
            result: the rejected portfolio.
        """
        self.result = result
        super().__init__(
            f"the solve finished {result.status!r} rather than optimal, with a worst residual of "
            f"{result.residuals.largest:.3g}. Pass strict=False to receive it anyway and judge it "
            f"yourself; result.residuals says which condition failed"
        )


@dataclass(frozen=True, eq=False)
class Portfolio:
    """The answer, in the units the question was asked in.

    Attributes:
        holdings: the optimal ``x``, one weight per asset.
        expected_return: ``mu.T @ x``.
        risk: ``sigma(x) = sqrt(x.T @ Sigma @ x)``, the standard deviation. Not the variance:
            eq. (1) is written in standard deviation, which is the whole reason the problem
            is conic rather than quadratic, and returning the variance here would quietly
            change units between the formulation and its answer.
        utility: ``mu.T @ x - lam * sigma(x)``, the quantity eq. (1) maximizes.
        status: the solver's own status. ``"optimal"`` unless ``strict=False`` was passed.
        residuals: §6's five conic KKT residuals, which are what certify the answer.
        metrics: what the solve cost, for a caller who is timing things.
        active: the names of the constraints active at the solution -- Success Criterion 3's
            "interpretable in terms of the active portfolio constraints".
    """

    holdings: Vector
    expected_return: float
    risk: float
    utility: float
    status: str
    residuals: Residuals
    metrics: Metrics
    active: str

    @property
    def is_optimal(self) -> bool:
        """Did the solve reach a certified optimum?"""
        return self.status == "optimal" and self.residuals.is_optimal()

    def warm(self) -> WarmStart:
        """What the next solve in a sequence should be handed.

        Exposed because warm starting is the reason to use an active-set method at all. A
        frontier sweep is then a loop over ``lam`` and nothing else.

        Returns:
            The warm start.

        Raises:
            ProblemError: if this portfolio was built without a solver solution behind it,
                which cannot happen through :func:`solve_portfolio`.
        """
        if self._warm is None:  # pragma: no cover - unreachable through the public entry
            raise ProblemError("warm", "this portfolio carries no solver state to warm start from")
        return self._warm

    _warm: WarmStart | None = None

    def __str__(self) -> str:
        """One line a caller can print: return, risk, utility and how it went."""
        return (
            f"portfolio: return {self.expected_return:.4%}, risk {self.risk:.4%}, "
            f"utility {self.utility:.6f} [{self.status}, residual {self.residuals.largest:.2g}]"
        )


def solve_portfolio(
    mu: Vector,
    Sigma: Matrix,  # noqa: N803 - the covariance is named as finance names it
    lam: float,
    *,
    inequalities: tuple[Matrix, Vector] | None = None,
    equalities: tuple[Matrix, Vector] | None = None,
    budget: bool = True,
    long_only: bool = False,
    scale: bool = False,
    strict: bool = True,
    warm: WarmStart | None = None,
    cache: Reuse | None = None,
) -> Portfolio:
    """Maximize ``mu.T @ x - lam * sigma(x)`` subject to linear constraints.

    Eq. (1), taken as a person would write it. The reduction to eq. (2), eq. (7) and the
    general SOCP happens inside.

    Args:
        mu: expected returns, one per asset.
        Sigma: the covariance matrix, symmetric and positive semidefinite. Rank deficiency
            is fine and is not a special case -- see ``docs/development/failure-modes.md``.
        lam: the risk-aversion parameter, strictly positive. Larger means more risk averse.
        inequalities: ``(A, b)`` for ``A @ x <= b``, or ``None``.
        equalities: ``(E, d)`` for ``E @ x = d``, or ``None``. Combined with ``budget``
            rather than replacing it.
        budget: whether to add ``sum(x) == 1``, the constraint almost every mandate has.
            Set ``False`` for a long-short book with a different normalization.
        long_only: whether to add ``x >= 0``. Off by default because eq. (1) does not
            require it and a long-short mandate is as ordinary as a long-only one.
        scale: whether to apply §13.3's Ruiz equilibration first. **Off** by default -- see
            the module docstring, which records why the obvious choice was the wrong one.
        strict: whether to raise on a non-optimal solve rather than returning it.
        warm: a previous :meth:`Portfolio.warm`, for a sequence of related problems.
        cache: a factorization cache carried across a sequence, from a previous
            :attr:`Portfolio.metrics` run. Ordinarily obtained by passing the same
            :class:`cosa.linear_algebra.reuse.Reuse` to every call in the sequence.

    Returns:
        The portfolio.

    Raises:
        NotOptimalError: if the solve did not reach a certified optimum and ``strict``.
        ProblemError: if the inputs do not describe a well-posed instance -- a non-square
            covariance, a non-positive ``lam``, mismatched constraint shapes.
    """
    portfolio = _assemble(mu, Sigma, lam, inequalities, equalities, budget=budget, long_only=long_only)
    problem = portfolio.to_socp()
    factors = equilibrate(problem) if scale else identity(problem)
    answer = solver.solve(
        factors.apply(problem),
        warm=_rescaled(warm, factors),
        reuse=cache if cache is not None else True,
    )
    z = factors.unscale_point(answer.z)
    holdings = z[: portfolio.num_assets]
    result = Portfolio(
        holdings=holdings,
        expected_return=portfolio.expected_return(holdings),
        risk=portfolio.std(holdings),
        utility=portfolio.utility(holdings),
        status=answer.status,
        residuals=answer.residuals,
        metrics=answer.metrics,
        active=answer.working_set.describe(),
        # The warm start carries the *unscaled* point. Equilibration depends on `c`, so two
        # calls in a sequence generally scale differently, and a point stored in one call's
        # scaled variables would mean something else in the next one's. Storing it in the
        # caller's variables makes it portable, and `_rescaled` maps it in on the way back.
        _warm=WarmStart(z=z, working_set=answer.working_set, multipliers=answer.multipliers, cache=cache),
    )
    if strict and not result.is_optimal:
        raise NotOptimalError(result)
    return result


def _assemble(
    mu: Vector,
    Sigma: Matrix,  # noqa: N803 - see solve_portfolio
    lam: float,
    inequalities: tuple[Matrix, Vector] | None,
    equalities: tuple[Matrix, Vector] | None,
    *,
    budget: bool,
    long_only: bool,
) -> MeanStdPortfolio:
    """Build eq. (1) from the caller's pieces.

    The four sources of constraints are combined in a fixed order -- caller inequalities,
    then ``long_only``, then caller equalities, then ``budget`` -- so that a row index in
    :attr:`Portfolio.active` means the same thing on every call with the same arguments.

    Args:
        mu: expected returns.
        Sigma: the covariance.
        lam: the risk aversion.
        inequalities: ``(A, b)`` or ``None``.
        equalities: ``(E, d)`` or ``None``.
        budget: whether to add ``sum(x) == 1``.
        long_only: whether to add ``x >= 0``.

    Returns:
        The eq. (1) instance.
    """
    returns = np.asarray(mu, dtype=np.float64).reshape(-1)
    portfolio = MeanStdPortfolio.unconstrained(mu=returns, Sigma=np.asarray(Sigma, dtype=np.float64), lam=lam)
    if inequalities is not None:
        rows, rhs = inequalities
        portfolio = portfolio.with_inequalities(np.asarray(rows, dtype=np.float64), np.asarray(rhs, dtype=np.float64))
    if long_only:
        portfolio = portfolio.with_inequalities(-np.eye(returns.size), np.zeros(returns.size))
    if equalities is not None:
        rows, rhs = equalities
        portfolio = portfolio.with_equalities(np.asarray(rows, dtype=np.float64), np.asarray(rhs, dtype=np.float64))
    if budget:
        portfolio = portfolio.with_equalities(np.ones((1, returns.size)), np.ones(1))
    return portfolio


def _rescaled(warm: WarmStart | None, factors: Scaling) -> WarmStart | None:
    """Move a warm start into the scaled problem's variables.

    A warm start's point lives in the caller's variables and the solve happens in the scaled
    ones, so the point has to be mapped across. The working set does not: it names *rows*,
    and equilibration rescales rows without reordering or removing any, so an index means
    the same thing on both sides. The multipliers would need
    :meth:`cosa.linear_algebra.scaling.Scaling.unscale_multipliers` run backwards, and are
    dropped instead -- they seed #23's curvature, which re-derives them every iteration
    anyway, so the cost of dropping them is at most one iteration and the cost of getting
    the direction wrong is a wrong Hessian.

    Args:
        warm: the caller's warm start, or ``None``.
        factors: the scaling being applied.

    Returns:
        The warm start in scaled variables, or ``None``.
    """
    if warm is None:
        return None
    return WarmStart(
        z=factors.scale_point(warm.z),
        working_set=warm.working_set,
        multipliers=None,
        cache=warm.cache,
    )
