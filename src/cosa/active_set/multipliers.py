"""Reading the problem's multipliers out of the direction subproblem's, and testing signs.

The KKT system of §4.3 returns one multiplier per working-set row. §6 wants three named
things instead -- ``y >= 0`` for the inequalities, ``nu`` free for the equalities, and
``w in Q`` for the cone -- and this module is the map between them. It is a short module
with one long derivation behind it, so the derivation is written out here rather than
left for a reader to redo.

**The map.** At a point where the direction vanishes, the subproblem's stationarity row
``rho * d + W.T @ nu_k = -g`` reduces to ``g + W.T @ nu_k = 0``, and ``g`` is ``c``. Write
``W_k`` out by its three blocks -- the active rows of ``A``, all of ``E``, and the cone's
rows -- and compare with the problem's stationarity in the one fixed convention,
``c + A.T @ y + E.T @ nu - G.T @ w = 0``:

* the inequality block matches term for term, so ``y`` on the active rows *is* ``nu_k``
  there, and zero elsewhere by complementarity;
* the equality block likewise, so ``nu`` is ``nu_k``'s equality part;
* the conic block does not match term for term, because the cone's row is not a row of
  ``G``. For a tangent factor the row is ``covector.T @ G_block`` with
  ``covector = (1, -u)``, so the block contributes ``nu_cone * G_block.T @ covector``
  against the convention's ``-G_block.T @ w``, giving

      w_block = -nu_cone * covector.

  For an apex factor the rows *are* ``G_block``, and the same comparison gives
  ``w_block = -nu_block``.

**One dual feasibility test, two shapes.** ``w in Q`` is the whole of the cone's sign
condition, and applying it to the two expressions above produces exactly the two tests
§8.1 distinguishes. At a tangent factor ``covector`` sits on the boundary of ``Q``, so
``w in Q`` reduces to the *scalar* condition ``nu_cone <= 0``. At an apex factor it does
not reduce at all: the condition is that ``-nu_block`` lie in ``Q``, a cone membership
test, which is the "normal-cone condition" §8.1 (``paper.tex:644``) asks for in place of a
hyperplane. The normal cone of a self-dual cone at its apex is ``-Q``, and that is where
this comes from.

So :func:`dual_cone_violation` does not branch on the status: it computes ``w`` and asks
whether ``w`` is in ``Q``. The branch is in what ``w`` *is*, and that is
:func:`from_direction`'s business.

**The convention is read, never restated.** Every sign above is a consequence of
:data:`cosa.SIGN_CONVENTION`, which #9 fixed precisely so that this module and the residual
computation cannot drift apart. The stationarity check delegates to
:meth:`cosa.SOCP.stationarity_residual` rather than spelling the equation out again, and
the inequality removal rule is :func:`cosa.active_set.updates.removal_candidate`, reused
rather than reimplemented: this module's job is to produce a ``y``, not to have a second
opinion about what a wrong-signed one means.

**Level 2 is what this module is measured against.** §14.2 (``paper.tex:1028``) asks that
"the computed multipliers satisfy the stationarity equations to within a prescribed
tolerance", and :meth:`Multipliers.stationarity_error` is that number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from cosa.active_set.working_set import ConeStatus, WorkingSet
from cosa.geometry.soc import TOLERANCE, slack
from cosa.geometry.tangent import tangent_covector
from cosa.problem.socp import SIGN_CONVENTION, ProblemError, _vector

if TYPE_CHECKING:
    from cosa import Vector
    from cosa.linear_algebra.kkt import Direction
    from cosa.problem.socp import SOCP

__all__ = [
    "STATIONARITY_TOLERANCE",
    "Multipliers",
    "dual_cone_violation",
    "from_direction",
    "is_dual_feasible",
]

STATIONARITY_TOLERANCE: Final = 1e-8
"""§14.2's "prescribed tolerance" for the stationarity residual.

Relative to the objective's scale, as :meth:`Multipliers.stationarity_error` applies it.
Looser than the geometry module's rounding-level tolerance and tighter than the
active-set logic's decision tolerances, which is the right place for it: a stationarity
residual is arithmetic that should be nearly exact, not a decision with hysteresis.
"""


@dataclass(frozen=True, eq=False)
class Multipliers:
    """The problem's dual variables, named as §6.1 names them.

    Full-length and indexed by the problem's rows rather than by the working set's, so a
    consumer never has to know which rows were active: an inactive inequality's ``y`` is
    zero, and an inactive cone factor's ``w`` block is zero. Both are what complementarity
    requires, not padding -- an inactive constraint that carried a nonzero multiplier would
    be a violation of complementarity, so the zeros are the answer.

    Attributes:
        y: the inequality multipliers, ``(m_ineq,)``, non-negative when dual feasible.
        nu: the equality multipliers, ``(m_eq,)``, free in sign.
        w: the conic dual variable, ``(cone.dim,)``, in ``K`` when dual feasible.
    """

    y: Vector
    nu: Vector
    w: Vector

    def __post_init__(self) -> None:
        """Coerce the three blocks to finite float vectors.

        Sizes are not checked against a problem, because an instance does not carry one:
        the same multipliers are meant to be readable against the *next* problem in a
        frontier sequence, which is what #30's warm start does with them.
        """
        set_block = object.__setattr__
        set_block(self, "y", _vector("y", self.y))
        set_block(self, "nu", _vector("nu", self.nu))
        set_block(self, "w", _vector("w", self.w))

    def stationarity_residual(self, problem: SOCP) -> Vector:
        """The Level 2 residual ``c + A.T @ y + E.T @ nu - G.T @ w``.

        Delegated to :meth:`cosa.SOCP.stationarity_residual`, which is the definitional
        home of the signs. Nothing here writes them out.

        Args:
            problem: the instance these multipliers belong to.

        Returns:
            The residual, ``(n,)``.
        """
        return problem.stationarity_residual(self.y, self.nu, self.w)

    def stationarity_error(self, problem: SOCP) -> float:
        """The Level 2 error: the residual's size, relative to the objective's.

        Relative rather than absolute, because a portfolio whose returns are around
        ``1e-3`` and a factor model whose objective is around ``1e6`` should not be held to
        the same absolute residual -- and §14.2 asks for a "prescribed tolerance" without
        saying which, so the choice is made here and made once.

        Args:
            problem: the instance these multipliers belong to.

        Returns:
            ``||residual||_inf / max(1, ||c||_inf)``, which is zero exactly at
            stationarity.
        """
        residual = float(np.abs(self.stationarity_residual(problem)).max(initial=0.0))
        return residual / max(1.0, float(np.abs(problem.c).max(initial=0.0)))

    def satisfies_stationarity(self, problem: SOCP, *, tolerance: float = STATIONARITY_TOLERANCE) -> bool:
        """Is Level 2 satisfied to the prescribed tolerance?

        Args:
            problem: the instance these multipliers belong to.
            tolerance: the prescribed tolerance.

        Returns:
            ``True`` if :meth:`stationarity_error` is within tolerance.
        """
        return self.stationarity_error(problem) <= tolerance

    def inequality_violation(self, *, tolerance: float = 0.0) -> float:
        """How far the most wrong-signed inequality multiplier is from its required sign.

        The required sign is read from :data:`cosa.SIGN_CONVENTION` rather than assumed, so
        that flipping the convention flips this with it.

        Args:
            tolerance: a violation at or below this counts as none.

        Returns:
            The largest violation, zero when every multiplier has the right sign.
        """
        worst = float((-self.y * SIGN_CONVENTION.inequality).max(initial=0.0))
        return worst if worst > tolerance else 0.0

    def __str__(self) -> str:
        """The three blocks' sizes and extremes, for a log line or a failure message."""
        return (
            f"multipliers: y[{self.y.size}] in "
            f"[{float(self.y.min(initial=0.0)):.3g}, {float(self.y.max(initial=0.0)):.3g}], "
            f"nu[{self.nu.size}], w[{self.w.size}]"
        )


def from_direction(
    problem: SOCP,
    working_set: WorkingSet,
    z: Vector,
    direction: Direction,
    *,
    tolerance: float = TOLERANCE,
) -> Multipliers:
    """Map the direction subproblem's multipliers onto the problem's.

    The derivation is in the module docstring; this is it in code. The mapping is exact
    whenever the direction vanishes, which is where the multipliers are meaningful --
    away from a stationary point of the working-set-constrained problem they are still
    the subproblem's multipliers, and reading them as the problem's is an approximation
    the caller is responsible for. That is why nothing here checks ``d == 0``: §4.1's
    iteration computes multipliers at every step, and the sign tests are useful before the
    direction is zero.

    Args:
        problem: the instance.
        working_set: the set the direction was computed for.
        z: the point the direction was computed at, needed for a tangent factor's ``u``.
        direction: the solved direction, whose :attr:`~cosa.Direction.layout` says which
            multiplier belongs to which row.
        tolerance: the vanishing-tail tolerance for the tangent covector.

    Returns:
        The problem's multipliers, full-length and zero-padded on the inactive rows.

    Raises:
        ApexError: if a factor's status is :attr:`cosa.ConeStatus.TANGENT` but its slack has
            no tangent.
        ProblemError: if the working set's shape does not match the problem's, or the
            direction's layout does not match the working set.
    """
    if working_set.num_inequalities != problem.num_inequalities or working_set.cone != problem.cone:
        raise ProblemError("shape", "the working set and the problem describe different instances")
    active, equality, cones = direction.layout.split(direction.multipliers)

    y = np.zeros(problem.num_inequalities)
    y[list(direction.layout.inequalities)] = active

    w = np.zeros(problem.cone.dim)
    point = _vector("z", z, size=problem.num_variables)
    conic = problem.cone_slack(point)
    for rows, block in zip(direction.layout.cones, cones, strict=True):
        span = problem.cone.slices[rows.factor]
        match rows.status:
            case ConeStatus.TANGENT:
                covector = tangent_covector(conic[span], tolerance=tolerance)
                w[span] = -float(block[0]) * covector
            case ConeStatus.APEX:
                w[span] = -block
            case ConeStatus.INACTIVE:  # pragma: no cover - the layout omits inactive factors
                continue
    return Multipliers(y=y, nu=np.array(equality, dtype=np.float64), w=w)


def dual_cone_violation(problem: SOCP, multipliers: Multipliers) -> tuple[float, ...]:
    """How far each factor's ``w`` block falls outside ``Q``, one number per factor.

    The single dual feasibility test the module docstring derives: ``w in Q``. It does not
    branch on the working-set status, because it does not need to -- a tangent factor's
    ``w`` and an apex factor's ``w`` are both just vectors, and the question asked of them
    is the same one. What differs is that at a tangent factor the answer is determined by
    one scalar and at an apex factor by a whole block, which is the difference §8.1 is
    pointing at.

    An inactive factor's block is zero, which is in ``Q``, so it never violates -- again
    what complementarity requires rather than a special case.

    Args:
        problem: the instance, for its cone product.
        multipliers: the multipliers to test.

    Returns:
        One violation per factor, ``||w_tail|| - w_head`` clipped at zero, so zero means
        dual feasible.
    """
    blocks = problem.cone.blocks(_vector("w", multipliers.w, size=problem.cone.dim))
    return tuple(max(0.0, -slack(block)) for block in blocks)


def is_dual_feasible(
    problem: SOCP,
    multipliers: Multipliers,
    *,
    tolerance: float = STATIONARITY_TOLERANCE,
) -> bool:
    """Are the multipliers dual feasible: ``y >= 0`` and every ``w`` block in ``Q``?

    The dual half of Level 3. Stationarity is the other half and is
    :meth:`Multipliers.satisfies_stationarity`; the two are kept apart because a violation
    of each means something different -- a wrong-signed ``y`` names a constraint to drop,
    while a stationarity residual names nothing and is simply arithmetic gone wrong.

    Args:
        problem: the instance.
        multipliers: the multipliers to test.
        tolerance: how large a violation counts as none.

    Returns:
        ``True`` if both conditions hold to within the tolerance.
    """
    if multipliers.inequality_violation(tolerance=tolerance) > 0.0:
        return False
    return max(dual_cone_violation(problem, multipliers), default=0.0) <= tolerance
