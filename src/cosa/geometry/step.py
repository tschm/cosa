"""How far a step may go: the ratio test, and the interval it produces.

§5.2 (``paper.tex:471``) says the feasible step is *"determined by the intersection of: the
linear feasible interval; the SOC feasible interval; any explicit step bounds"*, and calls
that intersection "the conic analogue of the classical active-set ratio test". This module
computes it.

**Two of the three intervals are here; the middle one is #18's.** The linear interval is
the classical ratio test -- for each inactive inequality with ``a_i.T @ p > 0``,

    alpha <= (b_i - a_i.T @ x) / (a_i.T @ p),

which is arithmetic. The conic interval comes from eq. (6)'s scalar quadratic and has
genuine subtleties -- a squaring step that is only valid while the right-hand side stays
non-negative, and a degenerate case where the quadratic collapses to a linear equation --
so it is its own issue. :func:`step_limit` intersects what exists; when #18 lands it
intersects three intervals instead of two, and nothing that calls it changes.

Until then :func:`step_limit` **refuses** a problem whose cone could bind rather than
ignoring it. That is the whole reason to name the missing piece here rather than leave it
implicit: a ratio test that silently omitted the conic bound would produce iterates outside
the cone, and §14.1's Level 1 invariant would start failing several modules away from the
cause.

**What blocks matters as much as how far.** The step length alone is not enough for the
loop: §7.1 adds the constraint that "reaches its boundary", so the ratio test has to say
*which* one. :class:`StepLimit` carries the index, and the loop of §4.1 uses it directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from cosa.problem.socp import ProblemError, _vector

if TYPE_CHECKING:
    from cosa import Vector
    from cosa.active_set.working_set import WorkingSet
    from cosa.problem.socp import SOCP

__all__ = [
    "TOLERANCE",
    "StepLimit",
    "linear_step",
    "step_limit",
]

TOLERANCE: Final = 1e-12
"""How large ``a_i.T @ p`` must be for a row to count as approaching its bound.

A rounding-level threshold, not an algorithmic one. A row with ``a_i.T @ p`` of ``1e-18``
is not approaching its bound; it is a row the working set has made orthogonal to the
direction, and dividing by it produces a step length of ``1e18`` that means nothing. The
activation *decision* -- how close counts as reached -- is
:data:`cosa.active_set.updates.ACTIVATION_TOLERANCE` and is deliberately much coarser.
"""


@dataclass(frozen=True)
class StepLimit:
    """How far a step may go, and what stops it.

    Attributes:
        alpha: the largest feasible step, possibly ``inf`` when nothing stops it.
        blocking: the inequality row that stops it, or ``None`` when the limit comes from
            somewhere else -- an explicit bound, the cone, or nothing at all.
        source: what produced the limit: ``"linear"``, ``"cone"``, ``"bound"``, or
            ``"unbounded"``. Named because the loop treats them differently: a linear
            block is a constraint to add, an unbounded step is a diagnosis, and the other
            two are neither.
    """

    alpha: float
    blocking: int | None = None
    source: str = "unbounded"

    @property
    def is_unbounded(self) -> bool:
        """Is the step unlimited -- which for a descent direction means the problem is?"""
        return not math.isfinite(self.alpha)

    def tighter_of(self, other: StepLimit) -> StepLimit:
        """The smaller of two limits, keeping whichever one's provenance wins.

        Ties go to ``self``, which matters only for which ``source`` is reported and never
        for the step taken.

        Args:
            other: the limit to compare against.

        Returns:
            The tighter limit.
        """
        return self if self.alpha <= other.alpha else other

    def __str__(self) -> str:
        """The step and its cause, for a log line."""
        blocked = "" if self.blocking is None else f" by row {self.blocking}"
        return f"alpha={self.alpha:.6g} ({self.source}{blocked})"


def linear_step(
    problem: SOCP,
    z: Vector,
    d: Vector,
    working_set: WorkingSet,
    *,
    tolerance: float = TOLERANCE,
) -> StepLimit:
    """§5.2's classical ratio test over the inactive inequalities.

    Only the inactive rows are examined. The active ones satisfy ``a_i.T @ p = 0`` by
    construction -- that is what the working set imposes -- so they cannot block, and
    including them would turn their rounding-level residuals into spurious step limits.

    A negative ratio is clamped to zero rather than discarded. It means the iterate is
    already a hair outside that row, which a finite-precision step can produce, and the
    honest response is a zero step that adds the row rather than a step that goes further
    out.

    Args:
        problem: the instance.
        z: the current point, assumed feasible.
        d: the direction.
        working_set: the current set, whose inactive rows are the candidates.
        tolerance: how large ``a_i.T @ d`` must be for a row to count.

    Returns:
        The limit, with :attr:`StepLimit.blocking` naming the row, or an unbounded limit
        when no inactive row is approached.
    """
    point = _vector("z", z, size=problem.num_variables)
    step = _vector("d", d, size=problem.num_variables)
    inactive = list(working_set.inactive_inequalities)
    if not inactive:
        return StepLimit(alpha=math.inf)

    rows = problem.A[inactive]
    approach = rows @ step
    slack = problem.b[inactive] - rows @ point
    blocking = approach > tolerance
    if not blocking.any():
        return StepLimit(alpha=math.inf)

    ratios = np.where(blocking, slack / np.where(blocking, approach, 1.0), math.inf)
    winner = int(np.argmin(ratios))
    return StepLimit(alpha=max(0.0, float(ratios[winner])), blocking=inactive[winner], source="linear")


def step_limit(
    problem: SOCP,
    z: Vector,
    d: Vector,
    working_set: WorkingSet,
    *,
    max_step: float = math.inf,
    tolerance: float = TOLERANCE,
) -> StepLimit:
    """The intersection §5.2 asks for: linear interval, explicit bounds, and the cone.

    The cone's interval is #18's and is not yet computed, so this refuses a problem whose
    cone could bind rather than returning a limit that ignores it. "Could bind" means any
    factor the direction moves at all.

    An earlier version of this guard exempted a *strictly interior* factor, on the reasoning
    that it could not be reached by a step the linear interval already bounds. That
    reasoning is wrong, and §14.1's invariant checker caught it: the linear interval bounds
    the step, but nothing makes that bound small enough to stay inside the cone, so a step
    from an interior point can leave it. The refusal is now unconditional on the direction
    moving the block, which is the only version that cannot produce an infeasible iterate.

    Args:
        problem: the instance.
        z: the current point, assumed feasible.
        d: the direction.
        working_set: the current set.
        max_step: an explicit upper bound on the step -- §5.2's third interval.
        tolerance: the ratio test's rounding-level threshold.

    Returns:
        The tightest of the intervals computed.

    Raises:
        ProblemError: if ``max_step`` is not positive, or if a cone factor could bind. The
            second is a scope refusal rather than a data error, and its message says so.
    """
    if max_step <= 0.0:
        raise ProblemError("max_step", f"an explicit step bound is positive, found {max_step}")
    _require_no_binding_cone(problem, z, d)

    limit = linear_step(problem, z, d, working_set, tolerance=tolerance)
    if math.isfinite(max_step):
        limit = limit.tighter_of(StepLimit(alpha=float(max_step), source="bound"))
    return limit


def _require_no_binding_cone(problem: SOCP, z: Vector, d: Vector) -> None:
    """Refuse a problem whose conic block could limit the step.

    Args:
        problem: the instance.
        z: the current point.
        d: the direction.

    Raises:
        ProblemError: if the direction moves any cone factor's slack.
    """
    if not len(problem.cone):
        return
    _vector("z", z, size=problem.num_variables)
    moves = problem.G @ _vector("d", d, size=problem.num_variables)
    for factor, block in enumerate(problem.cone.slices):
        if not np.abs(moves[block]).any():
            continue
        raise ProblemError(
            "cone",
            f"the direction moves cone factor {factor}, so the conic interval of eq. (6) is "
            "needed and it is issue #18's. Until then this is the polyhedral baseline of "
            "§9 Phase I",
        )
