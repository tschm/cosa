"""How far a step may go: the ratio test, and the interval it produces.

§5.2 (``paper.tex:471``) says the feasible step is *"determined by the intersection of: the
linear feasible interval; the SOC feasible interval; any explicit step bounds"*, and calls
that intersection "the conic analogue of the classical active-set ratio test". This module
computes it.

**All three intervals are here.** The linear one is the classical ratio test -- for each
inactive inequality with ``a_i.T @ p > 0``,

    alpha <= (b_i - a_i.T @ x) / (a_i.T @ p),

which is arithmetic. The conic one is §5.1's, and is where the milestone gets its name.

**Eq. (6) as printed in the plan is wrong, and this module implements the correction.**
Feasibility along a direction is ``||L(x + alpha*p)||_2 <= t + alpha*tau``. With
``r = L @ x`` and ``q = L @ p``, and provided the right-hand side is non-negative, squaring
*both* sides gives

    (||q||^2 - tau^2) alpha^2 + 2(r.T @ q - t*tau) alpha + (||r||^2 - t^2) <= 0.

The plan's eq. (6) (``paper.tex:457``) writes the leading coefficient as ``||q||^2``,
dropping the ``tau^2`` that comes from expanding ``(t + alpha*tau)^2``. The two agree only
when ``tau = 0`` -- that is, only when the step does not move the risk variable, which is
the one case the mean-standard-deviation problem never takes.

The difference is not small and not conservative. Three intersections computed by hand:

===========================================  =======  ===============  ===========
case                                           exact  eq. (6) as printed  corrected
===========================================  =======  ===============  ===========
``|alpha| <= 2 - alpha``                       1.0000           0.8284      1.0000
``0.5 alpha <= 1 - alpha``                     0.6667           0.4721      0.6667
a step along the ray towards the apex          5.0000           0.0000      5.0000
===========================================  =======  ===============  ===========

The third is the one that matters most: a direction that keeps the iterate exactly on the
cone's boundary all the way to the apex is feasible for every step up to ``t / (-tau)``,
and the printed formula admits *none of it*. A solver using it would refuse to move along
the boundary at all. The corrected coefficient is the Lorentz quadratic form of the slack
direction, which is what "does this direction leave the cone" is a question about.

**The right-hand side proviso is not a footnote either.** Squaring an inequality whose
right-hand side has gone negative admits steps that satisfy the square and violate the
original -- ``||r + alpha q||`` is non-negative and ``t + alpha tau`` is not. When
``tau < 0`` the right-hand side reaches zero at ``alpha = t / (-tau)`` and every larger step
is infeasible however small the quadratic gets, so :func:`cone_interval` intersects that
bound rather than trusting the quadratic alone.

**Starting feasible is what makes the root selection tractable**, and the corrected
coefficient makes it a three-way case rather than a one-way one. At a feasible point
``||r|| <= t``, so the constant term is non-positive and ``alpha = 0`` always satisfies the
quadratic. Then:

* ``||q|| > |tau|`` -- the parabola opens upward, zero lies between its roots, and the step
  is the upper one. The direction leaves the cone eventually.
* ``||q|| = |tau|`` -- the quadratic collapses to a linear inequality, and dividing by the
  leading coefficient would be dividing by zero. The boundary ray above is this case. Note
  that §8.1's ``||q|| = 0`` lands here only when ``tau`` also vanishes; with ``tau != 0`` a
  motionless tail puts ``||q|| < |tau|`` and so falls in the branch below, bounded by the
  right-hand side rather than by the quadratic. Both are handled; they are not the same
  branch.
* ``||q|| < |tau|`` -- the parabola opens *downward*, so the admissible set is the
  complement of an interval and the step is the *lower* root when that root is non-negative,
  and unbounded otherwise. This is the case where the slack direction is itself inside the
  cone, so the step never leaves: exactly the situation a one-sided root selection would get
  backwards.

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
    "ConeInterval",
    "StepLimit",
    "cone_interval",
    "cone_step",
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


@dataclass(frozen=True)
class ConeInterval:
    """The steps eq. (6) admits for one cone factor, as an interval.

    Attributes:
        lower: the smaller root, at most zero for a feasible starting point.
        upper: the largest feasible step, possibly ``inf``.
        degenerate: whether ``||q|| = 0`` collapsed the quadratic to a linear inequality.
            Recorded rather than hidden because it is the case §8.1 is about, seen from the
            step's side: the direction does not move ``L @ x`` at all.
        capped: whether :attr:`upper` comes from the right-hand side going negative rather
            than from the quadratic. When it does, eq. (6) is satisfied beyond this point
            and the original inequality is not, which is the trap squaring sets.
    """

    lower: float
    upper: float
    degenerate: bool = False
    capped: bool = False

    def contains(self, alpha: float) -> bool:
        """Is ``alpha`` in the admissible interval?

        Args:
            alpha: the step to test.

        Returns:
            ``True`` if the step is admissible.
        """
        return self.lower <= alpha <= self.upper


def cone_interval(s: Vector, ds: Vector, *, tolerance: float = TOLERANCE) -> ConeInterval:
    """Solve eq. (6) for one cone factor: the admissible interval in ``alpha``.

    Assumes ``s`` is in the cone, which is what makes the constant term non-positive and the
    root selection unambiguous -- see the module docstring. A point outside the cone still
    produces an answer, but it is the answer to a different question and the caller is
    responsible for not asking it.

    The roots are computed by the stable form ``-(b + sign(b) sqrt(disc)) / 2`` rather than
    the textbook quotient, because ``b^2 - 4ac`` and ``b^2`` are close whenever ``c`` is
    small -- which is exactly the case of interest, a point already near the boundary.

    Args:
        s: the conic slack of this factor, head first, presumed in the cone.
        ds: the slack direction of this factor, head first.
        tolerance: below which the leading coefficient counts as zero.

    Returns:
        The interval, whose upper end is the largest feasible step.

    Raises:
        ProblemError: if ``s`` and ``ds`` are not vectors of the same length, at least 2.
    """
    head, tail = _split("slack", s)
    step_head, step_tail = _split("slack direction", ds, size=1 + tail.size)

    # The leading coefficient is the Lorentz form of the slack direction, `||q||^2 - tau^2`,
    # not the `||q||^2` eq. (6) prints. See the module docstring for the arithmetic and for
    # three hand-computed cases where the difference changes the answer.
    leading = float(step_tail @ step_tail) - step_head * step_head
    middle = 2.0 * (float(tail @ step_tail) - head * step_head)
    constant = float(tail @ tail) - head * head

    scale = max(1.0, abs(head), float(np.abs(tail).max(initial=0.0)))
    negligible = tolerance * scale * scale
    cap = math.inf if step_head >= 0.0 else head / (-step_head)

    if abs(leading) <= negligible:
        # `||q|| = |tau|`: the quadratic is linear. This is the ray that runs along the
        # boundary to the apex; §8.1's motionless tail lands here only if `tau` is zero too.
        upper = -constant / middle if middle > negligible else math.inf
        return ConeInterval(lower=-math.inf, upper=min(upper, cap), degenerate=True, capped=cap < upper)

    discriminant = middle * middle - 4.0 * leading * constant
    if discriminant < 0.0:
        # No real roots. Opening downward this means the quadratic is negative everywhere,
        # so the cone never stops the step; opening upward it means the current point is not
        # in the cone, which is the caller's precondition to keep.
        upper = math.inf if leading < 0.0 else math.nan
        return ConeInterval(lower=-math.inf, upper=min(upper, cap), capped=cap < upper)

    root = math.sqrt(discriminant)
    branch = -(middle + math.copysign(root, middle if middle else 1.0)) / 2.0
    first, second = branch / leading, (constant / branch if branch else 0.0)
    lower, upper = min(first, second), max(first, second)

    if leading < 0.0:
        # Opening downward: admissible *outside* the roots. Zero is admissible, so it lies
        # outside them, and the question is which side. Comparing zero to the roots directly
        # is not robust -- at a point exactly on the cone's boundary the constant term is
        # zero, so one root *is* zero up to rounding, and its sign decides the branch. The
        # midpoint is a rounding error away from neither.
        if (lower + upper) / 2.0 > 0.0:
            upper, lower = lower, -math.inf
        else:
            upper, lower = math.inf, -math.inf
    return ConeInterval(lower=lower, upper=min(upper, cap), capped=cap < upper)


def cone_step(
    problem: SOCP,
    z: Vector,
    d: Vector,
    *,
    tolerance: float = TOLERANCE,
) -> StepLimit:
    """The tightest conic step over every factor of the cone product.

    Args:
        problem: the instance.
        z: the current point, presumed conically feasible.
        d: the direction.
        tolerance: below which a factor's leading coefficient counts as zero.

    Returns:
        The limit, with ``source="cone"`` when a factor bounds the step.
    """
    point = _vector("z", z, size=problem.num_variables)
    step = _vector("d", d, size=problem.num_variables)
    if not len(problem.cone):
        return StepLimit(alpha=math.inf)

    slack = problem.cone_slack(point)
    moves = problem.G @ step
    limit = StepLimit(alpha=math.inf)
    for block in problem.cone.slices:
        interval = cone_interval(slack[block], moves[block], tolerance=tolerance)
        if math.isfinite(interval.upper):
            limit = limit.tighter_of(StepLimit(alpha=max(0.0, interval.upper), source="cone"))
    return limit


def step_limit(
    problem: SOCP,
    z: Vector,
    d: Vector,
    working_set: WorkingSet,
    *,
    max_step: float = math.inf,
    tolerance: float = TOLERANCE,
) -> StepLimit:
    """The intersection §5.2 asks for: linear interval, SOC interval, explicit bounds.

    All three, now that #18 has supplied the middle one. The loop calls this and nothing
    else; when the cone was missing this refused rather than returning a limit that ignored
    it, and the call site did not change when it stopped refusing.

    Args:
        problem: the instance.
        z: the current point, assumed feasible.
        d: the direction.
        working_set: the current set.
        max_step: an explicit upper bound on the step -- §5.2's third interval.
        tolerance: the rounding-level threshold for both intervals.

    Returns:
        The tightest of the three.

    Raises:
        ProblemError: if ``max_step`` is not positive.
    """
    if max_step <= 0.0:
        raise ProblemError("max_step", f"an explicit step bound is positive, found {max_step}")

    limit = linear_step(problem, z, d, working_set, tolerance=tolerance)
    limit = limit.tighter_of(cone_step(problem, z, d, tolerance=tolerance))
    if math.isfinite(max_step):
        limit = limit.tighter_of(StepLimit(alpha=float(max_step), source="bound"))
    return limit


def _split(name: str, block: Vector, *, size: int | None = None) -> tuple[float, Vector]:
    """Split a cone block into head and tail, validating its shape.

    Args:
        name: what the block is, for the error message.
        block: the vector to split, head first.
        size: the length required, when it is already determined.

    Returns:
        The pair ``(head, tail)``.

    Raises:
        ProblemError: if the block is not a vector of length at least 2.
    """
    entries = _vector(name, block, size=size)
    if entries.size < 2:
        raise ProblemError(name, f"a cone block needs a head and a tail, found {entries.size} entries")
    return float(entries[0]), entries[1:]
