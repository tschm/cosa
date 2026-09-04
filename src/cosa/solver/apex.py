"""The apex branch: what to do at ``Lx = 0``, where no tangent hyperplane exists.

§8.1 (``paper.tex:623``) singles out the point ``L @ x = 0`` and says what to do about it:
handle it *"using the exact SOC membership and normal-cone conditions rather than a tangent
hyperplane"* (``paper.tex:644``). This module is that branch. It exists because the
geometry module deliberately refuses to serve the apex -- see
:class:`cosa.geometry.tangent.ApexError` -- and refusing is only useful if something else
answers.

**Why a hyperplane cannot work here, stated exactly.** The set of directions a feasible
step may take from a point ``s`` of the cone is the tangent cone of ``Q`` at ``s``, and the
apex is where its *shape* changes:

* at a **nonzero boundary point**, the tangent cone is a half-space,
  ``{ds : (1, -u).T @ ds >= 0}``. One linear inequality. Holding the iterate on the
  boundary means imposing it as an equality, which is eq. (3) -- one row of ``W_k``,
  indistinguishable from an active linear constraint. That is what makes a conic
  active-set method possible at all.
* at the **apex**, the tangent cone is ``Q`` itself. Because ``Q`` is a cone,
  ``0 + alpha * ds in Q`` for ``alpha > 0`` is simply ``ds in Q`` -- for *every* positive
  step, not merely small ones. And ``Q`` is not a half-space, not polyhedral, and not
  expressible by any finite set of rows.

The dual side changes in the same way. The normal cone at a nonzero boundary point is a
single ray, through ``(-1, u)``; at the apex it is ``-Q``, a full-dimensional cone. So the
dual feasibility test stops being a scalar sign condition and becomes a cone membership
test. Those two facts -- tangent cone becomes ``Q``, normal cone becomes ``-Q`` -- are
precisely what §8.1 means by "a different tangent and normal geometry", and they are the
two tools §8.1 names: exact membership, on the direction; normal-cone conditions, on the
multiplier.

**What this branch does.** Given a factor whose slack sits at the apex:

1. Hold the block, ``G_block @ d = 0``, and solve. This is the conservative member of the
   tangent cone -- ``ds = 0`` is always in ``Q`` -- and it is *not* a stall: the direction
   may still move ``x`` anywhere in the null space of ``L``, which on a rank-deficient
   covariance is a subspace of positive dimension. That is the case #24 exists for.
2. Recover the multipliers and apply the normal-cone condition, ``w_block in Q``. If it
   holds, the cone contributes a genuine active normal and holding the apex is justified.
3. If it does not hold, the apex should be released. Drop the factor and re-solve, then
   apply exact membership to the *direction*: accept it only if ``ds in Q``.

**The case this branch cannot resolve, and why that is worth saying.** If the multiplier
says release and the released direction is not in ``Q``, then the best feasible direction
lies on the boundary of ``Q`` -- and finding it is a conic projection, not a linear solve.
This module holds instead, and says so in :attr:`ApexDirection.reason`. Holding is
feasible and often still a descent direction, so nothing is broken; but a dual violation
that the branch cannot act on is exactly the shape of Risk 1
([#39](https://github.com/tschm/cosa/issues/39)) -- "the SOC working-set concept may be
insufficient". If that reason is ever seen in practice, #39 is the issue it belongs to.
#23's :func:`cosa.active_set.updates.deactivate_cones` is the rule that *would* authorize
the release; what it cannot supply is a feasible direction to release into, which is why
the case survived it. Recording it is
more useful than pretending the case does not arise.

**On eq. (7) that case is not an edge case -- it is the only case.** Releasing by dropping
the factor can *never* produce a feasible direction for the mean-standard-deviation
problem, and the reason is structural rather than numerical. In eq. (7) the variable ``t``
appears in exactly two places: the cone's head row, and the objective with coefficient
``lam > 0``. :meth:`cosa.MeanStdForm.to_socp` puts a zero ``t`` column in ``A`` and ``E``,
and :meth:`cosa.SOCP.as_mean_std` checks it. So once the factor is dropped, every row of
``W_k`` has a zero ``t`` column, the direction's ``t`` row reads
``rho * d_t + 0 = -lam``, and

    d_t = -lam / rho < 0.

The slack direction's head is that same ``d_t``, and a vector with a negative head is
never in ``Q``. So on eq. (7) an unjustified apex always comes back *blocked*: the
released direction is infeasible by arithmetic, not by bad luck, and no choice of ``rho``
or tolerance changes it.

The release branch is therefore exercised only by the *general* SOCP -- an instance whose
head variable is constrained or rewarded elsewhere -- which the representation of #9
deliberately supports and which the tests use. Keeping the branch is right: it is correct
for the general form, and it is what makes the eq. (7) result a *finding* rather than an
assumption. But anyone reading this expecting the apex to be released on a portfolio
problem should know that it will not be, and that the honest answer there is #39.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cosa.active_set.multipliers import STATIONARITY_TOLERANCE, Multipliers, dual_cone_violation, from_direction
from cosa.active_set.updates import set_cone_status
from cosa.active_set.working_set import ConeStatus, WorkingSet
from cosa.geometry.soc import TOLERANCE, is_apex, is_member
from cosa.linear_algebra.kkt import RHO, Direction, direction
from cosa.problem.socp import ProblemError, _vector

if TYPE_CHECKING:
    from cosa import Vector
    from cosa.problem.socp import SOCP

__all__ = [
    "ApexDirection",
    "apex_direction",
    "is_apex_optimal",
    "step_stays_in_the_cone",
]


@dataclass(frozen=True, eq=False)
class ApexDirection:
    """The apex branch's answer: a direction, and the reasoning that produced it.

    The reasoning is a field rather than a log line because it is the interesting output.
    Three outcomes are possible and they mean different things -- the apex is justified,
    the apex was released, or the apex should have been released and could not be -- and a
    caller that cannot tell them apart cannot report the third, which is the one that
    matters (see the module docstring on Risk 1).

    Attributes:
        direction: the direction to step along.
        working_set: the set the direction was computed for. Differs from the one handed in
            exactly when :attr:`released` is true.
        released: whether the factor was taken out of the working geometry.
        multipliers: the multipliers of the *held* apex, which is what the normal-cone
            condition was applied to.
        violation: how far the held apex's ``w_block`` fell outside ``Q``. Zero means the
            normal-cone condition held and the apex is justified.
        reason: one sentence saying which of the three outcomes this is, and why.
    """

    direction: Direction
    working_set: WorkingSet
    released: bool
    multipliers: Multipliers
    violation: float
    reason: str

    @property
    def is_blocked(self) -> bool:
        """Should the apex have been released, but the released direction was infeasible?

        The Risk 1 case. Distinguishable from the other two outcomes precisely so that it
        can be reported rather than silently absorbed.
        """
        return not self.released and self.violation > 0.0

    def __str__(self) -> str:
        """The outcome and its reason, for a log line or a failure message."""
        outcome = "released" if self.released else ("blocked" if self.is_blocked else "held")
        return f"apex {outcome}: {self.reason}"


def step_stays_in_the_cone(
    problem: SOCP,
    factor: int,
    step: Vector,
    *,
    tolerance: float = TOLERANCE,
) -> bool:
    """Exact membership on a *direction* at the apex: is ``ds`` in ``Q``?

    The replacement for the tangent condition, and the reason it is exact rather than
    linearized: from the apex, ``alpha * ds in Q`` for every ``alpha > 0`` exactly when
    ``ds in Q``, so there is no step length to be conservative about and nothing is being
    approximated. At a nonzero boundary point the same question would need eq. (6)'s
    quadratic, because the boundary curves; at the apex it needs one membership test.

    Args:
        problem: the instance, for its cone product and ``G``.
        factor: which factor of the cone product to test.
        step: a direction in variable space, ``(n,)``. Its slack direction is ``G @ step``.
        tolerance: the membership tolerance.

    Returns:
        ``True`` if a step along ``step`` keeps this factor's slack in its cone.

    Raises:
        ProblemError: if ``factor`` is not a factor of the problem's cone product.
    """
    _require_factor(problem, factor)
    slack_direction = problem.G @ _vector("step", step, size=problem.num_variables)
    return is_member(slack_direction[problem.cone.slices[factor]], tolerance=tolerance)


def is_apex_optimal(
    problem: SOCP,
    multipliers: Multipliers,
    factor: int,
    *,
    tolerance: float = STATIONARITY_TOLERANCE,
) -> bool:
    """The normal-cone condition: does this factor's ``w`` block lie in ``Q``?

    §8.1's second named tool. At the apex the normal cone of ``Q`` is ``-Q``, so a
    multiplier that keeps the point there must satisfy ``w_block in Q`` -- a cone
    membership test, not a sign test, which is the whole difference from a smooth boundary
    point. If it holds, the cone contributes a genuine active normal and holding the apex
    is justified.

    Args:
        problem: the instance.
        multipliers: the multipliers of the held apex.
        factor: which factor of the cone product to test.
        tolerance: how large a violation counts as none.

    Returns:
        ``True`` if the normal-cone condition holds for this factor.

    Raises:
        ProblemError: if ``factor`` is not a factor of the problem's cone product.
    """
    _require_factor(problem, factor)
    return dual_cone_violation(problem, multipliers)[factor] <= tolerance


def apex_direction(
    problem: SOCP,
    working_set: WorkingSet,
    z: Vector,
    factor: int = 0,
    *,
    rho: float = RHO,
    tolerance: float = TOLERANCE,
    dual_tolerance: float = STATIONARITY_TOLERANCE,
) -> ApexDirection:
    """Compute a direction at the apex, from membership and the normal cone.

    The three-step branch of the module docstring. It is a *branch*, not a solver: it
    answers "what direction, given that this factor's slack is at the apex" and leaves the
    step length to #18 and the loop to #20.

    Args:
        problem: the instance.
        working_set: the current set. The factor's status is forced to
            :attr:`cosa.ConeStatus.APEX` for the held solve, whatever it was, because that
            is what the geometry says it is.
        z: the current point, whose slack must be at the apex in this factor.
        factor: which factor of the cone product is at its apex.
        rho: the ``rho`` of ``H = rho*I``, passed through to the direction solve.
        tolerance: the geometric tolerance, for the apex and membership tests.
        dual_tolerance: the tolerance for the normal-cone condition.

    Returns:
        The direction, the working set it belongs to, and the reasoning.

    Raises:
        ProblemError: if ``factor`` is not a factor of the cone product, or if this
            factor's slack is *not* at the apex -- in which case the caller wanted the
            tangent branch and asking for this one is a bug, not a degenerate case.
        SingularKktError: if the working-set rows are linearly dependent.
    """
    _require_factor(problem, factor)
    point = _vector("z", z, size=problem.num_variables)
    span = problem.cone.slices[factor]
    if not is_apex(problem.cone_slack(point)[span], tolerance=tolerance):
        raise ProblemError(
            "apex",
            f"cone factor {factor} is not at its apex, so the tangent branch of eq. (3) applies and this one does not",
        )

    held_set = set_cone_status(working_set, factor, ConeStatus.APEX)
    held = direction(problem, held_set, point, rho=rho, tolerance=tolerance)
    held_multipliers = from_direction(problem, held_set, point, held, tolerance=tolerance)
    violation = dual_cone_violation(problem, held_multipliers)[factor]

    if violation <= dual_tolerance:
        return ApexDirection(
            direction=held,
            working_set=held_set,
            released=False,
            multipliers=held_multipliers,
            violation=violation,
            reason=(
                "the normal-cone condition holds -- w lies in Q, so the cone contributes a "
                "genuine active normal and the apex is where the iterate belongs"
            ),
        )

    released_set = set_cone_status(held_set, factor, ConeStatus.INACTIVE)
    released = direction(problem, released_set, point, rho=rho, tolerance=tolerance)
    if step_stays_in_the_cone(problem, factor, released.d, tolerance=tolerance):
        return ApexDirection(
            direction=released,
            working_set=released_set,
            released=True,
            multipliers=held_multipliers,
            violation=violation,
            reason=(
                f"w lay {violation:.3g} outside Q, so the apex is not justified, and the "
                "released direction satisfies exact membership -- it leaves the apex into "
                "the cone"
            ),
        )

    return ApexDirection(
        direction=held,
        working_set=held_set,
        released=False,
        multipliers=held_multipliers,
        violation=violation,
        reason=(
            f"w lay {violation:.3g} outside Q, so the apex is not justified, but the "
            "released direction leaves Q -- the best feasible direction is on the boundary "
            "of Q, which needs a conic projection rather than a linear solve. Held instead; "
            "see issue #39"
        ),
    )


def _require_factor(problem: SOCP, factor: int) -> None:
    """Reject an index that is not a factor of the problem's cone product.

    Args:
        problem: the instance.
        factor: the index to check.

    Raises:
        ProblemError: if the index is out of range.
    """
    if not 0 <= factor < len(problem.cone):
        raise ProblemError("cone", f"expected a factor in [0, {len(problem.cone)}), found {factor}")
