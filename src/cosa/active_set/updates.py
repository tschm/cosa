"""The rules that change the working set: what to add, what to drop, what the cone is.

§7 Active-Set Logic (``paper.tex:569``) is four short subsections, and this module is
them:

* **§7.1 activation.** An inactive inequality that reaches its boundary, ``a_i.T @ x =
  b_i``, joins the working set. :func:`activation_candidates` finds those rows;
  :func:`add_inequality` puts one in.
* **§7.2 deactivation.** An active inequality whose multiplier violates its required sign
  is a candidate for removal, and the plan asks for "the classical rule of removing the
  most strongly violating multiplier, subject to numerical tolerances".
  :func:`removal_candidate` is that rule; :func:`drop_inequality` performs the removal.
* **§7.3 SOC activation.** The cone becomes geometrically active when ``t - ||L @ x||_2``
  is small. :func:`activate_cones` applies that threshold to a conic slack.
* **§8.3 dependent-constraint removal.** Not a §7 rule at all, but an active-set change,
  which is why it is here rather than in the linear algebra: §8.3 lists it beside QR rank
  detection and regularization, and it is the only one of the three that alters *what the
  solver believes is active*. :func:`drop_dependent_rows` is it. Losing it among the
  factorization work is the mistake #25 explicitly warns about.
* **§7.4 SOC deactivation.** Deliberately *not* here. The plan calls it "a key research
  component" and says the decision must come from the conic multiplier and the normal-cone
  conditions rather than from the geometry, so it belongs to #23. What this module offers
  is :func:`set_cone_status`, the primitive that issue will drive.

Every function is pure: it takes a working set and returns a new one, or takes a working
set and returns an index. Nothing is mutated, and no function here solves anything -- the
multipliers :func:`removal_candidate` judges are computed by #13, from the KKT system
of #12.

**The required sign of a multiplier is read, not restated.** ``y >= 0`` for
``A @ z <= b`` is a consequence of the Lagrangian sign choice fixed in
:data:`cosa.SIGN_CONVENTION`, not an independent fact, so :func:`removal_candidate`
multiplies by ``SIGN_CONVENTION.inequality`` instead of hard-coding the direction of the
test. Flip the convention and this rule follows it; write the ``< 0`` out by hand here and
the two silently disagree, which is the failure mode #9 exists to prevent.

**Two tolerances, and they are not the geometry tolerance.**
:data:`ACTIVATION_TOLERANCE` decides when a constraint counts as reached and when a cone
counts as active -- §7.3's ``eps_act``. :data:`MULTIPLIER_TOLERANCE` decides when a
multiplier counts as wrong-signed. Both are algorithmic and coarse, and both are
deliberately looser than :data:`cosa.geometry.soc.TOLERANCE`, which only asks what a point
*is*. The separate activation and deactivation thresholds of §8.2 -- the hysteresis
``eps_on < eps_off`` that stops a nearly active cone oscillating -- are a further
refinement owned by #29; a single symmetric threshold is what §7 asks for and what is
here.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Final

from cosa.active_set.working_set import ConeStatus, WorkingSet
from cosa.geometry.soc import ConePosition, positions
from cosa.problem.socp import SIGN_CONVENTION, ProblemError, _vector

if TYPE_CHECKING:
    from cosa import Vector
    from cosa.problem.socp import SOCP

__all__ = [
    "ACTIVATION_TOLERANCE",
    "MULTIPLIER_TOLERANCE",
    "activate_cones",
    "activation_candidates",
    "add_inequality",
    "cone_status_for",
    "drop_dependent_rows",
    "drop_inequality",
    "inequality_slack",
    "removal_candidate",
    "set_cone_status",
]

ACTIVATION_TOLERANCE: Final = 1e-8
"""How close to its boundary a constraint must be to count as reached -- §7.3's eps_act."""

MULTIPLIER_TOLERANCE: Final = 1e-8
"""How wrong-signed a multiplier must be to count as violating its sign, per §7.2."""


def inequality_slack(problem: SOCP, z: Vector) -> Vector:
    """The linear slack ``b - A @ z``: zero on a constraint's boundary, negative outside.

    The counterpart of :meth:`cosa.SOCP.cone_slack` for the polyhedral block, and the
    quantity §7.1's activation test thresholds. It lives here rather than on
    :class:`cosa.SOCP` because it exists for this rule: the representation's job is to hold
    ``A`` and ``b``, and what "close to the boundary" means is an active-set decision.

    Args:
        problem: the instance.
        z: a point in variable space.

    Returns:
        The slack, ``(m_ineq,)``, in row order.
    """
    return problem.b - problem.A @ _vector("z", z, size=problem.num_variables)


def activation_candidates(
    problem: SOCP,
    z: Vector,
    working_set: WorkingSet,
    *,
    tolerance: float = ACTIVATION_TOLERANCE,
) -> tuple[int, ...]:
    """The inactive inequalities at their boundary: §7.1's "reaches its boundary".

    A row counts as reached when its slack is at or below ``tolerance * max(1, |b_i|)``,
    which includes rows that a step has overshot into infeasibility -- those are exactly
    the rows a Phase I iteration must add, and refusing to notice a row because the step
    passed slightly through it would leave the iterate outside the feasible set with
    nothing in the working set to pull it back.

    Args:
        problem: the instance.
        z: the current point in variable space.
        working_set: the current set, whose active rows are excluded from the result.
        tolerance: the relative activation tolerance.

    Returns:
        The candidate row indices, ascending. Empty when no inactive row is close.
    """
    slack = inequality_slack(problem, z)
    return tuple(
        index
        for index in working_set.inactive_inequalities
        if slack[index] <= tolerance * max(1.0, abs(float(problem.b[index])))
    )


def add_inequality(working_set: WorkingSet, index: int) -> WorkingSet:
    """Put inequality row ``index`` into the working set -- §7.1.

    Args:
        working_set: the set to extend.
        index: the row to add.

    Returns:
        A new set containing the row.

    Raises:
        ProblemError: if the row is not an inequality of this shape, or is already active.
            Already-active is an error rather than a no-op: a rule that adds a row it
            already holds has lost track of the set, and silently succeeding hides that.
    """
    if working_set.is_active(index):
        raise ProblemError("inequality", f"row {index} is already active")
    return replace(working_set, inequalities=(*working_set.inequalities, index))


def drop_inequality(working_set: WorkingSet, index: int) -> WorkingSet:
    """Take inequality row ``index`` out of the working set -- §7.2.

    Args:
        working_set: the set to shrink.
        index: the row to drop.

    Returns:
        A new set without the row.

    Raises:
        ProblemError: if the row is not an inequality of this shape, or is not active.
    """
    if not working_set.is_active(index):
        raise ProblemError("inequality", f"row {index} is not active, so it cannot be dropped")
    return replace(working_set, inequalities=tuple(i for i in working_set.inequalities if i != index))


def removal_candidate(
    working_set: WorkingSet,
    y: Vector,
    *,
    tolerance: float = MULTIPLIER_TOLERANCE,
) -> int | None:
    """The active inequality whose multiplier most strongly violates its sign -- §7.2.

    The classical rule. Under :data:`cosa.SIGN_CONVENTION` an inequality multiplier must
    be non-negative, so the violation of row ``i`` is ``-y_i * SIGN_CONVENTION.inequality``
    and the candidate is the row maximizing it, provided the violation exceeds the
    tolerance. A multiplier that is merely a rounding error below zero is not a reason to
    change the active set.

    Ties go to the lowest row index. That is arbitrary but deterministic, which is what
    matters: an arbitrary *and* unstable choice is how an active-set method cycles, and
    the anti-cycling rules that replace this tie-break belong to #29.

    Args:
        working_set: the current set, whose active rows are the ones judged.
        y: the inequality multipliers, ``(m_ineq,)``, indexed by row. Entries for inactive
            rows are ignored, so passing the full vector with zeros there -- which is what
            complementarity makes them -- is correct.
        tolerance: how negative a multiplier must be to count as violating.

    Returns:
        The row index to drop, or ``None`` if every active multiplier has the right sign,
        which is the dual-feasibility half of optimality for the polyhedral block.
    """
    multipliers = _vector("y", y, size=working_set.num_inequalities)
    violations = {index: -float(multipliers[index]) * SIGN_CONVENTION.inequality for index in working_set.inequalities}
    worst = max(violations, key=lambda index: (violations[index], -index), default=None)
    if worst is None or violations[worst] <= tolerance:
        return None
    return worst


def cone_status_for(position: ConePosition) -> ConeStatus:
    """The status the geometry alone argues for at a point in the given position.

    The one place the observation of :class:`cosa.geometry.soc.ConePosition` is turned into
    the decision of :class:`cosa.active_set.working_set.ConeStatus`:

    * interior means the cone constrains nothing locally, so :attr:`ConeStatus.INACTIVE`;
    * a nonzero boundary point is where eq. (3) applies, so :attr:`ConeStatus.TANGENT`;
    * the apex gets :attr:`ConeStatus.APEX`, because §8.1 has no tangent hyperplane there;
    * an *exterior* point gets :attr:`ConeStatus.TANGENT` as well. The block is infeasible,
      which the residual check of #22 is what notices; as far as the working set goes, the
      cone is the constraint the next direction has to respect, and treating it as inactive
      would compute a step that ignores the constraint being violated.

    Args:
        position: where the conic slack sits.

    Returns:
        The status the geometry argues for. §7.3 warns this is not the last word: a cone
        can be geometrically active and still carry no genuine active normal, which is
        #23's business.
    """
    match position:
        case ConePosition.INTERIOR:
            return ConeStatus.INACTIVE
        case ConePosition.APEX:
            return ConeStatus.APEX
        case ConePosition.BOUNDARY | ConePosition.EXTERIOR:
            return ConeStatus.TANGENT


def set_cone_status(working_set: WorkingSet, index: int, status: ConeStatus) -> WorkingSet:
    """Set cone factor ``index``'s status, whatever it currently is.

    The unconditional primitive. Activation has a rule of its own in
    :func:`activate_cones`; deactivation does not have one yet, on purpose -- §7.4 makes it
    a research question and #23 answers it -- so this is what that issue will build on.

    Args:
        working_set: the set to change.
        index: the cone factor.
        status: its new status.

    Returns:
        A new set with the status applied, or the same set when nothing changes.

    Raises:
        ProblemError: if the index is not a factor of this shape's cone product.
    """
    if working_set.status(index) is status:
        return working_set
    updated = list(working_set.cone_status)
    updated[index] = status
    return replace(working_set, cone_status=tuple(updated))


def activate_cones(
    problem: SOCP,
    z: Vector,
    working_set: WorkingSet,
    *,
    tolerance: float = ACTIVATION_TOLERANCE,
) -> WorkingSet:
    """Turn cones on, and correct the geometry of the ones already on -- §7.3.

    For each factor, the conic slack ``G @ z + h`` is classified at the activation
    tolerance and :func:`cone_status_for` says what the geometry argues for. What is done
    with that depends on the factor's current status:

    * an **inactive** factor is activated when the geometry says it should be. This is
      §7.3's ``t - ||L @ x||_2 <= eps_act``, applied through the cone predicates rather
      than by rewriting the threshold here.
    * an **active** factor has its geometry corrected -- tangent to apex, or apex to
      tangent -- because that is not a change of belief about whether the cone is active,
      only about which face of it is.
    * an active factor whose slack has grown is **left active**. Turning it off is
      deactivation, and §7.4 refuses to decide that on the geometry alone. This function
      is monotone in activity by design; #23 is what turns a cone off.

    Args:
        problem: the instance, for its cone product and conic slack.
        z: the current point in variable space.
        working_set: the current set.
        tolerance: the activation tolerance, ``eps_act``. Coarser than the geometry
            module's default, because it decides an algorithmic step rather than a fact.

    Returns:
        A new set with the cone statuses updated, or the same set when nothing changes.
    """
    slack = problem.cone_slack(z)
    updated = working_set
    for index, position in enumerate(positions(problem.cone, slack, tolerance=tolerance)):
        argued = cone_status_for(position)
        current = updated.status(index)
        if current.is_active and not argued.is_active:
            continue
        updated = set_cone_status(updated, index, argued)
    return updated


def drop_dependent_rows(
    problem: SOCP,
    working_set: WorkingSet,
    z: Vector,
    *,
    tolerance: float | None = None,
) -> tuple[WorkingSet, tuple[int, ...]]:
    """§8.3's dependent-constraint removal: drop active rows that add nothing.

    The item #25 flags as easy to lose, because it sits on a list with QR rank detection
    and regularization and is the only one of the three that is not linear algebra. A
    pivoted QR says *which* rows are redundant -- see
    :func:`cosa.linear_algebra.rank.analyse` -- and this decides which of those the working
    set is allowed to let go of.

    **Only inequalities may be dropped, and that is not a simplification.** §3.1 imposes
    every equality unconditionally, so an equality row is not the solver's to remove even
    when it is the redundant one; and a cone's rows encode a geometric belief that §7.4
    hands to the conic multiplier, so removing one is #23's decision and not a numerical
    repair. When the dependency lies entirely among rows that cannot be dropped, this
    returns the set unchanged and the caller falls back to regularization -- which is
    exactly why §8.3 lists both.

    Args:
        problem: the instance.
        working_set: the set whose rows may be dependent.
        z: the current point, needed to build the tangent rows.
        tolerance: the pivot threshold, or ``None`` for the default.

    Returns:
        The set with the droppable dependent rows removed, and the inequality row indices
        that were dropped. An empty tuple means nothing could be dropped -- either because
        the set was independent or because the redundancy was not the solver's to remove.
    """
    from cosa.linear_algebra.kkt import RowLayout, working_set_matrix
    from cosa.linear_algebra.rank import analyse

    analysis = analyse(working_set_matrix(problem, working_set, z), tolerance=tolerance)
    if not analysis.is_deficient:
        return working_set, ()

    layout = RowLayout.for_working_set(working_set)
    droppable = [
        layout.inequalities[position] for position in analysis.dependent if position < len(layout.inequalities)
    ]
    updated = working_set
    for index in droppable:
        updated = drop_inequality(updated, index)
    return updated, tuple(droppable)
