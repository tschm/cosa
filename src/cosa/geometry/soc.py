"""Where a point sits relative to a second-order cone: inside, on it, or at its apex.

The cone is the plan's ``Q^(m+1) = {(t, y) : ||y||_2 <= t}``, written head first as
:class:`cosa.SecondOrderCone` fixes it. This module answers the three questions the rest
of the solver asks about a point in it, and nothing else:

* is ``z`` in the cone at all -- primal feasibility of the conic block;
* is ``z`` on the boundary -- which is where §7.3 says the cone becomes geometrically
  active and where the tangent condition of eq. (3) is defined;
* is ``z`` at the apex -- which §8.1 (``paper.tex:623``) singles out, because ``L @ x = 0``
  makes ``u = L @ x / ||L @ x||`` undefined and the apex "has a different tangent and
  normal geometry from a smooth nonzero boundary point".

Phase II (``paper.tex:710``) asks for these as independent routines, tested on their own,
and that is exactly what they are: free functions over a vector, with no solver state and
no problem instance in sight. They are the scaffolding the step interval (#18) and the
prototype (#20) are debugged with, so they ship with their own tests.

**One tolerance, mixed absolute and relative.** Every predicate compares against
``tolerance * max(1, scale)``, where ``scale`` is the magnitude of the point unless the
caller passes a better one. Below unit scale the test is absolute, above it relative, so a
portfolio whose risk numbers live around ``1e-4`` and a factor model whose cone block lives
around ``1e6`` both get a sane answer from the same default. The apex is the case where the
point's own magnitude carries no information -- it is nearly zero by definition -- and
there the derived scale falls back to 1, making the test absolute; a caller that knows the
problem's scale should say so.

**This is not the activation tolerance.** :data:`TOLERANCE` decides what a point *is*, to
the precision floating point can distinguish. Deciding what the solver should *treat* as
active is a coarser, deliberately hysteretic question -- §7.3's ``eps_act`` and §8.2's
``eps_on < eps_off`` -- and belongs to the working-set logic, not here. Keeping the two
apart is what stops a rounding-level tolerance from being tuned for algorithmic stability.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Final

import numpy as np

from cosa.problem.socp import ProblemError, SecondOrderCone

if TYPE_CHECKING:
    from cosa import Vector
    from cosa.problem.socp import ConeProduct

__all__ = [
    "TOLERANCE",
    "ConePosition",
    "is_apex",
    "is_boundary",
    "is_interior",
    "is_member",
    "is_member_of_product",
    "magnitude",
    "position",
    "positions",
    "slack",
]

TOLERANCE: Final = 1e-9
"""The default numerical tolerance for every predicate in this module."""


class ConePosition(enum.Enum):
    """Where a point sits relative to the cone -- a fact about the point.

    Exhaustive and mutually exclusive, which is what makes it usable as the subject of a
    ``match``. It describes geometry only: whether the solver should *act* on the cone
    being active is a different question, answered by
    :class:`cosa.active_set.working_set.ConeStatus`, which also consults the conic
    multiplier because §7.3 warns that "geometric activity alone is not sufficient to
    establish optimality".

    Attributes:
        EXTERIOR: outside the cone, ``||y|| > t``. The conic block is infeasible.
        INTERIOR: strictly inside, ``||y|| < t``. The cone constrains nothing locally.
        BOUNDARY: on the boundary away from the apex, so ``u = y / ||y||`` exists and
            eq. (3) is the local analogue of an active linear constraint.
        APEX: at ``(0, 0)`` within tolerance. On the boundary too, but reported as its
            own case because §8.1 handles it through exact membership and normal-cone
            conditions rather than a tangent hyperplane.
    """

    EXTERIOR = "exterior"
    INTERIOR = "interior"
    BOUNDARY = "boundary"
    APEX = "apex"


def _split(z: Vector) -> tuple[float, Vector]:
    """Split a cone vector into its head and tail, validating its length.

    Args:
        z: a vector of length at least 2, head first.

    Returns:
        The pair ``(t, y)``, with the head as a scalar.

    Raises:
        ProblemError: if ``z`` is not a finite vector of length at least 2. A length-1
            block is the non-negative ray, which belongs in the linear inequalities.
    """
    entries = np.asarray(z, dtype=np.float64)
    if entries.ndim != 1:
        raise ProblemError("cone", f"expected a vector, found an array of shape {entries.shape}")
    return SecondOrderCone(dim=entries.size).split(entries)


def _threshold(tolerance: float, scale: float) -> float:
    """The comparison threshold: absolute below unit scale, relative above it.

    Args:
        tolerance: the relative tolerance.
        scale: the magnitude the comparison is relative to.

    Returns:
        ``tolerance * max(1, scale)``.
    """
    return tolerance * max(1.0, scale)


def slack(z: Vector) -> float:
    """The conic slack ``t - ||y||_2``: positive inside, zero on the boundary.

    The scalar §7.3 thresholds to decide geometric activity, and the quantity every
    predicate here is a sign test on.

    Args:
        z: a point in the cone's coordinates, head first.

    Returns:
        ``t - ||y||_2``.
    """
    head, tail = _split(z)
    return head - float(np.linalg.norm(tail))


def magnitude(z: Vector) -> float:
    """The scale of a cone point, ``max(|t|, ||y||_2)``.

    Used to relativize the tolerance when the caller does not supply a scale of its own.
    It is the two halves' larger norm rather than ``||z||`` so that the head and the tail
    are weighted the way the constraint weights them.

    Args:
        z: a point in the cone's coordinates, head first.

    Returns:
        ``max(|t|, ||y||_2)``, which is zero exactly at the apex.
    """
    head, tail = _split(z)
    return max(abs(head), float(np.linalg.norm(tail)))


def is_apex(z: Vector, *, tolerance: float = TOLERANCE, scale: float | None = None) -> bool:
    """Is ``z`` the apex ``(0, 0)``, within tolerance?

    A predicate of its own, not a corollary of :func:`is_boundary`, because §8.1 branches
    on it: at ``L @ x = 0`` the unit vector ``u`` is undefined and the tangent
    representation has nothing to say. Note that the apex *is* a boundary point, so
    :func:`is_boundary` also holds here; :func:`positions` reports the more specific case.

    A point within tolerance of the apex answers ``True`` whichever side of the boundary
    rounding put it on. That is deliberate: if the point cannot be distinguished from the
    apex, the apex branch is the one that has to run.

    Args:
        z: a point in the cone's coordinates, head first.
        tolerance: the numerical tolerance.
        scale: the problem scale to relativize against. The default derives it from ``z``,
            which is near zero here and so makes the test absolute -- pass the scale of
            the cone block explicitly on a badly scaled instance.

    Returns:
        ``True`` if both the head and the tail vanish to within tolerance.
    """
    size = magnitude(z)
    return size <= _threshold(tolerance, size if scale is None else scale)


def is_member(z: Vector, *, tolerance: float = TOLERANCE, scale: float | None = None) -> bool:
    """Is ``z`` in the cone, ``||y||_2 <= t``, within tolerance?

    Primal feasibility of one conic block. The tolerance is one-sided and forgiving: a
    point just outside by less than the threshold counts as a member, because that is what
    a feasibility check on a computed iterate has to mean.

    Args:
        z: a point in the cone's coordinates, head first.
        tolerance: the numerical tolerance.
        scale: the problem scale to relativize against, or ``None`` to derive it from
            ``z``.

    Returns:
        ``True`` if ``z`` is in the cone up to the tolerance.
    """
    return slack(z) >= -_threshold(tolerance, magnitude(z) if scale is None else scale)


def is_boundary(z: Vector, *, tolerance: float = TOLERANCE, scale: float | None = None) -> bool:
    """Is ``z`` on the boundary, ``t == ||y||_2``, within tolerance?

    True at the apex as well, which is the mathematically honest answer -- the apex
    satisfies ``t = ||y|| = 0``. Callers that need the tangent condition of eq. (3) must
    exclude it with :func:`is_apex`, or ask :func:`positions` and match on
    :attr:`ConePosition.BOUNDARY`, which already does.

    Args:
        z: a point in the cone's coordinates, head first.
        tolerance: the numerical tolerance.
        scale: the problem scale to relativize against, or ``None`` to derive it from
            ``z``.

    Returns:
        ``True`` if the conic slack vanishes to within tolerance.
    """
    return abs(slack(z)) <= _threshold(tolerance, magnitude(z) if scale is None else scale)


def is_interior(z: Vector, *, tolerance: float = TOLERANCE, scale: float | None = None) -> bool:
    """Is ``z`` strictly inside the cone, ``||y||_2 < t``, by more than the tolerance?

    The complement of :func:`is_boundary` within the cone, and the case in which the conic
    block constrains nothing locally: the cone is not in the working set and the direction
    subproblem ignores it.

    Args:
        z: a point in the cone's coordinates, head first.
        tolerance: the numerical tolerance.
        scale: the problem scale to relativize against, or ``None`` to derive it from
            ``z``.

    Returns:
        ``True`` if the conic slack is positive by more than the tolerance.
    """
    return slack(z) > _threshold(tolerance, magnitude(z) if scale is None else scale)


def position(z: Vector, *, tolerance: float = TOLERANCE, scale: float | None = None) -> ConePosition:
    """Classify ``z`` into the one case that describes it best.

    The apex is reported as :attr:`ConePosition.APEX` rather than as a boundary point,
    because that is the distinction §8.1 needs; everything else follows the slack's sign.

    Args:
        z: a point in the cone's coordinates, head first.
        tolerance: the numerical tolerance.
        scale: the problem scale to relativize against, or ``None`` to derive it from
            ``z``.

    Returns:
        The point's position, as one of the four exhaustive cases.
    """
    if is_apex(z, tolerance=tolerance, scale=scale):
        return ConePosition.APEX
    if is_boundary(z, tolerance=tolerance, scale=scale):
        return ConePosition.BOUNDARY
    if is_member(z, tolerance=tolerance, scale=scale):
        return ConePosition.INTERIOR
    return ConePosition.EXTERIOR


def positions(
    cone: ConeProduct,
    vector: Vector,
    *,
    tolerance: float = TOLERANCE,
    scale: float | None = None,
) -> tuple[ConePosition, ...]:
    """Classify every block of a product vector, one position per factor.

    The product form the solver actually holds: a conic slack ``G @ z + h`` is one vector,
    and the working set needs a per-factor verdict on it.

    Args:
        cone: the cone product the vector lives in.
        vector: a vector of length ``cone.dim``, laid out one block per factor.
        tolerance: the numerical tolerance.
        scale: the problem scale to relativize against, applied to every block, or
            ``None`` to derive one per block.

    Returns:
        One position per factor, in factor order. Empty for the empty product, which is
        the linear-programming case.
    """
    return tuple(position(block, tolerance=tolerance, scale=scale) for block in cone.blocks(vector))


def is_member_of_product(
    cone: ConeProduct,
    vector: Vector,
    *,
    tolerance: float = TOLERANCE,
    scale: float | None = None,
) -> bool:
    """Is every block of ``vector`` in its factor of the product?

    Args:
        cone: the cone product the vector lives in.
        vector: a vector of length ``cone.dim``.
        tolerance: the numerical tolerance.
        scale: the problem scale to relativize against, or ``None`` to derive one per
            block.

    Returns:
        ``True`` if ``vector in K``. Vacuously ``True`` for the empty product.
    """
    where = positions(cone, vector, tolerance=tolerance, scale=scale)
    return all(block is not ConePosition.EXTERIOR for block in where)
