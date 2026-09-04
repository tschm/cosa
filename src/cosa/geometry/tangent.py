"""The tangent hyperplane and the normal at a nonzero boundary point of the cone.

This is the module that lets a second-order cone enter a working set at all. §3.2
(``paper.tex:264``) writes the construction in the portfolio's own coordinates: at a
boundary point where ``t = ||L @ x||_2``, with

    u = L @ x / ||L @ x||_2,

a candidate direction ``(p, tau)`` keeps the point on the boundary exactly when

    tau - u.T @ L @ p = 0,                                              (eq. 3)

which the plan calls "the local analogue of an active linear constraint". One scalar
equation, exactly like one row of ``A @ p = 0`` -- that equivalence is what makes a conic
active-set method conceivable, and it is why an active cone costs the direction subproblem
a single row.

**Written in slack coordinates, not portfolio coordinates.** ``(t, L @ x)`` is the conic
slack ``G @ z + h`` of one factor, and ``(tau, L @ p)`` is the corresponding slack
direction ``G @ p``. Everything here therefore takes a cone vector ``s = (s_0, s_1)`` and
a slack direction, head first as :class:`cosa.SecondOrderCone` fixes it. Eq. (3) is
recovered by :func:`tangent_row`, which pushes the covector back through ``G`` to give a
row over ``z`` -- the form the KKT assembly of #12 needs. The generalization costs nothing
and buys the cone product: eq. (3) holds per factor, and the plan's ``L`` is just the tail
rows of that factor's ``G``.

**One covector, three jobs.** At a boundary point with ``s_1 != 0``, the vector

    w = (1, -u)

is simultaneously the covector of eq. (3), the outward normal's negative, and -- up to a
non-negative scale -- the *only* dual variable in ``Q`` that is complementary to ``s``.
That last one is not a coincidence and is worth stating, because it is what ties this
module to the sign convention of #9: complementarity ``w.T @ s = 0`` with ``w in Q`` and
``s`` on the boundary forces ``w = w_0 * (1, -u)``. Read the other way, an active cone's
multiplier has exactly one degree of freedom, its magnitude, and #13's multiplier
computation and #23's deactivation test are both statements about that single number. The
golden instances of ``tests/test_socp.py`` are the arithmetic check: their hand-derived
``w`` is this covector scaled by ``lam``.

**Curvature is here too, and it is what "the tangent alone" is missing.** §3.3
(``paper.tex:301``) says the tangent representation is the *initial* one and the final
algorithm is to be "formulated in terms of the primal-dual conic KKT conditions". The gap
between those two is exactly one derivative: the tangent gives the constraint's *gradient*
at a boundary point, and the boundary is curved, so a method that knows only the gradient
crawls along it. :func:`curvature` supplies the second derivative, and #23 puts it into the
direction subproblem weighted by the conic multiplier -- which is what makes that
computation primal-*dual* rather than primal with a dual afterthought.

**The apex is refused, loudly.** ``u`` does not exist at ``L @ x = 0``, and §8.1
(``paper.tex:623``) says so: the apex "has a different tangent and normal geometry from a
smooth nonzero boundary point". Every function here raises :class:`ApexError` rather than
returning a value there. Silently returning a garbage direction -- a zero row, an
arbitrary unit vector, a NaN -- is the failure mode the guard exists to prevent, because
the direction subproblem would accept it and the resulting step would be wrong in a way
no residual check attributes back to here. The apex branch is #24's, and it uses exact
membership and normal-cone conditions instead of a hyperplane.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cosa.geometry.soc import TOLERANCE, is_apex, is_boundary, magnitude, slack
from cosa.problem.socp import ProblemError, SecondOrderCone

if TYPE_CHECKING:
    from cosa import Matrix, Vector

__all__ = [
    "ApexError",
    "NotOnBoundaryError",
    "curvature",
    "outward_normal",
    "require_boundary",
    "tangent_covector",
    "tangent_residual",
    "tangent_row",
    "unit_tail",
]


class ApexError(ProblemError):
    """The tangent geometry was asked for where the tail vanishes and ``u`` does not exist.

    A subclass of :class:`cosa.ProblemError`, so a caller that catches problem errors
    catches this too, but a distinct type because it is the one error here that a *caller*
    is expected to handle rather than fix: reaching the apex is a legitimate state of the
    algorithm, and §8.1's answer is to take a different branch, not to correct the input.
    #24 is that branch.

    Named for the apex because that is the paper's name for the case, but the condition it
    tests is the paper's too, and it is slightly wider: §8.1 (``paper.tex:623``) is headed
    *The Point Lx = 0* and only then observes that the apex ``(t, L @ x) = (0, 0)`` has a
    different geometry. What is undefined is ``L @ x / ||L @ x||``, so the trigger is the
    *tail* vanishing. That also catches ``(s_0, 0)`` with ``s_0 != 0``, which is not the
    apex and is never on the boundary, but has no ``u`` either.
    """

    def __init__(self) -> None:
        """State the one thing that has gone wrong, since there is only one."""
        super().__init__(
            "apex",
            "the cone block's tail has vanished, so u = s_1 / ||s_1|| is undefined and there "
            "is no tangent hyperplane; see paper.tex:623 and take the normal-cone branch",
        )


class NotOnBoundaryError(ProblemError):
    """The tangent geometry was asked for at a point that is not on the boundary.

    Eq. (3) is a statement about a boundary point. Evaluating it strictly inside or
    strictly outside the cone would produce a hyperplane through a point that is not on the
    surface it is supposed to be tangent to -- arithmetic that runs and means nothing.
    Raised only by the routines that need the point to be on the boundary, and only when
    the caller asks to be checked.
    """

    def __init__(self, gap: float) -> None:
        """Report how far from the boundary the point was.

        Args:
            gap: the conic slack ``s_0 - ||s_1||``, positive inside and negative outside.
        """
        where = "strictly inside" if gap > 0.0 else "outside"
        super().__init__("boundary", f"the point is {where} the cone, with a conic slack of {gap:g}")


def _split(s: Vector) -> tuple[float, Vector]:
    """Split a cone vector into head and tail, validating its shape.

    Args:
        s: a vector of length at least 2, head first.

    Returns:
        The pair ``(s_0, s_1)``.

    Raises:
        ProblemError: if ``s`` is not a finite vector of length at least 2.
    """
    entries = np.asarray(s, dtype=np.float64)
    if entries.ndim != 1:
        raise ProblemError("cone", f"expected a vector, found an array of shape {entries.shape}")
    return SecondOrderCone(dim=entries.size).split(entries)


def unit_tail(s: Vector, *, tolerance: float = TOLERANCE, scale: float | None = None) -> Vector:
    """The unit vector ``u = s_1 / ||s_1||_2`` of §3.2.

    The plan's ``u = L @ x / ||L @ x||_2``, in slack coordinates. It points along the tail,
    which is the direction the cone's boundary curves around, and it is the only thing the
    tangent hyperplane and the complementary multiplier depend on.

    Args:
        s: the cone vector, head first. Only its tail is used, so this is well defined at
            any point with a nonzero tail -- boundary or not.
        tolerance: the tolerance for the vanishing-tail test.
        scale: the problem scale for that test, or ``None`` to derive it from ``s``.

    Returns:
        The unit vector along the tail, of length ``s.size - 1``.

    Raises:
        ApexError: if the tail vanishes, which includes the apex.
        ProblemError: if ``s`` is not a finite vector of length at least 2.
    """
    _, tail = _split(s)
    norm = float(np.linalg.norm(tail))
    # The precondition for `u` is a nonzero *tail*, which is a slightly wider condition
    # than "not the apex" and is the one §8.1 actually names. Testing it directly means
    # the apex needs no separate branch: its tail is zero, so it is caught here.
    if norm <= tolerance * max(1.0, magnitude(s) if scale is None else scale):
        raise ApexError()
    return tail / norm


def tangent_covector(s: Vector, *, tolerance: float = TOLERANCE, scale: float | None = None) -> Vector:
    """The covector ``(1, -u)`` of eq. (3), in this factor's slack coordinates.

    Applied to a slack direction ``(tau, ds_1)`` it gives ``tau - u.T @ ds_1``, so the
    tangent condition is ``tangent_covector(s) @ ds == 0``. It is also, up to a
    non-negative scale, the unique dual variable in ``Q`` complementary to ``s`` -- see the
    module docstring, and :data:`cosa.SIGN_CONVENTION` for why the multiplier lives in
    ``Q`` rather than in ``-Q``.

    Args:
        s: the boundary point, head first.
        tolerance: the tolerance for the vanishing-tail test.
        scale: the problem scale for that test, or ``None`` to derive it from ``s``.

    Returns:
        The covector ``(1, -u)``, of the same length as ``s``.

    Raises:
        ApexError: if ``s``'s tail vanishes, which includes the apex.
    """
    return np.concatenate([[1.0], -unit_tail(s, tolerance=tolerance, scale=scale)])


def outward_normal(s: Vector, *, tolerance: float = TOLERANCE, scale: float | None = None) -> Vector:
    """The outward normal ``(-1, u)`` to the cone's boundary at ``s``.

    The gradient of the constraint function ``g(s) = ||s_1||_2 - s_0``, which is
    non-positive on the cone, so it points out of the cone. The normal cone of ``Q`` at a
    nonzero boundary point is the ray it spans: ``N_Q(s) = {rho * (-1, u) : rho >= 0}``,
    a single ray, which is what makes a nonzero boundary point *smooth* and the apex not.

    It is exactly ``-``:func:`tangent_covector`. Both are provided because the sign is
    where mistakes live: the multiplier convention wants the one in ``Q``, and the
    geometry of "which way is out" wants the other, and a reader should never have to
    infer which a bare ``normal`` meant.

    Args:
        s: the boundary point, head first.
        tolerance: the tolerance for the vanishing-tail test.
        scale: the problem scale for that test, or ``None`` to derive it from ``s``.

    Returns:
        The outward normal ``(-1, u)``.

    Raises:
        ApexError: if ``s`` is at the apex, where the normal cone is not a ray.
    """
    return -tangent_covector(s, tolerance=tolerance, scale=scale)


def tangent_residual(
    s: Vector,
    ds: Vector,
    *,
    tolerance: float = TOLERANCE,
    scale: float | None = None,
) -> float:
    """The left-hand side of eq. (3): ``tau - u.T @ ds_1``.

    The linearized rate at which the slack ``s_0 - ||s_1||`` grows along ``ds``, so its
    sign classifies the direction at first order:

    * **zero** -- tangent. The direction stays on the boundary to first order, which is
      what the working set imposes on an active cone.
    * **positive** -- entering. The direction moves into the interior, and the cone stops
      being active along it.
    * **negative** -- leaving. The direction moves out of the cone, so a step along it is
      limited by the exact quadratic of eq. (6) rather than by this linearization -- that
      is #18's business, and this sign is only the warning.

    Args:
        s: the boundary point, head first.
        ds: the slack direction ``(tau, ds_1)``, head first, of the same length as ``s``.
        tolerance: the tolerance for the vanishing-tail test.
        scale: the problem scale for that test, or ``None`` to derive it from ``s``.

    Returns:
        ``tau - u.T @ ds_1``.

    Raises:
        ApexError: if ``s``'s tail vanishes, which includes the apex.
        ProblemError: if ``ds`` does not have the same length as ``s``.
    """
    unit = unit_tail(s, tolerance=tolerance, scale=scale)
    head, tail = SecondOrderCone(dim=unit.size + 1).split(ds)
    return head - float(unit @ tail)


def tangent_row(
    s: Vector,
    rows: Matrix,
    *,
    tolerance: float = TOLERANCE,
    scale: float | None = None,
) -> Vector:
    """Eq. (3) as a single row over the variables ``z``.

    Pushes :func:`tangent_covector` back through this factor's rows of ``G``: with
    ``rows = [g_0; G_1]`` and the slack direction ``ds = rows @ p``, eq. (3) reads

        (g_0 - u.T @ G_1) @ p = 0,

    which is one row of the ``W_k`` of §4.3 -- the same shape as an active inequality's
    ``a_i.T @ p = 0``. For eq. (7), where ``g_0`` selects ``t`` and ``G_1`` is ``L``, this
    is literally ``tau - u.T @ L @ p``.

    Args:
        s: the boundary point of this factor, head first.
        rows: this factor's rows of ``G``, ``(s.size, n)``, head row first.
        tolerance: the tolerance for the vanishing-tail test.
        scale: the problem scale for that test, or ``None`` to derive it from ``s``.

    Returns:
        The row, of length ``n``.

    Raises:
        ApexError: if ``s``'s tail vanishes, which includes the apex.
        ProblemError: if ``rows`` does not have one row per entry of ``s``.
    """
    covector = tangent_covector(s, tolerance=tolerance, scale=scale)
    block = np.asarray(rows, dtype=np.float64)
    if block.ndim != 2 or block.shape[0] != covector.size:
        raise ProblemError(
            "G",
            f"expected a matrix with {covector.size} rows, one per entry of the cone block, "
            f"found an array of shape {block.shape}",
        )
    return covector @ block


def require_boundary(s: Vector, *, tolerance: float = TOLERANCE, scale: float | None = None) -> None:
    """Check that ``s`` is on the boundary and away from the apex, or raise.

    The precondition of eq. (3), spelled out as its own call so that a caller can assert
    it once per iteration rather than trusting it. The routines above deliberately do
    *not* check it: :func:`unit_tail` is well defined off the boundary and the working-set
    logic legitimately evaluates a tangent row at an iterate that is a rounding error away
    from it. What is never legitimate is the apex, which every routine refuses regardless.

    Args:
        s: the point to check, head first.
        tolerance: the boundary and apex tolerance.
        scale: the problem scale, or ``None`` to derive it from ``s``.

    Raises:
        ApexError: if ``s``'s tail vanishes, which includes the apex.
        NotOnBoundaryError: if ``s`` is not on the boundary within the tolerance.
    """
    if is_apex(s, tolerance=tolerance, scale=scale):
        raise ApexError()
    if not is_boundary(s, tolerance=tolerance, scale=scale):
        raise NotOnBoundaryError(slack(s))


def curvature(s: Vector, *, tolerance: float = TOLERANCE, scale: float | None = None) -> Matrix:
    """The second derivative of ``g(s) = ||s_1|| - s_0`` at a boundary point.

    The constraint function is linear in the head and a norm in the tail, so all of its
    curvature lives in the tail block:

        d^2 g / ds^2 = [[0, 0], [0, (I - u u.T) / ||s_1||]]

    which is the projector off ``u``, scaled by the distance from the axis. Two properties
    make it the right thing to hand a direction subproblem:

    * it is **positive semidefinite**, so adding a non-negative multiple of it to ``rho*I``
      keeps the subproblem convex and its solution unique. That is not automatic for a
      Lagrangian Hessian in general -- it is a gift of the constraint being convex.
    * it is **singular along** ``u``. Moving radially -- straight out along the tail --
      changes ``||s_1||`` at a constant rate, so there is no curvature in that direction,
      and the matrix says so. The curvature is entirely in the directions that *turn*.

    It also grows without bound as ``||s_1||`` falls, which is the apex announcing itself
    from a distance: the boundary's curvature is what becomes infinite there, not merely the
    tangent's definition that fails. #24's branch is the answer, and this function refuses
    the point rather than returning an enormous matrix.

    Args:
        s: the boundary point, head first.
        tolerance: the vanishing-tail tolerance.
        scale: the problem scale for that test, or ``None`` to derive it from ``s``.

    Returns:
        The full ``(s.size, s.size)`` second derivative, in this factor's slack coordinates.
        The head row and column are zero.

    Raises:
        ApexError: if ``s``'s tail vanishes, which includes the apex.
    """
    unit = unit_tail(s, tolerance=tolerance, scale=scale)
    _, tail = _split(s)
    norm = float(np.linalg.norm(tail))
    block = (np.eye(unit.size) - np.outer(unit, unit)) / norm
    second = np.zeros((unit.size + 1, unit.size + 1))
    second[1:, 1:] = block
    return second
