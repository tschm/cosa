"""The direction subproblem's KKT system: assembled, factorized afresh, solved.

§4.2 (``paper.tex:352``) states the direction problem for a candidate ``(p, tau)`` as
eq. (4) -- minimize ``grad(f).T @ d + d.T @ H @ d / 2`` subject to the working-set
equations ``a_i.T @ p = 0``, ``E @ p = 0`` and, when the cone is active,
``tau - u.T @ L @ p = 0``. §4.3 (``paper.tex:397``) turns it into the saddle-point system

    [ H    W_k.T ] [ d_k  ]        [ g_k ]
    [ W_k    0   ] [ nu_k ]  =  -  [  0  ]

and this module is that system: :func:`assemble` builds it, :func:`solve` solves it, and
:func:`direction` does both. The plan's own comment applies -- this "creates a direct
connection with classical active-set QP implementations", and the connection is the point:
the conic row is *one more row of* ``W_k``, indistinguishable in shape from an active
linear inequality.

**H = rho*I, and rho is not a tuning parameter.** The objective is linear, so eq. (4) has
no curvature of its own and ``H`` is what makes the direction well defined at all --
§4.2's ``H = rho*I, rho > 0``. What is easy to miss is how little the value matters. From
``rho*d + W.T @ nu = -g`` and ``W @ d = 0`` one gets ``W @ W.T @ nu = -W @ g``, which has
no ``rho`` in it: **the multipliers are exactly rho-invariant, and the direction scales as
1/rho**. So ``rho`` sets the length of ``d`` and nothing else, and the ratio test of #14
and #18 normalizes that length away. It is a well-posedness device, not a knob.

Which is worth being explicit about because the plan flags the confusion itself: this
``rho`` is **not** the degeneracy regularization of §8.3 (``paper.tex:671``). That one
perturbs a rank-deficient ``W`` to make the system solvable at the price of a slightly
wrong answer, and it belongs to #25. This one perturbs nothing -- with ``W`` of full row
rank the system is exactly solvable and ``d`` is exactly the projection of ``-g/rho`` onto
the null space of ``W``. Two regularizations, two purposes, two knobs.

**Refactorized every iteration, deliberately.** §13.1 (``paper.tex:975``): *"The first
implementation will refactor the KKT matrix at every iteration. This minimizes
implementation complexity and provides a reliable reference."* :func:`solve` therefore
holds no state and caches nothing; one call is exactly one factorization. That is what
makes the "number of KKT factorizations" metric (``paper.tex:883``) have a reference
value at all, and it is the number #27's reuse has to beat while #26 checks it still gets
the same answer. Anything here that looked like an optimization would destroy the
baseline it exists to be.

**The row order is published, not implied.** ``nu_k`` has one entry per row of ``W_k``,
and #13 has to map those entries back onto ``y``, ``nu`` and ``w`` to test their signs.
Guessing the order would couple the two issues through an unwritten convention, so
:class:`RowLayout` states it -- active inequalities in ascending row order, then every
equality, then the cones in factor order -- and :meth:`RowLayout.split` performs the
inverse mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from cosa.active_set.working_set import ConeStatus, WorkingSet
from cosa.geometry.soc import TOLERANCE
from cosa.geometry.tangent import tangent_row
from cosa.problem.socp import ProblemError, _vector

if TYPE_CHECKING:
    from cosa import Matrix, Vector
    from cosa.problem.socp import SOCP

__all__ = [
    "RHO",
    "ConeRows",
    "Direction",
    "KktSystem",
    "RowLayout",
    "SingularKktError",
    "assemble",
    "direction",
    "solve",
    "working_set_matrix",
]

RHO: Final = 1.0
"""The default ``rho`` of ``H = rho*I``.

One, because the value is only the length of the direction -- see the module docstring --
and a direction of ``-g`` projected onto the working set's null space is the least
surprising thing for a reader stepping through an iteration to see.
"""


class SingularKktError(RuntimeError):
    """The KKT matrix is singular, so the multipliers are not determined.

    Raised rather than worked around, because what has gone wrong is precise and is
    somebody else's issue: ``W_k`` has linearly dependent rows. The direction ``d`` is
    still unique -- ``H`` is positive definite and the constraint ``W @ d = 0`` does not
    care about redundancy -- but ``nu_k`` is not, and returning one arbitrary member of an
    infinite family would hand #13's sign tests a number with no meaning.

    §8.3 (``paper.tex:671``) is where the plan addresses this, through rank detection and
    the removal of dependent constraints, and #25 owns it. Until then a degenerate working
    set is a diagnosable stop rather than a silent wrong answer -- see :func:`solve` for why
    that took an explicit rank test rather than catching ``LinAlgError``, and #33's
    degenerate-optimum family for the instance that showed the difference.
    """

    def __init__(self, rows: int, variables: int) -> None:
        """Report the shape of the system that failed.

        Args:
            rows: the number of working-set rows.
            variables: the number of variables.
        """
        super().__init__(
            f"the {variables + rows}-by-{variables + rows} KKT matrix is singular, so the "
            f"{rows} working-set rows are linearly dependent. The direction is still well "
            "defined; the multipliers are not. Rank detection is paper.tex:671 and issue #25"
        )


@dataclass(frozen=True)
class ConeRows:
    """The rows one active cone factor contributes to ``W_k``.

    Attributes:
        factor: the factor's index in the problem's cone product.
        status: the status that produced these rows, which says what they mean -- one
            tangent row for :attr:`cosa.ConeStatus.TANGENT`, the whole block pinned for
            :attr:`cosa.ConeStatus.APEX`.
        rows: where they sit in ``W_k``.
    """

    factor: int
    status: ConeStatus
    rows: slice


@dataclass(frozen=True)
class RowLayout:
    """The row order of ``W_k``, and how to read a multiplier vector back apart.

    Built from a working set alone, so it can be computed without assembling anything --
    which is what lets #13 know the layout of a ``nu_k`` it did not assemble, and #15 size
    its instrumentation.

    Attributes:
        inequalities: the ``A``-row index of each inequality row of ``W_k``, ascending.
        equalities: the ``E``-row index of each equality row, which is all of them in
            order, because §3.1 never drops one.
        cones: one entry per *active* factor, in factor order.
    """

    inequalities: tuple[int, ...]
    equalities: tuple[int, ...]
    cones: tuple[ConeRows, ...]

    @classmethod
    def for_working_set(cls, working_set: WorkingSet) -> RowLayout:
        """Derive the layout the assembly will use.

        Args:
            working_set: the set whose rows are being laid out.

        Returns:
            The layout: active inequalities first, then the equalities, then the cones.
        """
        start = len(working_set.inequalities) + working_set.num_equalities
        cones = []
        for factor, status in enumerate(working_set.cone_status):
            width = status.num_rows(working_set.cone.cones[factor])
            if width:
                cones.append(ConeRows(factor=factor, status=status, rows=slice(start, start + width)))
                start += width
        return cls(
            inequalities=working_set.inequalities,
            equalities=tuple(range(working_set.num_equalities)),
            cones=tuple(cones),
        )

    @property
    def num_rows(self) -> int:
        """The number of rows of ``W_k``, and so the length of ``nu_k``."""
        return (
            len(self.inequalities) + len(self.equalities) + sum(rows.rows.stop - rows.rows.start for rows in self.cones)
        )

    @property
    def inequality_rows(self) -> slice:
        """Where the active inequalities sit in ``W_k``."""
        return slice(0, len(self.inequalities))

    @property
    def equality_rows(self) -> slice:
        """Where the equalities sit in ``W_k``."""
        start = len(self.inequalities)
        return slice(start, start + len(self.equalities))

    def split(self, nu: Vector) -> tuple[Vector, Vector, tuple[Vector, ...]]:
        """Cut a multiplier vector into its inequality, equality and conic parts.

        The inverse of the layout. The inequality part is indexed by position, not by
        ``A``-row: entry ``j`` belongs to row ``self.inequalities[j]``. #13 is what turns
        that into a full-length ``y`` with zeros on the inactive rows.

        Args:
            nu: the multipliers, ``(num_rows,)``.

        Returns:
            The triple ``(active inequality multipliers, equality multipliers, one block
            per active cone factor)``.

        Raises:
            ProblemError: if ``nu`` is not of length :attr:`num_rows`.
        """
        multipliers = _vector("nu", nu, size=self.num_rows)
        return (
            multipliers[self.inequality_rows],
            multipliers[self.equality_rows],
            tuple(multipliers[rows.rows] for rows in self.cones),
        )


def working_set_matrix(
    problem: SOCP,
    working_set: WorkingSet,
    z: Vector,
    *,
    tolerance: float = TOLERANCE,
) -> Matrix:
    """Assemble ``W_k``: the working-set equations as rows over the variables.

    Three kinds of row, in the order :class:`RowLayout` publishes:

    * ``a_i.T`` for each active inequality -- §3.1's ``a_i.T @ p = 0``;
    * every row of ``E`` -- §3.1's ``E @ p = 0``, unconditionally;
    * for each active cone factor, either its single tangent row of eq. (3), via
      :func:`cosa.geometry.tangent.tangent_row`, or its whole block of ``G`` when the
      status is :attr:`cosa.ConeStatus.APEX`. Pinning the block holds the slack at the
      apex exactly, which is §8.1's "exact SOC membership" treatment rather than a
      hyperplane; the normal-cone conditions that decide *whether* to hold it there are
      #24's.

    A pinned apex block is only *usable* where the apex is reachable, and those are the same
    condition. Its ``cone.dim`` rows are ``1 + rank(L)``, so on a full-rank covariance they
    already determine the whole direction -- ``G`` is square and invertible, ``G @ d = 0``
    forces ``d = 0`` -- and one equality row on top of that makes the system dependent, which
    :func:`solve` now reports. That is not a limitation to work around: with ``L`` invertible,
    ``L @ x = 0`` implies ``x = 0``, so the apex is not reachable at any point a budget
    equality admits. Rank deficiency is what makes the apex reachable *and* what leaves the
    pinned system room to move, and #24's fixtures are rank-deficient for that reason.

    Args:
        problem: the instance.
        working_set: what is currently believed active.
        z: the current point, needed for the tangent rows' ``u``.
        tolerance: the vanishing-tail tolerance passed to the tangent row.

    Returns:
        ``W_k``, of shape ``(working_set.num_rows, n)``.

    Raises:
        ApexError: if a factor's status is :attr:`cosa.ConeStatus.TANGENT` but its slack
            has no tangent -- the working set believes something the geometry denies, and
            the guard of #17 is what surfaces it.
        ProblemError: if the working set's shape does not match the problem's.
    """
    _require_same_shape(problem, working_set)
    point = _vector("z", z, size=problem.num_variables)
    blocks: list[Matrix] = [problem.A[list(working_set.inequalities), :], problem.E]
    slack = problem.cone_slack(point)
    for status, block in zip(working_set.cone_status, problem.cone.slices, strict=True):
        rows = problem.G[block, :]
        match status:
            case ConeStatus.INACTIVE:
                continue
            case ConeStatus.TANGENT:
                blocks.append(tangent_row(slack[block], rows, tolerance=tolerance).reshape(1, -1))
            case ConeStatus.APEX:
                blocks.append(rows)
    return np.vstack(blocks)


@dataclass(frozen=True, eq=False)
class KktSystem:
    """The assembled saddle-point system of §4.3, ready to factorize.

    Kept as a value rather than solved on the spot so that the matrix can be inspected,
    its conditioning measured -- which is what #26's factorization comparison and #41's
    stability tracker will want -- and the same system solved for more than one
    right-hand side.

    Attributes:
        matrix: the ``(n + m, n + m)`` symmetric indefinite matrix
            ``[[rho*I, W.T], [W, 0]]``.
        rhs: the right-hand side ``(-g, 0)``.
        W: the working-set matrix, ``(m, n)``, kept because every consumer that checks
            ``W @ d = 0`` needs it and recovering it from the assembled matrix is silly.
        gradient: ``g_k``, the objective gradient, ``(n,)``.
        rho: the ``rho`` of ``H = rho*I``.
        layout: the row order of ``W``.
        regularization: the ``delta`` of §8.3 the system was assembled with, zero for an
            unregularized one. Recorded because a solution obtained from a regularized
            system answers a nearby question, and a consumer comparing two solutions needs
            to know which.
    """

    matrix: Matrix
    rhs: Vector
    W: Matrix
    gradient: Vector
    rho: float
    layout: RowLayout
    regularization: float = 0.0

    @property
    def num_variables(self) -> int:
        """The number of variables ``n``."""
        return self.gradient.size

    @property
    def num_rows(self) -> int:
        """The number of working-set rows ``m``."""
        return self.W.shape[0]


def assemble(
    problem: SOCP,
    working_set: WorkingSet,
    z: Vector,
    *,
    rho: float = RHO,
    tolerance: float = TOLERANCE,
    regularization: float = 0.0,
) -> KktSystem:
    """Build the system of §4.3 for the current working set and point.

    The gradient is ``problem.c``, because the SOCP objective ``c.T @ z`` is linear and so
    ``grad(f)`` is constant -- for eq. (7) that is ``(-mu, lam)``, exactly the ``grad(f)``
    §4.2 writes. Nothing here evaluates the objective at ``z``; ``z`` is needed only for
    the tangent rows.

    Args:
        problem: the instance.
        working_set: what is currently believed active.
        z: the current point.
        rho: the ``rho`` of ``H = rho*I``, which must be positive.
        tolerance: the vanishing-tail tolerance passed to the tangent rows.
        regularization: ``delta``, placed as ``-delta*I`` in the ``(2, 2)`` block. §8.3's
            regularization, and **not** ``rho``: this one perturbs the problem to make a
            dependent working set solvable, at the price of an answer to a nearby question,
            whereas ``rho`` perturbs nothing. Zero leaves the system exactly as §4.3 prints
            it. See #25.

    Returns:
        The assembled system.

    Raises:
        ProblemError: if ``rho`` is not a positive finite number, or if the working set's
            shape does not match the problem's.
    """
    if not np.isfinite(rho) or rho <= 0.0:
        raise ProblemError("rho", f"§4.2 takes rho > 0, found {rho}")
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ProblemError("regularization", f"§8.3 takes delta >= 0, found {regularization}")
    matrix_w = working_set_matrix(problem, working_set, z, tolerance=tolerance)
    rows, variables = matrix_w.shape
    matrix = np.zeros((variables + rows, variables + rows))
    matrix[:variables, :variables] = rho * np.eye(variables)
    matrix[:variables, variables:] = matrix_w.T
    matrix[variables:, :variables] = matrix_w
    if regularization:
        matrix[variables:, variables:] = -regularization * np.eye(rows)
    rhs = np.concatenate([-problem.c, np.zeros(rows)])
    return KktSystem(
        matrix=matrix,
        rhs=rhs,
        W=matrix_w,
        gradient=problem.c,
        rho=float(rho),
        layout=RowLayout.for_working_set(working_set),
        regularization=float(regularization),
    )


@dataclass(frozen=True, eq=False)
class Direction:
    """The solution of the direction subproblem: where to move, and at what multipliers.

    Attributes:
        d: the direction ``d_k = (p_k, tau_k)`` over all variables, ``(n,)``. The split
            into ``p`` and ``tau`` is the problem's, not this module's: for eq. (7) ``tau``
            is the last entry, because that is where :meth:`cosa.MeanStdForm.to_socp` puts
            ``t``.
        multipliers: ``nu_k``, one per working-set row, ``(m,)``. Laid out by
            :attr:`layout`.
        layout: the row order of the working-set matrix, so :meth:`RowLayout.split` can
            read :attr:`multipliers` apart.
        rho: the ``rho`` the direction was computed with. Recorded because ``d`` scales as
            ``1/rho`` while :attr:`multipliers` does not, so comparing two directions
            computed at different ``rho`` means rescaling one of them.
    """

    d: Vector
    multipliers: Vector
    layout: RowLayout
    rho: float

    def directional_derivative(self, gradient: Vector) -> float:
        """The rate of change of the objective along ``d``: ``g.T @ d``.

        Always ``-rho * ||d||^2``, hence never positive -- which is the one guarantee the
        direction subproblem has to make and the reason ``H`` is positive definite. It is
        computed from ``g`` rather than from that identity so that the identity stays a
        testable claim about the solve rather than an assumption baked into the reader.

        Args:
            gradient: the objective gradient the direction was computed for.

        Returns:
            ``g.T @ d``, which is zero exactly when ``d`` is zero.
        """
        return float(_vector("g", gradient, size=self.d.size) @ self.d)


def solve(system: KktSystem, *, rank_tolerance: float | None = None) -> Direction:
    """Factorize and solve one assembled system -- one call, one factorization.

    The factorization is ``numpy.linalg.solve``'s dense LU, which handles the symmetric
    indefinite matrix correctly and needs no SciPy. That is not the eventual answer: §13
    wants a sparse ``LDL^T`` or a null-space method, and #26 compares them. It is the
    *reference* answer, which is what §13.1 asks the first implementation to be, and it is
    deliberately the least clever thing that is right.

    **The rank of ``W`` is checked before the solve, not inferred from it.** The obvious
    implementation -- solve, and catch ``LinAlgError`` -- does not work, and the
    degenerate-optimum family of #33 is what showed it. ``numpy.linalg.solve`` raises only
    on an *exactly* zero pivot, and a genuinely dependent working set produces a pivot of
    ``1e-18`` rather than zero: LAPACK returns, the residual is small because LU always
    makes the residual small, and the multipliers are enormous garbage. So the check that
    made :class:`SingularKktError` a real behaviour rather than a defensive branch is an
    explicit rank test on ``W``, which is both more direct -- dependent rows are what the
    error is *about* -- and cheaper, ``W`` being smaller than the assembled matrix.

    An SVD per solve is a real cost, and it is the right one to pay here: §13.1 asks this
    implementation to be reliable rather than fast, and a reference that silently returns
    garbage on a degenerate instance is not a reference. #26 and #27, which are allowed to
    be fast, will need a cheaper test.

    Args:
        system: the assembled system.
        rank_tolerance: the singular-value threshold below which a direction of ``W``
            counts as absent, or ``None`` for ``numpy.linalg.matrix_rank``'s default of
            ``max(m, n) * eps * sigma_max``. Raising it catches *nearly* dependent rows
            too -- which is #25's problem, not this module's, so the knob is here and the
            policy is not.

    Returns:
        The direction and its multipliers.

    Raises:
        SingularKktError: if ``W`` has linearly dependent rows.
    """
    # A regularized system is nonsingular by construction, so the rank test is skipped:
    # a caller that asked for regularization has already decided to accept a nearby answer
    # rather than a refusal, and refusing anyway would make the option useless.
    if (
        not system.regularization
        and system.num_rows
        and int(np.linalg.matrix_rank(system.W, tol=rank_tolerance)) < system.num_rows
    ):
        raise SingularKktError(system.num_rows, system.num_variables)
    try:
        solution = np.linalg.solve(system.matrix, system.rhs)
    except np.linalg.LinAlgError as singular:  # pragma: no cover - the rank test catches this first
        raise SingularKktError(system.num_rows, system.num_variables) from singular
    variables = system.num_variables
    return Direction(
        d=solution[:variables],
        multipliers=solution[variables:],
        layout=system.layout,
        rho=system.rho,
    )


def direction(
    problem: SOCP,
    working_set: WorkingSet,
    z: Vector,
    *,
    rho: float = RHO,
    tolerance: float = TOLERANCE,
    rank_tolerance: float | None = None,
    regularization: float = 0.0,
) -> Direction:
    """Assemble and solve in one call -- what step 5 of the §4.1 iteration does.

    Args:
        problem: the instance.
        working_set: what is currently believed active.
        z: the current point.
        rho: the ``rho`` of ``H = rho*I``.
        tolerance: the vanishing-tail tolerance passed to the tangent rows.
        rank_tolerance: the singular-value threshold for the dependent-row check.
        regularization: §8.3's ``delta``. A positive value makes a dependent working set
            solvable instead of refused -- see :func:`assemble`.

    Returns:
        The direction and its multipliers.

    Raises:
        SingularKktError: if the working-set rows are linearly dependent and no
            regularization was asked for.
    """
    system = assemble(problem, working_set, z, rho=rho, tolerance=tolerance, regularization=regularization)
    return solve(system, rank_tolerance=rank_tolerance)


def _require_same_shape(problem: SOCP, working_set: WorkingSet) -> None:
    """Check that the working set describes this problem's shape.

    The working set deliberately carries a shape rather than a problem, so that a warm
    start can move it between instances -- which makes it worth checking that the instance
    it has been moved to is actually the same shape.

    Args:
        problem: the instance.
        working_set: the set to check against it.

    Raises:
        ProblemError: if the row counts or the cone product disagree.
    """
    found = (working_set.num_inequalities, working_set.num_equalities, working_set.cone)
    wanted = (problem.num_inequalities, problem.num_equalities, problem.cone)
    if found != wanted:
        raise ProblemError(
            "shape",
            f"the working set is over {found[0]} inequalities, {found[1]} equalities and "
            f"{len(found[2])} cones, the problem over {wanted[0]}, {wanted[1]} and {len(wanted[2])}",
        )
