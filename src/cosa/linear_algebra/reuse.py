"""Keeping the factorization across an iteration instead of throwing it away.

§13.2 (``paper.tex:983``) observes that "the working set changes by only a few constraints
per iteration, so there is substantial structure to exploit", and #27 is the exploiting.
#12 deliberately refactorizes every iteration so that there is an honest baseline to beat;
this module is what beats it.

**What is factorized, and why it is ``W.T`` rather than the saddle-point matrix.** The
obvious object to keep is the ``(n+m)``-by-``(n+m)`` KKT matrix itself, and it is the wrong
one. Adding a constraint borders it with a new row *and* a new column, so an update has to
touch both triangles; and #23 made the ``(1, 1)`` block move every iteration a cone is
active, so most of what was kept would be stale anyway.

``W.T`` has neither problem. A constraint entering the working set appends one *column* to
it, a constraint leaving deletes one, and the tangent row moving replaces one -- three
operations on the same object, each an ``O(n^2)`` sequence of Givens rotations against an
``O(n^2 m)`` refactorization. And a QR of ``W.T`` is exactly what the null-space route of
§13 needs: with ``W.T = Q R`` and ``Q = [Q1 Q2]`` split at column ``m``, the columns of
``Q2`` span ``{p : W @ p = 0}``, so

    Q2.T @ H @ Q2 @ v = -Q2.T @ g,      d = Q2 @ v,
    R1 @ nu = -Q1.T @ (g + H @ d),

which is one small symmetric solve and one triangular solve. When ``H`` is ``rho*I`` the
first collapses to ``v = -Q2.T @ g / rho`` and costs nothing at all.

**The third case is the one the paper flags, and it is the reason this module exists.**
§13.2 warns that "the SOC tangent changes continuously with ``x``, so this part is more
subtle than ordinary linear constraint updates" -- a linear row enters and leaves discretely,
while the tangent row moves on *every* iteration the cone is active. That sounds like it
defeats reuse, and it does defeat reuse of the whole factorization. What it does not defeat
is reuse of the *rest*: the tangent occupies known columns of ``W.T``, and replacing a column
is a delete followed by an insert. The linear structure, which is most of the matrix, is
never refactorized. :func:`replace` is that operation and it is why the counter falls on
instances whose cone is active throughout.

**What is *not* reused, stated plainly.** #23's curvature makes ``H`` change every
iteration, and result 10 of #43 records that the term has rank ``k - 2`` for a factor of
dimension ``k`` -- on eq. (7) that is ``n - 1``, which is not a low-rank update to route
around. So the reduced Hessian ``Q2.T @ H @ Q2`` is re-formed and re-factorized every
iteration a cone is active. It is ``(n - m)``-by-``(n - m)`` rather than ``(n + m)``-by-
``(n + m)``, and on an active-set method's working sets that is the cheap end; but it is
work, and pretending otherwise would make the metric a lie.

**The cache is keyed on the working set, not threaded through the loop.** :class:`Reuse`
holds one factorization and the set it belongs to, and each call compares the two. That is
deliberate: §4.1's iteration has six paths that can change a working set -- a blocking row,
a dropped row, a dependent-row removal, a cone activation, a cone deactivation, and the apex
branch -- and threading a stateful object through all six is how a stale factorization gets
used by mistake. Comparing is cheap, cannot go stale, and gives every path the update for
free.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import scipy.linalg

from cosa.geometry.soc import TOLERANCE
from cosa.linear_algebra.kkt import RHO, Direction, RowLayout, SingularKktError, working_set_matrix
from cosa.problem.socp import ProblemError

if TYPE_CHECKING:
    from cosa import Matrix, Vector
    from cosa.active_set.working_set import WorkingSet
    from cosa.problem.socp import SOCP

__all__ = [
    "UPDATE_BUDGET",
    "Factorization",
    "Reuse",
    "delete",
    "factorize",
    "insert",
    "replace",
]


_qr_insert = getattr(scipy.linalg, "qr_insert")  # noqa: B009
_qr_delete = getattr(scipy.linalg, "qr_delete")  # noqa: B009
"""SciPy's QR update routines, resolved once at import.

Reached through ``getattr`` rather than as attributes because ``scipy.linalg``'s type stubs
do not declare them -- they are runtime members behind a lazy ``__getattr__``, present since
SciPy 0.16 -- and a checker following the stubs rejects every call site. Resolving them here
puts the workaround in one place, next to the reason for it, instead of a suppression
comment on each of the three functions that use them.
"""

UPDATE_BUDGET: Final = 3
"""How many of §13.2's updates are worth doing before refactorizing instead.

Each update is an ``O(n^2)`` sequence of Givens rotations and a refactorization is
``O(n * m^2)``, so the break-even is around ``n / m^2`` updates -- a number the solver does
not know in advance and which is not worth measuring per call. Three is chosen instead
because it covers what the loop actually does in one iteration: the tangent row moves, one
bound is added, and one is dropped. A working set that changed more than that has been
restructured rather than updated, and refactorizing is both cheaper and less accumulative.
"""


@dataclass(frozen=True, eq=False)
class Factorization:
    """A full QR of ``W.T``, and the ``W`` it was taken of.

    ``W`` is carried alongside because every consumer needs it -- the reduced Hessian needs
    ``Q``, the multiplier recovery needs ``R``, and the update rules need to know which
    column a given working-set row occupies. Recomputing it from the working set at each use
    would be cheap but would also be a second source of truth about row order, which is
    exactly the sort of duplication that lets an update silently apply to the wrong column.

    Attributes:
        Q: the full ``(n, n)`` orthogonal factor, not the economic one -- the null-space
            basis is its *trailing* columns, which an economic QR does not have.
        R: the ``(n, m)`` upper trapezoidal factor.
        W: the working-set matrix this factors, ``(m, n)``.
    """

    Q: Matrix
    R: Matrix
    W: Matrix

    @property
    def rows(self) -> int:
        """How many rows the working set has, ``m``."""
        return self.W.shape[0]

    @property
    def variables(self) -> int:
        """How many variables the problem has, ``n``."""
        return self.W.shape[1]

    def null_space(self) -> Matrix:
        """An orthonormal basis of ``{p : W @ p = 0}``, ``(n, n - m)``.

        The trailing columns of ``Q``, which is the whole reason the factorization is kept
        in full form. No new arithmetic: this is a view of what is already there.

        Returns:
            The basis, whose columns are orthonormal by construction.
        """
        return self.Q[:, self.rows :]

    def solve(
        self,
        gradient: Vector,
        hessian: Matrix | None = None,
        *,
        rho: float = RHO,
        layout: RowLayout | None = None,
    ) -> Direction:
        """Solve the direction subproblem from this factorization.

        The null-space route written out in the module docstring. Two things are worth
        noting about the arithmetic rather than left implicit:

        * when ``hessian`` is ``None`` the reduced system is ``rho`` times the identity, so
          the direction is a projection and no second factorization happens at all. That is
          the common case on a polyhedral working set and it is where most of the saving is.
        * the multiplier solve is triangular and *not* least squares. A rank-deficient ``W``
          makes ``R1`` singular, and this reports that as :class:`SingularKktError` rather
          than returning the least-squares answer, because at this point in the loop a
          dependent working set is a thing to repair -- #25's business -- and not a thing to
          approximate around.

        Args:
            gradient: ``g``, the objective's gradient, ``(n,)``.
            hessian: the full ``H``, or ``None`` for ``rho*I``. Passing ``rho*I``
                explicitly is allowed and gives the same answer more slowly.
            rho: the ``rho`` of ``H = rho*I``, used only when ``hessian`` is ``None``.
            layout: the row order the multipliers are to be read against, or ``None`` for
                an empty one. Carried rather than derived: this module knows ``W`` as a
                matrix and nothing about which row means what, which is the separation that
                lets the update rules work on columns without understanding them.

        Returns:
            The direction and its multipliers, in the same convention
            :func:`cosa.linear_algebra.kkt.solve` uses.

        Raises:
            SingularKktError: if the working-set rows are linearly dependent.
        """
        order = layout if layout is not None else RowLayout(inequalities=(), equalities=(), cones=())
        basis = self.null_space()
        reduced_gradient = basis.T @ gradient
        if hessian is None:
            free = -reduced_gradient / rho
        else:
            reduced = basis.T @ hessian @ basis
            try:
                free = scipy.linalg.solve(reduced, -reduced_gradient, assume_a="sym")
            except (np.linalg.LinAlgError, ValueError) as error:  # pragma: no cover - a PSD H is not singular
                raise SingularKktError(self.rows, self.variables) from error
        d = basis @ free
        if not self.rows:
            return Direction(d=d, multipliers=np.zeros(0), layout=order, rho=rho)

        upper = self.R[: self.rows, :]
        if np.linalg.matrix_rank(upper) < self.rows:
            raise SingularKktError(self.rows, self.variables)
        residual = gradient + (d * rho if hessian is None else hessian @ d)
        nu = scipy.linalg.solve_triangular(upper, -self.Q[:, : self.rows].T @ residual)
        return Direction(d=d, multipliers=nu, layout=order, rho=rho)


def factorize(matrix: Matrix) -> Factorization:
    """Take a fresh QR of ``W.T``. The operation the other three exist to avoid.

    Args:
        matrix: the working-set matrix ``W``, ``(m, n)``.

    Returns:
        The factorization.

    Raises:
        SingularKktError: if ``W`` has more rows than columns. More constraints than
            variables makes the rows dependent by counting alone, and this is the same
            answer :func:`cosa.linear_algebra.kkt.solve` gives -- a degenerate working set
            is #25's to repair, and both routes must hand it the same signal or the loop's
            repair path becomes reachable from only one of them.
    """
    rows, variables = matrix.shape
    if rows > variables:
        raise SingularKktError(rows, variables)
    if not rows:
        return Factorization(Q=np.eye(variables), R=np.zeros((variables, 0)), W=matrix)
    Q, R = scipy.linalg.qr(matrix.T)  # noqa: N806 - the factors are named as linear algebra names them
    return Factorization(Q=Q, R=R, W=matrix)


def insert(factorization: Factorization, row: Vector, at: int) -> Factorization:
    """One constraint added: append its row to ``W`` as a column of ``W.T``.

    §13.2's first case. ``O(n^2)`` Givens rotations rather than a refactorization.

    Args:
        factorization: the current factorization.
        row: the new working-set row, ``(n,)``.
        at: the position it takes in ``W``, which must be where the row layout puts it --
            the working set is ordered, and inserting at the wrong index would factor a
            matrix that is not the one the multipliers get read against.

    Returns:
        The updated factorization.

    Raises:
        ProblemError: if the position is out of range or the row has the wrong length.
    """
    if not 0 <= at <= factorization.rows:
        raise ProblemError("at", f"expected a position in [0, {factorization.rows}], found {at}")
    column = np.asarray(row, dtype=np.float64).reshape(-1)
    if column.size != factorization.variables:
        raise ProblemError("row", f"expected {factorization.variables} entries, found {column.size}")
    Q, R = _qr_insert(factorization.Q, factorization.R, column, at, which="col")  # noqa: N806
    W = np.insert(factorization.W, at, column, axis=0)  # noqa: N806
    return Factorization(Q=Q, R=R, W=W)


def delete(factorization: Factorization, at: int) -> Factorization:
    """One constraint removed: delete its column from ``W.T``.

    §13.2's second case, and the cheaper of the two -- a deletion only has to re-triangularize
    the columns to the right of the one that left.

    Args:
        factorization: the current factorization.
        at: the position of the row leaving ``W``.

    Returns:
        The updated factorization.

    Raises:
        ProblemError: if the position is out of range.
    """
    if not 0 <= at < factorization.rows:
        raise ProblemError("at", f"expected a position in [0, {factorization.rows}), found {at}")
    if factorization.rows == 1:
        return factorize(np.zeros((0, factorization.variables)))
    Q, R = _qr_delete(factorization.Q, factorization.R, at, which="col")  # noqa: N806
    W = np.delete(factorization.W, at, axis=0)  # noqa: N806
    return Factorization(Q=Q, R=R, W=W)


def replace(factorization: Factorization, row: Vector, at: int) -> Factorization:
    """The tangent row moved: replace one column of ``W.T`` in place.

    §13.2's third case, the one the paper singles out as "more subtle than ordinary linear
    constraint updates" because the tangent changes continuously with ``x`` rather than
    discretely. The subtlety is real but it is about *frequency*, not about kind: the row
    changes every iteration the cone is active, so this update runs far more often than the
    other two -- and it is still an update. A delete followed by an insert leaves every
    other column's contribution to ``Q`` untouched, which on a working set of one tangent
    row and several bounds is most of the work saved.

    Args:
        factorization: the current factorization.
        row: the row's new value, ``(n,)``.
        at: its position in ``W``.

    Returns:
        The updated factorization.

    Raises:
        ProblemError: if the position is out of range or the row has the wrong length.
    """
    if not 0 <= at < factorization.rows:
        raise ProblemError("at", f"expected a position in [0, {factorization.rows}), found {at}")
    return insert(delete(factorization, at), row, at)


@dataclass(eq=False)
class Reuse:
    """One factorization, carried between iterations and updated to fit.

    The only mutable object in the solver, and it is mutable because it is a cache: its
    state is derived from the working set it was last asked about, so nothing downstream can
    read a wrong answer out of it -- at worst it re-factorizes when it could have updated.

    Attributes:
        held: the factorization currently cached, or ``None`` before the first solve.
        factorizations: how many full factorizations have been taken.
        updates: how many were avoided by an update instead.
    """

    held: Factorization | None = None
    factorizations: int = 0
    updates: int = 0

    def matrix_for(
        self,
        problem: SOCP,
        working_set: WorkingSet,
        z: Vector,
        *,
        tolerance: float = TOLERANCE,
    ) -> Factorization:
        """Return a factorization of this working set's ``W``, updating if it can.

        The three cases of §13.2 are recognized by *comparing matrices*, not by being told
        which happened. That is what makes the cache safe against the loop's six ways of
        changing a working set: a path that forgets to announce itself still gets a correct
        factorization, and the worst that happens is a refactorization that was avoidable.

        A row differing only in value, at the same position, with the same number of rows, is
        the tangent case; one extra or one missing row with the rest matching is an insert or
        a delete; anything else is a fresh factorization.

        Args:
            problem: the instance.
            working_set: what is currently believed active.
            z: the current point, needed for the tangent rows.
            tolerance: the vanishing-tail tolerance passed to the tangent rows.

        Returns:
            The factorization, cached for the next call.
        """
        wanted = working_set_matrix(problem, working_set, z, tolerance=tolerance)
        held = self.held
        updated = None if held is None else _update_to(held, wanted)
        if updated is None:
            self.held = factorize(wanted)
            self.factorizations += 1
        else:
            self.held = updated
            self.updates += 1
        return self.held

    def direction(
        self,
        problem: SOCP,
        working_set: WorkingSet,
        z: Vector,
        *,
        rho: float = RHO,
        tolerance: float = TOLERANCE,
        curvature: Matrix | None = None,
    ) -> Direction:
        """Solve the direction subproblem, reusing the factorization where possible.

        Signature-compatible with :func:`cosa.linear_algebra.kkt.direction` so that the
        loop can hold one or the other, and the comparison between them is a comparison of
        cost rather than of answers.

        Args:
            problem: the instance.
            working_set: what is currently believed active.
            z: the current point.
            rho: the ``rho`` of ``H = rho*I``.
            tolerance: the vanishing-tail tolerance passed to the tangent rows.
            curvature: #23's Lagrangian curvature, or ``None`` for ``H = rho*I``.

        Returns:
            The direction and its multipliers.

        Raises:
            SingularKktError: if the working-set rows are linearly dependent.
        """
        factorization = self.matrix_for(problem, working_set, z, tolerance=tolerance)
        hessian = None
        if curvature is not None and curvature.any():
            hessian = rho * np.eye(problem.num_variables) + curvature
        return factorization.solve(problem.c, hessian, rho=rho, layout=RowLayout.for_working_set(working_set))

    def __str__(self) -> str:
        """The two counters and the saving between them, for a log line."""
        total = self.factorizations + self.updates
        share = self.updates / total if total else 0.0
        return f"reuse: {self.factorizations} factorization(s), {self.updates} update(s) ({share:.0%} reused)"


def _update_to(held: Factorization, wanted: Matrix, *, budget: int = UPDATE_BUDGET) -> Factorization | None:
    """A short sequence of §13.2's updates taking ``held`` to ``wanted``, if one is short.

    A single update was the first design and it left most of the saving on the table. The
    loop routinely changes two things in one iteration -- a blocking row is added *and* the
    tangent moves, because the step that reached the row also moved ``x`` -- and a rule that
    only recognized one change refactorized on every such iteration. On a box-constrained
    portfolio at ``n = 150`` that was nine iterations in ten.

    So the difference is diffed rather than classified: rows common to both matrices are
    matched in order, the ones only in ``held`` are deleted, the ones only in ``wanted`` are
    inserted, and a row that changed value in place is replaced. Under
    :data:`UPDATE_BUDGET` operations that is still cheaper than a refactorization; over it,
    it is not, and this says so by declining.

    Args:
        held: the cached factorization.
        wanted: the working-set matrix now needed.
        budget: how many updates are worth doing before refactorizing instead.

    Returns:
        The updated factorization, or ``None`` when no short sequence reaches it.
    """
    current = held.W
    if current.shape[1] != wanted.shape[1]:
        return None
    plan = _plan(current, wanted, budget=budget)
    if plan is None:
        return None
    updated = held
    for operation, at, row in plan:
        if operation == "delete":
            updated = delete(updated, at)
        elif operation == "insert":
            updated = insert(updated, row, at)
        else:
            updated = replace(updated, row, at)
    return updated


def _plan(current: Matrix, wanted: Matrix, *, budget: int) -> list[tuple[str, int, Vector]] | None:
    """The edit script from ``current`` to ``wanted``, or ``None`` if it is longer than the budget.

    A real diff rather than a walk in step. The first version compared the two matrices row
    by row and gave up at the first mismatch it could not classify, which handled exactly one
    change per iteration -- and the loop's most ordinary iteration makes two, since the step
    that reached a blocking row also moved ``x`` and so moved the tangent. Worse, the cone's
    rows come *last* in the layout, so an inserted bound shifted every later row and the walk
    read the shift as a cascade of deletions. On a box-constrained portfolio at ``n = 150``
    that meant nine iterations in ten refactorized.

    :class:`difflib.SequenceMatcher` over the rows' bytes finds the real script instead. The
    overlapping part of a replacement opcode becomes :func:`replace` -- one operation, not a
    delete plus an insert -- which is what keeps a moving tangent row at unit cost.

    Args:
        current: the matrix held.
        wanted: the matrix needed.
        budget: the longest script worth executing.

    Returns:
        The script as ``(operation, position, row)`` triples against the *evolving* matrix,
        or ``None`` when it is longer than the budget.
    """
    matcher = difflib.SequenceMatcher(
        a=[row.tobytes() for row in current], b=[row.tobytes() for row in wanted], autojunk=False
    )
    script: list[tuple[str, int, Vector]] = []
    offset = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        common = min(i2 - i1, j2 - j1)
        for k in range(common):
            script.append(("replace", i1 + offset + k, wanted[j1 + k]))
        for _ in range(i2 - i1 - common):
            script.append(("delete", i1 + offset + common, current[i1]))
        offset -= i2 - i1 - common
        for k in range(j2 - j1 - common):
            script.append(("insert", i1 + offset + common + k, wanted[j1 + common + k]))
        offset += j2 - j1 - common
        if len(script) > budget:
            return None
    return script
