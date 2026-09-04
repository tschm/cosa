"""Getting a feasible point to start from, without borrowing one from an oracle.

§9 Phase I (``paper.tex:679``) lists "feasible initialization" first among the nine
components, and it is the one that cannot be skipped: an active-set method's whole
invariant is that every iterate is feasible (§14.1), so there is nothing to iterate from
until a feasible point exists.

**Three routes, in increasing cost, and the order matters.**

1. **A point the caller already has.** The instance generators all ship a
   :attr:`~cosa.experiments.portfolio.PortfolioInstance.witness`, chosen before their
   constraints were, and #30's warm start will supply the previous solve's answer. Checking
   a given point costs one matrix-vector product; earning one costs a solve.
2. **The least-norm solution of the equalities, with the cone's heads raised.** ``E z = d``
   is a linear system, so a particular solution is a least-squares call. If it happens to
   satisfy the inequalities, that is the whole job. The conic blocks then come free *for
   eq. (7)*, and only for it -- see below.
3. **An elastic Phase I.** Relax every inequality by one scalar ``s``, minimize ``s``, and
   start from a point that is feasible for the relaxation by construction. If the optimum
   has ``s <= 0`` the original is feasible and the ``z`` part is a start; if not, the
   original is infeasible and that is the answer, not a failure.

Route 3 is what makes this Phase I rather than a lookup: the relaxed problem is an
instance of the same class, so the *solver solves its own initialization*, which is why
:func:`elastic_problem` returns a problem rather than a point. The loop of
:mod:`cosa.solver.cosa` is what runs it, and the recursion terminates at depth one because
the elastic problem's start is known.

**The cone's head trick, and its exact limits.** In eq. (7) the risk variable ``t`` appears
in the cone's head row and nowhere else, so any ``x`` can be made conically feasible by
raising ``t`` -- which is what :meth:`cosa.MeanStdPortfolio.socp_point` does at the optimum
and what :func:`raise_free_heads` does here for an arbitrary point. That covers the whole
mean-standard-deviation family. It does *not* cover a general SOCP whose head variable is
constrained elsewhere, and :func:`feasible_start` says so rather than guessing: conic
initialization in that generality needs a conic Phase I, which needs the step interval of
#18, and by then #20 owns the question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from cosa.problem.socp import SOCP, ConeProduct, ProblemError, _vector
from cosa.solver.instrumentation import level_1_violations

if TYPE_CHECKING:
    from cosa import Vector

__all__ = [
    "TOLERANCE",
    "ElasticProblem",
    "NeedsPhaseOneError",
    "elastic_problem",
    "equality_particular_solution",
    "feasible_start",
    "raise_free_heads",
]

TOLERANCE: Final = 1e-9
"""How large a constraint violation may be for a point to count as a feasible start.

Tighter than §14.1's Level 1 tolerance on purpose: a *starting* point should be feasible
with room to spare, because the iterates that follow will each spend a little of that room
on rounding. Starting at the edge of the tolerance means the first accepted iterate may
fall outside it.
"""


class NeedsPhaseOneError(ProblemError):
    """No feasible start could be constructed cheaply, and the cheap routes are exhausted.

    Not "the problem is infeasible" -- that is a different and stronger claim, and
    :func:`elastic_problem` is what establishes it. This says only that routes 1 and 2 of
    the module docstring failed, so the caller must run route 3 or supply a point.

    A :class:`cosa.ProblemError`, because from a caller's point of view the input is
    incomplete: this instance needs a start and did not come with one.
    """

    def __init__(self, reason: str) -> None:
        """Say which cheap route failed and what the caller can do instead.

        Args:
            reason: what went wrong, phrased so the actionable part comes last.
        """
        super().__init__("start", reason)


def equality_particular_solution(problem: SOCP, *, tolerance: float = TOLERANCE) -> Vector:
    """The least-norm solution of ``E @ z = d``, which is route 2's starting guess.

    Least-norm rather than any solution, because it is the one that does not wander: a
    particular solution with a large component in the null space of ``E`` is just as valid
    and far more likely to violate the inequalities.

    Args:
        problem: the instance.
        tolerance: how well the system must be satisfied to count as solved.

    Returns:
        The least-norm ``z`` with ``E @ z = d``, or the zero vector when there are no
        equalities.

    Raises:
        NeedsPhaseOneError: if ``E @ z = d`` has no solution, which makes the instance
            infeasible outright -- the one infeasibility this module can prove on its own.
    """
    if not problem.num_equalities:
        return np.zeros(problem.num_variables)
    point = np.linalg.lstsq(problem.E, problem.d, rcond=None)[0]
    error = float(np.abs(problem.E @ point - problem.d).max(initial=0.0))
    if error > tolerance * max(1.0, float(np.abs(problem.d).max(initial=0.0))):
        reason = (
            f"E @ z = d has no solution -- the least-squares residual is {error:.3g} -- so the "
            "instance is infeasible before any inequality is considered"
        )
        raise NeedsPhaseOneError(reason)
    return np.ascontiguousarray(point)


def _free_head_variables(problem: SOCP) -> tuple[int, ...] | None:
    """The variable each cone factor's head selects, if every head is free to move.

    A head is *free* when its row of ``G`` selects a single variable with coefficient one
    and that variable appears in no other row of ``A``, ``E`` or ``G``. That is exactly
    eq. (7)'s ``t``, and it is the condition under which conic feasibility can be repaired
    without touching anything else.

    Args:
        problem: the instance.

    Returns:
        One variable index per cone factor, or ``None`` if any head is not free.
    """
    heads: list[int] = []
    for block in problem.cone.slices:
        row = problem.G[block][0]
        selected = np.flatnonzero(row)
        # Any single nonzero coefficient will do, not only one. The head is
        # `coefficient * point[variable] + h_head` and solving `head >= ||tail||` for the
        # variable is one division either way. Requiring exactly 1.0 was the original test
        # and it made the routine refuse every *equilibrated* instance, since §13.3 rescales
        # the head row by a positive factor -- which is how #37's public interface, whose
        # default is to equilibrate, found this.
        if selected.size != 1 or row[selected[0]] == 0.0:
            return None
        variable = int(selected[0])
        elsewhere = (
            np.abs(problem.A[:, variable]).sum()
            + np.abs(problem.E[:, variable]).sum()
            + np.abs(np.delete(problem.G[:, variable], block.start)).sum()
        )
        if elsewhere > 0.0 or variable in heads:
            return None
        heads.append(variable)
    return tuple(heads)


def raise_free_heads(problem: SOCP, z: Vector, *, margin: float = 0.0) -> Vector:
    """Make every conic block feasible by raising its head, where the head is free to rise.

    For eq. (7) this sets ``t = ||L @ x||`` -- or a little above it, with a margin -- which
    is conically feasible for any ``x`` and changes no other constraint, because ``t``
    appears nowhere else.

    Args:
        problem: the instance.
        z: a point, presumed feasible for the linear blocks.
        margin: how far *inside* the cone to place each block. A positive margin leaves the
            cone strictly inactive, which is what the polyhedral baseline wants: it keeps
            the conic interval of #18 out of the step calculation entirely.

    Returns:
        A point agreeing with ``z`` except in the cone's head variables.

    Raises:
        ProblemError: if any head is not free to move, in which case conic feasibility
            cannot be repaired without a conic Phase I.
    """
    point = _vector("z", z, size=problem.num_variables).copy()
    if not len(problem.cone):
        return point
    heads = _free_head_variables(problem)
    if heads is None:
        raise ProblemError(
            "cone",
            "at least one cone head is not a free variable, so conic feasibility cannot be "
            "repaired by raising it. Supply a feasible start instead",
        )
    for block, variable in zip(problem.cone.slices, heads, strict=True):
        # The head row selects `variable` alone and `variable` appears in no other row, so
        # the tail does not depend on it and the head is
        # `coefficient * point[variable] + h_head`. Solving `head >= ||tail|| + margin` is
        # one division.
        coefficient = float(problem.G[block][0][variable])
        tail = (problem.G[block] @ point + problem.h[block])[1:]
        wanted = float(np.linalg.norm(tail)) + margin - float(problem.h[block][0])
        point[variable] = wanted / coefficient
    return point


def feasible_start(
    problem: SOCP,
    start: Vector | None = None,
    *,
    tolerance: float = TOLERANCE,
    margin: float = 0.0,
) -> Vector:
    """Routes 1 and 2 of the module docstring: a given point, or one built cheaply.

    Args:
        problem: the instance.
        start: a point the caller believes feasible. Checked, not trusted.
        tolerance: how large a violation may be for a point to count as feasible.
        margin: how far inside the cone to place a repaired conic block.

    Returns:
        A feasible point.

    Raises:
        NeedsPhaseOneError: if neither route produces one. The caller's next move is
            :func:`elastic_problem`.
    """
    if start is not None:
        point = _vector("start", start, size=problem.num_variables)
        violations = level_1_violations(problem, point, tolerance=tolerance)
        if not violations:
            return np.ascontiguousarray(point)
        reason = (
            f"the point supplied is not feasible: {'; '.join(violations)}. Omit it to have one "
            "constructed, or correct it"
        )
        raise NeedsPhaseOneError(reason)

    point = equality_particular_solution(problem, tolerance=tolerance)
    if len(problem.cone):
        try:
            point = raise_free_heads(problem, point, margin=margin)
        except ProblemError as unrepairable:
            reason = f"no start was supplied and none could be built: {unrepairable}"
            raise NeedsPhaseOneError(reason) from unrepairable

    violations = level_1_violations(problem, point, tolerance=tolerance)
    if violations:
        reason = (
            f"the least-norm equality solution violates {'; '.join(violations)}, so an elastic "
            "Phase I is needed -- see elastic_problem"
        )
        raise NeedsPhaseOneError(reason)
    return point


@dataclass(frozen=True, eq=False)
class ElasticProblem:
    """The relaxed instance of route 3, together with everything needed to use it.

    Attributes:
        problem: the relaxed instance, over the variables ``(z, s)``. Its objective is
            ``s`` alone, so minimizing it minimizes the largest relaxation needed.
        start: a point feasible for :attr:`problem` by construction, so the solve that uses
            it needs no Phase I of its own and the recursion stops here.
        elastic: the index of the relaxation variable ``s``.
    """

    problem: SOCP
    start: Vector
    elastic: int

    def original_point(self, z: Vector) -> Vector:
        """Drop the relaxation variable, recovering a point of the original instance.

        Args:
            z: a point of :attr:`problem`.

        Returns:
            Its first ``n`` entries.
        """
        return _vector("z", z, size=self.elastic + 1)[: self.elastic]

    def relaxation(self, z: Vector) -> float:
        """How much relaxation a point of :attr:`problem` uses.

        Args:
            z: a point of :attr:`problem`.

        Returns:
            Its ``s`` entry. Non-positive means the original instance is feasible at
            :meth:`original_point`.
        """
        return float(_vector("z", z, size=self.elastic + 1)[self.elastic])


def elastic_problem(problem: SOCP, *, tolerance: float = TOLERANCE) -> ElasticProblem:
    """Build route 3: relax every inequality by one scalar and minimize it.

        min  s   s.t.  A @ z - s <= b,  E @ z = d,  -s <= 0

    Three decisions, each with a reason:

    **One scalar, not one per row.** A per-row relaxation minimizing ``sum(s_i)`` is the
    more common elastic formulation and finds a point that violates *few* constraints;
    minimizing a single ``s`` finds one that violates *none* by much, which is what a
    starting point wants. It also keeps the relaxed problem the same size plus one, so its
    solve costs what the original's does.

    **``s >= 0`` is required, and the relaxed problem is unbounded without it.** With only
    ``A @ z - s <= b``, a ``z`` running off along an unbounded edge takes ``s`` to minus
    infinity with it -- the row is satisfied by ever more slack. Bounding ``s`` below at
    zero makes the optimum ``max(0, the least uniform relaxation)``, which is the number
    the question is actually about.

    **The cone is dropped rather than relaxed.** Relaxing it would make ``s`` serve two
    purposes and would leave the cone *exactly* active at ``s = 0``, which is precisely the
    state the polyhedral step of #14 cannot handle. Dropping it and then raising the free
    heads afterwards -- which :func:`feasible_start` does, with a margin -- lands strictly
    inside the cone instead. An instance whose heads are not free never reaches here;
    :func:`feasible_start` refuses it first.

    Args:
        problem: the instance to relax.
        tolerance: how well the equalities must be solved.

    Returns:
        The relaxed instance, a feasible start for it, and the relaxation variable's index.

    Raises:
        NeedsPhaseOneError: if the equalities alone have no solution.
    """
    base = equality_particular_solution(problem, tolerance=tolerance)
    elastic = problem.num_variables
    rows = problem.num_inequalities

    relaxed = SOCP(
        c=np.concatenate([np.zeros(elastic), [1.0]]),
        A=np.vstack(
            [
                np.hstack([problem.A, -np.ones((rows, 1))]),
                np.concatenate([np.zeros(elastic), [-1.0]]).reshape(1, -1),
            ]
        ),
        b=np.concatenate([problem.b, [0.0]]),
        E=np.hstack([problem.E, np.zeros((problem.num_equalities, 1))]),
        d=problem.d,
        G=np.zeros((0, elastic + 1)),
        h=np.zeros(0),
        cone=ConeProduct(),
    )
    violation = float((problem.A @ base - problem.b).max(initial=0.0))
    return ElasticProblem(
        problem=relaxed,
        start=np.concatenate([base, [max(0.0, violation) + 1.0]]),
        elastic=elastic,
    )
