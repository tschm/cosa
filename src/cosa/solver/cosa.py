"""The iteration: working set, direction, step, multipliers, update -- until it stops.

§4.1 (``paper.tex:321``) writes the fundamental iteration as

    working set -> direction -> feasible step -> multiplier test -> working-set update

and §4.1's numbered list (``paper.tex:336``) spells it out in ten steps. This module is
that loop. §9 Phase I (``paper.tex:679``) is what it is being held to first: a conventional
active-set solver whose nine components -- feasible initialization, active-set
representation, direction computation, KKT solve, multiplier calculation, constraint
addition, constraint deletion, ratio test, termination checks -- are each somebody else's
module, and whose job here is to be *wired together correctly*.

Its purpose is stated by the issue that asks for it: "deliberately simple -- its job is to
be the thing you debug COSA against". So there is nothing clever below, and where a choice
existed the duller one was taken.

**Two exits, and the difference between them is the whole termination question.** The loop
stops when the projected gradient vanishes *and* every multiplier has the right sign. Those
are different conditions and they fail differently:

* a nonzero direction means the working set still permits improvement, so take the step;
* a zero direction with a wrong-signed multiplier means the working set is *too large* --
  the point is optimal for the constraints being held, but one of them should not be held.
  §7.2's rule names which, and the loop drops it and continues. This is the only place
  iterations are spent without moving.

Only when both hold is the point optimal, and :mod:`cosa.solver.termination`'s five
residuals are what certify it. The loop asks them rather than deciding for itself, because
§6 says they *are* the criterion and Success Criterion 2 asks that they be.

**Why the linear objective makes the step trivial.** With ``c.T @ z`` linear and ``d`` a
descent direction, the objective falls monotonically along the ray -- there is no interior
minimum to find, so the step is always the largest feasible one and no line search is
needed. That is a simplification the mean-standard-deviation formulation *earns*: it is
linear precisely because the risk term went into the cone. A quadratic objective would need
a step that stops at the unconstrained minimizer when that comes first.

**A degenerate working set is repaired, not surrendered to.** When the direction solve
refuses a dependent set, the loop applies §8.3 in the order that order matters: first
*dependent-constraint removal*, which fixes the working set and costs nothing thereafter;
and only if that cannot help -- because the redundancy is among rows the solver may not
drop -- *regularization*, which always works and answers a nearby question. #25 provides
both; this is where the choice between them is made.

The order is not a preference, it is a correctness ranking. Removal changes the working set
and leaves the arithmetic exact; regularization leaves the working set alone and makes the
arithmetic approximate, and the approximation is visible: at a stationary point the true
direction is exactly zero while the regularized one is ``O(delta)``, so on an instance with
nothing to block a step a large enough ``delta`` turns that residue into an apparent
improving direction and the loop concludes ``"unbounded"``. That is why
:data:`REGULARIZATION` is small and why removal is tried first.

**Scope: the cone is not handled here.** §9 Phase I is the polyhedral baseline, and
:func:`cosa.geometry.step.step_limit` refuses a problem whose cone could bind rather than
stepping through it. So this loop solves linear programs, including the elastic Phase I
problems it builds for itself, and instances whose cone stays strictly inactive. #20 adds
the conic activation and the exact step of #18, and the loop's shape does not change --
which is the point of putting the missing piece behind one function call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from cosa.active_set import updates
from cosa.active_set.multipliers import Multipliers, from_direction
from cosa.active_set.working_set import WorkingSet
from cosa.geometry.step import step_limit
from cosa.linear_algebra.kkt import RHO, SingularKktError
from cosa.problem.socp import ProblemError, _vector
from cosa.solver.initialization import NeedsPhaseOneError, elastic_problem, feasible_start, raise_free_heads
from cosa.solver.instrumentation import UNCHECKED, InvariantChecker, Metrics, Recorder
from cosa.solver.termination import Residuals, residuals

if TYPE_CHECKING:
    from cosa import Vector
    from cosa.problem.socp import SOCP

__all__ = [
    "MAX_ITERATIONS",
    "Solution",
    "solve",
]

REGULARIZATION: Final = 1e-10
"""§8.3's ``delta``, used only when dependent-constraint removal cannot repair a set.

Small, because it buys an answer to a *nearby* problem and the nearness is what is being
paid for: at ``1e-10`` the perturbation is far below any tolerance the solver tests
against, and far above the ``1e-18`` pivots that make a dependent system look solvable to
LAPACK. It is deliberately not ``rho`` and deliberately not tuned -- if an instance needs
it large enough to matter, the right response is to find out why the working set is
degenerate, which is what #36's failure-mode study is for.
"""

MAX_ITERATIONS: Final = 200
"""How many iterations before the loop gives up and says so.

An active-set method's iteration count is bounded by the number of working sets, which is
finite but astronomically large, so a limit is not a safety net -- it is the admission that
without the anti-cycling rules of #29 the loop can revisit a working set forever. Two
hundred is far above what any instance here needs and far below a wait anyone would sit
through.
"""

_STATIONARY: Final = 1e-10
"""How small a direction must be, relative to the iterate, to count as vanished.

Relative because the direction scales as ``1 / rho``: an absolute threshold would mean a
different thing at every ``rho``, which is exactly the confusion #12's docstring warns
against.
"""


@dataclass(frozen=True, eq=False)
class Solution:
    """What one solve produced, and enough about it to tell whether to believe it.

    Attributes:
        z: the final iterate.
        multipliers: the multipliers there.
        working_set: the working set held at termination -- Success Criterion 3's
            "interpretable in terms of the active portfolio constraints", available for
            :meth:`cosa.WorkingSet.describe`.
        residuals: §6's five, which are what :attr:`status` is decided by.
        status: ``"optimal"``, ``"iteration_limit"``, ``"unbounded"``, ``"infeasible"`` or
            ``"degenerate"``. A status other than ``"optimal"`` is a description of what
            stopped the loop, not an excuse: the other four each name a specific thing that
            happened.
        metrics: every quantity §11 and §12.3 ask for.
    """

    z: Vector
    multipliers: Multipliers
    working_set: WorkingSet
    residuals: Residuals
    status: str
    metrics: Metrics

    @property
    def is_optimal(self) -> bool:
        """Did the loop terminate on §14.3's Level 3 certificate?"""
        return self.status == "optimal"

    def objective(self, problem: SOCP) -> float:
        """The objective value at the final iterate.

        Args:
            problem: the instance solved.

        Returns:
            ``c.T @ z``.
        """
        return float(problem.c @ self.z)

    def __str__(self) -> str:
        """The status, the residuals and the iteration count, on one line."""
        return f"{self.status}: {self.residuals} [{self.metrics}]"


def solve(
    problem: SOCP,
    *,
    start: Vector | None = None,
    rho: float = RHO,
    max_iterations: int = MAX_ITERATIONS,
    tolerance: float | None = None,
    checker: InvariantChecker = UNCHECKED,
    recorder: Recorder | None = None,
    phase_one: bool = True,
    regularization: float = REGULARIZATION,
) -> Solution:
    """Run the §4.1 iteration until it terminates.

    Args:
        problem: the instance to solve.
        start: a feasible point to begin from, or ``None`` to construct one. A supplied
            point is checked, not trusted, and an infeasible one is refused rather than
            replaced -- see the note at the call site.
        rho: the ``rho`` of the direction subproblem's ``H = rho*I``. Affects the length of
            each direction and nothing else -- see #12.
        max_iterations: the iteration limit.
        tolerance: the residual tolerance for termination, or ``None`` for
            :data:`cosa.solver.termination.TOLERANCE`.
        checker: §14's invariant checker. Off by default so a benchmark pays nothing;
            the test suite passes :data:`cosa.solver.instrumentation.CHECKED`.
        recorder: a recorder to accumulate metrics into, or ``None`` for a fresh one.
        regularization: §8.3's ``delta``, used only as the fallback when dependent-constraint
            removal cannot repair a degenerate working set. Zero disables the fallback, so a
            dependency the working set may not resolve stops the loop with a
            ``"degenerate"`` status instead of being perturbed away.
        phase_one: whether to build and solve an elastic Phase I when no feasible start can
            be constructed cheaply. Set ``False`` by the recursive call, which is what
            stops the recursion at depth one.

    Returns:
        The solution, whatever its status.

    Raises:
        ProblemError: if the iteration limit is not positive, or the supplied start is not
            feasible.
    """
    if max_iterations < 1:
        raise ProblemError("max_iterations", f"expected at least one iteration, found {max_iterations}")
    stopping = residuals_tolerance(tolerance)
    recorder = recorder or Recorder()

    try:
        point = feasible_start(problem, start)
    except NeedsPhaseOneError:
        # Only the *constructed* routes fall through to Phase I. A start the caller supplied
        # and got wrong is a caller error and propagates: silently discarding it and solving
        # from somewhere else would hide the mistake and, for a warm start, would hide that
        # the warm start was not being used.
        if start is not None or not phase_one:
            raise
        point = _phase_one(problem, rho=rho, max_iterations=max_iterations, recorder=recorder)

    checker.accepted_iterate(problem, point)
    working_set = _working_set_at(problem, point)

    with recorder.solving():
        for _ in range(max_iterations):
            recorder.iteration()
            try:
                direction = recorder.solve_direction(problem, working_set, point, rho=rho)
            except SingularKktError:
                working_set, dropped = updates.drop_dependent_rows(problem, working_set, point)
                if dropped:
                    for _row in dropped:
                        recorder.constraint_removed()
                    continue
                if not regularization:
                    return _finish(problem, point, working_set, "degenerate", recorder, stopping)
                direction = recorder.solve_direction(
                    problem, working_set, point, rho=rho, regularization=regularization
                )

            found = from_direction(problem, working_set, point, direction)
            measured = residuals(problem, point, found)
            recorder.kkt_residual(measured.largest)

            if float(np.abs(direction.d).max(initial=0.0)) <= _STATIONARY * max(
                1.0, float(np.abs(point).max(initial=0.0))
            ):
                checker.computed_multipliers(problem, found)
                dropping = updates.removal_candidate(working_set, found.y)
                if dropping is None:
                    status = "optimal" if measured.is_optimal(tolerance=stopping) else "degenerate"
                    return Solution(
                        z=point,
                        multipliers=found,
                        working_set=working_set,
                        residuals=measured,
                        status=status,
                        metrics=recorder.metrics(),
                    )
                working_set = updates.drop_inequality(working_set, dropping)
                recorder.constraint_removed()
                continue

            limit = step_limit(problem, point, direction.d, working_set)
            if limit.is_unbounded:
                return _finish(problem, point, working_set, "unbounded", recorder, stopping)

            point = point + limit.alpha * direction.d
            checker.accepted_iterate(problem, point)
            if limit.blocking is not None:
                working_set = updates.add_inequality(working_set, limit.blocking)
                recorder.constraint_added()

    return _finish(problem, point, working_set, "iteration_limit", recorder, stopping)


def residuals_tolerance(tolerance: float | None) -> float:
    """The termination tolerance, defaulting to the residual module's.

    Args:
        tolerance: the caller's choice, or ``None``.

    Returns:
        The tolerance to compare residuals against.
    """
    from cosa.solver.termination import TOLERANCE

    return TOLERANCE if tolerance is None else float(tolerance)


def _working_set_at(problem: SOCP, z: Vector, *, tolerance: float = updates.ACTIVATION_TOLERANCE) -> WorkingSet:
    """The working set the §7.1 rule makes active at a point.

    The loop must start with a working set that *matches* its point: a set holding a row
    the point is not on would impose ``a_i.T @ p = 0`` about a constraint that is not
    tight, and the direction would refuse to move in a direction that is perfectly
    feasible.

    Args:
        problem: the instance.
        z: the starting point.
        tolerance: §7.1's activation tolerance.

    Returns:
        The set with every tight inequality active, and the cone left alone -- activating it
        is #20's.
    """
    working_set = WorkingSet.empty(problem)
    for index in updates.activation_candidates(problem, z, working_set, tolerance=tolerance):
        working_set = updates.add_inequality(working_set, index)
    return working_set


def _phase_one(problem: SOCP, *, rho: float, max_iterations: int, recorder: Recorder) -> Vector:
    """Route 3 of :mod:`cosa.solver.initialization`: solve the elastic relaxation.

    The recursion that makes this Phase I rather than a lookup. It terminates at depth one
    because the elastic problem comes with a feasible start, so its own call to
    :func:`feasible_start` succeeds and ``phase_one=False`` is never exercised.

    Args:
        problem: the instance needing a start.
        rho: passed through to the relaxed solve.
        max_iterations: passed through to the relaxed solve.
        recorder: the recorder, so the Phase I solve's iterations are counted too -- they
            are part of the cost of solving this instance.

    Returns:
        A feasible point of ``problem``.

    Raises:
        NeedsPhaseOneError: if the relaxation's optimum still needs a positive relaxation,
            which proves the instance infeasible.
    """
    elastic = elastic_problem(problem)
    relaxed = solve(
        elastic.problem,
        start=elastic.start,
        rho=rho,
        max_iterations=max_iterations,
        recorder=recorder,
        phase_one=False,
    )
    slack = elastic.relaxation(relaxed.z)
    if slack > feasible_tolerance():
        reason = (
            f"the least uniform relaxation of the inequalities is {slack:.3g} > 0, so the "
            f"instance is infeasible (the elastic Phase I terminated {relaxed.status})"
        )
        raise NeedsPhaseOneError(reason)

    point = elastic.original_point(relaxed.z)
    if len(problem.cone):
        point = raise_free_heads(problem, point, margin=1.0)
    return feasible_start(problem, point)


def feasible_tolerance() -> float:
    """The tolerance a Phase I relaxation must fall below for the instance to be feasible.

    Returns:
        :data:`cosa.solver.initialization.TOLERANCE`.
    """
    from cosa.solver.initialization import TOLERANCE

    return TOLERANCE


def _finish(
    problem: SOCP,
    z: Vector,
    working_set: WorkingSet,
    status: str,
    recorder: Recorder,
    tolerance: float,
) -> Solution:
    """Build a solution for an exit that is not the optimal one.

    The multipliers are recovered anyway, and the residuals with them: a caller told
    "iteration limit" still wants to know *how close* it got, and #36's failure-mode study
    is going to want exactly that number.

    Args:
        problem: the instance.
        z: the final iterate.
        working_set: the final set.
        status: what stopped the loop.
        recorder: the recorder to snapshot.
        tolerance: the residual tolerance, in case the point turns out to be optimal after
            all.

    Returns:
        The solution.
    """
    point = _vector("z", z, size=problem.num_variables)
    try:
        direction = recorder.solve_direction(problem, working_set, point)
        found = from_direction(problem, working_set, point, direction)
    except SingularKktError:
        found = Multipliers(
            y=np.zeros(problem.num_inequalities),
            nu=np.zeros(problem.num_equalities),
            w=np.zeros(problem.cone.dim),
        )
    measured = residuals(problem, point, found)
    if status != "unbounded" and measured.is_optimal(tolerance=tolerance):
        status = "optimal"
    return Solution(
        z=point,
        multipliers=found,
        working_set=working_set,
        residuals=measured,
        status=status,
        metrics=recorder.metrics(),
    )
