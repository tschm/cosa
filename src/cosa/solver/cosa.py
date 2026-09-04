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

**The cone joins the working set the same way a constraint does.** §7.3 says the cone
becomes geometrically active when ``t - ||L @ x||_2`` falls below ``eps_act``, and
:func:`cosa.active_set.updates.activate_cones` applies that after every accepted iterate --
in the same place, and for the same reason, that §7.1's rule adds a linear row. Once a
factor is active the direction subproblem carries its tangent row, so the next direction
stays on the boundary rather than pointing off it, and #18's exact step keeps the one after
that inside the cone. Those four things -- a working set of linear constraints, the tangent
representation, exact conic step lengths, multiplier-based updates -- are §9 Phase III's
list, and wiring them together is all this module does with them.

**Deactivation is the multiplier's decision, and on eq. (7) it provably never fires.**
§7.3 is explicit that geometric activity *"alone is not sufficient to establish
optimality"*, and §7.4 hands the other half to the conic multiplier and the normal cone.
:func:`cosa.active_set.updates.deactivate_cones` is that rule, and the loop consults it
wherever it consults §7.2's -- at a stationary point, after the inequality test, because a
drop is only meaningful once the working set has stopped permitting progress.

On the mean-standard-deviation problem it is inert, and the reason is worth writing down.
``t`` appears in exactly one linear row -- the cone's head -- and in the objective, with
coefficient ``lam``. Reading the direction subproblem's stationarity in the ``t`` slot gives

    rho * d_t + nu_cone = -lam,      so      w_0 = -nu_cone = lam + rho * d_t,

and at a stationary point ``d = 0``, so ``w_0 = lam``. A tangent factor's ``w`` is
``-nu_cone * (1, -u)``, whose tail has norm exactly ``|w_0|``, so ``w`` sits precisely on
the boundary of ``Q`` with a strictly positive head. It can be neither outside ``Q`` nor
zero. **An active cone at an eq. (7) optimum always contributes a genuine active normal**,
and §7.4's test says keep it, every time.

That is a result about the formulation, not a limitation of the rule: away from a
stationary point ``w_0 = lam + rho * d_t`` is free to go negative, and on a general SOCP
with several factors the test is a real one. It also removes the cycling hazard that a
geometric deactivation rule would have here -- there is nothing for the rule to release, so
there is nothing for §7.3 to re-acquire.

**A tangent step is retracted, and the reason is a result rather than an implementation
choice.** Two facts about eq. (7), both verified in ``tests/test_prototype.py``:

* at a boundary point, a direction satisfying eq. (3) has an *exact conic step of zero*.
  Tangency makes eq. (6)'s middle coefficient vanish, feasibility makes its constant term
  vanish, and its leading coefficient is ``||q||^2 - (u.T @ q)^2``, which Cauchy-Schwarz
  puts at or above zero. So ``a alpha^2 <= 0`` admits only ``alpha = 0``, except along the
  radial ray.
* the projected steepest-descent direction always pushes ``t`` *down*, by exactly
  ``lam / rho``. ``t`` appears in no linear row, so nothing in the working set opposes the
  objective's pull on it -- and from a boundary point that direction leaves the cone, which
  is again a step of zero.

Together: **an iterate on the cone's boundary cannot move at all** under a linear direction
and an exact conic step. That is not a bug to be fixed in the ratio test; it is the
curvature of the cone, and it is Risk 1 (#39) and §3.3's warning about becoming "an SQP
method with an SOC constraint" arriving together.

The prototype's answer is the one §3.3 sanctions for it -- *"COSA will initially use the
tangent representation to construct directions"* -- with the step made honest by a
**retraction**: move along the tangent, which leaves the cone, then restore feasibility by
raising the cone's head. That is free on eq. (7), where ``t`` appears only in the cone's
head row and the objective, and it is exactly what
:meth:`cosa.MeanStdPortfolio.socp_point` does at a solution. The retraction costs
``lam * delta_t``, which is second order in the step, while the tangent direction decreases
the objective at first order -- so a short enough step improves the objective, and a
backtracking search finds one. When no step improves it, the point is stationary for the
working set and the multiplier tests take over.

**#23's answer is to give the direction the curvature the tangent plane is missing.** A
tangent step leaves a curved surface because the subproblem's ``H = rho*I`` describes a
*plane*; the Lagrangian's Hessian describes the surface. §3.3 asks for a formulation "in
terms of the primal-dual conic KKT conditions", and
:func:`cosa.active_set.multipliers.lagrangian_curvature` is what that phrase buys: the dual
variable ``mu = w_0`` weights the cone's second derivative and enters the *primal* direction
computation, so the two are solved together rather than one after the other.

The multipliers needed to build it are the ones the last direction produced, which makes the
scheme a fixed point rather than an implicit system -- the first iteration uses zero and so
reproduces Waves 4-6 exactly. The retraction stays, because any method that keeps its
iterates exactly on a curved boundary must restore feasibility after a straight step; what
changes is how often it has to work hard, and the metrics say by how much.

**The apex is a branch, not a special case in the loop.** When a factor's slack reaches its
apex the tangent representation has nothing to say, and #24's
:func:`cosa.solver.apex.apex_direction` is what answers instead: exact membership on the
direction, the normal cone on the multiplier. The loop calls it and steps along whatever it
returns. On a factor model this is not an exotic path -- a rank-``k`` covariance over many
more assets has a large null space, so the minimum-risk portfolio has risk exactly zero and
the optimum *is* the apex.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from cosa.active_set import updates
from cosa.active_set.multipliers import Multipliers, from_direction, lagrangian_curvature
from cosa.active_set.working_set import WorkingSet
from cosa.geometry.soc import is_apex
from cosa.geometry.step import StepLimit, linear_step, step_limit
from cosa.linear_algebra.kkt import RHO, SingularKktError
from cosa.linear_algebra.reuse import Reuse
from cosa.problem.socp import ProblemError, _vector
from cosa.solver.anticycling import Guard, objective_of
from cosa.solver.apex import apex_direction
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

BACKTRACKS: Final = 40
"""How many halvings the retraction's line search tries before giving up on a direction.

Forty halvings reaches ``1e-12`` of the initial step, which is far below any tolerance the
solver tests against -- so exhausting them means no step along this direction improves the
objective, which is a statement about the point rather than about the search.
"""

ARMIJO: Final = 1e-4
"""How much of the predicted decrease a retracted step must actually deliver.

The standard sufficient-decrease constant. Small because the retraction's cost is second
order and the predicted decrease is first order, so almost any short step qualifies; the
condition is there to reject a step that has gone far enough for the curvature to eat the
improvement, not to be selective.
"""

MAX_ITERATIONS: Final = 1000
"""How many iterations before the loop gives up and says so.

An active-set method's iteration count is bounded by the number of working sets, which is
finite but astronomically large, so a limit is not a safety net -- it is the admission that
without the anti-cycling rules of #29 the loop can revisit a working set forever.

A thousand rather than the two hundred the polyhedral baseline needed, because the
retraction changes the cost model. Crawling along a *curved* boundary in tangent steps is a
first-order process: each step is limited by how far the retraction can be afforded, so the
count grows as the optimum is approached. #23's Lagrangian curvature roughly halves that -- measured across the family
suite, and on the ill-conditioned family it is the difference between an answer and the
iteration limit -- but the shape of the cost is unchanged, and #27's factorization reuse is
what addresses the rest. §9 Phase III (``paper.tex:740``) says correctness matters more than
speed at this stage, and this is where that is being spent.
"""

_PROGRESS: Final = 1e-12
"""How far an accepted step must move the iterate to count as progress.

Relative to the iterate, and far tighter than :data:`_STATIONARY` because it measures a
different thing: not whether the direction is small but whether the *step along it* achieved
anything. The two fail apart. A direction can sit an order of magnitude above the
stationarity threshold while every step along it is cut to nothing by the retraction, and
that is precisely the tail behaviour that used to consume the iteration budget.

Asking about the point rather than the direction is also scale-free in the way the direction
test is not: ``d`` scales as ``1 / rho`` and with the objective's magnitude, while "the
iterate did not change" means the same thing on every instance.
"""

_STATIONARY: Final = 1e-10
"""How small a direction must be, relative to the iterate, to count as vanished.

Relative because the direction scales as ``1 / rho``: an absolute threshold would mean a
different thing at every ``rho``, which is exactly the confusion #12's docstring warns
against.

**It is deliberately not the loop's only stopping test, and that is why it can stay tight.**
Near a solution the retraction cuts every step to nothing while ``d`` sits an order of
magnitude above this, so a threshold tuned to catch that would have to be loose enough to
stop early elsewhere -- and the value that did so was found by fitting, which is a bad way
to choose a constant. :data:`_PROGRESS` asks the question directly instead, and with it in
place this threshold makes no difference at ``1e-10`` or ``1e-9``: same statuses, same
answers. It is left where it started.
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
        status: ``"optimal"``, ``"iteration_limit"``, ``"unbounded"``, ``"degenerate"``,
            ``"stalled"`` or ``"blocked-at-apex"``. A status other than ``"optimal"`` is a
            description of what stopped the loop, not an excuse: each names a specific thing
            that happened. ``"blocked-at-apex"`` is the Risk 1 case #24 identified -- the
            apex is not justified and cannot be released, and §7.4's test cannot help
            because the release it would authorize is arithmetically unavailable. It
            belongs to #39 rather than to a bug.
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
    reuse: Reuse | bool | None = True,
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
        reuse: #27's factorization cache. ``True`` builds a fresh one, ``False`` restores
            §13.1's refactorize-every-iteration reference policy, and a
            :class:`cosa.linear_algebra.reuse.Reuse` carried in from a previous solve is
            what #30's warm start passes -- the answers are identical either way and only
            :attr:`cosa.solver.instrumentation.Metrics.factorizations` differs.

    Returns:
        The solution, whatever its status.

    Raises:
        ProblemError: if the iteration limit is not positive, or the supplied start is not
            feasible.
    """
    if max_iterations < 1:
        raise ProblemError("max_iterations", f"expected at least one iteration, found {max_iterations}")
    stopping = residuals_tolerance(tolerance)
    activation = updates.ACTIVATION_TOLERANCE
    recorder = recorder or Recorder()
    cache = Reuse() if reuse is True else (None if reuse is False else reuse)
    guard = Guard()

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
    # The Lagrangian curvature needs multipliers the direction has not produced yet, so the
    # first pass uses zero -- which is `H = rho*I`, the subproblem Waves 4-6 solved. From
    # then on each direction is built on the last one's duals: a fixed-point iteration, and
    # the cheapest honest reading of "primal-dual".
    previous = _no_multipliers(problem)

    with recorder.solving():
        for _ in range(max_iterations):
            recorder.iteration()
            recorder.working_set_seen(guard.saw(working_set))
            at_apex = _apex_factor(problem, point, working_set)
            if at_apex is not None:
                # §8.1's geometry, through #24's branch: exact membership on the direction
                # and the normal cone on the multiplier, because there is no tangent
                # hyperplane here to put in the working set.
                answer = apex_direction(problem, working_set, point, at_apex, rho=rho)
                before = working_set.cone_status
                working_set = answer.working_set
                for old_status, new_status in zip(before, working_set.cone_status, strict=True):
                    recorder.cone_changed(old_status, new_status)
                if not answer.released:
                    found = answer.multipliers
                    measured = residuals(problem, point, found)
                    recorder.kkt_residual(measured.largest)
                    status = "optimal" if measured.is_optimal(tolerance=stopping) else "blocked-at-apex"
                    return Solution(
                        z=point,
                        multipliers=found,
                        working_set=working_set,
                        residuals=measured,
                        status=status,
                        metrics=recorder.metrics(),
                    )
                direction = answer.direction
                recorder.kkt_residual(answer.violation)
                limit = step_limit(problem, point, direction.d, working_set)
                if limit.is_unbounded:  # pragma: no cover - a released apex is bounded by its own cone
                    return _finish(problem, point, working_set, "unbounded", recorder, stopping)
                point = point + limit.alpha * direction.d
                checker.accepted_iterate(problem, point)
                guard.accepted(objective_of(problem.c, point))
                working_set = updates.activate_cones(problem, point, working_set, tolerance=activation)
                continue

            unchanged = working_set
            curvature = lagrangian_curvature(problem, working_set, point, previous)
            try:
                direction = recorder.solve_direction(
                    problem, working_set, point, rho=rho, curvature=curvature, reuse=cache
                )
            except SingularKktError:
                working_set, dropped = updates.drop_dependent_rows(problem, working_set, point)
                if dropped:
                    for _row in dropped:
                        recorder.constraint_removed()
                    continue
                if not regularization:
                    return _finish(problem, point, working_set, "degenerate", recorder, stopping)
                direction = recorder.solve_direction(
                    problem,
                    working_set,
                    point,
                    rho=rho,
                    regularization=regularization,
                    curvature=curvature,
                )

            found = from_direction(problem, working_set, point, direction)
            previous = found
            measured = residuals(problem, point, found)
            recorder.kkt_residual(measured.largest)

            if float(np.abs(direction.d).max(initial=0.0)) <= _STATIONARY * max(
                1.0, float(np.abs(point).max(initial=0.0))
            ):
                checker.computed_multipliers(problem, found)
                revised = _revise(problem, working_set, found, recorder, guard, point)
                if revised is not None:
                    working_set = revised
                    continue
                status = "optimal" if measured.is_optimal(tolerance=stopping) else "degenerate"
                return Solution(
                    z=point,
                    multipliers=found,
                    working_set=working_set,
                    residuals=measured,
                    status=status,
                    metrics=recorder.metrics(),
                )

            if working_set.active_cones and _heads_are_free(problem):
                stepped = _retracted_step(problem, point, direction.d, working_set)
                if stepped is None:
                    # No step along this direction improves the objective once the cone has
                    # been restored, so the point is stationary for this working set even
                    # though the direction is not zero. The multiplier tests decide next.
                    revised = _revise(problem, working_set, found, recorder, guard, point)
                    if revised is not None:
                        working_set = revised
                        continue
                    status = "optimal" if measured.is_optimal(tolerance=stopping) else "stalled"
                    return Solution(
                        z=point,
                        multipliers=found,
                        working_set=working_set,
                        residuals=measured,
                        status=status,
                        metrics=recorder.metrics(),
                    )
                candidate, limit = stepped
            else:
                limit = step_limit(problem, point, direction.d, working_set)
                if limit.is_unbounded:
                    return _finish(problem, point, working_set, "unbounded", recorder, stopping)
                candidate = point + limit.alpha * direction.d

            # §17.2's merit safeguard, and #29's no-progress rule, which reach the same
            # place by different routes and so share one.
            #
            # The safeguard: the direction is a descent direction and the step was chosen
            # along it, so an accepted iterate *worse* than the best one seen means the
            # arithmetic and the geometry have disagreed, and continuing from it would
            # compound the disagreement. The rule: a step that moves the iterate by nothing
            # is not progress, whatever the ratio test called it -- near a solution the
            # retraction's corrections are second order and the loop can spend its whole
            # budget taking steps that change no digit of the point.
            #
            # A refused step and a step that achieved nothing are the same situation from
            # here on: the iterate stands, and whether that is optimality or a stall is the
            # multiplier tests' to decide. So a refusal is recorded as `stationary` rather
            # than handled, and the one exit below serves both.
            value = objective_of(problem.c, candidate)
            if guard.accepts(value):
                moved = float(np.abs(candidate - point).max(initial=0.0))
                stationary = moved <= _PROGRESS * max(1.0, float(np.abs(point).max(initial=0.0)))
                point = candidate
                guard.accepted(value)
                checker.accepted_iterate(problem, point)
                if limit.blocking is not None:
                    working_set = updates.add_inequality(working_set, limit.blocking)
                    recorder.constraint_added()
            else:
                stationary = True

            # §7.3: the cone's status follows the geometry -- but only where there is new
            # geometry to read. A step that moved the
            # iterate by nothing leaves the conic slack exactly as it was, so re-deriving the
            # cone's status from it can only restore what the iteration just released -- and
            # on eq. (7) at the apex that is a two-state cycle, APEX to INACTIVE and back,
            # which is what §8.2's oscillation looks like when the factor is at its apex
            # rather than near its boundary. Skipping the re-derivation lets the no-progress
            # rule below see an unchanged working set and stop.
            if not stationary:
                before = working_set.cone_status
                working_set = updates.activate_cones(problem, point, working_set, tolerance=activation)
                for old, new in zip(before, working_set.cone_status, strict=True):
                    recorder.cone_changed(old, new)

            # An iteration that moved the iterate by nothing *and* changed nothing about the
            # working set has achieved nothing at all, and repeating it will achieve nothing
            # again. A zero-length step is not itself evidence of that -- blocking at
            # `alpha = 0` is how a constraint gets added, and adding it is progress -- so
            # both halves are required. This is what catches the tail the direction test
            # cannot: near a solution the retraction cuts steps to nothing while `d` stays an
            # order of magnitude above `_STATIONARY`, and the loop would otherwise spend its
            # whole budget there.
            if stationary and working_set == unchanged:
                revised = _revise(problem, working_set, found, recorder, guard, point)
                if revised is not None:
                    working_set = revised
                    continue
                status = "optimal" if measured.is_optimal(tolerance=stopping) else "stalled"
                return Solution(
                    z=point,
                    multipliers=found,
                    working_set=working_set,
                    residuals=residuals(problem, point, found),
                    status=status,
                    metrics=recorder.metrics(),
                )

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


def _apex_factor(problem: SOCP, z: Vector, working_set: WorkingSet) -> int | None:
    """The first cone factor sitting at its apex, if any.

    Args:
        problem: the instance.
        z: the current point.
        working_set: the current set, whose statuses say which factors are in play.

    Returns:
        The factor's index, or ``None`` when no active factor is at its apex.
    """
    if not working_set.active_cones:
        return None
    slack = problem.cone_slack(z)
    for factor in working_set.active_cones:
        if is_apex(slack[problem.cone.slices[factor]]):
            return factor
    return None


def _heads_are_free(problem: SOCP) -> bool:
    """Can conic feasibility be restored by raising the cone's heads?

    True for eq. (7) and for every instance the portfolio families produce, false for a
    general SOCP whose head variable is constrained elsewhere. The retraction is only
    available where it is, and the loop falls back to the exact conic step otherwise --
    which on a boundary point means it stalls, honestly.

    Args:
        problem: the instance.

    Returns:
        Whether :func:`cosa.solver.initialization.raise_free_heads` will succeed.
    """
    try:
        raise_free_heads(problem, np.zeros(problem.num_variables))
    except ProblemError:
        return False
    return True


def _retracted_step(
    problem: SOCP,
    z: Vector,
    d: Vector,
    working_set: WorkingSet,
    *,
    backtracks: int = BACKTRACKS,
    armijo: float = ARMIJO,
) -> tuple[Vector, StepLimit] | None:
    """Step along the tangent and restore the cone, backtracking until the objective falls.

    The tangent direction leaves the cone at second order, so the retraction's cost is
    second order in the step while the direction's improvement is first order. A short
    enough step therefore improves the objective after retraction, and halving finds one.

    The step is bounded by the *linear* ratio test and not by the conic one, deliberately:
    the conic interval would report zero, which is true of the unretracted step and beside
    the point of a retracted one.

    Args:
        problem: the instance.
        z: the current point, feasible.
        d: the direction, tangent to every active cone.
        working_set: the current set, for the linear ratio test.
        backtracks: how many halvings to try.
        armijo: the fraction of the predicted decrease a step must deliver.

    Returns:
        The retracted point and the step that produced it, or ``None`` when no step
        improves the objective -- which means the point is stationary for this working set.
    """
    predicted = float(problem.c @ d)
    if predicted >= 0.0:
        return None
    linear = linear_step(problem, z, d, working_set)
    alpha = min(linear.alpha, 1.0) if math.isfinite(linear.alpha) else 1.0
    before = float(problem.c @ z)

    for _ in range(backtracks):
        candidate = raise_free_heads(problem, z + alpha * d)
        if float(problem.c @ candidate) <= before + armijo * alpha * predicted:
            blocking = linear.blocking if alpha == linear.alpha else None
            source = "linear" if blocking is not None else "retraction"
            return candidate, StepLimit(alpha=alpha, blocking=blocking, source=source)
        alpha /= 2.0
    return None


def _working_set_at(problem: SOCP, z: Vector, *, tolerance: float = updates.ACTIVATION_TOLERANCE) -> WorkingSet:
    """The working set §7.1 and §7.3 make active at a point.

    The loop must start with a working set that *matches* its point: a set holding a row
    the point is not on would impose ``a_i.T @ p = 0`` about a constraint that is not
    tight, and the direction would refuse to move in a direction that is perfectly feasible.
    The same argument applies to the cone, which is why §7.3's rule runs here too.

    Args:
        problem: the instance.
        z: the starting point.
        tolerance: §7.1's and §7.3's activation tolerance.

    Returns:
        The set with every tight inequality active and every geometrically active cone
        factor at its correct status.
    """
    working_set = WorkingSet.empty(problem)
    for index in updates.activation_candidates(problem, z, working_set, tolerance=tolerance):
        working_set = updates.add_inequality(working_set, index)
    return updates.activate_cones(problem, z, working_set, tolerance=tolerance)


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


def _no_multipliers(problem: SOCP) -> Multipliers:
    """Zero multipliers, the starting point of the curvature's fixed-point iteration.

    Zero rather than a guess: it makes the first direction subproblem exactly ``H = rho*I``,
    so the first step of a #23 solve is bit-for-bit the first step of a Wave 6 solve and any
    difference between them is attributable to the curvature rather than to initialization.

    Args:
        problem: the instance, for its shape.

    Returns:
        Multipliers of the right shape, all zero.
    """
    return Multipliers(
        y=np.zeros(problem.num_inequalities),
        nu=np.zeros(problem.num_equalities),
        w=np.zeros(problem.cone.dim),
    )


def _revise(
    problem: SOCP,
    working_set: WorkingSet,
    found: Multipliers,
    recorder: Recorder,
    guard: Guard,
    z: Vector,
) -> WorkingSet | None:
    """The multiplier test of §4.1's step 8: does any multiplier say to drop something?

    Both places the loop stops making progress ask this same question, so it is asked in
    one place. §7.2's inequality rule is tried first and §7.4's conic rule second, and the
    order is not arbitrary: a wrong-signed ``y`` is routine and cheap to act on, while
    releasing a cone changes the local description of the feasible set. Trying the cheaper
    repair first means the conic test is only ever asked at a point where nothing else
    explains the halt.

    Args:
        problem: the instance.
        working_set: the current set.
        found: the multipliers at this point.
        recorder: the metrics recorder, told about each removal and status change.
        guard: #29's anti-cycling state, which decides whether §7.2's rule or Bland's is in
            force. The switch lives there rather than here so that this function does not
            have to know there is one.
        z: the current point, for §8.2's hysteresis band.

    Returns:
        A revised working set to continue from, or ``None`` when every multiplier has the
        sign its constraint requires -- which is the point at which the residuals get to
        say whether that means optimal or merely stuck.
    """
    dropping = guard.candidate(working_set, found.y, tolerance=updates.MULTIPLIER_TOLERANCE)
    if dropping is not None:
        recorder.constraint_removed()
        return updates.drop_inequality(working_set, dropping)
    updated, dropped = updates.deactivate_cones(problem, working_set, found, z=z)
    if not dropped:
        return None
    for index in dropped:
        recorder.cone_changed(working_set.status(index), updated.status(index))
        recorder.constraint_removed()
    return updated
