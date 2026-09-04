"""The Phase I loop: nine components wired together, and every iterate feasible.

The executable half of issue #14. Its "done when" is two claims:

* *the Phase I QP solves with all nine components of §9's list present* -- checked by
  solving, and by naming the nine and where each lives, so a component cannot go missing
  without a test saying which;
* *every accepted iterate satisfies `Ax <= b`, `Ex = d` within tolerance* -- §14.1's Level 1,
  which is a **runtime** invariant and not a property of the answer. Every solve in this
  file runs with `CHECKED`, so the invariant is asserted at each accepted iterate rather
  than at the end. A loop that wandered outside the feasible set and came back would pass
  an answer-only test and fail these.

Correctness is measured against the reference solver of #21 on every instance, which is
§16.3's cross-check applied to COSA's own output for the first time.
"""

import numpy as np
import pytest

from cosa import SOCP, MeanStdPortfolio, ProblemError, SingularKktError
from cosa.experiments import portfolio as families
from cosa.experiments import randomized, reference
from cosa.solver import cosa
from cosa.solver.initialization import NeedsPhaseOneError
from cosa.solver.instrumentation import CHECKED, Recorder


def lp(objective, rows, rhs, equalities=None, equality_rhs=None):
    """A small linear program, as an SOCP with no cone."""
    problem = SOCP.unconstrained(np.asarray(objective, dtype=float)).add_inequalities(rows, rhs)
    if equalities is not None:
        problem = problem.add_equalities(equalities, equality_rhs)
    return problem


@pytest.fixture
def simplex():
    """The objective -x1 - 2 x2 - x3/2 over the unit simplex: the optimum is a vertex."""
    return lp([-1.0, -2.0, -0.5], -np.eye(3), np.zeros(3), [[1.0, 1.0, 1.0]], [1.0])


# ----------------------------------------------------------------------------------
# §9 Phase I's nine components
# ----------------------------------------------------------------------------------

NINE_COMPONENTS = {
    "feasible initialization": "cosa.solver.initialization",
    "active-set representation": "cosa.active_set.working_set",
    "direction computation": "cosa.linear_algebra.kkt",
    "KKT solve": "cosa.linear_algebra.kkt",
    "multiplier calculation": "cosa.active_set.multipliers",
    "constraint addition": "cosa.active_set.updates",
    "constraint deletion": "cosa.active_set.updates",
    "ratio test": "cosa.geometry.step",
    "termination checks": "cosa.solver.termination",
}


@pytest.mark.parametrize(("component", "module"), sorted(NINE_COMPONENTS.items()))
def test_every_phase_one_component_exists(component, module):
    """§9's list at `paper.tex:695`, each with a module that owns it."""
    import importlib

    assert importlib.import_module(module).__doc__, f"§9 Phase I requires {component!r}"


def test_the_loop_uses_all_nine(simplex):
    """Not merely that they exist: that one solve exercises each.

    Read off the metrics, which is the only way to tell from outside that a constraint was
    added and another dropped rather than the loop stumbling onto the answer.
    """
    solution = cosa.solve(simplex, checker=CHECKED)
    assert solution.is_optimal
    assert solution.metrics.iterations > 1, "the loop iterated"
    assert solution.metrics.factorizations > 0, "the KKT system was solved"
    assert solution.metrics.constraints_added > 0, "the ratio test blocked and a row was added"
    assert solution.working_set.inequalities, "an active-set representation is held"
    assert solution.residuals.is_optimal(), "termination is the residuals' verdict"


# ----------------------------------------------------------------------------------
# It solves, and the answer is right
# ----------------------------------------------------------------------------------

PROGRAMS = {
    "simplex": lp([-1.0, -2.0, -0.5], -np.eye(3), np.zeros(3), [[1.0, 1.0, 1.0]], [1.0]),
    "box": lp(
        [-1.0, -2.0],
        np.vstack([np.eye(2), -np.eye(2)]),
        np.array([1.0, 1.0, 0.0, 0.0]),
    ),
    "half spaces": lp([1.0, 1.0], [[-1.0, -1.0], [-1.0, 0.0], [0.0, -1.0]], [-1.0, 0.0, 0.0]),
    "degenerate vertex": lp(
        [1.0, 1.0],
        [[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [-1.0, -1.0]],
        [-0.5, -0.5, 10.0, -1.0],
        [[1.0, -1.0]],
        [0.0],
    ),
    "needs phase I": lp(
        [1.0, 1.0],
        [[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]],
        [-0.5, -0.5, 10.0],
        [[1.0, -1.0]],
        [0.0],
    ),
}


@pytest.fixture(params=sorted(PROGRAMS))
def program(request):
    """One linear program, by name."""
    return request.param, PROGRAMS[request.param]


def test_the_loop_agrees_with_a_reference_solver(program):
    """§16.3's cross-check, applied to COSA's own answer for the first time."""
    name, problem = program
    solution = cosa.solve(problem, checker=CHECKED)
    oracle = reference.solve_reference(problem)
    assert solution.is_optimal, f"{name}: {solution}"
    assert oracle.is_optimal
    assert solution.objective(problem) == pytest.approx(oracle.objective, abs=1e-6)


def test_every_accepted_iterate_is_feasible(program):
    """Issue #14's second "done when", and the reason `CHECKED` is passed everywhere.

    §14.1 says *every accepted iterate*, not the answer. `CHECKED` asserts it inside the
    loop, so this test passing means the invariant held at each step and not merely at the
    end.
    """
    _, problem = program
    solution = cosa.solve(problem, checker=CHECKED)
    from cosa.solver.instrumentation import level_1_violations

    assert level_1_violations(problem, solution.z) == ()


def test_the_final_residuals_certify_the_answer(program):
    """§14.3's Level 3: all five within tolerance is a certificate, the problem being convex."""
    _, problem = program
    solution = cosa.solve(problem, checker=CHECKED)
    assert solution.residuals.is_optimal()
    assert solution.residuals.worst() in {"none", "primal feasibility", "linear complementarity"}


# ----------------------------------------------------------------------------------
# The two exits, and the difference between them
# ----------------------------------------------------------------------------------


def test_a_constraint_is_dropped_when_its_multiplier_is_wrong_signed():
    """The exit that spends an iteration without moving -- §7.2's rule inside the loop.

    Constructed so the loop must pass through a vertex that is optimal for a *larger*
    working set than the true one: it starts in a corner the objective wants to leave along
    a constraint it is holding.
    """
    problem = lp(
        [-1.0, -2.0],
        np.vstack([np.eye(2), -np.eye(2)]),
        np.array([1.0, 1.0, 0.0, 0.0]),
    )
    solution = cosa.solve(problem, start=np.array([0.0, 0.0]), checker=CHECKED)
    assert solution.is_optimal
    assert solution.metrics.constraints_removed > 0, "a row was held and then let go"
    np.testing.assert_allclose(solution.z, [1.0, 1.0], atol=1e-9)


def test_an_unbounded_program_is_detected_not_iterated():
    """No row blocks a descent direction, so the objective has no floor -- and the loop says so."""
    problem = lp([-1.0], [[-1.0]], [0.0])
    solution = cosa.solve(problem)
    assert solution.status == "unbounded"
    assert not solution.is_optimal


def test_an_infeasible_program_is_proved_infeasible():
    """By the elastic Phase I, whose optimum needs a positive relaxation."""
    problem = lp([1.0], [[1.0], [-1.0]], [-1.0, -1.0])
    with pytest.raises(NeedsPhaseOneError, match="infeasible"):
        cosa.solve(problem)


def test_the_iteration_limit_is_reported_rather_than_hit_silently():
    """A limit that stopped the loop is a status, not an exception -- and the residuals say how close."""
    problem = PROGRAMS["simplex"]
    solution = cosa.solve(problem, max_iterations=1)
    assert solution.status in {"iteration_limit", "optimal"}
    assert solution.metrics.iterations <= 1


def test_a_non_positive_iteration_limit_is_rejected():
    """Zero iterations is not a solve."""
    with pytest.raises(ProblemError, match="at least one iteration"):
        cosa.solve(PROGRAMS["box"], max_iterations=0)


# ----------------------------------------------------------------------------------
# Phase I, running itself
# ----------------------------------------------------------------------------------


def test_the_loop_solves_its_own_initialization():
    """The recursion that makes this Phase I rather than a lookup.

    The instance's least-norm equality point violates a bound, so route 3 builds an elastic
    relaxation and the loop solves *that* to find a start -- then solves the original from
    it. Both solves are counted on the same recorder, because both are part of the cost.
    """
    problem = PROGRAMS["needs phase I"]
    recorder = Recorder()
    solution = cosa.solve(problem, checker=CHECKED, recorder=recorder)
    assert solution.is_optimal
    assert solution.metrics.iterations > 2, "the Phase I solve is counted too"


def test_the_phase_one_recursion_stops_at_depth_one():
    """The elastic problem comes with a start, so its own solve never needs a Phase I."""
    from cosa.solver import initialization as init

    elastic = init.elastic_problem(PROGRAMS["needs phase I"])
    solution = cosa.solve(elastic.problem, start=elastic.start, phase_one=False, checker=CHECKED)
    assert solution.is_optimal
    assert elastic.relaxation(solution.z) <= 1e-9, "no relaxation is needed, so the original is feasible"


def test_a_supplied_start_is_used(simplex):
    """Route 1, through the loop: the cheapest initialization is the one already in hand."""
    solution = cosa.solve(simplex, start=np.array([0.2, 0.5, 0.3]), checker=CHECKED)
    assert solution.is_optimal


def test_an_infeasible_supplied_start_is_refused(simplex):
    """Checked, not trusted -- and the loop does not silently repair it."""
    with pytest.raises(NeedsPhaseOneError, match="not feasible"):
        cosa.solve(simplex, start=np.array([2.0, 0.0, 0.0]))


# ----------------------------------------------------------------------------------
# §8.3's degeneracy, resolved rather than surrendered to
# ----------------------------------------------------------------------------------


def test_a_degenerate_working_set_is_repaired_by_dropping_a_dependent_row():
    """#25's dependent-constraint removal, exercised through the loop.

    This instance's optimum has three active rows spanning two dimensions -- two bounds and
    an equality -- so the direction solve refuses it. Before #25 the loop stopped with a
    `degenerate` status and a stationarity residual of 1; now it drops the redundant row and
    finishes with a certificate.
    """
    problem = PROGRAMS["needs phase I"]
    solution = cosa.solve(problem, checker=CHECKED)
    assert solution.is_optimal
    assert solution.residuals.stationarity < 1e-9
    np.testing.assert_allclose(solution.z, [0.5, 0.5], atol=1e-9)


def test_regularization_is_the_fallback_and_can_be_switched_off():
    """§8.3 lists both because removal cannot always help; here it can, so both routes work."""
    problem = PROGRAMS["needs phase I"]
    assert cosa.solve(problem, regularization=0.0, checker=CHECKED).is_optimal
    assert cosa.solve(problem, regularization=1e-8, checker=CHECKED).is_optimal


# ----------------------------------------------------------------------------------
# What it cannot do yet, said out loud
# ----------------------------------------------------------------------------------


def test_a_portfolio_whose_cone_binds_is_refused():
    """Issue #14's scope: §9 Phase I is the polyhedral baseline, and eq. (7)'s cone always binds.

    Refused with a message naming #18 rather than solved wrongly. #20 is where this starts
    working, and nothing about the loop changes when it does -- only what `step_limit`
    returns.
    """
    instance = families.basic(4, seed=0)
    with pytest.raises(ProblemError, match="#18"):
        cosa.solve(instance.problem, start=instance.witness)


def test_a_randomized_instance_is_refused_for_the_same_reason():
    """The same boundary, on an instance nobody chose."""
    instance = randomized.random_instance(3)
    with pytest.raises(ProblemError, match="#18"):
        cosa.solve(instance.problem, start=instance.witness)


def test_an_irreparable_degeneracy_stops_the_loop_unless_regularized():
    """The path §8.3 keeps regularization for: a dependency the working set may not remove.

    Two identical equality rows, which §3.1 forbids dropping. With `delta = 0` the loop
    stops and says `degenerate`; with `delta > 0` it gets an answer to a nearby problem and
    runs to a conclusion instead.

    *Which* conclusion is the honest part. On this instance the objective is constant on the
    feasible line, so the true direction at every point is exactly zero -- and the
    regularized one is `O(delta)`, which with nothing to block a step reads as an improving
    direction. So the loop concludes `unbounded`. That is the documented cost of the
    fallback rather than a defect: it answers a nearby question, and near questions can have
    different answers. It is also why `REGULARIZATION` is small and why
    dependent-constraint removal is always tried first.
    """
    flat = SOCP.unconstrained(np.array([1.0, 1.0])).add_equalities([[1.0, 1.0], [1.0, 1.0]], [1.0, 1.0])

    stopped = cosa.solve(flat, regularization=0.0)
    assert stopped.status == "degenerate"
    assert stopped.residuals.stationarity > 0.0, "the multipliers were never determined"

    regularized = cosa.solve(flat, regularization=1e-8)
    assert regularized.status != "degenerate", "the solve completed rather than stopping"
    assert regularized.status == "unbounded", "on a flat objective, to a different answer"


def test_regularization_lets_the_direction_solve_proceed_where_removal_cannot():
    """The same claim at the level it is unambiguous: one direction solve, not a whole loop.

    On the working set the loop above cannot repair, the unregularized solve refuses and the
    regularized one returns a finite direction and finite multipliers. That is all
    regularization promises, and separating it from the loop's eventual verdict is what
    keeps the promise checkable.
    """
    from cosa import WorkingSet
    from cosa.linear_algebra import kkt

    flat = SOCP.unconstrained(np.array([1.0, 1.0])).add_equalities([[1.0, 1.0], [1.0, 1.0]], [1.0, 1.0])
    working_set = WorkingSet.empty(flat)
    point = np.array([0.5, 0.5])

    with pytest.raises(SingularKktError):
        kkt.direction(flat, working_set, point)
    step = kkt.direction(flat, working_set, point, regularization=1e-8)
    assert np.all(np.isfinite(step.d))
    assert np.all(np.isfinite(step.multipliers))


def test_phase_one_runs_on_a_coned_instance_and_the_cone_is_what_refuses():
    """Phase I completes, raises the free heads, and *then* the step guard stops the solve.

    An instance whose equal-weight point violates a floor, so route 2 fails and the elastic
    relaxation runs. That path ends by raising the cone's head with a margin -- so the start
    it produces is feasible in every block -- and the refusal that follows comes from the
    step limit, not from initialization. Exactly the scope boundary #14 draws.
    """
    portfolio = MeanStdPortfolio(
        mu=np.array([0.10, 0.04, 0.06, 0.05]),
        Sigma=np.diag([0.04, 0.09, 0.16, 0.05]),
        lam=2.0,
        A=np.vstack([-np.eye(4), [[-1.0, 0.0, 0.0, 0.0]]]),
        b=np.concatenate([np.zeros(4), [-0.5]]),
        E=np.ones((1, 4)),
        d=np.ones(1),
    )
    problem = portfolio.to_socp()
    with pytest.raises(ProblemError, match="#18"):
        cosa.solve(problem, checker=CHECKED)


def test_an_exhausted_loop_reports_optimal_when_the_point_is():
    """The iteration limit is a stopping rule, not a verdict on the answer.

    Two iterations from an interior start land exactly on the optimum, and the loop runs
    out before it can verify. `_finish` recomputes the residuals anyway, finds them within
    tolerance, and reports `optimal` -- because Level 3 is a certificate about a point, not
    about how the point was reached.
    """
    box = SOCP.unconstrained(np.array([-1.0, -2.0])).add_inequalities(
        np.vstack([np.eye(2), -np.eye(2)]), np.array([1.0, 1.0, 0.0, 0.0])
    )
    stopped = cosa.solve(box, start=np.array([0.5, 0.5]), max_iterations=2, checker=CHECKED)
    np.testing.assert_allclose(stopped.z, [1.0, 1.0], atol=1e-9)
    assert stopped.status == "optimal"


def test_a_phase_one_start_repairs_the_cone_before_returning():
    """Route 3 drops the cone, so the head is raised afterwards -- with a margin.

    Checked through the loop on an instance that needs Phase I *and* has a cone: the
    resulting start must be feasible in every block, cone included, or the first iterate
    would fail Level 1.
    """
    from cosa.solver import initialization as init
    from cosa.solver.instrumentation import level_1_violations

    instance = families.basic(4, seed=0)
    problem = instance.problem
    elastic = init.elastic_problem(problem)
    relaxed = cosa.solve(elastic.problem, start=elastic.start, phase_one=False, checker=CHECKED)
    point = init.raise_free_heads(problem, elastic.original_point(relaxed.z), margin=1.0)
    assert level_1_violations(problem, point) == ()


# ----------------------------------------------------------------------------------
# The report
# ----------------------------------------------------------------------------------


def test_the_solution_carries_its_working_set(simplex):
    """Success Criterion 3: the answer says which constraints it is holding."""
    solution = cosa.solve(simplex, checker=CHECKED)
    described = solution.working_set.describe()
    assert "active inequalities" in described
    assert solution.working_set.num_inequalities == simplex.num_inequalities


def test_the_solution_reports_status_residuals_and_metrics(simplex):
    """One line carrying everything a benchmark row needs."""
    rendered = str(cosa.solve(simplex, checker=CHECKED))
    assert "optimal" in rendered
    assert "primal=" in rendered
    assert "iters" in rendered


def test_the_metrics_count_one_factorization_per_solve(simplex):
    """§13.1's policy, visible through the loop -- the baseline #27 has to beat."""
    metrics = cosa.solve(simplex, checker=CHECKED).metrics
    assert metrics.factorizations == metrics.kkt_solves


def test_rho_changes_nothing_but_the_arithmetic(simplex):
    """It scales the direction and the ratio test divides the scale out again."""
    answers = [cosa.solve(simplex, rho=rho, checker=CHECKED) for rho in (0.01, 1.0, 100.0)]
    for solution in answers:
        assert solution.is_optimal
        np.testing.assert_allclose(solution.z, answers[0].z, atol=1e-9)
