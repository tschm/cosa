"""Getting a feasible point to start from, by the three routes and in that order.

Part of issue #14 -- "feasible initialization", the first of §9 Phase I's nine components.
The three routes are cheap-to-expensive, and each is tested for what it does *and* for
handing off cleanly when it cannot: a route that silently returned an infeasible point
would break §14.1's Level 1 invariant at the very first iterate.

Route 3, the elastic Phase I, is only half here -- this file checks the relaxation it
builds, and `test_cosa.py` checks the solve that runs it, because running it needs the loop.
"""

import numpy as np
import pytest

from cosa import SOCP, ProblemError
from cosa.experiments import portfolio as families
from cosa.solver import initialization as init
from cosa.solver.instrumentation import level_1_violations


@pytest.fixture
def simplex():
    """The unit simplex in three variables: `sum(x) = 1`, `x >= 0`."""
    return (
        SOCP.unconstrained(np.array([-1.0, -2.0, -0.5]))
        .add_inequalities(-np.eye(3), np.zeros(3))
        .add_equalities([[1.0, 1.0, 1.0]], [1.0])
    )


# ----------------------------------------------------------------------------------
# Route 1: a point the caller already has
# ----------------------------------------------------------------------------------


def test_a_feasible_point_is_accepted(simplex):
    """The cheapest route, and the one #30's warm start will use."""
    point = np.array([0.5, 0.25, 0.25])
    np.testing.assert_allclose(init.feasible_start(simplex, point), point)


def test_a_supplied_point_is_checked_not_trusted(simplex):
    """A caller who is wrong about feasibility finds out here, not at iterate seventeen."""
    with pytest.raises(init.NeedsPhaseOneError, match="not feasible"):
        init.feasible_start(simplex, np.array([-1.0, 1.0, 1.0]))


def test_every_generated_instance_s_witness_is_accepted():
    """The generators promise a feasible witness; this is the promise being collected.

    Across all twelve families, structured and adversarial: the initializer and the
    generators agree about what feasible means.
    """
    for instance in (*families.all_families(seed=0), *families.all_robustness(seed=0)):
        assert init.feasible_start(instance.problem, instance.witness) is not None


# ----------------------------------------------------------------------------------
# Route 2: the least-norm equality solution
# ----------------------------------------------------------------------------------


def test_the_equality_solution_is_least_norm(simplex):
    """Least-norm rather than any solution, because it is the one that does not wander."""
    point = init.equality_particular_solution(simplex)
    np.testing.assert_allclose(simplex.E @ point, simplex.d, atol=1e-12)
    np.testing.assert_allclose(point, 1.0 / 3.0)


def test_a_problem_with_no_equalities_starts_at_the_origin():
    """Nothing to solve, so nothing is solved."""
    problem = SOCP.unconstrained(np.array([1.0, 1.0]))
    np.testing.assert_array_equal(init.equality_particular_solution(problem), np.zeros(2))


def test_inconsistent_equalities_are_proved_infeasible():
    """The one infeasibility this module can establish on its own, before any inequality."""
    problem = SOCP.unconstrained(np.array([1.0])).add_equalities([[1.0], [1.0]], [1.0, 2.0])
    with pytest.raises(init.NeedsPhaseOneError, match="no solution"):
        init.equality_particular_solution(problem)


def test_route_two_succeeds_when_the_equality_point_is_feasible(simplex):
    """The equal-weight point satisfies the long-only bounds, so no Phase I is needed."""
    np.testing.assert_allclose(init.feasible_start(simplex), 1.0 / 3.0)


def test_route_two_hands_off_when_the_equality_point_is_not_feasible():
    """And says which route comes next rather than failing vaguely."""
    problem = (
        SOCP.unconstrained(np.array([1.0, 1.0]))
        .add_inequalities([[-1.0, 0.0]], [-0.5])
        .add_equalities([[1.0, -1.0]], [0.0])
    )
    with pytest.raises(init.NeedsPhaseOneError, match="elastic_problem"):
        init.feasible_start(problem)


# ----------------------------------------------------------------------------------
# The cone's free head, and its exact limits
# ----------------------------------------------------------------------------------


def test_raising_a_free_head_makes_the_cone_feasible():
    """Eq. (7)'s `t` appears only in the cone's head, so raising it repairs nothing else."""
    instance = families.basic(4, seed=0)
    problem = instance.problem
    point = np.concatenate([np.full(4, 0.25), [0.0]])
    assert level_1_violations(problem, point) != ()
    raised = init.raise_free_heads(problem, point)
    assert level_1_violations(problem, raised) == ()
    np.testing.assert_array_equal(raised[:4], point[:4]), "only t moved"


def test_a_margin_leaves_the_cone_strictly_inactive():
    """Which is what the polyhedral baseline wants: the conic interval stays out of the step."""
    from cosa.geometry import soc

    instance = families.basic(4, seed=0)
    problem = instance.problem
    point = np.concatenate([np.full(4, 0.25), [0.0]])
    assert soc.is_boundary(problem.cone_slack(init.raise_free_heads(problem, point)))
    assert soc.is_interior(problem.cone_slack(init.raise_free_heads(problem, point, margin=1.0)))


def test_route_two_repairs_the_cone_automatically():
    """So the whole mean-standard-deviation family gets a start with no witness needed."""
    instance = families.basic(5, seed=0)
    point = init.feasible_start(instance.problem, margin=1.0)
    assert level_1_violations(instance.problem, point) == ()


def test_a_head_that_is_not_free_is_refused():
    """A general SOCP whose head is constrained elsewhere cannot be repaired this way.

    Refused rather than guessed at: conic initialization in that generality needs a conic
    Phase I, which needs #18's step interval.
    """
    from cosa import ConeProduct

    problem = SOCP(
        c=np.array([0.0, 1.0]),
        A=np.array([[0.0, 1.0]]),
        b=np.array([1.0]),
        E=np.zeros((0, 2)),
        d=np.zeros(0),
        G=np.array([[0.0, 1.0], [1.0, 0.0]]),
        h=np.zeros(2),
        cone=ConeProduct.from_dims(2),
    )
    with pytest.raises(ProblemError, match="not a free variable"):
        init.raise_free_heads(problem, np.array([5.0, 0.0]))


def test_a_problem_with_no_cone_needs_no_head_raising():
    """The polyhedral case short-circuits, rather than looking for heads that are not there."""
    problem = SOCP.unconstrained(np.array([1.0, 1.0])).add_inequalities([[1.0, 1.0]], [1.0])
    point = np.array([0.25, 0.25])
    np.testing.assert_array_equal(init.raise_free_heads(problem, point), point)


def test_a_repeated_head_variable_is_not_free():
    """Two cone factors sharing one head cannot both be repaired by moving it."""
    from cosa import ConeProduct

    problem = SOCP(
        c=np.array([0.0, 1.0]),
        A=np.zeros((0, 2)),
        b=np.zeros(0),
        E=np.zeros((0, 2)),
        d=np.zeros(0),
        G=np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        h=np.zeros(4),
        cone=ConeProduct.from_dims(2, 2),
    )
    with pytest.raises(ProblemError, match="not a free variable"):
        init.raise_free_heads(problem, np.zeros(2))


def test_a_head_row_with_any_single_coefficient_is_free():
    """A coefficient of two is as settable as a coefficient of one — it is one division.

    The regression test for a bug that cost two waves. This required *exactly* `1.0`, which
    no rescaled instance satisfies, so `_heads_are_free` reported `False` on such an
    instance, the retraction was silently unavailable, and an iterate on the cone's boundary
    could not move at all. That failure was diagnosed as a conditioning problem and had an
    ablation apparently confirming it. See `docs/development/failure-modes.md`.
    """
    from cosa import ConeProduct

    problem = SOCP(
        c=np.array([0.0, 1.0]),
        A=np.zeros((0, 2)),
        b=np.zeros(0),
        E=np.zeros((0, 2)),
        d=np.zeros(0),
        G=np.array([[0.0, 2.0], [1.0, 0.0]]),
        h=np.zeros(2),
        cone=ConeProduct.from_dims(2),
    )
    from cosa.geometry import soc

    raised = init.raise_free_heads(problem, np.array([3.0, 0.0]))
    slack = problem.cone_slack(raised)
    assert slack[0] == pytest.approx(abs(slack[1])), "the head was set to exactly the tail's norm"
    assert soc.is_member(slack, tolerance=1e-12)


def test_a_head_row_selecting_two_variables_is_not_free():
    """Setting one variable no longer determines the head, so there is nothing to solve."""
    from cosa import ConeProduct

    problem = SOCP(
        c=np.array([0.0, 1.0, 0.0]),
        A=np.zeros((0, 3)),
        b=np.zeros(0),
        E=np.zeros((0, 3)),
        d=np.zeros(0),
        G=np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]),
        h=np.zeros(2),
        cone=ConeProduct.from_dims(2),
    )
    with pytest.raises(ProblemError, match="not a free variable"):
        init.raise_free_heads(problem, np.zeros(3))


def test_route_two_hands_off_when_the_cone_cannot_be_repaired():
    """A general SOCP with a constrained head reaches the caller as a Phase I request."""
    from cosa import ConeProduct

    problem = SOCP(
        c=np.array([0.0, 1.0]),
        A=np.array([[0.0, 1.0]]),
        b=np.array([1.0]),
        E=np.zeros((0, 2)),
        d=np.zeros(0),
        G=np.array([[0.0, 1.0], [1.0, 0.0]]),
        h=np.zeros(2),
        cone=ConeProduct.from_dims(2),
    )
    with pytest.raises(init.NeedsPhaseOneError, match="none could be built"):
        init.feasible_start(problem)


# ----------------------------------------------------------------------------------
# Route 3: the elastic relaxation
# ----------------------------------------------------------------------------------


@pytest.fixture
def needs_phase_one():
    """An instance whose least-norm equality point violates a bound."""
    return (
        SOCP.unconstrained(np.array([1.0, 1.0]))
        .add_inequalities([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]], [-0.5, -0.5, 10.0])
        .add_equalities([[1.0, -1.0]], [0.0])
    )


def test_the_relaxation_adds_one_variable(needs_phase_one):
    """One scalar, not one per row -- it finds a point that violates *nothing* by much."""
    elastic = init.elastic_problem(needs_phase_one)
    assert elastic.problem.num_variables == needs_phase_one.num_variables + 1
    assert elastic.elastic == needs_phase_one.num_variables


def test_the_relaxation_minimizes_the_relaxation(needs_phase_one):
    """Its objective is `s` alone: the original objective plays no part in finding a start."""
    elastic = init.elastic_problem(needs_phase_one)
    np.testing.assert_array_equal(elastic.problem.c, np.array([0.0, 0.0, 1.0]))


def test_the_relaxation_bounds_its_own_variable_below(needs_phase_one):
    """Without `s >= 0` the relaxed problem is unbounded, which would make it useless.

    A `z` running off along an unbounded edge takes `s` to minus infinity with it -- the
    row is satisfied by ever more slack. The extra row is what stops that.
    """
    elastic = init.elastic_problem(needs_phase_one)
    assert elastic.problem.num_inequalities == needs_phase_one.num_inequalities + 1
    np.testing.assert_array_equal(elastic.problem.A[-1], np.array([0.0, 0.0, -1.0]))
    assert elastic.problem.b[-1] == 0.0


def test_the_relaxation_start_is_feasible_by_construction(needs_phase_one):
    """Which is what stops the Phase I recursion at depth one."""
    elastic = init.elastic_problem(needs_phase_one)
    assert level_1_violations(elastic.problem, elastic.start) == ()


def test_the_relaxation_drops_the_cone():
    """Rather than relaxing it, which would leave it exactly active at the optimum.

    Exactly active is the state the polyhedral step cannot handle; dropping it and raising
    the free heads afterwards lands strictly inside instead.
    """
    instance = families.basic(4, seed=0)
    elastic = init.elastic_problem(instance.problem)
    assert len(elastic.problem.cone) == 0


def test_the_relaxation_reads_its_own_answer_back(needs_phase_one):
    """The two accessors a caller needs: the point, and how much relaxation it used."""
    elastic = init.elastic_problem(needs_phase_one)
    point = np.array([1.0, 2.0, 0.25])
    np.testing.assert_array_equal(elastic.original_point(point), [1.0, 2.0])
    assert elastic.relaxation(point) == 0.25
