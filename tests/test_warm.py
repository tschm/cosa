"""§9 Phase VI: starting the next solve from the last one, and what it saves.

Issue #30. The claim under test is the project's central hypothesis in miniature — that an
active-set method's state transfers between related problems and an interior-point method's
does not — so the measurement is the point and the plumbing exists to make it possible.

Four things are carried and each is tested for the same two properties: it is used when it
fits, and it is discarded rather than misapplied when it does not. A warm start is a hint.
"""

from dataclasses import replace

import numpy as np
import pytest

from cosa import SOCP, ConeStatus, Multipliers, WorkingSet
from cosa.active_set import updates
from cosa.experiments import portfolio as families
from cosa.linear_algebra.reuse import Reuse
from cosa.solver import cosa as solver
from cosa.solver import warm as warmstart
from cosa.solver.instrumentation import iterations_saved


@pytest.fixture
def instance():
    """A box-constrained portfolio: bounds that go active, and a cone that does too."""
    return families.box(8, seed=0)


@pytest.fixture
def solved(instance):
    """That instance, solved, which is what a warm start is made from."""
    return solver.solve(instance.problem)


def frontier(instance, lams):
    """The §11 sequence: one SOCP per risk aversion, everything else held fixed.

    Only `c` changes between consecutive problems — `A`, `E`, `G` and the cone product are
    identical — which is exactly the structure a warm start is supposed to exploit, and
    exactly what `WarmStart.fits` checks for.
    """
    return [replace(instance.portfolio, lam=float(lam)).to_socp() for lam in lams]


# ----------------------------------------------------------------------------------
# The four things carried
# ----------------------------------------------------------------------------------


def test_a_warm_start_carries_all_four(instance, solved):
    """§9 Phase VI's list, and nothing quietly left out."""
    cache = Reuse()
    hint = warmstart.from_solution(solved, cache=cache)
    assert hint.z is solved.z
    assert hint.working_set is solved.working_set
    assert hint.multipliers is solved.multipliers
    assert hint.cache is cache


def test_the_cache_is_shared_rather_than_copied(instance, solved):
    """A sequence of solves accumulates one cache, which is what makes the reuse pay.

    Consecutive problems have nearly the same working sets, so the factorization the last
    solve ended on is usually one update away from the next solve's first.
    """
    cache = Reuse()
    problems = frontier(instance, [1.0, 1.2, 1.4])
    hint = None
    for problem in problems:
        answer = solver.solve(problem, warm=hint)
        hint = warmstart.from_solution(answer, cache=cache)
    assert cache.factorizations < len(problems)


def test_the_multipliers_seed_the_curvature(instance, solved):
    """#23 turned multipliers from an output into an input, and this is that input.

    A cold solve starts the curvature's fixed point at zero, so its first direction is
    `H = rho*I`. A warm one starts from the last problem's duals and so begins with a
    Hessian that is already approximately right.
    """
    hint = warmstart.from_solution(solved)
    carried = warmstart.seed(instance.problem, hint)[2]
    assert carried is solved.multipliers
    assert np.abs(warmstart.seed(instance.problem, None)[2].w).max() == 0.0


# ----------------------------------------------------------------------------------
# A hint, never a commitment
# ----------------------------------------------------------------------------------


def test_a_warm_start_fits_the_next_problem_on_a_frontier(instance, solved):
    """Only `c` changes, so the row indices still mean what they meant."""
    hint = warmstart.from_solution(solved)
    for problem in frontier(instance, [1.5, 2.0]):
        assert hint.fits(problem)


def test_a_warm_start_from_a_different_shape_is_discarded(solved):
    """A working set names rows by index, so different rows make it a wrong answer.

    Not merely a poor start — the indices would mean something else entirely.
    """
    other = families.box(5, seed=0).problem
    hint = warmstart.from_solution(solved)
    assert not hint.fits(other)
    point, believed, carried, _ = warmstart.seed(other, hint)
    assert point is None
    assert believed is None
    assert np.abs(carried.w).max() == 0.0


def test_a_point_of_the_right_length_survives_a_shape_mismatch_elsewhere(instance, solved):
    """The four parts are checked apart, so one that fits is kept even when another does not.

    Refusing everything unless everything matches would make the feature useless precisely
    where a sequence is most interesting, which is where the problems differ.
    """
    problem = instance.problem
    narrowed = SOCP(
        c=problem.c,
        A=problem.A[:1],
        b=problem.b[:1],
        E=problem.E,
        d=problem.d,
        G=problem.G,
        h=problem.h,
        cone=problem.cone,
    )
    hint = warmstart.from_solution(solved)
    assert hint.point(narrowed) is not None, "the point is the right length"
    assert hint.set_for(narrowed) is None, "the working set names rows that are gone"


def test_multipliers_of_the_wrong_shape_are_replaced_by_zeros(instance, solved):
    """Handing on a half-valid object is how a shape error becomes a wrong answer later."""
    hint = warmstart.WarmStart(
        z=solved.z,
        working_set=solved.working_set,
        multipliers=Multipliers(y=np.zeros(2), nu=np.zeros(1), w=np.zeros(3)),
    )
    assert hint.duals_for(instance.problem) is None


def test_no_warm_start_at_all_is_a_cold_solve(instance):
    """The `None` case answered in the same place as every other, not at the call site."""
    point, believed, carried, cache = warmstart.seed(instance.problem, None)
    assert (point, believed, cache) == (None, None, None)
    assert np.abs(carried.w).max() == 0.0


def test_an_explicit_start_wins_over_the_warm_start(instance, solved):
    """A caller who passes both means the one they passed."""
    elsewhere = instance.witness
    answer = solver.solve(instance.problem, start=elsewhere, warm=warmstart.from_solution(solved))
    assert answer.status == "optimal"


def test_the_believed_working_set_is_only_used_with_the_point_it_came_from(instance, solved):
    """A believed working set travels with the point it came from, or not at all.

    Applying last problem's active set to a freshly constructed point would assert an
    activity the geometry there has not been asked about.
    """
    hint = warmstart.WarmStart(
        z=np.full(instance.problem.num_variables, 1e6),
        working_set=updates.set_cone_status(WorkingSet.empty(instance.problem), 0, ConeStatus.TANGENT),
    )
    answer = solver.solve(instance.problem, warm=hint)
    assert answer.status == "optimal", "an infeasible hint falls back to construction"


def test_a_warm_start_reports_what_it_carries(solved):
    """A frontier trace needs to see which parts survived."""
    rendered = str(warmstart.from_solution(solved, cache=Reuse()))
    assert "active row" in rendered
    assert "multipliers" in rendered
    assert "factorization" in rendered


# ----------------------------------------------------------------------------------
# The "Done when": iterations saved is strictly positive across the frontier
# ----------------------------------------------------------------------------------


def test_warm_starting_saves_iterations_across_the_frontier(instance):
    """Success Criterion 4, and the reason the project chose an active-set method.

    Solve the same twelve-point frontier twice, once cold and once with each solve warm
    started from the last. The answers must agree — a warm start that changed the answer
    would be a bug, not a speed-up — and the work must fall.
    """
    problems = frontier(instance, np.linspace(1.0, 4.0, 12))
    cold_iterations = warm_iterations = 0
    cache = Reuse()
    hint = None
    for problem in problems:
        cold = solver.solve(problem)
        hot = solver.solve(problem, warm=hint)
        assert hot.status == "optimal"
        assert hot.objective(problem) == pytest.approx(cold.objective(problem), abs=1e-7)
        cold_iterations += cold.metrics.iterations
        warm_iterations += hot.metrics.iterations
        hint = warmstart.from_solution(hot, cache=cache)
    assert warm_iterations < cold_iterations, (cold_iterations, warm_iterations)
    assert cold_iterations - warm_iterations > 0.25 * cold_iterations


def test_the_saving_is_reported_by_the_metric_that_promised_it(instance):
    """§11's seventh quantity, `iterations saved by warm starts`, on two consecutive points."""
    first, second = frontier(instance, [2.0, 2.1])
    seeded = solver.solve(first)
    cold = solver.solve(second)
    hot = solver.solve(second, warm=warmstart.from_solution(seeded, cache=Reuse()))
    assert iterations_saved(cold.metrics, hot.metrics) > 0


def test_a_warm_start_from_a_distant_problem_still_terminates(instance):
    """A wrong guess must cost iterations, not correctness.

    §7.2 and §7.4 correct a bad working set; they do not depend on a good one.
    """
    near, far = frontier(instance, [1.0, 50.0])
    seeded = solver.solve(near)
    answer = solver.solve(far, warm=warmstart.from_solution(seeded))
    assert answer.status == "optimal"
    assert answer.residuals.is_optimal()
    assert answer.objective(far) == pytest.approx(solver.solve(far).objective(far), abs=1e-7)


def test_the_recursion_stops_at_depth_one():
    """The elastic solve is called with `phase_one=False` and must refuse, not recurse.

    Tested here rather than in `test_cosa.py` because #30 moved the check: a caller's
    `start` is now refused on its own path, so `phase_one` is the only thing left guarding
    the recursion and it needs a test of its own.
    """
    from cosa.solver.initialization import NeedsPhaseOneError

    problem = families.turnover(6, seed=0).problem
    with pytest.raises(NeedsPhaseOneError):
        solver.solve(problem, phase_one=False)
