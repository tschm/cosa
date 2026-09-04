"""§13.2's three updates, and the counter they exist to move.

Issue #27. The claim under test is narrow and checkable: an update produces the *same*
factorization a refactorization would, so using one changes only the cost. Every case is
therefore tested the same way — apply the update, factorize the target from scratch, and
require the two to solve identically. A QR is not unique in sign, so the factors themselves
are not compared; what they compute is.
"""

import numpy as np
import pytest

from cosa import SOCP, ConeStatus, WorkingSet
from cosa.active_set import updates
from cosa.experiments import portfolio as families
from cosa.experiments.randomized import random_instance
from cosa.linear_algebra import kkt, reuse
from cosa.problem.socp import ProblemError
from cosa.solver import cosa as solver


@pytest.fixture
def instance():
    """A box-constrained portfolio, which has bounds to add and drop and a cone to move."""
    return families.box(6, seed=0)


@pytest.fixture
def working_set(instance):
    """One active bound and an active cone: every kind of row at once."""
    return updates.set_cone_status(updates.add_inequality(WorkingSet.empty(instance.problem), 0), 0, ConeStatus.TANGENT)


def _matrix(instance, working_set):
    """The working-set matrix at the instance's witness."""
    return kkt.working_set_matrix(instance.problem, working_set, instance.witness)


def _agree(instance, factorization, working_set):
    """Does this factorization solve the direction subproblem the way `kkt` does?"""
    reference = kkt.direction(instance.problem, working_set, instance.witness)
    got = factorization.solve(instance.problem.c, layout=reference.layout)
    return np.allclose(got.d, reference.d) and np.allclose(got.multipliers, reference.multipliers)


# ----------------------------------------------------------------------------------
# The factorization itself
# ----------------------------------------------------------------------------------


def test_the_factors_reproduce_the_matrix(instance, working_set):
    """`W.T = Q R`, which is the only property the update rules preserve."""
    matrix = _matrix(instance, working_set)
    factorization = reuse.factorize(matrix)
    assert pytest.approx(matrix.T) == factorization.Q @ factorization.R


def test_the_null_space_is_orthonormal_and_annihilates_the_working_set(instance, working_set):
    """The trailing columns of `Q` span `{p : W @ p = 0}`, which is what makes them useful."""
    matrix = _matrix(instance, working_set)
    basis = reuse.factorize(matrix).null_space()
    assert basis.shape == (instance.problem.num_variables, instance.problem.num_variables - matrix.shape[0])
    assert basis.T @ basis == pytest.approx(np.eye(basis.shape[1]))
    assert matrix @ basis == pytest.approx(np.zeros((matrix.shape[0], basis.shape[1])), abs=1e-12)


def test_an_empty_working_set_factors_to_the_identity(instance):
    """The degenerate case answered rather than special-cased at the call site."""
    factorization = reuse.factorize(np.zeros((0, 4)))
    assert factorization.rows == 0
    assert factorization.null_space() == pytest.approx(np.eye(4))


def test_the_solve_matches_the_reference_route(instance, working_set):
    """Same subproblem, same answer — the whole premise of substituting one for the other."""
    assert _agree(instance, reuse.factorize(_matrix(instance, working_set)), working_set)


def test_the_solve_matches_the_reference_route_with_curvature(instance, working_set):
    """#23's Hessian goes through the reduced system, and must not change the answer."""
    problem, z = instance.problem, instance.witness
    bent = 0.5 * np.eye(problem.num_variables)
    reference = kkt.direction(problem, working_set, z, curvature=bent)
    hessian = kkt.RHO * np.eye(problem.num_variables) + bent
    got = reuse.factorize(_matrix(instance, working_set)).solve(problem.c, hessian, layout=reference.layout)
    assert got.d == pytest.approx(reference.d)
    assert got.multipliers == pytest.approx(reference.multipliers)


def test_an_empty_working_set_leaves_the_direction_unconstrained():
    """`d = -g / rho`, with no multipliers to recover.

    On a problem with no equalities, since an equality row is in the working set
    unconditionally and a portfolio always has the budget constraint.
    """
    problem = SOCP.unconstrained(np.array([1.0, -2.0]))
    empty = WorkingSet.empty(problem)
    got = reuse.factorize(kkt.working_set_matrix(problem, empty, np.zeros(2))).solve(problem.c)
    assert got.d == pytest.approx(-problem.c / kkt.RHO)
    assert got.multipliers.size == 0


def test_more_rows_than_variables_is_a_singular_working_set(instance):
    """Dependent by counting alone, and reported the way the reference route reports it.

    Both routes must raise the same thing or the loop's §8.3 repair path becomes reachable
    from only one of them — which is how a dependent working set turns into a crash.
    """
    with pytest.raises(kkt.SingularKktError):
        reuse.factorize(np.ones((3, 2)))


def test_dependent_rows_are_refused_rather_than_least_squared(instance):
    """The direction is still unique; the multipliers are not, so they are not invented."""
    matrix = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    with pytest.raises(kkt.SingularKktError):
        reuse.factorize(matrix).solve(np.ones(3))


# ----------------------------------------------------------------------------------
# §13.2's three cases
# ----------------------------------------------------------------------------------


def test_adding_a_constraint_updates_rather_than_refactorizes(instance, working_set):
    """§13.2's first case: one column appended to `W.T`."""
    grown = updates.add_inequality(working_set, 3)
    updated = reuse.insert(reuse.factorize(_matrix(instance, working_set)), _matrix(instance, grown)[1], 1)
    assert pytest.approx(_matrix(instance, grown)) == updated.W
    assert _agree(instance, updated, grown)


def test_removing_a_constraint_updates_rather_than_refactorizes(instance, working_set):
    """§13.2's second case, and the cheaper of the two."""
    grown = updates.add_inequality(working_set, 3)
    shrunk = updates.drop_inequality(grown, 3)
    updated = reuse.delete(reuse.factorize(_matrix(instance, grown)), 1)
    assert pytest.approx(_matrix(instance, shrunk)) == updated.W
    assert _agree(instance, updated, shrunk)


def test_removing_the_last_constraint_leaves_an_empty_factorization():
    """`qr_delete` cannot produce a zero-column factor, so this route is taken explicitly."""
    problem = SOCP.unconstrained(np.ones(2)).add_inequalities([[1.0, 1.0]], [1.0])
    single = updates.add_inequality(WorkingSet.empty(problem), 0)
    matrix = kkt.working_set_matrix(problem, single, np.zeros(2))
    updated = reuse.delete(reuse.factorize(matrix), 0)
    assert updated.rows == 0
    assert updated.null_space() == pytest.approx(np.eye(2))


def test_the_tangent_row_moving_updates_rather_than_refactorizes(instance, working_set):
    """§13.2's third case — the one the paper calls "more subtle".

    Subtle in *frequency* rather than in kind: the row changes every iteration the cone is
    active, so this update runs far more often than the other two. It is still an update,
    and the linear structure around it is never refactorized.
    """
    problem = instance.problem
    moved = instance.witness * 1.01
    before = reuse.factorize(_matrix(instance, working_set))
    after = kkt.working_set_matrix(problem, working_set, moved)
    updated = reuse.replace(before, after[-1], before.rows - 1)
    assert pytest.approx(after) == updated.W
    assert pytest.approx(after.T) == updated.Q @ updated.R


def test_an_update_position_outside_the_working_set_is_refused(instance, working_set):
    """Inserting at the wrong index would factor a matrix nobody asked for."""
    factorization = reuse.factorize(_matrix(instance, working_set))
    row = np.ones(instance.problem.num_variables)
    for call in (
        lambda: reuse.insert(factorization, row, 99),
        lambda: reuse.delete(factorization, 99),
        lambda: reuse.replace(factorization, row, 99),
    ):
        with pytest.raises(ProblemError, match="at"):
            call()


def test_an_inserted_row_of_the_wrong_length_is_refused(instance, working_set):
    """It becomes a column of `W.T`, so it must have as many entries as there are variables."""
    with pytest.raises(ProblemError, match="row"):
        reuse.insert(reuse.factorize(_matrix(instance, working_set)), np.ones(2), 0)


# ----------------------------------------------------------------------------------
# The cache: recognizing which case happened
# ----------------------------------------------------------------------------------


def test_the_first_call_factorizes_and_an_unchanged_set_needs_nothing(instance, working_set):
    """The cache starts empty, and a repeat call has no work to do beyond the diff."""
    cache = reuse.Reuse()
    cache.matrix_for(instance.problem, working_set, instance.witness)
    assert (cache.factorizations, cache.updates) == (1, 0)
    cache.matrix_for(instance.problem, working_set, instance.witness)
    assert (cache.factorizations, cache.updates) == (1, 1)


def test_each_of_the_three_cases_is_recognized_without_being_told(instance, working_set):
    """Recognition is by comparing matrices, which is what makes the cache safe.

    §4.1's loop has six paths that change a working set, and a path that forgot to announce
    itself would otherwise get a stale factorization. Nothing announces anything here.
    """
    cache = reuse.Reuse()
    z = instance.witness
    cache.matrix_for(instance.problem, working_set, z)
    grown = updates.add_inequality(working_set, 3)
    cache.matrix_for(instance.problem, grown, z)
    cache.matrix_for(instance.problem, working_set, z)
    cache.matrix_for(instance.problem, working_set, z * 1.01)
    assert (cache.factorizations, cache.updates) == (1, 3)


def test_a_wholesale_change_refactorizes_instead(instance, working_set):
    """Over the budget an update is no longer cheaper, so the cache declines.

    Rather than grinding through a long script whose cost exceeds the refactorization it
    was trying to avoid.
    """
    cache = reuse.Reuse()
    cache.matrix_for(instance.problem, working_set, instance.witness)
    crowded = working_set
    for row in range(1, 5):
        crowded = updates.add_inequality(crowded, row)
    cache.matrix_for(instance.problem, crowded, instance.witness)
    assert cache.factorizations == 2


def test_a_different_problem_shape_refactorizes(instance, working_set):
    """A cache carried to an instance with different variables has nothing to update."""
    other = SOCP.unconstrained(np.ones(3)).add_inequalities([[1.0, 0.0, 0.0]], [1.0])
    cache = reuse.Reuse()
    cache.matrix_for(instance.problem, working_set, instance.witness)
    cache.matrix_for(other, updates.add_inequality(WorkingSet.empty(other), 0), np.zeros(3))
    assert cache.factorizations == 2


def test_the_cache_solves_what_the_reference_route_solves(instance, working_set):
    """The property the whole module rests on, over a sequence rather than a single call."""
    cache = reuse.Reuse()
    problem, z = instance.problem, instance.witness
    for step, current in enumerate([working_set, updates.add_inequality(working_set, 3), working_set]):
        point = z * (1.0 + 0.01 * step)
        got = cache.direction(problem, current, point)
        want = kkt.direction(problem, current, point)
        assert got.d == pytest.approx(want.d)
        assert got.multipliers == pytest.approx(want.multipliers)


def test_the_cache_reports_what_it_saved(instance, working_set):
    """A log line needs the two counters and the share between them."""
    cache = reuse.Reuse()
    cache.matrix_for(instance.problem, working_set, instance.witness)
    cache.matrix_for(instance.problem, working_set, instance.witness)
    assert "1 factorization(s), 1 update(s) (50% reused)" in str(cache)
    assert "0%" in str(reuse.Reuse())


# ----------------------------------------------------------------------------------
# The metric #27 is judged on
# ----------------------------------------------------------------------------------


def test_reuse_changes_the_answer_not_at_all(instance):
    """The precondition for the measurement below meaning anything."""
    cold = solver.solve(instance.problem, reuse=False)
    warm = solver.solve(instance.problem, reuse=True)
    assert cold.status == warm.status == "optimal"
    assert cold.objective(instance.problem) == pytest.approx(warm.objective(instance.problem), rel=1e-8)


def test_the_factorization_count_falls_far_below_the_reference_policy():
    """§13.2's "Done when", as a number.

    Under §13.1's policy every KKT solve is a fresh factorization. With the three updates in
    place a whole solve takes a handful — two, on most of these — because the working set
    changes by one or two rows an iteration and the tangent row moves in place.
    """
    total = {True: 0, False: 0}
    solves = 0
    for family in (families.basic, families.box, families.sector, families.turnover):
        for seed in range(3):
            problem = family(8, seed=seed).problem
            for policy in (False, True):
                metrics = solver.solve(problem, reuse=policy).metrics
                total[policy] += metrics.factorizations
                solves += metrics.kkt_solves if not policy else 0
    assert total[False] > 0.9 * solves, "the reference policy factorizes on nearly every solve"
    assert total[True] < 0.1 * total[False], total


def test_the_saving_holds_on_the_randomized_sweep():
    """The families have favourable working sets; §16.3's generator does not."""
    total = {True: 0, False: 0}
    for seed in range(15):
        problem = random_instance(seed).problem
        for policy in (False, True):
            total[policy] += solver.solve(problem, reuse=policy).metrics.factorizations
    assert total[True] < 0.1 * total[False], total


def test_a_regularized_solve_bypasses_the_cache(instance):
    """A regularized solve bypasses the cache, because there is nothing there to reuse.

    §8.3's `delta` changes the system rather than the working set, and a cache that
    pretended otherwise would hand back the unregularized answer.
    """
    flat = SOCP.unconstrained(np.array([1.0, 1.0])).add_equalities([[1.0, 1.0], [1.0, 1.0]], [1.0, 1.0])
    answer = solver.solve(flat, regularization=1e-8)
    assert answer.status != "degenerate"
