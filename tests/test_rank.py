"""§8.3's four answers to a degenerate working set, and which one applies when.

The executable half of issue #25, whose "done when" is that linearly dependent active
constraints are *detected and resolved* -- by regularization or by dependent-constraint
removal -- without solver failure on the nearly-redundant instances of #33.

§8.3 names four things and this file checks all four, because the issue warns that one of
them is easy to lose: dependent-constraint *removal* changes the working set rather than
the linear algebra, so it lives in `active_set.updates` and is tested here alongside the
QR that decides what it may remove.

The instances are #33's, which is the point of having scheduled #33 first: before it
existed there was nothing to demonstrate a fix against, and #12's refusal to guess was a
branch nobody had seen fire.
"""

import numpy as np
import pytest

from cosa import SOCP, ProblemError, SingularKktError, WorkingSet
from cosa.active_set import updates
from cosa.experiments import portfolio as families
from cosa.experiments import reference
from cosa.linear_algebra import kkt
from cosa.linear_algebra import rank as rk


def active_set_at(problem, z, tolerance=1e-6):
    """The working set the §7 rules make active at a point."""
    working_set = WorkingSet.empty(problem)
    for index in updates.activation_candidates(problem, z, working_set, tolerance=tolerance):
        working_set = updates.add_inequality(working_set, index)
    return updates.activate_cones(problem, z, working_set)


# ----------------------------------------------------------------------------------
# QR-based rank detection
# ----------------------------------------------------------------------------------


def test_an_independent_matrix_is_full_rank():
    """The ordinary case, answered without drama."""
    analysis = rk.analyse(np.eye(3))
    assert analysis.rank == 3
    assert not analysis.is_deficient
    assert analysis.dependent == ()
    assert analysis.independent == (0, 1, 2)


def test_a_duplicated_row_is_detected():
    """One row adds nothing, so the rank is one below the row count."""
    analysis = rk.analyse(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))
    assert analysis.rank == 2
    assert analysis.is_deficient
    assert len(analysis.dependent) == 1
    assert set(analysis.dependent) <= {0, 2}, "one of the two duplicates, not the independent row"


def test_the_pivot_order_names_which_rows_to_drop():
    """The reason for a pivoted QR rather than an SVD.

    Rank detection is not the hard part of §8.3; deciding *which* rows to remove is, and
    the pivot order answers it directly. An SVD gives a number and leaves the choice to be
    reconstructed.
    """
    analysis = rk.analyse(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    assert analysis.rank == 2
    assert len(analysis.order) == 3
    assert set(analysis.independent) | set(analysis.dependent) == {0, 1, 2}
    kept = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])[list(analysis.independent)]
    assert np.linalg.matrix_rank(kept) == 2, "the retained rows span the same space"


def test_the_smallest_pivot_measures_near_dependence():
    """A full-rank set can still be nearly dependent, and the number says how nearly."""
    close = rk.analyse(np.array([[1.0, 0.0], [1.0, 1e-9]]))
    apart = rk.analyse(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert close.rank == 2
    assert close.smallest < 1e-8
    assert apart.smallest == pytest.approx(1.0)


def test_the_tolerance_decides_where_nearly_becomes_actually():
    """Which is #25's central judgement, so it is a parameter rather than a constant."""
    matrix = np.array([[1.0, 0.0], [1.0, 1e-9]])
    assert rk.analyse(matrix).rank == 2
    assert rk.analyse(matrix, tolerance=1e-6).rank == 1


def test_an_empty_matrix_has_rank_zero():
    """A working set with no rows is not degenerate, it is empty."""
    analysis = rk.analyse(np.zeros((0, 3)))
    assert analysis.rank == 0
    assert not analysis.is_deficient


def test_the_analysis_reports_itself():
    """A diagnosis needs the rank and how close the retained rows came to failing."""
    rendered = str(rk.analyse(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])))
    assert "rank 2/3" in rendered
    assert "smallest pivot" in rendered


def test_an_empty_matrix_has_no_tolerance_to_speak_of():
    """Nothing to be relative to, so the threshold is zero rather than a guess."""
    assert rk.rank_tolerance(np.zeros((0, 3))) == 0.0


def test_the_null_space_rejects_a_non_matrix():
    """Same contract as the rank analysis, on the same kind of input."""
    with pytest.raises(ProblemError, match="expected a matrix"):
        rk.null_space_basis(np.zeros(3))


def test_a_non_matrix_is_rejected():
    """A vector is not a working-set matrix."""
    with pytest.raises(ProblemError, match="expected a matrix"):
        rk.analyse(np.zeros(3))


# ----------------------------------------------------------------------------------
# Null-space handling
# ----------------------------------------------------------------------------------


def test_the_null_space_basis_is_orthonormal_and_annihilates_the_rows():
    """`W @ Z = 0` and `Z.T @ Z = I` -- the definition, checked."""
    matrix = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    basis = rk.null_space_basis(matrix)
    assert basis.shape == (3, 1)
    np.testing.assert_allclose(matrix @ basis, 0.0, atol=1e-12)
    np.testing.assert_allclose(basis.T @ basis, np.eye(1), atol=1e-12)


def test_the_null_space_survives_a_dependent_working_set():
    """The route that is *defined* where the saddle-point solve is not.

    A rank-deficient `W` leaves the direction perfectly well determined -- a null space
    does not care how many redundant rows described it. Only the multipliers are lost.
    """
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    basis = rk.null_space_basis(matrix)
    assert basis.shape == (2, 0), "these rows span everything, so the null space is trivial"
    dependent = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(dependent @ rk.null_space_basis(dependent), 0.0, atol=1e-12)


def test_no_rows_means_the_whole_space():
    """An empty working set constrains no direction."""
    np.testing.assert_allclose(rk.null_space_basis(np.zeros((0, 3))), np.eye(3))


def test_the_null_space_direction_matches_the_kkt_solve():
    """Two routes to the same direction, which is what makes the fallback trustworthy.

    On a *non*-degenerate set both are defined, so they can be compared; the point of the
    null-space route is that it keeps working when the other stops.
    """
    problem = SOCP.unconstrained(np.array([1.0, -2.0, 0.5])).add_equalities([[1.0, 1.0, 1.0]], [1.0])
    working_set = WorkingSet.empty(problem)
    z = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    matrix = kkt.working_set_matrix(problem, working_set, z)
    basis = rk.null_space_basis(matrix)
    projected = -(basis @ (basis.T @ problem.c))
    np.testing.assert_allclose(kkt.direction(problem, working_set, z).d, projected, atol=1e-12)


# ----------------------------------------------------------------------------------
# Regularization -- and it is not rho
# ----------------------------------------------------------------------------------


def test_regularization_makes_a_dependent_system_solvable():
    """§8.3's third answer: perturb, and get an answer to a nearby question."""
    problem = families.degenerate_optimum(5, seed=0).problem
    z = reference.solve_reference(problem).z
    working_set = active_set_at(problem, z)
    with pytest.raises(SingularKktError):
        kkt.direction(problem, working_set, z)
    step = kkt.direction(problem, working_set, z, regularization=1e-8)
    assert np.all(np.isfinite(step.d))
    assert np.all(np.isfinite(step.multipliers))


def test_regularization_appears_only_in_the_lower_block():
    """`[[rho*I, W.T], [W, -delta*I]]` -- the primal block is untouched."""
    problem = families.box(4, seed=0).problem
    z = reference.solve_reference(problem).z
    working_set = active_set_at(problem, z)
    plain = kkt.assemble(problem, working_set, z)
    regularized = kkt.assemble(problem, working_set, z, regularization=1e-6)
    n, m = problem.num_variables, working_set.num_rows
    np.testing.assert_allclose(regularized.matrix[:n, :], plain.matrix[:n, :])
    np.testing.assert_allclose(regularized.matrix[n:, :n], plain.matrix[n:, :n])
    np.testing.assert_allclose(regularized.matrix[n:, n:], -1e-6 * np.eye(m))


def test_regularization_is_recorded_on_the_system():
    """Because a regularized answer is an answer to a nearby problem, and a consumer must know."""
    problem = families.box(4, seed=0).problem
    z = reference.solve_reference(problem).z
    working_set = active_set_at(problem, z)
    assert kkt.assemble(problem, working_set, z).regularization == 0.0
    assert kkt.assemble(problem, working_set, z, regularization=1e-6).regularization == 1e-6


def test_regularization_is_not_rho():
    """The confusion the plan itself warns against, kept apart by construction.

    `rho` perturbs nothing -- it makes a linear objective's direction well defined and
    scales it. `delta` perturbs the problem. They enter different blocks and a change in
    one is visible where a change in the other is not.
    """
    problem = families.box(4, seed=0).problem
    z = reference.solve_reference(problem).z
    working_set = active_set_at(problem, z)
    n = problem.num_variables
    by_rho = kkt.assemble(problem, working_set, z, rho=7.0)
    by_delta = kkt.assemble(problem, working_set, z, regularization=7.0)
    np.testing.assert_allclose(np.diag(by_rho.matrix[:n, :n]), 7.0)
    np.testing.assert_allclose(np.diag(by_delta.matrix[:n, :n]), 1.0)


@pytest.mark.parametrize("delta", [-1.0, np.nan])
def test_a_nonsensical_regularization_is_rejected(delta):
    """§8.3 takes `delta >= 0`; a negative one would flip the block's sign."""
    problem = families.box(4, seed=0).problem
    with pytest.raises(ProblemError, match="delta >= 0"):
        kkt.assemble(
            problem,
            WorkingSet.empty(problem),
            np.zeros(problem.num_variables),
            regularization=delta,
        )


# ----------------------------------------------------------------------------------
# Dependent-constraint removal: the one that changes active-set logic
# ----------------------------------------------------------------------------------


def test_a_dependent_row_is_dropped_from_the_working_set():
    """§8.3's fourth answer, and the one #25 warns is easy to lose among the linear algebra."""
    instance = families.degenerate_optimum(6, seed=0)
    problem = instance.problem
    z = reference.solve_reference(problem).z
    working_set = active_set_at(problem, z)

    analysis = rk.analyse(kkt.working_set_matrix(problem, working_set, z))
    assert analysis.is_deficient

    repaired, dropped = updates.drop_dependent_rows(problem, working_set, z)
    assert dropped, "a row was removed"
    assert repaired.num_rows == working_set.num_rows - len(dropped)
    assert not rk.analyse(kkt.working_set_matrix(problem, repaired, z)).is_deficient


def test_removal_makes_the_direction_solve_succeed():
    """Which is the point: the working set is repaired, and nothing is perturbed."""
    instance = families.degenerate_optimum(6, seed=0)
    problem = instance.problem
    z = reference.solve_reference(problem).z
    working_set = active_set_at(problem, z)
    with pytest.raises(SingularKktError):
        kkt.direction(problem, working_set, z)
    repaired, _ = updates.drop_dependent_rows(problem, working_set, z)
    assert np.all(np.isfinite(kkt.direction(problem, repaired, z).d))


def test_an_independent_working_set_is_left_alone():
    """No dependency, no removal -- and the set comes back unchanged rather than rebuilt."""
    instance = families.box(5, seed=0)
    z = reference.solve_reference(instance.problem).z
    working_set = active_set_at(instance.problem, z)
    repaired, dropped = updates.drop_dependent_rows(instance.problem, working_set, z)
    assert dropped == ()
    assert repaired == working_set


def test_only_inequalities_are_dropped():
    """§3.1 never lets an equality go, and a cone's rows are §7.4's decision, not a repair.

    Constructed so the dependency is *entirely* among equalities: two identical equality
    rows. Nothing may be dropped, and the caller is left to regularize -- which is why §8.3
    lists both answers.
    """
    problem = SOCP.unconstrained(np.array([1.0, 1.0])).add_equalities([[1.0, 1.0], [1.0, 1.0]], [1.0, 1.0])
    working_set = WorkingSet.empty(problem)
    assert rk.analyse(kkt.working_set_matrix(problem, working_set, np.zeros(2))).is_deficient
    repaired, dropped = updates.drop_dependent_rows(problem, working_set, np.zeros(2))
    assert dropped == ()
    assert repaired == working_set


# ----------------------------------------------------------------------------------
# Issue #25's "done when", on #33's instances
# ----------------------------------------------------------------------------------


def test_the_nearly_redundant_family_is_nearly_but_not_actually_dependent():
    """Issue #25's "done when", on the family built to trigger it.

    At the default gap the near-duplicate pair has *full rank* with a smallest singular
    direction of about `gap`, so it is not removed -- which is correct, and is the
    distinction #25 has to get right: a row that differs in the ninth digit is independent,
    and dropping it would change the problem.

    This test is also what caught the family being wrong. Its first version perturbed the
    duplicated row's own coefficient, which leaves the two rows *parallel* -- algebraically
    rank one for any gap, so the family tested exact dependence and its docstring said
    otherwise. The perturbation now tilts the row instead.
    """
    instance = families.nearly_redundant(6, gap=1e-9, seed=0)
    problem = instance.problem
    pair = rk.analyse(np.vstack([problem.A[0], problem.A[-1]]))
    assert pair.rank == 2, "1e-9 apart is still independent"
    assert pair.smallest == pytest.approx(1e-9, rel=1e-3), "but only just"

    z = reference.solve_reference(problem).z
    working_set = active_set_at(problem, z)
    analysis = rk.analyse(kkt.working_set_matrix(problem, working_set, z))
    assert 0 in working_set.inequalities
    assert not analysis.is_deficient, "so the solver is not asked to remove anything"
    assert analysis.smallest < 1e-8, "and yet the working set is nearly dependent"


def test_the_solver_does_not_fail_on_the_nearly_redundant_family():
    """The literal "done when": no solver failure on the nearly-redundant instances.

    The direction solve succeeds and the multipliers are finite, on an instance whose
    working set has a singular direction of `1e-9`. Before #25 the only defence was #12's
    rank test, which correctly declined to remove anything here -- and correctly refused
    the *exactly* dependent case, which is the next test.
    """
    instance = families.nearly_redundant(6, gap=1e-9, seed=0)
    problem = instance.problem
    z = reference.solve_reference(problem).z
    working_set = active_set_at(problem, z)
    step = kkt.direction(problem, working_set, z)
    assert np.all(np.isfinite(step.d))
    assert np.all(np.isfinite(step.multipliers))


def test_a_gap_below_the_threshold_is_treated_as_dependent():
    """The sweep #33's `gap` knob exists for: at some point near becomes actually.

    Which is the judgement #25 owns, and the reason the threshold is a parameter rather
    than a constant.
    """
    for gap, deficient in ((1e-9, False), (1e-18, True)):
        instance = families.nearly_redundant(6, gap=gap, seed=0)
        problem = instance.problem
        pair = rk.analyse(np.vstack([problem.A[0], problem.A[-1]]))
        assert pair.is_deficient == deficient, f"gap {gap:g}"
