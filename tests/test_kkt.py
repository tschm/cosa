"""The direction subproblem: assembled, solved, and checked against what it claims.

The executable half of issue #12. The system of §4.3 is small enough that its solution can
be characterized completely, so these tests do not merely check that it runs -- they pin
every claim the module makes:

* `W @ d = 0` and `rho * d + W.T @ nu = -g`: the two block rows, which is what "solves the
  system" means.
* `d` is the projection of `-g / rho` onto the null space of `W`, computed independently
  here from an SVD. The KKT solve and the projection are different algorithms for the same
  vector, so agreement is evidence rather than tautology.
* `g.T @ d = -rho * ||d||^2`, so the direction is always a descent direction. The one
  guarantee eq. (4) has to make.
* `nu` is exactly rho-invariant and `d` scales as `1 / rho`. Stated in the module docstring
  as the reason `rho` is not a tuning parameter, and asserted here so it stays true.
* the conic row is eq. (3): the slack direction `G @ d` has zero tangent residual. That is
  the sentence "the SOC enters the working set like a linear constraint" made checkable.

Issue #12's "done when" adds that this stands as the reference M7 is measured against, so
the last section pins the two properties a reference has to have: one factorization per
solve, and a diagnosable failure rather than a plausible wrong answer when the working set
is degenerate.
"""

import numpy as np
import pytest

from cosa import (
    SOCP,
    ConeStatus,
    Direction,
    MeanStdPortfolio,
    ProblemError,
    RowLayout,
    SingularKktError,
    WorkingSet,
)
from cosa.active_set import updates
from cosa.geometry import tangent
from cosa.linear_algebra import kkt


@pytest.fixture
def portfolio():
    """Three assets, a budget equality and two weight caps."""
    return MeanStdPortfolio(
        mu=np.array([0.10, 0.04, 0.06]),
        Sigma=np.array([[0.04, 0.01, 0.00], [0.01, 0.09, 0.02], [0.00, 0.02, 0.16]]),
        lam=2.0,
        A=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        b=np.array([0.5, 0.5]),
        E=np.array([[1.0, 1.0, 1.0]]),
        d=np.array([1.0]),
    )


@pytest.fixture
def problem(portfolio):
    """The portfolio as the SOCP the system is assembled over."""
    return portfolio.to_socp()


@pytest.fixture
def point(portfolio):
    """A point whose first cap binds and whose conic slack is on the boundary."""
    return portfolio.socp_point(np.array([0.5, 0.2, 0.3]))


@pytest.fixture
def active(problem, point):
    """A working set with one inequality, the equality, and the cone tangent."""
    working_set = updates.add_inequality(WorkingSet.empty(problem), 0)
    return updates.activate_cones(problem, point, working_set)


def null_space_projection(matrix, vector):
    """Project ``vector`` onto the null space of ``matrix``, independently of the KKT solve.

    Via the SVD, which is a different algorithm from the saddle-point solve, so agreement
    between the two is a check and not a restatement.
    """
    if matrix.shape[0] == 0:
        return vector
    _, singular, right = np.linalg.svd(matrix)
    rank = int((singular > 1e-12 * max(1.0, singular.max())).sum())
    basis = right[rank:].T
    return basis @ (basis.T @ vector)


# ----------------------------------------------------------------------------------
# The published row order
# ----------------------------------------------------------------------------------


def test_the_row_order_is_inequalities_then_equalities_then_cones(problem, active):
    """The order #13 is entitled to rely on, stated by the layout rather than implied."""
    layout = RowLayout.for_working_set(active)
    assert layout.inequalities == (0,)
    assert layout.equalities == (0,)
    assert layout.inequality_rows == slice(0, 1)
    assert layout.equality_rows == slice(1, 2)
    assert [rows.rows for rows in layout.cones] == [slice(2, 3)]
    assert [rows.status for rows in layout.cones] == [ConeStatus.TANGENT]


def test_the_layout_agrees_with_the_working_set_on_the_row_count(problem, active):
    """Two independent counts of the same thing, so a drift between them is caught."""
    assert RowLayout.for_working_set(active).num_rows == active.num_rows


def test_an_inactive_cone_contributes_no_layout_entry(problem):
    """Only active factors get rows, so `cones` is not indexed by factor."""
    layout = RowLayout.for_working_set(WorkingSet.empty(problem))
    assert layout.cones == ()
    assert layout.num_rows == problem.num_equalities


def test_an_apex_cone_claims_its_whole_block(problem):
    """The exact-membership treatment of §8.1 is `cone.dim` rows, not one."""
    working_set = updates.set_cone_status(WorkingSet.empty(problem), 0, ConeStatus.APEX)
    (rows,) = RowLayout.for_working_set(working_set).cones
    assert rows.rows.stop - rows.rows.start == problem.cone.cones[0].dim
    assert rows.status is ConeStatus.APEX


def test_the_layout_splits_a_multiplier_vector_back_apart(problem, active):
    """The inverse mapping, which is how #13 gets from `nu_k` to `y`, `nu` and `w`."""
    layout = RowLayout.for_working_set(active)
    nu = np.array([1.0, 2.0, 3.0])
    inequality, equality, cones = layout.split(nu)
    np.testing.assert_array_equal(inequality, [1.0])
    np.testing.assert_array_equal(equality, [2.0])
    np.testing.assert_array_equal(cones[0], [3.0])


def test_the_split_rejects_a_vector_of_the_wrong_length(problem, active):
    """The length is the row count; a mismatch means the caller has the wrong layout."""
    with pytest.raises(ProblemError, match="expected 3 entries"):
        RowLayout.for_working_set(active).split(np.zeros(5))


# ----------------------------------------------------------------------------------
# W_k, row by row
# ----------------------------------------------------------------------------------


def test_the_working_set_matrix_holds_the_three_kinds_of_row(problem, active, point):
    """Active inequality rows of A, every row of E, then the cone's tangent row."""
    matrix = kkt.working_set_matrix(problem, active, point)
    assert matrix.shape == (3, problem.num_variables)
    np.testing.assert_allclose(matrix[0], problem.A[0])
    np.testing.assert_allclose(matrix[1], problem.E[0])
    np.testing.assert_allclose(matrix[2], tangent.tangent_row(problem.cone_slack(point), problem.G))


def test_an_apex_status_pins_the_whole_conic_block(problem, point):
    """Its rows are G's block itself, which holds the slack at the apex exactly."""
    working_set = updates.set_cone_status(WorkingSet.empty(problem), 0, ConeStatus.APEX)
    matrix = kkt.working_set_matrix(problem, working_set, point)
    np.testing.assert_allclose(matrix[problem.num_equalities :], problem.G)


def test_an_empty_working_set_gives_no_rows():
    """An unconstrained instance has an empty W, and the arithmetic still works out."""
    problem = SOCP.unconstrained(np.array([1.0, 2.0]))
    matrix = kkt.working_set_matrix(problem, WorkingSet.empty(problem), np.zeros(2))
    assert matrix.shape == (0, 2)


def test_a_tangent_status_at_the_apex_refuses_to_assemble(portfolio):
    """The working set believes something the geometry denies, and #17's guard says so.

    Reached honestly: a singular covariance makes the apex reachable at a nonzero
    portfolio, so this is a state the algorithm can actually be in -- and #24, not a
    hyperplane, is what should handle it.
    """
    singular = MeanStdPortfolio.unconstrained(mu=np.ones(3), Sigma=np.ones((3, 3)), lam=1.0)
    problem = singular.to_socp()
    z = singular.socp_point(np.array([1.0, -1.0, 0.0]))
    working_set = updates.set_cone_status(WorkingSet.empty(problem), 0, ConeStatus.TANGENT)
    with pytest.raises(tangent.ApexError):
        kkt.working_set_matrix(problem, working_set, z)


def test_a_working_set_of_the_wrong_shape_is_rejected(problem, point):
    """The set carries a shape so it can be reused; that is worth checking on arrival."""
    other = WorkingSet(num_inequalities=9, num_equalities=1, cone=problem.cone, cone_status=(ConeStatus.INACTIVE,))
    with pytest.raises(ProblemError, match="the working set is over 9 inequalities"):
        kkt.working_set_matrix(problem, other, point)


# ----------------------------------------------------------------------------------
# The assembled system
# ----------------------------------------------------------------------------------


def test_the_matrix_has_the_block_structure_of_section_four_three(problem, active, point):
    """[[rho*I, W.T], [W, 0]] and a right-hand side of (-g, 0), exactly as printed."""
    system = kkt.assemble(problem, active, point, rho=3.0)
    n, m = problem.num_variables, active.num_rows
    np.testing.assert_allclose(system.matrix[:n, :n], 3.0 * np.eye(n))
    np.testing.assert_allclose(system.matrix[:n, n:], system.W.T)
    np.testing.assert_allclose(system.matrix[n:, :n], system.W)
    np.testing.assert_allclose(system.matrix[n:, n:], np.zeros((m, m)))
    np.testing.assert_allclose(system.rhs, np.concatenate([-problem.c, np.zeros(m)]))


def test_the_matrix_is_symmetric(problem, active, point):
    """A saddle-point matrix is symmetric indefinite, and the assembly must not break that."""
    system = kkt.assemble(problem, active, point)
    np.testing.assert_allclose(system.matrix, system.matrix.T)


def test_the_gradient_is_the_objective_vector(problem, active, point):
    """`g` is `c`, because `c.T @ z` is linear -- for eq. (7), the plan's `(-mu, lam)`."""
    system = kkt.assemble(problem, active, point)
    np.testing.assert_allclose(system.gradient, problem.c)


@pytest.mark.parametrize("rho", [0.0, -1.0, np.inf, np.nan])
def test_a_non_positive_rho_is_rejected(problem, active, point, rho):
    """§4.2 takes rho > 0; at rho = 0 the direction is not defined at all."""
    with pytest.raises(ProblemError, match="rho > 0"):
        kkt.assemble(problem, active, point, rho=rho)


# ----------------------------------------------------------------------------------
# The solution, characterized completely
# ----------------------------------------------------------------------------------


def test_the_direction_satisfies_the_working_set_equations(problem, active, point):
    """`W @ d = 0`: the lower block row, and the whole purpose of the working set."""
    system = kkt.assemble(problem, active, point)
    solution = kkt.solve(system)
    np.testing.assert_allclose(system.W @ solution.d, 0.0, atol=1e-12)


def test_the_direction_satisfies_the_stationarity_row(problem, active, point):
    """`rho * d + W.T @ nu = -g`: the upper block row."""
    system = kkt.assemble(problem, active, point, rho=2.5)
    solution = kkt.solve(system)
    residual = system.rho * solution.d + system.W.T @ solution.multipliers + system.gradient
    np.testing.assert_allclose(residual, 0.0, atol=1e-12)


def test_the_direction_is_the_projected_negative_gradient(problem, active, point):
    """`d = -P g / rho` with `P` the null-space projector, computed here from an SVD.

    Two algorithms, one vector. This is the test that would catch a sign error, a
    transposed block or a row assembled into the wrong place, none of which the two block
    rows above would notice on their own.
    """
    system = kkt.assemble(problem, active, point, rho=1.5)
    solution = kkt.solve(system)
    expected = -null_space_projection(system.W, problem.c) / 1.5
    np.testing.assert_allclose(solution.d, expected, atol=1e-12)


def test_the_direction_is_a_descent_direction(problem, active, point):
    """`g.T @ d = -rho * ||d||^2 <= 0`, the one guarantee eq. (4) has to make."""
    system = kkt.assemble(problem, active, point, rho=0.75)
    solution = kkt.solve(system)
    assert solution.directional_derivative(system.gradient) == pytest.approx(-0.75 * float(solution.d @ solution.d))
    assert solution.directional_derivative(system.gradient) < 0.0


def test_an_empty_working_set_gives_the_steepest_descent_direction():
    """With nothing active, `d = -g / rho` and there is nothing to project onto."""
    problem = SOCP.unconstrained(np.array([1.0, -2.0]))
    solution = kkt.direction(problem, WorkingSet.empty(problem), np.zeros(2), rho=4.0)
    np.testing.assert_allclose(solution.d, -problem.c / 4.0)
    assert solution.multipliers.shape == (0,)


def test_a_fully_determined_working_set_gives_the_zero_direction():
    """`n` independent rows leave no room to move, which is what stationarity looks like."""
    problem = SOCP.unconstrained(np.array([1.0, -2.0])).add_equalities(np.eye(2), np.zeros(2))
    solution = kkt.direction(problem, WorkingSet.empty(problem), np.zeros(2))
    np.testing.assert_allclose(solution.d, 0.0, atol=1e-14)
    assert solution.directional_derivative(problem.c) == pytest.approx(0.0)


# ----------------------------------------------------------------------------------
# rho is a well-posedness device, not a knob
# ----------------------------------------------------------------------------------


def test_the_multipliers_are_independent_of_rho(problem, active, point):
    """`W @ W.T @ nu = -W @ g` has no rho in it, so neither do the multipliers.

    Which is what makes it safe for #13 to test their signs without knowing what rho the
    direction was computed at.
    """
    baseline = kkt.direction(problem, active, point, rho=1.0)
    for rho in (0.01, 7.0, 1e4):
        np.testing.assert_allclose(
            kkt.direction(problem, active, point, rho=rho).multipliers,
            baseline.multipliers,
            atol=1e-10,
        )


def test_the_direction_scales_as_one_over_rho(problem, active, point):
    """So rho sets a length, and the ratio test normalizes lengths away."""
    baseline = kkt.direction(problem, active, point, rho=1.0)
    for rho in (0.01, 7.0, 1e4):
        scaled = kkt.direction(problem, active, point, rho=rho)
        np.testing.assert_allclose(rho * scaled.d, baseline.d, atol=1e-10)
        assert scaled.rho == rho


def test_the_direction_records_the_rho_it_used(problem, active, point):
    """Because two directions at different rho are only comparable after rescaling."""
    assert kkt.direction(problem, active, point, rho=2.0).rho == 2.0
    assert kkt.direction(problem, active, point).rho == kkt.RHO


# ----------------------------------------------------------------------------------
# The conic row behaves like an active constraint
# ----------------------------------------------------------------------------------


def test_the_direction_keeps_the_active_cone_on_its_boundary(problem, active, point):
    """The slack direction has zero tangent residual: eq. (3), satisfied by the solve.

    The claim that a second-order cone can join a working set at all. Nothing in the KKT
    solve knows about cones -- it sees one more row -- so this is the row doing its job.
    """
    solution = kkt.direction(problem, active, point)
    slack = problem.cone_slack(point)
    assert tangent.tangent_residual(slack, problem.G @ solution.d) == pytest.approx(0.0, abs=1e-12)


def test_an_inactive_cone_does_not_constrain_the_direction(problem, portfolio):
    """A strictly interior slack leaves the cone out of W, so the direction may enter it."""
    interior = portfolio.socp_point(np.array([0.5, 0.2, 0.3])) + np.array([0.0, 0.0, 0.0, 0.5])
    working_set = updates.activate_cones(problem, interior, WorkingSet.empty(problem))
    assert working_set.active_cones == ()
    solution = kkt.direction(problem, working_set, interior)
    assert tangent.tangent_residual(problem.cone_slack(interior), problem.G @ solution.d) != 0.0


def test_an_apex_cone_holds_the_slack_direction_at_zero():
    """`G @ d = 0` on the pinned block, so a step keeps the slack at the apex exactly.

    On a *rank-deficient* covariance, which is the only place a pinned apex block leaves
    anything to solve for. With `L` invertible the block's `1 + n` rows already force
    `d = 0` on their own, and one equality on top makes the system dependent -- see
    `working_set_matrix`. Rank deficiency is what makes the apex reachable and what leaves
    the pinned direction room to move, so it is the honest setting for this test.
    """
    portfolio = MeanStdPortfolio.unconstrained(
        mu=np.array([0.10, 0.04, 0.06]), Sigma=np.ones((3, 3)), lam=1.0
    ).with_inequalities(np.vstack([np.eye(3), -np.eye(3)]), np.ones(6))
    problem = portfolio.to_socp()
    z = portfolio.socp_point(np.array([1.0, -1.0, 0.0]))
    working_set = updates.set_cone_status(WorkingSet.empty(problem), 0, ConeStatus.APEX)
    solution = kkt.direction(problem, working_set, z)
    np.testing.assert_allclose(problem.G @ solution.d, 0.0, atol=1e-12)
    assert np.linalg.norm(solution.d) > 1e-6, "and the direction is not merely zero"


def test_a_pinned_full_rank_apex_block_over_determines_the_direction(problem, point):
    """Its `1 + n` rows plus an equality are dependent, and the solve says so.

    Recorded as a test because the arithmetic is the reason the previous test needs a
    rank-deficient instance, and because it is the shape of dependency #25 will meet: not a
    duplicated constraint, but a block that is individually fine and collectively too much.
    """
    working_set = updates.set_cone_status(WorkingSet.empty(problem), 0, ConeStatus.APEX)
    assert working_set.num_rows > problem.num_variables
    with pytest.raises(SingularKktError, match="linearly dependent"):
        kkt.direction(problem, working_set, point)


def test_the_conic_row_costs_exactly_one_row(problem, active, point):
    """An active cone is no more expensive than an active linear constraint.

    The efficiency claim behind the tangent representation, and it is arithmetic: the
    system grows by one when the cone activates.
    """
    inactive = updates.add_inequality(WorkingSet.empty(problem), 0)
    without = kkt.assemble(problem, inactive, point)
    with_cone = kkt.assemble(problem, active, point)
    assert with_cone.num_rows == without.num_rows + 1


# ----------------------------------------------------------------------------------
# What a reference implementation has to be
# ----------------------------------------------------------------------------------


def test_a_degenerate_working_set_is_diagnosed_not_guessed(problem, point):
    """Two copies of the same active row make `nu` indeterminate, and the solve says so.

    Not a wrong answer with a plausible shape: §13.1 wants "a reliable reference", and a
    reference that silently returns one of infinitely many multiplier vectors is not one.
    #25 is the issue that makes this survivable.
    """
    duplicated = problem.add_inequalities(problem.A[0:1], problem.b[0:1])
    working_set = WorkingSet.empty(duplicated)
    working_set = updates.add_inequality(working_set, 0)
    working_set = updates.add_inequality(working_set, duplicated.num_inequalities - 1)
    with pytest.raises(SingularKktError, match="linearly dependent"):
        kkt.direction(duplicated, working_set, point)


def test_the_singular_error_names_the_issue_that_fixes_it(problem, point):
    """A diagnosable stop is only diagnosable if the message says where to go next."""
    duplicated = problem.add_equalities(problem.E[0:1], problem.d[0:1])
    with pytest.raises(SingularKktError, match="#25"):
        kkt.direction(duplicated, WorkingSet.empty(duplicated), point)


def test_solving_twice_gives_the_same_answer(problem, active, point):
    """`solve` holds no state, which is what "refactorize every iteration" requires.

    §13.1 asks the first implementation to refactor every time, so that the factorization
    count has a reference value for #27 to beat. A cache here would quietly destroy the
    baseline it exists to be.
    """
    system = kkt.assemble(problem, active, point)
    first, second = kkt.solve(system), kkt.solve(system)
    np.testing.assert_array_equal(first.d, second.d)
    np.testing.assert_array_equal(first.multipliers, second.multipliers)
    assert first is not second


def test_the_direction_is_a_plain_value(problem, active, point):
    """Frozen, so an iteration can keep the previous direction without defending it."""
    solution = kkt.direction(problem, active, point)
    assert isinstance(solution, Direction)
    with pytest.raises(AttributeError):
        solution.d = np.zeros(4)


def test_the_directional_derivative_checks_its_gradient(problem, active, point):
    """A gradient of the wrong length is a bug, not something to broadcast."""
    solution = kkt.direction(problem, active, point)
    with pytest.raises(ProblemError, match="expected 4 entries"):
        solution.directional_derivative(np.zeros(2))
