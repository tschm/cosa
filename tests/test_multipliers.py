"""Recovering the problem's multipliers from the subproblem's, and testing their signs.

The executable half of issue #13, whose "done when" is that multipliers computed from the
KKT system satisfy Level 2 stationarity (§14.2, `paper.tex:1028`) on the analytical
problems. The analytical problems are the two golden instances of `test_socp.py`, solved
by hand under #9's sign convention before any of this code existed -- so their `y`, `nu`
and `w` are arithmetic that cannot have been fitted to the implementation. Reproducing
them exactly is the strongest statement this module can make, and it is the first section
below.

The second claim is the one the module docstring derives: dual feasibility for the cone is
`w in Q`, and that single condition produces a *scalar* sign test at a tangent factor and a
*cone membership* test at an apex factor. Both are tested, and the tangent case is tested
against its own reduction (`nu_cone <= 0`) so that the derivation and the code are checked
against each other rather than one being assumed.
"""

import numpy as np
import pytest

from cosa import (
    SIGN_CONVENTION,
    SOCP,
    ConeStatus,
    MeanStdForm,
    MeanStdPortfolio,
    Multipliers,
    ProblemError,
    WorkingSet,
)
from cosa.active_set import multipliers as mult
from cosa.active_set import updates
from cosa.geometry import soc, tangent
from cosa.linear_algebra import kkt

# The two instances of `test_socp.py`, with the multipliers derived there by hand. Repeated
# rather than imported so that this file states what it is checking against.
#
# BOUND: one asset, mu = 1, lam = 1/2, x <= 1. Optimum x = t = 1, y = 1/2, w = (1/2, -1/2).
# BUDGET: two assets, mu = (2, 1), lam = 1, sum(x) = 1, Sigma = I. Optimum x = (1, 0),
# t = 1, nu = 1, w = (1, -1, 0).
GOLDEN = {
    "bound": {
        "form": MeanStdForm(
            mu=np.array([1.0]),
            lam=0.5,
            A=np.array([[1.0]]),
            b=np.array([1.0]),
            E=np.zeros((0, 1)),
            d=np.zeros(0),
            L=np.array([[1.0]]),
        ),
        "z": np.array([1.0, 1.0]),
        "active": (0,),
        "y": np.array([0.5]),
        "nu": np.zeros(0),
        "w": np.array([0.5, -0.5]),
    },
    "budget": {
        "form": MeanStdForm(
            mu=np.array([2.0, 1.0]),
            lam=1.0,
            A=np.zeros((0, 2)),
            b=np.zeros(0),
            E=np.array([[1.0, 1.0]]),
            d=np.array([1.0]),
            L=np.eye(2),
        ),
        "z": np.array([1.0, 0.0, 1.0]),
        "active": (),
        "y": np.zeros(0),
        "nu": np.array([1.0]),
        "w": np.array([1.0, -1.0, 0.0]),
    },
}


@pytest.fixture(params=sorted(GOLDEN))
def golden(request):
    """One hand-solved instance, its optimum, and the working set active there."""
    case = GOLDEN[request.param]
    problem = case["form"].to_socp()
    working_set = WorkingSet.empty(problem)
    for index in case["active"]:
        working_set = updates.add_inequality(working_set, index)
    working_set = updates.activate_cones(problem, case["z"], working_set)
    return problem, working_set, case


def recover(problem, working_set, z):
    """Solve the direction subproblem and map its multipliers onto the problem's."""
    step = kkt.direction(problem, working_set, z)
    return step, mult.from_direction(problem, working_set, z, step)


# ----------------------------------------------------------------------------------
# Issue #13's "done when": the analytical problems, exactly
# ----------------------------------------------------------------------------------


def test_the_recovered_multipliers_are_the_hand_derived_ones(golden):
    """`y`, `nu` and `w` all match arithmetic that predates this module.

    Not "satisfy the same conditions" -- *are the same numbers*. The multipliers at these
    optima are unique, so there is one right answer and this is it.
    """
    problem, working_set, case = golden
    _, recovered = recover(problem, working_set, case["z"])
    np.testing.assert_allclose(recovered.y, case["y"], atol=1e-12)
    np.testing.assert_allclose(recovered.nu, case["nu"], atol=1e-12)
    np.testing.assert_allclose(recovered.w, case["w"], atol=1e-12)


def test_level_2_stationarity_holds_on_the_analytical_problems(golden):
    """§14.2 to the prescribed tolerance -- issue #13's "done when", literally."""
    problem, working_set, case = golden
    _, recovered = recover(problem, working_set, case["z"])
    assert recovered.stationarity_error(problem) == pytest.approx(0.0, abs=1e-12)
    assert recovered.satisfies_stationarity(problem)


def test_the_recovered_multipliers_are_dual_feasible(golden):
    """`y >= 0` and `w in Q`: the dual half of the KKT conditions at these optima."""
    problem, working_set, case = golden
    _, recovered = recover(problem, working_set, case["z"])
    assert recovered.inequality_violation() == 0.0
    assert mult.dual_cone_violation(problem, recovered) == (0.0,)
    assert mult.is_dual_feasible(problem, recovered)


def test_the_direction_vanishes_where_the_multipliers_are_meaningful(golden):
    """The map is exact when `d = 0`, and at these optima it is.

    Stated as its own test because it is the hypothesis of the whole derivation: away from
    a stationary point the recovered numbers are the subproblem's multipliers, and reading
    them as the problem's is an approximation.
    """
    problem, working_set, case = golden
    step, _ = recover(problem, working_set, case["z"])
    np.testing.assert_allclose(step.d, 0.0, atol=1e-12)


def test_the_cone_multiplier_head_is_lambda(golden):
    """The convention's signature, recovered rather than assumed.

    `test_socp.py` asserts this of the hand-derived `w`; here it comes out of the KKT
    solve, so the two agree about the one number that pins the sign convention.
    """
    problem, working_set, case = golden
    _, recovered = recover(problem, working_set, case["z"])
    assert recovered.w[0] == pytest.approx(case["form"].lam)


# ----------------------------------------------------------------------------------
# The map itself
# ----------------------------------------------------------------------------------


@pytest.fixture
def portfolio():
    """Three assets, a budget equality and two caps -- a problem with all three blocks."""
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
def state(portfolio):
    """A problem, a point with one cap binding, and the working set active there."""
    problem = portfolio.to_socp()
    z = portfolio.socp_point(np.array([0.5, 0.2, 0.3]))
    working_set = updates.activate_cones(problem, z, updates.add_inequality(WorkingSet.empty(problem), 0))
    return problem, working_set, z


def test_the_multipliers_are_full_length_and_zero_off_the_working_set(state):
    """An inactive row's `y` is zero, which is what complementarity requires."""
    problem, working_set, z = state
    _, recovered = recover(problem, working_set, z)
    assert recovered.y.shape == (problem.num_inequalities,)
    assert recovered.nu.shape == (problem.num_equalities,)
    assert recovered.w.shape == (problem.cone.dim,)
    assert recovered.y[1] == 0.0, "row 1 is inactive, so its multiplier is zero"


def test_the_active_inequality_multipliers_are_the_subproblem_s(state):
    """The block that matches term for term -- `y` on the active rows *is* `nu_k` there."""
    problem, working_set, z = state
    step, recovered = recover(problem, working_set, z)
    active, _, _ = step.layout.split(step.multipliers)
    np.testing.assert_allclose(recovered.y[list(working_set.inequalities)], active)


def test_the_tangent_block_is_minus_nu_times_the_covector(state):
    """`w_block = -nu_cone * (1, -u)`, the one block that does not match term for term."""
    problem, working_set, z = state
    step, recovered = recover(problem, working_set, z)
    _, _, cones = step.layout.split(step.multipliers)
    covector = tangent.tangent_covector(problem.cone_slack(z))
    np.testing.assert_allclose(recovered.w, -float(cones[0][0]) * covector, atol=1e-12)


def test_an_inactive_cone_gets_a_zero_multiplier(portfolio):
    """Complementarity again: a strictly interior slack forces `w = 0`."""
    problem = portfolio.to_socp()
    interior = portfolio.socp_point(np.array([0.5, 0.2, 0.3])) + np.array([0.0, 0.0, 0.0, 0.5])
    working_set = updates.activate_cones(problem, interior, WorkingSet.empty(problem))
    assert working_set.active_cones == ()
    _, recovered = recover(problem, working_set, interior)
    np.testing.assert_array_equal(recovered.w, np.zeros(problem.cone.dim))
    assert mult.dual_cone_violation(problem, recovered) == (0.0,)


def test_the_apex_block_is_minus_the_subproblem_multipliers():
    """`w_block = -nu_block`, with no covector in sight -- the whole block is the row set.

    On a rank-deficient covariance, because that is where a pinned apex block is solvable
    at all: with `L` invertible its rows plus an equality are dependent, which is what
    `test_kkt.py` records.
    """
    portfolio = MeanStdPortfolio.unconstrained(
        mu=np.array([0.10, 0.04, 0.06]), Sigma=np.ones((3, 3)), lam=1.0
    ).with_inequalities(np.vstack([np.eye(3), -np.eye(3)]), np.ones(6))
    problem = portfolio.to_socp()
    z = portfolio.socp_point(np.array([1.0, -1.0, 0.0]))
    working_set = updates.set_cone_status(WorkingSet.empty(problem), 0, ConeStatus.APEX)
    step, recovered = recover(problem, working_set, z)
    _, _, cones = step.layout.split(step.multipliers)
    np.testing.assert_allclose(recovered.w, -cones[0], atol=1e-14)


def test_the_map_rejects_a_working_set_of_another_shape(state):
    """A set carried over from a differently shaped problem is a bug, caught on arrival."""
    problem, working_set, z = state
    other = WorkingSet(num_inequalities=9, num_equalities=1, cone=problem.cone, cone_status=(ConeStatus.INACTIVE,))
    step = kkt.direction(problem, working_set, z)
    with pytest.raises(ProblemError, match="different instances"):
        mult.from_direction(problem, other, z, step)


# ----------------------------------------------------------------------------------
# One dual feasibility test, two shapes
# ----------------------------------------------------------------------------------


def test_the_tangent_dual_test_reduces_to_a_scalar_sign(state):
    """`w in Q` at a tangent factor is exactly `nu_cone <= 0`.

    The derivation checked against the code. `w = -nu_cone * (1, -u)` and `(1, -u)` is on
    the boundary of `Q`, so membership survives only for a non-negative multiple -- which
    makes the cone's dual condition a scalar sign test, indistinguishable in cost from an
    inequality's.
    """
    problem, working_set, z = state
    step, recovered = recover(problem, working_set, z)
    _, _, cones = step.layout.split(step.multipliers)
    nu_cone = float(cones[0][0])
    violation = mult.dual_cone_violation(problem, recovered)[0]
    assert (violation == 0.0) == (nu_cone <= 1e-12)
    assert soc.is_member(recovered.w) == (nu_cone <= 1e-12)


def test_a_wrong_signed_cone_multiplier_is_reported_as_a_violation(portfolio):
    """Flip the sign and `w` leaves `Q` -- the test has teeth in both directions."""
    problem = portfolio.to_socp()
    z = portfolio.socp_point(np.array([0.5, 0.2, 0.3]))
    covector = tangent.tangent_covector(problem.cone_slack(z))
    good = Multipliers(y=np.zeros(2), nu=np.zeros(1), w=2.0 * covector)
    bad = Multipliers(y=np.zeros(2), nu=np.zeros(1), w=-2.0 * covector)
    assert mult.dual_cone_violation(problem, good) == (0.0,)
    assert mult.dual_cone_violation(problem, bad)[0] > 0.0


def test_the_apex_dual_test_is_a_cone_membership(portfolio):
    """At the apex it does not reduce -- the whole block has to lie in `Q`.

    §8.1's "normal-cone conditions" made concrete: the normal cone of a self-dual cone at
    its apex is `-Q`, so `w in Q` stops being a sign and becomes a membership. A block
    whose head is large enough passes; the same block with a smaller head does not, and no
    single scalar distinguishes them.
    """
    problem = portfolio.to_socp()
    inside = Multipliers(y=np.zeros(2), nu=np.zeros(1), w=np.array([1.0, 0.3, 0.2, 0.1]))
    outside = Multipliers(y=np.zeros(2), nu=np.zeros(1), w=np.array([0.1, 0.3, 0.2, 0.1]))
    assert mult.dual_cone_violation(problem, inside) == (0.0,)
    assert mult.dual_cone_violation(problem, outside)[0] > 0.0


def test_dual_feasibility_fails_on_a_wrong_signed_inequality():
    """`is_dual_feasible` is a conjunction, so either half can sink it."""
    problem = SOCP.unconstrained(np.array([1.0, 1.0])).add_inequalities([[1.0, 0.0]], [1.0])
    assert not mult.is_dual_feasible(problem, Multipliers(y=np.array([-1.0]), nu=np.zeros(0), w=np.zeros(0)))
    assert mult.is_dual_feasible(problem, Multipliers(y=np.array([1.0]), nu=np.zeros(0), w=np.zeros(0)))


def test_the_inequality_sign_is_read_from_the_convention():
    """`y >= 0` is a consequence of SIGN_CONVENTION, so the test reads it rather than assuming."""
    wrong = Multipliers(y=np.array([-1.0 * SIGN_CONVENTION.inequality]), nu=np.zeros(0), w=np.zeros(0))
    right = Multipliers(y=np.array([1.0 * SIGN_CONVENTION.inequality]), nu=np.zeros(0), w=np.zeros(0))
    assert wrong.inequality_violation() > 0.0
    assert right.inequality_violation() == 0.0


def test_a_rounding_level_sign_violation_is_below_the_tolerance():
    """Noise is not a reason to call the multipliers dual infeasible."""
    noisy = Multipliers(y=np.array([-1e-14]), nu=np.zeros(0), w=np.zeros(0))
    assert noisy.inequality_violation(tolerance=1e-12) == 0.0
    assert noisy.inequality_violation(tolerance=1e-16) > 0.0


# ----------------------------------------------------------------------------------
# The removal rule is reused, not reimplemented
# ----------------------------------------------------------------------------------


def test_the_recovered_y_drives_the_existing_removal_rule(state):
    """#13 produces a `y`; §7.2's rule already knows what to do with one.

    Composition rather than duplication: `updates.removal_candidate` is the classical
    most-violating rule and this module has no second opinion about it. A wrong-signed
    multiplier planted here is picked up there.
    """
    problem, working_set, z = state
    _, recovered = recover(problem, working_set, z)
    assert updates.removal_candidate(working_set, recovered.y) is None

    planted = Multipliers(y=np.array([-0.5, 0.0]), nu=recovered.nu, w=recovered.w)
    assert updates.removal_candidate(working_set, planted.y) == 0


# ----------------------------------------------------------------------------------
# The stationarity error itself
# ----------------------------------------------------------------------------------


def test_the_stationarity_error_is_relative_to_the_objective():
    """So a problem in percent and one in basis points are held to the same accuracy."""
    small = SOCP.unconstrained(np.array([1.0, 1.0]))
    large = SOCP.unconstrained(np.array([1e6, 1e6]))
    off_by_one = Multipliers(y=np.zeros(0), nu=np.zeros(0), w=np.zeros(0))
    assert off_by_one.stationarity_error(small) == pytest.approx(1.0)
    assert off_by_one.stationarity_error(large) == pytest.approx(1.0)


def test_the_stationarity_residual_delegates_to_the_problem(state):
    """One definitional home for the signs, and this is not it."""
    problem, working_set, z = state
    _, recovered = recover(problem, working_set, z)
    np.testing.assert_array_equal(
        recovered.stationarity_residual(problem),
        problem.stationarity_residual(recovered.y, recovered.nu, recovered.w),
    )


def test_multipliers_describe_themselves(state):
    """A log line or a failure message needs the shapes and the extremes."""
    problem, working_set, z = state
    _, recovered = recover(problem, working_set, z)
    rendered = str(recovered)
    assert "y[2]" in rendered
    assert "nu[1]" in rendered
    assert "w[4]" in rendered


def test_multipliers_coerce_their_blocks():
    """Lists in, arrays out -- the same contract every block in this package has."""
    recovered = Multipliers(y=[1.0, 2.0], nu=[], w=[3.0, 1.0])
    assert recovered.y.dtype == np.float64
    np.testing.assert_array_equal(recovered.y, [1.0, 2.0])


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_multipliers_reject_a_non_finite_entry(bad):
    """A NaN multiplier would make every sign test answer False, which is not an answer."""
    with pytest.raises(ProblemError, match="must be finite"):
        Multipliers(y=np.array([bad]), nu=np.zeros(0), w=np.zeros(0))
