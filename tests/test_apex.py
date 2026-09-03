"""The apex branch: exact membership on the direction, normal cone on the multiplier.

The executable half of issue #24, whose "done when" is that at `Lx = 0` the direction comes
from exact SOC membership and normal-cone conditions rather than a tangent hyperplane,
verified on a rank-deficient `Sigma` instance. The rank-deficient instance is the first
fixture below and is not contrived: #10's factorization produces a short `L` for any
singular covariance, and a singular covariance is what §12.4's "highly correlated assets"
family is made of.

Two things get tested harder than the rest.

**The geometry claim.** The module says the tangent cone of `Q` at the apex *is* `Q`, and
at a nonzero boundary point is a half-space. Both are checked numerically, by stepping,
because that pair of facts is the entire justification for the branch existing -- if the
tangent cone at the apex were a half-space after all, eq. (3) would suffice and #24 would
be unnecessary.

**All three outcomes are reachable.** A branch with an unreachable arm is a branch nobody
has tested, so each of held, released and blocked is exercised. Reaching *released* takes a
general SOCP rather than a portfolio, and that is a finding rather than a testing
convenience: on eq. (7) the released direction is infeasible by arithmetic, which
`test_eq_seven_can_never_release_the_apex` proves rather than asserts.
"""

import numpy as np
import pytest

from cosa import SOCP, ConeProduct, ConeStatus, MeanStdPortfolio, ProblemError, WorkingSet
from cosa.active_set import updates
from cosa.geometry import soc
from cosa.linear_algebra import kkt
from cosa.solver import apex


@pytest.fixture
def rank_deficient():
    """Three assets on a rank-1 covariance, boxed so the apex is reachable at `x != 0`.

    `Sigma = 1 1.T` has `L` proportional to a single row of ones, so `L @ x = 0` on the
    hyperplane `sum(x) = 0` -- which the box `-1 <= x <= 1` contains plenty of. No budget
    equality, precisely because `sum(x) = 1` would put the apex out of reach.
    """
    portfolio = MeanStdPortfolio.unconstrained(
        mu=np.array([0.10, 0.04, 0.06]), Sigma=np.ones((3, 3)), lam=1.0
    ).with_inequalities(np.vstack([np.eye(3), -np.eye(3)]), np.ones(6))
    return portfolio, portfolio.to_socp()


@pytest.fixture
def at_apex(rank_deficient):
    """A point of the rank-deficient instance whose conic slack is exactly the apex."""
    portfolio, problem = rank_deficient
    z = portfolio.socp_point(np.array([1.0, -1.0, 0.0]))
    assert soc.is_apex(problem.cone_slack(z)), "the fixture must actually be at the apex"
    return z


# ----------------------------------------------------------------------------------
# The geometry the branch rests on
# ----------------------------------------------------------------------------------


def test_the_tangent_cone_at_the_apex_is_the_cone_itself():
    """From the apex, `alpha * ds in Q` for every `alpha > 0` exactly when `ds in Q`.

    Checked over a range of step lengths and both kinds of direction, because "for every
    positive step, not merely small ones" is the part that distinguishes the apex from a
    smooth boundary point -- there, feasibility of a step is a quadratic condition and
    holds only up to some length.
    """
    apex_point = np.zeros(3)
    inside = np.array([2.0, 1.0, 1.0])
    outside = np.array([1.0, 2.0, 1.0])
    assert soc.is_member(inside)
    assert not soc.is_member(outside)
    for step in (1e-6, 1.0, 1e6):
        assert soc.is_member(apex_point + step * inside)
        assert not soc.is_member(apex_point + step * outside)


def test_the_tangent_cone_at_a_smooth_boundary_point_is_a_half_space():
    """There, a direction's admissibility is decided by one linear inequality.

    The contrast that makes the apex special. At `(5, 3, 4)` the covector `(1, -u)` decides
    which directions point into the cone, and it is a *hyperplane* test -- exactly what
    eq. (3) imposes as an equality and what the apex has no analogue of.
    """
    boundary = np.array([5.0, 3.0, 4.0])
    covector = np.array([1.0, -0.6, -0.8])
    entering = np.array([1.0, 0.0, 0.0])
    leaving = np.array([-1.0, 0.0, 0.0])
    assert float(covector @ entering) > 0.0
    assert float(covector @ leaving) < 0.0
    step = 1e-6
    assert soc.is_member(boundary + step * entering)
    assert not soc.is_member(boundary + step * leaving)


def test_exact_membership_is_the_condition_on_a_direction(rank_deficient, at_apex):
    """`step_stays_in_the_cone` is `ds in Q`, and it is exact rather than linearized."""
    _, problem = rank_deficient
    into = np.array([0.0, 0.0, 0.0, 1.0])
    assert apex.step_stays_in_the_cone(problem, 0, into)
    assert not apex.step_stays_in_the_cone(problem, 0, -into)


def test_the_membership_test_checks_its_factor_index(rank_deficient):
    """One cone here, so index 1 is a bug rather than a degenerate case."""
    _, problem = rank_deficient
    with pytest.raises(ProblemError, match=r"factor in \[0, 1\)"):
        apex.step_stays_in_the_cone(problem, 1, np.zeros(problem.num_variables))


# ----------------------------------------------------------------------------------
# Held: the normal-cone condition justifies the apex
# ----------------------------------------------------------------------------------


def test_the_apex_is_held_when_the_normal_cone_condition_holds(rank_deficient, at_apex):
    """Issue #24's "done when", on a rank-deficient `Sigma`: a direction from `w in Q`."""
    _, problem = rank_deficient
    result = apex.apex_direction(problem, WorkingSet.empty(problem), at_apex, 0)
    assert not result.released
    assert not result.is_blocked
    assert result.violation == 0.0
    assert "normal-cone condition holds" in result.reason
    assert soc.is_member(result.multipliers.w), "w lies in Q, which is what justified holding"


def test_holding_the_apex_keeps_the_slack_at_the_apex(rank_deficient, at_apex):
    """Exact membership, not a hyperplane: `G_block @ d = 0` holds the block outright."""
    _, problem = rank_deficient
    result = apex.apex_direction(problem, WorkingSet.empty(problem), at_apex, 0)
    np.testing.assert_allclose(problem.G @ result.direction.d, 0.0, atol=1e-12)
    assert apex.step_stays_in_the_cone(problem, 0, result.direction.d)


def test_holding_the_apex_is_not_a_stall(rank_deficient, at_apex):
    """The direction still moves `x` through the null space of `L`.

    The case #24 exists for, and the reason holding is a real answer rather than a
    surrender: on a rank-deficient covariance that null space has positive dimension, so
    there is somewhere to go while the risk stays exactly zero.
    """
    portfolio, problem = rank_deficient
    result = apex.apex_direction(problem, WorkingSet.empty(problem), at_apex, 0)
    assert np.linalg.norm(result.direction.d) > 1e-6, "the direction is not zero"
    assert result.direction.directional_derivative(problem.c) < 0.0, "and it descends"
    factor = portfolio.factor()
    np.testing.assert_allclose(factor @ result.direction.d[:-1], 0.0, atol=1e-12)


def test_the_held_working_set_marks_the_cone_at_its_apex(rank_deficient, at_apex):
    """Whatever the status was, the geometry says APEX and the branch records that."""
    _, problem = rank_deficient
    tangent_belief = updates.set_cone_status(WorkingSet.empty(problem), 0, ConeStatus.TANGENT)
    result = apex.apex_direction(problem, tangent_belief, at_apex, 0)
    assert result.working_set.status(0) is ConeStatus.APEX


def test_the_normal_cone_predicate_agrees_with_the_branch(rank_deficient, at_apex):
    """`is_apex_optimal` is the condition the branch applies, available on its own."""
    _, problem = rank_deficient
    result = apex.apex_direction(problem, WorkingSet.empty(problem), at_apex, 0)
    assert apex.is_apex_optimal(problem, result.multipliers, 0)


# ----------------------------------------------------------------------------------
# Blocked: the apex is unjustified and cannot be left
# ----------------------------------------------------------------------------------


def test_a_small_lambda_makes_the_apex_unjustified(rank_deficient):
    """`w = (lam, w_tail)` at a held apex, so a small enough `lam` puts `w` outside `Q`.

    Reached by turning the one knob that decides it. At a held apex the head of `w` is
    forced to `lam` by `t`'s stationarity, while the tail comes from the returns -- so the
    normal-cone condition is `||w_tail|| <= lam`, and a risk aversion below that fails it.
    """
    portfolio = MeanStdPortfolio.unconstrained(
        mu=np.array([0.10, 0.04, 0.06]), Sigma=np.ones((3, 3)), lam=0.01
    ).with_inequalities(np.vstack([np.eye(3), -np.eye(3)]), np.ones(6))
    problem = portfolio.to_socp()
    z = portfolio.socp_point(np.array([1.0, -1.0, 0.0]))
    result = apex.apex_direction(problem, WorkingSet.empty(problem), z, 0)
    assert result.violation > 0.0
    assert result.multipliers.w[0] == pytest.approx(0.01), "the head of w is lam"
    assert not soc.is_member(result.multipliers.w)
    assert result.is_blocked
    assert "#39" in result.reason


def test_eq_seven_can_never_release_the_apex():
    """On eq. (7) the released direction is infeasible by arithmetic, not by bad luck.

    The module's structural claim, checked rather than asserted. `t` appears only in the
    cone's head row and in the objective with coefficient `lam > 0`, so with the factor
    dropped every row of `W_k` has a zero `t` column, `t`'s stationarity row reads
    `rho * d_t = -lam`, and the slack direction's head is negative. A vector with a
    negative head is never in `Q`.

    Swept over `lam` and `rho` so the conclusion cannot be an artefact of either.
    """
    for lam in (0.001, 0.01, 0.5):
        portfolio = MeanStdPortfolio.unconstrained(
            mu=np.array([0.10, 0.04, 0.06]), Sigma=np.ones((3, 3)), lam=lam
        ).with_inequalities(np.vstack([np.eye(3), -np.eye(3)]), np.ones(6))
        problem = portfolio.to_socp()
        z = portfolio.socp_point(np.array([1.0, -1.0, 0.0]))
        released_set = updates.set_cone_status(WorkingSet.empty(problem), 0, ConeStatus.INACTIVE)
        for rho in (0.1, 1.0, 10.0):
            step = kkt.direction(problem, released_set, z, rho=rho)
            assert step.d[-1] == pytest.approx(-lam / rho), "d_t is forced by t's stationarity"
            assert not apex.step_stays_in_the_cone(problem, 0, step.d)


def test_a_blocked_apex_still_returns_a_feasible_direction(rank_deficient):
    """Blocked is not broken: the held direction is returned and it is feasible."""
    portfolio = MeanStdPortfolio.unconstrained(
        mu=np.array([0.10, 0.04, 0.06]), Sigma=np.ones((3, 3)), lam=0.01
    ).with_inequalities(np.vstack([np.eye(3), -np.eye(3)]), np.ones(6))
    problem = portfolio.to_socp()
    z = portfolio.socp_point(np.array([1.0, -1.0, 0.0]))
    result = apex.apex_direction(problem, WorkingSet.empty(problem), z, 0)
    assert result.is_blocked
    assert apex.step_stays_in_the_cone(problem, 0, result.direction.d)
    np.testing.assert_allclose(problem.G @ result.direction.d, 0.0, atol=1e-12)


# ----------------------------------------------------------------------------------
# Released: reachable only on the general form
# ----------------------------------------------------------------------------------


@pytest.fixture
def rewarding_head():
    """A general SOCP whose objective *rewards* the cone's head variable.

    Not a mean-standard-deviation portfolio -- `t` is not a risk bound here -- but a
    legitimate instance of the general form #9 deliberately supports, and the only shape
    that can release an apex. With `c = (0, -1)` and the slack laid out as `(t, x)`,
    dropping the factor drives `t` *up*, so the released direction's head is positive.
    """
    return SOCP(
        c=np.array([0.0, -1.0]),
        A=np.array([[1.0, 0.0]]),
        b=np.array([1.0]),
        E=np.zeros((0, 2)),
        d=np.zeros(0),
        G=np.array([[0.0, 1.0], [1.0, 0.0]]),
        h=np.zeros(2),
        cone=ConeProduct.from_dims(2),
    )


def test_the_apex_is_released_when_the_direction_stays_in_the_cone(rewarding_head):
    """The third outcome: `w` outside `Q`, and a released direction that is feasible."""
    result = apex.apex_direction(rewarding_head, WorkingSet.empty(rewarding_head), np.zeros(2), 0)
    assert result.released
    assert not result.is_blocked
    assert result.violation > 0.0
    assert "leaves the apex into the cone" in result.reason


def test_the_released_direction_leaves_the_apex(rewarding_head):
    """It moves the slack strictly into the cone, which is what releasing means."""
    result = apex.apex_direction(rewarding_head, WorkingSet.empty(rewarding_head), np.zeros(2), 0)
    slack_direction = rewarding_head.G @ result.direction.d
    assert soc.is_member(slack_direction)
    assert np.linalg.norm(slack_direction) > 1e-6
    assert soc.is_member(rewarding_head.cone_slack(np.zeros(2)) + 0.5 * slack_direction)


def test_releasing_deactivates_the_factor(rewarding_head):
    """The returned working set is the one the direction belongs to, not the one handed in."""
    handed_in = WorkingSet.empty(rewarding_head)
    result = apex.apex_direction(rewarding_head, handed_in, np.zeros(2), 0)
    assert result.working_set.status(0) is ConeStatus.INACTIVE
    assert handed_in.status(0) is ConeStatus.INACTIVE, "the input is unchanged either way"


def test_the_normal_cone_condition_fails_on_a_released_apex(rewarding_head):
    """`is_apex_optimal` and the branch agree that the apex was not justified."""
    result = apex.apex_direction(rewarding_head, WorkingSet.empty(rewarding_head), np.zeros(2), 0)
    assert not apex.is_apex_optimal(rewarding_head, result.multipliers, 0)


# ----------------------------------------------------------------------------------
# Refusals, and the report
# ----------------------------------------------------------------------------------


def test_the_branch_refuses_a_point_that_is_not_at_the_apex(rank_deficient):
    """Asking for the apex branch away from the apex is a bug, not a degenerate case."""
    portfolio, problem = rank_deficient
    z = portfolio.socp_point(np.array([0.5, 0.2, 0.3]))
    assert not soc.is_apex(problem.cone_slack(z))
    with pytest.raises(ProblemError, match="not at its apex"):
        apex.apex_direction(problem, WorkingSet.empty(problem), z, 0)


def test_the_branch_checks_its_factor_index(rank_deficient, at_apex):
    """Same range check as everywhere else in the package."""
    _, problem = rank_deficient
    with pytest.raises(ProblemError, match=r"factor in \[0, 1\)"):
        apex.apex_direction(problem, WorkingSet.empty(problem), at_apex, 3)


def test_the_report_names_the_outcome(rank_deficient, at_apex, rewarding_head):
    """Three outcomes, three distinguishable renderings -- so the third can be reported."""
    held = apex.apex_direction(problem := rank_deficient[1], WorkingSet.empty(problem), at_apex, 0)
    released = apex.apex_direction(rewarding_head, WorkingSet.empty(rewarding_head), np.zeros(2), 0)
    assert str(held).startswith("apex held:")
    assert str(released).startswith("apex released:")

    portfolio = MeanStdPortfolio.unconstrained(
        mu=np.array([0.10, 0.04, 0.06]), Sigma=np.ones((3, 3)), lam=0.01
    ).with_inequalities(np.vstack([np.eye(3), -np.eye(3)]), np.ones(6))
    blocked = apex.apex_direction(
        portfolio.to_socp(),
        WorkingSet.empty(portfolio.to_socp()),
        portfolio.socp_point(np.array([1.0, -1.0, 0.0])),
        0,
    )
    assert str(blocked).startswith("apex blocked:")


def test_the_tangent_module_still_refuses_what_this_one_answers(rank_deficient, at_apex):
    """The division of labour, asserted from both sides.

    #17 raises at the apex and #24 answers there. If the geometry module ever stopped
    refusing, this branch would be silently bypassed by a working set that believed the
    cone was tangent.
    """
    from cosa.geometry import tangent

    _, problem = rank_deficient
    with pytest.raises(tangent.ApexError):
        tangent.unit_tail(problem.cone_slack(at_apex))
    assert apex.apex_direction(problem, WorkingSet.empty(problem), at_apex, 0) is not None
