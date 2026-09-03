"""The tangent condition of eq. (3), the normal, and the refusal at the apex.

The executable half of issue #17. Three things are checked harder than the rest:

* **eq. (3) is a linearization, and it linearizes the right thing.** §16.1
  (`paper.tex:1104`) asks for tangent directions, directions entering the cone and
  directions leaving it, so the sign of `tangent_residual` is tested on all three -- and,
  beyond the case list, a tangent direction is stepped along to confirm the slack really
  is second-order in the step. A first-order condition that is not the first-order
  behaviour of the constraint would pass a sign test and still be wrong.
* **the covector is the multiplier.** `tangent_covector` claims to be simultaneously
  eq. (3)'s covector and the unique dual variable in `Q` complementary to the point. The
  golden instances of `test_socp.py` were solved by hand under the sign convention of #9,
  so their `w` is arithmetic that knows nothing about this module -- which makes it the
  check that the claim is true rather than merely consistent.
* **the apex raises.** Every routine, every way of reaching it. Issue #17's "done when" is
  that the routine raises rather than returning a value at `Lx = 0`, because a garbage
  unit vector is accepted by everything downstream.
"""

import numpy as np
import pytest

from cosa import SIGN_CONVENTION, MeanStdForm, ProblemError
from cosa.geometry import soc, tangent

# A 3-4-5 boundary point, so `u` is exact in binary and every expectation below is a
# number rather than a rounding of one.
BOUNDARY = np.array([5.0, 3.0, 4.0])
UNIT = np.array([0.6, 0.8])


# ----------------------------------------------------------------------------------
# u, the covector, and the normal
# ----------------------------------------------------------------------------------


def test_the_unit_tail_is_the_normalized_tail():
    """The unit tail u = s_1 / ||s_1||, the plan's L @ x / ||L @ x|| in slack coordinates."""
    np.testing.assert_allclose(tangent.unit_tail(BOUNDARY), UNIT)
    assert np.linalg.norm(tangent.unit_tail(BOUNDARY)) == pytest.approx(1.0)


def test_the_unit_tail_ignores_the_head():
    """The unit tail depends on the tail alone, so it is defined off the boundary too."""
    for head in (-2.0, 0.5, 5.0, 100.0):
        np.testing.assert_allclose(tangent.unit_tail(np.array([head, 3.0, 4.0])), UNIT)


def test_the_covector_is_one_and_minus_u():
    """The covector of eq. (3): applied to (tau, ds_1) it gives tau - u.T @ ds_1."""
    np.testing.assert_allclose(tangent.tangent_covector(BOUNDARY), [1.0, -0.6, -0.8])


def test_the_outward_normal_is_the_covector_negated():
    """(-1, u): the gradient of ||s_1|| - s_0, which points out of the cone."""
    np.testing.assert_allclose(tangent.outward_normal(BOUNDARY), -tangent.tangent_covector(BOUNDARY))
    np.testing.assert_allclose(tangent.outward_normal(BOUNDARY), [-1.0, 0.6, 0.8])


def test_the_covector_lies_on_the_boundary_of_the_cone():
    """||-u|| = 1 = its head, so the covector is itself a boundary point of Q.

    Which is what lets it be a dual variable at all: the cone is self-dual, so a
    multiplier must lie in Q, and this one lies on its boundary -- the only place a
    nonzero multiplier complementary to a boundary point can be.
    """
    assert soc.is_boundary(tangent.tangent_covector(BOUNDARY))
    assert soc.is_member(tangent.tangent_covector(BOUNDARY))


def test_the_covector_is_complementary_to_its_own_point():
    """w.T @ s = 0 at a boundary point, which is the conic complementarity condition."""
    assert float(tangent.tangent_covector(BOUNDARY) @ BOUNDARY) == pytest.approx(0.0)


def test_the_covector_is_the_hand_solved_multiplier():
    """The golden instance's w, derived by hand under #9's convention, is lam * (1, -u).

    The strongest statement this module makes, checked against arithmetic that predates
    it. `mu = (2, 1)`, `lam = 1`, `sum(x) = 1`, `Sigma = I` puts the optimum at
    `x = (1, 0)` with `t = 1`, and `test_socp.py` derives `w = (1, -1, 0)` from
    stationarity and complementarity alone.
    """
    form = MeanStdForm(
        mu=np.array([2.0, 1.0]),
        lam=1.0,
        A=np.zeros((0, 2)),
        b=np.zeros(0),
        E=np.array([[1.0, 1.0]]),
        d=np.array([1.0]),
        L=np.eye(2),
    )
    problem = form.to_socp()
    z = np.array([1.0, 0.0, 1.0])
    slack = problem.cone_slack(z)
    np.testing.assert_allclose(form.lam * tangent.tangent_covector(slack), [1.0, -1.0, 0.0], atol=1e-15)


def test_the_multiplier_direction_respects_the_sign_convention():
    """The covector is in Q, not -Q, which is what SIGN_CONVENTION.cone being -1 buys.

    Stated as a dependency rather than as a coincidence: the plan writes the dual
    condition as `w_soc in Q`, and #9 chose the sign of the cone term in the Lagrangian to
    make that true. Flip that choice and the multiplier would have to be the outward
    normal instead.
    """
    assert SIGN_CONVENTION.cone == -1.0
    assert soc.is_member(tangent.tangent_covector(BOUNDARY))
    assert not soc.is_member(tangent.outward_normal(BOUNDARY))


# ----------------------------------------------------------------------------------
# §16.1's case list: tangent, entering, leaving
# ----------------------------------------------------------------------------------

# At (5, 3, 4) with u = (0.6, 0.8): a tail direction orthogonal to u contributes nothing
# to u.T @ ds_1, so the residual is tau alone.
TANGENT_DIRECTION = np.array([0.0, 0.8, -0.6])
ENTERING_DIRECTION = np.array([1.0, 0.0, 0.0])
LEAVING_DIRECTION = np.array([-1.0, 0.0, 0.0])


def test_a_tangent_direction_has_zero_residual():
    """Eq. (3) holds: the direction stays on the boundary to first order."""
    assert tangent.tangent_residual(BOUNDARY, TANGENT_DIRECTION) == pytest.approx(0.0)


def test_a_direction_entering_the_cone_has_a_positive_residual():
    """The slack grows, so the cone stops being active along this direction."""
    assert tangent.tangent_residual(BOUNDARY, ENTERING_DIRECTION) > 0.0
    assert soc.is_interior(BOUNDARY + 0.1 * ENTERING_DIRECTION)


def test_a_direction_leaving_the_cone_has_a_negative_residual():
    """The slack shrinks, so a step along it is limited -- by #18's quadratic, not by this."""
    assert tangent.tangent_residual(BOUNDARY, LEAVING_DIRECTION) < 0.0
    assert not soc.is_member(BOUNDARY + 0.1 * LEAVING_DIRECTION)


@pytest.mark.parametrize("direction", [TANGENT_DIRECTION, ENTERING_DIRECTION, LEAVING_DIRECTION])
def test_the_residual_is_the_covector_applied_to_the_direction(direction):
    """One definition, two spellings, and they agree."""
    expected = float(tangent.tangent_covector(BOUNDARY) @ direction)
    assert tangent.tangent_residual(BOUNDARY, direction) == pytest.approx(expected)


def test_the_residual_is_the_first_order_rate_of_the_conic_slack():
    """The residual really is d/d(alpha) of s_0 - ||s_1|| at alpha = 0.

    Checked numerically, because this is the claim that makes eq. (3) a *linearization*
    rather than an unrelated linear equation that happens to vanish on the boundary.
    """
    for direction in (TANGENT_DIRECTION, ENTERING_DIRECTION, LEAVING_DIRECTION, np.array([0.3, -0.2, 0.7])):
        step = 1e-7
        forward = soc.slack(BOUNDARY + step * direction)
        backward = soc.slack(BOUNDARY - step * direction)
        derivative = (forward - backward) / (2.0 * step)
        assert derivative == pytest.approx(tangent.tangent_residual(BOUNDARY, direction), abs=1e-6)


def test_a_tangent_direction_leaves_a_convex_cone_at_second_order():
    """Tangency is first order only: the exact step still exits, quadratically.

    The reason #18 exists. A tangent direction is what the working set imposes, and the
    plan's eq. (6) is the exact condition; this test is the evidence that the two are not
    the same thing, so a step length taken from the linearization would be wrong.
    """
    losses = []
    for step in (1e-2, 1e-3, 1e-4):
        losses.append(-soc.slack(BOUNDARY + step * TANGENT_DIRECTION) / step**2)
    assert all(loss > 0.0 for loss in losses), "the slack goes negative, so the direction exits"
    # The ratio to alpha^2 is bounded as alpha shrinks, which is what "second order" means.
    assert max(losses) / min(losses) == pytest.approx(1.0, abs=0.05)


# ----------------------------------------------------------------------------------
# The row over z, which is what the KKT assembly consumes
# ----------------------------------------------------------------------------------


def test_the_tangent_row_is_the_covector_pushed_through_g():
    """tangent_row(s, G) @ p == tangent_residual(s, G @ p), for any p.

    The defining identity, and the whole reason the row form exists: the direction
    subproblem constrains `p`, while eq. (3) is stated about the slack direction.
    """
    rng = np.random.default_rng(0)
    conic = rng.normal(size=(3, 4))
    row = tangent.tangent_row(BOUNDARY, conic)
    for p in rng.normal(size=(10, 4)):
        assert float(row @ p) == pytest.approx(tangent.tangent_residual(BOUNDARY, conic @ p))


def test_the_tangent_row_of_eq_seven_is_tau_minus_u_l_p():
    """For eq. (7)'s G, the row is exactly the plan's `tau - u.T @ L @ p`.

    `G` has the head row selecting `t` and the tail rows holding `L`, so the row must come
    out as `(-u.T @ L, 1)` with `t` last -- which is eq. (3) written over `(p, tau)`.
    """
    form = MeanStdForm(
        mu=np.array([0.1, 0.05]),
        lam=1.0,
        A=np.zeros((0, 2)),
        b=np.zeros(0),
        E=np.zeros((0, 2)),
        d=np.zeros(0),
        L=np.array([[0.3, 0.1], [0.0, 0.2]]),
    )
    problem = form.to_socp()
    x = np.array([0.6, 0.4])
    z = np.concatenate([x, [float(np.linalg.norm(form.L @ x))]])
    slack = problem.cone_slack(z)
    unit = tangent.unit_tail(slack)
    row = tangent.tangent_row(slack, problem.G)
    np.testing.assert_allclose(row, np.concatenate([-(unit @ form.L), [1.0]]), atol=1e-15)


def test_the_tangent_row_rejects_a_block_of_the_wrong_height():
    """One row of G per entry of the cone block; anything else is a layout bug."""
    with pytest.raises(ProblemError, match="expected a matrix with 3 rows"):
        tangent.tangent_row(BOUNDARY, np.zeros((2, 4)))


# ----------------------------------------------------------------------------------
# The apex, refused every way it can be reached
# ----------------------------------------------------------------------------------

APEX_ROUTINES = [
    tangent.unit_tail,
    tangent.tangent_covector,
    tangent.outward_normal,
]


@pytest.mark.parametrize("routine", APEX_ROUTINES)
def test_every_routine_refuses_the_apex(routine):
    """Issue #17's "done when": raise rather than return a value at L @ x = 0."""
    with pytest.raises(tangent.ApexError, match="tail has vanished"):
        routine(np.zeros(3))


@pytest.mark.parametrize("routine", APEX_ROUTINES)
def test_every_routine_refuses_a_vanishing_tail_off_the_apex(routine):
    """(s_0, 0) has no u either, and it is not the apex -- §8.1's condition is Lx = 0.

    The wider guard matters: `(1, 0, 0)` is strictly inside the cone, so nothing about it
    looks degenerate, and a routine that only tested for the apex would hand back an
    arbitrary unit vector here.
    """
    with pytest.raises(tangent.ApexError):
        routine(np.array([1.0, 0.0, 0.0]))


def test_the_tangent_residual_refuses_the_apex():
    """The residual needs u as much as the covector does."""
    with pytest.raises(tangent.ApexError):
        tangent.tangent_residual(np.zeros(3), np.array([1.0, 1.0, 0.0]))


def test_the_tangent_row_refuses_the_apex():
    """So a working set that believes a cone is tangent at the apex cannot assemble.

    The failure the guard is really for: #11's `ConeStatus.TANGENT` is a *belief*, and this
    is what happens when the geometry contradicts it. #24 is the branch that should have
    run instead.
    """
    with pytest.raises(tangent.ApexError):
        tangent.tangent_row(np.zeros(3), np.eye(3))


def test_a_tail_within_tolerance_of_vanishing_is_refused():
    """A tail of 1e-15 is not a direction, and normalizing it amplifies pure rounding."""
    with pytest.raises(tangent.ApexError):
        tangent.unit_tail(np.array([1e-15, 1e-16, 1e-16]))


def test_the_guard_is_scale_aware():
    """On a badly scaled instance, "vanishing" has to mean vanishing relative to something."""
    small_tail = np.array([1e6, 1.0, 1.0])
    assert tangent.unit_tail(small_tail) is not None
    with pytest.raises(tangent.ApexError):
        tangent.unit_tail(small_tail, scale=1e12)


def test_a_looser_tolerance_widens_the_refusal():
    """The tolerance is a parameter, because #29 will want to vary the ones near it."""
    point = np.array([1.0, 1e-6, 0.0])
    assert tangent.unit_tail(point)[0] == pytest.approx(1.0)
    with pytest.raises(tangent.ApexError):
        tangent.unit_tail(point, tolerance=1e-4)


def test_the_apex_error_is_a_problem_error():
    """So a caller catching ProblemError catches this, but can still single it out."""
    assert issubclass(tangent.ApexError, ProblemError)


# ----------------------------------------------------------------------------------
# The boundary precondition, checked only when asked
# ----------------------------------------------------------------------------------


def test_require_boundary_accepts_a_boundary_point():
    """The precondition of eq. (3), and it holds here."""
    assert tangent.require_boundary(BOUNDARY) is None


@pytest.mark.parametrize("point", [np.array([6.0, 3.0, 4.0]), np.array([4.0, 3.0, 4.0])])
def test_require_boundary_rejects_a_point_off_the_boundary(point):
    """A hyperplane through a point that is not on the surface means nothing."""
    with pytest.raises(tangent.NotOnBoundaryError, match="conic slack"):
        tangent.require_boundary(point)


def test_require_boundary_rejects_the_apex_as_the_apex():
    """The apex is on the boundary, so it must be refused for the other reason."""
    with pytest.raises(tangent.ApexError):
        tangent.require_boundary(np.zeros(3))


def test_the_routines_do_not_require_the_boundary_themselves():
    """Deliberately: an iterate a rounding error off the boundary still has a tangent row.

    If every routine checked, the working-set logic would have to hold its iterates on the
    boundary to machine precision -- which no floating-point iteration does. The check is
    available; it is not imposed.
    """
    nearly = np.array([5.0 + 1e-7, 3.0, 4.0])
    assert not soc.is_boundary(nearly)
    np.testing.assert_allclose(tangent.unit_tail(nearly), UNIT)


# ----------------------------------------------------------------------------------
# Malformed input
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("point", [np.zeros(0), np.zeros(1)])
def test_a_block_too_short_to_be_a_cone_is_rejected(point):
    """Q^1 is the non-negative ray: a linear inequality, with no tangent geometry."""
    with pytest.raises(ProblemError, match="dim >= 2"):
        tangent.unit_tail(point)


def test_a_two_dimensional_block_is_rejected():
    """A matrix is not a point on the cone and must not be silently flattened."""
    with pytest.raises(ProblemError, match="expected a vector"):
        tangent.unit_tail(np.zeros((2, 2)))


def test_a_direction_of_the_wrong_length_is_rejected():
    """The slack direction lives in the same space as the slack."""
    with pytest.raises(ProblemError, match="expected 3 entries"):
        tangent.tangent_residual(BOUNDARY, np.array([1.0, 0.0]))
