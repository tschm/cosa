"""The mean-standard-deviation portfolio problem and its reduction to an SOCP.

The executable half of issue #10. Two claims carry the module and are tested hardest:

* `||L @ x||_2 == sqrt(x.T @ Sigma @ x)` to machine precision, **for any `Sigma >= 0`**,
  rank-deficient included. §2.1 (`paper.tex:132`) assumes only semidefiniteness, so a
  factorization that needs a definite covariance is not a factorization of this problem.
* eq. (1) maps to eq. (2). The max form and the min form are the same problem, and the
  auxiliary `t` does not change the answer -- which is checkable rather than arguable,
  because `MeanStdPortfolio.socp_point` lifts an `x` to the `(x, t)` where the two
  objectives must agree.

The third thing worth stating: a singular covariance is what makes the apex reachable at
a nonzero portfolio, which is the case §8.1 (`paper.tex:623`) has to branch on. That is
asserted here, at its source, rather than left as a remark.
"""

import numpy as np
import pytest

from cosa import ConePosition, MeanStdPortfolio, ProblemError
from cosa.geometry import soc
from cosa.problem.portfolio import covariance_factor, covariance_tolerance

# ----------------------------------------------------------------------------------
# Covariances to factor: the full-rank, the singular and the awkward
# ----------------------------------------------------------------------------------


def _random_covariance(assets, rank, seed):
    """A covariance of the given rank, as `B.T @ B` with `B` of that many rows."""
    factor = np.random.default_rng(seed).normal(size=(rank, assets))
    return factor.T @ factor


COVARIANCES = {
    "identity": (np.eye(4), 4),
    "diagonal": (np.diag([0.04, 0.09, 0.0025]), 3),
    "full rank": (_random_covariance(5, 5, 1), 5),
    "rank 3 of 6": (_random_covariance(6, 3, 2), 3),
    "rank 1 of 4": (_random_covariance(4, 1, 3), 1),
    "perfectly correlated": (np.ones((3, 3)), 1),
    "ill conditioned": (np.diag([1.0, 1e-8, 1e-16]), 2),
    "zero": (np.zeros((2, 2)), 0),
}


@pytest.fixture(params=sorted(COVARIANCES))
def covariance(request):
    """One covariance together with the rank it is known to have."""
    return COVARIANCES[request.param]


def test_the_factor_reproduces_the_covariance(covariance):
    """Sigma = L.T @ L, which is the only thing the factor has to be."""
    matrix, _ = covariance
    factor = covariance_factor(matrix)
    np.testing.assert_allclose(factor.T @ factor, matrix, atol=1e-12, rtol=1e-9)


def test_the_factor_turns_the_standard_deviation_into_a_norm(covariance):
    """||L @ x|| == sqrt(x.T @ Sigma @ x) to machine precision, over many directions.

    Issue #10's "done when", asserted over random `x` rather than at one point: the
    identity is what licenses replacing the risk term with a cone constraint, and it has
    to hold everywhere, not just where it was checked once.
    """
    matrix, _ = covariance
    factor = covariance_factor(matrix)
    rng = np.random.default_rng(11)
    for x in rng.normal(size=(25, matrix.shape[0])):
        quadratic = max(0.0, float(x @ matrix @ x))
        assert np.linalg.norm(factor @ x) == pytest.approx(np.sqrt(quadratic), abs=1e-12, rel=1e-9)


def test_the_factor_has_one_row_per_unit_of_rank(covariance):
    """A rank-k covariance gives a (k, n) factor -- and so the cheap cone Q^(1+k)."""
    matrix, rank = covariance
    factor = covariance_factor(matrix)
    assert factor.shape[1] == matrix.shape[0]
    assert factor.shape[0] == max(rank, 1), "the zero covariance keeps one row, so the risk term has a cone"
    assert np.linalg.matrix_rank(factor) == rank


def test_the_zero_covariance_factors_to_a_single_zero_row():
    """The degenerate limit, answered honestly rather than with an empty matrix."""
    factor = covariance_factor(np.zeros((3, 3)))
    np.testing.assert_array_equal(factor, np.zeros((1, 3)))


def test_the_rows_lead_with_the_dominant_risk_factor():
    """Rows in order of decreasing eigenvalue, so a truncation would keep the big ones."""
    factor = covariance_factor(np.diag([1.0, 9.0, 4.0]))
    np.testing.assert_allclose(np.linalg.norm(factor, axis=1), [3.0, 2.0, 1.0])


# ----------------------------------------------------------------------------------
# Rank deficiency is what makes the apex reachable
# ----------------------------------------------------------------------------------


def test_a_singular_covariance_puts_the_apex_within_reach():
    """There is an x != 0 with L @ x = 0, so the conic slack can sit at the apex.

    Not a numerical curiosity: this is how the case §8.1 treats separately actually
    arises, and it arises whenever the covariance is singular -- which the plan's
    "highly correlated assets" family (`paper.tex:942`) guarantees.
    """
    matrix = np.ones((3, 3))  # perfectly correlated: rank 1
    factor = covariance_factor(matrix)
    x = np.array([1.0, -1.0, 0.0])  # in the null space, and not the zero portfolio
    assert np.linalg.norm(x) > 0.0
    np.testing.assert_allclose(factor @ x, 0.0, atol=1e-12)

    portfolio = MeanStdPortfolio.unconstrained(mu=np.ones(3), Sigma=matrix, lam=1.0)
    problem = portfolio.to_socp()
    assert soc.position(problem.cone_slack(portfolio.socp_point(x))) is ConePosition.APEX


def test_a_definite_covariance_keeps_the_apex_at_the_origin_only():
    """With Sigma > 0, L @ x = 0 forces x = 0: the apex is unreachable elsewhere."""
    portfolio = MeanStdPortfolio.unconstrained(mu=np.ones(2), Sigma=np.eye(2), lam=1.0)
    problem = portfolio.to_socp()
    assert soc.position(problem.cone_slack(portfolio.socp_point(np.zeros(2)))) is ConePosition.APEX
    assert soc.position(problem.cone_slack(portfolio.socp_point(np.array([0.3, 0.7])))) is ConePosition.BOUNDARY


# ----------------------------------------------------------------------------------
# What is not a covariance
# ----------------------------------------------------------------------------------


def test_a_non_square_covariance_is_rejected():
    """A covariance is square; anything else is a modelling error, not a hard instance."""
    with pytest.raises(ProblemError, match="square"):
        covariance_factor(np.ones((2, 3)))


def test_an_asymmetric_covariance_is_rejected():
    """Asymmetry beyond rounding is repaired by nobody: it is reported."""
    with pytest.raises(ProblemError, match="symmetric"):
        covariance_factor(np.array([[1.0, 0.5], [0.2, 1.0]]))


def test_rounding_level_asymmetry_is_symmetrized_rather_than_refused():
    """A computed covariance is asymmetric in the last bit, and that is not an error."""
    matrix = np.array([[1.0, 0.5], [0.5 + 1e-17, 1.0]])
    factor = covariance_factor(matrix)
    np.testing.assert_allclose(factor.T @ factor, (matrix + matrix.T) / 2.0, atol=1e-12)


def test_an_indefinite_matrix_is_not_a_covariance():
    """§2.1 assumes Sigma >= 0. A negative eigenvalue means the input is wrong."""
    with pytest.raises(ProblemError, match="positive semidefinite"):
        covariance_factor(np.diag([1.0, -1.0]))


def test_a_negative_eigenvalue_within_the_tolerance_is_treated_as_zero():
    """The rounding-level negative eigenvalue a real estimator produces is absorbed."""
    factor = covariance_factor(np.diag([1.0, -1e-17]))
    assert factor.shape == (1, 2)


def test_a_caller_can_widen_the_tolerance_for_a_repaired_estimator():
    """A shrinkage estimator's larger negative eigenvalues are a modelling decision.

    The default is deliberately tight, so making that decision has to be explicit.
    """
    matrix = np.diag([1.0, -1e-9])
    with pytest.raises(ProblemError, match="positive semidefinite"):
        covariance_factor(matrix)
    assert covariance_factor(matrix, tolerance=1e-8).shape == (1, 2)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_non_finite_covariance_is_rejected(bad):
    """A NaN would propagate into every factor row and every subsequent solve."""
    with pytest.raises(ProblemError, match="must be finite"):
        covariance_factor(np.array([[1.0, 0.0], [0.0, bad]]))


@pytest.mark.parametrize("tolerance", [-1.0, np.nan])
def test_a_nonsensical_tolerance_is_rejected(tolerance):
    """A negative or NaN tolerance would make the rank decision meaningless."""
    with pytest.raises(ProblemError, match="finite and non-negative"):
        covariance_factor(np.eye(2), tolerance=tolerance)


def test_an_empty_covariance_is_rejected():
    """A portfolio over no assets has no risk term to build a cone for."""
    with pytest.raises(ProblemError, match="at least one asset"):
        covariance_factor(np.zeros((0, 0)))


def test_the_tolerance_helper_checks_the_shape_too():
    """Asking for the threshold of a non-covariance fails the way asking for its factor does.

    Same input, same complaint: a caller comparing ranks should not have to discover a
    shape error through a different message than the one the factorization would give.
    """
    with pytest.raises(ProblemError, match="square"):
        covariance_tolerance(np.ones((2, 3)))
    assert covariance_tolerance(np.zeros((0, 0))) == 0.0


def test_the_default_tolerance_scales_with_the_covariance():
    """The default is n * eps * lambda_max: relative to the covariance, zero for zero."""
    assert covariance_tolerance(np.zeros((3, 3))) == 0.0
    assert covariance_tolerance(np.diag([1e6, 1.0])) == pytest.approx(2 * np.finfo(float).eps * 1e6)


# ----------------------------------------------------------------------------------
# Eq. (1) to eq. (2): the same problem, minimized instead of maximized
# ----------------------------------------------------------------------------------


@pytest.fixture
def portfolio():
    """Three assets, a budget equality and a weight cap -- a small but complete eq. (1)."""
    return MeanStdPortfolio(
        mu=np.array([0.10, 0.04, 0.06]),
        Sigma=np.array([[0.04, 0.01, 0.00], [0.01, 0.09, 0.02], [0.00, 0.02, 0.16]]),
        lam=2.0,
        A=np.array([[1.0, 0.0, 0.0]]),
        b=np.array([0.5]),
        E=np.array([[1.0, 1.0, 1.0]]),
        d=np.array([1.0]),
    )


WEIGHTS = [
    np.array([0.5, 0.3, 0.2]),
    np.array([0.0, 0.0, 1.0]),
    np.array([1.0, -0.5, 0.5]),
    np.zeros(3),
]


@pytest.mark.parametrize("x", WEIGHTS)
def test_the_risk_term_is_the_standard_deviation(portfolio, x):
    """sigma(x) = sqrt(x.T @ Sigma @ x), and the factor computes the same number."""
    assert portfolio.std(x) == pytest.approx(np.sqrt(portfolio.variance(x)))
    assert portfolio.std(x) == pytest.approx(float(np.linalg.norm(portfolio.factor() @ x)))


@pytest.mark.parametrize("x", WEIGHTS)
def test_the_min_form_is_the_negated_max_form(portfolio, x):
    """Eq. (1) to eq. (2), first move: `cost` is `-utility` at every x, feasible or not."""
    assert portfolio.cost(x) == pytest.approx(-portfolio.utility(x))
    assert portfolio.utility(x) == pytest.approx(portfolio.expected_return(x) - portfolio.lam * portfolio.std(x))


@pytest.mark.parametrize("x", WEIGHTS)
def test_the_socp_objective_agrees_at_the_lifted_point(portfolio, x):
    """Eq. (1) to eq. (2), second move: `c.T @ z` at `z = (x, sigma(x))` is the cost.

    The auxiliary variable is the whole difference between the two forms, so this is the
    mapping itself rather than a consequence of it.
    """
    problem = portfolio.to_socp()
    z = portfolio.socp_point(x)
    assert float(problem.c @ z) == pytest.approx(portfolio.cost(x))
    assert z[-1] == pytest.approx(portfolio.std(x))


@pytest.mark.parametrize("x", WEIGHTS)
def test_the_lifted_point_sits_on_the_cone_boundary(portfolio, x):
    """The lift takes t = sigma(x), the smallest feasible t -- where lam > 0 drives it."""
    problem = portfolio.to_socp()
    assert soc.is_boundary(problem.cone_slack(portfolio.socp_point(x)))


def test_a_larger_t_is_feasible_and_strictly_worse(portfolio):
    """Why the optimum has t = sigma(x): slack in the cone is paid for at lam per unit."""
    problem = portfolio.to_socp()
    x = np.array([0.5, 0.3, 0.2])
    tight = portfolio.socp_point(x)
    slack = tight + np.array([0.0, 0.0, 0.0, 0.25])
    assert soc.is_interior(problem.cone_slack(slack))
    assert float(problem.c @ slack) == pytest.approx(float(problem.c @ tight) + portfolio.lam * 0.25)


# ----------------------------------------------------------------------------------
# The reduction carries the constraints, and round-trips
# ----------------------------------------------------------------------------------


def test_the_socp_carries_the_portfolio_constraints(portfolio):
    """A and E gain a zero column for t, and nothing else changes."""
    problem = portfolio.to_socp()
    assert problem.num_variables == portfolio.num_assets + 1
    np.testing.assert_array_equal(problem.A[:, :-1], portfolio.A)
    np.testing.assert_array_equal(problem.A[:, -1], 0.0)
    np.testing.assert_array_equal(problem.E[:, :-1], portfolio.E)
    np.testing.assert_array_equal(problem.E[:, -1], 0.0)
    np.testing.assert_array_equal(problem.b, portfolio.b)
    np.testing.assert_array_equal(problem.d, portfolio.d)


def test_the_socp_round_trips_back_to_eq_seven(portfolio):
    """to_socp then as_mean_std returns the same eq. (7) data, factor included."""
    form = portfolio.to_mean_std()
    recovered = portfolio.to_socp().as_mean_std()
    np.testing.assert_allclose(recovered.mu, form.mu)
    assert recovered.lam == pytest.approx(form.lam)
    np.testing.assert_allclose(recovered.A, form.A)
    np.testing.assert_allclose(recovered.b, form.b)
    np.testing.assert_allclose(recovered.E, form.E)
    np.testing.assert_allclose(recovered.d, form.d)
    np.testing.assert_allclose(recovered.L, form.L)


def test_the_cone_is_one_factor_of_the_factor_rank(portfolio):
    """Q^(1 + rank): the head is t and the tail rows are L."""
    problem = portfolio.to_socp()
    rank = portfolio.factor().shape[0]
    assert len(problem.cone) == 1
    assert problem.cone.cones[0].dim == 1 + rank


def test_a_rank_deficient_covariance_gives_a_smaller_cone():
    """Rank deficiency is not a special case downstream: it is a shorter tail."""
    portfolio = MeanStdPortfolio.unconstrained(mu=np.ones(5), Sigma=np.ones((5, 5)), lam=1.0)
    assert portfolio.to_socp().cone.cones[0].dim == 2


# ----------------------------------------------------------------------------------
# Building an instance up, and refusing a malformed one
# ----------------------------------------------------------------------------------


def test_an_unconstrained_instance_has_no_rows():
    """The starting point for the constraint-family generators of #19."""
    portfolio = MeanStdPortfolio.unconstrained(mu=np.ones(3), Sigma=np.eye(3), lam=1.0)
    assert portfolio.A.shape == (0, 3)
    assert portfolio.E.shape == (0, 3)
    assert portfolio.to_socp().num_inequalities == 0


def test_constraints_can_be_added_one_block_at_a_time():
    """Frozen instances, so each addition returns a new one and the original stands."""
    portfolio = MeanStdPortfolio.unconstrained(mu=np.ones(2), Sigma=np.eye(2), lam=1.0)
    grown = portfolio.with_equalities([[1.0, 1.0]], [1.0]).with_inequalities([[1.0, 0.0]], [0.6])
    assert grown.E.shape == (1, 2)
    assert grown.A.shape == (1, 2)
    assert portfolio.E.shape == (0, 2), "the original instance is unchanged"


def test_a_negative_variance_from_rounding_is_clipped():
    """A near-null-space direction rounds the quadratic form negative; sigma stays real."""
    portfolio = MeanStdPortfolio.unconstrained(mu=np.ones(2), Sigma=np.ones((2, 2)), lam=1.0)
    assert portfolio.variance(np.array([1.0, -1.0])) >= 0.0
    assert portfolio.std(np.array([1.0, -1.0])) == pytest.approx(0.0)


@pytest.mark.parametrize("lam", [0.0, -1.0, np.nan])
def test_a_non_positive_risk_aversion_is_rejected(lam):
    """§2.1 takes lam > 0; at lam = 0 the risk term vanishes and the cone is pointless."""
    with pytest.raises(ProblemError, match="lam > 0"):
        MeanStdPortfolio.unconstrained(mu=np.ones(2), Sigma=np.eye(2), lam=lam)


def test_an_empty_instance_is_rejected():
    """A portfolio over no assets is not a problem."""
    with pytest.raises(ProblemError, match="at least one asset"):
        MeanStdPortfolio.unconstrained(mu=np.zeros(0), Sigma=np.zeros((0, 0)), lam=1.0)


def test_a_covariance_of_the_wrong_size_is_rejected():
    """Sigma is (n, n) with n from mu, checked at construction rather than at solve time."""
    with pytest.raises(ProblemError, match="expected 3 rows"):
        MeanStdPortfolio.unconstrained(mu=np.ones(3), Sigma=np.eye(2), lam=1.0)


def test_a_constraint_block_of_the_wrong_width_is_rejected():
    """A row over the wrong number of assets is caught where it is handed over."""
    portfolio = MeanStdPortfolio.unconstrained(mu=np.ones(3), Sigma=np.eye(3), lam=1.0)
    with pytest.raises(ProblemError, match="expected 3 columns"):
        portfolio.with_inequalities([[1.0, 1.0]], [1.0])


def test_a_point_of_the_wrong_length_is_rejected(portfolio):
    """Every method taking x checks it, so a length bug surfaces at the call, not later."""
    with pytest.raises(ProblemError, match="expected 3 entries"):
        portfolio.std(np.ones(2))
