"""Deliverable 5's public interface: solving a portfolio without knowing what an SOCP is.

Issue #37. The claim under test is that a caller with a covariance matrix and a deadline can
get holdings out of this package from the README alone, and that what comes back is in the
units the question was asked in.

The README's example is executed here rather than paraphrased, because a README that has
stopped working is worse than no README.
"""

import numpy as np
import pytest

from cosa import NotOptimalError, Portfolio, solve_portfolio
from cosa.api import _rescaled
from cosa.linear_algebra.reuse import Reuse
from cosa.problem.socp import ProblemError


@pytest.fixture
def market():
    """A small factor market: returns, a low-rank-plus-diagonal covariance."""
    rng = np.random.default_rng(0)
    factors = rng.normal(size=(8, 4))
    return rng.normal(0.08, 0.03, 8), factors @ factors.T + 0.05 * np.eye(8)


# ----------------------------------------------------------------------------------
# The README's example
# ----------------------------------------------------------------------------------


def test_the_readme_example_runs(market):
    """The README's example, executed rather than paraphrased.

    A README that has stopped working is worse than no README.
    """
    returns, covariance = market
    answer = solve_portfolio(returns, covariance, lam=2.0, long_only=True)
    assert answer.is_optimal
    assert answer.holdings.sum() == pytest.approx(1.0)
    assert (answer.holdings >= -1e-9).all()
    assert "portfolio: return" in str(answer)


def test_the_answer_is_in_the_units_the_question_was_asked_in(market):
    """The result is in the units the question was asked in.

    A caller who supplied a covariance wants a standard deviation back, not `c.T @ z` over a
    lifted variable that includes an auxiliary.
    """
    returns, covariance = market
    answer = solve_portfolio(returns, covariance, lam=2.0, long_only=True)
    holdings = answer.holdings
    assert answer.expected_return == pytest.approx(float(returns @ holdings))
    assert answer.risk == pytest.approx(float(np.sqrt(holdings @ covariance @ holdings)))
    assert answer.utility == pytest.approx(answer.expected_return - 2.0 * answer.risk)


def test_risk_is_the_standard_deviation_and_not_the_variance(market):
    """Risk is the standard deviation, not the variance.

    Eq. (1) is written in standard deviation, which is the whole reason the problem is conic
    rather than quadratic, and returning the variance would change units silently.
    """
    returns, covariance = market
    answer = solve_portfolio(returns, covariance, lam=2.0, long_only=True)
    variance = float(answer.holdings @ covariance @ answer.holdings)
    assert answer.risk == pytest.approx(np.sqrt(variance))
    assert answer.risk != pytest.approx(variance)


# ----------------------------------------------------------------------------------
# Constraints
# ----------------------------------------------------------------------------------


def test_the_budget_constraint_is_added_by_default(market):
    """Almost every mandate has it, so almost every caller should not have to type it."""
    returns, covariance = market
    assert solve_portfolio(returns, covariance, lam=2.0).holdings.sum() == pytest.approx(1.0)


def test_the_budget_constraint_can_be_switched_off(market):
    """A long-short book with a different normalization is not an exotic case."""
    returns, covariance = market
    answer = solve_portfolio(returns, covariance, lam=6.0, budget=False)
    assert answer.holdings.sum() != pytest.approx(1.0)


def test_long_only_is_off_by_default():
    """Eq. (1) does not require it, and a long-short mandate is as ordinary as a long one.

    On a market with an asset whose expected return is negative, the unconstrained answer
    shorts it and the long-only answer does not.
    """
    rng = np.random.default_rng(3)
    factors = rng.normal(size=(6, 3))
    covariance = factors @ factors.T + 0.05 * np.eye(6)
    returns = np.array([0.10, 0.09, 0.08, 0.07, 0.06, -0.20])
    assert (solve_portfolio(returns, covariance, lam=1.0).holdings < -1e-6).any()
    assert (solve_portfolio(returns, covariance, lam=1.0, long_only=True).holdings >= -1e-9).all()


def test_caller_inequalities_bind(market):
    """A sector cap, expressed the way a mandate expresses it."""
    returns, covariance = market
    sector = np.zeros((1, 8))
    sector[0, :3] = 1.0
    answer = solve_portfolio(returns, covariance, lam=2.0, long_only=True, inequalities=(sector, np.array([0.25])))
    assert float((sector @ answer.holdings)[0]) <= 0.25 + 1e-8


def test_caller_equalities_bind(market):
    """And combine with the budget rather than replacing it."""
    returns, covariance = market
    pinned = np.zeros((1, 8))
    pinned[0, 0] = 1.0
    answer = solve_portfolio(returns, covariance, lam=2.0, equalities=(pinned, np.array([0.2])))
    assert answer.holdings[0] == pytest.approx(0.2)
    assert answer.holdings.sum() == pytest.approx(1.0)


def test_the_active_constraints_are_reported_by_name(market):
    """Success Criterion 3: "interpretable in terms of the active portfolio constraints"."""
    returns, covariance = market
    answer = solve_portfolio(returns, covariance, lam=2.0, long_only=True)
    assert "active" in answer.active


def test_a_malformed_instance_is_refused(market):
    """The shape checks the internal representation already does, reached through the door."""
    returns, _ = market
    with pytest.raises(ProblemError):
        solve_portfolio(returns, np.eye(3), lam=1.0)
    with pytest.raises(ProblemError):
        solve_portfolio(returns, np.eye(8), lam=-1.0)


# ----------------------------------------------------------------------------------
# Refusing to hand back a position it does not stand behind
# ----------------------------------------------------------------------------------


def test_a_non_optimal_solve_raises(market, monkeypatch):
    """A non-optimal solve raises rather than returning.

    A portfolio interface that silently returned holdings from a stalled solve would be
    handing someone a position to trade.
    """
    returns, covariance = market
    monkeypatch.setattr("cosa.api.solver.solve", _stalling(returns.size))
    with pytest.raises(NotOptimalError, match="rather than optimal"):
        solve_portfolio(returns, covariance, lam=2.0)


def test_the_rejected_result_is_attached(market, monkeypatch):
    """Nothing is lost by raising: a caller who wants to look anyway can."""
    returns, covariance = market
    monkeypatch.setattr("cosa.api.solver.solve", _stalling(returns.size))
    with pytest.raises(NotOptimalError) as raised:
        solve_portfolio(returns, covariance, lam=2.0)
    assert isinstance(raised.value.result, Portfolio)
    assert raised.value.result.status == "stalled"


def test_strict_can_be_switched_off(market, monkeypatch):
    """For a caller who would rather have the answer and judge it themselves."""
    returns, covariance = market
    monkeypatch.setattr("cosa.api.solver.solve", _stalling(returns.size))
    answer = solve_portfolio(returns, covariance, lam=2.0, strict=False)
    assert not answer.is_optimal
    assert answer.status == "stalled"


def _stalling(assets):
    """A `solve` that returns a stalled answer, for the refusal tests.

    Induced rather than found: no instance in this file stalls, which is the point of every
    other test here.
    """
    from dataclasses import replace

    from cosa.solver.cosa import solve

    def stalled(problem, **kwargs):
        """Solve normally, then relabel the answer as a stall.

        The original is captured before the patch goes in, or the patched name would call
        itself.
        """
        return replace(solve(problem, **kwargs), status="stalled")

    return stalled


# ----------------------------------------------------------------------------------
# Warm starting, which is the reason to use this at all
# ----------------------------------------------------------------------------------


def test_a_frontier_sweep_is_a_loop_over_lam(market):
    """The README's second example, and the shape a caller should be able to write.

    The measurement is the same one #35 makes internally, reached through the public door:
    warm starting roughly halves the work on a sweep whose working sets mostly transfer.
    """
    returns, covariance = market
    cache = Reuse()
    previous = None
    cold = warm = 0
    for lam in np.geomspace(1.0, 6.0, 12):
        alone = solve_portfolio(returns, covariance, lam=lam, long_only=True)
        seeded = solve_portfolio(returns, covariance, lam=lam, long_only=True, warm=previous, cache=cache)
        assert seeded.utility == pytest.approx(alone.utility, abs=1e-9)
        cold += alone.metrics.iterations
        warm += seeded.metrics.iterations
        previous = seeded.warm()
    assert warm < cold
    assert cold - warm > 0.25 * cold


def test_the_warm_start_carries_the_unscaled_point(market):
    """Equilibration depends on `c`, so two calls in a sequence generally scale differently.

    A point stored in one call's scaled variables would mean something else in the next
    one's, which is a bug that shows up as a warm start saving nothing.
    """
    returns, covariance = market
    answer = solve_portfolio(returns, covariance, lam=2.0, long_only=True, scale=True)
    carried = answer.warm()
    assert carried.z[: returns.size] == pytest.approx(answer.holdings)


def test_rescaling_a_warm_start_drops_only_the_multipliers(market):
    """Only the multipliers are dropped when a warm start moves into scaled variables.

    The working set names rows, and equilibration rescales rows without reordering them, so
    an index means the same thing on both sides. The multipliers do not, and seeding #23's
    curvature with the wrong Hessian is worse than seeding it with none.
    """
    returns, covariance = market
    answer = solve_portfolio(returns, covariance, lam=2.0, long_only=True)
    from cosa.linear_algebra.scaling import identity
    from cosa.problem.portfolio import MeanStdPortfolio

    problem = MeanStdPortfolio.unconstrained(mu=returns, Sigma=covariance, lam=2.0).to_socp()
    moved = _rescaled(answer.warm(), identity(problem))
    assert moved is not None
    assert moved.working_set is answer.warm().working_set
    assert moved.multipliers is None
    assert _rescaled(None, identity(problem)) is None


# ----------------------------------------------------------------------------------
# Equilibration, and why the default is what it is
# ----------------------------------------------------------------------------------


def test_equilibration_is_available_and_off(market):
    """Equilibration is available and off by default.

    Off because #36's study found it costs iterations everywhere and rescues nothing. That
    was not the first answer: it looked like the mitigation for `badly scaled` until
    building this interface exposed why that family was really failing. The flag remains for
    a caller whose units are genuinely pathological.
    """
    returns, covariance = market
    plain = solve_portfolio(returns, covariance, lam=2.0, long_only=True)
    scaled = solve_portfolio(returns, covariance, lam=2.0, long_only=True, scale=True)
    assert plain.utility == pytest.approx(scaled.utility, abs=1e-8)
    assert plain.metrics.iterations <= scaled.metrics.iterations
