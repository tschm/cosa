"""The five conic KKT residuals of §6, and the certificate they constitute.

The executable half of issue #22. Its "done when" is that all five are computed, that they
constitute the primary termination criterion (Success Criterion 2, `paper.tex:1321`), and
that Level 3 optimality (`paper.tex:1033`) holds at termination.

The first section checks all five against the hand-solved instances of #9, where every one
must be exactly zero. The second checks that each can *fail* -- five residuals that could
never be nonzero would satisfy the letter of the issue and certify nothing -- and that each
fails for its own reason, which is the argument for reporting five numbers rather than one.
"""

import numpy as np
import pytest

from cosa import MeanStdForm, MeanStdPortfolio, Multipliers, ProblemError, WorkingSet
from cosa.active_set import multipliers as mult
from cosa.active_set import updates
from cosa.experiments import portfolio as families
from cosa.experiments import reference
from cosa.geometry import tangent
from cosa.linear_algebra import kkt
from cosa.solver import termination as term

# The two instances of `test_socp.py`, with their hand-derived optima and multipliers.
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
        "multipliers": Multipliers(y=np.array([0.5]), nu=np.zeros(0), w=np.array([0.5, -0.5])),
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
        "multipliers": Multipliers(y=np.zeros(0), nu=np.array([1.0]), w=np.array([1.0, -1.0, 0.0])),
    },
}


@pytest.fixture(params=sorted(GOLDEN))
def golden(request):
    """One hand-solved instance with its optimum and multipliers."""
    case = GOLDEN[request.param]
    return case["form"].to_socp(), case["z"], case["multipliers"]


@pytest.fixture
def portfolio():
    """A three-asset instance with all three constraint blocks."""
    return MeanStdPortfolio(
        mu=np.array([0.10, 0.04, 0.06]),
        Sigma=np.diag([0.04, 0.09, 0.16]),
        lam=2.0,
        A=np.array([[1.0, 0.0, 0.0]]),
        b=np.array([0.5]),
        E=np.array([[1.0, 1.0, 1.0]]),
        d=np.array([1.0]),
    )


# ----------------------------------------------------------------------------------
# All five, at a certified optimum
# ----------------------------------------------------------------------------------


def test_every_residual_vanishes_at_the_hand_solved_optimum(golden):
    """Issue #22's "done when": Level 3 holds, from arithmetic that predates the module."""
    problem, z, found = golden
    measured = term.residuals(problem, z, found)
    assert measured.primal == pytest.approx(0.0, abs=1e-14)
    assert measured.dual == pytest.approx(0.0, abs=1e-14)
    assert measured.stationarity == pytest.approx(0.0, abs=1e-14)
    assert measured.linear_complementarity == pytest.approx(0.0, abs=1e-14)
    assert measured.cone_complementarity == pytest.approx(0.0, abs=1e-14)
    assert measured.is_optimal()
    assert measured.worst() == "none"


def test_the_five_are_named_by_the_paper(golden):
    """§6's list at `paper.tex:566`, field for field."""
    problem, z, found = golden
    measured = term.residuals(problem, z, found)
    for field in ("primal", "dual", "stationarity", "linear_complementarity", "cone_complementarity"):
        assert hasattr(measured, field)


def test_the_criterion_is_the_worst_of_the_five(golden):
    """One number to stop on, and it is a maximum rather than a sum."""
    problem, z, found = golden
    measured = term.residuals(problem, z, found)
    assert measured.largest == max(
        measured.primal,
        measured.dual,
        measured.stationarity,
        measured.linear_complementarity,
        measured.cone_complementarity,
    )


# ----------------------------------------------------------------------------------
# Each one can fail, and each fails for its own reason
# ----------------------------------------------------------------------------------


def test_a_primal_violation_is_reported(portfolio):
    """§14.1's Level 1, as one of the five."""
    problem = portfolio.to_socp()
    z = portfolio.socp_point(np.array([0.9, 0.05, 0.05]))
    zeros = Multipliers(y=np.zeros(1), nu=np.zeros(1), w=np.zeros(4))
    measured = term.residuals(problem, z, zeros)
    assert measured.primal > 0.0
    assert measured.worst() in {"primal feasibility", "stationarity"}


def test_a_wrong_signed_inequality_multiplier_is_a_dual_violation(portfolio):
    """`y >= 0` is dual feasibility, and a negative entry says so."""
    problem = portfolio.to_socp()
    z = portfolio.socp_point(np.array([0.5, 0.2, 0.3]))
    found = Multipliers(y=np.array([-1.0]), nu=np.zeros(1), w=np.zeros(4))
    assert term.residuals(problem, z, found).dual > 0.0


def test_a_multiplier_outside_the_cone_is_a_dual_violation(portfolio):
    """`w in Q` is the other half, and it is a cone membership rather than a sign."""
    problem = portfolio.to_socp()
    z = portfolio.socp_point(np.array([0.5, 0.2, 0.3]))
    outside = Multipliers(y=np.zeros(1), nu=np.zeros(1), w=np.array([0.1, 1.0, 0.0, 0.0]))
    assert term.residuals(problem, z, outside).dual > 0.0


def test_a_stationarity_violation_is_reported(portfolio):
    """§14.2's Level 2, as one of the five."""
    problem = portfolio.to_socp()
    z = portfolio.socp_point(np.array([0.5, 0.2, 0.3]))
    zeros = Multipliers(y=np.zeros(1), nu=np.zeros(1), w=np.zeros(4))
    assert term.residuals(problem, z, zeros).stationarity > 0.0


def test_linear_complementarity_fails_on_an_inactive_row_with_a_multiplier(portfolio):
    """`y_i * (a_i.T @ x - b_i) = 0` -- a positive multiplier on a slack row breaks it.

    The residual that catches a working set disagreeing with its own multipliers: the row
    is not active, so its multiplier should be zero, and complementarity is where that
    shows up rather than in stationarity.
    """
    problem = portfolio.to_socp()
    z = portfolio.socp_point(np.array([0.2, 0.3, 0.5]))
    assert float((problem.b - problem.A @ z)[0]) > 0.1, "the row is slack here"
    found = Multipliers(y=np.array([1.0]), nu=np.zeros(1), w=np.zeros(4))
    assert term.residuals(problem, z, found).linear_complementarity > 0.0


def test_cone_complementarity_fails_when_the_dual_is_not_on_the_complementary_face(portfolio):
    """`w.T @ s = 0` is a stronger condition than it looks.

    With `w in Q` and `s in Q` both self-dual the inner product is non-negative, so
    demanding it vanish forces the pair onto complementary faces. A `w` inside the cone
    rather than on the ray #13 derives fails it, even though it is perfectly dual feasible.
    """
    problem = portfolio.to_socp()
    z = portfolio.socp_point(np.array([0.5, 0.2, 0.3]))
    interior = Multipliers(y=np.zeros(1), nu=np.zeros(1), w=np.array([1.0, 0.0, 0.0, 0.0]))
    measured = term.residuals(problem, z, interior)
    assert measured.dual == 0.0, "this w is in Q"
    assert measured.cone_complementarity > 0.0, "but not complementary to the slack"


def test_the_complementary_ray_satisfies_cone_complementarity(portfolio):
    """And the ray #13 derives does satisfy it, which is the other half of the claim."""
    problem = portfolio.to_socp()
    z = portfolio.socp_point(np.array([0.5, 0.2, 0.3]))
    covector = tangent.tangent_covector(problem.cone_slack(z))
    on_ray = Multipliers(y=np.zeros(1), nu=np.zeros(1), w=2.0 * covector)
    assert term.residuals(problem, z, on_ray).cone_complementarity == pytest.approx(0.0, abs=1e-14)


def test_every_w_is_complementary_at_the_apex():
    """Because the slack is zero, so the condition says nothing -- which is why #24 exists.

    Recorded as a test because it is the reason the apex needs a normal-cone branch: SOC
    complementarity, the condition that pins `w` to a ray at a smooth boundary point,
    imposes nothing at all here.
    """
    portfolio = MeanStdPortfolio.unconstrained(mu=np.ones(3), Sigma=np.ones((3, 3)), lam=1.0).with_inequalities(
        np.vstack([np.eye(3), -np.eye(3)]), np.ones(6)
    )
    problem = portfolio.to_socp()
    z = portfolio.socp_point(np.array([1.0, -1.0, 0.0]))
    np.testing.assert_allclose(problem.cone_slack(z), 0.0, atol=1e-15)
    for head in (0.5, 1.0, 5.0):
        anything = Multipliers(y=np.zeros(6), nu=np.zeros(0), w=np.array([head, 0.1]))
        assert term.residuals(problem, z, anything).cone_complementarity == pytest.approx(0.0)


# ----------------------------------------------------------------------------------
# The criterion, and its report
# ----------------------------------------------------------------------------------


def test_the_worst_residual_is_named(portfolio):
    """The first thing a diagnosis wants, and the argument for five numbers over one."""
    problem = portfolio.to_socp()
    z = portfolio.socp_point(np.array([0.5, 0.2, 0.3]))
    found = Multipliers(y=np.array([-5.0]), nu=np.zeros(1), w=np.zeros(4))
    assert term.residuals(problem, z, found).worst() in {"dual feasibility", "stationarity"}


def test_the_tolerance_is_a_parameter(golden):
    """§14.3 asks for "a prescribed tolerance", so it is prescribed rather than fixed."""
    problem, z, found = golden
    perturbed = Multipliers(y=found.y, nu=found.nu, w=found.w + 1e-7)
    measured = term.residuals(problem, z, perturbed)
    assert not measured.is_optimal(tolerance=1e-12)
    assert measured.is_optimal(tolerance=1e-4)


def test_the_residuals_are_relative_to_their_data():
    """A portfolio in percent and one in millions are held to the same standard."""
    natural = families.box(6, seed=0)
    scaled = families.badly_scaled(6, seed=0)
    zeros = Multipliers(
        y=np.zeros(natural.problem.num_inequalities),
        nu=np.zeros(natural.problem.num_equalities),
        w=np.zeros(natural.problem.cone.dim),
    )
    assert term.residuals(natural.problem, natural.witness, zeros).primal == pytest.approx(0.0, abs=1e-9)
    assert term.residuals(scaled.problem, scaled.witness, zeros).primal == pytest.approx(0.0, abs=1e-9)


def test_the_report_shows_all_five(golden):
    """One line, five numbers -- what a solver log entry carries."""
    problem, z, found = golden
    rendered = str(term.residuals(problem, z, found))
    for label in ("primal=", "dual=", "stat=", "comp_lin=", "comp_soc="):
        assert label in rendered


def test_the_complementarity_shortcut_reports_the_worse_of_the_two(portfolio):
    """For a report that wants four numbers rather than five."""
    problem = portfolio.to_socp()
    z = portfolio.socp_point(np.array([0.2, 0.3, 0.5]))
    found = Multipliers(y=np.array([1.0]), nu=np.zeros(1), w=np.zeros(4))
    measured = term.residuals(problem, z, found)
    assert measured.complementarity == max(measured.linear_complementarity, measured.cone_complementarity)


def test_the_residuals_check_their_blocks(portfolio):
    """A multiplier vector of the wrong length is a bug, caught where it arrives."""
    problem = portfolio.to_socp()
    z = portfolio.socp_point(np.array([0.5, 0.2, 0.3]))
    with pytest.raises(ProblemError, match="expected 1 entries"):
        term.residuals(problem, z, Multipliers(y=np.zeros(5), nu=np.zeros(1), w=np.zeros(4)))


# ----------------------------------------------------------------------------------
# Level 3 at a real optimum
# ----------------------------------------------------------------------------------


def test_level_3_holds_at_a_reference_optimum(portfolio):
    """The recovered multipliers at a solved instance certify optimality.

    To the reference solver's accuracy rather than to machine precision -- the point came
    from Clarabel, and the residuals are as small as that point is optimal. The same
    observation `test_instrumentation.py` makes about Level 2, and it bites harder here:
    stationarity comes out around `6e-6` on a point whose own duality gap is `1e-8`,
    because recovering multipliers at an approximate point amplifies its error. The exact
    Level 3 check is the hand-solved one at the top of this file, where every residual is
    zero to machine precision.
    """
    problem = portfolio.to_socp()
    z = reference.solve_reference(problem).z
    working_set = WorkingSet.empty(problem)
    for index in updates.activation_candidates(problem, z, working_set, tolerance=1e-6):
        working_set = updates.add_inequality(working_set, index)
    working_set = updates.activate_cones(problem, z, working_set)
    found = mult.from_direction(problem, working_set, z, kkt.direction(problem, working_set, z))
    measured = term.residuals(problem, z, found)
    assert measured.is_optimal(tolerance=1e-4)
    assert measured.primal < 1e-7, "the point itself is feasible to much better than that"
