"""Diagonal scaling across §13.3's five targets, and where it does and does not help.

The executable half of issue #28. Its "done when" has two parts and they come out
differently, which is the interesting result in this file:

* *all five targets are scaled* -- checked against §13.3's list, name by name.
* *conditioning measurably improves on the ill-conditioned-covariance robustness
  instances* -- **it does not**, and `test_the_ill_conditioned_family_needs_no_scaling`
  measures why. The covariance's condition number is `1e10` and the assembled KKT matrix's
  is about `10`, because the tangent representation compresses `L` into a single row and a
  single row has no spectrum. The equilibration correctly returns the identity, there being
  nothing to equilibrate.

  So the issue's premise is false, and for a reason worth having: insensitivity of the KKT
  system to covariance conditioning is a *property of the method*, and §17
  (`paper.tex:1345`) says a characterization of "how their geometry differs fundamentally
  from polyhedral active-set methods" would be a valuable result.

Where scaling does earn its keep is unit mismatch -- which, read again, is what §13.3's
five targets are all about: variables, constraints, returns, SOC variables. On
`badly_scaled`, whose weights are in basis points and whose risk variable is in millions,
the KKT condition number falls thirteen orders of magnitude. That is
`test_scaling_fixes_a_unit_mismatch`, and it is the demonstration the "done when" was
after.

Everything else here is correctness: the scaled problem must be *the same problem*, so its
solution, its multipliers, its working set and its tangent geometry all have to correspond
exactly.
"""

import numpy as np
import pytest

from cosa import Multipliers, ProblemError, Scaling, WorkingSet
from cosa.active_set import multipliers as mult
from cosa.active_set import updates
from cosa.experiments import portfolio as families
from cosa.experiments import reference
from cosa.geometry import soc, tangent
from cosa.linear_algebra import kkt
from cosa.linear_algebra import scaling as sc


@pytest.fixture
def instance():
    """A box-constrained instance with all three blocks and an active cone."""
    return families.box(6, seed=0)


def active_set_at(problem, z, tolerance=1e-6):
    """The working set that the §7 rules make active at ``z``."""
    working_set = WorkingSet.empty(problem)
    for index in updates.activation_candidates(problem, z, working_set, tolerance=tolerance):
        working_set = updates.add_inequality(working_set, index)
    return updates.activate_cones(problem, z, working_set)


# ----------------------------------------------------------------------------------
# §13.3's five targets
# ----------------------------------------------------------------------------------

FIVE_TARGETS = (
    "portfolio variables",
    "covariance matrices",
    "linear constraints",
    "expected returns",
    "SOC variables",
)


@pytest.mark.parametrize("target", FIVE_TARGETS)
def test_every_named_target_is_scaled(instance, target):
    """§13.3's list, checked name by name -- issue #28's first "done when"."""
    described = sc.equilibrate(instance.problem).describe()
    assert target in described


def test_the_description_reports_the_spread_per_target(instance):
    """Coverage plus magnitude, so "all five are scaled" is checkable rather than claimed."""
    described = sc.equilibrate(families.badly_scaled(6, seed=0).problem).describe()
    assert described.count("scale(s) in") == len(FIVE_TARGETS)


def test_a_target_the_instance_lacks_is_reported_as_absent():
    """A linear program has no cone to scale, and the description says so rather than lying."""
    from cosa import SOCP

    problem = SOCP.unconstrained(np.array([1.0, 1.0])).add_inequalities([[1.0, 1.0]], [1.0])
    described = sc.equilibrate(problem).describe()
    assert "none in this instance" in described


def test_the_covariance_is_reached_through_the_cone_s_tail_rows(instance):
    """No separate covariance knob, because `L` is the tail of `G` and scaling it suffices.

    Asserted as arithmetic: the scaled problem's implied covariance is
    `kappa^2 * D @ Sigma @ D`, which is what scaling those rows and columns means.
    """
    problem = instance.problem
    scaling = Scaling(
        variables=np.append(np.full(instance.num_assets, 2.0), 3.0),
        inequalities=np.ones(problem.num_inequalities),
        equalities=np.ones(problem.num_equalities),
        cones=np.array([5.0]),
        objective=1.0,
    )
    scaled = scaling.apply(problem)
    factor, scaled_factor = problem.G[1:], scaled.G[1:]
    expected = 5.0 * factor * scaling.variables
    np.testing.assert_allclose(scaled_factor, expected)
    np.testing.assert_allclose(
        scaled_factor.T @ scaled_factor,
        25.0 * np.diag(scaling.variables) @ (factor.T @ factor) @ np.diag(scaling.variables),
        atol=1e-12,
    )


def test_the_objective_scale_covers_both_mu_and_lambda(instance):
    """Scaling `mu` alone would change the problem; scaling all of `c` does not."""
    problem = instance.problem
    scaled = sc.equilibrate(problem).apply(problem)
    ratios = scaled.c[np.abs(problem.c) > 0] / problem.c[np.abs(problem.c) > 0]
    assert np.allclose(ratios / ratios[0], 1.0), "one factor, applied to every entry of c"


# ----------------------------------------------------------------------------------
# The cone's one-scale-per-block constraint
# ----------------------------------------------------------------------------------


def test_the_cone_carries_one_scale_per_factor(instance):
    """Not per row: `||y|| <= t` is not preserved by scaling head and tail differently."""
    scaling = sc.equilibrate(instance.problem)
    assert scaling.cones.shape == (len(instance.problem.cone),)
    assert scaling.cones.shape != (instance.problem.cone.dim,)


def test_scaling_a_block_by_one_factor_preserves_membership():
    """Which is the whole licence for scaling a conic block at all: `Q` is a cone."""
    inside = np.array([2.0, 1.0, 1.0])
    outside = np.array([1.0, 2.0, 1.0])
    for kappa in (1e-6, 0.5, 1e6):
        assert soc.is_member(kappa * inside)
        assert not soc.is_member(kappa * outside)


def test_scaling_head_and_tail_differently_would_break_the_constraint():
    """The counterfactual, stated so the design constraint is not merely asserted."""
    boundary = np.array([1.0, 1.0])
    assert soc.is_boundary(boundary)
    assert not soc.is_member(boundary * np.array([0.5, 1.0])), "a different constraint"
    assert soc.is_interior(boundary * np.array([2.0, 1.0])), "also a different constraint"


def test_the_tangent_direction_is_scale_invariant(instance):
    """`u` is unchanged, so the scaling cannot move the working set.

    A scaling that changed `u` would change which directions eq. (3) admits, and then "the
    same problem, better conditioned" would be false.
    """
    problem = instance.problem
    z = reference.solve_reference(problem).z
    scaling = sc.equilibrate(families.badly_scaled(6, seed=0).problem)
    scaled_problem = families.badly_scaled(6, seed=0).problem
    scaled = scaling.apply(scaled_problem)

    original = tangent.unit_tail(scaled_problem.cone_slack(z / np.append(np.full(6, 1e-4), 1e6)))
    after = tangent.unit_tail(scaled.cone_slack(scaling.scale_point(z / np.append(np.full(6, 1e-4), 1e6))))
    np.testing.assert_allclose(after, original, atol=1e-9)


def test_the_working_set_is_unchanged_by_scaling(instance):
    """Same rows active, same cone status -- the set is over row indices, not values."""
    problem = instance.problem
    z = reference.solve_reference(problem).z
    scaling = sc.equilibrate(problem)
    scaled = scaling.apply(problem)
    assert active_set_at(problem, z) == active_set_at(scaled, scaling.scale_point(z))


# ----------------------------------------------------------------------------------
# The scaled problem is the same problem
# ----------------------------------------------------------------------------------


def test_the_point_maps_back_and_forth(instance):
    """`scale_point` and `unscale_point` are inverses, so nothing is lost."""
    scaling = sc.equilibrate(instance.problem)
    z = instance.witness
    np.testing.assert_allclose(scaling.unscale_point(scaling.scale_point(z)), z, atol=1e-12)


def test_a_feasible_point_stays_feasible(instance):
    """Both directions, since a scaling that changed feasibility would not be a scaling."""
    problem = instance.problem
    scaling = sc.equilibrate(problem)
    scaled = scaling.apply(problem)
    point = scaling.scale_point(instance.witness)
    assert np.all(scaled.A @ point <= scaled.b + 1e-9)
    np.testing.assert_allclose(scaled.E @ point, scaled.d, atol=1e-9)
    assert soc.is_member_of_product(scaled.cone, scaled.cone_slack(point), tolerance=1e-9)


def test_the_optimum_corresponds_under_the_scaling():
    """Solve the scaled problem, unscale, and get the original's answer.

    The property that licences solving a scaled instance at all -- checked on the badly
    scaled family, where the scaling is not the identity and so the claim has content.

    The tolerance is loose on purpose, and the looseness is the subject of the next test:
    the *unscaled* instance has a condition number of `2e14`, so the answer it is being
    compared against is itself only good to about four digits. Comparing to `1e-9` here
    would be asserting that a badly conditioned solve is accurate, which it is not.
    """
    instance = families.badly_scaled(6, seed=0)
    problem = instance.problem
    scaling = sc.equilibrate(problem)
    scaled = scaling.apply(problem)

    direct = reference.solve_reference(problem)
    indirect = reference.solve_reference(scaled)
    np.testing.assert_allclose(scaling.unscale_point(indirect.z), direct.z, rtol=1e-3)
    assert indirect.objective / scaling.objective == pytest.approx(direct.objective, rel=1e-3)


def test_pre_scaling_does_not_help_the_reference_solver():
    """A finding, recorded so #34 does not expect scaling to move the reference numbers.

    The intuitive claim -- scale the problem, get a better answer -- is false here, and
    measurably so: solving `badly_scaled` through `equilibrate` lands about an order of
    magnitude *further* from the truth than solving it directly. Clarabel equilibrates
    internally, so it has already done this work; pre-scaling only adds a second change of
    units and its rounding.

    Which locates what :mod:`cosa.linear_algebra.scaling` is actually for. It is not a
    pre-processing step to make an oracle happier -- it is conditioning for *COSA's own*
    dense factorization, where nothing else is doing the job. `test_scaling_fixes_a_unit_
    mismatch` is the claim that survives measurement; this one is the claim that does not.
    """
    assets = 6
    truth = reference.solve_reference(families.box(assets, seed=0).problem).z[:assets]
    problem = families.badly_scaled(assets, seed=0).problem
    units = np.full(assets, 1e-4)

    direct = reference.solve_reference(problem).z[:assets] * units
    scaling = sc.equilibrate(problem)
    through_scaling = scaling.unscale_point(reference.solve_reference(scaling.apply(problem)).z)[:assets] * units

    assert float(np.abs(direct - truth).max()) < 1e-4, "both paths land close"
    assert float(np.abs(through_scaling - truth).max()) < 1e-3
    assert float(np.abs(through_scaling - truth).max()) > float(np.abs(direct - truth).max()), (
        "and the scaled path is not the more accurate one"
    )


def test_the_multipliers_correspond_under_the_scaling():
    """`y = R y_hat / sigma`, `nu = S nu_hat / sigma`, `w = kappa w_hat / sigma`.

    Derived in the module docstring and checked here against the multipliers the original
    problem produces, so the derivation is not taken on trust.
    """
    instance = families.badly_scaled(6, seed=0)
    problem = instance.problem
    z = reference.solve_reference(problem).z
    working_set = active_set_at(problem, z)
    original = mult.from_direction(problem, working_set, z, kkt.direction(problem, working_set, z))

    scaling = sc.equilibrate(problem)
    scaled = scaling.apply(problem)
    point = scaling.scale_point(z)
    scaled_multipliers = mult.from_direction(scaled, working_set, point, kkt.direction(scaled, working_set, point))
    recovered = scaling.unscale_multipliers(problem, scaled_multipliers)

    np.testing.assert_allclose(recovered.y, original.y, atol=1e-6)
    np.testing.assert_allclose(recovered.nu, original.nu, atol=1e-6)
    np.testing.assert_allclose(recovered.w, original.w, atol=1e-6)


def test_unscaling_preserves_dual_feasibility():
    """Every factor is positive, so `y >= 0` and `w in Q` survive in both directions."""
    problem = families.box(4, seed=0).problem
    scaling = sc.equilibrate(problem)
    feasible = Multipliers(
        y=np.ones(problem.num_inequalities),
        nu=np.zeros(problem.num_equalities),
        w=np.concatenate([[2.0], np.zeros(problem.cone.dim - 1)]),
    )
    recovered = scaling.unscale_multipliers(problem, feasible)
    assert recovered.inequality_violation() == 0.0
    assert mult.dual_cone_violation(problem, recovered) == (0.0,)


def test_the_identity_scaling_changes_nothing(instance):
    """A baseline to measure against, and the answer when there is nothing to fix."""
    problem = instance.problem
    scaled = sc.identity(problem).apply(problem)
    for block in ("c", "A", "b", "E", "d", "G", "h"):
        np.testing.assert_array_equal(getattr(scaled, block), getattr(problem, block))


# ----------------------------------------------------------------------------------
# Where scaling helps, and where it has nothing to do
# ----------------------------------------------------------------------------------


def test_scaling_fixes_a_unit_mismatch():
    """Issue #28's "done when", on the instance the premise actually holds for.

    Weights in basis points and a risk variable in millions -- an ordinary modelling
    slip, and exactly what §13.3's five targets are about. Thirteen orders of magnitude
    off the KKT condition number.
    """
    instance = families.badly_scaled(6, seed=0)
    problem = instance.problem
    z = reference.solve_reference(problem).z
    working_set = active_set_at(problem, z)

    before = sc.kkt_condition(problem, working_set, z)
    scaling = sc.equilibrate(problem)
    after = sc.kkt_condition(scaling.apply(problem), working_set, scaling.scale_point(z))

    assert before > 1e10, "the unit mismatch really does wreck the conditioning"
    assert after < 1e3, "and the scaling really does fix it"
    assert before / after > 1e6


def test_the_ill_conditioned_family_needs_no_scaling():
    """Issue #28's premise, measured -- and it does not hold. This is the finding.

    `cond(Sigma) = 1e10` by construction, and the assembled KKT matrix is *well
    conditioned anyway*. The tangent representation puts `L` into the system as one row,
    `g_0 - u.T @ L`, and one row has no spectrum to inherit. Every column maximum of the
    stacked constraint matrix is exactly `1` -- the bounds are `±1`, the budget is ones --
    so the equilibration is the identity, correctly.

    Recorded as an assertion rather than a comment so that if the method ever *does* start
    inheriting the covariance's conditioning, this test fails and says so.
    """
    instance = families.ill_conditioned(8, condition=1e10, seed=0)
    problem = instance.problem
    z = reference.solve_reference(problem).z
    working_set = active_set_at(problem, z)

    assert np.linalg.cond(instance.portfolio.Sigma) > 1e9
    assert sc.kkt_condition(problem, working_set, z) < 1e3, "the KKT matrix does not inherit it"

    scaling = sc.equilibrate(problem)
    np.testing.assert_allclose(scaling.variables, 1.0)
    np.testing.assert_allclose(scaling.cones, 1.0)


def test_the_conic_row_is_why_the_covariance_does_not_reach_the_system():
    """One row of `W_k`, whatever the rank of `L`.

    The mechanism behind the finding above, isolated: an active cone contributes a single
    row, so the covariance enters the saddle-point system only through the direction `u`
    and never through its spectrum.
    """
    instance = families.ill_conditioned(8, seed=0)
    problem = instance.problem
    z = reference.solve_reference(problem).z
    working_set = active_set_at(problem, z)
    matrix = kkt.working_set_matrix(problem, working_set, z)
    conic_rows = matrix.shape[0] - len(working_set.inequalities) - working_set.num_equalities
    assert conic_rows == 1
    assert np.linalg.cond(matrix @ matrix.T) < 1e3


def test_scaling_a_well_scaled_problem_is_harmless(instance):
    """It must not make things worse where there is nothing to fix."""
    problem = instance.problem
    z = reference.solve_reference(problem).z
    working_set = active_set_at(problem, z)
    scaling = sc.equilibrate(problem)
    before = sc.kkt_condition(problem, working_set, z)
    after = sc.kkt_condition(scaling.apply(problem), working_set, scaling.scale_point(z))
    assert after <= before * 1.5


# ----------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["variables", "inequalities", "cones"])
def test_a_non_positive_scale_is_rejected(instance, field):
    """A zero or negative scale is not invertible, so it is not a scaling."""
    problem = instance.problem
    scales = {
        "variables": np.ones(problem.num_variables),
        "inequalities": np.ones(problem.num_inequalities),
        "equalities": np.ones(problem.num_equalities),
        "cones": np.ones(len(problem.cone)),
        "objective": 1.0,
    }
    scales[field] = np.zeros_like(scales[field])
    with pytest.raises(ProblemError, match="must be positive"):
        Scaling(**scales)


@pytest.mark.parametrize("objective", [0.0, -1.0, np.inf])
def test_a_non_positive_objective_scale_is_rejected(instance, objective):
    """Scaling `c` by zero deletes the objective; by a negative number, maximizes it."""
    problem = instance.problem
    with pytest.raises(ProblemError, match="objective scale"):
        Scaling(
            variables=np.ones(problem.num_variables),
            inequalities=np.ones(problem.num_inequalities),
            equalities=np.ones(problem.num_equalities),
            cones=np.ones(len(problem.cone)),
            objective=objective,
        )


def test_a_scaling_of_the_wrong_shape_is_rejected(instance):
    """Applying a scaling built for another problem is a bug, caught on arrival."""
    other = sc.identity(families.box(4, seed=0).problem)
    with pytest.raises(ProblemError, match="but applied"):
        other.apply(instance.problem)
    with pytest.raises(ProblemError, match="but applied"):
        other.unscale_multipliers(instance.problem, Multipliers(y=np.zeros(8), nu=np.zeros(1), w=np.zeros(5)))


def test_a_negative_sweep_count_is_rejected(instance):
    """Zero rounds is the identity and is meaningful; negative is not."""
    with pytest.raises(ProblemError, match="sweep count"):
        sc.equilibrate(instance.problem, rounds=-1)


def test_zero_rounds_is_the_identity_up_to_the_objective(instance):
    """The sweep count is a knob, and its floor behaves."""
    scaling = sc.equilibrate(instance.problem, rounds=0)
    np.testing.assert_allclose(scaling.variables, 1.0)
    np.testing.assert_allclose(scaling.cones, 1.0)


def test_a_variable_in_no_constraint_is_left_alone():
    """A column of zeros has no conditioning to improve, and dividing by it would be worse."""
    from cosa import SOCP

    problem = SOCP.unconstrained(np.array([1.0, 1.0])).add_inequalities([[1.0, 0.0]], [1.0])
    scaling = sc.equilibrate(problem)
    assert np.all(np.isfinite(scaling.variables))
    assert scaling.variables[1] == 1.0


def test_the_condition_number_is_of_the_matrix_that_gets_factorized(instance):
    """Not of the covariance -- which, per the finding above, is the whole distinction."""
    problem = instance.problem
    z = reference.solve_reference(problem).z
    working_set = active_set_at(problem, z)
    assert sc.kkt_condition(problem, working_set, z) == pytest.approx(
        np.linalg.cond(kkt.assemble(problem, working_set, z).matrix)
    )
