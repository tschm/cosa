"""§12.4's six adversarial families, and the diagnostic that proves each one is adversarial.

The executable half of issue #33, whose "done when" is that all six families generate on
demand and each is runnable as a standalone diagnostic. There is no solver yet -- #20 owns
that -- so "runnable as a standalone diagnostic" is `diagnose`, which measures the
pathology rather than asserting it.

Every family gets two tests: it exists and solves, and **it is actually pathological**.
The second is the one that earns its keep, and Wave 2 supplied the cautionary tale: the
box family's cap was once loose enough that its optimum was bit-for-bit identical to the
basic family's, so a family that was supposed to stress the active-set logic tested
nothing at all. A robustness family that stops being adversarial fails the same way and is
worth just as little, so each one asserts its own measured number here.

Two of the six are load-bearing for issues that have not started:
`test_the_degenerate_family_makes_the_kkt_system_singular` is what makes #25 demonstrable,
and `test_the_nearly_active_family_straddles_the_activation_tolerance` is what makes #29's
hysteresis a real problem rather than a hypothetical one.
"""

import numpy as np
import pytest

from cosa import ProblemError, SingularKktError, WorkingSet
from cosa.active_set import updates
from cosa.experiments import portfolio as families
from cosa.experiments import reference
from cosa.geometry import soc
from cosa.linear_algebra import kkt

# §12.4's list, in the order the paper gives it, paired with the generator that makes it.
SIX_FAMILIES = {
    "nearly redundant constraints": families.nearly_redundant,
    "highly correlated assets": families.highly_correlated,
    "ill-conditioned covariance matrices": families.ill_conditioned,
    "nearly active SOC constraints": families.nearly_active_cone,
    "degenerate optimal solutions": families.degenerate_optimum,
    "many simultaneously active portfolio bounds": families.many_active_bounds,
}


@pytest.fixture(params=sorted(SIX_FAMILIES))
def family(request):
    """One of §12.4's six, generated at its default size and seed."""
    return SIX_FAMILIES[request.param]()


# ----------------------------------------------------------------------------------
# Issue #33's "done when", over all six
# ----------------------------------------------------------------------------------


def test_all_six_families_exist():
    """§12.4 names six; `all_robustness` returns six, and they are these six."""
    assert len(SIX_FAMILIES) == 6
    generated = families.all_robustness(seed=0)
    assert len(generated) == 6
    assert len({instance.name for instance in generated}) == 6


def test_every_family_generates_on_demand(family):
    """The first half of the "done when": a callable, a seed, an instance."""
    assert family.problem.num_variables > 0
    assert family.num_assets > 0
    assert len(family.names.inequalities) == family.problem.num_inequalities


def test_every_family_is_feasible_at_its_witness(family):
    """Adversarial does not mean infeasible.

    A family that breaks its own witness's feasibility would be testing the generator
    rather than the solver.
    """
    z = family.witness
    problem = family.problem
    assert np.all(problem.A @ z <= problem.b + 1e-9)
    np.testing.assert_allclose(problem.E @ z, problem.d, atol=1e-9)
    assert soc.is_member_of_product(problem.cone, problem.cone_slack(z), tolerance=1e-9)


def test_every_family_still_has_an_answer(family):
    """A reference solver reaches an optimum on all six.

    §12.4's purpose is to "identify failure modes", which requires that the instance *have*
    a right answer to fail to find. A family so hostile that nothing can solve it would
    tell us nothing about COSA.
    """
    solution = reference.solve_reference(family.problem)
    assert solution.is_optimal, f"{family.name}: {solution.status}"


def test_every_family_can_be_diagnosed(family):
    """The second half of the "done when": a standalone diagnostic, per family."""
    at = family.witness if family.name.startswith("nearly-active") else None
    diagnosis = families.diagnose(family, at)
    assert diagnosis.instance == family.name
    assert diagnosis.num_assets if hasattr(diagnosis, "num_assets") else True
    assert family.name in str(diagnosis)


def test_the_diagnosis_reports_every_measured_pathology(family):
    """One line carrying conditioning, rank, active set and conic slack together.

    Which is what makes it a diagnostic rather than a number: the six families are hard in
    six different ways, and telling them apart needs all of it.
    """
    rendered = str(families.diagnose(family, family.witness))
    for label in ("cond(Sigma)", "rank=", "active=", "conic slack=", "risk="):
        assert label in rendered


# ----------------------------------------------------------------------------------
# Each family is actually the pathology it claims
# ----------------------------------------------------------------------------------


def test_the_nearly_redundant_family_has_a_near_duplicate_row():
    """Two rows differing by `gap`, and both active at the optimum.

    Not exactly duplicated: an exactly dependent row is caught by any rank test, and the
    hard case -- the one #25 has to get right -- is a row that differs in the tenth digit.
    """
    instance = families.nearly_redundant(8, gap=1e-9, seed=0)
    rows = instance.problem.A
    difference = np.abs(rows[0] - rows[-1]).max()
    assert difference == pytest.approx(1e-9, rel=1e-6)
    assert "near-duplicate" in instance.names.inequalities[-1]

    # Nearly dependent, not dependent: the pair has full rank and a smallest singular
    # direction of about `gap`. That distinction is the whole family, and it took #25's
    # rank detection to notice the first version of this generator did not have it -- it
    # perturbed the row's magnitude, leaving the two rows exactly parallel.
    from cosa.linear_algebra import rank as rk

    pair = rk.analyse(np.vstack([rows[0], rows[-1]]))
    assert pair.rank == 2, "independent"
    assert pair.smallest == pytest.approx(1e-9, rel=1e-3), "but only by `gap`"


def test_the_nearly_redundant_gap_is_a_knob():
    """#25 will need to sweep it: how small a gap can rank detection still see?"""
    wide = families.nearly_redundant(6, gap=1e-3, seed=0)
    narrow = families.nearly_redundant(6, gap=1e-12, seed=0)
    assert np.abs(wide.problem.A[0] - wide.problem.A[-1]).max() == pytest.approx(1e-3, rel=1e-6)
    assert np.abs(narrow.problem.A[0] - narrow.problem.A[-1]).max() == pytest.approx(1e-12, rel=1e-6)


def test_the_nearly_redundant_family_rejects_a_negative_gap():
    """A negative perturbation is not a perturbation."""
    with pytest.raises(ProblemError, match="non-negative"):
        families.nearly_redundant(6, gap=-1e-9, seed=0)


@pytest.mark.parametrize(("weight_unit", "risk_unit"), [(0.0, 1.0), (1.0, -1.0)])
def test_the_badly_scaled_family_rejects_a_non_positive_unit(weight_unit, risk_unit):
    """A unit of zero collapses the variable; a negative one flips its sign."""
    with pytest.raises(ProblemError, match="units are positive"):
        families.badly_scaled(6, weight_unit=weight_unit, risk_unit=risk_unit, seed=0)


def test_the_highly_correlated_family_is_nearly_rank_one():
    """Correlation `1 - eps` puts the smallest eigenvalue at `eps` times the largest.

    A *controlled* approach to rank deficiency, unlike the ill-conditioned family, which
    spreads the whole spectrum. The distinction matters because a factorization can survive
    one and not the other.
    """
    instance = families.highly_correlated(8, correlation=1.0 - 1e-8, seed=0)
    eigenvalues = np.linalg.eigvalsh(instance.portfolio.Sigma)
    assert eigenvalues.max() / eigenvalues.min() > 1e7
    correlation = instance.portfolio.Sigma / np.sqrt(
        np.outer(np.diag(instance.portfolio.Sigma), np.diag(instance.portfolio.Sigma))
    )
    off_diagonal = correlation[~np.eye(8, dtype=bool)]
    assert off_diagonal.min() > 1.0 - 1e-7


def test_the_highly_correlated_family_makes_the_tangent_ill_determined():
    """`||L @ x||` is tiny along the near-null direction, so `u` is decided by rounding.

    The gap between "refused" and "meaningless" that this family probes: #17's guard does
    not fire, because the tail has not vanished -- it has merely stopped carrying
    information.
    """
    instance = families.highly_correlated(8, correlation=1.0 - 1e-10, seed=0)
    factor = instance.portfolio.factor()
    null = np.linalg.svd(factor)[2][-1]
    assert float(np.linalg.norm(factor @ null)) < 1e-4
    assert float(np.linalg.norm(factor @ np.ones(8))) > 1e-2, "but not in every direction"


@pytest.mark.parametrize("correlation", [-0.1, 1.0, 1.5])
def test_the_correlated_family_rejects_an_impossible_correlation(correlation):
    """At exactly one it is rank one, not *nearly* singular -- use `rank=1` for that."""
    with pytest.raises(ProblemError, match="correlation in"):
        families.highly_correlated(6, correlation=correlation, seed=0)


def test_the_ill_conditioned_family_has_the_condition_number_it_claims():
    """Exact rather than approximate, so #28's work can be swept against it."""
    instance = families.ill_conditioned(8, condition=1e10, seed=0)
    assert np.linalg.cond(instance.portfolio.Sigma) == pytest.approx(1e10, rel=1e-3)


def test_the_ill_conditioned_family_is_full_rank_despite_its_conditioning():
    """The pathology is the spread, not a null space -- that is the correlated family's."""
    instance = families.ill_conditioned(8, seed=0)
    diagnosis = families.diagnose(instance)
    assert diagnosis.covariance_rank == 8
    assert diagnosis.covariance_condition > 1e9


def test_the_nearly_active_family_has_a_tiny_positive_conic_slack():
    """The witness sits inside the cone by `gap`, not on its boundary."""
    instance = families.nearly_active_cone(8, gap=1e-9, seed=0)
    slack = instance.problem.cone_slack(instance.witness)
    assert float(slack[0] - np.linalg.norm(slack[1:])) == pytest.approx(1e-9, rel=1e-6)
    assert families.diagnose(instance, instance.witness).conic_slack == pytest.approx(1e-9, rel=1e-6)


def test_the_nearly_active_family_straddles_the_activation_tolerance():
    """Strictly interior, and yet activated -- which is the oscillation §8.2 warns about.

    Both answers are correct, and that is exactly the problem: the geometry says interior,
    the activation rule says active, and without the separate on/off tolerances of #29 an
    iterate here can flip between the two states forever. This is the instance that makes
    that concrete.
    """
    instance = families.nearly_active_cone(8, gap=1e-9, seed=0)
    problem, witness = instance.problem, instance.witness

    assert soc.is_interior(problem.cone_slack(witness), tolerance=1e-12), "strictly inside"
    activated = updates.activate_cones(problem, witness, WorkingSet.empty(problem), tolerance=1e-8)
    assert activated.active_cones == (0,), "and yet active at §7.3's tolerance"

    not_activated = updates.activate_cones(problem, witness, WorkingSet.empty(problem), tolerance=1e-12)
    assert not_activated.active_cones == (), "the verdict flips with the tolerance"


def test_the_nearly_active_family_rejects_a_non_positive_gap():
    """At zero the cone is exactly active, which is the ordinary case and needs no family."""
    with pytest.raises(ProblemError, match="positive slack"):
        families.nearly_active_cone(8, gap=0.0, seed=0)


def test_the_degenerate_family_has_one_feasible_portfolio():
    """A cap of exactly `1/n` with a budget of one admits the equal-weight point alone."""
    instance = families.degenerate_optimum(8, seed=0)
    np.testing.assert_allclose(instance.portfolio.b[:8], 1.0 / 8)
    x = reference.solve_reference(instance.problem).z[:8]
    np.testing.assert_allclose(x, 1.0 / 8, atol=1e-7)


def test_the_degenerate_family_has_a_dependent_active_set():
    """`n` bounds plus the budget, of which only `n` are independent.

    The textbook definition of primal degeneracy: the primal solution is unique and the
    multipliers are not.
    """
    instance = families.degenerate_optimum(8, seed=0)
    diagnosis = families.diagnose(instance)
    assert diagnosis.active_rows == 8
    assert diagnosis.independent_rows == 9, "eight bounds and the budget"
    assert diagnosis.active_rank == 8, "but the budget row is their sum"
    assert diagnosis.is_primal_degenerate


def test_the_degenerate_family_makes_the_kkt_system_singular():
    """What #25 has to make survivable, and what #12 currently refuses to guess at.

    The whole reason this family is scheduled before M7. Until it existed, "rank detection
    and dependent-constraint removal" had nothing to detect -- and #12's refusal to return
    one arbitrary member of an infinite family of multipliers was a defensive branch nobody
    had seen fire.
    """
    instance = families.degenerate_optimum(6, seed=0)
    problem = instance.problem
    z = reference.solve_reference(problem).z
    working_set = WorkingSet.empty(problem)
    for index in updates.activation_candidates(problem, z, working_set, tolerance=1e-6):
        working_set = updates.add_inequality(working_set, index)
    working_set = updates.activate_cones(problem, z, working_set)

    with pytest.raises(SingularKktError, match="linearly dependent"):
        kkt.direction(problem, working_set, z)


def test_the_many_bounds_family_is_large_but_not_degenerate():
    """The contrast with the family above, and the reason both exist.

    An active-set method can be perfectly correct when the active set is merely *large* and
    fail when it is *dependent*. One family conflating the two would not say which.
    """
    instance = families.many_active_bounds(20, seed=0)
    diagnosis = families.diagnose(instance)
    assert diagnosis.active_rows >= 15, "most bounds bind"
    assert not diagnosis.is_primal_degenerate, "but the active set is independent"


def test_the_many_bounds_family_leaves_the_box_some_interior():
    """`slack > 1`, so the feasible set is not a single point."""
    instance = families.many_active_bounds(10, slack=1.1, seed=0)
    np.testing.assert_allclose(instance.problem.b[:10], 1.1 / 10)
    with pytest.raises(ProblemError, match="slack > 1"):
        families.many_active_bounds(10, slack=1.0, seed=0)


# ----------------------------------------------------------------------------------
# The diagnostic itself
# ----------------------------------------------------------------------------------


def test_a_diagnosis_can_be_taken_at_a_given_point():
    """For the family whose pathology is in the iterate rather than in the data."""
    instance = families.nearly_active_cone(8, seed=0)
    at_optimum = families.diagnose(instance)
    at_witness = families.diagnose(instance, instance.witness)
    assert at_witness.status == "given"
    assert at_witness.conic_slack > at_optimum.conic_slack


def test_a_diagnosis_of_a_healthy_instance_finds_nothing_wrong():
    """The measurement has to be able to say "fine", or it is not a measurement."""
    diagnosis = families.diagnose(families.box(8, seed=0))
    assert not diagnosis.is_primal_degenerate
    assert diagnosis.covariance_condition < 1e3
    assert diagnosis.covariance_rank == 8
    assert "DEGENERATE" not in str(diagnosis)


def test_the_diagnosis_reports_the_risk_at_the_point():
    """A tiny risk means the point is near the apex, where the tangent is ill-determined."""
    instance = families.box(6, seed=0)
    diagnosis = families.diagnose(instance)
    assert diagnosis.risk == pytest.approx(
        instance.portfolio.std(reference.solve_reference(instance.problem).z[:6]), rel=1e-9
    )


def test_the_diagnosis_validates_the_point_it_is_given():
    """A point of the wrong length is a bug, caught where it is handed over."""
    with pytest.raises(ProblemError, match="expected 9 entries"):
        families.diagnose(families.box(8, seed=0), np.zeros(3))


def test_every_robustness_family_is_reproducible_from_its_seed(family):
    """Same seed, same instance -- a pathology that moves cannot be regression-tested."""
    regenerated = SIX_FAMILIES[next(name for name, make in SIX_FAMILIES.items() if make().name == family.name)]()
    np.testing.assert_array_equal(regenerated.problem.A, family.problem.A)
    np.testing.assert_array_equal(regenerated.problem.c, family.problem.c)
    np.testing.assert_array_equal(regenerated.witness, family.witness)
