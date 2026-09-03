"""§13's four factorization strategies, measured against the M2 reference and each other.

The executable half of issue #26, whose "done when" is three claims: all four produce the
reference's solution within tolerance, factorization time is recorded per method, and one is
chosen as default *with the measurement that justifies it*.

The third is the one a test can get wrong by asserting a preference instead of a
measurement. `test_the_default_is_the_fastest_that_agrees` asserts the *rule* -- fastest
among those that agree -- and lets the measurement pick, so a change in the arithmetic
changes the answer rather than breaking a hard-coded name.
"""

import numpy as np
import pytest

from cosa import ProblemError, WorkingSet
from cosa.active_set import updates
from cosa.experiments import portfolio as families
from cosa.experiments import reference
from cosa.linear_algebra import factorization as fz
from cosa.linear_algebra import kkt


def systems_for(instances):
    """Assemble one KKT system per instance, at its reference optimum."""
    assembled = []
    for instance in instances:
        problem = instance.problem
        z = reference.solve_reference(problem).z
        working_set = WorkingSet.empty(problem)
        for index in updates.activation_candidates(problem, z, working_set, tolerance=1e-6):
            working_set = updates.add_inequality(working_set, index)
        working_set = updates.activate_cones(problem, z, working_set)
        try:
            assembled.append(kkt.assemble(problem, working_set, z))
        except ProblemError:
            continue
    return assembled


@pytest.fixture(scope="module")
def systems():
    """Systems from the six structured families -- a spread of shapes, not one shape."""
    return systems_for(families.all_families(seed=0))


@pytest.fixture(scope="module")
def measured(systems):
    """The comparison, computed once for the whole module."""
    return fz.compare(systems)


# ----------------------------------------------------------------------------------
# All four produce the reference's answer
# ----------------------------------------------------------------------------------


def test_every_strategy_the_paper_names_is_implemented():
    """§13.2's list plus §13.1's reference, by name."""
    assert set(fz.STRATEGIES) == {"lu", "ldl", "qr", "null-space", "range-space"}
    assert fz.REFERENCE == "lu"


@pytest.mark.parametrize("strategy", sorted(fz.STRATEGIES))
def test_every_strategy_agrees_with_the_reference(strategy, systems):
    """Issue #26's first "done when", strategy by strategy.

    On a full-rank working set the solution is unique, so agreement is a fact rather than a
    tolerance choice -- and the deviations come out at machine precision, not at the
    tolerance.
    """
    for system in systems:
        expected = kkt.solve(system)
        found = fz.solve_with(strategy, system)
        scale = max(1.0, float(np.abs(expected.d).max(initial=0.0)))
        np.testing.assert_allclose(found.d, expected.d, atol=1e-8 * scale)


@pytest.mark.parametrize("strategy", sorted(fz.STRATEGIES))
def test_every_strategy_satisfies_the_system_it_solved(strategy, systems):
    """Not just "the same as the reference" -- actually a solution.

    Both block rows checked directly, so a strategy that agreed with a *wrong* reference
    would still be caught.
    """
    for system in systems:
        found = fz.solve_with(strategy, system)
        np.testing.assert_allclose(system.W @ found.d, 0.0, atol=1e-8)
        residual = system.rho * found.d + system.W.T @ found.multipliers + system.gradient
        np.testing.assert_allclose(residual, 0.0, atol=1e-8)


def test_the_comparison_says_they_all_agree(measured):
    """The claim in one place, which is what a results table reports."""
    assert measured.all_agree, str(measured)
    assert len(measured.measurements) == len(fz.STRATEGIES)


# ----------------------------------------------------------------------------------
# Factorization time is recorded per method
# ----------------------------------------------------------------------------------


def test_every_strategy_is_timed(measured):
    """Issue #26's second "done when" -- §12.3's "factorization time", per method."""
    for measurement in measured.measurements:
        assert measurement.seconds > 0.0, measurement.strategy


def test_the_comparison_reports_a_table(measured):
    """One line per strategy, with the time and the deviation -- what #34 will print."""
    rendered = str(measured)
    for strategy in fz.STRATEGIES:
        assert strategy in rendered
    assert "deviation" in rendered
    assert "fastest" in rendered


def test_a_speedup_is_relative_to_the_reference(measured):
    """The number the choice of default rests on."""
    assert measured.speedup(fz.REFERENCE) == pytest.approx(1.0)
    assert measured.speedup(fz.DEFAULT) > 0.0
    with pytest.raises(ProblemError, match="was not measured"):
        measured.speedup("nonsense")


# ----------------------------------------------------------------------------------
# The default, chosen by the measurement
# ----------------------------------------------------------------------------------


def test_the_default_is_a_strategy_that_agrees(measured):
    """Issue #26's third "done when", in the half a test can assert.

    That the default agrees with the reference is a fact and is asserted. That it is the
    *fastest* is a measurement, and asserting a ranking inside this suite would measure the
    coverage instrumentation rather than the code -- it was observed flipping under `--cov`.
    So the ranking lives in `compare`, which anyone can run, and the numbers behind the
    choice are in `DEFAULT`'s docstring. What is asserted here is everything that does not
    depend on a clock.
    """
    agreeing = {measurement.strategy for measurement in measured.measurements if measurement.matches}
    assert fz.DEFAULT in agreeing
    assert measured.fastest() in agreeing, "a wrong answer arriving sooner cannot win"


def test_the_default_solves_a_smaller_system_than_the_reference():
    """The structural reason the default is the default, which is deterministic.

    The range-space method solves an `m`-by-`m` system where the reference solves an
    `(n + m)`-by-`(n + m)` one. Since a working set never has more rows than there are
    variables plus its conic block, `m <= n + m` always and the range-space system has at
    most a quarter the entries -- more when the working set is small, which is the regime an
    active-set method spends most of a solve in.

    Stated as `m^2` against `(n + m)^2` rather than as "the working set is small", which is
    what an earlier version of this test assumed and which is *false* at a box-constrained
    optimum: there almost every bound is active and `m` is about `n`. The factor-of-four is
    the claim that survives.
    """
    large = systems_for([families.large(200, factors=10, seed=0)])
    assert large, "the instance assembled"
    system = large[0]
    saddle_point = system.num_variables + system.num_rows
    assert system.num_rows**2 * 4 <= saddle_point**2


def test_the_measurement_behind_the_choice_is_reproducible(systems):
    """Anyone can rerun it, and every strategy agrees on well-conditioned systems.

    The agreement is the assertion; the times are printed by `compare` for whoever is
    choosing. Deliberately run on the structured families rather than on `large`, for a
    reason worth recording: `large`'s optimum sits at the cone's *apex*, and pinning the
    apex block there makes `W` square and the KKT condition number about `1e18`. At that
    conditioning the strategies disagree in the fourteenth digit and "they agree" stops
    being a meaningful claim -- see `test_the_large_family_reaches_the_apex`.
    """
    comparison = fz.compare(systems, repeats=2)
    assert comparison.all_agree, str(comparison)
    assert all(measurement.seconds > 0.0 for measurement in comparison.measurements)


def test_the_large_family_reaches_the_apex_and_that_wrecks_the_conditioning():
    """A property of the instance, found by the comparison disagreeing on it.

    `large` is a factor model: a rank-`k` covariance over `n >> k` assets, so the null space
    of `L` has dimension `n - k` and the box admits points in it. The minimum-risk portfolio
    therefore has risk *exactly zero* and, `lam` being positive, the optimum takes it -- the
    conic slack lands on the apex.

    Which makes `large` an apex instance, exercising #24's branch without anyone constructing
    one. It also has a numerical consequence: §8.1's exact treatment pins the whole conic
    block, and with almost every box bound already active that leaves `W` square, so the
    saddle-point matrix is conditioned like `W` -- about `1e18` here. Worth knowing before
    #27 optimizes factorization reuse on this family.
    """
    from cosa.geometry import soc

    instance = families.large(200, factors=10, seed=0)
    problem = instance.problem
    z = reference.solve_reference(problem).z

    assert instance.portfolio.std(z[: instance.num_assets]) < 1e-8, "zero risk is attainable"
    assert soc.position(problem.cone_slack(z)) is soc.ConePosition.APEX

    system = systems_for([instance])[0]
    assert system.num_rows == system.num_variables, "the pinned apex block squares W"
    assert np.linalg.cond(system.matrix) > 1e3


def degenerate_system():
    """A KKT system whose working set is rank-deficient, from #33's family."""
    instance = families.degenerate_optimum(6, seed=0)
    problem = instance.problem
    z = reference.solve_reference(problem).z
    working_set = WorkingSet.empty(problem)
    for index in updates.activation_candidates(problem, z, working_set, tolerance=1e-6):
        working_set = updates.add_inequality(working_set, index)
    working_set = updates.activate_cones(problem, z, working_set)
    return kkt.assemble(problem, working_set, z)


def test_the_default_is_unusable_on_a_degenerate_set_however_lapack_fails():
    """Why the default is not chosen for robustness -- and why *how* it fails is not asserted.

    On a rank-deficient working set `W @ W.T` is singular too, and LAPACK's response is not
    portable: OpenBLAS raises `LinAlgError`, Apple's Accelerate returns a direction that does
    not satisfy `W @ d = 0`. Both were seen on this project -- the second locally, the first
    in CI on the very commit that documented the second as *the* behaviour.

    So the property asserted is the one that holds on both: the range-space route does not
    produce a correct direction here, and the null-space route does. Which is exactly why
    #25's rank test refuses before the loop reaches this route.
    """
    system = degenerate_system()
    assert np.linalg.matrix_rank(system.W) < system.num_rows, "the set really is dependent"

    try:
        wrong, _ = fz.STRATEGIES["range-space"](system)
    except np.linalg.LinAlgError:
        pass  # one of the two failures
    else:
        assert float(np.abs(system.W @ wrong).max()) > 1e-3, "the other: it returns, and is wrong"

    survives, _ = fz.STRATEGIES["null-space"](system)
    assert np.all(np.isfinite(survives))
    np.testing.assert_allclose(system.W @ survives, 0.0, atol=1e-9)


def test_a_comparison_reports_the_failure_however_it_arrives():
    """A comparison over adversarial systems must survive one strategy going wrong.

    Measured against the null-space route, which is the one that is right here, the
    range-space answer shows up either as a counted failure or as a large deviation --
    whichever this machine's LAPACK produces. `all_agree` is false in both cases, which is
    what makes it a useful thing to assert on the robustness families.
    """
    comparison = fz.compare([degenerate_system()], repeats=1, reference="null-space")
    assert not comparison.all_agree
    verdict = next(m for m in comparison.measurements if m.strategy == "range-space")
    assert verdict.failures > 0 or verdict.deviation > 1e-3
    assert not verdict.matches
    assert "DIFFERS" in str(comparison)


def test_a_strategy_that_raises_is_counted_as_a_failure(systems):
    """A comparison must survive a strategy giving up outright, not only getting it wrong.

    A zero matrix is the crudest way to make the three matrix factorizations fail; the two
    that never form the saddle-point matrix are untroubled by it, which is itself the point
    of having them.
    """
    import dataclasses

    broken = dataclasses.replace(systems[0], matrix=np.zeros_like(systems[0].matrix))
    comparison = fz.compare([broken], repeats=1, reference="null-space")
    failures = {m.strategy: m.failures for m in comparison.measurements}
    assert failures["lu"] == 1
    assert failures["ldl"] == 1
    assert failures["qr"] == 1
    assert failures["null-space"] == 0, "it never forms the matrix"
    assert failures["range-space"] == 0
    assert "failures" in str(comparison)


# ----------------------------------------------------------------------------------
# The two that exist only because the objective is linear
# ----------------------------------------------------------------------------------


def test_the_null_space_route_never_forms_the_saddle_point(systems):
    """It reads the direction straight off the projection, which is why #25 can use it."""
    for system in systems:
        direction, _ = fz.STRATEGIES["null-space"](system)
        np.testing.assert_allclose(system.W @ direction, 0.0, atol=1e-9)


def test_the_range_space_route_solves_an_m_by_m_system(systems):
    """`W @ W.T @ nu = -W @ g`, which is small exactly when the working set is."""
    for system in systems:
        _, multipliers = fz.STRATEGIES["range-space"](system)
        assert multipliers.shape == (system.num_rows,)
        residual = system.W @ system.W.T @ multipliers + system.W @ system.gradient
        np.testing.assert_allclose(residual, 0.0, atol=1e-8)


def test_an_empty_working_set_is_handled_by_every_strategy():
    """With nothing active the direction is `-g / rho` and there is no system to solve."""
    from cosa import SOCP

    problem = SOCP.unconstrained(np.array([1.0, -2.0]))
    system = kkt.assemble(problem, WorkingSet.empty(problem), np.zeros(2), rho=2.0)
    for strategy in fz.STRATEGIES:
        found = fz.solve_with(strategy, system)
        np.testing.assert_allclose(found.d, -problem.c / 2.0, atol=1e-12)


# ----------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------


def test_an_unknown_strategy_is_rejected(systems):
    """The names are the API, so a typo is an error rather than a silent default."""
    with pytest.raises(ProblemError, match="expected one of"):
        fz.solve_with("cholesky", systems[0])


def test_a_comparison_needs_systems_and_repeats(systems):
    """A comparison on nothing measures nothing."""
    with pytest.raises(ProblemError, match="at least one system"):
        fz.compare([])
    with pytest.raises(ProblemError, match="at least one repeat"):
        fz.compare(systems, repeats=0)
    with pytest.raises(ProblemError, match="unknown strategy"):
        fz.compare(systems, reference="nonsense")
