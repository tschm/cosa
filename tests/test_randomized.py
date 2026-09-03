"""The seeded random generator, and §16.3's cross-check run on every instance it makes.

The executable half of issue #32, whose "done when" is two claims: randomized problems
generate reproducibly from a seed, and each is cross-checked against the reference solver
within the prescribed tolerance. Both are asserted here twice over -- once on a fixed
sample of seeds, so a regression shows up at the same seed every time, and once as
property-based tests under the `property` marker, so the space is actually swept.

The two styles answer different questions and neither replaces the other. The fixed sample
is a regression net: seed 76 is the worst Clarabel-versus-SCS disagreement found in the
range, and it stays in the suite because it is the instance that would notice a tolerance
being tightened by mistake. The property tests are a search: they draw seeds the author
never looked at, which is the only way the *shape* axes -- dimension, rank, conditioning,
active-set structure -- get combined in ways nobody chose.

`hypothesis` shrinks a failure to a small counterexample and prints the seed that produced
it, which together with `RandomSpec.reproduce` is what issue #32 means by "reproducible
seeds recorded with any failure".
"""

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cosa import ConeStatus, ProblemError, WorkingSet
from cosa.active_set import updates
from cosa.experiments import randomized, reference
from cosa.geometry import soc

# Fixed seeds: a spread of shapes, plus the two that produced the largest reference-solver
# disagreement over the first two hundred. They are named here so a failure names them too.
SAMPLE = (0, 1, 2, 5, 7, 13, 42, 52, 76, 100)


def feasible(problem, z, tolerance=1e-9):
    """Is ``z`` feasible for every block of ``problem``, cone included?"""
    return (
        bool(np.all(problem.A @ z <= problem.b + tolerance))
        and bool(np.all(np.abs(problem.E @ z - problem.d) <= tolerance))
        and soc.is_member_of_product(problem.cone, problem.cone_slack(z), tolerance=tolerance)
    )


# ----------------------------------------------------------------------------------
# The specification a seed draws
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SAMPLE)
def test_a_spec_is_reproducible_from_its_seed(seed):
    """The same seed draws the same specification, field for field."""
    assert randomized.random_spec(seed) == randomized.random_spec(seed)


@pytest.mark.parametrize("seed", SAMPLE)
def test_a_spec_stays_inside_its_declared_ranges(seed):
    """Every axis is bounded, and `tight` is bounded by two things at once.

    The cap at `assets - 1` is the one worth checking: it keeps the tight rows plus the
    budget equality from pinning the witness down completely, which would make it a
    degenerate vertex -- a case worth testing deliberately, in #33 and #36, and not worth
    stumbling into here where it could not be told apart from a bug.
    """
    spec = randomized.random_spec(seed)
    assert 2 <= spec.assets <= randomized.MAX_ASSETS
    assert 1 <= spec.rank <= spec.assets
    assert 1.0 <= spec.condition <= randomized.MAX_CONDITION
    assert 0 <= spec.rows <= spec.assets
    assert 0 <= spec.tight <= min(spec.rows, spec.assets - 1)


def test_different_seeds_draw_different_specs():
    """Otherwise the generator would be a constant with extra steps."""
    specs = {randomized.random_spec(seed) for seed in range(50)}
    assert len(specs) > 40


def test_a_spec_prints_its_own_reproduction():
    """Issue #32's "reproducible seeds recorded with any failure", literally."""
    spec = randomized.random_spec(7)
    assert spec.reproduce() == "cosa.experiments.randomized.random_instance(7)"
    assert "s7" in spec.name
    assert spec.reproduce() in str(spec)


def test_a_spec_reports_whether_the_apex_is_reachable():
    """Rank deficiency is what puts the apex within reach at a nonzero portfolio."""
    deficient = [randomized.random_spec(seed) for seed in range(60)]
    assert any(spec.is_rank_deficient for spec in deficient)
    assert any(not spec.is_rank_deficient for spec in deficient)
    for spec in deficient:
        assert spec.is_rank_deficient == (spec.rank < spec.assets)


@pytest.mark.parametrize("seed", [-1, randomized.MAX_SEED + 1])
def test_a_seed_outside_the_range_is_rejected(seed):
    """The range is what makes a seed portable across runs and platforms."""
    with pytest.raises(ProblemError, match="expected a seed"):
        randomized.random_spec(seed)


def test_a_maximum_below_two_assets_is_rejected():
    """A one-asset portfolio has a budget equality and nothing to choose."""
    with pytest.raises(ProblemError, match="at least 2 assets"):
        randomized.random_spec(0, max_assets=1)


# ----------------------------------------------------------------------------------
# The instance built around a witness
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SAMPLE)
def test_an_instance_is_feasible_at_its_witness(seed):
    """Feasible because a feasible point was chosen before the constraints were."""
    instance = randomized.random_instance(seed)
    assert feasible(instance.problem, instance.witness)


@pytest.mark.parametrize("seed", SAMPLE)
def test_the_drawn_active_set_structure_is_the_actual_one(seed):
    """Exactly `tight` random rows are active at the witness -- no more, no fewer.

    The axis a naive generator misses. Random rows with random right-hand sides are almost
    surely all slack, so without this the interesting states -- several constraints active
    at once -- would never be generated, and the test suite would never know.
    """
    spec = randomized.random_spec(seed)
    instance = randomized.random_instance(seed)
    slack = instance.problem.b - instance.problem.A @ instance.witness
    assert int((np.abs(slack) <= 1e-12).sum()) == spec.tight


@pytest.mark.parametrize("seed", SAMPLE)
def test_the_witness_is_interior_to_the_long_only_bounds(seed):
    """No weight at zero, so a long-only bound is never accidentally active.

    Otherwise the drawn structure and the actual one would drift apart, and the test above
    would be measuring the Dirichlet draw rather than the generator.
    """
    instance = randomized.random_instance(seed)
    assert instance.witness[: instance.num_assets].min() > 1e-6


@pytest.mark.parametrize("seed", SAMPLE)
def test_an_instance_matches_its_spec(seed):
    """The shape actually built is the shape drawn: assets, rank, rows and names."""
    spec = randomized.random_spec(seed)
    instance = randomized.random_instance(seed)
    assert instance.num_assets == spec.assets
    assert instance.portfolio.factor().shape[0] == spec.rank
    assert instance.problem.num_inequalities == spec.assets + spec.rows
    assert instance.problem.cone.cones[0].dim == spec.rank + 1
    assert instance.name == spec.name
    assert len(instance.names.inequalities) == instance.problem.num_inequalities


@pytest.mark.parametrize("seed", SAMPLE)
def test_an_instance_is_reproducible_from_its_seed(seed):
    """Bit for bit, which is what makes a recorded seed worth recording."""
    first, second = randomized.random_instance(seed), randomized.random_instance(seed)
    np.testing.assert_array_equal(first.problem.c, second.problem.c)
    np.testing.assert_array_equal(first.problem.A, second.problem.A)
    np.testing.assert_array_equal(first.problem.b, second.problem.b)
    np.testing.assert_array_equal(first.problem.G, second.problem.G)
    np.testing.assert_array_equal(first.witness, second.witness)


def test_a_rank_deficient_draw_puts_the_apex_within_reach():
    """The reason rank is an axis: it is how the apex branch gets exercised at all.

    Nobody constructs an apex instance here -- the generator produces them, because a
    singular covariance has a null space and the witness can be moved along it.
    """
    seed = next(s for s in range(100) if randomized.random_spec(s).rank == 1)
    instance = randomized.random_instance(seed)
    factor = instance.portfolio.factor()
    null = np.linalg.svd(factor)[2][1:]
    direction = null[0]
    at_apex = instance.portfolio.socp_point(direction * 0.0)
    assert soc.position(instance.problem.cone_slack(at_apex)) is soc.ConePosition.APEX
    assert np.linalg.norm(factor @ direction) < 1e-12, "the null space is nonempty"


def test_an_instance_is_bounded_because_it_lives_on_the_simplex():
    """The budget equality plus the long-only bounds make the feasible set compact."""
    instance = randomized.random_instance(3)
    np.testing.assert_allclose(instance.portfolio.E, np.ones((1, instance.num_assets)))
    np.testing.assert_allclose(instance.portfolio.A[: instance.num_assets], -np.eye(instance.num_assets))


def test_a_randomized_instance_is_interchangeable_with_a_structured_one():
    """Same type, so the cross-check, the names and the working set all just work."""
    instance = randomized.random_instance(7)
    z = reference.solve_reference(instance.problem).z
    working_set = updates.activate_cones(instance.problem, z, WorkingSet.empty(instance.problem))
    assert working_set.status(0) is ConeStatus.TANGENT
    assert "risk" in working_set.describe(instance.names)


# ----------------------------------------------------------------------------------
# §16.3's cross-check on the fixed sample
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SAMPLE)
def test_every_sampled_instance_is_cross_checked(seed):
    """Issue #32's "done when": solvable, and the references agree within tolerance."""
    instance = randomized.random_instance(seed)
    check = reference.cross_check(instance.problem, name=instance.name)
    assert check.all_optimal, str(check)
    assert check.agrees, str(check)


def test_the_worst_sampled_disagreement_is_still_inside_the_tolerance():
    """Seed 76 is the largest Clarabel-versus-SCS gap in the first two hundred draws.

    Kept as a regression net around :data:`cosa.experiments.reference.BACKEND_ACCURACY`:
    the gap here is about 1e-5, which is why SCS is held to 1e-4 rather than to §16.3's
    1e-6. Tighten that figure and this test says so.
    """
    check = reference.cross_check(randomized.random_instance(76).problem, name="s76")
    if len(check.solutions) < 2:
        pytest.skip("only one reference solver is installed, so there is nothing to disagree")
    assert check.gap > reference.OBJECTIVE_TOLERANCE, "this instance is why the tolerance widens"
    assert check.agrees, str(check)
    assert check.tolerance > check.requested_tolerance


# ----------------------------------------------------------------------------------
# The cross-check machinery itself
# ----------------------------------------------------------------------------------


def test_the_cross_check_compares_the_references_with_each_other_when_given_no_objective():
    """The strongest available claim before #20 exists: reference against reference."""
    check = reference.cross_check(randomized.random_instance(1).problem, name="pairwise")
    assert check.objective is None
    assert len(check.solutions) == len(reference.available_solvers())
    assert check.agrees, str(check)


def test_the_cross_check_compares_an_objective_when_given_one():
    """The seam #20 plugs into: pass COSA's objective and nothing else changes."""
    instance = randomized.random_instance(1)
    truth = reference.solve_reference(instance.problem).objective
    assert reference.cross_check(instance.problem, truth, name="cosa").agrees
    assert not reference.cross_check(instance.problem, truth + 1.0, name="wrong").agrees


def test_the_tolerance_widens_to_the_least_accurate_solver():
    """A comparison cannot be tighter than its least accurate participant."""
    problem = randomized.random_instance(1).problem
    precise = reference.cross_check(problem, name="clarabel", solvers=(reference.CvxpySolver("CLARABEL"),))
    assert precise.tolerance == precise.requested_tolerance == reference.OBJECTIVE_TOLERANCE

    if not reference.CvxpySolver("SCS").is_available():
        pytest.skip("SCS is not installed")
    loose = reference.cross_check(problem, name="scs", solvers=(reference.CvxpySolver("SCS"),))
    assert loose.tolerance == reference.BACKEND_ACCURACY["SCS"]
    assert "widened from" in str(loose)


def test_a_single_solver_with_no_objective_has_nothing_to_compare():
    """Reported as a zero gap, which is a fact about the comparison, not about the answer."""
    check = reference.cross_check(
        randomized.random_instance(1).problem,
        name="alone",
        solvers=(reference.CvxpySolver("CLARABEL"),),
    )
    assert check.gap == 0.0
    assert check.agrees


def test_the_cross_check_reports_the_instance_and_the_gap():
    """A failure message that does not name the instance is not a failure message."""
    rendered = str(reference.cross_check(randomized.random_instance(1).problem, name="named-instance"))
    assert "named-instance" in rendered
    assert "gap" in rendered
    assert "CLARABEL" in rendered


def test_the_cross_check_needs_a_solver():
    """Reaching this in CI means §16.3's open fallback is broken, not merely absent."""
    with pytest.raises(reference.SolverUnavailableError, match=r"paper\.tex:1126"):
        reference.cross_check(randomized.random_instance(1).problem, solvers=())


# ----------------------------------------------------------------------------------
# Property-based: the sweep the fixed sample cannot do
# ----------------------------------------------------------------------------------

SEEDS = st.integers(min_value=0, max_value=randomized.MAX_SEED)

# Solver calls are slow relative to hypothesis's default deadline, and the deadline
# measures wall clock rather than work, so it is disabled rather than tuned. The
# `function_scoped_fixture` health check is irrelevant here -- these tests take no
# fixtures -- but `too_slow` would fire on the cross-check property, whose cost is the
# point of it.
PROFILE = settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@pytest.mark.property
@given(seed=SEEDS)
@PROFILE
def test_any_seed_gives_a_feasible_instance(seed):
    """Over the whole seed space: the witness is feasible, always.

    The claim that makes every other property here worth testing. If it can fail, then
    §16.3's "every randomly generated test problem" quietly becomes "every one that
    happened to be feasible".
    """
    instance = randomized.random_instance(seed)
    assert feasible(instance.problem, instance.witness), randomized.random_spec(seed).reproduce()


@pytest.mark.property
@given(seed=SEEDS)
@PROFILE
def test_any_seed_gives_the_active_set_structure_it_drew(seed):
    """The `tight` count holds across the whole space, not just the sampled seeds."""
    spec = randomized.random_spec(seed)
    instance = randomized.random_instance(seed)
    slack = instance.problem.b - instance.problem.A @ instance.witness
    assert int((np.abs(slack) <= 1e-12).sum()) == spec.tight, spec.reproduce()


@pytest.mark.property
@given(seed=SEEDS)
@PROFILE
def test_any_seed_gives_a_reproducible_instance(seed):
    """Issue #32's first "done when", swept rather than sampled."""
    first, second = randomized.random_instance(seed), randomized.random_instance(seed)
    np.testing.assert_array_equal(first.problem.A, second.problem.A)
    np.testing.assert_array_equal(first.problem.b, second.problem.b)
    np.testing.assert_array_equal(first.witness, second.witness)


@pytest.mark.property
@given(seed=SEEDS)
@PROFILE
def test_the_covariance_factor_is_exact_on_any_drawn_market(seed):
    """||L @ x|| == sqrt(x.T @ Sigma @ x) across six orders of conditioning.

    #10 asserts this on hand-picked covariances. Here the conditioning is drawn, which is
    where an eigenvalue cut that is too aggressive would show up.
    """
    instance = randomized.random_instance(seed)
    factor = instance.portfolio.factor()
    x = instance.witness[: instance.num_assets]
    quadratic = float(x @ instance.portfolio.Sigma @ x)
    assert float(np.linalg.norm(factor @ x)) == pytest.approx(np.sqrt(max(0.0, quadratic)), abs=1e-12, rel=1e-8)


@pytest.mark.property
@given(seed=SEEDS)
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_any_seed_is_cross_checked_against_a_reference_solver(seed):
    """§16.3, as written: every randomly generated problem, compared to a reference.

    The one property here that calls a solver, so it draws fewer examples. It is also the
    only one that could fail for a reason outside this repository -- which is what
    `CrossCheck.__str__` naming the solvers and the gap is for.
    """
    instance = randomized.random_instance(seed)
    check = reference.cross_check(instance.problem, name=instance.name)
    assert check.all_optimal, f"{check} -- {randomized.random_spec(seed).reproduce()}"
    assert check.agrees, f"{check} -- {randomized.random_spec(seed).reproduce()}"
