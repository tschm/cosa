"""The portfolio test-problem families of §10, and the large instance of §12.1.

The executable half of issues #19 and #31. Both have the same "done when" -- every family
instantiates, and a reference solver confirms each is feasible and bounded -- so the first
section checks exactly that, over all six families at once, and the sections after it check
that each family is actually the thing §10 asked for.

That second part matters more than it sounds. A generator that produces a *solvable*
problem has done half its job; a generator whose box constraints never bind, or whose
sector caps are a relabelling of the long-only bounds, produces solvable problems that
test nothing. §10.2 wants a family whose bounds become active, so
`test_the_box_binds_where_the_long_only_bounds_did_not` asserts that they do -- and that
assertion is what caught the original default cap of 0.4, which left this family's optimum
bit-for-bit identical to the basic family's.

The file lives under a name of its own rather than `test_portfolio.py`, which
`problem/portfolio.py` already owns. Two subpackages have a module called `portfolio` --
the plan's own table names both -- so the one-test-file-per-module convention needs a
tiebreak, and the subpackage is it.
"""

import numpy as np
import pytest

from cosa import ConeStatus, ProblemError, WorkingSet
from cosa.active_set import updates
from cosa.experiments import portfolio as families
from cosa.experiments import reference
from cosa.problem.portfolio import covariance_tolerance

# ----------------------------------------------------------------------------------
# Issue #19's and #31's "done when", over every family at once
# ----------------------------------------------------------------------------------


@pytest.fixture(params=[instance.name for instance in families.all_families(seed=0)])
def instance(request):
    """One family's instance, at the default size and seed."""
    return next(one for one in families.all_families(seed=0) if one.name == request.param)


def test_every_family_is_feasible_at_its_witness(instance):
    """Feasible by construction: the witness satisfies every block, cone included.

    The claim the module makes about itself, and the one that keeps the reference-solver
    check below from being a test of the generator's luck.
    """
    z = instance.witness
    problem = instance.problem
    assert np.all(problem.A @ z <= problem.b + 1e-12)
    np.testing.assert_allclose(problem.E @ z, problem.d, atol=1e-12)
    slack = problem.cone_slack(z)
    assert float(np.linalg.norm(slack[1:])) <= slack[0] + 1e-12


def test_every_family_is_solved_to_optimality_by_a_reference_solver(instance):
    """Bounded as well as feasible: an unbounded instance comes back as `unbounded`."""
    solution = reference.solve_reference(instance.problem)
    assert solution.is_optimal, f"{instance.name}: {solution.status}"
    assert np.isfinite(solution.objective)


def test_every_family_agrees_across_the_available_reference_solvers(instance):
    """§16.3's cross-check, run on the structured families as well as the random ones."""
    check = reference.cross_check(instance.problem, name=instance.name)
    assert check.all_optimal, str(check)
    assert check.agrees, str(check)


def test_every_family_beats_its_own_witness(instance):
    """The optimum is at least as good as the feasible point the family was built around.

    A weak claim, and worth making anyway: it is the one assertion that ties the witness
    and the solve together, so a witness that is feasible for a *different* problem than
    the one solved would fail here rather than pass both preceding tests.
    """
    solution = reference.solve_reference(instance.problem)
    assert solution.objective <= float(instance.problem.c @ instance.witness) + 1e-6


def test_every_family_names_all_of_its_rows(instance):
    """One name per row, so a working-set description never falls back to an index.

    Success Criterion 3 asks for decisions interpretable in terms of the active portfolio
    constraints, which is only true of generated instances if the generator supplies the
    names -- and only checkable if the counts line up.
    """
    assert len(instance.names.inequalities) == instance.problem.num_inequalities
    assert len(instance.names.equalities) == instance.problem.num_equalities
    assert len(instance.names.cones) == len(instance.problem.cone)


def test_every_family_carries_its_seed_in_its_name(instance):
    """So a failing benchmark row names the instance that produced it."""
    assert instance.name.endswith("-s0")
    assert f"n{instance.num_assets}" in instance.name


def test_every_family_has_one_risk_cone(instance):
    """The conic part never changes across the families -- only the polyhedral part does.

    Which is what makes a failure attributable: if the basic family solves and the sector
    family does not, the difference is linear rows, not geometry.
    """
    assert len(instance.problem.cone) == 1
    assert instance.problem.cone.cones[0].dim == instance.portfolio.factor().shape[0] + 1


def test_every_family_offers_a_working_set_over_its_own_shape(instance):
    """The convenience #20's prototype starts from, matching the problem it will solve."""
    working_set = instance.working_set()
    assert working_set == WorkingSet.empty(instance.problem)
    assert working_set.num_inequalities == instance.problem.num_inequalities


def test_the_instance_describes_itself(instance):
    """A benchmark row or a failure message needs the shape, not just the name."""
    rendered = str(instance)
    assert instance.name in rendered
    assert f"{instance.num_assets} assets" in rendered


# ----------------------------------------------------------------------------------
# The synthetic market: exact rank, exact conditioning
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("rank", [1, 3, 5])
def test_the_market_has_the_rank_it_was_asked_for(rank):
    """Exactly, which a random `B.T @ B` would only manage on average."""
    market = families.synthetic_market(5, seed=1, rank=rank)
    assert market.factor().shape[0] == rank
    assert np.linalg.matrix_rank(market.Sigma) == rank


def test_a_full_rank_market_is_the_default():
    """`rank=None` means as many risk factors as assets."""
    assert families.synthetic_market(4, seed=1).factor().shape[0] == 4


@pytest.mark.parametrize("condition", [1.0, 1e2, 1e6])
def test_the_market_has_the_conditioning_it_was_asked_for(condition):
    """The nonzero eigenvalues span exactly the requested ratio.

    The knob #33's ill-conditioned family will turn, so it has to be a knob and not a
    suggestion.
    """
    market = families.synthetic_market(6, seed=2, condition=condition)
    eigenvalues = np.linalg.eigvalsh(market.Sigma)
    positive = eigenvalues[eigenvalues > covariance_tolerance(market.Sigma)]
    assert positive.max() / positive.min() == pytest.approx(condition, rel=1e-6)


def test_the_market_is_not_accidentally_diagonal():
    """A diagonal covariance makes the risk term separable and hides conic errors."""
    covariance = families.synthetic_market(5, seed=3).Sigma
    off_diagonal = covariance - np.diag(np.diag(covariance))
    assert np.abs(off_diagonal).max() > 1e-3


def test_the_market_is_scaled_to_the_requested_volatility():
    """The average asset variance is the target, so `lam` means the same thing at any size."""
    market = families.synthetic_market(20, seed=4, volatility=0.3)
    assert float(np.mean(np.diag(market.Sigma))) == pytest.approx(0.09, rel=1e-9)


def test_the_market_is_reproducible_from_its_seed():
    """The same seed gives the same market, bit for bit."""
    first = families.synthetic_market(5, seed=7)
    second = families.synthetic_market(5, seed=7)
    np.testing.assert_array_equal(first.mu, second.mu)
    np.testing.assert_array_equal(first.Sigma, second.Sigma)
    assert not np.array_equal(first.mu, families.synthetic_market(5, seed=8).mu)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"assets": 0}, "at least one asset"),
        ({"assets": 4, "rank": 5}, "rank in"),
        ({"assets": 4, "condition": 0.5}, "condition number"),
        ({"assets": 4, "volatility": 0.0}, "volatility"),
    ],
)
def test_the_market_rejects_impossible_parameters(kwargs, match):
    """A generator that silently reinterprets its arguments is worse than one that stops."""
    with pytest.raises(ProblemError, match=match):
        families.synthetic_market(seed=0, **kwargs)


# ----------------------------------------------------------------------------------
# §10.1 basic: eq. (8)
# ----------------------------------------------------------------------------------


def test_the_basic_family_is_eq_eight():
    """1.T @ x = 1 and x >= 0, the latter as ordinary rows of A."""
    instance = families.basic(5, seed=0)
    portfolio = instance.portfolio
    np.testing.assert_allclose(portfolio.E, np.ones((1, 5)))
    np.testing.assert_allclose(portfolio.d, [1.0])
    np.testing.assert_allclose(portfolio.A, -np.eye(5))
    np.testing.assert_allclose(portfolio.b, np.zeros(5))


def test_the_basic_family_is_long_only_at_its_optimum():
    """The bounds are in reach of the working-set logic, so the solver honours them."""
    instance = families.basic(6, seed=1)
    x = reference.solve_reference(instance.problem).z[: instance.num_assets]
    assert x.min() >= -1e-8
    assert float(x.sum()) == pytest.approx(1.0, abs=1e-8)


def test_the_basic_witness_is_the_equal_weight_portfolio():
    """The interior point every family but turnover is built around."""
    instance = families.basic(4, seed=0)
    np.testing.assert_allclose(instance.witness[:4], 0.25)


# ----------------------------------------------------------------------------------
# §10.2 box: the family whose bounds become active
# ----------------------------------------------------------------------------------


def test_the_box_family_adds_two_blocks_of_bounds():
    """Uppers first, then lowers, so a row index maps to a bound without arithmetic."""
    instance = families.box(5, lower=0.05, upper=0.5, seed=0)
    assert instance.problem.num_inequalities == 10
    np.testing.assert_allclose(instance.portfolio.A[:5], np.eye(5))
    np.testing.assert_allclose(instance.portfolio.b[:5], 0.5)
    np.testing.assert_allclose(instance.portfolio.A[5:], -np.eye(5))
    np.testing.assert_allclose(instance.portfolio.b[5:], -0.05)


@pytest.mark.parametrize("assets", [8, 20])
def test_the_box_binds_where_the_long_only_bounds_did_not(assets):
    """§10.2's whole purpose: the cap has to be active at the optimum.

    The test that earns its keep. With a fixed cap of 0.4 -- the module's first default --
    the box optimum was identical to the basic optimum on this market, objective and active
    set alike: the family had collapsed into the one it exists to differ from. The default
    now scales with the asset count, and this asserts the consequence rather than the
    intention.
    """
    plain = families.basic(assets, seed=0)
    boxed = families.box(assets, seed=0)
    plain_objective = reference.solve_reference(plain.problem).objective
    boxed_objective = reference.solve_reference(boxed.problem).objective
    assert boxed_objective > plain_objective + 1e-6, "the cap must actually cost something"

    z = reference.solve_reference(boxed.problem).z
    tight = np.abs(boxed.problem.b - boxed.problem.A @ z) <= 1e-6
    assert tight[:assets].sum() >= 2, "several upper bounds are active, as §10.2 expects"


def test_the_default_box_scales_with_the_asset_count():
    """`BOX_WIDTH` times the equal weight, not a fixed number of percent."""
    for assets in (5, 25):
        instance = families.box(assets, seed=0)
        np.testing.assert_allclose(instance.portfolio.b[:assets], families.BOX_WIDTH / assets)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lower": 0.5, "upper": 0.2}, "lower <= upper"),
        ({"upper": 0.05}, "fully-invested"),
        ({"lower": 0.5, "upper": 0.6}, "fully-invested"),
    ],
)
def test_the_box_rejects_bounds_that_cannot_be_fully_invested(kwargs, match):
    """Caught at construction, where the arithmetic is one line, not at solve time.

    Both directions: a cap too low to reach a total of one, and a floor too high to stay
    at one. The second needs an explicit upper bound, because a floor above the default
    cap trips the ordering check first -- which is itself the right answer, just a
    different one.
    """
    with pytest.raises(ProblemError, match=match):
        families.box(8, seed=0, **kwargs)


# ----------------------------------------------------------------------------------
# §10.3 sector: nontrivial combinations of active constraints
# ----------------------------------------------------------------------------------


def test_the_sector_family_deals_assets_round_robin():
    """Interleaved, not contiguous: every sector row overlaps every part of the weights.

    A contiguous partition would make a sector cap a bound on a slice, and the
    "nontrivial combinations" §10.3 asks for would be trivial.
    """
    instance = families.sector(6, sectors=3, seed=0)
    membership = instance.portfolio.A[6:]
    np.testing.assert_allclose(membership[0], [1, 0, 0, 1, 0, 0])
    np.testing.assert_allclose(membership[1], [0, 1, 0, 0, 1, 0])
    np.testing.assert_allclose(membership[2], [0, 0, 1, 0, 0, 1])


def test_every_asset_is_in_exactly_one_sector():
    """A partition, so the sector caps sum to a bound on the whole portfolio."""
    instance = families.sector(9, sectors=4, seed=0)
    membership = instance.portfolio.A[9:]
    np.testing.assert_allclose(membership.sum(axis=0), np.ones(9))


def test_the_sector_caps_bind_at_the_optimum():
    """Tight enough to matter: a cap no optimum reaches is not a test of anything."""
    loose = families.basic(9, seed=0)
    capped = families.sector(9, sectors=3, cap=0.34, seed=0)
    assert reference.solve_reference(capped.problem).objective > (
        reference.solve_reference(loose.problem).objective + 1e-6
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sectors": 0}, "sector count"),
        ({"sectors": 20}, "sector count"),
        ({"sectors": 2, "cap": 0.4}, "cannot be fully invested"),
    ],
)
def test_the_sector_family_rejects_impossible_partitions(kwargs, match):
    """Two sectors capped at 0.4 hold 0.8, which is not a fully invested portfolio."""
    with pytest.raises(ProblemError, match=match):
        families.sector(9, seed=0, **kwargs)


# ----------------------------------------------------------------------------------
# §10.4 factor exposure: correlated active constraints
# ----------------------------------------------------------------------------------


def test_the_factor_family_adds_a_band_per_factor():
    """`l_f <= F @ x <= u_f` as two blocks, uppers then lowers."""
    instance = families.factor_exposure(6, factors=2, seed=0)
    assert instance.problem.num_inequalities == 6 + 4
    upper, lower = instance.portfolio.A[6:8], instance.portfolio.A[8:]
    np.testing.assert_allclose(upper, -lower)


def test_the_factor_bands_are_centred_on_the_equal_weight_exposure():
    """Which is what makes the family feasible by construction rather than by luck.

    Independent bounds on a random `F` are infeasible far more often than not, so the
    bands are placed around the witness's own exposure.
    """
    instance = families.factor_exposure(6, factors=3, width=0.25, seed=0)
    exposures = instance.portfolio.A[6:9]
    centre = exposures @ np.full(6, 1.0 / 6)
    np.testing.assert_allclose(instance.portfolio.b[6:9], centre + 0.25)
    np.testing.assert_allclose(instance.portfolio.b[9:], -(centre - 0.25))


def test_the_factor_rows_are_dense_and_correlated_with_each_other():
    """Dense Gaussian rows, which is what makes the active set nearly dependent.

    The property §10.4 is testing for, and the reason this family is where #25's rank
    detection will first be needed.
    """
    exposures = families.factor_exposure(8, factors=3, seed=0).portfolio.A[8:11]
    assert np.abs(exposures).min() > 1e-6, "no structural zeros"
    correlations = np.corrcoef(exposures)
    off = correlations[~np.eye(3, dtype=bool)]
    assert np.abs(off).max() > 0.1


def test_changing_the_factor_count_does_not_move_the_market():
    """The factor matrix comes from a stream of its own, so two rows stay comparable."""
    two = families.factor_exposure(6, factors=2, seed=0)
    three = families.factor_exposure(6, factors=3, seed=0)
    np.testing.assert_array_equal(two.portfolio.mu, three.portfolio.mu)
    np.testing.assert_array_equal(two.portfolio.Sigma, three.portfolio.Sigma)


@pytest.mark.parametrize(("kwargs", "match"), [({"factors": 0}, "at least one factor"), ({"width": 0.0}, "width")])
def test_the_factor_family_rejects_impossible_parameters(kwargs, match):
    """A band of zero width is a pair of equalities in disguise, not a band."""
    with pytest.raises(ProblemError, match=match):
        families.factor_exposure(8, seed=0, **kwargs)


# ----------------------------------------------------------------------------------
# §10.5 turnover: the family that needed auxiliary variables
# ----------------------------------------------------------------------------------


def test_the_turnover_family_carries_one_auxiliary_variable_per_asset():
    """`(x, t, delta)`: the variable vector §10.5 forces, built with `SOCP.augment`."""
    instance = families.turnover(6, seed=0)
    assert instance.num_auxiliary == 6
    assert instance.problem.num_variables == 6 + 1 + 6


def test_the_turnover_rows_bound_the_absolute_trade():
    """Two blocks giving `delta_i >= |x_i - x_old_i|`, then one row capping the total.

    No `delta >= 0` rows are needed: the two blocks together already imply it, which is
    worth asserting because adding them would be the obvious redundant thing to do.
    """
    instance = families.turnover(4, budget=0.3, seed=0)
    problem = instance.problem
    assert problem.num_inequalities == 4 + 4 + 4 + 1
    up, down, cap = problem.A[4:8], problem.A[8:12], problem.A[12]
    np.testing.assert_allclose(up[:, :4], np.eye(4))
    np.testing.assert_allclose(up[:, 5:], -np.eye(4))
    np.testing.assert_allclose(down[:, :4], -np.eye(4))
    np.testing.assert_allclose(down[:, 5:], -np.eye(4))
    np.testing.assert_allclose(cap, np.concatenate([np.zeros(5), np.ones(4)]))
    assert problem.b[12] == pytest.approx(0.3)


def test_the_turnover_witness_is_the_previous_portfolio_with_no_trading():
    """`x = x_old`, `delta = 0`: feasible for any positive budget, by construction."""
    instance = families.turnover(5, seed=0)
    previous = instance.witness[:5]
    assert float(previous.sum()) == pytest.approx(1.0)
    assert previous.min() > 0.0
    np.testing.assert_allclose(instance.witness[6:], 0.0)
    assert instance.witness[5] == pytest.approx(instance.portfolio.std(previous))


def test_the_turnover_budget_binds_at_the_optimum():
    """A budget the optimum ignores is not a turnover constraint."""
    instance = families.turnover(8, budget=0.05, seed=0)
    z = reference.solve_reference(instance.problem).z
    traded = float(z[9:].sum())
    assert traded == pytest.approx(0.05, abs=1e-6)


def test_a_generous_turnover_budget_recovers_the_unconstrained_optimum():
    """The other end: with enough budget, the family is the basic family again."""
    plain = families.basic(8, seed=0)
    generous = families.turnover(8, budget=10.0, seed=0)
    assert reference.solve_reference(generous.problem).objective == pytest.approx(
        reference.solve_reference(plain.problem).objective, abs=1e-6
    )


def test_the_turnover_family_rejects_a_non_positive_budget():
    """A budget of zero pins the portfolio, which is not a rebalancing problem."""
    with pytest.raises(ProblemError, match="turnover budget"):
        families.turnover(8, budget=0.0, seed=0)


# ----------------------------------------------------------------------------------
# §12.1's large instance: many rows, a small cone
# ----------------------------------------------------------------------------------


def test_the_large_family_is_low_rank_in_the_cone_and_large_in_the_rows():
    """`Q^(1 + factors)` against `2 * assets` linear rows -- the shape of a factor model."""
    instance = families.large(120, factors=8, seed=0)
    assert instance.problem.cone.cones[0].dim == 9
    assert instance.problem.num_inequalities == 240


def test_the_large_family_has_a_large_active_set():
    """A large problem with three active constraints would not test an active-set method."""
    instance = families.large(60, factors=5, seed=0)
    z = reference.solve_reference(instance.problem).z
    tight = int((np.abs(instance.problem.b - instance.problem.A @ z) <= 1e-6).sum())
    assert tight >= instance.num_assets // 2


def test_the_large_family_solves_at_a_realistic_size():
    """Five hundred assets is not a stress test; it is the size the plan is aimed at."""
    instance = families.large(500, factors=20, seed=0)
    assert reference.solve_reference(instance.problem).is_optimal


def test_the_large_family_rejects_more_factors_than_assets():
    """A factor model with more factors than assets is not low rank."""
    with pytest.raises(ProblemError, match="factor count"):
        families.large(10, factors=20, seed=0)


# ----------------------------------------------------------------------------------
# The families feed the working set, which is the point of generating them
# ----------------------------------------------------------------------------------


def test_a_generated_instance_drives_the_working_set_logic():
    """The end-to-end reason these exist: a real instance, a real active set, real names.

    Not a solver test -- #20 owns that. It checks that the pieces of Wave 1 and Wave 2 fit:
    a generated instance yields a point, the point yields activation candidates and a cone
    status, and the resulting working set describes itself in the family's own vocabulary.
    """
    instance = families.box(6, seed=0)
    z = reference.solve_reference(instance.problem).z
    working_set = WorkingSet.empty(instance.problem)
    for index in updates.activation_candidates(instance.problem, z, working_set, tolerance=1e-6):
        working_set = updates.add_inequality(working_set, index)
    working_set = updates.activate_cones(instance.problem, z, working_set)

    assert working_set.inequalities, "some bound is active at the optimum"
    assert working_set.status(0) is ConeStatus.TANGENT, "the risk cone binds at the optimum"
    description = working_set.describe(instance.names)
    assert "upper bound on asset" in description
    assert "fully invested" in description
    assert "risk" in description
