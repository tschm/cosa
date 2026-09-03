"""The SOCP representation, the cone product, and the fixed sign convention.

The executable half of issue #9. Three things are load-bearing here and are tested
harder than the rest, because four later issues consume them:

* the cone is a Cartesian product, and its blocks are laid out head first;
* an instance of eq. (7) round-trips through the general form and back;
* the sign convention is one convention. The two golden instances at the bottom are
  hand-solved, so their multipliers pin the signs: a consumer that flips one --
  computes multipliers of the opposite sign, or assembles a KKT system with `+G.T @ w`
  -- fails `test_flipped_signs_are_not_stationary` and
  `test_sign_convention_is_the_one_fixed_convention`.
"""

import numpy as np
import pytest

from cosa import (
    SIGN_CONVENTION,
    SOCP,
    ConeProduct,
    MeanStdForm,
    ProblemError,
    SecondOrderCone,
    SignConvention,
)


@pytest.fixture
def mean_std():
    """Eq. (7) with two assets, a budget equality, a box inequality and a full L."""
    return MeanStdForm(
        mu=np.array([0.10, 0.04]),
        lam=2.0,
        A=np.array([[1.0, 0.0]]),
        b=np.array([0.7]),
        E=np.array([[1.0, 1.0]]),
        d=np.array([1.0]),
        L=np.array([[0.2, 0.05], [0.0, 0.1]]),
    )


# ----------------------------------------------------------------------------------
# One cone
# ----------------------------------------------------------------------------------


def test_cone_carries_a_head_and_a_tail():
    """The plan's Q^(m+1): total dimension, and the m of the tail."""
    cone = SecondOrderCone(dim=4)
    assert cone.dim == 4
    assert cone.tail_dim == 3


@pytest.mark.parametrize("dim", [1, 0, -1])
def test_cone_rejects_a_dimension_below_two(dim):
    """Q^1 is the non-negative ray, which is a linear inequality, not a cone."""
    with pytest.raises(ProblemError, match="dim >= 2"):
        SecondOrderCone(dim=dim)


def test_cone_splits_head_first():
    """Head first, never head last -- the layout every consumer relies on."""
    head, tail = SecondOrderCone(dim=3).split(np.array([5.0, 3.0, 4.0]))
    assert head == 5.0
    np.testing.assert_array_equal(tail, [3.0, 4.0])


def test_cone_split_rejects_the_wrong_length():
    """A block of the wrong length is a bug, not something to broadcast away."""
    with pytest.raises(ProblemError, match="expected 3 entries"):
        SecondOrderCone(dim=3).split(np.array([1.0, 2.0]))


# ----------------------------------------------------------------------------------
# The Cartesian product
# ----------------------------------------------------------------------------------


def test_product_of_several_cones():
    """K = Q^2 x Q^3 x Q^2: the shape the plan's multiple-cone section targets."""
    product = ConeProduct.from_dims(2, 3, 2)
    assert len(product) == 3
    assert product.dim == 7
    assert product.slices == (slice(0, 2), slice(2, 5), slice(5, 7))
    assert [cone.dim for cone in product.cones] == [2, 3, 2]


def test_product_blocks_partition_a_vector_in_factor_order():
    """Blocks are consecutive and ordered, so G's rows line up with the factors."""
    product = ConeProduct.from_dims(2, 3)
    blocks = product.blocks(np.arange(5.0))
    np.testing.assert_array_equal(blocks[0], [0.0, 1.0])
    np.testing.assert_array_equal(blocks[1], [2.0, 3.0, 4.0])


def test_product_split_is_a_head_and_tail_per_factor():
    """The per-factor form the conic working-set logic will iterate over."""
    pairs = ConeProduct.from_dims(2, 3).split(np.array([1.0, 0.5, 2.0, 1.0, 1.5]))
    assert pairs[0][0] == 1.0
    np.testing.assert_array_equal(pairs[0][1], [0.5])
    assert pairs[1][0] == 2.0
    np.testing.assert_array_equal(pairs[1][1], [1.0, 1.5])


def test_product_rejects_a_vector_of_the_wrong_length():
    """A dual variable that does not match the cone is caught at the boundary."""
    with pytest.raises(ProblemError, match="expected 5 entries"):
        ConeProduct.from_dims(2, 3).blocks(np.zeros(4))


def test_empty_product_is_the_linear_program():
    """No cone at all is allowed: that instance is an LP, which is a useful case."""
    product = ConeProduct()
    assert len(product) == 0
    assert product.dim == 0
    assert product.slices == ()


# ----------------------------------------------------------------------------------
# The general form and its validation
# ----------------------------------------------------------------------------------


def test_unconstrained_is_the_objective_alone():
    """The starting point for building an instance up a block at a time."""
    problem = SOCP.unconstrained(np.array([1.0, 2.0, 3.0]))
    assert problem.num_variables == 3
    assert problem.num_inequalities == 0
    assert problem.num_equalities == 0
    assert problem.cone.dim == 0
    assert problem.A.shape == (0, 3)
    assert problem.G.shape == (0, 3)


def test_blocks_are_coerced_to_float_and_copied():
    """An instance owns its data: integer input is fine, later mutation is not."""
    objective = np.array([1, 2])
    problem = SOCP.unconstrained(objective)
    objective[0] = 99
    assert problem.c.dtype == np.float64
    np.testing.assert_array_equal(problem.c, [1.0, 2.0])


def test_lists_are_accepted():
    """Nothing forces a caller to reach for NumPy to write down a small instance."""
    problem = SOCP.unconstrained([1.0, -1.0]).add_inequalities([[1.0, 1.0]], [2.0])
    np.testing.assert_array_equal(problem.A, [[1.0, 1.0]])
    np.testing.assert_array_equal(problem.b, [2.0])


def _valid_blocks():
    """The keyword arguments of a small, valid two-variable instance.

    Returns:
        A dict of blocks that :class:`SOCP` accepts, for a test to spoil one of.
    """
    return {
        "c": np.array([1.0, 1.0]),
        "A": np.zeros((1, 2)),
        "b": np.zeros(1),
        "E": np.zeros((1, 2)),
        "d": np.zeros(1),
        "G": np.eye(2),
        "h": np.zeros(2),
        "cone": ConeProduct.from_dims(2),
    }


@pytest.mark.parametrize(
    ("block", "value", "match"),
    [
        ("c", np.zeros(0), "at least one variable"),
        ("c", np.zeros((2, 1)), "expected a vector"),
        ("A", np.zeros((1, 3)), "expected 2 columns"),
        ("A", np.zeros(2), "expected a matrix"),
        ("b", np.zeros(2), "expected 1 entries"),
        ("E", np.zeros((1, 5)), "expected 2 columns"),
        ("d", np.zeros(3), "expected 1 entries"),
        ("G", np.eye(3), "expected 2 rows"),
        ("h", np.zeros(1), "expected 2 entries"),
        ("c", np.array([1.0, np.nan]), "must be finite"),
        ("G", np.array([[1.0, 0.0], [0.0, np.inf]]), "must be finite"),
    ],
)
def test_validation_catches_a_spoiled_block(block, value, match):
    """Shape agreement, cone dimension, emptiness and finiteness, all on construction."""
    blocks = _valid_blocks()
    blocks[block] = value
    with pytest.raises(ProblemError, match=match):
        SOCP(**blocks)


def test_cone_dimension_must_match_the_conic_rows():
    """G's row count is the cone's dimension: the two cannot drift apart."""
    blocks = _valid_blocks()
    blocks["cone"] = ConeProduct.from_dims(3)
    with pytest.raises(ProblemError, match="expected 3 rows"):
        SOCP(**blocks)


def test_cone_slack_is_the_vector_the_cone_is_about():
    """G @ z + h, laid out head first -- the primal side of the conic condition."""
    problem = SOCP(**{**_valid_blocks(), "h": np.array([1.0, 2.0])})
    np.testing.assert_array_equal(problem.cone_slack(np.array([3.0, 4.0])), [4.0, 6.0])


# ----------------------------------------------------------------------------------
# Room for auxiliary variables
# ----------------------------------------------------------------------------------


def test_augment_appends_variables_that_are_zero_everywhere(mean_std):
    """The turnover pattern: new variables enter no existing row."""
    problem = mean_std.to_socp()
    grown = problem.augment(2, c=np.array([0.5, 0.5]))
    assert grown.num_variables == problem.num_variables + 2
    np.testing.assert_array_equal(grown.c[-2:], [0.5, 0.5])
    np.testing.assert_array_equal(grown.A[:, -2:], np.zeros((problem.num_inequalities, 2)))
    np.testing.assert_array_equal(grown.E[:, -2:], np.zeros((problem.num_equalities, 2)))
    np.testing.assert_array_equal(grown.G[:, -2:], np.zeros((problem.cone.dim, 2)))


def test_augment_defaults_to_variables_outside_the_objective(mean_std):
    """An auxiliary variable that only appears in constraints costs nothing to add."""
    grown = mean_std.to_socp().augment(1)
    assert grown.c[-1] == 0.0


def test_augment_needs_at_least_one_variable(mean_std):
    """Augmenting by nothing is a caller bug, so it says so."""
    with pytest.raises(ProblemError, match="at least one variable"):
        mean_std.to_socp().augment(0)


def test_turnover_shaped_augmentation(mean_std):
    """Auxiliary variables plus linear inequalities, the plan's turnover recipe.

    ``|x_i - x_old_i| <= u_i`` becomes two inequalities per asset in the augmented
    variable vector. The point of the test is that the representation takes it
    without being reopened.
    """
    problem = mean_std.to_socp()
    assets = mean_std.num_assets
    grown = problem.augment(assets)
    previous = np.array([0.5, 0.5])
    selector = np.zeros((assets, grown.num_variables))
    selector[:, :assets] = np.eye(assets)
    bound = np.zeros((assets, grown.num_variables))
    bound[:, -assets:] = np.eye(assets)
    turnover = grown.add_inequalities(
        np.vstack([selector - bound, -selector - bound]),
        np.concatenate([previous, -previous]),
    )
    assert turnover.num_inequalities == problem.num_inequalities + 2 * assets
    assert turnover.num_variables == assets + 1 + assets


def test_add_equalities_appends_rows(mean_std):
    """The equality block grows the same way, and stays consistent with d."""
    problem = mean_std.to_socp()
    grown = problem.add_equalities(np.ones((1, problem.num_variables)), np.array([2.0]))
    assert grown.num_equalities == problem.num_equalities + 1
    assert grown.d[-1] == 2.0


def test_add_inequalities_rejects_rows_of_the_wrong_width(mean_std):
    """New rows are checked against the current variable count, not the old one."""
    problem = mean_std.to_socp().augment(1)
    with pytest.raises(ProblemError, match="expected 4 columns"):
        problem.add_inequalities(np.ones((1, 3)), np.array([1.0]))


def test_add_cone_grows_the_product(mean_std):
    """A second cone is an addition to the product, not a new problem class."""
    problem = mean_std.to_socp()
    cone = SecondOrderCone(dim=2)
    rows = np.zeros((2, problem.num_variables))
    rows[0, -1] = 1.0
    rows[1, 0] = 1.0
    grown = problem.add_cone(cone, rows, np.zeros(2))
    assert len(grown.cone) == 2
    assert grown.cone.dim == problem.cone.dim + 2
    assert grown.G.shape == (problem.cone.dim + 2, problem.num_variables)


def test_add_cone_rejects_the_wrong_number_of_rows(mean_std):
    """The new factor's dimension and its rows of G are the same number."""
    problem = mean_std.to_socp()
    with pytest.raises(ProblemError, match="expected 2 rows"):
        problem.add_cone(SecondOrderCone(dim=2), np.zeros((3, problem.num_variables)), np.zeros(2))


# ----------------------------------------------------------------------------------
# Ill-posedness that is visible without solving
# ----------------------------------------------------------------------------------


def test_a_well_posed_instance_is_not_trivially_infeasible(mean_std):
    """The common case: nothing visibly wrong, and no opinion offered about the rest."""
    assert mean_std.to_socp().trivially_infeasible() is None


def test_an_empty_inequality_row_with_a_negative_bound_is_empty(mean_std):
    """0 <= -1 constrains nothing and rules out everything."""
    problem = mean_std.to_socp()
    spoiled = problem.add_inequalities(np.zeros((1, problem.num_variables)), np.array([-1.0]))
    assert spoiled.trivially_infeasible() == (
        "an inequality row constrains no variable and has a negative right-hand side"
    )


def test_an_empty_equality_row_with_a_nonzero_bound_is_empty(mean_std):
    """0 = 1 is the other half of the same modelling error."""
    problem = mean_std.to_socp()
    spoiled = problem.add_equalities(np.zeros((1, problem.num_variables)), np.array([1.0]))
    assert spoiled.trivially_infeasible() == (
        "an equality row constrains no variable and has a nonzero right-hand side"
    )


def test_an_empty_row_with_a_consistent_bound_is_only_redundant(mean_std):
    """0 <= 1 and 0 = 0 are redundant rows, which is not the same as infeasible."""
    problem = mean_std.to_socp()
    padded = problem.add_inequalities(np.zeros((1, problem.num_variables)), np.array([1.0]))
    padded = padded.add_equalities(np.zeros((1, padded.num_variables)), np.array([0.0]))
    assert padded.trivially_infeasible() is None


# ----------------------------------------------------------------------------------
# Eq. (7) and the round trip
# ----------------------------------------------------------------------------------


def test_mean_std_maps_into_the_general_form(mean_std):
    """The general form takes z = (x, t), c = (-mu, lam) and one factor Q^(1 + k)."""
    problem = mean_std.to_socp()
    assets = mean_std.num_assets
    assert problem.num_variables == assets + 1
    np.testing.assert_array_equal(problem.c, [-0.10, -0.04, 2.0])
    assert len(problem.cone) == 1
    assert problem.cone.dim == 1 + mean_std.L.shape[0]
    np.testing.assert_array_equal(problem.G[0], [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(problem.G[1:, :assets], mean_std.L)
    np.testing.assert_array_equal(problem.h, np.zeros(problem.cone.dim))
    np.testing.assert_array_equal(problem.A[:, -1], [0.0])
    np.testing.assert_array_equal(problem.E[:, -1], [0.0])


def test_eq_7_round_trips(mean_std):
    """The issue's "done when": out to the general form, and back unchanged."""
    recovered = mean_std.to_socp().as_mean_std()
    assert recovered.lam == mean_std.lam
    for block in ("mu", "A", "b", "E", "d", "L"):
        np.testing.assert_array_equal(getattr(recovered, block), getattr(mean_std, block))


def test_round_trip_survives_an_instance_without_linear_constraints():
    """The cone on its own round-trips too, with empty polyhedral blocks."""
    form = MeanStdForm(
        mu=np.array([1.0]),
        lam=0.5,
        A=np.zeros((0, 1)),
        b=np.zeros(0),
        E=np.zeros((0, 1)),
        d=np.zeros(0),
        L=np.array([[1.0]]),
    )
    recovered = form.to_socp().as_mean_std()
    np.testing.assert_array_equal(recovered.mu, [1.0])
    assert recovered.A.shape == (0, 1)


@pytest.mark.parametrize(
    ("spoil", "match"),
    [
        (lambda p: p.add_cone(SecondOrderCone(dim=2), _cone_rows(p), np.zeros(2)), "exactly one cone"),
        (lambda p: p.augment(1), "must select t"),
        (lambda p: p.add_inequalities(_last_variable_row(p), np.array([1.0])), "linear inequalities"),
        (lambda p: p.add_equalities(_last_variable_row(p), np.array([1.0])), "linear equalities"),
    ],
)
def test_as_mean_std_rejects_an_instance_of_another_shape(mean_std, spoil, match):
    """Reading eq. (7) back is only defined on instances that are in its shape."""
    with pytest.raises(ProblemError, match=match):
        spoil(mean_std.to_socp()).as_mean_std()


def _cone_rows(problem):
    """Rows of G for one extra Q^2 factor over an existing instance.

    Args:
        problem: the instance to extend.

    Returns:
        A ``(2, n)`` matrix whose head row selects the last variable.
    """
    rows = np.zeros((2, problem.num_variables))
    rows[0, -1] = 1.0
    return rows


def _last_variable_row(problem):
    """A single constraint row that touches only the last variable.

    Args:
        problem: the instance the row is for.

    Returns:
        A ``(1, n)`` matrix selecting the last variable.
    """
    row = np.zeros((1, problem.num_variables))
    row[0, -1] = 1.0
    return row


def test_as_mean_std_rejects_a_nonzero_offset(mean_std):
    """Eq. (7) has h = 0; an instance with an offset is a more general problem."""
    problem = mean_std.to_socp()
    shifted = SOCP(
        c=problem.c,
        A=problem.A,
        b=problem.b,
        E=problem.E,
        d=problem.d,
        G=problem.G,
        h=np.ones(problem.cone.dim),
        cone=problem.cone,
    )
    with pytest.raises(ProblemError, match="h must be zero"):
        shifted.as_mean_std()


@pytest.mark.parametrize(
    ("block", "value", "match"),
    [
        ("mu", np.zeros(0), "at least one asset"),
        ("lam", 0.0, "lam > 0"),
        ("lam", -1.0, "lam > 0"),
        ("lam", float("nan"), "lam > 0"),
        ("L", np.zeros((1, 3)), "expected 2 columns"),
        ("L", np.zeros((0, 2)), "at least one row"),
        ("A", np.zeros((1, 3)), "expected 2 columns"),
    ],
)
def test_mean_std_validation(mean_std, block, value, match):
    """Eq. (7)'s own blocks are validated against the number of assets."""
    blocks = {name: getattr(mean_std, name) for name in ("mu", "lam", "A", "b", "E", "d", "L")}
    blocks[block] = value
    with pytest.raises(ProblemError, match=match):
        MeanStdForm(**blocks)


# ----------------------------------------------------------------------------------
# The one sign convention
# ----------------------------------------------------------------------------------


def test_sign_convention_is_the_one_fixed_convention():
    """The signs are pinned here.

    A consumer that assembles stationarity itself -- the KKT matrix, the multiplier
    computation -- reads these three numbers rather than writing signs out. Changing
    one is a change to the derivation and to every consumer, so it fails here first.
    """
    assert SignConvention(inequality=1.0, equality=1.0, cone=-1.0) == SIGN_CONVENTION


def test_stationarity_residual_uses_the_convention(mean_std):
    """The residual is exactly c + A.T @ y + E.T @ nu - G.T @ w, and nothing else."""
    problem = mean_std.to_socp()
    rng = np.random.default_rng(9)
    y = rng.uniform(size=problem.num_inequalities)
    nu = rng.normal(size=problem.num_equalities)
    w = rng.normal(size=problem.cone.dim)
    expected = (
        problem.c
        + SIGN_CONVENTION.inequality * (problem.A.T @ y)
        + SIGN_CONVENTION.equality * (problem.E.T @ nu)
        + SIGN_CONVENTION.cone * (problem.G.T @ w)
    )
    np.testing.assert_allclose(problem.stationarity_residual(y, nu, w), expected)


@pytest.mark.parametrize(("block", "value"), [("y", np.zeros(2)), ("nu", np.zeros(2)), ("w", np.zeros(2))])
def test_stationarity_residual_checks_the_multiplier_shapes(mean_std, block, value):
    """A multiplier vector of the wrong length is caught, not broadcast."""
    problem = mean_std.to_socp()
    multipliers = {"y": np.zeros(1), "nu": np.zeros(1), "w": np.zeros(problem.cone.dim)}
    multipliers[block] = value
    with pytest.raises(ProblemError, match=f"^{block}: expected"):
        problem.stationarity_residual(**multipliers)


# Two instances solved by hand, so that every number below is a fact about the
# convention rather than a recording of what the code happened to produce.
#
# BOUND: one asset, mu = 1, lam = 1/2, x <= 1. The objective is -x + t/2 with
# t = |x|, so it decreases along x and the bound binds: x = t = 1. Stationarity gives
# w_t = lam = 1/2 and y = 1 + w_x; cone complementarity w.T @ (t, x) = 0 then forces
# w_x = -1/2 and so y = 1/2.
#
# BUDGET: two assets, mu = (2, 1), lam = 1, sum(x) = 1, Sigma = I. The risk term is
# isotropic, so the optimum puts everything in the better asset: x = (1, 0), t = 1.
# Stationarity gives w_t = lam = 1, w_1 = nu - 2, w_2 = nu - 1, and cone
# complementarity forces nu = 1, hence w = (1, -1, 0), on the boundary of Q.
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
        "y": np.array([0.5]),
        "nu": np.zeros(0),
        "w": np.array([0.5, -0.5]),
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
        "y": np.zeros(0),
        "nu": np.array([1.0]),
        "w": np.array([1.0, -1.0, 0.0]),
    },
}


@pytest.fixture(params=sorted(GOLDEN))
def golden(request):
    """One hand-solved instance together with its primal point and multipliers."""
    return GOLDEN[request.param]


def test_golden_instance_is_primal_feasible(golden):
    """The hand-solved point satisfies every block, cone included."""
    problem = golden["form"].to_socp()
    z = golden["z"]
    assert np.all(problem.A @ z <= problem.b + 1e-12)
    np.testing.assert_allclose(problem.E @ z, problem.d)
    head, tail = problem.cone.split(problem.cone_slack(z))[0]
    assert np.linalg.norm(tail) <= head + 1e-12


def test_golden_instance_is_stationary(golden):
    """The residual vanishes at the hand-solved multipliers, in this convention."""
    problem = golden["form"].to_socp()
    residual = problem.stationarity_residual(golden["y"], golden["nu"], golden["w"])
    np.testing.assert_allclose(residual, np.zeros(problem.num_variables), atol=1e-12)


def test_golden_multipliers_are_dual_feasible(golden):
    """Dual feasibility: y >= 0 for A @ z <= b, and w in K, the cone being self-dual."""
    assert np.all(golden["y"] >= 0.0)
    head, tail = golden["form"].to_socp().cone.split(golden["w"])[0]
    assert np.linalg.norm(tail) <= head + 1e-12


def test_golden_multipliers_are_complementary(golden):
    """Both complementarity conditions, in the same convention as stationarity."""
    problem = golden["form"].to_socp()
    z = golden["z"]
    np.testing.assert_allclose(golden["y"] * (problem.A @ z - problem.b), 0.0, atol=1e-12)
    np.testing.assert_allclose(golden["w"] @ problem.cone_slack(z), 0.0, atol=1e-12)


def test_the_cone_multiplier_head_is_lambda(golden):
    """The convention's signature: t's stationarity reads w_t = lam, not -lam.

    A consumer that fixes the opposite sign produces w_t = -lam, which is outside Q
    for lam > 0 and so fails dual feasibility as well as this assertion.
    """
    head, _ = golden["form"].to_socp().cone.split(golden["w"])[0]
    assert head == pytest.approx(golden["form"].lam)


@pytest.mark.parametrize("flip", ["y", "nu", "w"])
def test_flipped_signs_are_not_stationary(golden, flip):
    """Negating any block of the multipliers breaks stationarity.

    This is what makes the convention checkable rather than merely written down: the
    opposite convention does not also satisfy it, so a consumer that adopts one
    silently cannot agree with the residuals.
    """
    problem = golden["form"].to_socp()
    multipliers = {name: golden[name].copy() for name in ("y", "nu", "w")}
    if multipliers[flip].size == 0:
        pytest.skip(f"this instance has no {flip} block to flip")
    multipliers[flip] = -multipliers[flip]
    residual = problem.stationarity_residual(**multipliers)
    assert np.linalg.norm(residual) > 1e-9
