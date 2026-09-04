"""§7.4 and §9 Phase IV: the working set decided by the conic multiplier, not the geometry.

Issue #23. Three things are under test, and they are separable:

* :func:`cosa.geometry.tangent.curvature`, the second derivative of the cone constraint --
  the object "the tangent alone" is missing;
* :func:`cosa.active_set.multipliers.lagrangian_curvature`, which weights it by the conic
  multiplier and so makes the direction subproblem primal-*dual*;
* :func:`cosa.active_set.updates.deactivate_cones`, §7.4's rule, and the derivation that
  says it can never fire at an eq. (7) optimum.

The last of those is a claim about the *formulation*, so it is tested twice: once by
constructing multipliers that violate, to show the rule is real, and once by solving actual
portfolios and reading the multiplier off the answer, to show that eq. (7) never produces
such a multiplier.
"""

import numpy as np
import pytest

from cosa import SOCP, ConeStatus, Multipliers, ProblemError, SecondOrderCone, WorkingSet
from cosa.active_set import multipliers as mult
from cosa.active_set import updates
from cosa.experiments import portfolio as families
from cosa.experiments.randomized import random_instance
from cosa.geometry import soc, tangent
from cosa.linear_algebra import kkt
from cosa.solver import cosa as solver


@pytest.fixture
def instance():
    """A portfolio whose optimum has the cone active, which is the interesting case."""
    return families.basic(6, seed=3)


@pytest.fixture
def active(instance):
    """That instance's witness, its working set with the cone tangent, and the problem."""
    working_set = updates.set_cone_status(WorkingSet.empty(instance.problem), 0, ConeStatus.TANGENT)
    return instance.problem, instance.witness, working_set


# ----------------------------------------------------------------------------------
# The second derivative
# ----------------------------------------------------------------------------------


def test_the_curvature_is_the_projector_off_the_axis():
    """`(I - u u.T) / ||s_1||` in the tail block, and zero in the head row and column."""
    s = np.array([5.0, 3.0, 4.0])
    second = tangent.curvature(s)
    assert second.shape == (3, 3)
    assert not second[0].any()
    assert not second[:, 0].any()
    unit = np.array([0.6, 0.8])
    assert second[1:, 1:] == pytest.approx((np.eye(2) - np.outer(unit, unit)) / 5.0)


def test_there_is_no_curvature_along_the_radial_direction():
    """Moving straight out along the tail changes `||s_1||` at a constant rate.

    So the second derivative is singular in that direction, and says so exactly. This is
    the same degeneracy that makes the radial ray the one direction an exact conic step can
    move along -- #18's result, seen from the second derivative rather than the first.
    """
    s = np.array([13.0, 5.0, 12.0])
    radial = np.concatenate([[0.0], s[1:] / np.linalg.norm(s[1:])])
    assert tangent.curvature(s) @ radial == pytest.approx(np.zeros(3), abs=1e-12)


def test_the_curvature_is_positive_semidefinite():
    """Which is what keeps the direction subproblem convex once it is added to `rho*I`.

    Not a numerical accident: `g(s) = ||s_1|| - s_0` is convex, so its Hessian is positive
    semidefinite wherever it exists, and this is that fact in arithmetic.
    """
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = rng.normal(size=5)
        s[0] = float(np.linalg.norm(s[1:]))
        assert float(np.linalg.eigvalsh(tangent.curvature(s)).min()) >= -1e-12


def test_the_curvature_grows_as_the_apex_is_approached():
    """`1 / ||s_1||` blows up, which is the apex announcing itself from a distance."""
    near = tangent.curvature(np.array([1e-3, 1e-3, 0.0]))
    far = tangent.curvature(np.array([1.0, 1.0, 0.0]))
    assert float(np.abs(near).max()) > 100.0 * float(np.abs(far).max())


def test_the_apex_has_no_curvature_because_it_has_no_derivative():
    """Refused rather than returned as an enormous matrix. #24's branch is the answer."""
    with pytest.raises(tangent.ApexError):
        tangent.curvature(np.zeros(3))


def test_the_curvature_matches_a_finite_difference():
    """The definitional check: it is the second derivative of `g`, not merely a formula."""
    s = np.array([5.0, 3.0, 4.0])

    def g(point):
        """The cone constraint as a scalar function of the slack."""
        return float(np.linalg.norm(point[1:]) - point[0])

    step = 1e-5
    numerical = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            shift_i, shift_j = np.zeros(3), np.zeros(3)
            shift_i[i] = shift_j[j] = step
            numerical[i, j] = (g(s + shift_i + shift_j) - g(s + shift_i) - g(s + shift_j) + g(s)) / step**2
    assert tangent.curvature(s) == pytest.approx(numerical, abs=1e-4)


# ----------------------------------------------------------------------------------
# The Lagrangian's curvature: the dual variable entering the primal computation
# ----------------------------------------------------------------------------------


def test_an_inactive_cone_contributes_nothing(instance):
    """No active factor, no term -- and so the subproblem is exactly Wave 6's."""
    problem, witness = instance.problem, instance.witness
    empty = WorkingSet.empty(problem)
    zero = solver._no_multipliers(problem)
    assert not mult.lagrangian_curvature(problem, empty, witness, zero).any()


def test_a_zero_multiplier_contributes_nothing(active):
    """`mu = 0` says the constraint is not being pushed against, so it bends nothing."""
    problem, witness, working_set = active
    zero = solver._no_multipliers(problem)
    assert not mult.lagrangian_curvature(problem, working_set, witness, zero).any()


def test_the_contribution_is_the_multiplier_times_the_pushed_forward_curvature(active):
    """`mu * G.T @ grad^2 g @ G`, spelled out here and assembled there."""
    problem, witness, working_set = active
    slack = problem.cone_slack(witness)
    unit = tangent.unit_tail(slack)
    found = Multipliers(
        y=np.zeros(problem.num_inequalities),
        nu=np.zeros(problem.num_equalities),
        w=2.5 * np.concatenate([[1.0], -unit]),
    )
    built = mult.lagrangian_curvature(problem, working_set, witness, found)
    expected = 2.5 * (problem.G.T @ tangent.curvature(slack) @ problem.G)
    assert built == pytest.approx(expected)


def test_the_multiplier_is_read_off_the_head_of_w(active):
    """`w = mu * covector` and `covector`'s head is one, so `mu` is `w[0]`.

    Doubling the multiplier doubles the term, which is the linearity the derivation claims.
    """
    problem, witness, working_set = active
    unit = tangent.unit_tail(problem.cone_slack(witness))
    covector = np.concatenate([[1.0], -unit])

    def built(weight):
        """The curvature term for a `w` of the given magnitude along the covector."""
        found = Multipliers(
            y=np.zeros(problem.num_inequalities),
            nu=np.zeros(problem.num_equalities),
            w=weight * covector,
        )
        return mult.lagrangian_curvature(problem, working_set, witness, found)

    assert built(2.0) == pytest.approx(2.0 * built(1.0))


def test_a_dual_infeasible_multiplier_is_clipped_rather_than_trusted(active):
    """A negative `mu` would make the subproblem non-convex, so it contributes zero.

    The factor it belongs to is one §7.4 is about to remove; it should not be allowed to
    bend the direction on its way out.
    """
    problem, witness, working_set = active
    unit = tangent.unit_tail(problem.cone_slack(witness))
    found = Multipliers(
        y=np.zeros(problem.num_inequalities),
        nu=np.zeros(problem.num_equalities),
        w=-3.0 * np.concatenate([[1.0], -unit]),
    )
    assert not mult.lagrangian_curvature(problem, working_set, witness, found).any()


def test_an_apex_factor_contributes_nothing(instance):
    """`g` is not differentiable there, so there is no second derivative to weight."""
    problem = instance.problem
    apex = np.zeros(problem.num_variables)
    working_set = updates.set_cone_status(WorkingSet.empty(problem), 0, ConeStatus.APEX)
    found = Multipliers(
        y=np.zeros(problem.num_inequalities),
        nu=np.zeros(problem.num_equalities),
        w=np.concatenate([[1.0], np.zeros(problem.cone.dim - 1)]),
    )
    assert soc.is_apex(problem.cone_slack(apex))
    assert not mult.lagrangian_curvature(problem, working_set, apex, found).any()


def test_the_curvature_refuses_a_working_set_of_the_wrong_shape(instance):
    """The same shape guard `from_direction` applies, for the same reason."""
    problem = instance.problem
    other = WorkingSet.empty(SOCP.unconstrained(np.ones(2)))
    with pytest.raises(ProblemError, match="different instances"):
        mult.lagrangian_curvature(problem, other, instance.witness, solver._no_multipliers(problem))


# ----------------------------------------------------------------------------------
# The subproblem it produces
# ----------------------------------------------------------------------------------


def test_the_kkt_block_becomes_the_lagrangian_hessian(active):
    """`rho*I + curvature`, in the `(1, 1)` block and nowhere else."""
    problem, witness, working_set = active
    plain = kkt.assemble(problem, working_set, witness)
    bent = kkt.assemble(problem, working_set, witness, curvature=np.eye(problem.num_variables))
    n = problem.num_variables
    assert bent.matrix[:n, :n] == pytest.approx(plain.matrix[:n, :n] + np.eye(n))
    assert bent.matrix[n:, :] == pytest.approx(plain.matrix[n:, :])
    assert bent.rhs == pytest.approx(plain.rhs)


def test_no_curvature_is_the_wave_six_subproblem(active):
    """`None` must be bit-for-bit `H = rho*I`, or the comparison below means nothing."""
    problem, witness, working_set = active
    plain = kkt.assemble(problem, working_set, witness)
    explicit = kkt.assemble(problem, working_set, witness, curvature=None)
    assert np.array_equal(plain.matrix, explicit.matrix)


def test_a_curvature_of_the_wrong_shape_is_refused(active):
    """It is added to the variable block, so it must have the variable block's shape."""
    problem, witness, working_set = active
    with pytest.raises(ProblemError, match="curvature"):
        kkt.assemble(problem, working_set, witness, curvature=np.eye(problem.num_variables + 1))


def test_the_curvature_reaches_the_direction_through_the_one_call_form(active):
    """`direction` is `assemble` then `solve`, and it must pass the term along."""
    problem, witness, working_set = active
    bent = np.eye(problem.num_variables)
    assert kkt.direction(problem, working_set, witness, curvature=bent).d != pytest.approx(
        kkt.direction(problem, working_set, witness).d
    )


# ----------------------------------------------------------------------------------
# §7.4: deactivation on the multiplier
# ----------------------------------------------------------------------------------


def _multipliers(problem, w):
    """Multipliers with a given `w` and no linear part."""
    return Multipliers(y=np.zeros(problem.num_inequalities), nu=np.zeros(problem.num_equalities), w=w)


def test_a_factor_whose_multiplier_leaves_the_cone_is_released(active):
    """`w not in Q`: the normal points a way the problem does not want held."""
    problem, _, working_set = active
    outside = np.concatenate([[-1.0], np.zeros(problem.cone.dim - 1)])
    updated, dropped = updates.deactivate_cones(problem, working_set, _multipliers(problem, outside))
    assert dropped == (0,)
    assert updated.status(0) is ConeStatus.INACTIVE


def test_a_factor_with_no_multiplier_at_all_is_released(active):
    """`w = 0` is in `Q` and contributes nothing, so the cone is doing no work.

    Keeping it costs a row in `W` for no reason; releasing it lets the next direction move
    off the boundary, and §7.3 re-acquires it in one geometric test if it should not have.
    """
    problem, _, working_set = active
    updated, dropped = updates.deactivate_cones(problem, working_set, _multipliers(problem, np.zeros(problem.cone.dim)))
    assert dropped == (0,)
    assert updated.status(0) is ConeStatus.INACTIVE


def test_a_factor_with_a_genuine_active_normal_is_kept(active):
    """A nonzero `w` in `Q` is exactly what §7.4 asks for, so nothing happens."""
    problem, witness, working_set = active
    unit = tangent.unit_tail(problem.cone_slack(witness))
    found = _multipliers(problem, np.concatenate([[1.0], -unit]))
    updated, dropped = updates.deactivate_cones(problem, working_set, found)
    assert dropped == ()
    assert updated is working_set


def test_an_inactive_factor_is_not_a_candidate(instance):
    """It is already off; there is nothing to release and no multiplier worth reading."""
    problem = instance.problem
    empty = WorkingSet.empty(problem)
    _, dropped = updates.deactivate_cones(problem, empty, _multipliers(problem, np.zeros(problem.cone.dim)))
    assert dropped == ()


def test_the_apex_test_is_a_cone_membership_not_a_scalar(instance):
    """§8.1's distinction, in the one place it changes an answer.

    At a tangent factor `w` is pinned to the ray through the covector and the test reduces
    to a sign. At an apex factor `w` is a free block, so a head that is positive is not
    enough -- the tail has to fit under it.
    """
    problem = instance.problem
    apex_set = updates.set_cone_status(WorkingSet.empty(problem), 0, ConeStatus.APEX)
    tail = np.zeros(problem.cone.dim - 1)
    tail[0] = 5.0
    inside = _multipliers(problem, np.concatenate([[9.0], tail]))
    outside = _multipliers(problem, np.concatenate([[1.0], tail]))
    assert updates.deactivate_cones(problem, apex_set, inside)[1] == ()
    assert updates.deactivate_cones(problem, apex_set, outside)[1] == (0,)


def test_deactivation_is_the_multiplier_and_not_the_slack(instance):
    """The rule §7.4 insists on, and the reason it insists.

    The iterate here is well inside the cone -- the geometry has no case at all for holding
    the factor -- and the rule keeps it anyway, because the multiplier says it contributes a
    genuine active normal. A slack-based rule would deactivate, the unconstrained direction
    would walk straight back, and §7.3 would reactivate: that cycle is what this avoids.
    """
    problem = instance.problem
    interior = instance.witness.copy()
    interior[-1] += 10.0
    working_set = updates.set_cone_status(WorkingSet.empty(problem), 0, ConeStatus.TANGENT)
    unit = tangent.unit_tail(problem.cone_slack(interior))
    found = _multipliers(problem, np.concatenate([[1.0], -unit]))
    assert soc.is_interior(problem.cone_slack(interior))
    assert updates.deactivate_cones(problem, working_set, found)[1] == ()


# ----------------------------------------------------------------------------------
# The derivation: on eq. (7) the rule provably never fires
# ----------------------------------------------------------------------------------


def test_an_active_cone_at_an_optimum_always_has_a_genuine_active_normal():
    """The module docstring's derivation, checked against actual solves.

    `t` appears in one linear row and in the objective with coefficient `lam`, so the
    direction subproblem's stationarity in the `t` slot gives `w_0 = lam + rho * d_t`, which
    at a stationary point is `lam`. The tangent structure then forces `||w_tail|| = w_0`, so
    `w` sits on the boundary of `Q` with a strictly positive head: neither outside `Q` nor
    zero, whatever the instance.
    """
    for seed in range(6):
        instance = families.basic(7, seed=seed)
        answer = solver.solve(instance.problem)
        assert answer.status == "optimal"
        if not answer.working_set.active_cones:
            continue
        w = answer.multipliers.w
        assert float(w[0]) == pytest.approx(instance.portfolio.lam, rel=1e-6)
        assert float(np.linalg.norm(w[1:])) == pytest.approx(float(w[0]), rel=1e-6)
        assert updates.deactivate_cones(instance.problem, answer.working_set, answer.multipliers)[1] == ()


def test_the_solver_never_releases_a_cone_on_a_portfolio():
    """The same statement as a property of the loop rather than of its answer.

    Every cone that joins a working set on eq. (7) is still in it at termination -- which is
    a finding about the formulation, not about the rule: `lam > 0` is what makes the head of
    `w` positive, and a problem whose objective did not charge for risk would not. Read
    through §7.4 directly, so that a rule which had fired somewhere along the way and left
    the factor reactivated would still be caught.
    """
    for family in (families.basic, families.box, families.sector, families.turnover):
        instance = family(6, seed=2)
        answer = solver.solve(instance.problem)
        assert answer.status == "optimal"
        assert answer.working_set.active_cones == (0,)
        assert updates.deactivate_cones(instance.problem, answer.working_set, answer.multipliers)[1] == ()


# ----------------------------------------------------------------------------------
# What the curvature buys, measured
# ----------------------------------------------------------------------------------


def _iterations(problem, *, bend):
    """Solve with the Lagrangian curvature on or off, and report the count and the status."""
    if bend:
        answer = solver.solve(problem)
    else:
        real = solver.lagrangian_curvature
        solver.lagrangian_curvature = lambda *_args, **_kwargs: None
        try:
            answer = solver.solve(problem)
        finally:
            solver.lagrangian_curvature = real
    return answer.metrics.iterations, answer.status


def test_the_curvature_halves_the_iteration_count():
    """§9 Phase IV's point, as a number rather than an argument.

    Across the family suite the Lagrangian Hessian roughly halves the work. A first-order
    method on a curved surface converges linearly and this one does not have to.
    """
    total = {True: 0, False: 0}
    for family in (families.basic, families.sector, families.factor_exposure):
        for seed in range(4):
            problem = family(8, seed=seed).problem
            for bend in (True, False):
                total[bend] += _iterations(problem, bend=bend)[0]
    assert total[True] < 0.75 * total[False], total


def test_the_curvature_solves_instances_that_rho_i_cannot():
    """Not merely faster: the difference between an answer and an iteration limit.

    `ill_conditioned` is the family built for #33's robustness question, and under
    `H = rho*I` the tangent-plane step is short enough that the loop runs out of iterations
    with a residual around `1e-2`. With the cone's curvature in the subproblem it converges.
    """
    problem = families.ill_conditioned(8, seed=0).problem
    assert _iterations(problem, bend=False)[1] == "iteration_limit"
    bent = solver.solve(problem)
    assert bent.status == "optimal"
    assert bent.residuals.is_optimal()


def test_the_first_direction_is_unchanged_by_the_curvature(active):
    """The fixed point starts at zero, so iteration one is bit-for-bit Wave 6's.

    Which is what makes the comparison above attributable: any difference between the two
    solves comes from the curvature, not from a different starting subproblem.
    """
    problem, witness, working_set = active
    zero = solver._no_multipliers(problem)
    built = mult.lagrangian_curvature(problem, working_set, witness, zero)
    assert np.array_equal(
        kkt.direction(problem, working_set, witness, curvature=built).d,
        kkt.direction(problem, working_set, witness).d,
    )


# ----------------------------------------------------------------------------------
# §7.4 inside the loop, on a problem that is not eq. (7)
# ----------------------------------------------------------------------------------


_ON_THE_BOUNDARY = np.array([1.0, 1.0, 0.0])
"""The start :func:`_cone_wants_releasing` is entered at: feasible, and on the cone."""


def _cone_wants_releasing() -> SOCP:
    """A three-variable SOCP whose first working set holds a cone it should not.

    ``z = (t, y1, y2)``, the cone is ``||y|| <= t`` on the variables themselves, the
    equalities pin ``y = (1, 0)``, and the objective *rewards* a large ``t`` up to a bound.
    The optimum is therefore strictly inside the cone at ``t = 5``, but the constructed
    start is on the boundary at ``t = 1`` -- supplied rather than constructed, because
    ``t`` carries an upper bound and so is not one of the free heads
    :func:`cosa.solver.initialization.raise_free_heads` may raise --
    so §7.3 activates the factor at the start and the tangent row then pins the direction to
    zero. Nothing geometric can undo that: the iterate is exactly on the boundary and the
    working set is exactly stationary. Only the multiplier knows the cone is being held
    backwards, and it says so loudly, with ``w = (-1, 1, 0)``.

    It is deliberately *not* a portfolio: eq. (7) charges ``lam`` for ``t`` and so can never
    produce this, which is the whole content of the derivation above.
    """
    return (
        SOCP.unconstrained(np.array([-1.0, 0.0, 0.0]))
        .add_inequalities([[1.0, 0.0, 0.0]], [5.0])
        .add_equalities([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [1.0, 0.0])
        .add_cone(SecondOrderCone(3), np.eye(3), np.zeros(3))
    )


def test_a_wrongly_held_cone_is_released_and_the_solve_completes():
    """§7.4 driving the loop: the release is what makes the instance solvable at all."""
    problem = _cone_wants_releasing()
    answer = solver.solve(problem, start=_ON_THE_BOUNDARY)
    assert answer.status == "optimal"
    assert answer.z == pytest.approx(np.array([5.0, 1.0, 0.0]))
    assert answer.working_set.status(0) is ConeStatus.INACTIVE
    assert answer.residuals.is_optimal()


def test_the_released_factor_is_the_one_whose_multiplier_left_the_cone():
    """Read the offending multiplier directly, so the mechanism is pinned and not inferred."""
    problem = _cone_wants_releasing()
    start = _ON_THE_BOUNDARY
    working_set = updates.set_cone_status(WorkingSet.empty(problem), 0, ConeStatus.TANGENT)
    found = mult.from_direction(problem, working_set, start, kkt.direction(problem, working_set, start))
    assert found.w == pytest.approx(np.array([-1.0, 1.0, 0.0]))
    assert mult.dual_cone_violation(problem, found)[0] == pytest.approx(2.0)
    assert updates.deactivate_cones(problem, working_set, found)[1] == (0,)


def test_without_the_rule_that_instance_never_moves():
    """The counterfactual, which is what makes the release load-bearing rather than tidy.

    With §7.4 switched off the loop reaches a stationary point of a working set it has no
    way to leave -- the direction is zero, no inequality is active to drop, and the cone is
    held -- so it reports the point as optimal when it is nothing of the kind. The residuals
    are the ones that catch it, which is Success Criterion 2 doing its job.
    """
    problem = _cone_wants_releasing()
    real = updates.deactivate_cones
    updates.deactivate_cones = lambda _p, w, _m, **_k: (w, ())
    try:
        answer = solver.solve(problem, start=_ON_THE_BOUNDARY)
    finally:
        updates.deactivate_cones = real
    assert answer.z == pytest.approx(np.array([1.0, 1.0, 0.0]))
    assert not answer.residuals.is_optimal()


def test_the_curvature_is_what_makes_the_randomized_sweep_converge():
    """§16.3's generator, which is the hardest thing the solver is pointed at.

    ``experiments/randomized.py`` randomizes the *shape* -- dimension, rank, conditioning,
    active-set structure -- so a sweep over it is not a family with a favourable geometry.
    Over the first two hundred seeds the curvature takes the iteration count from 62631 to
    19386 and the number of instances that reach an optimum from 162 to 192. Thirty seeds
    are checked here, which is enough to see the gap without making the suite slow, and the
    assertions are loose because the point is the direction and not the width.
    """
    solved = {True: 0, False: 0}
    total = {True: 0, False: 0}
    for seed in range(30):
        problem = random_instance(seed).problem
        for bend in (True, False):
            iterations, status = _iterations(problem, bend=bend)
            total[bend] += iterations
            solved[bend] += status == "optimal"
    assert solved[True] > solved[False], solved
    assert total[True] < 0.5 * total[False], total
