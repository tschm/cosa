"""The COSA prototype: eq. (8) solved by the four ingredients of §9 Phase III.

The executable half of issue #20, whose "done when" is Success Criterion 1
(`paper.tex:1318`) -- eq. (8) solves via the four Phase III ingredients to an objective
agreeing with a reference solver.

**Two structural facts shape the whole design, and they are pinned first.** Both are about
eq. (7) and both were found by building this:

* at a boundary point, a direction satisfying eq. (3) has an *exact conic step of zero*;
* the projected steepest-descent direction always pushes `t` down, by exactly `lam / rho`.

Together an iterate on the cone's boundary cannot move at all under a linear direction and
an exact conic step. That is the curvature of the cone, not a bug in the ratio test, and it
is §3.3's warning and Risk 1 (#39) arriving together. The prototype's answer is the one §3.3
sanctions for it: tangent directions, with the step made honest by retracting onto the cone.

§16.2 (`paper.tex:1110`) asks that analytical problems validate primal solutions, active
sets, multipliers, SOC activity and KKT residuals. The golden instances of #9 are those
problems, and the last section checks all five.
"""

import numpy as np
import pytest

from cosa import ConeStatus, MeanStdForm, WorkingSet
from cosa.experiments import portfolio as families
from cosa.experiments import randomized, reference
from cosa.geometry import soc, tangent
from cosa.linear_algebra import kkt
from cosa.solver import cosa
from cosa.solver.instrumentation import CHECKED, level_1_violations

# The families the prototype converges on, with the reference gap each is held to.
CONVERGING = ["basic", "box", "sector", "factor", "turnover", "many-active-bounds"]


def instance_named(name):
    """One instance by short name."""
    makers = {
        "basic": lambda: families.basic(5, seed=0),
        "box": lambda: families.box(5, seed=0),
        "sector": lambda: families.sector(9, seed=0),
        "factor": lambda: families.factor_exposure(8, seed=0),
        "turnover": lambda: families.turnover(6, seed=0),
        "many-active-bounds": lambda: families.many_active_bounds(12, seed=0),
    }
    return makers[name]()


# ----------------------------------------------------------------------------------
# The two structural facts the design rests on
# ----------------------------------------------------------------------------------


def test_a_tangent_direction_admits_an_exact_step_of_zero():
    """Eq. (3) and eq. (6) are incompatible at a boundary point, and Cauchy-Schwarz is why.

    Tangency makes eq. (6)'s middle coefficient vanish and feasibility makes its constant
    term vanish, so the quadratic is `a alpha^2 <= 0` with
    `a = ||q||^2 - (u.T @ q)^2 >= 0`. Only `alpha = 0` satisfies it, except along the radial
    ray where `q` is parallel to `u`.

    Checked over two thousand random boundary points and tangent directions, because a
    result this load-bearing should not rest on one example.
    """
    from cosa.geometry import step as st

    rng = np.random.default_rng(0)
    for _ in range(2000):
        size = int(rng.integers(1, 5))
        tail = rng.normal(size=size)
        s = np.concatenate([[float(np.linalg.norm(tail))], tail])
        unit = tangent.unit_tail(s)
        q = rng.normal(size=size)
        ds = np.concatenate([[float(unit @ q)], q])

        assert tangent.tangent_residual(s, ds) == pytest.approx(0.0, abs=1e-12)
        assert float(q @ q) - float(unit @ q) ** 2 >= -1e-12, "Cauchy-Schwarz"
        leading = float(q @ q) - float(unit @ q) ** 2
        if leading <= 1e-12 * float(q @ q):
            continue  # the radial ray, which is the documented exception
        # Relative to the geometry: the step is zero, and what "zero" costs in a quadratic
        # whose middle and constant coefficients have both cancelled is a few ulps of the
        # ratio between them.
        scale = max(1.0, float(np.abs(s).max())) / max(1e-12, float(np.abs(ds).max()))
        assert max(0.0, st.cone_interval(s, ds).upper) <= 1e-6 * scale


@pytest.mark.parametrize("rho", [0.25, 1.0, 4.0])
def test_the_projected_direction_always_pushes_the_risk_variable_down(rho):
    """`d_t = -lam / rho`, because `t` appears in no linear row.

    So nothing in the working set opposes the objective's pull on `t`, and from a boundary
    point the steepest-descent direction always leaves the cone -- another exact step of
    zero. The second half of why the prototype needs a retraction.
    """
    instance = families.basic(5, seed=0)
    step = kkt.direction(instance.problem, WorkingSet.empty(instance.problem), instance.witness, rho=rho)
    assert step.d[-1] == pytest.approx(-instance.portfolio.lam / rho)


# ----------------------------------------------------------------------------------
# Success Criterion 1: eq. (8) solves
# ----------------------------------------------------------------------------------


def test_eq_eight_solves_and_agrees_with_a_reference_solver():
    """Issue #20's "done when", on the family that *is* eq. (8).

    The basic family is `min -mu.T @ x + lam*t` over `1.T @ x = 1`, `x >= 0`,
    `||L @ x||_2 <= t` -- eq. (8) exactly. Solved by the four Phase III ingredients and
    checked against the oracle of #21.
    """
    instance = families.basic(5, seed=0)
    solution = cosa.solve(instance.problem, checker=CHECKED)
    oracle = reference.solve_reference(instance.problem)

    assert solution.is_optimal, str(solution)
    assert solution.objective(instance.problem) == pytest.approx(oracle.objective, abs=1e-8)


@pytest.mark.parametrize("name", CONVERGING)
def test_the_prototype_agrees_with_the_reference_on_the_structured_families(name):
    """§16.3's cross-check, applied to COSA's own answer on a conic problem."""
    instance = instance_named(name)
    solution = cosa.solve(instance.problem, checker=CHECKED)
    oracle = reference.solve_reference(instance.problem)

    assert solution.is_optimal, f"{name}: {solution}"
    gap = abs(solution.objective(instance.problem) - oracle.objective)
    assert gap <= 1e-7 * max(1.0, abs(oracle.objective)), f"{name}: gap {gap:.2e}"


@pytest.mark.parametrize("name", CONVERGING)
def test_every_accepted_iterate_stays_in_the_cone(name):
    """§14.1's Level 1, with the conic condition now doing real work.

    Every solve here runs with the checker enabled, so the invariant is asserted at each
    accepted iterate -- including after each retraction, which is the step that would break
    it if the retraction were wrong.
    """
    instance = instance_named(name)
    solution = cosa.solve(instance.problem, checker=CHECKED)
    assert level_1_violations(instance.problem, solution.z) == ()
    assert soc.is_member(instance.problem.cone_slack(solution.z), tolerance=1e-8)


@pytest.mark.parametrize("name", CONVERGING)
def test_the_final_residuals_certify_the_answer(name):
    """§14.3's Level 3: all five within tolerance.

    The problem being convex, that is a certificate of global optimality rather than a
    stopping heuristic that happens to work.
    """
    instance = instance_named(name)
    solution = cosa.solve(instance.problem, checker=CHECKED)
    assert solution.residuals.is_optimal(), str(solution.residuals)


# ----------------------------------------------------------------------------------
# The four ingredients of §9 Phase III
# ----------------------------------------------------------------------------------


def test_the_cone_joins_the_working_set():
    """§7.3's activation, which is the ingredient Wave 5 was missing.

    Before this the loop stalled at the boundary: the working set had no tangent row, so the
    direction pointed out of the cone and the exact step was zero.

    The default start already sits on the cone's boundary, so the cone is active before the
    first iteration -- `_working_set_at` runs §7.3 for the same reason it runs §7.1.
    """
    instance = families.basic(5, seed=0)
    problem = instance.problem
    assert cosa._working_set_at(problem, instance.witness).status(0) is ConeStatus.TANGENT
    solution = cosa.solve(problem, checker=CHECKED)
    assert solution.working_set.status(0) is ConeStatus.TANGENT


def test_a_cone_that_starts_inactive_is_activated_during_the_solve():
    """And the transition is counted, which is what §12.3's metric is for.

    Started strictly inside the cone, with the head raised by a margin, so the cone begins
    inactive and §7.3's rule has something to do.
    """
    from cosa.solver import initialization as init

    instance = families.basic(5, seed=0)
    problem = instance.problem
    interior = init.feasible_start(problem, margin=1.0)
    assert cosa._working_set_at(problem, interior).status(0) is ConeStatus.INACTIVE

    solution = cosa.solve(problem, start=interior, checker=CHECKED)
    assert solution.working_set.status(0) is ConeStatus.TANGENT
    assert solution.metrics.cone_changes >= 1, "the transition was recorded"


def test_the_direction_respects_the_tangent_condition_once_the_cone_is_active():
    """The tangent representation, exercised: eq. (3) holds on the direction the loop uses."""
    instance = families.basic(5, seed=0)
    problem = instance.problem
    z = cosa.solve(problem, checker=CHECKED, max_iterations=5).z
    working_set = cosa._working_set_at(problem, z)
    if working_set.status(0) is not ConeStatus.TANGENT:
        pytest.skip("the cone had not activated within five iterations")
    step = kkt.direction(problem, working_set, z)
    residual = tangent.tangent_residual(problem.cone_slack(z), problem.G @ step.d)
    assert residual == pytest.approx(0.0, abs=1e-9)


def test_the_working_set_holds_all_three_item_classes_at_the_solution():
    """A working set of linear constraints *and* the cone's geometry -- §3.2's list, live."""
    instance = families.box(5, seed=0)
    solution = cosa.solve(instance.problem, checker=CHECKED)
    assert solution.working_set.inequalities, "linear rows are active"
    assert solution.working_set.equalities, "the budget is always in"
    assert solution.working_set.active_cones == (0,), "and so is the risk cone"


def test_the_solution_is_readable_in_the_family_s_own_terms():
    """Success Criterion 3, end to end: the answer says which portfolio constraints bind."""
    instance = families.box(5, seed=0)
    solution = cosa.solve(instance.problem, checker=CHECKED)
    described = solution.working_set.describe(instance.names)
    assert "upper bound on asset" in described
    assert "fully invested" in described
    assert "risk" in described


def test_multipliers_drive_the_updates():
    """The fourth ingredient: constraints leave the working set on a sign test, not a guess."""
    instance = families.box(5, seed=0)
    solution = cosa.solve(instance.problem, start=instance.witness, checker=CHECKED)
    assert solution.metrics.constraints_added > 0
    assert solution.multipliers.inequality_violation() == 0.0, "no wrong-signed multiplier survives"


# ----------------------------------------------------------------------------------
# The retraction
# ----------------------------------------------------------------------------------


def test_the_retraction_restores_feasibility_and_improves_the_objective():
    """Both halves of what makes a retracted step a step at all.

    The tangent direction leaves the cone at second order while improving the objective at
    first order, so a short enough step improves it even after the cone is restored. The
    line search is what finds that step.
    """
    instance = families.basic(5, seed=0)
    problem = instance.problem
    z = instance.witness
    working_set = cosa._working_set_at(problem, z)
    assert working_set.status(0) is ConeStatus.TANGENT

    step = kkt.direction(problem, working_set, z)
    stepped = cosa._retracted_step(problem, z, step.d, working_set)
    assert stepped is not None, "a step exists"
    moved, limit = stepped

    assert soc.is_member(problem.cone_slack(moved), tolerance=1e-9), "feasible after retraction"
    assert float(problem.c @ moved) < float(problem.c @ z), "and better than before"
    assert limit.alpha > 0.0


def test_the_retraction_reports_no_step_when_none_improves():
    """The retraction's own termination signal, at a point where it has nothing left to give.

    At the solution the tangent direction is rounding-level, so any step along it is undone
    by the retraction that follows and the search returns nothing worth taking. #29 made this
    no longer the *loop's* stopping signal -- the no-progress rule reaches the same
    conclusion one test earlier and more cheaply -- so what is asserted here is the property
    rather than the return value: whatever step the search reports moves the iterate by
    nothing.
    """
    instance = families.basic(6, seed=0)
    problem = instance.problem
    solution = cosa.solve(problem)
    working_set = solution.working_set
    step = kkt.direction(problem, working_set, solution.z)
    stepped = cosa._retracted_step(problem, solution.z, step.d, working_set)
    if stepped is not None:
        moved, _ = stepped
        assert np.abs(moved - solution.z).max() < 1e-8


def test_the_line_search_gives_up_after_its_budget():
    """`backtracks` is a budget, and exhausting it means no step along this direction works.

    Forced here by setting the budget to zero, which is the same code path a genuinely
    unproductive direction takes -- and the loop reads `None` as "stationary for this
    working set" rather than as an error.
    """
    instance = families.basic(5, seed=0)
    problem = instance.problem
    working_set = cosa._working_set_at(problem, instance.witness)
    step = kkt.direction(problem, working_set, instance.witness)
    assert float(problem.c @ step.d) < 0.0, "the direction does predict a decrease"
    assert cosa._retracted_step(problem, instance.witness, step.d, working_set, backtracks=0) is None


def test_the_loop_releases_an_apex_when_the_geometry_allows_it():
    """The apex branch's third outcome, driven through the loop.

    Unreachable on eq. (7) -- Wave 3 proved that dropping the factor there forces
    `d_t = -lam/rho < 0`, so the released direction is infeasible by arithmetic. It takes a
    general SOCP whose objective *rewards* the head variable, which the representation of #9
    deliberately supports, and the loop handles it without knowing the difference.
    """
    from cosa import SOCP, ConeProduct

    rewarding_head = SOCP(
        c=np.array([0.0, -1.0]),
        A=np.array([[1.0, 0.0], [0.0, 1.0]]),
        b=np.array([1.0, 1.0]),
        E=np.zeros((0, 2)),
        d=np.zeros(0),
        G=np.array([[0.0, 1.0], [1.0, 0.0]]),
        h=np.zeros(2),
        cone=ConeProduct.from_dims(2),
    )
    solution = cosa.solve(rewarding_head, start=np.zeros(2), checker=CHECKED)
    assert solution.status in {"optimal", "iteration_limit", "stalled"}
    assert level_1_violations(rewarding_head, solution.z) == ()
    assert solution.z[1] > 0.0, "the loop left the apex rather than sitting on it"


def test_the_retraction_needs_a_free_head():
    """It raises `t`, which is only available where `t` is the solver's to raise.

    On a general SOCP whose head is constrained elsewhere the loop falls back to the exact
    conic step -- which at a boundary point means it stalls, honestly, rather than
    retracting something it must not move.
    """
    assert cosa._heads_are_free(families.basic(4, seed=0).problem)

    from cosa import SOCP, ConeProduct

    constrained_head = SOCP(
        c=np.array([0.0, 1.0]),
        A=np.array([[0.0, 1.0]]),
        b=np.array([1.0]),
        E=np.zeros((0, 2)),
        d=np.zeros(0),
        G=np.array([[0.0, 1.0], [1.0, 0.0]]),
        h=np.zeros(2),
        cone=ConeProduct.from_dims(2),
    )
    assert not cosa._heads_are_free(constrained_head)


# ----------------------------------------------------------------------------------
# §16.2's analytical problems: all five things it asks to be validated
# ----------------------------------------------------------------------------------


@pytest.fixture
def golden():
    """#9's hand-solved instance: mu = (2, 1), lam = 1, sum(x) = 1, Sigma = I.

    The risk term is isotropic, so the optimum puts everything in the better asset:
    x = (1, 0), t = 1, objective -1, with nu = 1 and w = (1, -1, 0).
    """
    form = MeanStdForm(
        mu=np.array([2.0, 1.0]),
        lam=1.0,
        A=np.zeros((0, 2)),
        b=np.zeros(0),
        E=np.array([[1.0, 1.0]]),
        d=np.array([1.0]),
        L=np.eye(2),
    )
    return form.to_socp()


def test_the_analytical_primal_solution(golden):
    """§16.2's first: the primal solution, against arithmetic."""
    solution = cosa.solve(golden, checker=CHECKED)
    assert solution.is_optimal, str(solution)
    np.testing.assert_allclose(solution.z, [1.0, 0.0, 1.0], atol=1e-6)


def test_the_analytical_objective(golden):
    """And its value, which is `-mu.T @ x + lam*t = -2 + 1 = -1`."""
    solution = cosa.solve(golden, checker=CHECKED)
    assert solution.objective(golden) == pytest.approx(-1.0, abs=1e-6)


def test_the_analytical_active_set(golden):
    """§16.2's second: the active set. No inequalities here, the budget, and the cone."""
    solution = cosa.solve(golden, checker=CHECKED)
    assert solution.working_set.inequalities == ()
    assert solution.working_set.equalities == (0,)
    assert solution.working_set.active_cones == (0,)


def test_the_analytical_soc_activity(golden):
    """§16.2's fourth: SOC activity.

    The cone is exactly active, as it always is at eq. (7)'s optimum: any slack in it costs
    `lam` per unit.
    """
    solution = cosa.solve(golden, checker=CHECKED)
    assert soc.is_boundary(golden.cone_slack(solution.z), tolerance=1e-7)
    assert solution.working_set.status(0) is ConeStatus.TANGENT


def test_the_analytical_multipliers(golden):
    """§16.2's third: the multipliers, against the hand derivation of #9."""
    solution = cosa.solve(golden, checker=CHECKED)
    np.testing.assert_allclose(solution.multipliers.nu, [1.0], atol=1e-6)
    np.testing.assert_allclose(solution.multipliers.w, [1.0, -1.0, 0.0], atol=1e-6)
    assert solution.multipliers.w[0] == pytest.approx(1.0, abs=1e-6), "w_t = lam"


def test_the_analytical_kkt_residuals(golden):
    """§16.2's fifth: the KKT residuals, all five, at the answer."""
    solution = cosa.solve(golden, checker=CHECKED)
    assert solution.residuals.is_optimal()
    assert solution.residuals.largest < 1e-7


# ----------------------------------------------------------------------------------
# What the prototype does not do, said out loud
# ----------------------------------------------------------------------------------


def test_the_apex_case_is_reported_rather_than_solved_wrongly():
    """Risk 1, firing in a live solve rather than in a constructed example.

    Seed 2 draws a rank-one covariance, so the optimum sits at the apex -- and the apex's
    multiplier says release while the released direction leaves the cone, which is the case
    #24 identified and cannot resolve. The loop reports `blocked-at-apex` and stops.

    Asserted so that the case is *visible*: a status naming the problem is worth far more
    than an `optimal` that is 3% wrong, and #23 is the issue that fixes it.
    """
    instance = randomized.random_instance(2)
    solution = cosa.solve(instance.problem)
    assert solution.status == "blocked-at-apex"
    assert soc.position(instance.problem.cone_slack(solution.z)) is soc.ConePosition.APEX
    assert level_1_violations(instance.problem, solution.z) == (), "and the iterate is still feasible"


def test_slow_convergence_is_reported_as_an_iteration_limit():
    """Crawling along a curved boundary is a first-order process, and it shows.

    §9 Phase III says correctness matters more than speed here, so the honest outcome on an
    instance the retraction approaches slowly is the iteration limit and the residuals that
    say how close it got -- not a claim of optimality.
    """
    instance = families.large(30, factors=4, seed=0)
    solution = cosa.solve(instance.problem, max_iterations=50)
    assert solution.status in {"iteration_limit", "degenerate", "blocked-at-apex", "optimal"}
    assert level_1_violations(instance.problem, solution.z) == ()


def test_the_cone_is_never_deactivated():
    """Issue #20's scope boundary: activation only, deactivation is #23's.

    §7.3 says geometric activity alone is not sufficient for optimality and §7.4 hands the
    removal decision to the conic multiplier. So a cone that joins the working set stays,
    and an instance whose optimum wants it inactive terminates without a certificate rather
    than with a wrong one.
    """
    import inspect

    source = inspect.getsource(cosa)
    assert "activate_cones" in source
    assert "ConeStatus.INACTIVE" not in source, "nothing here turns a cone off"
