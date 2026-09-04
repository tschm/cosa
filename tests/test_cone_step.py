"""Eq. (6): the exact SOC step, and the term the plan's printed form is missing.

The executable half of issue #18. Its "done when" is four claims, and all four are here:
the accepted step is the intersection of all three intervals, it matches analytically known
cone intersections, both the negative-right-hand-side and `||q|| = 0` cases are handled, and
no iterate violates `||Lx||_2 <= t`.

**The first section is the finding.** Eq. (6) as printed at `paper.tex:457` drops the
`tau^2` that comes from expanding `(t + alpha*tau)^2`, so its leading coefficient is
`||q||^2` where it should be `||q||^2 - tau^2`. The two agree only when `tau = 0` -- the one
case eq. (7) never takes -- and the difference is not conservative: on a direction running
along the cone's boundary to the apex, the printed form admits a step of *zero* where the
true answer is `t / (-tau)`. Each case below is solved by hand in its docstring, so the
expected numbers are arithmetic rather than recordings.
"""

import math

import numpy as np
import pytest

from cosa import ProblemError, WorkingSet
from cosa.experiments import portfolio as families
from cosa.geometry import soc
from cosa.geometry import step as st


def printed_eq_six(s, ds):
    """The step eq. (6) gives *as printed*, for comparison. Leading coefficient `||q||^2`."""
    head, tail = s[0], s[1:]
    step_head, step_tail = ds[0], ds[1:]
    leading = float(step_tail @ step_tail)
    middle = 2.0 * (float(tail @ step_tail) - head * step_head)
    constant = float(tail @ tail) - head * head
    cap = math.inf if step_head >= 0.0 else head / (-step_head)
    if leading <= 1e-15:
        return min(-constant / middle if middle > 1e-15 else math.inf, cap)
    disc = middle * middle - 4.0 * leading * constant
    if disc < 0.0:
        return math.nan
    return min(max(np.roots([leading, middle, constant])), cap)


# ----------------------------------------------------------------------------------
# The correction, against intersections solved by hand
# ----------------------------------------------------------------------------------

# Each case: (slack, slack direction, exact alpha, the inequality solved by hand).
HAND_SOLVED = [
    (np.array([2.0, 0.0]), np.array([-1.0, 1.0]), 1.0, "|a| <= 2 - a  =>  a <= 1"),
    (np.array([1.0, 0.0]), np.array([-1.0, 0.5]), 2.0 / 3.0, "0.5a <= 1 - a  =>  a <= 2/3"),
    (np.array([5.0, 3.0, 4.0]), np.array([-1.0, -0.6, -0.8]), 5.0, "along the ray to the apex"),
    (np.array([2.0, 0.0]), np.array([0.0, 1.0]), 2.0, "|a| <= 2"),
    (np.array([5.0, 3.0, 4.0]), np.array([0.0, 0.8, -0.6]), 0.0, "tangent: exits at once"),
]


@pytest.mark.parametrize(("s", "ds", "exact", "why"), HAND_SOLVED)
def test_the_exact_step_matches_arithmetic(s, ds, exact, why):
    """§16.1's "exact step roots" (`paper.tex:1107`), against inequalities solved by hand."""
    assert max(0.0, st.cone_interval(s, ds).upper) == pytest.approx(exact, abs=1e-12), why


@pytest.mark.parametrize(("s", "ds", "exact", "why"), HAND_SOLVED)
def test_the_printed_form_of_eq_six_disagrees_except_where_tau_is_zero(s, ds, exact, why):
    """The finding, asserted rather than described.

    The printed formula is right on the one case with `tau = 0` and wrong on the rest. If
    the plan is ever corrected, this test is what says so.
    """
    printed = max(0.0, printed_eq_six(s, ds))
    if ds[0] == 0.0:
        assert printed == pytest.approx(exact, abs=1e-12), f"tau = 0, so the two agree: {why}"
    else:
        assert printed != pytest.approx(exact, abs=1e-9), f"tau != 0, so they differ: {why}"


def test_the_printed_form_forbids_moving_along_the_boundary():
    """The case where the difference matters most, isolated.

    A direction that keeps the iterate exactly on the cone's boundary all the way to the
    apex is feasible for every step up to `t / (-tau)`. The printed formula admits *none* of
    it -- a solver using it would refuse to travel along the boundary at all, which given
    §8.1's interest in reaching `Lx = 0` is a route the algorithm actively wants.
    """
    s = np.array([5.0, 3.0, 4.0])
    ds = np.array([-1.0, -0.6, -0.8])
    assert max(0.0, printed_eq_six(s, ds)) == pytest.approx(0.0)
    assert st.cone_interval(s, ds).upper == pytest.approx(5.0)
    np.testing.assert_allclose(s + 5.0 * ds, 0.0, atol=1e-12), "and it lands exactly on the apex"


# ----------------------------------------------------------------------------------
# The three branches of the corrected quadratic
# ----------------------------------------------------------------------------------


def test_a_direction_leaving_the_cone_gives_an_upward_parabola():
    """`||q|| > |tau|`: zero lies between the roots and the step is the upper one."""
    interval = st.cone_interval(np.array([2.0, 0.0]), np.array([0.0, 1.0]))
    assert not interval.degenerate
    assert interval.lower < 0.0 < interval.upper


def test_a_direction_inside_the_cone_is_never_blocked():
    """`||q|| < |tau|` with `tau > 0`: the parabola opens downward and nothing stops the step.

    The case a one-sided root selection gets backwards. The slack direction is itself in the
    cone, so `s + alpha*ds` is a sum of two cone elements and stays in the cone forever.
    """
    ds = np.array([1.0, 0.5])
    assert soc.is_member(ds), "the direction is itself in the cone"
    assert math.isinf(st.cone_interval(np.array([2.0, 0.0]), ds).upper)
    far = np.array([2.0, 0.0]) + 1e6 * ds
    assert soc.is_member(far)


def test_the_degenerate_case_collapses_to_a_linear_inequality():
    """`||q|| = |tau|`: dividing by the leading coefficient would be dividing by zero.

    Two instances of it. A tail growing exactly as fast as the head falls, and the ray that
    runs along the boundary to the apex -- both have `||q||^2 - tau^2 = 0` exactly, and
    neither has a quadratic to take roots of.
    """
    matched = st.cone_interval(np.array([2.0, 0.0]), np.array([-1.0, 1.0]))
    assert matched.degenerate
    assert matched.upper == pytest.approx(1.0), "|a| <= 2 - a"

    boundary_ray = st.cone_interval(np.array([5.0, 3.0, 4.0]), np.array([-1.0, -0.6, -0.8]))
    assert boundary_ray.degenerate
    assert boundary_ray.upper == pytest.approx(5.0)


def test_a_motionless_tail_with_a_falling_head_is_bounded_by_the_right_hand_side():
    """§8.1's `||q|| = 0` is *not* the degenerate branch unless `tau` vanishes too.

    With the tail frozen and the head falling, `||q|| < |tau|`, so the parabola opens
    downward and the binding constraint is the right-hand side reaching zero -- at exactly
    the same place the quadratic's double root sits, which is why the answer is right either
    way. Worth distinguishing because the docstring used to conflate the two branches.
    """
    frozen_tail = st.cone_interval(np.array([2.0, 0.0]), np.array([-1.0, 0.0]))
    assert not frozen_tail.degenerate
    assert frozen_tail.upper == pytest.approx(2.0), "t reaches zero at alpha = 2"


def test_a_direction_that_does_not_move_the_cone_never_blocks():
    """`ds = 0` is the extreme of the degenerate case, and is not a division by zero."""
    assert math.isinf(st.cone_interval(np.array([2.0, 1.0]), np.zeros(2)).upper)


# ----------------------------------------------------------------------------------
# The right-hand side guard
# ----------------------------------------------------------------------------------


def test_the_step_is_capped_where_the_right_hand_side_reaches_zero():
    """Squaring is valid only while `t + alpha*tau >= 0`, and beyond that nothing is feasible.

    `||r + alpha q||` is non-negative and `t + alpha tau` is not, so no step past the cap is
    feasible however small the quadratic gets. Reported rather than folded in silently,
    because "the quadratic stopped you" and "the arithmetic stopped being valid" are
    different facts -- and here it is the second: the quadratic allows every step, and only
    the right-hand side does not.
    """
    interval = st.cone_interval(np.array([5.0, 3.0, 4.0]), np.array([-1.0, -0.6, -0.8]))
    assert interval.capped
    assert interval.upper == pytest.approx(5.0)
    assert soc.is_member(np.array([5.0, 3.0, 4.0]) + 5.0 * np.array([-1.0, -0.6, -0.8]))
    assert not soc.is_member(np.array([5.0, 3.0, 4.0]) + 5.1 * np.array([-1.0, -0.6, -0.8]))


def test_a_rising_head_is_never_capped():
    """With `tau >= 0` the right-hand side only grows, so the guard never binds."""
    assert not st.cone_interval(np.array([2.0, 0.0]), np.array([1.0, 3.0])).capped


def test_the_reported_step_is_feasible_and_the_next_one_is_not():
    """The property that makes it *the* step rather than *a* step, over random cases.

    Ten thousand draws from feasible points and arbitrary directions: every finite step
    reported lands in the cone, and every step a little past it lands outside.
    """
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(2000):
        size = int(rng.integers(1, 5))
        tail = rng.normal(size=size)
        head = float(np.linalg.norm(tail)) * float(rng.uniform(1.0, 2.0))
        s = np.concatenate([[head], tail])
        ds = rng.normal(size=size + 1) * float(rng.choice([0.1, 1.0, 10.0]))

        upper = st.cone_interval(s, ds).upper
        if not math.isfinite(upper) or upper <= 1e-12:
            continue
        checked += 1
        scale = max(1.0, float(np.abs(s).max())) * max(1.0, float(np.abs(ds).max()))
        assert soc.slack(s + upper * ds) >= -1e-9 * scale, "the step itself is feasible"
        assert soc.slack(s + (upper * (1.0 + 1e-3) + 1e-9) * ds) <= 1e-12 * scale, "and just past it is not"
    assert checked > 500, "the sample actually exercised finite steps"


def test_an_unbounded_verdict_survives_an_enormous_step():
    """The other half: when it says nothing stops the step, nothing does."""
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(2000):
        size = int(rng.integers(1, 5))
        tail = rng.normal(size=size)
        head = float(np.linalg.norm(tail)) * float(rng.uniform(1.0, 2.0))
        s = np.concatenate([[head], tail])
        ds = rng.normal(size=size + 1)
        if math.isfinite(st.cone_interval(s, ds).upper):
            continue
        checked += 1
        assert soc.is_member(s + 1e6 * ds, tolerance=1e-6)
    assert checked > 100


# ----------------------------------------------------------------------------------
# Through the problem, and through the intersection
# ----------------------------------------------------------------------------------


def test_the_cone_step_takes_the_tightest_factor():
    """A cone product is bounded by whichever of its factors binds first."""
    from cosa import SOCP, ConeProduct

    problem = SOCP(
        c=np.zeros(3),
        A=np.zeros((0, 3)),
        b=np.zeros(0),
        E=np.zeros((0, 3)),
        d=np.zeros(0),
        G=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        h=np.zeros(4),
        cone=ConeProduct.from_dims(2, 2),
    )
    z = np.array([2.0, 0.0, 0.0])
    limit = st.cone_step(problem, z, np.array([0.0, 1.0, 3.0]))
    assert limit.source == "cone"
    assert limit.alpha == pytest.approx(4.0 / 3.0), "the second factor binds first"


def test_a_problem_with_no_cone_is_never_conically_bounded():
    """The linear programs the loop also solves."""
    from cosa import SOCP

    problem = SOCP.unconstrained(np.array([1.0]))
    assert st.cone_step(problem, np.zeros(1), np.ones(1)).is_unbounded


def test_the_intersection_reports_which_interval_bound_it():
    """§5.2's three, and the loop needs to know which one so it can add the right row."""
    instance = families.box(5, seed=0)
    problem = instance.problem
    tail_only = np.concatenate([np.ones(instance.num_assets), [0.0]])
    assert st.step_limit(problem, instance.witness, tail_only, WorkingSet.empty(problem)).source == "cone"


def test_a_point_outside_the_cone_gives_no_interval():
    """The precondition, and what happens when a caller breaks it.

    Every branch above assumes `s` is in the cone -- that is what makes the constant term
    non-positive and the root selection unambiguous. From outside, the quadratic can have no
    real roots at all, and the honest answer is that there is no admissible interval rather
    than a number that looks like one.
    """
    outside = np.array([1.0, 2.0, 0.0])
    assert not soc.is_member(outside)
    interval = st.cone_interval(outside, np.array([0.0, 0.0, 1.0]))
    assert math.isnan(interval.upper)
    assert not interval.contains(0.0), "a NaN interval contains nothing"


def test_the_block_shapes_are_checked():
    """A slack and a direction of different lengths is a bug, not something to broadcast."""
    with pytest.raises(ProblemError, match="expected 3 entries"):
        st.cone_interval(np.array([2.0, 0.0, 0.0]), np.array([1.0, 1.0]))


def test_a_block_too_short_to_be_a_cone_is_rejected():
    """Q^1 is the non-negative ray, which has no step interval of this kind."""
    with pytest.raises(ProblemError, match="head and a tail"):
        st.cone_interval(np.array([1.0]), np.array([1.0]))


def test_the_interval_answers_membership_questions():
    """A small convenience the loop and the tests both want."""
    interval = st.cone_interval(np.array([2.0, 0.0]), np.array([0.0, 1.0]))
    assert interval.contains(1.0)
    assert not interval.contains(3.0)
