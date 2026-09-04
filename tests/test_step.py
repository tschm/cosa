"""The ratio test: how far a step may go, and what stops it.

§5.2's three intervals (`paper.tex:471`) -- linear, conic, explicit -- and their
intersection. The linear half arrived with issue #14's loop; the conic half is #18's and is
tested in `test_cone_step.py`, which is where eq. (6)'s correction is pinned down. This
file covers the ratio test, the intersection, and the report.
"""

import math

import numpy as np
import pytest

from cosa import SOCP, ProblemError, WorkingSet
from cosa.active_set import updates
from cosa.experiments import portfolio as families
from cosa.geometry import soc
from cosa.geometry import step as st


@pytest.fixture
def box():
    """A unit box in two variables, as four inequality rows: x <= 1, -x <= 0."""
    return SOCP.unconstrained(np.array([-1.0, -1.0])).add_inequalities(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], [1.0, 1.0, 0.0, 0.0]
    )


# ----------------------------------------------------------------------------------
# §5.2's ratio test
# ----------------------------------------------------------------------------------


def test_the_step_stops_at_the_first_row_it_reaches(box):
    """`alpha <= (b_i - a_i.T @ x) / (a_i.T @ p)` over the rows being approached."""
    limit = st.linear_step(box, np.array([0.0, 0.0]), np.array([1.0, 0.5]), WorkingSet.empty(box))
    assert limit.alpha == pytest.approx(1.0)
    assert limit.blocking == 0
    assert limit.source == "linear"


def test_the_blocking_row_is_the_tightest_ratio_not_the_first(box):
    """Which row blocks is arithmetic, not order."""
    limit = st.linear_step(box, np.array([0.0, 0.0]), np.array([0.5, 1.0]), WorkingSet.empty(box))
    assert limit.blocking == 1
    assert limit.alpha == pytest.approx(1.0)


def test_a_direction_moving_away_from_every_row_is_unbounded():
    """No row is approached, so nothing stops the step -- which the loop reads as unbounded.

    On a half-space rather than the box: a box is bounded in every direction by
    construction, so there is no direction to test this with.
    """
    problem = SOCP.unconstrained(np.array([-1.0, -1.0])).add_inequalities([[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0])
    limit = st.linear_step(problem, np.array([0.0, 0.0]), np.array([-1.0, -1.0]), WorkingSet.empty(problem))
    assert limit.is_unbounded
    assert limit.blocking is None


def test_active_rows_are_not_candidates(box):
    """They satisfy `a_i.T @ p = 0` by construction, so they cannot block.

    Excluded rather than relied upon to produce a huge ratio: their residual is
    rounding-level, and dividing by rounding is how a spurious step limit appears.
    """
    working_set = updates.add_inequality(WorkingSet.empty(box), 0)
    limit = st.linear_step(box, np.array([1.0, 0.0]), np.array([1.0, 1.0]), working_set)
    assert limit.blocking == 1, "row 0 is active, so row 1 blocks instead"


def test_a_row_barely_moved_towards_does_not_block(box):
    """The threshold is rounding-level, and a row orthogonal to the direction is orthogonal."""
    limit = st.linear_step(box, np.array([0.0, 0.0]), np.array([1e-18, 1.0]), WorkingSet.empty(box))
    assert limit.blocking == 1


def test_a_negative_ratio_is_clamped_to_zero(box):
    """An iterate a hair outside a row gives a zero step that adds it, not one that goes further.

    Which is what a finite-precision step can produce, and the honest response to it: stop,
    add the row, and let the working set pull the next direction back inside.
    """
    limit = st.linear_step(box, np.array([1.0 + 1e-14, 0.0]), np.array([1.0, 0.0]), WorkingSet.empty(box))
    assert limit.alpha == 0.0
    assert limit.blocking == 0


def test_a_problem_with_no_inequalities_never_blocks():
    """The degenerate case, answered rather than special-cased at the call site."""
    problem = SOCP.unconstrained(np.array([1.0]))
    assert st.linear_step(problem, np.zeros(1), np.ones(1), WorkingSet.empty(problem)).is_unbounded


# ----------------------------------------------------------------------------------
# §5.2's intersection
# ----------------------------------------------------------------------------------


def test_an_explicit_bound_can_be_the_tightest(box):
    """The third of §5.2's three intervals."""
    limit = st.step_limit(box, np.zeros(2), np.array([1.0, 0.0]), WorkingSet.empty(box), max_step=0.25)
    assert limit.alpha == pytest.approx(0.25)
    assert limit.source == "bound"
    assert limit.blocking is None


def test_a_loose_bound_leaves_the_linear_row_in_charge(box):
    """The intersection keeps the tighter, and says which one it was."""
    limit = st.step_limit(box, np.zeros(2), np.array([1.0, 0.0]), WorkingSet.empty(box), max_step=10.0)
    assert limit.alpha == pytest.approx(1.0)
    assert limit.source == "linear"


def test_the_tighter_of_two_limits_wins():
    """The combinator the intersection is built from, on its own."""
    linear = st.StepLimit(alpha=2.0, blocking=3, source="linear")
    bound = st.StepLimit(alpha=0.5, source="bound")
    assert linear.tighter_of(bound) is bound
    assert bound.tighter_of(linear) is bound


def test_a_non_positive_explicit_bound_is_rejected(box):
    """A step bound of zero is not a bound, it is a refusal to move."""
    with pytest.raises(ProblemError, match="step bound is positive"):
        st.step_limit(box, np.zeros(2), np.ones(2), WorkingSet.empty(box), max_step=0.0)


# ----------------------------------------------------------------------------------
# The conic interval, now that #18 has supplied it
# ----------------------------------------------------------------------------------


def test_the_cone_bounds_the_step_on_a_portfolio():
    """What Wave 4 refused to answer, answered: the cone is one of the three intervals.

    The direction moves `L @ x` while leaving `t` alone, so the tail grows and the head does
    not and the cone is what stops the step. `np.ones` would not do: it raises `t` faster
    than the tail, which is a direction into the cone's interior and is never blocked.
    """
    instance = families.basic(4, seed=0)
    problem = instance.problem
    direction = np.concatenate([np.ones(instance.num_assets), [0.0]])
    limit = st.step_limit(problem, instance.witness, direction, WorkingSet.empty(problem))
    assert limit.source == "cone"
    assert math.isfinite(limit.alpha)
    assert limit.alpha == pytest.approx(0.0), "the witness is exactly on the boundary"


def test_a_step_from_a_strictly_interior_cone_is_still_bounded():
    """The regression test for a bug the invariant checker caught in Wave 4.

    That version exempted a strictly interior factor from the guard, reasoning that the
    linear interval already bounded the step. It does bound it -- but not by enough to stay
    inside the cone. Now the interval is computed rather than assumed away, and a step from
    an interior point lands inside the cone.
    """
    instance = families.basic(4, seed=0)
    problem = instance.problem
    interior = instance.witness.copy()
    interior[-1] += 10.0
    direction = np.concatenate([np.full(instance.num_assets, 100.0), [0.0]])
    limit = st.step_limit(problem, interior, direction, WorkingSet.empty(problem))

    assert soc.is_interior(problem.cone_slack(interior))
    assert math.isfinite(limit.alpha)
    assert soc.is_member(problem.cone_slack(interior + limit.alpha * direction), tolerance=1e-9)


def test_the_tightest_of_the_three_intervals_wins():
    """§5.2's intersection, with all three present."""
    instance = families.box(5, seed=0)
    problem = instance.problem
    direction = np.ones(problem.num_variables)
    unbounded = st.step_limit(problem, instance.witness, direction, WorkingSet.empty(problem))
    capped = st.step_limit(
        problem, instance.witness, direction, WorkingSet.empty(problem), max_step=unbounded.alpha / 2
    )
    assert capped.source == "bound"
    assert capped.alpha == pytest.approx(unbounded.alpha / 2)


# ----------------------------------------------------------------------------------
# The report
# ----------------------------------------------------------------------------------


def test_the_limit_reports_its_cause(box):
    """A log line needs to know whether a row blocked, and which."""
    rendered = str(st.linear_step(box, np.zeros(2), np.array([1.0, 0.0]), WorkingSet.empty(box)))
    assert "alpha=1" in rendered
    assert "linear" in rendered
    assert "row 0" in rendered


def test_an_unbounded_limit_says_so():
    """And an unbounded step names no row, because none stopped it."""
    limit = st.StepLimit(alpha=math.inf)
    assert limit.is_unbounded
    assert "unbounded" in str(limit)
