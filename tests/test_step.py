"""The ratio test: how far a step may go, and what stops it.

The linear half of §5.2 (`paper.tex:471`), which is what issue #14's loop needs. The conic
half is eq. (6) and is #18's; this file also pins the *refusal* that stands in for it, so
that the missing interval cannot be silently skipped.
"""

import math

import numpy as np
import pytest

from cosa import SOCP, ProblemError, WorkingSet
from cosa.active_set import updates
from cosa.experiments import portfolio as families
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
# The conic interval is #18's, and its absence is loud
# ----------------------------------------------------------------------------------


def test_a_cone_that_could_bind_is_refused_not_ignored():
    """Issue #14's scope boundary, enforced rather than documented.

    A ratio test that silently omitted the conic bound would produce iterates outside the
    cone, and §14.1's Level 1 invariant would start failing several modules away from the
    cause. So it refuses, and the message names the issue that fixes it.
    """
    instance = families.basic(4, seed=0)
    problem = instance.problem
    with pytest.raises(ProblemError, match="#18"):
        st.step_limit(problem, instance.witness, np.ones(problem.num_variables), WorkingSet.empty(problem))


def test_a_strictly_interior_cone_is_refused_too():
    """The guard is unconditional on the direction moving the block, and has to be.

    An earlier version exempted a strictly interior factor, reasoning that the linear
    interval already bounded the step. It does bound it -- but not by enough to stay inside
    the cone, so a step from an interior point could leave it. §14.1's checker caught that
    on a Phase I solve; this test is what stops it coming back.
    """
    instance = families.basic(4, seed=0)
    problem = instance.problem
    interior = instance.witness.copy()
    interior[-1] += 10.0
    from cosa.geometry import soc

    assert soc.is_interior(problem.cone_slack(interior)), "strictly inside, and still refused"
    with pytest.raises(ProblemError, match="#18"):
        st.step_limit(problem, interior, np.ones(problem.num_variables), WorkingSet.empty(problem))


def test_a_direction_that_does_not_move_the_cone_does_not_refuse():
    """The other safe case: the cone is on its boundary but the step leaves it alone."""
    instance = families.basic(4, seed=0)
    problem = instance.problem
    still = np.zeros(problem.num_variables)
    assert st.step_limit(problem, instance.witness, still, WorkingSet.empty(problem)).is_unbounded


def test_a_problem_with_no_cone_never_refuses(box):
    """The linear programs the Phase I loop actually runs on."""
    assert st.step_limit(box, np.zeros(2), np.array([1.0, 0.0]), WorkingSet.empty(box)).alpha == pytest.approx(1.0)


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
