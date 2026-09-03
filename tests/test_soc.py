"""The SOC predicates: membership, boundary and apex.

The executable half of issue #16, and the reason §9 Phase II (`paper.tex:723`) insists
these routines be "tested independently": M4's step interval and M5's prototype are
debugged *through* them, so a predicate that is quietly wrong turns into a solver bug that
looks like an algorithmic one.

The case list is §16.1 (`paper.tex:1100`) -- points inside the cone, points outside it,
boundary points -- plus the case the plan singles out separately in §8.1: the apex.
"""

import numpy as np
import pytest

from cosa import ConePosition, ConeProduct, ProblemError
from cosa.geometry import soc

# ----------------------------------------------------------------------------------
# The slack and the scale, which every predicate is a sign test on
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("z", "expected"),
    [
        ((5.0, 3.0, 4.0), 0.0),  # 3-4-5: exactly on the boundary
        ((6.0, 3.0, 4.0), 1.0),  # a unit inside
        ((4.0, 3.0, 4.0), -1.0),  # a unit outside
        ((0.0, 0.0), 0.0),  # the apex
        ((-1.0, 0.0), -1.0),  # a negative head is outside by its own size
    ],
)
def test_slack_is_the_head_minus_the_tail_norm(z, expected):
    """The slack t - ||y||: positive inside, zero on the boundary, negative outside."""
    assert soc.slack(np.array(z)) == pytest.approx(expected)


def test_magnitude_is_the_larger_of_the_two_halves():
    """The scale the tolerance is relativized against, zero only at the apex."""
    assert soc.magnitude(np.array([5.0, 3.0, 4.0])) == pytest.approx(5.0)
    assert soc.magnitude(np.array([1.0, 3.0, 4.0])) == pytest.approx(5.0)
    assert soc.magnitude(np.zeros(3)) == 0.0


# ----------------------------------------------------------------------------------
# The three predicates, over §16.1's case list
# ----------------------------------------------------------------------------------

INSIDE = [(1.0, 0.0), (2.0, 1.0, 1.0), (10.0, 3.0, 4.0), (1e-6, 0.0, 0.0)]
OUTSIDE = [(0.0, 1.0), (1.0, 2.0), (4.0, 3.0, 4.0), (-1.0, 0.0), (-5.0, 3.0, 4.0)]
BOUNDARY = [(1.0, 1.0), (5.0, 3.0, 4.0), (2.0, 2.0, 0.0), (np.sqrt(2.0), 1.0, 1.0)]


@pytest.mark.parametrize("z", INSIDE)
def test_points_inside_the_cone(z):
    """Inside: a member, in the interior, and neither boundary nor apex."""
    point = np.array(z)
    assert soc.is_member(point)
    assert soc.is_interior(point)
    assert not soc.is_boundary(point)
    assert not soc.is_apex(point)
    assert soc.position(point) is ConePosition.INTERIOR


@pytest.mark.parametrize("z", OUTSIDE)
def test_points_outside_the_cone(z):
    """Outside: not a member, and none of the three inside cases."""
    point = np.array(z)
    assert not soc.is_member(point)
    assert not soc.is_interior(point)
    assert not soc.is_boundary(point)
    assert not soc.is_apex(point)
    assert soc.position(point) is ConePosition.EXTERIOR


@pytest.mark.parametrize("z", BOUNDARY)
def test_points_on_the_cone(z):
    """On the boundary: a member, on the boundary, not in the interior, not the apex."""
    point = np.array(z)
    assert soc.is_member(point)
    assert soc.is_boundary(point)
    assert not soc.is_interior(point)
    assert not soc.is_apex(point)
    assert soc.position(point) is ConePosition.BOUNDARY


@pytest.mark.parametrize("dim", [2, 3, 7])
def test_the_apex_is_its_own_case(dim):
    """The apex is a member and a boundary point, and is reported as the apex."""
    apex = np.zeros(dim)
    assert soc.is_apex(apex)
    assert soc.is_member(apex)
    assert soc.is_boundary(apex), "the apex satisfies t = ||y|| = 0, so it is on the boundary"
    assert not soc.is_interior(apex)
    assert soc.position(apex) is ConePosition.APEX


def test_the_apex_predicate_is_not_the_boundary_predicate():
    """A boundary point away from the apex is not the apex -- the distinction of §8.1."""
    point = np.array([5.0, 3.0, 4.0])
    assert soc.is_boundary(point)
    assert not soc.is_apex(point)


@pytest.mark.parametrize("side", [1.0, -1.0])
def test_a_point_indistinguishable_from_the_apex_is_the_apex(side):
    """Whichever side rounding put it on: if it is not distinguishable, the apex branch runs.

    The reason the apex predicate is a two-sided test on the magnitude rather than a
    corollary of membership. At `L @ x = 0` the tangent direction `u` does not exist, and
    a point 1e-15 outside the cone has no more of a tangent than one exactly at the apex.
    """
    point = np.array([side * 1e-15, 1e-15])
    assert soc.is_apex(point)
    assert soc.position(point) is ConePosition.APEX


# ----------------------------------------------------------------------------------
# The tolerance, and the scale it is relative to
# ----------------------------------------------------------------------------------


def test_the_tolerance_is_relative_above_unit_scale():
    """The same absolute violation is noise at scale 1e6 and a real violation at scale 1.

    The mixed absolute/relative convention, and the reason it exists: a factor model's
    cone block lives far from unit scale, and a fixed absolute tolerance would call every
    such point infeasible.
    """
    violation = 1e-7
    large = np.array([1e6, 1e6 + violation])
    small = np.array([1.0, 1.0 + violation])
    assert soc.is_member(large), "1e-7 out of 1e6 is below the relative tolerance"
    assert not soc.is_member(small), "1e-7 out of 1 is above it"


def test_an_explicit_scale_overrides_the_derived_one():
    """A caller that knows the problem's scale can say so, and is believed."""
    point = np.array([1.0, 1.0 + 1e-7])
    assert not soc.is_member(point)
    assert soc.is_member(point, scale=1e6)


def test_a_wider_tolerance_admits_a_wider_band():
    """The tolerance is a parameter, not a constant, because #29 will need to vary it."""
    point = np.array([1.0, 1.0 + 1e-5])
    assert not soc.is_boundary(point)
    assert soc.is_boundary(point, tolerance=1e-4)


def test_the_default_tolerance_is_the_documented_one():
    """The default is a fact consumers rely on, so it is asserted rather than assumed."""
    assert soc.TOLERANCE == 1e-9


# ----------------------------------------------------------------------------------
# Malformed input
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("z", [np.zeros(0), np.zeros(1)])
def test_a_block_too_short_to_be_a_cone_is_rejected(z):
    """Q^1 is the non-negative ray, which is a linear inequality, not a cone."""
    with pytest.raises(ProblemError, match="dim >= 2"):
        soc.slack(z)


def test_a_two_dimensional_block_is_rejected():
    """A matrix is not a point in the cone, and must not be silently flattened."""
    with pytest.raises(ProblemError, match="expected a vector"):
        soc.slack(np.zeros((2, 2)))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_non_finite_entry_is_rejected(bad):
    """A NaN slack would make every predicate answer False, which is not an answer."""
    with pytest.raises(ProblemError, match="must be finite"):
        soc.slack(np.array([1.0, bad]))


# ----------------------------------------------------------------------------------
# The product form, which is what the solver actually holds
# ----------------------------------------------------------------------------------


def test_positions_classifies_every_block():
    """One verdict per factor, in factor order -- what the working set consumes."""
    cone = ConeProduct.from_dims(2, 3, 2)
    vector = np.concatenate([[2.0, 1.0], [5.0, 3.0, 4.0], [0.0, 0.0]])
    assert soc.positions(cone, vector) == (
        ConePosition.INTERIOR,
        ConePosition.BOUNDARY,
        ConePosition.APEX,
    )


def test_product_membership_needs_every_block():
    """`vector in K` is a conjunction: one exterior block makes the whole thing false."""
    cone = ConeProduct.from_dims(2, 2)
    assert soc.is_member_of_product(cone, np.array([2.0, 1.0, 2.0, 1.0]))
    assert not soc.is_member_of_product(cone, np.array([2.0, 1.0, 1.0, 2.0]))


def test_the_empty_product_is_the_linear_programming_case():
    """No factors, no positions, and membership holds vacuously."""
    cone = ConeProduct()
    assert soc.positions(cone, np.zeros(0)) == ()
    assert soc.is_member_of_product(cone, np.zeros(0))


def test_a_product_vector_of_the_wrong_length_is_rejected():
    """The block layout is the representation; a length mismatch is a bug, not a reshape."""
    with pytest.raises(ProblemError, match="expected 5 entries"):
        soc.positions(ConeProduct.from_dims(2, 3), np.zeros(4))
