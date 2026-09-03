"""The working set, its three item classes, and the rules that change it.

The executable half of issue #11. The "done when" is two claims, and both are here:

* the set holds all three item classes of §3.2 (`paper.tex:268`) *simultaneously* --
  active linear inequalities, equality constraints, and the active geometry of the cone;
* its state is human-readable in terms of the active portfolio constraints and the SOC
  geometry, which is Success Criterion 3 (`paper.tex:1324`) and therefore an output
  requirement rather than a debugging convenience.

The rules of §7 are tested for what they decide, not for how: the most-violating removal
rule is checked against `SIGN_CONVENTION` rather than against a hard-coded `< 0`, because
a consumer that restates the convention instead of reading it is the failure mode #9
exists to prevent.
"""

import numpy as np
import pytest

from cosa import (
    SIGN_CONVENTION,
    SOCP,
    ConePosition,
    ConeProduct,
    ConeStatus,
    ConstraintNames,
    MeanStdPortfolio,
    ProblemError,
    SecondOrderCone,
    WorkingSet,
)
from cosa.active_set import updates


@pytest.fixture
def portfolio():
    """Three assets, a budget equality and two weight caps -- one cone, six rows of shape."""
    return MeanStdPortfolio(
        mu=np.array([0.10, 0.04, 0.06]),
        Sigma=np.diag([0.04, 0.09, 0.16]),
        lam=2.0,
        A=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        b=np.array([0.5, 0.5]),
        E=np.array([[1.0, 1.0, 1.0]]),
        d=np.array([1.0]),
    )


@pytest.fixture
def problem(portfolio):
    """The portfolio as the general SOCP the working set is built over."""
    return portfolio.to_socp()


@pytest.fixture
def empty(problem):
    """The starting working set: nothing chosen, and the equalities in it regardless."""
    return WorkingSet.empty(problem)


# ----------------------------------------------------------------------------------
# All three item classes at once
# ----------------------------------------------------------------------------------


def test_the_set_holds_the_three_item_classes_simultaneously(empty, problem):
    """§3.2's list, all three present in one object -- issue #11's "done when"."""
    working_set = updates.add_inequality(empty, 0)
    working_set = updates.set_cone_status(working_set, 0, ConeStatus.TANGENT)

    assert working_set.inequalities == (0,)
    assert working_set.equalities == (0,)
    assert working_set.cone_status == (ConeStatus.TANGENT,)
    assert working_set.num_rows == 1 + problem.num_equalities + 1


def test_every_equality_is_always_in_the_set(problem):
    """§3.1 imposes E @ p = 0 unconditionally, so there is no subset to choose."""
    working_set = WorkingSet.empty(problem)
    assert working_set.equalities == tuple(range(problem.num_equalities))
    assert len(working_set.equalities) == problem.num_equalities


def test_no_rule_can_drop_an_equality(empty):
    """There is deliberately no API for it: the rules that exist are §7.1 and §7.2."""
    assert not hasattr(updates, "drop_equality")
    assert "equalities" not in {field for field in vars(empty) if not field.startswith("_")}


def test_the_starting_set_has_nothing_active_but_the_equalities(empty, problem):
    """Where Phase I begins."""
    assert empty.inequalities == ()
    assert empty.active_cones == ()
    assert empty.inactive_inequalities == (0, 1)
    assert empty.num_rows == problem.num_equalities


def test_the_set_carries_the_shape_and_not_the_problem(empty, problem):
    """Shape only, so a warm start can hand it to the next problem in the sequence.

    #30's frontier walk reuses the working set of one lambda for the next: the same
    shape, different data. A set that held a problem instance could not be reused that
    way without lying about which problem it describes.
    """
    assert empty.num_inequalities == problem.num_inequalities
    assert empty.num_equalities == problem.num_equalities
    assert empty.cone == problem.cone
    assert not any(isinstance(value, SOCP) for value in vars(empty).values())


# ----------------------------------------------------------------------------------
# The representation validates itself
# ----------------------------------------------------------------------------------


def test_indices_are_normalized_to_ascending_order():
    """Ascending, so the KKT row order is a function of the set and not of the path.

    Two iterations that arrive at the same active set must assemble the same matrix, or
    the factorization reuse of #27 compares systems that are not the same system.
    """
    working_set = WorkingSet(num_inequalities=4, num_equalities=0, inequalities=(3, 0, 2))
    assert working_set.inequalities == (0, 2, 3)


def test_a_repeated_index_is_rejected():
    """A row active twice means the caller has lost track of the set."""
    with pytest.raises(ProblemError, match="cannot be active twice"):
        WorkingSet(num_inequalities=3, num_equalities=0, inequalities=(1, 1))


@pytest.mark.parametrize("index", [-1, 3, 99])
def test_an_index_outside_the_shape_is_rejected(index):
    """The shape is what the set is over; an index outside it is not a row."""
    with pytest.raises(ProblemError, match=r"index in \[0, 3\)"):
        WorkingSet(num_inequalities=3, num_equalities=0, inequalities=(index,))


def test_a_negative_row_count_is_rejected():
    """A shape with negative rows describes no problem."""
    with pytest.raises(ProblemError, match="non-negative"):
        WorkingSet(num_inequalities=-1, num_equalities=0)


def test_there_is_exactly_one_status_per_cone_factor():
    """The status tuple is indexed by factor, so its length is part of the shape."""
    with pytest.raises(ProblemError, match="one status per cone factor"):
        WorkingSet(
            num_inequalities=0,
            num_equalities=0,
            cone=ConeProduct.from_dims(3, 3),
            cone_status=(ConeStatus.INACTIVE,),
        )


def test_a_status_must_be_a_status():
    """A string that looks like a status is not one -- the enum is the vocabulary."""
    with pytest.raises(ProblemError, match="not a ConeStatus"):
        WorkingSet(
            num_inequalities=0,
            num_equalities=0,
            cone=ConeProduct.from_dims(3),
            cone_status=("tangent",),
        )


def test_the_set_is_frozen(empty):
    """Every rule returns a new set, so an iteration can keep the previous one for free."""
    with pytest.raises(AttributeError):
        empty.inequalities = (0,)


def test_equal_sets_are_equal_and_hashable(problem):
    """Structural comparison, because "did this iteration change the set?" is the question.

    Hashability is what lets #29 keep the sequence of visited sets in a container and
    notice a cycle.
    """
    first = updates.add_inequality(WorkingSet.empty(problem), 1)
    second = updates.add_inequality(WorkingSet.empty(problem), 1)
    assert first == second
    assert hash(first) == hash(second)
    assert len({first, second}) == 1
    assert first != updates.add_inequality(WorkingSet.empty(problem), 0)


# ----------------------------------------------------------------------------------
# How many rows each item class contributes
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [(ConeStatus.INACTIVE, 0), (ConeStatus.TANGENT, 1), (ConeStatus.APEX, 5)],
)
def test_a_cone_contributes_rows_according_to_its_status(status, expected):
    """One row for a tangent hyperplane, the whole block at the apex, none when inactive.

    The tangent case costing a single row is what makes the tangent representation
    attractive: an active cone is no more expensive than an active linear constraint.
    The apex has no hyperplane to use, per §8.1, so the block is held exactly.
    """
    assert status.num_rows(SecondOrderCone(dim=5)) == expected


def test_the_row_count_adds_up_over_all_three_classes(problem):
    """num_rows is the size of the W block of §13's KKT system."""
    working_set = WorkingSet.empty(problem)
    working_set = updates.add_inequality(working_set, 0)
    working_set = updates.add_inequality(working_set, 1)
    working_set = updates.set_cone_status(working_set, 0, ConeStatus.APEX)
    conic = problem.cone.cones[0].dim
    assert working_set.num_rows == 2 + problem.num_equalities + conic


def test_only_a_non_inactive_status_counts_as_active():
    """is_active is the one place "in the working set" is decided for a cone."""
    assert not ConeStatus.INACTIVE.is_active
    assert ConeStatus.TANGENT.is_active
    assert ConeStatus.APEX.is_active


# ----------------------------------------------------------------------------------
# §7.1: adding an inequality that has reached its boundary
# ----------------------------------------------------------------------------------


def test_a_constraint_at_its_boundary_is_a_candidate(portfolio, problem, empty):
    """a_i.T @ x = b_i, which is §7.1's activation condition verbatim."""
    x = np.array([0.5, 0.2, 0.3])  # the first cap binds, the second does not
    z = portfolio.socp_point(x)
    assert updates.activation_candidates(problem, z, empty) == (0,)


def test_a_constraint_the_step_overshot_is_also_a_candidate(portfolio, problem, empty):
    """An overshoot has to be noticed, or nothing in the set pulls the iterate back."""
    z = portfolio.socp_point(np.array([0.7, 0.2, 0.1]))
    assert 0 in updates.activation_candidates(problem, z, empty)
    assert updates.inequality_slack(problem, z)[0] < 0.0


def test_an_already_active_constraint_is_not_a_candidate(portfolio, problem, empty):
    """Candidates are drawn from the inactive rows, so a rule cannot add a row twice."""
    z = portfolio.socp_point(np.array([0.5, 0.5, 0.0]))
    assert updates.activation_candidates(problem, z, empty) == (0, 1)
    working_set = updates.add_inequality(empty, 0)
    assert updates.activation_candidates(problem, z, working_set) == (1,)


def test_a_constraint_far_from_its_boundary_is_not_a_candidate(portfolio, problem, empty):
    """The whole point of the tolerance: only rows that have been reached are added."""
    z = portfolio.socp_point(np.array([0.1, 0.1, 0.8]))
    assert updates.activation_candidates(problem, z, empty) == ()


@pytest.mark.parametrize(("cap", "reached"), [(1.0, False), (1e6, True)])
def test_the_activation_tolerance_is_relative_to_the_right_hand_side(cap, reached):
    """A cap of 1e6 and a cap of 1 do not deserve the same absolute threshold.

    The same absolute shortfall of 1e-4 is a real distance from a cap of 1 and pure
    rounding on a cap of 1e6, and an absolute tolerance would have to be wrong about one
    of them.
    """
    problem = SOCP.unconstrained(np.array([1.0])).add_inequalities([[1.0]], [cap])
    working_set = WorkingSet.empty(problem)
    candidates = updates.activation_candidates(problem, np.array([cap - 1e-4]), working_set, tolerance=1e-8)
    assert candidates == ((0,) if reached else ())


def test_adding_a_row_puts_it_in_the_set(empty):
    """§7.1's effect, and the original set is untouched."""
    working_set = updates.add_inequality(empty, 1)
    assert working_set.is_active(1)
    assert not working_set.is_active(0)
    assert not empty.is_active(1), "the set is frozen, so the original is unchanged"


def test_adding_a_row_twice_is_an_error(empty):
    """Not a no-op: a rule adding a row it already holds has lost track of the set."""
    working_set = updates.add_inequality(empty, 1)
    with pytest.raises(ProblemError, match="already active"):
        updates.add_inequality(working_set, 1)


def test_querying_a_row_outside_the_shape_is_an_error(empty):
    """Out of range is a bug in the caller, reported where it happens."""
    with pytest.raises(ProblemError, match=r"index in \[0, 2\)"):
        empty.is_active(7)


# ----------------------------------------------------------------------------------
# §7.2: dropping the most strongly violating multiplier
# ----------------------------------------------------------------------------------


def test_the_removal_candidate_is_the_most_strongly_violating_row(problem):
    """The classical rule: among wrong-signed multipliers, the worst one goes."""
    working_set = updates.add_inequality(updates.add_inequality(WorkingSet.empty(problem), 0), 1)
    assert updates.removal_candidate(working_set, np.array([-0.1, -0.9])) == 1
    assert updates.removal_candidate(working_set, np.array([-0.9, -0.1])) == 0


def test_a_correctly_signed_multiplier_is_no_candidate(problem):
    """All multipliers right-signed is the dual-feasibility half of optimality."""
    working_set = updates.add_inequality(updates.add_inequality(WorkingSet.empty(problem), 0), 1)
    assert updates.removal_candidate(working_set, np.array([0.5, 1.5])) is None


def test_a_rounding_level_violation_is_below_the_tolerance(problem):
    """Subject to numerical tolerances, as §7.2 puts it: noise is no reason to re-decide."""
    working_set = updates.add_inequality(WorkingSet.empty(problem), 0)
    assert updates.removal_candidate(working_set, np.array([-1e-14, 0.0])) is None
    assert updates.removal_candidate(working_set, np.array([-1e-14, 0.0]), tolerance=1e-16) == 0


def test_inactive_multipliers_are_ignored(problem):
    """Complementarity makes them zero; a stale nonzero entry must not drive a removal."""
    working_set = updates.add_inequality(WorkingSet.empty(problem), 0)
    assert updates.removal_candidate(working_set, np.array([0.5, -9.0])) is None


def test_a_tie_goes_to_the_lowest_index(problem):
    """Arbitrary but deterministic. An arbitrary *and* unstable choice is how one cycles."""
    working_set = updates.add_inequality(updates.add_inequality(WorkingSet.empty(problem), 0), 1)
    assert updates.removal_candidate(working_set, np.array([-0.5, -0.5])) == 0


def test_the_required_sign_is_read_from_the_convention(problem):
    """The rule consumes SIGN_CONVENTION rather than restating `y < 0`.

    Under this convention an inequality multiplier must be non-negative. The test states
    the dependency the way the code does, so that flipping the convention flips both
    together rather than leaving them silently disagreeing.
    """
    working_set = updates.add_inequality(WorkingSet.empty(problem), 0)
    wrong_signed = np.array([-1.0 * SIGN_CONVENTION.inequality, 0.0])
    right_signed = -wrong_signed
    assert updates.removal_candidate(working_set, wrong_signed) == 0
    assert updates.removal_candidate(working_set, right_signed) is None


def test_dropping_a_row_takes_it_out(problem):
    """§7.2's effect."""
    working_set = updates.add_inequality(updates.add_inequality(WorkingSet.empty(problem), 0), 1)
    assert updates.drop_inequality(working_set, 0).inequalities == (1,)


def test_dropping_an_inactive_row_is_an_error(empty):
    """Same reason adding an active one is: the caller's model of the set is wrong."""
    with pytest.raises(ProblemError, match="not active"):
        updates.drop_inequality(empty, 0)


def test_a_multiplier_vector_of_the_wrong_length_is_rejected(empty):
    """The vector is indexed by row, so its length is part of the contract."""
    with pytest.raises(ProblemError, match="expected 2 entries"):
        updates.removal_candidate(empty, np.zeros(5))


# ----------------------------------------------------------------------------------
# §7.3: the cone's geometry, and the deactivation that is deliberately absent
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        (ConePosition.INTERIOR, ConeStatus.INACTIVE),
        (ConePosition.BOUNDARY, ConeStatus.TANGENT),
        (ConePosition.APEX, ConeStatus.APEX),
        (ConePosition.EXTERIOR, ConeStatus.TANGENT),
    ],
)
def test_the_geometry_argues_for_a_status(position, expected):
    """The one place an observed position becomes a working-set decision.

    The exterior case is the interesting one: the block is infeasible, and the cone is
    exactly the constraint the next direction has to respect, so treating it as inactive
    would compute a step that ignores a violated constraint.
    """
    assert updates.cone_status_for(position) is expected


def test_a_cone_at_its_boundary_is_activated(portfolio, problem, empty):
    """§7.3: t - ||L @ x|| small enough, applied through the predicates, not restated."""
    z = portfolio.socp_point(np.array([0.3, 0.3, 0.4]))
    working_set = updates.activate_cones(problem, z, empty)
    assert working_set.status(0) is ConeStatus.TANGENT
    assert working_set.active_cones == (0,)


def test_a_cone_the_iterate_is_inside_stays_inactive(portfolio, problem, empty):
    """A strictly interior slack constrains nothing locally."""
    z = portfolio.socp_point(np.array([0.3, 0.3, 0.4])) + np.array([0.0, 0.0, 0.0, 0.5])
    assert updates.activate_cones(problem, z, empty).status(0) is ConeStatus.INACTIVE


def test_activation_recognizes_the_apex(problem, empty):
    """The apex gets its own status, because §8.1 gives it its own geometry."""
    z = np.zeros(problem.num_variables)
    assert updates.activate_cones(problem, z, empty).status(0) is ConeStatus.APEX


def test_activation_corrects_the_geometry_of_an_active_cone(portfolio, problem, empty):
    """Apex to tangent is not a deactivation: the cone stays active, the face changes."""
    at_apex = updates.activate_cones(problem, np.zeros(problem.num_variables), empty)
    assert at_apex.status(0) is ConeStatus.APEX
    moved = updates.activate_cones(problem, portfolio.socp_point(np.array([0.3, 0.3, 0.4])), at_apex)
    assert moved.status(0) is ConeStatus.TANGENT


def test_activation_never_turns_a_cone_off(portfolio, problem, empty):
    """§7.4 refuses to decide deactivation on the geometry alone, so neither does this.

    The rule is monotone in activity by design. A cone whose slack has grown stays
    active until the conic multiplier and the normal-cone conditions say otherwise, which
    is #23's question -- the plan calls it "a key research component".
    """
    active = updates.activate_cones(problem, portfolio.socp_point(np.array([0.3, 0.3, 0.4])), empty)
    assert active.status(0) is ConeStatus.TANGENT
    interior = portfolio.socp_point(np.array([0.3, 0.3, 0.4])) + np.array([0.0, 0.0, 0.0, 0.5])
    assert updates.activate_cones(problem, interior, active).status(0) is ConeStatus.TANGENT


def test_the_deactivation_primitive_is_available_for_issue_23(problem, empty):
    """What #23 will drive: unconditional, and no rule of its own attached yet."""
    active = updates.set_cone_status(empty, 0, ConeStatus.TANGENT)
    assert updates.set_cone_status(active, 0, ConeStatus.INACTIVE).status(0) is ConeStatus.INACTIVE


def test_setting_an_unchanged_status_returns_the_same_set(empty):
    """No churn on a no-op, so an unchanged iteration allocates nothing."""
    assert updates.set_cone_status(empty, 0, ConeStatus.INACTIVE) is empty


def test_a_cone_index_outside_the_product_is_an_error(empty):
    """One status per factor means one valid index range."""
    with pytest.raises(ProblemError, match=r"index in \[0, 1\)"):
        updates.set_cone_status(empty, 1, ConeStatus.TANGENT)


def test_a_linear_program_has_no_cone_geometry_to_track():
    """The empty product is a legitimate shape, and the polyhedral half works alone."""
    problem = SOCP.unconstrained(np.array([1.0, 1.0])).add_inequalities([[1.0, 1.0]], [1.0])
    working_set = WorkingSet.empty(problem)
    assert working_set.num_cones == 0
    assert updates.activate_cones(problem, np.zeros(2), working_set) == working_set


# ----------------------------------------------------------------------------------
# Success Criterion 3: the state, in the user's terms
# ----------------------------------------------------------------------------------


@pytest.fixture
def names():
    """Names for the portfolio's rows, as a caller who knows the model would supply them."""
    return ConstraintNames(
        inequalities=("cap on AAPL", "cap on MSFT"),
        equalities=("budget",),
        cones=("risk",),
    )


def test_the_description_names_the_active_portfolio_constraints(portfolio, problem, names):
    """Success Criterion 3: read in terms of the constraints, not of row indices."""
    working_set = updates.add_inequality(WorkingSet.empty(problem), 0)
    working_set = updates.activate_cones(problem, portfolio.socp_point(np.array([0.5, 0.1, 0.4])), working_set)
    description = working_set.describe(names)

    assert "cap on AAPL" in description
    assert "cap on MSFT" in description
    assert "budget" in description
    assert "risk" in description


def test_the_description_covers_all_three_item_classes(problem, names):
    """Inequalities active and inactive, the equalities, and the cone geometry."""
    description = WorkingSet.empty(problem).describe(names)
    for heading in ("active inequalities", "inactive inequalities", "equalities", "cone geometry"):
        assert heading in description


def test_the_description_says_which_geometry_is_active(problem, names):
    """Not merely that the cone is active: which face of it, and at what cost in rows."""
    tangent = updates.set_cone_status(WorkingSet.empty(problem), 0, ConeStatus.TANGENT)
    apex = updates.set_cone_status(WorkingSet.empty(problem), 0, ConeStatus.APEX)
    assert "tangent hyperplane" in tangent.describe(names)
    assert "at the apex" in apex.describe(names)
    assert "strictly inside" in WorkingSet.empty(problem).describe(names)


def test_unnamed_rows_fall_back_to_their_index(problem):
    """Naming is optional and partial: name the two that matter, leave the rest.

    A portfolio with four hundred box bounds should not need four hundred strings before
    its working set can be read.
    """
    description = WorkingSet.empty(problem).describe(ConstraintNames(inequalities=("cap on AAPL",)))
    assert "cap on AAPL (#0)" in description
    assert "inequality #1" in description


def test_the_description_is_deterministic(problem):
    """Rows in index order, so two iterates' descriptions can be diffed."""
    shape = {"num_inequalities": 4, "num_equalities": 1, "cone": problem.cone, "cone_status": (ConeStatus.TANGENT,)}
    first = WorkingSet(**shape, inequalities=(2, 0))
    second = WorkingSet(**shape, inequalities=(0, 2))
    assert first.describe() == second.describe()


def test_str_is_the_description(problem):
    """So a log line or a traceback carries the state without anyone arranging it."""
    working_set = WorkingSet.empty(problem)
    assert str(working_set) == working_set.describe()


def test_a_linear_program_says_so_in_its_description():
    """The empty cone product is reported as what it means, not as an empty section."""
    problem = SOCP.unconstrained(np.array([1.0])).add_inequalities([[1.0]], [1.0])
    assert "linear program" in WorkingSet.empty(problem).describe()
