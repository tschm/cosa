"""§17.2's four remedies and §8.2's hysteresis: making the loop provably stop.

Issue #29. The four remedies the paper names are tested where each of them lives:

* the **anti-cycling rule** and **lexicographic selection** are one rule,
  `anticycling.lexicographic_candidate`, and the `Guard` that arms it;
* **multiplier tolerances** are `updates.MULTIPLIER_TOLERANCE`, in place since #11, tested
  here for the property that makes it a remedy rather than a constant;
* the **merit-function safeguard** is `Guard.accepts`.

§8.2's hysteresis is the fifth item and the one with teeth, so it gets a regression test
that exhibits the oscillation rather than asserting its absence: with a single threshold the
cone's status flips on every iterate, and with the band it does not flip at all.
"""

import numpy as np
import pytest

from cosa import SOCP, ConeStatus, Multipliers, WorkingSet
from cosa.active_set import updates
from cosa.experiments import portfolio as families
from cosa.experiments.randomized import random_instance
from cosa.geometry import tangent
from cosa.solver import anticycling
from cosa.solver import cosa as solver


@pytest.fixture
def box():
    """Four bounds on two variables: enough active rows to have a choice about dropping."""
    return SOCP.unconstrained(np.array([-1.0, -1.0])).add_inequalities(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], [1.0, 1.0, 0.0, 0.0]
    )


# ----------------------------------------------------------------------------------
# Bland's rule
# ----------------------------------------------------------------------------------


def test_bland_takes_the_lowest_index_where_the_fast_rule_takes_the_worst(box):
    """The entire content of the remedy, in one comparison.

    Row 1 violates by more, row 0 violates first. §7.2's rule optimizes and can rotate at a
    degenerate vertex; Bland's does not optimize and therefore cannot.
    """
    working_set = updates.add_inequality(updates.add_inequality(WorkingSet.empty(box), 0), 1)
    y = np.array([-1.0, -5.0, 0.0, 0.0])
    assert updates.removal_candidate(working_set, y) == 1
    assert anticycling.lexicographic_candidate(working_set, y, tolerance=0.0) == 0


def test_bland_names_nothing_when_every_multiplier_has_its_sign(box):
    """The rule is a tie-break among violators, not a reason to drop something."""
    working_set = updates.add_inequality(WorkingSet.empty(box), 0)
    assert anticycling.lexicographic_candidate(working_set, np.ones(4), tolerance=0.0) is None


def test_an_inactive_rows_multiplier_is_not_consulted(box):
    """Only active rows are candidates; an inactive one's `y` is zero by complementarity."""
    working_set = updates.add_inequality(WorkingSet.empty(box), 2)
    y = np.array([-9.0, 0.0, 0.0, 0.0])
    assert anticycling.lexicographic_candidate(working_set, y, tolerance=0.0) is None


def test_a_violation_within_tolerance_is_not_a_violation(box):
    """§17.2's second remedy, which is why the tolerance is a remedy and not a constant.

    Dropping on rounding is itself a way to cycle: the row comes straight back, its
    multiplier is rounding-level again, and it goes straight out.
    """
    working_set = updates.add_inequality(WorkingSet.empty(box), 0)
    y = np.array([-1e-12, 0.0, 0.0, 0.0])
    assert anticycling.lexicographic_candidate(working_set, y, tolerance=1e-9) is None
    assert anticycling.lexicographic_candidate(working_set, y, tolerance=0.0) == 0


# ----------------------------------------------------------------------------------
# The guard: when the switch happens
# ----------------------------------------------------------------------------------


def test_a_working_set_may_recur_a_few_times_before_the_rule_switches(box):
    """Recurrence is ordinary; recurrence without progress is not.

    The loop drops a row, steps, blocks on it again and is back where it was having genuinely
    moved. Arming on the first repeat would trade speed away for nothing.
    """
    guard = anticycling.Guard()
    here = updates.add_inequality(WorkingSet.empty(box), 0)
    away = updates.add_inequality(WorkingSet.empty(box), 1)
    for visit in range(1, anticycling.REVISITS + 1):
        assert guard.saw(here) == visit
        assert not guard.armed
        guard.saw(away)
    guard.saw(here)
    assert guard.armed


def test_staying_on_one_working_set_is_iterating_not_returning(box):
    """The distinction between cycling and slowness, and the reason it is worth having.

    A solve that spends nine hundred iterations refining a point under a single working set
    is converging slowly. Counting that as nine hundred revisits would arm an anti-cycling
    rule against a problem it cannot help with, and report a cycle that is not there -- which
    is exactly what the counter did before this, and it hid the apex oscillation that was.
    """
    guard = anticycling.Guard()
    working_set = updates.add_inequality(WorkingSet.empty(box), 0)
    for _ in range(50):
        assert guard.saw(working_set) == 1
    assert not guard.armed


def test_two_different_working_sets_are_counted_apart(box):
    """The key is the set, not the point: a zero-length step is how a row gets added."""
    guard = anticycling.Guard()
    first = updates.add_inequality(WorkingSet.empty(box), 0)
    second = updates.add_inequality(WorkingSet.empty(box), 1)
    for _ in range(anticycling.REVISITS + 1):
        guard.saw(first)
        guard.saw(second)
    assert len(guard.visits) == 2
    assert guard.visits[first.inequalities, first.cone_status] == anticycling.REVISITS + 1


def test_the_cone_status_is_part_of_a_working_sets_identity(box):
    """Two sets with the same rows and different cone geometry are different sets."""
    instance = families.basic(4, seed=0)
    guard = anticycling.Guard()
    empty = WorkingSet.empty(instance.problem)
    guard.saw(empty)
    guard.saw(updates.set_cone_status(empty, 0, ConeStatus.TANGENT))
    assert len(guard.visits) == 2


def test_arming_is_permanent(box):
    """Disarming on the first sign of progress is how a solver cycles a second time."""
    guard = anticycling.Guard()
    here = updates.add_inequality(WorkingSet.empty(box), 0)
    away = updates.add_inequality(WorkingSet.empty(box), 1)
    for _ in range(anticycling.REVISITS + 1):
        guard.saw(here)
        guard.saw(away)
    guard.saw(updates.add_inequality(WorkingSet.empty(box), 2))
    assert guard.armed


def test_the_guard_switches_which_row_it_names(box):
    """The one place the switch happens, so no caller has to know there is one."""
    guard = anticycling.Guard()
    working_set = updates.add_inequality(updates.add_inequality(WorkingSet.empty(box), 0), 1)
    away = updates.add_inequality(WorkingSet.empty(box), 2)
    y = np.array([-1.0, -5.0, 0.0, 0.0])
    assert guard.candidate(working_set, y, tolerance=0.0) == 1
    for _ in range(anticycling.REVISITS + 1):
        guard.saw(working_set)
        guard.saw(away)
    assert guard.candidate(working_set, y, tolerance=0.0) == 0


def test_the_guard_reports_what_it_has_seen(box):
    """A log line needs the distinct sets, the worst revisit count and the rule in force."""
    guard = anticycling.Guard()
    guard.saw(updates.add_inequality(WorkingSet.empty(box), 0))
    assert "most-violating" in str(guard)
    assert "1 working set(s)" in str(guard)


# ----------------------------------------------------------------------------------
# The merit safeguard
# ----------------------------------------------------------------------------------


def test_the_first_iterate_is_always_accepted():
    """There is nothing to compare it against, and refusing it would refuse every solve."""
    assert anticycling.Guard().accepts(1e9)


def test_an_improving_iterate_is_accepted():
    """The ordinary case, and the one the descent direction promises."""
    guard = anticycling.Guard()
    guard.accepted(1.0)
    assert guard.accepts(0.5)


def test_a_worsening_iterate_is_refused():
    """A net increase means the arithmetic and the geometry have disagreed.

    The direction was a descent direction and the step was chosen along it, so there is no
    honest way for the objective to have risen.
    """
    guard = anticycling.Guard()
    guard.accepted(1.0)
    assert not guard.accepts(2.0)


def test_a_rounding_level_increase_is_tolerated():
    """The retraction genuinely raises the objective before the direction pays for it."""
    guard = anticycling.Guard()
    guard.accepted(1.0)
    assert guard.accepts(1.0 + anticycling.MERIT_SLACK / 2)


def test_asking_does_not_record():
    """A rejected candidate must not poison the record it was rejected against."""
    guard = anticycling.Guard()
    guard.accepted(1.0)
    guard.accepts(5.0)
    assert guard.objective == 1.0


def test_the_record_keeps_the_best_not_the_last():
    """A loop that accepted a slightly worse iterate must still be held to its best."""
    guard = anticycling.Guard()
    guard.accepted(1.0)
    guard.accepted(2.0)
    assert guard.objective == 1.0


def test_the_merit_function_is_the_objective():
    """The merit function is the objective, because there is nothing else to trade.

    A penalty merit function exists to weigh feasibility against optimality, and this loop
    never leaves the feasible set.
    """
    assert anticycling.objective_of(np.array([1.0, -2.0]), np.array([3.0, 4.0])) == pytest.approx(-5.0)


# ----------------------------------------------------------------------------------
# §8.2's hysteresis, exhibited rather than asserted
# ----------------------------------------------------------------------------------


def _wobbling_slacks():
    """Slacks alternating either side of `eps_on`, which is the oscillation §8.2 describes."""
    below, above = updates.ACTIVATION_TOLERANCE / 2, updates.ACTIVATION_TOLERANCE * 1.5
    return [below, above, below, above, below]


def _status_changes(*, hysteresis):
    """Run the wobble through §7.3 and §7.4 and count how often the cone's status flips.

    The multiplier supplied is a genuine active normal every time, so nothing but the
    geometry can release the factor -- which is exactly the situation §8.2 is about.
    """
    instance = families.basic(4, seed=0)
    problem = instance.problem
    working_set = updates.set_cone_status(WorkingSet.empty(problem), 0, ConeStatus.TANGENT)
    changes = 0
    for gap in _wobbling_slacks():
        point = instance.witness.copy()
        point[-1] += gap
        unit = tangent.unit_tail(problem.cone_slack(point))
        found = Multipliers(
            y=np.zeros(problem.num_inequalities),
            nu=np.zeros(problem.num_equalities),
            w=np.concatenate([[1.0], -unit]),
        )
        before = working_set.status(0)
        working_set, _ = updates.deactivate_cones(problem, working_set, found, z=point, hysteresis=hysteresis)
        working_set = updates.activate_cones(problem, point, working_set)
        changes += working_set.status(0) is not before
    return changes


def test_a_single_threshold_makes_the_cone_status_oscillate():
    """The regression instance #29 asks for: the cycle, exhibited.

    With `eps_off == eps_on` an iterate a hair outside the threshold is called interior and
    the factor is released; the next iterate is a hair inside and §7.3 puts it straight back.
    Nothing about the problem changed between them.
    """
    assert _status_changes(hysteresis=updates.ACTIVATION_TOLERANCE) >= 4


def test_the_band_stops_it_dead():
    """§8.2's `eps_on < eps_off`, and the same wobble producing no change at all."""
    assert _status_changes(hysteresis=updates.DEACTIVATION_TOLERANCE) == 0


def test_the_band_still_releases_a_demonstrably_interior_factor():
    """Hysteresis must not become a refusal to ever let go.

    A factor well outside `eps_off` is released, because complementarity says a strictly
    interior slack forces `w = 0` and there is no active normal there to hold.
    """
    instance = families.basic(4, seed=0)
    problem = instance.problem
    interior = instance.witness.copy()
    interior[-1] += 1.0
    working_set = updates.set_cone_status(WorkingSet.empty(problem), 0, ConeStatus.TANGENT)
    unit = tangent.unit_tail(problem.cone_slack(interior))
    found = Multipliers(
        y=np.zeros(problem.num_inequalities),
        nu=np.zeros(problem.num_equalities),
        w=np.concatenate([[1.0], -unit]),
    )
    assert updates.deactivate_cones(problem, working_set, found, z=interior)[1] == (0,)


def test_without_a_point_the_geometric_clause_is_skipped():
    """The multiplier alone is still a complete rule, which is what #23 established."""
    instance = families.basic(4, seed=0)
    problem = instance.problem
    interior = instance.witness.copy()
    interior[-1] += 1.0
    working_set = updates.set_cone_status(WorkingSet.empty(problem), 0, ConeStatus.TANGENT)
    unit = tangent.unit_tail(problem.cone_slack(interior))
    found = Multipliers(
        y=np.zeros(problem.num_inequalities),
        nu=np.zeros(problem.num_equalities),
        w=np.concatenate([[1.0], -unit]),
    )
    assert updates.deactivate_cones(problem, working_set, found)[1] == ()


# ----------------------------------------------------------------------------------
# The "Done when": no benchmark instance cycles
# ----------------------------------------------------------------------------------


def test_no_structured_family_cycles():
    """Every family reaches an optimum, which a cycling solve cannot do."""
    for family in (
        families.basic,
        families.box,
        families.sector,
        families.turnover,
        families.factor_exposure,
        families.nearly_redundant,
        families.highly_correlated,
        families.ill_conditioned,
        families.nearly_active_cone,
        families.degenerate_optimum,
        families.many_active_bounds,
    ):
        for seed in range(2):
            answer = solver.solve(family(8, seed=seed).problem)
            assert answer.status == "optimal", (family.__name__, seed, answer.status)


def test_no_randomized_instance_cycles():
    """§16.3's generator randomizes the shape, including the degeneracy that causes cycling.

    Asserted on the *returns* counter rather than on the status, because the two questions
    are different and conflating them is what made this hard to see. An instance that
    exhausts the iteration limit may be cycling or may simply be converging slowly under one
    working set, and only the counter distinguishes them. Before #29 the worst instance
    returned to a single working set 486 times, alternating `APEX` and `INACTIVE`; now the
    worst returns twice.
    """
    worst = max(solver.solve(random_instance(seed).problem).metrics.working_set_revisits for seed in range(40))
    assert worst <= anticycling.REVISITS, worst


def test_the_apex_oscillation_is_the_one_that_was_actually_there():
    """The regression instance, found rather than constructed.

    §8.2 describes oscillation "between active and inactive states" and its examples are
    about a *nearly* active cone. The cycle these instances actually exhibited was at the
    apex: #24's branch releases a factor it cannot justify, the released direction cannot be
    travelled -- result 4, and arithmetic rather than tolerance -- so the step moves nothing,
    and §7.3 reads the same unchanged geometry and puts the factor straight back. Seed 8 did
    that 486 times.

    The fix is to not re-derive a status from geometry that has not changed, which is why
    this asserts on the iterate rather than on the answer: the loop must *stop*, and what it
    stops with is #39's business.
    """
    answer = solver.solve(random_instance(8).problem)
    assert answer.metrics.working_set_revisits <= anticycling.REVISITS
    assert answer.metrics.iterations < 1000


def test_the_nearly_active_cone_family_is_the_one_this_was_built_for():
    """#33's family puts an iterate inside the activation band, which is §8.2's scenario."""
    answer = solver.solve(families.nearly_active_cone(8, seed=0).problem)
    assert answer.status == "optimal"
    assert answer.residuals.is_optimal()


def test_a_refused_iterate_sends_the_loop_to_the_multiplier_tests():
    """§17.2's merit safeguard inside the loop rather than on its own.

    The direction is a descent direction and the step was chosen along it, so an accepted
    iterate that is worse than the best seen means the arithmetic and the geometry have
    disagreed — which no instance here does, so the refusal is induced. What is under test
    is what the loop does *with* a refusal: it must not step, and it must fall through to the
    multiplier tests exactly as a stall does, rather than looping on the same rejected
    candidate until the iteration limit.
    """
    real = anticycling.Guard.accepts
    anticycling.Guard.accepts = lambda _self, _value: False
    try:
        answers = [
            solver.solve(family(5, seed=0).problem) for family in (families.basic, families.box, families.sector)
        ]
    finally:
        anticycling.Guard.accepts = real
    for answer in answers:
        assert answer.status in {"optimal", "stalled", "degenerate"}
        assert answer.metrics.iterations < 50, "a refusal must end the loop, not repeat inside it"
