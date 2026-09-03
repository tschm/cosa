"""The counters §11 and §12.3 promise, and the per-iterate invariants of §14.

The executable half of issue #15. Its "done when" has three parts, and the third is the
one worth stating carefully:

* *every metric the frontier and benchmarking sections name* -- checked field by field
  against the two lists, so a quantity the paper promises cannot go missing.
* *the frontier experiment can be written without touching solver internals* -- checked by
  writing the shape of it: `test_a_solve_can_be_measured_without_touching_internals`
  drives the pieces that exist through a `Recorder` and reads every number back off
  `Metrics`.
* *the Level 1/2 invariants are asserted per-iterate under test* -- §14.1 says "every
  accepted iterate", which is a runtime condition, so both levels are tested for what they
  do at a *bad* iterate as well as a good one. An invariant that cannot fail is not an
  invariant.

The solver loop itself is #20's, so what runs here is a fragment of it built out of Wave 1
and Wave 2: activation candidates, a direction solve, a multiplier recovery. That is enough
to exercise every counter, and it is deliberately not a solver.
"""

import numpy as np
import pytest

from cosa import ConeStatus, MeanStdPortfolio, Multipliers, ProblemError, WorkingSet
from cosa.active_set import multipliers as mult
from cosa.active_set import updates
from cosa.experiments import portfolio as families
from cosa.experiments import reference
from cosa.solver import instrumentation as inst

# The seven quantities §11 (`paper.tex:878`) promises the frontier experiment will measure,
# and the six §12.3 (`paper.tex:925`) adds. Written out so that the "done when" is checked
# against the paper rather than against the implementation.
FRONTIER_QUANTITIES = {
    "number of active-set iterations": "iterations",
    "number of constraints added": "constraints_added",
    "number of constraints removed": "constraints_removed",
    "number of KKT factorizations": "factorizations",
    "total runtime": "runtime",
    "KKT residual": "kkt_residual",
    # "number of iterations saved by warm starts" is a difference between two solves, so it
    # is `iterations_saved` rather than a field -- see the module docstring.
}

BENCHMARK_QUANTITIES = {
    "wall-clock time": "runtime",
    "number of iterations": "iterations",
    "number of KKT solves": "kkt_solves",
    "number of active-set changes": "active_set_changes",
    "factorization time": "factorization_time",
    "memory usage where relevant": "peak_memory",
}


@pytest.fixture
def instance():
    """A box-constrained instance whose optimum has several bounds active."""
    return families.box(6, seed=0)


@pytest.fixture
def optimum(instance):
    """The reference solver's optimum, as a point Level 1 must accept."""
    return reference.solve_reference(instance.problem).z


# ----------------------------------------------------------------------------------
# Every metric the paper names
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(("quantity", "field"), sorted(FRONTIER_QUANTITIES.items()))
def test_the_frontier_quantities_are_all_recorded(quantity, field):
    """§11's list, checked name by name -- issue #15's first "done when"."""
    assert hasattr(inst.Metrics(), field), f"§11 promises {quantity!r}"


@pytest.mark.parametrize(("quantity", "field"), sorted(BENCHMARK_QUANTITIES.items()))
def test_the_benchmark_quantities_are_all_recorded(quantity, field):
    """§12.3's list, likewise."""
    assert hasattr(inst.Metrics(), field), f"§12.3 promises {quantity!r}"


def test_iterations_saved_is_a_comparison_not_a_counter():
    """§11's seventh quantity needs two solves, so it is a function of two `Metrics`."""
    cold = inst.Metrics(iterations=17)
    warm = inst.Metrics(iterations=4)
    assert inst.iterations_saved(cold, warm) == 13
    assert not hasattr(inst.Metrics(), "iterations_saved")


def test_a_warm_start_that_costs_more_is_reported_as_negative():
    """Not clamped: a warm start that loses is a finding, and #35 exists to find it."""
    assert inst.iterations_saved(inst.Metrics(iterations=3), inst.Metrics(iterations=9)) == -6


def test_active_set_changes_counts_all_three_kinds():
    """§12.3's "active-set changes" is adds, drops and cone transitions together."""
    metrics = inst.Metrics(constraints_added=3, constraints_removed=2, cone_changes=1)
    assert metrics.active_set_changes == 6


def test_metrics_are_immutable():
    """So two solves can be compared without either being able to change."""
    with pytest.raises(AttributeError):
        inst.Metrics().iterations = 5


def test_metrics_describe_themselves():
    """A benchmark table row needs one line, not a field-by-field read."""
    rendered = str(inst.Metrics(iterations=4, factorizations=4, kkt_residual=1e-9, peak_memory=2_000_000))
    assert "4 iters" in rendered
    assert "4 factorizations" in rendered
    assert "mem=2.0MB" in rendered


def test_memory_is_absent_unless_it_was_tracked():
    """§12.3 asks for it "where relevant", so relevance has to be opted into."""
    assert inst.Metrics().peak_memory is None
    assert "mem=" not in str(inst.Metrics())


# ----------------------------------------------------------------------------------
# The recorder, over a fragment of the eventual loop
# ----------------------------------------------------------------------------------


def test_a_solve_can_be_measured_without_touching_internals(instance, optimum):
    """Issue #15's second "done when": every number read off `Metrics`, not off the solver.

    The shape #35's frontier experiment will take. It is not a solver -- #20 owns the loop
    -- but it exercises every counter through the public surface of the pieces that exist,
    which is the property the "done when" is really about.
    """
    problem = instance.problem
    recorder = inst.Recorder(track_memory=True)

    with recorder.solving():
        working_set = WorkingSet.empty(problem)
        for _ in range(3):
            recorder.iteration()
            for index in updates.activation_candidates(problem, optimum, working_set, tolerance=1e-6):
                working_set = updates.add_inequality(working_set, index)
                recorder.constraint_added()
            before = working_set.status(0)
            working_set = updates.activate_cones(problem, optimum, working_set)
            recorder.cone_changed(before, working_set.status(0))
            step = recorder.solve_direction(problem, working_set, optimum)
            recovered = mult.from_direction(problem, working_set, optimum, step)
            recorder.kkt_residual(recovered.stationarity_error(problem))

    metrics = recorder.metrics()
    assert metrics.iterations == 3
    assert metrics.constraints_added > 0
    assert metrics.cone_changes == 1, "the cone activates once and then stays put"
    assert metrics.kkt_solves == 3
    assert metrics.factorizations == 3
    assert metrics.runtime > 0.0
    assert metrics.factorization_time > 0.0
    assert metrics.factorization_time <= metrics.runtime
    assert metrics.kkt_residual < 1e-6
    assert metrics.peak_memory is not None
    assert metrics.peak_memory > 0


def test_routing_the_direction_solve_counts_one_factorization_each(instance, optimum):
    """#12 guarantees one call is one factorization; counting at the call site keeps it true.

    The alternative -- incrementing a counter beside the call -- is one edit away from being
    wrong, and the number it produces is the baseline #27's whole result is measured
    against.
    """
    problem = instance.problem
    recorder = inst.Recorder()
    working_set = WorkingSet.empty(problem)
    for _ in range(5):
        recorder.solve_direction(problem, working_set, optimum)
    metrics = recorder.metrics()
    assert metrics.factorizations == 5
    assert metrics.kkt_solves == 5
    assert metrics.factorizations == metrics.kkt_solves, "§13.1's policy, made visible"


def test_the_solve_direction_helper_returns_what_the_kkt_module_returns(instance, optimum):
    """A counting wrapper, not a reimplementation."""
    from cosa.linear_algebra import kkt

    problem = instance.problem
    working_set = WorkingSet.empty(problem)
    counted = inst.Recorder().solve_direction(problem, working_set, optimum, rho=2.0)
    direct = kkt.direction(problem, working_set, optimum, rho=2.0)
    np.testing.assert_array_equal(counted.d, direct.d)
    assert counted.rho == direct.rho == 2.0


def test_drops_are_counted_separately_from_adds():
    """§11 lists them as two quantities, so they are two counters."""
    recorder = inst.Recorder()
    recorder.constraint_added()
    recorder.constraint_removed()
    recorder.constraint_removed()
    metrics = recorder.metrics()
    assert (metrics.constraints_added, metrics.constraints_removed) == (1, 2)
    assert metrics.active_set_changes == 3


def test_a_no_op_cone_transition_is_not_counted():
    """The number means what it says: how often the conic geometry actually moved."""
    recorder = inst.Recorder()
    recorder.cone_changed(ConeStatus.TANGENT, ConeStatus.TANGENT)
    recorder.cone_changed(ConeStatus.TANGENT, ConeStatus.APEX)
    assert recorder.metrics().cone_changes == 1


def test_the_last_residual_recorded_is_the_one_reported():
    """The residual that describes a solve is the one it finished at."""
    recorder = inst.Recorder()
    for value in (1.0, 1e-3, 1e-9):
        recorder.kkt_residual(value)
    assert recorder.metrics().kkt_residual == 1e-9


def test_a_snapshot_does_not_change_as_the_recorder_continues():
    """Which is the whole reason `Metrics` is a separate, frozen object."""
    recorder = inst.Recorder()
    recorder.iteration()
    early = recorder.metrics()
    recorder.iteration()
    assert early.iterations == 1
    assert recorder.metrics().iterations == 2


def test_memory_is_not_tracked_unless_asked():
    """Tracking allocation slows every allocation, which would contaminate the runtime."""
    recorder = inst.Recorder()
    with recorder.solving():
        pass
    assert recorder.metrics().peak_memory is None


def test_runtime_accumulates_across_solves():
    """A warm-start study times two solves on one recorder if it wants to."""
    recorder = inst.Recorder()
    with recorder.solving():
        pass
    first = recorder.metrics().runtime
    with recorder.solving():
        pass
    assert recorder.metrics().runtime >= first


def test_factorization_time_is_recorded_even_when_the_body_raises():
    """A failed factorization still took time, and the counter still counted it."""
    recorder = inst.Recorder()
    with pytest.raises(ValueError, match="boom"), recorder.factorizing():
        raise ValueError("boom")
    metrics = recorder.metrics()
    assert metrics.factorizations == 1
    assert metrics.factorization_time >= 0.0


# ----------------------------------------------------------------------------------
# §14.1: Level 1 at every accepted iterate
# ----------------------------------------------------------------------------------


def test_level_1_holds_at_a_feasible_point(instance, optimum):
    """The optimum satisfies all three of §14.1's conditions."""
    assert inst.level_1_violations(instance.problem, optimum) == ()


def test_level_1_holds_at_every_family_s_witness():
    """Feasible by construction means Level 1 by construction, on all six families."""
    for family in families.all_families(seed=0):
        assert inst.level_1_violations(family.problem, family.witness) == (), family.name


def test_level_1_names_the_row_that_fails(instance, optimum):
    """A per-iterate invariant that does not say what went wrong leaves the work undone."""
    bad = optimum.copy()
    bad[0] += 1.0
    violations = inst.level_1_violations(instance.problem, bad)
    assert any("A @ z <= b fails at row" in line for line in violations)
    assert any("E @ z = d fails at row" in line for line in violations)


def test_level_1_reports_every_violated_condition_not_just_the_first(instance, optimum):
    """Which *combination* broke is the diagnostic."""
    bad = optimum.copy()
    bad[0] += 1.0
    assert len(inst.level_1_violations(instance.problem, bad)) == 3


def test_level_1_catches_an_infeasible_cone(instance, optimum):
    """§14.1's third condition, `||L @ x|| <= t`, checked through the cone predicates."""
    bad = optimum.copy()
    bad[-1] -= 0.5
    violations = inst.level_1_violations(instance.problem, bad)
    assert violations == ("||L @ x|| <= t fails at cone factor 0",)


def test_level_1_is_relative_to_each_block_s_scale():
    """So a budget of 1 and a notional of 1e9 are held to comparable relative accuracy."""
    small = families.box(4, seed=0).problem
    large = families.badly_scaled(4, seed=0).problem
    point = np.zeros(small.num_variables)
    point[-1] = 1.0
    assert inst.level_1_violations(small, point, tolerance=1.0) == ()
    assert inst.level_1_violations(large, point, tolerance=1.0) == ()


# ----------------------------------------------------------------------------------
# §14.2: Level 2 at every multiplier computation
# ----------------------------------------------------------------------------------


def test_level_2_inherits_the_accuracy_of_the_point_it_is_evaluated_at(instance, optimum):
    """Stationarity holds at a reference optimum -- to the reference solver's accuracy.

    Worth a test of its own because the number surprises: the residual here is around
    `1e-7`, comfortably outside `STATIONARITY_TOLERANCE`, and nothing is wrong. Level 2 is
    a statement about multipliers computed *at a point*, and this point came from Clarabel,
    which converges to a duality gap around `1e-8`. The recovered multipliers are exactly
    as stationary as the point they were recovered at is optimal, and no tighter.

    So the default tolerance is the right one for a point COSA produced and the wrong one
    for a point borrowed from an interior-point solver. `test_multipliers.py` checks the
    exact case, at the hand-solved optima where the residual is zero to machine precision.
    """
    problem = instance.problem
    working_set = WorkingSet.empty(problem)
    for index in updates.activation_candidates(problem, optimum, working_set, tolerance=1e-6):
        working_set = updates.add_inequality(working_set, index)
    working_set = updates.activate_cones(problem, optimum, working_set)
    step = inst.Recorder().solve_direction(problem, working_set, optimum)
    recovered = mult.from_direction(problem, working_set, optimum, step)

    assert inst.level_2_violations(problem, recovered) != (), "the default tolerance is tighter than Clarabel"
    assert inst.level_2_violations(problem, recovered, tolerance=1e-6) == ()
    assert recovered.stationarity_error(problem) < 1e-6


def test_level_2_catches_wrong_multipliers(instance):
    """Zeros are not stationary for a nonzero objective, and the checker says so."""
    problem = instance.problem
    zeros = Multipliers(
        y=np.zeros(problem.num_inequalities),
        nu=np.zeros(problem.num_equalities),
        w=np.zeros(problem.cone.dim),
    )
    violations = inst.level_2_violations(problem, zeros)
    assert len(violations) == 1
    assert "stationarity residual" in violations[0]


# ----------------------------------------------------------------------------------
# The checker is opt-in, and both states are real
# ----------------------------------------------------------------------------------


def test_the_enabled_checker_raises_on_a_bad_iterate(instance, optimum):
    """Issue #15's third "done when": the invariant is asserted, not merely computable."""
    bad = optimum.copy()
    bad[0] += 1.0
    with pytest.raises(inst.InvariantViolationError, match="Level 1 violated"):
        inst.CHECKED.accepted_iterate(instance.problem, bad)


def test_the_enabled_checker_raises_on_bad_multipliers(instance):
    """Level 2, at every multiplier computation."""
    problem = instance.problem
    zeros = Multipliers(
        y=np.zeros(problem.num_inequalities),
        nu=np.zeros(problem.num_equalities),
        w=np.zeros(problem.cone.dim),
    )
    with pytest.raises(inst.InvariantViolationError, match="Level 2 violated"):
        inst.CHECKED.computed_multipliers(problem, zeros)


def test_the_enabled_checker_is_silent_on_a_good_iterate(instance, optimum):
    """It only fires when something is wrong."""
    assert inst.CHECKED.accepted_iterate(instance.problem, optimum) is None


def test_the_disabled_checker_skips_level_2_as_well(instance):
    """Both levels are behind the same switch, so a benchmark pays for neither."""
    problem = instance.problem
    zeros = Multipliers(
        y=np.zeros(problem.num_inequalities),
        nu=np.zeros(problem.num_equalities),
        w=np.zeros(problem.cone.dim),
    )
    assert inst.UNCHECKED.computed_multipliers(problem, zeros) is None


def test_the_disabled_checker_checks_nothing(instance, optimum):
    """So a benchmark cannot accidentally include the cost, or the failure."""
    bad = optimum.copy()
    bad[0] += 1.0
    assert inst.UNCHECKED.accepted_iterate(instance.problem, bad) is None
    assert inst.UNCHECKED.enabled is False


def test_the_default_checker_is_off():
    """§12.3's performance numbers must not be measured with the checker running."""
    assert inst.InvariantChecker().enabled is False


def test_a_checker_can_carry_its_own_tolerances():
    """For #36, which will want to see how far the invariants bend before they break."""
    lenient = inst.InvariantChecker(enabled=True, primal=1.0)
    instance = families.box(4, seed=0)
    nearly = instance.witness.copy()
    nearly[0] += 0.5
    assert inst.level_1_violations(instance.problem, nearly) != ()
    assert lenient.accepted_iterate(instance.problem, nearly) is None


def test_the_violation_error_carries_the_level_and_the_lines(instance, optimum):
    """So a caller can report which level failed without parsing the message."""
    bad = optimum.copy()
    bad[0] += 1.0
    with pytest.raises(inst.InvariantViolationError) as raised:
        inst.CHECKED.accepted_iterate(instance.problem, bad)
    assert raised.value.level == 1
    assert len(raised.value.violations) == 3


def test_the_violation_error_is_an_assertion_error():
    """§14.1 says *must*: this is not a state the algorithm is allowed to be in."""
    assert issubclass(inst.InvariantViolationError, AssertionError)


def test_the_checker_validates_its_point(instance):
    """A point of the wrong length is a bug, caught where it is handed over."""
    with pytest.raises(ProblemError, match="expected 7 entries"):
        inst.CHECKED.accepted_iterate(instance.problem, np.zeros(3))


# ----------------------------------------------------------------------------------
# The invariants hold on the instances that exist
# ----------------------------------------------------------------------------------


def test_level_1_holds_at_every_robustness_family_s_witness():
    """Even the adversarial instances are feasible where they say they are.

    Worth checking separately: a robustness family is built to break something, and if it
    broke its own witness's feasibility the family would be testing the generator rather
    than the solver.
    """
    for family in families.all_robustness(seed=0):
        assert inst.level_1_violations(family.problem, family.witness) == (), family.name


def test_a_rank_deficient_instance_satisfies_level_1_at_its_apex():
    """The apex is feasible, so §14.1 must accept an iterate that sits on it."""
    portfolio = MeanStdPortfolio.unconstrained(
        mu=np.array([0.10, 0.04, 0.06]), Sigma=np.ones((3, 3)), lam=1.0
    ).with_inequalities(np.vstack([np.eye(3), -np.eye(3)]), np.ones(6))
    problem = portfolio.to_socp()
    z = portfolio.socp_point(np.array([1.0, -1.0, 0.0]))
    assert inst.level_1_violations(problem, z) == ()
