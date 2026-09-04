"""§12's comparison study, in the four modes the paper asks for.

Issue #34. What is tested is that the study measures the right things and that Success
Criterion 5 — every generated problem's objective agrees with the reference within the
prescribed tolerance — actually holds. The performance numbers are reported rather than
asserted, because a test that pinned wall-clock time would fail on a busy machine and tell
nobody anything; what is asserted about performance is that it is *measured*, which is a
claim that has been wrong before.
"""

import pytest

from cosa.experiments import benchmarks
from cosa.experiments.reference import SolverUnavailableError, default_solver
from cosa.solver.termination import Residuals


@pytest.fixture(scope="module")
def oracle():
    """A reference solver, or a skip. The study runs without one; this file does not."""
    try:
        return default_solver()
    except SolverUnavailableError as missing:  # pragma: no cover - CI installs one
        pytest.skip(str(missing))


@pytest.fixture(scope="module")
def comparisons(oracle):
    """One pass of the study, shared."""
    return benchmarks.benchmark(8, seeds=(0,), oracle=oracle, large=40)


# ----------------------------------------------------------------------------------
# Success Criterion 5
# ----------------------------------------------------------------------------------


def test_every_objective_agrees_with_the_reference(comparisons):
    """§16.3's requirement and Success Criterion 5, which is the study's reason to exist."""
    assert benchmarks.disagreements(comparisons) == ()


def test_the_agreement_is_actually_checked(comparisons):
    """A study that silently had no reference would report the same thing as one that agreed.

    So the count of comparisons with something to compare against is asserted too — this is
    the distinction `Accuracy.reference is None` exists to keep.
    """
    checked = [c for c in comparisons if c.accuracy.reference is not None]
    assert len(checked) >= len(comparisons) - 1, "only the sequence mode has no single reference"


def test_agreement_is_vacuously_true_without_a_reference():
    """And is reported as such rather than as agreement."""
    accuracy = benchmarks.Accuracy(
        residuals=Residuals(
            primal=0.0, dual=0.0, stationarity=0.0, linear_complementarity=0.0, cone_complementarity=0.0
        ),
        objective=1.0,
        reference=None,
        gap=float("inf"),
        expected_return=0.0,
        deviation=0.0,
    )
    assert accuracy.agrees
    assert "no reference" in str(accuracy)


# ----------------------------------------------------------------------------------
# The four modes
# ----------------------------------------------------------------------------------


def test_all_four_modes_are_reported(comparisons):
    """§12 lists four and separates them because they are four different questions."""
    assert {comparison.mode for comparison in comparisons} == set(benchmarks.MODES)


def test_every_mode_reaches_an_optimum(comparisons):
    """A mode that failed would make its metrics meaningless."""
    assert all(comparison.status == "optimal" for comparison in comparisons)


def test_the_large_mode_is_the_only_one_that_measures_memory(comparisons):
    """§12.3 asks for memory "where relevant" and `tracemalloc` roughly doubles a solve.

    The large-problem mode is the only one where the answer could be interesting, so it is
    the only one that pays.
    """
    tracked = {c.mode for c in comparisons if c.performance.metrics.peak_memory is not None}
    assert tracked == {"large"}


def test_the_sequence_mode_aggregates_a_whole_frontier(comparisons):
    """Because that is how a caller tracing a frontier experiences the cost."""
    sequence = next(c for c in comparisons if c.mode == "sequence")
    assert sequence.performance.metrics.iterations > 100
    assert "frontier" in sequence.instance


# ----------------------------------------------------------------------------------
# §12.2's and §12.3's tables
# ----------------------------------------------------------------------------------


def test_the_accuracy_table_carries_what_paper_asks_for(comparisons):
    """Five residuals, objective, expected return, standard deviation."""
    accuracy = comparisons[0].accuracy
    assert accuracy.residuals.is_optimal()
    assert accuracy.deviation > 0.0
    assert accuracy.expected_return != 0.0


def test_the_performance_table_carries_what_paper_asks_for(comparisons):
    """Wall clock, iterations, KKT solves, active-set changes, factorization time."""
    for comparison in comparisons:
        metrics = comparison.performance.metrics
        assert metrics.runtime > 0.0, f"{comparison.mode}: runtime was never being recorded"
        assert metrics.iterations > 0
        assert metrics.kkt_solves > 0
        assert metrics.factorization_time > 0.0


def test_runtime_is_measured_from_inside_the_solve(comparisons):
    """The regression test for a bug this study found.

    `Metrics.runtime` was zero for every solve ever recorded: `cosa.solve` builds its
    `Solution` from inside the `with recorder.solving()` block, so a runtime latched in that
    block's `finally` was not yet set when anybody read it. It had been wrong since #15 and
    #34 is the first consumer with a reason to look.
    """
    assert all(comparison.performance.metrics.runtime > 0.0 for comparison in comparisons)


# ----------------------------------------------------------------------------------
# The wall-clock finding
# ----------------------------------------------------------------------------------


def test_the_reference_is_faster_and_the_study_says_so(comparisons):
    """The finding, reported rather than buried.

    CVXPY spends most of its time building a problem rather than solving one, so this
    comparison ought to flatter COSA — and it does not. §20 asks for a characterization of
    *when* conic active-set methods work well, which is not a claim that they always do.
    """
    speedups = [c.speedup for c in comparisons if c.speedup is not None]
    assert speedups
    assert max(speedups) < 1.0, "if this ever fails, the paper's results section changes"


def test_the_iteration_counts_are_where_the_structure_is(comparisons):
    """Which is why §12.3 asks for wall clock and iteration counts side by side."""
    warm = [c for c in comparisons if c.mode == "warm"]
    cold = [c for c in comparisons if c.mode == "cold"]
    assert sum(c.performance.metrics.factorizations for c in warm) <= sum(
        c.performance.metrics.factorizations for c in cold
    )


# ----------------------------------------------------------------------------------
# The report
# ----------------------------------------------------------------------------------


def test_the_report_runs(oracle):
    """It is what the paper's results section is written from."""
    text = benchmarks.report(6, seeds=(0,), oracle=oracle, large=20)
    assert "Success Criterion 5" in text
    assert "wall clock vs reference" in text
    for mode in benchmarks.MODES:
        assert mode in text


def test_a_comparison_renders_its_own_row(comparisons):
    """One line carrying both tables, which is what a results table row is."""
    rendered = str(comparisons[0])
    assert "cold" in rendered
    assert "obj" in rendered
    assert "iters" in rendered


class _Unavailable:
    """A reference solver that is installed and never works, which is a real case.

    An unlicensed commercial backend behaves exactly like this: `is_available` says yes
    because the package imports, and every solve fails on the license. The study has to
    report COSA's own numbers rather than crash, and has to mark the agreement column as
    having nothing to compare against rather than as agreement.
    """

    name = "unavailable"
    accuracy = 1e-9

    def is_available(self) -> bool:
        """It claims to be."""
        return True

    def solve(self, problem):
        """And then is not."""
        raise SolverUnavailableError(self.name, "no license")


def test_the_study_runs_without_a_working_reference():
    """Every mode still reports, and nothing claims agreement it did not check."""
    comparisons = benchmarks.benchmark(6, seeds=(0,), oracle=_Unavailable(), large=15)
    assert {comparison.mode for comparison in comparisons} == set(benchmarks.MODES)
    assert all(comparison.accuracy.reference is None for comparison in comparisons)
    assert benchmarks.disagreements(comparisons) == (), "nothing to disagree with"
    assert all(comparison.speedup is None for comparison in comparisons)


def test_no_reference_installed_is_not_an_error(monkeypatch):
    """CI without the `reference` extra must still be able to run the study."""

    def missing(*_args, **_kwargs):
        """Stand in for `default_solver` on a machine with no reference installed."""
        raise SolverUnavailableError("none", "not installed")

    monkeypatch.setattr(benchmarks, "default_solver", missing)
    comparisons = benchmarks.benchmark(6, seeds=(0,), large=15)
    assert all(comparison.accuracy.reference is None for comparison in comparisons)
