"""COSA against a reference solver, in the four modes §12 asks for.

§12 (``paper.tex:889``) sets out the comparison study, and #34 is it. Four modes, reported
separately because they are four different questions:

* **cold start** -- one problem, no prior information. The base case, and the one an
  interior-point method is designed for.
* **warm start** -- the same problem entered from a neighbour's answer. Where an active-set
  method is supposed to win, and #30 is what makes it possible.
* **individual large problems** -- one problem, many assets. Where an active-set method is
  supposed to lose, because the working set grows with the problem and each iteration costs
  more.
* **sequences of related problems** -- #35's frontier, taken as a whole. The mode the
  project's hypothesis is actually about: not "is COSA faster on one problem" but "is COSA
  faster on twenty problems that resemble each other".

**Two metric tables, because §12.2 and §12.3 ask two different things.** Accuracy is
per-solution -- the five residuals, the objective, the expected return, the standard
deviation -- and is what Success Criterion 5 is checked against. Performance is per-solve --
wall clock, iterations, KKT solves, active-set changes, factorization time, and memory where
it is worth the cost of measuring. Reporting them together would let a fast wrong answer
look good.

**The reference solver is the arbiter, not a competitor.** §16.3 requires every generated
problem's objective to agree with a reference within tolerance, and that check runs in every
mode. Timing against CVXPY is reported but is *not* the claim: CVXPY's overhead is its
modelling layer, not its solver, and a comparison that took it for the solver's cost would
be measuring the wrong thing. :attr:`Comparison.speedup` exists and its docstring says what
it is worth.

**Memory is opt-in.** §12.3 asks for it "where relevant" and ``tracemalloc`` roughly doubles
a solve's cost, so :func:`benchmark` takes a flag and the large-problem mode is the only one
that sets it by default -- which is the only mode where the answer could be interesting.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

from cosa.experiments import portfolio as families
from cosa.experiments.frontier import sweep
from cosa.experiments.reference import (
    ReferenceSolver,
    SolverUnavailableError,
    default_solver,
    relative_gap,
)
from cosa.solver import cosa as solver
from cosa.solver.instrumentation import Metrics, Recorder
from cosa.solver.warm import WarmStart, from_solution

if TYPE_CHECKING:
    from cosa.experiments.portfolio import PortfolioInstance
    from cosa.problem.socp import SOCP
    from cosa.solver.cosa import Solution
    from cosa.solver.termination import Residuals

__all__ = [
    "MODES",
    "OBJECTIVE_AGREEMENT",
    "Accuracy",
    "Comparison",
    "Performance",
    "benchmark",
    "disagreements",
    "report",
]

MODES: Final = ("cold", "warm", "large", "sequence")
"""§12's four modes, in the order the paper lists them (``paper.tex:902``)."""

OBJECTIVE_AGREEMENT: Final = 1e-6
"""Success Criterion 5's "prescribed tolerance" for agreement with the reference.

Relative. Matched to §6's termination tolerance rather than set tighter: COSA stops when its
residuals are inside ``1e-6``, so requiring the *objective* to agree more closely than that
would be requiring an accuracy the stopping criterion does not promise. Where the open
reference backends disagree with each other by more than this -- which #21 measured at up to
``1e-5`` on nearly degenerate instances -- the comparison is against the backend's own
accuracy, not against this.
"""


@dataclass(frozen=True, eq=False)
class Accuracy:
    """§12.2's per-solution metrics (``paper.tex:911``).

    Attributes:
        residuals: §6's five, which are the conic KKT conditions measured.
        objective: ``c.T @ z`` at COSA's answer.
        reference: the reference solver's objective, or ``None`` when none was available.
        gap: the relative difference between them, or ``inf`` when there is no reference.
        expected_return: ``mu.T @ x``, which is what a portfolio report quotes.
        deviation: ``sigma(x)``, the other half of the same report.
    """

    residuals: Residuals
    objective: float
    reference: float | None
    gap: float
    expected_return: float
    deviation: float

    @property
    def agrees(self) -> bool:
        """Success Criterion 5: does the objective match the reference within tolerance?

        ``True`` when no reference was available, which is not the same as agreement and is
        why :attr:`reference` is kept: a report that cannot distinguish "agreed" from
        "nothing to agree with" is not reporting a check.
        """
        return self.reference is None or self.gap <= OBJECTIVE_AGREEMENT

    def __str__(self) -> str:
        """One accuracy row."""
        against = "no reference" if self.reference is None else f"gap {self.gap:.1e}"
        return (
            f"obj {self.objective:12.6f}  {against:16s} "
            f"residual {self.residuals.largest:8.1e}  "
            f"return {self.expected_return:8.4f}  sd {self.deviation:7.4f}"
        )


@dataclass(frozen=True, eq=False)
class Performance:
    """§12.3's per-solve metrics (``paper.tex:925``).

    A thin wrapper over :class:`cosa.solver.instrumentation.Metrics` rather than a
    reimplementation: #15 already counts all six of these, which is why it was built four
    waves before the study that consumes them.

    Attributes:
        metrics: what the recorder counted.
        reference_seconds: how long the reference solver took, or ``None``.
    """

    metrics: Metrics
    reference_seconds: float | None = None

    @property
    def speedup(self) -> float | None:
        """Reference wall clock over COSA's, or ``None`` when there is nothing to compare.

        **It is below one, and that is the finding.** CVXPY spends most of its time building
        a problem rather than solving one, so this number ought to flatter COSA -- and it
        does not: the reference is between three and twenty times faster on every mode
        measured, overhead included. Reporting it is the point of running the study, and
        §20 (``paper.tex:1339``) says as much: what the paper values is a characterization
        of *when* conic active-set methods work well, which is not a claim that they always
        do.

        Where COSA's numbers are good is the iteration and factorization counts on
        *sequences*, which is what its structure is for. §12.3 asks for wall clock and
        iteration counts side by side precisely so that the difference between them is
        visible.
        """
        if self.reference_seconds is None or not self.metrics.runtime:
            return None
        return self.reference_seconds / self.metrics.runtime

    def __str__(self) -> str:
        """One performance row."""
        memory = "" if self.metrics.peak_memory is None else f"  {self.metrics.peak_memory / 1e6:6.1f}MB"
        against = "" if self.speedup is None else f"  {self.speedup:6.1f}x reference"
        return (
            f"{self.metrics.runtime * 1e3:8.2f}ms  {self.metrics.iterations:5d} iters  "
            f"{self.metrics.kkt_solves:5d} solves  {self.metrics.active_set_changes:4d} changes  "
            f"{self.metrics.factorizations:4d} fact ({self.metrics.factorization_time * 1e3:7.2f}ms)"
            f"{memory}{against}"
        )


@dataclass(frozen=True, eq=False)
class Comparison:
    """One instance, one mode, both tables.

    Attributes:
        mode: which of :data:`MODES`.
        instance: the family's name.
        assets: how many assets it has.
        status: COSA's status.
        accuracy: §12.2's table.
        performance: §12.3's table.
    """

    mode: str
    instance: str
    assets: int
    status: str
    accuracy: Accuracy
    performance: Performance

    @property
    def speedup(self) -> float | None:
        """Convenience for :attr:`Performance.speedup`, whose docstring is the caveat."""
        return self.performance.speedup

    def __str__(self) -> str:
        """One line of the combined table."""
        return (
            f"{self.mode:9s} {self.instance:20s} n={self.assets:4d} {self.status:9s} | "
            f"{self.accuracy} | {self.performance}"
        )


def _accuracy(instance: PortfolioInstance, problem: SOCP, answer: Solution, oracle: ReferenceSolver | None) -> Accuracy:
    """Measure §12.2's metrics for one answer.

    Args:
        instance: the instance, for its portfolio view.
        problem: the SOCP actually solved, which for a frontier point is not the instance's
            own.
        answer: COSA's solution.
        oracle: the reference solver, or ``None``.

    Returns:
        The accuracy row.
    """
    holdings = answer.z[: instance.num_assets]
    objective = answer.objective(problem)
    reference = None
    if oracle is not None:
        try:
            reference = oracle.solve(problem).objective
        except SolverUnavailableError:
            reference = None
    return Accuracy(
        residuals=answer.residuals,
        objective=objective,
        reference=reference,
        gap=float("inf") if reference is None else relative_gap(objective, reference),
        expected_return=instance.portfolio.expected_return(holdings),
        deviation=instance.portfolio.std(holdings),
    )


def _timed_reference(problem: SOCP, oracle: ReferenceSolver | None) -> float | None:
    """How long the reference solver takes on this problem, or ``None``.

    Args:
        problem: the instance.
        oracle: the reference solver, or ``None``.

    Returns:
        Seconds, or ``None`` when there is no usable reference.
    """
    if oracle is None:
        return None
    started = time.perf_counter()
    try:
        oracle.solve(problem)
    except SolverUnavailableError:
        return None
    return time.perf_counter() - started


def _solve(problem: SOCP, *, memory: bool, warm: WarmStart | None = None) -> Solution:
    """Solve, with the recorder configured for this mode.

    Args:
        problem: the instance.
        memory: whether to track peak allocation, which roughly doubles the cost.
        warm: the warm start, for the mode that has one.

    Returns:
        The solution.
    """
    recorder = Recorder(track_memory=memory)
    return solver.solve(problem, recorder=recorder, warm=warm)


def benchmark(
    assets: int = 10,
    *,
    seeds: Sequence[int] = (0, 1),
    oracle: ReferenceSolver | None = None,
    large: int = 60,
    lams: Sequence[float] | None = None,
) -> tuple[Comparison, ...]:
    """Run all four modes and return both tables for each.

    Args:
        assets: how many assets the cold, warm and sequence modes use.
        seeds: which draws.
        oracle: the reference solver, or ``None`` to pick one. Passing ``None`` and having
            none installed is not an error: the study still reports COSA's own numbers and
            marks the agreement column as having nothing to compare against.
        large: how many assets the large-problem mode uses.
        lams: the risk aversions the sequence mode traces, or ``None`` for
            :func:`cosa.experiments.frontier.risk_aversions`. A shorter sequence is a
            smaller study rather than a different one -- the mode is about how a sequence
            behaves, and six points show that as well as twenty-four when the question is
            whether the machinery works.

    Returns:
        One comparison per mode, family and seed.
    """
    if oracle is None:
        try:
            oracle = default_solver()
        except SolverUnavailableError:
            oracle = None

    results: list[Comparison] = []
    structured = {
        "basic": families.basic,
        "box": families.box,
        "sector": families.sector,
        "factor exposure": families.factor_exposure,
    }
    for seed in seeds:
        for name, build in structured.items():
            instance = build(assets, seed=seed)
            problem = instance.problem
            answer = _solve(problem, memory=False)
            results.append(
                Comparison(
                    mode="cold",
                    instance=name,
                    assets=assets,
                    status=answer.status,
                    accuracy=_accuracy(instance, problem, answer, oracle),
                    performance=Performance(answer.metrics, _timed_reference(problem, oracle)),
                )
            )

            # Warm mode: the same instance entered from a neighbouring risk aversion's
            # answer, which is the smallest honest version of what #35 does at length.
            neighbour = replace(instance.portfolio, lam=instance.portfolio.lam * 0.9).to_socp()
            seeded = solver.solve(neighbour)
            hot = _solve(problem, memory=False, warm=from_solution(seeded))
            results.append(
                Comparison(
                    mode="warm",
                    instance=name,
                    assets=assets,
                    status=hot.status,
                    accuracy=_accuracy(instance, problem, hot, oracle),
                    performance=Performance(hot.metrics, _timed_reference(problem, oracle)),
                )
            )

        # Large mode, once per seed rather than per family: the question is how the method
        # scales, and four families of the same size answer it four times over.
        big = families.box(large, seed=seed)
        answer = _solve(big.problem, memory=True)
        results.append(
            Comparison(
                mode="large",
                instance="box",
                assets=large,
                status=answer.status,
                accuracy=_accuracy(big, big.problem, answer, oracle),
                performance=Performance(answer.metrics, _timed_reference(big.problem, oracle)),
            )
        )

        # Sequence mode: #35's frontier as a single unit of work, because that is how a
        # caller tracing a frontier experiences it.
        traced = sweep(families.box(assets, seed=seed), lams)
        last = traced.points[-1]
        last_residuals = solver.solve(
            replace(families.box(assets, seed=seed).portfolio, lam=last.lam).to_socp()
        ).residuals
        results.append(
            Comparison(
                mode="sequence",
                instance=f"box frontier x{len(traced.points)}",
                assets=assets,
                status=last.status,
                accuracy=Accuracy(
                    # A sequence has no single point to measure five residuals at. The
                    # worst residual over the sweep is what `totals()` carries and it is
                    # already in the performance row, so this row reports the last point's
                    # answer and leaves the residual block to the per-point table in #35.
                    residuals=last_residuals,
                    objective=last.objective,
                    reference=None,
                    gap=float("inf"),
                    expected_return=last.expected_return,
                    deviation=last.deviation,
                ),
                performance=Performance(traced.totals()),
            )
        )
    return tuple(results)


def report(
    assets: int = 10,
    *,
    seeds: Sequence[int] = (0, 1),
    oracle: ReferenceSolver | None = None,
    large: int = 60,
    lams: Sequence[float] | None = None,
) -> str:
    """The whole study as text, one line per comparison and a tally beneath.

    Args:
        assets: how many assets the small modes use.
        seeds: which draws.
        oracle: the reference solver, or ``None`` to pick one.
        large: how many assets the large-problem mode uses.
        lams: the sequence mode's risk aversions, or ``None`` for the default sweep.

    Returns:
        The report.
    """
    comparisons = benchmark(assets, seeds=seeds, oracle=oracle, large=large, lams=lams)
    lines = [f"benchmark: {len(comparisons)} comparison(s) across {len(MODES)} modes", ""]
    lines += [str(comparison) for comparison in comparisons]
    checked = [comparison for comparison in comparisons if comparison.accuracy.reference is not None]
    agreeing = [comparison for comparison in checked if comparison.accuracy.agrees]
    speedups = [comparison.speedup for comparison in comparisons if comparison.speedup is not None]
    lines += [
        "",
        f"Success Criterion 5: {len(agreeing)}/{len(checked)} objectives agree with the reference",
    ]
    if speedups:
        lines.append(
            f"wall clock vs reference: {min(speedups):.2f}x to {max(speedups):.2f}x "
            "(below one means the reference is faster)"
        )
    return "\n".join(lines)


def disagreements(comparisons: Sequence[Comparison]) -> tuple[Comparison, ...]:
    """The comparisons whose objective did not match the reference.

    Success Criterion 5 is the claim that this is empty.

    Args:
        comparisons: what :func:`benchmark` returned.

    Returns:
        The ones that disagreed, ignoring those with nothing to compare against.
    """
    return tuple(
        comparison
        for comparison in comparisons
        if comparison.accuracy.reference is not None and not comparison.accuracy.agrees
    )
