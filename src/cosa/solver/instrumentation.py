"""Counting what the experiments promise to measure, and checking what must always hold.

Two jobs that belong together because both are about watching a solve rather than
performing one.

**The counters.** §11 (``paper.tex:878``) promises the frontier experiment will measure
seven quantities, and §12.3 (``paper.tex:925``) names six more for the benchmark tables.
:class:`Metrics` is their union, and :class:`Recorder` is how a solve fills one in. Built
now rather than at M11 for a reason that has nothing to do with tidiness: if the counters
are not in place from the M2 baseline onward, then when #27 replaces the
refactorize-every-iteration policy of §13.1 there is no *before* number to compare its
factorization count against, and #35's frontier experiment has to reach into solver
internals after the fact to reconstruct what happened. A metric that was never recorded
cannot be recovered.

One of the seven is not a counter. "Number of iterations saved by warm starts" is a
difference between two solves, so it is :func:`iterations_saved` over two
:class:`Metrics` rather than a field that a single solve could fill in. Recording it as a
counter would require a solve to know what a different solve did.

**The invariants.** §14 (``paper.tex:1012``) sets out three levels of validation, and two
of them are *runtime* conditions rather than final-answer tests:

* **Level 1** (``paper.tex:1019``) -- *"Every accepted iterate must satisfy, within
  tolerance, ``Ax <= b``, ``Ex = d``, ``||Lx|| <= t``"*. Every accepted iterate, not the
  answer: an active-set method that wanders outside the feasible set and comes back has
  broken this even if it terminates correctly, and the ratio test is exactly the thing
  that can break it.
* **Level 2** (``paper.tex:1028``) -- the computed multipliers satisfy stationarity to a
  prescribed tolerance. Checked at every multiplier computation, for the same reason.

:class:`InvariantChecker` asserts both, and is **opt-in**: :data:`UNCHECKED` in production,
:data:`CHECKED` under test. Opt-in because Level 1 costs a matrix-vector product per
iterate and Level 2 costs another, which is a real fraction of a small solve's runtime --
and because the performance numbers §12.3 asks for must not be measured with the checker
running. The default is off so that a benchmark cannot accidentally include it; the test
suite passes :data:`CHECKED` explicitly.

Neither class knows anything about the solver loop, which is #20's. They are what #20 will
be handed.
"""

from __future__ import annotations

import contextlib
import time
import tracemalloc
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from cosa.active_set.multipliers import STATIONARITY_TOLERANCE, Multipliers
from cosa.geometry.soc import ConePosition, positions
from cosa.linear_algebra.kkt import RHO, direction
from cosa.problem.socp import _vector

if TYPE_CHECKING:
    from cosa import Vector
    from cosa.active_set.working_set import ConeStatus, WorkingSet
    from cosa.linear_algebra.kkt import Direction
    from cosa.problem.socp import SOCP

__all__ = [
    "CHECKED",
    "PRIMAL_TOLERANCE",
    "UNCHECKED",
    "InvariantChecker",
    "InvariantViolationError",
    "Metrics",
    "Recorder",
    "iterations_saved",
    "level_1_violations",
    "level_2_violations",
]

PRIMAL_TOLERANCE: Final = 1e-8
"""§14.1's "within tolerance" for feasibility of an accepted iterate.

Looser than the geometry module's rounding-level tolerance, because an iterate is the
output of a ratio test and a linear solve rather than an exact construction, and tighter
than the activation tolerance, because a point that is *inside* the activation band should
still be feasible.
"""


class InvariantViolationError(AssertionError):
    """An invariant §14 says must always hold did not.

    An ``AssertionError`` rather than a ``RuntimeError``, deliberately. This is not a
    condition a caller can handle or a state the algorithm is allowed to be in: §14.1 says
    *must*. If it fires, the solver is wrong, and the only useful response is to stop with
    the violated conditions named. It is raised explicitly rather than through an ``assert``
    statement so that it survives ``python -O``.
    """

    def __init__(self, level: int, violations: tuple[str, ...]) -> None:
        """Name the level and list every condition that failed.

        Every condition rather than the first: which *combination* broke is the diagnostic,
        and a checker that stops at the first violation hides it.

        Args:
            level: which of §14's validation levels, ``1`` or ``2``.
            violations: one line per violated condition.
        """
        super().__init__(f"Level {level} violated: " + "; ".join(violations))
        self.level = level
        self.violations = violations


@dataclass(frozen=True)
class Metrics:
    """A snapshot of one solve: every quantity §11 and §12.3 name.

    Immutable, so two solves can be compared without either being able to change. A
    :class:`Recorder` produces one with :meth:`Recorder.metrics`.

    Attributes:
        iterations: active-set iterations -- §11's first quantity and §12.3's "number of
            iterations".
        constraints_added: §7.1 activations.
        constraints_removed: §7.2 deactivations.
        cone_changes: how many times a cone factor's status changed. Counted separately
            from the linear add/drop because §7.4 makes it a different decision, and
            because a study of whether the conic working set churns needs it apart.
        factorizations: §11's "number of KKT factorizations" and the number #27 has to
            beat. Under §13.1's policy this equals :attr:`kkt_solves`; once #27 lands it
            will not, and the gap is the result.
        kkt_solves: §12.3's "number of KKT solves".
        runtime: total wall-clock seconds, §11's "total runtime" and §12.3's "wall-clock
            time".
        factorization_time: seconds inside factorizations, §12.3's "factorization time".
        kkt_residual: §11's "KKT residual" -- the last one recorded, which is the one that
            describes the answer.
        peak_memory: peak bytes allocated during the solve, or ``None`` when memory was not
            tracked. §12.3 asks for it "where relevant", and tracking it costs enough that
            relevance has to be opted into.
    """

    iterations: int = 0
    constraints_added: int = 0
    constraints_removed: int = 0
    cone_changes: int = 0
    factorizations: int = 0
    kkt_solves: int = 0
    runtime: float = 0.0
    factorization_time: float = 0.0
    kkt_residual: float = float("nan")
    peak_memory: int | None = None

    @property
    def active_set_changes(self) -> int:
        """§12.3's "number of active-set changes": every add, drop and cone transition."""
        return self.constraints_added + self.constraints_removed + self.cone_changes

    def __str__(self) -> str:
        """One line, for a benchmark table row or a log entry."""
        memory = "" if self.peak_memory is None else f" mem={self.peak_memory / 1e6:.1f}MB"
        return (
            f"{self.iterations} iters, {self.active_set_changes} changes "
            f"(+{self.constraints_added}/-{self.constraints_removed}/~{self.cone_changes}), "
            f"{self.factorizations} factorizations, {self.kkt_solves} solves, "
            f"{self.runtime * 1e3:.1f}ms ({self.factorization_time * 1e3:.1f}ms factorizing), "
            f"residual {self.kkt_residual:.3g}{memory}"
        )


def iterations_saved(cold: Metrics, warm: Metrics) -> int:
    """§11's seventh quantity: iterations saved by warm starting.

    A function of two solves rather than a field of one, because that is what it is. A
    negative result is meaningful and is not clamped: a warm start that costs *more*
    iterations than a cold one is a finding, and #35's experiment exists partly to find out
    whether that happens.

    Args:
        cold: the cold-start solve.
        warm: the warm-started solve of the same problem.

    Returns:
        ``cold.iterations - warm.iterations``.
    """
    return cold.iterations - warm.iterations


class Recorder:
    """Accumulates the counters of :class:`Metrics` over one solve.

    Mutable, unlike almost everything else in this package, because counting is what it
    does. The immutable half is :meth:`metrics`, which snapshots it -- so a solve can hand
    out a value that will not change underneath its reader.

    The intended shape of use, and the reason :meth:`solve_direction` exists:

        recorder = Recorder()
        with recorder.solving():
            while not converged:
                recorder.iteration()
                step = recorder.solve_direction(problem, working_set, z)
                ...
        metrics = recorder.metrics()

    Routing the direction solve through the recorder is what makes the factorization count
    trustworthy. #12 guarantees one call is one factorization; counting it at the call site
    means the guarantee and the counter cannot drift, whereas a hand-incremented counter
    beside the call is one edit away from being wrong -- and the number it produces is the
    baseline #27's whole result is measured against.
    """

    def __init__(self, *, track_memory: bool = False) -> None:
        """Start a recorder with every counter at zero.

        Args:
            track_memory: whether to sample peak allocation with ``tracemalloc`` inside
                :meth:`solving`. Off by default: it slows every allocation in the process,
                which would contaminate the runtime §12.3 also asks for.
        """
        self._track_memory = track_memory
        self._iterations = 0
        self._added = 0
        self._removed = 0
        self._cone_changes = 0
        self._factorizations = 0
        self._kkt_solves = 0
        self._runtime = 0.0
        self._factorization_time = 0.0
        self._residual = float("nan")
        self._peak_memory: int | None = None

    @contextlib.contextmanager
    def solving(self) -> Iterator[Recorder]:
        """Time the whole solve, and sample peak memory if that was asked for.

        Yields:
            This recorder, so the ``with`` statement can bind it.
        """
        started_tracing = self._track_memory and not tracemalloc.is_tracing()
        if started_tracing:
            tracemalloc.start()
        start = time.perf_counter()
        try:
            yield self
        finally:
            self._runtime += time.perf_counter() - start
            if self._track_memory:
                self._peak_memory = tracemalloc.get_traced_memory()[1]
                if started_tracing:
                    tracemalloc.stop()

    @contextlib.contextmanager
    def factorizing(self) -> Iterator[None]:
        """Count one KKT factorization and time it -- §12.3's "factorization time".

        Yields:
            Nothing; the block's body is the factorization.
        """
        self._factorizations += 1
        start = time.perf_counter()
        try:
            yield
        finally:
            self._factorization_time += time.perf_counter() - start

    def solve_direction(
        self,
        problem: SOCP,
        working_set: WorkingSet,
        z: Vector,
        *,
        rho: float = RHO,
    ) -> Direction:
        """Solve the direction subproblem, counting one factorization and one solve.

        Args:
            problem: the instance.
            working_set: what is currently believed active.
            z: the current point.
            rho: the ``rho`` of ``H = rho*I``.

        Returns:
            The direction, exactly as :func:`cosa.linear_algebra.kkt.direction` returns it.
        """
        self._kkt_solves += 1
        with self.factorizing():
            return direction(problem, working_set, z, rho=rho)

    def iteration(self) -> None:
        """Count one active-set iteration."""
        self._iterations += 1

    def constraint_added(self) -> None:
        """Count one §7.1 activation."""
        self._added += 1

    def constraint_removed(self) -> None:
        """Count one §7.2 deactivation."""
        self._removed += 1

    def cone_changed(self, before: ConeStatus, after: ConeStatus) -> None:
        """Count one cone status transition, if it is one.

        A no-op transition is not counted, so the number means what it says: how often the
        conic working geometry actually moved.

        Args:
            before: the status before.
            after: the status after.
        """
        if before is not after:
            self._cone_changes += 1

    def kkt_residual(self, residual: float) -> None:
        """Record the KKT residual -- §11's sixth quantity.

        The last value recorded is the one :attr:`Metrics.kkt_residual` reports, because
        the residual that describes a solve is the one it finished at.

        Args:
            residual: the residual.
        """
        self._residual = float(residual)

    def metrics(self) -> Metrics:
        """Snapshot the counters.

        Returns:
            An immutable :class:`Metrics` that will not change as this recorder continues.
        """
        return Metrics(
            iterations=self._iterations,
            constraints_added=self._added,
            constraints_removed=self._removed,
            cone_changes=self._cone_changes,
            factorizations=self._factorizations,
            kkt_solves=self._kkt_solves,
            runtime=self._runtime,
            factorization_time=self._factorization_time,
            kkt_residual=self._residual,
            peak_memory=self._peak_memory,
        )


def level_1_violations(problem: SOCP, z: Vector, *, tolerance: float = PRIMAL_TOLERANCE) -> tuple[str, ...]:
    """§14.1's three conditions, checked at a point, with the failures described.

    Returns descriptions rather than a bare boolean because the point of a per-iterate
    invariant is to say what went wrong at the iterate it went wrong at -- a solver that
    reports "infeasible iterate" without naming the row leaves the reader to reconstruct it.

    Tolerances are relative to each block's own scale, so a portfolio with a budget of 1
    and one with a notional of ``1e9`` are held to comparable *relative* accuracy.

    Args:
        problem: the instance.
        z: the iterate to check.
        tolerance: the relative feasibility tolerance.

    Returns:
        One description per violated condition, empty when Level 1 holds.
    """
    point = _vector("z", z, size=problem.num_variables)
    violations: list[str] = []

    if problem.num_inequalities:
        excess = problem.A @ point - problem.b
        bound = tolerance * np.maximum(1.0, np.abs(problem.b))
        worst = int(np.argmax(excess - bound))
        if excess[worst] > bound[worst]:
            violations.append(f"A @ z <= b fails at row {worst} by {excess[worst]:.3g}")

    if problem.num_equalities:
        error = np.abs(problem.E @ point - problem.d)
        bound = tolerance * np.maximum(1.0, np.abs(problem.d))
        worst = int(np.argmax(error - bound))
        if error[worst] > bound[worst]:
            violations.append(f"E @ z = d fails at row {worst} by {error[worst]:.3g}")

    for factor, where in enumerate(positions(problem.cone, problem.cone_slack(point), tolerance=tolerance)):
        if where is ConePosition.EXTERIOR:
            violations.append(f"||L @ x|| <= t fails at cone factor {factor}")
    return tuple(violations)


def level_2_violations(
    problem: SOCP,
    multipliers: Multipliers,
    *,
    tolerance: float = STATIONARITY_TOLERANCE,
) -> tuple[str, ...]:
    """§14.2's condition, checked on computed multipliers.

    Args:
        problem: the instance.
        multipliers: the multipliers to check.
        tolerance: the prescribed stationarity tolerance.

    Returns:
        One description if stationarity fails, empty otherwise.
    """
    error = multipliers.stationarity_error(problem)
    if error > tolerance:
        return (f"stationarity residual {error:.3g} exceeds {tolerance:.3g}",)
    return ()


@dataclass(frozen=True)
class InvariantChecker:
    """Asserts §14's Levels 1 and 2, when enabled.

    A value rather than a switch, so a solve is handed the checker it should use instead of
    consulting global state. :data:`CHECKED` and :data:`UNCHECKED` are the two a caller
    normally wants; a custom one is for a study that needs its own tolerances -- #36's
    degeneracy work, which will want to see how far the invariants bend before they break.

    Attributes:
        enabled: whether to check at all. ``False`` makes every method a no-op, so a
            solver can call them unconditionally and a benchmark pays nothing.
        primal: the Level 1 feasibility tolerance.
        stationarity: the Level 2 stationarity tolerance.
    """

    enabled: bool = False
    primal: float = PRIMAL_TOLERANCE
    stationarity: float = STATIONARITY_TOLERANCE

    def accepted_iterate(self, problem: SOCP, z: Vector) -> None:
        """Assert Level 1 at an iterate the solver is about to accept.

        Args:
            problem: the instance.
            z: the iterate.

        Raises:
            InvariantViolationError: if any of §14.1's three conditions fails.
        """
        if not self.enabled:
            return
        violations = level_1_violations(problem, z, tolerance=self.primal)
        if violations:
            raise InvariantViolationError(1, violations)

    def computed_multipliers(self, problem: SOCP, multipliers: Multipliers) -> None:
        """Assert Level 2 at a multiplier computation.

        Args:
            problem: the instance.
            multipliers: the multipliers just computed.

        Raises:
            InvariantViolationError: if stationarity fails.
        """
        if not self.enabled:
            return
        violations = level_2_violations(problem, multipliers, tolerance=self.stationarity)
        if violations:
            raise InvariantViolationError(2, violations)


UNCHECKED: Final = InvariantChecker(enabled=False)
"""The production checker: no checks, no cost. The default everywhere."""

CHECKED: Final = InvariantChecker(enabled=True)
"""The test checker: both levels asserted at every opportunity.

What the test suite passes, so that the invariants of §14 are exercised on every solve a
test performs rather than only where someone remembered to assert them.
"""
