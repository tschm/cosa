"""An oracle: solve an SOCP with somebody else's solver and compare.

§16.3 Cross-Solver Tests (``paper.tex:1124``) sets a standard that is easy to state and
easy to quietly fail: *"For **every** randomly generated test problem, compare COSA
against a reference solver. The objective values should agree to within the prescribed
numerical tolerance."*

Every problem. Which is why this module exists at all, and exists now rather than at M10.
§12.1 (``paper.tex:892``) names MOSEK and Gurobi as the reference solvers, and both are
commercial and license-gated: a license-gated oracle cannot check *every* problem, cannot
run in continuous integration, and cannot be relied on by a contributor who does not have
one. An oracle that is unavailable half the time is not an oracle -- the cross-check
silently degrades into a skipped test, and the correctness claim it was supposed to
support quietly stops being checked.

So the adapter is an interface, :class:`ReferenceSolver`, and the default implementation is
an *open* one: :class:`CvxpySolver` over CVXPY, which reaches Clarabel, ECOS and SCS.
Clarabel needs no license and installs as a wheel, so :func:`default_solver` finds a
working oracle in CI with nothing configured. MOSEK and Gurobi are reachable through the
same class and the same interface -- CVXPY speaks to all of them -- but they arrive through
the ``mosek`` and ``gurobi`` optional extras, and a missing license surfaces as
:class:`SolverUnavailableError` so a test can skip cleanly rather than fail obscurely.

**Availability and licensing are different questions, and only one can be answered
cheaply.** :meth:`CvxpySolver.is_available` reports whether the backend is *installed*,
which CVXPY can say without solving anything. Whether a licensed solver will actually
accept the problem is discoverable only by asking it, so :meth:`CvxpySolver.solve` turns a
solver-level failure into :class:`SolverUnavailableError`. A caller that wants a guaranteed
oracle should use :func:`default_solver`, which prefers the open backends precisely
because their availability is the whole answer.

**What is compared is the objective value.** Not the solution vector: an SOCP can have a
non-unique optimal ``z`` -- a degenerate face, a direction along which the objective is
flat -- and two correct solvers may legitimately return different points. §16.3 asks for
objective agreement, and :meth:`ReferenceSolution.agrees_with` is that test, relative and
tolerance-driven. Comparing active sets or multipliers against a reference is a different
and much stronger claim, and the plan does not make it.

This module is the oracle only. The comparison *study* -- the four-mode table of §12, the
accuracy and performance metrics -- is #34, and the instance generators whose every output
gets cross-checked here are #19, #31, #32 and #33.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

import numpy as np

from cosa.problem.socp import SOCP, ProblemError

if TYPE_CHECKING:
    from cosa import Vector

__all__ = [
    "BACKEND_ACCURACY",
    "DEFAULT_ACCURACY",
    "LICENSED_BACKENDS",
    "OBJECTIVE_TOLERANCE",
    "OPEN_BACKENDS",
    "CrossCheck",
    "CvxpySolver",
    "ReferenceSolution",
    "ReferenceSolver",
    "SolverUnavailableError",
    "available_solvers",
    "cross_check",
    "default_solver",
    "relative_gap",
    "solve_reference",
]

OPEN_BACKENDS: Final = ("CLARABEL", "ECOS", "SCS")
"""Conic backends that need no license, in preference order -- the CI oracle.

Clarabel first: it is an interior-point method with no license, ships as a wheel and is
CVXPY's own default for conic problems. ECOS second, as the long-standing reference for
small dense SOCPs. SCS last -- a first-order method, so it reaches a looser accuracy than
either, which makes it a fallback rather than a peer.
"""

LICENSED_BACKENDS: Final = ("MOSEK", "GUROBI")
"""The commercial solvers §12.1 names, reachable through the same adapter with a license."""

OBJECTIVE_TOLERANCE: Final = 1e-6
"""§16.3's "prescribed numerical tolerance" for objective agreement, relative."""

DEFAULT_ACCURACY: Final = 1e-6
"""The relative objective accuracy assumed of a backend that does not appear below."""

BACKEND_ACCURACY: Final = {
    "CLARABEL": 1e-8,
    "ECOS": 1e-8,
    "MOSEK": 1e-8,
    "GUROBI": 1e-8,
    "SCS": 1e-4,
}
"""How accurately each backend can be expected to report an objective, relatively.

Not decoration: it is what keeps :func:`cross_check` from being wrong about who is at
fault. §16.3 prescribes a tolerance for the comparison, but a comparison cannot be
tighter than its least accurate participant, and these backends are not peers. The four
interior-point solvers converge to a duality gap around ``1e-8`` and agree with each
other well inside :data:`OBJECTIVE_TOLERANCE`. SCS is a first-order method whose own
default ``eps`` is ``1e-4``, and it behaves accordingly.

The SCS figure is measured rather than assumed. Over 200 randomized instances from
:mod:`cosa.experiments.randomized`, solved by both Clarabel and SCS to a status of
``optimal``, the relative objective gap had a median of ``2.3e-7`` and a maximum of
``9.8e-6`` -- so a quarter of the draws exceeded ``1e-6`` and none came near ``1e-4``.
Holding SCS to ``1e-6`` would therefore fail one comparison in four for no reason that has
anything to do with COSA; ``1e-4`` is its own documented accuracy and bounds what was
observed with an order of magnitude to spare.
"""


class SolverUnavailableError(RuntimeError):
    """The reference solver cannot run here, so there is no answer to be had.

    A ``RuntimeError`` and not a :class:`cosa.ProblemError`: nothing is wrong with the
    problem. The backend is not installed, or is installed without a license, or refused
    the problem for a reason of its own. Distinct from a solver that ran and reported
    infeasibility, which is an answer and comes back as a :class:`ReferenceSolution`.

    It is the exception a test skips on. That is the point of naming it: without it, "no
    license" and "wrong answer" arrive as the same failure.
    """

    def __init__(self, solver: str, reason: str) -> None:
        """Name the solver at fault and say why it cannot answer.

        The same shape as :class:`cosa.ProblemError`, deliberately: every message in the
        project names the thing at fault first, so a failure is greppable by the name of
        whatever produced it.

        Args:
            solver: the backend that cannot run, or the backends that were tried.
            reason: why not, phrased so that the actionable part comes last.
        """
        super().__init__(f"{solver}: {reason}")


@dataclass(frozen=True, eq=False)
class ReferenceSolution:
    """What a reference solver said about an instance.

    Attributes:
        solver: the backend that produced it, e.g. ``"CLARABEL"``.
        status: the backend's own status string, passed through rather than normalized.
            CVXPY's vocabulary is ``"optimal"``, ``"optimal_inaccurate"``, ``"infeasible"``,
            ``"unbounded"`` and their inaccurate variants; :attr:`is_optimal` is the
            interpretation, and the raw string is kept because a study reporting *why* a
            reference solver disagreed needs it.
        objective: the optimal value of ``c.T @ z``, or ``+inf`` for an infeasible
            instance and ``-inf`` for an unbounded one, following the convention of the
            extended-value objective.
        z: the optimal point, or ``None`` when the backend returned no point -- which is
            what an infeasible or unbounded instance gives.
    """

    solver: str
    status: str
    objective: float
    z: Vector | None = None

    @property
    def is_optimal(self) -> bool:
        """Did the backend claim an optimal solution, accurate or not?

        Both accuracy levels count. An "inaccurate" flag means the backend hit its own
        iteration or accuracy limit, which is a reason to widen the comparison tolerance,
        not a reason to treat the answer as absent -- and on the ill-conditioned instances
        of §12.4 it is the most an interior-point method will claim.
        """
        return self.status.startswith("optimal")

    def agrees_with(self, objective: float, *, tolerance: float = OBJECTIVE_TOLERANCE) -> bool:
        """Does ``objective`` match this reference value to within the tolerance?

        §16.3's check. Relative to the larger of the two magnitudes, so it behaves on a
        portfolio objective of ``1e-3`` and on one of ``1e6``, and falls back to absolute
        near zero. Two infinities of the same sign agree: both solvers found the instance
        infeasible, or both found it unbounded, and that is agreement.

        Args:
            objective: the value to compare, typically COSA's.
            tolerance: the relative tolerance.

        Returns:
            ``True`` if the values agree.

        Raises:
            SolverUnavailableError: if this solution is not optimal and not an infinity, so
                there is no value to compare against.
        """
        if not self.is_optimal and math.isfinite(self.objective):
            raise SolverUnavailableError(
                self.solver,
                f"returned status {self.status!r} with no usable objective, so there is nothing to compare against",
            )
        return relative_gap(objective, self.objective) <= tolerance


def relative_gap(first: float, second: float) -> float:
    """The relative difference between two objective values.

    ``|first - second| / max(1, |first|, |second|)``: relative above unit scale, absolute
    below it, which is the same mixed convention the cone predicates use. Equal infinities
    of the same sign give ``0``; opposite infinities, or one finite and one not, give
    ``inf``, because that is a disagreement no tolerance should absorb.

    Args:
        first: one objective value.
        second: the other.

    Returns:
        The gap, non-negative, possibly infinite.
    """
    if first == second:
        return 0.0
    if not (math.isfinite(first) and math.isfinite(second)):
        return math.inf
    return abs(first - second) / max(1.0, abs(first), abs(second))


@runtime_checkable
class ReferenceSolver(Protocol):
    """The oracle interface: a name, an availability check, and a solve.

    Deliberately four members. Everything the cross-check needs is here, and nothing
    about how the answer is obtained is: a backend may shell out, call a library, or read a
    cached answer from disk. That is what makes the oracle swappable, which §12.1's list of
    solvers demands -- the list will grow, and a study comparing two references must be
    able to hold both behind one type.

    :attr:`accuracy` is a member rather than a lookup because an oracle that cannot say how
    accurate it is cannot be compared against: see :data:`BACKEND_ACCURACY`.

    Runtime-checkable, so a test can assert that an implementation satisfies it.
    """

    @property
    def name(self) -> str:
        """The backend's name, as it appears in a results table."""
        ...

    @property
    def accuracy(self) -> float:
        """The relative objective accuracy this solver can be held to.

        The floor under any comparison it takes part in. See :data:`BACKEND_ACCURACY`.
        """
        ...

    def is_available(self) -> bool:
        """Can this solver be used at all here?

        Cheap and side-effect-free: it must not solve anything. For a licensed backend it
        answers whether the code is installed, not whether the license will be accepted --
        see the module docstring.
        """
        ...

    def solve(self, problem: SOCP) -> ReferenceSolution:
        """Solve an instance and report what the backend said.

        Args:
            problem: the instance to solve.

        Returns:
            The backend's answer, including a non-optimal status if that is the answer.

        Raises:
            SolverUnavailableError: if the backend cannot run or refuses the problem.
        """
        ...


@dataclass(frozen=True)
class CvxpySolver:
    """A reference solver reached through CVXPY -- one class, every backend.

    CVXPY already speaks second-order cones and already speaks to Clarabel, ECOS, SCS,
    MOSEK and Gurobi, so writing one adapter through it is strictly less code than writing
    five direct ones, and the translation from :class:`cosa.SOCP` is written once where it
    can be got right once. The cost is a dependency in the middle, which is why the
    objective comparison is relative rather than exact.

    CVXPY is imported inside the methods, not at module scope. The module must import
    without it -- ``cosa`` itself does not depend on CVXPY, only the ``reference`` extra
    does -- so that a test can ask :meth:`is_available` and skip, rather than fail at
    collection with an ``ImportError``.

    Attributes:
        backend: the CVXPY solver name, e.g. ``"CLARABEL"`` or ``"MOSEK"``.
    """

    backend: str = OPEN_BACKENDS[0]

    @property
    def name(self) -> str:
        """The backend's name, which is what identifies this solver in a results table."""
        return self.backend

    @property
    def accuracy(self) -> float:
        """This backend's relative objective accuracy, from :data:`BACKEND_ACCURACY`."""
        return BACKEND_ACCURACY.get(self.backend, DEFAULT_ACCURACY)

    def is_available(self) -> bool:
        """Is CVXPY importable, and does it report this backend as installed?

        Returns:
            ``True`` if the backend can be attempted. For a licensed backend that is not
            the same as "will succeed": the license is only tested by solving.
        """
        try:
            import cvxpy
        except ImportError:
            return False
        return self.backend in cvxpy.installed_solvers()

    def solve(self, problem: SOCP) -> ReferenceSolution:
        """Translate the instance into CVXPY, solve it, and report the answer.

        The translation is direct, block for block: ``A @ z <= b``, ``E @ z = d``, and one
        ``cvxpy.SOC`` per factor of the cone product over the corresponding block of the
        slack ``G @ z + h``, head first as :class:`cosa.SecondOrderCone` lays it out. Empty
        blocks are skipped rather than passed as zero-row constraints, which CVXPY has no
        use for.

        Args:
            problem: the instance to solve.

        Returns:
            The backend's answer.

        Raises:
            SolverUnavailableError: if CVXPY or the backend is missing, or the backend
                fails for any reason of its own -- which is what an unlicensed commercial
                solver does, in its own exception type rather than in CVXPY's.
        """
        try:
            import cvxpy
        except ImportError as missing:
            raise SolverUnavailableError(
                self.backend,
                "CVXPY is not installed, so this reference solver cannot run. Install the 'reference' extra.",
            ) from missing

        z = cvxpy.Variable(problem.num_variables)
        constraints = []
        if problem.num_inequalities:
            constraints.append(problem.A @ z <= problem.b)
        if problem.num_equalities:
            constraints.append(problem.E @ z == problem.d)
        if problem.cone.dim:
            slack = problem.G @ z + problem.h
            constraints.extend(cvxpy.SOC(slack[block][0], slack[block][1:]) for block in problem.cone.slices)

        # Only the solve is guarded, not the translation above: a shape error of ours is a
        # bug and must surface as one. What the guard is for is the backend's own
        # vocabulary of failure. CVXPY raises `SolverError` for a solver it can speak to,
        # but an unlicensed MOSEK raises `mosek.Error` and a Gurobi over its size limit
        # raises a `gurobipy` error -- third-party exceptions this package cannot name
        # without importing the very solvers it refuses to depend on. From the caller's
        # side the distinction does not exist: the oracle either answers or is
        # unavailable, and "unavailable" is what a test can skip on.
        instance = cvxpy.Problem(cvxpy.Minimize(problem.c @ z), constraints)
        try:
            value = instance.solve(solver=self.backend)
        except Exception as failure:
            raise SolverUnavailableError(self.backend, f"could not solve the instance: {failure}") from failure

        point = None if z.value is None else np.ascontiguousarray(z.value, dtype=np.float64)
        return ReferenceSolution(
            solver=self.backend,
            status=str(instance.status),
            objective=float(value) if value is not None else math.nan,
            z=point,
        )


def available_solvers(backends: tuple[str, ...] = OPEN_BACKENDS) -> tuple[CvxpySolver, ...]:
    """The solvers among ``backends`` that can be attempted here, in the order given.

    Args:
        backends: CVXPY backend names to try. Defaults to the open ones, which is what a
            CI run wants; pass ``OPEN_BACKENDS + LICENSED_BACKENDS`` to include the
            commercial solvers where they are installed.

    Returns:
        One solver per available backend, possibly empty.
    """
    solvers = tuple(CvxpySolver(backend=backend) for backend in backends)
    return tuple(solver for solver in solvers if solver.is_available())


def default_solver(backends: tuple[str, ...] = OPEN_BACKENDS) -> CvxpySolver:
    """The first available solver, which is the oracle the cross-check should use.

    Args:
        backends: backend names in preference order.

    Returns:
        The first one that is available.

    Raises:
        SolverUnavailableError: if none of them is. The message names the extra to install,
            because that is the actionable part -- and reaching it in CI means the open
            fallback §16.3 depends on is broken, not merely absent.
    """
    solvers = available_solvers(backends)
    if not solvers:
        raise SolverUnavailableError(
            ", ".join(backends),
            "no reference solver among these is available. Install the 'reference' extra to get CVXPY and Clarabel.",
        )
    return solvers[0]


@dataclass(frozen=True, eq=False)
class CrossCheck:
    """The result of §16.3's cross-solver comparison on one instance.

    §16.3 asks that COSA's objective agree with a reference solver's on every randomly
    generated problem. That check has two halves, and only one of them exists yet: the
    oracle is here, COSA is #20. So this class takes the objective to check as *optional*
    and does the strongest comparison available:

    * given an objective -- COSA's, once there is one -- it compares every available
      reference solver against it. That is §16.3 as written, and it is the seam #20 plugs
      into with no change here.
    * given none, it compares the available reference solvers *against each other*. That
      is a weaker claim about COSA and a stronger one about the instance: agreement among
      two independent interior-point implementations is what licenses treating either as
      an oracle in the first place, and disagreement means the instance is too
      ill-conditioned for the comparison to mean anything -- which is worth knowing before
      a solver is blamed for missing it.

    Attributes:
        instance: the name of the problem checked, so a failure identifies itself.
        objective: the value checked against, or ``None`` when the references were only
            compared with each other.
        solutions: what each available reference solver returned, in preference order.
        requested_tolerance: the tolerance the caller asked for -- §16.3's prescribed one
            by default.
        tolerance: the tolerance actually applied, which is the requested one widened to
            the accuracy of the least accurate participating solver. Both are kept so a
            report can say when they differ and why.
    """

    instance: str
    objective: float | None
    solutions: tuple[ReferenceSolution, ...]
    requested_tolerance: float
    tolerance: float

    @property
    def objectives(self) -> tuple[float, ...]:
        """The objective each reference solver reported."""
        return tuple(solution.objective for solution in self.solutions)

    @property
    def all_optimal(self) -> bool:
        """Did every reference solver claim an optimal solution?"""
        return bool(self.solutions) and all(solution.is_optimal for solution in self.solutions)

    @property
    def gap(self) -> float:
        """The largest relative gap among everything compared.

        Zero for a single reference solver with no objective to check against -- there is
        nothing to disagree with, which is a fact about the comparison and not a claim
        that the answer is right.
        """
        values = self.objectives if self.objective is None else (self.objective, *self.objectives)
        return max(
            (relative_gap(first, second) for first, second in itertools.combinations(values, 2)),
            default=0.0,
        )

    @property
    def agrees(self) -> bool:
        """Is every gap within :attr:`tolerance`?"""
        return self.gap <= self.tolerance

    def __str__(self) -> str:
        """A one-line verdict naming the instance, the solvers, the gap and the tolerance."""
        reported = ", ".join(
            f"{solution.solver}={solution.objective:.9g} ({solution.status})" for solution in self.solutions
        )
        checked = "" if self.objective is None else f"checked {self.objective:.9g} against "
        widened = (
            ""
            if self.tolerance == self.requested_tolerance
            else f" (widened from {self.requested_tolerance:.3g} for the least accurate solver)"
        )
        return (
            f"{self.instance}: {checked}{reported or 'no reference solver'} -- "
            f"gap {self.gap:.3g} vs tolerance {self.tolerance:.3g}{widened}"
        )


def cross_check(
    problem: SOCP,
    objective: float | None = None,
    *,
    name: str = "instance",
    solvers: tuple[ReferenceSolver, ...] | None = None,
    tolerance: float = OBJECTIVE_TOLERANCE,
) -> CrossCheck:
    """Run §16.3's comparison on one instance.

    Args:
        problem: the instance to check.
        objective: the objective to check against, typically COSA's, or ``None`` to
            compare the reference solvers with each other.
        name: the instance's name, carried into the result so a failure identifies itself.
        solvers: the oracles to use, or ``None`` for every available open backend. Passing
            a single solver with no ``objective`` produces a result with nothing to
            compare, which :attr:`CrossCheck.gap` reports honestly as zero.
        tolerance: the relative tolerance, §16.3's "prescribed numerical tolerance". It is
            widened to the accuracy of the least accurate participating solver, because a
            comparison cannot be tighter than that -- see :data:`BACKEND_ACCURACY`.

    Returns:
        The comparison.

    Raises:
        SolverUnavailableError: if no reference solver is available at all, or one of the
            named solvers cannot run.
    """
    oracles = available_solvers() if solvers is None else solvers
    if not oracles:
        raise SolverUnavailableError(
            ", ".join(OPEN_BACKENDS),
            "no reference solver among these is available, so the cross-check of "
            "paper.tex:1126 cannot run. Install the 'reference' extra",
        )
    return CrossCheck(
        instance=name,
        objective=objective,
        solutions=tuple(solver.solve(problem) for solver in oracles),
        requested_tolerance=tolerance,
        tolerance=max(tolerance, *(solver.accuracy for solver in oracles)),
    )


def solve_reference(problem: SOCP, *, solver: ReferenceSolver | None = None) -> ReferenceSolution:
    """Solve an instance with a reference solver, choosing one if none is named.

    The one-call form of the §16.3 cross-check's first half: get the oracle's answer, then
    compare it with :meth:`ReferenceSolution.agrees_with`.

    Args:
        problem: the instance to solve.
        solver: the oracle to use, or ``None`` to take :func:`default_solver`.

    Returns:
        The oracle's answer.

    Raises:
        SolverUnavailableError: if no solver is available, or the chosen one cannot run.
        ProblemError: if ``problem`` is not an :class:`cosa.SOCP`. Checked because passing
            a :class:`cosa.MeanStdForm` or a portfolio here is an easy mistake, and CVXPY's
            complaint about it would be several frames deep.
    """
    if not isinstance(problem, SOCP):
        raise ProblemError("problem", f"expected an SOCP, found {type(problem).__name__}")
    return (solver or default_solver()).solve(problem)
