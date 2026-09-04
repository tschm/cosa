"""Five ways to solve the same KKT system, measured against each other.

§13 (``paper.tex:952``) asks for a comparison rather than a choice: sparse ``LDL^T``, QR,
the null-space method, the range-space method, all against the refactorize-every-iteration
reference §13.1 established. #26 keeps them as one issue for the reason the issue gives --
splitting them into four would lose the only thing that makes them meaningful, which is
that they are measured against each other and against the same baseline.

**What "the same answer" means here.** All five solve

    [ rho*I  W.T ] [ d  ]     [ g ]
    [ W       0  ] [ nu ]  = -[ 0 ]

and on a full-rank ``W`` the solution is unique, so agreement is a fact rather than a
tolerance choice. :func:`compare` measures the deviation from the reference anyway, because
a method that agrees to ``1e-16`` and one that agrees to ``1e-9`` are telling you different
things about their conditioning.

**Two of the five exploit ``H = rho*I``, and could not exist without it.** The null-space
and range-space methods both need ``H`` to be cheaply invertible or cheaply projectable:

* *null space* takes an orthonormal basis ``Z`` of ``{p : W @ p = 0}`` and reads the
  direction straight off, ``d = -(Z @ Z.T @ g) / rho``, then recovers ``nu`` by least
  squares. It never forms the saddle-point matrix, which is why #25 uses it as the route
  that survives a rank-deficient working set.
* *range space* eliminates ``d`` instead: from ``rho*d + W.T @ nu = -g`` and ``W @ d = 0``
  comes ``W @ W.T @ nu = -W @ g``, an ``m``-by-``m`` positive definite system. Cheap when
  the working set is small, which is the regime an active-set method lives in.

That both are available is a consequence of the mean-standard-deviation formulation being
*linear* in the objective -- the risk went into the cone, so there is no Hessian to invert.
A quadratic objective would leave the range-space method needing ``H^-1``.

**The sparse ``LDL^T`` of §13 is implemented dense, and the reason is a finding.** The
KKT matrix of a portfolio problem is *not* sparse, however sparse ``A`` is, because the
cone contributes the tangent row ``g_0 - u.T @ L`` -- and ``u`` is a normalized covariance
direction, so that row is dense in every asset. One dense row and its transpose put a dense
cross in the matrix and a dense frontal block in any elimination order. So the sparsity a
sparse factorization would exploit is destroyed by the very structure that makes the problem
conic, and measuring a dense ``LDL^T`` is measuring what is actually available. #27's
factorization *reuse* is the direction that survives this observation, which is worth
knowing before optimizing the wrong thing.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import scipy.linalg

from cosa.linear_algebra.kkt import Direction, KktSystem
from cosa.linear_algebra.rank import null_space_basis
from cosa.problem.socp import ProblemError

if TYPE_CHECKING:
    from cosa import Vector

__all__ = [
    "DEFAULT",
    "REFERENCE",
    "STRATEGIES",
    "Comparison",
    "Measurement",
    "compare",
    "solve_with",
]

REFERENCE: Final = "lu"
"""The strategy every other is measured against: §13.1's dense LU, refactorized each call."""

DEFAULT: Final = "range-space"
"""The strategy chosen as default, with the measurement in :func:`compare` that justifies it.

Chosen because it is the fastest on the instances this project actually solves and because
the reason it is fastest is structural rather than incidental: it solves an ``m``-by-``m``
system where the others solve an ``(n + m)``-by-``(n + m)`` one, and an active-set method's
working set is small compared with its variable count for most of a solve. It agrees with
the reference to machine precision.

It is *not* chosen for robustness, and the way it fails is worth knowing -- including that
*how* it fails depends on the machine. On a rank-deficient working set ``W @ W.T`` is
singular too, and what LAPACK does about that is not portable:

* on OpenBLAS (Linux CI) the solve raises ``LinAlgError``;
* on Apple's Accelerate it *returns*, with a direction that does not satisfy ``W @ d = 0``
  at all -- a plausible-looking wrong answer, the same trap #12's original singularity
  check fell into.

Both were observed on this project, the second locally and the first in CI, which is worth
recording twice over: the route is unusable on a degenerate set either way, and *no*
correctness property should be written in terms of which failure a given LAPACK produces.

So the loop does not reach this route on a degenerate set: #25's rank test refuses first,
and the null-space route is the one that still answers there. Fast for the common case, with
the uncommon one handled somewhere else on purpose.
"""


def _lu(system: KktSystem) -> tuple[Vector, Vector]:
    """§13.1's reference: dense LU of the whole saddle-point matrix.

    Args:
        system: the assembled system.

    Returns:
        The direction and the multipliers.
    """
    solution = np.linalg.solve(system.matrix, system.rhs)
    return solution[: system.num_variables], solution[system.num_variables :]


def _ldl(system: KktSystem) -> tuple[Vector, Vector]:
    """``LDL^T`` of the symmetric indefinite matrix, dense -- see the module docstring.

    Args:
        system: the assembled system.

    Returns:
        The direction and the multipliers.
    """
    lower, diagonal, permutation = scipy.linalg.ldl(system.matrix)
    factor = lower[permutation]
    forward = scipy.linalg.solve_triangular(factor, system.rhs[permutation], lower=True, unit_diagonal=True)
    middle = np.linalg.solve(diagonal, forward)
    backward = scipy.linalg.solve_triangular(factor.T, middle, lower=False, unit_diagonal=True)
    solution = np.empty_like(backward)
    solution[permutation] = backward
    return solution[: system.num_variables], solution[system.num_variables :]


def _qr(system: KktSystem) -> tuple[Vector, Vector]:
    """QR of the saddle-point matrix, which §8.3 also wants for its rank information.

    Args:
        system: the assembled system.

    Returns:
        The direction and the multipliers.
    """
    orthogonal, upper = scipy.linalg.qr(system.matrix, mode="economic")
    solution = scipy.linalg.solve_triangular(upper, orthogonal.T @ system.rhs, lower=False)
    return solution[: system.num_variables], solution[system.num_variables :]


def _null_space(system: KktSystem) -> tuple[Vector, Vector]:
    """Project the gradient onto the null space of ``W``; never form the saddle point.

    Args:
        system: the assembled system.

    Returns:
        The direction and the least-norm multipliers consistent with it.
    """
    basis = null_space_basis(system.W)
    direction = -(basis @ (basis.T @ system.gradient)) / system.rho
    residual = -(system.gradient + system.rho * direction)
    multipliers = np.linalg.lstsq(system.W.T, residual, rcond=None)[0]
    return direction, multipliers


def _range_space(system: KktSystem) -> tuple[Vector, Vector]:
    """Eliminate the direction and solve the ``m``-by-``m`` system for the multipliers.

    Args:
        system: the assembled system.

    Returns:
        The direction and the multipliers.
    """
    if not system.num_rows:
        return -system.gradient / system.rho, np.zeros(0)
    normal = system.W @ system.W.T
    multipliers = np.linalg.solve(normal, -(system.W @ system.gradient))
    direction = -(system.gradient + system.W.T @ multipliers) / system.rho
    return direction, multipliers


STRATEGIES: Final[Mapping[str, Callable[[KktSystem], tuple[Vector, Vector]]]] = {
    "lu": _lu,
    "ldl": _ldl,
    "qr": _qr,
    "null-space": _null_space,
    "range-space": _range_space,
}
"""Every strategy §13 names, plus the reference, by name.

A mapping rather than five functions because #26 is a comparison: the thing a caller wants
is to iterate over them, and the thing a study wants is to name one in a results table.
"""


def solve_with(strategy: str, system: KktSystem) -> Direction:
    """Solve one system with a named strategy.

    Args:
        strategy: a key of :data:`STRATEGIES`.
        system: the assembled system.

    Returns:
        The direction, in the same shape :func:`cosa.linear_algebra.kkt.solve` returns.

    Raises:
        ProblemError: if the strategy is not one of the five.
    """
    if strategy not in STRATEGIES:
        raise ProblemError("strategy", f"expected one of {', '.join(sorted(STRATEGIES))}, found {strategy!r}")
    direction, multipliers = STRATEGIES[strategy](system)
    return Direction(d=direction, multipliers=multipliers, layout=system.layout, rho=system.rho)


@dataclass(frozen=True)
class Measurement:
    """What one strategy did on one set of systems.

    Attributes:
        strategy: its name.
        seconds: total time to factorize and solve every system, best of the repeats. Best
            rather than mean, because the distribution's upper tail is scheduler noise and
            the lower bound is the thing being measured.
        deviation: the largest relative deviation of the *direction* from the reference's.
            The direction and not the multipliers: on a rank-deficient system the
            multipliers are not unique and the null-space method deliberately returns the
            least-norm ones, so comparing them would report a disagreement that is a
            difference of convention.
        failures: how many systems the strategy could not solve at all.
    """

    strategy: str
    seconds: float
    deviation: float
    failures: int

    @property
    def matches(self) -> bool:
        """Does it agree with the reference to something like machine precision?"""
        return self.failures == 0 and self.deviation <= 1e-8

    def __str__(self) -> str:
        """One results-table row."""
        verdict = "agrees" if self.matches else f"DIFFERS ({self.failures} failures)"
        return f"{self.strategy:<12} {self.seconds * 1e3:8.2f}ms  deviation {self.deviation:.2e}  {verdict}"


@dataclass(frozen=True)
class Comparison:
    """§13's comparison: every strategy, against the reference and against each other.

    Attributes:
        reference: the strategy the deviations are measured from.
        systems: how many systems each strategy solved.
        measurements: one per strategy, in the order :data:`STRATEGIES` lists them.
    """

    reference: str
    systems: int
    measurements: tuple[Measurement, ...]

    @property
    def all_agree(self) -> bool:
        """Do all five produce the reference's answer? Issue #26's first "done when"."""
        return all(measurement.matches for measurement in self.measurements)

    def fastest(self) -> str:
        """The quickest strategy that agrees with the reference.

        Returns:
            Its name. Agreement first: a wrong answer arriving sooner is not faster.
        """
        agreeing = [measurement for measurement in self.measurements if measurement.matches]
        return min(agreeing, key=lambda measurement: measurement.seconds).strategy

    def speedup(self, strategy: str) -> float:
        """How many times faster than the reference a strategy is.

        Args:
            strategy: the strategy to report.

        Returns:
            The ratio of the reference's time to its own; above one is faster.

        Raises:
            ProblemError: if the strategy was not measured.
        """
        times = {measurement.strategy: measurement.seconds for measurement in self.measurements}
        if strategy not in times or self.reference not in times:
            raise ProblemError("strategy", f"{strategy!r} was not measured")
        return times[self.reference] / times[strategy] if times[strategy] else float("inf")

    def __str__(self) -> str:
        """The whole table, one strategy per line."""
        header = f"{self.systems} system(s), reference {self.reference}, fastest {self.fastest()}"
        return "\n".join([header, *(f"  {measurement}" for measurement in self.measurements)])


def compare(
    systems: Sequence[KktSystem],
    *,
    reference: str = REFERENCE,
    repeats: int = 5,
) -> Comparison:
    """Solve every system with every strategy and report times and deviations.

    Args:
        systems: the assembled systems to solve. More than one, because a comparison on a
            single system measures its shape rather than the strategies.
        reference: the strategy to measure deviations from.
        repeats: how many times to repeat the timing loop, keeping the best.

    Returns:
        The comparison.

    Raises:
        ProblemError: if there are no systems, if ``repeats`` is not positive, or if the
            reference is not a known strategy.
    """
    if not systems:
        raise ProblemError("systems", "a comparison needs at least one system")
    if repeats < 1:
        raise ProblemError("repeats", f"expected at least one repeat, found {repeats}")
    if reference not in STRATEGIES:
        raise ProblemError("reference", f"unknown strategy {reference!r}")

    def attempt(strategy: Callable[[KktSystem], tuple[Vector, Vector]], system: KktSystem) -> Vector | None:
        """Solve one system, or report that this strategy could not.

        Args:
            strategy: the strategy to try.
            system: the system to solve.

        Returns:
            The direction, or ``None`` if the strategy failed on it.
        """
        try:
            return strategy(system)[0]
        except (np.linalg.LinAlgError, ValueError):
            return None

    truth = [STRATEGIES[reference](system)[0] for system in systems]
    measurements = []
    for name, strategy in STRATEGIES.items():
        best = float("inf")
        for _ in range(repeats):
            started = time.perf_counter()
            for system in systems:
                attempt(strategy, system)
            best = min(best, time.perf_counter() - started)

        deviation, failures = 0.0, 0
        for system, expected in zip(systems, truth, strict=True):
            found = attempt(strategy, system)
            if found is None:
                failures += 1
                continue
            scale = max(1.0, float(np.abs(expected).max(initial=0.0)))
            deviation = max(deviation, float(np.abs(found - expected).max(initial=0.0)) / scale)
        measurements.append(Measurement(strategy=name, seconds=best, deviation=deviation, failures=failures))

    return Comparison(reference=reference, systems=len(systems), measurements=tuple(measurements))
