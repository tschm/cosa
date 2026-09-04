"""Starting the next solve from the last one: §9 Phase VI, and the project's central bet.

§9 Phase VI (``paper.tex:767``) calls warm starts "expected to be one of the strongest
potential advantages of COSA", and that expectation is the reason the project exists. An
interior-point method cannot warm start usefully -- its iterates live strictly inside the
feasible set and a previous *solution* is on the boundary, which is the worst possible place
to begin a central path from. An active-set method's state is a *combinatorial* object, the
working set, and a nearby problem usually has a nearby one. That asymmetry is the whole
hypothesis, and #35's frontier experiment is where it gets tested.

**Four things are carried, and §9 lists them in the order of how much they are worth.**

* the **previous primal solution**, ``z``. Cheapest to reuse and cheapest to check: it is
  either feasible for the new instance or it is not, and :func:`cosa.solve` already refuses
  a start it cannot verify rather than silently replacing it.
* the **previous working set**. The valuable one. A cold solve spends most of its iterations
  discovering which constraints are active; a warm one is told. On a frontier sequence the
  answer is usually right, and when it is wrong §7.2 and §7.4 correct it in a few iterations
  rather than rediscovering it in a hundred.
* the **previous multipliers**, which #23 turned from an output into an input: they weight
  the Lagrangian curvature, so a warm start begins with the *right* Hessian instead of
  spending its first iterations rebuilding one from zero.
* the **previous factorizations**, which is why §9 Phase VI depends on Phase V and why #30
  hard-blocks on #27. A :class:`cosa.linear_algebra.reuse.Reuse` carried across problems
  starts the next solve one update away from its first direction rather than one
  factorization.

**What makes a warm start transferable is the shape, not the numbers.** Along a frontier
only ``lam`` changes, so ``A``, ``E``, ``G`` and the cone product are identical between
consecutive problems and only ``c`` moves. :meth:`WarmStart.fits` checks exactly that, and
checks it structurally rather than trusting the caller: a working set is a tuple of row
indices, and applying one to a problem with different rows is not a bad start but a wrong
answer.

**A warm start is a hint, never a commitment.** Every part of it is checked and any part may
be discarded: an infeasible point falls back to construction, a working set whose rows no
longer exist is dropped, multipliers of the wrong length are replaced by zeros. The
alternative -- refusing to warm start unless everything matches -- would make the feature
useless precisely where a sequence is most interesting, which is where the problems differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from cosa.active_set.multipliers import Multipliers
from cosa.active_set.working_set import WorkingSet
from cosa.linear_algebra.reuse import Reuse

if TYPE_CHECKING:
    from cosa import Vector
    from cosa.problem.socp import SOCP
    from cosa.solver.cosa import Solution

__all__ = [
    "WarmStart",
    "from_solution",
    "seed",
]


@dataclass(frozen=True, eq=False)
class WarmStart:
    """Everything one solve can hand the next.

    Attributes:
        z: the previous primal solution.
        working_set: the working set held there.
        multipliers: the multipliers there, or ``None``.
        cache: the factorization cache the previous solve built, or ``None``. Shared rather
            than copied, so a sequence of solves accumulates one cache -- which is the point:
            consecutive problems have nearly the same working sets, so the factorization the
            last solve ended on is usually one update from the next solve's first.
    """

    z: Vector
    working_set: WorkingSet
    multipliers: Multipliers | None = None
    cache: Reuse | None = None

    def fits(self, problem: SOCP) -> bool:
        """Is this warm start applicable to ``problem`` at all?

        Structural, not numerical. A working set names rows by index, so it transfers exactly
        when the rows it names are the same rows -- which along a frontier they are, because
        only ``c`` changes. Anything else and the indices mean something different, which is
        not a poor start but a wrong one.

        Args:
            problem: the instance about to be solved.

        Returns:
            ``True`` when the shapes agree.
        """
        return (
            self.z.size == problem.num_variables
            and self.working_set.num_inequalities == problem.num_inequalities
            and self.working_set.num_equalities == problem.num_equalities
            and self.working_set.cone == problem.cone
        )

    def point(self, problem: SOCP) -> Vector | None:
        """The previous solution, if it is the right length for this problem.

        Feasibility is *not* checked here, and deliberately: :func:`cosa.solve` checks it,
        refuses what it cannot verify, and falls back to construction. Checking twice would
        mean two definitions of feasible.

        Args:
            problem: the instance.

        Returns:
            The point, or ``None`` when it cannot apply.
        """
        return self.z if self.z.size == problem.num_variables else None

    def set_for(self, problem: SOCP) -> WorkingSet | None:
        """The previous working set, if its rows are this problem's rows.

        Args:
            problem: the instance.

        Returns:
            The working set, or ``None`` when it does not fit.
        """
        return self.working_set if self.fits(problem) else None

    def duals_for(self, problem: SOCP) -> Multipliers | None:
        """The previous multipliers, if they are the right shape to seed the curvature.

        Only the conic block matters for what these are used for -- #23's
        :func:`cosa.active_set.multipliers.lagrangian_curvature` reads ``w`` and nothing
        else -- but the whole object is checked, because handing on a half-valid one is how
        a shape error becomes a wrong answer three modules later.

        Args:
            problem: the instance.

        Returns:
            The multipliers, or ``None``.
        """
        if self.multipliers is None:
            return None
        found = self.multipliers
        matches = (
            found.y.size == problem.num_inequalities
            and found.nu.size == problem.num_equalities
            and found.w.size == problem.cone.dim
        )
        return found if matches else None

    def __str__(self) -> str:
        """What is being carried, for a log line or a frontier trace."""
        rows = len(self.working_set.inequalities) + len(self.working_set.active_cones)
        parts = [f"{rows} active row(s)"]
        if self.multipliers is not None:
            parts.append("multipliers")
        if self.cache is not None:
            parts.append(str(self.cache))
        return "warm start: " + ", ".join(parts)


def from_solution(solution: Solution, *, cache: Reuse | None = None) -> WarmStart:
    """Build a warm start from a finished solve.

    Takes all four of §9 Phase VI's items, including the ones a caller might think are not
    worth carrying. Multipliers from a solve that did *not* reach optimality are still worth
    having: they are the best estimate available and #23's curvature only needs them to be
    approximately right, since it re-derives them every iteration anyway.

    Args:
        solution: the previous solve.
        cache: the factorization cache that solve used, if the caller kept one. Not read off
            the solution because a :class:`cosa.solver.cosa.Solution` does not carry it --
            the cache is a cost record, not part of the answer, and putting it in the answer
            would make two solves of the same problem compare unequal.

    Returns:
        The warm start.
    """
    return WarmStart(
        z=solution.z,
        working_set=solution.working_set,
        multipliers=solution.multipliers,
        cache=cache,
    )


def seed(problem: SOCP, warm: WarmStart | None) -> tuple[Vector | None, WorkingSet | None, Multipliers, Reuse | None]:
    """Unpack a warm start into what :func:`cosa.solve` needs, discarding what does not fit.

    One place where "a warm start is a hint, never a commitment" is enforced, so that the
    loop never has to ask whether a part of one applies.

    Args:
        problem: the instance about to be solved.
        warm: the warm start, or ``None``.

    Returns:
        The starting point or ``None``, the starting working set or ``None``, the multipliers
        to seed the curvature with (zeros when there are none to carry), and the factorization
        cache or ``None``.
    """
    if warm is None:
        return None, None, _zero_multipliers(problem), None
    return (
        warm.point(problem),
        warm.set_for(problem),
        warm.duals_for(problem) or _zero_multipliers(problem),
        warm.cache,
    )


def _zero_multipliers(problem: SOCP) -> Multipliers:
    """Multipliers of the right shape and no content.

    Args:
        problem: the instance, for its shape.

    Returns:
        All-zero multipliers, which make #23's curvature term vanish and so make the first
        direction of a cold solve exactly the one Wave 6 computed.
    """
    return Multipliers(
        y=np.zeros(problem.num_inequalities),
        nu=np.zeros(problem.num_equalities),
        w=np.zeros(problem.cone.dim),
    )
