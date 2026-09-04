"""Making the loop terminate: Bland's rule when it stops making progress, and a merit guard.

§17.2 (``paper.tex:1145``) says "as in classical active-set methods, the algorithm may cycle
among working sets" and names four remedies. #29 implements them; this module is three of
the four, and the fourth -- multiplier tolerances -- is
:data:`cosa.active_set.updates.MULTIPLIER_TOLERANCE`, which has been in place since #11 and
is left where it is rather than moved here to make a list look complete.

**Why cycling is possible at all.** An active-set method changes its working set on
information that is exactly true only at a stationary point of the current one. A degenerate
vertex has more constraints active than the dimension needs, so several working sets describe
it; the loop can drop a row, take a step of length zero, add a different row, and arrive back
where it started with nothing changed but the iteration counter. Nothing in §7's rules
forbids that, because §7's rules are about which multiplier is most wrong, and at a
degenerate vertex "most wrong" can rotate.

**The remedy is to stop optimizing the choice.** §7.2's rule picks the *most strongly
violating* multiplier, which is a good heuristic and is what makes the method fast. Bland's
rule picks the *lowest-indexed* violating one instead, which is a bad heuristic and is
provably finite: it imposes a total order on working sets that the iteration must descend, so
no set can recur. :func:`lexicographic_candidate` is that rule, and :class:`Guard` is what
decides when to switch to it.

**Switching rather than always using it** is the whole design. Bland's rule costs iterations
on ordinary instances -- it ignores the magnitude of the violation, which is the information
that makes the fast rule fast -- so it is held in reserve and armed only once a working set
has been seen more than :data:`REVISITS` times. That is the standard compromise and it is
also the honest one: a solver that never cycles because it is always slow has not solved the
problem, it has paid for it.

**The merit safeguard is separate and cheaper.** §17.2's fourth remedy is a merit function,
and :meth:`Guard.accepts` is it: an accepted iterate may not increase the objective by more
than rounding. That does not prevent cycling among working sets at a single point -- the
objective is constant there, which is exactly the difficulty -- but it does prevent the other
failure it is confused with, a step that makes things worse and is taken anyway because the
direction said it would not. The two are different and both are cheap.

**What this module deliberately does not do.** It does not detect cycling by comparing
iterates. Two visits to the same *point* are ordinary -- a zero-length step is how a
constraint gets added -- and only a repeated *working set* is evidence of anything. Keying
on the working set alone is why :meth:`Guard.saw` can be a dictionary lookup rather than a
geometric test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import numpy as np

from cosa.problem.socp import SIGN_CONVENTION

if TYPE_CHECKING:
    from cosa import Vector, WorkingSet

__all__ = [
    "MERIT_SLACK",
    "REVISITS",
    "Guard",
    "lexicographic_candidate",
    "objective_of",
]

REVISITS: Final = 3
"""How often one working set may recur before Bland's rule is armed.

Not one. A working set legitimately recurs: the loop drops a row, takes a step that blocks
on it again, and is back where it was having genuinely moved. Two visits are evidence of
nothing, three are evidence of little, and by the fourth the loop is doing something it
cannot explain -- which is the point at which trading speed for a termination guarantee is
the right trade.
"""

MERIT_SLACK: Final = 1e-9
"""How much an accepted iterate may worsen the objective before the guard objects.

Relative to the objective's own scale. Not zero, because the retraction genuinely does
increase the objective by ``lam * delta_t`` before the direction's first-order decrease pays
for it, and a backtracking search that has already accepted a step has already established
the net is favourable. What this catches is a net *increase*, which no accepted step should
ever produce.
"""


def lexicographic_candidate(working_set: WorkingSet, y: Vector, *, tolerance: float) -> int | None:
    """Bland's rule: the *lowest-indexed* violating multiplier, not the most violating.

    §17.2's "lexicographic selection" and "anti-cycling rule" are the same rule under two
    names, and this is it. The contrast with
    :func:`cosa.active_set.updates.removal_candidate` is the entire content: that one asks
    *how badly* each multiplier violates its sign and picks the worst, this one asks only
    *whether* each violates and picks the first. Ignoring the magnitude is what makes it
    finite -- the rule depends on nothing that can rotate at a degenerate vertex, so the
    sequence of working sets is strictly ordered and cannot return.

    The required sign is read from :data:`cosa.SIGN_CONVENTION` rather than written out, for
    the reason #9 exists: two rules that disagree about the direction of the test would drop
    different rows and one of them would be wrong.

    Args:
        working_set: the current set, whose active rows are the candidates.
        y: the inequality multipliers, indexed by the problem's rows.
        tolerance: how large a violation counts as none.

    Returns:
        The lowest-indexed active row whose multiplier violates its required sign, or
        ``None`` when none does.
    """
    for index in working_set.inequalities:
        if -float(y[index]) * SIGN_CONVENTION.inequality > tolerance:
            return index
    return None


@dataclass(eq=False)
class Guard:
    """What the loop has already seen, and what it will do about seeing it again.

    Mutable for the same reason :class:`cosa.linear_algebra.reuse.Reuse` is: it is a record
    of the past, and a solve has exactly one past.

    Attributes:
        visits: how many times each working set has been *returned to*, keyed by the set's
            identity -- its active inequalities and its cone statuses, which together are
            everything the direction subproblem depends on besides the point. Consecutive
            iterations holding one set count once between them: staying put is iterating,
            and only leaving and coming back is evidence of a cycle.
        current: the set the last call reported, so that staying can be told from returning.
        objective: the best objective value accepted so far, or ``None`` before the first.
        armed: whether Bland's rule is in force. Once armed it stays armed: disarming on the
            first sign of progress is how a solver cycles a second time.
    """

    visits: dict[tuple[object, ...], int] = field(default_factory=dict)
    current: tuple[object, ...] | None = None
    objective: float | None = None
    armed: bool = False

    def saw(self, working_set: WorkingSet) -> int:
        """Record a visit to this working set and arm Bland's rule if it is time.

        A run of iterations that all hold the same set is one visit, not many. That
        distinction is the difference between measuring cycling and measuring slowness, and
        getting it wrong hides both: a solve that spends nine hundred iterations refining a
        point under one working set is converging slowly, and counting that as nine hundred
        revisits would arm an anti-cycling rule against a problem it cannot help with while
        reporting a cycle that is not there.

        Args:
            working_set: the set the loop is about to compute a direction for.

        Returns:
            How many times this set has been returned to, including now.
        """
        key = (working_set.inequalities, working_set.cone_status)
        if key == self.current:
            return self.visits[key]
        self.current = key
        count = self.visits.get(key, 0) + 1
        self.visits[key] = count
        if count > REVISITS:
            self.armed = True
        return count

    def accepts(self, value: float) -> bool:
        """§17.2's merit safeguard: may an iterate with this objective be accepted?

        Args:
            value: the objective at the candidate iterate.

        Returns:
            ``True`` unless the objective has risen by more than :data:`MERIT_SLACK`
            relative to the best seen. Recording the value is left to :meth:`accepted`, so
            that asking is free of consequence and a rejected candidate does not poison the
            record.
        """
        if self.objective is None:
            return True
        return value <= self.objective + MERIT_SLACK * max(1.0, abs(self.objective))

    def accepted(self, value: float) -> None:
        """Record an accepted iterate's objective.

        Args:
            value: the objective there.
        """
        self.objective = value if self.objective is None else min(self.objective, value)

    def candidate(self, working_set: WorkingSet, y: Vector, *, tolerance: float) -> int | None:
        """Which inequality to drop, under whichever rule is currently in force.

        The single place the switch happens, so that no caller has to know there is one.

        Args:
            working_set: the current set.
            y: the inequality multipliers.
            tolerance: how large a violation counts as none.

        Returns:
            The row to drop, or ``None`` when every multiplier has its required sign.
        """
        from cosa.active_set.updates import removal_candidate

        if self.armed:
            return lexicographic_candidate(working_set, y, tolerance=tolerance)
        return removal_candidate(working_set, y, tolerance=tolerance)

    def __str__(self) -> str:
        """The distinct sets seen, the worst revisit count, and whether Bland's rule is on."""
        worst = max(self.visits.values(), default=0)
        rule = "Bland" if self.armed else "most-violating"
        return f"guard: {len(self.visits)} working set(s), most-revisited {worst}x, {rule} rule"


def objective_of(gradient: Vector, z: Vector) -> float:
    """The linear objective ``c.T @ z``, for the merit guard.

    A one-line helper rather than an inline expression so that the merit function has a name
    and can be pointed at. §17.2 says "merit-function safeguards" in the plural and for a
    linear objective over a convex feasible set there is only one worth having -- the
    objective itself. A penalty merit function exists to trade feasibility against
    optimality, and this loop never leaves the feasible set, so there is nothing to trade.

    Args:
        gradient: ``c``.
        z: the point.

    Returns:
        ``c.T @ z``.
    """
    return float(np.asarray(gradient) @ np.asarray(z))
