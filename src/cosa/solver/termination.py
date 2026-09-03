"""The five conic KKT residuals, and the stopping criterion they constitute.

§6 (``paper.tex:500``) ends by naming what the final implementation will measure
(``paper.tex:566``): primal feasibility, dual feasibility, stationarity, linear
complementarity, SOC complementarity. *"These residuals define the primary termination
criterion."* This module is those five and that criterion.

**Why five and not one.** They could be summed into a single number, and the sum would be a
worse thing to stop on: each residual fails for a different reason and names a different
repair. A primal violation says the ratio test let an iterate out of the feasible set; a
dual violation names a constraint to drop; a stationarity residual is arithmetic gone
wrong and names nothing; a complementarity violation says the working set disagrees with
the multipliers about which constraints are active. Success Criterion 2
(``paper.tex:1321``) asks that the termination criterion be "based on mathematically
meaningful conic KKT residuals", and meaningful is what a sum is not.

**SOC complementarity is the one that is not obvious.** For a linear inequality,
complementarity is ``y_i * (a_i.T @ x - b_i) = 0`` -- a product of two scalars, zero when
either is. For the cone it is ``w.T @ s = 0`` with ``s`` the conic slack, and that is a
stronger statement than it looks: with ``w in Q`` and ``s in Q``, both self-dual, the inner
product is *non-negative*, so demanding it vanish forces the pair onto complementary faces.
At a nonzero boundary point that pins ``w`` to a ray -- the one #13 derives -- and at the
apex it says nothing at all, every ``w in Q`` being complementary to ``s = 0``. Which is
exactly why the apex needs #24's normal-cone branch and not this.

**Level 3 is the whole of it.** §14.3 (``paper.tex:1033``) says the final solution must
satisfy the full conic KKT conditions, and that *"because the problem is convex,
satisfaction of the KKT conditions provides a certificate of global optimality under the
usual constraint qualification assumptions"*. So :meth:`Residuals.is_optimal` is not a
heuristic stopping rule that happens to work -- it is a certificate, and the reason this
module exists rather than a tangent-residual threshold.

The one thing it does not do is decide *what to change* when a residual is nonzero. That
is the active-set logic of §7 and the conic working-set logic of #23; this module reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from cosa.active_set.multipliers import Multipliers
from cosa.geometry.soc import slack
from cosa.problem.socp import _vector

if TYPE_CHECKING:
    from cosa import Vector
    from cosa.problem.socp import SOCP

__all__ = [
    "TOLERANCE",
    "Residuals",
    "residuals",
]

TOLERANCE: Final = 1e-6
"""The default tolerance every residual is measured against.

One number for all five, because they are all relative and all dimensionless once scaled
by the data they come from -- see :func:`residuals`. A criterion with five separate knobs
would need five justifications, and §6 offers one.

**Set by what the algorithm can reach, not by what would be nice.** It began at ``1e-8``,
which is right where exact arithmetic is available -- the hand-solved instances of #9 reach
a residual of *zero*, and the polyhedral loop of #14 reaches ``1e-17``. But the prototype's
conic path converges by retraction, and a backtracking line search locates the cone's
boundary to about the square root of machine precision. Its stationarity residual therefore
floors out near ``1e-8``, and whether a given instance lands just above or just below is
decided by the BLAS: the ``sector`` family reaches ``9e-9`` on one numpy build and
``1.08e-8`` on another, which made the difference between ``optimal`` and ``stalled``.

A tolerance a method can only just reach is a tolerance that reports its own rounding as a
failure. ``1e-6`` is an order of magnitude above the floor, which is far tighter than
§16.3's cross-solver agreement needs and leaves the residual meaning what it says. Where
exactness *is* available nothing is given up -- those instances still come out at zero, and
the tests that care assert it directly rather than through this default.
"""


@dataclass(frozen=True)
class Residuals:
    """The five residuals of §6, each non-negative and each relative to its own data.

    Relative, so that a portfolio whose returns are in percent and one whose notional is in
    millions are held to the same standard. The scaling is by the largest magnitude in the
    quantities the residual is built from, floored at one, which is the same mixed
    absolute/relative convention the geometry module uses.

    Attributes:
        primal: how far the iterate is outside the feasible set -- the largest violation of
            ``A @ z <= b``, ``E @ z = d`` or ``G @ z + h in K``. §14.1's Level 1.
        dual: how far the multipliers are outside their cones -- the largest violation of
            ``y >= 0`` or ``w in K``.
        stationarity: the size of ``c + A.T @ y + E.T @ nu - G.T @ w``. §14.2's Level 2.
        linear_complementarity: the largest ``|y_i * (a_i.T @ z - b_i)|``.
        cone_complementarity: the largest ``|w_j.T @ s_j|`` over the cone's factors.
    """

    primal: float
    dual: float
    stationarity: float
    linear_complementarity: float
    cone_complementarity: float

    @property
    def largest(self) -> float:
        """The worst of the five -- the single number a stopping test compares."""
        return max(
            self.primal,
            self.dual,
            self.stationarity,
            self.linear_complementarity,
            self.cone_complementarity,
        )

    @property
    def complementarity(self) -> float:
        """The worse of the two complementarity residuals, for a report that wants four."""
        return max(self.linear_complementarity, self.cone_complementarity)

    def is_optimal(self, *, tolerance: float = TOLERANCE) -> bool:
        """Does this iterate satisfy §14.3's Level 3, and so certify global optimality?

        The problem is convex, so all five residuals within tolerance is a certificate
        rather than a guess -- under the constraint qualification §14.3 names.

        Args:
            tolerance: the tolerance every residual must be within.

        Returns:
            ``True`` if the largest residual is within tolerance.
        """
        return self.largest <= tolerance

    def worst(self) -> str:
        """Which residual is largest, named -- the first thing a diagnosis wants.

        Returns:
            The name of the largest residual, or ``"none"`` when every one is exactly zero.
        """
        named = {
            "primal feasibility": self.primal,
            "dual feasibility": self.dual,
            "stationarity": self.stationarity,
            "linear complementarity": self.linear_complementarity,
            "SOC complementarity": self.cone_complementarity,
        }
        largest = max(named.values())
        return "none" if largest == 0.0 else max(named, key=lambda name: named[name])

    def __str__(self) -> str:
        """All five on one line, for a log entry or a termination report."""
        return (
            f"primal={self.primal:.3g} dual={self.dual:.3g} stat={self.stationarity:.3g} "
            f"comp_lin={self.linear_complementarity:.3g} comp_soc={self.cone_complementarity:.3g}"
        )


def _relative(value: float, scale: float) -> float:
    """Divide a residual by the scale of the data it came from, floored at one.

    Args:
        value: the absolute residual.
        scale: the magnitude of the data.

    Returns:
        The relative residual, non-negative.
    """
    return max(0.0, value) / max(1.0, scale)


def residuals(problem: SOCP, z: Vector, multipliers: Multipliers) -> Residuals:
    """Compute all five residuals of §6 at a primal-dual point.

    Args:
        problem: the instance.
        z: the primal iterate.
        multipliers: the dual iterate, as #13 recovers it.

    Returns:
        The five residuals.

    Raises:
        ProblemError: if any block's length disagrees with the problem's shape.
    """
    # `nu` is validated by the stationarity residual below, which is the only place it is
    # used: complementarity does not apply to an equality, whose slack is zero by definition.
    point = _vector("z", z, size=problem.num_variables)
    y = _vector("y", multipliers.y, size=problem.num_inequalities)
    w = _vector("w", multipliers.w, size=problem.cone.dim)

    conic_slack = problem.cone_slack(point)
    linear_slack = problem.b - problem.A @ point
    equality_error = problem.E @ point - problem.d

    primal = max(
        float((-linear_slack).max(initial=0.0)) / max(1.0, float(np.abs(problem.b).max(initial=0.0))),
        float(np.abs(equality_error).max(initial=0.0)) / max(1.0, float(np.abs(problem.d).max(initial=0.0))),
        max(
            (
                _relative(-slack(block), float(np.abs(block).max(initial=0.0)))
                for block in problem.cone.blocks(conic_slack)
            ),
            default=0.0,
        ),
    )

    dual = max(
        _relative(float((-y).max(initial=0.0)), float(np.abs(y).max(initial=0.0))),
        max(
            (_relative(-slack(block), float(np.abs(block).max(initial=0.0))) for block in problem.cone.blocks(w)),
            default=0.0,
        ),
    )

    return Residuals(
        primal=primal,
        dual=dual,
        stationarity=multipliers.stationarity_error(problem),
        linear_complementarity=_relative(
            float(np.abs(y * linear_slack).max(initial=0.0)),
            float(np.abs(y).max(initial=0.0)) * float(np.abs(linear_slack).max(initial=0.0)),
        ),
        cone_complementarity=max(
            (
                _relative(
                    abs(float(dual_block @ slack_block)),
                    float(np.abs(dual_block).max(initial=0.0)) * float(np.abs(slack_block).max(initial=0.0)),
                )
                for dual_block, slack_block in zip(
                    problem.cone.blocks(w), problem.cone.blocks(conic_slack), strict=True
                )
            ),
            default=0.0,
        ),
    )
