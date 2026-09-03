"""Diagonal scaling of an instance, and the one constraint the cone puts on it.

§13.3 (``paper.tex:996``) lists five things to scale -- portfolio variables, covariance
matrices, linear constraints, expected returns, SOC variables -- and gives the reason in
one sentence: *"Good scaling is particularly important because the SOC couples ``t`` and
``Lx``"* (``paper.tex:1008``). That sentence is also the whole design constraint, so it is
worth unpacking before the code.

**The cone forces one scale per block, and that is not a convenience.** A diagonal scaling
of the variables and a row scaling of the constraints are free to be arbitrary: any
positive diagonal preserves ``A @ z <= b``. The conic block is not. ``(t, y) in Q`` says
``||y|| <= t``, and scaling ``t`` by ``2`` while leaving ``y`` alone changes *which points
satisfy it* -- it is a different constraint, not the same one better conditioned. What may
be done is scale the whole block by a single positive ``kappa``, because ``Q`` is a cone:
``kappa * s in Q`` exactly when ``s in Q``. So :attr:`Scaling.cones` carries one number
per factor, applied to head and tail together, and there is no way to make it a vector
without breaking the problem. §13.3's sentence is a statement about this.

Two consequences fall out, both useful:

* the unit vector ``u = s_1 / ||s_1||`` is **scale-invariant**. Scaling the block by
  ``kappa`` scales the tail by ``kappa`` and leaves its direction alone, so eq. (3)'s
  tangent geometry is untouched by anything this module does. A scaling that changed ``u``
  would change the working set, and then "the same problem, better conditioned" would be
  false.
* the dual variable's cone membership survives. ``w`` unscales by a *positive* multiple of
  each block, so ``w in Q`` before and after are the same statement -- which is what makes
  :meth:`Scaling.unscale_multipliers` safe to apply to multipliers whose signs have already
  been tested.

**The algorithm is Ruiz equilibration, with that constraint imposed.** A few rounds of
alternately dividing each column by the square root of its largest entry and each row by
the square root of its own, with the conic rows sharing one factor per block. Then the
objective is normalized. It is a standard method and deliberately not a clever one: #26 is
where factorization strategies get compared, and a scaling that is hard to reason about
would confound it.

**What scaling turns out *not* to fix, measured.** Issue #28's "done when" expects
conditioning to improve on the ill-conditioned-covariance instances of §12.4. On this
implementation it does not, and the reason is worth recording because it is a fact about
the algorithm rather than about the scaling.

Take :func:`cosa.experiments.portfolio.ill_conditioned`, whose covariance has a condition
number of ``1e10`` by construction. At its optimum the assembled KKT matrix has a condition
number of about **10**, and ``cond(W @ W.T)`` is about **18**. The covariance's spectrum
never reaches the saddle-point system, because the tangent representation compresses ``L``
into a *single row*, ``g_0 - u.T @ L``. One row has no spectrum to be ill-conditioned. The
long-only bounds are ``±1`` and the budget row is ones, so every column maximum of the
stacked constraint matrix is exactly ``1`` and :func:`equilibrate` returns the identity --
correctly, there being nothing to equilibrate.

That is a result rather than a disappointment. §17 (``paper.tex:1345``) says a valuable
outcome would be "a clear characterization of when conic active-set methods work well, why
they work well, and how their geometry differs fundamentally from polyhedral active-set
methods", and insensitivity of the KKT system to covariance conditioning is exactly such a
characterization: an interior-point method forms ``L.T @ L`` or works with all of ``L`` at
every iteration and inherits its conditioning, while a tangent-representation active-set
method touches ``L`` only through one direction. Whether it *survives* an ill-conditioned
covariance elsewhere -- in the factorization of #10, in the step interval of #18 -- is a
separate question this does not answer.

Where scaling does earn its keep is unit mismatch, which is what §13.3's five targets are
actually about: on :func:`cosa.experiments.portfolio.badly_scaled`, whose weights are in
basis points and whose risk variable is in millions, the KKT condition number falls from
about ``2e14`` to about ``10``. Thirteen orders of magnitude, and the instance is an
ordinary modelling error rather than a contrived one.

That claim is about the conditioning of the matrix COSA factorizes, and deliberately not
about anyone else's answer. Pre-scaling an instance before handing it to the *reference*
solver makes its answer slightly **worse**, because Clarabel equilibrates internally and a
second change of units only adds rounding -- measured in
``test_pre_scaling_does_not_help_the_reference_solver``. This module is conditioning for
the dense factorization of #12, where nothing else is doing the job.

**How the five named targets map onto three fields.** The plan lists five things and this
module has :attr:`Scaling.variables`, :attr:`Scaling.inequalities`,
:attr:`Scaling.equalities`, :attr:`Scaling.cones` and :attr:`Scaling.objective` -- so the
correspondence is worth writing down, and :meth:`Scaling.describe` prints it:

* *portfolio variables* -- the leading entries of :attr:`Scaling.variables`;
* *SOC variables* -- the ``t`` entry of :attr:`Scaling.variables`, together with
  :attr:`Scaling.cones`;
* *covariance matrices* -- reached indirectly and exactly. ``L`` is the tail rows of ``G``,
  so scaling those rows by ``kappa`` and their columns by ``D`` replaces ``Sigma`` by
  ``kappa^2 * D @ Sigma @ D``. There is no separate covariance knob because there does not
  need to be one;
* *linear constraints* -- :attr:`Scaling.inequalities` and :attr:`Scaling.equalities`;
* *expected returns* -- :attr:`Scaling.objective`, which scales all of ``c`` and so both
  ``mu`` and ``lam``. Scaling ``mu`` alone would change the problem; scaling ``c`` does
  not change its argmin.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

import numpy as np

from cosa.active_set.multipliers import Multipliers
from cosa.linear_algebra.kkt import RHO, assemble
from cosa.problem.socp import ProblemError, _vector

if TYPE_CHECKING:
    from cosa import Vector
    from cosa.active_set.working_set import WorkingSet
    from cosa.problem.socp import SOCP

__all__ = [
    "ROUNDS",
    "Scaling",
    "equilibrate",
    "identity",
    "kkt_condition",
]

ROUNDS: Final = 10
"""How many equilibration sweeps :func:`equilibrate` performs.

Ruiz equilibration converges geometrically and ten rounds is far past the point of
diminishing returns on problems of this shape; it is cheap enough -- two norms per sweep --
that stopping early buys nothing worth the extra parameter.
"""


def _positive(name: str, values: Vector, size: int) -> Vector:
    """Coerce a scale vector and check that every entry is positive and finite.

    Args:
        name: the field's name, for the error message.
        values: the scales.
        size: how many are required.

    Returns:
        The scales as a float vector.

    Raises:
        ProblemError: if any entry is not positive, or the length is wrong.
    """
    scales = _vector(name, values, size=size)
    if scales.size and float(scales.min()) <= 0.0:
        raise ProblemError(name, f"every scale must be positive, found {float(scales.min())}")
    return scales


@dataclass(frozen=True, eq=False)
class Scaling:
    """A diagonal rescaling of an instance, and the maps back from its solution.

    Every field is a vector of positive numbers, so the scaling is invertible and the
    original problem is recoverable exactly -- which is the only reason it is safe to solve
    a scaled instance at all.

    Attributes:
        variables: ``D``, one positive scale per variable. The scaled variable is
            ``z_hat = z / D``, so ``z = D * z_hat``.
        inequalities: ``R``, one positive scale per row of ``A``.
        equalities: ``S``, one positive scale per row of ``E``.
        cones: ``kappa``, **one** positive scale per factor of the cone product, applied to
            the whole block. See the module docstring for why it cannot be per row.
        objective: ``sigma``, a single positive scale for ``c``.
    """

    variables: Vector
    inequalities: Vector
    equalities: Vector
    cones: Vector
    objective: float

    def __post_init__(self) -> None:
        """Coerce every field and reject a non-positive scale.

        Raises:
            ProblemError: if any scale is not positive and finite.
        """
        set_field = object.__setattr__
        set_field(self, "variables", _positive("variables", self.variables, self.variables_size))
        set_field(self, "inequalities", _positive("inequalities", self.inequalities, self.inequalities_size))
        set_field(self, "equalities", _positive("equalities", self.equalities, self.equalities_size))
        set_field(self, "cones", _positive("cones", self.cones, self.cones_size))
        set_field(self, "objective", float(self.objective))
        if not np.isfinite(self.objective) or self.objective <= 0.0:
            raise ProblemError("objective", f"the objective scale must be positive, found {self.objective}")

    @property
    def variables_size(self) -> int:
        """How many variables this scaling is for."""
        return int(np.asarray(self.variables).size)

    @property
    def inequalities_size(self) -> int:
        """How many inequality rows this scaling is for."""
        return int(np.asarray(self.inequalities).size)

    @property
    def equalities_size(self) -> int:
        """How many equality rows this scaling is for."""
        return int(np.asarray(self.equalities).size)

    @property
    def cones_size(self) -> int:
        """How many cone factors this scaling is for."""
        return int(np.asarray(self.cones).size)

    def _block_scales(self, problem: SOCP) -> Vector:
        """Expand the per-factor cone scales to one entry per conic row.

        Args:
            problem: the instance, for its cone product's block widths.

        Returns:
            A vector of length ``cone.dim``.
        """
        widths = [cone.dim for cone in problem.cone.cones]
        return np.repeat(self.cones, widths) if widths else np.zeros(0)

    def apply(self, problem: SOCP) -> SOCP:
        """Rescale an instance, producing an equivalent one.

        Equivalent in the strong sense: the feasible sets correspond under
        ``z = D * z_hat``, the objectives differ by the positive factor ``sigma``, and the
        multipliers correspond under :meth:`unscale_multipliers`. So a solver may work
        entirely in the scaled problem and nothing is lost.

        Args:
            problem: the instance to rescale.

        Returns:
            The scaled instance.

        Raises:
            ProblemError: if the scaling's shape does not match the problem's.
        """
        self._require_shape(problem)
        blocks = self._block_scales(problem)
        return replace(
            problem,
            c=self.objective * self.variables * problem.c,
            A=self.inequalities[:, None] * problem.A * self.variables,
            b=self.inequalities * problem.b,
            E=self.equalities[:, None] * problem.E * self.variables,
            d=self.equalities * problem.d,
            G=blocks[:, None] * problem.G * self.variables,
            h=blocks * problem.h,
        )

    def scale_point(self, z: Vector) -> Vector:
        """Map a point of the original problem into the scaled one: ``z / D``.

        Args:
            z: a point in the original variables.

        Returns:
            The corresponding point in the scaled variables.
        """
        return _vector("z", z, size=self.variables_size) / self.variables

    def unscale_point(self, z: Vector) -> Vector:
        """Map a point of the scaled problem back: ``D * z_hat``.

        Args:
            z: a point in the scaled variables.

        Returns:
            The corresponding point in the original variables.
        """
        return self.variables * _vector("z", z, size=self.variables_size)

    def unscale_multipliers(self, problem: SOCP, multipliers: Multipliers) -> Multipliers:
        """Map the scaled problem's multipliers back to the original's.

        ``y = R * y_hat / sigma``, ``nu = S * nu_hat / sigma``, and each conic block
        ``w_block = kappa * w_hat_block / sigma``. Derived by substituting the scaled blocks
        into the scaled stationarity equation and dividing out ``D``; every factor is
        positive, so both ``y >= 0`` and ``w in Q`` are preserved in both directions.

        Args:
            problem: the *original* instance, for its cone product's block widths.
            multipliers: the multipliers of the scaled problem.

        Returns:
            The original problem's multipliers.

        Raises:
            ProblemError: if the scaling's shape does not match the problem's.
        """
        self._require_shape(problem)
        return Multipliers(
            y=self.inequalities * multipliers.y / self.objective,
            nu=self.equalities * multipliers.nu / self.objective,
            w=self._block_scales(problem) * multipliers.w / self.objective,
        )

    def describe(self) -> str:
        """The five targets of §13.3 and the range of scales applied to each.

        Issue #28's "done when" begins "all five targets are scaled", which is a claim
        about coverage rather than about numbers -- so this reports coverage, naming each
        target the plan names and the spread of factors it received.

        Returns:
            A five-line description, with no trailing newline.
        """
        spread = _spread
        assets = self.variables[:-1] if self.variables_size > 1 else self.variables
        return "\n".join(
            [
                "scaling, over §13.3's five targets:",
                spread("portfolio variables", assets),
                spread("covariance matrices (via the cone's tail rows)", self.cones),
                spread("linear constraints", np.concatenate([self.inequalities, self.equalities])),
                spread("expected returns (the whole objective)", np.array([self.objective])),
                spread("SOC variables (t, and each cone block)", np.append(self.cones, self.variables[-1:])),
            ]
        )

    def _require_shape(self, problem: SOCP) -> None:
        """Reject a problem this scaling was not built for.

        Args:
            problem: the instance to check.

        Raises:
            ProblemError: if any block count disagrees.
        """
        found = (self.variables_size, self.inequalities_size, self.equalities_size, self.cones_size)
        wanted = (problem.num_variables, problem.num_inequalities, problem.num_equalities, len(problem.cone))
        if found != wanted:
            raise ProblemError(
                "scaling",
                f"built for {found} (variables, inequalities, equalities, cones) but applied to a problem of {wanted}",
            )


def identity(problem: SOCP) -> Scaling:
    """The scaling that changes nothing, for a baseline to measure against.

    Args:
        problem: the instance whose shape to match.

    Returns:
        A scaling of all ones.
    """
    return Scaling(
        variables=np.ones(problem.num_variables),
        inequalities=np.ones(problem.num_inequalities),
        equalities=np.ones(problem.num_equalities),
        cones=np.ones(len(problem.cone)),
        objective=1.0,
    )


def equilibrate(problem: SOCP, *, rounds: int = ROUNDS) -> Scaling:
    """Compute a Ruiz equilibration of the instance, respecting the cone's block constraint.

    Each sweep divides every column of the stacked constraint matrix by the square root of
    its largest magnitude and every row by the square root of its own, driving both towards
    one. The conic rows are treated as one row per *block* for this purpose: their scale is
    taken from the largest entry anywhere in the block, so head and tail always receive the
    same factor.

    A column or block that is entirely zero is left alone rather than divided by zero -- a
    variable that appears in no constraint, or a cone whose rows are all zero, has no
    conditioning to improve.

    Args:
        problem: the instance to scale.
        rounds: how many sweeps to perform.

    Returns:
        The scaling. Apply it with :meth:`Scaling.apply`.

    Raises:
        ProblemError: if ``rounds`` is negative.
    """
    if rounds < 0:
        raise ProblemError("rounds", f"expected a non-negative sweep count, found {rounds}")

    variables = np.ones(problem.num_variables)
    inequalities = np.ones(problem.num_inequalities)
    equalities = np.ones(problem.num_equalities)
    cones = np.ones(len(problem.cone))
    widths = [cone.dim for cone in problem.cone.cones]

    for _ in range(rounds):
        scaled = Scaling(
            variables=variables,
            inequalities=inequalities,
            equalities=equalities,
            cones=cones,
            objective=1.0,
        ).apply(problem)
        stacked = np.vstack([scaled.A, scaled.E, scaled.G])

        columns = np.abs(stacked).max(axis=0, initial=0.0)
        variables = variables / np.sqrt(np.where(columns > 0.0, columns, 1.0))

        inequalities = inequalities / np.sqrt(_row_scale(scaled.A))
        equalities = equalities / np.sqrt(_row_scale(scaled.E))
        if widths:
            # One scale per block, not per row: the whole point of the cone constraint.
            blocks = np.abs(scaled.G).max(axis=1, initial=0.0)
            largest = np.array([block.max(initial=0.0) for block in np.split(blocks, np.cumsum(widths)[:-1])])
            cones = cones / np.sqrt(np.where(largest > 0.0, largest, 1.0))

    objective = np.abs(variables * problem.c).max(initial=0.0)
    return Scaling(
        variables=variables,
        inequalities=inequalities,
        equalities=equalities,
        cones=cones,
        objective=1.0 / objective if objective > 0.0 else 1.0,
    )


def _spread(name: str, values: Vector) -> str:
    """Render one target's name and the range of scales it received.

    Args:
        name: the target's name, as §13.3 lists it.
        values: the scales applied to it.

    Returns:
        One line, reporting "none in this instance" for a target the problem does not have.
    """
    entries = np.asarray(values)
    if not entries.size:
        return f"  {name}: none in this instance"
    return f"  {name}: {entries.size} scale(s) in [{float(entries.min()):.3g}, {float(entries.max()):.3g}]"


def _row_scale(block: Vector) -> Vector:
    """The largest magnitude in each row, with zero rows reported as one.

    Args:
        block: a matrix.

    Returns:
        One positive number per row.
    """
    largest = np.abs(block).max(axis=1, initial=0.0)
    return np.where(largest > 0.0, largest, 1.0)


def kkt_condition(
    problem: SOCP,
    working_set: WorkingSet,
    z: Vector,
    *,
    rho: float = RHO,
) -> float:
    """The condition number of the assembled KKT matrix at a point.

    The measurement #28's "done when" is stated in terms of -- "conditioning measurably
    improves" -- so it is a function here rather than something a test computes for itself.
    It is the conditioning of the matrix that actually gets factorized, which is the number
    that matters: the covariance's own condition number is an input, and improving it is
    only interesting if the saddle-point system inherits the improvement.

    Args:
        problem: the instance.
        working_set: what is believed active, which decides ``W_k``.
        z: the point, needed for the tangent rows.
        rho: the ``rho`` of ``H = rho*I``.

    Returns:
        The 2-norm condition number, infinite for a singular matrix.
    """
    return float(np.linalg.cond(assemble(problem, working_set, z, rho=rho).matrix))
