"""The primal SOCP, its cone, and the one fixed sign convention.

The project plan's eq. (7) is the problem COSA solves:

    min  -mu.T @ x + lam * t
    s.t. A @ x <= b,  E @ x = d,  (t, L @ x) in Q

This module represents it, and represents it in the slightly more general shape the
plan's "General SOCPs" section targets, because three decisions have to be settled
here rather than in the issues that consume them.

**The cone is a Cartesian product.** ``K = Q^(n_1) x ... x Q^(n_J)``, even though the
solver starts by handling a single cone. The general form is
``min c.T @ z s.t. A @ z <= b, E @ z = d, G @ z + h in K``, which is what the plan's
generalization section asks for and what makes Success Criterion 7 (modularity)
reachable without reopening the representation. Eq. (7) is the instance of it with
``z = (x, t)``, ``c = (-mu, lam)``, ``G = [[0, 1], [L, 0]]`` and ``h = 0``;
:class:`MeanStdForm` is that instance's data, and it round-trips through
:meth:`MeanStdForm.to_socp` and :meth:`SOCP.as_mean_std`.

**Auxiliary variables are cheap.** Turnover constraints are expressed with auxiliary
variables and linear inequalities, so nothing here may assume that the variable vector
is exactly ``(x, t)``. It is ``z``, of whatever length, and :meth:`SOCP.augment`,
:meth:`SOCP.add_inequalities`, :meth:`SOCP.add_equalities` and :meth:`SOCP.add_cone`
grow an instance without rebuilding it.

**The sign convention is fixed here, once.** The plan defers it deliberately -- *"The
exact sign convention will be fixed in the implementation and must be used consistently
throughout the derivation and code"* -- so this is that choice. With multipliers
``y`` for the inequalities, ``nu`` for the equalities and ``w`` for the cone, the
Lagrangian is

    Lagr(z, y, nu, w) = c.T @ z + y.T @ (A @ z - b) + nu.T @ (E @ z - d)
                        - w.T @ (G @ z + h)

which makes stationarity

    c + A.T @ y + E.T @ nu - G.T @ w = 0

with ``y >= 0`` and ``w in K``. The cone term is *subtracted*: that is what keeps the
dual variable in ``K`` itself rather than in ``-K``, and the plan states the dual
condition as ``w_soc in Q``. The second-order cone is self-dual under the head-first
representation used here, so ``K* = K`` and no separate dual-cone type exists.

Every consumer reads the signs from :data:`SIGN_CONVENTION` or calls
:meth:`SOCP.stationarity_residual`; none of them writes the signs out again. The full
statement, including what it implies for eq. (7) (namely ``w_t = lam``) and where it
differs from the plan's printed stationarity display, is in
``docs/development/sign-convention.md``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from cosa import Matrix, Vector

__all__ = [
    "SIGN_CONVENTION",
    "SOCP",
    "ConeProduct",
    "MeanStdForm",
    "ProblemError",
    "SecondOrderCone",
    "SignConvention",
]


class ProblemError(ValueError):
    """The data does not describe a well-formed problem instance.

    A ``ValueError``, because that is what a caller handing over the wrong array
    shape expects to catch. It is raised for shape disagreement, for a non-finite
    entry, for an empty problem and for a cone dimension that cannot hold a cone --
    never for infeasibility, which is the solver's business and not the
    representation's.
    """

    def __init__(self, block: str, problem: str) -> None:
        """Name the block at fault and say what is wrong with it.

        Args:
            block: the name the offending block goes by in the problem, e.g. ``"A"``.
            problem: what is wrong with it, phrased as expected-versus-found.
        """
        super().__init__(f"{block}: {problem}")


def _vector(block: str, value: Vector, *, size: int | None = None) -> Vector:
    """Coerce a block to a finite one-dimensional float array.

    Args:
        block: the name the block goes by in the problem, used in error messages.
        value: the data to coerce.
        size: the length the instance requires, when it is already determined.

    Returns:
        A contiguous ``float64`` array of shape ``(size,)``.

    Raises:
        ProblemError: if the data is not one-dimensional, has the wrong length, or
            carries a NaN or an infinity.
    """
    array = np.array(value, dtype=np.float64, order="C")
    if array.ndim != 1:
        raise ProblemError(block, f"expected a vector, found an array of shape {array.shape}")
    if size is not None and array.size != size:
        raise ProblemError(block, f"expected {size} entries, found {array.size}")
    if not np.isfinite(array).all():
        raise ProblemError(block, "every entry must be finite, found a NaN or an infinity")
    return array


def _matrix(block: str, value: Matrix, *, rows: int | None = None, columns: int | None = None) -> Matrix:
    """Coerce a block to a finite two-dimensional float array.

    A block with no rows is legitimate: a problem without inequalities carries an
    ``A`` of shape ``(0, n)``, and the arithmetic works out without a special case.

    Args:
        block: the name the block goes by in the problem, used in error messages.
        value: the data to coerce.
        rows: the number of rows the instance requires, when it is determined.
        columns: the number of columns the instance requires, when it is determined.

    Returns:
        A contiguous ``float64`` array of shape ``(rows, columns)``.

    Raises:
        ProblemError: if the data is not two-dimensional, has the wrong shape, or
            carries a NaN or an infinity.
    """
    array = np.array(value, dtype=np.float64, order="C")
    if array.ndim != 2:
        raise ProblemError(block, f"expected a matrix, found an array of shape {array.shape}")
    if rows is not None and array.shape[0] != rows:
        raise ProblemError(block, f"expected {rows} rows, found {array.shape[0]}")
    if columns is not None and array.shape[1] != columns:
        raise ProblemError(block, f"expected {columns} columns, found {array.shape[1]}")
    if not np.isfinite(array).all():
        raise ProblemError(block, "every entry must be finite, found a NaN or an infinity")
    return array


@dataclass(frozen=True)
class SecondOrderCone:
    """One second-order cone, written head first.

    ``Q^dim = {(s_0, s_1) in R x R^(dim-1) : ||s_1||_2 <= s_0}``. The head ``s_0`` is
    entry 0 and the tail ``s_1`` is everything after it -- head first, never head
    last. That ordering is part of the fixed representation: it is the layout of the
    primal slack ``G @ z + h``, of the dual variable ``w``, and of anything either of
    them is split into. The cone is self-dual in this representation, so the same
    class describes where ``w`` lives.

    Attributes:
        dim: the total dimension, ``1 + len(tail)``.
    """

    dim: int

    def __post_init__(self) -> None:
        """Reject a cone too small to carry both a head and a tail.

        Raises:
            ProblemError: if ``dim < 2``. ``Q^1`` is the non-negative ray, which is a
                linear inequality and belongs in ``A @ z <= b``, not in the cone.
        """
        if self.dim < 2:
            raise ProblemError("cone", f"a second-order cone needs a head and a tail, so dim >= 2, found {self.dim}")

    @property
    def tail_dim(self) -> int:
        """The length of the tail, ``dim - 1`` -- the ``m`` of the plan's ``Q^(m+1)``."""
        return self.dim - 1

    def split(self, block: Vector) -> tuple[float, Vector]:
        """Split a vector in this cone's coordinates into its head and its tail.

        Args:
            block: a vector of length :attr:`dim`, in this cone's coordinates.

        Returns:
            The pair ``(head, tail)``, with the head as a scalar.
        """
        entries = _vector("cone block", block, size=self.dim)
        return float(entries[0]), entries[1:]


@dataclass(frozen=True)
class ConeProduct:
    """The Cartesian product ``K = Q^(n_1) x ... x Q^(n_J)``, in that order.

    The solver starts out handling a single cone, and the plan's generalization
    section then tracks "the active geometry of multiple cones". Representing the
    product from the start is what keeps that a matter of extending the working-set
    logic rather than the problem class.

    The empty product is allowed and means ``K = {0}`` over no rows at all: the
    instance is then a linear program, which is a useful degenerate case for testing
    the polyhedral half of the algorithm on its own.

    Attributes:
        cones: the factors, in the order their blocks appear in ``G @ z + h``.
    """

    cones: tuple[SecondOrderCone, ...] = ()

    @classmethod
    def from_dims(cls, *dims: int) -> ConeProduct:
        """Build the product from its factors' dimensions.

        Args:
            *dims: one total dimension per factor, each at least 2.

        Returns:
            The product of the corresponding second-order cones.
        """
        return cls(tuple(SecondOrderCone(dim) for dim in dims))

    @property
    def dim(self) -> int:
        """The dimension of the product, and so the number of rows of ``G``."""
        return sum(cone.dim for cone in self.cones)

    @property
    def slices(self) -> tuple[slice, ...]:
        """One slice per factor, indexing that factor's block of a product vector."""
        bounds = tuple(itertools.accumulate((cone.dim for cone in self.cones), initial=0))
        return tuple(slice(start, stop) for start, stop in itertools.pairwise(bounds))

    def blocks(self, vector: Vector) -> tuple[Vector, ...]:
        """Cut a product vector into one sub-vector per factor.

        Args:
            vector: a vector of length :attr:`dim`, e.g. a slack or a dual variable.

        Returns:
            One view per factor, in factor order.
        """
        entries = _vector("cone vector", vector, size=self.dim)
        return tuple(entries[block] for block in self.slices)

    def split(self, vector: Vector) -> tuple[tuple[float, Vector], ...]:
        """Cut a product vector into one ``(head, tail)`` pair per factor.

        Args:
            vector: a vector of length :attr:`dim`, e.g. a slack or a dual variable.

        Returns:
            One ``(head, tail)`` pair per factor, in factor order.
        """
        return tuple(cone.split(block) for cone, block in zip(self.cones, self.blocks(vector), strict=True))

    def __len__(self) -> int:
        """The number of factors -- one, for the problems the solver starts with."""
        return len(self.cones)


@dataclass(frozen=True)
class SignConvention:
    """The sign each dual block carries in the stationarity equation.

    Read from here rather than written out again, so that a consumer assembling a KKT
    system and a consumer computing a residual cannot drift apart. The single instance
    is :data:`SIGN_CONVENTION`; the reasoning is in the module docstring and in
    ``docs/development/sign-convention.md``.

    Attributes:
        inequality: the sign of ``A.T @ y``, with ``y >= 0`` for ``A @ z <= b``.
        equality: the sign of ``E.T @ nu``, with ``nu`` free.
        cone: the sign of ``G.T @ w``, with ``w in K`` for ``G @ z + h in K``.
    """

    inequality: float
    equality: float
    cone: float


SIGN_CONVENTION: Final = SignConvention(inequality=1.0, equality=1.0, cone=-1.0)
"""The one sign convention, fixed once and consumed everywhere."""


@dataclass(frozen=True, eq=False, kw_only=True)
class SOCP:
    """A second-order cone program, validated on construction.

        min  c.T @ z
        s.t. A @ z <= b
             E @ z = d
             G @ z + h in K

    Every block is required and explicit: a problem with no inequalities carries an
    ``A`` of shape ``(0, n)`` and a ``b`` of shape ``(0,)``. :meth:`unconstrained`
    plus the ``add_*`` methods build an instance up a block at a time instead.

    Arrays are copied into contiguous ``float64`` on construction, so an instance does
    not alias the caller's buffers and no consumer has to defend against an integer
    dtype. Instances are frozen; every method that changes one returns a new,
    re-validated instance. Two instances compare by identity rather than block by
    block, because elementwise array comparison does not answer that question:
    compare the blocks that matter with ``numpy.testing``.

    Attributes:
        c: the objective, one entry per variable. Its length defines ``n``.
        A: the inequality matrix, ``(m_ineq, n)``.
        b: the inequality right-hand side, ``(m_ineq,)``.
        E: the equality matrix, ``(m_eq, n)``.
        d: the equality right-hand side, ``(m_eq,)``.
        G: the conic matrix, ``(cone.dim, n)``.
        h: the conic offset, ``(cone.dim,)``.
        cone: the cone ``K``, as a Cartesian product of second-order cones.
    """

    c: Vector
    A: Matrix
    b: Vector
    E: Matrix
    d: Vector
    G: Matrix
    h: Vector
    cone: ConeProduct

    def __post_init__(self) -> None:
        """Coerce every block and check that the shapes agree with each other.

        Raises:
            ProblemError: if the objective is empty -- an instance with no variables
                is not a problem -- or if any block disagrees in shape or carries a
                non-finite entry.
        """
        set_block = object.__setattr__
        set_block(self, "c", _vector("c", self.c))
        if self.c.size < 1:
            raise ProblemError("c", "an instance needs at least one variable, found an empty objective")
        columns = self.c.size
        set_block(self, "A", _matrix("A", self.A, columns=columns))
        set_block(self, "b", _vector("b", self.b, size=self.A.shape[0]))
        set_block(self, "E", _matrix("E", self.E, columns=columns))
        set_block(self, "d", _vector("d", self.d, size=self.E.shape[0]))
        set_block(self, "G", _matrix("G", self.G, rows=self.cone.dim, columns=columns))
        set_block(self, "h", _vector("h", self.h, size=self.cone.dim))

    @classmethod
    def unconstrained(cls, c: Vector) -> SOCP:
        """The objective on its own: no rows anywhere, and the empty cone product.

        The starting point for building an instance incrementally, which is what the
        generators of auxiliary-variable problems such as turnover want.

        Args:
            c: the objective, one entry per variable.

        Returns:
            The instance ``min c.T @ z`` with an empty feasible-set description.
        """
        objective = _vector("c", c)
        return cls(
            c=objective,
            A=np.zeros((0, objective.size)),
            b=np.zeros(0),
            E=np.zeros((0, objective.size)),
            d=np.zeros(0),
            G=np.zeros((0, objective.size)),
            h=np.zeros(0),
            cone=ConeProduct(),
        )

    @property
    def num_variables(self) -> int:
        """The length ``n`` of the variable vector ``z``."""
        return self.c.size

    @property
    def num_inequalities(self) -> int:
        """The number of rows of ``A``."""
        return self.A.shape[0]

    @property
    def num_equalities(self) -> int:
        """The number of rows of ``E``."""
        return self.E.shape[0]

    def cone_slack(self, z: Vector) -> Vector:
        """The conic slack ``G @ z + h``, the vector the cone membership is about.

        Args:
            z: a point in variable space.

        Returns:
            The slack, laid out head first, one block per factor of :attr:`cone`.
        """
        return self.G @ _vector("z", z, size=self.num_variables) + self.h

    def stationarity_residual(self, y: Vector, nu: Vector, w: Vector) -> Vector:
        """The stationarity residual, in the one fixed sign convention.

        ``c + A.T @ y + E.T @ nu - G.T @ w``, with the signs taken from
        :data:`SIGN_CONVENTION`. This is the definitional home of those signs: the
        residual set and termination criterion, the multiplier computation and the
        conic working-set logic all reduce their stationarity check to this, rather
        than each spelling the signs out.

        The multipliers' own feasibility -- ``y >= 0`` and ``w in K`` -- is not checked
        here. This is the stationarity block alone.

        Args:
            y: the inequality multipliers, ``(m_ineq,)``.
            nu: the equality multipliers, ``(m_eq,)``.
            w: the conic dual variable, ``(cone.dim,)``.

        Returns:
            The residual, ``(n,)``. Zero exactly when the point is stationary for
            these multipliers.
        """
        inequality = _vector("y", y, size=self.num_inequalities)
        equality = _vector("nu", nu, size=self.num_equalities)
        conic = _vector("w", w, size=self.cone.dim)
        return (
            self.c
            + SIGN_CONVENTION.inequality * (self.A.T @ inequality)
            + SIGN_CONVENTION.equality * (self.E.T @ equality)
            + SIGN_CONVENTION.cone * (self.G.T @ conic)
        )

    def augment(self, columns: int, *, c: Vector | None = None) -> SOCP:
        """Append fresh variables, zero in every existing row.

        The auxiliary variables of a turnover constraint arrive this way: augment,
        then add the inequalities that tie the new variables to the old ones.

        Args:
            columns: how many variables to append.
            c: their objective coefficients, or ``None`` for variables that do not
                enter the objective.

        Returns:
            A new instance with ``columns`` more variables.

        Raises:
            ProblemError: if ``columns`` is not positive.
        """
        if columns < 1:
            raise ProblemError("columns", f"augmenting adds at least one variable, found {columns}")
        cost = np.zeros(columns) if c is None else _vector("c", c, size=columns)
        return replace(
            self,
            c=np.concatenate([self.c, cost]),
            A=np.hstack([self.A, np.zeros((self.num_inequalities, columns))]),
            E=np.hstack([self.E, np.zeros((self.num_equalities, columns))]),
            G=np.hstack([self.G, np.zeros((self.cone.dim, columns))]),
        )

    def add_inequalities(self, rows: Matrix, rhs: Vector) -> SOCP:
        """Append rows to ``A @ z <= b``.

        Args:
            rows: the new rows of ``A``, ``(k, n)``.
            rhs: their right-hand sides, ``(k,)``.

        Returns:
            A new instance carrying the additional inequalities.
        """
        return replace(
            self,
            A=np.vstack([self.A, _matrix("new inequality rows", rows, columns=self.num_variables)]),
            b=np.concatenate([self.b, _vector("new inequality right-hand side", rhs)]),
        )

    def add_equalities(self, rows: Matrix, rhs: Vector) -> SOCP:
        """Append rows to ``E @ z = d``.

        Args:
            rows: the new rows of ``E``, ``(k, n)``.
            rhs: their right-hand sides, ``(k,)``.

        Returns:
            A new instance carrying the additional equalities.
        """
        return replace(
            self,
            E=np.vstack([self.E, _matrix("new equality rows", rows, columns=self.num_variables)]),
            d=np.concatenate([self.d, _vector("new equality right-hand side", rhs)]),
        )

    def add_cone(self, cone: SecondOrderCone, rows: Matrix, offset: Vector) -> SOCP:
        """Append a factor to the cone product, with the rows it constrains.

        Args:
            cone: the new factor.
            rows: its rows of ``G``, ``(cone.dim, n)``, head row first.
            offset: its entries of ``h``, ``(cone.dim,)``, head entry first.

        Returns:
            A new instance whose cone is the product with one more factor.
        """
        return replace(
            self,
            G=np.vstack([self.G, _matrix("new cone rows", rows, rows=cone.dim, columns=self.num_variables)]),
            h=np.concatenate([self.h, _vector("new cone offset", offset, size=cone.dim)]),
            cone=ConeProduct((*self.cone.cones, cone)),
        )

    def trivially_infeasible(self) -> str | None:
        """Why the feasible set is empty by inspection, if it visibly is.

        Only rows that constrain no variable are examined, which is the one kind of
        emptiness that costs nothing to see and is always a modelling error rather
        than a hard instance. Everything else -- including conic emptiness, which
        needs the cone predicates -- is the solver's Phase I to discover.

        Returns:
            A one-line reason, or ``None`` if nothing is visibly wrong.
        """
        empty_rows = ~self.A.any(axis=1)
        if bool(np.any(self.b[empty_rows] < 0.0)):
            return "an inequality row constrains no variable and has a negative right-hand side"
        empty_rows = ~self.E.any(axis=1)
        if bool(np.any(self.d[empty_rows] != 0.0)):
            return "an equality row constrains no variable and has a nonzero right-hand side"
        return None

    def as_mean_std(self) -> MeanStdForm:
        """Read the instance back as the data of eq. (7).

        The inverse of :meth:`MeanStdForm.to_socp`, and only defined on instances in
        that shape: one cone, ``t`` as the last variable, ``t`` appearing nowhere but
        the cone's head row, and ``h = 0``.

        Returns:
            The eq. (7) data ``mu``, ``lam``, ``A``, ``b``, ``E``, ``d``, ``L``.

        Raises:
            ProblemError: if the instance is not in the shape of eq. (7).
        """
        if len(self.cone) != 1:
            raise ProblemError("cone", f"eq. (7) has exactly one cone, this instance has {len(self.cone)}")
        last = self.num_variables - 1
        head = np.zeros(self.num_variables)
        head[last] = 1.0
        reasons = (
            (bool(np.array_equal(self.G[0], head)), "the cone's head row must select t, the last variable"),
            (not self.G[1:, last].any(), "t must not appear in the cone's tail rows"),
            (not self.h.any(), "eq. (7) has no conic offset, so h must be zero"),
            (not self.A[:, last].any(), "t must not appear in the linear inequalities"),
            (not self.E[:, last].any(), "t must not appear in the linear equalities"),
        )
        for holds, reason in reasons:
            if not holds:
                raise ProblemError("instance", f"not in the shape of eq. (7): {reason}")
        return MeanStdForm(
            mu=-self.c[:last],
            lam=float(self.c[last]),
            A=self.A[:, :last],
            b=self.b,
            E=self.E[:, :last],
            d=self.d,
            L=self.G[1:, :last],
        )


@dataclass(frozen=True, eq=False, kw_only=True)
class MeanStdForm:
    """The data of eq. (7), the mean-standard-deviation SOCP.

        min  -mu.T @ x + lam * t
        s.t. A @ x <= b
             E @ x = d
             (t, L @ x) in Q

    ``L`` is any factor of the covariance with ``Sigma = L.T @ L``, which turns the
    portfolio standard deviation into ``||L @ x||_2 <= t``. Choosing that factor from
    a covariance matrix, and generating the portfolio constraints, belong to the
    portfolio builder; this class is the shape the builder produces and the solver's
    entry point consumes.

    :meth:`to_socp` maps it into the general :class:`SOCP`, putting ``t`` last and the
    risk cone as the single factor of the product. :meth:`SOCP.as_mean_std` maps it
    back, so an instance round-trips.

    Attributes:
        mu: expected returns, one entry per asset. Its length defines ``n``.
        lam: the risk-aversion parameter, strictly positive as in the plan.
        A: the inequality matrix over the assets, ``(m_ineq, n)``.
        b: the inequality right-hand side, ``(m_ineq,)``.
        E: the equality matrix over the assets, ``(m_eq, n)``.
        d: the equality right-hand side, ``(m_eq,)``.
        L: a factor of the covariance, ``(k, n)`` with ``Sigma = L.T @ L``.
    """

    mu: Vector
    lam: float
    A: Matrix
    b: Vector
    E: Matrix
    d: Vector
    L: Matrix

    def __post_init__(self) -> None:
        """Coerce every block and check the shapes against the number of assets.

        Raises:
            ProblemError: if there are no assets, if ``lam`` is not a positive finite
                number, or if any block disagrees in shape.
        """
        set_block = object.__setattr__
        set_block(self, "mu", _vector("mu", self.mu))
        if self.mu.size < 1:
            raise ProblemError("mu", "an instance needs at least one asset, found an empty mu")
        set_block(self, "lam", float(self.lam))
        if not math.isfinite(self.lam) or self.lam <= 0.0:
            raise ProblemError("lam", f"the plan takes lam > 0, found {self.lam}")
        assets = self.mu.size
        set_block(self, "A", _matrix("A", self.A, columns=assets))
        set_block(self, "b", _vector("b", self.b, size=self.A.shape[0]))
        set_block(self, "E", _matrix("E", self.E, columns=assets))
        set_block(self, "d", _vector("d", self.d, size=self.E.shape[0]))
        set_block(self, "L", _matrix("L", self.L, columns=assets))
        if self.L.shape[0] < 1:
            raise ProblemError("L", "a risk term needs at least one row, found a factor with none")

    @property
    def num_assets(self) -> int:
        """The number of assets ``n``, the length of ``x``."""
        return self.mu.size

    def to_socp(self) -> SOCP:
        """Assemble eq. (7) as a general :class:`SOCP`.

        The variable vector is ``z = (x, t)``, so ``t`` is the last variable and the
        cone is the single factor ``Q^(1 + k)``, with the head row of ``G`` selecting
        ``t`` and its tail rows holding ``L``.

        Returns:
            The instance, validated.
        """
        assets = self.num_assets
        rows = self.L.shape[0]
        conic = np.zeros((1 + rows, assets + 1))
        conic[0, assets] = 1.0
        conic[1:, :assets] = self.L
        return SOCP(
            c=np.concatenate([-self.mu, [self.lam]]),
            A=np.hstack([self.A, np.zeros((self.A.shape[0], 1))]),
            b=self.b,
            E=np.hstack([self.E, np.zeros((self.E.shape[0], 1))]),
            d=self.d,
            G=conic,
            h=np.zeros(1 + rows),
            cone=ConeProduct.from_dims(1 + rows),
        )
