"""The mean-standard-deviation portfolio problem, and its reduction to an SOCP.

This is the problem COSA exists for. §2.1 (``paper.tex:120``) states it as eq. (1),

    max  mu.T @ x - lam * sqrt(x.T @ Sigma @ x)   s.t.  A @ x <= b,  E @ x = d,

and §2.3 (``paper.tex:208``) says why it is worth the trouble: variance has units of
return squared, standard deviation has the units of return itself, ``[sigma] = [mu]``, so
``mu.T @ x - lam * sigma(x)`` is directly interpretable as return minus risk. The price of
that interpretation is a second-order cone instead of a quadratic, and paying it is the
project's whole premise.

:class:`MeanStdPortfolio` is eq. (1). :meth:`MeanStdPortfolio.to_socp` walks it down to
eq. (2) and then to the general :class:`cosa.SOCP`:

* **eq. (1) to eq. (2)** is two moves: negate, because the solver minimizes, and introduce
  the auxiliary ``t`` with ``||L @ x||_2 <= t``. The negation is exact and total --
  :meth:`MeanStdPortfolio.cost` is ``-utility`` at every ``x``, feasible or not. The
  auxiliary variable is an *enlargement*: eq. (2) has an ``x``-and-``t`` feasible set, and
  it agrees with eq. (1) because ``lam > 0`` pushes ``t`` down onto ``t = ||L @ x||``.
  :meth:`MeanStdPortfolio.socp_point` performs that lift, so the two objectives can be
  compared at the same point rather than argued about.
* **eq. (2) to the cone** needs a factor ``L`` with ``Sigma = L.T @ L``, which turns
  ``sqrt(x.T @ Sigma @ x)`` into ``||L @ x||_2``. That is :func:`covariance_factor`.

**Rank deficiency is the interesting case, not the edge case.** §2.1 assumes only
``Sigma >= 0`` (``paper.tex:132``), never ``Sigma > 0``, so a plain Cholesky is not
enough: it fails on the first singular covariance it meets. And singular covariances are
not exotic here -- the plan's own robustness family "highly correlated assets"
(``paper.tex:942``) produces them, more assets than return observations produces them, and
an exact factor model produces them by construction. It matters algorithmically rather than
just numerically: when ``Sigma`` is singular there is an ``x != 0`` with ``L @ x = 0``, so
the conic slack can reach the apex ``(0, 0)`` at a nonzero portfolio, which is precisely
the case §8.1 (``paper.tex:623``) has to branch on. The apex is reachable *because* the
covariance is singular.

:func:`covariance_factor` therefore goes through a symmetric eigendecomposition rather
than a pivoted Cholesky. Both handle rank deficiency; the eigendecomposition is preferred
because it is unconditionally stable on an indefinite input -- so a covariance that is
*not* PSD is diagnosed rather than silently half-factored -- and because the eigenvalues it
already computed are what the rank decision, the conditioning report and the
ill-conditioned robustness family all want to look at.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from cosa.problem.socp import SOCP, MeanStdForm, ProblemError, _matrix, _vector

if TYPE_CHECKING:
    from cosa import Matrix, Vector

__all__ = [
    "MeanStdPortfolio",
    "covariance_factor",
    "covariance_tolerance",
]


def _eigenvalue_cut(assets: int, largest: float) -> float:
    """The default eigenvalue tolerance, ``n * eps * lambda_max``.

    Args:
        assets: the dimension ``n`` of the covariance.
        largest: the largest eigenvalue magnitude.

    Returns:
        The tolerance, ``0.0`` for the zero matrix.
    """
    return assets * float(np.finfo(np.float64).eps) * largest


def covariance_tolerance(covariance: Matrix) -> float:
    """The default eigenvalue tolerance for a covariance: ``n * eps * lambda_max``.

    The threshold ``numpy.linalg.matrix_rank`` uses, and for the same reason: it is the
    magnitude at which an eigenvalue is indistinguishable from zero given the rounding
    already present in an ``n``-by-``n`` symmetric decomposition. Discarding eigenvalues
    below it perturbs ``Sigma`` by at most that much, which is why
    :func:`covariance_factor` can drop rank *and* still reproduce ``Sigma`` to machine
    precision.

    It is deliberately tight, and it is the right default only for a covariance that was
    computed carefully. A shrinkage or repaired estimator can carry negative eigenvalues
    orders of magnitude larger than this; that is a modelling decision, and a caller who
    has made it should pass a tolerance saying so rather than have one guessed here.

    Exposed because a caller comparing ranks, or deciding whether a covariance is
    "nearly" singular, needs the number :func:`covariance_factor` would have used. That
    function does not call this one -- it has the eigenvalues in hand already and does not
    pay for a second decomposition.

    Args:
        covariance: the matrix ``Sigma``, symmetric and positive semidefinite.

    Returns:
        The tolerance, or ``0.0`` for the zero matrix, whose rank is unambiguous.

    Raises:
        ProblemError: if ``Sigma`` is not a square, finite matrix.
    """
    matrix = _matrix("Sigma", covariance)
    if matrix.shape[0] != matrix.shape[1]:
        raise ProblemError("Sigma", f"a covariance is square, found shape {matrix.shape}")
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    largest = float(np.abs(eigenvalues).max()) if eigenvalues.size else 0.0
    return _eigenvalue_cut(matrix.shape[0], largest)


def covariance_factor(covariance: Matrix, *, tolerance: float | None = None) -> Matrix:
    """A factor ``L`` with ``Sigma = L.T @ L``, valid for any ``Sigma >= 0``.

    Built from the symmetric eigendecomposition ``Sigma = V @ diag(w) @ V.T``: with the
    non-negligible eigenvalues ``w_k`` and their eigenvectors ``V_k``,

        L = diag(sqrt(w_k)) @ V_k.T

    so ``L`` has one row per unit of rank and ``||L @ x||_2 == sqrt(x.T @ Sigma @ x)`` to
    machine precision. A rank-deficient covariance simply produces a short ``L``, with no
    special case anywhere: a rank-``k`` covariance over ``n`` assets gives a ``(k, n)``
    factor and a cone ``Q^(1 + k)``, which is also the cheap representation.

    ``Sigma`` is symmetrized as ``(Sigma + Sigma.T) / 2`` before the decomposition. That is
    not a repair of asymmetric input -- input that is asymmetric by more than the tolerance
    is rejected -- but the removal of the rounding-level asymmetry a computed covariance
    carries, so that the eigendecomposition sees exactly the matrix the factor claims to
    reproduce.

    The zero covariance is the degenerate limit and is answered honestly with a single zero
    row: ``||L @ x|| = 0 = sqrt(x.T @ 0 @ x)`` for every ``x``, and the conic slack sits at
    the apex everywhere. The row is there because the risk term has to have a cone to live
    in; a factor with no rows would leave eq. (2) with no conic block at all.

    Args:
        covariance: the matrix ``Sigma``, square, symmetric and positive semidefinite.
        tolerance: the magnitude below which an eigenvalue counts as zero. Eigenvalues in
            ``[-tolerance, tolerance]`` are dropped; anything below ``-tolerance`` makes
            the matrix not positive semidefinite. Defaults to
            :func:`covariance_tolerance`.

    Returns:
        The factor ``L``, of shape ``(rank, n)`` -- or ``(1, n)`` of zeros when the rank is
        zero. Its rows are in order of decreasing eigenvalue, so the leading rows carry
        the dominant risk factors.

    Raises:
        ProblemError: if ``Sigma`` is not square, is empty, is asymmetric by more than a
            rounding-level threshold, carries a non-finite entry, or has an eigenvalue
            below ``-tolerance``, which means it is not a covariance.
    """
    matrix = _matrix("Sigma", covariance)
    assets, columns = matrix.shape
    if assets != columns:
        raise ProblemError("Sigma", f"a covariance is square, found shape {matrix.shape}")
    if assets < 1:
        raise ProblemError("Sigma", "an instance needs at least one asset, found an empty covariance")
    if tolerance is not None and (not math.isfinite(tolerance) or tolerance < 0.0):
        raise ProblemError("tolerance", f"the eigenvalue tolerance must be finite and non-negative, found {tolerance}")

    epsilon = float(np.finfo(np.float64).eps)
    asymmetry = float(np.abs(matrix - matrix.T).max())
    if asymmetry > max(_eigenvalue_cut(assets, float(np.abs(matrix).max())), epsilon):
        raise ProblemError("Sigma", f"a covariance is symmetric, found |Sigma - Sigma.T| up to {asymmetry:g}")

    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    cut = _eigenvalue_cut(assets, float(np.abs(eigenvalues).max())) if tolerance is None else float(tolerance)
    smallest = float(eigenvalues[0])
    if smallest < -cut:
        raise ProblemError(
            "Sigma",
            f"a covariance is positive semidefinite, found an eigenvalue of {smallest:g} "
            f"against a tolerance of {cut:g}",
        )

    # `eigh` returns ascending eigenvalues; reverse so the dominant risk factor leads.
    keep = eigenvalues > cut
    if not keep.any():
        return np.zeros((1, assets))
    scale = np.sqrt(eigenvalues[keep])[::-1]
    directions = eigenvectors[:, keep][:, ::-1]
    return np.ascontiguousarray(scale[:, None] * directions.T)


@dataclass(frozen=True, eq=False, kw_only=True)
class MeanStdPortfolio:
    """Eq. (1): the mean-standard-deviation portfolio problem, in covariance form.

        max  mu.T @ x - lam * sqrt(x.T @ Sigma @ x)
        s.t. A @ x <= b
             E @ x = d

    The problem as a portfolio manager states it, holding the covariance itself rather
    than a factor of it. :meth:`to_mean_std` chooses the factor and hands over eq. (7)'s
    :class:`cosa.MeanStdForm`; :meth:`to_socp` continues to the general
    :class:`cosa.SOCP`. Everything below the factorization is the representation's
    business, not this class's.

    Every block is required and explicit, as in :class:`cosa.SOCP`: a problem with no
    inequalities carries an ``A`` of shape ``(0, n)``. :meth:`unconstrained` plus
    :meth:`with_inequalities` and :meth:`with_equalities` build one up instead, which is
    what the constraint-family generators want.

    Arrays are copied into contiguous ``float64`` on construction and instances are
    frozen, so nothing here aliases a caller's buffer and every method that changes an
    instance returns a new one. Instances compare by identity, for the same reason
    :class:`cosa.SOCP` does: elementwise array comparison does not answer that question.

    Attributes:
        mu: expected returns, one entry per asset. Its length defines ``n``.
        Sigma: the covariance, ``(n, n)``, symmetric and positive semidefinite -- not
            necessarily positive definite.
        lam: the risk-aversion parameter, strictly positive as §2.1 takes it.
        A: the inequality matrix, ``(m_ineq, n)``.
        b: the inequality right-hand side, ``(m_ineq,)``.
        E: the equality matrix, ``(m_eq, n)``.
        d: the equality right-hand side, ``(m_eq,)``.
    """

    mu: Vector
    Sigma: Matrix  # the paper's name for the covariance, as with the A, E and L of socp.py
    lam: float
    A: Matrix
    b: Vector
    E: Matrix
    d: Vector

    def __post_init__(self) -> None:
        """Coerce every block and check it against the number of assets.

        The covariance is checked for shape and finiteness here and for symmetry and
        semidefiniteness in :func:`covariance_factor`, which has the eigenvalues in hand.
        Construction is therefore cheap: it does no decomposition.

        Raises:
            ProblemError: if there are no assets, if ``lam`` is not a positive finite
                number, or if any block disagrees in shape.
        """
        set_block = object.__setattr__
        set_block(self, "mu", _vector("mu", self.mu))
        if self.mu.size < 1:
            raise ProblemError("mu", "an instance needs at least one asset, found an empty mu")
        assets = self.mu.size
        set_block(self, "Sigma", _matrix("Sigma", self.Sigma, rows=assets, columns=assets))
        set_block(self, "lam", float(self.lam))
        if not math.isfinite(self.lam) or self.lam <= 0.0:
            raise ProblemError("lam", f"the plan takes lam > 0, found {self.lam}")
        set_block(self, "A", _matrix("A", self.A, columns=assets))
        set_block(self, "b", _vector("b", self.b, size=self.A.shape[0]))
        set_block(self, "E", _matrix("E", self.E, columns=assets))
        set_block(self, "d", _vector("d", self.d, size=self.E.shape[0]))

    @classmethod
    def unconstrained(cls, *, mu: Vector, Sigma: Matrix, lam: float) -> MeanStdPortfolio:  # noqa: N803
        """The objective on its own: no inequalities and no equalities.

        The starting point for building an instance incrementally. A portfolio problem
        with no constraints at all is unusual but not degenerate -- ``lam > 0`` and
        ``Sigma >= 0`` keep it bounded whenever ``mu`` is in the row space of ``Sigma`` --
        and it is the cleanest instance to test the cone geometry on.

        Args:
            mu: expected returns, one entry per asset.
            Sigma: the covariance, ``(n, n)``.
            lam: the risk-aversion parameter.

        Returns:
            The instance with an empty constraint description.
        """
        assets = _vector("mu", mu).size
        return cls(
            mu=mu,
            Sigma=Sigma,
            lam=lam,
            A=np.zeros((0, assets)),
            b=np.zeros(0),
            E=np.zeros((0, assets)),
            d=np.zeros(0),
        )

    @property
    def num_assets(self) -> int:
        """The number of assets ``n``, the length of ``x``."""
        return self.mu.size

    def with_inequalities(self, rows: Matrix, rhs: Vector) -> MeanStdPortfolio:
        """Append rows to ``A @ x <= b``.

        Args:
            rows: the new rows of ``A``, ``(k, n)``.
            rhs: their right-hand sides, ``(k,)``.

        Returns:
            A new instance carrying the additional inequalities.
        """
        return replace(
            self,
            A=np.vstack([self.A, _matrix("new inequality rows", rows, columns=self.num_assets)]),
            b=np.concatenate([self.b, _vector("new inequality right-hand side", rhs)]),
        )

    def with_equalities(self, rows: Matrix, rhs: Vector) -> MeanStdPortfolio:
        """Append rows to ``E @ x = d``.

        Args:
            rows: the new rows of ``E``, ``(k, n)``.
            rhs: their right-hand sides, ``(k,)``.

        Returns:
            A new instance carrying the additional equalities.
        """
        return replace(
            self,
            E=np.vstack([self.E, _matrix("new equality rows", rows, columns=self.num_assets)]),
            d=np.concatenate([self.d, _vector("new equality right-hand side", rhs)]),
        )

    def factor(self, *, tolerance: float | None = None) -> Matrix:
        """The factor ``L`` of this instance's covariance, ``Sigma = L.T @ L``.

        Recomputed on each call rather than cached: the decomposition is the expensive
        part of building the problem, and it happens once per solve at
        :meth:`to_mean_std`, not once per iteration.

        Args:
            tolerance: the eigenvalue tolerance, passed to :func:`covariance_factor`.

        Returns:
            The factor, of shape ``(rank, n)``.
        """
        return covariance_factor(self.Sigma, tolerance=tolerance)

    def expected_return(self, x: Vector) -> float:
        """The portfolio expected return ``r(x) = mu.T @ x`` of §2.1.

        Args:
            x: portfolio weights, ``(n,)``.

        Returns:
            The expected return.
        """
        return float(self.mu @ _vector("x", x, size=self.num_assets))

    def variance(self, x: Vector) -> float:
        """The portfolio variance ``x.T @ Sigma @ x``, clipped at zero.

        Clipped because a semidefinite ``Sigma`` and an ``x`` in its near-null space give
        a quadratic form that rounds to a small negative number, and a negative variance
        is never a fact about the portfolio -- only about the arithmetic.

        Args:
            x: portfolio weights, ``(n,)``.

        Returns:
            The variance, non-negative.
        """
        weights = _vector("x", x, size=self.num_assets)
        return max(0.0, float(weights @ self.Sigma @ weights))

    def std(self, x: Vector) -> float:
        """The portfolio standard deviation ``sigma(x) = sqrt(x.T @ Sigma @ x)`` of §2.1.

        The quantity the cone constraint bounds, and the one carrying the same units as
        the expected return -- which per §2.3 is the reason the problem is conic at all.

        Args:
            x: portfolio weights, ``(n,)``.

        Returns:
            The standard deviation, non-negative.
        """
        return math.sqrt(self.variance(x))

    def utility(self, x: Vector) -> float:
        """Eq. (1)'s objective, ``mu.T @ x - lam * sigma(x)``, to be maximized.

        Return minus risk in the units of return. Reported as the portfolio number a user
        cares about; the solver never sees it, because it minimizes :meth:`cost`.

        Args:
            x: portfolio weights, ``(n,)``.

        Returns:
            The utility.
        """
        return self.expected_return(x) - self.lam * self.std(x)

    def cost(self, x: Vector) -> float:
        """Eq. (2)'s objective at the ``t`` eq. (1) implies: ``-utility(x)``.

        The minimization form, and exactly ``-``:meth:`utility` at every ``x``. It equals
        ``c.T @ z`` of :meth:`to_socp` evaluated at :meth:`socp_point`, which is the whole
        content of the eq. (1)-to-eq. (2) mapping and is asserted as such in the tests.

        Args:
            x: portfolio weights, ``(n,)``.

        Returns:
            The cost, ``-mu.T @ x + lam * sigma(x)``.
        """
        return -self.utility(x)

    def socp_point(self, x: Vector) -> Vector:
        """Lift ``x`` to eq. (2)'s variable ``z = (x, t)`` with ``t = sigma(x)``.

        Eq. (2) has one variable more than eq. (1), so the two objectives can only be
        compared at a point after this lift. The ``t`` chosen is the smallest feasible one,
        which is where ``lam > 0`` drives it anyway: any larger ``t`` is conically feasible
        but costs ``lam`` per unit, so the optimum of eq. (2) always has ``t = sigma(x)``
        and the two problems have the same solution set in ``x``.

        Args:
            x: portfolio weights, ``(n,)``.

        Returns:
            The point ``z = (x, sigma(x))``, of length ``n + 1``.
        """
        weights = _vector("x", x, size=self.num_assets)
        return np.concatenate([weights, [self.std(weights)]])

    def to_mean_std(self, *, tolerance: float | None = None) -> MeanStdForm:
        """Reduce to eq. (7) by replacing the covariance with a factor of it.

        The one lossy-looking step in the chain, and it is not lossy: ``L`` reproduces
        ``Sigma`` to machine precision, and the risk term ``sqrt(x.T @ Sigma @ x)`` becomes
        ``||L @ x||_2`` exactly.

        Args:
            tolerance: the eigenvalue tolerance, passed to :func:`covariance_factor`.

        Returns:
            Eq. (7)'s data, with ``L`` of shape ``(rank, n)``.
        """
        return MeanStdForm(
            mu=self.mu,
            lam=self.lam,
            A=self.A,
            b=self.b,
            E=self.E,
            d=self.d,
            L=self.factor(tolerance=tolerance),
        )

    def to_socp(self, *, tolerance: float | None = None) -> SOCP:
        """Reduce all the way to the general :class:`cosa.SOCP` the solver consumes.

        Eq. (1) to eq. (2) to eq. (7) to the general form, in one call. The variable vector
        is ``z = (x, t)`` with ``t`` last, and the cone is the single factor
        ``Q^(1 + rank)``.

        Args:
            tolerance: the eigenvalue tolerance, passed to :func:`covariance_factor`.

        Returns:
            The instance, validated.
        """
        return self.to_mean_std(tolerance=tolerance).to_socp()
