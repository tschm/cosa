"""Detecting a rank-deficient working set, and the null space that survives it.

§8.3 (``paper.tex:661``) is four sentences and a list: *"The working-set KKT matrix may
become singular when active constraints are linearly dependent. The implementation will
detect rank deficiency and investigate: QR-based rank detection; null-space methods;
regularization; dependent-constraint removal."* This module is the first two. The third is
a parameter of :func:`cosa.linear_algebra.kkt.solve`; the fourth changes the *working set*
rather than the linear algebra and so lives in :mod:`cosa.active_set.updates`, which is
where the issue warns it is easy to lose.

**QR with column pivoting, not an SVD.** #12's dependent-row check uses
``numpy.linalg.matrix_rank``, which is an SVD -- correct, and it answers only *how many*
directions there are. Detecting rank deficiency is not the hard part of §8.3; deciding
*which rows to remove* is, and a pivoted QR answers that directly. Its pivot order is the
rows of ``W`` sorted by how much each adds to the span, so the last ``m - rank`` of them
are a set that can be dropped without changing the row space at all. An SVD gives a rank
and leaves the choice of rows to be reconstructed.

Pivoted QR is the one thing NumPy does not have, and it is why SciPy is now a declared
dependency. ``docs/development/architecture.md`` predicted this: *"whichever issue first
needs a factorization adds scipy in the same change that imports it"*, naming M7 as the
milestone. #25 is that issue and this is that change.

**The direction never needed any of this.** A rank-deficient ``W`` leaves the *direction*
perfectly well defined -- ``d`` is the projection of ``-g / rho`` onto the null space of
``W``, and a null space does not care how many redundant rows described it. What is
undetermined is ``nu``, because dependent rows can trade multiplier mass between
themselves. :func:`null_space_basis` is the route that computes the direction without ever
forming the saddle-point matrix, and it is the "null-space method" §8.3 asks for: on a
degenerate working set it is not merely better conditioned, it is *defined* where the
saddle-point solve is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg

from cosa.problem.socp import ProblemError

if TYPE_CHECKING:
    from cosa import Matrix

__all__ = [
    "RankAnalysis",
    "analyse",
    "null_space_basis",
    "rank_tolerance",
]


def rank_tolerance(matrix: Matrix) -> float:
    """The default threshold below which a pivot counts as zero.

    ``max(m, n) * eps * |R[0, 0]|`` -- the same shape of threshold
    ``numpy.linalg.matrix_rank`` uses, with the leading diagonal entry of the pivoted ``R``
    standing in for the largest singular value. Relative, so it means the same thing on a
    working set of unit-norm bound rows and on one of factor exposures.

    Args:
        matrix: the matrix whose rank is in question.

    Returns:
        The tolerance, or ``0.0`` for an empty matrix.
    """
    entries = np.asarray(matrix, dtype=np.float64)
    if entries.size == 0:
        return 0.0
    largest = float(np.abs(entries).max())
    return max(entries.shape) * float(np.finfo(np.float64).eps) * largest


@dataclass(frozen=True)
class RankAnalysis:
    """What a pivoted QR says about a working-set matrix.

    Attributes:
        rows: how many rows ``W`` has.
        rank: how many of them are linearly independent.
        order: the rows in pivot order, most informative first. Its first :attr:`rank`
            entries are an independent subset; the rest are the redundant ones.
        smallest: the magnitude of the smallest pivot kept, which says how *nearly*
            dependent the retained rows are. A small value on a full-rank set is the
            nearly-redundant case of §12.4, and is the number #29's tolerances will care
            about.
    """

    rows: int
    rank: int
    order: tuple[int, ...]
    smallest: float

    @property
    def is_deficient(self) -> bool:
        """Are there more rows than independent directions among them?"""
        return self.rank < self.rows

    @property
    def independent(self) -> tuple[int, ...]:
        """An independent subset of the rows, in ascending order.

        Ascending rather than pivot order, because a working set's row order is ascending
        by construction and a caller comparing the two should not have to sort.
        """
        return tuple(sorted(self.order[: self.rank]))

    @property
    def dependent(self) -> tuple[int, ...]:
        """The rows that can be removed without changing the row space, ascending.

        "Can" rather than "should": which of a dependent group to drop is a choice, and the
        pivoted QR's answer -- drop the ones that added least -- is a good default and not
        the only defensible one. The *working set* may also refuse to drop a particular row,
        which is :func:`cosa.active_set.updates.drop_dependent_rows`'s problem.
        """
        return tuple(sorted(self.order[self.rank :]))

    def __str__(self) -> str:
        """The rank and the smallest pivot, for a diagnosis."""
        return f"rank {self.rank}/{self.rows}, smallest pivot {self.smallest:.3g}"


def analyse(matrix: Matrix, *, tolerance: float | None = None) -> RankAnalysis:
    """Rank-detect a working-set matrix by QR with column pivoting.

    The factorization is of ``W.T`` rather than ``W``, because column pivoting reorders
    *columns* and the rows of ``W`` are what need ordering. So the pivots come back as row
    indices of ``W`` directly.

    Args:
        matrix: the working-set matrix ``W``, of shape ``(m, n)``.
        tolerance: the pivot threshold, or ``None`` for :func:`rank_tolerance`.

    Returns:
        The analysis.

    Raises:
        ProblemError: if ``matrix`` is not two-dimensional.
    """
    entries = np.asarray(matrix, dtype=np.float64)
    if entries.ndim != 2:
        raise ProblemError("W", f"expected a matrix, found an array of shape {entries.shape}")
    rows = entries.shape[0]
    if rows == 0:
        return RankAnalysis(rows=0, rank=0, order=(), smallest=0.0)

    _, upper, pivots = scipy.linalg.qr(entries.T, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(upper))
    cut = rank_tolerance(entries) if tolerance is None else float(tolerance)
    rank = int((diagonal > cut).sum())
    return RankAnalysis(
        rows=rows,
        rank=rank,
        order=tuple(int(pivot) for pivot in pivots),
        smallest=float(diagonal[rank - 1]) if rank else 0.0,
    )


def null_space_basis(matrix: Matrix, *, tolerance: float | None = None) -> Matrix:
    """An orthonormal basis for the null space of ``W`` -- §8.3's null-space route.

    Computed from the SVD rather than from the pivoted QR above. The QR of ``W.T`` gives an
    orthonormal basis for the *row* space, and the null space is its orthogonal complement,
    which would need the full ``Q`` including its silent columns -- an SVD says it directly
    and is the numerically safer of the two when ``W`` is nearly dependent, which is
    precisely when this route is being used.

    The direction of the subproblem is then ``d = -(Z @ Z.T @ g) / rho``, which is defined
    for *any* ``W``, dependent rows included. That is what makes this a fallback worth
    having rather than an optimization: the saddle-point solve has no answer on a degenerate
    working set, and this one does.

    Args:
        matrix: the working-set matrix ``W``, of shape ``(m, n)``.
        tolerance: the singular-value threshold, or ``None`` for :func:`rank_tolerance`.

    Returns:
        A matrix of shape ``(n, n - rank)`` whose columns are an orthonormal basis of
        ``{p : W @ p = 0}``. The identity when ``W`` has no rows.
    """
    entries = np.asarray(matrix, dtype=np.float64)
    if entries.ndim != 2:
        raise ProblemError("W", f"expected a matrix, found an array of shape {entries.shape}")
    columns = entries.shape[1]
    if entries.shape[0] == 0:
        return np.eye(columns)

    _, singular, right = np.linalg.svd(entries)
    cut = rank_tolerance(entries) if tolerance is None else float(tolerance)
    rank = int((singular > cut).sum())
    return np.ascontiguousarray(right[rank:].T)
