"""COSA: a conic active-set algorithm for second-order cone programs.

COSA solves second-order cone programs by maintaining an explicit working set of
active constraints, extending the classical active-set philosophy from polyhedral
optimisation to conic optimisation. The motivating application is
mean-standard-deviation portfolio optimisation, where the linear portfolio
constraints have a natural active-set structure and the risk term
``sqrt(x.T @ Sigma @ x)`` enters as a second-order cone.

The full project plan lives in ``docs/paper/paper.tex``; the package layout and the
decisions behind it are recorded in ``docs/development/architecture.md``.

Subpackages:
    problem: Problem representation (``socp``, ``portfolio``).
    geometry: Second-order cone geometry (``soc``, ``tangent``, ``step``).
    active_set: Working set and its updates (``working_set``, ``multipliers``, ``updates``).
    linear_algebra: KKT systems and factorizations (``kkt``, ``factorization``, ``scaling``).
    solver: The solver itself (``cosa``, ``initialization``, ``termination``).
    experiments: Numerical studies (``portfolio``, ``frontier``, ``benchmarks``).

Attributes:
    Vector: A one-dimensional float array, used for ``x``, ``mu``, ``b`` and ``d``.
    Matrix: A two-dimensional float array, used for ``A``, ``E``, ``L`` and ``Sigma``.
"""

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]

# Deliberately narrow. Nothing else is public yet: every module named in the
# subpackage list above arrives with its own issue, and each one extends this
# surface as it lands rather than reserving a name in advance.
__all__ = ["Matrix", "Vector"]
