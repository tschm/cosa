"""COSA: a conic active-set algorithm for second-order cone programs.

COSA solves second-order cone programs by maintaining an explicit working set of
active constraints, extending the classical active-set philosophy from polyhedral
optimisation to conic optimisation. The motivating application is
mean-standard-deviation portfolio optimisation, where the linear portfolio
constraints have a natural active-set structure and the risk term
``sqrt(x.T @ Sigma @ x)`` enters as a second-order cone.

The full project plan lives in ``docs/paper/paper.tex``; the package layout and the
decisions behind it are recorded in ``docs/development/architecture.md``, and the one
fixed sign convention for the conic KKT conditions in
``docs/development/sign-convention.md``.

Subpackages:
    problem: Problem representation (``socp``, ``portfolio``).
    geometry: Second-order cone geometry (``soc``, ``tangent``, ``step``).
    active_set: Working set and its updates (``working_set``, ``multipliers``, ``updates``).
    linear_algebra: KKT systems and factorizations (``kkt``, ``factorization``, ``scaling``).
    solver: The solver itself (``cosa``, ``initialization``, ``termination``).
    experiments: Numerical studies (``reference``, ``portfolio``, ``frontier``,
        ``benchmarks``).

Attributes:
    Vector: A one-dimensional float array, used for ``x``, ``mu``, ``b`` and ``d``.
    Matrix: A two-dimensional float array, used for ``A``, ``E``, ``L`` and ``Sigma``.
"""

import numpy as np
from numpy.typing import NDArray

from cosa.active_set.working_set import ConeStatus, ConstraintNames, WorkingSet
from cosa.geometry.soc import ConePosition
from cosa.geometry.tangent import ApexError
from cosa.linear_algebra.kkt import Direction, RowLayout, SingularKktError
from cosa.problem.portfolio import MeanStdPortfolio
from cosa.problem.socp import (
    SIGN_CONVENTION,
    SOCP,
    ConeProduct,
    MeanStdForm,
    ProblemError,
    SecondOrderCone,
    SignConvention,
)

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]

# Deliberately narrow: every module named in the subpackage list above extends this
# surface as it lands, rather than reserving a name in advance for code that does not
# exist.
#
# What lands here is the shared *vocabulary* -- the array aliases, the problem
# representation, and the types that cross subpackage boundaries -- and not every public
# name of every module. The routines stay where their context is, reached as
# `cosa.geometry.soc.is_boundary` or `cosa.active_set.updates.removal_candidate`, because
# their names only mean something next to their module: `cosa.slack` and `cosa.position`
# would be unreadable at the root, and `cosa.geometry.soc.slack` is not.
#
# The line runs between the *library* and the *harness*. The algorithm's vocabulary is
# here -- the problem, the working set, the cone's position and status, the direction and
# its row layout, and the two errors a solver loop has to catch. The experiment harness of
# `cosa.experiments` is not: its instance families, its random specifications and its
# reference-solver oracle are how the library is exercised, not part of what it offers.
__all__ = [
    "SIGN_CONVENTION",
    "SOCP",
    "ApexError",
    "ConePosition",
    "ConeProduct",
    "ConeStatus",
    "ConstraintNames",
    "Direction",
    "Matrix",
    "MeanStdForm",
    "MeanStdPortfolio",
    "ProblemError",
    "RowLayout",
    "SecondOrderCone",
    "SignConvention",
    "SingularKktError",
    "Vector",
    "WorkingSet",
]
