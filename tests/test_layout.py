"""Every package named in the architecture block imports, and the public surface holds.

This is the executable half of issue #8's "done when": the layout is only real if it
imports. It is deliberately cheap -- it asserts structure, not behaviour, because the
modules inside these packages arrive with their own issues.
"""

import importlib

import pytest

SUBPACKAGES = [
    "problem",
    "geometry",
    "active_set",
    "linear_algebra",
    "solver",
    "experiments",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    """Each of the six subpackages imports and carries a docstring."""
    module = importlib.import_module(f"cosa.{name}")
    assert module.__doc__, f"cosa.{name} has no docstring"


# Every name `cosa.__init__` re-exports. The list grows as the modules named in the
# subpackage docstrings land, and it is spelled out here so that a name arriving or
# leaving is a deliberate edit to the public surface rather than a side effect.
PUBLIC_SURFACE = [
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
    "Multipliers",
    "ProblemError",
    "RowLayout",
    "Scaling",
    "SecondOrderCone",
    "SignConvention",
    "SingularKktError",
    "Vector",
    "WorkingSet",
]

# What the root deliberately does *not* re-export: the routines. They are reached as
# `cosa.geometry.soc.is_boundary` or `cosa.active_set.updates.removal_candidate`, because
# a name like `slack` or `position` only means something next to its module.
NOT_AT_THE_ROOT = [
    "Factorization",
    "Guard",
    "Reuse",
    "covariance_factor",
    "curvature",
    "deactivate_cones",
    "equilibrate",
    "is_boundary",
    "lagrangian_curvature",
    "position",
    "positions",
    "lexicographic_candidate",
    "objective_of",
    "removal_candidate",
    "slack",
    "tangent_row",
    "working_set_matrix",
    # Types too, when the name is generic enough that the module is what gives it meaning.
    # `cosa.Recorder` and `cosa.Metrics` say nothing about what is being recorded or
    # measured; `instrumentation.Recorder` says it exactly. Same rule as the routines.
    "Metrics",
    "Recorder",
    "InvariantChecker",
]

# The other half of the rule: `cosa.experiments` is the harness that exercises the
# library, not part of what the library offers, so none of it reaches the root either.
NOT_THE_LIBRARY = [
    "CrossCheck",
    "CvxpySolver",
    "PortfolioInstance",
    "RandomSpec",
    "cross_check",
    "random_instance",
    "solve_reference",
]


def test_public_surface():
    """The package exports the array aliases, the problem representation, and no more."""
    import cosa

    assert cosa.__all__ == PUBLIC_SURFACE
    for name in cosa.__all__:
        assert hasattr(cosa, name), f"cosa.__all__ names {name}, which is missing"


def test_the_root_exports_types_and_not_routines():
    """The rule the surface follows, asserted so that a drift from it is deliberate."""
    import cosa

    for name in NOT_AT_THE_ROOT:
        assert not hasattr(cosa, name), f"{name} belongs to its module, not to the root"


def test_the_root_exports_the_library_and_not_the_harness():
    """`cosa.experiments` exercises the library; it is not part of its surface."""
    import cosa

    for name in NOT_THE_LIBRARY:
        assert not hasattr(cosa, name), f"{name} is harness, not library"


def test_array_aliases_are_usable():
    """The aliases annotate real arrays -- they are the vocabulary every module shares."""
    import numpy as np

    import cosa

    vector: cosa.Vector = np.zeros(3, dtype=np.float64)
    matrix: cosa.Matrix = np.eye(3, dtype=np.float64)
    assert vector.shape == (3,)
    assert matrix.shape == (3, 3)
