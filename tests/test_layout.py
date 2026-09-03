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


def test_public_surface():
    """The package exports the shared array aliases and nothing undeclared."""
    import cosa

    assert cosa.__all__ == ["Matrix", "Vector"]
    for name in cosa.__all__:
        assert hasattr(cosa, name), f"cosa.__all__ names {name}, which is missing"


def test_array_aliases_are_usable():
    """The aliases annotate real arrays -- they are the vocabulary every module shares."""
    import numpy as np

    import cosa

    vector: cosa.Vector = np.zeros(3, dtype=np.float64)
    matrix: cosa.Matrix = np.eye(3, dtype=np.float64)
    assert vector.shape == (3,)
    assert matrix.shape == (3, 3)
