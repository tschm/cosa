"""Success Criterion 7: a second cone, added without editing the solver loop.

Issue #38. §20's seventh criterion asks that "the resulting implementation is sufficiently
modular to support extensions beyond the initial portfolio application", and §18.1
(`paper.tex:1265`) names the concrete extension: several factors `(t_j, L_j x) in Q_j`
instead of one.

That is a pass/fail claim, so it needs a demonstration rather than an assurance. This file
is the demonstration: it builds a two-cone SOCP that no portfolio generator produces, solves
it with the ordinary entry point, and checks the answer against a reference solver. Nothing
in `cosa/solver/` is touched, monkeypatched or subclassed — if that were necessary the
criterion would have failed, and the test would be the place it showed.
"""

import numpy as np
import pytest

from cosa import SOCP, ConeStatus, SecondOrderCone
from cosa.experiments.reference import SolverUnavailableError, default_solver
from cosa.solver import cosa as solver


@pytest.fixture(scope="module")
def oracle():
    """A reference solver, or a skip: the point of this file is agreement with one."""
    try:
        return default_solver()
    except SolverUnavailableError as missing:  # pragma: no cover - CI installs one
        pytest.skip(str(missing))


def two_cone_problem():
    """A six-variable SOCP with two independent second-order cones.

    Variables `(x1, x2, x3, x4, t1, t2)`. Each cone has its own head and its own pair of tail
    rows drawn from `x`; the objective charges the two heads at different rates and the
    linear budget couples all four `x`. Nothing about this is a portfolio — eq. (7) has one
    cone, one head and one `L` — and that is the point: §18.1 says the working set has to
    handle several factors one at a time, and this is several factors.
    """
    problem = SOCP.unconstrained(np.array([-1.0, -0.8, -0.6, -0.5, 0.5, 0.4]))
    problem = problem.add_inequalities([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [0.5])
    problem = problem.add_equalities([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0]], [1.0])
    first = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    second = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.8, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.2, 0.0, 0.0],
        ]
    )
    # Nonzero offsets on the tails, so neither factor's tail can be driven to zero. Without
    # them the optimum sits at an apex the multiplier cannot justify, which is a real
    # limitation of the method (§14.4) but not the thing this file is testing.
    problem = problem.add_cone(SecondOrderCone(3), first, np.array([0.0, 0.05, 0.05]))
    return problem.add_cone(SecondOrderCone(3), second, np.array([0.0, 0.05, 0.05]))


def three_cone_problem():
    """The same, with a third factor over a fresh head and an overlapping tail.

    The tails overlap in `x1` and `x3`, so the three factors are not separable and the
    working set genuinely has to track them apart rather than treating the product as one
    large cone.
    """
    problem = two_cone_problem()
    grown = SOCP(
        c=np.concatenate([problem.c, [0.3]]),
        A=np.hstack([problem.A, np.zeros((problem.num_inequalities, 1))]),
        b=problem.b,
        E=np.hstack([problem.E, np.zeros((problem.num_equalities, 1))]),
        d=problem.d,
        G=np.hstack([problem.G, np.zeros((problem.cone.dim, 1))]),
        h=problem.h,
        cone=problem.cone,
    )
    rows = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    return grown.add_cone(SecondOrderCone(3), rows, np.array([0.0, 0.05, 0.05]))


# ----------------------------------------------------------------------------------
# The demonstration
# ----------------------------------------------------------------------------------


def test_a_two_cone_problem_solves_through_the_ordinary_entry_point():
    """No new module, no new argument, no branch on the number of factors."""
    answer = solver.solve(two_cone_problem())
    assert answer.status == "optimal"
    assert answer.residuals.is_optimal()


def test_it_agrees_with_a_reference_solver(oracle):
    """Modularity that produced a wrong answer would not be modularity."""
    problem = two_cone_problem()
    ours = solver.solve(problem)
    theirs = oracle.solve(problem)
    assert theirs.is_optimal
    assert theirs.agrees_with(ours.objective(problem))


def test_both_factors_are_tracked_independently():
    """The working set carries a state per factor, which is what makes this work.

    A single active/inactive flag for "the cone" would be unable to describe a solution with
    one factor on its boundary and one strictly inside, which is what this instance has.
    """
    answer = solver.solve(two_cone_problem())
    statuses = answer.working_set.cone_status
    assert len(statuses) == 2
    assert set(statuses) != {ConeStatus.INACTIVE}, "at least one factor is active here"


def test_the_multipliers_have_a_block_per_factor():
    """§6's `w in K` is a Cartesian product, and the recovery respects the product."""
    problem = two_cone_problem()
    answer = solver.solve(problem)
    blocks = problem.cone.blocks(answer.multipliers.w)
    assert len(blocks) == 2
    assert all(block.size == 3 for block in blocks)


def test_the_residuals_certify_a_product_cone():
    """Complementarity over a product is a sum over factors, not a special case."""
    answer = solver.solve(two_cone_problem())
    assert answer.residuals.cone_complementarity < 1e-8
    assert answer.residuals.dual < 1e-8


def test_three_cones_work_too(oracle):
    """One and two could both be accidents of the implementation; three is a pattern."""
    problem = three_cone_problem()
    answer = solver.solve(problem)
    assert answer.status == "optimal"
    assert len(answer.working_set.cone_status) == 3
    assert oracle.solve(problem).agrees_with(answer.objective(problem))


def test_nothing_private_is_reached_into():
    """The mechanical half of the criterion, checked by parsing rather than by grepping.

    A demonstration that needed to monkeypatch the loop, subclass a solver type or reach past
    a leading underscore would be showing the criterion failing, not passing. The file's own
    syntax tree is inspected rather than its text, because a text search finds this
    docstring — and because the next person to extend this file will not read it first.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(Path(__file__).read_text())
    imported = [
        alias.name.split(".")[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom | ast.Import)
        for alias in node.names
    ]
    assert not [name for name in imported if name.startswith("_")], imported

    attributes = [node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)]
    assert not [name for name in attributes if name.startswith("_")], attributes

    arguments = [
        argument.arg for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) for argument in node.args.args
    ]
    assert "monkeypatch" not in arguments
