"""The reference-solver oracle, and the promise that it is always there.

The executable half of issue #21. Its "done when" is a claim about *availability*, not
about accuracy: any generated problem can be solved by at least one reference solver with
no license present, so §16.3's "for **every** randomly generated test problem, compare
COSA against a reference solver" (`paper.tex:1126`) is a check that actually runs.

`test_an_open_reference_solver_is_available_without_a_license` is therefore the load-
bearing test in this file. If it fails in CI, the cross-solver check has quietly stopped
being a check, and every correctness claim resting on it is unsupported -- which is the
failure mode this issue exists to prevent, and the reason it is not deferred to M10.

The commercial solvers §12.1 names are exercised through the same interface and skipped
cleanly when they are not installed or not licensed.
"""

import sys

import numpy as np
import pytest

from cosa import SOCP, MeanStdForm, MeanStdPortfolio, ProblemError
from cosa.experiments import reference

# ----------------------------------------------------------------------------------
# Instances whose optimal objective is known by hand
# ----------------------------------------------------------------------------------

# Solved by hand, so the numbers below are facts about the problems rather than
# recordings of what a solver produced.
#
# BOUND: one asset, mu = 1, lam = 1/2, x <= 1, Sigma = 1. The objective -x + t/2 with
# t = |x| decreases along x, so the bound binds: x = t = 1 and the value is -1/2.
#
# BUDGET: two assets, mu = (2, 1), lam = 1, sum(x) = 1, Sigma = I. The risk term is
# isotropic, so everything goes into the better asset: x = (1, 0), t = 1, value -1.
GOLDEN = {
    "bound": (
        MeanStdForm(
            mu=np.array([1.0]),
            lam=0.5,
            A=np.array([[1.0]]),
            b=np.array([1.0]),
            E=np.zeros((0, 1)),
            d=np.zeros(0),
            L=np.array([[1.0]]),
        ),
        -0.5,
    ),
    "budget": (
        MeanStdForm(
            mu=np.array([2.0, 1.0]),
            lam=1.0,
            A=np.zeros((0, 2)),
            b=np.zeros(0),
            E=np.array([[1.0, 1.0]]),
            d=np.array([1.0]),
            L=np.eye(2),
        ),
        -1.0,
    ),
}


@pytest.fixture(params=sorted(GOLDEN))
def golden(request):
    """One hand-solved instance together with its known optimal value."""
    form, objective = GOLDEN[request.param]
    return form.to_socp(), objective


# ----------------------------------------------------------------------------------
# The promise: an oracle with no license
# ----------------------------------------------------------------------------------


def test_an_open_reference_solver_is_available_without_a_license():
    """Issue #21's "done when", and the one test in this file that must never skip.

    §16.3 requires *every* generated problem to be cross-checked. A license-gated oracle
    cannot satisfy "every" and cannot run in CI at all, so the open fallback is not a
    convenience -- it is what makes the requirement satisfiable.
    """
    solver = reference.default_solver()
    assert solver.is_available()
    assert solver.name in reference.OPEN_BACKENDS


def test_the_open_backends_need_no_license():
    """Stated as a property of the list, so a licensed solver cannot be slipped into it."""
    assert set(reference.OPEN_BACKENDS).isdisjoint(reference.LICENSED_BACKENDS)


def test_available_solvers_filters_out_what_is_not_installed():
    """An oracle that is not installed is not offered, so a caller never has to guess."""
    available = reference.available_solvers()
    assert available, "no open reference solver is installed -- the CI fallback is broken"
    assert all(solver.is_available() for solver in available)
    assert [solver.name for solver in available] == [
        backend for backend in reference.OPEN_BACKENDS if reference.CvxpySolver(backend=backend).is_available()
    ]


def test_the_first_available_backend_is_preferred():
    """Preference order is a property of the list, not of whatever happened to install."""
    available = [solver.name for solver in reference.available_solvers()]
    assert reference.default_solver().name == available[0]


def test_no_available_solver_is_reported_as_unavailable_not_as_a_crash():
    """The message has to name the extra to install: that is the actionable part."""
    with pytest.raises(reference.SolverUnavailableError, match="reference"):
        reference.default_solver(backends=("NOT_A_SOLVER",))


# ----------------------------------------------------------------------------------
# The adapter is an interface, so the oracle is swappable
# ----------------------------------------------------------------------------------


def test_the_cvxpy_adapter_satisfies_the_interface():
    """§12.1's solver list will grow, and a study comparing two must hold both as one type."""
    assert isinstance(reference.CvxpySolver(), reference.ReferenceSolver)


def test_a_solver_reports_its_own_name():
    """The name is what identifies a row in the comparison tables of #34."""
    assert reference.CvxpySolver(backend="CLARABEL").name == "CLARABEL"


def test_an_unknown_backend_is_unavailable_rather_than_an_error():
    """is_available must be cheap and total: it answers, it does not raise."""
    assert reference.CvxpySolver(backend="NOT_A_SOLVER").is_available() is False


def test_an_unknown_backend_fails_as_unavailable(golden):
    """A backend that cannot run raises SolverUnavailable, which a test can skip on.

    The distinction that matters: "no license" and "wrong answer" must not arrive as the
    same failure.
    """
    problem, _ = golden
    with pytest.raises(reference.SolverUnavailableError):
        reference.CvxpySolver(backend="NOT_A_SOLVER").solve(problem)


def test_the_adapter_survives_cvxpy_not_being_installed(monkeypatch, golden):
    """The module must import and answer without CVXPY, or a test cannot skip on it.

    `cosa` does not depend on CVXPY -- only the `reference` extra does -- so
    `is_available` has to return False rather than raise `ImportError` at collection
    time. Simulated by poisoning the import rather than by uninstalling anything: a
    `None` in `sys.modules` is what makes `import cvxpy` fail.
    """
    monkeypatch.setitem(sys.modules, "cvxpy", None)
    solver = reference.CvxpySolver()
    assert solver.is_available() is False
    problem, _ = golden
    with pytest.raises(reference.SolverUnavailableError, match="reference"):
        solver.solve(problem)


# ----------------------------------------------------------------------------------
# The oracle answers correctly, on problems whose answer is known
# ----------------------------------------------------------------------------------


def test_the_oracle_finds_the_hand_solved_optimum(golden):
    """The oracle is checked against arithmetic before anything is checked against it."""
    problem, objective = golden
    solution = reference.solve_reference(problem)
    assert solution.is_optimal
    assert solution.objective == pytest.approx(objective, abs=1e-6)
    assert solution.agrees_with(objective)


def test_every_available_open_backend_agrees_on_the_optimum(golden):
    """Cross-solver agreement among the references themselves, which §16.3 presumes."""
    problem, objective = golden
    for solver in reference.available_solvers():
        solution = solver.solve(problem)
        assert solution.agrees_with(objective, tolerance=1e-4), f"{solver.name} disagreed"


def test_the_oracle_solves_a_portfolio_built_from_a_covariance():
    """The path a generated instance actually takes: eq. (1) to eq. (7) to the oracle."""
    portfolio = MeanStdPortfolio(
        mu=np.array([0.10, 0.04, 0.06]),
        Sigma=np.diag([0.04, 0.09, 0.16]),
        lam=2.0,
        A=np.array([[1.0, 0.0, 0.0]]),
        b=np.array([0.5]),
        E=np.array([[1.0, 1.0, 1.0]]),
        d=np.array([1.0]),
    )
    solution = reference.solve_reference(portfolio.to_socp())
    assert solution.is_optimal

    x = solution.z[:-1]
    assert solution.objective == pytest.approx(portfolio.cost(x), abs=1e-6)
    assert float(np.sum(x)) == pytest.approx(1.0, abs=1e-6)
    assert x[0] <= 0.5 + 1e-6


def test_the_oracle_solves_a_rank_deficient_instance():
    """A singular covariance is a shorter cone tail and nothing else, oracle included.

    With `Sigma = ones(3, 3)` the risk term sees only `sum(x)`, which the budget
    equality already pins, so the cone stops bounding the objective altogether and the
    long-only bounds are the only thing that does. Worth recording: on a rank-deficient
    instance, boundedness moves from the conic block to the polyhedral one -- and
    without the bounds this same instance is unbounded, which the oracle duly says.
    """
    portfolio = (
        MeanStdPortfolio.unconstrained(mu=np.array([0.10, 0.04, 0.06]), Sigma=np.ones((3, 3)), lam=1.0)
        .with_equalities([[1.0, 1.0, 1.0]], [1.0])
        .with_inequalities(-np.eye(3), np.zeros(3))
    )
    assert portfolio.to_socp().cone.cones[0].dim == 2, "rank 1, so the cone tail is one row"
    assert reference.solve_reference(portfolio.to_mean_std().to_socp()).is_optimal

    unbounded = MeanStdPortfolio.unconstrained(
        mu=portfolio.mu, Sigma=portfolio.Sigma, lam=portfolio.lam
    ).with_equalities([[1.0, 1.0, 1.0]], [1.0])
    assert reference.solve_reference(unbounded.to_socp()).objective == -np.inf

    solution = reference.solve_reference(portfolio.to_socp())
    assert solution.is_optimal
    assert solution.objective == pytest.approx(portfolio.cost(solution.z[:-1]), abs=1e-6)
    assert solution.objective == pytest.approx(0.90, abs=1e-6), "all in on the best asset"


def test_the_oracle_solves_an_instance_with_no_cone_at_all():
    """The empty cone product is a linear program, and the translation must not choke.

    The polyhedral half of the algorithm is tested against LPs, so the oracle has to
    handle a problem with no conic block -- and, symmetrically, one with no linear rows.
    """
    lp = SOCP.unconstrained(np.array([1.0, 1.0])).add_inequalities([[-1.0, 0.0], [0.0, -1.0]], [-1.0, -2.0])
    solution = reference.solve_reference(lp)
    assert solution.is_optimal
    assert solution.objective == pytest.approx(3.0, abs=1e-6)


def test_the_oracle_reports_infeasibility_as_an_answer():
    """An infeasible instance is a result, not a failure of the oracle."""
    infeasible = SOCP.unconstrained(np.array([1.0])).add_inequalities([[1.0], [-1.0]], [-1.0, -1.0])
    solution = reference.solve_reference(infeasible)
    assert not solution.is_optimal
    assert solution.objective == np.inf
    assert solution.z is None
    assert solution.agrees_with(np.inf), "both solvers calling it infeasible is agreement"


def test_the_oracle_reports_unboundedness_as_an_answer():
    """The other extended-value case, with the sign the convention implies."""
    solution = reference.solve_reference(SOCP.unconstrained(np.array([1.0])))
    assert not solution.is_optimal
    assert solution.objective == -np.inf
    assert solution.agrees_with(-np.inf)


# ----------------------------------------------------------------------------------
# The comparison itself: objective values, to a prescribed tolerance
# ----------------------------------------------------------------------------------


def test_the_prescribed_tolerance_is_the_documented_one():
    """§16.3's "prescribed numerical tolerance", named once so studies cite one number."""
    assert reference.OBJECTIVE_TOLERANCE == 1e-6


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (1.0, 1.0, 0.0),
        (0.0, 1e-9, 1e-9),  # absolute below unit scale
        (1e6, 1e6 + 1.0, 1e-6),  # relative above it
        (np.inf, np.inf, 0.0),  # both infeasible: agreement
        (-np.inf, -np.inf, 0.0),  # both unbounded: agreement
        (np.inf, -np.inf, np.inf),  # opposite verdicts: no tolerance absorbs that
        (1.0, np.inf, np.inf),  # one finite, one not
    ],
)
def test_the_relative_gap(first, second, expected):
    """Relative above unit scale, absolute below it, and symmetric in its arguments."""
    assert reference.relative_gap(first, second) == pytest.approx(expected)
    assert reference.relative_gap(second, first) == pytest.approx(expected)


def test_agreement_is_relative_to_the_objective_scale():
    """A portfolio objective of 1e-3 and one of 1e6 get the same treatment."""
    small = reference.ReferenceSolution(solver="test", status="optimal", objective=1e-3)
    large = reference.ReferenceSolution(solver="test", status="optimal", objective=1e6)
    assert small.agrees_with(1e-3 + 1e-9)
    assert not small.agrees_with(1e-3 + 1e-4)
    assert large.agrees_with(1e6 + 0.5)


def test_a_wider_tolerance_can_be_prescribed():
    """§12.4's ill-conditioned instances are where a reference solver loses digits."""
    solution = reference.ReferenceSolution(solver="test", status="optimal_inaccurate", objective=1.0)
    assert not solution.agrees_with(1.001)
    assert solution.agrees_with(1.001, tolerance=1e-2)


def test_an_inaccurate_status_still_counts_as_an_answer():
    """It is a reason to widen the tolerance, not a reason to treat the answer as absent."""
    assert reference.ReferenceSolution(solver="test", status="optimal_inaccurate", objective=1.0).is_optimal


def test_a_failed_solve_has_no_objective_to_compare_against():
    """Comparing against a solver that did not solve anything is not a check."""
    solution = reference.ReferenceSolution(solver="test", status="solver_error", objective=0.0)
    with pytest.raises(reference.SolverUnavailableError, match="nothing to compare"):
        solution.agrees_with(0.0)


# ----------------------------------------------------------------------------------
# The licensed backends, skipped cleanly
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", reference.LICENSED_BACKENDS)
def test_a_licensed_backend_works_through_the_same_interface(backend, golden):
    """§12.1's MOSEK and Gurobi, when a license happens to be present.

    Skipped rather than failed when it is not, which is the whole reason
    `SolverUnavailableError` is a named exception: an unlicensed machine must be able to run
    the suite, and an unlicensed *solve* is indistinguishable from an uninstalled one
    until it is attempted.
    """
    solver = reference.CvxpySolver(backend=backend)
    if not solver.is_available():
        pytest.skip(f"{backend} is not installed; install the '{backend.lower()}' extra")
    problem, objective = golden
    try:
        solution = solver.solve(problem)
    except reference.SolverUnavailableError as unlicensed:
        pytest.skip(f"{backend} is installed but unusable here: {unlicensed}")
    assert solution.agrees_with(objective, tolerance=1e-4)


# ----------------------------------------------------------------------------------
# Handing the oracle the wrong thing
# ----------------------------------------------------------------------------------


def test_a_non_socp_is_rejected_at_the_call():
    """MeanStdForm is an easy mistake, and CVXPY's complaint would be frames deep."""
    form, _ = GOLDEN["budget"]
    with pytest.raises(ProblemError, match="expected an SOCP"):
        reference.solve_reference(form)
