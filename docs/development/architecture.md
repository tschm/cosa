# Architecture and layout decisions

The record of decisions taken in [#8](https://github.com/tschm/cosa/issues/8), so the
issues that follow do not each re-litigate them. The source of truth for *what* COSA is
remains [the project plan](../paper/paper.pdf); this page is only about how the code is
arranged and what it is built on.

## Package layout

Six subpackages under `src/cosa/`, taken from the project plan's Software Architecture
section:

| Package | Holds | Modules the plan names |
| --- | --- | --- |
| `problem` | Problem representation | `socp`, `portfolio` |
| `geometry` | Second-order cone geometry | `soc`, `tangent`, `step` |
| `active_set` | The working set and its updates | `working_set`, `multipliers`, `updates` |
| `linear_algebra` | KKT systems, factorizations, scaling | `kkt`, `factorization`, `scaling` |
| `solver` | The iteration itself | `cosa`, `initialization`, `termination` |
| `experiments` | Numerical studies | `portfolio`, `frontier`, `benchmarks` |

Every package exists, each with a docstring naming the modules it will own. That is
deliberate: it means each later issue has an unambiguous home, and the module names
above are authoritative even before the files exist. Landed so far: `problem/socp.py`,
`problem/portfolio.py`, `geometry/soc.py`, `geometry/tangent.py`,
`geometry/step.py`, `active_set/working_set.py`, `active_set/updates.py`,
`active_set/multipliers.py`, `linear_algebra/factorization.py`, `linear_algebra/kkt.py`, `linear_algebra/rank.py`,
`linear_algebra/scaling.py`, `solver/apex.py`, `solver/cosa.py`,
`solver/initialization.py`, `solver/instrumentation.py`, `solver/termination.py`,
`experiments/reference.py`, `experiments/portfolio.py` and `experiments/randomized.py`.
The rest are still names.

Five modules in the list above are **not** in the plan's table, and all five are the same
kind of omission: the plan describes the *algorithm* and the *studies*, and is silent
about the scaffolding both need.

- `solver/apex.py`, the branch of [#24](https://github.com/tschm/cosa/issues/24). §8.1
  calls the apex "a distinct direction computation inside the solver" without giving it a
  module. It cannot live in `geometry/`, which is where its two ingredients are, because
  it composes them with the working set and the KKT solve — it is the first module that
  reaches into all three lower subpackages, which is what "inside the solver" means.
- `linear_algebra/rank.py`, §8.3's rank detection and null-space route for
  [#25](https://github.com/tschm/cosa/issues/25). The plan's `factorization.py` is about
  *reuse* across iterations (#27); detecting a degenerate working set is a different job
  with a different lifetime, and it is needed several waves earlier.
- `solver/instrumentation.py`, the counters and invariants of
  [#15](https://github.com/tschm/cosa/issues/15). §11 and §12.3 promise thirteen measured
  quantities between them and §14 sets out two runtime invariants; the plan's `cosa.py`,
  `initialization.py` and `termination.py` are the algorithm, not its measurement.

- `experiments/reference.py`, the reference-solver oracle of
  [#21](https://github.com/tschm/cosa/issues/21). The plan puts solver comparison in
  `benchmarks`, but the oracle and the study are different things with different
  lifetimes -- §16.3 wants *every* generated problem cross-checked from M5 onward, while
  `benchmarks` is the M10 comparison table. Folding the oracle into `benchmarks` would
  have made every test that needs an oracle depend on the study module. `benchmarks`
  still belongs to [#34](https://github.com/tschm/cosa/issues/34).
- `experiments/randomized.py`, the seeded random generator of
  [#32](https://github.com/tschm/cosa/issues/32). §16.3 requires a comparison for "every
  randomly generated test problem" and never says where those come from; the six families
  in `experiments/portfolio.py` are fixed shapes, so no amount of reseeding makes them
  randomly generated in the sense the requirement means. The generator randomizes the
  *shape* -- dimension, rank, conditioning, active-set structure -- which is a different
  job from producing a named family, so it is a different module.

## One deviation from the plan: where tests live

The plan's architecture block nests the test modules **inside** the package:

```text
cosa/
    ...
    tests/
        test_soc.py
        test_step.py
        ...
```

This repository does not do that. Tests live in a **top-level `tests/`**, because
`pytest.ini` sets `testpaths = tests` and the packaging test asserts the src-layout.
Repo layout wins over the plan's sketch.

What survives from the plan is the *naming*: a module at `src/cosa/geometry/soc.py` is
tested by `tests/test_soc.py`, so the plan's five test-module names still say which test
file a given piece of work belongs in.

That convention needs one tiebreak, because the plan's own table names two modules
`portfolio` -- `problem.portfolio` and `experiments.portfolio`. A flat `tests/` directory
cannot hold two `test_portfolio.py`, so the subpackage breaks the tie: the problem
representation is tested by `tests/test_portfolio.py` and the instance families by
`tests/test_portfolio_families.py`.

Those are the only intentional divergences. Where the plan and the repository disagree
about anything else, treat it as a bug in one of them and say so on the issue.

## Numerical stack

**NumPy is declared now.** It is a hard dependency: `cosa.Vector` and `cosa.Matrix` are
NumPy array aliases, and they are the vocabulary every later module shares, so there is
no version of this project that does not import NumPy at the top level.

**SciPy is the recorded choice for the M7 work, and is deliberately not declared yet.**
The sparse `LDL^T` and QR factorizations, the null-space and range-space methods and the
rank detection that
[M7](https://github.com/tschm/cosa/milestone/7) needs are all reachable through
`scipy.linalg` and `scipy.sparse.linalg`, and that is the intended route. It is left
undeclared until something imports it for two reasons:

- `deptry` runs over `src/` on every CI run and fails a declared dependency that nothing
  imports (`DEP002`). Declaring SciPy today would mean either a red gate or an ignore
  entry that suppresses a real check for every future dependency too.
- The plan explicitly leaves the decision open -- *"The exact language and numerical
  libraries can be selected at the start of implementation"* -- and asks only that the
  option be recorded. This is that record.

The floor is `scipy>=1.11.2` rather than `>=1.11`, and the reason is the
lowest-direct-resolution gate: 1.11.1 predates Python 3.12 and has no wheel for it, so
`uv sync --resolution lowest-direct` tried to build it from source and the Cython step
failed in CI. 1.11.2 is the first release with `cp312` wheels, which mirrors why the NumPy
floor is 1.26. A dependency floor is a claim that the project works there, and that gate is
what checks the claim.

**That prediction came true at #25, exactly as written.** §8.3 asks for "QR-based rank
detection", pivoted QR is the one factorization NumPy does not have, and
`linear_algebra/rank.py` is the change that both imports SciPy and declares it. The
milestone was M7, as recorded.

The rest of the paragraph still stands: whichever issue first needs a factorization adds
`scipy` to `[project].dependencies` in the same change that imports it. That issue was
**not**
[#12](https://github.com/tschm/cosa/issues/12), which had the first real reason to want
one: its KKT solve uses `numpy.linalg.solve`'s dense LU, which handles the symmetric
indefinite saddle-point matrix correctly and is the least clever thing that is right --
which is exactly what §13.1 asks a reference implementation to be. The sparse `LDL^T` and
null-space methods are still ahead, and still SciPy's.

The dependent-row check in that solve is `numpy.linalg.matrix_rank`, an SVD per solve, and
it is there because the obvious alternative does not work: `numpy.linalg.solve` raises only
on an *exactly* zero pivot, so a genuinely degenerate working set produced a pivot of
`1e-18`, returned successfully, and handed back enormous garbage. #33's degenerate-optimum
family is what surfaced it. Paying for an SVD is the right trade in a module §13.1 asks to
be reliable rather than fast; #26 and #27 will need something cheaper. If a dedicated LDL package
turns out to beat SciPy, that is a decision for
[#26](https://github.com/tschm/cosa/issues/26), which compares the strategies; nothing
above prevents it.

A lower-level implementation, as the plan suggests, is a question for after the algorithm
is stable. Nothing here forecloses it.

**CVXPY and Clarabel are a test dependency and an optional extra, never a runtime
dependency.** They are how [#21](https://github.com/tschm/cosa/issues/21)'s oracle
reaches an open conic solver, and `cosa.experiments.reference` imports CVXPY *inside its
methods* so that the module -- and therefore the test suite -- imports without it. The
same `deptry` logic as above applies: nothing under `src/` imports CVXPY at module
scope, so it belongs in the `test` dependency group (where CI installs it) and in the
`reference` extra (where a user of the library opts into it), not in
`[project].dependencies`.

The `mosek` and `gurobi` extras exist for the same reason in reverse: §12.1 names both
as reference solvers, and both are license-gated, so the suite must run without them and
skip cleanly. That is what `SolverUnavailableError` is for -- and it earns its keep: the
gate installs every extra, so an unlicensed MOSEK is *present* in CI, and it fails in its
own exception type rather than CVXPY's.

**`hypothesis` is a test-group dependency**, added by
[#32](https://github.com/tschm/cosa/issues/32). It was already assumed: `pytest.ini`
registered a `property` marker that nothing used, and `rhiza-task hypothesis-test`
reported `no hypothesis/property tests collected` on every run. The property tests in
`tests/test_randomized.py` are what that gate was waiting for.

## Public surface

`cosa/__init__.py` exports the array aliases every module shares:

```python
from cosa import Matrix, Vector
```

Both are `numpy.typing.NDArray[numpy.float64]`. `Vector` annotates `x`, `mu`, `b`, `d`;
`Matrix` annotates `A`, `E`, `L`, `Sigma`. They are aliases rather than distinct types on
purpose -- the distinction they carry is for the reader, not for the type checker.

The surface is narrow by choice. Each module named in the table above extends `__all__`
as it lands, rather than reserving names in advance for code that does not exist.
`problem/socp.py` was the first to do so, adding `SOCP`, `MeanStdForm`,
`SecondOrderCone`, `ConeProduct`, `ProblemError`, `SignConvention` and
`SIGN_CONVENTION`; `problem/portfolio.py` adds `MeanStdPortfolio`, `geometry/soc.py`
adds `ConePosition`, `geometry/tangent.py` adds `ApexError`,
`active_set/working_set.py` adds `WorkingSet`, `ConeStatus` and `ConstraintNames`, and
`linear_algebra/kkt.py` adds `Direction`, `RowLayout` and `SingularKktError`,
`active_set/multipliers.py` adds `Multipliers`, and `linear_algebra/scaling.py` adds
`Scaling`.

Two rules decide what gets in, and both are asserted in `tests/test_layout.py` so that
drifting from either is a deliberate edit rather than a side effect.

**The root holds types, not routines.** A module's functions stay in the module, reached
as `cosa.geometry.soc.is_boundary`, `cosa.geometry.tangent.tangent_row` or
`cosa.active_set.updates.removal_candidate`. The reason is legibility rather than
tidiness: `cosa.slack` and `cosa.position` say nothing about what they are the slack or
the position *of*, while `cosa.geometry.soc.slack` says it exactly.

The same test applies to types whose names are generic. `Recorder`, `Metrics` and
`InvariantChecker` all cross subpackage boundaries and would qualify on that ground, but
`cosa.Recorder` says nothing about what is being recorded, so they stay in
`solver.instrumentation`.

**The root holds the library, not the harness.** The algorithm's vocabulary is at the
root -- the problem, the working set, the cone's position and status, the direction and
its row layout, and the two errors a solver loop has to catch. All of `cosa.experiments`
stays where it is: its instance families, its random specifications and its
reference-solver oracle are how the library is exercised, not part of what it offers, and
a `cosa.PortfolioInstance` would suggest otherwise.

## `H` is not `rho*I`

§4.2 writes the direction subproblem's matrix as `H = rho*I`, and Waves 4–6 used exactly
that. [#23](https://github.com/tschm/cosa/issues/23) replaces it with the Hessian of the
Lagrangian, `rho*I + sum_j mu_j * grad^2 g_j`, where `g_j` is the `j`-th cone constraint
and `mu_j` its multiplier. Three consequences are worth stating where they can be found.

**Where the second derivative lives.** In `geometry/tangent.py`, next to the first one,
rather than in a `geometry/curvature.py` of its own. Both are the local geometry of the
same object at the same point, both refuse the apex for the same reason, and splitting
them would mean two modules that must agree about what `u` is. The module is named for
what it was first used for, not for the whole of what it holds.

**Why this is not "an SQP method with an SOC constraint",** which §3.3 explicitly warns
against. An SQP method would linearize the conic constraint into a *general* nonlinear
program and hand the result to a QP solver. What happens here is narrower: the constraint
stays a cone, the working set stays conic — a factor is `INACTIVE`, `TANGENT` or `APEX`,
never a row of a linearized system — and the curvature is exact rather than an
approximation to a Hessian nobody can write down. The Lagrangian Hessian is how a
*multiplier* enters a *primal* computation, and that coupling is what "primal-dual" names.

**What it costs and what it buys.** It costs one `n × n` symmetric update per iteration,
formed from a matrix the assembly already has, and nothing in the factorization: the
`(1, 1)` block was dense before and is dense after. Across the eleven portfolio families
it roughly halves the iteration count, and on the `ill_conditioned` family it is the
difference between an answer and the iteration limit. The measurement is
`tests/test_conic_logic.py`.

The multipliers it needs are the previous iteration's, which makes the scheme a fixed
point rather than an implicit system. The first iteration uses zero, so a #23 solve and a
Wave 6 solve begin identically and every difference between them is attributable.

## The other decision recorded once

The sign convention for the conic KKT conditions lives on
[its own page](sign-convention.md), for the same reason this one exists: the plan defers
it to the implementation, and four later issues consume it. Read it before touching
multipliers, residuals or the working set.
