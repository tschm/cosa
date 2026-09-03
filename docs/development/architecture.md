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

Every package exists now and is empty apart from a docstring naming the modules it will
own. That is deliberate: it means each later issue has an unambiguous home, and the
module names above are authoritative even though the files do not exist yet.

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

This is the only intentional divergence. Where the plan and the repository disagree
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

So: whichever issue first needs a factorization adds `scipy` to
`[project].dependencies` in the same change that imports it. If a dedicated LDL package
turns out to beat SciPy, that is a decision for
[#26](https://github.com/tschm/cosa/issues/26), which compares the strategies; nothing
above prevents it.

A lower-level implementation, as the plan suggests, is a question for after the algorithm
is stable. Nothing here forecloses it.

## Public surface

`cosa/__init__.py` exports exactly two names today:

```python
from cosa import Matrix, Vector
```

Both are `numpy.typing.NDArray[numpy.float64]`. `Vector` annotates `x`, `mu`, `b`, `d`;
`Matrix` annotates `A`, `E`, `L`, `Sigma`. They are aliases rather than distinct types on
purpose -- the distinction they carry is for the reader, not for the type checker.

The surface is narrow by choice. Each module named in the table above extends `__all__`
as it lands, rather than reserving names in advance for code that does not exist.
