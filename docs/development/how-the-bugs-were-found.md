# How the bugs were actually found

Not a list of bugs. A list of the *mechanisms* that surfaced them, because every one of them
was found the same way and none of them was found by reading.

This page exists because it is the most transferable thing the project produced. The
algorithmic results are in [failure-modes.md](failure-modes.md), the architectural ones in
[architecture.md](architecture.md), and the mathematical ones in the paper. What is here is
the observation that ties them together:

> **Every claim this project made about itself that was not cross-checked against something
> external turned out to be wrong at least once.**

Eleven instances follow. They are worth reading as a set rather than individually, because
the pattern is what matters: in each case the code was doing something wrong, the test suite
was green, and the thing that noticed was a *different kind of check* rather than a more
careful one of the same kind.

## Silent failure is the default failure mode

**A guard that never fired.** `kkt.solve` raised `SingularKktError` on a dependent working
set — except LAPACK raises only on an *exactly* zero pivot, and a dependent set gives
`1e-18`. The guard had never once executed. Found by the degenerate-optimum instance family,
within minutes of that family existing. The fix was an explicit rank test.

**A test family that wasn't testing anything.** The box-constrained family's bound was loose
enough that its optimum was bit-for-bit identical to the unconstrained family's. Two
"different" families, one problem.

**A pathology that wasn't pathological.** The nearly-redundant family perturbed a row's
*magnitude*, which leaves the pair exactly parallel rather than nearly so. Found by rank
detection disagreeing with the docstring that described what it should find.

**A step guard exempting the case it existed for.** A version of the ratio test skipped the
conic interval for a strictly interior cone, reasoning that the linear interval already
bounded the step. It does bound it — not by enough to stay inside the cone. Found by the
per-iterate invariant checker, which is why that checker was built four waves before the
experiment that consumes its metrics.

**A metric that was zero for four waves.** `Metrics.runtime` was `0.0` for every solve ever
recorded. `solve` builds its `Solution` — and so calls `recorder.metrics()` — from *inside*
the `with recorder.solving()` block, so a runtime latched in that block's `finally` was not
yet set when anybody read it. Peak memory had the same shape of bug. The benchmark study was
the first consumer with a reason to look at the number, and it looked wrong immediately.

**A calibration that held on the seeds I chose.** A cross-solver accuracy tolerance held over
200 seeds and failed on one I had not picked. Found by a property-based test, which picks the
seeds you would not.

**Three that only CI could catch:** a dependency floor asserted rather than resolved; a LAPACK
behaviour measured on one platform and documented as *the* behaviour; and a termination
tolerance set exactly at its own achievable floor.

**An assertion with a four-iteration margin.** `assert max(apex) < min(others)` held locally
by four iterations and failed in CI at `68 < 67`. A claim a different BLAS can overturn is not
the claim worth making — the real effect was a third of the work, which totals across seeds
show without depending on which seed landed where.

## Three that were mine rather than the code's

Worth separating, because they are failures of *process* and the process is the thing a
reader can copy.

**A `--amend` after a hook-aborted commit.** A pre-commit hook reformatted files and aborted
the commit; the follow-up `git commit --amend` therefore amended the *previous* wave's commit,
folding an entire wave into it under a message naming the wrong issues. Found by the next
wave's pull request reporting a conflict — not by anything local, because locally everything
was consistent.

**A fix whose reasoning was void.** A CI timeout came from tests that each performed several
expensive runs. The first fix made those runs module-scoped fixtures, *ordered* so that each
was requested first by a different test. That reasoning does not survive `pytest -n auto`,
which the CI suite uses: xdist distributes tests across workers in an order nothing in the
file controls, so one worker's setup built all of them anyway. It then passed one job and
failed another **at the same commit**, which is what a boundary-condition timeout looks like.
The real fix was to make the studies *parameterizable* so a small run is genuinely small.

**A verification that verified nothing.** The local check of that second fix ran
`pytest -n auto` with the plugin not installed. pytest exits with `unrecognized arguments: -n`
and the run does nothing. It was noticed only because the output looked wrong.

## The one that succeeded silently

The worst of the eleven, and the reason this page leads with the sentence it does.

The failure-mode study classified every instance family and reported thirty-nine of
thirty-nine solved. It reached that verdict by consulting the solver's own KKT residuals —
so `solved` meant *"COSA is satisfied with itself"*. On twelve of thirteen families that is
the same thing. On the thirteenth the solver terminates reporting `optimal`, with all five
residuals under `1e-11`, at a point a reference solver beats by 3.4% — and the reference's
point is feasible for COSA's *own* feasibility check to `1e-11`.

Success Criterion 5 had asked for agreement with a reference on *every* generated problem
from the start. The study was not asking.

**And it confirmed two contradictory hypotheses in succession, each with sound evidence.**
The same family was first diagnosed as a conditioning failure that scaling fixes — with a
clean before-and-after table — and then, after the real cause of its stall was found, as
solving unaided. Both diagnoses were self-certified. Both were wrong.

> A study that certifies itself will confirm whichever hypothesis it started with.

The fix was not a better residual. It was a fifth verdict, `wrong`, meaning *optimal,
certified, and disagreeing* — and a fourth, `unchecked`, for when no reference is available,
because "nothing disagreed with me" and "a reference agreed with me" are different claims and
collapsing them is what caused this.

## What actually did the finding

Ranked by how much they earned, which is not the order anyone would guess:

1. **A reference solver**, consulted per instance rather than per benchmark. It is the only
   mechanism on this list that can catch a wrong answer with a clean certificate.
2. **Per-iterate invariant assertions** — the feasibility and stationarity checkers. Off by
   default so a benchmark pays nothing, on in the whole test suite.
3. **CI on hardware that is not mine**, three Python versions and a different BLAS. Four of
   the eleven were platform-dependent or margin-dependent and could not fail locally.
4. **Property-based tests**, for the seeds a person would not choose.
5. **Instance families built to be pathological**, each asserting its own pathology so that a
   family which stops being hard fails rather than quietly passing.
6. **Counterfactual ablation** — solving with a mitigation removed. Sound, and note that it
   is the mechanism that *confirmed* the wrong diagnosis twice: an ablation measures what
   changed, not why.

The ordering has a moral. The mechanisms that found the most were the ones that compared
against something the code could not influence. The ones that found least were the ones that
asked the code about itself, however rigorously.
