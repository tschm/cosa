# Failure modes and degeneracy

What COSA does on the hard instances, why, and which mitigation is responsible. This is
[#36](https://github.com/tschm/cosa/issues/36) and §12.4 of the plan, and it exists to
answer Success Criterion 7's neighbour — *"COSA's numerical behaviour on degenerate and
ill-conditioned problems is understood"*.

**Every number here is generated.** `cosa.experiments.failures.report()` produces this
table; nothing below is typed by hand. A claim about numerical behaviour that is written
down once and never re-run stops being true silently, which is the failure mode this
document is most at risk of.

```python
from cosa.experiments import failures

print(failures.report())
```

## The result

Thirteen families, three seeds each, twenty assets. **Thirty-six solved, three diagnosed,
none undiagnosed.**

| verdict | meaning | count |
| --- | --- | --- |
| `solved` | optimal, and §6's five residuals certify it | 36 |
| `loose` | the loop called it optimal, the residuals do not quite agree | 0 |
| `diagnosed` | stopped, and named what happened | 3 |
| `undiagnosed` | the iteration limit, which names nothing | 0 |

The distinction between the last two is the point of the study. `degenerate`, `stalled` and
`blocked-at-apex` each name a specific thing that happened and each has an issue behind it;
`iteration_limit` is the solver saying it does not know. That none of the thirty-nine solves
ends there is the strongest single statement in this document.

## Per family

| family | verdict | residual | iterations | notes |
| --- | --- | --- | --- | --- |
| basic | solved | 5e-11 | 90 | |
| box | solved | 8e-11 | 48 | |
| sector | solved | 5e-11 | 90 | |
| factor exposure | solved | 2e-10 | 139 | rank-deficient `Sigma` |
| turnover | solved | 2e-10 | 336 | most iterations of the structured families |
| large | solved | 7e-11 | 33 | **optimum at the apex** |
| nearly redundant | solved | 9e-11 | 49 | |
| highly correlated | solved | 1e-15 | 40 | |
| ill conditioned | solved | 9e-11 | 174 | `cond(Sigma)` around `1e10` |
| nearly active cone | solved | 5e-11 | 90 | §8.2's band |
| degenerate optimum | solved | 2e-16 | 4 | dependent active set |
| many active bounds | solved | 1e-10 | 42 | |
| badly scaled | **stalled** | 5e-2 | 29 | see below |

### Behaviour at the apex

`large` and `factor exposure` build a rank-`k` covariance over `n >> k` assets, so `L` has
an `(n-k)`-dimensional null space, the minimum-risk portfolio has risk exactly zero, and
`lam > 0` takes it. **Their optima are at the apex, and nobody designed them that way** —
it is a consequence of low-rank covariance, which is how factor models are built.

`large` is consistently *faster* than the basic family rather than harder — 179 iterations
against 253 across three seeds, a saving of 29%. Stated on totals rather than on a single
seed: the first version of this claim compared per-seed extremes, held locally by four
iterations, and failed in CI at `68 < 67`. That the apex is cheap is worth stating plainly
because it is where the tangent representation has nothing to say and where
[#24](https://github.com/tschm/cosa/issues/24)'s branch takes over: exact membership on the
direction, the normal cone on the multiplier. On these instances the branch is not a
degenerate case being survived, it is the fast path.

The apex failure that does occur is `blocked-at-apex`, and it does not appear here at all —
it appears on §16.3's randomized generator, at three instances in two hundred. What happens
is that the branch cannot justify holding the apex and cannot release it either: on eq. (7)
the released direction is infeasible by arithmetic, because `t` appears in exactly two
places and dropping the factor forces `d_t = -lam/rho < 0`. That is
[#39](https://github.com/tschm/cosa/issues/39), it is a genuine limitation, and the solver
reports it rather than looping.

### Behaviour on rank-deficient `Sigma`

Covered above and by `highly correlated`, whose residual is at rounding level. Rank
deficiency turns out not to be a difficulty for this formulation: `covariance_factor`
handles any `Sigma ⪰ 0` by eigendecomposition and simply produces a shorter `L`, and a
shorter `L` makes the cone *smaller*, not worse conditioned. The KKT system does not inherit
the covariance's conditioning at all — the tangent representation puts `L` into the system as
a **single row**, and one row has no spectrum.

### Behaviour on degenerate optima and dependent active sets

`degenerate optimum` solves in **four iterations** with a residual at rounding level, which
looks like the study failing to find anything hard. It is not: the family's difficulty is
that the active rows at the optimum are linearly dependent, so the KKT matrix is singular
and the multipliers are not determined. §8.3's dependent-row removal repairs the working set
before the direction solve, and the repair is exact — which is why the answer is *better*
than the well-conditioned families' rather than worse.

This family is also the one that found the bug: [#12](https://github.com/tschm/cosa/issues/12)'s
singular-KKT guard never fired, because LAPACK raises only on an exactly zero pivot and a
dependent set gives `1e-18`. The guard became an explicit rank test. The family existed for
about ten minutes before it earned its place.

## The one failure: `badly scaled`

The only family that does not solve, at all three seeds, with residuals around `5e-2`
to `8e-2`. It stalls: the retraction can find no step that improves the objective, and the
multiplier tests have nothing to drop.

**§13.3's Ruiz equilibration is the mitigation, and the evidence is before-and-after.**

| | verdict | residual | iterations |
| --- | --- | --- | --- |
| as given | `stalled` | 5.0e-2 | 29 |
| equilibrated | `optimal` | 9.7e-7 | 1000 (the whole budget) |

`badly scaled` is also the only family where [#29](https://github.com/tschm/cosa/issues/29)'s
anti-cycling guard actually arms: at seeds 1 and 2 the solve returns to a single working set
nine and four times respectively, past the threshold that switches §7.2's most-violating rule
for Bland's. Twelve of the thirteen families never return to a working set at all. That the
guard arms exactly where the conditioning is worst is the expected relationship — and those
solves still terminate, with a diagnosis rather than an iteration limit, which is what the
guard promises.

Equilibration converts a stall into a certified optimum. It is not a comfortable win — the
solve uses its entire iteration budget and lands just inside §6's `1e-6` tolerance — and
saying so is more useful than reporting "solved". What it establishes is that the failure is
one of *conditioning* rather than of the algorithm: the same iteration, on the same problem
written in better units, converges.

This also settles where [#28](https://github.com/tschm/cosa/issues/28)'s scaling earns its
keep. Not on ill-conditioned covariances — `ill conditioned` has `cond(Sigma) ≈ 1e10` and
solves without help, for the single-row reason above. On **unit mismatch**, where the
constraint matrix's entries span fourteen orders of magnitude and the equilibration brings
the KKT condition number from `2e14` to about `10`.

## Which mitigation addressed which failure

§12.4 asks for this specifically. A mitigation is measured by removing it and re-solving.

| mitigation | changed an outcome | changed the cost |
| --- | --- | --- |
| §13.3 Ruiz equilibration | **yes** — `badly scaled`, `stalled` → `optimal` | slower, and worth it |
| §8.3 regularization | no | no |
| §13.2 factorization reuse | no | factorizations 98.9% → 1.4% of solves |

Two things this table says that are easy to miss.

**§8.3's regularization never changes an outcome, and that is the right result rather than a
disappointing one.** The loop tries dependent-row *removal* first and falls back to
regularization only when the dependency lies among rows it may not drop — equalities, and
the cone's own rows. On these families removal always succeeds. Regularization is the
answer to a nearby question and it is correct that the solver never has to ask one.

**§13.2's reuse changes no verdict by design.** It is a cost policy, not a numerical one: it
computes the same direction by a different route, agreeing to `1e-16`. Its effect is the
factorization counter, and it is measured in
[architecture.md](architecture.md#what-reuse-turned-out-to-be-worth) rather than here.

## What is not ablated, and why

Four things that protect this solver cannot be switched off through the public interface,
and the reason is the same in each case: a solver without them is not a worse solver but a
different one that does not converge, and an ablation that always reports "iteration limit"
measures nothing.

- **#23's Lagrangian curvature.** Measured where it landed: across the eleven families
  9738 iterations to 4691, with four `ill conditioned` instances going from the iteration
  limit to an answer.
- **#29's anti-cycling guard and no-progress rule.** Measured where they landed: the worst
  randomized instance's returns to a single working set, 486 to 2.
- **#18's exact conic step** and **#20's retraction**, which are the step, not a policy
  about it.

Their before-and-after evidence lives in `tests/test_conic_logic.py` and
`tests/test_anticycling.py`, next to the code each one changed.

## Open

- **`badly scaled` needs equilibration to be applied**, and the solver does not apply it.
  Whether it should is [#37](https://github.com/tschm/cosa/issues/37)'s question, since it
  is a decision about the public interface rather than about the algorithm.
- **`blocked-at-apex` on 3 of 200 randomized instances.** [#39](https://github.com/tschm/cosa/issues/39).
  Not a tolerance problem and not fixable by a working-set rule; the release the multiplier
  authorizes is arithmetically unavailable.
- **Slow convergence on 5 of 200 randomized instances**, which run to the iteration limit
  with one working set and no cycling. A convergence-rate question rather than a robustness
  one, and the natural place to look is the retraction: it is a first-order process on a
  curved boundary, and #23's curvature improved the *direction* without changing that.
