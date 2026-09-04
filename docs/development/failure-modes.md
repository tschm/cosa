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

Thirteen families, three seeds each, twenty assets. **All thirty-nine solved.**

| verdict | meaning | count |
| --- | --- | --- |
| `solved` | optimal, and §6's five residuals certify it | 39 |
| `loose` | the loop called it optimal, the residuals do not quite agree | 0 |
| `diagnosed` | stopped, and named what happened | 0 |
| `undiagnosed` | the iteration limit, which names nothing | 0 |

The distinction between the last two is still the point of the study, even though both
columns are now empty. `degenerate`, `stalled` and `blocked-at-apex` each name a specific
thing that happened and each has an issue behind it; `iteration_limit` is the solver saying
it does not know. Keeping them apart is what let the one failure below be diagnosed
correctly — and the first diagnosis was wrong.

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
| badly scaled | solved | 1e-11 | 1000 (the whole budget) | see below |

### Behaviour at the apex

`large` and `factor exposure` build a rank-`k` covariance over `n >> k` assets, so `L` has
an `(n-k)`-dimensional null space, the minimum-risk portfolio has risk exactly zero, and
`lam > 0` takes it. **Their optima are at the apex, and nobody designed them that way** —
it is a consequence of low-rank covariance, which is how factor models are built.

`large` is the *fastest* family in the table, at 33 iterations. That is worth stating
plainly because the apex is where the tangent representation has nothing to say and where
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

## The failure that was not what it looked like: `badly scaled`

This section had a different conclusion two waves ago, and how it changed is more useful
than what it now says.

`badly scaled` puts a constraint matrix spanning fourteen orders of magnitude in front of
the solver. It used to **stall** at all three seeds, with residuals around `5e-2`, and
§13.3's Ruiz equilibration rescued it — `stalled` at `5e-2` became `optimal` at `9.7e-7`.
That is a clean before-and-after and it pointed at conditioning, so equilibration became the
mitigation for this family and, briefly, the default for the public interface.

**It was the wrong diagnosis.** Building that public interface exposed the real cause.
`raise_free_heads` — the routine that repairs conic feasibility by setting `t = ||Lx||` —
required the cone's head row to select its variable with a coefficient of *exactly one*:

```python
if selected.size != 1 or row[selected[0]] != 1.0:
    return None
```

Nothing needs that. The head is `coefficient * t + h_head` and solving `head >= ||tail||`
for `t` is one division whatever the coefficient is. But `badly_scaled`'s head row does not
have a unit coefficient, so the routine refused, `_heads_are_free` reported `False`, **the
retraction was silently unavailable**, and an iterate on the cone's boundary could not move
at all. That is not a conditioning failure. It is Risk 1 — the boundary-immobility result —
arriving with its remedy switched off.

With the restriction removed:

| | verdict | residual | iterations |
| --- | --- | --- | --- |
| as given | `optimal` | **1e-11** | 5000 |
| equilibrated | `optimal` | 9.7e-7 | 5000, and one seed hits the limit |

The family solves to eleven digits without help, and **equilibrating makes it five orders of
magnitude worse.** Equilibration was never fixing the conditioning; it was perturbing the
head row's coefficient, which is a different thing entirely and worked by accident.

Three things follow.

- **The public interface does not equilibrate.** `solve_portfolio(..., scale=False)` is the
  default; the flag remains for a caller whose units are genuinely pathological.
- **§13.3's scaling has no family it rescues.** It costs between 30% and 80% more iterations
  on every family and changes no verdict. That is a real result about this formulation and
  it is consistent with an earlier one: the KKT system does not inherit the covariance's
  conditioning, because the tangent puts `L` in as a single row. There is less conditioning
  here to fix than there looks to be.
- **A "diagnosed" stop is a hypothesis, not a conclusion.** `stalled` named the symptom
  correctly and the study attributed it to the wrong cause, for two waves, with a
  before-and-after table apparently confirming it. The ablation was sound; what it measured
  was a coincidence.

`badly scaled` still costs the entire iteration budget, which is the honest remaining
caveat: it converges, and slowly.

## Which mitigation addressed which failure

§12.4 asks for this specifically. A mitigation is measured by removing it and re-solving.

| mitigation | changed an outcome | changed the cost |
| --- | --- | --- |
| §8.3 regularization | no | no |
| §13.2 factorization reuse | no | factorizations 98.9% → 1.4%, iterations +30% |
| §13.3 Ruiz equilibration | no | iterations +30% to +80%, and one residual five orders worse |

**No mitigation changes an outcome any more**, which is a stronger statement than the one
this table used to make. Everything solves without help.

Two more things it says that are easy to miss.

**§8.3's regularization never changes an outcome, and that is the right result rather than a
disappointing one.** The loop tries dependent-row *removal* first and falls back to
regularization only when the dependency lies among rows it may not drop — equalities, and
the cone's own rows. On these families removal always succeeds. Regularization is the
answer to a nearby question and it is correct that the solver never has to ask one.

**§13.2's reuse changes no verdict by design, and does change the iteration count.** It is a
cost policy, not a numerical one: it computes the same direction by a different route,
agreeing to `1e-16` at any given point. But agreeing to `1e-16` is not agreeing exactly, and
over a few hundred iterations the two trajectories separate — about 30% more iterations
under reuse on these families, against 98.9% fewer factorizations. Both halves are in
[architecture.md](architecture.md#what-reuse-turned-out-to-be-worth).

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

- **`badly scaled` converges, and spends the whole iteration budget doing it.** It is no
  longer a robustness failure; it is the slow-convergence question below wearing a different
  hat.
- **`blocked-at-apex` on 3 of 200 randomized instances.** [#39](https://github.com/tschm/cosa/issues/39).
  Not a tolerance problem and not fixable by a working-set rule; the release the multiplier
  authorizes is arithmetically unavailable.
- **Slow convergence on 5 of 200 randomized instances**, which run to the iteration limit
  with one working set and no cycling. A convergence-rate question rather than a robustness
  one, and the natural place to look is the retraction: it is a first-order process on a
  curved boundary, and #23's curvature improved the *direction* without changing that.
