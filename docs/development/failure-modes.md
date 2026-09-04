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

Thirteen families, three seeds each, twenty assets. **Thirty-six solved, three wrong.**

| verdict | meaning | count |
| --- | --- | --- |
| `solved` | optimal, §6's residuals certify it, *and* a reference solver agrees | 36 |
| `wrong` | optimal and certified, and **disagreeing with the reference** | **3** |
| `loose` | the loop called it optimal, the residuals do not quite agree | 0 |
| `unchecked` | no reference was available, so nothing was verified | 0 |
| `diagnosed` | stopped, and named what happened | 0 |
| `undiagnosed` | the iteration limit, which names nothing | 0 |

**`wrong` is the category that matters, and it did not exist until the study stopped
certifying itself.** A `diagnosed` stop is honest and an `undiagnosed` one is at least
visible. A `wrong` answer looks like success: the loop reports `optimal`, all five of §6's
residuals are inside their tolerance, and the answer is several percent away from the
reference's. Only a cross-check catches it — which is what Success Criterion 5 has always
asked for, and what this study was not doing.

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
| badly scaled | **wrong** | 1e-11 residual, 3.4e-2 gap | 1000 (the whole budget) | see below |

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

## The one failure: `badly scaled`, and two wrong diagnoses before the right one

This section has had three conclusions. The first two were wrong, and how they were wrong is
more useful than the current one.

`badly scaled` puts a constraint matrix spanning fourteen orders of magnitude in front of the
solver — `|A|` around `1e-4`, `|G|` and `|c|` around `1e6`.

**First conclusion: a conditioning failure that scaling fixes.** The family stalled at a
residual around `5e-2`, and §13.3's Ruiz equilibration converted the stall into an apparent
optimum at `9.7e-7`. Clean before-and-after, and wrong.

**Second conclusion: an initialization bug, and it solves unaided.** Building the public
interface exposed that `raise_free_heads` — the routine that repairs conic feasibility by
setting `t = ||Lx||` — required the cone's head row to select its variable with a coefficient
of *exactly one*:

```python
if selected.size != 1 or row[selected[0]] != 1.0:
    return None
```

Nothing needs that; the head is `coefficient * t + h_head` and solving `head >= ||tail||` is
one division. But no *rescaled* instance satisfies it, so `_heads_are_free` reported `False`,
**the retraction was silently unavailable**, and an iterate on the cone's boundary could not
move at all. That is Risk 1 — the boundary-immobility result — arriving with its remedy
switched off, and equilibration had been "fixing" it by perturbing the head row's
coefficient. With the restriction removed the family runs to completion with every residual
under `1e-11`, which the study reported as `solved`.

**It was not solved.** That verdict was self-certified: the study trusted §6's residuals and
never asked a reference solver. It should have.

| | objective | feasibility of that point |
| --- | --- | --- |
| COSA | −0.00524 | worst `Az−b` = −2.5e-2, `\|Ez−d\|` = 1.6e-15, cone slack 0 |
| CLARABEL | **−0.03956** | worst `Az−b` = −5.0e-11, `\|Ez−d\|` = 2.2e-16, cone slack 1.5e-12 |

The reference's point is feasible for COSA's *own* feasibility check to `1e-11`, and its
objective is 7.5× better. The direction from COSA's point to it is a feasible descent
direction with derivative −0.0343. No tolerance argument survives that: a better feasible
point demonstrably exists.

### Why the residual does not notice

Both statements are true at once, and that is the whole difficulty.

- The stationarity residual there is **1.93e-05 absolute**.
- §14.2 reports it relative to the objective's scale, dividing by `max(1, |c|_inf)` — and
  `|c|_inf` is `2e6`. So it prints as **9.7e-12**.

A convex problem cannot have an *exactly* satisfied KKT system at a suboptimal point, and
this one does not: the residual is real. It is simply small enough, relative to data of size
`1e6`, that a relative certificate cannot distinguish it from zero — while the conditioning
amplifies it into a 3.4% objective error. **The certificate is relatively satisfied and the
answer is wrong, and neither of those is a mistake in the other.**

That is a statement about what a relative KKT residual can promise on an ill-conditioned
instance, and it is the most important thing in this document. It also means §6's tolerance
is not a sufficient stopping criterion on its own for instances of this kind — a finding for
[#22](https://github.com/tschm/cosa/issues/22) and
[#41](https://github.com/tschm/cosa/issues/41) rather than one this study can fix.

### And equilibration still does not help

Worth stating, since it was the first conclusion. Equilibrated, the same instance lands at
−0.00540 against the reference's −0.03956 — a gap of 3.4e-2, indistinguishable from the
unscaled solve's. Scaling changed the *appearance* of the residual and never the answer.

## Which mitigation addressed which failure

§12.4 asks for this specifically. A mitigation is measured by removing it and re-solving.

| mitigation | changed an outcome | changed the cost |
| --- | --- | --- |
| §8.3 regularization | no | no |
| §13.2 factorization reuse | no | factorizations 98.9% → 1.4%, iterations +30% |
| §13.3 Ruiz equilibration | no | iterations +30% to +80%, and one residual five orders worse |

**No mitigation changes an outcome**, including for the family that fails: nothing rescues
`badly scaled`, equilibration included.

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

- **`badly scaled` terminates claiming optimality at a wrong answer.** The open question is
  not the family, it is the criterion: a relative KKT residual cannot certify an instance
  whose data spans fourteen orders of magnitude, and §6 currently has no absolute floor.
  That belongs to [#22](https://github.com/tschm/cosa/issues/22).
- **Every "solved" verdict now requires a reference solver to agree**, so a study run without
  one reports `unchecked` rather than `solved`. The self-certified version of this study was
  wrong for two waves in two different directions.
- **`blocked-at-apex` on 3 of 200 randomized instances.** [#39](https://github.com/tschm/cosa/issues/39).
  Not a tolerance problem and not fixable by a working-set rule; the release the multiplier
  authorizes is arithmetically unavailable.
- **Slow convergence on 5 of 200 randomized instances**, which run to the iteration limit
  with one working set and no cycling. A convergence-rate question rather than a robustness
  one, and the natural place to look is the retraction: it is a first-order process on a
  curved boundary, and #23's curvature improved the *direction* without changing that.
