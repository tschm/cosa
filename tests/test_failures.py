"""§12.4's study: what COSA does on the hard instances, and which mitigation is why.

Issue #36. The study itself is `cosa.experiments.failures`, and its narrative is
`docs/development/failure-modes.md`. What is tested here is that the study measures what it
claims to and that its headline claims hold — so that the document, which is written from
this output, cannot quietly stop being true.

Every assertion is on a *classification* rather than on a number. The residual `basic`
achieves is not a property worth pinning; that it is certified, that nothing runs to the
iteration limit undiagnosed, and that equilibration is what rescues `badly scaled` are.
"""

import pytest

from cosa.experiments import failures
from cosa.solver import anticycling


@pytest.fixture(scope="module")
def outcomes():
    """One pass of the study, shared — thirty-nine solves is not worth repeating per test."""
    return failures.study(seeds=(0, 1, 2))


@pytest.fixture(scope="module")
def ablations():
    """One pass of the ablation, likewise."""
    return failures.ablate(seed=0)


# ----------------------------------------------------------------------------------
# Success Criterion 6: each family solves or has its failure mode documented
# ----------------------------------------------------------------------------------


def test_every_family_is_covered(outcomes):
    """Structured and robustness families alike.

    The structured ones are in the study because "which mitigation addressed which failure"
    is only meaningful next to instances that never needed one.
    """
    assert {outcome.family for outcome in outcomes} == set(failures.FAMILIES)


def test_nothing_stops_without_naming_a_reason(outcomes):
    """The strongest single claim in the study, and the only one that is a hard failure.

    `degenerate`, `stalled` and `blocked-at-apex` each name a specific thing that happened
    and each has an issue behind it. `iteration_limit` names nothing — it is the solver
    saying it does not know — and none of the thirty-nine solves ends there.
    """
    assert failures.undiagnosed(outcomes) == ()


def test_almost_nothing_revisits_a_working_set(outcomes):
    """#29's guarantee on the families, and where the guard actually earns its keep.

    Returning to a working set more often than `REVISITS` is what arms Bland's rule. On
    twelve of the thirteen families it never happens; on `badly scaled` it does, at up to
    nine returns — the same family whose conditioning makes it stall. That the guard arms
    exactly where the numerics are worst is the expected relationship, and that those solves
    still *terminate*, with a diagnosis rather than an iteration limit, is what the guard
    promises.
    """
    revisiting = failures.cycling(outcomes, threshold=anticycling.REVISITS)
    assert {outcome.family for outcome in revisiting} <= {"badly scaled"}
    assert all(outcome.verdict == "diagnosed" for outcome in revisiting)


def test_twelve_of_thirteen_families_solve(outcomes):
    """Thirty-six of thirty-nine, and every failure is the same family.

    This assertion has been wrong twice, in opposite directions, which is worth knowing
    before trusting it. It first said `badly scaled` stalls; then, after
    `raise_free_heads` was fixed, that everything solves. Both were self-certified: the
    study trusted §6's residuals and did not ask a reference solver. It should have —
    `badly scaled` terminates with every residual under `1e-11` at a point 3.4% away from
    the reference's, and the reference's point is feasible for COSA's own check to `1e-11`
    with a strictly better objective.
    """
    unsolved = [outcome for outcome in outcomes if outcome.verdict != "solved"]
    assert {outcome.family for outcome in unsolved} == {"badly scaled"}
    assert len(unsolved) == 3, "all three seeds"


def test_the_one_failure_is_a_wrong_answer_with_a_clean_certificate(outcomes):
    """The category this study exists to surface, and the worst one there is.

    A `diagnosed` stop is honest and an `undiagnosed` one is at least visible. This looks
    like success: the loop reports `optimal`, all five of §6's residuals are inside their
    tolerance, and the answer is wrong. Only the reference check catches it.
    """
    disagreeing = failures.wrong(outcomes)
    assert len(disagreeing) == 3
    for outcome in disagreeing:
        assert outcome.status == "optimal", "the loop is satisfied"
        assert outcome.certified, "and so are the residuals"
        assert outcome.gap is not None
        assert outcome.gap > 1e-3, "and the answer is wrong anyway"


def test_the_residual_is_small_and_the_answer_is_wrong_at_the_same_time():
    """Both statements are true, which is the whole difficulty.

    A convex problem cannot have an exactly satisfied KKT system at a suboptimal point, and
    the residual here is not exactly zero: it is around `2e-5` absolute, reported as `1e-11`
    because §14.2 normalizes stationarity by `max(1, |c|_inf)` and this instance's `|c|_inf`
    is `2e6`. So the residual is genuinely small in relative terms and the answer is
    genuinely wrong — a conditioning result rather than an arithmetic mistake, and one that a
    relative certificate cannot express.
    """
    import numpy as np

    from cosa.experiments import portfolio as generators
    from cosa.solver import cosa as solver

    instance = generators.badly_scaled(20, seed=0)
    answer = solver.solve(instance.problem)
    absolute = float(np.abs(answer.multipliers.stationarity_residual(instance.problem)).max())
    assert answer.residuals.stationarity < 1e-9, "relatively negligible"
    assert absolute > 1e-6, "and absolutely not"
    assert float(np.abs(instance.problem.c).max()) > 1e5, "which is what the normalizer divides by"


def test_a_solved_verdict_needs_a_reference_and_not_only_a_certificate(outcomes):
    """`solved` is three claims at once, and the third is the one that was missing.

    The loop must report optimal, §6's residuals must confirm it, *and* a reference solver
    must agree about the objective. "Nothing disagreed with me" and "a reference agreed with
    me" are different claims, and conflating them is what let a wrong answer read as solved.
    """
    for outcome in outcomes:
        if outcome.verdict == "solved":
            assert outcome.status == "optimal"
            assert outcome.certified
            assert outcome.gap is not None
            assert outcome.gap <= 1e-6


def test_without_a_reference_the_verdict_is_unchecked_rather_than_solved():
    """Skipping the check must not look like passing it."""
    subset = {name: failures.FAMILIES[name] for name in ("basic", "box")}
    blind = failures.study(20, seeds=(0,), families=subset, oracle=False)
    assert {outcome.verdict for outcome in blind} == {"unchecked"}
    assert all(outcome.gap is None for outcome in blind)


# ----------------------------------------------------------------------------------
# The apex and the rank-deficient covariance, which are not special cases here
# ----------------------------------------------------------------------------------


def test_the_apex_family_is_faster_than_the_ordinary_one(outcomes):
    """The apex is not a degenerate case being survived, it is the fast path.

    `large` builds a rank-`k` covariance over `n >> k` assets, so the minimum-risk portfolio
    has risk exactly zero and `lam > 0` takes it — its optimum *is* the apex. Asserting that
    #24's branch is *fast* is more useful than asserting it merely terminates.

    Compared on totals across seeds rather than on per-seed extremes, and with a margin. The
    first version of this test asserted that every apex solve beat every solve of three named
    families, which held locally with a margin of four iterations and failed in CI at
    `68 < 67`. A claim that a different BLAS can overturn is not the claim worth making: the
    real effect is a third of the work, and the totals show it without depending on which
    seed happened to land where.
    """
    apex = sum(o.iterations for o in outcomes if o.family == "large")
    ordinary = sum(o.iterations for o in outcomes if o.family == "basic")
    assert all(o.verdict == "solved" for o in outcomes if o.family == "large")
    assert apex < 0.85 * ordinary, (apex, ordinary)


def test_rank_deficiency_is_not_a_difficulty(outcomes):
    """Rank deficiency is not a difficulty for this formulation.

    A shorter `L` makes the cone smaller, not worse conditioned, and the KKT system does not
    inherit the covariance's conditioning at all — the tangent puts `L` in as one row.
    """
    for family in ("factor exposure", "highly correlated", "ill conditioned"):
        assert all(outcome.verdict == "solved" for outcome in outcomes if outcome.family == family)


def test_the_degenerate_optimum_solves_almost_immediately(outcomes):
    """The degenerate optimum solves almost immediately, and that is the repair working.

    Its active rows are dependent, §8.3 removes them, and the removal is exact — which is
    why the answer is better than the well-conditioned families' rather than worse.
    """
    degenerate = [outcome for outcome in outcomes if outcome.family == "degenerate optimum"]
    assert all(outcome.verdict == "solved" for outcome in degenerate)
    assert max(outcome.iterations for outcome in degenerate) < 10


# ----------------------------------------------------------------------------------
# The one failure, and the mitigation that addresses it
# ----------------------------------------------------------------------------------


def test_badly_scaled_spends_the_whole_budget_and_still_gets_it_wrong(outcomes):
    """Fourteen orders of magnitude in the constraint matrix, and the honest verdict.

    Earlier versions of this test asserted, in turn, that the family stalls and that it
    solves. It does neither: it terminates claiming optimality, having used its entire
    iteration budget, at a point the reference beats by several percent.
    """
    scaled = [outcome for outcome in outcomes if outcome.family == "badly scaled"]
    assert all(outcome.verdict == "wrong" for outcome in scaled)
    assert all(outcome.exhausted for outcome in scaled)


def test_no_mitigation_changes_an_outcome(ablations):
    """§12.4's fourth item, answered in the negative — including for the family that fails.

    Nothing rescues `badly scaled`. In particular §13.3's equilibration does not, which is
    the second time that question has been answered: it appeared to rescue the family when
    the family was stalling, and once the stall was traced to `raise_free_heads` the family
    ran to completion — with the wrong answer, equilibrated or not.
    """
    for ablation in ablations:
        assert ablation.with_it.verdict == ablation.without_it.verdict, ablation.mitigation


def test_equilibration_costs_iterations_and_buys_nothing(ablations):
    """§13.3's scaling has no family it rescues, on this formulation.

    Consistent with an earlier result rather than surprising given it: the KKT system does
    not inherit the covariance's conditioning, because the tangent representation puts `L`
    into it as a single row and one row has no spectrum. There is less conditioning here to
    fix than there looks to be.
    """
    scaling = [ablation for ablation in ablations if "equilibration" in ablation.mitigation]
    costlier = [ablation for ablation in scaling if ablation.with_it.iterations > ablation.without_it.iterations]
    assert len(costlier) > len(scaling) / 2, "equilibration is a net cost on most families"


def test_regularization_never_changes_an_outcome(ablations):
    """The right result rather than a disappointing one.

    The loop tries dependent-row *removal* first and falls back to regularization only when
    the dependency lies among rows it may not drop. On these families removal always
    succeeds, so the solver never has to answer a nearby question.
    """
    for ablation in ablations:
        if "regularization" in ablation.mitigation:
            assert ablation.with_it.verdict == ablation.without_it.verdict, ablation.family


def test_reuse_changes_no_verdict_by_design(ablations):
    """It is a cost policy, not a numerical one: the same direction by a different route.

    Agreeing to `1e-16` is not agreeing exactly, though, and over a few hundred iterations
    the two trajectories separate — so the *iteration* count does move, and the verdict
    does not.
    """
    for ablation in ablations:
        if "reuse" in ablation.mitigation:
            assert ablation.with_it.verdict == ablation.without_it.verdict, ablation.family


def test_reuse_does_change_the_factorization_count(ablations):
    """Which is the whole of what it promises, and is measured in `architecture.md`."""
    reuse = [ablation for ablation in ablations if "reuse" in ablation.mitigation]
    assert sum(a.with_it.factorizations for a in reuse) < sum(a.without_it.factorizations for a in reuse) / 10


# ----------------------------------------------------------------------------------
# The report
# ----------------------------------------------------------------------------------


def test_the_report_is_the_document():
    """`docs/development/failure-modes.md` is written from this, so it has to run."""
    text = failures.report(seeds=(0,))
    assert "failure-mode study" in text
    assert "verdicts:" in text
    assert "ablations that mattered:" in text
    assert "badly scaled" in text


def test_an_outcome_renders_its_own_row(outcomes):
    """One table row per solve, with the two caveats that are not the verdict."""
    rendered = {outcome.family: str(outcome) for outcome in outcomes}
    assert "solved" in rendered["basic"]
    assert "used the whole budget" in rendered["badly scaled"]


def test_an_ablation_renders_its_own_row(ablations):
    """Without and with, in that order, because the reader is asking what it bought."""
    rendered = str(next(a for a in ablations if "equilibration" in a.mitigation and a.family == "badly scaled"))
    assert "wrong -> wrong" in rendered
    assert "iters" in rendered


def test_the_report_says_that_nothing_mattered():
    """A study whose ablation section is empty must say so rather than print a blank.

    "None" is the current answer and it is a result, so the section is checked for content
    rather than for existence.
    """
    text = failures.report(seeds=(0,))
    section = text.split("ablations that mattered:")[1]
    assert "none" in section, "nothing changes an outcome any more, and the report says so"


# ----------------------------------------------------------------------------------
# The classification itself, on constructed outcomes
# ----------------------------------------------------------------------------------


def _outcome(status, *, certified=True, iterations=10, gap=0.0):
    """An outcome with a given status, for testing the classification directly."""
    return failures.Outcome(
        family="synthetic",
        status=status,
        residual=0.0 if certified else 1.0,
        iterations=iterations,
        factorizations=1,
        revisits=1,
        certified=certified,
        gap=gap,
    )


def test_the_five_verdicts_are_distinguished():
    """Constructed rather than found, because the study does not produce most of them.

    That is a good problem to have and a bad reason to stop testing the classification: it
    is the study's central instrument, and an instrument only exercised when something is
    broken is not one you can trust when something breaks.
    """
    assert _outcome("optimal").verdict == "solved"
    assert _outcome("optimal", gap=1.0).verdict == "wrong"
    assert _outcome("optimal", gap=None).verdict == "unchecked"
    assert _outcome("optimal", certified=False).verdict == "loose"
    assert _outcome("stalled").verdict == "diagnosed"
    assert _outcome("degenerate").verdict == "diagnosed"
    assert _outcome("blocked-at-apex").verdict == "diagnosed"
    assert _outcome("iteration_limit").verdict == "undiagnosed"


def test_a_wrong_answer_outranks_a_clean_certificate():
    """The order of the tests in `verdict`, asserted rather than left to reading order.

    An outcome that is certified *and* disagrees with the reference is `wrong`, not `solved`.
    Getting that precedence backwards is exactly the bug this classification was extended to
    catch.
    """
    assert _outcome("optimal", certified=True, gap=1.0).verdict == "wrong"


def test_only_the_iteration_limit_counts_as_undiagnosed():
    """`degenerate`, `stalled` and `blocked-at-apex` each have an issue behind them."""
    every = [_outcome(status) for status in ("optimal", "stalled", "degenerate", "iteration_limit")]
    assert [outcome.family for outcome in failures.undiagnosed(every)] == ["synthetic"]


def test_a_mitigation_can_matter_by_cost_alone():
    """The weak form: no verdict changed, but the work did.

    Worth reporting separately from the strong form. A mitigation that never changes an
    outcome but halves the work is still doing something, and one that does neither is not
    earning its place.
    """
    cheap = failures.Ablation(
        mitigation="synthetic",
        family="synthetic",
        with_it=_outcome("optimal", iterations=10),
        without_it=_outcome("optimal", iterations=100),
    )
    neither = failures.Ablation(
        mitigation="synthetic",
        family="synthetic",
        with_it=_outcome("optimal", iterations=10),
        without_it=_outcome("optimal", iterations=10),
    )
    changed = failures.Ablation(
        mitigation="synthetic",
        family="synthetic",
        with_it=_outcome("optimal"),
        without_it=_outcome("iteration_limit"),
    )
    assert cheap.mattered, "the weak form: same verdict, ten times the work"
    assert changed.mattered, "the strong form: the verdict itself moved"
    assert not neither.mattered


# ----------------------------------------------------------------------------------
# The oracle argument, and what happens when there is not one
# ----------------------------------------------------------------------------------


class _Unavailable:
    """A reference solver that is installed and never works.

    An unlicensed commercial backend behaves exactly like this: importable, and failing on
    every solve. The study must report `unchecked` rather than crash or claim agreement.
    """

    name = "unavailable"
    accuracy = 1e-9

    def is_available(self) -> bool:
        """It claims to be."""
        return True

    def solve(self, problem):
        """And then is not."""
        from cosa.experiments.reference import SolverUnavailableError

        raise SolverUnavailableError(self.name, "no license")


def test_a_broken_oracle_leaves_the_verdict_unchecked():
    """Not `solved`, and not a crash. Failing to verify is not the same as verifying."""
    subset = {name: failures.FAMILIES[name] for name in ("basic",)}
    blind = failures.study(20, seeds=(0,), families=subset, oracle=_Unavailable())
    assert [outcome.verdict for outcome in blind] == ["unchecked"]
    assert "no reference" in str(blind[0])


def test_no_oracle_installed_leaves_the_verdict_unchecked(monkeypatch):
    """A machine without the `reference` extra must still be able to run the study."""
    from cosa.experiments.reference import SolverUnavailableError

    def missing(*_args, **_kwargs):
        """Stand in for `default_solver` where nothing is installed."""
        raise SolverUnavailableError("none", "not installed")

    monkeypatch.setattr(failures, "default_solver", missing)
    subset = {name: failures.FAMILIES[name] for name in ("basic",)}
    blind = failures.study(20, seeds=(0,), families=subset, oracle=True)
    assert [outcome.verdict for outcome in blind] == ["unchecked"]


def test_an_oracle_can_be_passed_in_directly():
    """So a caller with a licensed backend can use it, and a test can use a stub."""
    from cosa.experiments.reference import default_solver

    subset = {name: failures.FAMILIES[name] for name in ("basic",)}
    checked = failures.study(20, seeds=(0,), families=subset, oracle=default_solver())
    assert [outcome.verdict for outcome in checked] == ["solved"]
