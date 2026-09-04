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


def test_everything_solves(outcomes):
    """All thirty-nine, which is a stronger claim than this file used to make.

    `badly scaled` used to stall; the cause turned out to be `raise_free_heads` refusing any
    cone head row whose coefficient was not exactly one, which made the retraction silently
    unavailable. See `docs/development/failure-modes.md` — the interesting part is that the
    first diagnosis was wrong and had an ablation apparently confirming it.
    """
    assert [outcome for outcome in outcomes if outcome.verdict != "solved"] == []


def test_a_solved_verdict_means_the_residuals_agree(outcomes):
    """`solved` is a certificate, not a status string.

    §6's five residuals are what say so, which is the distinction between `solved` and
    `loose` — a loop that called a point optimal on a criterion the residuals do not confirm.
    """
    for outcome in outcomes:
        if outcome.verdict == "solved":
            assert outcome.status == "optimal"
            assert outcome.certified


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


def test_badly_scaled_converges_and_spends_the_whole_budget(outcomes):
    """The honest remaining caveat: it converges, to eleven digits, and slowly.

    Fourteen orders of magnitude in the constraint matrix is not a conditioning failure for
    this formulation — the KKT system does not inherit it, because the tangent puts `L` in as
    a single row. What it costs is iterations.
    """
    scaled = [outcome for outcome in outcomes if outcome.family == "badly scaled"]
    assert all(outcome.verdict == "solved" for outcome in scaled)
    assert all(outcome.residual < 1e-9 for outcome in scaled)
    assert all(outcome.exhausted for outcome in scaled)


def test_no_mitigation_changes_an_outcome(ablations):
    """§12.4's fourth item, answered in the negative — which is the stronger answer.

    Everything solves without help. The counterfactual is still the only form of the claim
    that can be wrong, and it is still worth running: it is what caught the earlier version
    of this file asserting that equilibration rescued `badly scaled`, on evidence that was
    sound and a conclusion that was not.
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
    assert "solved -> solved" in rendered
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


def _outcome(status, *, certified=True, iterations=10):
    """An outcome with a given status, for testing the classification directly."""
    return failures.Outcome(
        family="synthetic",
        status=status,
        residual=0.0 if certified else 1.0,
        iterations=iterations,
        factorizations=1,
        revisits=1,
        certified=certified,
    )


def test_the_four_verdicts_are_distinguished():
    """Constructed rather than found, because the study no longer produces three of them.

    That is a good problem to have and a bad reason to stop testing the classification: the
    distinction between a stop that names what happened and one that does not is the study's
    central instrument, and an instrument that is only exercised when something is broken is
    not one you can trust when something breaks.
    """
    assert _outcome("optimal").verdict == "solved"
    assert _outcome("optimal", certified=False).verdict == "loose"
    assert _outcome("stalled").verdict == "diagnosed"
    assert _outcome("degenerate").verdict == "diagnosed"
    assert _outcome("blocked-at-apex").verdict == "diagnosed"
    assert _outcome("iteration_limit").verdict == "undiagnosed"


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
