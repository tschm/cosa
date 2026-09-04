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


def test_the_only_family_that_revisits_is_the_one_that_stalls(outcomes):
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


def test_almost_everything_solves(outcomes):
    """Thirty-six of thirty-nine, and the three that do not are all one family."""
    unsolved = [outcome for outcome in outcomes if outcome.verdict != "solved"]
    assert {outcome.family for outcome in unsolved} == {"badly scaled"}


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


def test_badly_scaled_stalls_and_says_so(outcomes):
    """A diagnosed stop, not a silent wrong answer and not an iteration limit."""
    scaled = [outcome for outcome in outcomes if outcome.family == "badly scaled"]
    assert all(outcome.status == "stalled" for outcome in scaled)
    assert all(outcome.residual > 1e-3 for outcome in scaled), "it is genuinely far from optimal"


def test_equilibration_is_what_rescues_it(ablations):
    """§12.4's fourth item: which mitigation addressed which failure, with evidence.

    The evidence is a counterfactual — the same instance, solved with the mitigation and
    without — because that is the only form of the claim that can be wrong.
    """
    rescue = next(
        ablation
        for ablation in ablations
        if ablation.family == "badly scaled" and "equilibration" in ablation.mitigation
    )
    assert rescue.without_it.verdict == "diagnosed"
    assert rescue.with_it.verdict == "solved"
    assert rescue.mattered


def test_equilibration_is_not_free(ablations):
    """It converts a stall into a certified optimum and spends the whole budget doing it.

    Saying so is more useful than reporting "solved": what the mitigation establishes is
    that the failure is one of *conditioning* rather than of the algorithm, and the cost is
    part of that finding.
    """
    rescue = next(
        ablation
        for ablation in ablations
        if ablation.family == "badly scaled" and "equilibration" in ablation.mitigation
    )
    assert rescue.with_it.exhausted
    assert rescue.with_it.iterations > 10 * rescue.without_it.iterations


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
    """It is a cost policy, not a numerical one: the same direction by a different route."""
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
    assert "start supplied" in rendered["badly scaled"], "its cone head is not a free variable"


def test_an_ablation_renders_its_own_row(ablations):
    """Without and with, in that order, because the reader is asking what it bought."""
    rendered = str(next(a for a in ablations if "equilibration" in a.mitigation and a.family == "badly scaled"))
    assert "diagnosed -> solved" in rendered


def test_the_report_names_the_mitigation_that_mattered():
    """A study whose ablation section were empty would still print, and would say nothing.

    So the section is checked for content rather than for existence.
    """
    text = failures.report(seeds=(0,))
    section = text.split("ablations that mattered:")[1]
    assert "equilibration" in section
    assert "none" not in section
