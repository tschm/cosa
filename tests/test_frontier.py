"""§11's principal experiment: does warm starting a frontier pay?

Issue #35. The sequence is `min -mu.T @ x + lam_k * sigma(x)` for `lam_1 ... lam_N`, solved
twice — cold, and each point warm started from the last. §11 names seven quantities and the
experiment reports all seven; what is tested here is that the comparison is *sound*, and
that the structure it finds is real rather than an artefact of one instance.

The finding is not a single number. Warm starting pays on the points where the carried
working set turns out to be right and costs on the points where it does not, and the sign
of the total is decided by the mix. That split is asserted on three instances, because a
result seen once is a coincidence.
"""

import itertools

import numpy as np
import pytest

from cosa.experiments import frontier
from cosa.experiments import portfolio as families


@pytest.fixture(scope="module")
def traced():
    """One sweep, shared — twenty-four points solved twice is not worth repeating."""
    return frontier.sweep(families.box(8, seed=0))


# ----------------------------------------------------------------------------------
# The sequence itself
# ----------------------------------------------------------------------------------


def test_the_lambdas_are_spaced_geometrically():
    """`lam` is a ratio — return traded against risk — so its structure is in its logarithm.

    A linear sweep of the same range spends most of its points where the frontier is nearly
    flat, which is where warm starting is easiest and so the least informative place to
    measure it.
    """
    lams = frontier.risk_aversions(1.0, 8.0, 4)
    ratios = [later / earlier for earlier, later in itertools.pairwise(lams)]
    assert ratios == pytest.approx([ratios[0]] * len(ratios))


def test_every_point_solves(traced):
    """A sweep with a failure in it measures nothing."""
    assert all(point.status == "optimal" for point in traced.points)


def test_only_c_changes_along_the_sequence():
    """The structural fact the whole hypothesis rests on, and it is not generic.

    `lam` enters eq. (7) as the coefficient of `t` in the objective and nowhere else, so a
    working set transfers *exactly*. A sequence that perturbed `Sigma` would change `G`, and
    the transfer would be a guess.
    """
    instance = families.box(6, seed=0)
    first = frontier._at(instance.portfolio, 1.0)
    second = frontier._at(instance.portfolio, 3.0)
    assert np.array_equal(first.A, second.A)
    assert np.array_equal(first.E, second.E)
    assert np.array_equal(first.G, second.G)
    assert first.cone == second.cone
    assert not np.array_equal(first.c, second.c)


def test_the_frontier_is_monotone(traced):
    """Risk falls as `lam` rises, which is what a frontier means.

    A property of the answers rather than of the solver, and the cheapest check that the
    sweep traced a frontier rather than a sequence of unrelated optima — a violation would
    mean one point is wrong however good its residuals look.
    """
    assert traced.is_monotone


# ----------------------------------------------------------------------------------
# What a warm start may not do
# ----------------------------------------------------------------------------------


def test_warm_starting_does_not_change_the_answer(traced):
    """The precondition for every other number meaning anything.

    A warm start that changed the answer would be a bug, not a speed-up. The saving is in
    how the answer was reached.
    """
    assert traced.agrees
    assert max(point.gap for point in traced.points) <= frontier.AGREEMENT


def test_the_comparison_is_paired(traced):
    """Each `lam` is solved both ways against the same instance in the same process.

    Comparing a cold sweep against a warm sweep in aggregate would confound the warm start
    with everything else that differs between two runs.
    """
    assert traced.cold_iterations == sum(point.cold.iterations for point in traced.points)
    assert traced.warm_iterations == sum(point.warm.iterations for point in traced.points)


# ----------------------------------------------------------------------------------
# §11's seven quantities
# ----------------------------------------------------------------------------------


def test_all_seven_quantities_are_reported(traced):
    """§11 names seven and the report carries seven, six per point and one across two."""
    point = traced.points[-1]
    assert point.warm.iterations >= 0
    assert point.warm.constraints_added >= 0
    assert point.warm.constraints_removed >= 0
    assert point.warm.factorizations >= 0
    assert point.warm.runtime > 0.0
    assert np.isfinite(point.warm.kkt_residual)
    assert isinstance(point.saved, int)


def test_a_negative_saving_is_reported_rather_than_clamped():
    """A warm start that costs more iterations than a cold one is a finding.

    This experiment exists partly to find out whether that happens, and it does — so the
    quantity must be able to express it.
    """
    losses = [point for point in frontier.sweep(families.box(12, seed=0)).points if point.saved < 0]
    assert losses, "the box(12) sweep is the instance where warm starting sometimes loses"


def test_the_sweep_summarizes_itself(traced):
    """A headline a reader can act on: work saved, and whether the answers survived."""
    rendered = str(traced)
    assert "frontier: 24 points" in rendered
    assert "answers agree" in rendered
    assert "working set unchanged" in rendered


def test_the_totals_take_the_worst_residual_not_the_last(traced):
    """A sweep is only as good as its least converged point."""
    totals = traced.totals()
    assert totals.kkt_residual == max(point.warm.kkt_residual for point in traced.points)
    assert totals.iterations == traced.warm_iterations


# ----------------------------------------------------------------------------------
# The finding
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("assets", [8, 10, 12])
def test_warm_starting_pays_where_the_working_set_transfers(assets):
    """The result, on three instances so that it is a result and not a coincidence.

    Warm starting saves substantially on the points where the carried working set turns out
    to be right, and *costs* on the points where the loop has to correct it. Correcting a
    belief is more expensive than acquiring one: a cold solve discovers the active set on
    the way in, while a warm one has to undo a wrong answer first and then discover it
    anyway.
    """
    traced = frontier.sweep(families.box(assets, seed=0))
    assert traced.stable, "some point must carry its working set intact"
    assert traced.changed, "and some must not, or there is nothing to compare"
    assert traced.saving_on(traced.stable) > 0.1
    assert traced.saving_on(traced.changed) < 0.0


def test_the_overall_sign_is_decided_by_the_mix():
    """Which is why a single headline number is the wrong way to report this.

    `box(8)` saves 44% overall and `box(12)` loses 7%, with the *per-group* savings almost
    identical between them. What differs is how many points had to correct their working
    set.
    """
    small = frontier.sweep(families.box(8, seed=0))
    large = frontier.sweep(families.box(12, seed=0))
    assert small.share > large.share
    assert len(small.changed) < len(large.changed)


def test_the_saving_is_positive_on_the_default_sweep():
    """#30's "done when" — iterations saved strictly positive across the frontier sequence.

    On the default instance, and stated as the conditional claim it is: the sweep whose
    working sets mostly transfer saves work. `test_the_overall_sign_is_decided_by_the_mix`
    is the other half and is what stops this being read as a universal.
    """
    traced = frontier.sweep()
    assert traced.saved > 0
    assert traced.share > 0.0


def test_the_report_runs():
    """`docs` and #34 both read this, so it has to produce text."""
    text = frontier.report(families.box(6, seed=0), lams=frontier.risk_aversions(1.0, 3.0, 5))
    assert "frontier: 5 points" in text
    assert "monotone frontier: True" in text
    assert "warm totals:" in text
