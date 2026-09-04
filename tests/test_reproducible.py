"""Deliverable 9: one command that regenerates every number in the study documents.

Issue #37. "Reproducible experiments" decomposes into four things — a pinned environment,
recorded seeds, a one-command rerun, and stored result artifacts — and each is tested for
here, because each of them fails silently.
"""

import subprocess
import sys

import pytest

from cosa.experiments import __main__ as runner
from cosa.experiments import failures

# These are integration tests: every artifact is a whole study. The suite runs under
# `pytest -n auto`, so tests are distributed across workers in an order nothing here
# controls — which rules out sharing expensive module-scoped fixtures between them, because
# whichever test a worker happens to schedule first pays for all of them and can exceed the
# 60s per-test timeout. That is what the first version of this file did, and it failed in CI
# while passing locally.
#
# Each test therefore performs its own runs, and the runs are made small instead: three
# families rather than thirteen, and eight assets rather than twenty. `SMALL` is what makes
# both possible — `large` is the family that forces twenty assets, and leaving it out lets
# everything else shrink with it.
SMALL = {name: failures.FAMILIES[name] for name in ("basic", "box", "degenerate optimum")}
"""A three-family study: enough to be a study, cheap enough to run several times."""

ASSETS = 8
"""Small, which `SMALL` permits and the full family set would not."""

POINTS = 6
"""A six-point frontier rather than twenty-four, which is the other half of `cheaply`.

Both knobs are properties a study should have anyway — which families, and how long a
sequence — rather than test-only hooks. That they also make the suite fast is a
consequence of the study being parameterized rather than the reason it is.
"""


_FLAGS = [
    "--assets",
    str(ASSETS),
    "--seeds",
    "0",
    "--points",
    str(POINTS),
    "--families",
    *SMALL,
]
"""The command-line spelling of `cheaply`, so the two paths test the same size of study."""


@pytest.fixture
def out(tmp_path):
    """A directory to write artifacts into, thrown away afterwards."""
    return tmp_path / "experiments"


def cheaply(out, *, seeds=(0,)):
    """One artifact run, small enough that a test can afford two of them.

    Two rather than three. The first version of this file did three in one test, which fits
    locally and does not in CI — the runner there is six to eight times slower under coverage
    and `-n auto` contention, and the 60s per-test timeout is not generous at that ratio. No
    test below performs more than two.
    """
    return runner.run(out, assets=ASSETS, seeds=seeds, families=SMALL, points=POINTS)


def test_one_run_writes_every_artifact(out):
    """Four files, and the fourth is not padding — see `test_the_environment_is_recorded`."""
    written = cheaply(out)
    assert [path.name for path in written] == list(runner.ARTIFACTS)
    assert all(path.exists() and path.read_text().strip() for path in written)


def test_the_artifacts_are_the_studies(out):
    """Each written by the module that owns it, so a study and its artifact cannot drift."""
    named = {path.name: path.read_text() for path in cheaply(out)}
    assert "failure-mode study" in named["failure-modes.txt"]
    assert "frontier:" in named["frontier.txt"]
    assert "Success Criterion 5" in named["benchmarks.txt"]


def test_the_environment_is_recorded(out):
    """The environment is recorded beside the numbers it produced.

    A study whose numbers cannot be attributed to a version of anything is not reproducible,
    only re-runnable. The reference backends are listed because a comparison study's numbers
    depend on which oracle was available, and a report that does not say which one it used
    cannot be checked.
    """
    recorded = next(path for path in cheaply(out) if path.name == "environment.txt").read_text()
    for field in ("cosa", "python", "platform", "numpy", "scipy", "reference"):
        assert field in recorded
    assert sys.version.split()[0] in recorded


def test_the_same_seeds_give_the_same_study(out):
    """The same seeds give the same study.

    Which is what "recorded defaults" is worth: the command with no flags reproduces the
    committed files.
    """
    once = cheaply(out / "a")
    again = cheaply(out / "b")
    assert once[0].read_text() == again[0].read_text()


def test_a_different_seed_gives_a_different_study(out):
    """The other half, and the one that stops the test above passing on a constant.

    Split from it rather than asserted alongside, because together they need three runs and
    a test that needs three runs is a test that times out in CI.
    """
    once = cheaply(out / "a")
    elsewhere = cheaply(out / "b", seeds=(7,))
    assert elsewhere[0].read_text() != once[0].read_text()


def test_the_run_is_deterministic(out):
    """The same seeds give byte-identical artifacts, which is not automatic.

    Every generator takes a seed and every study is a pure function of its arguments; a
    single unseeded `default_rng()` anywhere below would break this and nothing else would
    notice.

    Two artifacts are exempt and both for the same honest reason: `benchmarks.txt` reports
    wall-clock time, and `environment.txt` reports a platform. Neither is reproducible across
    runs and pretending otherwise would be worse than saying so — which is why the frontier
    report, which *is* committed and diffed, has no timing column at all.
    """
    timing = {"benchmarks.txt", "environment.txt"}
    once = cheaply(out / "a", seeds=(0, 1))
    again = cheaply(out / "b", seeds=(0, 1))
    for one, other in zip(once, again, strict=True):
        if one.name in timing:
            continue
        assert one.read_text() == other.read_text(), one.name


def test_a_subset_is_a_smaller_study_not_a_different_one(out):
    """The knob these tests lean on, checked rather than assumed.

    A subset changes which families are reported and nothing else about how they are
    reported, so the artifact is still a study — which is what makes running a small one a
    valid test of the machinery rather than a test of a special case.
    """
    text = failures.report(ASSETS, seeds=(0,), families=SMALL)
    assert f"{len(SMALL)} families" in text
    assert "verdicts:" in text
    for name in SMALL:
        assert name in text
    assert "ill conditioned" not in text


def test_the_command_line_entry_point_works(out):
    """`python -m cosa.experiments`, which is what the README tells a reader to run."""
    assert runner.main(["--out", str(out), *_FLAGS]) == 0
    assert (out / "environment.txt").exists()


def test_it_runs_as_a_module(out):
    """It runs as a module, in a subprocess.

    Because `python -m` resolving is exactly what a README promises and exactly what an
    `__main__` guard can get wrong.
    """
    finished = subprocess.run(
        [sys.executable, "-m", "cosa.experiments", "--out", str(out), *_FLAGS],
        capture_output=True,
        text=True,
        check=False,
    )
    assert finished.returncode == 0, finished.stderr
    assert "wrote" in finished.stdout
    assert (out / "benchmarks.txt").exists()


def test_the_committed_artifacts_are_present():
    """The repository carries a generated copy, or the README's claim is to nothing.

    Only presence is checked, not content: the numbers move whenever the solver does, and a
    test that pinned them would fail on every improvement rather than on a regression.
    """
    from pathlib import Path

    committed = Path(__file__).resolve().parent.parent / "docs" / "experiments"
    assert committed.is_dir()
    for name in runner.ARTIFACTS:
        assert (committed / name).exists(), name
