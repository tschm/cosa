"""One command that regenerates every number in the study documents.

Deliverable 9 (``paper.tex:1254``) asks for "reproducible experiments": a pinned
environment, recorded seeds, a one-command rerun, and stored result artifacts. #37 is that,
and this module is the command:

    uv run python -m cosa.experiments

**The four artifacts are the four studies**, each written by the module that owns it:

* ``failure-modes.txt`` -- §12.4's per-family classification and mitigation ablation, #36.
* ``frontier.txt`` -- §11's warm-started frontier sweep, #35.
* ``benchmarks.txt`` -- §12's four-mode comparison against a reference solver, #34.
* ``environment.txt`` -- what produced the other three.

The last one is not padding. A study whose numbers cannot be attributed to a version of
anything is not reproducible, only re-runnable, and the difference is exactly the thing that
makes a results table trustworthy two years later. Python, NumPy, SciPy, the installed
reference backends and the package's own version are recorded beside the numbers they
produced.

**Seeds are arguments, not constants.** Every study takes them and every default is written
down here rather than buried, so a reader who wants a different draw changes one flag and a
reader who wants *this* draw gets it by running the command with no flags at all.

**It writes files rather than printing.** A study that only prints has to be re-run to be
cited, which means the number in a document and the number a reader gets are related by
hope. Writing them under ``docs/reports`` puts the artifacts next to the prose that quotes
them, and a diff shows when they stop agreeing.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import scipy

from cosa.experiments import benchmarks, failures, frontier
from cosa.experiments.reference import OPEN_BACKENDS, CvxpySolver

if TYPE_CHECKING:
    from cosa.experiments.portfolio import PortfolioInstance

__all__ = ["ARTIFACTS", "environment", "main", "run"]

ARTIFACTS: Final = ("failure-modes.txt", "frontier.txt", "benchmarks.txt", "environment.txt")
"""What one run writes, in the order it writes them."""

DEFAULT_OUT: Final = Path("docs/experiments")
"""Where the artifacts go: next to the prose that quotes them, and tracked.

Not ``docs/reports``, which is gitignored and belongs to the coverage and test-report
machinery. An artifact a study is cited from has to be committed or the citation is to
something nobody else has.
"""


def environment() -> str:
    """What produced the other artifacts.

    Returns:
        A block naming the interpreter, the platform, the numerical stack, the installed
        reference backends and the package version. Backends are listed because a
        comparison study's numbers depend on which oracle was available, and a report that
        does not say which one it used cannot be checked.
    """
    try:
        version = importlib.metadata.version("cosa")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - always installed here
        version = "unknown"
    available = [backend for backend in OPEN_BACKENDS if CvxpySolver(backend).is_available()]
    return "\n".join(
        [
            "environment",
            "",
            f"cosa       {version}",
            f"python     {sys.version.split()[0]}",
            f"platform   {platform.platform()}",
            f"numpy      {np.__version__}",
            f"scipy      {scipy.__version__}",
            f"reference  {', '.join(available) if available else 'none installed'}",
        ]
    )


def run(
    out: Path = DEFAULT_OUT,
    *,
    assets: int = 20,
    seeds: tuple[int, ...] = (0, 1, 2),
    families: Mapping[str, Callable[..., PortfolioInstance]] | None = None,
    points: int | None = None,
) -> tuple[Path, ...]:
    """Regenerate every artifact.

    Args:
        out: the directory to write into. Created if it does not exist.
        assets: how many assets the studies use. Twenty is the failure study's floor --
            ``large`` builds a rank-ten covariance and refuses an instance it cannot make
            low-rank -- and the other two scale down from it.
        seeds: which draws.
        families: which instance families the failure study covers, or ``None`` for all of
            them. A subset makes the whole run cheap, which is what the test suite asks for
            -- and note that ``large`` is what forces ``assets`` to twenty, so a subset
            without it can use a far smaller instance.
        points: how many risk aversions the two sequence studies trace, or ``None`` for
            :data:`cosa.experiments.frontier.LAMBDAS`. The other half of what makes a small
            run small.

    Returns:
        The paths written, in :data:`ARTIFACTS` order.
    """
    out.mkdir(parents=True, exist_ok=True)
    chosen = failures.FAMILIES if families is None else families
    lams = None if points is None else frontier.risk_aversions(count=points)
    contents = (
        failures.report(assets, seeds=seeds, families=chosen),
        frontier.report(assets=max(8, assets // 2), seed=seeds[0], lams=lams),
        benchmarks.report(max(8, assets // 2), seeds=seeds[:2], large=assets * 3, lams=lams),
        environment(),
    )
    written = []
    for name, text in zip(ARTIFACTS, contents, strict=True):
        path = out / name
        path.write_text(text + "\n")
        written.append(path)
    return tuple(written)


def main(argv: list[str] | None = None) -> int:
    """The command-line entry point.

    Every parameter of :func:`run` is exposed, including the two that make a run small. A
    command line that could only produce the full study would be a command line a reader
    could not use to look at one family.

    Args:
        argv: arguments, or ``None`` for ``sys.argv``.

    Returns:
        A process exit code, zero on success.
    """
    parser = argparse.ArgumentParser(prog="python -m cosa.experiments", description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where to write the artifacts")
    parser.add_argument("--assets", type=int, default=20, help="how many assets the studies use")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="which draws")
    parser.add_argument(
        "--families",
        nargs="+",
        choices=sorted(failures.FAMILIES),
        help="which instance families the failure study covers (default: all of them)",
    )
    parser.add_argument(
        "--points",
        type=int,
        help=f"how many frontier points the sequence studies trace (default: {frontier.LAMBDAS})",
    )
    args = parser.parse_args(argv)
    chosen = None if args.families is None else {name: failures.FAMILIES[name] for name in args.families}
    for path in run(args.out, assets=args.assets, seeds=tuple(args.seeds), families=chosen, points=args.points):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess, not imported
    raise SystemExit(main())
