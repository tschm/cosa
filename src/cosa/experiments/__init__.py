"""Numerical studies: the evidence for or against the project's hypothesis.

Modules (each owned by its own issue):
    reference: The reference-solver oracle every generated problem is cross-checked
        against, with an open backend so the check runs unlicensed and in CI. Landed.
    portfolio: The portfolio test-problem families and the robustness instances. Landed.
    randomized: §16.3's seeded generator, which randomizes the problem's *shape* rather
        than reseeding a fixed one. Landed.
    failures: §12.4's failure-mode and degeneracy study -- every family classified, every
        switchable mitigation ablated. Landed; its narrative is
        ``docs/development/failure-modes.md``.
    frontier: §11's efficient-frontier sequence over lambda, solved cold and warm, which
        is where warm starting either pays off or does not. Landed.
    benchmarks: §12's comparison against reference SOCP solvers, in four modes with both
        metric tables. Landed.
"""
