"""Numerical studies: the evidence for or against the project's hypothesis.

Modules (each owned by its own issue):
    reference: The reference-solver oracle every generated problem is cross-checked
        against, with an open backend so the check runs unlicensed and in CI. Landed.
    portfolio: The portfolio test-problem families and the robustness instances.
    frontier: The efficient-frontier sequence over lambda, which is where warm
        starting either pays off or does not.
    benchmarks: Comparison against reference SOCP solvers.
"""
