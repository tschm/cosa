"""KKT systems, their factorizations, and the scaling that conditions them.

Modules (each owned by its own issue):
    kkt: The direction subproblem's saddle-point system of §4.3, refactorized every
        iteration as §13.1 asks a reference implementation to be. Landed.
    rank: QR-based rank detection and the null-space route that survives a degenerate
        working set. Landed.
    factorization: §13's four strategies measured against the M2 reference, and the
        default the measurement chose. Landed; the *reuse* across iterations that §13.2
        asks for is #27's and is still ahead.
    scaling: Diagonal equilibration across §13.3's five named targets, with the cone's
        one-scale-per-block constraint imposed. Landed.
"""
