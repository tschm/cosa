"""KKT systems, their factorizations, and the scaling that conditions them.

Modules (each owned by its own issue):
    kkt: The direction subproblem's saddle-point system of §4.3, refactorized every
        iteration as §13.1 asks a reference implementation to be. Landed.
    factorization: Reuse across iterations -- rank-one updates when a constraint is
        added or dropped, and the subtler update when the SOC tangent moves.
    scaling: Diagonal equilibration across §13.3's five named targets, with the cone's
        one-scale-per-block constraint imposed. Landed.
"""
