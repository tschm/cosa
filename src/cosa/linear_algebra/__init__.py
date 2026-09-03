"""The repeated KKT solve, and the numerical work that makes it fast and stable.

Modules (each owned by its own issue):
    kkt: Assembly and solution of the working-set KKT system. The first
        implementation refactorizes every iteration on purpose, as the reference
        the later strategies are measured against.
    factorization: Factorization strategies, rank detection and reuse.
    scaling: Scaling of variables, covariance, constraints and cone variables.
"""
