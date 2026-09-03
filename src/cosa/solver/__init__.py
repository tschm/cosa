"""The solver: one iteration is working set, direction, step, multipliers, update.

Modules (each owned by its own issue):
    apex: The branch at ``L @ x = 0``, where §8.1 replaces the tangent hyperplane with
        exact membership and normal-cone conditions. Landed.
    instrumentation: The counters §11 and §12.3 promise to measure, and the per-iterate
        assertions of §14's Levels 1 and 2. Landed.
    cosa: The iteration itself.
    initialization: Feasible starting points, and warm starts from a previous
        solution, working set, multipliers and factorizations.
    termination: The conic KKT residuals that constitute the stopping criterion.
"""
