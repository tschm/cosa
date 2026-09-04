"""The solver: one iteration is working set, direction, step, multipliers, update.

Modules (each owned by its own issue):
    apex: The branch at ``L @ x = 0``, where §8.1 replaces the tangent hyperplane with
        exact membership and normal-cone conditions. Landed.
    instrumentation: The counters §11 and §12.3 promise to measure, and the per-iterate
        assertions of §14's Levels 1 and 2. Landed.
    initialization: Feasible starting points, by three routes ending in an elastic
        Phase I the solver runs on itself. Landed; warm starts from a previous solution
        are #30's.
    termination: The five conic KKT residuals of §6 that constitute the stopping
        criterion. Landed.
    cosa: The iteration itself -- §9 Phase III's four ingredients wired together.
        Landed, with activation only; §7.4's deactivation is #23's.
"""
