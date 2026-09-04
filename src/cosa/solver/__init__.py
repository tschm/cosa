"""The solver: one iteration is working set, direction, step, multipliers, update.

Modules (each owned by its own issue):
    anticycling: §17.2's remedies -- Bland's rule, the guard that arms it, and the merit
        safeguard -- plus the no-progress rule that turned out to be what the loop actually
        needed. Landed.
    apex: The branch at ``L @ x = 0``, where §8.1 replaces the tangent hyperplane with
        exact membership and normal-cone conditions. Landed.
    instrumentation: The counters §11 and §12.3 promise to measure, and the per-iterate
        assertions of §14's Levels 1 and 2. Landed.
    initialization: Feasible starting points, by three routes ending in an elastic
        Phase I the solver runs on itself. Landed; warm starts from a previous solution
        are #30's.
    termination: The five conic KKT residuals of §6 that constitute the stopping
        criterion. Landed.
    warm: §9 Phase VI's warm starts -- the previous solution, working set, multipliers and
        factorization cache, carried into the next solve. Landed.
    cosa: The iteration itself -- §9 Phase III's four ingredients wired together, then
        §9 Phase IV's conic working-set logic on top: the direction subproblem carries the
        Lagrangian's conic curvature, and §7.4 decides deactivation on the multiplier.
        Landed.
"""
