"""The solver: one iteration is working set, direction, step, multipliers, update.

Modules (each owned by its own issue):
    cosa: The iteration itself.
    initialization: Feasible starting points, and warm starts from a previous
        solution, working set, multipliers and factorizations.
    termination: The conic KKT residuals that constitute the stopping criterion.
"""
