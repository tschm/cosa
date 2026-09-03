"""The working set: what COSA believes is active, and how that belief changes.

The working set holds three kinds of object at once: active linear inequalities,
equality constraints, and the currently active geometry of the second-order cone.

Modules (each owned by its own issue):
    working_set: The representation itself.
    multipliers: Multiplier recovery from the KKT system, and the sign tests that
        identify removal candidates.
    updates: Constraint addition and deletion, and the SOC status transitions.
"""
