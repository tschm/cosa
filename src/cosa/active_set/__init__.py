"""The working set: what COSA believes is active, and how that belief changes.

The working set holds three kinds of object at once: active linear inequalities,
equality constraints, and the currently active geometry of the second-order cone.

Modules (each owned by its own issue):
    working_set: The representation itself, and the rendering of it that Success
        Criterion 3 asks for. Landed.
    multipliers: Multiplier recovery from the KKT system, and the sign tests that
        identify removal candidates.
    updates: Constraint addition and deletion, and the SOC status transitions.
        Landed, less the SOC *deactivation* rule, which §7.4 makes a research question.
"""
