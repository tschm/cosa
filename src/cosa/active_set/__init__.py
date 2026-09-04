"""The working set: what COSA believes is active, and how that belief changes.

The working set holds three kinds of object at once: active linear inequalities,
equality constraints, and the currently active geometry of the second-order cone.

Modules (each owned by its own issue):
    working_set: The representation itself, and the rendering of it that Success
        Criterion 3 asks for. Landed.
    multipliers: Multiplier recovery from the KKT system, the sign tests that identify
        removal candidates, and the Lagrangian curvature that sends the multipliers back
        into the primal direction computation. Landed.
    updates: Constraint addition and deletion, and the SOC status transitions in both
        directions -- §7.3's activation on the geometry and §7.4's deactivation on the
        conic multiplier. Landed.
"""
