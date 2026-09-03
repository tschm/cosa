"""Second-order cone geometry, testable independently of the solver.

The cone is ``Q = {(t, y) : ||y|| <= t}``, and the solver reaches it only through
this subpackage.

Modules (each owned by its own issue):
    soc: Membership, boundary and apex predicates.
    tangent: The unit vector ``u = L @ x / ||L @ x||`` and the tangent condition
        ``tau - u.T @ L @ p = 0`` at a nonzero boundary point.
    step: The exact step interval from the scalar quadratic in ``alpha``, and its
        intersection with the linear step bounds.
"""
