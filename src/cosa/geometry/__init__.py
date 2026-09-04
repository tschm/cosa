"""Second-order cone geometry, testable independently of the solver.

The cone is ``Q = {(t, y) : ||y|| <= t}``, and the solver reaches it only through
this subpackage.

Modules (each owned by its own issue):
    soc: Membership, boundary and apex predicates. Landed.
    tangent: The unit vector ``u = L @ x / ||L @ x||``, the tangent condition
        ``tau - u.T @ L @ p = 0`` at a nonzero boundary point, and the boundary's
        *curvature* -- the second derivative the tangent plane leaves out, which is what
        #23's direction subproblem is built on. Landed.
    step: The exact step interval from the scalar quadratic in ``alpha``, and its
        intersection with the linear step bounds. Landed -- with eq. (6) corrected; see
        the module docstring.
"""
