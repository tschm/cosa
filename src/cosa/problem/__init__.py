"""Problem representation: the SOCP and the portfolio problem that reduces to it.

Modules:
    socp: The primal SOCP of the paper's eq. (7), with the cone as a Cartesian
        product of second-order cones, and the one fixed sign convention. Landed.
    portfolio: The mean-standard-deviation portfolio problem, including the
        factorization ``Sigma = L.T @ L`` that turns the standard deviation into
        the cone constraint ``||L @ x|| <= t``. It produces the
        ``cosa.problem.socp.MeanStdForm`` this package already defines.
"""
