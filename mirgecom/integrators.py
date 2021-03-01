"""Functions for time integration.

Time integrators
^^^^^^^^^^^^^^^^
.. autofunction:: rk4_step
.. autofunction:: lsrk4_step
.. autofunction:: lsrk144_step
.. autofunction:: euler_step
"""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__author__ = """
Center for Exascale-Enabled Scramjet Design
University of Illinois, Urbana, IL 61801
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import mirgecom.butcher_tableau as bt


def rk4_step(state, t, dt, rhs):
    """Take one step using 4th order Runge-Kutta."""
    k1 = rhs(t, state)
    k2 = rhs(t+dt/2, state + dt/2*k1)
    k3 = rhs(t+dt/2, state + dt/2*k2)
    k4 = rhs(t+dt, state + dt*k3)
    return state + dt/6*(k1 + 2*k2 + 2*k3 + k4)


def lsrk4_step(state, t, dt, rhs):
    """
    Take one step using Carpenter-Kennedy low storage 4th order Runge-Kutta.

    LSERK coefficients from [Hesthaven_2008]_, Section 3.4.
    """
    p = state
    k = p * 0.

    for i in range(5):
        k = bt._LSRK4_A[i]*k + dt*rhs(t + bt._LSRK4_C[i]*dt, p)
        p = p + bt._LSRK4_B[i]*k

    return p


def lsrk144_step(state, t, dt, rhs):
    """
    Take one step using the low storage 14-stage 4th order Runge-Kutta method.

    LSRK coefficients are summarized in Table 3 of Niegemann, Diehl, and
    Busch (2012): https://doi.org/10.1016/j.jcp.2011.09.003.
    """
    p = state
    k = p * 0.

    for i in range(14):
        k = bt._LSRK144_A[i]*k + dt*rhs(t + bt._LSRK144_C[i]*dt, p)
        p = p + bt._LSRK144_B[i]*k

    return p


def euler_step(state, t, dt, rhs):
    """Take one step using forward Euler time integration."""
    return state + dt*rhs(t, state)
