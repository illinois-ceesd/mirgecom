"""Functions for time integration.

Time integrators
^^^^^^^^^^^^^^^^
.. autofunction:: rk4_step
.. autofunction:: lsrk4_step
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

import numpy as np

_LSRK4_A = np.array([
    0.,
    -567301805773/1357537059087,
    -2404267990393/2016746695238,
    -3550918686646/2091501179385,
    -1275806237668/842570457699])

_LSRK4_B = np.array([
    1432997174477/9575080441755,
    5161836677717/13612068292357,
    1720146321549/2090206949498,
    3134564353537/4481467310338,
    2277821191437/14882151754819])

_LSRK4_C = np.array([
    0.,
    1432997174477/9575080441755,
    2526269341429/6820363962896,
    2006345519317/3224310063776,
    2802321613138/2924317926251])


def rk4_step(state, t, dt, rhs):
    """Take one step using 4th order Runge-Kutta."""
    k1 = rhs(t, state)
    k2 = rhs(t+dt/2, state + dt/2*k1)
    k3 = rhs(t+dt/2, state + dt/2*k2)
    k4 = rhs(t+dt, state + dt*k3)
    return state + dt/6*(k1 + 2*k2 + 2*k3 + k4)


def lsrk4_step(state, t, dt, rhs):
    """Take one step using low storage 4th order Runge-Kutta."""
    """LSERK coefficients from [Hesthaven_2008]_, Section 3.4"""

    p = state
    k = p * 0.

    for i in range(5):
        k = _LSRK4_A[i]*k + dt*rhs(t + _LSRK4_C[i]*dt, p)
        p = p + _LSRK4_B[i]*k

    return p


def euler_step(state, t, dt, rhs):
    """Take one step using forward Euler time integration."""
    return state + dt*rhs(t, state)
