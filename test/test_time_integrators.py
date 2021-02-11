"""Test time integrators."""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
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
import logging

logger = logging.getLogger(__name__)


def test_rk4_order():
    """Test that RK4 is actually 4th order."""
    from mirgecom.integrators import rk4_step

    def rhs(t, state):
        return np.exp(t)

    def exact_soln(t):
        return np.exp(t)

    state = 1.0
    from pytools.convergence import EOCRecorder
    rk4_eoc = EOCRecorder()

    dt = 1.0
    for refine in [1, 2, 4, 8]:
        dt = dt / refine
        t = 0
        state = exact_soln(t)

        while t < 4:
            state = rk4_step(state, t, dt, rhs)
            t = t + dt

        error = np.abs(state - exact_soln(t)) / exact_soln(t)
        rk4_eoc.add_data_point(dt, error)

    logger.info(f"RK4 EOC = {rk4_eoc}")
    assert (
        rk4_eoc.order_estimate() >= 3.99
    )


def test_euler_order():
    """Test that Euler integrator is actually 1st order."""
    from mirgecom.integrators import euler_step

    def rhs(t, state):
        return np.exp(t)

    def exact_soln(t):
        return np.exp(t)

    state = 1.0
    from pytools.convergence import EOCRecorder
    euler_eoc = EOCRecorder()

    dt = .1  # go easy on ye olde Euler
    for refine in [1, 2, 4, 8]:
        dt = dt / refine
        t = 0.0
        state = exact_soln(t)

        while t < 4:
            state = euler_step(state, t, dt, rhs)
            t = t + dt

        error = np.abs(state - exact_soln(t)) / exact_soln(t)
        euler_eoc.add_data_point(dt, error)

    logger.info(f"Euler EOC = {euler_eoc}")
    assert (
        euler_eoc.order_estimate() >= .99
    )
