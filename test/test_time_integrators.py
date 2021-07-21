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
import pytest
import importlib
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

from mirgecom.integrators import (euler_step,
                                  lsrk54_step,
                                  lsrk144_step,
                                  rk4_step)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(("integrator", "method_order"),
                         [(euler_step, 1),
                          (lsrk54_step, 4),
                          (lsrk144_step, 4),
                          (rk4_step, 4)])
def test_integration_order(integrator, method_order):
    """Test that time integrators have correct order."""

    def exact_soln(t):
        return np.exp(-t)

    def rhs(t, state):
        return -np.exp(-t)

    from pytools.convergence import EOCRecorder
    integrator_eoc = EOCRecorder()

    dt = 1.0
    for refine in [1, 2, 4, 8]:
        dt = dt / refine
        t = 0
        state = exact_soln(t)

        while t < 4:
            state = integrator(state, t, dt, rhs)
            t = t + dt

        error = np.abs(state - exact_soln(t)) / exact_soln(t)
        integrator_eoc.add_data_point(dt, error)

    logger.info(f"Time Integrator EOC:\n = {integrator_eoc}")
    assert integrator_eoc.order_estimate() >= method_order - .01


leap_spec = importlib.util.find_spec("leap")
found = leap_spec is not None
if found:
    from leap.rk import (
        ODE23MethodBuilder, ODE45MethodBuilder,
        ForwardEulerMethodBuilder,
        MidpointMethodBuilder, HeunsMethodBuilder,
        RK3MethodBuilder, RK4MethodBuilder, RK5MethodBuilder,
        LSRK4MethodBuilder,
        SSPRK22MethodBuilder, SSPRK33MethodBuilder,
        )
    from leap.rk.imex import KennedyCarpenterIMEXARK4MethodBuilder
    from mirgecom.steppers import advance_state

    @pytest.mark.parametrize(("method", "method_order"), [
        (ODE23MethodBuilder("y", use_high_order=False), 2),
        (ODE23MethodBuilder("y", use_high_order=True), 3),
        (ODE45MethodBuilder("y", use_high_order=False), 4),
        (ODE45MethodBuilder("y", use_high_order=True), 5),
        (ForwardEulerMethodBuilder("y"), 1),
        (MidpointMethodBuilder("y"), 2),
        (HeunsMethodBuilder("y"), 2),
        (RK3MethodBuilder("y"), 3),
        (RK4MethodBuilder("y"), 4),
        (RK5MethodBuilder("y"), 5),
        (LSRK4MethodBuilder("y"), 4),
        (KennedyCarpenterIMEXARK4MethodBuilder("y", use_implicit=False,
            explicit_rhs_name="y"), 4),
        (SSPRK22MethodBuilder("y"), 2),
        (SSPRK33MethodBuilder("y"), 3),
        ])
    def test_leapgen_integration_order(actx_factory, method, method_order):
        """Test that time integrators have correct order."""
        actx = actx_factory()

        def exact_soln(t):
            return np.exp(-t)

        def rhs(t, y):
            return -np.exp(-t)

        from pytools.convergence import EOCRecorder
        integrator_eoc = EOCRecorder()

        dt = 1.0
        for refine in [1, 2, 4, 8]:
            dt = dt / refine
            t = 0
            state = exact_soln(t)

            t_final = 4
            step = 0

            (step, t, state) = \
                advance_state(rhs=rhs, timestepper=method, dt=dt,
                              state=state, t=t, t_final=t_final,
                              component_id="y", actx=actx)

            error = np.abs(state - exact_soln(t)) / exact_soln(t)
            integrator_eoc.add_data_point(dt, error)

        logger.info(f"Time Integrator EOC:\n = {integrator_eoc}")
        assert integrator_eoc.order_estimate() >= method_order - .1
