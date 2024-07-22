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

import importlib
import logging

import numpy as np
import pytest
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context as pytest_generate_tests,
)

from mirgecom.integrators import (
    euler_step,
    lsrk54_step,
    lsrk144_step,
    rk4_step,
    ssprk43_step,
)


logger = logging.getLogger(__name__)


@pytest.mark.parametrize(("integrator", "method_order"),
                         [(euler_step, 1),
                          (lsrk54_step, 4),
                          (lsrk144_step, 4),
                          (rk4_step, 4),
                          (ssprk43_step, 3)])
@pytest.mark.parametrize("local_dt", [True, False])
def test_integrator_order(integrator, method_order, local_dt):
    """Test that time integrators have correct order."""

    def exact_soln(t):
        return np.exp(-t)

    def rhs(t, state):
        return -np.exp(-t)

    from pytools.convergence import EOCRecorder
    integrator_eoc = EOCRecorder()

    dt = (np.asarray([0.5, 1.0, 1.5, 2.0]) if local_dt else 1.0)
    max_steps = 5

    for refine in [1, 2, 4, 8]:
        # These are multi-valued when local_dt
        dt = dt / refine
        t = 0*dt
        state = exact_soln(t)

        for _ in range(max_steps):
            state = integrator(state, t, dt, rhs)
            t = t + dt

        if local_dt:
            # Use the max error among multi-"cells" for local_dt
            error = max(np.abs(state - exact_soln(t)) / exact_soln(t))
            integrator_eoc.add_data_point(dt[0], error)
        else:
            error = np.abs(state - exact_soln(t)) / exact_soln(t)
            integrator_eoc.add_data_point(dt, error)

    logger.info(f"Time Integrator EOC:\n = {integrator_eoc}")
    assert integrator_eoc.order_estimate() >= method_order - .01


@pytest.mark.parametrize(("integrator", "method_order"),
                         [(euler_step, 1),
                          (lsrk54_step, 4),
                          (lsrk144_step, 4),
                          (rk4_step, 4),
                          (ssprk43_step, 3)])
@pytest.mark.parametrize("local_dt", [True, False])
def test_state_advancer(integrator, method_order, local_dt):
    """Test that time integrators have correct order."""

    def exact_soln(t):
        return np.exp(-t)

    def rhs(t, state):
        return -np.exp(-t)

    from pytools.convergence import EOCRecorder
    integrator_eoc = EOCRecorder()

    dt = (np.asarray([0.5, 1.0, 1.5, 2.0]) if local_dt else 1.0)
    max_steps = 5 if local_dt else None
    t_final = 5*dt

    for refine in [1, 2, 4, 8]:
        # These are multi-valued when local_dt
        dt = dt / refine
        t = 0*dt
        state = exact_soln(t)

        advanced_step, advanced_t, advanced_state = \
            advance_state(rhs=rhs, timestepper=integrator, dt=dt,
                          state=state, t=t, t_final=t_final,
                          max_steps=max_steps, local_dt=local_dt,
                          istep=0)

        expected_soln = exact_soln(advanced_t)

        if local_dt:
            # Use the max error among multi-"cells" for local_dt
            error = max(np.abs(advanced_state - expected_soln)
                        / expected_soln)
            integrator_eoc.add_data_point(dt[0], error)
        else:
            error = (
                np.abs(advanced_state - expected_soln) / expected_soln
            )
            integrator_eoc.add_data_point(dt, error)

    logger.info(f"Time Integrator EOC:\n = {integrator_eoc}")
    assert integrator_eoc.order_estimate() >= method_order - .01


leap_spec = importlib.util.find_spec("leap")
found = leap_spec is not None
if found:
    from leap.rk import (
        ForwardEulerMethodBuilder,
        HeunsMethodBuilder,
        LSRK4MethodBuilder,
        MidpointMethodBuilder,
        ODE23MethodBuilder,
        ODE45MethodBuilder,
        RK3MethodBuilder,
        RK4MethodBuilder,
        RK5MethodBuilder,
        SSPRK22MethodBuilder,
        SSPRK33MethodBuilder,
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
    def test_leapgen_integration_order(method, method_order):
        """Test that time integrators have correct order."""
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
                              component_id="y")

            error = np.abs(state - exact_soln(t)) / exact_soln(t)
            integrator_eoc.add_data_point(dt, error)

        logger.info(f"Time Integrator EOC:\n = {integrator_eoc}")
        assert integrator_eoc.order_estimate() >= method_order - .1
