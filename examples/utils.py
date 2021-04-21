"""Utilities for running the examples."""

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

from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state, advance_state_leap


def leap_setup(method, component_id, rhs, t, dt, state):
    timestepper = method(component_id)
    code = timestepper.generate()
    from dagrt.codegen import PythonCodeGenerator
    codegen = PythonCodeGenerator(class_name="Method")
    interp = codegen.get_class(code)(function_map={
        "<func>" + component_id: rhs,
        })
    interp.set_up(t_start=t, dt_start=dt, context={component_id: state})

    return interp


def advance_example(rhs, timestepper, checkpoint, get_timestep, state, t_final,
                    component_id="state", t=0.0, istep=0, logmgr=None,
                    eos=None, dim=None):

    if timestepper is rk4_step:
        (current_step, current_t, current_state) = \
            advance_state(rhs=rhs, timestepper=timestepper,
                        checkpoint=checkpoint,
                        get_timestep=get_timestep, state=state,
                        t=t, t_final=t_final, istep=istep,
                        logmgr=logmgr, eos=eos, dim=dim)
    else:
        # The timestepper should either be a Leap
        # method object, or something is broken.
        from leap import MethodBuilder
        if isinstance(timestepper, MethodBuilder):
            (current_step, current_t, current_state) = \
                advance_state_leap(rhs=rhs, timestepper=timestepper,
                              checkpoint=checkpoint,
                              get_timestep=get_timestep, state=state,
                              t=t, t_final=t_final, component_id=component_id,
                              istep=istep, logmgr=logmgr, eos=eos, dim=dim)
        else:
            raise ValueError("Timestepper unrecognizable")

    return current_step, current_t, current_state

# vim: foldmethod=marker
