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


__doc__ = """
.. autofunction:: rk4_step
"""


def rk4_step(state, t, dt, rhs):
    """
    Implements a generic RK4 time step state/rhs pair.
    """
    k1 = rhs(t, state)
    k2 = rhs(t+dt/2, state + dt/2*k1)
    k3 = rhs(t+dt/2, state + dt/2*k2)
    k4 = rhs(t+dt, state + dt*k3)
    return state + dt/6*(k1 + 2*k2 + 2*k3 + k4)


def rk4_stepper(rhs, checkpoint, get_timestep,
                state, t=0.0, t_final=1.0, istep=0):
    """
    Implements a generic RK4 time stepping loop for a state/rhs pair.
    """
    if t_final <= t:
        return(istep, t, state)

    while t < t_final:

        dt = get_timestep(state=state)
        if dt < 0:
            return (istep, t, state)

        status = checkpoint(state=state, step=istep, t=t, dt=dt)
        if status != 0:
            return (istep, t, state)

        state = rk4_step(state, t, dt, rhs)

        t += dt
        istep += 1

    return (istep, t, state)
