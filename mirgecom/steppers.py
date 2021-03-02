"""Helper functions for advancing a gas state.

.. autofunction:: advance_state
"""

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

from logpyle import set_dt
from mirgecom.logging_quantities import set_sim_state


def advance_state(rhs, timestepper, checkpoint, get_timestep,
                  state, t_final, t=0.0, istep=0, logmgr=None, eos=None, dim=None):
    """Advance state from some time (t) to some time (t_final).

    Parameters
    ----------
    rhs
        Function that should return the time derivative of the state
    timestepper
        Object of type `TimestepperBase` that advances the state from
        t=time to t=(time+dt), and returns the advanced state.
    checkpoint
        Function is user-defined and can be used to preform simulation status
        reporting, viz, and restart i/o.  A non-zero return code from this function
        indicates that this function should stop gracefully.
    get_timestep
        Function that should return dt for the next step. This interface allows
        user-defined adaptive timestepping. A negative return value indicated that
        the stepper should stop gracefully.
    state: numpy.ndarray
        Agglomerated object array containing at least the state variables that
        will be advanced by this stepper
    t_final: float
        Simulated time at which to stop
    t: float
        Time at which to start
    istep: int
        Step number from which to start

    Returns
    -------
    istep: int
        the current step number
    t: float
        the current time
    state: numpy.ndarray
    """
    if t_final <= t:
        return istep, t, state

    while t < t_final:

        if logmgr:
            logmgr.tick_before()

        dt = get_timestep(state=state)
        if dt < 0:
            return istep, t, state

        checkpoint(state=state, step=istep, t=t, dt=dt)

        state = timestepper.step(state=state, t=t, dt=dt, rhs=rhs)

        t += dt
        istep += 1

        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, state, eos)
            logmgr.tick_after()

    return istep, t, state
