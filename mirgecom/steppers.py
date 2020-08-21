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


def advance_state(rhs, timestepper, checkpoint, get_timestep,
                  state, t=0.0, t_final=1.0, istep=0):
    """
    Implements a generic state advancement routine

    Parameters
    ----------
    rhs: function
        Function that should return the time derivative of the state
    timestepper: function
        Function that advances the state from t=time to t=(time+dt), and
        returns the advanced state.
    checkpoint: function
        Function is user-defined and can be used to preform simulation status
        reporting, viz, and restart i/o.  A non-zero return code from this function
        indicates that this function should stop gracefully.
    get_timestep: function
        Function that should return dt for the next step. This interface allows
        user-defined adaptive timestepping. A negative return value indicated that
        the stepper should stop gracefully.

    Returns
    -------
    istep, t, state: the current step number, time, and state, respectively
    """
    if t_final <= t:
        return istep, t, state

    while t < t_final:

        dt = get_timestep(state=state)
        if dt < 0:
            return istep, t, state

        status = checkpoint(state=state, step=istep, t=t, dt=dt)
        if status != 0:
            return istep, t, state

        state = timestepper(state=state, t=t, dt=dt, rhs=rhs)

        t += dt
        istep += 1

    return istep, t, state
