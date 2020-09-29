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

from logpyle import (LogManager, add_general_quantities,
        add_simulation_quantities, add_run_info, IntervalTimer,
        set_dt, LogQuantity)

class Pressure(LogQuantity):
  def __init__(self, discr, current_state, eos):
        LogQuantity.__init__(self, "pressure", "p")

        from mirgecom.euler import split_conserved
        cv = split_conserved(discr.dim, current_state)
        self.dv = eos.dependent_vars(cv)

        self.discr = discr

  def __call__(self):
    from functools import partial
    _min = partial(self.discr.nodal_min, "vol")
    _max = partial(self.discr.nodal_max, "vol")

    return _min(self.dv.pressure)

class Temperature(LogQuantity):
  def __init__(self, discr, current_state, eos):
        LogQuantity.__init__(self, "temperature", "K")

        from mirgecom.euler import split_conserved
        cv = split_conserved(discr.dim, current_state)
        self.dv = eos.dependent_vars(cv)

        self.discr = discr

  def __call__(self):
    from functools import partial
    _min = partial(self.discr.nodal_min, "vol")
    _max = partial(self.discr.nodal_max, "vol")

    return _min(self.dv.temperature)


def advance_state(rhs, timestepper, checkpoint, get_timestep,
                  state, t_final, t=0.0, istep=0, logmgr=None, discr=None, eos=None):
    """Advance state from some time (t) to some time (t_final).

    Parameters
    ----------
    rhs
        Function that should return the time derivative of the state
    timestepper
        Function that advances the state from t=time to t=(time+dt), and
        returns the advanced state.
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

    logmgr.add_quantity(Pressure(discr, state, eos))
    logmgr.add_quantity(Temperature(discr, state, eos))
    logmgr.add_watches(["pressure", "temperature"])

    while t < t_final:
        if logmgr:
            logmgr.tick_before()

        dt = get_timestep(state=state)
        if dt < 0:
            return istep, t, state

        checkpoint(state=state, step=istep, t=t, dt=dt)

        state = timestepper(state=state, t=t, dt=dt, rhs=rhs)

        t += dt
        istep += 1

        set_dt(logmgr, dt)
        if logmgr:
            logmgr.tick_after()

    return istep, t, state
