"""Helper functions for advancing a gas state.

.. autofunction:: advance_state
.. autofunction:: generate_singlerate_leap_advancer
"""

__copyright__ = """
Copyright (C) 2020-21 University of Illinois Board of Trustees
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


def _advance_state_stepper_func(rhs, timestepper,
                                state, t_final, dt=0,
                                t=0.0, istep=0,
                                get_timestep=None,
                                pre_step_callback=None,
                                post_step_callback=None,
                                logmgr=None, eos=None, dim=None):
    """Advance state from some time (t) to some time (t_final).

    Parameters
    ----------
    rhs
        Function that should return the time derivative of the state.
        This function should take time and state as arguments, with
        a call with signature ``rhs(t, state)``.
    timestepper
        Function that advances the state from t=time to t=(time+dt), and
        returns the advanced state. Has a call with signature
        ``timestepper(state, t, dt, rhs)``.
    get_timestep
        Function that should return dt for the next step. This interface allows
        user-defined adaptive timestepping. A negative return value indicated that
        the stepper should stop gracefully. Takes only state as an argument.
    state: numpy.ndarray
        Agglomerated object array containing at least the state variables that
        will be advanced by this stepper
    t_final: float
        Simulated time at which to stop
    t: float
        Time at which to start
    dt: float
        Initial timestep size to use, optional if dt is adaptive
    istep: int
        Step number from which to start
    pre_step_callback
        An optional user-defined function, with signature:
        ``state, dt = pre_step_callback(step, t, dt, state)``,
        to be called before the timestepper is called for that particular step.
    post_step_callback
        An optional user-defined function, with signature:
        ``state, dt = post_step_callback(step, t, dt, state)``,
        to be called after the timestepper is called for that particular step.

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

        if get_timestep:
            dt = get_timestep(state=state, t=t, dt=dt)

        if pre_step_callback is not None:
            state, dt = pre_step_callback(state=state, step=istep, t=t, dt=dt)

        state = timestepper(state=state, t=t, dt=dt, rhs=rhs)
        t += dt
        istep += 1

        if post_step_callback is not None:
            state, dt = post_step_callback(state=state, step=istep, t=t, dt=dt)

        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, state, eos)
            logmgr.tick_after()

    return istep, t, state


def _advance_state_leap(rhs, timestepper, state,
                        t_final, dt=0,
                        component_id="state",
                        t=0.0, istep=0,
                        get_timestep=None,
                        pre_step_callback=None,
                        post_step_callback=None,
                        logmgr=None, eos=None, dim=None):
    """Advance state from some time *t* to some time *t_final* using :mod:`leap`.

    Parameters
    ----------
    rhs
        Function that should return the time derivative of the state.
        This function should take time and state as arguments, with
        a call with signature ``rhs(t, state)``.
    timestepper
        An instance of :class:`leap.MethodBuilder`.
    get_timestep
        Function that should return dt for the next step. This interface allows
        user-defined adaptive timestepping. A negative return value indicated that
        the stepper should stop gracefully.
    state: numpy.ndarray
        Agglomerated object array containing at least the state variables that
        will be advanced by this stepper
    t_final: float
        Simulated time at which to stop
    component_id
        State id (required input for leap method generation)
    t: float
        Time at which to start
    dt: float
        Initial timestep size to use, optional if dt is adaptive
    istep: int
        Step number from which to start
    pre_step_callback
        An optional user-defined function, with signature:
        ``state, dt = pre_step_callback(step, t, dt, state)``,
        to be called before the timestepper is called for that particular step.
    post_step_callback
        An optional user-defined function, with signature:
        ``state, dt = post_step_callback(step, t, dt, state)``,
        to be called after the timestepper is called for that particular step.

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

    # Generate code for Leap method.
    if get_timestep:
        dt = get_timestep(state=state, t=t, dt=dt)

    stepper_cls = generate_singlerate_leap_advancer(timestepper, component_id,
                                                    rhs, t, dt, state)
    while t < t_final:

        if get_timestep:
            dt = get_timestep(state=state, t=t, dt=dt)

        if dt < 0:
            return istep, t, state

        if pre_step_callback is not None:
            state, dt = pre_step_callback(state=state,
                                          step=istep,
                                          t=t, dt=dt)

        # Leap interface here is *a bit* different.
        for event in stepper_cls.run(t_end=t+dt):
            if isinstance(event, stepper_cls.StateComputed):
                state = event.state_component
                t += dt

                if post_step_callback is not None:
                    state, dt = post_step_callback(state=state,
                                                   step=istep,
                                                   t=t, dt=dt)

                istep += 1

    return istep, t, state


def generate_singlerate_leap_advancer(timestepper, component_id, rhs, t, dt,
                                      state):
    """Generate Leap code to advance all state at the same timestep, without substepping.

    Parameters
    ----------
    timestepper
        An instance of :class:`leap.MethodBuilder` that advances the state
        from t=time to t=(time+dt), and returns the advanced state.
    component_id
        State id (required input for leap method generation)
    rhs
        Function that should return the time derivative of the state.
        This function should take time and state as arguments, with
        a call looking like rhs(t, state).
    t: float
        Time at which to start
    dt: float
        Initial timestep to be set by leap method
    state: numpy.ndarray
        Agglomerated object array containing at least the state variables that
        will be advanced by this stepper

    Returns
    -------
    dagrt.codegen.python.StepperInterface
        Python class implementing leap method, and generated by dagrt
    """
    code = timestepper.generate()
    from dagrt.codegen import PythonCodeGenerator
    codegen = PythonCodeGenerator(class_name="Method")
    stepper_cls = codegen.get_class(code)(function_map={
        "<func>" + component_id: rhs,
        })
    stepper_cls.set_up(t_start=t, dt_start=dt, context={component_id: state})

    return stepper_cls


def advance_state(rhs, timestepper, state, t_final,
                  component_id="state",
                  t=0.0, istep=0, dt=0,
                  get_timestep=None,
                  pre_step_callback=None,
                  post_step_callback=None,
                  logmgr=None, eos=None, dim=None):
    """Determine what stepper we're using and advance the state from (t) to (t_final).

    Parameters
    ----------
    rhs
        Function that should return the time derivative of the state.
        This function should take time and state as arguments, with
        a call with signature``rhs(t, state)``.
    timestepper
        This is either a user-defined function that advances the state
        from t=time to t=(time+dt) and returns the advanced state
        with call signature ``timestepper(state, t, dt, rhs)``, or
        an instance of :class:`leap.MethodBuilder`. If it's the latter, we are
        responsible for generating timestepper code from the method instructions
        before using it, as well as providing context in the form of the state
        to be integrated, the initial time and timestep, and the RHS function.
    component_id
        State id (required input for leap method generation)
    get_timestep
        Function that should return dt for the next step. This interface allows
        user-defined adaptive timestepping. A negative return value indicated that
        the stepper should stop gracefully. Takes only state as an argument.
    state: numpy.ndarray
        Agglomerated object array containing at least the state variables that
        will be advanced by this stepper
    t_final: float
        Simulated time at which to stop
    t: float
        Time at which to start
    dt: float
        Initial timestep size to use, optional if dt is adaptive
    istep: int
        Step number from which to start
    pre_step_callback
        An optional user-defined function, with signature:
        ``state, dt = pre_step_callback(step, t, dt, state)``,
        to be called before the timestepper is called for that particular step.
    post_step_callback
        An optional user-defined function, with signature:
        ``state, dt = post_step_callback(step, t, dt, state)``,
        to be called after the timestepper is called for that particular step.

    Returns
    -------
    istep: int
        the current step number
    t: float
        the current time
    state: numpy.ndarray
    """
    # The timestepper should either be a Leap
    # method object, or a user-passed function.
    # First, check if we have leap.
    import sys
    leap_timestepper = False

    if ((logmgr is not None) or (dim is not None) or (eos is not None)):
        from warnings import warn
        warn("Passing logmgr, dim, or eos into the stepper is a deprecated stepper "
             "signature. See the examples for the current and preferred usage.",
             DeprecationWarning, stacklevel=2)

    if get_timestep is not None:
        from warnings import warn
        warn("Passing the get_timestep function into the stepper is deprecated. "
             "Users should use the dt argument for constant timestep, and "
             "perform any dt modification in the {pre,post}-step callbacks.",
             DeprecationWarning, stacklevel=2)

    if "leap" in sys.modules:
        # The timestepper can still either be a leap method generator
        # or a user-passed function.
        from leap import MethodBuilder
        if isinstance(timestepper, MethodBuilder):
            leap_timestepper = True

    if leap_timestepper:
        (current_step, current_t, current_state) = \
            _advance_state_leap(
                rhs=rhs, timestepper=timestepper,
                get_timestep=get_timestep, state=state,
                t=t, t_final=t_final, dt=dt,
                pre_step_callback=pre_step_callback,
                post_step_callback=post_step_callback,
                component_id=component_id,
                istep=istep, logmgr=logmgr, eos=eos, dim=dim
            )
    else:
        (current_step, current_t, current_state) = \
            _advance_state_stepper_func(
                rhs=rhs, timestepper=timestepper,
                get_timestep=get_timestep, state=state,
                t=t, t_final=t_final, dt=dt,
                pre_step_callback=pre_step_callback,
                post_step_callback=post_step_callback,
                istep=istep,
                logmgr=logmgr, eos=eos, dim=dim
            )

    return current_step, current_t, current_state
