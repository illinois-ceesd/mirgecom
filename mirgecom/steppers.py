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

import numpy as np
from functools import partial
from mirgecom.utils import force_evaluation
from pytools import memoize_in
from arraycontext import get_container_context_recursively_opt


def _compile_timestepper(actx, timestepper, rhs):
    """Create lazy evaluation version of the timestepper."""
    @memoize_in(actx, ("mirgecom_compiled_operator",
                       timestepper, rhs))
    def get_timestepper():
        return actx.compile(
            lambda state, t, dt: timestepper(
                state=state, t=t, dt=dt, rhs=rhs))

    return get_timestepper()


def _compile_rhs(actx, rhs):
    """Create lazy evaluation version of the rhs."""
    if actx is None:
        return rhs

    @memoize_in(actx, ("mirgecom_compiled_rhs",
                       rhs))
    def get_rhs():
        return actx.compile(rhs)

    return get_rhs()


def _is_unevaluated(actx, ary):
    """Check if an array contains an unevaluated :module:`pytato` expression."""
    from arraycontext import serialize_container, NotAnArrayContainerError
    try:
        iterable = serialize_container(ary)
        for _, subary in iterable:
            if _is_unevaluated(actx, subary):
                return True
        return False
    except NotAnArrayContainerError:
        import pytato as pt
        return isinstance(ary, pt.Array) and not isinstance(ary, pt.DataWrapper)


def _advance_state_stepper_func(rhs, timestepper, state, t_final, dt=0,
                                t=0.0, istep=0, pre_step_callback=None,
                                post_step_callback=None, force_eval=None,
                                local_dt=False, max_steps=None, compile_rhs=True,
                                compile_timestepper=False):
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
    max_steps: int
        Optional parameter indicating maximum number of steps to take
    pre_step_callback
        An optional user-defined function, with signature:
        ``state, dt = pre_step_callback(step, t, dt, state)``,
        to be called before the timestepper is called for that particular step.
    post_step_callback
        An optional user-defined function, with signature:
        ``state, dt = post_step_callback(step, t, dt, state)``,
        to be called after the timestepper is called for that particular step.
    force_eval
        An optional boolean indicating whether to force lazy evaluation between
        timesteps. By default, attempts to deduce whether this is necessary based
        on the behavior of the timestepper.
    local_dt
        An optional boolean indicating whether *dt* is uniform or cell-local in
        the domain.
    compile_rhs
        An optional boolean indicating whether *rhs* can be compiled.
    compile_timestepper
        An optional boolean indicating whether *timestepper* can be compiled.

    Returns
    -------
    istep: int
        the current step number
    t: float
        the current time
    state: numpy.ndarray
    """
    actx = get_container_context_recursively_opt(state)

    if local_dt:
        if max_steps is None:
            raise ValueError("max_steps must be given for local_dt mode.")
        marching_loc = istep
        marching_limit = max_steps
    else:
        t = np.float64(t)
        marching_loc = t
        marching_limit = t_final

    if marching_loc >= marching_limit:
        return istep, t, state

    state = force_evaluation(actx, state)

    if compile_timestepper:
        maybe_compiled_timestepper = _compile_timestepper(actx, timestepper, rhs)
    else:
        if compile_rhs:
            maybe_compiled_rhs = _compile_rhs(actx, rhs)
        else:
            maybe_compiled_rhs = rhs
        maybe_compiled_timestepper = partial(timestepper, rhs=maybe_compiled_rhs)

    while marching_loc < marching_limit:
        if max_steps is not None:
            if max_steps <= istep:
                return istep, t, state

        if pre_step_callback is not None:
            state, dt = pre_step_callback(state=state, step=istep, t=t, dt=dt)

        if force_eval:
            state = force_evaluation(actx, state)

        state = maybe_compiled_timestepper(state=state, t=t, dt=dt)

        if force_eval is None:
            if _is_unevaluated(actx, state):
                force_eval = True
                from warnings import warn
                warn(
                    "Deduced force_eval=True for this timestepper. This can have a "
                    "nontrivial performance impact. If you know that your "
                    "timestepper does not require per-step forced evaluation, "
                    "explicitly set force_eval=False.", stacklevel=2)
            else:
                force_eval = False

        if force_eval:
            state = force_evaluation(actx, state)

        istep += 1

        if local_dt:
            dt = force_evaluation(actx, dt)
            t = force_evaluation(actx, t)
            t = t + dt
            marching_loc = istep
        else:
            t += dt
            marching_loc = t

        if post_step_callback is not None:
            state, dt = post_step_callback(state=state, step=istep, t=t, dt=dt)

    return istep, t, state


def _advance_state_leap(rhs, timestepper, state, t_final, dt=0,
                        component_id="state", t=0.0, istep=0,
                        pre_step_callback=None, post_step_callback=None,
                        force_eval=None, compile_rhs=True):
    """Advance state from some time *t* to some time *t_final* using :mod:`leap`.

    Parameters
    ----------
    rhs
        Function that should return the time derivative of the state.
        This function should take time and state as arguments, with
        a call with signature ``rhs(t, state)``.
    timestepper
        An instance of :class:`leap.MethodBuilder`.
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
    force_eval
        An optional boolean indicating whether to force lazy evaluation between
        timesteps. By default, attempts to deduce whether this is necessary based
        on the behavior of the timestepper.
    compile_rhs
        An optional boolean indicating whether *rhs* can be compiled.

    Returns
    -------
    istep: int
        the current step number
    t: float
        the current time
    state: numpy.ndarray
    """
    actx = get_container_context_recursively_opt(state)

    t = np.float64(t)
    state = force_evaluation(actx, state)

    if t_final <= t:
        return istep, t, state

    if compile_rhs:
        maybe_compiled_rhs = _compile_rhs(actx, rhs)
    else:
        maybe_compiled_rhs = rhs
    stepper_cls = generate_singlerate_leap_advancer(timestepper, component_id,
                                                    maybe_compiled_rhs, t, dt, state)

    while t < t_final:

        if pre_step_callback is not None:
            state, dt = pre_step_callback(state=state,
                                          step=istep,
                                          t=t, dt=dt)
            stepper_cls.state = state
            stepper_cls.dt = dt

        if force_eval:
            state = force_evaluation(actx, state)

        # Leap interface here is *a bit* different.
        for event in stepper_cls.run(t_end=t+dt):
            if isinstance(event, stepper_cls.StateComputed):
                state = event.state_component

                if force_eval is None:
                    if _is_unevaluated(actx, state):
                        force_eval = True
                        from warnings import warn
                        warn(
                            "Deduced force_eval=True for this timestepper. This "
                            "can have a nontrivial performance impact. If you know "
                            "that your timestepper does not require per-step forced "
                            "evaluation, explicitly set force_eval=False.",
                            stacklevel=2)
                    else:
                        force_eval = False

                if force_eval:
                    state = force_evaluation(actx, state)
                    stepper_cls.state = state

                t += dt

                if post_step_callback is not None:
                    state, dt = post_step_callback(state=state,
                                                   step=istep,
                                                   t=t, dt=dt)
                    stepper_cls.state = state
                    stepper_cls.dt = dt

                istep += 1

    return istep, t, state


def generate_singlerate_leap_advancer(timestepper, component_id, rhs, t, dt,
                                      state):
    """Generate Leap code to advance all states with uniform dt, without substepping.

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


def advance_state(rhs, timestepper, state, t_final, t=0, istep=0, dt=0,
                  max_steps=None, component_id="state", pre_step_callback=None,
                  post_step_callback=None, force_eval=None, local_dt=False,
                  compile_rhs=True, compile_timestepper=False):
    """Determine what stepper to use and advance the state from (t) to (t_final).

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
    max_steps: int
        Optional parameter indicating maximum number of steps to take
    pre_step_callback
        An optional user-defined function, with signature:
        ``state, dt = pre_step_callback(step, t, dt, state)``,
        to be called before the timestepper is called for that particular step.
    post_step_callback
        An optional user-defined function, with signature:
        ``state, dt = post_step_callback(step, t, dt, state)``,
        to be called after the timestepper is called for that particular step.
    force_eval
        An optional boolean indicating whether to force lazy evaluation between
        timesteps. By default, attempts to deduce whether this is necessary based
        on the behavior of the timestepper.
    local_dt
        An optional boolean indicating whether *dt* is uniform or cell-local in
        the domain.
    compile_rhs
        An optional boolean indicating whether *rhs* can be compiled.
    compile_timestepper
        An optional boolean indicating whether *timestepper* can be compiled.

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

    if "leap" in sys.modules:
        # The timestepper can still either be a leap method generator
        # or a user-passed function.
        from leap import MethodBuilder
        if isinstance(timestepper, MethodBuilder):
            leap_timestepper = True
            if local_dt:
                raise ValueError("Local timestepping is not supported for Leap-based"
                                 " integrators.")
    if leap_timestepper:
        if compile_timestepper:
            raise ValueError(
                "Leap timestepper is not compatible with compile_timestepper=True")
        (current_step, current_t, current_state) = \
            _advance_state_leap(
                rhs=rhs, timestepper=timestepper,
                state=state, t=t, t_final=t_final, dt=dt, istep=istep,
                pre_step_callback=pre_step_callback,
                post_step_callback=post_step_callback,
                component_id=component_id,
                force_eval=force_eval,
                compile_rhs=compile_rhs,
            )
    else:
        (current_step, current_t, current_state) = \
            _advance_state_stepper_func(
                rhs=rhs, timestepper=timestepper,
                state=state, t=t, t_final=t_final, dt=dt,
                pre_step_callback=pre_step_callback,
                post_step_callback=post_step_callback,
                istep=istep, force_eval=force_eval,
                max_steps=max_steps, local_dt=local_dt,
                compile_rhs=compile_rhs,
                compile_timestepper=compile_timestepper,
            )

    return current_step, current_t, current_state
