Writing drivers for :mod:`mirgecom`
==============================================

:mod:`mirgecom` provides a great deal of flexibility in what users
are able to do with their drivers. This section of the documentation
is designed to introduce the tools, and recommended practices for
designing drivers for :mod:`mirgecom` simulations.

Timestepper callbacks
------------------------
User callbacks, specifically *pre_step_callback* and *post_step_callback*
are used to inject user code into the main timestepping loop of :mod:`mirgecom`.

Consider the following code from the time stepper, :mod:`~mirgecom.steppers`:

    | while t < t_final:
    |   if pre_step_callback:
    |     state, dt = pre_step_callback(step, t, dt, state)
    |   state = rk4(dt, state)
    |   t = t + dt
    |   step = step + 1
    |   if post_step_callback:
    |     state, dt = post_step_callback(step, t, dt, state)

The above snippet is typical of the time stepping loop logic, and includes
a model call to a Runge-Kutta 4th order time integrator to advance the state.
Users may optionally provide the *pre_step_callback* and/or a *post_step_callback*
that are called before, and after the time integration call, respectively. These
callbacks are typically used for the following:

* Periodically report simulation status (e.g. step, t, state-dependent vars (dv))
* Check on the health of the simulation (by range checking *state*, or *dv*)
* Write restart and/or visualization files
* Implement adaptive timestepping (i.e. modify *dt*)
* Implement solution filtering or limiting (i.e. modify *state*)

.. important::
   The modification of *state* carries inherent danger. Only do this if you know
   what you are doing. One should never modify the state in between time steps
   when using time integrators that carry the state history - because any change
   to the state invalidates the time integrator's historical version.

All user callbacks *must* follow the signature provided above and in the documentation
of :mod:`~mirgecom.steppers`.  That is, user callbacks must return the *state*, and
*dt* to the stepper, even if it just returns the values passed in by the stepper.

