Time stepping for simulations
=============================

This section will discuss matters pertaining to time stepping in *MIRGE-Com*
simulations.


Time stepping modes
-------------------

*MIRGE-Com* fluid simluations support two "modes" for time-stepping:

  * Constant DT - DT fixed in time, CFL changes
  * Constant CFL - CFL fixed in time, DT changes

*MIRGE-Com* timestepping modes are actuated at the driver level by
passing the `constant_cfl` argument to the `get_simulation_timestep`
utility in the `simutils` module.  The `get_simulation_timestep`
utility is designed to return a conservative estimate for the
maximum stable DT.

Choosing a simulation timestep
------------------------------

It is important to consider the stability limits of the physics and
mesh configurations when choosing a simulation timestep.  If the
simulation DT is too high, then the simulation may be unstable.  On
the other hand, if DT is too small, then the simulation may be
unnecessarily expensive or altogether unproductive.

When running long-running simulations or simulations that cover
a range of different flow conditions, *Constant CFL* mode can be
useful at keeping the system-wide DT down to stable levels.

As a quick rule-of-thumb for *MIRGE-Com* fluid simulations; to choose
a stable timestep, the user should ensure that the constant CFL, or the
constant DT is such that CFL is:

   * < .8 for inviscid/advective-only systems
   * < .4 for viscous/diffusive systems

Some useful analysis for determining the theoretical limits for simulation
DT will follow in forthcoming sections.

Advective stability
^^^^^^^^^^^^^^^^^^^
 - Advective timescale

Diffusive stability
^^^^^^^^^^^^^^^^^^^
 - Diffusive timescale
 - Reynolds number 
 - Peclet number

Chemical reaction stability
^^^^^^^^^^^^^^^^^^^^^^^^^^^
 - Reaction rate timescale
 - Damköhler number

Putting it all together for fluid simulations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inviscid part
"""""""""""""

Thermal diffusion
"""""""""""""""""
 
Mixtures
""""""""

Combustion
""""""""""

Examples
--------

Stability of the acoustic pulse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stability of Poiseuille flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stability of diffusive, inert mixture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stability of reactive mixture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
