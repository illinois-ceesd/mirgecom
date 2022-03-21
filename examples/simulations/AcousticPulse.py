"""Demonstrate acoustic pulse, and adiabatic slip wall."""

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

import logging
import numpy as np

from meshmode.mesh import BTAG_ALL

from mirgecom.simulation_prototypes import FluidSimulation
from mirgecom.boundary import AdiabaticSlipBoundary
from mirgecom.io import make_init_message
from mirgecom.initializers import (
    Lump,
    AcousticPulse
)

logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


class Pulse(FluidSimulation):
    """Reactive mixture with CNS operator."""

    def __init__(self, **kwargs):
        """Initialize the dummy simulation."""
        super().__init__(**kwargs)

        initname = self._initializer.__class__.__name__
        eosname = self._gas_model.eos.__class__.__name__
        init_message = make_init_message(dim=self._dim, order=self._order,
                                         nelements=self._local_mesh.nelements,
                                         global_nelements=self._global_nelements,
                                         dt=self._sim_dt, t_final=self._t_final,
                                         nviz=self._nviz, cfl=self._sim_cfl,
                                         nstatus=self._nstatus,
                                         constant_cfl=self._constant_cfl,
                                         initname=initname,
                                         eosname=eosname, casename=self._casename)

        if self._rank == 0:
            logger.info(init_message)

    def configure_simulation(self):
        """Customize the configuration parameters."""
        self._name = "Pulse"
        self._order = self._configit("element_order", 1)
        self._dim = self._configit("spatial_dimensions", 3)
        self._debug = False

        self._sim_cfl = self._configit("CFL", 1)
        self._sim_dt = self._configit("timestep_dt", .01)
        self._t_final = self._configit("final_time", .1)
        self._max_steps = self._configit("max_steps", 1000000)

        self._nstatus = self._configit("nstep_status", 1)
        self._nhealth = self._configit("nstep_health", 1)
        self._nviz = self._configit("nstep_viz", 10)
        self._nrestart = self._configit("nstep_restart", 5)

        # user limits
        self._max_pressure = self._configit("maximum_pressure", 1.5)
        self._min_pressure = self._configit("minimum_pressure", .8)

        # Acoustic pulse parameters
        origin = np.zeros(shape=(self._dim,))
        self._pulse_center = self._configit("pulse_center", origin)
        self._pulse_amplitude = self._configit("pulse_amplitude", 1.0)
        self._pulse_width = self._configit("pulse_width", .1)

        self._init_velocity = self._configit("initial_velocity",
                                                np.zeros(shape=(self._dim,)))
        self._initializer = Lump(dim=self._dim, center=self._pulse_center,
                                 velocity=self._init_velocity, rhoamp=0.0)

        self._acoustic_pulse = AcousticPulse(dim=self._dim,
                                             amplitude=self._pulse_amplitude,
                                             width=self._pulse_width,
                                             center=self._pulse_center)

        # Acoustic pulse custom defaults for mesh gen
        self._box_ll = self._configit("box_ll", (-1,)*self._dim)
        self._box_ur = self._configit("box_ur", (1,)*self._dim)
        self._nel_per_axis = self._configit("nelements_per_axis", (16,)*self._dim)

        self._boundaries = {BTAG_ALL: AdiabaticSlipBoundary()}

    def get_initial_fluid_state(self):
        """Get the initial fluid state at time=0."""
        from arraycontext import thaw
        from mirgecom.gas_model import make_fluid_state
        nodes = thaw(self._discr.nodes(), self._actx)
        initial_cv = self._initializer(x_vec=nodes, eos=self._gas_model.eos)
        return make_fluid_state(
            cv=self._acoustic_pulse(x_vec=nodes, cv=initial_cv,
                                    eos=self._gas_model.eos),
            gas_model=self._gas_model)

    def get_boundaries(self):
        """Get the boundary condition dictionary for this case."""
        return self._boundaries

SimulationApplication = Pulse  # noqa
