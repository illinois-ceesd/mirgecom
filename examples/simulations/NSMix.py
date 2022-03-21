"""Define the :class:`mirgecom.simutil.SimulationApplication` for NSMix."""

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

from meshmode.mesh import BTAG_ALL

from mirgecom.simulation_prototypes import ViscousReactiveMixture
from mirgecom.boundary import IsothermalNoSlipBoundary
from mirgecom.io import make_init_message


logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


class NSMix(ViscousReactiveMixture):
    """Reactive mixture with CNS operator."""

    def __init__(self, **kwargs):
        """Initialize the dummy simulation."""
        super().__init__(**kwargs)

        self._boundaries = {
            BTAG_ALL: IsothermalNoSlipBoundary(wall_temperature=self._tseed)
        }

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
        """Configure customization for the simulation."""
        self._name = "NSMix"
        self._order = self._configit("element_order", 1)
        self._dim = self._configit("spatial_dimensions", 2)
        self._debug = False

        # This example runs only 3 steps by default (to keep CI ~short)
        self._sim_cfl = self._configit("CFL", .1)
        self._sim_dt = self._configit("timestep_dt", 1e-9)
        self._t_final = self._configit("final_time", 3e-9)
        self._max_steps = self._configit("max_steps", 1000000)

        self._nstatus = self._configit("nstep_status", 1)
        self._nhealth = self._configit("nstep_health", 1)
        self._nviz = self._configit("nstep_viz", 100)
        self._nrestart = self._configit("nstep_restart", 1000)

    def get_initial_fluid_state(self):
        """Return the initial fluid state at time = 0."""
        from arraycontext import thaw
        from mirgecom.gas_model import make_fluid_state
        nodes = thaw(self._discr.nodes(), self._actx)
        initial_cv = self._initializer(x_vec=nodes, eos=self._gas_model.eos)
        initial_tseed = self._tseed * (self._discr.zeros(self._actx)+1.)
        return make_fluid_state(cv=initial_cv, temperature_seed=initial_tseed,
                                gas_model=self._gas_model)

    def get_boundaries(self):
        """Get the boundary condition dictionary for this case."""
        return self._boundaries

SimulationApplication = NSMix  # noqa

# vim: foldmethod=marker
