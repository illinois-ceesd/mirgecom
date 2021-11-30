""":mod:`mirgecom.gas_model` provides utilities to deal with gases.

Fluid State Handling
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: FluidState
.. autofunction: make_fluid_state
.. autofunction: make_fluid_state_on_boundary
.. autofunction: make_fluid_state_trace_pairs

"""

__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
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
import numpy as np  # noqa
from meshmode.dof_array import DOFArray  # noqa
from dataclasses import dataclass
from arraycontext import dataclass_array_container
from mirgecom.fluid import ConservedVars
from mirgecom.eos import (
    GasEOS,
    EOSDependentVars
)
from mirgecom.transport import (
    TransportModel,
    TransportModelRequired,
    TransportDependentVars
)


@dataclass(frozen=True)
class GasModel:
    r"""Physical gas model for calculating fluid state-dependent quantities.

    .. attribute:: eos
    .. attribute:: transport_model
    """

    eos: GasEOS
    transport: TransportModel = None


@dataclass_array_container
@dataclass(frozen=True)
class FluidState:
    r"""Gas model-consistent fluid state.

    Data attributes
    ^^^^^^^^^^^^^^^
    .. attribute:: cv

        :class:`~mirgecom.fluid.ConservedVars` for the fluid conserved state

    .. attribute:: dv

        :class:`~mirgecom.eos.EOSDependentVars` for the fluid state-dependent
        quantities corresponding to the chosen equation of state.

    .. attribute:: tv
        :class:`~mirgecom.transport.TransportDependentVars` for the fluid
        state-dependent transport properties.

    Properties
    ^^^^^^^^^^
    .. autoattribute:: array_context
    .. autoattribute:: dim
    .. autoattribute:: nspecies
    .. autoattribute:: pressure
    .. autoattribute:: temperature
    .. autoattribute:: velocity
    .. autoattribute:: speed
    .. autoattribute:: speed_of_sound
    .. autoattribute:: mass_density
    .. autoattribute:: momentum_density
    .. autoattribute:: energy_density
    .. autoattribute:: species_mass_density
    .. autoattribute:: species_mass_fractions
    """

    cv: ConservedVars
    dv: EOSDependentVars
    tv: TransportDependentVars = None

    @property
    def array_context(self):
        """Return an array context for the :class:`ConservedVars` object."""
        return self.cv.array_context

    @property
    def dim(self):
        """Return the number of physical dimensions."""
        return self.cv.dim

    @property
    def nspecies(self):
        """Return the number of physical dimensions."""
        return self.cv.nspecies

    @property
    def pressure(self):
        """Return the gas pressure."""
        return self.dv.pressure

    @property
    def temperature(self):
        """Return the gas temperature."""
        return self.dv.temperature

    @property
    def mass_density(self):
        """Return the gas density."""
        return self.cv.mass

    @property
    def momentum_density(self):
        """Return the gas momentum density."""
        return self.cv.momentum

    @property
    def energy_density(self):
        """Return the gas total energy density."""
        return self.cv.energy

    @property
    def species_mass_density(self):
        """Return the gas species densities."""
        return self.cv.species_mass

    @property
    def velocity(self):
        """Return the gas velocity."""
        return self.cv.velocity

    @property
    def speed(self):
        """Return the gas speed."""
        return self.cv.speed

    @property
    def species_mass_fractions(self):
        """Return the species mass fractions y = species_mass / mass."""
        return self.cv.species_mass_fractions

    @property
    def speed_of_sound(self):
        """Return the speed of sound in the gas."""
        return self.dv.speed_of_sound

    def _get_transport_property(self, name):
        """Grab a transport property if transport model is present."""
        if self.tv is None:
            raise TransportModelRequired("Viscous transport model not provided.")
        return getattr(self.tv, name)

    @property
    def viscosity(self):
        """Return the fluid viscosity."""
        return self._get_transport_property("viscosity")

def make_fluid_state(cv, gas_model, temperature_seed=None):
    """Create a fluid state from the conserved vars and equation of state."""
    dv = gas_model.eos.dependent_vars(cv, temperature_seed=temperature_seed)
    tv = None
    if gas_model.transport is not None:
        tv = gas_model.transport.dependent_vars(eos=gas_model.eos, cv=cv)
    return FluidState(cv=cv, dv=dv, tv=tv)


def project_fluid_state(discr, btag, fluid_state, gas_model):
    """Create a fluid state from the conserved vars and equation of state."""
    """Create a fluid state from volume :class:`FluidState` *fluid_state*
    by projection onto the boundary and ensuring thermal consistency.
    """
    cv_sd = discr.project("vol", btag, fluid_state.cv)
    temperature_seed = None
    if fluid_state.cv.nspecies > 0:
        temperature_seed = discr.project("vol", btag, fluid_state.dv.temperature)
    return make_fluid_state(cv=cv_sd, gas_model=gas_model,
                            temperature_seed=temperature_seed)


def _getattr_ish(obj, name):
    if obj is None:
        return None
    else:
        return getattr(obj, name)


def make_fluid_state_trace_pairs(cv_pairs, gas_model, temperature_seed_pairs=None):
    """Create a fluid state from the conserved vars and equation of state."""
    from grudge.trace_pair import TracePair
    if temperature_seed_pairs is None:
        temperature_seed_pairs = [None] * len(cv_pairs)
    return [TracePair(
        cv_pair.dd,
        interior=make_fluid_state(cv_pair.int, gas_model,
                                  temperature_seed=_getattr_ish(tseed_pair, "int")),
        exterior=make_fluid_state(cv_pair.ext, gas_model,
                                  temperature_seed=_getattr_ish(tseed_pair, "ext")))
        for cv_pair, tseed_pair in zip(cv_pairs, temperature_seed_pairs)]


def make_fluid_state_interior_trace_pair(discr, state, gas_model):
    """Create a fluid state from the conserved vars and equation of state."""
    from grudge.eager import interior_trace_pair
    from grudge.trace_pair import TracePair
    cv_tpair = interior_trace_pair(discr, state.cv)
    tseed_pair = None
    if state.nspecies > 0:
        tseed_pair = interior_trace_pair(discr, state.dv.temperature)
    return TracePair(
        cv_tpair.dd,
        interior=make_fluid_state(cv_tpair.int, gas_model,
                                  _getattr_ish(tseed_pair, "int")),
        exterior=make_fluid_state(cv_tpair.ext, gas_model,
                                  _getattr_ish(tseed_pair, "ext")))
