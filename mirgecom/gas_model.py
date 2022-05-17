""":mod:`mirgecom.gas_model` provides utilities to deal with gases.

Physical Gas Model Encapsulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: GasModel

Fluid State Encapsulation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: FluidState
.. autoclass:: ViscousFluidState

Fluid State Handling Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: make_fluid_state
.. autofunction:: project_fluid_state
.. autofunction:: make_fluid_state_trace_pairs
.. autofunction:: make_operator_fluid_states
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
from functools import partial
from meshmode.dof_array import DOFArray  # noqa
from dataclasses import dataclass
from arraycontext import dataclass_array_container
from mirgecom.fluid import ConservedVars
from mirgecom.eos import (
    GasEOS,
    GasDependentVars,
    MixtureDependentVars,
    MixtureEOSNeededError
)
from mirgecom.transport import (
    TransportModel,
    GasTransportVars
)
from grudge.dof_desc import DOFDesc, as_dofdesc
from grudge.trace_pair import (
    interior_trace_pairs,
    tracepair_with_discr_tag
)


@dataclass(frozen=True)
class GasModel:
    r"""Physical gas model for calculating fluid state-dependent quantities.

    .. attribute:: eos

        A gas equation of state to provide thermal properties.

    .. attribute:: transport_model

        A gas transport model to provide transport properties.  None for inviscid
        models.
    """

    eos: GasEOS
    transport: TransportModel = None


@dataclass_array_container
@dataclass(frozen=True)
class FluidState:
    r"""Gas model-consistent fluid state.

    .. attribute:: cv

        Fluid conserved quantities

    .. attribute:: dv

        Fluid state-dependent quantities corresponding to the chosen equation of
        state.

    .. autoattribute:: array_context
    .. autoattribute:: dim
    .. autoattribute:: nspecies
    .. autoattribute:: pressure
    .. autoattribute:: temperature
    .. autoattribute:: velocity
    .. autoattribute:: speed
    .. autoattribute:: wavespeed
    .. autoattribute:: speed_of_sound
    .. autoattribute:: mass_density
    .. autoattribute:: momentum_density
    .. autoattribute:: energy_density
    .. autoattribute:: species_mass_density
    .. autoattribute:: species_mass_fractions
    .. autoattribute:: species_enthalpies
    """

    cv: ConservedVars
    dv: GasDependentVars

    @property
    def array_context(self):
        """Return the relevant array context for this object."""
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

    @property
    def wavespeed(self):
        """Return the characteristic wavespeed."""
        return self.cv.speed + self.dv.speed_of_sound

    @property
    def is_viscous(self):
        """Indicate if this is a viscous state."""
        return isinstance(self, ViscousFluidState)

    @property
    def is_mixture(self):
        """Indicate if this is a state resulting from a mixture gas model."""
        return isinstance(self.dv, MixtureDependentVars)

    def _get_mixture_property(self, name):
        """Grab a mixture property if EOS is a :class:`~mirgecom.eos.MixtureEOS`."""
        if not self.is_mixture:
            raise \
                MixtureEOSNeededError("Mixture EOS required for mixture properties.")
        return getattr(self.dv, name)

    @property
    def species_enthalpies(self):
        """Return the fluid species enthalpies."""
        return self._get_mixture_property("species_enthalpies")


@dataclass_array_container
@dataclass(frozen=True)
class ViscousFluidState(FluidState):
    r"""Gas model-consistent fluid state for viscous gas models.

    .. attribute:: tv

        Viscous fluid state-dependent transport properties.

    .. autoattribute:: viscosity
    .. autoattribute:: bulk_viscosity
    .. autoattribute:: species_diffusivity
    .. autoattribute:: thermal_conductivity
    """

    tv: GasTransportVars

    @property
    def viscosity(self):
        """Return the fluid viscosity."""
        return self.tv.viscosity

    @property
    def bulk_viscosity(self):
        """Return the fluid bulk viscosity."""
        return self.tv.bulk_viscosity

    @property
    def thermal_conductivity(self):
        """Return the fluid thermal conductivity."""
        return self.tv.thermal_conductivity

    @property
    def species_diffusivity(self):
        """Return the fluid species diffusivities."""
        return self.tv.species_diffusivity


def make_fluid_state(cv, gas_model, temperature_seed=None):
    """Create a fluid state from the conserved vars and physical gas model.

    Parameters
    ----------
    cv: :class:`~mirgecom.fluid.ConservedVars`

        The gas conserved state

    gas_model: :class:`~mirgecom.gas_model.GasModel`

        The physical model for the gas/fluid.

    temperature_seed: :class:`~meshmode.dof_array.DOFArray` or float

        An optional array or number with the temperature to use as a seed
        for a temperature evaluation for the created fluid state

    Returns
    -------
    :class:`~mirgecom.gas_model.FluidState`

        Thermally consistent fluid state
    """
    dv = gas_model.eos.dependent_vars(cv, temperature_seed=temperature_seed)
    if gas_model.transport is not None:
        tv = gas_model.transport.transport_vars(cv=cv, dv=dv, eos=gas_model.eos)
        return ViscousFluidState(cv=cv, dv=dv, tv=tv)
    return FluidState(cv=cv, dv=dv)


def project_fluid_state(discr, src, tgt, state, gas_model):
    """Project a fluid state onto a boundary consistent with the gas model.

    If required by the gas model, (e.g. gas is a mixture), this routine will
    ensure that the returned state is thermally consistent.

    Parameters
    ----------
    discr: :class:`~grudge.eager.EagerDGDiscretization`

        A discretization collection encapsulating the DG elements

    src:

        A :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one
        indicating where the state is currently defined
        (e.g. "vol" or "all_faces")

    tgt:

        A :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one
        indicating where to interpolate/project the state
        (e.g. "all_faces" or a boundary tag *btag*)

    state: :class:`~mirgecom.gas_model.FluidState`

        The full fluid conserved and thermal state

    gas_model: :class:`~mirgecom.gas_model.GasModel`

        The physical model constructs for the gas_model

    Returns
    -------
    :class:`~mirgecom.gas_model.FluidState`

        Thermally consistent fluid state
    """
    cv_sd = discr.project(src, tgt, state.cv)
    temperature_seed = None
    if state.is_mixture:
        temperature_seed = discr.project(src, tgt, state.dv.temperature)
    return make_fluid_state(cv=cv_sd, gas_model=gas_model,
                            temperature_seed=temperature_seed)


def _getattr_ish(obj, name):
    if obj is None:
        return None
    else:
        return getattr(obj, name)


def make_fluid_state_trace_pairs(cv_pairs, gas_model, temperature_seed_pairs=None):
    """Create a fluid state from the conserved vars and equation of state.

    This routine helps create a thermally consistent fluid state out of a collection
    of  CV (:class:`~mirgecom.fluid.ConservedVars`) pairs.  It is useful for creating
    consistent boundary states for partition boundaries.

    Parameters
    ----------
    cv_pairs: list of :class:`~grudge.trace_pair.TracePair`

        List of tracepairs of fluid CV (:class:`~mirgecom.fluid.ConservedVars`) for
        each boundary on which the thermally consistent state is desired

    gas_model: :class:`~mirgecom.gas_model.GasModel`

        The physical model constructs for the gas_model

    temperature_seed_pairs: list of :class:`~grudge.trace_pair.TracePair`

        List of tracepairs of :class:`~meshmode.dof_array.DOFArray` with the
        temperature seeds to use in creation of the thermally consistent states.

    Returns
    -------
    List of :class:`~grudge.trace_pair.TracePair`

        List of tracepairs of thermally consistent states
        (:class:`~mirgecom.gas_model.FluidState`) for each boundary in the input set
    """
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


class _FluidCVTag:
    pass


class _FluidTemperatureTag:
    pass


def make_operator_fluid_states(discr, volume_state, gas_model, boundaries,
                               quadrature_tag=None):
    """Prepare gas model-consistent fluid states for use in fluid operators.

    This routine prepares a model-constistent fluid state for each of the volume and
    all interior and domain boundaries using the quadrature representation if
    one is given. The input *volume_state* is projected to the quadrature domain
    (if any), along with the model-consistent dependent quantities.

    .. note::

        When running MPI-distributed, volume state conserved quantities
        (ConservedVars), and for mixtures, temperatures will be communicated over
        partition boundaries inside this routine.

    Parameters
    ----------
    discr: :class:`~grudge.eager.EagerDGDiscretization`

        A discretization collection encapsulating the DG elements

    volume_state: :class:`~mirgecom.gas_model.FluidState`

        The full fluid conserved and thermal state

    gas_model: :class:`~mirgecom.gas_model.GasModel`

        The physical model constructs for the gas_model

    boundaries
        Dictionary of boundary functions, one for each valid btag

    quadrature_tag
        An optional identifier denoting a particular quadrature
        discretization to use during operator evaluations.
        The default value is *None*.

    Returns
    -------
    :class:`~mirgecom.gas_model.FluidState`

        Thermally consistent fluid state for the volume, fluid state trace pairs
        for the internal boundaries, and a dictionary of fluid states keyed by
        valid btag in boundaries, all on the quadrature grid (if specified).
    """
    dd_base_vol = DOFDesc("vol")
    dd_quad_vol = DOFDesc("vol", quadrature_tag)

    # project pair to the quadrature discretization and update dd to quad
    interp_to_surf_quad = partial(tracepair_with_discr_tag, discr, quadrature_tag)

    domain_boundary_states_quad = {
        btag: project_fluid_state(discr, dd_base_vol,
                                  as_dofdesc(btag).with_discr_tag(quadrature_tag),
                                  volume_state, gas_model)
        for btag in boundaries
    }

    # performs MPI communication of CV if needed
    cv_interior_pairs = [
        # Get the interior trace pairs onto the surface quadrature
        # discretization (if any)
        interp_to_surf_quad(tpair=tpair)
        for tpair in interior_trace_pairs(discr, volume_state.cv, tag=_FluidCVTag)
    ]

    tseed_interior_pairs = None
    if volume_state.is_mixture:
        # If this is a mixture, we need to exchange the temperature field because
        # mixture pressure (used in the inviscid flux calculations) depends on
        # temperature and we need to seed the temperature calculation for the
        # (+) part of the partition boundary with the remote temperature data.
        tseed_interior_pairs = [
            # Get the interior trace pairs onto the surface quadrature
            # discretization (if any)
            interp_to_surf_quad(tpair=tpair)
            for tpair in interior_trace_pairs(discr, volume_state.temperature,
                                              tag=_FluidTemperatureTag)]

    interior_boundary_states_quad = \
        make_fluid_state_trace_pairs(cv_interior_pairs, gas_model,
                                     tseed_interior_pairs)

    # Interpolate the fluid state to the volume quadrature grid
    # (this includes the conserved and dependent quantities)
    volume_state_quad = project_fluid_state(discr, dd_base_vol, dd_quad_vol,
                                            volume_state, gas_model)

    return \
        volume_state_quad, interior_boundary_states_quad, domain_boundary_states_quad
