""":mod:`mirgecom.gas_model` provides utilities to deal with gases.

Physical Gas Model Encapsulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: GasModel

Fluid State Encapsulation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: FluidState
.. autoclass:: ViscousFluidState
.. autoclass:: PorousFlowFluidState

Fluid State Handling Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: make_fluid_state
.. autofunction:: replace_fluid_state
.. autofunction:: project_fluid_state
.. autofunction:: make_fluid_state_trace_pairs
.. autofunction:: make_operator_fluid_states
.. autofunction:: make_entropy_projected_fluid_state
.. autofunction:: conservative_to_entropy_vars
.. autofunction:: entropy_to_conservative_vars
"""

__copyright__ = """
Copyright (C) 2023 University of Illinois Board of Trustees
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

from functools import partial
from dataclasses import dataclass
from typing import Optional
import numpy as np  # noqa
from arraycontext import dataclass_array_container
from grudge.dof_desc import (
    DD_VOLUME_ALL,
    VolumeDomainTag,
    DISCR_TAG_BASE,
)
import grudge.op as op
from grudge.trace_pair import (
    interior_trace_pairs,
    tracepair_with_discr_tag
)
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
from mirgecom.utils import normalize_boundaries
from mirgecom.wall_model import PorousWallVars, PorousFlowModel


@dataclass(frozen=True)
class GasModel:
    r"""Physical gas model for calculating fluid state-dependent quantities.

    .. attribute:: eos

        A gas equation of state to provide thermal properties.

    .. attribute:: transport

        A gas transport model to provide transport properties. None for inviscid
        models.
    """

    eos: GasEOS
    transport: Optional[TransportModel] = None


@dataclass_array_container
@dataclass(frozen=True, eq=False)
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
    .. autoattribute:: smoothness_mu
    .. autoattribute:: smoothness_kappa
    .. autoattribute:: smoothness_d
    .. autoattribute:: smoothness_beta
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
    def smoothness_mu(self):
        """Return the smoothness_mu field."""
        return self.dv.smoothness_mu

    @property
    def smoothness_kappa(self):
        """Return the smoothness_kappa field."""
        return self.dv.smoothness_kappa

    @property
    def smoothness_d(self):
        """Return the smoothness_d field."""
        return self.dv.smoothness_d

    @property
    def smoothness_beta(self):
        """Return the smoothness_beta field."""
        return self.dv.smoothness_beta

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


@dataclass_array_container
@dataclass(frozen=True, eq=False)
class PorousFlowFluidState(ViscousFluidState):
    """Dependent variables for the (porous) fluid state.

    .. attribute:: wv

        Porous media flow state-dependent properties.
    """

    wv: PorousWallVars


def make_fluid_state(cv, gas_model,
                     temperature_seed=None,
                     smoothness_mu=None,
                     smoothness_kappa=None,
                     smoothness_d=None,
                     smoothness_beta=None,
                     material_densities=None,
                     limiter_func=None, limiter_dd=None):
    """Create a fluid state from the conserved vars and physical gas model.

    Parameters
    ----------
    cv: :class:`~mirgecom.fluid.ConservedVars`

        The gas conserved state

    gas_model: :class:`~mirgecom.gas_model.GasModel` or
        :class:`~mirgecom.wall_model.PorousFlowModel`

        The physical model for the gas/fluid.

    temperature_seed: :class:`~meshmode.dof_array.DOFArray` or float

        An optional array or number with the temperature to use as a seed
        for a temperature evaluation for the created fluid state

    smoothness_mu: :class:`~meshmode.dof_array.DOFArray`

        Optional array containing the smoothness parameter for extra shear
        viscosity in the artificial viscosity.

    smoothness_kappa: :class:`~meshmode.dof_array.DOFArray`

        Optional array containing the smoothness parameter for extra thermal
        conductivity in the artificial viscosity.

    smoothness_d: :class:`~meshmode.dof_array.DOFArray`

        Optional array containing the smoothness parameter for extra species
        diffusivity in the artificial viscosity.

    smoothness_beta: :class:`~meshmode.dof_array.DOFArray`

        Optional array containing the smoothness parameter for extra bulk
        viscosity in the artificial viscosity.

    material_densities: :class:`~meshmode.dof_array.DOFArray` or np.ndarray

        Optional quantity containing the mass of the porous solid.

    limiter_func:

        Callable function to limit the fluid conserved quantities to physically
        valid and realizable values.

    Returns
    -------
    :class:`~mirgecom.gas_model.FluidState`

        Thermally consistent fluid state
    """
    actx = cv.array_context

    # FIXME work-around for now
    smoothness_mu = (actx.np.zeros_like(cv.mass) if smoothness_mu
                     is None else smoothness_mu)
    smoothness_kappa = (actx.np.zeros_like(cv.mass) if smoothness_kappa
                        is None else smoothness_kappa)
    smoothness_beta = (actx.np.zeros_like(cv.mass) if smoothness_beta
                       is None else smoothness_beta)
    smoothness_d = (actx.np.zeros_like(cv.mass) if smoothness_d
                        is None else smoothness_d)

    if isinstance(gas_model, GasModel):
        pressure = None
        temperature = None

        if limiter_func:
            rv = limiter_func(cv=cv, temperature_seed=temperature_seed,
                              gas_model=gas_model, dd=limiter_dd)
            if isinstance(rv, np.ndarray):
                cv, pressure, temperature = rv
            else:
                cv = rv

        if temperature is None:
            temperature = gas_model.eos.temperature(
                cv=cv, temperature_seed=temperature_seed)
        if pressure is None:
            pressure = gas_model.eos.pressure(cv=cv, temperature=temperature)

        dv = GasDependentVars(
            temperature=temperature,
            pressure=pressure,
            speed_of_sound=gas_model.eos.sound_speed(cv, temperature),
            smoothness_mu=smoothness_mu,
            smoothness_kappa=smoothness_kappa,
            smoothness_d=smoothness_d,
            smoothness_beta=smoothness_beta
        )

        from mirgecom.eos import MixtureEOS
        if isinstance(gas_model.eos, MixtureEOS):
            dv = MixtureDependentVars(
                temperature=dv.temperature,
                pressure=dv.pressure,
                speed_of_sound=dv.speed_of_sound,
                smoothness_mu=dv.smoothness_mu,
                smoothness_kappa=dv.smoothness_kappa,
                smoothness_d=dv.smoothness_d,
                smoothness_beta=dv.smoothness_beta,
                species_enthalpies=gas_model.eos.species_enthalpies(cv, temperature)
            )

        if gas_model.transport is not None:
            tv = gas_model.transport.transport_vars(cv=cv, dv=dv, eos=gas_model.eos)
            return ViscousFluidState(cv=cv, dv=dv, tv=tv)

        return FluidState(cv=cv, dv=dv)

    # TODO ideally, we want to avoid using "gas model" because the name contradicts
    # its usage with solid+fluid.
    elif isinstance(gas_model, PorousFlowModel):

        # FIXME per previous review, think of a way to de-couple wall and fluid.
        # ~~~ we need to squeeze wall_eos in gas_model because this is easily
        # accessible everywhere in the code

        tau = gas_model.decomposition_progress(material_densities)
        wv = PorousWallVars(
            material_densities=material_densities,
            tau=tau,
            density=gas_model.solid_density(material_densities),
            void_fraction=gas_model.wall_eos.void_fraction(tau=tau),
            permeability=gas_model.wall_eos.permeability(tau=tau),
            tortuosity=gas_model.wall_eos.tortuosity(tau=tau)
        )

        temperature = gas_model.get_temperature(cv=cv, wv=wv,
            tseed=temperature_seed)

        pressure = gas_model.get_pressure(cv, wv, temperature)

        if limiter_func:
            cv = limiter_func(cv=cv, wv=wv, pressure=pressure,
                              temperature=temperature, dd=limiter_dd)

        dv = MixtureDependentVars(
            temperature=temperature,
            pressure=pressure,
            speed_of_sound=gas_model.eos.sound_speed(cv, temperature),
            smoothness_mu=smoothness_mu,
            smoothness_kappa=smoothness_kappa,
            smoothness_d=smoothness_d,
            smoothness_beta=smoothness_beta,
            species_enthalpies=gas_model.eos.species_enthalpies(cv, temperature),
        )

        tv = gas_model.transport.transport_vars(cv=cv, dv=dv, wv=wv,
            eos=gas_model.eos, wall_eos=gas_model.wall_eos)

        return PorousFlowFluidState(cv=cv, dv=dv, tv=tv, wv=wv)

    else:
        raise TypeError("Invalid type for gas_model")


def project_fluid_state(dcoll, src, tgt, state, gas_model, limiter_func=None,
                        entropy_stable=False):
    """Project a fluid state onto a boundary consistent with the gas model.

    If required by the gas model, (e.g. gas is a mixture), this routine will
    ensure that the returned state is thermally consistent.

    Parameters
    ----------
    dcoll: :class:`~grudge.discretization.DiscretizationCollection`

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

    limiter_func:

        Callable function to limit the fluid conserved quantities to physically
        valid and realizable values.

    Returns
    -------
    :class:`~mirgecom.gas_model.FluidState`

        Thermally consistent fluid state
    """
    cv_sd = op.project(dcoll, src, tgt, state.cv)

    temperature_seed = None
    if state.is_mixture:
        temperature_seed = op.project(dcoll, src, tgt, state.dv.temperature)

    if entropy_stable:
        temp_state = make_fluid_state(cv=cv_sd, gas_model=gas_model,
                                      temperature_seed=temperature_seed,
                                      limiter_func=limiter_func, limiter_dd=tgt)
        gamma = gas_model.eos.gamma(temp_state.cv, temp_state.temperature)
        ev_sd = conservative_to_entropy_vars(gamma, temp_state)
        cv_sd = entropy_to_conservative_vars(gamma, ev_sd)

    smoothness_mu = None
    if state.dv.smoothness_mu is not None:
        smoothness_mu = op.project(dcoll, src, tgt, state.dv.smoothness_mu)

    smoothness_kappa = None
    if state.dv.smoothness_kappa is not None:
        smoothness_kappa = op.project(dcoll, src, tgt, state.dv.smoothness_kappa)

    smoothness_d = None
    if state.dv.smoothness_d is not None:
        smoothness_d = op.project(dcoll, src, tgt, state.dv.smoothness_d)

    smoothness_beta = None
    if state.dv.smoothness_beta is not None:
        smoothness_beta = op.project(dcoll, src, tgt, state.dv.smoothness_beta)

    material_densities = None
    if isinstance(gas_model, PorousFlowModel):
        material_densities = op.project(dcoll, src, tgt, state.wv.material_densities)

    return make_fluid_state(cv=cv_sd, gas_model=gas_model,
                            temperature_seed=temperature_seed,
                            smoothness_mu=smoothness_mu,
                            smoothness_kappa=smoothness_kappa,
                            smoothness_d=smoothness_d,
                            smoothness_beta=smoothness_beta,
                            material_densities=material_densities,
                            limiter_func=limiter_func, limiter_dd=tgt)


def _getattr_ish(obj, name):
    if obj is None:
        return None
    else:
        return getattr(obj, name)


def make_fluid_state_trace_pairs(cv_pairs, gas_model,
                                 temperature_seed_pairs=None,
                                 smoothness_mu_pairs=None,
                                 smoothness_kappa_pairs=None,
                                 smoothness_d_pairs=None,
                                 smoothness_beta_pairs=None,
                                 material_densities_pairs=None,
                                 limiter_func=None):
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

    limiter_func:

        Callable function to limit the fluid conserved quantities to physically
        valid and realizable values.

    Returns
    -------
    List of :class:`~grudge.trace_pair.TracePair`

        List of tracepairs of thermally consistent states
        (:class:`~mirgecom.gas_model.FluidState`) for each boundary in the input set
    """
    from grudge.trace_pair import TracePair
    if temperature_seed_pairs is None:
        temperature_seed_pairs = [None] * len(cv_pairs)
    if smoothness_mu_pairs is None:
        smoothness_mu_pairs = [None] * len(cv_pairs)
    if smoothness_kappa_pairs is None:
        smoothness_kappa_pairs = [None] * len(cv_pairs)
    if smoothness_d_pairs is None:
        smoothness_d_pairs = [None] * len(cv_pairs)
    if smoothness_beta_pairs is None:
        smoothness_beta_pairs = [None] * len(cv_pairs)
    if material_densities_pairs is None:
        material_densities_pairs = [None] * len(cv_pairs)
    return [TracePair(
        cv_pair.dd,
        interior=make_fluid_state(
            cv_pair.int, gas_model,
            temperature_seed=_getattr_ish(tseed_pair, "int"),
            smoothness_mu=_getattr_ish(smoothness_mu_pair, "int"),
            smoothness_kappa=_getattr_ish(smoothness_kappa_pair, "int"),
            smoothness_d=_getattr_ish(smoothness_d_pair, "int"),
            smoothness_beta=_getattr_ish(smoothness_beta_pair, "int"),
            material_densities=_getattr_ish(material_densities_pair, "int"),
            limiter_func=limiter_func, limiter_dd=cv_pair.dd),
        exterior=make_fluid_state(
            cv_pair.ext, gas_model,
            temperature_seed=_getattr_ish(tseed_pair, "ext"),
            smoothness_mu=_getattr_ish(smoothness_mu_pair, "ext"),
            smoothness_kappa=_getattr_ish(smoothness_kappa_pair, "ext"),
            smoothness_d=_getattr_ish(smoothness_d_pair, "ext"),
            smoothness_beta=_getattr_ish(smoothness_beta_pair, "ext"),
            material_densities=_getattr_ish(material_densities_pair, "ext"),
            limiter_func=limiter_func, limiter_dd=cv_pair.dd))
        for cv_pair,
            tseed_pair,
            smoothness_mu_pair,
            smoothness_kappa_pair,
            smoothness_d_pair,
            smoothness_beta_pair,
            material_densities_pair in zip(
                cv_pairs, temperature_seed_pairs,
                smoothness_mu_pairs, smoothness_kappa_pairs, smoothness_d_pairs,
                smoothness_beta_pairs, material_densities_pairs)]


class _FluidCVTag:
    pass


class _FluidTemperatureTag:
    pass


class _FluidSmoothnessMuTag:
    pass


class _FluidSmoothnessKappaTag:
    pass


class _FluidSmoothnessDiffTag:
    pass


class _FluidSmoothnessBetaTag:
    pass


class _WallDensityTag:
    pass


def make_operator_fluid_states(
        dcoll, volume_state, gas_model, boundaries, quadrature_tag=DISCR_TAG_BASE,
        dd=DD_VOLUME_ALL, comm_tag=None, limiter_func=None, entropy_stable=False):
    """Prepare gas model-consistent fluid states for use in fluid operators.

    This routine prepares a model-consistent fluid state for each of the volume and
    all interior and domain boundaries, using the quadrature representation if
    one is given. The input *volume_state* is projected to the quadrature domain
    (if any), along with the model-consistent dependent quantities.

    .. note::

        When running MPI-distributed, volume state conserved quantities
        (ConservedVars), and for mixtures, temperatures will be communicated over
        partition boundaries inside this routine.

    Parameters
    ----------
    dcoll: :class:`~grudge.discretization.DiscretizationCollection`

        A discretization collection encapsulating the DG elements

    volume_state: :class:`~mirgecom.gas_model.FluidState`

        The full fluid conserved and thermal state

    gas_model: :class:`~mirgecom.gas_model.GasModel`

        The physical model constructs for the gas_model

    boundaries
        Dictionary of boundary functions, one for each valid
        :class:`~grudge.dof_desc.BoundaryDomainTag`.

    quadrature_tag
        An identifier denoting a particular quadrature discretization to use during
        operator evaluations.

    dd: grudge.dof_desc.DOFDesc
        the DOF descriptor of the discretization on which *volume_state* lives. Must
        be a volume on the base discretization.

    comm_tag: Hashable

        Tag for distributed communication

    limiter_func:

        Callable function to limit the fluid conserved quantities to physically
        valid and realizable values.

    Returns
    -------
    (:class:`~mirgecom.gas_model.FluidState`, :class:`~grudge.trace_pair.TracePair`,
     dict)

        Thermally consistent fluid state for the volume, fluid state trace pairs
        for the internal boundaries, and a dictionary of fluid states keyed by
        boundary domain tags in *boundaries*, all on the quadrature grid (if
        specified).
    """
    boundaries = normalize_boundaries(boundaries)

    if not isinstance(dd.domain_tag, VolumeDomainTag):
        raise TypeError("dd must represent a volume")
    if dd.discretization_tag != DISCR_TAG_BASE:
        raise ValueError("dd must belong to the base discretization")

    dd_vol = dd
    dd_vol_quad = dd_vol.with_discr_tag(quadrature_tag)

    # project pair to the quadrature discretization and update dd to quad
    interp_to_surf_quad = partial(tracepair_with_discr_tag, dcoll, quadrature_tag)

    domain_boundary_states_quad = {
        bdtag: project_fluid_state(
            dcoll, dd_vol, dd_vol_quad.with_domain_tag(bdtag),
            volume_state, gas_model, limiter_func=limiter_func,
            entropy_stable=entropy_stable)
        for bdtag in boundaries
    }

    # performs MPI communication of CV if needed
    cv_interior_pairs = [
        # Get the interior trace pairs onto the surface quadrature
        # discretization (if any)
        interp_to_surf_quad(tpair=tpair)
        for tpair in interior_trace_pairs(
            dcoll, volume_state.cv, volume_dd=dd_vol,
            comm_tag=(_FluidCVTag, comm_tag))
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
            for tpair in interior_trace_pairs(
                dcoll, volume_state.temperature, volume_dd=dd_vol,
                comm_tag=(_FluidTemperatureTag, comm_tag))]

    smoothness_mu_interior_pairs = None
    if volume_state.smoothness_mu is not None:
        smoothness_mu_interior_pairs = [
            interp_to_surf_quad(tpair=tpair)
            for tpair in interior_trace_pairs(
                dcoll, volume_state.smoothness_mu, volume_dd=dd_vol,
                comm_tag=(_FluidSmoothnessMuTag, comm_tag))]

    smoothness_kappa_interior_pairs = None
    if volume_state.smoothness_kappa is not None:
        smoothness_kappa_interior_pairs = [
            interp_to_surf_quad(tpair=tpair)
            for tpair in interior_trace_pairs(
                dcoll, volume_state.smoothness_kappa, volume_dd=dd_vol,
                comm_tag=(_FluidSmoothnessKappaTag, comm_tag))]

    smoothness_d_interior_pairs = None
    if volume_state.smoothness_d is not None:
        smoothness_d_interior_pairs = [
            interp_to_surf_quad(tpair=tpair)
            for tpair in interior_trace_pairs(
                dcoll, volume_state.smoothness_d, volume_dd=dd_vol,
                comm_tag=(_FluidSmoothnessDiffTag, comm_tag))]

    smoothness_beta_interior_pairs = None
    if volume_state.smoothness_beta is not None:
        smoothness_beta_interior_pairs = [
            interp_to_surf_quad(tpair=tpair)
            for tpair in interior_trace_pairs(
                dcoll, volume_state.smoothness_beta, volume_dd=dd_vol,
                comm_tag=(_FluidSmoothnessBetaTag, comm_tag))]

    material_densities_interior_pairs = None
    if isinstance(gas_model, PorousFlowModel):
        material_densities_interior_pairs = [
            interp_to_surf_quad(tpair=tpair)
            for tpair in interior_trace_pairs(
                dcoll, volume_state.wv.material_densities, volume_dd=dd_vol,
                comm_tag=(_WallDensityTag, comm_tag))]

    interior_boundary_states_quad = make_fluid_state_trace_pairs(
        cv_pairs=cv_interior_pairs,
        gas_model=gas_model,
        temperature_seed_pairs=tseed_interior_pairs,
        smoothness_mu_pairs=smoothness_mu_interior_pairs,
        smoothness_kappa_pairs=smoothness_kappa_interior_pairs,
        smoothness_d_pairs=smoothness_d_interior_pairs,
        smoothness_beta_pairs=smoothness_beta_interior_pairs,
        material_densities_pairs=material_densities_interior_pairs,
        limiter_func=limiter_func)

    # Interpolate the fluid state to the volume quadrature grid
    # (this includes the conserved and dependent quantities)
    volume_state_quad = project_fluid_state(
        dcoll, dd_vol, dd_vol_quad, volume_state, gas_model,
        limiter_func=limiter_func, entropy_stable=entropy_stable)

    return \
        volume_state_quad, interior_boundary_states_quad, domain_boundary_states_quad


def replace_fluid_state(
        state, gas_model, *, mass=None, energy=None, momentum=None,
        species_mass=None, temperature_seed=None, limiter_func=None,
        limiter_dd=None):
    """Create a new fluid state from an existing one with modified data.

    Parameters
    ----------
    state: :class:`~mirgecom.gas_model.FluidState`

        The full fluid conserved and thermal state

    gas_model: :class:`~mirgecom.gas_model.GasModel`

        The physical model for the gas/fluid.

    mass: :class:`~meshmode.dof_array.DOFArray` or :class:`numpy.ndarray`

        Optional :class:`~meshmode.dof_array.DOFArray` for scalars or object array of
        :class:`~meshmode.dof_array.DOFArray` for vector quantities corresponding
        to the mass continuity equation.

    energy: :class:`~meshmode.dof_array.DOFArray` or :class:`numpy.ndarray`

        Optional :class:`~meshmode.dof_array.DOFArray` for scalars or object array of
        :class:`~meshmode.dof_array.DOFArray` for vector quantities corresponding
        to the energy conservation equation.

    momentum: :class:`numpy.ndarray`

        Optional object array (:class:`numpy.ndarray`) with shape ``(ndim,)``
        of :class:`~meshmode.dof_array.DOFArray` , or an object array with shape
        ``(ndim, ndim)`` respectively for scalar or vector quantities corresponding
        to the ndim equations of momentum conservation.

    species_mass: :class:`numpy.ndarray`

        Optional object array (:class:`numpy.ndarray`) with shape ``(nspecies,)``
        of :class:`~meshmode.dof_array.DOFArray`, or an object array with shape
        ``(nspecies, ndim)`` respectively for scalar or vector quantities
        corresponding to the `nspecies` species mass conservation equations.

    temperature_seed: :class:`~meshmode.dof_array.DOFArray` or float

        Optional array or number with the temperature to use as a seed
        for a temperature evaluation for the created fluid state

    limiter_func:

        Callable function to limit the fluid conserved quantities to physically
        valid and realizable values.

    Returns
    -------
    :class:`~mirgecom.gas_model.FluidState`

        The new fluid conserved and thermal state
    """
    new_cv = state.cv.replace(
        mass=(mass if mass is not None else state.cv.mass),
        energy=(energy if energy is not None else state.cv.energy),
        momentum=(momentum if momentum is not None else state.cv.momentum),
        species_mass=(
            species_mass if species_mass is not None else state.cv.species_mass))

    new_tseed = (
        temperature_seed
        if temperature_seed is not None
        else state.temperature)

    material_densities = (None
        if isinstance(gas_model, PorousFlowModel) is False
        else state.wv.material_densities)

    return make_fluid_state(
        cv=new_cv,
        gas_model=gas_model,
        temperature_seed=new_tseed,
        smoothness_mu=state.smoothness_mu,
        smoothness_kappa=state.smoothness_kappa,
        smoothness_d=state.smoothness_d,
        smoothness_beta=state.smoothness_beta,
        material_densities=material_densities,
        limiter_func=limiter_func,
        limiter_dd=limiter_dd)


def make_entropy_projected_fluid_state(
        discr, dd_vol, dd_faces, state, entropy_vars, gamma, gas_model):
    """Projects the entropy vars to target manifold, computes the CV from that."""
    from grudge.interpolation import volume_and_surface_quadrature_interpolation

    # Interpolate to the volume and surface (concatenated) quadrature
    # discretizations: v = [v_vol, v_surf]
    ev_quad = volume_and_surface_quadrature_interpolation(
        discr, dd_vol, dd_faces, entropy_vars)

    temperature_seed = None
    if state.is_mixture:
        temperature_seed = volume_and_surface_quadrature_interpolation(
            discr, dd_vol, dd_faces, state.temperature)
        gamma = volume_and_surface_quadrature_interpolation(
            discr, dd_vol, dd_faces, gamma)

    # Convert back to conserved varaibles and use to make the new fluid state
    cv_modified = entropy_to_conservative_vars(gamma, ev_quad)

    return make_fluid_state(cv=cv_modified,
                            gas_model=gas_model,
                            temperature_seed=temperature_seed)


# TODO: Revisit for Entropy Stable flamelet mixture
def conservative_to_entropy_vars(gamma, state):
    """Compute the entropy variables from conserved variables.

    Converts from conserved variables
    (density, momentum, total energy, species densities)
    into entropy variables, per eqn. 4.5 of [Chandrashekar_2013]_.

    Parameters
    ----------
    state: :class:`~mirgecom.gas_model.FluidState`
        The full fluid conserved and thermal state

    Returns
    -------
    ConservedVars
        The entropy variables
    """
    from mirgecom.fluid import make_conserved

    dim = state.dim
    actx = state.array_context

    rho = state.mass_density
    u = state.velocity
    p = state.pressure
    y_species = state.species_mass_fractions

    u_square = np.dot(u, u)
    s = actx.np.log(p) - gamma*actx.np.log(rho)
    rho_p = rho / p
    ev_mass = ((gamma - s) / (gamma - 1)) - 0.5 * rho_p * u_square
    ev_spec = y_species / (gamma - 1)

    return make_conserved(dim, mass=ev_mass, energy=-rho_p, momentum=rho_p * u,
                          species_mass=ev_spec)


def entropy_to_conservative_vars(gamma, ev: ConservedVars):
    """Compute the conserved variables from entropy variables *ev*.

    Converts from entropy variables into conserved variables
    (density, momentum, total energy, species density) per
    eqn. 4.5 of [Chandrashekar_2013]_.

    Parameters
    ----------
    ev: ConservedVars
        The entropy variables

    Returns
    -------
    ConservedVars
        The fluid conserved variables
    """
    from mirgecom.fluid import make_conserved

    dim = ev.dim
    actx = ev.array_context
    # See Hughes, Franca, Mallet (1986) A new finite element
    # formulation for CFD: (DOI: 10.1016/0045-7825(86)90127-1)
    inv_gamma_minus_one = 1/(gamma - 1)

    # Convert to entropy `-rho * s` used by Hughes, France, Mallet (1986)
    ev_state = ev * (gamma - 1)
    v1 = ev_state.mass
    v234 = ev_state.momentum
    v5 = ev_state.energy
    v6ns = ev_state.species_mass

    v_square = np.dot(v234, v234)
    s = gamma - v1 + v_square/(2*v5)
    iota = ((gamma - 1) / (-v5)**gamma)**(inv_gamma_minus_one)
    rho_iota = iota * actx.np.exp(-s * inv_gamma_minus_one)
    mass = -rho_iota * v5
    spec_mass = mass * v6ns

    return make_conserved(
        dim,
        mass=mass,
        energy=rho_iota * (1 - v_square/(2*v5)),
        momentum=rho_iota * v234,
        species_mass=spec_mass
    )
