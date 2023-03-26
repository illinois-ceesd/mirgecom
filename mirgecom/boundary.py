""":mod:`mirgecom.boundary` provides methods and constructs for boundary treatments.

Boundary Treatment Interfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: FluidBoundary

Boundary Conditions Base Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: PrescribedFluidBoundary

Boundary Conditions
^^^^^^^^^^^^^^^^^^^

.. autoclass:: DummyBoundary
.. autoclass:: AdiabaticSlipBoundary
.. autoclass:: AdiabaticNoslipMovingBoundary
.. autoclass:: IsothermalNoSlipBoundary
.. autoclass:: FarfieldBoundary
.. autoclass:: RiemannInflowBoundary
.. autoclass:: RiemannOutflowBoundary
.. autoclass:: PressureOutflowBoundary
.. autoclass:: IsothermalWallBoundary
.. autoclass:: AdiabaticNoslipWallBoundary
.. autoclass:: LinearizedOutflowBoundary
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

from warnings import warn
import numpy as np
from functools import partial
from arraycontext import get_container_context_recursively
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.discretization.connection import FACE_RESTR_ALL
from grudge.dof_desc import as_dofdesc
from mirgecom.fluid import make_conserved
from grudge.trace_pair import TracePair
import grudge.op as op
from mirgecom.viscous import viscous_facial_flux_central
from mirgecom.flux import num_flux_central
from mirgecom.gas_model import make_fluid_state
from pytools.obj_array import make_obj_array

from mirgecom.inviscid import inviscid_facial_flux_rusanov

from abc import ABCMeta, abstractmethod


def _ldg_bnd_flux_for_grad(internal_quantity, external_quantity):
    return external_quantity


def _get_normal_axes(seed_vector):
    actx = get_container_context_recursively(seed_vector)
    vec_dim, = seed_vector.shape

    vec_mag = actx.np.sqrt(np.dot(seed_vector, seed_vector))
    seed_vector = seed_vector / vec_mag

    if vec_dim == 1:
        return seed_vector,

    if vec_dim == 2:
        vector_2 = 0*seed_vector
        vector_2[0] = -1.*seed_vector[1]
        vector_2[1] = 1.*seed_vector[0]
        return seed_vector, vector_2

    if vec_dim == 3:
        x_comp = seed_vector[0]
        y_comp = seed_vector[1]
        z_comp = seed_vector[2]
        zsign = z_comp / actx.np.abs(z_comp)

        a = vec_mag * zsign
        b = z_comp + a

        vector_2 = 0*seed_vector
        vector_2[0] = a*b - x_comp*x_comp
        vector_2[1] = -x_comp*y_comp
        vector_2[2] = -x_comp*b
        vec_mag2 = actx.np.sqrt(np.dot(vector_2, vector_2))
        vector_2 = vector_2 / vec_mag2
        x_comp_2 = vector_2[0]
        y_comp_2 = vector_2[1]
        z_comp_2 = vector_2[2]

        vector_3 = 0*vector_2
        vector_3[0] = y_comp*z_comp_2 - y_comp_2*z_comp
        vector_3[1] = x_comp_2*z_comp - x_comp*z_comp_2
        vector_3[2] = x_comp*y_comp_2 - y_comp*x_comp_2

    return seed_vector, vector_2, vector_3


def _get_rotation_matrix(principal_direction):
    principal_axes = _get_normal_axes(principal_direction)
    dim, = principal_direction.shape
    comps = []

    for d in range(dim):
        axis = principal_axes[d]
        for i in range(dim):
            comps.append(axis[i])

    comps = make_obj_array(comps)
    return comps.reshape(dim, dim)


def _identical_state(state_minus, **kwargs):
    return state_minus


def _identical_temperature(
        dcoll, dd_bdry, gas_model, state_minus, **kwargs):
    return state_minus.temperature


def _identical_grad_cv(grad_cv_minus, **kwargs):
    return grad_cv_minus


def _identical_grad_temperature(dcoll, dd_bdry, grad_t_minus, **kwargs):
    return grad_t_minus


def _identical_grad_av(grad_av_minus, **kwargs):
    return grad_av_minus


class FluidBoundary(metaclass=ABCMeta):
    r"""Abstract interface to fluid boundary treatment.

    .. automethod:: inviscid_divergence_flux
    .. automethod:: viscous_divergence_flux
    .. automethod:: cv_gradient_flux
    .. automethod:: temperature_gradient_flux
    """

    @abstractmethod
    def inviscid_divergence_flux(self, dcoll, dd_bdry, gas_model, state_minus,
                                 numerical_flux_func, **kwargs):
        """Get the inviscid boundary flux for the divergence operator.

        This routine returns the facial flux used in the divergence
        of the inviscid fluid transport flux.

        Parameters
        ----------
        dcoll: :class:`~grudge.discretization.DiscretizationCollection`

            A discretization collection encapsulating the DG elements

        state_minus: :class:`~mirgecom.gas_model.FluidState`

            Fluid state object with the conserved state, and dependent
            quantities for the (-) side of the boundary specified by
            *dd_bdry*.

        dd_bdry:

            Boundary DOF descriptor (or object convertible to one) indicating which
            domain boundary to process

        gas_model: :class:`~mirgecom.gas_model.GasModel`

            Physical gas model including equation of state, transport,
            and kinetic properties as required by fluid state

        numerical_flux_func:

            Function should return the numerical flux corresponding to
            the divergence of the inviscid transport flux. This function
            is typically backed by an approximate Riemann solver, such as
            :func:`~mirgecom.inviscid.inviscid_facial_flux_rusanov`.

        Returns
        -------
        :class:`mirgecom.fluid.ConservedVars`
        """

    @abstractmethod
    def viscous_divergence_flux(self, dcoll, dd_bdry, gas_model, state_minus,
                                grad_cv_minus, grad_t_minus,
                                numerical_flux_func, **kwargs):
        """Get the viscous boundary flux for the divergence operator.

        This routine returns the facial flux used in the divergence
        of the viscous fluid transport flux.

        Parameters
        ----------
        dcoll: :class:`~grudge.discretization.DiscretizationCollection`

            A discretization collection encapsulating the DG elements

        dd_bdry:

            Boundary DOF descriptor (or object convertible to one) indicating which
            domain boundary to process

        state_minus: :class:`~mirgecom.gas_model.FluidState`

            Fluid state object with the conserved state, and dependent
            quantities for the (-) side of the boundary specified
            by *dd_bdry*.

        grad_cv_minus: :class:`~mirgecom.fluid.ConservedVars`

            The gradient of the conserved quantities on the (-) side
            of the boundary specified by *dd_bdry*.

        grad_t_minus: numpy.ndarray

            The gradient of the fluid temperature on the (-) side
            of the boundary specified by *dd_bdry*.

        gas_model: :class:`~mirgecom.gas_model.GasModel`

            Physical gas model including equation of state, transport,
            and kinetic properties as required by fluid state

        numerical_flux_func:

            Function should return the numerical flux corresponding to
            the divergence of the viscous transport flux. This function
            is typically backed by a helper, such as
            :func:`~mirgecom.viscous.viscous_facial_flux_central`.

        Returns
        -------
        :class:`mirgecom.fluid.ConservedVars`
        """

    @abstractmethod
    def cv_gradient_flux(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the boundary flux for the gradient of the fluid conserved variables.

        This routine returns the facial flux used by the gradient operator to
        compute the gradient of the fluid solution on a domain boundary.

        Parameters
        ----------
        dcoll: :class:`~grudge.discretization.DiscretizationCollection`

            A discretization collection encapsulating the DG elements

        dd_bdry:

            Boundary DOF descriptor (or object convertible to one) indicating which
            domain boundary to process

        state_minus: :class:`~mirgecom.gas_model.FluidState`

            Fluid state object with the conserved state, and dependent
            quantities for the (-) side of the boundary specified by
            *dd_bdry*.

        gas_model: :class:`~mirgecom.gas_model.GasModel`

            Physical gas model including equation of state, transport,
            and kinetic properties as required by fluid state

        Returns
        -------
        :class:`mirgecom.fluid.ConservedVars`
        """

    @abstractmethod
    def temperature_gradient_flux(self, dcoll, dd_bdry, gas_model, state_minus,
                                  **kwargs):
        """Get the boundary flux for the gradient of the fluid temperature.

        This method returns the boundary flux to be used by the gradient
        operator when computing the gradient of the fluid temperature at a
        domain boundary.

        Parameters
        ----------
        dcoll: :class:`~grudge.discretization.DiscretizationCollection`

            A discretization collection encapsulating the DG elements

        dd_bdry:

            Boundary DOF descriptor (or object convertible to one) indicating which
            domain boundary to process

        state_minus: :class:`~mirgecom.gas_model.FluidState`

            Fluid state object with the conserved state, and dependent
            quantities for the (-) side of the boundary specified by
            *dd_bdry*.

        gas_model: :class:`~mirgecom.gas_model.GasModel`

            Physical gas model including equation of state, transport,
            and kinetic properties as required by fluid state

        Returns
        -------
        numpy.ndarray
        """


# This class is a FluidBoundary that provides default implementations of
# the abstract methods in FluidBoundary. This class will be eliminated
# by resolution of https://github.com/illinois-ceesd/mirgecom/issues/576.
# TODO: Don't do this. Make every boundary condition implement its own
# version of the FluidBoundary methods.
class PrescribedFluidBoundary(FluidBoundary):
    r"""Abstract interface to a prescribed fluid boundary treatment.

    .. automethod:: __init__
    .. automethod:: inviscid_divergence_flux
    .. automethod:: viscous_divergence_flux
    .. automethod:: cv_gradient_flux
    .. automethod:: temperature_gradient_flux
    .. automethod:: av_flux
    """

    def __init__(self,
                 # returns the flux to be used in div op (prescribed flux)
                 inviscid_flux_func=None,
                 # returns CV+, to be used in num flux func (prescribed soln)
                 boundary_state_func=None,
                 # Flux to be used in grad(Temperature) op
                 temperature_gradient_flux_func=None,
                 # Function returns boundary temperature_plus
                 boundary_temperature_func=None,
                 # Function returns the flux to be used in grad(cv)
                 cv_gradient_flux_func=None,
                 # Function computes the numerical flux for a gradient
                 gradient_numerical_flux_func=None,
                 # Function computes the flux to be used in the div op
                 viscous_flux_func=None,
                 # Returns the boundary value for grad(cv)
                 boundary_gradient_cv_func=None,
                 # Returns the boundary value for grad(temperature)
                 boundary_gradient_temperature_func=None,
                 # For artificial viscosity - grad fluid soln on boundary
                 boundary_grad_av_func=None,
                 ):
        """Initialize the PrescribedFluidBoundary and methods."""
        self._bnd_state_func = boundary_state_func
        self._temperature_grad_flux_func = temperature_gradient_flux_func
        self._inviscid_flux_func = inviscid_flux_func
        self._bnd_temperature_func = boundary_temperature_func
        self._grad_num_flux_func = gradient_numerical_flux_func
        self._cv_gradient_flux_func = cv_gradient_flux_func
        self._viscous_flux_func = viscous_flux_func
        self._bnd_grad_cv_func = boundary_gradient_cv_func
        self._bnd_grad_temperature_func = boundary_gradient_temperature_func
        self._av_num_flux_func = num_flux_central
        self._bnd_grad_av_func = boundary_grad_av_func

        if not self._bnd_grad_av_func:
            self._bnd_grad_av_func = _identical_grad_av

        if not self._inviscid_flux_func and not self._bnd_state_func:
            from warnings import warn
            warn("Using dummy boundary: copies interior solution.", stacklevel=2)

        if not self._inviscid_flux_func:
            self._inviscid_flux_func = self._inviscid_flux_for_prescribed_state

        if not self._bnd_state_func:
            self._bnd_state_func = _identical_state

        if not self._bnd_temperature_func:
            self._bnd_temperature_func = self._temperature_for_prescribed_state
        if not self._grad_num_flux_func:
            # self._grad_num_flux_func = num_flux_central
            self._grad_num_flux_func = _ldg_bnd_flux_for_grad

        if not self._cv_gradient_flux_func:
            self._cv_gradient_flux_func = self._gradient_flux_for_prescribed_cv
        if not self._temperature_grad_flux_func:
            self._temperature_grad_flux_func = \
                self._gradient_flux_for_prescribed_temperature

        if not self._viscous_flux_func:
            self._viscous_flux_func = self._viscous_flux_for_prescribed_state
        if not self._bnd_grad_cv_func:
            self._bnd_grad_cv_func = _identical_grad_cv
        if not self._bnd_grad_temperature_func:
            self._bnd_grad_temperature_func = _identical_grad_temperature

    def _boundary_quantity(self, dcoll, dd_bdry, quantity, local=False, **kwargs):
        """Get a boundary quantity on local boundary, or projected to "all_faces"."""
        dd_allfaces = dd_bdry.with_boundary_tag(FACE_RESTR_ALL)
        return quantity if local else op.project(dcoll,
            dd_bdry, dd_allfaces, quantity)

    def _boundary_state_pair(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        return TracePair(dd_bdry,
                         interior=state_minus,
                         exterior=self._bnd_state_func(dcoll=dcoll, dd_bdry=dd_bdry,
                                                       gas_model=gas_model,
                                                       state_minus=state_minus,
                                                       **kwargs))
    # The following methods provide default implementations of the fluid
    # boundary functions and helpers in an effort to eliminate much
    # repeated code. They will be eliminated by the resolution of
    # https://github.com/illinois-ceesd/mirgecom/issues/576.

    # {{{ Default boundary helpers

    # Returns temperature(+) for boundaries that prescribe CV(+)
    def _temperature_for_prescribed_state(self, dcoll, dd_bdry,
                                          gas_model, state_minus, **kwargs):
        boundary_state = self._bnd_state_func(dcoll=dcoll, dd_bdry=dd_bdry,
                                              gas_model=gas_model,
                                              state_minus=state_minus,
                                              **kwargs)
        return boundary_state.temperature

    # Returns the flux to be used by the gradient operator when computing the
    # gradient of the fluid solution on boundaries that prescribe CV(+).
    def _gradient_flux_for_prescribed_cv(self, dcoll, dd_bdry, gas_model,
                                         state_minus, **kwargs):
        # Use prescribed external state and gradient numerical flux function
        boundary_state = self._bnd_state_func(dcoll=dcoll, dd_bdry=dd_bdry,
                                              gas_model=gas_model,
                                              state_minus=state_minus,
                                              **kwargs)
        cv_pair = TracePair(dd_bdry,
                            interior=state_minus.cv,
                            exterior=boundary_state.cv)

        actx = state_minus.array_context
        nhat = actx.thaw(dcoll.normal(dd_bdry))
        from arraycontext import outer
        return outer(self._grad_num_flux_func(cv_pair.int, cv_pair.ext), nhat)

    # Returns the flux to be used by the gradient operator when computing the
    # gradient of fluid temperature using prescribed fluid temperature(+).
    def _gradient_flux_for_prescribed_temperature(self, dcoll, dd_bdry, gas_model,
                                                  state_minus, **kwargs):
        # Feed a boundary temperature to numerical flux for grad op
        actx = state_minus.array_context
        nhat = actx.thaw(dcoll.normal(dd_bdry))
        bnd_tpair = TracePair(dd_bdry,
                              interior=state_minus.temperature,
                              exterior=self._bnd_temperature_func(
                                  dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
                                  state_minus=state_minus, **kwargs))
        from arraycontext import outer
        return outer(self._grad_num_flux_func(bnd_tpair.int, bnd_tpair.ext), nhat)

    # Returns the flux to be used by the divergence operator when computing the
    # divergence of inviscid fluid transport flux using the boundary's
    # prescribed CV(+).
    def _inviscid_flux_for_prescribed_state(
            self, dcoll, dd_bdry, gas_model, state_minus,
            numerical_flux_func=inviscid_facial_flux_rusanov, **kwargs):
        # Use a prescribed boundary state and the numerical flux function
        boundary_state_pair = self._boundary_state_pair(dcoll=dcoll, dd_bdry=dd_bdry,
                                                        gas_model=gas_model,
                                                        state_minus=state_minus,
                                                        **kwargs)
        normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))
        return numerical_flux_func(boundary_state_pair, gas_model, normal)

    # Returns the flux to be used by the divergence operator when computing the
    # divergence of viscous fluid transport flux using the boundary's
    # prescribed CV(+).
    def _viscous_flux_for_prescribed_state(
            self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
            grad_t_minus, numerical_flux_func=viscous_facial_flux_central, **kwargs):

        state_pair = self._boundary_state_pair(
            dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
            state_minus=state_minus, **kwargs)

        grad_cv_pair = \
            TracePair(dd_bdry, interior=grad_cv_minus,
                      exterior=self._bnd_grad_cv_func(
                          dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
                          state_minus=state_minus, grad_cv_minus=grad_cv_minus,
                          grad_t_minus=grad_t_minus))

        grad_t_pair = \
            TracePair(
                dd_bdry, interior=grad_t_minus,
                exterior=self._bnd_grad_temperature_func(
                    dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
                    state_minus=state_minus, grad_cv_minus=grad_cv_minus,
                    grad_t_minus=grad_t_minus))

        return numerical_flux_func(
            dcoll=dcoll, gas_model=gas_model, state_pair=state_pair,
            grad_cv_pair=grad_cv_pair, grad_t_pair=grad_t_pair)

    # }}} Default boundary helpers

    def inviscid_divergence_flux(self, dcoll, dd_bdry, gas_model, state_minus,
                                 numerical_flux_func=inviscid_facial_flux_rusanov,
                                 **kwargs):
        """Get the inviscid boundary flux for the divergence operator."""
        dd_bdry = as_dofdesc(dd_bdry)
        return self._inviscid_flux_func(dcoll, dd_bdry, gas_model, state_minus,
                                        numerical_flux_func=numerical_flux_func,
                                        **kwargs)

    def cv_gradient_flux(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the cv flux for *dd_bdry* for use in the gradient operator."""
        dd_bdry = as_dofdesc(dd_bdry)
        return self._cv_gradient_flux_func(
            dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
            state_minus=state_minus, **kwargs)

    def temperature_gradient_flux(self, dcoll, dd_bdry, gas_model, state_minus,
                                  **kwargs):
        """Get the T flux for *dd_bdry* for use in the gradient operator."""
        dd_bdry = as_dofdesc(dd_bdry)
        return self._temperature_grad_flux_func(dcoll, dd_bdry, gas_model,
                                                state_minus, **kwargs)

    def viscous_divergence_flux(self, dcoll, dd_bdry, gas_model, state_minus,
                                grad_cv_minus, grad_t_minus,
                                numerical_flux_func=viscous_facial_flux_central,
                                **kwargs):
        """Get the viscous flux for *dd_bdry* for use in the divergence operator."""
        dd_bdry = as_dofdesc(dd_bdry)
        return self._viscous_flux_func(dcoll=dcoll, dd_bdry=dd_bdry,
                                       gas_model=gas_model,
                                       state_minus=state_minus,
                                       grad_cv_minus=grad_cv_minus,
                                       grad_t_minus=grad_t_minus,
                                       numerical_flux_func=numerical_flux_func,
                                       **kwargs)

    # {{{ Boundary interface for artificial viscosity

    def av_flux(self, dcoll, dd_bdry, diffusion, **kwargs):
        """Get the diffusive fluxes for the AV operator API."""
        dd_bdry = as_dofdesc(dd_bdry)
        grad_av_minus = op.project(dcoll, dd_bdry.untrace(), dd_bdry, diffusion)
        actx = get_container_context_recursively(grad_av_minus)
        nhat = actx.thaw(dcoll.normal(dd_bdry))
        grad_av_plus = self._bnd_grad_av_func(
            dcoll=dcoll, dd_bdry=dd_bdry, grad_av_minus=grad_av_minus, **kwargs)
        bnd_grad_pair = TracePair(dd_bdry, interior=grad_av_minus,
                                  exterior=grad_av_plus)
        num_flux = self._av_num_flux_func(bnd_grad_pair.int, bnd_grad_pair.ext)@nhat
        return self._boundary_quantity(dcoll, dd_bdry, num_flux, **kwargs)

    # }}}


class DummyBoundary(PrescribedFluidBoundary):
    """Boundary type that assigns boundary-adjacent solution to the boundary."""

    def __init__(self):
        """Initialize the DummyBoundary boundary type."""
        PrescribedFluidBoundary.__init__(self)


class _SlipBoundaryComponent:
    def momentum_plus(self, mom_minus, normal):
        return mom_minus - 2.0*np.dot(mom_minus, normal)*normal

    def momentum_bc(self, mom_minus, normal):
        # set the normal momentum to 0
        return mom_minus - np.dot(mom_minus, normal)*normal

    def grad_momentum_bc(
            self, state_minus, state_bc, grad_cv_minus, normal):
        # normal velocity on the surface is zero,
        vel_bc = state_bc.velocity

        from mirgecom.fluid import velocity_gradient
        grad_v_minus = velocity_gradient(state_minus.cv, grad_cv_minus)

        # rotate the velocity gradient tensor into the normal direction
        rotation_matrix = _get_rotation_matrix(normal)
        grad_v_normal = rotation_matrix@grad_v_minus@rotation_matrix.T

        # set the normal component of the tangential velocity to 0
        for i in range(state_minus.dim-1):
            grad_v_normal[i+1][0] = 0.*grad_v_normal[i+1][0]

        # get the gradient on the boundary in the global coordiate space
        grad_v_bc = rotation_matrix.T@grad_v_normal@rotation_matrix

        # construct grad(mom)
        return (
            state_minus.mass_density*grad_v_bc
            + np.outer(vel_bc, grad_cv_minus.mass))


class _NoSlipBoundaryComponent:
    def momentum_plus(self, mom_minus, normal):
        return -mom_minus

    def momentum_bc(self, mom_minus, normal):
        return 0.*mom_minus


class _AdiabaticBoundaryComponent:
    def grad_temperature_bc(self, grad_t_minus, normal):
        return grad_t_minus - np.dot(grad_t_minus, normal)*normal


class _IsothermalBoundaryComponent:
    def __init__(self, t_bc):
        self._t_bc = t_bc

    def temperature_plus(self, state_minus):
        return 2*self._t_bc - state_minus.temperature

    def temperature_bc(self, state_minus):
        return self._t_bc + 0*state_minus.temperature


class _ImpermeableBoundaryComponent:
    def grad_species_mass_bc(self, state_minus, grad_cv_minus, normal):
        nspecies = len(state_minus.species_mass_density)

        grad_species_mass_bc = 1.*grad_cv_minus.species_mass
        if nspecies > 0:
            from mirgecom.fluid import species_mass_fraction_gradient
            grad_y_minus = species_mass_fraction_gradient(state_minus.cv,
                                                          grad_cv_minus)
            grad_y_bc = grad_y_minus - np.outer(grad_y_minus@normal, normal)
            grad_species_mass_bc = 0.*grad_y_bc

            for i in range(nspecies):
                grad_species_mass_bc[i] = \
                    (state_minus.mass_density*grad_y_bc[i]
                     + state_minus.species_mass_fractions[i]*grad_cv_minus.mass)

        return grad_species_mass_bc


# Note: callback function inputs have no default on purpose so that they can't be
# accidentally omitted
def _inviscid_flux_for_prescribed_state_mengaldo(
        dcoll, dd_bdry, gas_model, state_minus, *, state_plus_func,
        numerical_flux_func=inviscid_facial_flux_rusanov,
        **kwargs):
    """Compute the inviscid boundary flux."""
    if state_plus_func is None:
        state_plus_func = _identical_state

    dd_bdry = as_dofdesc(dd_bdry)
    normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))

    state_plus = state_plus_func(
        dcoll, dd_bdry, gas_model, state_minus, **kwargs)
    state_pair = TracePair(dd_bdry, interior=state_minus, exterior=state_plus)

    return numerical_flux_func(state_pair, gas_model, normal)


# Note: callback function inputs have no default on purpose so that they can't be
# accidentally omitted
def _viscous_flux_for_prescribed_state_mengaldo(
        dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
        grad_t_minus, *, state_bc_func, grad_cv_bc_func, grad_temperature_bc_func,
        numerical_flux_func=viscous_facial_flux_central,
        **kwargs):
    """Return the boundary flux for the divergence of the viscous flux."""
    if state_bc_func is None:
        state_bc_func = _identical_state
    if grad_cv_bc_func is None:
        grad_cv_bc_func = _identical_grad_cv
    if grad_temperature_bc_func is None:
        grad_temperature_bc_func = _identical_grad_temperature

    dd_bdry = as_dofdesc(dd_bdry)
    normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))

    state_bc = state_bc_func(
        dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
        state_minus=state_minus, **kwargs)

    grad_cv_bc = grad_cv_bc_func(
        dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
        state_minus=state_minus, state_bc=state_bc, grad_cv_minus=grad_cv_minus,
        grad_t_minus=grad_t_minus, **kwargs)

    grad_t_bc = grad_temperature_bc_func(
        dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
        state_minus=state_minus, grad_cv_minus=grad_cv_minus,
        grad_t_minus=grad_t_minus, **kwargs)

    from mirgecom.viscous import viscous_flux
    return viscous_flux(state_bc, grad_cv_bc, grad_t_bc) @ normal


class AdiabaticSlipBoundary(PrescribedFluidBoundary):
    r"""Boundary condition implementing inviscid slip boundary.

    a.k.a. Reflective inviscid wall boundary

    This class implements an adiabatic reflective slip boundary given by
    $\mathbf{q^{+}} = [\rho^{-}, (\rho{E})^{-}, (\rho\vec{V})^{-}
    - 2((\rho\vec{V})^{-}\cdot\hat{\mathbf{n}}) \hat{\mathbf{n}}]$
    wherein the normal component of velocity at the wall is 0, and tangential
    components are preserved. This class implements an adiabatic slip wall
    consistent with the prescription by [Mengaldo_2014]_ and correspond to the
    characteristic boundary conditions described in detail in [Poinsot_1992]_.

    .. automethod:: __init__
    .. automethod:: state_plus
    .. automethod:: state_bc
    .. automethod:: grad_cv_bc
    .. automethod:: grad_temperature_bc
    .. automethod:: grad_av_plus
    """

    def __init__(self):
        """Initialize the boundary condition object."""
        self.inviscid_flux = partial(
            _inviscid_flux_for_prescribed_state_mengaldo,
            state_plus_func=self.state_plus)

        self.viscous_flux = partial(
            _viscous_flux_for_prescribed_state_mengaldo,
            state_bc_func=self.state_bc,
            grad_cv_bc_func=self.grad_cv_bc,
            grad_temperature_bc_func=self.grad_temperature_bc)

        PrescribedFluidBoundary.__init__(
            self,
            boundary_state_func=self.state_bc,
            inviscid_flux_func=self.inviscid_flux,
            viscous_flux_func=self.viscous_flux,
            boundary_gradient_cv_func=self.grad_cv_bc,
            boundary_gradient_temperature_func=self.grad_temperature_bc)

        self._adiabatic = _AdiabaticBoundaryComponent()
        self._slip = _SlipBoundaryComponent()
        self._impermeable = _ImpermeableBoundaryComponent()

    def state_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the exterior solution on the boundary.

        The exterior solution is set such that there will be vanishing
        flux through the boundary, preserving mass, momentum (magnitude) and
        energy.
        rho_plus = rho_minus
        v_plus = v_minus - 2 * (v_minus . n_hat) * n_hat
        mom_plus = rho_plus * v_plus
        E_plus = E_minus
        """
        dd_bdry = as_dofdesc(dd_bdry)
        normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))

        mom_plus = self._slip.momentum_plus(state_minus.momentum_density, normal)

        # Energy is the same, don't need to compute
        cv_plus = make_conserved(
            state_minus.dim,
            mass=state_minus.mass_density,
            energy=state_minus.energy_density,
            momentum=mom_plus,
            species_mass=state_minus.species_mass_density)

        # we'll need this when we go to production
        return make_fluid_state(cv=cv_plus, gas_model=gas_model,
                                temperature_seed=state_minus.temperature,
                                smoothness=state_minus.smoothness)

    def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Return state with zero normal-component velocity for an adiabatic wall."""
        dd_bdry = as_dofdesc(dd_bdry)
        normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))

        mom_bc = self._slip.momentum_bc(state_minus.momentum_density, normal)

        energy_bc = (
            gas_model.eos.internal_energy(state_minus.cv)
            + 0.5*np.dot(mom_bc, mom_bc)/state_minus.mass_density)

        cv_bc = make_conserved(
            state_minus.dim,
            mass=state_minus.mass_density,
            energy=energy_bc,
            momentum=mom_bc,
            species_mass=state_minus.species_mass_density)

        # we'll need this when we go to production
        return make_fluid_state(cv=cv_bc, gas_model=gas_model,
                                temperature_seed=state_minus.temperature,
                                smoothness=state_minus.smoothness)

    def grad_cv_bc(
            self, dcoll, dd_bdry, gas_model, state_minus, state_bc, grad_cv_minus,
            **kwargs):
        """
        Return external grad(CV) used in the boundary calculation of viscous flux.

        Specify the velocity gradients on the external state to ensure zero
        energy and momentum flux due to shear stresses.

        Gradients of species mass fractions are set to zero in the normal direction
        to ensure zero flux of species across the boundary.
        """
        dd_bdry = as_dofdesc(dd_bdry)
        normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))

        grad_mom_bc = self._slip.grad_momentum_bc(
            state_minus, state_bc, grad_cv_minus, normal)

        grad_species_mass_bc = self._impermeable.grad_species_mass_bc(
            state_minus, grad_cv_minus, normal)

        return make_conserved(
            grad_cv_minus.dim,
            mass=grad_cv_minus.mass,
            energy=grad_cv_minus.energy,
            momentum=grad_mom_bc,
            species_mass=grad_species_mass_bc)

    def grad_temperature_bc(self, dcoll, dd_bdry, gas_model, grad_t_minus, **kwargs):
        """
        Compute temperature gradient on the boundary.

        Impose the opposite normal component to enforce zero energy flux
        from conduction.
        """
        dd_bdry = as_dofdesc(dd_bdry)
        normal = grad_t_minus[0].array_context.thaw(dcoll.normal(dd_bdry))

        return self._adiabatic.grad_temperature_bc(grad_t_minus, normal)

    # FIXME: Remove this?
    def grad_av_plus(self, dcoll, dd_bdry, grad_av_minus, **kwargs):
        """Get the exterior grad(Q) on the boundary for artificial viscosity."""
        # Grab some boundary-relevant data
        dim, = grad_av_minus.mass.shape
        actx = get_container_context_recursively(grad_av_minus)
        nhat = actx.thaw(dcoll.normal(dd_bdry))

        # Subtract 2*wall-normal component of q
        # to enforce q=0 on the wall
        s_mom_normcomp = np.outer(nhat,
                                  np.dot(grad_av_minus.momentum, nhat))
        s_mom_flux = grad_av_minus.momentum - 2*s_mom_normcomp

        # flip components to set a neumann condition
        return make_conserved(
            dim,
            mass=-grad_av_minus.mass,
            energy=-grad_av_minus.energy,
            momentum=-s_mom_flux,
            species_mass=-grad_av_minus.species_mass)


class AdiabaticNoslipMovingBoundary(PrescribedFluidBoundary):
    r"""Boundary condition implementing a no-slip moving boundary.

    This function is deprecated and should be replaced by
    :class:`~mirgecom.boundary.AdiabaticNoslipWallBoundary`

    .. automethod:: adiabatic_noslip_state
    .. automethod:: adiabatic_noslip_grad_av
    """

    def __init__(self, wall_velocity=None, dim=2):
        """Initialize boundary device."""
        warn("AdiabaticNoslipMovingBoundary is deprecated. Use "
             "AdiabaticNoSlipWallBoundary instead.", DeprecationWarning,
             stacklevel=2)

        PrescribedFluidBoundary.__init__(
            self, boundary_state_func=self.adiabatic_noslip_state,
            boundary_temperature_func=_identical_temperature,
            boundary_grad_av_func=self.adiabatic_noslip_grad_av,
        )

        # Check wall_velocity (assumes dim is correct)
        if wall_velocity is None:
            wall_velocity = np.zeros(shape=(dim,))
        if len(wall_velocity) != dim:
            raise ValueError(f"Specified wall velocity must be {dim}-vector.")
        self._wall_velocity = wall_velocity

    def adiabatic_noslip_state(
            self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the exterior solution on the boundary.

        Sets the external state s.t. $v^+ = -v^-$, giving vanishing contact velocity
        in the approximate Riemann solver used to compute the inviscid flux.
        """
        wall_pen = 2.0 * self._wall_velocity * state_minus.mass_density
        ext_mom = wall_pen - state_minus.momentum_density  # no-slip

        # Form the external boundary solution with the new momentum
        cv = make_conserved(
            state_minus.dim,
            mass=state_minus.mass_density,
            energy=state_minus.energy_density,
            momentum=ext_mom,
            species_mass=state_minus.species_mass_density)
        return make_fluid_state(cv=cv, gas_model=gas_model,
                                temperature_seed=state_minus.temperature,
                                smoothness=state_minus.smoothness)

    def adiabatic_noslip_grad_av(self, grad_av_minus, **kwargs):
        """Get the exterior solution on the boundary for artificial viscosity."""
        return -grad_av_minus


class IsothermalNoSlipBoundary(PrescribedFluidBoundary):
    r"""Isothermal no-slip viscous wall boundary.

    This function is deprecated and should be replaced by
    :class:`~mirgecom.boundary.IsothermalWallBoundary`

    .. automethod:: isothermal_noslip_state
    .. automethod:: temperature_bc
    """

    def __init__(self, wall_temperature=300):
        """Initialize the boundary condition object."""
        warn("IsothermalNoSlipBoundary is deprecated. Use IsothermalWallBoundary "
             "instead.", DeprecationWarning, stacklevel=2)

        self._wall_temp = wall_temperature
        PrescribedFluidBoundary.__init__(
            self, boundary_state_func=self.isothermal_noslip_state,
            boundary_temperature_func=self.temperature_bc
        )

    def isothermal_noslip_state(
            self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        r"""Get the interior and exterior solution (*state_minus*) on the boundary.

        Sets the external state s.t. $v^+ = -v^-$, giving vanishing contact velocity
        in the approximate Riemann solver used to compute the inviscid flux.
        """
        temperature_wall = self._wall_temp + 0*state_minus.mass_density
        velocity_plus = -state_minus.velocity
        mass_frac_plus = state_minus.species_mass_fractions

        internal_energy_plus = gas_model.eos.get_internal_energy(
            temperature=temperature_wall, species_mass_fractions=mass_frac_plus)

        total_energy_plus = state_minus.mass_density*(internal_energy_plus
                                           + .5*np.dot(velocity_plus, velocity_plus))

        cv_plus = make_conserved(
            state_minus.dim, mass=state_minus.mass_density, energy=total_energy_plus,
            momentum=-state_minus.momentum_density,
            species_mass=state_minus.species_mass_density
        )
        tseed = state_minus.temperature if state_minus.is_mixture else None
        return make_fluid_state(cv=cv_plus, gas_model=gas_model,
                                temperature_seed=tseed,
                                smoothness=state_minus.smoothness)

    def temperature_bc(self, state_minus, **kwargs):
        r"""Get temperature value to weakly prescribe wall bc.

        Returns $2T_\text{wall} - T^-$ so that a central gradient flux
        will get the correct $T_\text{wall}$ BC.
        """
        return 2*self._wall_temp - state_minus.temperature


class FarfieldBoundary(PrescribedFluidBoundary):
    r"""Farfield boundary treatment.

    This class implements a farfield boundary as described by
    [Mengaldo_2014]_ eqn. 30 and eqn. 42.  The boundary condition is implemented
    as:

    .. math::
        q^{+} = q_\infty

    and the gradients

    .. math::
        \nabla q_{bc} = \nabla q^{-}

    .. automethod:: __init__
    .. automethod:: farfield_state
    .. automethod:: temperature_bc
    """

    def __init__(self, numdim, free_stream_pressure,
                 free_stream_velocity, free_stream_temperature,
                 free_stream_mass_fractions=None):
        """Initialize the boundary condition object."""
        if len(free_stream_velocity) != numdim:
            raise ValueError("Free-stream velocity must be of ambient dimension.")

        self._temperature = free_stream_temperature
        self._pressure = free_stream_pressure
        self._species_mass_fractions = free_stream_mass_fractions
        self._velocity = free_stream_velocity

        PrescribedFluidBoundary.__init__(
            self, boundary_state_func=self.farfield_state
        )

    def farfield_state(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the exterior solution on the boundary."""
        free_stream_mass_fractions = (0.*state_minus.species_mass_fractions
                                      + self._species_mass_fractions)

        free_stream_temperature = 0.*state_minus.temperature + self._temperature
        free_stream_pressure = 0.*state_minus.pressure + self._pressure
        free_stream_velocity = 0.*state_minus.velocity + self._velocity

        free_stream_density = gas_model.eos.get_density(
            pressure=free_stream_pressure, temperature=free_stream_temperature,
            species_mass_fractions=free_stream_mass_fractions)

        free_stream_internal_energy = gas_model.eos.get_internal_energy(
            temperature=free_stream_temperature,
            species_mass_fractions=free_stream_mass_fractions)

        free_stream_total_energy = \
            free_stream_density*(free_stream_internal_energy
                                 + .5*np.dot(free_stream_velocity,
                                             free_stream_velocity))
        free_stream_spec_mass = free_stream_density * free_stream_mass_fractions

        cv_infinity = make_conserved(
            state_minus.dim, mass=free_stream_density,
            energy=free_stream_total_energy,
            momentum=free_stream_density*free_stream_velocity,
            species_mass=free_stream_spec_mass
        )

        return make_fluid_state(cv=cv_infinity, gas_model=gas_model,
                                temperature_seed=free_stream_temperature,
                                smoothness=state_minus.smoothness)

    def temperature_bc(self, state_minus, **kwargs):
        """Return farfield temperature for use in grad(temperature)."""
        return 0*state_minus.temperature + self._temperature


class PressureOutflowBoundary(PrescribedFluidBoundary):
    r"""Outflow boundary treatment with prescribed pressure.

    This class implements an outflow boundary as described by
    [Mengaldo_2014]_.  The boundary condition is implemented as:

    .. math::

        \rho^+ &= \rho^-

        \rho\mathbf{Y}^+ &= \rho\mathbf{Y}^-

        \rho\mathbf{V}^+ &= \rho\mathbf{V}^-

    For an ideal gas at super-sonic flow conditions, i.e. when:

    .. math::

       \rho\mathbf{V} \cdot \hat{\mathbf{n}} \ge c,

    then the pressure is extrapolated from interior points:

    .. math::

        P^+ = P^-

    Otherwise, if the flow is sub-sonic, then the prescribed boundary pressure,
    $P^+$, is used. In both cases, the energy is computed as:

    .. math::

        \rho{E}^+ = \frac{\left(2~P^+ - P^-\right)}{\left(\gamma-1\right)}
        + \frac{1}{2}\rho^+\left(\mathbf{V}^+\cdot\mathbf{V}^+\right).

    For mixtures, the pressure is imposed or extrapolated in a similar fashion
    to the ideal gas case.
    However, the total energy depends on the temperature to account for the
    species enthalpy and variable specific heat at constant volume. For super-sonic
    flows, it is extrapolated from interior points:

    .. math::

       T^+ = T^-

    while for sub-sonic flows, it is evaluated using ideal gas law

    .. math::

        T^+ = \frac{P^+}{R_{mix} \rho^+}

    .. automethod:: __init__
    .. automethod:: outflow_state
    """

    def __init__(self, boundary_pressure=101325):
        """Initialize the boundary condition object."""
        self._pressure = boundary_pressure
        PrescribedFluidBoundary.__init__(
            self, boundary_state_func=self.outflow_state,
            inviscid_flux_func=self.inviscid_boundary_flux,
            viscous_flux_func=self.viscous_boundary_flux,
            boundary_temperature_func=self.temperature_bc,
            boundary_gradient_cv_func=self.grad_cv_bc
        )

    def outflow_state(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the exterior solution on the boundary.

        This is the partially non-reflective boundary state described by
        [Mengaldo_2014]_ eqn. 40 if super-sonic, 41 if sub-sonic.

        For super-sonic outflow, the interior flow properties (minus) are
        extrapolated to the exterior point (plus).
        For sub-sonic outflow, the pressure is imposed on the external point.

        For mixtures, the internal energy is obtained via temperature, which comes
        from ideal gas law with the mixture-weighted gas constant.
        For ideal gas, the internal energy is obtained directly from pressure.
        """
        actx = state_minus.array_context
        nhat = actx.thaw(dcoll.normal(dd_bdry))
        # boundary-normal velocity
        boundary_vel = np.dot(state_minus.velocity, nhat)*nhat
        boundary_speed = actx.np.sqrt(np.dot(boundary_vel, boundary_vel))
        speed_of_sound = state_minus.speed_of_sound
        kinetic_energy = gas_model.eos.kinetic_energy(state_minus.cv)
        gamma = gas_model.eos.gamma(state_minus.cv, state_minus.temperature)

        # evaluate internal energy based on prescribed pressure
        pressure_plus = 2.0*self._pressure - state_minus.pressure
        if state_minus.is_mixture:
            gas_const = gas_model.eos.gas_const(state_minus.cv)
            temp_plus = (
                actx.np.where(actx.np.greater(boundary_speed, speed_of_sound),
                state_minus.temperature,
                pressure_plus/(state_minus.cv.mass*gas_const))
            )

            internal_energy = state_minus.cv.mass*(
                gas_model.eos.get_internal_energy(temp_plus,
                                            state_minus.species_mass_fractions))
        else:
            boundary_pressure = actx.np.where(actx.np.greater(boundary_speed,
                                                              speed_of_sound),
                                              state_minus.pressure, pressure_plus)
            internal_energy = boundary_pressure/(gamma - 1.0)

        total_energy = internal_energy + kinetic_energy
        cv_outflow = make_conserved(dim=state_minus.dim, mass=state_minus.cv.mass,
                                    momentum=state_minus.cv.momentum,
                                    energy=total_energy,
                                    species_mass=state_minus.cv.species_mass)

        return make_fluid_state(cv=cv_outflow, gas_model=gas_model,
                                temperature_seed=state_minus.temperature,
                                smoothness=state_minus.smoothness)

    def outflow_state_for_diffusion(self, dcoll, dd_bdry, gas_model,
                                           state_minus, **kwargs):
        """Return state."""
        actx = state_minus.array_context
        nhat = actx.thaw(dcoll.normal(dd_bdry))

        # boundary-normal velocity
        boundary_vel = np.dot(state_minus.velocity, nhat)*nhat
        boundary_speed = actx.np.sqrt(np.dot(boundary_vel, boundary_vel))
        speed_of_sound = state_minus.speed_of_sound
        kinetic_energy = gas_model.eos.kinetic_energy(state_minus.cv)
        gamma = gas_model.eos.gamma(state_minus.cv, state_minus.temperature)

        # evaluate internal energy based on prescribed pressure
        pressure_plus = self._pressure + 0.0*state_minus.pressure
        if state_minus.is_mixture:
            gas_const = gas_model.eos.gas_const(state_minus.cv)
            temp_plus = (
                actx.np.where(actx.np.greater(boundary_speed, speed_of_sound),
                state_minus.temperature,
                pressure_plus/(state_minus.cv.mass*gas_const))
            )

            internal_energy = state_minus.cv.mass*(
                gas_model.eos.get_internal_energy(
                    temp_plus, state_minus.species_mass_fractions)
            )
        else:
            boundary_pressure = (
                actx.np.where(actx.np.greater(boundary_speed, speed_of_sound),
                              state_minus.pressure, pressure_plus)
            )
            internal_energy = (boundary_pressure / (gamma - 1.0))

        cv_plus = make_conserved(
            state_minus.dim, mass=state_minus.mass_density,
            energy=kinetic_energy + internal_energy,
            momentum=state_minus.momentum_density,
            species_mass=state_minus.species_mass_density
        )
        return make_fluid_state(cv=cv_plus, gas_model=gas_model,
                                temperature_seed=state_minus.temperature)

    def inviscid_boundary_flux(self, dcoll, dd_bdry, gas_model, state_minus,
            numerical_flux_func=inviscid_facial_flux_rusanov, **kwargs):
        """."""
        outflow_state = self.outflow_state(
            dcoll, dd_bdry, gas_model, state_minus)
        state_pair = TracePair(dd_bdry, interior=state_minus, exterior=outflow_state)

        actx = state_minus.array_context
        normal = actx.thaw(dcoll.normal(dd_bdry))
        return numerical_flux_func(state_pair, gas_model, normal)

    def temperature_bc(self, state_minus, **kwargs):
        """Get temperature value used in grad(T)."""
        return state_minus.temperature

    def grad_cv_bc(self, state_minus, grad_cv_minus, normal, **kwargs):
        """Return grad(CV) to be used in the boundary calculation of viscous flux."""
        return grad_cv_minus

    def grad_temperature_bc(self, grad_t_minus, normal, **kwargs):
        """Return grad(temperature) to be used in viscous flux at wall."""
        return grad_t_minus

    def viscous_boundary_flux(self, dcoll, dd_bdry, gas_model, state_minus,
                          grad_cv_minus, grad_t_minus,
                          numerical_flux_func=viscous_facial_flux_central,
                                           **kwargs):
        """Return the boundary flux for the divergence of the viscous flux."""
        from mirgecom.viscous import viscous_flux
        actx = state_minus.array_context
        normal = actx.thaw(dcoll.normal(dd_bdry))

        state_plus = self.outflow_state_for_diffusion(dcoll=dcoll,
            dd_bdry=dd_bdry, gas_model=gas_model, state_minus=state_minus)

        grad_cv_plus = self.grad_cv_bc(state_minus=state_minus,
                                       grad_cv_minus=grad_cv_minus,
                                       normal=normal, **kwargs)
        grad_t_plus = self.grad_temperature_bc(grad_t_minus, normal)

        # Note that [Mengaldo_2014]_ uses F_v(Q_bc, dQ_bc) here and
        # *not* the numerical viscous flux as advised by [Bassi_1997]_.
        f_ext = viscous_flux(state=state_plus, grad_cv=grad_cv_plus,
                             grad_t=grad_t_plus)

        return f_ext@normal


class RiemannInflowBoundary(PrescribedFluidBoundary):
    r"""Inflow boundary treatment.

    This class implements an Riemann invariant for inflow boundary as described by
    [Mengaldo_2014]_.

    .. automethod:: __init__
    .. automethod:: inflow_state
    """

    def __init__(self, free_stream_state_func):
        """Initialize the boundary condition object."""
        self.free_stream_state_func = free_stream_state_func

        PrescribedFluidBoundary.__init__(
            self, boundary_state_func=self.inflow_state
        )

    def inflow_state(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the exterior solution on the boundary.

        This is the partially non-reflective boundary state described by
        [Mengaldo_2014]_ eqn. 40 if super-sonic, 41 if sub-sonic.
        """
        actx = state_minus.array_context
        nhat = actx.thaw(dcoll.normal(dd_bdry))

        ones = 0.0*nhat[0] + 1.0

        free_stream_state = self.free_stream_state_func(
            dcoll, dd_bdry, gas_model, state_minus, **kwargs)

        v_plus = np.dot(free_stream_state.velocity, nhat)
        rho_plus = free_stream_state.mass_density
        c_plus = free_stream_state.speed_of_sound
        gamma_plus = gas_model.eos.gamma(free_stream_state.cv,
                                         free_stream_state.temperature)

        v_minus = np.dot(state_minus.velocity, nhat)
        gamma_minus = gas_model.eos.gamma(state_minus.cv,
                                          temperature=state_minus.temperature)
        c_minus = state_minus.speed_of_sound
        r_minus = v_plus - 2*c_plus/(gamma_plus - 1)*ones

        # eqs. 17 and 19
        r_plus_subsonic = v_minus + 2*c_minus/(gamma_minus - 1)
        r_plus_supersonic = v_plus + 2*c_plus/(gamma_plus - 1)
        r_plus = actx.np.where(actx.np.greater(v_minus, c_minus),
                               r_plus_supersonic, r_plus_subsonic)

        velocity_boundary = (r_minus + r_plus)/2
        velocity_boundary = (
            free_stream_state.velocity + (velocity_boundary - v_plus)*nhat
        )

        c_boundary = (gamma_plus - 1)*(r_plus - r_minus)/4

        # isentropic relations, using minus state (Eq. 23 and 24)
        gamma_boundary = 1.0*gamma_plus
        entropy_boundary = \
            c_plus**2/(gamma_boundary*rho_plus**(gamma_boundary-1))
        rho_boundary = (
            c_boundary**2/(gamma_boundary * entropy_boundary)
        )**(1.0/(gamma_plus-1.0))  # in the reference, Eq. 24 lacks the exponent.
        pressure_boundary = rho_boundary * c_boundary**2 / gamma_boundary

        species_mass_boundary = None
        if free_stream_state.is_mixture:
            energy_boundary = rho_boundary * (
                gas_model.eos.get_internal_energy(
                    temperature=free_stream_state.temperature,
                    species_mass_fractions=free_stream_state.species_mass_fractions)
            ) + 0.5*rho_boundary*np.dot(velocity_boundary, velocity_boundary)

            species_mass_boundary = (
                rho_boundary * free_stream_state.species_mass_fractions
            )
        else:
            energy_boundary = (
                pressure_boundary / (gamma_boundary - 1)
                + 0.5*rho_boundary*np.dot(velocity_boundary, velocity_boundary)
            )

        boundary_cv = make_conserved(dim=state_minus.dim, mass=rho_boundary,
                                     energy=energy_boundary,
                                     momentum=rho_boundary * velocity_boundary,
                                     species_mass=species_mass_boundary)

        return make_fluid_state(cv=boundary_cv, gas_model=gas_model,
                                temperature_seed=state_minus.temperature,
                                smoothness=state_minus.smoothness)


class RiemannOutflowBoundary(PrescribedFluidBoundary):
    r"""Outflow boundary treatment.

    This class implements an Riemann invariant for outflow boundary as described
    by [Mengaldo_2014]_. Note that the "minus" and "plus" are different from the
    reference to the current Mirge-COM definition.

    This boundary condition assume isentropic flow, so the regions where it can
    be applied are not general. Far-field regions are adequate, but not
    viscous-dominated regions of the flow (such as a boundary layer).

    .. automethod:: __init__
    .. automethod:: outflow_state
    """

    def __init__(self, dim, free_stream_state_func):
        """Initialize the boundary condition object."""
        self.free_stream_state_func = free_stream_state_func

        PrescribedFluidBoundary.__init__(
            self, boundary_state_func=self.outflow_state
        )

    def outflow_state(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the exterior solution on the boundary.

        This is the Riemann Invariant Boundary Condition described by
        [Mengaldo_2014]_ in eqs. 8 to 18.
        """
        actx = state_minus.array_context
        nhat = actx.thaw(dcoll.normal(dd_bdry))

        ones = 0.0*nhat[0] + 1.0

        free_stream_state = self.free_stream_state_func(
            dcoll, dd_bdry, gas_model, state_minus, **kwargs)

        v_plus = np.dot(free_stream_state.velocity*ones, nhat)
        c_plus = free_stream_state.speed_of_sound
        gamma_plus = gas_model.eos.gamma(free_stream_state.cv,
                                         free_stream_state.temperature)

        rho_minus = state_minus.mass_density
        v_minus = np.dot(state_minus.velocity, nhat)
        c_minus = state_minus.speed_of_sound
        gamma_minus = gas_model.eos.gamma(
            state_minus.cv, temperature=state_minus.temperature)

        # eqs 17 and 27
        r_plus = v_plus - 2.0*c_plus/(gamma_plus - 1.0)
        r_minus_subsonic = v_minus + 2.0*c_minus/(gamma_minus - 1.0)
        r_minus_supersonic = v_minus - 2.0*c_minus/(gamma_minus - 1.0)
        r_minus = actx.np.where(actx.np.greater(v_minus, c_minus),
                                r_minus_supersonic, r_minus_subsonic)

        velocity_boundary = (r_minus + r_plus)/2.0
        velocity_boundary = (
            state_minus.velocity + (velocity_boundary - v_minus)*nhat
        )
        gamma_boundary = 1.0*gamma_minus

        c_boundary = (gamma_minus - 1.0)*(r_minus - r_plus)/4.0

        # isentropic relations, using minus state (Eq. 24 and 29)
        entropy_boundary = \
            c_minus**2/(gamma_boundary*rho_minus**(gamma_boundary-1.0))
        rho_boundary = (
            c_boundary**2/(gamma_boundary * entropy_boundary)
        )**(1.0/(gamma_minus-1.0))  # in the reference, Eq. 24 lacks the exponent.
        pressure_boundary = rho_boundary*c_boundary**2/gamma_boundary

        species_mass_boundary = None
        if free_stream_state.is_mixture:

            # using gas constant based on state_minus species
            gas_const = gas_model.eos.gas_const(state_minus.cv)
            temperature_boundary = pressure_boundary/(gas_const*rho_boundary)

            energy_boundary = rho_boundary * (
                gas_model.eos.get_internal_energy(
                    temperature_boundary, free_stream_state.species_mass_fractions)
            ) + 0.5*rho_boundary*np.dot(velocity_boundary, velocity_boundary)

            # extrapolate species
            species_mass_boundary = (
                rho_boundary * state_minus.species_mass_fractions
            )
        else:
            energy_boundary = (
                pressure_boundary / (gamma_boundary - 1)
                + 0.5*rho_boundary*np.dot(velocity_boundary, velocity_boundary)
            )

        boundary_cv = make_conserved(dim=state_minus.dim, mass=rho_boundary,
                                     energy=energy_boundary,
                                     momentum=rho_boundary*velocity_boundary,
                                     species_mass=species_mass_boundary)

        return make_fluid_state(cv=boundary_cv, gas_model=gas_model,
                                temperature_seed=state_minus.temperature)


class IsothermalWallBoundary(PrescribedFluidBoundary):
    r"""Isothermal viscous wall boundary.

    This class implements an isothermal no-slip wall consistent with the prescription
    by [Mengaldo_2014]_.

    .. automethod:: __init__
    .. automethod:: state_plus
    .. automethod:: state_bc
    .. automethod:: temperature_plus
    .. automethod:: grad_cv_bc
    """

    def __init__(self, wall_temperature=300):
        """Initialize the boundary condition object."""
        self.inviscid_flux = partial(
            _inviscid_flux_for_prescribed_state_mengaldo,
            state_plus_func=self.state_plus)

        self.viscous_flux = partial(
            _viscous_flux_for_prescribed_state_mengaldo,
            state_bc_func=self.state_bc,
            grad_cv_bc_func=self.grad_cv_bc,
            grad_temperature_bc_func=None)

        PrescribedFluidBoundary.__init__(
            self,
            boundary_state_func=self.state_bc,
            inviscid_flux_func=self.inviscid_flux,
            viscous_flux_func=self.viscous_flux,
            boundary_temperature_func=self.temperature_plus,
            boundary_gradient_cv_func=self.grad_cv_bc)

        self._isothermal = _IsothermalBoundaryComponent(wall_temperature)
        self._no_slip = _NoSlipBoundaryComponent()
        self._impermeable = _ImpermeableBoundaryComponent()

    def state_plus(self, dcoll, dd_bdry, gas_model, state_minus,
                   **kwargs):  # noqa D400
        """
        Return state that cancels interior velocity and has the respective
        internal energy.
        """
        dd_bdry = as_dofdesc(dd_bdry)
        normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))

        mom_plus = self._no_slip.momentum_plus(state_minus.momentum_density, normal)

        # FIXME: Should we be setting the internal energy here?
        cv_plus = make_conserved(
            state_minus.dim,
            mass=state_minus.mass_density,
            energy=state_minus.energy_density,
            momentum=mom_plus,
            species_mass=state_minus.species_mass_density)

        return make_fluid_state(cv=cv_plus, gas_model=gas_model,
                                temperature_seed=state_minus.temperature,
                                smoothness=state_minus.smoothness)

    def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Return state with zero-velocity and the respective internal energy."""
        dd_bdry = as_dofdesc(dd_bdry)
        normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))

        mom_bc = self._no_slip.momentum_bc(state_minus.momentum_density, normal)

        t_bc = self._isothermal.temperature_bc(state_minus)

        internal_energy_bc = gas_model.eos.get_internal_energy(
            temperature=t_bc,
            species_mass_fractions=state_minus.species_mass_fractions)

        # Velocity is pinned to 0 here, no kinetic energy
        total_energy_bc = state_minus.mass_density*internal_energy_bc

        cv_bc = make_conserved(
            state_minus.dim,
            mass=state_minus.mass_density,
            energy=total_energy_bc,
            momentum=mom_bc,
            species_mass=state_minus.species_mass_density)

        return make_fluid_state(cv=cv_bc, gas_model=gas_model,
                                temperature_seed=state_minus.temperature,
                                smoothness=state_minus.smoothness)

    def temperature_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Get temperature value used in grad(T)."""
        return self._isothermal.temperature_plus(state_minus)

    def grad_cv_bc(
            self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus, **kwargs):
        """Return grad(CV) to be used in the boundary calculation of viscous flux."""
        dd_bdry = as_dofdesc(dd_bdry)
        normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))

        grad_species_mass_bc = self._impermeable.grad_species_mass_bc(
            state_minus, grad_cv_minus, normal)

        return make_conserved(
            grad_cv_minus.dim,
            mass=grad_cv_minus.mass,
            energy=grad_cv_minus.energy,
            momentum=grad_cv_minus.momentum,
            species_mass=grad_species_mass_bc)


class AdiabaticNoslipWallBoundary(PrescribedFluidBoundary):
    r"""Adiabatic viscous wall boundary.

    This class implements an adiabatic no-slip wall consistent with the prescription
    by [Mengaldo_2014]_.

    .. automethod:: __init__
    .. automethod:: state_plus
    .. automethod:: state_bc
    .. automethod:: grad_cv_bc
    .. automethod:: grad_temperature_bc
    .. automethod:: grad_av_plus
    """

    def __init__(self):
        """Initialize the boundary condition object."""
        self.inviscid_flux = partial(
            _inviscid_flux_for_prescribed_state_mengaldo,
            state_plus_func=self.state_plus)

        self.viscous_flux = partial(
            _viscous_flux_for_prescribed_state_mengaldo,
            state_bc_func=self.state_bc,
            grad_cv_bc_func=self.grad_cv_bc,
            grad_temperature_bc_func=self.grad_temperature_bc)

        PrescribedFluidBoundary.__init__(
            self,
            boundary_state_func=self.state_bc,
            inviscid_flux_func=self.inviscid_flux,
            viscous_flux_func=self.viscous_flux,
            boundary_gradient_cv_func=self.grad_cv_bc,
            boundary_gradient_temperature_func=self.grad_temperature_bc)

        self._adiabatic = _AdiabaticBoundaryComponent()
        self._no_slip = _NoSlipBoundaryComponent()
        self._impermeable = _ImpermeableBoundaryComponent()

    def state_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Return state that cancels interior velocity."""
        dd_bdry = as_dofdesc(dd_bdry)
        normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))

        mom_plus = self._no_slip.momentum_plus(state_minus.momentum_density, normal)

        # Energy is the same, don't need to compute
        cv_plus = make_conserved(
            state_minus.dim,
            mass=state_minus.mass_density,
            energy=state_minus.energy_density,
            momentum=mom_plus,
            species_mass=state_minus.species_mass_density)

        return make_fluid_state(cv=cv_plus, gas_model=gas_model,
                                temperature_seed=state_minus.temperature,
                                smoothness=state_minus.smoothness)

    def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Return state with zero-velocity."""
        dd_bdry = as_dofdesc(dd_bdry)
        normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))

        mom_bc = self._no_slip.momentum_bc(state_minus.momentum_density, normal)

        # FIXME: Should we modify kinetic energy here? If not, add a comment
        # explaining why
        cv_bc = make_conserved(
            state_minus.dim,
            mass=state_minus.mass_density,
            energy=state_minus.energy_density,
            momentum=mom_bc,
            species_mass=state_minus.species_mass_density)

        return make_fluid_state(cv=cv_bc, gas_model=gas_model,
                                temperature_seed=state_minus.temperature,
                                smoothness=state_minus.smoothness)

    def grad_cv_bc(
            self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus, **kwargs):
        """Return grad(CV) to be used in the boundary calculation of viscous flux."""
        dd_bdry = as_dofdesc(dd_bdry)
        normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))

        grad_species_mass_bc = self._impermeable.grad_species_mass_bc(
            state_minus, grad_cv_minus, normal)

        # Note we don't need to tweak grad(rhoE) here as it is unused.
        # Tweaks to grad(rhoV) (i.e. grad(V)) are ineffective as we have V=0 at
        # the wall
        return make_conserved(
            grad_cv_minus.dim,
            mass=grad_cv_minus.mass,
            energy=grad_cv_minus.energy,
            momentum=grad_cv_minus.momentum,
            species_mass=grad_species_mass_bc)

    def grad_temperature_bc(
            self, dcoll, dd_bdry, gas_model, grad_t_minus, **kwargs):
        """Return grad(temperature) to be used in viscous flux at wall."""
        dd_bdry = as_dofdesc(dd_bdry)
        normal = grad_t_minus[0].array_context.thaw(dcoll.normal(dd_bdry))

        return self._adiabatic.grad_temperature_bc(grad_t_minus, normal)

    # FIXME: Remove this?
    def grad_av_plus(self, grad_av_minus, **kwargs):
        """Get the exterior solution on the boundary for artificial viscosity."""
        return -grad_av_minus


class LinearizedOutflowBoundary(PrescribedFluidBoundary):
    r"""Characteristics outflow BCs for linearized Euler equations.

    Implement non-reflecting outflow based on characteristic variables for
    the Euler equations assuming small perturbations based on [Giles_1988]_.
    The equations assume an uniform, steady flow and linerize the Euler eqs.
    in this reference state, yielding a linear equation in the form

    .. math::
        \frac{\partial U}{\partial t} + A \frac{\partial U}{\partial x} +
        B \frac{\partial U}{\partial y} = 0

    where where U is the vector of perturbation (primitive) variables and
    the coefficient matrices A and B are constant matrices based on the
    uniform, steady variables.

    Using the linear hyperbolic system theory, this equation can be further
    simplified by ignoring the y-axis terms (tangent) such that wave propagation
    occurs only along the x-axis direction (normal). Then, the eigendecomposition
    results in a orthogonal system where the wave have characteristic directions
    of propagations and enable the creation of non-reflecting outflow boundaries.

    This can also be applied for Navier-Stokes equations in regions where
    viscous effects are not dominant, such as the far-field.
    """

    def __init__(self, free_stream_state=None,
                 free_stream_density=None,
                 free_stream_velocity=None,
                 free_stream_pressure=None,
                 free_stream_species_mass_fractions=None):
        """Initialize the boundary condition object."""
        if free_stream_state is None:
            self._ref_mass = free_stream_density
            self._ref_velocity = free_stream_velocity
            self._ref_pressure = free_stream_pressure
            self._spec_mass_fracs = free_stream_species_mass_fractions
        else:
            self._ref_mass = free_stream_state.cv.mass
            self._ref_velocity = free_stream_state.velocity
            self._ref_pressure = free_stream_state.pressure
            self._spec_mass_fracs = free_stream_state.cv.species_mass_fractions

        PrescribedFluidBoundary.__init__(
            self, boundary_state_func=self.outflow_state
        )

    def outflow_state(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Non-reflecting outflow."""
        actx = state_minus.array_context
        nhat = actx.thaw(dcoll.normal(dd_bdry))

        rtilde = state_minus.cv.mass - self._ref_mass
        utilde = state_minus.velocity[0] - self._ref_velocity[0]
        vtilde = state_minus.velocity[1] - self._ref_velocity[1]
        ptilde = state_minus.dv.pressure - self._ref_pressure

        un_tilde = +utilde*nhat[0] + vtilde*nhat[1]
        ut_tilde = -utilde*nhat[1] + vtilde*nhat[0]

        a = state_minus.speed_of_sound

        c1 = -rtilde*a**2 + ptilde
        c2 = self._ref_mass*a*ut_tilde
        c3 = self._ref_mass*a*un_tilde + ptilde
        c4 = 0.0  # zero-out the last characteristic variable
        r_tilde_bnd = 1.0/(a**2)*(-c1 + 0.5*c3 + 0.5*c4)
        un_tilde_bnd = 1.0/(self._ref_mass*a)*(0.5*c3 - 0.5*c4)
        ut_tilde_bnd = 1.0/(self._ref_mass*a)*c2
        p_tilde_bnd = 0.5*c3 + 0.5*c4

        mass = r_tilde_bnd + self._ref_mass
        u_x = self._ref_velocity[0] + (nhat[0]*un_tilde_bnd - nhat[1]*ut_tilde_bnd)
        u_y = self._ref_velocity[1] + (nhat[1]*un_tilde_bnd + nhat[0]*ut_tilde_bnd)
        pressure = p_tilde_bnd + self._ref_pressure

        kin_energy = 0.5*mass*(u_x**2 + u_y**2)
        if state_minus.is_mixture:
            gas_const = gas_model.eos.gas_const(state_minus.cv)
            temperature = self._ref_pressure/(self._ref_mass*gas_const)
            int_energy = mass*gas_model.eos.get_internal_energy(
                temperature, self._spec_mass_fracs)
        else:
            int_energy = pressure/(gas_model.eos.gamma() - 1.0)

        boundary_cv = (
            make_conserved(dim=state_minus.dim, mass=mass,
                           energy=kin_energy + int_energy,
                           momentum=make_obj_array([u_x*mass, u_y*mass]),
                           species_mass=state_minus.cv.species_mass)
        )

        return make_fluid_state(cv=boundary_cv, gas_model=gas_model,
                                temperature_seed=state_minus.temperature)
