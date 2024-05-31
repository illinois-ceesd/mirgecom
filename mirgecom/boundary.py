""":mod:`mirgecom.boundary` provides methods and constructs for boundary treatments.

Boundary Treatment Interfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: FluidBoundary

Boundary Conditions Base Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: PrescribedFluidBoundary
.. autoclass:: MengaldoBoundaryCondition

Boundary Conditions
^^^^^^^^^^^^^^^^^^^

.. autoclass:: DummyBoundary
.. autoclass:: FarfieldBoundary
.. autoclass:: PressureOutflowBoundary
.. autoclass:: RiemannInflowBoundary
.. autoclass:: RiemannOutflowBoundary
.. autoclass:: IsothermalSlipWallBoundary
.. autoclass:: IsothermalWallBoundary
.. autoclass:: AdiabaticSlipBoundary
.. autoclass:: AdiabaticNoslipWallBoundary
.. autoclass:: LinearizedOutflowBoundary
.. autoclass:: LinearizedInflowBoundary
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

from abc import ABCMeta, abstractmethod
import numpy as np
import grudge.op as op
from arraycontext import outer, get_container_context_recursively
from meshmode.discretization.connection import FACE_RESTR_ALL
from grudge.dof_desc import as_dofdesc
from grudge.trace_pair import TracePair
from pytools.obj_array import make_obj_array
from mirgecom.fluid import (
    ConservedVars,
    make_conserved
)
from mirgecom.gas_model import (
    make_fluid_state, replace_fluid_state,
    get_cv_from_state_container
)
from mirgecom.utils import project_from_base
from mirgecom.viscous import viscous_facial_flux_central, viscous_flux
from mirgecom.inviscid import inviscid_facial_flux_rusanov, inviscid_flux


def _ldg_bnd_flux_for_grad(internal_quantity, external_quantity):
    # Default for prescribed boundary; sends (+) bnd value for gradient flux
    return external_quantity


def _get_normal_axes(seed_vector):
    actx = get_container_context_recursively(seed_vector)
    vec_dim, = seed_vector.shape

    vec_mag = actx.np.sqrt(np.dot(seed_vector, seed_vector))
    seed_vector = seed_vector / vec_mag

    if vec_dim == 1:
        return seed_vector,  # pylint: disable=trailing-comma-tuple

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


class _SlipBoundaryComponent:
    """Helper class for slip boundaries, consistent with [Mengaldo_2014]_."""

    def momentum_plus(self, mom_minus, normal):
        return mom_minus - 2.0*np.dot(mom_minus, normal)*normal

    def momentum_bc(self, mom_minus, normal):
        # set the normal momentum to 0
        return mom_minus - np.dot(mom_minus, normal)*normal

    def grad_velocity_bc(
            self, state_minus, state_bc, grad_cv_minus, normal):
        from mirgecom.fluid import velocity_gradient

        cv_minus = get_cv_from_state_container(state_minus)

        grad_v_minus = velocity_gradient(cv_minus, grad_cv_minus)

        # rotate the velocity gradient tensor into the normal direction
        rotation_matrix = _get_rotation_matrix(normal)
        grad_v_normal = rotation_matrix@grad_v_minus@rotation_matrix.T

        # set the normal component of the tangential velocity to 0
        for i in range(state_minus.dim-1):
            grad_v_normal[i+1][0] = 0.*grad_v_normal[i+1][0]

        # get the gradient on the boundary in the global coordiate space
        return rotation_matrix.T@grad_v_normal@rotation_matrix


class _NoSlipBoundaryComponent:
    """Helper class for no-slip boundaries, consistent with [Mengaldo_2014]_."""

    def momentum_plus(self, mom_minus, **kwargs):
        return -mom_minus

    def momentum_bc(self, mom_minus, **kwargs):
        return 0.*mom_minus


class _AdiabaticBoundaryComponent:
    """Helper class for adiabatic boundaries, consistent with [Mengaldo_2014]_."""

    def grad_temperature_bc(self, grad_t_minus, normal):
        return grad_t_minus - np.dot(grad_t_minus, normal)*normal


class _ImpermeableBoundaryComponent:
    """Helper class for impermeable boundaries, consistent with [Mengaldo_2014]_."""

    # Routine seems fine for flamelets, where just like before but nspecies=1
    # For flamelet mixtures, Z is the one transported scalar
    def grad_species_mass_bc(self, state_minus, grad_cv_minus, normal):
        # This will return nspecies=1 for flamelet
        nspecies = len(state_minus.species_mass_density)
        # This will return [grad(Z)] for flamelet [grad(Y_alpha)] for mixture
        cv_minus = get_cv_from_state_container(state_minus)
        grad_species_mass_bc = 1.*grad_cv_minus.species_mass
        if nspecies > 0:
            from mirgecom.fluid import species_mass_fraction_gradient
            grad_y_minus = species_mass_fraction_gradient(cv_minus, grad_cv_minus)
            grad_y_bc = grad_y_minus - np.outer(grad_y_minus@normal, normal)
            grad_species_mass_bc = 0.*grad_y_bc

            for i in range(nspecies):
                grad_species_mass_bc[i] = \
                    (cv_minus.mass*grad_y_bc[i]
                     + cv_minus.species_mass_fractions[i]*grad_cv_minus.mass)

        return grad_species_mass_bc


# Bare minimum interface to work in CNS Operator
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


class MengaldoBoundaryCondition(FluidBoundary):
    r"""Abstract interface to Megaldo fluid boundary treatment.

    Mengaldo boundary conditions are those described by [Mengaldo_2014]_, and
    with slight mods for flow boundaries from [Poinsot_1992]_ where noted.

    Base class implementations
    --------------------------
    .. automethod:: inviscid_divergence_flux
    .. automethod:: viscous_divergence_flux
    .. automethod:: cv_gradient_flux
    .. automethod:: temperature_gradient_flux

    Abstract Mengaldo interface
    ---------------------------
    .. automethod:: state_bc
    .. automethod:: grad_cv_bc
    .. automethod:: temperature_bc
    .. automethod:: grad_temperature_bc
    .. automethod:: state_plus
    """

    @abstractmethod
    def state_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the boundary state to be used for inviscid fluxes.

        This routine returns a boundary state that is designed to
        be used in an approximate Riemann solver, like HLL, or HLLC.

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

        Returns
        -------
        :class:`mirgecom.gas_model.FluidState`
        """

    @abstractmethod
    def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the boundary condition on the fluid state.

        This routine returns the exact value of the boundary condition
        of the fluid state. These are the values we want to enforce
        at the boundary. It is used in the calculation of the gradient
        of the conserved quantities, and in the calculation of the
        viscous fluxes.

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

        Returns
        -------
        :class:`mirgecom.gas_model.FluidState`
        """

    @abstractmethod
    def grad_cv_bc(self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
                   normal, **kwargs):
        """Get the boundary condition on the fluid state.

        This routine returns the exact value of the boundary condition
        of the fluid state. These are the values we want to enforce
        at the boundary. It is used in the calculation of the gradient
        of the conserved quantities, and in the calculation of the
        viscous fluxes.

        Parameters
        ----------
        state_minus: :class:`~mirgecom.gas_model.FluidState`

            Fluid state object with the conserved state, and dependent
            quantities for the (-) side of the boundary specified by
            *dd_bdry*.

        grad_cv_minus: :class:`~mirgecom.fluid.ConservedVars`

            ConservedVars object with the gradient of the fluid
            conserved variables on the (-) side of the boundary.

        normal: numpy.ndarray
            Unit normal vector to the boundary

        Returns
        -------
        :class:`mirgecom.gas_model.FluidState`
        """

    @abstractmethod
    def grad_temperature_bc(self, dcoll, dd_bdry, grad_t_minus, normal, **kwargs):
        r"""Get the boundary condition on the temperature gradient.

        This routine returns the boundary condition on the gradient of the
        temperature, $(\nabla{T})_\text{bc}$.  This value is used in the
        calculation of the heat flux.

        Parameters
        ----------
        dcoll: :class:`~grudge.discretization.DiscretizationCollection`

            A discretization collection encapsulating the DG elements

        dd_bdry:

            Boundary DOF descriptor (or object convertible to one) indicating which
            domain boundary to process

        grad_t_minus: numpy.ndarray
            Gradient of the temperature on the (-) side of the boundary.

        normal: numpy.ndarray
            Unit normal vector to the boundary

        Returns
        -------
        :class:`mirgecom.fluid.ConservedVars`
        """

    @abstractmethod
    def temperature_bc(self, dcoll, dd_bdry, state_minus, **kwargs):
        r"""Get boundary contition on the temperature.

        This routine returns the temperature boundary condition, $T_\text{bc}$.
        This value is used in the calcuation of the temperature gradient,
        $\nabla{T}$.

        Parameters
        ----------
        dcoll: :class:`~grudge.discretization.DiscretizationCollection`

            A discretization collection encapsulating the DG elements

        dd_bdry:

            Boundary DOF descriptor (or object convertible to one) indicating which
            domain boundary to process

        state_minus: :class:`~mirgecom.gas_model.FluidState`

            Fluid state object with the conserved state, and dependent
            quantities for the (-) side of the boundary.

        Returns
        -------
        :class:`meshmode.dof_array.DOFArray`
        """

    def inviscid_divergence_flux(self, dcoll, dd_bdry, gas_model, state_minus,
                                 numerical_flux_func=inviscid_facial_flux_rusanov,
                                 **kwargs):
        """Get the inviscid boundary flux for the divergence operator.

        This routine returns the facial flux used in the divergence
        of the inviscid fluid transport flux. Mengaldo BCs use the
        approximate Riemann solver specified by the *numerical_flux_func*
        to calculate the flux.  The boundary implementation must provide
        the :meth:`state_plus` to set the exterior state used in the
        Riemann solver.

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
        dd_bdry = as_dofdesc(dd_bdry)
        state_plus = self.state_plus(
            dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
            state_minus=state_minus, **kwargs)
        boundary_state_pair = TracePair(dd=dd_bdry,
                                        interior=state_minus,
                                        exterior=state_plus)
        normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))
        return numerical_flux_func(boundary_state_pair, gas_model, normal)

    def viscous_divergence_flux(self, dcoll, dd_bdry, gas_model, state_minus,
                                grad_cv_minus, grad_t_minus,
                                numerical_flux_func=viscous_facial_flux_central,
                                **kwargs):
        r"""Get the viscous boundary flux for the divergence operator.

        This routine returns the facial flux used in the divergence
        of the viscous fluid transport flux, ($f_v$). The Mengaldo boundary
        treatment sends back the face-normal component of the physical
        viscous flux calculated with the boundary conditions:

        .. math::
            f_v = F_v\left(\text{CV}_\text{bc}, (\nabla{\text{CV}})_\text{bc},
            (\nabla{T})_\text{bc}\right) \cdot \hat{n}

        where $F_v(.,.,.)$ is the viscous flux function and it is called with
        the boundary conditions of $\text{CV}$, $\nabla\text{CV}$, and
        temperature gradient.

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

            Unused!

        Returns
        -------
        :class:`mirgecom.fluid.ConservedVars`
        """
        dd_bdry = as_dofdesc(dd_bdry)

        actx = state_minus.array_context
        normal = actx.thaw(dcoll.normal(dd_bdry))

        state_bc = self.state_bc(dcoll=dcoll, dd_bdry=dd_bdry,
                                 gas_model=gas_model,
                                 state_minus=state_minus, **kwargs)

        grad_cv_bc = self.grad_cv_bc(dcoll, dd_bdry, gas_model,
                                     state_minus=state_minus,
                                     grad_cv_minus=grad_cv_minus,
                                     normal=normal, **kwargs)

        grad_t_bc = self.grad_temperature_bc(dcoll, dd_bdry,
                                             grad_t_minus=grad_t_minus,
                                             normal=normal, **kwargs)

        # Note that [Mengaldo_2014]_ uses F_v(Q_bc, dQ_bc) here and
        # *not* the numerical viscous flux as advised by [Bassi_1997]_.
        f_ext = viscous_flux(state=state_bc, grad_cv=grad_cv_bc,
                             grad_t=grad_t_bc)
        return f_ext@normal

    def cv_gradient_flux(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        r"""Get the boundary flux for the gradient of the fluid conserved variables.

        This routine returns the facial flux used by the gradient operator to
        compute the gradient of the fluid solution on a domain boundary. The
        Mengaldo boundary treatment sends back $\text{CV}_bc~\mathbf{\hat{n}}$.

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
        # Mengaldo Eqn (50)+
        state_bc = self.state_bc(dcoll, dd_bdry, gas_model, state_minus, **kwargs)
        actx = state_minus.array_context
        nhat = actx.thaw(dcoll.normal(dd_bdry))
        cv_bc = get_cv_from_state_container(state_bc)
        return outer(cv_bc, nhat)

    def temperature_gradient_flux(self, dcoll, dd_bdry, gas_model, state_minus,
                                  **kwargs):
        r"""Get the boundary flux for the gradient of the fluid temperature.

        This method returns the boundary flux to be used by the gradient
        operator when computing the gradient of the fluid temperature at a
        domain boundary. The Mengaldo boundary treatment sends back
        $T_bc~\mathbf{\hat{n}}$.

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
        # Mengaldo Eqn (50)+
        temperature_bc = self.temperature_bc(dcoll, dd_bdry, state_minus, **kwargs)
        actx = state_minus.array_context
        nhat = actx.thaw(dcoll.normal(dd_bdry))

        return outer(temperature_bc, nhat)


class PrescribedFluidBoundary(FluidBoundary):
    r"""Abstract interface to a prescribed fluid boundary treatment.

    .. automethod:: __init__
    .. automethod:: inviscid_divergence_flux
    .. automethod:: viscous_divergence_flux
    .. automethod:: cv_gradient_flux
    .. automethod:: temperature_gradient_flux
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
            # Default sends BC value for gradient flux
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
    # boundary functions and helpers.

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

        cv_minus = get_cv_from_state_container(state_minus)
        cv_plus = get_cv_from_state_container(boundary_state)

        actx = state_minus.array_context
        nhat = actx.thaw(dcoll.normal(dd_bdry))

        return outer(self._grad_num_flux_func(cv_minus, cv_plus), nhat)

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

        return outer(self._grad_num_flux_func(bnd_tpair.int, bnd_tpair.ext), nhat)

    # Returns the flux to be used by the divergence operator when computing the
    # divergence of inviscid fluid transport flux using the boundary's
    # prescribed CV(+).
    def _inviscid_flux_for_prescribed_state(
            self, dcoll, dd_bdry, gas_model, state_minus,
            numerical_flux_func=inviscid_facial_flux_rusanov, **kwargs):
        # Use a prescribed boundary state and the numerical flux function
        dd_bdry = as_dofdesc(dd_bdry)
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

        grad_cv_pair = TracePair(
            dd_bdry,
            interior=grad_cv_minus,
            exterior=self._bnd_grad_cv_func(
                dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
                state_minus=state_minus, grad_cv_minus=grad_cv_minus,
                grad_t_minus=grad_t_minus))

        grad_t_pair = TracePair(
            dd_bdry,
            interior=grad_t_minus,
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


class DummyBoundary(FluidBoundary):
    """Boundary type that assigns boundary-adjacent solution to the boundary.

    .. automethod:: inviscid_divergence_flux
    .. automethod:: viscous_divergence_flux
    .. automethod:: cv_gradient_flux
    .. automethod:: temperature_gradient_flux
    """

    def __init__(self):
        from warnings import warn
        warn("Using dummy boundary: copies interior solution.", stacklevel=2)

    def inviscid_divergence_flux(self, dcoll, dd_bdry, gas_model, state_minus,
                                 numerical_flux_func=inviscid_facial_flux_rusanov,
                                 **kwargs):
        """Get the inviscid boundary flux for the divergence operator."""
        normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))
        return inviscid_flux(state_minus)@normal

    def cv_gradient_flux(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the cv flux for *dd_bdry* for use in the gradient operator."""
        normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))
        cv_minus = get_cv_from_state_container(state_minus)
        return outer(cv_minus, normal)

    def temperature_gradient_flux(self, dcoll, dd_bdry, gas_model, state_minus,
                                  **kwargs):
        """Get the T flux for *dd_bdry* for use in the gradient operator."""
        normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))
        return state_minus.temperature*normal

    def viscous_divergence_flux(self, dcoll, dd_bdry, gas_model, state_minus,
                                grad_cv_minus, grad_t_minus,
                                numerical_flux_func=viscous_facial_flux_central,
                                **kwargs):
        """Get the viscous flux for *dd_bdry* for use in the divergence operator."""
        normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))
        return viscous_flux(state_minus, grad_cv_minus, grad_t_minus)@normal


class AdiabaticSlipBoundary(MengaldoBoundaryCondition):
    r"""Boundary condition implementing inviscid slip boundary.

    This class implements an adiabatic slip wall consistent with the prescription
    by [Mengaldo_2014]_.

    .. automethod:: __init__
    .. automethod:: state_plus
    .. automethod:: state_bc
    .. automethod:: temperature_bc
    .. automethod:: grad_cv_bc
    .. automethod:: grad_temperature_bc
    """

    def __init__(self):
        """Initialize the BC object."""
        self._slip = _SlipBoundaryComponent()
        self._impermeable = _ImpermeableBoundaryComponent()
        self._adiabatic = _AdiabaticBoundaryComponent()

    def state_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Return state with reflected normal-component velocity."""
        actx = state_minus.array_context

        # Grab a unit normal to the boundary
        nhat = actx.thaw(dcoll.normal(dd_bdry))

        # set the normal momentum to 0
        cv_minus = get_cv_from_state_container(state_minus)
        mom_plus = self._slip.momentum_plus(cv_minus.momentum, nhat)
        return replace_fluid_state(state_minus, gas_model, momentum=mom_plus)

    def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Return state with zero normal-component velocity."""
        actx = state_minus.array_context

        # Grab a unit normal to the boundary
        nhat = actx.thaw(dcoll.normal(dd_bdry))

        # set the normal momentum to 0
        cv_minus = get_cv_from_state_container(state_minus)
        mom_bc = self._slip.momentum_bc(cv_minus.momentum, nhat)

        energy_bc = (
            gas_model.eos.internal_energy(cv_minus)
            + 0.5*np.dot(mom_bc, mom_bc)/cv_minus.mass)

        return replace_fluid_state(
            state_minus, gas_model,
            energy=energy_bc,
            momentum=mom_bc)

    def temperature_bc(self, dcoll, dd_bdry, state_minus, **kwargs):
        """Return temperature for use in grad(temperature)."""
        return state_minus.temperature

    def grad_temperature_bc(self, dcoll, dd_bdry, grad_t_minus, normal, **kwargs):
        """Compute temperature gradient on the plus state.

        Impose the opposite normal component to enforce zero energy flux
        from conduction.
        """
        return self._adiabatic.grad_temperature_bc(grad_t_minus, normal)

    def grad_cv_bc(self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
                   normal, **kwargs):
        """
        Return external grad(CV) used in the boundary calculation of viscous flux.

        Specify the velocity gradients on the external state to ensure zero
        energy and momentum flux due to shear stresses.

        Gradients of species mass fractions are set to zero in the normal direction
        to ensure zero flux of species across the boundary.
        """
        dd_bdry = as_dofdesc(dd_bdry)
        normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))
        state_bc = self.state_bc(
            dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
            state_minus=state_minus, **kwargs)

        grad_v_bc = self._slip.grad_velocity_bc(
            state_minus, state_bc, grad_cv_minus, normal)
        cv_bc = get_cv_from_state_container(state_bc)
        grad_mom_bc = (
            cv_bc.mass * grad_v_bc
            + np.outer(cv_bc.velocity, grad_cv_minus.mass))

        grad_species_mass_bc = self._impermeable.grad_species_mass_bc(
            state_minus, grad_cv_minus, normal)

        return grad_cv_minus.replace(
            momentum=grad_mom_bc,
            species_mass=grad_species_mass_bc)


class FarfieldBoundary(MengaldoBoundaryCondition):
    r"""Farfield boundary treatment.

    This class implements a farfield boundary as described by
    [Mengaldo_2014]_ eqn. 30 and eqn. 42. The boundary condition is implemented
    as:

    .. math::
        q^{+} = q_\infty

    and the gradients

    .. math::
        \nabla q_{bc} = \nabla q^{-}

    .. automethod:: __init__
    .. automethod:: state_plus
    .. automethod:: state_bc
    .. automethod:: temperature_bc
    .. automethod:: grad_cv_bc
    .. automethod:: grad_temperature_bc
    """

    # This is OK, user needs to be aware that for flamelet, Z is specified
    # in the farfield mixture, *not* Y.  *NOT OK* for NN flamelet, where
    # we need |grad(Z)| to compute temperature - so will be called with CV-only.
    def __init__(self, free_stream_pressure, free_stream_velocity,
                 free_stream_temperature, free_stream_mass_fractions=None):
        """Initialize the boundary condition object."""
        self._temperature = free_stream_temperature
        self._pressure = free_stream_pressure
        self._species_mass_fractions = free_stream_mass_fractions
        self._velocity = free_stream_velocity

    def state_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the exterior solution on the boundary."""
        cv_minus = get_cv_from_state_container(state_minus)
        free_stream_mass_fractions = (0.*cv_minus.species_mass_fractions
                                      + self._species_mass_fractions)

        free_stream_temperature = 0.*state_minus.temperature + self._temperature
        free_stream_pressure = 0.*state_minus.pressure + self._pressure
        free_stream_velocity = 0.*state_minus.velocity + self._velocity

        # For flamelet, EOS already expects to get Z from this function call
        # and *not* species_mass_fractions.
        y_or_z = free_stream_mass_fractions
        free_stream_density = gas_model.eos.get_density(
            pressure=free_stream_pressure, temperature=free_stream_temperature,
            species_mass_fractions=y_or_z)

        free_stream_internal_energy = gas_model.eos.get_internal_energy(
            temperature=free_stream_temperature,
            species_mass_fractions=y_or_z)

        free_stream_total_energy = \
            free_stream_density*(free_stream_internal_energy
                                 + .5*np.dot(free_stream_velocity,
                                             free_stream_velocity))
        # This call puts the correct free_stream_density back into the right
        # shape for transported scalars (regardless of mixture, or flamelet)
        free_stream_spec_mass = free_stream_density * y_or_z

        cv_infinity = make_conserved(
            state_minus.dim, mass=free_stream_density,
            energy=free_stream_total_energy,
            momentum=free_stream_density*free_stream_velocity,
            species_mass=free_stream_spec_mass)

        return make_fluid_state(cv=cv_infinity, gas_model=gas_model,
                                temperature_seed=free_stream_temperature,
                                smoothness_mu=state_minus.smoothness_mu,
                                smoothness_kappa=state_minus.smoothness_kappa,
                                smoothness_beta=state_minus.smoothness_beta)

    def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Return BC fluid state."""
        return self.state_plus(dcoll, dd_bdry, gas_model, state_minus, **kwargs)

    def temperature_bc(self, dcoll, dd_bdry, state_minus, **kwargs):
        """Return farfield temperature for use in grad(temperature)."""
        actx = state_minus.array_context
        return actx.np.zeros_like(state_minus.temperature) + self._temperature

    def grad_cv_bc(self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
                   normal, **kwargs):
        """Return grad(CV) to be used in the boundary calculation of viscous flux."""
        return grad_cv_minus

    def grad_temperature_bc(self, dcoll, dd_bdry, grad_t_minus, normal, **kwargs):
        """Return grad(temperature) to be used in viscous flux at wall."""
        return grad_t_minus


class PressureOutflowBoundary(MengaldoBoundaryCondition):
    r"""Outflow boundary treatment with prescribed pressure.

    This class implements an outflow boundary as described by
    [Mengaldo_2014]_. The boundary condition is implemented as:

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
    .. automethod:: state_plus
    .. automethod:: state_bc
    .. automethod:: temperature_bc
    .. automethod:: grad_cv_bc
    .. automethod:: grad_temperature_bc
    """

    def __init__(self, boundary_pressure=101325):
        """Initialize the boundary condition object."""
        self._pressure = boundary_pressure

    def state_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
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
            # This form should work for flamelet, too
            y_or_z = state_minus.cv.species_mass_fractions
            gas_const = gas_model.eos.gas_const(species_mass_fractions=y_or_z)
            temp_plus = (
                actx.np.where(actx.np.greater(boundary_speed, speed_of_sound),
                              state_minus.temperature,
                              pressure_plus/(state_minus.cv.mass*gas_const)))

            internal_energy = state_minus.cv.mass*(
                gas_model.eos.get_internal_energy(temp_plus, y_or_z))
        else:
            boundary_pressure = (
                actx.np.where(actx.np.greater(boundary_speed, speed_of_sound),
                              state_minus.pressure, pressure_plus))
            internal_energy = boundary_pressure/(gamma - 1.0)

        total_energy = internal_energy + kinetic_energy

        cv_outflow = make_conserved(dim=state_minus.dim, mass=state_minus.cv.mass,
                                    momentum=state_minus.cv.momentum,
                                    energy=total_energy,
                                    species_mass=state_minus.cv.species_mass)

        return make_fluid_state(cv=cv_outflow, gas_model=gas_model,
                                temperature_seed=state_minus.temperature,
                                smoothness_mu=state_minus.smoothness_mu,
                                smoothness_kappa=state_minus.smoothness_kappa,
                                smoothness_beta=state_minus.smoothness_beta)

    def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
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
        pressure_plus = self._pressure + actx.np.zeros_like(state_minus.pressure)
        if state_minus.is_mixture:
            y_or_z = state_minus.cv.species_mass_fractions
            gas_const = gas_model.eos.gas_const(
                species_mass_fractions=y_or_z)
            temp_plus = (
                actx.np.where(actx.np.greater(boundary_speed, speed_of_sound),
                              state_minus.temperature,
                              pressure_plus/(state_minus.cv.mass*gas_const)))

            internal_energy = state_minus.cv.mass*(
                gas_model.eos.get_internal_energy(
                    temp_plus, species_mass_fractions=y_or_z))
        else:
            boundary_pressure = (
                actx.np.where(actx.np.greater(boundary_speed, speed_of_sound),
                              state_minus.pressure, pressure_plus))
            internal_energy = boundary_pressure / (gamma - 1.0)

        cv_plus = make_conserved(
            state_minus.dim, mass=state_minus.mass_density,
            energy=kinetic_energy + internal_energy,
            momentum=state_minus.momentum_density,
            species_mass=state_minus.species_mass_density)

        return make_fluid_state(cv=cv_plus, gas_model=gas_model,
                                temperature_seed=state_minus.temperature,
                                smoothness_mu=state_minus.smoothness_mu,
                                smoothness_kappa=state_minus.smoothness_kappa,
                                smoothness_beta=state_minus.smoothness_beta)

    def temperature_bc(self, dcoll, dd_bdry, state_minus, **kwargs):
        """Get temperature value used in grad(T)."""
        return state_minus.temperature

    def grad_cv_bc(self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
                   normal, **kwargs):
        """Return grad(CV) to be used in the boundary calculation of viscous flux."""
        return grad_cv_minus

    def grad_temperature_bc(self, dcoll, dd_bdry, grad_t_minus, normal, **kwargs):
        """Return grad(temperature) to be used in viscous flux at wall."""
        return grad_t_minus


class RiemannInflowBoundary(MengaldoBoundaryCondition):
    r"""Inflow boundary treatment.

    This class implements an Riemann invariant inflow boundary condition
    as described by [Mengaldo_2014]_.

    .. automethod:: __init__
    .. automethod:: state_plus
    .. automethod:: state_bc
    .. automethod:: temperature_bc
    .. automethod:: grad_cv_bc
    .. automethod:: grad_temperature_bc
    """

    def __init__(self, cv, temperature):
        """Initialize the boundary condition object."""
        self._cv_plus = cv
        self._t_plus = temperature

    def state_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the exterior solution on the boundary.

        This is the partially non-reflective boundary state described by
        [Mengaldo_2014]_ eqn. 40 if super-sonic, 41 if sub-sonic.
        """
        actx = state_minus.array_context
        nhat = actx.thaw(dcoll.normal(dd_bdry))

        v_plus = np.dot(self._cv_plus.velocity, nhat)
        rho_plus = self._cv_plus.mass
        c_plus = gas_model.eos.sound_speed(self._cv_plus, self._t_plus)
        gamma_plus = gas_model.eos.gamma(self._cv_plus, self._t_plus)

        v_minus = np.dot(state_minus.velocity, nhat)
        gamma_minus = gas_model.eos.gamma(state_minus.cv,
                                          temperature=state_minus.temperature)
        c_minus = state_minus.speed_of_sound
        r_minus = v_plus - 2*c_plus/(gamma_plus - 1)

        # eqs. 17 and 19
        r_plus_subsonic = v_minus + 2*c_minus/(gamma_minus - 1)
        r_plus_supersonic = v_plus + 2*c_plus/(gamma_plus - 1)
        r_plus = actx.np.where(actx.np.greater(v_minus, c_minus),
                               r_plus_supersonic, r_plus_subsonic)

        velocity_boundary = (r_minus + r_plus)/2
        velocity_boundary = (
            self._cv_plus.velocity + (velocity_boundary - v_plus)*nhat
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

        if state_minus.is_mixture:
            # Flamelet OK (I think)
            y_or_z_plus = self._cv_plus.species_mass_fractions
            energy_boundary = rho_boundary * (
                gas_model.eos.get_internal_energy(temperature=self._t_plus,
                                                  species_mass_fractions=y_or_z_plus)
                + 0.5*np.dot(velocity_boundary, velocity_boundary))

            species_mass_boundary = \
                rho_boundary * y_or_z_plus
        else:
            energy_boundary = (
                pressure_boundary / (gamma_boundary - 1)
                + 0.5*rho_boundary*np.dot(velocity_boundary, velocity_boundary))

            # in case of passive scalars
            species_mass_boundary = \
                rho_boundary * state_minus.cv.species_mass_fractions

        boundary_cv = make_conserved(dim=state_minus.dim, mass=rho_boundary,
                                     energy=energy_boundary,
                                     momentum=rho_boundary * velocity_boundary,
                                     species_mass=species_mass_boundary)

        return make_fluid_state(cv=boundary_cv, gas_model=gas_model,
                                temperature_seed=state_minus.temperature,
                                smoothness_mu=state_minus.smoothness_mu,
                                smoothness_kappa=state_minus.smoothness_kappa,
                                smoothness_beta=state_minus.smoothness_beta)

    def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Return BC fluid state."""
        return self.state_plus(dcoll, dd_bdry, gas_model, state_minus, **kwargs)

    def temperature_bc(self, dcoll, dd_bdry, state_minus, **kwargs):
        """Get temperature value used in grad(T)."""
        return state_minus.temperature

    def grad_cv_bc(self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
                   normal, **kwargs):
        """Return grad(CV) to be used in the boundary calculation of viscous flux."""
        return grad_cv_minus

    def grad_temperature_bc(self, dcoll, dd_bdry, grad_t_minus, normal, **kwargs):
        """Return grad(temperature) to be used in viscous flux at wall."""
        return grad_t_minus


class RiemannOutflowBoundary(MengaldoBoundaryCondition):
    r"""Outflow boundary treatment.

    This class implements an Riemann invariant for outflow boundary as described
    by [Mengaldo_2014]_. Note that the "minus" and "plus" are different from the
    reference to the current Mirge-COM definition.

    This boundary condition assume isentropic flow, so the regions where it can
    be applied are not general. Far-field regions are adequate, but not
    viscous-dominated regions of the flow (such as a boundary layer).

    .. automethod:: __init__
    .. automethod:: state_plus
    .. automethod:: state_bc
    .. automethod:: temperature_bc
    .. automethod:: grad_cv_bc
    .. automethod:: grad_temperature_bc
    """

    def __init__(self, cv, temperature):
        """Initialize the boundary condition object."""
        self._cv_plus = cv
        self._t_plus = temperature

        from warnings import warn
        warn("Using RiemannOutflowBoundary is not recommended.", stacklevel=2)

    def state_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the exterior solution on the boundary.

        This is the Riemann Invariant Boundary Condition described by
        [Mengaldo_2014]_ in eqs. 8 to 18.
        """
        actx = state_minus.array_context
        nhat = actx.thaw(dcoll.normal(dd_bdry))

        v_plus = np.dot(self._cv_plus.velocity, nhat)
        c_plus = gas_model.eos.sound_speed(self._cv_plus, self._t_plus)
        gamma_plus = gas_model.eos.gamma(self._cv_plus, self._t_plus)

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

        if state_minus.is_mixture:

            # using gas constant based on state_minus species
            y_or_z = state_minus.cv.species_mass_fractions
            gas_const = gas_model.eos.gas_const(
                species_mass_fractions=y_or_z)
            temperature_boundary = pressure_boundary/(gas_const*rho_boundary)

            energy_boundary = rho_boundary * (
                gas_model.eos.get_internal_energy(
                    temperature_boundary, species_mass_fractions=y_or_z)
                + 0.5*np.dot(velocity_boundary, velocity_boundary))
        else:
            energy_boundary = (
                pressure_boundary / (gamma_boundary - 1)
                + 0.5*rho_boundary*np.dot(velocity_boundary, velocity_boundary))

        # extrapolate species (appropriately done for mixture/flamelet/passive)
        species_mass_boundary = rho_boundary * state_minus.species_mass_fractions

        boundary_cv = make_conserved(dim=state_minus.dim, mass=rho_boundary,
                                     energy=energy_boundary,
                                     momentum=rho_boundary*velocity_boundary,
                                     species_mass=species_mass_boundary)

        return make_fluid_state(cv=boundary_cv, gas_model=gas_model,
                                temperature_seed=state_minus.temperature,
                                smoothness_mu=state_minus.smoothness_mu,
                                smoothness_kappa=state_minus.smoothness_kappa,
                                smoothness_beta=state_minus.smoothness_beta)

    def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Return BC fluid state."""
        return self.state_plus(dcoll, dd_bdry, gas_model, state_minus, **kwargs)

    def temperature_bc(self, dcoll, dd_bdry, state_minus, **kwargs):
        """Get temperature value used in grad(T)."""
        return state_minus.temperature

    def grad_cv_bc(self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
                   normal, **kwargs):
        """Return grad(CV) to be used in the boundary calculation of viscous flux."""
        return grad_cv_minus

    def grad_temperature_bc(self, dcoll, dd_bdry, grad_t_minus, normal, **kwargs):
        """Return grad(temperature) to be used in viscous flux at wall."""
        return grad_t_minus


class IsothermalSlipWallBoundary(MengaldoBoundaryCondition):
    r"""Isothermal viscous slip wall boundary.

    This class implements an isothermal slip wall consistent with the prescription
    by [Mengaldo_2014]_.

    .. automethod:: __init__
    .. automethod:: state_bc
    .. automethod:: temperature_bc
    .. automethod:: grad_cv_bc
    .. automethod:: grad_temperature_bc
    .. automethod:: state_plus
    """

    def __init__(self, wall_temperature=300):
        """Initialize the boundary condition object."""
        self._wall_temp = wall_temperature
        self._slip = _SlipBoundaryComponent()
        self._impermeable = _ImpermeableBoundaryComponent()

    def temperature_bc(self, dcoll, dd_bdry, state_minus, **kwargs):
        """Get temperature value used in grad(T)."""
        actx = state_minus.array_context
        cv_minus = get_cv_from_state_container(state_minus)
        wall_temp = project_from_base(dcoll, dd_bdry, self._wall_temp)
        return actx.np.zeros_like(cv_minus.mass) + wall_temp

    def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Return BC fluid state."""
        dd_bdry = as_dofdesc(dd_bdry)

        cv_minus = get_cv_from_state_container(state_minus)
        nhat = state_minus.array_context.thaw(dcoll.normal(dd_bdry))

        mom_bc = self._slip.momentum_bc(cv_minus.momentum, nhat)
        t_bc = self.temperature_bc(dcoll, dd_bdry, state_minus, **kwargs)
        # Update supports flamelet
        y_or_z = cv_minus.species_mass_fractions
        internal_energy_bc = gas_model.eos.get_internal_energy(
            temperature=t_bc, species_mass_fractions=y_or_z)

        total_energy_bc = (
            cv_minus.mass * internal_energy_bc
            + 0.5*np.dot(mom_bc, mom_bc)/cv_minus.mass)

        return replace_fluid_state(
            state_minus, gas_model,
            energy=total_energy_bc,
            momentum=mom_bc)

    def grad_cv_bc(self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
                   normal, **kwargs):
        """Return grad(CV) used in the boundary calculation of viscous flux.

        Specify the velocity gradients on the external state to ensure zero
        energy and momentum flux due to shear stresses.

        Gradients of species mass fractions are set to zero in the normal direction
        to ensure zero flux of species across the boundary.
        """
        # Should be fine as-is for flamelet
        dd_bdry = as_dofdesc(dd_bdry)
        normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))
        cv_minus = get_cv_from_state_container(state_minus)
        cv_bc = self.state_bc(
            dcoll=dcoll, dd_bdry=dd_bdry, gas_model=gas_model,
            state_minus=cv_minus, **kwargs)

        grad_v_bc = self._slip.grad_velocity_bc(
            cv_minus, cv_bc, grad_cv_minus, normal)

        grad_mom_bc = (
            cv_bc.mass * grad_v_bc
            + np.outer(cv_bc.velocity, grad_cv_minus.mass))

        grad_species_mass_bc = self._impermeable.grad_species_mass_bc(
            state_minus, grad_cv_minus, normal)

        return grad_cv_minus.replace(
            momentum=grad_mom_bc,
            species_mass=grad_species_mass_bc)

    def grad_temperature_bc(self, dcoll, dd_bdry, grad_t_minus, normal, **kwargs):
        """Return BC on grad(temperature)."""
        # Mengaldo Eqns (50-51)
        return grad_t_minus

    def state_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Return state with reflected normal-component velocity."""
        actx = state_minus.array_context
        cv_minus = get_cv_from_state_container(state_minus)
        # Grab a unit normal to the boundary
        nhat = actx.thaw(dcoll.normal(dd_bdry))

        # set the normal momentum to 0
        mom_plus = self._slip.momentum_plus(cv_minus.momentum, nhat)
        return replace_fluid_state(state_minus, gas_model, momentum=mom_plus)


class IsothermalWallBoundary(MengaldoBoundaryCondition):
    r"""Isothermal viscous wall boundary.

    This class implements an isothermal no-slip wall consistent with the prescription
    by [Mengaldo_2014]_.

    .. automethod:: __init__
    .. automethod:: state_bc
    .. automethod:: temperature_bc
    .. automethod:: grad_cv_bc
    .. automethod:: grad_temperature_bc
    .. automethod:: state_plus
    """

    def __init__(self, wall_temperature=300):
        """Initialize the boundary condition object."""
        self._wall_temp = wall_temperature
        self._no_slip = _NoSlipBoundaryComponent()
        self._impermeable = _ImpermeableBoundaryComponent()

    def temperature_bc(self, dcoll, dd_bdry, state_minus, **kwargs):
        """Get temperature value used in grad(T)."""
        actx = state_minus.array_context
        wall_temp = project_from_base(dcoll, dd_bdry, self._wall_temp)
        cv_minus = get_cv_from_state_container(state_minus)
        return actx.np.zeros_like(cv_minus.mass) + wall_temp

    def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Return BC fluid state."""
        # Mengaldo Eqn (48)
        dd_bdry = as_dofdesc(dd_bdry)
        cv_minus = get_cv_from_state_container(state_minus)
        mom_bc = self._no_slip.momentum_bc(cv_minus.momentum)
        t_bc = self.temperature_bc(dcoll, dd_bdry, state_minus, **kwargs)

        # Updated for flamelet support
        y_or_z = cv_minus.species_mass_fractions
        internal_energy_bc = gas_model.eos.get_internal_energy(
            temperature=t_bc, species_mass_fractions=y_or_z)

        # Velocity is pinned to 0 here, no kinetic energy
        total_energy_bc = cv_minus.mass*internal_energy_bc

        return replace_fluid_state(
            state_minus, gas_model,
            energy=total_energy_bc,
            momentum=mom_bc)

    def grad_cv_bc(
            self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus, normal,
            **kwargs):
        """Return grad(CV) to be used in the boundary calculation of viscous flux."""
        grad_species_mass_bc = self._impermeable.grad_species_mass_bc(
            state_minus, grad_cv_minus, normal)

        return grad_cv_minus.replace(species_mass=grad_species_mass_bc)

    def grad_temperature_bc(self, dcoll, dd_bdry, grad_t_minus, normal, **kwargs):
        """Return BC on grad(temperature)."""
        # Mengaldo Eqns (50-51)
        return grad_t_minus

    def state_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Return fluid state to use in calculation of inviscid flux."""
        # Mengaldo Eqn (45)
        cv_minus = get_cv_from_state_container(state_minus)
        mom_plus = self._no_slip.momentum_plus(cv_minus.momentum)
        return replace_fluid_state(state_minus, gas_model, momentum=mom_plus)


class AdiabaticNoslipWallBoundary(MengaldoBoundaryCondition):
    r"""Adiabatic viscous wall boundary.

    This class implements an adiabatic no-slip wall consistent with the prescription
    by [Mengaldo_2014]_.

    .. automethod:: __init__
    .. automethod:: grad_cv_bc
    .. automethod:: temperature_bc
    .. automethod:: state_plus
    .. automethod:: state_bc
    .. automethod:: grad_temperature_bc
    """

    def __init__(self):
        """Initialize the BC object."""
        self._no_slip = _NoSlipBoundaryComponent()
        self._impermeable = _ImpermeableBoundaryComponent()
        self._adiabatic = _AdiabaticBoundaryComponent()

    def temperature_bc(self, dcoll, dd_bdry, state_minus, **kwargs):
        """Get temperature value used in grad(T)."""
        return state_minus.temperature

    def state_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Return state with zero-velocity."""
        dd_bdry = as_dofdesc(dd_bdry)
        cv_minus = get_cv_from_state_container(state_minus)
        mom_plus = self._no_slip.momentum_plus(cv_minus.momentum)
        return replace_fluid_state(state_minus, gas_model, momentum=mom_plus)

    def state_bc(self, dcoll, dd_bdry, gas_model,
                            state_minus, **kwargs):
        """Return state with zero-velocity."""
        dd_bdry = as_dofdesc(dd_bdry)
        cv_minus = get_cv_from_state_container(state_minus)

        mom_bc = self._no_slip.momentum_bc(cv_minus.momentum)
        if isinstance(state_minus, ConservedVars):
            t_bc = cv_minus.mass*0 + 300.
        else:
            t_bc = self.temperature_bc(dcoll, dd_bdry, state_minus)

        # Updated for flamelet support
        y_or_z = state_minus.cv.species_mass_fractions
        internal_energy_bc = gas_model.eos.get_internal_energy(
            temperature=t_bc, species_mass_fractions=y_or_z)

        # Velocity is pinned to 0 here, no kinetic energy
        total_energy_bc = state_minus.mass_density*internal_energy_bc

        return replace_fluid_state(
            state_minus, gas_model,
            energy=total_energy_bc,
            momentum=mom_bc)

    def grad_cv_bc(self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
                   normal, **kwargs):
        """Return grad(CV) to be used in the boundary calculation of viscous flux."""
        dd_bdry = as_dofdesc(dd_bdry)
        normal = state_minus.array_context.thaw(dcoll.normal(dd_bdry))

        grad_species_mass_bc = self._impermeable.grad_species_mass_bc(
            state_minus, grad_cv_minus, normal)

        return grad_cv_minus.replace(species_mass=grad_species_mass_bc)

    def grad_temperature_bc(self, dcoll, dd_bdry, grad_t_minus, normal, **kwargs):
        """Return grad(temperature) to be used in viscous flux at wall."""
        return self._adiabatic.grad_temperature_bc(grad_t_minus, normal)


# This BC needs updated for Flamelet - it's harder to do than for others
class LinearizedOutflowBoundary(MengaldoBoundaryCondition):
    r"""Characteristics outflow BCs for linearized Euler equations.

    Implement non-reflecting outflow based on characteristic variables for
    the Euler equations assuming small perturbations based on [Giles_1988]_.
    The implementation assumes an uniform, steady flow in which the Euler
    equations are linearized about, yielding

    .. math::
        \frac{\partial U}{\partial t} + A \frac{\partial U}{\partial x} +
        B \frac{\partial U}{\partial y} = 0

    where U is the vector of perturbation (primitive) variables and
    the coefficient matrices A and B are constant matrices based on the
    uniform, steady variables.

    Using the linear hyperbolic system theory, this equation can be further
    simplified by ignoring the y-axis terms (tangent) such that wave propagation
    occurs only along the x-axis direction (normal). Then, the eigendecomposition
    results in a orthogonal system where the wave have characteristic directions
    of propagations and enable the creation of non-reflecting outflow boundaries.

    This can also be applied for Navier-Stokes equations in regions where
    viscous effects are not dominant, such as the far-field.

    .. automethod:: __init__
    .. automethod:: state_plus
    .. automethod:: state_bc
    .. automethod:: temperature_bc
    .. automethod:: grad_cv_bc
    .. automethod:: grad_temperature_bc
    """

    def __init__(
            self, free_stream_state=None, free_stream_density=None,
            free_stream_velocity=None, free_stream_pressure=None,
            free_stream_species_mass_fractions=None):
        """Initialize the BC object."""
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

        if self._ref_velocity.shape[0] > 2:
            raise ValueError("This BC only supports 1 or 2-dimensional inputs.")

        if free_stream_species_mass_fractions is None:
            from warnings import warn
            warn("Species mass fractions set to None; "
                 "using internal values.", stacklevel=2)

    def state_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Non-reflecting outflow."""
        dim = state_minus.dim
        actx = state_minus.array_context
        nhat = actx.thaw(dcoll.normal(dd_bdry))

        rtilde = state_minus.cv.mass - self._ref_mass
        utilde = state_minus.velocity - self._ref_velocity
        ptilde = state_minus.dv.pressure - self._ref_pressure

        rotation_matrix = _get_rotation_matrix(nhat)
        ur_tilde = rotation_matrix@utilde

        a = state_minus.speed_of_sound

        # zero-out the last, out-going characteristic variable
        c1 = -rtilde*a**2 + ptilde
        c3 = self._ref_mass*a*ur_tilde[0] + ptilde
        c4 = 0.0  # -self._ref_mass*a*un_tilde + ptilde

        r_tilde_bnd = 1.0/(a**2)*(-c1 + 0.5*c3 + 0.5*c4)
        un_tilde_bnd = 1.0/(self._ref_mass*a)*(0.5*c3 - 0.5*c4)
        p_tilde_bnd = 0.5*c3 + 0.5*c4

        if dim == 1:
            velocity = self._ref_velocity + un_tilde_bnd
        else:
            c2 = self._ref_mass*a*ur_tilde[1]
            ut_tilde_bnd = 1.0/(self._ref_mass*a)*c2
            ur_tilde_bnd = make_obj_array([un_tilde_bnd, ut_tilde_bnd])
            velocity = self._ref_velocity + rotation_matrix.T@ur_tilde_bnd

        mass = r_tilde_bnd + self._ref_mass
        pressure = p_tilde_bnd + self._ref_pressure

        # Here need to handle this differently for mixture vs. flamelet
        # ---
        if self._spec_mass_fracs is None:
            y_or_z = state_minus.cv.species_mass_fractions
        else:
            y_or_z = self._spec_mass_fracs

        kin_energy = 0.5*mass*np.dot(velocity, velocity)
        if state_minus.is_mixture:
            gas_const = gas_model.eos.gas_const(
                species_mass_fractions=y_or_z)
            temperature = pressure/(mass*gas_const)
            int_energy = mass*gas_model.eos.get_internal_energy(
                temperature, y_or_z)
        else:
            int_energy = pressure/(gas_model.eos.gamma() - 1.0)
        # -----

        boundary_cv = make_conserved(dim=dim,
                                     mass=mass,
                                     energy=kin_energy + int_energy,
                                     momentum=mass*velocity,
                                     species_mass=mass*y_or_z)

        return make_fluid_state(cv=boundary_cv, gas_model=gas_model,
                                temperature_seed=state_minus.temperature,
                                smoothness_mu=state_minus.smoothness_mu,
                                smoothness_kappa=state_minus.smoothness_kappa,
                                smoothness_beta=state_minus.smoothness_beta)

    def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Return BC fluid state."""
        return self.state_plus(dcoll, dd_bdry, gas_model, state_minus, **kwargs)

    def temperature_bc(self, dcoll, dd_bdry, state_minus, **kwargs):
        """Return temperature for use in grad(temperature)."""
        return state_minus.temperature

    def grad_temperature_bc(self, dcoll, dd_bdry, grad_t_minus, normal, **kwargs):
        """Return grad(T) to be used in the boundary calculation of viscous flux."""
        return grad_t_minus

    def grad_cv_bc(self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
                   normal, **kwargs):
        """Return grad(CV) to be used in the boundary calculation of viscous flux."""
        return grad_cv_minus


# This BC needs updated for Flamelet - it's harder to do than for others
class LinearizedInflowBoundary(MengaldoBoundaryCondition):
    r"""Characteristics inflow BCs for linearized Euler equations.

    .. automethod:: __init__
    .. automethod:: state_plus
    .. automethod:: state_bc
    .. automethod:: temperature_bc
    .. automethod:: grad_cv_bc
    .. automethod:: grad_temperature_bc
    """

    def __init__(
            self, free_stream_state=None, free_stream_velocity=None,
            free_stream_pressure=None, free_stream_density=None,
            free_stream_species_mass_fractions=None):
        """Initialize the BC object."""
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

        if self._ref_velocity.shape[0] > 2:
            raise ValueError("This BC only supports 1 or 2-dimensional inputs.")

        if free_stream_species_mass_fractions is None:
            self._spec_mass_fracs = np.empty((0,), dtype=object)

    def state_plus(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Non-reflecting inflow."""
        actx = state_minus.array_context
        nhat = -1.0*actx.thaw(dcoll.normal(dd_bdry))

        # rtilde = state_minus.cv.mass - self._ref_mass
        utilde = state_minus.velocity - self._ref_velocity
        ptilde = state_minus.dv.pressure - self._ref_pressure

        # get the normal component of velocity. The tangential is not required.
        un_tilde = utilde@nhat

        a = state_minus.speed_of_sound

        # zero-out the first three, incoming characteristics
        c1 = 0.0  # -rtilde*a**2 + ptilde
        c2 = 0.0  # self._ref_mass*a*ut_tilde  # noqa
        c3 = 0.0  # self._ref_mass*a*un_tilde + ptilde
        c4 = -1.0*self._ref_mass*a*un_tilde + ptilde

        r_tilde_bnd = 1.0/(a**2)*(-c1 + 0.5*c3 + 0.5*c4)
        un_tilde_bnd = 1.0/(self._ref_mass*a)*(0.5*c3 - 0.5*c4)
        # ut_tilde_bnd = 1.0/(self._ref_mass*a)*c2
        p_tilde_bnd = 0.5*c3 + 0.5*c4

        mass = r_tilde_bnd + self._ref_mass
        velocity = self._ref_velocity + un_tilde_bnd*nhat
        pressure = p_tilde_bnd + self._ref_pressure

        kin_energy = 0.5*mass*np.dot(velocity, velocity)
        # Needs updated for flamelet
        # ---
        y_or_z = self._spec_mass_fracs
        if state_minus.is_mixture:
            gas_const = gas_model.eos.gas_const(
                species_mass_fractions=y_or_z)
            temperature = self._ref_pressure/(self._ref_mass*gas_const)
            int_energy = mass*gas_model.eos.get_internal_energy(
                temperature, y_or_z)
        else:
            int_energy = pressure/(gas_model.eos.gamma() - 1.0)
        # ----

        boundary_cv = make_conserved(dim=state_minus.dim,
                                     mass=mass,
                                     energy=kin_energy + int_energy,
                                     momentum=mass*velocity,
                                     species_mass=mass*y_or_z)

        return make_fluid_state(cv=boundary_cv, gas_model=gas_model,
                                temperature_seed=state_minus.temperature)

    def state_bc(self, dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        """Return BC fluid state."""
        return self.state_plus(dcoll, dd_bdry, gas_model, state_minus, **kwargs)

    def temperature_bc(self, dcoll, dd_bdry, state_minus, **kwargs):
        """Return temperature for use in grad(temperature)."""
        return state_minus.temperature

    def grad_temperature_bc(self, dcoll, dd_bdry, grad_t_minus, normal, **kwargs):
        """Return grad(T) used in the boundary calculation of viscous flux."""
        return grad_t_minus

    def grad_cv_bc(self, dcoll, dd_bdry, gas_model, state_minus, grad_cv_minus,
                   normal, **kwargs):
        """Return grad(CV) used in the boundary calculation of viscous flux."""
        return grad_cv_minus
