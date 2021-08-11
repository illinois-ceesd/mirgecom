r""":mod:`mirgecom.viscous` provides helper functions for viscous flow.

Viscous Flux Calculation
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: viscous_flux
.. autofunction:: viscous_stress_tensor
.. autofunction:: diffusive_flux
.. autofunction:: conductive_heat_flux
.. autofunction:: diffusive_heat_flux
.. autofunction:: viscous_facial_flux

Viscous Time Step Computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: get_viscous_timestep
.. autofunction:: get_viscous_cfl
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

import numpy as np
from pytools.obj_array import make_obj_array
from mirgecom.fluid import (
    velocity_gradient,
    species_mass_fraction_gradient,
    make_conserved
)
from meshmode.dof_array import thaw


def viscous_stress_tensor(discr, eos, cv, grad_cv):
    r"""Compute the viscous stress tensor.

    The viscous stress tensor $\tau$ is defined by:

    .. math::

        \mathbf{\tau} = \mu\left(\nabla{\mathbf{v}}
        +\left(\nabla{\mathbf{v}}\right)^T\right) + (\mu_B - \frac{2\mu}{3})
        \left(\nabla\cdot\mathbf{v}\right)

    Parameters
    ----------
    eos: :class:`~mirgecom.eos.GasEOS`
        A gas equation of state with a non-empty
        :class:`~mirgecom.transport.TransportModel`.
    cv: :class:`~mirgecom.fluid.ConservedVars`
        Fluid state
    grad_cv: :class:`~mirgecom.fluid.ConservedVars`
        Gradient of the fluid state

    Returns
    -------
    numpy.ndarray
        The viscous stress tensor
    """
    dim = cv.dim
    transport = eos.transport_model()

    mu_b = transport.bulk_viscosity(eos, cv)
    mu = transport.viscosity(eos, cv)

    grad_v = velocity_gradient(discr, cv, grad_cv)
    div_v = np.trace(grad_v)

    return mu*(grad_v + grad_v.T) + (mu_b - 2*mu/3)*div_v*np.eye(dim)


def diffusive_flux(discr, eos, cv, grad_cv):
    r"""Compute the species diffusive flux vector, ($\mathbf{J}_{\alpha}$).

    The species diffussive flux is defined by:

    .. math::

        \mathbf{J}_{\alpha} = -\rho{d}_{(\alpha)}\nabla{Y_{\alpha}}~~
        (\mathtt{no~implied~sum}),

    with species diffusivities ${d}_{\alpha}$, and species mass
    fractions ${Y}_{\alpha}$.

    Parameters
    ----------
    eos: :class:`~mirgecom.eos.GasEOS`
        A gas equation of state with a non-empty
        :class:`~mirgecom.transport.TransportModel`
    cv: :class:`~mirgecom.fluid.ConservedVars`
        Fluid state
    grad_cv: :class:`~mirgecom.fluid.ConservedVars`
        Gradient of the fluid state

    Returns
    -------
    numpy.ndarray
        The species diffusive flux vector, $\mathbf{J}_{\alpha}$
    """
    nspecies = cv.nspecies
    transport = eos.transport_model()

    grad_y = species_mass_fraction_gradient(discr, cv, grad_cv)
    d = transport.species_diffusivity(eos, cv)

    # TODO: Better way?
    obj_ary = -make_obj_array([cv.mass*d[i]*grad_y[i] for i in range(nspecies)])
    diffusive_flux = np.empty(shape=(nspecies, discr.dim), dtype=object)
    for idx, v in enumerate(obj_ary):
        diffusive_flux[idx] = v

    return diffusive_flux


def conductive_heat_flux(discr, eos, cv, grad_t):
    r"""Compute the conductive heat flux, ($\mathbf{q}_{c}$).

    The conductive heat flux is defined by:

    .. math::

        \mathbf{q}_{c} = -\kappa\nabla{T},

    with thermal conductivity $\kappa$, and gas temperature $T$.

    Parameters
    ----------
    eos: :class:`~mirgecom.eos.GasEOS`
        A gas equation of state with a non-empty
        :class:`~mirgecom.transport.TransportModel`
    cv: :class:`~mirgecom.fluid.ConservedVars`
        Fluid state
    grad_t: numpy.ndarray
        Gradient of the fluid temperature

    Returns
    -------
    numpy.ndarray
        The conductive heat flux vector
    """
    transport = eos.transport_model()
    return -transport.thermal_conductivity(eos, cv)*grad_t


def diffusive_heat_flux(discr, eos, cv, j):
    r"""Compute the diffusive heat flux, ($\mathbf{q}_{d}$).

    The diffusive heat flux is defined by:

    .. math::

        \mathbf{q}_{d} = \sum_{\alpha=1}^{\mathtt{Nspecies}}{h}_{\alpha}
        \mathbf{J}_{\alpha},

    with species specific enthalpy ${h}_{\alpha}$ and diffusive flux
    ($\mathbf{J}_{\alpha}$) defined as:

    .. math::

        \mathbf{J}_{\alpha} = -\rho{d}_{\alpha}\nabla{Y}_{\alpha},

    where ${Y}_{\alpha}$ is the vector of species mass fractions.

    Parameters
    ----------
    eos: mirgecom.eos.GasEOS
        A gas equation of state with a non-empty
        :class:`~mirgecom.transport.TransportModel`
    cv: :class:`~mirgecom.fluid.ConservedVars`
        Fluid state
    j: numpy.ndarray
        The species diffusive flux vector

    Returns
    -------
    numpy.ndarray
        The total diffusive heath flux vector
    """
    nspec = cv.nspecies
    if nspec > 0:
        try:
            h_alpha = eos.species_enthalpies(cv)
            return sum(h_alpha[i]*j[i] for i in range(nspec))
        except NotImplementedError:
            return 0
    return 0


# TODO: We could easily make this more general (dv, grad_dv)
# so that user can define arbitrary array containers for *dv*,
# enabling support for other models and simultaneous eqns.
def viscous_flux(discr, eos, cv, grad_cv, grad_t):
    r"""Compute the viscous flux vectors.

    The viscous fluxes are:

    .. math::

        \mathbf{F}_V = [0,\tau\cdot\mathbf{v} - \mathbf{q},
        \tau_{:i},-\mathbf{J}_\alpha],

    with fluid velocity ($\mathbf{v}$), viscous stress tensor
    ($\mathbf{\tau}$), and diffusive flux for each species
    ($\mathbf{J}_\alpha$).

    .. note::

        The fluxes are returned as a :class:`mirgecom.fluid.ConservedVars`
        object with a *dim-vector* for each conservation equation. See
        :class:`mirgecom.fluid.ConservedVars` for more information about
        how the fluxes are represented.

    Parameters
    ----------
    eos: :class:`~mirgecom.eos.GasEOS`
        A gas equation of state with a non-empty
        :class:`~mirgecom.transport.TransportModel`
    cv: :class:`~mirgecom.fluid.ConservedVars`
        Fluid state
    grad_cv: :class:`~mirgecom.fluid.ConservedVars`
        Gradient of the fluid state
    grad_t: numpy.ndarray
        Gradient of the fluid temperature

    Returns
    -------
    :class:`~mirgecom.fluid.ConservedVars`
        The viscous transport flux vector
    """
    transport = eos.transport_model()
    if transport is None:
        return 0

    dim = cv.dim
    viscous_mass_flux = 0 * cv.momentum

    nspecies = cv.nspecies
    if nspecies > 0:
        j = diffusive_flux(discr, eos, cv, grad_cv)
        heat_flux_diffusive = diffusive_heat_flux(discr, eos, cv, j)
    else:
        # passes the right (empty) shape for diffusive flux when no species
        j = cv.momentum * cv.species_mass.reshape(-1, 1)
        heat_flux_diffusive = 0

    tau = viscous_stress_tensor(discr, eos, cv, grad_cv)
    viscous_energy_flux = (
        np.dot(tau, cv.velocity)
        - conductive_heat_flux(discr, eos, cv, grad_t)
        - heat_flux_diffusive
    )

    return make_conserved(dim,
            mass=viscous_mass_flux,
            energy=viscous_energy_flux,
            momentum=tau, species_mass=-j)


def viscous_facial_flux(discr, eos, cv_tpair, grad_cv_tpair, grad_t_tpair,
                        local=False):
    """Return the viscous flux across a face given the solution on both sides.

    Parameters
    ----------
    eos: :class:`~mirgecom.eos.GasEOS`
        Implementing the pressure and temperature functions for
        returning pressure and temperature as a function of the state q.
    cv_tpair: :class:`grudge.trace_pair.TracePair`
        Trace pair of :class:`~mirgecom.fluid.ConservedVars` with the fluid solution
        on the faces
    grad_cv_tpair: :class:`grudge.trace_pair.TracePair`
        Trace pair of :class:`~mirgecom.fluid.ConservedVars` with the gradient of the
        fluid solution on the faces
    grad_t_tpair: :class:`grudge.trace_pair.TracePair`
        Trace pair of temperature gradient on the faces.
    local: bool
        Indicates whether to skip projection of fluxes to "all_faces" or not. If
        set to *False* (the default), the returned fluxes are projected to
        "all_faces."  If set to *True*, the returned fluxes are not projected to
        "all_faces"; remaining instead on the boundary restriction.

    Returns
    -------
    :class:`~mirgecom.fluid.ConservedVars`
        The viscous transport flux in the face-normal direction on "all_faces" or
        local to the sub-discretization depending on *local* input parameter
    """
    actx = cv_tpair.int.array_context
    normal = thaw(actx, discr.normal(cv_tpair.dd))

    # todo: user-supplied flux routine
    f_int = viscous_flux(discr, eos, cv_tpair.int, grad_cv_tpair.int,
                         grad_t_tpair.int)
    f_ext = viscous_flux(discr, eos, cv_tpair.ext, grad_cv_tpair.ext,
                         grad_t_tpair.ext)

    # This bit hard-codes BR1 central flux, use mirgecom.flux.central?
    f_avg = 0.5*(f_int + f_ext)
    flux_weak = make_conserved(cv_tpair.int.dim, q=(f_avg.join() @ normal))

    if local is False:
        return discr.project(cv_tpair.dd, "all_faces", flux_weak)
    return flux_weak


def get_viscous_timestep(discr, eos, cv):
    """Routine returns the the node-local maximum stable viscous timestep.

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization
        the discretization to use
    eos: :class:`~mirgecom.eos.GasEOS`
        An equation of state implementing the speed_of_sound method
    cv: :class:`~mirgecom.fluid.ConservedVars`
        Fluid solution

    Returns
    -------
    :class:`~meshmode.dof_array.DOFArray`
        The maximum stable timestep at each node.
    """
    from grudge.dt_utils import characteristic_lengthscales
    from mirgecom.fluid import compute_wavespeed

    length_scales = characteristic_lengthscales(cv.array_context, discr)

    mu = 0
    transport = eos.transport_model()
    if transport:
        mu = transport.viscosity(eos, cv)

    return(
        length_scales / (compute_wavespeed(eos, cv) + mu / length_scales)
    )


def get_viscous_cfl(discr, eos, dt, cv):
    """Calculate and return node-local CFL based on current state and timestep.

    Parameters
    ----------
    discr: :class:`grudge.eager.EagerDGDiscretization`
        the discretization to use
    eos: :class:`~mirgecom.eos.GasEOS`
        Implementing the pressure and temperature functions for
        returning pressure and temperature as a function of the state q.
    dt: float or :class:`~meshmode.dof_array.DOFArray`
        A constant scalar dt or node-local dt
    cv: :class:`~mirgecom.fluid.ConservedVars`
        The fluid conserved variables

    Returns
    -------
    :class:`~meshmode.dof_array.DOFArray`
        The CFL at each node.
    """
    return dt / get_viscous_timestep(discr, eos=eos, cv=cv)
