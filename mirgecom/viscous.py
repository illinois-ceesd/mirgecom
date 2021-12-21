r""":mod:`mirgecom.viscous` provides helper functions for viscous flow.

Viscous Flux Calculation
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: viscous_flux
.. autofunction:: viscous_stress_tensor
.. autofunction:: diffusive_flux
.. autofunction:: conductive_heat_flux
.. autofunction:: diffusive_heat_flux
.. autofunction:: viscous_facial_flux
.. autofunction:: viscous_flux_central

Viscous Time Step Computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: get_viscous_timestep
.. autofunction:: get_viscous_cfl
.. autofunction:: get_local_max_species_diffusivity
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
from grudge.trace_pair import TracePair
from meshmode.dof_array import thaw, DOFArray

from mirgecom.flux import divergence_flux_central
from mirgecom.fluid import (
    velocity_gradient,
    species_mass_fraction_gradient,
    make_conserved
)


# low level routine works with numpy arrays and can be tested without
# a full grid + fluid state, etc
def _compute_viscous_stress_tensor(dim, mu, mu_b, grad_v):
    return mu*(grad_v + grad_v.T) + (mu_b - 2*mu/3)*np.trace(grad_v)*np.eye(dim)


def viscous_stress_tensor(state, grad_cv):
    r"""Compute the viscous stress tensor.

    The viscous stress tensor $\tau$ is defined by:

    .. math::

        \mathbf{\tau} = \mu\left(\nabla{\mathbf{v}}
        +\left(\nabla{\mathbf{v}}\right)^T\right) + (\mu_B - \frac{2\mu}{3})
        \left(\nabla\cdot\mathbf{v}\right)

    Parameters
    ----------
    state: :class:`~mirgecom.gas_model.FluidState`

        Full conserved and thermal state of fluid

    grad_cv: :class:`~mirgecom.fluid.ConservedVars`

        Gradient of the fluid state

    Returns
    -------
    numpy.ndarray

        The viscous stress tensor
    """
    return _compute_viscous_stress_tensor(
        dim=state.dim, mu=state.viscosity, mu_b=state.bulk_viscosity,
        grad_v=velocity_gradient(state.cv, grad_cv))


# low level routine works with numpy arrays and can be tested without
# a full grid + fluid state, etc
def _compute_diffusive_flux(density, d_alpha, grad_y):
    return -density*d_alpha.reshape(-1, 1)*grad_y


def diffusive_flux(state, grad_cv):
    r"""Compute the species diffusive flux vector, ($\mathbf{J}_{\alpha}$).

    The species diffusive flux is defined by:

    .. math::

        \mathbf{J}_{\alpha} = -\rho{d}_{(\alpha)}\nabla{Y_{\alpha}}~~
        (\mathtt{no~implied~sum}),

    with species diffusivities ${d}_{\alpha}$, and species mass
    fractions ${Y}_{\alpha}$.

    Parameters
    ----------
    state: :class:`~mirgecom.gas_model.FluidState`

        Full fluid conserved and thermal state

    grad_cv: :class:`~mirgecom.fluid.ConservedVars`

        Gradient of the fluid state

    Returns
    -------
    numpy.ndarray

        The species diffusive flux vector, $\mathbf{J}_{\alpha}$
    """
    return _compute_diffusive_flux(state.mass_density, state.species_diffusivity,
                                   species_mass_fraction_gradient(state.cv, grad_cv))


# low level routine works with numpy arrays and can be tested without
# a full grid + fluid state, etc
def _compute_conductive_heat_flux(grad_t, kappa):
    return -kappa*grad_t


def conductive_heat_flux(state, grad_t):
    r"""Compute the conductive heat flux, ($\mathbf{q}_{c}$).

    The conductive heat flux is defined by:

    .. math::

        \mathbf{q}_{c} = -\kappa\nabla{T},

    with thermal conductivity $\kappa$, and gas temperature $T$.

    Parameters
    ----------
    state: :class:`~mirgecom.gas_model.FluidState`

        Full fluid conserved and thermal state

    grad_t: numpy.ndarray

        Gradient of the fluid temperature

    Returns
    -------
    numpy.ndarray

        The conductive heat flux vector
    """
    return _compute_conductive_heat_flux(grad_t, state.thermal_conductivity)


# low level routine works with numpy arrays and can be tested without
# a full grid + fluid state, etc
def _compute_diffusive_heat_flux(j, h_alpha):
    return sum(h_alpha.reshape(-1, 1) * j)


def diffusive_heat_flux(state, j):
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
    state: :class:`~mirgecom.gas_model.FluidState`

        Full fluid conserved and thermal state

    j: numpy.ndarray

        The species diffusive flux vector

    Returns
    -------
    numpy.ndarray

        The total diffusive heat flux vector
    """
    if state.is_mixture:
        return _compute_diffusive_heat_flux(j, state.species_enthalpies)
    return 0


def viscous_flux(state, grad_cv, grad_t):
    r"""Compute the viscous flux vectors.

    The viscous fluxes are:

    .. math::

        \mathbf{F}_V = [0,\tau\cdot\mathbf{v} - \mathbf{q},
        \tau,-\mathbf{J}_\alpha],

    with fluid velocity ($\mathbf{v}$), viscous stress tensor
    ($\mathbf{\tau}$), heat flux ($\mathbf{q}$), and diffusive flux
    for each species ($\mathbf{J}_\alpha$).

    .. note::

        The fluxes are returned as a :class:`mirgecom.fluid.ConservedVars`
        object with a *dim-vector* for each conservation equation. See
        :class:`mirgecom.fluid.ConservedVars` for more information about
        how the fluxes are represented.

    Parameters
    ----------
    state: :class:`~mirgecom.gas_model.FluidState`

        Full fluid conserved and thermal state

    grad_cv: :class:`~mirgecom.fluid.ConservedVars`

        Gradient of the fluid state

    grad_t: numpy.ndarray

        Gradient of the fluid temperature

    Returns
    -------
    :class:`~mirgecom.fluid.ConservedVars` or float

        The viscous transport flux vector if viscous transport properties
        are provided, scalar zero otherwise.
    """
    if not state.is_viscous:
        import warnings
        warnings.warn("Viscous fluxes requested for inviscid state.")
        return 0

    viscous_mass_flux = 0 * state.momentum_density
    tau = viscous_stress_tensor(state, grad_cv)
    j = diffusive_flux(state, grad_cv)

    viscous_energy_flux = (
        np.dot(tau, state.velocity) - diffusive_heat_flux(state, j)
        - conductive_heat_flux(state, grad_t)
    )

    return make_conserved(state.dim,
            mass=viscous_mass_flux,
            energy=viscous_energy_flux,
            momentum=tau, species_mass=-j)


def viscous_flux_central(discr, state_pair, grad_cv_pair, grad_t_pair, **kwargs):
    r"""Return a central viscous facial flux for the divergence operator.

    The central flux is defined as:

    .. math::

        f_{\text{central}} = \frac{1}{2}\left(\mathbf{f}_v^+
        + \mathbf{f}_v^-\right)\cdot\hat{\mathbf{n}},

    with viscous fluxes ($\mathbf{f}_v$), and the outward pointing
    face normal ($\hat{\mathbf{n}}$).

    Parameters
    ----------
    discr: :class:`~grudge.eager.EagerDGDiscretization`

        The discretization to use

    state_pair: :class:`~grudge.trace_pair.TracePair`

        Trace pair of :class:`~mirgecom.gas_model.FluidState` with the full fluid
        conserved and thermal state on the faces

    grad_cv_pair: :class:`~grudge.trace_pair.TracePair`

        Trace pair of :class:`~mirgecom.fluid.ConservedVars` with the gradient of the
        fluid solution on the faces

    grad_t_pair: :class:`~grudge.trace_pair.TracePair`

        Trace pair of temperature gradient on the faces.

    Returns
    -------
    :class:`~mirgecom.fluid.ConservedVars`

        The viscous transport flux in the face-normal direction on "all_faces" or
        local to the sub-discretization depending on *local* input parameter
    """
    actx = state_pair.int.array_context
    normal = thaw(actx, discr.normal(state_pair.dd))

    f_int = viscous_flux(state_pair.int, grad_cv_pair.int,
                         grad_t_pair.int)
    f_ext = viscous_flux(state_pair.ext, grad_cv_pair.ext,
                         grad_t_pair.ext)
    f_pair = TracePair(state_pair.dd, interior=f_int, exterior=f_ext)

    return divergence_flux_central(f_pair, normal)


def viscous_facial_flux(discr, gas_model, state_pair, grad_cv_pair, grad_t_pair,
                        numerical_flux_func=viscous_flux_central, local=False):
    """Return the viscous facial flux for the divergence operator.

    Parameters
    ----------
    discr: :class:`~grudge.eager.EagerDGDiscretization`

        The discretization to use

    state_pair: :class:`~grudge.trace_pair.TracePair`

        Trace pair of :class:`~mirgecom.gas_model.FluidState` with the full fluid
        conserved and thermal state on the faces

    grad_cv_pair: :class:`~grudge.trace_pair.TracePair`

        Trace pair of :class:`~mirgecom.fluid.ConservedVars` with the gradient of the
        fluid solution on the faces

    grad_t_pair: :class:`~grudge.trace_pair.TracePair`

        Trace pair of temperature gradient on the faces.

    local: bool

        Indicates whether to skip projection of fluxes to "all_faces" or not. If
        set to *False* (the default), the returned fluxes are projected to
        "all_faces".  If set to *True*, the returned fluxes are not projected to
        "all_faces"; remaining instead on the boundary restriction.

    Returns
    -------
    :class:`~mirgecom.fluid.ConservedVars`

        The viscous transport flux in the face-normal direction on "all_faces" or
        local to the sub-discretization depending on *local* input parameter
    """
    num_flux = numerical_flux_func(discr=discr, gas_model=gas_model,
                                   state_pair=state_pair,
                                   grad_cv_pair=grad_cv_pair,
                                   grad_t_pair=grad_t_pair)
    dd = state_pair.dd
    dd_allfaces = dd.with_dtag("all_faces")
    return num_flux if local else discr.project(dd, dd_allfaces, num_flux)


def get_viscous_timestep(discr, state):
    """Routine returns the the node-local maximum stable viscous timestep.

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization

        the discretization to use

    state: :class:`~mirgecom.gas_model.FluidState`

        Full fluid conserved and thermal state

    Returns
    -------
    :class:`~meshmode.dof_array.DOFArray`

        The maximum stable timestep at each node.
    """
    from grudge.dt_utils import characteristic_lengthscales

    length_scales = characteristic_lengthscales(state.array_context, discr)

    mu = 0
    d_alpha_max = 0
    if state.is_viscous:
        mu = state.viscosity
        d_alpha_max = \
            get_local_max_species_diffusivity(
                state.array_context,
                state.species_diffusivity
            )

    return(
        length_scales / (state.wavespeed
        + ((mu + d_alpha_max) / length_scales))
    )


def get_viscous_cfl(discr, dt, state):
    """Calculate and return node-local CFL based on current state and timestep.

    Parameters
    ----------
    discr: :class:`~grudge.eager.EagerDGDiscretization`

        the discretization to use

    dt: float or :class:`~meshmode.dof_array.DOFArray`

        A constant scalar dt or node-local dt

    state: :class:`~mirgecom.gas_model.FluidState`

        The full fluid conserved and thermal state

    Returns
    -------
    :class:`~meshmode.dof_array.DOFArray`

        The CFL at each node.
    """
    return dt / get_viscous_timestep(discr, state=state)


def get_local_max_species_diffusivity(actx, d_alpha):
    """Return the maximum species diffusivity at every point.

    Parameters
    ----------
    actx: :class:`~arraycontext.ArrayContext`

        Array context to use

    d_alpha: numpy.ndarray

        Species diffusivities

    Returns
    -------
    :class:`~meshmode.dof_array.DOFArray`

        The maximum species diffusivity
    """
    if len(d_alpha) == 0:
        return 0
    if not isinstance(d_alpha[0], DOFArray):
        return max(d_alpha)

    from functools import reduce
    return reduce(actx.np.maximum, d_alpha)
