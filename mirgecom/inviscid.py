r""":mod:`mirgecom.inviscid` provides helper functions for inviscid flow.

Inviscid Flux Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: inviscid_flux
.. autofunction:: inviscid_facial_flux
.. autofunction:: inviscid_flux_rusanov

Entropy Stable Flux Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: entropy_conserving_flux_chandrashekar
.. autofunction:: entropy_stable_facial_flux

Inviscid Time Step Computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: get_inviscid_timestep
.. autofunction:: get_inviscid_cfl
"""

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

import numpy as np
import grudge.op as op

from arraycontext import thaw, outer

from mirgecom.fluid import (
    make_conserved,
    ConservedVars,
    conservative_to_primitive_vars
)
from mirgecom.eos import GasEOS

from meshmode.dof_array import DOFArray

from pytools.obj_array import make_obj_array


def inviscid_flux(state):
    r"""Compute the inviscid flux vectors from fluid conserved vars *cv*.

    The inviscid fluxes are
    $(\rho\vec{V},(\rho{E}+p)\vec{V},\rho(\vec{V}\otimes\vec{V})
    +p\mathbf{I}, \rho{Y_s}\vec{V})$

    .. note::

        The fluxes are returned as a :class:`mirgecom.fluid.ConservedVars`
        object with a *dim-vector* for each conservation equation. See
        :class:`mirgecom.fluid.ConservedVars` for more information about
        how the fluxes are represented.

    Parameters
    ----------
    state: :class:`~mirgecom.gas_model.FluidState`

        Full fluid conserved and thermal state.

    Returns
    -------
    :class:`~mirgecom.fluid.ConservedVars`

        A CV object containing the inviscid flux vector for each
        conservation equation.
    """
    mass_flux = state.momentum_density
    energy_flux = state.velocity * (state.energy_density + state.pressure)
    mom_flux = (
        state.mass_density * np.outer(state.velocity, state.velocity)
        + np.eye(state.dim)*state.pressure
    )
    species_mass_flux = (  # reshaped: (nspecies, dim)
        state.velocity * state.species_mass_density.reshape(-1, 1)
    )
    return make_conserved(state.dim, mass=mass_flux, energy=energy_flux,
                          momentum=mom_flux, species_mass=species_mass_flux)


def inviscid_flux_rusanov(state_pair, gas_model, normal, **kwargs):
    r"""High-level interface for inviscid facial flux using Rusanov numerical flux.

    The Rusanov inviscid numerical flux is calculated as:

    .. math::

        F^{*}_{\mathtt{LFR}} = \frac{1}{2}(\mathbf{F}(q^-)
        +\mathbf{F}(q^+)) \cdot \hat{n} + \frac{\lambda}{2}(q^{-} - q^{+}),

    where $q^-, q^+$ are the fluid solution state on the interior and the
    exterior of the face on which the flux is to be calculated, $\mathbf{F}$ is
    the inviscid fluid flux, $\hat{n}$ is the face normal, and $\lambda$ is the
    *local* maximum fluid wavespeed.
    """
    actx = state_pair.int.array_context
    lam = actx.np.maximum(state_pair.int.wavespeed, state_pair.ext.wavespeed)
    from mirgecom.flux import num_flux_lfr
    return num_flux_lfr(f_minus=inviscid_flux(state_pair.int)@normal,
                        f_plus=inviscid_flux(state_pair.ext)@normal,
                        q_minus=state_pair.int.cv,
                        q_plus=state_pair.ext.cv, lam=lam)


def inviscid_flux_hll(state_pair, gas_model, normal, **kwargs):
    r"""High-level interface for inviscid facial flux using HLL numerical flux.

    The Harten, Lax, van Leer approximate riemann numerical flux is calculated as:

    .. math::

        F^{*}_{\mathtt{HLL}} = \frac{1}{2}(\mathbf{F}(q^-)
        +\mathbf{F}(q^+)) \cdot \hat{n} + \frac{\lambda}{2}(q^{-} - q^{+}),

    where $q^-, q^+$ are the fluid solution state on the interior and the
    exterior of the face on which the flux is to be calculated, $\mathbf{F}$ is
    the inviscid fluid flux, $\hat{n}$ is the face normal, and $\lambda$ is the
    *local* maximum fluid wavespeed.
    """
    # calculate left/right wavespeeds
    actx = state_pair.int.array_context
    ones = 0.*state_pair.int.mass_density + 1.

    # note for me, treat the interior state as left and the exterior state as right
    # pressure estimate
    p_int = state_pair.int.pressure
    p_ext = state_pair.ext.pressure
    u_int = np.dot(state_pair.int.velocity, normal)
    u_ext = np.dot(state_pair.ext.velocity, normal)
    rho_int = state_pair.int.mass_density
    rho_ext = state_pair.ext.mass_density
    c_int = state_pair.int.speed_of_sound
    c_ext = state_pair.ext.speed_of_sound

    p_star = (0.5*(p_int + p_ext) + (1./8.)*(u_int - u_ext)
              * (rho_int + rho_ext) * (c_int + c_ext))

    gamma_int = gas_model.eos.gamma(state_pair.int.cv)
    gamma_ext = gas_model.eos.gamma(state_pair.ext.cv)

    q_int = 1 + (gamma_int + 1)/(2*gamma_int)*(p_star/p_int - 1)
    q_ext = 1 + (gamma_ext + 1)/(2*gamma_ext)*(p_star/p_ext - 1)

    pres_check_int = actx.np.greater(p_star, p_int)
    pres_check_ext = actx.np.greater(p_star, p_ext)

    q_int = actx.np.where(pres_check_int, q_int, ones)
    q_ext = actx.np.where(pres_check_ext, q_ext, ones)

    q_int = actx.np.sqrt(q_int)
    q_ext = actx.np.sqrt(q_ext)

    # left (internal), and right (external) wave speed estimates
    # can alternatively use the roe estimated states to find the wave speeds
    wavespeed_int = u_int - c_int*q_int
    wavespeed_ext = u_ext + c_ext*q_ext

    from mirgecom.flux import hll_flux_driver
    return hll_flux_driver(state_pair, inviscid_flux,
                           wavespeed_int, wavespeed_ext, normal)


def inviscid_facial_flux(discr, gas_model, state_pair,
                         numerical_flux_func=inviscid_flux_rusanov, local=False):
    r"""Return the numerical inviscid flux for the divergence operator.

    Different numerical fluxes may be used through the specificiation of
    the *numerical_flux_func*. By default, a Rusanov-type flux is used.

    Parameters
    ----------
    discr: :class:`~grudge.eager.EagerDGDiscretization`

        The discretization collection to use

    state_pair: :class:`~grudge.trace_pair.TracePair`

        Trace pair of :class:`~mirgecom.gas_model.FluidState` for the face upon
        which the flux calculation is to be performed

    local: bool

        Indicates whether to skip projection of fluxes to "all_faces" or not. If
        set to *False* (the default), the returned fluxes are projected to
        "all_faces."  If set to *True*, the returned fluxes are not projected to
        "all_faces"; remaining instead on the boundary restriction.

    Returns
    -------
    :class:`~mirgecom.fluid.ConservedVars`

        A CV object containing the scalar numerical fluxes at the input faces.
        The returned fluxes are scalar because they've already been dotted with
        the face normals as required by the divergence operator for which they
        are being computed.
    """
    normal = thaw(discr.normal(state_pair.dd), state_pair.int.array_context)
    num_flux = numerical_flux_func(state_pair, gas_model, normal)
    dd = state_pair.dd
    dd_allfaces = dd.with_dtag("all_faces")
    return num_flux if local else discr.project(dd, dd_allfaces, num_flux)


def entropy_conserving_flux_chandrashekar(
        discr, eos: GasEOS, cv_ll: ConservedVars, cv_rr: ConservedVars):
    """Compute the entropy conservative fluxes from states *cv_ll* and *cv_rr*.

    This routine implements the two-point volume flux based on the entropy
    conserving and kinetic energy preserving two-point flux in:
    - Chandrashekar (2013) Kinetic Energy Preserving and Entropy Stable Finite
    Volume Schemes for Compressible Euler and Navier-Stokes Equations
    [DOI](https://doi.org/10.4208/cicp.170712.010313a)

    Parameters
    ----------
    discr: :class:`~grudge.eager.EagerDGDiscretization`

        The discretization collection to use

    cv_ll: :class:`~mirgecom.fluid.ConservedVars`

        The conserved variables for the "left" state

    cv_rr: :class:`~mirgecom.fluid.ConservedVars`

        The conserved variables for the "right" state

    Returns
    -------
    :class:`~mirgecom.fluid.ConservedVars`

        A CV object containing the matrix-valued two-point flux vectors
        for each conservation equation.
    """
    dim = discr.dim
    actx = cv_ll.array_context

    def ln_mean(x: DOFArray, y: DOFArray, epsilon=1e-4):
        f2 = (x * (x - 2 * y) + y * y) / (x * (x + 2 * y) + y * y)
        return actx.np.where(
            actx.np.less(f2, epsilon),
            (x + y) / (2 + f2*2/3 + f2*f2*2/5 + f2*f2*f2*2/7),
            (y - x) / actx.np.log(y / x)
        )

    rho_ll, u_ll, p_ll, rho_species_ll = conservative_to_primitive_vars(eos, cv_ll)
    rho_rr, u_rr, p_rr, rho_species_rr = conservative_to_primitive_vars(eos, cv_rr)

    beta_ll = 0.5 * rho_ll / p_ll
    beta_rr = 0.5 * rho_rr / p_rr
    specific_kin_ll = 0.5 * sum(v**2 for v in u_ll)
    specific_kin_rr = 0.5 * sum(v**2 for v in u_rr)

    rho_avg = 0.5 * (rho_ll + rho_rr)
    rho_mean = ln_mean(rho_ll,  rho_rr)
    rho_species_mean = make_obj_array(
        [ln_mean(rho_ll_i, rho_rr_i)
         for rho_ll_i, rho_rr_i in zip(rho_species_ll, rho_species_rr)])

    beta_mean = ln_mean(beta_ll, beta_rr)
    beta_avg = 0.5 * (beta_ll + beta_rr)

    u_avg = 0.5 * (u_ll + u_rr)
    p_mean = 0.5 * rho_avg / beta_avg
    velocity_square_avg = specific_kin_ll + specific_kin_rr

    mass_flux = rho_mean * u_avg
    momentum_flux = np.outer(mass_flux, u_avg) + np.eye(dim) * p_mean
    energy_flux = (
        mass_flux * 0.5 * (
            1/(eos.gamma() - 1)/beta_mean - velocity_square_avg)
        + np.dot(momentum_flux, u_avg)
    )
    species_mass_flux = rho_species_mean.reshape(-1, 1) * u_avg

    return ConservedVars(
        mass=mass_flux,
        energy=energy_flux,
        momentum=momentum_flux,
        species_mass=species_mass_flux
    )


def entropy_stable_facial_flux(discr, gas_model, state_pair, local=False):
    r"""Return the entropy stable inviscid numerical flux.

    This facial flux routine is "entropy stable" in the sense that
    it computes the flux average component of the interface fluxes
    using an entropy conservative two-point flux
    (e.g. :func:`entropy_conserving_flux_chandrashekar`). Additional
    dissipation is imposed by penalizing the "jump" of the state across
    interfaces.

    Parameters
    ----------
    discr: :class:`~grudge.eager.EagerDGDiscretization`

        The discretization collection to use

    state_pair: :class:`~grudge.trace_pair.TracePair`

        Trace pair of :class:`~mirgecom.gas_model.FluidState` for the face upon
        which the flux calculation is to be performed

    local: bool

        Indicates whether to skip projection of fluxes to "all_faces" or not. If
        set to *False* (the default), the returned fluxes are projected to
        "all_faces."  If set to *True*, the returned fluxes are not projected to
        "all_faces"; remaining instead on the boundary restriction.

    Returns
    -------
    :class:`~mirgecom.fluid.ConservedVars`

        A CV object containing the scalar numerical fluxes at the input faces.
        The returned fluxes are scalar because they've already been dotted with
        the face normals.
    """
    actx = state_pair.int.array_context
    cv_ll = state_pair.int.cv
    cv_rr = state_pair.ext.cv
    flux = entropy_conserving_flux_chandrashekar(discr, gas_model.eos,
                                                 cv_ll, cv_rr)

    # FIXME: Need to refactor the numerical fluxes into the
    # "flux average" and "dissipation" components

    # This calculates the local maximum eigenvalue of the flux Jacobian
    # for a single component gas, i.e. the element-local max wavespeed |v| + c.
    lam = actx.np.maximum(state_pair.int.wavespeed, state_pair.ext.wavespeed)

    normal = thaw(actx, discr.normal(state_pair.dd))
    result = (flux - lam*outer(cv_rr - cv_ll, normal)/2) @ normal
    if local is False:
        dd = state_pair.dd
        return op.project(discr, dd, dd.with_dtag("all_faces"), result)
    return result


def get_inviscid_timestep(discr, state):
    """Return node-local stable timestep estimate for an inviscid fluid.

    The maximum stable timestep is computed from the acoustic wavespeed.

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization

        the discretization to use

    state: :class:`~mirgecom.gas_model.FluidState`

        Full fluid conserved and thermal state

    Returns
    -------
    class:`~meshmode.dof_array.DOFArray`

        The maximum stable timestep at each node.
    """
    from grudge.dt_utils import characteristic_lengthscales
    return (
        characteristic_lengthscales(state.array_context, discr)
        / state.wavespeed
    )


def get_inviscid_cfl(discr, state, dt):
    """Return node-local CFL based on current state and timestep.

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
    return dt / get_inviscid_timestep(discr, state=state)
