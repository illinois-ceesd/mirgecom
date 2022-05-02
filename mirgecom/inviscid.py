r""":mod:`mirgecom.inviscid` provides helper functions for inviscid flow.

Inviscid Flux Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: inviscid_flux
.. autofunction:: inviscid_facial_flux_rusanov
.. autofunction:: inviscid_facial_flux_hll
.. autofunction:: inviscid_facial_flux
.. autofunction:: inviscid_flux_on_element_boundary

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
from arraycontext import thaw
from mirgecom.fluid import make_conserved


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
    species_mass_flux = \
        state.velocity*state.species_mass_density.reshape(-1, 1)

    return make_conserved(state.dim, mass=mass_flux, energy=energy_flux,
                          momentum=mom_flux, species_mass=species_mass_flux)


def inviscid_facial_flux_rusanov(state_pair, gas_model, normal, **kwargs):
    r"""High-level interface for inviscid facial flux using Rusanov numerical flux.

    The Rusanov or Local Lax-Friedrichs (LLF) inviscid numerical flux is calculated
    as:

    .. math::

        F^{*}_{\mathtt{Rusanov}} = \frac{1}{2}(\mathbf{F}(q^-)
        +\mathbf{F}(q^+)) \cdot \hat{n} + \frac{\lambda}{2}(q^{-} - q^{+}),

    where $q^-, q^+$ are the fluid solution state on the interior and the
    exterior of the face where the Rusanov flux is to be calculated, $\mathbf{F}$ is
    the inviscid fluid flux, $\hat{n}$ is the face normal, and $\lambda$ is the
    *local* maximum fluid wavespeed.

    Parameters
    ----------
    state_pair: :class:`~grudge.trace_pair.TracePair`

        Trace pair of :class:`~mirgecom.gas_model.FluidState` for the face upon
        which the flux calculation is to be performed

    gas_model: :class:`~mirgecom.gas_model.GasModel`

        Physical gas model including equation of state, transport,
        and kinetic properties as required by fluid state

    normal: numpy.ndarray

        The element interface normals

    Returns
    -------
    :class:`~mirgecom.fluid.ConservedVars`

        A CV object containing the scalar numerical fluxes at the input faces.
        The returned fluxes are scalar because they've already been dotted with
        the face normals as required by the divergence operator for which they
        are being computed.
    """
    actx = state_pair.int.array_context
    lam = actx.np.maximum(state_pair.int.wavespeed, state_pair.ext.wavespeed)
    from mirgecom.flux import num_flux_lfr
    return num_flux_lfr(f_minus_normal=inviscid_flux(state_pair.int)@normal,
                        f_plus_normal=inviscid_flux(state_pair.ext)@normal,
                        q_minus=state_pair.int.cv,
                        q_plus=state_pair.ext.cv, lam=lam)


def inviscid_facial_flux_hll(state_pair, gas_model, normal, **kwargs):
    r"""High-level interface for inviscid facial flux using HLL numerical flux.

    The Harten, Lax, van Leer approximate riemann numerical flux is calculated as:

    .. math::

        f^{*}_{\mathtt{HLL}} = \frac{\left(s^+f^--s^-f^++s^+s^-\left(q^+-q^-\right)
        \right)}{\left(s^+ - s^-\right)}

    where $f^{\mp}$, $q^{\mp}$, and $s^{\mp}$ are the interface-normal fluxes, the
    states, and the wavespeeds for the interior (-) and exterior (+) of the
    interface respectively.

    Details about how the parameters and fluxes are calculated can be found in
    Section 10.3 of [Toro_2009]_.

    Parameters
    ----------
    state_pair: :class:`~grudge.trace_pair.TracePair`

        Trace pair of :class:`~mirgecom.gas_model.FluidState` for the face upon
        which the flux calculation is to be performed

    gas_model: :class:`~mirgecom.gas_model.GasModel`

        Physical gas model including equation of state, transport,
        and kinetic properties as required by fluid state

    normal: numpy.ndarray

        The element interface normals

    Returns
    -------
    :class:`~mirgecom.fluid.ConservedVars`

        A CV object containing the scalar numerical fluxes at the input faces.
        The returned fluxes are scalar because they've already been dotted with
        the face normals as required by the divergence operator for which they
        are being computed.
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

    gamma_int = gas_model.eos.gamma(state_pair.int.cv, state_pair.int.temperature)
    gamma_ext = gas_model.eos.gamma(state_pair.ext.cv, state_pair.ext.temperature)

    q_int = 1 + (gamma_int + 1)/(2*gamma_int)*(p_star/p_int - 1)
    q_ext = 1 + (gamma_ext + 1)/(2*gamma_ext)*(p_star/p_ext - 1)

    pres_check_int = actx.np.greater(p_star, p_int)
    pres_check_ext = actx.np.greater(p_star, p_ext)

    q_int_rt = actx.np.sqrt(actx.np.where(pres_check_int, q_int, ones))
    q_ext_rt = actx.np.sqrt(actx.np.where(pres_check_ext, q_ext, ones))

    # left (internal), and right (external) wave speed estimates
    # can alternatively use the roe estimated states to find the wave speeds
    s_minus = u_int - c_int*q_int_rt
    s_plus = u_ext + c_ext*q_ext_rt

    f_minus_normal = inviscid_flux(state_pair.int)@normal
    f_plus_normal = inviscid_flux(state_pair.ext)@normal

    q_minus = state_pair.int.cv
    q_plus = state_pair.ext.cv

    from mirgecom.flux import num_flux_hll
    return num_flux_hll(f_minus_normal, f_plus_normal, q_minus, q_plus, s_minus,
                        s_plus)


def inviscid_facial_flux(discr, gas_model, state_pair,
                         numerical_flux_func=inviscid_facial_flux_rusanov,
                         local=False, **kwargs):
    r"""Return the numerical inviscid flux for the divergence operator.

    Different numerical fluxes may be used through the specificiation of
    the *numerical_flux_func*. By default, a Rusanov-type flux is used.

    Parameters
    ----------
    discr: :class:`~grudge.eager.EagerDGDiscretization`

        The discretization collection to use

    gas_model: :class:`~mirgecom.gas_model.GasModel`

        Physical gas model including equation of state, transport,
        and kinetic properties as required by fluid state

    state_pair: :class:`~grudge.trace_pair.TracePair`

        Trace pair of :class:`~mirgecom.gas_model.FluidState` for the face upon
        which the flux calculation is to be performed

    numerical_flux_func:

        Callable that returns the interface-normal numerical flux

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
    num_flux = numerical_flux_func(state_pair, gas_model, normal, **kwargs)
    dd = state_pair.dd
    dd_allfaces = dd.with_dtag("all_faces")
    return num_flux if local else discr.project(dd, dd_allfaces, num_flux)


def inviscid_flux_on_element_boundary(
        discr, gas_model, boundaries, interior_state_pairs,
        domain_boundary_states, quadrature_tag=None,
        numerical_flux_func=inviscid_facial_flux_rusanov, time=0.0):
    """Compute the inviscid boundary fluxes for the divergence operator.

    This routine encapsulates the computation of the inviscid contributions
    to the boundary fluxes for use by the divergence operator. Its existence
    is intended to allow multiple operators (e.g. Euler and Navier-Stokes) to
    perform the computation without duplicating code.

    Parameters
    ----------
    discr: :class:`~grudge.eager.EagerDGDiscretization`
        A discretization collection encapsulating the DG elements

    gas_model: :class:`~mirgecom.gas_model.GasModel`
        The physical model constructs for the gas_model

    boundaries
        Dictionary of boundary functions, one for each valid btag

    interior_state_pairs
        A :class:`~mirgecom.gas_model.FluidState` TracePair for each internal face.

    domain_boundary_states
       A dictionary of boundary-restricted :class:`~mirgecom.gas_model.FluidState`,
       keyed by btags in *boundaries*.

    quadrature_tag
        An optional identifier denoting a particular quadrature
        discretization to use during operator evaluations.
        The default value is *None*.

    numerical_flux_func
        The numerical flux function to use in computing the boundary flux.

    time: float
        Time
    """
    from grudge.dof_desc import as_dofdesc

    # Compute interface contributions
    inviscid_flux_bnd = (

        # Interior faces
        sum(inviscid_facial_flux(discr, gas_model, state_pair,
                                 numerical_flux_func)
            for state_pair in interior_state_pairs)

        # Domain boundary faces
        + sum(
            boundaries[btag].inviscid_divergence_flux(
                discr, as_dofdesc(btag).with_discr_tag(quadrature_tag), gas_model,
                state_minus=domain_boundary_states[btag],
                numerical_flux_func=numerical_flux_func, time=time)
            for btag in boundaries)
    )

    return inviscid_flux_bnd


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
