r""":mod:`mirgecom.inviscid` provides helper functions for inviscid flow.

Inviscid Flux Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: inviscid_flux
.. autofunction:: inviscid_facial_flux

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
from meshmode.dof_array import thaw
from mirgecom.fluid import compute_wavespeed
from grudge.trace_pair import TracePair
from mirgecom.flux import divergence_flux_lfr
from mirgecom.fluid import make_conserved


def inviscid_flux(discr, eos, cv):
    r"""Compute the inviscid flux vectors from fluid conserved vars *cv*.

    The inviscid fluxes are
    $(\rho\vec{V},(\rho{E}+p)\vec{V},\rho(\vec{V}\otimes\vec{V})
    +p\mathbf{I}, \rho{Y_s}\vec{V})$

    .. note::

        The fluxes are returned as a :class:`mirgecom.fluid.ConservedVars`
        object with a *dim-vector* for each conservation equation. See
        :class:`mirgecom.fluid.ConservedVars` for more information about
        how the fluxes are represented.
    """
    dim = cv.dim
    p = eos.pressure(cv)

    mom = cv.momentum

    return make_conserved(
        dim, mass=mom, energy=mom * (cv.energy + p) / cv.mass,
        momentum=np.outer(mom, mom) / cv.mass + np.eye(dim)*p,
        species_mass=(  # reshaped: (nspecies, dim)
            (mom / cv.mass) * cv.species_mass.reshape(-1, 1)))


def inviscid_facial_flux(discr, eos, cv_tpair, local=False):
    r"""Return the flux across a face given the solution on both sides *q_tpair*.

    This flux is currently hard-coded to use a Rusanov-type  local Lax-Friedrichs
    (LFR) numerical flux at element boundaries. The numerical inviscid flux $F^*$ is
    calculated as:

    .. math::

        \mathbf{F}^{*}_{\mathtt{LFR}} = \frac{1}{2}(\mathbf{F}(q^-)
        +\mathbf{F}(q^+)) \cdot \hat{n} + \frac{\lambda}{2}(q^{-} - q^{+}),

    where $q^-, q^+$ are the fluid solution state on the interior and the
    exterior of the face on which the LFR flux is to be calculated, $\mathbf{F}$ is
    the inviscid fluid flux, $\hat{n}$ is the face normal, and $\lambda$ is the
    *local* maximum fluid wavespeed.

    Parameters
    ----------
    eos: mirgecom.eos.GasEOS
        Implementing the pressure and temperature functions for
        returning pressure and temperature as a function of the state q.

    q_tpair: :class:`grudge.trace_pair.TracePair`
        Trace pair for the face upon which flux calculation is to be performed

    local: bool
        Indicates whether to skip projection of fluxes to "all_faces" or not. If
        set to *False* (the default), the returned fluxes are projected to
        "all_faces."  If set to *True*, the returned fluxes are not projected to
        "all_faces"; remaining instead on the boundary restriction.
    """
    actx = cv_tpair.int.array_context

    flux_tpair = TracePair(cv_tpair.dd,
                           interior=inviscid_flux(discr, eos, cv_tpair.int),
                           exterior=inviscid_flux(discr, eos, cv_tpair.ext))

    # This calculates the local maximum eigenvalue of the flux Jacobian
    # for a single component gas, i.e. the element-local max wavespeed |v| + c.
    lam = actx.np.maximum(
        compute_wavespeed(eos=eos, cv=cv_tpair.int),
        compute_wavespeed(eos=eos, cv=cv_tpair.ext)
    )

    normal = thaw(actx, discr.normal(cv_tpair.dd))

    # todo: user-supplied flux routine
    flux_weak = divergence_flux_lfr(cv_tpair, flux_tpair, normal=normal, lam=lam)

    if local is False:
        return discr.project(cv_tpair.dd, "all_faces", flux_weak)

    return flux_weak


def get_inviscid_timestep(discr, eos, cv):
    """Return node-local stable timestep estimate for an inviscid fluid.

    The maximum stable timestep is computed from the acoustic wavespeed.

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization
        the discretization to use
    eos: mirgecom.eos.GasEOS
        Implementing the pressure and temperature functions for
        returning pressure and temperature as a function of the state q.
    cv: :class:`~mirgecom.fluid.ConservedVars`
        Fluid solution
    Returns
    -------
    class:`~meshmode.dof_array.DOFArray`
        The maximum stable timestep at each node.
    """
    from grudge.dt_utils import characteristic_lengthscales
    from mirgecom.fluid import compute_wavespeed
    return (
        characteristic_lengthscales(cv.array_context, discr)
        / compute_wavespeed(eos, cv)
    )


def get_inviscid_cfl(discr, eos, dt, cv):
    """Return node-local CFL based on current state and timestep.

    Parameters
    ----------
    discr: :class:`grudge.eager.EagerDGDiscretization`
        the discretization to use
    eos: mirgecom.eos.GasEOS
        Implementing the pressure and temperature functions for
        returning pressure and temperature as a function of the state q.
    dt: float or :class:`~meshmode.dof_array.DOFArray`
        A constant scalar dt or node-local dt
    cv: :class:`~mirgecom.fluid.ConservedVars`
        The fluid conserved variables

    Returns
    -------
    :class:`meshmode.dof_array.DOFArray`
        The CFL at each node.
    """
    return dt / get_inviscid_timestep(discr, eos=eos, cv=cv)
