r""":mod:`mirgecom.navierstokes` methods and utils for compressible Navier-Stokes.

Compressible Navier-Stokes equations:

.. math::

    \partial_t \mathbf{Q} + \nabla\cdot\mathbf{F}_{I} = \nabla\cdot\mathbf{F}_{V}

where:

-  fluid state $\mathbf{Q} = [\rho, \rho{E}, \rho\mathbf{v}, \rho{Y}_\alpha]$
-  with fluid density $\rho$, flow energy $E$, velocity $\mathbf{v}$, and vector
   of species mass fractions ${Y}_\alpha$, where $1\le\alpha\le\mathtt{nspecies}$.
-  inviscid flux $\mathbf{F}_{I} = [\rho\mathbf{v},(\rho{E} + p)\mathbf{v}
   ,(\rho(\mathbf{v}\otimes\mathbf{v})+p\mathbf{I}), \rho{Y}_\alpha\mathbf{v}]$
-  viscous flux $\mathbf{F}_V = [0,((\tau\cdot\mathbf{v})-\mathbf{q}),\tau_{:i}
   ,J_{\alpha}]$
-  viscous stress tensor $\mathbf{\tau} = \mu(\nabla\mathbf{v}+(\nabla\mathbf{v})^T)
   + (\mu_B - \frac{2}{3}\mu)(\nabla\cdot\mathbf{v})$
-  diffusive flux for each species $J_\alpha = \rho{D}_{\alpha}\nabla{Y}_{\alpha}$
-  total heat flux $\mathbf{q}=\mathbf{q}_c+\mathbf{q}_d$, is the sum of:
    -  conductive heat flux $\mathbf{q}_c = -\kappa\nabla{T}$
    -  diffusive heat flux $\mathbf{q}_d = \sum{h_{\alpha} J_{\alpha}}$
-  fluid pressure $p$, temperature $T$, and species specific enthalpies $h_\alpha$
-  fluid viscosity $\mu$, bulk viscosity $\mu_{B}$, fluid heat conductivity $\kappa$,
   and species diffusivities $D_{\alpha}$.

RHS Evaluation
^^^^^^^^^^^^^^

.. autofunction:: ns_operator
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
from grudge.symbolic.primitives import TracePair
from grudge.trace_pair import interior_trace_pairs
from mirgecom.inviscid import (
    inviscid_flux,
    inviscid_facial_flux
)
from mirgecom.viscous import (
    viscous_flux,
    viscous_facial_flux
)
from mirgecom.flux import (
    gradient_flux_central
)
from mirgecom.fluid import make_conserved
from mirgecom.operators import (
    div_operator, grad_operator
)
from meshmode.dof_array import thaw


def ns_operator(discr, eos, boundaries, cv, t=0.0,
                dv=None):
    r"""Compute RHS of the Navier-Stokes equations.

    Returns
    -------
    numpy.ndarray
        The right-hand-side of the Navier-Stokes equations:

        .. math::

            \partial_t \mathbf{Q} = \nabla\cdot(\mathbf{F}_V - \mathbf{F}_I)

    Parameters
    ----------
    cv: :class:`~mirgecom.fluid.ConservedVars`
        Fluid solution

    boundaries
        Dictionary of boundary functions, one for each valid btag

    t
        Time

    eos: mirgecom.eos.GasEOS
        Implementing the pressure and temperature functions for
        returning pressure and temperature as a function of the state q.
        Implementing the transport properties including heat conductivity,
        and species diffusivities type(mirgecom.transport.TransportModel).

    Returns
    -------
    numpy.ndarray
        Agglomerated object array of DOF arrays representing the RHS of the
        Navier-Stokes equations.
    """
    dim = discr.dim
    actx = cv.array_context
    if dv is None:
        dv = eos.dependent_vars(cv)

    def _elbnd_flux(discr, compute_interior_flux, compute_boundary_flux,
                    int_tpairs, boundaries):
        return (sum(compute_interior_flux(part_tpair)
                    for part_tpair in int_tpairs)
                + sum(compute_boundary_flux(btag) for btag in boundaries))

    def grad_flux_interior(int_tpair):
        normal = thaw(actx, discr.normal(int_tpair.dd))
        # Hard-coding central per [Bassi_1997]_ eqn 13
        flux_weak = gradient_flux_central(int_tpair, normal)
        return discr.project(int_tpair.dd, "all_faces", flux_weak)

    def grad_flux_bnd(btag):
        return boundaries[btag].cv_gradient_flux(
            discr, btag=btag, cv=cv, eos=eos, time=t
        )

    cv_interior = interior_trace_pairs(discr, cv)
    cv_flux_bnd = _elbnd_flux(discr, grad_flux_interior, grad_flux_bnd,
                              cv_interior, boundaries)

    # [Bassi_1997]_ eqn 15 (s = grad_q)
    grad_cv = make_conserved(dim, q=grad_operator(discr, cv.join(),
                                                  cv_flux_bnd.join()))

    # Temperature gradient for conductive heat flux: [Ihme_2014]_ eqn (3b)
    # - now computed, *not* communicated
    def t_grad_flux_bnd(btag):
        return boundaries[btag].t_gradient_flux(discr, btag=btag, cv=cv, eos=eos,
                                                time=t)

    gas_t = dv.temperature
    temperature_interior = [TracePair(pair.dd,
                                      interior=eos.temperature(pair.int),
                                      exterior=eos.temperature(pair.ext))
                            for pair in cv_interior]

    t_flux_bnd = _elbnd_flux(discr, grad_flux_interior, t_grad_flux_bnd,
                             temperature_interior, boundaries)
    grad_t = grad_operator(discr, gas_t, t_flux_bnd)
    grad_t_interior = interior_trace_pairs(discr, grad_t)

    # inviscid parts
    def finv_divergence_flux_interior(cv_tpair):
        return inviscid_facial_flux(discr, eos=eos, cv_tpair=cv_tpair)

    # inviscid part of bcs applied here
    def finv_divergence_flux_boundary(btag):
        return boundaries[btag].inviscid_divergence_flux(
            discr, btag=btag, eos=eos, cv=cv, time=t
        )

    # viscous parts
    grad_cv_interior = interior_trace_pairs(discr, grad_cv)

    # glob the inputs together in a tuple to use the _elbnd_flux wrapper
    viscous_flux_inputs = [
        (cv_pair, grad_cv_pair, grad_t_pair)
        for cv_pair, grad_cv_pair, grad_t_pair in
        zip(cv_interior, grad_cv_interior, grad_t_interior)]

    # viscous fluxes across interior faces (including partition and periodic bnd)
    def fvisc_divergence_flux_interior(tpair_tuple):
        cv_pair_int = tpair_tuple[0]
        s_pair_int = tpair_tuple[1]
        dt_pair_int = tpair_tuple[2]
        return viscous_facial_flux(discr, eos, cv_pair_int, s_pair_int, dt_pair_int)

    # viscous part of bcs applied here
    def fvisc_divergence_flux_boundary(btag):
        return boundaries[btag].viscous_divergence_flux(discr, btag, eos=eos,
                                                        cv=cv, grad_cv=grad_cv,
                                                        grad_t=grad_t, time=t)

    vol_term = (
        viscous_flux(discr, eos=eos, cv=cv, grad_cv=grad_cv, grad_t=grad_t)
        - inviscid_flux(discr, pressure=dv.pressure, cv=cv)
    )

    bnd_term = (
        _elbnd_flux(
            discr, fvisc_divergence_flux_interior, fvisc_divergence_flux_boundary,
            viscous_flux_inputs, boundaries)
        - _elbnd_flux(
            discr, finv_divergence_flux_interior, finv_divergence_flux_boundary,
            cv_interior, boundaries)
    )
    # NS RHS
    return div_operator(discr, vol_term, bnd_term)
