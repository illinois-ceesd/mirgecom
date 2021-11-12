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
from grudge.eager import (
    interior_trace_pair,
    cross_rank_trace_pairs
)
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
                    int_tpair, xrank_pairs, boundaries):
        return (compute_interior_flux(int_tpair)
                + sum(compute_interior_flux(part_tpair)
                      for part_tpair in xrank_pairs)
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

    cv_int_tpair = interior_trace_pair(discr, cv)
    cv_part_pairs = [
        TracePair(part_tpair.dd,
                  interior=part_tpair.int,
                  exterior=part_tpair.ext)
        for part_tpair in cross_rank_trace_pairs(discr, cv)]
    cv_flux_bnd = _elbnd_flux(discr, grad_flux_interior, grad_flux_bnd,
                              cv_int_tpair, cv_part_pairs, boundaries)

    # [Bassi_1997]_ eqn 15 (s = grad_q)
    grad_cv = grad_operator(discr, cv, cv_flux_bnd)

    # Temperature gradient for conductive heat flux: [Ihme_2014]_ eqn (3b)
    # - now computed, *not* communicated
    def t_grad_flux_bnd(btag):
        return boundaries[btag].t_gradient_flux(discr, btag=btag, cv=cv, eos=eos,
                                                time=t)

    gas_t = dv.temperature
    t_int_tpair = TracePair("int_faces",
                            interior=eos.temperature(cv_int_tpair.int),
                            exterior=eos.temperature(cv_int_tpair.ext))
    t_part_pairs = [
        TracePair(part_tpair.dd,
                  interior=eos.temperature(part_tpair.int),
                  exterior=eos.temperature(part_tpair.ext))
        for part_tpair in cv_part_pairs]
    t_flux_bnd = _elbnd_flux(discr, grad_flux_interior, t_grad_flux_bnd,
                             t_int_tpair, t_part_pairs, boundaries)
    grad_t = grad_operator(discr, gas_t, t_flux_bnd)

    # inviscid parts
    def finv_divergence_flux_interior(cv_tpair):
        return inviscid_facial_flux(discr, eos=eos, cv_tpair=cv_tpair)

    # inviscid part of bcs applied here
    def finv_divergence_flux_boundary(btag):
        return boundaries[btag].inviscid_divergence_flux(
            discr, btag=btag, eos=eos, cv=cv, time=t
        )

    # viscous parts
    s_int_pair = interior_trace_pair(discr, grad_cv)
    s_part_pairs = [TracePair(xrank_tpair.dd,
                             interior=xrank_tpair.int,
                             exterior=xrank_tpair.ext)
                    for xrank_tpair in cross_rank_trace_pairs(discr, grad_cv)]
    delt_int_pair = interior_trace_pair(discr, grad_t)
    delt_part_pairs = cross_rank_trace_pairs(discr, grad_t)
    num_partition_interfaces = len(cv_part_pairs)

    # glob the inputs together in a tuple to use the _elbnd_flux wrapper
    visc_part_inputs = [
        (cv_part_pairs[bnd_index], s_part_pairs[bnd_index],
         delt_part_pairs[bnd_index])
        for bnd_index in range(num_partition_interfaces)]

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
            (cv_int_tpair, s_int_pair, delt_int_pair), visc_part_inputs, boundaries)
        - _elbnd_flux(
            discr, finv_divergence_flux_interior, finv_divergence_flux_boundary,
            cv_int_tpair, cv_part_pairs, boundaries)
    )

    # NS RHS
    return div_operator(discr, vol_term, bnd_term)
