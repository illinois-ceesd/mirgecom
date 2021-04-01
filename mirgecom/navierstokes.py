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
from grudge.eager import (
    interior_trace_pair,
    cross_rank_trace_pairs
)
from mirgecom.inviscid import (
    inviscid_flux,
    interior_inviscid_flux
)
from mirgecom.viscous import (
    viscous_flux,
    interior_viscous_flux
)
from mirgecom.flux import (
    central_scalar_flux
)
from mirgecom.fluid import split_conserved
from mirgecom.operators import (
    dg_grad,
    element_boundary_flux
)
from meshmode.dof_array import thaw


def ns_operator(discr, eos, boundaries, q, t=0.0):
    r"""Compute RHS of the Navier-Stokes equations.

    Returns
    -------
    numpy.ndarray
        The right-hand-side of the Navier-Stokes equations:

        .. math::

            \partial_t \mathbf{Q} = \nabla\cdot(\mathbf{F}_V - \mathbf{F}_I)

    Parameters
    ----------
    q
        State array which expects at least the canonical conserved quantities
        (mass, energy, momentum) for the fluid at each point. For multi-component
        fluids, the conserved quantities should include
        (mass, energy, momentum, species_mass), where *species_mass* is a vector
        of species masses.

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
    cv = split_conserved(dim, q)
    actx = cv.mass.array_context

    def scalar_flux_interior(int_tpair):
        normal = thaw(actx, discr.normal(int_tpair.dd))
        # Hard-coding central per [Bassi_1997]_ eqn 13
        flux_weak = central_scalar_flux(int_tpair, normal)
        return discr.project(int_tpair.dd, "all_faces", flux_weak)

    def q_flux_bnd(btag):
        return boundaries[btag].q_flux(discr, btag, q)

    # [Bassi_1997]_ eqn 15 (s = grad_q)
    grad_q = dg_grad(discr, scalar_flux_interior, q_flux_bnd, boundaries, q)

    gas_t = eos.temperature(cv)

    def t_flux_bnd(btag):
        return boundaries[btag].t_flux(discr, btag, eos=eos, time=t, q=q, t=gas_t)

    # Temperature gradient for conductive heat flux: [Ihme_2014]_ eqn (3b)
    grad_t = dg_grad(discr, scalar_flux_interior, t_flux_bnd, boundaries, gas_t)

    # inviscid part
    def compute_inviscid_flux_interior(q_tpair):
        return interior_inviscid_flux(discr, eos, q_tpair)

    def compute_inviscid_flux_boundary(btag):
        return boundaries[btag].inviscid_flux(discr, btag, eos=eos, t=t, q=q)

    # inviscid boundary flux
    inv_flux_bnd = element_boundary_flux(discr, compute_inviscid_flux_interior,
                                         compute_inviscid_flux_boundary,
                                         boundaries, q)

    # Things are more complicated here because of multiple arguments to flux
    # functions - some of which need communication.
    # viscous boundary
    # what is the role in initiating comm?
    q_int_pair = interior_trace_pair(discr, q)
    q_part_pairs = cross_rank_trace_pairs(discr, q)
    t_int_pair = interior_trace_pair(discr, gas_t)
    t_part_pairs = cross_rank_trace_pairs(discr, gas_t)
    s_int_pair = interior_trace_pair(discr, grad_q)
    s_part_pairs = cross_rank_trace_pairs(discr, grad_q)
    delt_int_pair = interior_trace_pair(discr, grad_t)
    delt_part_pairs = cross_rank_trace_pairs(discr, grad_t)
    num_partition_interfaces = len(q_part_pairs)

    # - internal boundaries
    visc_flux_bnd = interior_viscous_flux(discr, eos, q_int_pair,
                                          s_int_pair, t_int_pair, delt_int_pair)

    for bnd_index in range(num_partition_interfaces):
        visc_flux_bnd += interior_viscous_flux(discr, eos,
                                               q_part_pairs[bnd_index],
                                               s_part_pairs[bnd_index],
                                               t_part_pairs[bnd_index],
                                               delt_part_pairs[bnd_index])

    # - domain boundaries (viscous bc's applied here)
    visc_flux_bnd += sum(bnd.viscous_flux(discr, btag, eos=eos, time=t, q=q,
                                          grad_q=grad_q, t=gas_t, grad_t=grad_t)
                         for btag, bnd in boundaries.items())

    # NS RHS
    return discr.inverse_mass(
        discr.weak_div(inviscid_flux(discr, eos, q)
                       + viscous_flux(discr, eos, q=q, grad_q=grad_q,
                                      t=gas_t, grad_t=grad_t))
        - discr.face_mass(inv_flux_bnd + visc_flux_bnd)
    )
