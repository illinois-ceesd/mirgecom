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
    central_scalar_flux
)
from mirgecom.fluid import (
    split_conserved,
    join_conserved_vectors
)
from mirgecom.operators import (
    elbnd_flux,
    dg_div_low,
    dg_grad_low
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

    def get_q_flux_bnd(btag):
        return boundaries[btag].get_q_flux(discr, btag, eos, q, time=t)

    q_int_tpair = interior_trace_pair(discr, q)
    q_part_pairs = cross_rank_trace_pairs(discr, q)
    q_flux_bnd = elbnd_flux(discr, scalar_flux_interior, get_q_flux_bnd,
                            q_int_tpair, q_part_pairs, boundaries)

    # [Bassi_1997]_ eqn 15 (s = grad_q)
    grad_q = join_conserved_vectors(dim, dg_grad_low(discr, q, q_flux_bnd))

    # Temperature gradient for conductive heat flux: [Ihme_2014]_ eqn (3b)
    # - now computed, *not* communicated
    def get_t_flux_bnd(btag):
        return boundaries[btag].get_t_flux(discr, btag, eos, q, time=t)
    gas_t = eos.temperature(cv)
    t_int_tpair = TracePair("int_faces",
                            interior=eos.temperature(
                                split_conserved(dim, q_int_tpair.int)),
                            exterior=eos.temperature(
                                split_conserved(dim, q_int_tpair.ext)))
    t_part_pairs = [
        TracePair(part_tpair.dd,
                  interior=eos.temperature(split_conserved(dim, part_tpair.int)),
                  exterior=eos.temperature(split_conserved(dim, part_tpair.ext)))
        for part_tpair in q_part_pairs]
    t_flux_bnd = elbnd_flux(discr, scalar_flux_interior, get_t_flux_bnd,
                            t_int_tpair, t_part_pairs, boundaries)
    grad_t = dg_grad_low(discr, gas_t, t_flux_bnd)

    # inviscid parts
    def finv_interior_face(q_tpair):
        return inviscid_facial_flux(discr, eos, q_tpair)

    # inviscid part of bcs applied here
    def finv_domain_boundary(btag):
        return boundaries[btag].get_inviscid_flux(discr, btag, eos=eos, q=q,
                                                  time=t)

    # viscous parts
    s_int_pair = interior_trace_pair(discr, grad_q)
    s_part_pairs = cross_rank_trace_pairs(discr, grad_q)
    delt_int_pair = interior_trace_pair(discr, grad_t)
    delt_part_pairs = cross_rank_trace_pairs(discr, grad_t)
    num_partition_interfaces = len(q_part_pairs)

    # glob the inputs together in a tuple to use the elbnd_flux wrapper
    visc_part_inputs = [
        (q_part_pairs[bnd_index], s_part_pairs[bnd_index],
         t_part_pairs[bnd_index], delt_part_pairs[bnd_index])
        for bnd_index in range(num_partition_interfaces)]

    # viscous fluxes across interior faces (including partition and periodic bnd)
    def fvisc_interior_face(tpair_tuple):
        qpair_int = tpair_tuple[0]
        spair_int = tpair_tuple[1]
        tpair_int = tpair_tuple[2]
        dtpair_int = tpair_tuple[3]
        return viscous_facial_flux(discr, eos, qpair_int, spair_int,
                                   tpair_int, dtpair_int)

    # viscous part of bcs applied here
    def visc_bnd_flux(btag):
        return boundaries[btag].get_viscous_flux(discr, btag, eos=eos,
                                                 q=q, grad_q=grad_q,
                                                 grad_t=grad_t, time=t)

    # NS RHS
    return dg_div_low(
        discr, (  # volume part
            viscous_flux(discr, eos, q=q, grad_q=grad_q, t=gas_t, grad_t=grad_t)
            - inviscid_flux(discr, eos, q)),
        elbnd_flux(  # viscous boundary
            discr, fvisc_interior_face, visc_bnd_flux,
            (q_int_tpair, s_int_pair, t_int_tpair, delt_int_pair),
            visc_part_inputs, boundaries)
        - elbnd_flux(  # inviscid boundary
            discr, finv_interior_face, finv_domain_boundary,
            q_int_tpair, q_part_pairs, boundaries)
    )
