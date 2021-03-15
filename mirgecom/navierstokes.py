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
-  viscous flux $\mathbf{F}_V = [0,((\mathbf{v}\cdot\tau)-\mathbf{q}),\tau_{:i}
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
from grudge.eager import (
    interior_trace_pair,
    cross_rank_trace_pairs
)
from mirgecom.fluid import (
    compute_wavespeed,
    split_conserved,
)
from mirgecom.inviscid import (
    inviscid_flux
)
from functools import partial
from mirgecom.flux import lfr_flux


def ns_operator(discr, eos, tv_model, boundaries, q, t=0.0):
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

    tv_model: mirgecom.transport.TransportModel
        Implementing the transport properties including heat conductivity,
        and species diffusivities.

    Returns
    -------
    numpy.ndarray
        Agglomerated object array of DOF arrays representing the RHS of the
        Navier-Stokes equations.
    """
    from mirgecom.inviscid import inviscid_flux
    inviscid_flux_vol = inviscid_flux(discr, eos, q)

    from mirgecom.viscous import (
        viscous_stress_tensor,
        diffusive_flux,
        convective_heat_flux,
        diffusive_heat_flux,
        viscous_flux,
    )

    from mirgecom.transport import (
        fluid_viscosity
        heat_conductivity,
        species_diffusvity
    )

    tau = viscous_stress_tensor()
    kappa = heat_conductivity() 
    j = diffusive_flux()
    q = convective_heat_flux(q, kappa) +  diffusive_heat_flux(q, j)
    viscous_flux_vol = viscous_flux(discr, eos=eos, tv_model=tv_model,
                                    tau=tau, q=q, j=j)

    vol_flux = inviscid_flux_vol - viscous_flux_vol
    dflux = discr.weak_div(vol_flux)

    interior_face_flux = _facial_flux(
        discr, eos=eos, q_tpair=interior_trace_pair(discr, q))

    # Domain boundaries
    domain_boundary_flux = sum(
        _facial_flux(
            discr,
            q_tpair=boundaries[btag].boundary_pair(discr,
                                                   eos=eos,
                                                   btag=btag,
                                                   t=t,
                                                   q=q),
            eos=eos
        )
        for btag in boundaries
    )

    # Flux across partition boundaries
    partition_boundary_flux = sum(
        _facial_flux(discr, eos=eos, q_tpair=part_pair)
        for part_pair in cross_rank_trace_pairs(discr, q)
    )

    return discr.inverse_mass(
        dflux - discr.face_mass(interior_face_flux + domain_boundary_flux
                                + partition_boundary_flux)
    )
