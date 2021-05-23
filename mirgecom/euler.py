r""":mod:`mirgecom.euler` helps solve Euler's equations of gas dynamics.

Euler's equations of gas dynamics:

.. math::

    \partial_t \mathbf{Q} = -\nabla\cdot{\mathbf{F}} +
    (\mathbf{F}\cdot\hat{n})_{\partial\Omega}

where:

-  state $\mathbf{Q} = [\rho, \rho{E}, \rho\vec{V}, \rho{Y}_\alpha]$
-  flux $\mathbf{F} = [\rho\vec{V},(\rho{E} + p)\vec{V},
   (\rho(\vec{V}\otimes\vec{V}) + p*\mathbf{I}), \rho{Y}_\alpha\vec{V}]$,
-  unit normal $\hat{n}$ to the domain boundary $\partial\Omega$,
-  vector of species mass fractions ${Y}_\alpha$,
   with $1\le\alpha\le\mathtt{nspecies}$.

RHS Evaluation
^^^^^^^^^^^^^^

.. autofunction:: euler_operator

Logging Helpers
^^^^^^^^^^^^^^^

.. autofunction:: units_for_logging
.. autofunction:: extract_vars_for_logging
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

from arraycontext.container.traversal import thaw

import grudge.op as op

from mirgecom.fluid import (
    compute_wavespeed,
    split_conserved,
)
from mirgecom.flux import lfr_flux
from mirgecom.inviscid import inviscid_flux

from functools import partial


def _facial_flux(dcoll, eos, q_tpair, local=False):
    """Return the flux across a face given the solution on both sides *q_tpair*.

    Parameters
    ----------
    dcoll: :class:`grudge.discretization.DiscretizationCollection`
        An object containing connections and mappings to different
        discretizations over the mesh.

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
    actx = q_tpair[0].int.array_context
    dim = dcoll.dim

    euler_flux = partial(inviscid_flux, dcoll, eos)
    lam = actx.np.maximum(
        compute_wavespeed(dim, eos=eos, cv=split_conserved(dim, q_tpair.int)),
        compute_wavespeed(dim, eos=eos, cv=split_conserved(dim, q_tpair.ext))
    )
    normal = thaw(op.normal(dcoll, q_tpair.dd), actx)

    # todo: user-supplied flux routine
    flux_weak = lfr_flux(q_tpair, flux_func=euler_flux, normal=normal, lam=lam)

    if local is False:
        return op.project(dcoll, q_tpair.dd, "all_faces", flux_weak)
    return flux_weak


def euler_operator(dcoll, eos, boundaries, q, t=0.0):
    r"""Compute RHS of the Euler flow equations.

    Returns
    -------
    numpy.ndarray
        The right-hand-side of the Euler flow equations:

        .. math::

            \dot{\mathbf{q}} = - \nabla\cdot\mathbf{F} +
                (\mathbf{F}\cdot\hat{n})_{\partial\Omega}

    Parameters
    ----------
    dcoll: :class:`grudge.discretization.DiscretizationCollection`
        An object containing connections and mappings to different
        discretizations over the mesh.

    eos: mirgecom.eos.GasEOS
        Implementing the pressure and temperature functions for
        returning pressure and temperature as a function of the state q.

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

    Returns
    -------
    numpy.ndarray
        Agglomerated object array of DOF arrays representing the RHS of the Euler
        flow equations.
    """
    interior_face_flux = _facial_flux(
        dcoll, eos=eos,
        q_tpair=op.interior_trace_pair(dcoll, q)
    )

    # Domain boundaries
    domain_boundary_flux = sum(
        _facial_flux(
            dcoll,
            eos=eos,
            q_tpair=boundaries[btag].boundary_pair(dcoll,
                                                   eos=eos,
                                                   btag=btag,
                                                   t=t,
                                                   q=q),
        )
        for btag in boundaries
    )

    # Flux across partition boundaries
    partition_boundary_flux = sum(
        _facial_flux(dcoll, eos=eos, q_tpair=part_pair)
        for part_pair in op.cross_rank_trace_pairs(dcoll, q)
    )

    return op.inverse_mass(
        dcoll,
        op.weak_local_div(dcoll, inviscid_flux(dcoll, eos, q))
        - op.face_mass(
            dcoll,
            interior_face_flux + domain_boundary_flux
            + partition_boundary_flux
        )
    )


def inviscid_operator(dcoll, eos, boundaries, q, t=0.0):
    """Interface :function:`euler_operator` with backwards-compatible API."""
    from warnings import warn
    warn("Do not call inviscid_operator; it is now called euler_operator. This"
         "function will disappear August 1, 2021", DeprecationWarning, stacklevel=2)
    return euler_operator(dcoll, eos, boundaries, q, t)


# By default, run unitless
NAME_TO_UNITS = {
    "mass": "",
    "energy": "",
    "momentum": "",
    "temperature": "",
    "pressure": ""
}


def units_for_logging(quantity: str) -> str:
    """Return unit for quantity."""
    return NAME_TO_UNITS[quantity]


def extract_vars_for_logging(dim: int, state: np.ndarray, eos) -> dict:
    """Extract state vars."""
    cv = split_conserved(dim, state)
    dv = eos.dependent_vars(cv)

    from mirgecom.utils import asdict_shallow
    name_to_field = asdict_shallow(cv)
    name_to_field.update(asdict_shallow(dv))
    return name_to_field
