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

Operator Boundary Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: EulerBoundaryInterface

Logging Helpers
^^^^^^^^^^^^^^^

.. autofunction:: units_for_logging
.. autofunction:: extract_vars_for_logging
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
from mirgecom.fluid import split_conserved
from mirgecom.inviscid import (
    inviscid_flux,
    inviscid_facial_flux,
    InviscidBoundaryInterface
)
from grudge.eager import (
    interior_trace_pair,
    cross_rank_trace_pairs
)
from mirgecom.operators import dg_div_low as dg_div


class EulerBoundaryInterface(InviscidBoundaryInterface):
    """Interface for boundary treatment of the Euler operator."""

    pass


def euler_operator(discr, eos, boundaries, q, t=0.0):
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

    Returns
    -------
    numpy.ndarray
        Agglomerated object array of DOF arrays representing the RHS of the Euler
        flow equations.
    """
    inviscid_flux_vol = inviscid_flux(discr, eos, q)
    inviscid_flux_bnd = (
        inviscid_facial_flux(discr, eos=eos, q_tpair=interior_trace_pair(discr, q))
        + sum(inviscid_facial_flux(discr, eos=eos, q_tpair=part_tpair)
              for part_tpair in cross_rank_trace_pairs(discr, q))
        + sum(boundaries[btag].get_inviscid_flux(discr, btag=btag, q=q, eos=eos)
              for btag in boundaries)
    )

    return -dg_div(discr, inviscid_flux_vol, inviscid_flux_bnd)


def inviscid_operator(discr, eos, boundaries, q, t=0.0):
    """Interface :function:`euler_operator` with backwards-compatible API."""
    from warnings import warn
    warn("Do not call inviscid_operator; it is now called euler_operator. This"
         "function will disappear August 1, 2021", DeprecationWarning, stacklevel=2)
    return euler_operator(discr, eos, boundaries, q, t)


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
