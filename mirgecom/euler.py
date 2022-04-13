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

from meshmode.discretization.connection import FACE_RESTR_ALL
from grudge.dof_desc import DD_VOLUME_ALL, DISCR_TAG_BASE

from mirgecom.gas_model import make_operator_fluid_states
from mirgecom.inviscid import (
    inviscid_flux,
    inviscid_flux_rusanov,
    inviscid_boundary_flux_for_divergence_operator
)
from mirgecom.operators import div_operator


def euler_operator(discr, state, gas_model, boundaries, time=0.0,
                   inviscid_numerical_flux_func=inviscid_flux_rusanov,
                   quadrature_tag=DISCR_TAG_BASE, volume_dd=DD_VOLUME_ALL):
    r"""Compute RHS of the Euler flow equations.

    Parameters
    ----------
    state: :class:`~mirgecom.gas_model.FluidState`

        Fluid state object with the conserved state, and dependent
        quantities.

    boundaries

        Dictionary of boundary functions, one for each valid
        :class:`~grudge.dof_desc.BoundaryDomainTag`

    time

        Time

    gas_model: :class:`~mirgecom.gas_model.GasModel`

        Physical gas model including equation of state, transport,
        and kinetic properties as required by fluid state

    quadrature_tag

        An optional identifier denoting a particular quadrature
        discretization to use during operator evaluations.

    volume_dd: grudge.dof_desc.DOFDesc
        The DOF descriptor of the volume on which to apply the operator.

    Returns
    -------
    :class:`mirgecom.fluid.ConservedVars`

        The right-hand-side of the Euler flow equations:

        .. math::

            \dot{\mathbf{q}} = - \nabla\cdot\mathbf{F} +
                (\mathbf{F}\cdot\hat{n})_{\partial\Omega}
    """
    dd_quad_vol = volume_dd.with_discr_tag(quadrature_tag)
    dd_quad_allfaces = dd_quad_vol.trace(FACE_RESTR_ALL)

    volume_state_quad, interior_boundary_states_quad, domain_boundary_states_quad = \
        make_operator_fluid_states(discr, state, gas_model, boundaries,
                                    quadrature_tag, volume_dd=volume_dd)

    # Compute volume contributions
    inviscid_flux_vol = inviscid_flux(volume_state_quad)

    # Compute interface contributions
    inviscid_flux_bnd = inviscid_boundary_flux_for_divergence_operator(
        discr, gas_model, boundaries, interior_boundary_states_quad,
        domain_boundary_states_quad, quadrature_tag=quadrature_tag,
        numerical_flux_func=inviscid_numerical_flux_func, time=time,
        volume_dd=volume_dd)

    return -div_operator(discr, dd_quad_vol, dd_quad_allfaces,
                         inviscid_flux_vol, inviscid_flux_bnd)


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


def extract_vars_for_logging(dim: int, state, eos) -> dict:
    """Extract state vars."""
    dv = eos.dependent_vars(state)

    from mirgecom.utils import asdict_shallow
    name_to_field = asdict_shallow(state)
    name_to_field.update(asdict_shallow(dv))
    return name_to_field
