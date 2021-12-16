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

from mirgecom.inviscid import (
    inviscid_flux,
    inviscid_facial_flux,
    inviscid_flux_rusanov
)
from mirgecom.operators import div_operator
from mirgecom.gas_model import (
    project_fluid_state,
    make_fluid_state_trace_pairs
)
from grudge.trace_pair import (
    TracePair,
    interior_trace_pairs
)
from grudge.dof_desc import DOFDesc, as_dofdesc

import grudge.op as op


def euler_operator(discr, state, gas_model, boundaries, time=0.0,
                   inviscid_numerical_flux_func=inviscid_flux_rusanov,
                   quadrature_tag=None):
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
    state: :class:`~mirgecom.gas_model.FluidState`

        Fluid state object with the conserved state, and dependent
        quantities.

    boundaries

        Dictionary of boundary functions, one for each valid btag

    time

        Time

    gas_model: :class:`~mirgecom.gas_model.GasModel`

        Physical gas model including equation of state, transport,
        and kinetic properties as required by fluid state

    quadrature_tag
        An optional identifier denoting a particular quadrature
        discretization to use during operator evaluations.
        The default value is *None*.

    Returns
    -------
    :class:`mirgecom.fluid.ConservedVars`

        Agglomerated object array of DOF arrays representing the RHS of the Euler
        flow equations.
    """
    dd_base = as_dofdesc("vol")
    dd_vol = DOFDesc("vol", quadrature_tag)
    dd_faces = DOFDesc("all_faces", quadrature_tag)

    def interp_to_surf_quad(utpair):
        local_dd = utpair.dd
        local_dd_quad = local_dd.with_discr_tag(quadrature_tag)
        return TracePair(
            local_dd_quad,
            interior=op.project(discr, local_dd, local_dd_quad, utpair.int),
            exterior=op.project(discr, local_dd, local_dd_quad, utpair.ext)
        )

    boundary_states = {
        btag: project_fluid_state(
            discr, dd_base,
            # Make sure we get the state on the quadrature grid
            # restricted to the tag *btag*
            as_dofdesc(btag).with_discr_tag(quadrature_tag),
            state, gas_model) for btag in boundaries
    }

    cv_interior_pairs = [
        # Get the interior trace pairs onto the surface quadrature
        # discretization (if any)
        interp_to_surf_quad(tpair)
        for tpair in interior_trace_pairs(discr, state.cv)
    ]

    tseed_interior_pairs = None
    if state.is_mixture > 0:
        # If this is a mixture, we need to exchange the temperature field because
        # mixture pressure (used in the inviscid flux calculations) depends on
        # temperature and we need to seed the temperature calculation for the
        # (+) part of the partition boundary with the remote temperature data.
        tseed_interior_pairs = [
            # Get the interior trace pairs onto the surface quadrature
            # discretization (if any)
            interp_to_surf_quad(tpair)
            for tpair in interior_trace_pairs(discr, state.temperature)
        ]

    interior_states = make_fluid_state_trace_pairs(cv_interior_pairs,
                                                   gas_model,
                                                   tseed_interior_pairs)

    # Interpolate the fluid state to the volume quadrature grid
    # (conserved vars and derived vars if applicable)
    state_quad = project_fluid_state(discr, dd_base, dd_vol, state, gas_model)

    inviscid_flux_vol = inviscid_flux(state_quad)
    inviscid_flux_bnd = (

        # Domain boundaries
        sum(boundaries[btag].inviscid_divergence_flux(
            discr,
            # Make sure we get the state on the quadrature grid
            # restricted to the tag *btag*
            as_dofdesc(btag).with_discr_tag(quadrature_tag),
            gas_model,
            state_minus=boundary_states[btag],
            time=time,
            numerical_flux_func=inviscid_numerical_flux_func)
            for btag in boundaries)

        # Interior boundaries
        + sum(inviscid_facial_flux(discr, gas_model=gas_model, state_pair=state_pair,
                                   numerical_flux_func=inviscid_numerical_flux_func)
              for state_pair in interior_states)
    )

    return -div_operator(discr, dd_vol, dd_faces,
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
