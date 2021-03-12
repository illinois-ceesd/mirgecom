r""":mod:`mirgecom.artificial viscosity` Artificial viscocity for Euler.

Artificial viscosity for the Euler Equations:

.. math::

    \partial_t \mathbf{Q} = \nabla\cdot{\varepsilon\nabla\mathbf{Q}}

where:

-  state $\mathbf{Q} = [\rho, \rho{E}, \rho\vec{V}, \rho{Y}_\alpha]$
-  artifical viscosity coefficient $\varepsilon$

To evalutate the second order derivative the problem is recast as a set of first
 order problems:

.. math::

    \partial_t \mathbf{Q} = \nabla\cdot{\mathbf{R}}
    \mathbf{R} = \varepsilon\nabla\mathbf{Q}

where $\mathbf{R}$ is an intermediate variable.

RHS Evaluation
^^^^^^^^^^^^^^

.. autofunction:: artificial_viscosity
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

# from dataclasses import dataclass

import numpy as np
from pytools.obj_array import (
    obj_array_vectorize,
    obj_array_vectorize_n_args,
)
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import interior_trace_pair, cross_rank_trace_pairs
from grudge.symbolic.primitives import TracePair
# from mirgecom.euler import split_conserved, join_conserved
from mirgecom.tag_cells import smoothness_indicator


def _facial_flux_r(discr, q_tpair):

    actx = q_tpair.int.array_context

    flux_dis = q_tpair.avg

    normal = thaw(actx, discr.normal(q_tpair.dd))

    flux_out = flux_dis * normal

    # Can't do it here... "obj arrays not allowed on compute device"
    # def flux_calc(flux):
    #    return (flux * normal)
    # flux_out = obj_array_vectorize(flux_calc,flux_dis)

    return discr.project(q_tpair.dd, "all_faces", flux_out)


def _facial_flux_q(discr, q_tpair):

    actx = q_tpair[0].int.array_context

    normal = thaw(actx, discr.normal(q_tpair.dd))

    flux_out = np.dot(q_tpair.avg, normal)

    return discr.project(q_tpair.dd, "all_faces", flux_out)


def artificial_viscosity(discr, t, eos, boundaries, r, alpha, **kwargs):
    r"""Compute artifical viscosity for the euler equations."""
    # Get smoothness indicator
    epsilon = np.zeros((2 + discr.dim,), dtype=object)
    indicator = smoothness_indicator(r[0], discr, **kwargs)
    for i in range(2 + discr.dim):
        epsilon[i] = indicator

    # compute dissapation flux

    # Cannot call weak_grad on obj of nd arrays
    # use obj_array_vectorize as work around
    dflux_r = obj_array_vectorize(discr.weak_grad, r)

    # interior face flux

    # Doesn't work: something related to obj on compute device
    # Not 100% on reason
    # qin = interior_trace_pair(discr,r)
    # iff_r = _facial_flux(discr,q_tpair=qin)

    # Work around?
    def my_facialflux_r_interior(q):
        qin = interior_trace_pair(discr, q)
        return _facial_flux_r(discr, q_tpair=qin)

    iff_r = obj_array_vectorize(my_facialflux_r_interior, r)

    # partition boundaries flux
    # flux across partition boundaries
    def my_facialflux_r_partition(q):
        qin = cross_rank_trace_pairs(discr, q)
        return sum(_facial_flux_r(discr, q_tpair=part_pair) for part_pair in qin)

    pbf_r = obj_array_vectorize(my_facialflux_r_partition, r)

    # True boundary implementation
    # Okay, not sure about this...
    # What I am attempting:
    #       1. Loop through all the boundaries
    #       2. Define a function my_TP that performes the trace pair for the given
    #          boundary given a solution variable
    #       3. Get the external solution from the boundary routine
    #       4. Get hte projected internal solution
    #       5. Compute the boundary flux as a sum over boundaries, using the
    #          obj_array_vectorize to pass each solution variable one at a time
    # DO I really need to do this like this?
    dbf_r = 0.0 * iff_r
    for btag in boundaries:

        def my_facialflux_r_boundary(sol_ext, sol_int):
            q_tpair = TracePair(
                btag,
                interior=sol_int,
                exterior=sol_ext,
            )
            return _facial_flux_r(discr, q_tpair=q_tpair)

        r_ext = boundaries[btag].exterior_soln(discr, eos=eos, btag=btag, t=t, q=r)
        r_int = discr.project("vol", btag, r)
        dbf_r = dbf_r + obj_array_vectorize_n_args(
            my_facialflux_r_boundary, r_ext, r_int
        )

    # Compute q, half way done!
    # q = discr.inverse_mass(-alpha*(dflux_r-discr.face_mass(iff_r + pbf_r + dbf_r)))
    q = discr.inverse_mass(
        -alpha * epsilon * (dflux_r - discr.face_mass(iff_r + pbf_r + dbf_r))
    )

    # flux of q

    # Again we need to vectorize
    # q is a object array of object arrays (dim,) of DOFArrays (?)
    dflux_q = obj_array_vectorize(discr.weak_div, q)

    # interior face flux of q
    def my_facialflux_q_interior(q):
        qin = interior_trace_pair(discr, q)
        iff_q = _facial_flux_q(discr, q_tpair=qin)
        return iff_q

    iff_q = obj_array_vectorize(my_facialflux_q_interior, q)

    # flux across partition boundaries
    def my_facialflux_q_partition(q):
        qin = cross_rank_trace_pairs(discr, q)
        return sum(_facial_flux_q(discr, q_tpair=part_pair) for part_pair in qin)

    pbf_q = obj_array_vectorize(my_facialflux_q_partition, q)

    dbf_q = 0.0 * iff_q
    for btag in boundaries:

        def my_facialflux_q_boundary(sol_ext, sol_int):
            q_tpair = TracePair(btag, interior=sol_int, exterior=sol_ext)
            return _facial_flux_q(discr, q_tpair=q_tpair)

        q_ext = boundaries[btag].av(discr, eos=eos, btag=btag, t=t, q=q)
        q_int = discr.project("vol", btag, q)
        dbf_q = dbf_q + obj_array_vectorize_n_args(
            my_facialflux_q_boundary, q_ext, q_int
        )

    # Return the rhs contribution
    return discr.inverse_mass(-dflux_q + discr.face_mass(iff_q + pbf_q + dbf_q))
