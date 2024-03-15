""":mod:`mirgecom.wave` computes the rhs of the wave equation.

.. autofunction:: wave_operator
"""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees"
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
import numpy.linalg as la  # noqa
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.trace_pair import TracePair, interior_trace_pairs
import grudge.op as op
from arraycontext import get_container_context_recursively

def _flux(dcoll, v, w_tpair):

    actx = get_container_context_recursively(w_tpair.int)
    # normal = actx.thaw(dcoll.normal(w_tpair.dd))

    # flux_weak = np.dot(w_tpair.avg*v, normal)
    flux_weak = w_tpair.diff
    return op.project(dcoll, w_tpair.dd, "all_faces", flux_weak)


class _ScalarTag:
    pass


def continuity_operator(dcoll, v, u, *, comm_tag=None):
    """Compute the RHS of the wave equation.

    Parameters
    ----------
    dcoll: grudge.discretization.DiscretizationCollection
        the discretization collection to use
    v: numpy.ndarray
        transport velocity vector
    u: DOF array
        scalar to transport
    comm_tag: Hashable
        Tag for distributed communication

    Returns
    -------
    float
        the RHS of the scalar transport equation
    """
    dir_u = op.project(dcoll, "vol", BTAG_ALL, u)
    dir_bc = -dir_u
    itp = interior_trace_pairs(dcoll, u, comm_tag=(_ScalarTag, comm_tag))

    transport_flux = u*v
    vol_term = -op.weak_local_div(dcoll, transport_flux)

    interior_facial_flux = sum(_flux(dcoll, v=v, w_tpair=tpair) for tpair in itp)
    print(f"{interior_facial_flux=}")

    # boundary_facial_flux = _flux(dcoll, v=v,
    #                             w_tpair=TracePair(BTAG_ALL, interior=dir_u,
    #                                               exterior=dir_bc))
    surface_fluxes = interior_facial_flux # + boundary_facial_flux
    surf_term = op.face_mass(dcoll, surface_fluxes)

    rhs = surf_term
    # rhs = op.inverse_mass(dcoll, vol_term + surf_term)
    return rhs
