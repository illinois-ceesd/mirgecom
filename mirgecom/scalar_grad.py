""":mod:`mirgecom.scalar_grad` computes a scalar field gradient.

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
from pytools.obj_array import flat_obj_array
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.trace_pair import TracePair, interior_trace_pairs
import grudge.op as op


def _flux(dcoll, u_tpair):

    actx = u_tpair.int.array_context
    normal = actx.thaw(dcoll.normal(u_tpair.dd))

    flux_weak = normal*u_tpair.avg

    return op.project(dcoll, u_tpair.dd, "all_faces", flux_weak)


class _ScalarGradTag:
    pass


def scalar_grad_operator(dcoll, u, *, comm_tag=None):
    """Compute the RHS of the wave equation.

    Parameters
    ----------
    dcoll: grudge.discretization.DiscretizationCollection
        the discretization collection to use
    u: DOF Array
        scalar field for which the gradient is computed
    comm_tag: Hashable
        Tag for distributed communication

    Returns
    -------
    numpy.ndarray
        an object array of DOF arrays, representing the ODE RHS
    """

    dir_u = op.project(dcoll, "vol", BTAG_ALL, u)
    dir_bval = dir_u
    dir_bc = -dir_u

    vol_term = -op.weak_local_grad(dcoll, u)

    dom_bnd_flux = _flux(dcoll, u_tpair=TracePair(BTAG_ALL, interior=dir_bval,
                                                  exterior=dir_bc))

    itp = interior_trace_pairs(dcoll, u, comm_tag=(_ScalarGradTag, comm_tag))
    el_bnd_flux = sum(_flux(dcoll, u_tpair=tpair)
                      for tpair in itp)

    surf_term = op.face_mass(dcoll, dom_bnd_flux + el_bnd_flux)

    return op.inverse_mass(dcoll, vol_term + surf_term) 
