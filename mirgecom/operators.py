r""":mod:`mirgecom.operators` provides helper functions for composing DG operators.

Calculus
^^^^^^^^

.. autofunction:: dg_grad
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
from pytools.obj_array import obj_array_vectorize
from grudge.eager import (
    interior_trace_pair,
    cross_rank_trace_pairs
)


def dg_grad(discr, compute_interior_flux, compute_boundary_flux, boundaries, u):
    r"""Compute a DG gradient for the input *u*.

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization
        the discretization to use
    u: Union[meshmode.dof_array.DOFArray, numpy.ndarray]
        the DOF array (or object array of DOF arrays) to which the operator should be
        applied

    Returns
    -------
    meshmode.dof_array.DOFArray or numpy.ndarray
        the dg gradient operator applied to *u*
    """
    if isinstance(u, np.ndarray):
        vol_part = obj_array_vectorize(discr.weak_grad, u)
    else:
        vol_part = discr.weak_grad(u)

    bnd_flux = compute_interior_flux(interior_trace_pair(discr, u))
    bnd_flux_part = sum(compute_interior_flux(p_pair) for p_pair in
                        cross_rank_trace_pairs(discr, u))
    bnd_flux_bc = sum(compute_boundary_flux(btag, u) for btag in boundaries)

    return -discr.inverse_mass(vol_part
                               - discr.face_mass(bnd_flux+bnd_flux_part+bnd_flux_bc))
