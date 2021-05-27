r""":mod:`mirgecom.operators` provides helper functions for composing DG operators.

.. autofunction:: dg_grad
.. autofunction:: dg_div
.. autofunction:: element_boundary_flux
.. autofunction:: elbnd_flux
.. autofunction:: jump
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
from grudge.eager import (
    interior_trace_pair,
    cross_rank_trace_pairs
)


# placeholder awaits resolution on grudge PR #71
def jump(trace_pair):
    r"""Return the "jump" in the quantities represented by the *trace_pair*.

    The jump in a quantity $\mathbf{q}$ is denoted $[\mathbf{q}]$ and is
    defined by:
    .. math:
        [\mathbf{q}] = \mathbf{q}^+ - \mathbf{q}^-

    Parameters
    ----------
    trace_pair: :class:`grudge.sym.TracePair`
    Represents the quantity for which the jump is to be calculated.

    Returns
    -------
    like(trace_pair.int)
    """
    return trace_pair.ext - trace_pair.int


def elbnd_flux(discr, compute_interior_flux, compute_boundary_flux,
               int_tpair, xrank_pairs, boundaries):
    """Generically compute flux across element boundaries."""
    return (compute_interior_flux(int_tpair)
            + sum(compute_interior_flux(part_tpair)
                  for part_tpair in xrank_pairs)
            + sum(compute_boundary_flux(btag) for btag in boundaries))


def element_boundary_flux(discr, compute_interior_flux, compute_boundary_flux,
                          boundaries, u):
    """Generically compute flux across element boundaries for simple f(u) flux."""
    return elbnd_flux(discr, compute_interior_flux, compute_boundary_flux,
                      interior_trace_pair(discr, u),
                      cross_rank_trace_pairs(discr, u), boundaries)


def dg_grad(discr, interior_u, bndry_flux):
    r"""Compute a DG gradient for the input *u*.

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization
        the discretization to use
    compute_interior_flux:
        function taking a `grudge.sym.TracePair` and returning the numerical flux
        for the corresponding interior boundary.
    compute_boundary_flux:
        function taking a boundary tag and returning the numerical flux
        for the corresponding domain boundary.
    u: meshmode.dof_array.DOFArray or numpy.ndarray
        the DOF array (or object array of DOF arrays) to which the operator should be
        applied

    Returns
    -------
    meshmode.dof_array.DOFArray or numpy.ndarray
        the dg gradient operator applied to *u*
    """
    from grudge.op import weak_local_grad
    return -discr.inverse_mass(weak_local_grad(discr, interior_u, nested=False)
                               - discr.face_mass(bndry_flux))


def dg_div(discr, vol_flux, bnd_flux):
    r"""Compute a DG divergence for the flux vectors given in *vol_flux* and *bnd_flux*.

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization
        the discretization to use
    vol_flux: np.ndarray
        the volume flux term in the element
    bnd_flux: np.ndarray
        the boundary fluxes across the faces of the element
    Returns
    -------
    meshmode.dof_array.DOFArray or numpy.ndarray
        the dg divergence operator applied to the flux of *u*.
    """
    from grudge.op import weak_local_div
    return -discr.inverse_mass(weak_local_div(discr, vol_flux)
                               - discr.face_mass(bnd_flux))
