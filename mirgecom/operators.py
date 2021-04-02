r""":mod:`mirgecom.operators` provides helper functions for composing DG operators.

.. autofunction:: element_local_grad
.. autofunction:: weak_grad
.. autofunction:: dg_grad
.. autofunction:: dg_div
.. autofunction:: dg_div_low
.. autofunction:: element_boundary_flux
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


def element_boundary_flux(discr, compute_interior_flux, compute_boundary_flux,
                          boundaries, u):
    """Generically compute flux across element boundaries for simple f(u) flux."""
    return (compute_interior_flux(interior_trace_pair(discr, u))
            + sum(compute_interior_flux(part_tpair)
                  for part_tpair in cross_rank_trace_pairs(discr, u))
            + sum(compute_boundary_flux(btag) for btag in boundaries))


def element_local_grad(discr, u):
    r"""Compute an element-local gradient for the input volume function *u*.

    This function simply wraps :func:`grudge.eager.grad` and adds support for
    vector-valued volume functions.

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization
        the discretization to use
    u: meshmode.dof_array.DOFArray or numpy.ndarray
        the DOF array (or object array of DOF arrays) to which the operator should be
        applied

    Returns
    -------
    meshmode.dof_array.DOFArray or numpy.ndarray
        the dg gradient operator applied to *u*
    """
    if isinstance(u, np.ndarray):
        return obj_array_vectorize(discr.grad, u)
    else:
        return discr.grad(u)


def weak_grad(discr, u):
    r"""Compute an element-local gradient for the input function *u*.

    This function simply wraps :func:`grudge.eager.weak_grad` and adds support for
    vector-valued volume functions.

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization
        the discretization to use
    u: meshmode.dof_array.DOFArray or numpy.ndarray
        the DOF array (or object array of DOF arrays) to which the operator should be
        applied

    Returns
    -------
    meshmode.dof_array.DOFArray or numpy.ndarray
        the dg gradient operator applied to *u*
    """
    if isinstance(u, np.ndarray):
        return obj_array_vectorize(discr.weak_grad, u)
    else:
        return discr.weak_grad(u)


def dg_grad(discr, compute_interior_flux, compute_boundary_flux, boundaries, u):
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
    return -discr.inverse_mass(
        weak_grad(discr, u) - discr.face_mass(
            element_boundary_flux(discr, compute_interior_flux,
                                  compute_boundary_flux, boundaries, u)
            )
        )


def dg_div_low(discr, vol_flux, bnd_flux):
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
    return -discr.inverse_mass(discr.weak_div(vol_flux)-discr.face_mass(bnd_flux))


def dg_div(discr, compute_vol_flux, compute_interior_flux,
           compute_boundary_flux, boundaries, u):
    r"""Compute a DG divergence for the vector fluxes computed for *u*.

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
        the dg divergence operator applied to the flux of *u*.
    """
    return dg_div_low(
        discr, compute_vol_flux(),
        element_boundary_flux(discr, compute_interior_flux,
                              compute_boundary_flux, boundaries, u)
    )
