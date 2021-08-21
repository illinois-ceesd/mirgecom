r""":mod:`mirgecom.operators` provides helper functions for composing DG operators.

.. autofunction:: grad_operator
.. autofunction:: div_operator
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


def grad_operator(discr, u, flux):
    r"""Compute a DG gradient for the input *u* with flux given by *flux*.

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization
        the discretization to use
    u: meshmode.dof_array.DOFArray or numpy.ndarray
        the DOF array (or object array of DOF arrays) to which the operator should be
        applied
    flux: numpy.ndarray
        the boundary fluxes across the faces of the element
    Returns
    -------
    meshmode.dof_array.DOFArray or numpy.ndarray
        the dg gradient operator applied to *u*
    """
    from grudge.op import weak_local_grad
    return -discr.inverse_mass(weak_local_grad(discr, u, nested=False)
                               - discr.face_mass(flux))


def div_operator(discr, u, flux):
    r"""Compute a DG divergence of vector-valued function *u* with flux given by *flux*.

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization
        the discretization to use
    u: numpy.ndarray
        the vector-valued function for which divergence is to be calculated
    flux: numpy.ndarray
        the boundary fluxes across the faces of the element
    Returns
    -------
    meshmode.dof_array.DOFArray or numpy.ndarray
        the dg divergence operator applied to vector-valued function *u*.
    """
    from grudge.op import weak_local_div
    return -discr.inverse_mass(weak_local_div(discr, u)
                               - discr.face_mass(flux))
