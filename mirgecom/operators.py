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

import grudge.op as op


def grad_operator(discr, dd_vol, dd_faces, volume_fluxes, boundary_fluxes):
    r"""Compute a DG gradient for the input *u* with flux given by *flux*.

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization
        the discretization to use
    dd_vol: grudge.dof_desc.DOFDesc
        the degree-of-freedom tag associated with the volume discrezation.
        This determines the type of quadrature to be used.
    dd_faces: grudge.dof_desc.DOFDesc
        the degree-of-freedom tag associated with the surface discrezation.
        This determines the type of quadrature to be used.
    volume_fluxes: :class:`mirgecom.fluid.ConservedVars`
        the vector-valued function for which divergence is to be calculated
    boundary_fluxes: :class:`mirgecom.fluid.ConservedVars`
        the boundary fluxes across the faces of the element
    Returns
    -------
    meshmode.dof_array.DOFArray or numpy.ndarray
        the dg gradient operator applied to *u*
    """
    return -discr.inverse_mass(
        op.weak_local_grad(discr, dd_vol, volume_fluxes)
        - op.face_mass(discr, dd_faces, boundary_fluxes)
    )


def div_operator(discr, dd_vol, dd_faces, volume_fluxes, boundary_fluxes):
    r"""Compute a DG divergence of vector-valued function *u* with flux given by *flux*.

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization
        the discretization to use
    dd_vol: grudge.dof_desc.DOFDesc
        the degree-of-freedom tag associated with the volume discrezation.
        This determines the type of quadrature to be used.
    dd_faces: grudge.dof_desc.DOFDesc
        the degree-of-freedom tag associated with the surface discrezation.
        This determines the type of quadrature to be used.
    volume_fluxes: :class:`mirgecom.fluid.ConservedVars`
        the vector-valued function for which divergence is to be calculated
    boundary_fluxes: :class:`mirgecom.fluid.ConservedVars`
        the boundary fluxes across the faces of the element
    Returns
    -------
    :class:`mirgecom.fluid.ConservedVars`
        the dg divergence operator applied to vector-valued function *u*.
    """
    return -discr.inverse_mass(
        op.weak_local_div(discr, dd_vol, volume_fluxes)
        - op.face_mass(discr, dd_faces, boundary_fluxes)
    )
