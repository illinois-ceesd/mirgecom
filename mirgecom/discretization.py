"""Provide some wrappers for centralizing grudge discretization creation.

Discretization creation
-----------------------

.. autofunction:: create_discretization_collection
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
import logging

logger = logging.getLogger(__name__)


# Centralize the discretization collection creation routine so that
# we can replace it more easily when we refactor the drivers and
# examples to use discretization collections, and change it centrally
# when we want to change it.
def create_discretization_collection(actx, volume_meshes, order, *,
                                     mpi_communicator=None, quadrature_order=-1,
                                     tensor_product_elements=False):
    """Create and return a grudge DG discretization collection."""
    from warnings import warn
    if mpi_communicator is not None:
        warn(
            "mpi_communicator argument is deprecated and will disappear in Q4 2022.",
            DeprecationWarning, stacklevel=2)

    if tensor_product_elements:
        warn(
            "tensor_product_elements argument is not needed and will vanish soon.",
            DeprecationWarning, stacklevel=2)

    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD, DISCR_TAG_MODAL
    from grudge.discretization import make_discretization_collection
    from meshmode.discretization.poly_element import (
        InterpolatoryEdgeClusteredGroupFactory,
        QuadratureGroupFactory,
        ModalGroupFactory
    )
    from meshmode.mesh import Mesh, TensorProductElementGroup

    if quadrature_order < 0:
        mesh_is_tpe = False
        if isinstance(volume_meshes, Mesh):
            the_mesh = volume_meshes
        else:
            the_mesh = next(iter(volume_meshes.values()))
        mesh_is_tpe = \
            isinstance(next(iter(the_mesh.groups())), TensorProductElementGroup)

        quadrature_order = order if mesh_is_tpe else 2*order

    return make_discretization_collection(
        actx, volume_meshes,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: InterpolatoryEdgeClusteredGroupFactory(order),
            DISCR_TAG_MODAL: ModalGroupFactory(order),
            DISCR_TAG_QUAD: QuadratureGroupFactory(quadrature_order)
        }
    )
