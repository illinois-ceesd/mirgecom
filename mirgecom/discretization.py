"""Provide some wrappers for centralizing grudge discretization creation.

Discretization creation
-----------------------

.. autofunction:: create_dg_discretization
.. autofunction:: create_dg_discretization_collection
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


def create_dg_discretization(actx, mesh, order):
    """Create and return a grudge DG discretization object."""
    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    from grudge.eager import EagerDGDiscretization
    from meshmode.discretization.poly_element import \
        QuadratureSimplexGroupFactory, \
        PolynomialWarpAndBlendGroupFactory
    discr = EagerDGDiscretization(
        actx, mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: PolynomialWarpAndBlendGroupFactory(order),
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(3*order),
        }
    )
    return discr
