"""Multiphysics module for MIRGE-Com."""

__copyright__ = """
Copyright (C) 2023 University of Illinois Board of Trustees
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

__doc__ = """
.. automodule:: mirgecom.multiphysics.thermally_coupled_fluid_wall

.. autofunction:: make_interface_boundaries
"""


def make_interface_boundaries(bdry_factories, inter_volume_tpairs):
    """
    Create volume-pairwise interface boundaries from inter-volume data.

    Return a :class:`dict` mapping a (directional) pair of
    :class:`grudge.dof_desc.DOFDesc` *(other_vol_dd, self_vol_dd)* representing
    neighboring volumes to a :class:`dict` of boundary objects. Specifically,
    *interface_boundaries[other_vol_dd, self_vol_dd]* maps each interface boundary
    :class:`~grudge.dof_desc.DOFDesc` to a boundary object.

    Parameters
    ----------

    bdry_factories

        Mapping from directional volume :class:`~grudge.dof_desc.DOFDesc` pair to
        a function that takes a :class:`grudge.trace_pair.TracePair` and returns
        an interface boundary object.

    inter_volume_tpairs

        Mapping from directional volume :class:`~grudge.dof_desc.DOFDesc` pair to
        a :class:`list` of :class:`grudge.trace_pair.TracePair` (as is returned by
        :func:`grudge.trace_pair.inter_volume_trace_pairs`).
    """
    return {
        vol_dd_pair: {
            tpair.dd: bdry_factories[vol_dd_pair](tpair)
            for tpair in tpairs}
        for vol_dd_pair, tpairs in inter_volume_tpairs.items()}
