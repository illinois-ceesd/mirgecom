"""Useful bits and bobs.

.. autoclass:: StatisticsAccumulator
.. autofunction:: asdict_shallow
.. autofunction:: force_evaluation
.. autofunction:: force_compile
.. autofunction:: normalize_boundaries
.. autofunction:: project_from_base
.. autofunction:: mask_from_elements
"""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
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

from typing import Optional
from arraycontext import tag_axes
from meshmode.dof_array import DOFArray
from meshmode.transform_metadata import (
    DiscretizationElementAxisTag,
    DiscretizationDOFAxisTag)


def asdict_shallow(dc_instance) -> dict:
    """Convert a dataclass into a dict.

    What :func:`dataclasses.asdict` should have been: no
    recursion, no deep copy. Simply turn one layer of
    a dataclass into a :class:`dict`.
    """
    from dataclasses import fields
    return {attr.name: getattr(dc_instance, attr.name)
            for attr in fields(dc_instance)}


class StatisticsAccumulator:
    """Class that provides statistical functions for multiple values.

    .. automethod:: __init__
    .. automethod:: add_value
    .. automethod:: sum
    .. automethod:: mean
    .. automethod:: max
    .. automethod:: min
    .. autoattribute:: num_values
    """

    def __init__(self, scale_factor: float = 1) -> None:
        """Initialize an empty StatisticsAccumulator object.

        Parameters
        ----------
        scale_factor
            Scale returned statistics by this factor.
        """
        # Number of values stored in the StatisticsAccumulator
        self.num_values: int = 0

        self._sum: float = 0
        self._min: Optional[float] = None
        self._max: Optional[float] = None
        self.scale_factor = scale_factor

    def add_value(self, v: float) -> None:
        """Add a new value to the statistics."""
        if v is None:
            return
        self.num_values += 1
        self._sum += v
        if self._min is None or v < self._min:
            self._min = v
        if self._max is None or v > self._max:
            self._max = v

    def sum(self) -> Optional[float]:
        """Return the sum of added values."""
        if self.num_values == 0:
            return None

        return self._sum * self.scale_factor

    def mean(self) -> Optional[float]:
        """Return the mean of added values."""
        if self.num_values == 0:
            return None

        return self._sum / self.num_values * self.scale_factor

    def max(self) -> Optional[float]:
        """Return the max of added values."""
        if self.num_values == 0 or self._max is None:
            return None

        return self._max * self.scale_factor

    def min(self) -> Optional[float]:
        """Return the min of added values."""
        if self.num_values == 0 or self._min is None:
            return None

        return self._min * self.scale_factor


def force_evaluation(actx, x):
    """Force evaluation of a (possibly lazy) array."""
    if actx is None:
        return x
    return actx.freeze_thaw(x)


def force_compile(actx, f, *args):
    """Force compilation of *f* with *args*."""
    new_args = [force_evaluation(actx, arg) for arg in args]
    f_compiled = actx.compile(f)
    f_compiled(*new_args)
    return f_compiled


def normalize_boundaries(boundaries):
    """
    Normalize the keys of *boundaries*.

    Promotes boundary tags to :class:`grudge.dof_desc.BoundaryDomainTag`.
    """
    from grudge.dof_desc import as_dofdesc
    return {
        as_dofdesc(key).domain_tag: bdry
        for key, bdry in boundaries.items()}


def project_from_base(dcoll, tgt_dd, field):
    """Project *field* from *DISCR_TAG_BASE* to the same discr. as *tgt_dd*."""
    from grudge.dof_desc import DISCR_TAG_BASE, as_dofdesc
    from grudge.op import project

    tgt_dd = as_dofdesc(tgt_dd)

    if tgt_dd.discretization_tag is not DISCR_TAG_BASE:
        tgt_dd_base = tgt_dd.with_discr_tag(DISCR_TAG_BASE)
        return project(dcoll, tgt_dd_base, tgt_dd, field)
    else:
        return field


def mask_from_elements(dcoll, dd, actx, elements):
    """Get a :class:`~meshmode.dof_array.DOFArray` mask corresponding to *elements*.

    Returns
    -------
    mask: :class:`meshmode.dof_array.DOFArray`
        A DOF array containing $1$ for elements that are in *elements* and $0$
        for elements that aren't.
    """
    discr = dcoll.discr_from_dd(dd)
    mesh = discr.mesh
    zeros = discr.zeros(actx)

    group_arrays = []

    for igrp in range(len(mesh.groups)):
        start_elem_nr = mesh.base_element_nrs[igrp]
        end_elem_nr = start_elem_nr + mesh.groups[igrp].nelements
        grp_elems = elements[
            (elements >= start_elem_nr)
            & (elements < end_elem_nr)] - start_elem_nr
        grp_ary_np = actx.to_numpy(zeros[igrp])
        grp_ary_np[grp_elems] = 1
        group_arrays.append(actx.from_numpy(grp_ary_np))

    return tag_axes(actx, {
        0: DiscretizationElementAxisTag(),
        1: DiscretizationDOFAxisTag()
    }, DOFArray(actx, tuple(group_arrays)))
