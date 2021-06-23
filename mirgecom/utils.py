"""Useful bits and bobs.

.. autofunction:: asdict_shallow
.. autofunction:: outer
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

__doc__ = """
.. autoclass:: StatisticsAccumulator
.. autofunction:: asdict_shallow
.. autofunction:: outer
"""

from typing import Optional
import numpy as np


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
        self.num_values: int = 0
        """Number of values stored in the StatisticsAccumulator."""

        self._sum = 0
        self._min = None
        self._max = None
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
        if self.num_values == 0:
            return None

        return self._max * self.scale_factor

    def min(self) -> Optional[float]:
        """Return the min of added values."""
        if self.num_values == 0:
            return None

        return self._min * self.scale_factor


def outer(a, b, scalar_types=None):
    """
    Compute the outer product of *a* and *b*.

    Tweaks the behavior of :func:`numpy.outer` to return a lower-dimensional
    object if either/both of *a* and *b* are scalars (whereas :func:`numpy.outer`
    always returns a matrix).

    Parameters
    ----------
    a
        A scalar, :class:`numpy.ndarray`, or :class:`arraycontext.ArrayContainer`.
    b
        A scalar, :class:`numpy.ndarray`, or :class:`arraycontext.ArrayContainer`.
    scalar_types
        A :class:`list` of types that should be treated as scalars. Defaults to
        [:class:`numbers.Number`, :class:`meshmode.dof_array.DOFArray`].
    """
    if scalar_types is None:
        from numbers import Number
        from meshmode.dof_array import DOFArray
        scalar_types = [Number, DOFArray]

    def is_scalar(x):
        for t in scalar_types:
            if isinstance(x, t):
                return True
        return False

    if is_scalar(a) or is_scalar(b):
        return a*b
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.outer(a, b)
    else:
        from arraycontext import map_array_container
        return map_array_container(lambda x: outer(x, b), a)
