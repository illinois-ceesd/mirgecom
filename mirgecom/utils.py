"""Useful bits and bobs.

.. autofunction:: asdict_shallow
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
"""

from typing import Optional


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
