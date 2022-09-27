""":mod:`mirgecom.wall_model` provides utilities to deal with wall materials.

Physical Wall Model Encapsulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: WallModel

"""

__copyright__ = """
Copyright (C) 2022 University of Illinois Board of Trustees
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
import numpy as np  # noqa
from dataclasses import dataclass
from typing import Union

from meshmode.dof_array import DOFArray


# TODO: Figure out what actually belongs here and what should be in state,
# then generalize like EOS/transport model
@dataclass(frozen=True)
class WallModel:
    """Physical wall model for calculating wall state-dependent quantities.

    .. attribute:: density

        The density of the material.

    .. attribute:: heat_capacity

        The specific heat capacity of the material.

    .. attribute:: thermal_conductivity

        The thermal conductivity of the material.
    """

    density: Union[float, DOFArray]
    heat_capacity: Union[float, DOFArray]
    thermal_conductivity: Union[float, DOFArray]

    def thermal_diffusivity(self):
        """Compute the thermal diffusivity of the material."""
        return self.thermal_conductivity/(self.density * self.heat_capacity)
