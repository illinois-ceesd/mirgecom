""":mod:`mirgecom.multiphysics.simple_oxidation` handles Y2 oxidation model."""

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

from dataclasses import dataclass, fields
import numpy as np
from meshmode.dof_array import DOFArray
from arraycontext import (
    dataclass_array_container,
    with_container_arithmetic,
    get_container_context_recursively
)


@with_container_arithmetic(bcast_obj_array=False,
                           bcast_container_types=(DOFArray, np.ndarray),
                           matmul=True,
                           _cls_has_array_context_attr=True,
                           rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class WallVars:
    r"""Class of conserved variables.

    .. attribute:: mass
    .. attribute:: energy
    .. attribute:: ox_mass
    """

    mass: DOFArray
    energy: DOFArray
    ox_mass: DOFArray

    @property
    def array_context(self):
        """Return an array context for the :class:`WallVars` object."""
        return get_container_context_recursively(self.mass)

    def __reduce__(self):
        """Return a tuple reproduction of self for pickling."""
        return (WallVars, tuple(getattr(self, f.name)
                                    for f in fields(WallVars)))


@dataclass_array_container
@dataclass(frozen=True)
class WallDependentVars:
    """Dependent variables for the oxidation model."""

    tau: DOFArray
    thermal_conductivity: DOFArray
    oxygen_diffusivity: DOFArray
    temperature: DOFArray


class SimpleOxidationWallModel:
    """Variables dependent on the wall state.

    .. automethod:: __init__
    .. automethod:: eval_tau
    .. automethod:: eval_temperature
    .. automethod:: enthalpy
    .. automethod:: heat_capacity
    .. automethod:: thermal_conductivity
    .. automethod:: thermal_diffusivity
    .. automethod:: oxygen_diffusivity
    """

    def __init__(self, solid_data, wall_sample_mask,
                 enthalpy_func, heat_capacity_func, thermal_conductivity_func,
                 oxygen_diffusivity_func):
        """
        Initialize the boundary condition.

        Parameters
        ----------
        solid_data: 
            The class with the solid properties of the desired material.
        wall_sample_mask: meshmode.dof_array.DOFArray
            Array with 1 for the reactive wall and 0 otherwise
        enthalpy_func: 
            function that computes the enthalpy of the entire wall.
            Must include the non-reactive part of the wall, if existing.
        heat_capacity_func: 
            function that computes the heat capacity of the entire wall
            Must include the non-reactive part of the wall, if existing.
        thermal_conductivity_func: 
            function that computes the thermal conductivity of the entire wall.
            Must include the non-reactive part of the wall, if existing.
        oxygen_diffusivity_func: 
            function that computes the oxygen diffusivity inside the wall.
            Must include the non-reactive part of the wall, if existing.
        """
        self._fiber = solid_data
        self._sample_mask = wall_sample_mask
        self._enthalpy_func = enthalpy_func
        self._heat_capacity_func = heat_capacity_func
        self._thermal_conductivity_func = thermal_conductivity_func
        self._oxygen_diffusivity_func = oxygen_diffusivity_func

    def eval_tau(self, wv):
        r"""Evaluate the progress ratio of the oxidation.

        Where $\tau=1$, the material is locally virgin. On the other hand, if
        $\tau=0$, then the fibers were all consumed.

        Parameters
        ----------
        wv: :class:`WallVars`
            the class of conserved variables for the oxidation

        Returns
        -------
        tau: meshmode.dof_array.DOFArray
        """
        virgin = (self._fiber.intrinsic_density()
                  * self._fiber.solid_volume_fraction(tau=1.0))
        return (
            (1.0 - (virgin - wv.mass)/virgin) * self._sample_mask  # fiber
            + 1.0*(1.0 - self._sample_mask)  # inert
        )

    def eval_temperature(self, wv, tseed):
        """Evaluate the temperature using Newton iteration.

        Parameters
        ----------
        wv: :class:`WallVars`

        Returns
        -------
        temperature: meshmode.dof_array.DOFArray
        """
        temp = tseed*1.0
        for _ in range(0, 3):
            h = self.enthalpy(temp)
            cp = self.heat_capacity(temp)
            temp = temp - (h - wv.energy/wv.mass)/cp
        return temp

    def enthalpy(self, temperature):
        """Return the enthalpy of the wall as a function of temperature.

        Parameters
        ----------
        temperature: meshmode.dof_array.DOFArray

        Returns
        -------
        enthalpy: meshmode.dof_array.DOFArray
            the wall enthalpy, including all of its components
        """
        return self._enthalpy_func(temperature=temperature)

    def heat_capacity(self, temperature):
        """Return the heat capacity of the fibers as a function of temperature.

        Parameters
        ----------
        temperature: meshmode.dof_array.DOFArray

        Returns
        -------
        heat capacity: meshmode.dof_array.DOFArray
            the wall heat capacity, including all of its components
        """
        return self._heat_capacity_func(temperature=temperature)

    def thermal_conductivity(self, temperature, tau):
        """Return the thermal conductivity of the wall.

        It is a function of temperature and oxidation progress. As the fibers
        are oxidized, they reduce their cross area and, consequenctly, their
        hability to conduct heat.

        Parameters
        ----------
        temperature: meshmode.dof_array.DOFArray
        tau: meshmode.dof_array.DOFArray

        Returns
        -------
        thermal_conductivity: meshmode.dof_array.DOFArray
            the wall thermal conductivity, including all of its components
        """
        return self._thermal_conductivity_func(temperature=temperature, tau=tau)

    def thermal_diffusivity(self, mass, temperature, tau,
                            thermal_conductivity=None):
        """Thermal diffusivity of the wall.

        Parameters
        ----------
        mass: meshmode.dof_array.DOFArray
        temperature: meshmode.dof_array.DOFArray
        tau: meshmode.dof_array.DOFArray
        thermal_conductivity: meshmode.dof_array.DOFArray
            Optional. If not given, it will be evaluated using
            :func:`thermal_conductivity`

        Returns
        -------
        thermal_diffusivity: meshmode.dof_array.DOFArray
            the wall thermal diffusivity, including all of its components
        """
        if thermal_conductivity is None:
            thermal_conductivity = self.thermal_conductivity(
                temperature=temperature, tau=tau)
        return thermal_conductivity/(mass * self.heat_capacity(temperature))

    def oxygen_diffusivity(self, temperature):
        """Mass diffusivity of oxygen through the (porous) wall.

        Parameters
        ----------
        temperature: meshmode.dof_array.DOFArray

        Returns
        -------
        oxygen_diffusivity: meshmode.dof_array.DOFArray
            the oxygen diffusivity inside the wall
        """
        return self._oxygen_diffusivity_func(temperature)

    def dependent_vars(self, wv, tseed):
        """Get the dependent variables."""
        tau = self.eval_tau(wv=wv)
        temperature = self.eval_temperature(wv=wv, tseed=tseed)
        kappa = self.thermal_conductivity(temperature=temperature, tau=tau)
        oxygen_diffusivity = self.oxygen_diffusivity(temperature=temperature)
        return WallDependentVars(
            tau=tau,
            thermal_conductivity=kappa,
            temperature=temperature,
            oxygen_diffusivity=oxygen_diffusivity)
