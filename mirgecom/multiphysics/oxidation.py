""":mod:`mirgecom.multiphysics.oxidation` handles Y3 oxidation model."""

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
    r"""Class of conserved variables for wall oxidation.

    .. attribute:: solid_mass
    .. attribute:: species_mass
    .. attribute:: energy
    """

    solid_mass: DOFArray
    energy: DOFArray
    species_mass: numpy.ndarray

    @property
    def array_context(self):
        """Return an array context for the :class:`WallVars` object."""
        return get_container_context_recursively(self.energy)

    def __reduce__(self):
        """Return a tuple reproduction of self for pickling."""
        return (WallVars, tuple(getattr(self, f.name) for f in fields(WallVars)))


@dataclass_array_container
@dataclass(frozen=True)
class WallDependentVars:
    """Dependent variables for the Y3 oxidation model."""

    tau: DOFArray
    thermal_conductivity: DOFArray
    species_diffusivity: DOFArray
    species_viscosity: DOFArray
    temperature: DOFArray


class OxidationWallModel:
    """Oxidation model used for the Y3 oxidation model.

    This class evaluates the variables dependent on the wall state. It requires
    functions are inputs in order to consider the different materials employed
    at wall (for instance, carbon fibers, graphite, alumina, steel etc). The
    number and type of material is case dependent and, hence, must be defined
    at the respective simulation driver.

    .. automethod:: __init__
    .. automethod:: eval_tau
    .. automethod:: eval_temperature
    .. automethod:: enthalpy
    .. automethod:: heat_capacity
    .. automethod:: thermal_conductivity
    .. automethod:: thermal_diffusivity
    .. automethod:: species_diffusivity
    """

    def __init__(self, solid_data, gas_data, wall_sample_mask,
                 enthalpy_func, heat_capacity_func, thermal_conductivity_func,
                 species_diffusivity_func):
        """
        Initialize the model.

        Parameters
        ----------
        solid_data:
            The class with the solid properties of the desired material.
        wall_sample_mask: meshmode.dof_array.DOFArray
            Array with 1 for the reactive part of the wall and 0 otherwise.
        enthalpy_func:
            function that computes the enthalpy of the entire wall.
            Must include the non-reactive part of the wall, if existing.
        heat_capacity_func:
            function that computes the heat capacity of the entire wall
            Must include the non-reactive part of the wall, if existing.
        thermal_conductivity_func:
            function that computes the thermal conductivity of the entire wall.
            Must include the non-reactive part of the wall, if existing.
        species_diffusivity_func:
            function that computes the species mass diffusivity inside the wall.
            Must include the non-reactive part of the wall, if existing.
        """
        self._fiber = solid_data
        self._sample_mask = wall_sample_mask
        self._enthalpy_func = enthalpy_func
        self._heat_capacity_func = heat_capacity_func
        self._thermal_conductivity_func = thermal_conductivity_func
        self._species_diffusivity_func = species_diffusivity_func

    def eval_tau(self, wv) -> DOFArray:
        r"""Evaluate the progress ratio $\tau$ of the oxidation.

        Where $\tau=1$, the material is locally virgin. On the other hand, if
        $\tau=0$, then the fibers were all consumed.

        Parameters
        ----------
        wv: :class:`WallVars`
            the class of conserved variables for the oxidation
        """
        virgin = (self._fiber.intrinsic_density()
                  * self._fiber.solid_volume_fraction(tau=1.0))
        return (
            (1.0 - (virgin - wv.mass)/virgin) * self._sample_mask  # fiber
            + 1.0*(1.0 - self._sample_mask)  # inert
        )

    def eval_temperature(self, wv, tseed) -> DOFArray:
        r"""Evaluate the temperature using Newton iteration.

        .. math::
            T^{n+1} = T^n - \frac{h_s(T^n) - e(t)}{C_{p_s}(T^n)}

        where $h_s(T^n)$ and $C_{p_s}(T^n)$ are evaluated every iteration $n$
        until convergence towards the conserved energy $e(t)$. Note that, for
        solids, both enthalpy and internal are exactly the same.

        Parameters
        ----------
        wv: :class:`WallVars`
        tseed: numbers.Number or meshmode.dof_array.DOFArray
            Initial guess for the temperature
        """
        if isinstance(tseed, DOFArray) is False:
            temp = tseed + wv.energy*0.0
        else:
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
        """Return the heat capacity of the wall.

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
        """Return the effective thermal conductivity of the wall.

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

    def species_diffusivity(self, temperature):
        """Mass diffusivity of gaseous species through the (porous) wall.

        Parameters
        ----------
        temperature: meshmode.dof_array.DOFArray

        Returns
        -------
        species_diffusivity: meshmode.dof_array.DOFArray
            the species mass diffusivity inside the wall
        """
        return self._species_diffusivity_func(temperature)

    def dependent_vars(self, wv, tseed):
        """Get the dependent variables."""
        tau = self.eval_tau(wv=wv)
        temperature = self.eval_temperature(wv=wv, tseed=tseed)
        kappa = self.thermal_conductivity(temperature=temperature, tau=tau)
        species_diffusivity = self.species_diffusivity(temperature=temperature)
        return WallDependentVars(
            tau=tau,
            thermal_conductivity=kappa,
            temperature=temperature,
            species_diffusivity=species_diffusivity)
