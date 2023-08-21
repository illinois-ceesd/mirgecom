""":mod:`~mirgecom.materials.initializer` returns wall materials initialization.

.. autoclass:: SolidWallInitializer
.. autoclass:: PorousWallInitializer
"""

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

from pytools.obj_array import make_obj_array
from mirgecom.fluid import make_conserved
from mirgecom.wall_model import SolidWallConservedVars


class SolidWallInitializer:
    """Initializer for heat conduction only materials."""

    def __init__(self, temperature):
        self._temp = temperature

    def __call__(self, x_vec, wall_model):
        """Evaluate the wall+gas properties for porous materials.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Nodal coordinates
        wall_model: :class:`mirgecom.wall_model.SolidWallModel`
            Equation of state class
        """
        actx = x_vec[0].array_context
        mass = wall_model.density() + actx.np.zeros_like(x_vec[0])
        energy = mass * wall_model.enthalpy(self._temp)
        return SolidWallConservedVars(mass=mass, energy=energy)


class PorousWallInitializer:
    """Initializer for porous materials."""

    def __init__(self, temperature, species, material_densities,
                 pressure=None, density=None):

        self._pres = pressure
        self._mass = density
        self._y = species
        self._temp = temperature
        self._wall_density = material_densities

    def __call__(self, dim, x_vec, gas_model):
        """Evaluate the wall+gas properties for porous materials.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Nodal coordinates
        gas_model: :class:`mirgecom.wall_model.PorousFlowModel`
            Equation of state class
        """
        actx = x_vec[0].array_context
        zeros = actx.np.zeros_like(x_vec[0])

        temperature = self._temp + zeros
        species_mass_frac = self._y + zeros
        wall_density = self._wall_density + zeros

        tau = gas_model.decomposition_progress(wall_density)

        eps_gas = gas_model.wall_eos.void_fraction(tau)
        if self._mass is None:
            pressure = self._pres + zeros
            eps_rho_gas = eps_gas*gas_model.eos.get_density(pressure,
                temperature, species_mass_frac)
        else:
            density = self._mass + zeros
            eps_rho_gas = eps_gas*density

        # internal energy (kinetic energy is neglected)
        eps_rho_solid = sum(wall_density)
        bulk_energy = (
            eps_rho_solid*gas_model.wall_eos.enthalpy(temperature, tau)
            + eps_rho_gas*gas_model.eos.get_internal_energy(temperature,
                                                            species_mass_frac)
        )

        momentum = make_obj_array([zeros, zeros])

        species_mass = eps_rho_gas*species_mass_frac

        return make_conserved(dim=dim, mass=eps_rho_gas, energy=bulk_energy,
            momentum=momentum, species_mass=species_mass)
