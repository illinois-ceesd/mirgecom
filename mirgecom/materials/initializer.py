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

    def __call__(self, actx, x_vec, wall_model):
        mass = wall_model.density()
        energy = mass * wall_model.enthalpy(self._temp)
        return SolidWallConservedVars(mass=mass, energy=energy)


class PorousWallInitializer:
    """Initializer for porous materials."""

    def __init__(self, pressure, temperature, species, material_densities):

        self._pres = pressure
        self._y = species
        self._temp = temperature
        self._wall_density = material_densities

    def __call__(self, actx, x_vec, gas_model):

        zeros = actx.np.zeros_like(x_vec[0])

        pressure = self._pres + zeros
        temperature = self._temp + zeros
        species_mass_frac = self._y + zeros

        tau = gas_model.wall.decomposition_progress(self._wall_density)

        eps_gas = gas_model.wall.void_fraction(tau)
        eps_rho_gas = eps_gas*gas_model.eos.get_density(pressure, temperature,
                                                        species_mass_frac)

        # internal energy (kinetic energy is neglected)
        eps_rho_solid = sum(self._wall_density)
        bulk_energy = (
            eps_rho_solid*gas_model.wall.enthalpy(temperature, tau)
            + eps_rho_gas*gas_model.eos.get_internal_energy(temperature,
                                                            species_mass_frac)
        )

        momentum = make_obj_array([zeros, zeros])

        species_mass = eps_rho_gas*species_mass_frac

        return make_conserved(dim=2, mass=eps_rho_gas, energy=bulk_energy,
            momentum=momentum, species_mass=species_mass)
