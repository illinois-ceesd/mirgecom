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

import numpy as np
from pytools.obj_array import make_obj_array
from mirgecom.fluid import make_conserved
from mirgecom.wall_model import SolidWallConservedVars
from mirgecom.wall_model import PorousWallVars


class SolidWallInitializer:
    """State initializer for wall models solving heat-diffusion equation.

    This class computes the initial condition for either solid or porous
    materials, and/or their combination, subject or not to ablation.
    """

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

        Returns
        -------
        wv: :class:`mirgecom.wall_model.SolidWallConservedVars`
            The conserved variables for heat-conduction only materials.
        """
        actx = x_vec[0].array_context
        mass = wall_model.density() + actx.np.zeros_like(x_vec[0])
        energy = mass * wall_model.enthalpy(self._temp)
        return SolidWallConservedVars(mass=mass, energy=energy)


class PorousWallInitializer:
    """State initializer for porous materials in the unified-domain solver."""

    def __init__(
            self, *, temperature, material_densities, species_mass_fractions=None,
            velocity=None, pressure=None, density=None, porous_region=None):
        """Initialize the object for porous materials.

        Parameters
        ----------
        porous_region:
            Field describing the homogeneous fluid (0) and porous material (1)
            portions of the flow. Only used for unified-domain solver without
            explicit coupling.
        """
        self._velocity = velocity
        self._pres = pressure
        self._mass = density
        self._temp = temperature
        self._wall_density = material_densities
        self._porous_region = porous_region

        if species_mass_fractions is not None:
            self._y = species_mass_fractions
        else:
            self._y = np.empty((0,), dtype=object)

    def __call__(self, x_vec, gas_model, return_wv=False):
        """Evaluate the wall+gas properties for porous materials.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Nodal coordinates
        gas_model: :class:`mirgecom.wall_model.PorousFlowModel`
            Equation of state class

        Returns
        -------
        cv: :class:`mirgecom.fluid.ConservedVars`
            The conserved variables for porous-media flows. It depends on
            both gas and porous material properties.
        wall_density: numpy.ndarray or :class:`meshmode.dof_array.DOFArray`
            The densities of each one of the materials
        wv: :class:`mirgecom.fluid.wall_model.PorousWallVars`
            Wall dependent variables
        """
        actx = x_vec[0].array_context
        zeros = actx.np.zeros_like(x_vec[0])
        ones = zeros + 1.0
        dim = x_vec.shape[0]

        if self._porous_region is not None:
            porous_region = self._porous_region + zeros
        else:
            porous_region = ones

        # wall-only properties
        material_densities = self._wall_density * porous_region
        tau = gas_model.decomposition_progress(material_densities)
        eps_rho_solid = gas_model.solid_density(material_densities)

        wv = PorousWallVars(
            material_densities=material_densities,
            tau=tau,
            density=eps_rho_solid,
            void_fraction=gas_model.wall_eos.void_fraction(tau=tau),
            permeability=gas_model.wall_eos.permeability(tau=tau),
            tortuosity=gas_model.wall_eos.tortuosity(tau=tau)
        )

        # coupled properties
        temperature = self._temp + zeros

        nspecies = len(self._y)
        if nspecies > 0:
            species_mass_frac = make_obj_array([self._y[i] + zeros
                                                for i in range(nspecies)])
        else:
            species_mass_frac = self._y

        eps_gas = gas_model.wall_eos.void_fraction(tau)
        if self._mass is None:
            pressure = self._pres + zeros
            eps_rho_gas = eps_gas*gas_model.eos.get_density(
                pressure, temperature, species_mass_frac)
        else:
            eps_rho_gas = eps_gas*self._mass

        # TODO gotta define better the velocity..
        if self._velocity is not None:
            velocity = self._velocity/eps_gas
            momentum = make_obj_array([eps_rho_gas*velocity[i] for i in range(dim)])
            # momentum = (1.0-self._porous_region) * (eps_rho_gas*self._velocity)
        else:
            momentum = make_obj_array([zeros for _ in range(dim)])

        bulk_energy = (
            eps_rho_solid*gas_model.wall_eos.enthalpy(temperature, tau)
            + eps_rho_gas*gas_model.eos.get_internal_energy(temperature,
                                                            species_mass_frac)
            + 0.5*np.dot(momentum, momentum)/eps_rho_gas
        )

        species_mass = eps_rho_gas*species_mass_frac

        cv = make_conserved(dim=dim, mass=eps_rho_gas, energy=bulk_energy,
                            momentum=momentum, species_mass=species_mass)

        if return_wv:
            return cv, wv
        return cv, wv.material_densities
