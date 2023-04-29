r""":mod:`mirgecom.materials.carbon_fiber` evaluate carbon fiber data.

.. autoclass:: Oxidation
.. autoclass:: SolidProperties
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

from meshmode.dof_array import DOFArray


mw_o = 15.999
mw_o2 = mw_o*2
mw_co = 28.010
univ_gas_const = 8314.46261815324


class Oxidation():
    """Evaluate the source terms for the oxidation of carbon fibers."""

    def puma_effective_surface_area(self, progress):
        """Polynomial fit based on PUMA data."""
        # Original fit function: -1.1012e5*x**2 - 0.0646e5*x + 1.1794e5
        # Rescale by x==0 value and rearrange
        return 1.1794e5*(1.0 - 0.0547736137*progress - 0.9336950992*progress**2)

    def _get_wall_effective_surface_area_fiber(self, progress):
        """Evaluate the effective surface of the fibers."""
        return self.puma_effective_surface_area(progress)

    def get_source_terms(self, temperature, tau, ox_mass):
        """Return the effective source terms for the oxidation."""
        actx = temperature.array_context
        progress = 1.0-tau
        eff_surf_area = self._get_wall_effective_surface_area_fiber(progress)
        alpha = (
            (0.00143+0.01*actx.np.exp(-1450.0/temperature))
            / (1.0+0.0002*actx.np.exp(13000.0/temperature)))
        k = alpha*actx.np.sqrt(
            (univ_gas_const*temperature)/(2.0*np.pi*mw_o2))
        return (mw_co/mw_o2 + mw_o/mw_o2 - 1)*ox_mass*k*eff_surf_area


class GasProperties():
    """Evaluate properties of the gas permeating the fibers."""

    def __init__(self, pyrometheus_mechanism):
        self._pyro = pyrometheus_mechanism

    def oxygen_diffusivity(self, temperature):
        """Return the mass diffusivity of the gaseous species."""
        return 1e-4 + temperature*0.0


class SolidProperties():
    """Model for calculating wall quantities."""

    def intrinsic_density(self):
        """Return the intrinsic density of the fibers."""
        return 1600.0

    def solid_heat_capacity(self, temperature: DOFArray) -> DOFArray:
        """Evaluate the heat capacity of the fibers."""
        return (
            + 1.461303669323e-14*temperature**5 - 1.862489701581e-10*temperature**4
            + 9.685398830530e-07*temperature**3 - 2.599755262540e-03*temperature**2
            + 3.667295510844e+00*temperature - 7.816218435655e+01)

    def solid_enthalpy(self, temperature) -> DOFArray:
        """Evaluate the solid enthalpy of the fibers."""
        return (
            - 1.279887694729e-11*temperature**5 + 1.491175465285e-07*temperature**4
            - 6.994595296860e-04*temperature**3 + 1.691564018108e+00*temperature**2
            - 3.441837408320e+01*temperature - 1.235438104496e+05)

    # ~~~~~~~~ fiber conductivity
    def experimental_kappa(self, temperature):
        """Experimental data of thermal conductivity."""
        return (
            1.766e-10 * temperature**3
            - 4.828e-7 * temperature**2
            + 6.252e-4 * temperature
            + 6.707e-3)

    def puma_kappa(self, progress):
        """Numerical data of thermal conductivity evaluated by PUMA."""
        return 0.0988*progress**2 - 0.2751*progress + 0.201

    def solid_thermal_conductivity(self, temperature: DOFArray,
                                   tau: DOFArray) -> DOFArray:
        """Evaluate the thermal conductivity of the fibers.

        It employs a rescaling of the experimental data based on the fiber
        shrinkage during the oxidation.
        """
        progress = 1.0-tau
        return (
            self.experimental_kappa(temperature)
            * self.puma_kappa(progress) / self.puma_kappa(0)
        )

    def solid_volume_fraction(self, tau) -> DOFArray:
        """Void fraction filled by gas around the fibers."""
        return 0.10*tau

    def solid_emissivity(self, tau):
        """Emissivity for energy radiation."""
        return 0.9 + tau*0.0
