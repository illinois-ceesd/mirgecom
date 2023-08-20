r""":mod:`mirgecom.materials.carbon_fiber` evaluate carbon fiber data.

.. autoclass:: Oxidation
.. autoclass:: FiberEOS
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

from typing import Optional
from abc import abstractmethod
import numpy as np
from meshmode.dof_array import DOFArray
from mirgecom.wall_model import PorousWallEOS


class Oxidation:
    """Abstract interface for wall oxidation model.

    .. automethod:: get_source_terms
    """

    @abstractmethod
    def get_source_terms(self, temperature: DOFArray,
            tau: DOFArray, rhoY_o2: DOFArray) -> DOFArray:  # noqa N803
        r"""Source terms of fiber oxidation."""
        raise NotImplementedError()


# TODO per MTC review, can we generalize the oxidation model?
# should we keep this in the driver?
class Y2_Oxidation_Model(Oxidation):  # noqa N801
    """Evaluate the source terms for the Y2 model of carbon fiber oxidation.

    .. automethod:: puma_effective_surface_area
    .. automethod:: get_source_terms
    """

    def puma_effective_surface_area(self, progress) -> DOFArray:
        """Polynomial fit based on PUMA data.

        Parameters
        ----------
        progress: meshmode.dof_array.DOFArray
            the rate of decomposition of the fibers
        """
        # Original fit function: -1.1012e5*x**2 - 0.0646e5*x + 1.1794e5
        # Rescale by x==0 value and rearrange
        return 1.1794e5*(1.0 - 0.0547736137*progress - 0.9336950992*progress**2)

    def _get_wall_effective_surface_area_fiber(self, progress) -> DOFArray:
        """Evaluate the effective surface of the fibers.

        Parameters
        ----------
        progress: meshmode.dof_array.DOFArray
            the rate of decomposition of the fibers
        """
        return self.puma_effective_surface_area(progress)

    def get_source_terms(self, temperature, tau, rhoY_o2) -> DOFArray:  # noqa N803
        """Return the effective source terms for the oxidation.

        Parameters
        ----------
        temperature: meshmode.dof_array.DOFArray
        tau: meshmode.dof_array.DOFArray
            the progress ratio of the oxidation
        ox_mass: meshmode.dof_array.DOFArray
            the mass fraction of oxygen
        """
        actx = temperature.array_context

        mw_o = 15.999
        mw_o2 = mw_o*2
        mw_co = 28.010
        univ_gas_const = 8314.46261815324

        eff_surf_area = self._get_wall_effective_surface_area_fiber(1.0-tau)
        alpha = (
            (0.00143+0.01*actx.np.exp(-1450.0/temperature))
            / (1.0+0.0002*actx.np.exp(13000.0/temperature)))
        k = alpha*actx.np.sqrt(
            (univ_gas_const*temperature)/(2.0*np.pi*mw_o2))
        return (mw_co/mw_o2 + mw_o/mw_o2 - 1)*rhoY_o2*k*eff_surf_area


class FiberEOS(PorousWallEOS):
    """Evaluate the properties of the solid state containing only fibers.

    .. automethod:: void_fraction
    .. automethod:: enthalpy
    .. automethod:: heat_capacity
    .. automethod:: thermal_conductivity
    .. automethod:: volume_fraction
    .. automethod:: permeability
    .. automethod:: emissivity
    .. automethod:: tortuosity
    .. automethod:: decomposition_progress
    """

    def __init__(self, char_mass, virgin_mass):
        """Bulk density considering the porosity and intrinsic density."""
        self._char_mass = char_mass
        self._virgin_mass = virgin_mass

    def void_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Return the volumetric fraction $\epsilon$ filled with gas.

        The fractions of gas and solid phases must sum to one,
        $\epsilon_g + \epsilon_s = 1$. Both depend only on the oxidation
        progress ratio $\tau$.
        """
        return 1.0 - self.volume_fraction(tau)

    def enthalpy(self, temperature: DOFArray, tau: Optional[DOFArray]) -> DOFArray:
        r"""Evaluate the solid enthalpy $h_s$ of the fibers."""
        return (
            - 1.279887694729e-11*temperature**5 + 1.491175465285e-07*temperature**4
            - 6.994595296860e-04*temperature**3 + 1.691564018108e+00*temperature**2
            - 3.441837408320e+01*temperature - 1.235438104496e+05)

    def heat_capacity(self, temperature: DOFArray,
                      tau: Optional[DOFArray]) -> DOFArray:
        r"""Evaluate the heat capacity $C_{p_s}$ of the fibers."""
        return (
            + 1.461303669323e-14*temperature**5 - 1.862489701581e-10*temperature**4
            + 9.685398830530e-07*temperature**3 - 2.599755262540e-03*temperature**2
            + 3.667295510844e+00*temperature - 7.816218435655e+01)

    # ~~~~~~~~ fiber conductivity
    def _experimental_kappa(self, temperature: DOFArray) -> DOFArray:
        """Experimental data of thermal conductivity."""
        return (
            1.766e-10 * temperature**3
            - 4.828e-7 * temperature**2
            + 6.252e-4 * temperature
            + 6.707e-3)

    def _puma_kappa(self, progress: DOFArray) -> DOFArray:
        """Numerical data of thermal conductivity evaluated by PUMA."""
        # FIXME the fully decomposed conductivity is given by the air, which
        # is already accounted for in our model.
        return 0.0988*progress**2 - 0.2751*progress + 0.201

    def thermal_conductivity(self, temperature: DOFArray,
                                   tau: DOFArray) -> DOFArray:
        r"""Evaluate the thermal conductivity $\kappa$ of the fibers.

        It employs a rescaling of the experimental data based on the fiber
        shrinkage during the oxidation.
        """
        progress = 1.0-tau
        return (
            self._experimental_kappa(temperature)
            * self._puma_kappa(progress) / self._puma_kappa(progress=0.0)
        )

    # ~~~~~~~~ other properties
    def volume_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Fraction $\phi$ occupied by the solid."""
        # FIXME Should this be a quadratic function?
        return 0.10*tau

    def permeability(self, tau: DOFArray) -> DOFArray:
        r"""Permeability $K$ of the porous material."""
        # FIXME find a relation to make it change as a function of "tau"
        actx = tau.array_context
        return 6.0e-11 + actx.np.zeros_like(tau)

    def emissivity(self, tau: DOFArray) -> DOFArray:
        """Emissivity for energy radiation."""
        actx = tau.array_context
        return 0.9 + actx.np.zeros_like(tau)

    def tortuosity(self, tau: DOFArray) -> DOFArray:
        r"""Tortuosity $\eta$ affects the species diffusivity."""
        # FIXME find a relation to make it change as a function of "tau"
        actx = tau.array_context
        return 1.1 + actx.np.zeros_like(tau)

    def decomposition_progress(self, mass: DOFArray) -> DOFArray:
        r"""Evaluate the mass loss progress ratio $\tau$ of the oxidation.

        Where $\tau=1$, the material is locally virgin. On the other hand, if
        $\tau=0$, then the oxidation is locally complete and the fibers were
        all consumed.
        """
        return 1.0 - (self._virgin_mass - mass)/self._virgin_mass
