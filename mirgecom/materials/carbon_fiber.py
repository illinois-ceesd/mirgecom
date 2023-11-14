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

from abc import abstractmethod
import numpy as np
from meshmode.dof_array import DOFArray
from mirgecom.wall_model import PorousWallEOS
from pytools.obj_array import make_obj_array


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
    r"""Evaluate the properties of the solid state containing only fibers.

    The properties are obtained as a function of oxidation progress. It can
    be computed based on the mass $m$, which is related to the void fraction
    $\epsilon$ and radius $r$ as:

    .. math::
        \tau = \frac{m}{m_0} = \frac{\rho_i \epsilon}{\rho_i \epsilon_0}
             = \frac{r^2}{r_0^2}

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

    def __init__(self, dim, anisotropic_direction, char_mass, virgin_mass):
        """Bulk density considering the porosity and intrinsic density."""
        self._char_mass = char_mass
        self._virgin_mass = virgin_mass
        self._dim = dim
        self._anisotropic_dir = anisotropic_direction

        if anisotropic_direction > dim:
            raise ValueError("Anisotropic axis must be less or equal than dim.")

    def void_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Return the volumetric fraction $\epsilon$ filled with gas.

        The fractions of gas and solid phases must sum to one,
        $\epsilon_g + \epsilon_s = 1$. Both depend only on the oxidation
        progress ratio $\tau$.
        """
        return 1.0 - self.volume_fraction(tau)

    def enthalpy(self, temperature, tau=None) -> DOFArray:
        r"""Evaluate the solid enthalpy $h_s$ of the fibers."""
        return (
            - 3.37112113e-11*temperature**5 + 3.13156695e-07*temperature**4
            - 1.17026962e-03*temperature**3 + 2.29194901e+00*temperature**2
            - 3.62422269e+02*temperature**1 - 5.96993843e+04)

    def heat_capacity(self, temperature, tau=None) -> DOFArray:
        r"""Evaluate the heat capacity $C_{p_s}$ of the fibers.

        The coefficients are obtained with the analytical derivative of the
        enthalpy fit.
        """
        return (
            - 1.68556056e-10*temperature**4 + 1.25262678e-06*temperature**3
            - 3.51080885e-03*temperature**2 + 4.58389802e+00*temperature**1
            - 3.62422269e+02)

    # ~~~~~~~~ fiber conductivity
    def thermal_conductivity(self, temperature, tau=None) -> DOFArray:
        r"""Evaluate the thermal conductivity $\kappa$ of the fibers.

        It accounts for anisotropy and oxidation progress.
        """
        kappa_ij = (
            + 2.86518890e-24*temperature**5 - 2.13976832e-20*temperature**4
            + 3.36320767e-10*temperature**3 - 6.14199551e-07*temperature**2
            + 7.92469194e-04*temperature**1 + 1.18270446e-01)

        kappa_k = (
            - 1.89693642e-24*temperature**5 + 1.43737973e-20*temperature**4
            + 1.93072961e-10*temperature**3 - 3.52595953e-07*temperature**2
            + 4.54935976e-04*temperature**1 + 5.08960039e-02)

        # initialize with the in-plane value
        kappa = make_obj_array([kappa_ij for _ in range(self._dim)])
        # modify only the normal direction
        kappa[self._anisotropic_dir] = kappa_k

        # account for fiber shrinkage via "tau"
        return kappa*tau

    # ~~~~~~~~ other properties
    def volume_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Fraction $\phi$ occupied by the solid."""
        return 0.12*tau

    def permeability(self, tau: DOFArray) -> DOFArray:
        r"""Permeability $K$ of the porous material."""
        # FIXME find a relation to make it change as a function of "tau"
        actx = tau.array_context
        permeability = np.zeros(self._dim,)
        permeability[:] = 5.57e-11 + actx.np.zeros_like(tau)
        permeability[self._anisotropic_dir] = 2.62e-11 + actx.np.zeros_like(tau)
        return permeability

    def emissivity(self, temperature=None, tau=None) -> DOFArray:
        """Emissivity for energy radiation."""
        return (
            + 2.26413679e-18*temperature**5 - 2.03008004e-14*temperature**4
            + 7.05300324e-11*temperature**3 - 1.22131715e-07*temperature**2
            + 1.21137817e-04*temperature**1 + 8.66656964e-01)

    def tortuosity(self, tau: DOFArray) -> DOFArray:
        r"""Tortuosity $\eta$ affects the species diffusivity."""
        # FIXME find a relation to make it change as a function of "tau"
        actx = tau.array_context
        return 1.1 + actx.np.zeros_like(tau)

    def decomposition_progress(self, mass: DOFArray) -> DOFArray:
        r"""Evaluate the mass loss progress ratio $\tau$ of the oxidation."""
        return mass/self._virgin_mass
