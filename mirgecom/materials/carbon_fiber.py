r""":mod:`mirgecom.materials.carbon_fiber` evaluate carbon fiber data.

.. autoclass:: Oxidation
.. autoclass:: Y2_Oxidation_Model
.. autoclass:: Y3_Oxidation_Model
.. autoclass:: OxidationModel
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

from typing import Optional, Union
from abc import abstractmethod
import numpy as np
from pytools.obj_array import make_obj_array
from meshmode.dof_array import DOFArray
from mirgecom.wall_model import PorousWallEOS


class Oxidation:
    """Abstract interface for wall oxidation models.

    .. automethod:: get_source_terms
    """

    @abstractmethod
    def get_source_terms(
            self, temperature: DOFArray, tau: DOFArray, rhoY_o2: DOFArray,  # noqa N803
            rhoY_o: Optional[DOFArray] = None) -> Union[DOFArray, tuple]:
        r"""Source terms of fiber oxidation.

        Parameters
        ----------
        temperature:
            gas and solid temperature assuming local-equilibrium
        tau:
            the progress ratio of the oxidation: 1 for virgin, 0 for fully oxidized
        rhoY_o2:
            the mass fraction of molecular oxygen
        rhoY_o:
            the mass fraction of atomic oxygen
        """
        raise NotImplementedError()


class Y2_Oxidation_Model(Oxidation):  # noqa N801
    r"""Evaluate the source terms for carbon fiber oxidation in Y2 model.

    .. automethod:: get_source_terms
    """

    def _get_wall_effective_surface_area_fiber(self, tau: DOFArray) -> DOFArray:
        """Evaluate the effective surface of the fibers."""
        # Polynomial fit based on PUMA data:
        # Original fit function: -1.1012e5*x**2 - 0.0646e5*x + 1.1794e5
        # Rescale by x==0 value and rearrange
        progress = 1.0-tau
        return 1.1794e5*(1.0 - 0.0547736137*progress - 0.9336950992*progress**2)

    def get_source_terms(self, temperature: DOFArray, tau: DOFArray,
                         rhoY_o2: DOFArray, rhoY_o=None) -> DOFArray:  # noqa N803
        """Return the effective source terms for fiber oxidation."""
        actx = temperature.array_context

        mw_o = 15.999
        mw_o2 = mw_o*2
        mw_co = 28.010
        univ_gas_const = 8314.46261815324

        eff_surf_area = self._get_wall_effective_surface_area_fiber(tau)
        alpha = (
            (0.00143+0.01*actx.np.exp(-1450.0/temperature))
            / (1.0+0.0002*actx.np.exp(13000.0/temperature)))
        k = alpha*actx.np.sqrt(
            (univ_gas_const*temperature)/(2.0*np.pi*mw_o2))
        return (mw_co/mw_o2 + mw_o/mw_o2 - 1)*rhoY_o2*k*eff_surf_area


class Y3_Oxidation_Model(Oxidation):  # noqa N801
    r"""Evaluate the source terms for carbon fiber oxidation in Y3 model.

    Follows [Martin_2013]_ by using a single reaction given by

    .. math::
        C_{(s)} + O_2 \to CO_2

    The degradation is computed as

    .. math::
        \frac{\partial \rho_s}{\partial t} =
        \rho_C \frac{\partial \epsilon}{\partial t} = \dot{\omega}_\text{het}

    The fiber radius as a function of mass loss $\tau$ is given by

    .. math::
        \tau = \frac{m}{m_0} = \frac{\epsilon}{\epsilon_0} =
        \frac{\pi r^2/L}{\pi r_0^2/L} = \frac{r^2}{r_0^2}

    and leads to the wall recession as

    .. math::
        \frac{\partial \epsilon}{\partial t} =
        \frac{\partial \epsilon}{\partial r} \frac{\partial r}{\partial t} =
        2 \frac{\epsilon_0}{r_0^2} r = 2 \frac{\epsilon_0}{r_0} \sqrt{\tau}

    .. automethod:: __init__
    .. automethod:: get_source_terms
    """

    def __init__(self, wall_material, initial_fiber_radius=5.5e-6, arrhenius=1e5,
                 activation_energy=-120000.0):
        self._material = wall_material
        self._init_fiber_radius = initial_fiber_radius
        self._arrhenius = arrhenius
        self._Ea = activation_energy

    def _get_wall_effective_surface_area_fiber(self, tau) -> DOFArray:
        r"""Evaluate the effective surface of the fibers."""
        actx = tau.array_context
        zeros = actx.np.zeros_like(tau)
        return (
            2.0 * self._material.volume_fraction(tau=1.0)/self._init_fiber_radius
            * actx.np.sqrt(actx.np.maximum(tau, zeros))  # only use positive tau
        )

    def get_source_terms(self, temperature: DOFArray, tau: DOFArray,
            rhoY_o2: DOFArray, rhoY_o=None):  # noqa N803
        r"""Return the effective source terms for the oxidation.

        Returns
        -------
        sources: tuple
            the tuple ($\omega_{C}$, $\omega_{O_2}$, $\omega_{CO_2}$)
        """
        actx = temperature.array_context

        # ignore negative amounts of O2
        rhoY_o2 = actx.np.maximum(rhoY_o2, actx.np.zeros_like(rhoY_o2))  # noqa N806

        mw_c = 12.011
        mw_o = 15.999
        mw_o2 = mw_o*2
        mw_co2 = 44.010
        univ_gas_const = 8.31446261815324  # J/(K-mol)

        eff_surf_area = self._get_wall_effective_surface_area_fiber(tau)

        k_f = self._arrhenius*actx.np.exp(self._Ea/(univ_gas_const*temperature))

        m_dot_c = - rhoY_o2/mw_o2 * mw_c * eff_surf_area * k_f
        m_dot_o2 = - rhoY_o2/mw_o2 * mw_o2 * eff_surf_area * k_f
        m_dot_co2 = + rhoY_o2/mw_o2 * mw_co2 * eff_surf_area * k_f

        return m_dot_c, m_dot_o2, m_dot_co2


class OxidationModel(Oxidation):
    """Evaluate the source terms for the carbon fiber oxidation.

    The user must specify the functions for oxidation in the driver.
    (Tentatively) Generalizing this class makes it easier for adding new,
    more complex models and for UQ runs.

    .. automethod:: get_source_terms
    """

    def __init__(self, surface_area_func, oxidation_func):
        """Initialize the user-specified oxidation class.

        Parameters
        ----------
        surface_area_func:
            Function prescribing how the fiber area changes during oxidation.
        oxidation_func:
            Reaction rate for the oxidation model.
        """
        self._surface_func = surface_area_func
        self._oxidation_func = oxidation_func

    def get_source_terms(
            self, temperature: DOFArray, tau: DOFArray, rhoY_o2: DOFArray,  # noqa N803
            rhoY_o: Optional[DOFArray] = None) -> DOFArray:
        """Return the effective source terms for the oxidation."""
        area = self._surface_func(tau)
        if rhoY_o is None:
            return self._oxidation_func(temperature=temperature, fiber_area=area,
                                        rhoY_o2=rhoY_o2)
        return self._oxidation_func(temperature=temperature, fiber_area=area,
                                    rhoY_o2=rhoY_o2, rhoY_o=rhoY_o)


class FiberEOS(PorousWallEOS):
    r"""Evaluate the properties of the solid state containing only fibers.

    The properties are obtained as a function of oxidation progress. It can
    be computed based on the mass $m$, which is related to the void fraction
    $\epsilon$ and radius $r$ as:

    .. math::
        \tau = \frac{m}{m_0} = \frac{\rho_i \epsilon}{\rho_i \epsilon_0}
             = \frac{r^2}{r_0^2}

    .. automethod:: __init__
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

    def __init__(self, dim, anisotropic_direction, char_mass, virgin_mass,
                 timescale=1.0):
        """Bulk density considering the porosity and intrinsic density.

        Parameters
        ----------
        dim: int
            geometrical dimension of the problem.
        anisotropic_direction: int
            For orthotropic materials, this indicates the normal direction
            where the properties are different than in-plane.
        char_mass: float
            final mass when the decomposition is complete.
        virgin_mass: float
            initial mass of the material.
        timescale: float
            Modifies the thermal conductivity and the radiation emission to
            increase/decrease the wall time-scale. Defaults to 1.0 (no changes).
        """
        self._char_mass = char_mass
        self._virgin_mass = virgin_mass
        self._dim = dim
        self._anisotropic_dir = anisotropic_direction
        self._timescale = timescale

        if anisotropic_direction >= dim:
            raise ValueError("Anisotropic axis must be less than dim.")

    def void_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Return the volumetric fraction $\epsilon$ filled with gas.

        The fractions of gas and solid phases must sum to one,
        $\epsilon_g + \epsilon_s = 1$. Both depend only on the oxidation
        progress ratio $\tau$.
        """
        return 1.0 - self.volume_fraction(tau)

    def enthalpy(self, temperature: DOFArray,
                 tau: Optional[DOFArray] = None) -> DOFArray:
        r"""Evaluate the solid enthalpy $h_s$ of the fibers."""
        return (
            - 3.37112113e-11*temperature**5 + 3.13156695e-07*temperature**4
            - 1.17026962e-03*temperature**3 + 2.29194901e+00*temperature**2
            - 3.62422269e+02*temperature**1 - 5.96993843e+04)

    def heat_capacity(self, temperature: DOFArray,
                      tau: Optional[DOFArray] = None) -> DOFArray:
        r"""Evaluate the heat capacity $C_{p_s}$ of the fibers.

        The coefficients are obtained with the analytical derivative of the
        enthalpy fit.
        """
        return (
            - 1.68556056e-10*temperature**4 + 1.25262678e-06*temperature**3
            - 3.51080885e-03*temperature**2 + 4.58389802e+00*temperature**1
            - 3.62422269e+02)

    # ~~~~~~~~ fiber conductivity
    def thermal_conductivity(self, temperature, tau) -> np.ndarray:
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

        # initialize with the in-plane value then modify the normal direction
        kappa = make_obj_array([kappa_ij for _ in range(self._dim)])
        kappa[self._anisotropic_dir] = kappa_k

        # account for fiber shrinkage via "tau"
        # XXX check if here is the best place for timescale
        return kappa*tau*self._timescale

    # ~~~~~~~~ other properties
    def volume_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Fraction $\phi$ occupied by the solid."""
        return 0.12*tau

    def permeability(self, tau: DOFArray) -> np.ndarray:
        r"""Permeability $K$ of the porous material."""
        # FIXME find a relation to make it change as a function of "tau"
        # TODO: the relation depends on the coupling model. Postpone it for now.
        actx = tau.array_context
        permeability = make_obj_array([5.57e-11 + actx.np.zeros_like(tau)
                                       for _ in range(0, self._dim)])
        permeability[self._anisotropic_dir] = 2.62e-11 + actx.np.zeros_like(tau)

        return permeability

    def emissivity(self, temperature: DOFArray,  # type: ignore[override]
                   tau: Optional[DOFArray] = None) -> DOFArray:
        """Emissivity for energy radiation."""
        # XXX check if here is the best place for timescale
        return self._timescale * (
            + 2.26413679e-18*temperature**5 - 2.03008004e-14*temperature**4
            + 7.05300324e-11*temperature**3 - 1.22131715e-07*temperature**2
            + 1.21137817e-04*temperature**1 + 8.66656964e-01)

    def tortuosity(self, tau: DOFArray) -> DOFArray:
        r"""Tortuosity $\eta$ affects the species diffusivity.

        .. math:
            D_{eff} = \frac{D_i^(m)}{\eta}
        """
        return 1.1*tau + 1.0*(1.0 - tau)

    def decomposition_progress(self, mass: DOFArray) -> DOFArray:
        r"""Evaluate the mass loss progress ratio $\tau$ of the oxidation."""
        return mass/self._virgin_mass
