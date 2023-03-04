r"""
:mod:`mirgecom.transport` provides methods/utils for transport properties.

Transport Models
^^^^^^^^^^^^^^^^
This module is designed provide Transport Model objects used to compute and
manage the transport properties in viscous flows.  The transport properties
currently implemented are the dynamic viscosity ($\mu$), the bulk viscosity
($\mu_{B}$), the thermal conductivity ($\kappa$), and the species diffusivities
($d_{\alpha}$).

.. autoclass:: GasTransportVars
.. autoclass:: TransportModel
.. autoclass:: SimpleTransport
.. autoclass:: PowerLawTransport
.. autoclass:: MixtureAveragedTransport
.. autoclass:: ArtificialViscosityTransportDiv

Exceptions
^^^^^^^^^^
.. autoexception:: TransportModelError
"""

__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
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
from dataclasses import dataclass
from arraycontext import dataclass_array_container
import numpy as np
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.dof_array import DOFArray
from mirgecom.fluid import ConservedVars
from mirgecom.eos import GasEOS, GasDependentVars


class TransportModelError(Exception):
    """Indicate that transport model is required for model evaluation."""

    pass


@dataclass_array_container
@dataclass(frozen=True)
class GasTransportVars:
    """State-dependent quantities for :class:`TransportModel`.

    Prefer individual methods for model use, use this
    structure for visualization or probing.

    .. attribute:: bulk_viscosity
    .. attribute:: viscosity
    .. attribute:: volume_viscosity
    .. attribute:: thermal_conductivity
    .. attribute:: species_diffusivity
    """

    bulk_viscosity: np.ndarray
    viscosity: np.ndarray
    volume_viscosity: np.ndarray
    thermal_conductivity: np.ndarray
    species_diffusivity: np.ndarray


class TransportModel:
    r"""Abstract interface to thermo-diffusive transport model class.

    Transport model classes are responsible for
    computing relations between fluid or gas state variables and
    thermo-diffusive transport properties for those fluids.

    .. automethod:: bulk_viscosity
    .. automethod:: viscosity
    .. automethod:: thermal_conductivity
    .. automethod:: species_diffusivity
    .. automethod:: volume_viscosity
    .. automethod:: transport_vars
    """

    def bulk_viscosity(self, cv: ConservedVars,
                       dv: Optional[GasDependentVars] = None,
                       eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the bulk viscosity for the gas (${\mu}_{B}$)."""
        raise NotImplementedError()

    def viscosity(self, cv: ConservedVars,
                  dv: Optional[GasDependentVars] = None,
                  eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the gas dynamic viscosity, $\mu$."""
        raise NotImplementedError()

    def volume_viscosity(self, cv: ConservedVars,
                         dv: Optional[GasDependentVars] = None,
                         eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the 2nd coefficent of viscosity, $\lambda$."""
        raise NotImplementedError()

    def thermal_conductivity(self, cv: ConservedVars,
                             dv: Optional[GasDependentVars] = None,
                             eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the gas thermal_conductivity, $\kappa$."""
        raise NotImplementedError()

    def species_diffusivity(self, cv: ConservedVars,
                            dv: Optional[GasDependentVars] = None,
                            eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the vector of species diffusivities, ${d}_{\alpha}$."""
        raise NotImplementedError()

    def transport_vars(self, cv: ConservedVars,
                       dv: Optional[GasDependentVars] = None,
                       eos: Optional[GasEOS] = None) -> GasTransportVars:
        r"""Compute the transport properties from the conserved state."""
        return GasTransportVars(
            bulk_viscosity=self.bulk_viscosity(cv=cv, dv=dv, eos=eos),
            viscosity=self.viscosity(cv=cv, dv=dv, eos=eos),
            volume_viscosity=self.volume_viscosity(cv=cv, dv=dv, eos=eos),
            thermal_conductivity=self.thermal_conductivity(cv=cv, dv=dv, eos=eos),
            species_diffusivity=self.species_diffusivity(cv=cv, dv=dv, eos=eos)
        )


class SimpleTransport(TransportModel):
    r"""Transport model with uniform, constant properties.

    Inherits from (and implements) :class:`TransportModel`.

    .. automethod:: __init__
    .. automethod:: bulk_viscosity
    .. automethod:: viscosity
    .. automethod:: volume_viscosity
    .. automethod:: species_diffusivity
    .. automethod:: thermal_conductivity
    """

    def __init__(self, bulk_viscosity=0, viscosity=0, thermal_conductivity=0,
                 species_diffusivity=None):
        """Initialize uniform, constant transport properties."""
        if species_diffusivity is None:
            species_diffusivity = np.empty((0,), dtype=object)
        self._mu_bulk = bulk_viscosity
        self._mu = viscosity
        self._kappa = thermal_conductivity
        self._d_alpha = species_diffusivity

    def bulk_viscosity(self, cv: ConservedVars,
                       dv: Optional[GasDependentVars] = None,
                       eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the bulk viscosity for the gas, $\mu_{B}$."""
        return self._mu_bulk*(0*cv.mass + 1.0)

    def viscosity(self, cv: ConservedVars,
                  dv: Optional[GasDependentVars] = None,
                  eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the gas dynamic viscosity, $\mu$."""
        return self._mu*(0*cv.mass + 1.0)

    def volume_viscosity(self, cv: ConservedVars,
                         dv: Optional[GasDependentVars] = None,
                         eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the 2nd viscosity coefficent, $\lambda$.

        In this transport model, the second coefficient of viscosity is defined as:

        .. math::

            \lambda = \left(\mu_{B} - \frac{2\mu}{3}\right)

        """
        return (self._mu_bulk - 2 * self._mu / 3)*(0*cv.mass + 1.0)

    def thermal_conductivity(self, cv: ConservedVars,
                             dv: Optional[GasDependentVars] = None,
                             eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the gas thermal_conductivity, $\kappa$."""
        return self._kappa*(0*cv.mass + 1.0)

    def species_diffusivity(self, cv: ConservedVars,
                            dv: Optional[GasDependentVars] = None,
                            eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the vector of species diffusivities, ${d}_{\alpha}$."""
        return self._d_alpha*(0*cv.mass + 1.0)


class PowerLawTransport(TransportModel):
    r"""Transport model with simple power law properties.

    Inherits from (and implements) :class:`TransportModel` based on a
    temperature-dependent power law.

    .. automethod:: __init__
    .. automethod:: bulk_viscosity
    .. automethod:: viscosity
    .. automethod:: volume_viscosity
    .. automethod:: species_diffusivity
    .. automethod:: thermal_conductivity
    """

    # air-like defaults here
    def __init__(self, scaling_factor=1.0, alpha=0.6, beta=4.093e-7, sigma=2.5,
                 n=.666, species_diffusivity=None, lewis=None):
        """Initialize power law coefficients and parameters.

        Parameters
        ----------
        alpha: float
            The bulk viscosity parameter. The default value is "air".

        beta: float
            The dynamic viscosity linear parameter. The default value is "air".

        n: float
            The temperature exponent for dynamic viscosity. The default value
            is "air".

        sigma: float
            The heat conductivity linear parameter. The default value is "air".

        scaling_factor: float
            Scaling factor to artifically increase or decrease the transport
            coefficients. The default is to keep the physical value, i.e., 1.0.

        lewis: numpy.ndarray
            If required, the Lewis number specify the relation between the
            thermal conductivity and the species diffusivities.
        """
        if species_diffusivity is None and lewis is None:
            species_diffusivity = np.empty((0,), dtype=object)
        self._scaling_factor = scaling_factor
        self._alpha = alpha
        self._beta = beta
        self._sigma = sigma
        self._n = n
        self._d_alpha = species_diffusivity
        self._lewis = lewis

    def bulk_viscosity(self, cv: ConservedVars,  # type: ignore[override]
                       dv: GasDependentVars,
                       eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the bulk viscosity for the gas, $\mu_{B}$.

        .. math::

            \mu_{B} = \alpha\mu

        """
        return self._alpha * self.viscosity(cv, dv)

    # TODO: Should this be memoized? Avoid multiple calls?
    def viscosity(self, cv: ConservedVars,  # type: ignore[override]
                  dv: GasDependentVars,
                  eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the gas dynamic viscosity, $\mu$.

        $\mu = \beta{T}^n$
        """
        return self._scaling_factor * self._beta * dv.temperature**self._n

    def volume_viscosity(self, cv: ConservedVars,  # type: ignore[override]
                         dv: GasDependentVars,
                         eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the 2nd viscosity coefficent, $\lambda$.

        In this transport model, the second coefficient of viscosity is defined as:

        .. math::

            \lambda = \left(\alpha - \frac{2}{3}\right)\mu

        """
        return (self._alpha - 2.0/3.0) * self.viscosity(cv, dv)

    def thermal_conductivity(self, cv: ConservedVars,  # type: ignore[override]
                             dv: GasDependentVars, eos: GasEOS) -> DOFArray:
        r"""Get the gas thermal conductivity, $\kappa$.

        .. math::

            \kappa = \sigma\mu{C}_{v}
        """
        return (
            self._sigma * self.viscosity(cv, dv)
            * eos.heat_capacity_cv(cv, dv.temperature)
        )

    def species_diffusivity(self, cv: ConservedVars,  # type: ignore[override]
                            dv: GasDependentVars, eos: GasEOS) -> DOFArray:
        r"""Get the vector of species diffusivities, ${d}_{\alpha}$.

        The species diffusivities can be specified directly or based on the
        user-imposed Lewis number $Le$ of the mixture and the heat capacity at
        constant pressure $C_p$:

        .. math::

            d_{\alpha} = \frac{\kappa}{\rho \; Le \; C_p}
        """
        if self._lewis is not None:
            return (self._sigma * self.viscosity(cv, dv)/(
                cv.mass*self._lewis*eos.gamma(cv, dv.temperature))
            )
        return self._d_alpha*(0*cv.mass + 1.)


class MixtureAveragedTransport(TransportModel):
    r"""Transport model with mixture averaged transport properties.

    Inherits from (and implements) :class:`TransportModel` based on a
    temperature-dependent fit from Pyrometheus/Cantera weighted by the mixture
    composition. The mixture-averaged rules used follow those discussed in
    chapter 12 from [Kee_2003]_.

    .. automethod:: __init__
    .. automethod:: bulk_viscosity
    .. automethod:: viscosity
    .. automethod:: volume_viscosity
    .. automethod:: species_diffusivity
    .. automethod:: thermal_conductivity
    """

    def __init__(self, pyrometheus_mech, alpha=0.6, factor=1.0,
                 prandtl=None, lewis=None):
        r"""Initialize power law coefficients and parameters.

        Parameters
        ----------
        pyrometheus_mech: :class:`~pyrometheus.thermochem_example.Thermochemistry`
            The :mod:`pyrometheus`  mechanism
            :class:`~pyrometheus.thermochem_example.Thermochemistry`
            object that is generated by the user with a call to
            *pyrometheus.get_thermochem_class*. To create the mechanism
            object, users need to provide a mechanism input file. Several example
            mechanisms are provided in `mirgecom/mechanisms/` and can be used through
            the :meth:`mirgecom.mechanisms.get_mechanism_input`.

        alpha: float
            The bulk viscosity parameter. The default value is "air".

        factor: float
            Scaling factor to artifically increase or decrease the transport
            coefficients. The default is to keep the physical value, i.e., 1.0.

        prandtl: float
            If required, the Prandtl number specify the relation between the
            fluid viscosity and the thermal conductivity.

        lewis: numpy.ndarray
            If required, the Lewis number specify the relation between the
            thermal conductivity and the species diffusivities.
        """
        self._pyro_mech = pyrometheus_mech
        self._alpha = alpha
        self._factor = factor
        self._prandtl = prandtl
        self._lewis = lewis
        if self._lewis is not None:
            if (len(self._lewis) != self._pyro_mech.num_species):
                raise ValueError("Lewis number should match number of species")

    def viscosity(self, cv: ConservedVars,  # type: ignore[override]
                  dv: GasDependentVars,
                  eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the mixture dynamic viscosity, $\mu^{(m)}$.

        The viscosity depends on the mixture composition given by $X_k$ mole
        fraction and pure species viscosity $\mu_k$ of the individual species.
        The latter depends on the temperature and it is evaluated by Pyrometheus.

        .. math::

            \mu^{(m)} = \sum_{k=1}^{K} \frac{X_k \mu_k}{\sum_{j=1}^{K} X_j\phi_{kj}}

        .. math::

            \phi_{kj} = \frac{1}{\sqrt{8}}
            \left( 1 + \frac{W_k}{W_j} \right)^{-\frac{1}{2}}
            \left( 1 + \left[ \frac{\mu_k}{\mu_j} \right]^{\frac{1}{2}}
            \left[ \frac{W_j}{W_k} \right]^{\frac{1}{4}} \right)^2

        """
        return (
            self._factor*self._pyro_mech.get_mixture_viscosity_mixavg(
                dv.temperature, cv.species_mass_fractions)
        )

    def bulk_viscosity(self, cv: ConservedVars,  # type: ignore[override]
                       dv: GasDependentVars,
                       eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the bulk viscosity for the gas, $\mu_{B}$.

        .. math::

            \mu_{B} = \alpha\mu

        """
        return self._alpha*self.viscosity(cv, dv)

    def volume_viscosity(self, cv: ConservedVars,  # type: ignore[override]
                         dv: GasDependentVars,
                         eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the 2nd viscosity coefficent, $\lambda$.

        In this transport model, the second coefficient of viscosity is defined as:

        .. math::

            \lambda = \left(\alpha - \frac{2}{3}\right)\mu

        """
        return (self._alpha - 2.0/3.0)*self.viscosity(cv, dv)

    def thermal_conductivity(self, cv: ConservedVars,  # type: ignore[override]
                             dv: GasDependentVars,
                             eos: GasEOS) -> DOFArray:
        r"""Get the gas thermal_conductivity, $\kappa$.

        The thermal conductivity can be obtained from Pyrometheus using a
        mixture averaged rule considering the species individual heat
        conductivity and mole fractions:

        .. math::

            \kappa = \frac{1}{2} \left( \sum_{k=1}^{K} X_k \lambda_k +
               \frac{1}{\sum_{k=1}^{K} \frac{X_k}{\lambda_k} }\right)


        or based on the user-imposed Prandtl number of
        the mixture $Pr$ and the heat capacity at constant pressure $C_p$:

        .. math::

            \kappa = \frac{\mu C_p}{Pr}

        """
        if self._prandtl is not None:
            return 1.0/self._prandtl*(
                eos.heat_capacity_cp(cv, dv.temperature)*self.viscosity(cv, dv))
        return self._factor*(self._pyro_mech.get_mixture_thermal_conductivity_mixavg(
            dv.temperature, cv.species_mass_fractions,))

    def species_diffusivity(self, cv: ConservedVars,  # type: ignore[override]
                            dv: GasDependentVars,
                            eos: GasEOS) -> DOFArray:
        r"""Get the vector of species diffusivities, ${d}_{i}$.

        The species diffusivities can be obtained directly from Pyrometheus using
        a mixture averaged rule considering the species binary mass diffusivities
        $d_{ij}$ and the mass fractions $Y_i$

        .. math::

            d_{i}^{(m)} = \frac{1 - Y_i}{\sum_{j\ne i} \frac{X_j}{d_{ij}}}

        or based on the user-imposed Lewis number $Le$ of the mixture and the
        heat capacity at constant pressure $C_p$:

        .. math::

            d_{i} = \frac{\kappa}{\rho \; Le \; C_p}

        """
        if self._lewis is not None:
            return (self.thermal_conductivity(cv, dv, eos)/(
                cv.mass*self._lewis*eos.heat_capacity_cp(cv, dv.temperature))
            )
        return self._factor*(
            self._pyro_mech.get_species_mass_diffusivities_mixavg(
                dv.temperature, dv.pressure, cv.species_mass_fractions)
        )


class ArtificialViscosityTransportDiv(TransportModel):
    r"""Transport model for add artificial viscosity.

    Inherits from (and implements) :class:`TransportModel`.

    Takes a physical transport model and adds the artificial viscosity
    contribution to it. Defaults to simple transport with inviscid settings.
    This is equivalent to inviscid flow with artifical viscosity enabled.

    .. automethod:: __init__
    .. automethod:: bulk_viscosity
    .. automethod:: viscosity
    .. automethod:: volume_viscosity
    .. automethod:: species_diffusivity
    .. automethod:: thermal_conductivity
    """

    def __init__(self,
                 av_mu, av_prandtl, physical_transport=None,
                 av_species_diffusivity=None):
        """Initialize uniform, constant transport properties."""
        if physical_transport is None:
            self._physical_transport = SimpleTransport()
        else:
            self._physical_transport = physical_transport

        if av_species_diffusivity is None:
            av_species_diffusivity = np.empty((0,), dtype=object)

        self._av_mu = av_mu
        self._av_prandtl = av_prandtl

    def av_viscosity(self, cv, dv, eos):
        r"""Get the artificial viscosity for the gas."""
        actx = cv.array_context
        return self._av_mu*actx.np.sqrt(np.dot(cv.velocity, cv.velocity)
                                        + dv.speed_of_sound**2)

    def bulk_viscosity(self, cv: ConservedVars,  # type: ignore[override]
                       dv: GasDependentVars,
                       eos: GasEOS) -> DOFArray:
        r"""Get the bulk viscosity for the gas, $\mu_{B}$."""
        return self._physical_transport.bulk_viscosity(cv, dv)

    def viscosity(self, cv: ConservedVars,  # type: ignore[override]
                  dv: GasDependentVars,
                  eos: GasEOS) -> DOFArray:
        r"""Get the gas dynamic viscosity, $\mu$."""
        return (dv.smoothness*self.av_viscosity(cv, dv, eos)
                + self._physical_transport.viscosity(cv, dv))

    def volume_viscosity(self, cv: ConservedVars,  # type: ignore[override]
                         dv: GasDependentVars,
                         eos: GasEOS) -> DOFArray:
        r"""Get the 2nd viscosity coefficent, $\lambda$.

        In this transport model, the second coefficient of viscosity is defined as:

        $\lambda = \left(\mu_{B} - \frac{2\mu}{3}\right)$
        """
        return (dv.smoothness*self.av_viscosity(cv, dv, eos)
                + self._physical_transport.volume_viscosity(cv, dv))

    def thermal_conductivity(self, cv: ConservedVars,  # type: ignore[override]
                             dv: GasDependentVars,
                             eos: GasEOS) -> DOFArray:
        r"""Get the gas thermal_conductivity, $\kappa$."""
        mu = self.av_viscosity(cv, dv, eos)
        av_kappa = (dv.smoothness*mu
                    * eos.heat_capacity_cp(cv, dv.temperature)/self._av_prandtl)
        return av_kappa + self._physical_transport.thermal_conductivity(
            cv, dv, eos)

    def species_diffusivity(self, cv: ConservedVars,
                            dv: Optional[GasDependentVars] = None,
                            eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the vector of species diffusivities, ${d}_{\alpha}$."""
        return self._physical_transport.species_diffusivity(cv, dv, eos)
