""":mod:`~mirgecom.materials.tacot` evaluates TACOT-related data.

    TACOT is the Theoretical Ablative Composite for Open Testing, a surrogate
    composite material closely related to PICA (Phenolic Impregnated Carbon
    Ablator) proposed as an open-source material for the Ablation Workshop.

.. autoclass:: Pyrolysis
.. autoclass:: TacotEOS
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
from typing import Optional
from meshmode.dof_array import DOFArray
from pytools.obj_array import make_obj_array
from mirgecom.wall_model import PorousWallEOS


class Pyrolysis:
    r"""Evaluate the source terms for the pyrolysis decomposition of TACOT.

    The source terms follow as Arrhenius-like equation given by

    .. math::

        \dot{\omega}_i^p = \mathcal{A}_{i} T^{n_{i}}
        \exp\left(- \frac{E_{i}}{RT} \right)
        \left( \frac{\epsilon_i \rho_i -
            \epsilon^c_i \rho^c_i}{\epsilon^0_i \rho^0_i} \right)^{m_i}

    For TACOT, 2 different reactions, which are assumed to only happen after
    a minimum temperature, are considered based on the resin constituents.
    The third reaction is the fiber oxidation, which is not handled here for now.

    .. automethod:: __init__
    .. automethod:: get_decomposition_parameters
    .. automethod:: get_source_terms
    """

    def __init__(self, *,
                 virgin_mass=120.0, char_mass=60.0, fiber_mass=160.0,
                 pre_exponential=(12000.0, 4.48e9),
                 decomposition_temperature=(333.3, 555.6)):
        """Initialize TACOT parameters."""
        self._virgin_mass = virgin_mass
        self._char_mass = char_mass
        self._fiber_mass = fiber_mass

        if len(decomposition_temperature) != 2:
            raise ValueError("TACOT model requires 2 starting temperatures.")
        self._Tcrit = np.array(decomposition_temperature)

        if len(pre_exponential) != 2:
            raise ValueError("TACOT model requires 2 pre-exponentials.")
        self._pre_exp = np.array(pre_exponential)

    def get_decomposition_parameters(self):
        """Return the parameters of TACOT decomposition for inspection."""
        return {"virgin_mass": self._virgin_mass,
                "char_mass": self._char_mass,
                "fiber_mass": self._fiber_mass,
                "reaction_weights": np.array([self._virgin_mass*0.25,
                                              self._virgin_mass*0.75]),
                "pre_exponential": self._pre_exp,
                "temperature": self._Tcrit}

    def get_source_terms(self, temperature, chi):
        r"""Return the source terms of pyrolysis decomposition for TACOT.

        Parameters
        ----------
        temperature: :class:`~meshmode.dof_array.DOFArray`
            The temperature of the bulk material.

        chi: numpy.ndarray
            Either the solid mass $\rho_i$ of all fractions of the resin or
            the progress ratio $\chi$ of the decomposition. The actual
            parameter depends on the modeling itself.

        Returns
        -------
        source: numpy.ndarray
            The source terms for the pyrolysis
        """
        actx = temperature.array_context

        # The density parameters are hard-coded for TACOT, depending on
        # virgin and char volume fractions.
        w1 = self._virgin_mass*0.25*(chi[0]/(self._virgin_mass*0.25))**3
        w2 = self._virgin_mass*0.75*(chi[1]/(self._virgin_mass*0.75) - 2./3.)**3

        return make_obj_array([
            # reaction 1
            actx.np.where(actx.np.less(temperature, self._Tcrit[0]),
                0.0, (-w1 * self._pre_exp[0] * actx.np.exp(-8556.000/temperature))
                ),
            # reaction 2
            actx.np.where(actx.np.less(temperature, self._Tcrit[1]),
                0.0, (-w2 * self._pre_exp[1] * actx.np.exp(-20444.44/temperature))
                ),
            # fiber oxidation: include in the RHS but don't do anything with it.
            actx.np.zeros_like(temperature)])


class TacotEOS(PorousWallEOS):
    """Evaluate the properties of the solid state containing resin and fibers.

    A linear weighting between the virgin and chared states is applied to
    yield the material properties. Polynomials were generated offline to avoid
    interpolation and they are not valid for temperatures above 3200K.

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

    def __init__(self, char_mass, virgin_mass):
        """Bulk density considering the porosity and intrinsic density.

        Parameters
        ----------
        virgin_mass: float
            initial mass of the material. The fiber and all resin components
            must be considered.
        char_mass: float
            final mass when the resin decomposition is complete.
        """
        self._char_mass = char_mass
        self._virgin_mass = virgin_mass

    def void_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Return the volumetric fraction $\epsilon$ filled with gas.

        The fractions of gas and solid phases must sum to one,
        $\epsilon_g + \epsilon_s = 1$. Both depend only on the pyrolysis
        progress ratio $\tau$.
        """
        return 1.0 - self.volume_fraction(tau)  # type: ignore

    def enthalpy(self, temperature: DOFArray,
                 tau: Optional[DOFArray] = None) -> DOFArray:
        """Solid enthalpy as a function of pyrolysis progress."""
        virgin = (
            - 1.360688853105e-11*temperature**5  # type: ignore
            + 1.521029626150e-07*temperature**4  # type: ignore
            - 6.733769958659e-04*temperature**3  # type: ignore
            + 1.497082282729e+00*temperature**2  # type: ignore
            + 3.009865156984e+02*temperature  # type: ignore
            - 1.062767983774e+06)

        char = (
            - 1.279887694729e-11*temperature**5  # type: ignore
            + 1.491175465285e-07*temperature**4  # type: ignore
            - 6.994595296860e-04*temperature**3  # type: ignore
            + 1.691564018109e+00*temperature**2  # type: ignore
            - 3.441837408320e+01*temperature  # type: ignore
            - 1.235438104496e+05)

        return virgin*tau + char*(1.0 - tau)  # type: ignore

    def heat_capacity(self, temperature: DOFArray,
                      tau: Optional[DOFArray] = None) -> DOFArray:
        r"""Solid heat capacity $C_{p_s}$ as a function of pyrolysis progress."""
        actx = temperature.array_context

        virgin = actx.np.where(actx.np.less(temperature, 2222.0),
            + 4.122658916891e-14*temperature**5  # type: ignore
                               - 4.430937180604e-10*temperature**4  # type: ignore
            + 1.872060335623e-06*temperature**3  # type: ignore
                               - 3.951464865603e-03*temperature**2  # type: ignore
            + 4.291080938736e+00*temperature  # type: ignore
                               + 1.397594340362e+01,
            2008.8139143251735)

        char = (
            + 1.461303669323e-14*temperature**5  # type: ignore
            - 1.862489701581e-10*temperature**4  # type: ignore
            + 9.685398830530e-07*temperature**3  # type: ignore
            - 2.599755262540e-03*temperature**2  # type: ignore
            + 3.667295510844e+00*temperature  # type: ignore
            - 7.816218435655e+01)

        return virgin*tau + char*(1.0 - tau)  # type: ignore

    def thermal_conductivity(self, temperature: DOFArray,
                             tau: Optional[DOFArray] = None) -> DOFArray:
        """Solid thermal conductivity as a function of pyrolysis progress."""
        virgin = (
            + 2.31290019732353e-17*temperature**5  # type: ignore
            - 2.167785032562e-13*temperature**4  # type: ignore
            + 8.24498395180905e-10*temperature**3  # type: ignore
            - 1.221612456223e-06*temperature**2  # type: ignore
            + 8.46459266618945e-04*temperature  # type: ignore
            + 2.387112689755e-01)

        char = (
            - 7.378279908877e-18*temperature**5  # type: ignore
            + 4.709353498411e-14*temperature**4  # type: ignore
            + 1.530236899258e-11*temperature**3  # type: ignore
            - 2.305611352452e-07*temperature**2  # type: ignore
            + 3.668624886569e-04*temperature  # type: ignore
            + 3.120898814888e-01)

        return virgin*tau + char*(1.0 - tau)  # type: ignore

    def volume_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Fraction $\phi$ occupied by the solid."""
        fiber = 0.10
        virgin = 0.10
        char = 0.05
        return virgin*tau + char*(1.0 - tau) + fiber  # type: ignore

    def permeability(self, tau: DOFArray) -> DOFArray:
        r"""Permeability $K$ of the composite material."""
        virgin = 1.6e-11
        char = 2.0e-11
        return virgin*tau + char*(1.0 - tau)  # type: ignore

    def emissivity(self, temperature=None, tau=None) -> DOFArray:
        """Emissivity for energy radiation."""
        virgin = 0.8
        char = 0.9
        return virgin*tau + char*(1.0 - tau)

    def tortuosity(self, tau: DOFArray) -> DOFArray:
        r"""Tortuosity $\eta$ affects the species diffusivity."""
        virgin = 1.2
        char = 1.1
        return virgin*tau + char*(1.0 - tau)  # type: ignore

    def decomposition_progress(self, mass: DOFArray) -> DOFArray:
        r"""Evaluate the progress ratio $\tau$ of the phenolics decomposition.

        Where $\tau=1$, the material is locally virgin. On the other hand, if
        $\tau=0$, then the pyrolysis is locally complete and only charred
        material exists:

        .. math::
            \tau = \frac{\rho_0}{\rho_0 - \rho_c}
                    \left( 1 - \frac{\rho_c}{\rho(t)} \right)
        """
        char_mass = self._char_mass
        virgin_mass = self._virgin_mass
        return virgin_mass/(virgin_mass
                            - char_mass)*(1.0 - char_mass/mass)
