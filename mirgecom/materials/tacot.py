""":mod:`~mirgecom.materials.tacot` evaluates TACOT-related data.

    TACOT is the Theoretical Ablative Composite for Open Testing, a surrogate
    composite material closely related to PICA (Phenolic Impregnated Carbon
    Ablator) proposed as an open-source material for the Ablation Workshop.

.. autoclass:: Pyrolysis
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
from pytools.obj_array import make_obj_array
# from mirgecom.fluid import make_conserved
from mirgecom.wall_model import PorousWallDegradationModel


# TODO generalize?
class Pyrolysis:
    r"""Evaluate the source terms for the pyrolysis decomposition.

    The source terms follow as Arrhenius-like equation given by

    .. math::

        \dot{\omega}_i^p = \mathcal{A}_{i} T^{n_{i}}
        \exp\left(- \frac{E_{i}}{RT} \right)
        \left( \frac{\epsilon_i \rho_i -
            \epsilon^c_i \rho^c_i}{\epsilon^0_i \rho^0_i} \right)^{m_i}

    For TACOT, 2 different reactions, which are assumed to only happen after
    a minimum temperature, are considered based on the resin constituents.
    The third reaction is the fiber oxidation, which is not handled here for now.

    .. automethod:: get_source_terms
    """

    def __init__(self):
        """Temperature in which each reaction starts."""
        self._Tcrit = np.array([333.3, 555.6])

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

        # The density parameters are specific for TACOT. They depend on the
        # virgin and char volume fraction.
        return make_obj_array([
            # reaction 1
            actx.np.where(actx.np.less(temperature, self._Tcrit[0]),
            0.0, (
                -(30.*((chi[0] - 0.00)/30.)**3)*12000.
                * actx.np.exp(-8556.000/temperature))),
            actx.np.where(actx.np.less(temperature, self._Tcrit[1]),
            # reaction 2
            0.0, (
                -(90.*((chi[1] - 60.0)/90.)**3)*4.48e9
                * actx.np.exp(-20444.44/temperature))),
            # fiber oxidation: include in the RHS but dont do anything with it.
            temperature*0.0])


class SolidProperties(PorousWallDegradationModel):
    """Evaluate the properties of the solid state containing resin and fibers.

    A linear weighting between the virgin and chared states is applied to
    yield the material properties. Polynomials were generated offline to avoid
    interpolation and they are not valid for temperatures above 3200K.

    .. automethod:: void_fraction
    .. automethod:: decomposition_progress
    .. automethod:: enthalpy
    .. automethod:: heat_capacity
    .. automethod:: thermal_conductivity
    .. automethod:: volume_fraction
    .. automethod:: permeability
    .. automethod:: emissivity
    .. automethod:: tortuosity
    """

    def __init__(self, char_mass, virgin_mass):
        """Bulk density considering the porosity and intrinsic density.

        The fiber and all resin components must be considered.
        """
        self._char_mass = char_mass
        self._virgin_mass = virgin_mass

    def void_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Return the volumetric fraction $\epsilon$ filled with gas.

        The fractions of gas and solid phases must sum to one,
        $\epsilon_g + \epsilon_s = 1$. Both depend only on the pyrolysis
        progress ratio $\tau$.
        """
        return 1.0 - self.volume_fraction(tau)

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
        return virgin_mass/(virgin_mass - char_mass)*(1.0 - char_mass/mass)

    def enthalpy(self, temperature: DOFArray, tau: DOFArray) -> DOFArray:
        """Solid enthalpy as a function of pyrolysis progress."""
        virgin = (
            - 1.360688853105e-11*temperature**5 + 1.521029626150e-07*temperature**4
            - 6.733769958659e-04*temperature**3 + 1.497082282729e+00*temperature**2
            + 3.009865156984e+02*temperature - 1.062767983774e+06)

        char = (
            - 1.279887694729e-11*temperature**5 + 1.491175465285e-07*temperature**4
            - 6.994595296860e-04*temperature**3 + 1.691564018109e+00*temperature**2
            - 3.441837408320e+01*temperature - 1.235438104496e+05)

        return virgin*tau + char*(1.0 - tau)

    def heat_capacity(self, temperature: DOFArray,
                      tau: DOFArray) -> DOFArray:
        r"""Solid heat capacity $C_{p_s}$ as a function of pyrolysis progress."""
        actx = temperature.array_context

        virgin = actx.np.where(actx.np.less(temperature, 2222.0),
            + 4.122658916891e-14*temperature**5 - 4.430937180604e-10*temperature**4
            + 1.872060335623e-06*temperature**3 - 3.951464865603e-03*temperature**2
            + 4.291080938736e+00*temperature + 1.397594340362e+01,
            2008.8139143251735)

        char = (
            + 1.461303669323e-14*temperature**5 - 1.862489701581e-10*temperature**4
            + 9.685398830530e-07*temperature**3 - 2.599755262540e-03*temperature**2
            + 3.667295510844e+00*temperature - 7.816218435655e+01)

        return virgin*tau + char*(1.0 - tau)

    def thermal_conductivity(self, temperature: DOFArray,
                             tau: DOFArray) -> DOFArray:
        """Solid thermal conductivity as a function of pyrolysis progress."""
        virgin = (
            + 2.31290019732353e-17*temperature**5 - 2.167785032562e-13*temperature**4
            + 8.24498395180905e-10*temperature**3 - 1.221612456223e-06*temperature**2
            + 8.46459266618945e-04*temperature + 2.387112689755e-01)

        char = (
            - 7.378279908877e-18*temperature**5 + 4.709353498411e-14*temperature**4
            + 1.530236899258e-11*temperature**3 - 2.305611352452e-07*temperature**2
            + 3.668624886569e-04*temperature + 3.120898814888e-01)

        return virgin*tau + char*(1.0 - tau)

    def volume_fraction(self, tau: DOFArray) -> DOFArray:
        r"""Fraction $\phi$ occupied by the solid."""
        fiber = 0.10
        virgin = 0.10
        char = 0.05
        return virgin*tau + char*(1.0 - tau) + fiber

    def permeability(self, tau: DOFArray) -> DOFArray:
        r"""Permeability $K$ of the porous material."""
        virgin = 1.6e-11
        char = 2.0e-11
        return virgin*tau + char*(1.0 - tau)

    def emissivity(self, tau: DOFArray) -> DOFArray:
        """Emissivity for energy radiation."""
        virgin = 0.8
        char = 0.9
        return virgin*tau + char*(1.0 - tau)

    def tortuosity(self, tau: DOFArray) -> DOFArray:
        r"""Tortuosity $\eta$ affects the species diffusivity."""
        virgin = 1.2
        char = 1.1
        return virgin*tau + char*(1.0 - tau)
