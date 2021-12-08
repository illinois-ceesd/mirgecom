r""":mod:`mirgecom.thermochemistry` provides a wrapper class for :mod:`pyrometheus`..

.. autofunction:: make_pyrometheus_mechanism_class
.. autofunction:: make_pyrometheus_mechanism
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


def _pyro_thermochem_wrapper_class(cantera_soln, temperature_niter=5):
    """Return a MIRGE-compatible wrapper for a :mod:`pyrometheus` mechanism class.

    Dynamically creates a class that inherits from a
    :class:`pyrometheus.Thermochemistry` class and overrides a couple of the methods
    to adapt it to :mod:`mirgecom`'s needs.

        - get_concentrations: overrides :class:`pyrometheus.Thermochemistry` version
        of  the same function, pinning any negative concentrations due to slightly
        negative massfractions (which are OK) back to 0.
        - get_temperature: MIRGE-specific interface to use a hard-coded Newton solver
        to find a temperature from an input state.

    Parameters
    ----------
    cantera_soln: Cantera solution
        Cantera solution from which to create the thermochemical mechanism
    temperature_niter: integer
        Number of Newton iterations in `get_temperature` (default=5)
    """
    import pyrometheus as pyro
    pyro_class = pyro.get_thermochem_class(cantera_soln)

    class PyroWrapper(pyro_class):

        # This bit disallows negative concentrations and instead
        # pins them to 0. mass_fractions can sometimes be slightly
        # negative and that's ok.
        def get_concentrations(self, rho, mass_fractions):
            concs = self.iwts * rho * mass_fractions
            # ensure non-negative concentrations
            zero = self._pyro_zeros_like(concs[0])
            for i in range(self.num_species):
                concs[i] = self.usr_np.where(self.usr_np.less(concs[i], 0),
                                             zero, concs[i])
            return concs

        # This is the temperature update for *get_temperature*.  Having this
        # separated out allows it to be used in the fluid drivers for evaluating
        # convergence of the temperature calculation.
        def get_temperature_update_energy(self, e_in, t_in, y):
            pv_func = self.get_mixture_specific_heat_cv_mass
            he_func = self.get_mixture_internal_energy_mass
            return (e_in - he_func(t_in, y)) / pv_func(t_in, y)

        # This hard-codes the number of Newton iterations because the convergence
        # check is not compatible with lazy evaluation. Instead, we plan to check
        # the temperature residual at simulation health checking time.
        # FIXME: Occasional convergence check is other-than-ideal; revisit asap.
        # - could adapt dt or num_iter on temperature convergence?
        # - can pass-in num_iter?
        def get_temperature(self, energy, temperature_guess, species_mass_fractions):
            """Compute the temperature of the mixture from thermal energy.

            Parameters
            ----------
            energy: :class:`~meshmode.dof_array.DOFArray`
                The internal (thermal) energy of the mixture.
            temperature_guess: :class:`~meshmode.dof_array.DOFArray`
                An initial starting temperature for the Newton iterations.
            species_mass_fractions: numpy.ndarray
                An object array of :class:`~meshmode.dof_array.DOFArray` with the
                mass fractions of the mixture species.

            Returns
            -------
            :class:`~meshmode.dof_array.DOFArray`
                The mixture temperature after a fixed number of Newton iterations.
            """
            num_iter = temperature_niter
            t_i = temperature_guess
            for _ in range(num_iter):
                t_i = t_i + self.get_temperature_update_energy(
                    energy, t_i, species_mass_fractions
                )
            return t_i

    return PyroWrapper


def make_pyrometheus_mechanism_class(cantera_soln, temperature_niter=5):
    """Create a :mod:`pyrometheus` thermochemical (or equivalent) mechanism class.

    This routine creates and returns an instance of a :mod:`pyrometheus`
    thermochemical mechanism for use in a MIRGE-Com fluid EOS.

    Parameters
    ----------
    actx: :class:`arraycontext.ArrayContext`
        Array context from which to get the numpy-like namespace for
        :mod:`pyrometheus`
    cantera_soln:
        Cantera Solution for the thermochemical mechanism to be used

    Returns
    -------
    :mod:`pyrometheus` ThermoChem class
    """
    return _pyro_thermochem_wrapper_class(cantera_soln, temperature_niter)


# backwards compat
def make_pyrometheus_mechanism(actx, cantera_soln):
    """Create a :mod:`pyrometheus` thermochemical (or equivalent) mechanism.

    This routine creates and returns an instance of a :mod:`pyrometheus`
    thermochemical mechanism for use in a MIRGE-Com fluid EOS.

    Parameters
    ----------
    actx: :class:`arraycontext.ArrayContext`
        Array context from which to get the numpy-like namespace for
        :mod:`pyrometheus`
    cantera_soln:
        Cantera Solution for the thermochemical mechanism to be used

    Returns
    -------
    :mod:`pyrometheus` ThermoChem class
    """
    from warnings import warn
    warn("make_pyrometheus_mechanism is deprecated and will disappear in Q1/2022")
    return _pyro_thermochem_wrapper_class(cantera_soln)(actx.np)
