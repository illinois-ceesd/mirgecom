r""":mod:`mirgecom.thermochemistry` provides a wrapper class for :mod:`pyrometheus`..

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


def _pyro_thermochem_wrapper_class(cantera_soln):
    """Return a MIRGE-compatible wrapper for a :mod:`pyrometheus` mechanism class.

    Dynamically creates a class that inherits from a :mod:`pyrometheus` instance
    and overrides a couple of the functions for MIRGE-Com compatibility.
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

        # This hard-codes the Newton iterations to 10 because the convergence
        # check is not compatible with lazy evaluation. Instead, we plan to check
        # the temperature residual at simulation health checking time.
        # FIXME: Occasional convergence check is other-than-ideal; revisit asap.
        def get_temperature(self, enthalpy_or_energy, t_guess, y, do_energy=False):
            if do_energy is False:
                pv_fun = self.get_mixture_specific_heat_cp_mass
                he_fun = self.get_mixture_enthalpy_mass
            else:
                pv_fun = self.get_mixture_specific_heat_cv_mass
                he_fun = self.get_mixture_internal_energy_mass

            num_iter = 10
            ones = self._pyro_zeros_like(enthalpy_or_energy) + 1.0
            t_i = t_guess * ones

            for _ in range(num_iter):
                f = enthalpy_or_energy - he_fun(t_i, y)
                j = -pv_fun(t_i, y)
                dt = -f / j
                t_i += dt
                # if self._pyro_norm(dt, np.inf) < tol:

            return t_i

        # This hard-codes the Newton iterations to 10 because the convergence
        # check is not compatible with lazy evaluation. Instead, we plan to check
        # the temperature residual at simulation health checking time.
        # FIXME: Occasional convergence check is other-than-ideal; revisit asap.
        def get_temperature_residual(self, enthalpy_or_energy, t_guess, y,
                                     do_energy=False):
            if do_energy is False:
                pv_fun = self.get_mixture_specific_heat_cp_mass
                he_fun = self.get_mixture_enthalpy_mass
            else:
                pv_fun = self.get_mixture_specific_heat_cv_mass
                he_fun = self.get_mixture_internal_energy_mass

            ones = self._pyro_zeros_like(enthalpy_or_energy) + 1.0
            t_i = t_guess * ones

            f = enthalpy_or_energy - he_fun(t_i, y)
            j = -pv_fun(t_i, y)

            return -f / j

    return PyroWrapper


def make_pyrometheus_mechanism(actx, cantera_soln):
    """Create a :mod:`pyrometheus` thermochemical (or equivalent) mechanism object.

    This routine creates and returns an instance of a :mod:`pyrometheus`
    thermochemical mechanism for use in a MIRGE-Com fluid EOS. It requires a
    Cantera Solution and an array context.

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
    return _pyro_thermochem_wrapper_class(cantera_soln)(actx.np)
