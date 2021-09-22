r""":mod:`mirgecom.thermochem` provides a wrapper class for :mod:`pyrometheus`..

.. autofunction:: make_thermochemistry_class
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


def make_thermochemisty_class(cantera_soln):
    """Create thermochemistry class."""
    import pyrometheus as pyro
    pyro_class = pyro.get_thermochem_class(cantera_soln)

    class ThermoChem(pyro_class):

        def get_concentrations(self, rho, mass_fractions):
            concs = self.iwts * rho * mass_fractions
            # ensure non-negative concentrations
            zero = self._pyro_zeros_like(concs[0])
            for i, conc in enumerate(concs):
                concs[i] = self.usr_np.where(self.usr_np.less(concs[i], 0),
                                             zero, concs[i])
            return concs

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

    return ThermoChem
