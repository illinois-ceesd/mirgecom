r""":mod:`mirgecom.thermochemistry` provides a wrapper class for :mod:`pyrometheus`.

This module provides an interface to the
`Pyrometheus Thermochemistry <https://github.com/pyrometheus>`_ package's
:class:`~pyrometheus.thermochem_example.Thermochemistry` object which provides a
thermal and chemical kinetics model for the the :class:`mirgecom.eos.MixtureEOS`,
and some helper routines to create the wrapper class.

   .. note::
    The wrapper addresses a couple of issues with the default interface:

    - Lazy eval is currently incapable of dealing with data-dependent
      behavior (like that of an iterative Newton solve). This wrapper allows us to
      hard-code the number of Newton iterations to *temperature_niter*.

    - Small species mass fractions can trigger reaction rates which drive species
      fractions significantly negative over a single timestep. The wrapper provides
      the *zero_level* parameter to set concentrations falling below *zero_level*
      to be pinned to zero.

.. autofunction:: get_pyrometheus_wrapper_class
.. autofunction:: get_pyrometheus_wrapper_class_from_cantera
.. autofunction:: get_thermochemistry_class_by_mechanism_name
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


def get_pyrometheus_wrapper_class(pyro_class, temperature_niter=5, zero_level=0.):
    """Return a MIRGE-compatible wrapper for a :mod:`pyrometheus` mechanism class.

    Dynamically creates a class that inherits from a
    :class:`~pyrometheus.thermochem_example.Thermochemistry` class and overrides a
    couple of the methods to adapt it to :mod:`mirgecom`'s needs.

    - get_concentrations: overrides
      :class:`~pyrometheus.thermochem_example.Thermochemistry` version of the same
      function, pinning any concentrations less than the *zero_level* due to small or
      slightly negative mass fractions (which are OK) back to 0.

    - get_temperature: MIRGE-specific interface to use a hard-coded Newton solver
      to find a temperature from an input state. This routine hard-codes the number
      of Newton solve iterations to *temperature_niter*.

    - get_heat_release:
      evaluate heat release due to reactions.

    Parameters
    ----------
    pyro_class: :class:`~pyrometheus.thermochem_example.Thermochemistry`
        Pyro thermochemical mechanism to wrap
    temperature_niter: int
        Number of Newton iterations in `get_temperature` (default=5)
    zero_level: float
        Squash concentrations below this level to 0. (default=0.)
    """

    class PyroWrapper(pyro_class):

        # This bit disallows negative concentrations (or user-defined floor)
        # and instead pins them to 0. Sometimes, mass_fractions can be slightly
        # negative and that's ok.
        # This only affects chemistry-related evaluations and does not interfere
        # with the actual fluid state.
        def get_concentrations(self, rho, mass_fractions):
            # concs = self.inv_molecular_weights * rho * mass_fractions
            concs = self.inv_molecular_weights * rho * mass_fractions
            # ensure non-negative concentrations
            zero = self._pyro_zeros_like(concs[0])
            for i in range(self.num_species):
                concs[i] = self.usr_np.where(self.usr_np.less(concs[i], zero_level),
                                             zero, concs[i])
            return concs

        # This is the temperature update for *get_temperature*. Having this
        # separated out allows it to be used in the fluid drivers for evaluating
        # convergence of the temperature calculation.
        def get_temperature_update_energy(self, e_in, t_in, y):
            pv_func = self.get_mixture_specific_heat_cv_mass
            he_func = self.get_mixture_internal_energy_mass
            return (e_in - he_func(t_in, y)) / pv_func(t_in, y)

        # This is the temperature update for *get_temperature*. Having this
        # separated out allows it to be used in the fluid drivers for evaluating
        # convergence of the temperature calculation.
        def get_temperature_update_enthalpy(self, h_in, t_in, y):
            pv_func = self.get_mixture_specific_heat_cp_mass
            he_func = self.get_mixture_enthalpy_mass
            return (h_in - he_func(t_in, y)) / pv_func(t_in, y)

        # This is the temperature update wrapper for *get_temperature*. It returns
        # the appropriate temperature update for the energy vs. the enthalpy
        # version of *get_temperature*.
        def get_temperature_update(self, e_or_h, t_in, y, use_energy=True):
            if use_energy:
                return self.get_temperature_update_energy(e_or_h, t_in, y)
            return self.get_temperature_update_enthalpy(e_or_h, t_in, y)

        # This hard-codes the number of Newton iterations because the convergence
        # check is not compatible with lazy evaluation. Instead, we plan to check
        # the temperature residual at simulation health checking time.
        # FIXME: Occasional convergence check is other-than-ideal; revisit asap.
        # - could adapt dt or num_iter on temperature convergence?
        # - can pass-in num_iter?
        def get_temperature(self, energy_or_enthalpy, temperature_guess,
                            species_mass_fractions, use_energy=True):
            """Compute the temperature of the mixture from thermal energy.

            Parameters
            ----------
            energy_or_enthalpy: :class:`~meshmode.dof_array.DOFArray`
                The internal (thermal) energy or enthalpy of the mixture. If
                enthalpy is passed, then *use_energy* should be set `False`.
            temperature_guess: :class:`~meshmode.dof_array.DOFArray`
                An initial starting temperature for the Newton iterations.
            species_mass_fractions: numpy.ndarray
                An object array of :class:`~meshmode.dof_array.DOFArray` with the
                mass fractions of the mixture species.
            use_energy: bool
                Indicates whether the energy or enthalpy version of the routine
                will be used. Defaults `True` for the energy version.

            Returns
            -------
            :class:`~meshmode.dof_array.DOFArray`
                The mixture temperature after a fixed number of Newton iterations.
            """
            num_iter = temperature_niter

            # if calorically perfect gas (constant heat capacities)
            if num_iter == 0:
                if use_energy:
                    heat_cap = self.get_mixture_specific_heat_cv_mass(
                        temperature_guess*0.0, species_mass_fractions)
                else:
                    heat_cap = self.get_mixture_specific_heat_cp_mass(
                        temperature_guess*0.0, species_mass_fractions)
                return energy_or_enthalpy/heat_cap

            # if thermally perfect gas
            t_i = temperature_guess
            for _ in range(num_iter):
                t_i = t_i + self.get_temperature_update(
                    energy_or_enthalpy, t_i, species_mass_fractions, use_energy
                )
            return t_i

        # Compute heat release rate due to chemistry.
        # Only used for visualization/post-processing.
        def get_heat_release_rate(self, state):
            w_dot = self.get_net_production_rates(state.cv.mass, state.temperature,
                                                  state.species_mass_fractions)

            h_a = self.get_species_enthalpies_rt(state.temperature)

            heat_rls = state.cv.mass*0.0
            for i in range(self.num_species):
                # heat_rls = heat_rls - h_a[i]*w_dot[i]/(self.molecular_weights[i])
                heat_rls = heat_rls - h_a[i]*w_dot[i]/(self.wts[i])

            return heat_rls*self.gas_constant*state.temperature

    return PyroWrapper


def get_pyrometheus_wrapper_class_from_cantera(cantera_soln, temperature_niter=5,
                                               zero_level=0.):
    """Return a MIRGE-compatible wrapper for a :mod:`pyrometheus` mechanism class.

    Cantera-based interface that creates a Pyrometheus mechanism
    :class:`~pyrometheus.thermochem_example.Thermochemistry` class on-the-fly using
    a Cantera solution.

    Parameters
    ----------
    cantera_soln:
        Cantera solution from which to create the thermochemical mechanism
    temperature_niter: int
        Number of Newton iterations in `get_temperature` (default=5)
    zero_level: float
        Squash concentrations below this level to 0. (default=0.)
    """
    import pyrometheus as pyro
    pyro_class = pyro.get_thermochem_class(cantera_soln)
    return get_pyrometheus_wrapper_class(pyro_class,
                                         temperature_niter=temperature_niter,
                                         zero_level=zero_level)


def get_thermochemistry_class_by_mechanism_name(mechanism_name: str,
                                                temperature_niter=5,
                                                zero_level=0.):
    """Grab a pyrometheus mechanism class from the mech name."""
    from mirgecom.mechanisms import get_mechanism_input
    mech_input_source = get_mechanism_input(mechanism_name)
    from cantera import Solution
    cantera_soln = Solution(name="gas", yaml=mech_input_source)
    return get_pyrometheus_wrapper_class_from_cantera(
        cantera_soln, temperature_niter=temperature_niter, zero_level=zero_level)
