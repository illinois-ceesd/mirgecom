"""
.. autoclass:: Thermochemistry
"""


import numpy as np


class Thermochemistry:
    """
    .. attribute:: model_name
    .. attribute:: num_elements
    .. attribute:: num_species
    .. attribute:: num_reactions
    .. attribute:: num_falloff
    .. attribute:: one_atm

        Returns 1 atm in SI units of pressure (Pa).

    .. attribute:: gas_constant
    .. attribute:: species_names
    .. attribute:: species_indices

    .. automethod:: get_specific_gas_constant
    .. automethod:: get_density
    .. automethod:: get_pressure
    .. automethod:: get_mix_molecular_weight
    .. automethod:: get_concentrations
    .. automethod:: get_mixture_specific_heat_cp_mass
    .. automethod:: get_mixture_specific_heat_cv_mass
    .. automethod:: get_mixture_enthalpy_mass
    .. automethod:: get_mixture_internal_energy_mass
    .. automethod:: get_species_specific_heats_r
    .. automethod:: get_species_enthalpies_rt
    .. automethod:: get_species_entropies_r
    .. automethod:: get_species_gibbs_rt
    .. automethod:: get_equilibrium_constants
    .. automethod:: get_temperature
    .. automethod:: __init__
    """

    def __init__(self, usr_np=np):
        """Initialize thermochemistry object for a mechanism.

        Parameters
        ----------
        usr_np
            :mod:`numpy`-like namespace providing at least the following functions,
            for any array ``X`` of the bulk array type:

            - ``usr_np.log(X)`` (like :data:`numpy.log`)
            - ``usr_np.log10(X)`` (like :data:`numpy.log10`)
            - ``usr_np.exp(X)`` (like :data:`numpy.exp`)
            - ``usr_np.where(X > 0, X_yes, X_no)`` (like :func:`numpy.where`)
            - ``usr_np.linalg.norm(X, np.inf)`` (like :func:`numpy.linalg.norm`)

            where the "bulk array type" is a type that offers arithmetic analogous
            to :class:`numpy.ndarray` and is used to hold all types of (potentialy
            volumetric) "bulk data", such as temperature, pressure, mass fractions,
            etc. This parameter defaults to *actual numpy*, so it can be ignored
            unless it is needed by the user (e.g. for purposes of
            GPU processing or automatic differentiation).

        """

        self.usr_np = usr_np
        self.model_name = 'mechs/gri30.yaml'
        self.num_elements = 5
        self.num_species = 53
        self.num_reactions = 325
        self.num_falloff = 29

        self.one_atm = 101325.0
        self.gas_constant = 8314.46261815324
        self.big_number = 1.0e300

        self.species_names = ['H2', 'H', 'O', 'O2', 'OH', 'H2O', 'HO2', 'H2O2', 'C', 'CH', 'CH2', 'CH2(S)', 'CH3', 'CH4', 'CO', 'CO2', 'HCO', 'CH2O', 'CH2OH', 'CH3O', 'CH3OH', 'C2H', 'C2H2', 'C2H3', 'C2H4', 'C2H5', 'C2H6', 'HCCO', 'CH2CO', 'HCCOH', 'N', 'NH', 'NH2', 'NH3', 'NNH', 'NO', 'NO2', 'N2O', 'HNO', 'CN', 'HCN', 'H2CN', 'HCNN', 'HCNO', 'HOCN', 'HNCO', 'NCO', 'N2', 'AR', 'C3H7', 'C3H8', 'CH2CHO', 'CH3CHO']
        self.species_indices = {'H2': 0, 'H': 1, 'O': 2, 'O2': 3, 'OH': 4, 'H2O': 5, 'HO2': 6, 'H2O2': 7, 'C': 8, 'CH': 9, 'CH2': 10, 'CH2(S)': 11, 'CH3': 12, 'CH4': 13, 'CO': 14, 'CO2': 15, 'HCO': 16, 'CH2O': 17, 'CH2OH': 18, 'CH3O': 19, 'CH3OH': 20, 'C2H': 21, 'C2H2': 22, 'C2H3': 23, 'C2H4': 24, 'C2H5': 25, 'C2H6': 26, 'HCCO': 27, 'CH2CO': 28, 'HCCOH': 29, 'N': 30, 'NH': 31, 'NH2': 32, 'NH3': 33, 'NNH': 34, 'NO': 35, 'NO2': 36, 'N2O': 37, 'HNO': 38, 'CN': 39, 'HCN': 40, 'H2CN': 41, 'HCNN': 42, 'HCNO': 43, 'HOCN': 44, 'HNCO': 45, 'NCO': 46, 'N2': 47, 'AR': 48, 'C3H7': 49, 'C3H8': 50, 'CH2CHO': 51, 'CH3CHO': 52}

        self.wts = np.array([2.016, 1.008, 15.999, 31.998, 17.007, 18.015, 33.006, 34.014, 12.011, 13.018999999999998, 14.027, 14.027, 15.035, 16.043, 28.009999999999998, 44.009, 29.018, 30.026, 31.034, 31.034, 32.042, 25.029999999999998, 26.037999999999997, 27.046, 28.054, 29.061999999999998, 30.07, 41.028999999999996, 42.037, 42.037, 14.007, 15.015, 16.023, 17.031, 29.022, 30.006, 46.005, 44.013, 31.014000000000003, 26.018, 27.025999999999996, 28.034, 41.033, 43.025, 43.025, 43.025, 42.016999999999996, 28.014, 39.95, 43.089, 44.097, 43.045, 44.053])
        self.iwts = 1/self.wts

    def _pyro_zeros_like(self, argument):
        # FIXME: This is imperfect, as a NaN will stay a NaN.
        return 0 * argument

    def _pyro_make_array(self, res_list):
        """This works around (e.g.) numpy.exp not working with object
        arrays of numpy scalars. It defaults to making object arrays, however
        if an array consists of all scalars, it makes a "plain old"
        :class:`numpy.ndarray`.

        See ``this numpy bug <https://github.com/numpy/numpy/issues/18004>`__
        for more context.
        """

        from numbers import Number
        all_numbers = all(isinstance(e, Number) for e in res_list)

        dtype = np.float64 if all_numbers else np.object
        result = np.empty((len(res_list),), dtype=dtype)

        # 'result[:] = res_list' may look tempting, however:
        # https://github.com/numpy/numpy/issues/16564
        for idx in range(len(res_list)):
            result[idx] = res_list[idx]

        return result

    def _pyro_norm(self, argument, normord):
        """This works around numpy.linalg norm not working with scalars.

        If the argument is a regular ole number, it uses :func:`numpy.abs`,
        otherwise it uses ``usr_np.linalg.norm``.
        """
        # Wrap norm for scalars

        from numbers import Number

        if isinstance(argument, Number):
            return np.abs(argument)
        return self.usr_np.linalg.norm(argument, normord)

    def species_name(self, species_index):
        return self.species_name[species_index]

    def species_index(self, species_name):
        return self.species_indices[species_name]

    def get_specific_gas_constant(self, mass_fractions):
        return self.gas_constant * (
                    + self.iwts[0]*mass_fractions[0]
                    + self.iwts[1]*mass_fractions[1]
                    + self.iwts[2]*mass_fractions[2]
                    + self.iwts[3]*mass_fractions[3]
                    + self.iwts[4]*mass_fractions[4]
                    + self.iwts[5]*mass_fractions[5]
                    + self.iwts[6]*mass_fractions[6]
                    + self.iwts[7]*mass_fractions[7]
                    + self.iwts[8]*mass_fractions[8]
                    + self.iwts[9]*mass_fractions[9]
                    + self.iwts[10]*mass_fractions[10]
                    + self.iwts[11]*mass_fractions[11]
                    + self.iwts[12]*mass_fractions[12]
                    + self.iwts[13]*mass_fractions[13]
                    + self.iwts[14]*mass_fractions[14]
                    + self.iwts[15]*mass_fractions[15]
                    + self.iwts[16]*mass_fractions[16]
                    + self.iwts[17]*mass_fractions[17]
                    + self.iwts[18]*mass_fractions[18]
                    + self.iwts[19]*mass_fractions[19]
                    + self.iwts[20]*mass_fractions[20]
                    + self.iwts[21]*mass_fractions[21]
                    + self.iwts[22]*mass_fractions[22]
                    + self.iwts[23]*mass_fractions[23]
                    + self.iwts[24]*mass_fractions[24]
                    + self.iwts[25]*mass_fractions[25]
                    + self.iwts[26]*mass_fractions[26]
                    + self.iwts[27]*mass_fractions[27]
                    + self.iwts[28]*mass_fractions[28]
                    + self.iwts[29]*mass_fractions[29]
                    + self.iwts[30]*mass_fractions[30]
                    + self.iwts[31]*mass_fractions[31]
                    + self.iwts[32]*mass_fractions[32]
                    + self.iwts[33]*mass_fractions[33]
                    + self.iwts[34]*mass_fractions[34]
                    + self.iwts[35]*mass_fractions[35]
                    + self.iwts[36]*mass_fractions[36]
                    + self.iwts[37]*mass_fractions[37]
                    + self.iwts[38]*mass_fractions[38]
                    + self.iwts[39]*mass_fractions[39]
                    + self.iwts[40]*mass_fractions[40]
                    + self.iwts[41]*mass_fractions[41]
                    + self.iwts[42]*mass_fractions[42]
                    + self.iwts[43]*mass_fractions[43]
                    + self.iwts[44]*mass_fractions[44]
                    + self.iwts[45]*mass_fractions[45]
                    + self.iwts[46]*mass_fractions[46]
                    + self.iwts[47]*mass_fractions[47]
                    + self.iwts[48]*mass_fractions[48]
                    + self.iwts[49]*mass_fractions[49]
                    + self.iwts[50]*mass_fractions[50]
                    + self.iwts[51]*mass_fractions[51]
                    + self.iwts[52]*mass_fractions[52]
                )

    def get_density(self, p, temperature, mass_fractions):
        mmw = self.get_mix_molecular_weight(mass_fractions)
        rt = self.gas_constant * temperature
        return p * mmw / rt

    def get_pressure(self, rho, temperature, mass_fractions):
        mmw = self.get_mix_molecular_weight(mass_fractions)
        rt = self.gas_constant * temperature
        return rho * rt / mmw

    def get_mix_molecular_weight(self, mass_fractions):
        return 1/(
                    + self.iwts[0]*mass_fractions[0]
                    + self.iwts[1]*mass_fractions[1]
                    + self.iwts[2]*mass_fractions[2]
                    + self.iwts[3]*mass_fractions[3]
                    + self.iwts[4]*mass_fractions[4]
                    + self.iwts[5]*mass_fractions[5]
                    + self.iwts[6]*mass_fractions[6]
                    + self.iwts[7]*mass_fractions[7]
                    + self.iwts[8]*mass_fractions[8]
                    + self.iwts[9]*mass_fractions[9]
                    + self.iwts[10]*mass_fractions[10]
                    + self.iwts[11]*mass_fractions[11]
                    + self.iwts[12]*mass_fractions[12]
                    + self.iwts[13]*mass_fractions[13]
                    + self.iwts[14]*mass_fractions[14]
                    + self.iwts[15]*mass_fractions[15]
                    + self.iwts[16]*mass_fractions[16]
                    + self.iwts[17]*mass_fractions[17]
                    + self.iwts[18]*mass_fractions[18]
                    + self.iwts[19]*mass_fractions[19]
                    + self.iwts[20]*mass_fractions[20]
                    + self.iwts[21]*mass_fractions[21]
                    + self.iwts[22]*mass_fractions[22]
                    + self.iwts[23]*mass_fractions[23]
                    + self.iwts[24]*mass_fractions[24]
                    + self.iwts[25]*mass_fractions[25]
                    + self.iwts[26]*mass_fractions[26]
                    + self.iwts[27]*mass_fractions[27]
                    + self.iwts[28]*mass_fractions[28]
                    + self.iwts[29]*mass_fractions[29]
                    + self.iwts[30]*mass_fractions[30]
                    + self.iwts[31]*mass_fractions[31]
                    + self.iwts[32]*mass_fractions[32]
                    + self.iwts[33]*mass_fractions[33]
                    + self.iwts[34]*mass_fractions[34]
                    + self.iwts[35]*mass_fractions[35]
                    + self.iwts[36]*mass_fractions[36]
                    + self.iwts[37]*mass_fractions[37]
                    + self.iwts[38]*mass_fractions[38]
                    + self.iwts[39]*mass_fractions[39]
                    + self.iwts[40]*mass_fractions[40]
                    + self.iwts[41]*mass_fractions[41]
                    + self.iwts[42]*mass_fractions[42]
                    + self.iwts[43]*mass_fractions[43]
                    + self.iwts[44]*mass_fractions[44]
                    + self.iwts[45]*mass_fractions[45]
                    + self.iwts[46]*mass_fractions[46]
                    + self.iwts[47]*mass_fractions[47]
                    + self.iwts[48]*mass_fractions[48]
                    + self.iwts[49]*mass_fractions[49]
                    + self.iwts[50]*mass_fractions[50]
                    + self.iwts[51]*mass_fractions[51]
                    + self.iwts[52]*mass_fractions[52]
                )

    def get_concentrations(self, rho, mass_fractions):
        return self.iwts * rho * mass_fractions

    def get_mass_average_property(self, mass_fractions, spec_property):
        return sum([mass_fractions[i] * spec_property[i] * self.iwts[i]
                    for i in range(self.num_species)])

    def get_mixture_specific_heat_cp_mass(self, temperature, mass_fractions):
        cp0_r = self.get_species_specific_heats_r(temperature)
        cpmix = self.get_mass_average_property(mass_fractions, cp0_r)
        return self.gas_constant * cpmix

    def get_mixture_specific_heat_cv_mass(self, temperature, mass_fractions):
        cp0_r = self.get_species_specific_heats_r(temperature) - 1.0
        cpmix = self.get_mass_average_property(mass_fractions, cp0_r)
        return self.gas_constant * cpmix

    def get_mixture_enthalpy_mass(self, temperature, mass_fractions):
        h0_rt = self.get_species_enthalpies_rt(temperature)
        hmix = self.get_mass_average_property(mass_fractions, h0_rt)
        return self.gas_constant * temperature * hmix

    def get_mixture_internal_energy_mass(self, temperature, mass_fractions):
        e0_rt = self.get_species_enthalpies_rt(temperature) - 1.0
        emix = self.get_mass_average_property(mass_fractions, e0_rt)
        return self.gas_constant * temperature * emix

    def get_species_specific_heats_r(self, temperature):
        return self._pyro_make_array([
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.3372792 + -4.94024731e-05*temperature + 4.99456778e-07*temperature**2 + -1.79566394e-10*temperature**3 + 2.00255376e-14*temperature**4, 2.34433112 + 0.00798052075*temperature + -1.9478151e-05*temperature**2 + 2.01572094e-08*temperature**3 + -7.37611761e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.50000001 + -2.30842973e-11*temperature + 1.61561948e-14*temperature**2 + -4.73515235e-18*temperature**3 + 4.98197357e-22*temperature**4, 2.5 + 7.05332819e-13*temperature + -1.99591964e-15*temperature**2 + 2.30081632e-18*temperature**3 + -9.27732332e-22*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.56942078 + -8.59741137e-05*temperature + 4.19484589e-08*temperature**2 + -1.00177799e-11*temperature**3 + 1.22833691e-15*temperature**4, 3.1682671 + -0.00327931884*temperature + 6.64306396e-06*temperature**2 + -6.12806624e-09*temperature**3 + 2.11265971e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.28253784 + 0.00148308754*temperature + -7.57966669e-07*temperature**2 + 2.09470555e-10*temperature**3 + -2.16717794e-14*temperature**4, 3.78245636 + -0.00299673416*temperature + 9.84730201e-06*temperature**2 + -9.68129509e-09*temperature**3 + 3.24372837e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.09288767 + 0.000548429716*temperature + 1.26505228e-07*temperature**2 + -8.79461556e-11*temperature**3 + 1.17412376e-14*temperature**4, 3.99201543 + -0.00240131752*temperature + 4.61793841e-06*temperature**2 + -3.88113333e-09*temperature**3 + 1.3641147e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.03399249 + 0.00217691804*temperature + -1.64072518e-07*temperature**2 + -9.7041987e-11*temperature**3 + 1.68200992e-14*temperature**4, 4.19864056 + -0.0020364341*temperature + 6.52040211e-06*temperature**2 + -5.48797062e-09*temperature**3 + 1.77197817e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.0172109 + 0.00223982013*temperature + -6.3365815e-07*temperature**2 + 1.1424637e-10*temperature**3 + -1.07908535e-14*temperature**4, 4.30179801 + -0.00474912051*temperature + 2.11582891e-05*temperature**2 + -2.42763894e-08*temperature**3 + 9.29225124e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.16500285 + 0.00490831694*temperature + -1.90139225e-06*temperature**2 + 3.71185986e-10*temperature**3 + -2.87908305e-14*temperature**4, 4.27611269 + -0.000542822417*temperature + 1.67335701e-05*temperature**2 + -2.15770813e-08*temperature**3 + 8.62454363e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.49266888 + 4.79889284e-05*temperature + -7.2433502e-08*temperature**2 + 3.74291029e-11*temperature**3 + -4.87277893e-15*temperature**4, 2.55423955 + -0.000321537724*temperature + 7.33792245e-07*temperature**2 + -7.32234889e-10*temperature**3 + 2.66521446e-13*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.87846473 + 0.000970913681*temperature + 1.44445655e-07*temperature**2 + -1.30687849e-10*temperature**3 + 1.76079383e-14*temperature**4, 3.48981665 + 0.000323835541*temperature + -1.68899065e-06*temperature**2 + 3.16217327e-09*temperature**3 + -1.40609067e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.87410113 + 0.00365639292*temperature + -1.40894597e-06*temperature**2 + 2.60179549e-10*temperature**3 + -1.87727567e-14*temperature**4, 3.76267867 + 0.000968872143*temperature + 2.79489841e-06*temperature**2 + -3.85091153e-09*temperature**3 + 1.68741719e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.29203842 + 0.00465588637*temperature + -2.01191947e-06*temperature**2 + 4.17906e-10*temperature**3 + -3.39716365e-14*temperature**4, 4.19860411 + -0.00236661419*temperature + 8.2329622e-06*temperature**2 + -6.68815981e-09*temperature**3 + 1.94314737e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.28571772 + 0.00723990037*temperature + -2.98714348e-06*temperature**2 + 5.95684644e-10*temperature**3 + -4.67154394e-14*temperature**4, 3.6735904 + 0.00201095175*temperature + 5.73021856e-06*temperature**2 + -6.87117425e-09*temperature**3 + 2.54385734e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 0.074851495 + 0.0133909467*temperature + -5.73285809e-06*temperature**2 + 1.22292535e-09*temperature**3 + -1.0181523e-13*temperature**4, 5.14987613 + -0.0136709788*temperature + 4.91800599e-05*temperature**2 + -4.84743026e-08*temperature**3 + 1.66693956e-11*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.71518561 + 0.00206252743*temperature + -9.98825771e-07*temperature**2 + 2.30053008e-10*temperature**3 + -2.03647716e-14*temperature**4, 3.57953347 + -0.00061035368*temperature + 1.01681433e-06*temperature**2 + 9.07005884e-10*temperature**3 + -9.04424499e-13*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.85746029 + 0.00441437026*temperature + -2.21481404e-06*temperature**2 + 5.23490188e-10*temperature**3 + -4.72084164e-14*temperature**4, 2.35677352 + 0.00898459677*temperature + -7.12356269e-06*temperature**2 + 2.45919022e-09*temperature**3 + -1.43699548e-13*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.77217438 + 0.00495695526*temperature + -2.48445613e-06*temperature**2 + 5.89161778e-10*temperature**3 + -5.33508711e-14*temperature**4, 4.22118584 + -0.00324392532*temperature + 1.37799446e-05*temperature**2 + -1.33144093e-08*temperature**3 + 4.33768865e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 1.76069008 + 0.00920000082*temperature + -4.42258813e-06*temperature**2 + 1.00641212e-09*temperature**3 + -8.8385564e-14*temperature**4, 4.79372315 + -0.00990833369*temperature + 3.73220008e-05*temperature**2 + -3.79285261e-08*temperature**3 + 1.31772652e-11*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.69266569 + 0.00864576797*temperature + -3.7510112e-06*temperature**2 + 7.87234636e-10*temperature**3 + -6.48554201e-14*temperature**4, 3.86388918 + 0.00559672304*temperature + 5.93271791e-06*temperature**2 + -1.04532012e-08*temperature**3 + 4.36967278e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.770799 + 0.007871497*temperature + -2.656384e-06*temperature**2 + 3.944431e-10*temperature**3 + -2.112616e-14*temperature**4, 2.106204 + 0.007216595*temperature + 5.338472e-06*temperature**2 + -7.377636e-09*temperature**3 + 2.07561e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 1.78970791 + 0.0140938292*temperature + -6.36500835e-06*temperature**2 + 1.38171085e-09*temperature**3 + -1.1706022e-13*temperature**4, 5.71539582 + -0.0152309129*temperature + 6.52441155e-05*temperature**2 + -7.10806889e-08*temperature**3 + 2.61352698e-11*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.16780652 + 0.00475221902*temperature + -1.83787077e-06*temperature**2 + 3.04190252e-10*temperature**3 + -1.7723277e-14*temperature**4, 2.88965733 + 0.0134099611*temperature + -2.84769501e-05*temperature**2 + 2.94791045e-08*temperature**3 + -1.09331511e-11*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.14756964 + 0.00596166664*temperature + -2.37294852e-06*temperature**2 + 4.67412171e-10*temperature**3 + -3.61235213e-14*temperature**4, 0.808681094 + 0.0233615629*temperature + -3.55171815e-05*temperature**2 + 2.80152437e-08*temperature**3 + -8.50072974e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.016724 + 0.0103302292*temperature + -4.68082349e-06*temperature**2 + 1.01763288e-09*temperature**3 + -8.62607041e-14*temperature**4, 3.21246645 + 0.00151479162*temperature + 2.59209412e-05*temperature**2 + -3.57657847e-08*temperature**3 + 1.47150873e-11*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.03611116 + 0.0146454151*temperature + -6.71077915e-06*temperature**2 + 1.47222923e-09*temperature**3 + -1.25706061e-13*temperature**4, 3.95920148 + -0.00757052247*temperature + 5.70990292e-05*temperature**2 + -6.91588753e-08*temperature**3 + 2.69884373e-11*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 1.95465642 + 0.0173972722*temperature + -7.98206668e-06*temperature**2 + 1.75217689e-09*temperature**3 + -1.49641576e-13*temperature**4, 4.30646568 + -0.00418658892*temperature + 4.97142807e-05*temperature**2 + -5.99126606e-08*temperature**3 + 2.30509004e-11*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 1.0718815 + 0.0216852677*temperature + -1.00256067e-05*temperature**2 + 2.21412001e-09*temperature**3 + -1.9000289e-13*temperature**4, 4.29142492 + -0.0055015427*temperature + 5.99438288e-05*temperature**2 + -7.08466285e-08*temperature**3 + 2.68685771e-11*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.6282058 + 0.0040853401*temperature + -1.5934547e-06*temperature**2 + 2.8626052e-10*temperature**3 + -1.9407832e-14*temperature**4, 2.2517214 + 0.017655021*temperature + -2.3729101e-05*temperature**2 + 1.7275759e-08*temperature**3 + -5.0664811e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.51129732 + 0.00900359745*temperature + -4.16939635e-06*temperature**2 + 9.23345882e-10*temperature**3 + -7.94838201e-14*temperature**4, 2.1358363 + 0.0181188721*temperature + -1.73947474e-05*temperature**2 + 9.34397568e-09*temperature**3 + -2.01457615e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.9238291 + 0.00679236*temperature + -2.5658564e-06*temperature**2 + 4.4987841e-10*temperature**3 + -2.9940101e-14*temperature**4, 1.2423733 + 0.031072201*temperature + -5.0866864e-05*temperature**2 + 4.3137131e-08*temperature**3 + -1.4014594e-11*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.4159429 + 0.00017489065*temperature + -1.1902369e-07*temperature**2 + 3.0226245e-11*temperature**3 + -2.0360982e-15*temperature**4, 2.5),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.7836928 + 0.001329843*temperature + -4.2478047e-07*temperature**2 + 7.8348501e-11*temperature**3 + -5.504447e-15*temperature**4, 3.4929085 + 0.00031179198*temperature + -1.4890484e-06*temperature**2 + 2.4816442e-09*temperature**3 + -1.0356967e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.8347421 + 0.0032073082*temperature + -9.3390804e-07*temperature**2 + 1.3702953e-10*temperature**3 + -7.9206144e-15*temperature**4, 4.2040029 + -0.0021061385*temperature + 7.1068348e-06*temperature**2 + -5.6115197e-09*temperature**3 + 1.6440717e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.6344521 + 0.005666256*temperature + -1.7278676e-06*temperature**2 + 2.3867161e-10*temperature**3 + -1.2578786e-14*temperature**4, 4.2860274 + -0.004660523*temperature + 2.1718513e-05*temperature**2 + -2.2808887e-08*temperature**3 + 8.2638046e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.7667544 + 0.0028915082*temperature + -1.041662e-06*temperature**2 + 1.6842594e-10*temperature**3 + -1.0091896e-14*temperature**4, 4.3446927 + -0.0048497072*temperature + 2.0059459e-05*temperature**2 + -2.1726464e-08*temperature**3 + 7.9469539e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.2606056 + 0.0011911043*temperature + -4.2917048e-07*temperature**2 + 6.9457669e-11*temperature**3 + -4.0336099e-15*temperature**4, 4.2184763 + -0.004638976*temperature + 1.1041022e-05*temperature**2 + -9.3361354e-09*temperature**3 + 2.803577e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.8847542 + 0.0021723956*temperature + -8.2806906e-07*temperature**2 + 1.574751e-10*temperature**3 + -1.0510895e-14*temperature**4, 3.9440312 + -0.001585429*temperature + 1.6657812e-05*temperature**2 + -2.0475426e-08*temperature**3 + 7.8350564e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.8230729 + 0.0026270251*temperature + -9.5850874e-07*temperature**2 + 1.6000712e-10*temperature**3 + -9.7752303e-15*temperature**4, 2.2571502 + 0.011304728*temperature + -1.3671319e-05*temperature**2 + 9.6819806e-09*temperature**3 + -2.9307182e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.9792509 + 0.0034944059*temperature + -7.8549778e-07*temperature**2 + 5.7479594e-11*temperature**3 + -1.9335916e-16*temperature**4, 4.5334916 + -0.0056696171*temperature + 1.8473207e-05*temperature**2 + -1.7137094e-08*temperature**3 + 5.5454573e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.7459805 + 4.3450775e-05*temperature + 2.9705984e-07*temperature**2 + -6.8651806e-11*temperature**3 + 4.4134173e-15*temperature**4, 3.6129351 + -0.00095551327*temperature + 2.1442977e-06*temperature**2 + -3.1516323e-10*temperature**3 + -4.6430356e-13*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.8022392 + 0.0031464228*temperature + -1.0632185e-06*temperature**2 + 1.6619757e-10*temperature**3 + -9.799757e-15*temperature**4, 2.2589886 + 0.01005117*temperature + -1.3351763e-05*temperature**2 + 1.0092349e-08*temperature**3 + -3.0089028e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.209703 + 0.0029692911*temperature + -2.8555891e-07*temperature**2 + -1.63555e-10*temperature**3 + 3.0432589e-14*temperature**4, 2.851661 + 0.0056952331*temperature + 1.07114e-06*temperature**2 + -1.622612e-09*temperature**3 + -2.3511081e-13*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.8946362 + 0.0039895959*temperature + -1.598238e-06*temperature**2 + 2.9249395e-10*temperature**3 + -2.0094686e-14*temperature**4, 2.5243194 + 0.015960619*temperature + -1.8816354e-05*temperature**2 + 1.212554e-08*temperature**3 + -3.2357378e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1382.0), 6.59860456 + 0.00302778626*temperature + -1.07704346e-06*temperature**2 + 1.71666528e-10*temperature**3 + -1.01439391e-14*temperature**4, 2.64727989 + 0.0127505342*temperature + -1.04794236e-05*temperature**2 + 4.41432836e-09*temperature**3 + -7.57521466e-13*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1368.0), 5.89784885 + 0.00316789393*temperature + -1.11801064e-06*temperature**2 + 1.77243144e-10*temperature**3 + -1.04339177e-14*temperature**4, 3.78604952 + 0.00688667922*temperature + -3.21487864e-06*temperature**2 + 5.17195767e-10*temperature**3 + 1.19360788e-14*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1478.0), 6.22395134 + 0.00317864004*temperature + -1.09378755e-06*temperature**2 + 1.70735163e-10*temperature**3 + -9.95021955e-15*temperature**4, 3.63096317 + 0.00730282357*temperature + -2.28050003e-06*temperature**2 + -6.61271298e-10*temperature**3 + 3.62235752e-13*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.1521845 + 0.0023051761*temperature + -8.8033153e-07*temperature**2 + 1.4789098e-10*temperature**3 + -9.0977996e-15*temperature**4, 2.8269308 + 0.0088051688*temperature + -8.3866134e-06*temperature**2 + 4.8016964e-09*temperature**3 + -1.3313595e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.92664 + 0.0014879768*temperature + -5.68476e-07*temperature**2 + 1.0097038e-10*temperature**3 + -6.753351e-15*temperature**4, 3.298677 + 0.0014082404*temperature + -3.963222e-06*temperature**2 + 5.641515e-09*temperature**3 + -2.444854e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.5, 2.5),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 7.7026987 + 0.016044203*temperature + -5.283322e-06*temperature**2 + 7.629859e-10*temperature**3 + -3.9392284e-14*temperature**4, 1.0515518 + 0.02599198*temperature + 2.380054e-06*temperature**2 + -1.9609569e-08*temperature**3 + 9.373247e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 7.5341368 + 0.018872239*temperature + -6.2718491e-06*temperature**2 + 9.1475649e-10*temperature**3 + -4.7838069e-14*temperature**4, 0.93355381 + 0.026424579*temperature + 6.1059727e-06*temperature**2 + -2.1977499e-08*temperature**3 + 9.5149253e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.97567 + 0.008130591*temperature + -2.743624e-06*temperature**2 + 4.070304e-10*temperature**3 + -2.176017e-14*temperature**4, 3.409062 + 0.010738574*temperature + 1.891492e-06*temperature**2 + -7.158583e-09*temperature**3 + 2.867385e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.4041108 + 0.011723059*temperature + -4.2263137e-06*temperature**2 + 6.8372451e-10*temperature**3 + -4.0984863e-14*temperature**4, 4.7294595 + -0.0031932858*temperature + 4.7534921e-05*temperature**2 + -5.7458611e-08*temperature**3 + 2.1931112e-11*temperature**4),
                ])

    def get_species_enthalpies_rt(self, temperature):
        return self._pyro_make_array([
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.3372792 + -2.470123655e-05*temperature + 1.6648559266666665e-07*temperature**2 + -4.48915985e-11*temperature**3 + 4.00510752e-15*temperature**4 + -950.158922 / temperature, 2.34433112 + 0.003990260375*temperature + -6.4927169999999995e-06*temperature**2 + 5.03930235e-09*temperature**3 + -1.4752235220000002e-12*temperature**4 + -917.935173 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.50000001 + -1.154214865e-11*temperature + 5.385398266666667e-15*temperature**2 + -1.1837880875e-18*temperature**3 + 9.96394714e-23*temperature**4 + 25473.6599 / temperature, 2.5 + 3.526664095e-13*temperature + -6.653065466666667e-16*temperature**2 + 5.7520408e-19*temperature**3 + -1.855464664e-22*temperature**4 + 25473.6599 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.56942078 + -4.298705685e-05*temperature + 1.3982819633333334e-08*temperature**2 + -2.504444975e-12*temperature**3 + 2.4566738199999997e-16*temperature**4 + 29217.5791 / temperature, 3.1682671 + -0.00163965942*temperature + 2.2143546533333334e-06*temperature**2 + -1.53201656e-09*temperature**3 + 4.22531942e-13*temperature**4 + 29122.2592 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.28253784 + 0.00074154377*temperature + -2.526555563333333e-07*temperature**2 + 5.236763875e-11*temperature**3 + -4.33435588e-15*temperature**4 + -1088.45772 / temperature, 3.78245636 + -0.00149836708*temperature + 3.282434003333333e-06*temperature**2 + -2.4203237725e-09*temperature**3 + 6.48745674e-13*temperature**4 + -1063.94356 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.09288767 + 0.000274214858*temperature + 4.216840933333333e-08*temperature**2 + -2.19865389e-11*temperature**3 + 2.34824752e-15*temperature**4 + 3858.657 / temperature, 3.99201543 + -0.00120065876*temperature + 1.5393128033333333e-06*temperature**2 + -9.702833325e-10*temperature**3 + 2.7282294e-13*temperature**4 + 3615.08056 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.03399249 + 0.00108845902*temperature + -5.469083933333333e-08*temperature**2 + -2.426049675e-11*temperature**3 + 3.36401984e-15*temperature**4 + -30004.2971 / temperature, 4.19864056 + -0.00101821705*temperature + 2.17346737e-06*temperature**2 + -1.371992655e-09*temperature**3 + 3.54395634e-13*temperature**4 + -30293.7267 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.0172109 + 0.001119910065*temperature + -2.1121938333333332e-07*temperature**2 + 2.85615925e-11*temperature**3 + -2.1581707e-15*temperature**4 + 111.856713 / temperature, 4.30179801 + -0.002374560255*temperature + 7.0527630333333326e-06*temperature**2 + -6.06909735e-09*temperature**3 + 1.8584502480000002e-12*temperature**4 + 294.80804 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.16500285 + 0.00245415847*temperature + -6.337974166666666e-07*temperature**2 + 9.27964965e-11*temperature**3 + -5.7581661e-15*temperature**4 + -17861.7877 / temperature, 4.27611269 + -0.0002714112085*temperature + 5.5778567000000005e-06*temperature**2 + -5.394270325e-09*temperature**3 + 1.724908726e-12*temperature**4 + -17702.5821 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.49266888 + 2.39944642e-05*temperature + -2.414450066666667e-08*temperature**2 + 9.357275725e-12*temperature**3 + -9.745557859999999e-16*temperature**4 + 85451.2953 / temperature, 2.55423955 + -0.000160768862*temperature + 2.44597415e-07*temperature**2 + -1.8305872225e-10*temperature**3 + 5.33042892e-14*temperature**4 + 85443.8832 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.87846473 + 0.0004854568405*temperature + 4.814855166666666e-08*temperature**2 + -3.267196225e-11*temperature**3 + 3.5215876599999997e-15*temperature**4 + 71012.4364 / temperature, 3.48981665 + 0.0001619177705*temperature + -5.629968833333334e-07*temperature**2 + 7.905433175e-10*temperature**3 + -2.8121813400000003e-13*temperature**4 + 70797.2934 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.87410113 + 0.00182819646*temperature + -4.6964865666666667e-07*temperature**2 + 6.504488725e-11*temperature**3 + -3.75455134e-15*temperature**4 + 46263.604 / temperature, 3.76267867 + 0.0004844360715*temperature + 9.316328033333334e-07*temperature**2 + -9.627278825e-10*temperature**3 + 3.37483438e-13*temperature**4 + 46004.0401 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.29203842 + 0.002327943185*temperature + -6.706398233333333e-07*temperature**2 + 1.044765e-10*temperature**3 + -6.7943273e-15*temperature**4 + 50925.9997 / temperature, 4.19860411 + -0.001183307095*temperature + 2.744320733333333e-06*temperature**2 + -1.6720399525e-09*temperature**3 + 3.88629474e-13*temperature**4 + 50496.8163 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.28571772 + 0.003619950185*temperature + -9.957144933333333e-07*temperature**2 + 1.48921161e-10*temperature**3 + -9.34308788e-15*temperature**4 + 16775.5843 / temperature, 3.6735904 + 0.001005475875*temperature + 1.9100728533333335e-06*temperature**2 + -1.7177935625e-09*temperature**3 + 5.08771468e-13*temperature**4 + 16444.9988 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 0.074851495 + 0.00669547335*temperature + -1.9109526966666665e-06*temperature**2 + 3.057313375e-10*temperature**3 + -2.0363046000000002e-14*temperature**4 + -9468.34459 / temperature, 5.14987613 + -0.0068354894*temperature + 1.63933533e-05*temperature**2 + -1.211857565e-08*temperature**3 + 3.33387912e-12*temperature**4 + -10246.6476 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.71518561 + 0.001031263715*temperature + -3.329419236666667e-07*temperature**2 + 5.7513252e-11*temperature**3 + -4.07295432e-15*temperature**4 + -14151.8724 / temperature, 3.57953347 + -0.00030517684*temperature + 3.3893811e-07*temperature**2 + 2.26751471e-10*temperature**3 + -1.808848998e-13*temperature**4 + -14344.086 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.85746029 + 0.00220718513*temperature + -7.382713466666667e-07*temperature**2 + 1.30872547e-10*temperature**3 + -9.44168328e-15*temperature**4 + -48759.166 / temperature, 2.35677352 + 0.004492298385*temperature + -2.3745208966666665e-06*temperature**2 + 6.14797555e-10*temperature**3 + -2.8739909599999997e-14*temperature**4 + -48371.9697 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.77217438 + 0.00247847763*temperature + -8.281520433333334e-07*temperature**2 + 1.472904445e-10*temperature**3 + -1.067017422e-14*temperature**4 + 4011.91815 / temperature, 4.22118584 + -0.00162196266*temperature + 4.593314866666667e-06*temperature**2 + -3.328602325e-09*temperature**3 + 8.6753773e-13*temperature**4 + 3839.56496 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 1.76069008 + 0.00460000041*temperature + -1.4741960433333332e-06*temperature**2 + 2.5160303e-10*temperature**3 + -1.7677112800000002e-14*temperature**4 + -13995.8323 / temperature, 4.79372315 + -0.004954166845*temperature + 1.2440666933333332e-05*temperature**2 + -9.482131525e-09*temperature**3 + 2.63545304e-12*temperature**4 + -14308.9567 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.69266569 + 0.004322883985*temperature + -1.2503370666666667e-06*temperature**2 + 1.96808659e-10*temperature**3 + -1.2971084020000001e-14*temperature**4 + -3242.50627 / temperature, 3.86388918 + 0.00279836152*temperature + 1.9775726366666668e-06*temperature**2 + -2.6133003e-09*temperature**3 + 8.73934556e-13*temperature**4 + -3193.91367 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.770799 + 0.0039357485*temperature + -8.854613333333334e-07*temperature**2 + 9.8610775e-11*temperature**3 + -4.225232e-15*temperature**4 + 127.83252 / temperature, 2.106204 + 0.0036082975*temperature + 1.7794906666666667e-06*temperature**2 + -1.844409e-09*temperature**3 + 4.15122e-13*temperature**4 + 978.6011 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 1.78970791 + 0.0070469146*temperature + -2.12166945e-06*temperature**2 + 3.454277125e-10*temperature**3 + -2.3412044e-14*temperature**4 + -25374.8747 / temperature, 5.71539582 + -0.00761545645*temperature + 2.17480385e-05*temperature**2 + -1.7770172225e-08*temperature**3 + 5.2270539599999996e-12*temperature**4 + -25642.7656 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.16780652 + 0.00237610951*temperature + -6.1262359e-07*temperature**2 + 7.6047563e-11*temperature**3 + -3.5446554e-15*temperature**4 + 67121.065 / temperature, 2.88965733 + 0.00670498055*temperature + -9.4923167e-06*temperature**2 + 7.369776125e-09*temperature**3 + -2.18663022e-12*temperature**4 + 66839.3932 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.14756964 + 0.00298083332*temperature + -7.9098284e-07*temperature**2 + 1.1685304275e-10*temperature**3 + -7.22470426e-15*temperature**4 + 25935.9992 / temperature, 0.808681094 + 0.01168078145*temperature + -1.18390605e-05*temperature**2 + 7.003810925e-09*temperature**3 + -1.700145948e-12*temperature**4 + 26428.9807 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.016724 + 0.0051651146*temperature + -1.5602744966666665e-06*temperature**2 + 2.5440822e-10*temperature**3 + -1.725214082e-14*temperature**4 + 34612.8739 / temperature, 3.21246645 + 0.00075739581*temperature + 8.640313733333333e-06*temperature**2 + -8.941446175e-09*temperature**3 + 2.94301746e-12*temperature**4 + 34859.8468 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.03611116 + 0.00732270755*temperature + -2.2369263833333335e-06*temperature**2 + 3.680573075e-10*temperature**3 + -2.51412122e-14*temperature**4 + 4939.88614 / temperature, 3.95920148 + -0.003785261235*temperature + 1.9033009733333333e-05*temperature**2 + -1.7289718825e-08*temperature**3 + 5.3976874600000004e-12*temperature**4 + 5089.77593 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 1.95465642 + 0.0086986361*temperature + -2.6606888933333333e-06*temperature**2 + 4.380442225e-10*temperature**3 + -2.99283152e-14*temperature**4 + 12857.52 / temperature, 4.30646568 + -0.00209329446*temperature + 1.65714269e-05*temperature**2 + -1.497816515e-08*temperature**3 + 4.61018008e-12*temperature**4 + 12841.6265 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 1.0718815 + 0.01084263385*temperature + -3.3418689e-06*temperature**2 + 5.535300025e-10*temperature**3 + -3.8000578e-14*temperature**4 + -11426.3932 / temperature, 4.29142492 + -0.00275077135*temperature + 1.998127626666667e-05*temperature**2 + -1.7711657125e-08*temperature**3 + 5.37371542e-12*temperature**4 + -11522.2055 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.6282058 + 0.00204267005*temperature + -5.311515666666667e-07*temperature**2 + 7.156513e-11*temperature**3 + -3.8815663999999995e-15*temperature**4 + 19327.215 / temperature, 2.2517214 + 0.0088275105*temperature + -7.909700333333333e-06*temperature**2 + 4.31893975e-09*temperature**3 + -1.0132962200000001e-12*temperature**4 + 20059.449 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.51129732 + 0.004501798725*temperature + -1.3897987833333332e-06*temperature**2 + 2.308364705e-10*temperature**3 + -1.589676402e-14*temperature**4 + -7551.05311 / temperature, 2.1358363 + 0.00905943605*temperature + -5.798249133333333e-06*temperature**2 + 2.33599392e-09*temperature**3 + -4.0291523e-13*temperature**4 + -7042.91804 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.9238291 + 0.00339618*temperature + -8.552854666666667e-07*temperature**2 + 1.124696025e-10*temperature**3 + -5.9880202e-15*temperature**4 + 7264.626 / temperature, 1.2423733 + 0.0155361005*temperature + -1.6955621333333335e-05*temperature**2 + 1.078428275e-08*temperature**3 + -2.8029188e-12*temperature**4 + 8031.6143 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.4159429 + 8.7445325e-05*temperature + -3.967456333333333e-08*temperature**2 + 7.55656125e-12*temperature**3 + -4.0721964e-16*temperature**4 + 56133.773 / temperature, 2.5 + 56104.637 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.7836928 + 0.0006649215*temperature + -1.4159349e-07*temperature**2 + 1.958712525e-11*temperature**3 + -1.1008894000000001e-15*temperature**4 + 42120.848 / temperature, 3.4929085 + 0.00015589599*temperature + -4.963494666666667e-07*temperature**2 + 6.2041105e-10*temperature**3 + -2.0713934e-13*temperature**4 + 41880.629 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.8347421 + 0.0016036541*temperature + -3.1130268e-07*temperature**2 + 3.42573825e-11*temperature**3 + -1.58412288e-15*temperature**4 + 22171.957 / temperature, 4.2040029 + -0.00105306925*temperature + 2.3689449333333334e-06*temperature**2 + -1.402879925e-09*temperature**3 + 3.2881434e-13*temperature**4 + 21885.91 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.6344521 + 0.002833128*temperature + -5.759558666666667e-07*temperature**2 + 5.96679025e-11*temperature**3 + -2.5157571999999998e-15*temperature**4 + -6544.6958 / temperature, 4.2860274 + -0.0023302615*temperature + 7.239504333333334e-06*temperature**2 + -5.70222175e-09*temperature**3 + 1.65276092e-12*temperature**4 + -6741.7285 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.7667544 + 0.0014457541*temperature + -3.4722066666666664e-07*temperature**2 + 4.2106485e-11*temperature**3 + -2.0183792e-15*temperature**4 + 28650.697 / temperature, 4.3446927 + -0.0024248536*temperature + 6.6864863333333334e-06*temperature**2 + -5.431616e-09*temperature**3 + 1.58939078e-12*temperature**4 + 28791.973 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.2606056 + 0.00059555215*temperature + -1.4305682666666666e-07*temperature**2 + 1.736441725e-11*temperature**3 + -8.0672198e-16*temperature**4 + 9920.9746 / temperature, 4.2184763 + -0.002319488*temperature + 3.6803406666666667e-06*temperature**2 + -2.33403385e-09*temperature**3 + 5.607154e-13*temperature**4 + 9844.623 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.8847542 + 0.0010861978*temperature + -2.7602302e-07*temperature**2 + 3.9368775e-11*temperature**3 + -2.102179e-15*temperature**4 + 2316.4983 / temperature, 3.9440312 + -0.0007927145*temperature + 5.552604e-06*temperature**2 + -5.1188565e-09*temperature**3 + 1.56701128e-12*temperature**4 + 2896.6179 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.8230729 + 0.00131351255*temperature + -3.195029133333333e-07*temperature**2 + 4.000178e-11*temperature**3 + -1.95504606e-15*temperature**4 + 8073.4048 / temperature, 2.2571502 + 0.005652364*temperature + -4.557106333333333e-06*temperature**2 + 2.42049515e-09*temperature**3 + -5.8614364e-13*temperature**4 + 8741.7744 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.9792509 + 0.00174720295*temperature + -2.6183259333333334e-07*temperature**2 + 1.43698985e-11*temperature**3 + -3.8671832e-17*temperature**4 + 11750.582 / temperature, 4.5334916 + -0.00283480855*temperature + 6.1577356666666665e-06*temperature**2 + -4.2842735e-09*temperature**3 + 1.10909146e-12*temperature**4 + 11548.297 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.7459805 + 2.17253875e-05*temperature + 9.901994666666666e-08*temperature**2 + -1.71629515e-11*temperature**3 + 8.8268346e-16*temperature**4 + 51536.188 / temperature, 3.6129351 + -0.000477756635*temperature + 7.147659e-07*temperature**2 + -7.87908075e-11*temperature**3 + -9.286071199999999e-14*temperature**4 + 51708.34 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.8022392 + 0.0015732114*temperature + -3.544061666666667e-07*temperature**2 + 4.15493925e-11*temperature**3 + -1.9599514e-15*temperature**4 + 14407.292 / temperature, 2.2589886 + 0.005025585*temperature + -4.450587666666667e-06*temperature**2 + 2.52308725e-09*temperature**3 + -6.0178056e-13*temperature**4 + 14712.633 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.209703 + 0.00148464555*temperature + -9.518630333333333e-08*temperature**2 + -4.088875e-11*temperature**3 + 6.0865178e-15*temperature**4 + 27677.109 / temperature, 2.851661 + 0.00284761655*temperature + 3.5704666666666666e-07*temperature**2 + -4.05653e-10*temperature**3 + -4.7022162e-14*temperature**4 + 28637.82 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.8946362 + 0.00199479795*temperature + -5.32746e-07*temperature**2 + 7.31234875e-11*temperature**3 + -4.0189372e-15*temperature**4 + 53452.941 / temperature, 2.5243194 + 0.0079803095*temperature + -6.272118e-06*temperature**2 + 3.031385e-09*temperature**3 + -6.4714756e-13*temperature**4 + 54261.984 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1382.0), 6.59860456 + 0.00151389313*temperature + -3.5901448666666666e-07*temperature**2 + 4.2916632e-11*temperature**3 + -2.02878782e-15*temperature**4 + 17966.1339 / temperature, 2.64727989 + 0.0063752671*temperature + -3.4931411999999996e-06*temperature**2 + 1.10358209e-09*temperature**3 + -1.515042932e-13*temperature**4 + 19299.0252 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1368.0), 5.89784885 + 0.001583946965*temperature + -3.7267021333333335e-07*temperature**2 + 4.4310786e-11*temperature**3 + -2.0867835399999997e-15*temperature**4 + -3706.53331 / temperature, 3.78604952 + 0.00344333961*temperature + -1.0716262133333332e-06*temperature**2 + 1.2929894175e-10*temperature**3 + 2.38721576e-15*temperature**4 + -2826.984 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1478.0), 6.22395134 + 0.00158932002*temperature + -3.6459585e-07*temperature**2 + 4.268379075e-11*temperature**3 + -1.99004391e-15*temperature**4 + -16659.9344 / temperature, 3.63096317 + 0.003651411785*temperature + -7.601666766666667e-07*temperature**2 + -1.653178245e-10*temperature**3 + 7.244715039999999e-14*temperature**4 + -15587.3636 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.1521845 + 0.00115258805*temperature + -2.9344384333333333e-07*temperature**2 + 3.6972745e-11*temperature**3 + -1.81955992e-15*temperature**4 + 14004.123 / temperature, 2.8269308 + 0.0044025844*temperature + -2.7955378e-06*temperature**2 + 1.2004241e-09*temperature**3 + -2.662719e-13*temperature**4 + 14682.477 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.92664 + 0.0007439884*temperature + -1.8949200000000001e-07*temperature**2 + 2.5242595e-11*temperature**3 + -1.3506701999999999e-15*temperature**4 + -922.7977 / temperature, 3.298677 + 0.0007041202*temperature + -1.3210739999999999e-06*temperature**2 + 1.41037875e-09*temperature**3 + -4.889707999999999e-13*temperature**4 + -1020.8999 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.5 + -745.375 / temperature, 2.5 + -745.375 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 7.7026987 + 0.0080221015*temperature + -1.7611073333333333e-06*temperature**2 + 1.90746475e-10*temperature**3 + -7.8784568e-15*temperature**4 + 8298.4336 / temperature, 1.0515518 + 0.01299599*temperature + 7.933513333333333e-07*temperature**2 + -4.90239225e-09*temperature**3 + 1.8746494e-12*temperature**4 + 10631.863 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 7.5341368 + 0.0094361195*temperature + -2.0906163666666664e-06*temperature**2 + 2.286891225e-10*temperature**3 + -9.567613800000001e-15*temperature**4 + -16467.516 / temperature, 0.93355381 + 0.0132122895*temperature + 2.0353242333333332e-06*temperature**2 + -5.49437475e-09*temperature**3 + 1.90298506e-12*temperature**4 + -13958.52 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.97567 + 0.0040652955*temperature + -9.145413333333334e-07*temperature**2 + 1.017576e-10*temperature**3 + -4.352034e-15*temperature**4 + 490.3218 / temperature, 3.409062 + 0.005369287*temperature + 6.304973333333333e-07*temperature**2 + -1.78964575e-09*temperature**3 + 5.73477e-13*temperature**4 + 1521.4766 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.4041108 + 0.0058615295*temperature + -1.4087712333333332e-06*temperature**2 + 1.709311275e-10*temperature**3 + -8.1969726e-15*temperature**4 + -22593.122 / temperature, 4.7294595 + -0.0015966429*temperature + 1.5844973666666667e-05*temperature**2 + -1.436465275e-08*temperature**3 + 4.3862224e-12*temperature**4 + -21572.878 / temperature),
                ])

    def get_species_entropies_r(self, temperature):
        return self._pyro_make_array([
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.3372792*self.usr_np.log(temperature) + -4.94024731e-05*temperature + 2.49728389e-07*temperature**2 + -5.985546466666667e-11*temperature**3 + 5.0063844e-15*temperature**4 + -3.20502331, 2.34433112*self.usr_np.log(temperature) + 0.00798052075*temperature + -9.7390755e-06*temperature**2 + 6.7190698e-09*temperature**3 + -1.8440294025e-12*temperature**4 + 0.683010238),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.50000001*self.usr_np.log(temperature) + -2.30842973e-11*temperature + 8.0780974e-15*temperature**2 + -1.5783841166666668e-18*temperature**3 + 1.2454933925e-22*temperature**4 + -0.446682914, 2.5*self.usr_np.log(temperature) + 7.05332819e-13*temperature + -9.9795982e-16*temperature**2 + 7.669387733333333e-19*temperature**3 + -2.31933083e-22*temperature**4 + -0.446682853),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.56942078*self.usr_np.log(temperature) + -8.59741137e-05*temperature + 2.097422945e-08*temperature**2 + -3.3392599666666663e-12*temperature**3 + 3.070842275e-16*temperature**4 + 4.78433864, 3.1682671*self.usr_np.log(temperature) + -0.00327931884*temperature + 3.32153198e-06*temperature**2 + -2.0426887466666666e-09*temperature**3 + 5.281649275e-13*temperature**4 + 2.05193346),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.28253784*self.usr_np.log(temperature) + 0.00148308754*temperature + -3.789833345e-07*temperature**2 + 6.982351833333333e-11*temperature**3 + -5.41794485e-15*temperature**4 + 5.45323129, 3.78245636*self.usr_np.log(temperature) + -0.00299673416*temperature + 4.923651005e-06*temperature**2 + -3.2270983633333334e-09*temperature**3 + 8.109320925e-13*temperature**4 + 3.65767573),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.09288767*self.usr_np.log(temperature) + 0.000548429716*temperature + 6.3252614e-08*temperature**2 + -2.93153852e-11*temperature**3 + 2.9353094e-15*temperature**4 + 4.4766961, 3.99201543*self.usr_np.log(temperature) + -0.00240131752*temperature + 2.308969205e-06*temperature**2 + -1.29371111e-09*temperature**3 + 3.41028675e-13*temperature**4 + -0.103925458),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.03399249*self.usr_np.log(temperature) + 0.00217691804*temperature + -8.2036259e-08*temperature**2 + -3.2347329e-11*temperature**3 + 4.2050248e-15*temperature**4 + 4.9667701, 4.19864056*self.usr_np.log(temperature) + -0.0020364341*temperature + 3.260201055e-06*temperature**2 + -1.82932354e-09*temperature**3 + 4.429945425e-13*temperature**4 + -0.849032208),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.0172109*self.usr_np.log(temperature) + 0.00223982013*temperature + -3.16829075e-07*temperature**2 + 3.808212333333334e-11*temperature**3 + -2.697713375e-15*temperature**4 + 3.78510215, 4.30179801*self.usr_np.log(temperature) + -0.00474912051*temperature + 1.057914455e-05*temperature**2 + -8.0921298e-09*temperature**3 + 2.32306281e-12*temperature**4 + 3.71666245),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.16500285*self.usr_np.log(temperature) + 0.00490831694*temperature + -9.50696125e-07*temperature**2 + 1.2372866199999999e-10*temperature**3 + -7.197707625e-15*temperature**4 + 2.91615662, 4.27611269*self.usr_np.log(temperature) + -0.000542822417*temperature + 8.36678505e-06*temperature**2 + -7.192360433333333e-09*temperature**3 + 2.1561359075e-12*temperature**4 + 3.43505074),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.49266888*self.usr_np.log(temperature) + 4.79889284e-05*temperature + -3.6216751e-08*temperature**2 + 1.2476367633333334e-11*temperature**3 + -1.2181947325e-15*temperature**4 + 4.80150373, 2.55423955*self.usr_np.log(temperature) + -0.000321537724*temperature + 3.668961225e-07*temperature**2 + -2.440782963333333e-10*temperature**3 + 6.66303615e-14*temperature**4 + 4.53130848),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.87846473*self.usr_np.log(temperature) + 0.000970913681*temperature + 7.22228275e-08*temperature**2 + -4.3562616333333334e-11*temperature**3 + 4.401984575e-15*temperature**4 + 5.48497999, 3.48981665*self.usr_np.log(temperature) + 0.000323835541*temperature + -8.44495325e-07*temperature**2 + 1.0540577566666666e-09*temperature**3 + -3.515226675e-13*temperature**4 + 2.08401108),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.87410113*self.usr_np.log(temperature) + 0.00365639292*temperature + -7.04472985e-07*temperature**2 + 8.672651633333333e-11*temperature**3 + -4.693189175e-15*temperature**4 + 6.17119324, 3.76267867*self.usr_np.log(temperature) + 0.000968872143*temperature + 1.397449205e-06*temperature**2 + -1.2836371766666668e-09*temperature**3 + 4.218542975e-13*temperature**4 + 1.56253185),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.29203842*self.usr_np.log(temperature) + 0.00465588637*temperature + -1.005959735e-06*temperature**2 + 1.3930200000000002e-10*temperature**3 + -8.492909125e-15*temperature**4 + 8.62650169, 4.19860411*self.usr_np.log(temperature) + -0.00236661419*temperature + 4.1164811e-06*temperature**2 + -2.2293866033333336e-09*temperature**3 + 4.857868425e-13*temperature**4 + -0.769118967),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.28571772*self.usr_np.log(temperature) + 0.00723990037*temperature + -1.49357174e-06*temperature**2 + 1.98561548e-10*temperature**3 + -1.167885985e-14*temperature**4 + 8.48007179, 3.6735904*self.usr_np.log(temperature) + 0.00201095175*temperature + 2.86510928e-06*temperature**2 + -2.2903914166666666e-09*temperature**3 + 6.35964335e-13*temperature**4 + 1.60456433),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 0.074851495*self.usr_np.log(temperature) + 0.0133909467*temperature + -2.866429045e-06*temperature**2 + 4.076417833333333e-10*temperature**3 + -2.54538075e-14*temperature**4 + 18.437318, 5.14987613*self.usr_np.log(temperature) + -0.0136709788*temperature + 2.459002995e-05*temperature**2 + -1.6158100866666668e-08*temperature**3 + 4.1673489e-12*temperature**4 + -4.64130376),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.71518561*self.usr_np.log(temperature) + 0.00206252743*temperature + -4.994128855e-07*temperature**2 + 7.6684336e-11*temperature**3 + -5.0911929e-15*temperature**4 + 7.81868772, 3.57953347*self.usr_np.log(temperature) + -0.00061035368*temperature + 5.08407165e-07*temperature**2 + 3.023352946666667e-10*temperature**3 + -2.2610612475e-13*temperature**4 + 3.50840928),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.85746029*self.usr_np.log(temperature) + 0.00441437026*temperature + -1.10740702e-06*temperature**2 + 1.7449672933333335e-10*temperature**3 + -1.18021041e-14*temperature**4 + 2.27163806, 2.35677352*self.usr_np.log(temperature) + 0.00898459677*temperature + -3.561781345e-06*temperature**2 + 8.197300733333333e-10*temperature**3 + -3.5924887e-14*temperature**4 + 9.90105222),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.77217438*self.usr_np.log(temperature) + 0.00495695526*temperature + -1.242228065e-06*temperature**2 + 1.9638725933333335e-10*temperature**3 + -1.3337717775e-14*temperature**4 + 9.79834492, 4.22118584*self.usr_np.log(temperature) + -0.00324392532*temperature + 6.8899723e-06*temperature**2 + -4.438136433333333e-09*temperature**3 + 1.0844221625e-12*temperature**4 + 3.39437243),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 1.76069008*self.usr_np.log(temperature) + 0.00920000082*temperature + -2.211294065e-06*temperature**2 + 3.3547070666666664e-10*temperature**3 + -2.2096391e-14*temperature**4 + 13.656323, 4.79372315*self.usr_np.log(temperature) + -0.00990833369*temperature + 1.86610004e-05*temperature**2 + -1.2642842033333333e-08*temperature**3 + 3.2943163e-12*temperature**4 + 0.6028129),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.69266569*self.usr_np.log(temperature) + 0.00864576797*temperature + -1.8755056e-06*temperature**2 + 2.6241154533333335e-10*temperature**3 + -1.6213855025e-14*temperature**4 + 5.81043215, 3.86388918*self.usr_np.log(temperature) + 0.00559672304*temperature + 2.966358955e-06*temperature**2 + -3.4844004000000002e-09*temperature**3 + 1.092418195e-12*temperature**4 + 5.47302243),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.770799*self.usr_np.log(temperature) + 0.007871497*temperature + -1.328192e-06*temperature**2 + 1.3148103333333333e-10*temperature**3 + -5.28154e-15*temperature**4 + 2.929575, 2.106204*self.usr_np.log(temperature) + 0.007216595*temperature + 2.669236e-06*temperature**2 + -2.459212e-09*temperature**3 + 5.189025e-13*temperature**4 + 13.152177),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 1.78970791*self.usr_np.log(temperature) + 0.0140938292*temperature + -3.182504175e-06*temperature**2 + 4.6057028333333335e-10*temperature**3 + -2.9265055e-14*temperature**4 + 14.5023623, 5.71539582*self.usr_np.log(temperature) + -0.0152309129*temperature + 3.262205775e-05*temperature**2 + -2.3693562966666666e-08*temperature**3 + 6.53381745e-12*temperature**4 + -1.50409823),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.16780652*self.usr_np.log(temperature) + 0.00475221902*temperature + -9.18935385e-07*temperature**2 + 1.0139675066666666e-10*temperature**3 + -4.43081925e-15*temperature**4 + 6.63589475, 2.88965733*self.usr_np.log(temperature) + 0.0134099611*temperature + -1.423847505e-05*temperature**2 + 9.826368166666667e-09*temperature**3 + -2.733287775e-12*temperature**4 + 6.22296438),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.14756964*self.usr_np.log(temperature) + 0.00596166664*temperature + -1.18647426e-06*temperature**2 + 1.55804057e-10*temperature**3 + -9.030880325e-15*temperature**4 + -1.23028121, 0.808681094*self.usr_np.log(temperature) + 0.0233615629*temperature + -1.775859075e-05*temperature**2 + 9.338414566666667e-09*temperature**3 + -2.125182435e-12*temperature**4 + 13.9397051),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.016724*self.usr_np.log(temperature) + 0.0103302292*temperature + -2.340411745e-06*temperature**2 + 3.3921096e-10*temperature**3 + -2.1565176025e-14*temperature**4 + 7.78732378, 3.21246645*self.usr_np.log(temperature) + 0.00151479162*temperature + 1.29604706e-05*temperature**2 + -1.1921928233333333e-08*temperature**3 + 3.678771825e-12*temperature**4 + 8.51054025),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.03611116*self.usr_np.log(temperature) + 0.0146454151*temperature + -3.355389575e-06*temperature**2 + 4.907430766666667e-10*temperature**3 + -3.142651525e-14*temperature**4 + 10.3053693, 3.95920148*self.usr_np.log(temperature) + -0.00757052247*temperature + 2.85495146e-05*temperature**2 + -2.3052958433333332e-08*temperature**3 + 6.747109325e-12*temperature**4 + 4.09733096),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 1.95465642*self.usr_np.log(temperature) + 0.0173972722*temperature + -3.99103334e-06*temperature**2 + 5.840589633333333e-10*temperature**3 + -3.7410394e-14*temperature**4 + 13.4624343, 4.30646568*self.usr_np.log(temperature) + -0.00418658892*temperature + 2.485714035e-05*temperature**2 + -1.9970886866666665e-08*temperature**3 + 5.7627251e-12*temperature**4 + 4.70720924),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 1.0718815*self.usr_np.log(temperature) + 0.0216852677*temperature + -5.01280335e-06*temperature**2 + 7.380400033333333e-10*temperature**3 + -4.75007225e-14*temperature**4 + 15.1156107, 4.29142492*self.usr_np.log(temperature) + -0.0055015427*temperature + 2.99719144e-05*temperature**2 + -2.3615542833333335e-08*temperature**3 + 6.717144275e-12*temperature**4 + 2.66682316),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.6282058*self.usr_np.log(temperature) + 0.0040853401*temperature + -7.9672735e-07*temperature**2 + 9.542017333333333e-11*temperature**3 + -4.851958e-15*temperature**4 + -3.9302595, 2.2517214*self.usr_np.log(temperature) + 0.017655021*temperature + -1.18645505e-05*temperature**2 + 5.758586333333334e-09*temperature**3 + -1.266620275e-12*temperature**4 + 12.490417),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.51129732*self.usr_np.log(temperature) + 0.00900359745*temperature + -2.084698175e-06*temperature**2 + 3.0778196066666667e-10*temperature**3 + -1.9870955025e-14*temperature**4 + 0.632247205, 2.1358363*self.usr_np.log(temperature) + 0.0181188721*temperature + -8.6973737e-06*temperature**2 + 3.11465856e-09*temperature**3 + -5.036440375e-13*temperature**4 + 12.215648),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.9238291*self.usr_np.log(temperature) + 0.00679236*temperature + -1.2829282e-06*temperature**2 + 1.4995947e-10*temperature**3 + -7.48502525e-15*temperature**4 + -7.6017742, 1.2423733*self.usr_np.log(temperature) + 0.031072201*temperature + -2.5433432e-05*temperature**2 + 1.4379043666666668e-08*temperature**3 + -3.5036485e-12*temperature**4 + 13.874319),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.4159429*self.usr_np.log(temperature) + 0.00017489065*temperature + -5.9511845e-08*temperature**2 + 1.0075415e-11*temperature**3 + -5.0902455e-16*temperature**4 + 4.6496096, 2.5*self.usr_np.log(temperature) + 4.1939087),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.7836928*self.usr_np.log(temperature) + 0.001329843*temperature + -2.12390235e-07*temperature**2 + 2.6116167e-11*temperature**3 + -1.37611175e-15*temperature**4 + 5.7407799, 3.4929085*self.usr_np.log(temperature) + 0.00031179198*temperature + -7.445242e-07*temperature**2 + 8.272147333333333e-10*temperature**3 + -2.58924175e-13*temperature**4 + 1.8483278),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.8347421*self.usr_np.log(temperature) + 0.0032073082*temperature + -4.6695402e-07*temperature**2 + 4.567651e-11*temperature**3 + -1.9801536e-15*temperature**4 + 6.5204163, 4.2040029*self.usr_np.log(temperature) + -0.0021061385*temperature + 3.5534174e-06*temperature**2 + -1.8705065666666667e-09*temperature**3 + 4.11017925e-13*temperature**4 + -0.14184248),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.6344521*self.usr_np.log(temperature) + 0.005666256*temperature + -8.639338e-07*temperature**2 + 7.955720333333333e-11*temperature**3 + -3.1446965e-15*temperature**4 + 6.5662928, 4.2860274*self.usr_np.log(temperature) + -0.004660523*temperature + 1.08592565e-05*temperature**2 + -7.602962333333334e-09*temperature**3 + 2.06595115e-12*temperature**4 + -0.62537277),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.7667544*self.usr_np.log(temperature) + 0.0028915082*temperature + -5.20831e-07*temperature**2 + 5.614198e-11*temperature**3 + -2.522974e-15*temperature**4 + 4.4705067, 4.3446927*self.usr_np.log(temperature) + -0.0048497072*temperature + 1.00297295e-05*temperature**2 + -7.242154666666667e-09*temperature**3 + 1.986738475e-12*temperature**4 + 2.977941),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.2606056*self.usr_np.log(temperature) + 0.0011911043*temperature + -2.1458524e-07*temperature**2 + 2.3152556333333334e-11*temperature**3 + -1.008402475e-15*temperature**4 + 6.3693027, 4.2184763*self.usr_np.log(temperature) + -0.004638976*temperature + 5.520511e-06*temperature**2 + -3.1120451333333334e-09*temperature**3 + 7.0089425e-13*temperature**4 + 2.2808464),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.8847542*self.usr_np.log(temperature) + 0.0021723956*temperature + -4.1403453e-07*temperature**2 + 5.24917e-11*temperature**3 + -2.62772375e-15*temperature**4 + -0.11741695, 3.9440312*self.usr_np.log(temperature) + -0.001585429*temperature + 8.328906e-06*temperature**2 + -6.825142e-09*temperature**3 + 1.9587641e-12*temperature**4 + 6.3119917),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.8230729*self.usr_np.log(temperature) + 0.0026270251*temperature + -4.7925437e-07*temperature**2 + 5.3335706666666663e-11*temperature**3 + -2.443807575e-15*temperature**4 + -2.2017207, 2.2571502*self.usr_np.log(temperature) + 0.011304728*temperature + -6.8356595e-06*temperature**2 + 3.227326866666667e-09*temperature**3 + -7.3267955e-13*temperature**4 + 10.757992),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.9792509*self.usr_np.log(temperature) + 0.0034944059*temperature + -3.9274889e-07*temperature**2 + 1.9159864666666668e-11*temperature**3 + -4.833979e-17*temperature**4 + 8.6063728, 4.5334916*self.usr_np.log(temperature) + -0.0056696171*temperature + 9.2366035e-06*temperature**2 + -5.712364666666667e-09*temperature**3 + 1.386364325e-12*temperature**4 + 1.7498417),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.7459805*self.usr_np.log(temperature) + 4.3450775e-05*temperature + 1.4852992e-07*temperature**2 + -2.2883935333333333e-11*temperature**3 + 1.103354325e-15*temperature**4 + 2.7867601, 3.6129351*self.usr_np.log(temperature) + -0.00095551327*temperature + 1.07214885e-06*temperature**2 + -1.0505441e-10*temperature**3 + -1.1607589e-13*temperature**4 + 3.9804995),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.8022392*self.usr_np.log(temperature) + 0.0031464228*temperature + -5.3160925e-07*temperature**2 + 5.539919e-11*temperature**3 + -2.44993925e-15*temperature**4 + 1.5754601, 2.2589886*self.usr_np.log(temperature) + 0.01005117*temperature + -6.6758815e-06*temperature**2 + 3.3641163333333337e-09*temperature**3 + -7.522257e-13*temperature**4 + 8.9164419),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.209703*self.usr_np.log(temperature) + 0.0029692911*temperature + -1.42779455e-07*temperature**2 + -5.451833333333334e-11*temperature**3 + 7.60814725e-15*temperature**4 + -4.444478, 2.851661*self.usr_np.log(temperature) + 0.0056952331*temperature + 5.3557e-07*temperature**2 + -5.408706666666667e-10*temperature**3 + -5.87777025e-14*temperature**4 + 8.9927511),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.8946362*self.usr_np.log(temperature) + 0.0039895959*temperature + -7.99119e-07*temperature**2 + 9.749798333333333e-11*temperature**3 + -5.0236715e-15*temperature**4 + -5.1030502, 2.5243194*self.usr_np.log(temperature) + 0.015960619*temperature + -9.408177e-06*temperature**2 + 4.041846666666667e-09*temperature**3 + -8.0893445e-13*temperature**4 + 11.67587),
                self.usr_np.where(self.usr_np.greater(temperature, 1382.0), 6.59860456*self.usr_np.log(temperature) + 0.00302778626*temperature + -5.3852173e-07*temperature**2 + 5.7222176e-11*temperature**3 + -2.535984775e-15*temperature**4 + -10.3306599, 2.64727989*self.usr_np.log(temperature) + 0.0127505342*temperature + -5.2397118e-06*temperature**2 + 1.4714427866666667e-09*temperature**3 + -1.893803665e-13*temperature**4 + 10.7332972),
                self.usr_np.where(self.usr_np.greater(temperature, 1368.0), 5.89784885*self.usr_np.log(temperature) + 0.00316789393*temperature + -5.5900532e-07*temperature**2 + 5.9081048e-11*temperature**3 + -2.608479425e-15*temperature**4 + -6.18167825, 3.78604952*self.usr_np.log(temperature) + 0.00688667922*temperature + -1.60743932e-06*temperature**2 + 1.72398589e-10*temperature**3 + 2.9840197e-15*temperature**4 + 5.63292162),
                self.usr_np.where(self.usr_np.greater(temperature, 1478.0), 6.22395134*self.usr_np.log(temperature) + 0.00317864004*temperature + -5.46893775e-07*temperature**2 + 5.6911721e-11*temperature**3 + -2.4875548875e-15*temperature**4 + -8.38224741, 3.63096317*self.usr_np.log(temperature) + 0.00730282357*temperature + -1.140250015e-06*temperature**2 + -2.2042376600000002e-10*temperature**3 + 9.0558938e-14*temperature**4 + 6.19457727),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.1521845*self.usr_np.log(temperature) + 0.0023051761*temperature + -4.40165765e-07*temperature**2 + 4.929699333333333e-11*temperature**3 + -2.2744499e-15*temperature**4 + -2.544266, 2.8269308*self.usr_np.log(temperature) + 0.0088051688*temperature + -4.1933067e-06*temperature**2 + 1.6005654666666665e-09*temperature**3 + -3.32839875e-13*temperature**4 + 9.5504646),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.92664*self.usr_np.log(temperature) + 0.0014879768*temperature + -2.84238e-07*temperature**2 + 3.3656793333333334e-11*temperature**3 + -1.68833775e-15*temperature**4 + 5.980528, 3.298677*self.usr_np.log(temperature) + 0.0014082404*temperature + -1.981611e-06*temperature**2 + 1.8805050000000002e-09*temperature**3 + -6.112135e-13*temperature**4 + 3.950372),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.5*self.usr_np.log(temperature) + 4.366, 2.5*self.usr_np.log(temperature) + 4.366),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 7.7026987*self.usr_np.log(temperature) + 0.016044203*temperature + -2.641661e-06*temperature**2 + 2.5432863333333333e-10*temperature**3 + -9.848071e-15*temperature**4 + -15.48018, 1.0515518*self.usr_np.log(temperature) + 0.02599198*temperature + 1.190027e-06*temperature**2 + -6.536523e-09*temperature**3 + 2.34331175e-12*temperature**4 + 21.122559),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 7.5341368*self.usr_np.log(temperature) + 0.018872239*temperature + -3.13592455e-06*temperature**2 + 3.0491883000000003e-10*temperature**3 + -1.195951725e-14*temperature**4 + -17.892349, 0.93355381*self.usr_np.log(temperature) + 0.026424579*temperature + 3.05298635e-06*temperature**2 + -7.3258329999999994e-09*temperature**3 + 2.378731325e-12*temperature**4 + 19.201691),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.97567*self.usr_np.log(temperature) + 0.008130591*temperature + -1.371812e-06*temperature**2 + 1.356768e-10*temperature**3 + -5.4400425e-15*temperature**4 + -5.045251, 3.409062*self.usr_np.log(temperature) + 0.010738574*temperature + 9.45746e-07*temperature**2 + -2.386194333333333e-09*temperature**3 + 7.1684625e-13*temperature**4 + 9.55829),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.4041108*self.usr_np.log(temperature) + 0.011723059*temperature + -2.11315685e-06*temperature**2 + 2.2790817e-10*temperature**3 + -1.024621575e-14*temperature**4 + -3.4807917, 4.7294595*self.usr_np.log(temperature) + -0.0031932858*temperature + 2.37674605e-05*temperature**2 + -1.9152870333333333e-08*temperature**3 + 5.482778e-12*temperature**4 + 4.1030159),
                ])

    def get_species_gibbs_rt(self, temperature):
        h0_rt = self.get_species_enthalpies_rt(temperature)
        s0_r = self.get_species_entropies_r(temperature)
        return h0_rt - s0_r

    def get_equilibrium_constants(self, temperature):
        rt = self.gas_constant * temperature
        c0 = self.usr_np.log(self.one_atm / rt)

        g0_rt = self.get_species_gibbs_rt(temperature)
        return self._pyro_make_array([
                    g0_rt[3] + -1*2.0*g0_rt[2] + -1*-1.0*c0,
                    g0_rt[4] + -1*(g0_rt[1] + g0_rt[2]) + -1*-1.0*c0,
                    g0_rt[1] + g0_rt[4] + -1*(g0_rt[0] + g0_rt[2]),
                    g0_rt[3] + g0_rt[4] + -1*(g0_rt[6] + g0_rt[2]),
                    g0_rt[6] + g0_rt[4] + -1*(g0_rt[7] + g0_rt[2]),
                    g0_rt[14] + g0_rt[1] + -1*(g0_rt[9] + g0_rt[2]),
                    g0_rt[1] + g0_rt[16] + -1*(g0_rt[10] + g0_rt[2]),
                    g0_rt[14] + g0_rt[0] + -1*(g0_rt[11] + g0_rt[2]),
                    g0_rt[1] + g0_rt[16] + -1*(g0_rt[11] + g0_rt[2]),
                    g0_rt[17] + g0_rt[1] + -1*(g0_rt[12] + g0_rt[2]),
                    g0_rt[12] + g0_rt[4] + -1*(g0_rt[13] + g0_rt[2]),
                    g0_rt[15] + -1*(g0_rt[14] + g0_rt[2]) + -1*-1.0*c0,
                    g0_rt[14] + g0_rt[4] + -1*(g0_rt[16] + g0_rt[2]),
                    g0_rt[15] + g0_rt[1] + -1*(g0_rt[16] + g0_rt[2]),
                    g0_rt[16] + g0_rt[4] + -1*(g0_rt[17] + g0_rt[2]),
                    g0_rt[17] + g0_rt[4] + -1*(g0_rt[18] + g0_rt[2]),
                    g0_rt[17] + g0_rt[4] + -1*(g0_rt[19] + g0_rt[2]),
                    g0_rt[18] + g0_rt[4] + -1*(g0_rt[20] + g0_rt[2]),
                    g0_rt[19] + g0_rt[4] + -1*(g0_rt[20] + g0_rt[2]),
                    g0_rt[9] + g0_rt[14] + -1*(g0_rt[21] + g0_rt[2]),
                    g0_rt[1] + g0_rt[27] + -1*(g0_rt[22] + g0_rt[2]),
                    g0_rt[21] + g0_rt[4] + -1*(g0_rt[22] + g0_rt[2]),
                    g0_rt[10] + g0_rt[14] + -1*(g0_rt[22] + g0_rt[2]),
                    g0_rt[28] + g0_rt[1] + -1*(g0_rt[23] + g0_rt[2]),
                    g0_rt[12] + g0_rt[16] + -1*(g0_rt[24] + g0_rt[2]),
                    g0_rt[17] + g0_rt[12] + -1*(g0_rt[25] + g0_rt[2]),
                    g0_rt[25] + g0_rt[4] + -1*(g0_rt[26] + g0_rt[2]),
                    2.0*g0_rt[14] + g0_rt[1] + -1*(g0_rt[27] + g0_rt[2]) + -1*c0,
                    g0_rt[27] + g0_rt[4] + -1*(g0_rt[28] + g0_rt[2]),
                    g0_rt[10] + g0_rt[15] + -1*(g0_rt[28] + g0_rt[2]),
                    g0_rt[15] + g0_rt[2] + -1*(g0_rt[14] + g0_rt[3]),
                    g0_rt[16] + g0_rt[6] + -1*(g0_rt[17] + g0_rt[3]),
                    g0_rt[6] + -1*(g0_rt[1] + g0_rt[3]) + -1*-1.0*c0,
                    g0_rt[6] + -1*(g0_rt[1] + g0_rt[3]) + -1*-1.0*c0,
                    g0_rt[6] + -1*(g0_rt[1] + g0_rt[3]) + -1*-1.0*c0,
                    g0_rt[6] + -1*(g0_rt[1] + g0_rt[3]) + -1*-1.0*c0,
                    g0_rt[6] + -1*(g0_rt[1] + g0_rt[3]) + -1*-1.0*c0,
                    g0_rt[2] + g0_rt[4] + -1*(g0_rt[1] + g0_rt[3]),
                    g0_rt[0] + -1*2.0*g0_rt[1] + -1*-1.0*c0,
                    g0_rt[0] + -1*2.0*g0_rt[1] + -1*-1.0*c0,
                    g0_rt[0] + -1*2.0*g0_rt[1] + -1*-1.0*c0,
                    g0_rt[0] + -1*2.0*g0_rt[1] + -1*-1.0*c0,
                    g0_rt[5] + -1*(g0_rt[1] + g0_rt[4]) + -1*-1.0*c0,
                    g0_rt[5] + g0_rt[2] + -1*(g0_rt[1] + g0_rt[6]),
                    g0_rt[0] + g0_rt[3] + -1*(g0_rt[1] + g0_rt[6]),
                    2.0*g0_rt[4] + -1*(g0_rt[1] + g0_rt[6]),
                    g0_rt[0] + g0_rt[6] + -1*(g0_rt[1] + g0_rt[7]),
                    g0_rt[5] + g0_rt[4] + -1*(g0_rt[1] + g0_rt[7]),
                    g0_rt[8] + g0_rt[0] + -1*(g0_rt[9] + g0_rt[1]),
                    g0_rt[12] + -1*(g0_rt[10] + g0_rt[1]) + -1*-1.0*c0,
                    g0_rt[9] + g0_rt[0] + -1*(g0_rt[11] + g0_rt[1]),
                    g0_rt[13] + -1*(g0_rt[12] + g0_rt[1]) + -1*-1.0*c0,
                    g0_rt[12] + g0_rt[0] + -1*(g0_rt[13] + g0_rt[1]),
                    g0_rt[17] + -1*(g0_rt[1] + g0_rt[16]) + -1*-1.0*c0,
                    g0_rt[14] + g0_rt[0] + -1*(g0_rt[1] + g0_rt[16]),
                    g0_rt[18] + -1*(g0_rt[17] + g0_rt[1]) + -1*-1.0*c0,
                    g0_rt[19] + -1*(g0_rt[17] + g0_rt[1]) + -1*-1.0*c0,
                    g0_rt[0] + g0_rt[16] + -1*(g0_rt[17] + g0_rt[1]),
                    g0_rt[20] + -1*(g0_rt[18] + g0_rt[1]) + -1*-1.0*c0,
                    g0_rt[17] + g0_rt[0] + -1*(g0_rt[18] + g0_rt[1]),
                    g0_rt[12] + g0_rt[4] + -1*(g0_rt[18] + g0_rt[1]),
                    g0_rt[11] + g0_rt[5] + -1*(g0_rt[18] + g0_rt[1]),
                    g0_rt[20] + -1*(g0_rt[19] + g0_rt[1]) + -1*-1.0*c0,
                    g0_rt[18] + g0_rt[1] + -1*(g0_rt[19] + g0_rt[1]),
                    g0_rt[17] + g0_rt[0] + -1*(g0_rt[19] + g0_rt[1]),
                    g0_rt[12] + g0_rt[4] + -1*(g0_rt[19] + g0_rt[1]),
                    g0_rt[11] + g0_rt[5] + -1*(g0_rt[19] + g0_rt[1]),
                    g0_rt[18] + g0_rt[0] + -1*(g0_rt[20] + g0_rt[1]),
                    g0_rt[19] + g0_rt[0] + -1*(g0_rt[20] + g0_rt[1]),
                    g0_rt[22] + -1*(g0_rt[21] + g0_rt[1]) + -1*-1.0*c0,
                    g0_rt[23] + -1*(g0_rt[22] + g0_rt[1]) + -1*-1.0*c0,
                    g0_rt[24] + -1*(g0_rt[23] + g0_rt[1]) + -1*-1.0*c0,
                    g0_rt[22] + g0_rt[0] + -1*(g0_rt[23] + g0_rt[1]),
                    g0_rt[25] + -1*(g0_rt[24] + g0_rt[1]) + -1*-1.0*c0,
                    g0_rt[23] + g0_rt[0] + -1*(g0_rt[24] + g0_rt[1]),
                    g0_rt[26] + -1*(g0_rt[25] + g0_rt[1]) + -1*-1.0*c0,
                    g0_rt[24] + g0_rt[0] + -1*(g0_rt[25] + g0_rt[1]),
                    g0_rt[25] + g0_rt[0] + -1*(g0_rt[26] + g0_rt[1]),
                    g0_rt[11] + g0_rt[14] + -1*(g0_rt[1] + g0_rt[27]),
                    g0_rt[0] + g0_rt[27] + -1*(g0_rt[28] + g0_rt[1]),
                    g0_rt[12] + g0_rt[14] + -1*(g0_rt[28] + g0_rt[1]),
                    g0_rt[28] + g0_rt[1] + -1*(g0_rt[1] + g0_rt[29]),
                    g0_rt[17] + -1*(g0_rt[14] + g0_rt[0]) + -1*-1.0*c0,
                    g0_rt[1] + g0_rt[5] + -1*(g0_rt[0] + g0_rt[4]),
                    g0_rt[7] + -1*2.0*g0_rt[4] + -1*-1.0*c0,
                    g0_rt[5] + g0_rt[2] + -1*2.0*g0_rt[4],
                    g0_rt[5] + g0_rt[3] + -1*(g0_rt[6] + g0_rt[4]),
                    g0_rt[5] + g0_rt[6] + -1*(g0_rt[7] + g0_rt[4]),
                    g0_rt[5] + g0_rt[6] + -1*(g0_rt[7] + g0_rt[4]),
                    g0_rt[14] + g0_rt[1] + -1*(g0_rt[8] + g0_rt[4]),
                    g0_rt[1] + g0_rt[16] + -1*(g0_rt[9] + g0_rt[4]),
                    g0_rt[17] + g0_rt[1] + -1*(g0_rt[10] + g0_rt[4]),
                    g0_rt[9] + g0_rt[5] + -1*(g0_rt[10] + g0_rt[4]),
                    g0_rt[17] + g0_rt[1] + -1*(g0_rt[11] + g0_rt[4]),
                    g0_rt[20] + -1*(g0_rt[12] + g0_rt[4]) + -1*-1.0*c0,
                    g0_rt[10] + g0_rt[5] + -1*(g0_rt[12] + g0_rt[4]),
                    g0_rt[11] + g0_rt[5] + -1*(g0_rt[12] + g0_rt[4]),
                    g0_rt[12] + g0_rt[5] + -1*(g0_rt[13] + g0_rt[4]),
                    g0_rt[15] + g0_rt[1] + -1*(g0_rt[14] + g0_rt[4]),
                    g0_rt[14] + g0_rt[5] + -1*(g0_rt[16] + g0_rt[4]),
                    g0_rt[5] + g0_rt[16] + -1*(g0_rt[17] + g0_rt[4]),
                    g0_rt[17] + g0_rt[5] + -1*(g0_rt[18] + g0_rt[4]),
                    g0_rt[17] + g0_rt[5] + -1*(g0_rt[19] + g0_rt[4]),
                    g0_rt[18] + g0_rt[5] + -1*(g0_rt[20] + g0_rt[4]),
                    g0_rt[19] + g0_rt[5] + -1*(g0_rt[20] + g0_rt[4]),
                    g0_rt[1] + g0_rt[27] + -1*(g0_rt[21] + g0_rt[4]),
                    g0_rt[28] + g0_rt[1] + -1*(g0_rt[22] + g0_rt[4]),
                    g0_rt[1] + g0_rt[29] + -1*(g0_rt[22] + g0_rt[4]),
                    g0_rt[21] + g0_rt[5] + -1*(g0_rt[22] + g0_rt[4]),
                    g0_rt[12] + g0_rt[14] + -1*(g0_rt[22] + g0_rt[4]),
                    g0_rt[22] + g0_rt[5] + -1*(g0_rt[23] + g0_rt[4]),
                    g0_rt[23] + g0_rt[5] + -1*(g0_rt[24] + g0_rt[4]),
                    g0_rt[25] + g0_rt[5] + -1*(g0_rt[26] + g0_rt[4]),
                    g0_rt[5] + g0_rt[27] + -1*(g0_rt[28] + g0_rt[4]),
                    g0_rt[7] + g0_rt[3] + -1*2.0*g0_rt[6],
                    g0_rt[7] + g0_rt[3] + -1*2.0*g0_rt[6],
                    g0_rt[17] + g0_rt[4] + -1*(g0_rt[10] + g0_rt[6]),
                    g0_rt[13] + g0_rt[3] + -1*(g0_rt[12] + g0_rt[6]),
                    g0_rt[19] + g0_rt[4] + -1*(g0_rt[12] + g0_rt[6]),
                    g0_rt[15] + g0_rt[4] + -1*(g0_rt[14] + g0_rt[6]),
                    g0_rt[7] + g0_rt[16] + -1*(g0_rt[17] + g0_rt[6]),
                    g0_rt[14] + g0_rt[2] + -1*(g0_rt[8] + g0_rt[3]),
                    g0_rt[21] + g0_rt[1] + -1*(g0_rt[8] + g0_rt[10]),
                    g0_rt[22] + g0_rt[1] + -1*(g0_rt[8] + g0_rt[12]),
                    g0_rt[16] + g0_rt[2] + -1*(g0_rt[9] + g0_rt[3]),
                    g0_rt[10] + g0_rt[1] + -1*(g0_rt[9] + g0_rt[0]),
                    g0_rt[17] + g0_rt[1] + -1*(g0_rt[9] + g0_rt[5]),
                    g0_rt[22] + g0_rt[1] + -1*(g0_rt[9] + g0_rt[10]),
                    g0_rt[23] + g0_rt[1] + -1*(g0_rt[9] + g0_rt[12]),
                    g0_rt[24] + g0_rt[1] + -1*(g0_rt[9] + g0_rt[13]),
                    g0_rt[27] + -1*(g0_rt[9] + g0_rt[14]) + -1*-1.0*c0,
                    g0_rt[14] + g0_rt[16] + -1*(g0_rt[9] + g0_rt[15]),
                    g0_rt[28] + g0_rt[1] + -1*(g0_rt[9] + g0_rt[17]),
                    g0_rt[22] + g0_rt[14] + -1*(g0_rt[9] + g0_rt[27]),
                    -0.17364695002734*temperature,
                    g0_rt[12] + g0_rt[1] + -1*(g0_rt[10] + g0_rt[0]),
                    g0_rt[22] + g0_rt[0] + -1*2.0*g0_rt[10],
                    g0_rt[24] + g0_rt[1] + -1*(g0_rt[10] + g0_rt[12]),
                    2.0*g0_rt[12] + -1*(g0_rt[10] + g0_rt[13]),
                    g0_rt[28] + -1*(g0_rt[10] + g0_rt[14]) + -1*-1.0*c0,
                    g0_rt[23] + g0_rt[14] + -1*(g0_rt[10] + g0_rt[27]),
                    g0_rt[10] + g0_rt[47] + -1*(g0_rt[11] + g0_rt[47]),
                    g0_rt[48] + g0_rt[10] + -1*(g0_rt[48] + g0_rt[11]),
                    g0_rt[14] + g0_rt[1] + g0_rt[4] + -1*(g0_rt[11] + g0_rt[3]) + -1*c0,
                    g0_rt[14] + g0_rt[5] + -1*(g0_rt[11] + g0_rt[3]),
                    g0_rt[12] + g0_rt[1] + -1*(g0_rt[11] + g0_rt[0]),
                    g0_rt[20] + -1*(g0_rt[11] + g0_rt[5]) + -1*-1.0*c0,
                    g0_rt[10] + g0_rt[5] + -1*(g0_rt[11] + g0_rt[5]),
                    g0_rt[24] + g0_rt[1] + -1*(g0_rt[11] + g0_rt[12]),
                    2.0*g0_rt[12] + -1*(g0_rt[11] + g0_rt[13]),
                    g0_rt[10] + g0_rt[14] + -1*(g0_rt[11] + g0_rt[14]),
                    g0_rt[10] + g0_rt[15] + -1*(g0_rt[11] + g0_rt[15]),
                    g0_rt[17] + g0_rt[14] + -1*(g0_rt[11] + g0_rt[15]),
                    g0_rt[25] + g0_rt[12] + -1*(g0_rt[26] + g0_rt[11]),
                    g0_rt[19] + g0_rt[2] + -1*(g0_rt[12] + g0_rt[3]),
                    g0_rt[17] + g0_rt[4] + -1*(g0_rt[12] + g0_rt[3]),
                    g0_rt[13] + g0_rt[6] + -1*(g0_rt[12] + g0_rt[7]),
                    g0_rt[26] + -1*2.0*g0_rt[12] + -1*-1.0*c0,
                    g0_rt[25] + g0_rt[1] + -1*2.0*g0_rt[12],
                    g0_rt[13] + g0_rt[14] + -1*(g0_rt[12] + g0_rt[16]),
                    g0_rt[13] + g0_rt[16] + -1*(g0_rt[17] + g0_rt[12]),
                    g0_rt[18] + g0_rt[13] + -1*(g0_rt[12] + g0_rt[20]),
                    g0_rt[19] + g0_rt[13] + -1*(g0_rt[12] + g0_rt[20]),
                    g0_rt[23] + g0_rt[13] + -1*(g0_rt[24] + g0_rt[12]),
                    g0_rt[25] + g0_rt[13] + -1*(g0_rt[26] + g0_rt[12]),
                    g0_rt[14] + g0_rt[1] + -1*g0_rt[16] + -1*c0,
                    g0_rt[14] + g0_rt[1] + -1*g0_rt[16] + -1*c0,
                    g0_rt[14] + g0_rt[6] + -1*(g0_rt[16] + g0_rt[3]),
                    g0_rt[17] + g0_rt[6] + -1*(g0_rt[18] + g0_rt[3]),
                    g0_rt[17] + g0_rt[6] + -1*(g0_rt[19] + g0_rt[3]),
                    g0_rt[14] + g0_rt[16] + -1*(g0_rt[21] + g0_rt[3]),
                    g0_rt[22] + g0_rt[1] + -1*(g0_rt[21] + g0_rt[0]),
                    g0_rt[17] + g0_rt[16] + -1*(g0_rt[23] + g0_rt[3]),
                    g0_rt[22] + g0_rt[0] + -1*g0_rt[24] + -1*c0,
                    g0_rt[24] + g0_rt[6] + -1*(g0_rt[25] + g0_rt[3]),
                    2.0*g0_rt[14] + g0_rt[4] + -1*(g0_rt[27] + g0_rt[3]) + -1*c0,
                    g0_rt[22] + 2.0*g0_rt[14] + -1*2.0*g0_rt[27] + -1*c0,
                    g0_rt[47] + g0_rt[2] + -1*(g0_rt[30] + g0_rt[35]),
                    g0_rt[35] + g0_rt[2] + -1*(g0_rt[30] + g0_rt[3]),
                    g0_rt[1] + g0_rt[35] + -1*(g0_rt[30] + g0_rt[4]),
                    g0_rt[47] + g0_rt[3] + -1*(g0_rt[37] + g0_rt[2]),
                    2.0*g0_rt[35] + -1*(g0_rt[37] + g0_rt[2]),
                    g0_rt[47] + g0_rt[4] + -1*(g0_rt[1] + g0_rt[37]),
                    g0_rt[6] + g0_rt[47] + -1*(g0_rt[37] + g0_rt[4]),
                    g0_rt[47] + g0_rt[2] + -1*g0_rt[37] + -1*c0,
                    g0_rt[36] + g0_rt[4] + -1*(g0_rt[6] + g0_rt[35]),
                    g0_rt[36] + -1*(g0_rt[35] + g0_rt[2]) + -1*-1.0*c0,
                    g0_rt[35] + g0_rt[3] + -1*(g0_rt[36] + g0_rt[2]),
                    g0_rt[35] + g0_rt[4] + -1*(g0_rt[1] + g0_rt[36]),
                    g0_rt[1] + g0_rt[35] + -1*(g0_rt[31] + g0_rt[2]),
                    g0_rt[0] + g0_rt[30] + -1*(g0_rt[1] + g0_rt[31]),
                    g0_rt[1] + g0_rt[38] + -1*(g0_rt[31] + g0_rt[4]),
                    g0_rt[5] + g0_rt[30] + -1*(g0_rt[31] + g0_rt[4]),
                    g0_rt[38] + g0_rt[2] + -1*(g0_rt[31] + g0_rt[3]),
                    g0_rt[35] + g0_rt[4] + -1*(g0_rt[31] + g0_rt[3]),
                    g0_rt[1] + g0_rt[47] + -1*(g0_rt[30] + g0_rt[31]),
                    g0_rt[0] + g0_rt[38] + -1*(g0_rt[5] + g0_rt[31]),
                    g0_rt[47] + g0_rt[4] + -1*(g0_rt[31] + g0_rt[35]),
                    g0_rt[1] + g0_rt[37] + -1*(g0_rt[31] + g0_rt[35]),
                    g0_rt[31] + g0_rt[4] + -1*(g0_rt[32] + g0_rt[2]),
                    g0_rt[1] + g0_rt[38] + -1*(g0_rt[32] + g0_rt[2]),
                    g0_rt[0] + g0_rt[31] + -1*(g0_rt[1] + g0_rt[32]),
                    g0_rt[5] + g0_rt[31] + -1*(g0_rt[32] + g0_rt[4]),
                    g0_rt[1] + g0_rt[47] + -1*g0_rt[34] + -1*c0,
                    g0_rt[1] + g0_rt[47] + -1*g0_rt[34] + -1*c0,
                    g0_rt[6] + g0_rt[47] + -1*(g0_rt[34] + g0_rt[3]),
                    g0_rt[47] + g0_rt[4] + -1*(g0_rt[34] + g0_rt[2]),
                    g0_rt[31] + g0_rt[35] + -1*(g0_rt[34] + g0_rt[2]),
                    g0_rt[0] + g0_rt[47] + -1*(g0_rt[1] + g0_rt[34]),
                    g0_rt[5] + g0_rt[47] + -1*(g0_rt[34] + g0_rt[4]),
                    g0_rt[13] + g0_rt[47] + -1*(g0_rt[12] + g0_rt[34]),
                    g0_rt[38] + -1*(g0_rt[1] + g0_rt[35]) + -1*-1.0*c0,
                    g0_rt[35] + g0_rt[4] + -1*(g0_rt[38] + g0_rt[2]),
                    g0_rt[0] + g0_rt[35] + -1*(g0_rt[1] + g0_rt[38]),
                    g0_rt[5] + g0_rt[35] + -1*(g0_rt[38] + g0_rt[4]),
                    g0_rt[6] + g0_rt[35] + -1*(g0_rt[38] + g0_rt[3]),
                    g0_rt[14] + g0_rt[30] + -1*(g0_rt[39] + g0_rt[2]),
                    g0_rt[1] + g0_rt[46] + -1*(g0_rt[39] + g0_rt[4]),
                    g0_rt[40] + g0_rt[4] + -1*(g0_rt[39] + g0_rt[5]),
                    g0_rt[46] + g0_rt[2] + -1*(g0_rt[39] + g0_rt[3]),
                    g0_rt[1] + g0_rt[40] + -1*(g0_rt[39] + g0_rt[0]),
                    g0_rt[14] + g0_rt[35] + -1*(g0_rt[46] + g0_rt[2]),
                    g0_rt[14] + g0_rt[31] + -1*(g0_rt[1] + g0_rt[46]),
                    g0_rt[14] + g0_rt[1] + g0_rt[35] + -1*(g0_rt[46] + g0_rt[4]) + -1*c0,
                    g0_rt[14] + g0_rt[47] + -1*(g0_rt[30] + g0_rt[46]),
                    g0_rt[15] + g0_rt[35] + -1*(g0_rt[46] + g0_rt[3]),
                    g0_rt[14] + g0_rt[30] + -1*g0_rt[46] + -1*c0,
                    g0_rt[14] + g0_rt[37] + -1*(g0_rt[46] + g0_rt[35]),
                    g0_rt[15] + g0_rt[47] + -1*(g0_rt[46] + g0_rt[35]),
                    g0_rt[39] + g0_rt[1] + -1*g0_rt[40] + -1*c0,
                    g0_rt[1] + g0_rt[46] + -1*(g0_rt[40] + g0_rt[2]),
                    g0_rt[14] + g0_rt[31] + -1*(g0_rt[40] + g0_rt[2]),
                    g0_rt[39] + g0_rt[4] + -1*(g0_rt[40] + g0_rt[2]),
                    g0_rt[1] + g0_rt[44] + -1*(g0_rt[40] + g0_rt[4]),
                    g0_rt[1] + g0_rt[45] + -1*(g0_rt[40] + g0_rt[4]),
                    g0_rt[14] + g0_rt[32] + -1*(g0_rt[40] + g0_rt[4]),
                    g0_rt[41] + -1*(g0_rt[1] + g0_rt[40]) + -1*-1.0*c0,
                    g0_rt[10] + g0_rt[47] + -1*(g0_rt[41] + g0_rt[30]),
                    g0_rt[39] + g0_rt[30] + -1*(g0_rt[8] + g0_rt[47]),
                    g0_rt[40] + g0_rt[30] + -1*(g0_rt[9] + g0_rt[47]),
                    g0_rt[42] + -1*(g0_rt[9] + g0_rt[47]) + -1*-1.0*c0,
                    g0_rt[40] + g0_rt[31] + -1*(g0_rt[10] + g0_rt[47]),
                    g0_rt[40] + g0_rt[31] + -1*(g0_rt[11] + g0_rt[47]),
                    g0_rt[39] + g0_rt[2] + -1*(g0_rt[8] + g0_rt[35]),
                    g0_rt[14] + g0_rt[30] + -1*(g0_rt[8] + g0_rt[35]),
                    g0_rt[40] + g0_rt[2] + -1*(g0_rt[9] + g0_rt[35]),
                    g0_rt[1] + g0_rt[46] + -1*(g0_rt[9] + g0_rt[35]),
                    g0_rt[16] + g0_rt[30] + -1*(g0_rt[9] + g0_rt[35]),
                    g0_rt[1] + g0_rt[45] + -1*(g0_rt[10] + g0_rt[35]),
                    g0_rt[40] + g0_rt[4] + -1*(g0_rt[10] + g0_rt[35]),
                    g0_rt[1] + g0_rt[43] + -1*(g0_rt[10] + g0_rt[35]),
                    g0_rt[1] + g0_rt[45] + -1*(g0_rt[11] + g0_rt[35]),
                    g0_rt[40] + g0_rt[4] + -1*(g0_rt[11] + g0_rt[35]),
                    g0_rt[1] + g0_rt[43] + -1*(g0_rt[11] + g0_rt[35]),
                    g0_rt[5] + g0_rt[40] + -1*(g0_rt[12] + g0_rt[35]),
                    g0_rt[41] + g0_rt[4] + -1*(g0_rt[12] + g0_rt[35]),
                    g0_rt[14] + g0_rt[1] + g0_rt[47] + -1*(g0_rt[42] + g0_rt[2]) + -1*c0,
                    g0_rt[40] + g0_rt[35] + -1*(g0_rt[42] + g0_rt[2]),
                    g0_rt[16] + g0_rt[47] + g0_rt[2] + -1*(g0_rt[42] + g0_rt[3]) + -1*c0,
                    g0_rt[1] + g0_rt[16] + g0_rt[47] + -1*(g0_rt[42] + g0_rt[4]) + -1*c0,
                    g0_rt[10] + g0_rt[47] + -1*(g0_rt[1] + g0_rt[42]),
                    g0_rt[15] + g0_rt[31] + -1*(g0_rt[45] + g0_rt[2]),
                    g0_rt[14] + g0_rt[38] + -1*(g0_rt[45] + g0_rt[2]),
                    g0_rt[46] + g0_rt[4] + -1*(g0_rt[45] + g0_rt[2]),
                    g0_rt[14] + g0_rt[32] + -1*(g0_rt[1] + g0_rt[45]),
                    g0_rt[0] + g0_rt[46] + -1*(g0_rt[1] + g0_rt[45]),
                    g0_rt[5] + g0_rt[46] + -1*(g0_rt[45] + g0_rt[4]),
                    g0_rt[15] + g0_rt[32] + -1*(g0_rt[45] + g0_rt[4]),
                    g0_rt[14] + g0_rt[31] + -1*g0_rt[45] + -1*c0,
                    g0_rt[1] + g0_rt[45] + -1*(g0_rt[1] + g0_rt[43]),
                    g0_rt[40] + g0_rt[4] + -1*(g0_rt[1] + g0_rt[43]),
                    g0_rt[14] + g0_rt[32] + -1*(g0_rt[1] + g0_rt[43]),
                    g0_rt[1] + g0_rt[45] + -1*(g0_rt[1] + g0_rt[44]),
                    g0_rt[14] + g0_rt[43] + -1*(g0_rt[27] + g0_rt[35]),
                    g0_rt[1] + g0_rt[41] + -1*(g0_rt[12] + g0_rt[30]),
                    g0_rt[0] + g0_rt[40] + -1*(g0_rt[12] + g0_rt[30]),
                    g0_rt[0] + g0_rt[32] + -1*(g0_rt[1] + g0_rt[33]),
                    g0_rt[5] + g0_rt[32] + -1*(g0_rt[33] + g0_rt[4]),
                    g0_rt[32] + g0_rt[4] + -1*(g0_rt[33] + g0_rt[2]),
                    g0_rt[14] + g0_rt[38] + -1*(g0_rt[15] + g0_rt[31]),
                    g0_rt[46] + g0_rt[35] + -1*(g0_rt[39] + g0_rt[36]),
                    g0_rt[15] + g0_rt[37] + -1*(g0_rt[46] + g0_rt[36]),
                    g0_rt[14] + g0_rt[35] + -1*(g0_rt[15] + g0_rt[30]),
                    -0.17364695002734*temperature,
                    g0_rt[51] + g0_rt[1] + -1*(g0_rt[24] + g0_rt[2]),
                    g0_rt[52] + g0_rt[1] + -1*(g0_rt[25] + g0_rt[2]),
                    g0_rt[5] + g0_rt[3] + -1*(g0_rt[6] + g0_rt[4]),
                    -0.17364695002734*temperature,
                    g0_rt[12] + -1*(g0_rt[9] + g0_rt[0]) + -1*-1.0*c0,
                    -0.17364695002734*temperature,
                    g0_rt[17] + g0_rt[2] + -1*(g0_rt[10] + g0_rt[3]),
                    -0.17364695002734*temperature,
                    -0.17364695002734*temperature,
                    g0_rt[51] + g0_rt[2] + -1*(g0_rt[23] + g0_rt[3]),
                    g0_rt[22] + g0_rt[6] + -1*(g0_rt[23] + g0_rt[3]),
                    g0_rt[51] + g0_rt[4] + -1*(g0_rt[52] + g0_rt[2]),
                    -0.17364695002734*temperature,
                    -0.17364695002734*temperature,
                    g0_rt[51] + g0_rt[0] + -1*(g0_rt[52] + g0_rt[1]),
                    -0.17364695002734*temperature,
                    -0.17364695002734*temperature,
                    -0.17364695002734*temperature,
                    -0.17364695002734*temperature,
                    g0_rt[51] + -1*(g0_rt[28] + g0_rt[1]) + -1*-1.0*c0,
                    -0.17364695002734*temperature,
                    -0.17364695002734*temperature,
                    -0.17364695002734*temperature,
                    g0_rt[12] + g0_rt[16] + -1*(g0_rt[51] + g0_rt[1]),
                    g0_rt[28] + g0_rt[0] + -1*(g0_rt[51] + g0_rt[1]),
                    g0_rt[28] + g0_rt[5] + -1*(g0_rt[51] + g0_rt[4]),
                    g0_rt[18] + g0_rt[16] + -1*(g0_rt[51] + g0_rt[4]),
                    g0_rt[50] + -1*(g0_rt[25] + g0_rt[12]) + -1*-1.0*c0,
                    g0_rt[49] + g0_rt[4] + -1*(g0_rt[50] + g0_rt[2]),
                    g0_rt[49] + g0_rt[0] + -1*(g0_rt[50] + g0_rt[1]),
                    g0_rt[49] + g0_rt[5] + -1*(g0_rt[50] + g0_rt[4]),
                    g0_rt[50] + g0_rt[6] + -1*(g0_rt[49] + g0_rt[7]),
                    g0_rt[49] + g0_rt[13] + -1*(g0_rt[50] + g0_rt[12]),
                    g0_rt[49] + -1*(g0_rt[24] + g0_rt[12]) + -1*-1.0*c0,
                    g0_rt[25] + g0_rt[17] + -1*(g0_rt[49] + g0_rt[2]),
                    g0_rt[50] + -1*(g0_rt[49] + g0_rt[1]) + -1*-1.0*c0,
                    g0_rt[25] + g0_rt[12] + -1*(g0_rt[49] + g0_rt[1]),
                    g0_rt[25] + g0_rt[18] + -1*(g0_rt[49] + g0_rt[4]),
                    g0_rt[50] + g0_rt[3] + -1*(g0_rt[49] + g0_rt[6]),
                    -0.17364695002734*temperature,
                    2.0*g0_rt[25] + -1*(g0_rt[49] + g0_rt[12]),
                ])

    def get_temperature(self, enthalpy_or_energy, t_guess, y, do_energy=False):
        if do_energy is False:
            pv_fun = self.get_mixture_specific_heat_cp_mass
            he_fun = self.get_mixture_enthalpy_mass
        else:
            pv_fun = self.get_mixture_specific_heat_cv_mass
            he_fun = self.get_mixture_internal_energy_mass

        num_iter = 500
        tol = 1.0e-6
        ones = self._pyro_zeros_like(enthalpy_or_energy) + 1.0
        t_i = t_guess * ones

        for _ in range(num_iter):
            f = enthalpy_or_energy - he_fun(t_i, y)
            j = -pv_fun(t_i, y)
            dt = -f / j
            t_i += dt
            if self._pyro_norm(dt, np.inf) < tol:
                return t_i

        raise RuntimeError("Temperature iteration failed to converge")

    def get_falloff_rates(self, temperature, concentrations, k_fwd):
        ones = self._pyro_zeros_like(temperature) + 1.0
        k_high = self._pyro_make_array([
            self.usr_np.exp(16.70588231586044 + -1*(1200.1785873945562 / temperature)),
            600000000000.0001,
            self.usr_np.exp(30.262909956065194 + -0.534*self.usr_np.log(temperature) + -1*(269.72566995533845 / temperature)),
            self.usr_np.exp(20.809443533187462 + 0.48*self.usr_np.log(temperature) + -1*(-130.8370787096791 / temperature)),
            self.usr_np.exp(20.107079697522593 + 0.454*self.usr_np.log(temperature) + -1*(1811.5903205955567 / temperature)),
            self.usr_np.exp(20.107079697522593 + 0.454*self.usr_np.log(temperature) + -1*(1308.370787096791 / temperature)),
            self.usr_np.exp(20.77680660387444 + 0.5*self.usr_np.log(temperature) + -1*(43.27687988089385 / temperature)),
            self.usr_np.exp(21.611157094298868 + 0.515*self.usr_np.log(temperature) + -1*(25.160976674938286 / temperature)),
            100000000000000.02*temperature**-1.0,
            self.usr_np.exp(22.446032434687513 + -1*(1207.7268803970378 / temperature)),
            self.usr_np.exp(22.528270532924488 + 0.27*self.usr_np.log(temperature) + -1*(140.9014693796544 / temperature)),
            self.usr_np.exp(20.107079697522593 + 0.454*self.usr_np.log(temperature) + -1*(915.8595509677536 / temperature)),
            self.usr_np.exp(33.88677115768191 + -0.99*self.usr_np.log(temperature) + -1*(795.0868629280499 / temperature)),
            self.usr_np.exp(10.6689553946757 + 1.5*self.usr_np.log(temperature) + -1*(40056.27486650175 / temperature)),
            74000000000.00002*temperature**-0.37,
            self.usr_np.exp(35.56481799074396 + -1.43*self.usr_np.log(temperature) + -1*(669.2819795533584 / temperature)),
            50000000000.00001,
            self.usr_np.exp(20.51254480563076 + 0.5*self.usr_np.log(temperature) + -1*(2269.5200960794336 / temperature)),
            self.usr_np.exp(33.80896522997915 + -1.16*self.usr_np.log(temperature) + -1*(576.1863658560868 / temperature)),
            self.usr_np.exp(31.846107295846778 + -1.18*self.usr_np.log(temperature) + -1*(329.1055749081928 / temperature)),
            self.usr_np.exp(29.710462657608385 + 0.44*self.usr_np.log(temperature) + -1*(43664.358921687905 / temperature)),
            self.usr_np.exp(25.09397871172002 + -1*(28190.358266600855 / temperature)),
            33000000000.000004,
            3100000000.0000005*temperature**0.15,
            self.usr_np.exp(21.401299379696308 + 0.43*self.usr_np.log(temperature) + -1*(-186.19122739454332 / temperature)),
            self.usr_np.exp(20.002747459590335 + 0.422*self.usr_np.log(temperature) + -1*(-883.1502812903339 / temperature)),
            9430000000.000002,
            self.usr_np.exp(7.843848638152472 + 1.6*self.usr_np.log(temperature) + -1*(2868.3513409429647 / temperature)),
            36130000000.00001,
                ])

        k_low = self._pyro_make_array([
            self.usr_np.exp(20.215768003273094 + -1*(1509.6586004962971 / temperature)),
            self.usr_np.exp(46.09092257303419 + -2.76*self.usr_np.log(temperature) + -1*(805.1512535980252 / temperature)),
            self.usr_np.exp(63.13297182861224 + -4.76*self.usr_np.log(temperature) + -1*(1227.8556617369884 / temperature)),
            self.usr_np.exp(42.350749824532706 + -2.57*self.usr_np.log(temperature) + -1*(213.86830173697544 / temperature)),
            self.usr_np.exp(60.10622931831569 + -4.82*self.usr_np.log(temperature) + -1*(3286.0235537469403 / temperature)),
            self.usr_np.exp(56.050499592221364 + -4.8*self.usr_np.log(temperature) + -1*(2797.9006062531375 / temperature)),
            self.usr_np.exp(59.037099382212084 + -4.65*self.usr_np.log(temperature) + -1*(2556.35523017373 / temperature)),
            self.usr_np.exp(82.12949370292915 + -7.44*self.usr_np.log(temperature) + -1*(7085.331031662621 / temperature)),
            self.usr_np.exp(63.491553350821555 + -4.8*self.usr_np.log(temperature) + -1*(956.117113647655 / temperature)),
            self.usr_np.exp(79.62289422852989 + -7.27*self.usr_np.log(temperature) + -1*(3633.2450318610886 / temperature)),
            self.usr_np.exp(55.59851446847831 + -3.86*self.usr_np.log(temperature) + -1*(1670.6888512159023 / temperature)),
            self.usr_np.exp(82.38223772401966 + -7.62*self.usr_np.log(temperature) + -1*(3507.440148486397 / temperature)),
            self.usr_np.exp(81.278612893528 + -7.08*self.usr_np.log(temperature) + -1*(3364.022581439249 / temperature)),
            self.usr_np.exp(49.97762777047805 + -3.42*self.usr_np.log(temperature) + -1*(42446.56765062089 / temperature)),
            self.usr_np.exp(28.463930238863654 + -0.9*self.usr_np.log(temperature) + -1*(-855.4732069479018 / temperature)),
            self.usr_np.exp(70.46384715094126 + -5.92*self.usr_np.log(temperature) + -1*(1580.1093351861243 / temperature)),
            self.usr_np.exp(51.646413239482754 + -3.74*self.usr_np.log(temperature) + -1*(974.2330168536105 / temperature)),
            self.usr_np.exp(63.159338704452985 + -5.11*self.usr_np.log(temperature) + -1*(3570.342590173743 / temperature)),
            self.usr_np.exp(74.31399475265133 + -6.36*self.usr_np.log(temperature) + -1*(2536.226448833779 / temperature)),
            self.usr_np.exp(81.81425368641372 + -7.03*self.usr_np.log(temperature) + -1*(1389.892351523591 / temperature)),
            self.usr_np.exp(110.98150931075307 + -9.3*self.usr_np.log(temperature) + -1*(49214.870376179286 / temperature)),
            self.usr_np.exp(27.180035492518574 + -1*(28502.35437737009 / temperature)),
            self.usr_np.exp(46.38817409650213 + -3.4*self.usr_np.log(temperature) + -1*(956.117113647655 / temperature)),
            self.usr_np.exp(44.01148103135436 + -3.16*self.usr_np.log(temperature) + -1*(372.38245478908664 / temperature)),
            self.usr_np.exp(45.321890694949374 + -2.8*self.usr_np.log(temperature) + -1*(296.8995247642718 / temperature)),
            self.usr_np.exp(82.90499191865092 + -7.63*self.usr_np.log(temperature) + -1*(1939.4080821042432 / temperature)),
            self.usr_np.exp(157.5727349584867 + -16.82*self.usr_np.log(temperature) + -1*(6574.563205161375 / temperature)),
            self.usr_np.exp(132.3459625893287 + -14.6*self.usr_np.log(temperature) + -1*(9143.498923672574 / temperature)),
            self.usr_np.exp(128.12831981076212 + -13.545*self.usr_np.log(temperature) + -1*(5715.0642419454825 / temperature)),
                ])

        reduced_pressure = self._pyro_make_array([
            (0.5*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 3.5*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + 6.0*concentrations[3] + concentrations[1] + concentrations[2] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[0]/k_high[0],
            (0.7*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[1]/k_high[1],
            (0.7*concentrations[48] + 3.0*concentrations[26] + 3.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[2]/k_high[2],
            (0.7*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[3]/k_high[3],
            (3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[48] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[4]/k_high[4],
            (3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[48] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[5]/k_high[5],
            (3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[48] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[6]/k_high[6],
            (3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[48] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[7]/k_high[7],
            (0.7*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[8]/k_high[8],
            (0.7*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[9]/k_high[9],
            (0.7*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[10]/k_high[10],
            (0.7*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[11]/k_high[11],
            (0.7*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[12]/k_high[12],
            (0.7*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[13]/k_high[13],
            (0.7*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[14]/k_high[14],
            (3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[48] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[15]/k_high[15],
            (0.7*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[16]/k_high[16],
            (0.7*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[17]/k_high[17],
            (3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[48] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[18]/k_high[18],
            (0.7*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[19]/k_high[19],
            (0.7*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[20]/k_high[20],
            (0.625*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[21]/k_high[21],
            (0.7*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[22]/k_high[22],
            (concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[23]/k_high[23],
            (0.7*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[24]/k_high[24],
            (0.7*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[25]/k_high[25],
            (0.7*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[26]/k_high[26],
            (0.7*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[27]/k_high[27],
            (0.7*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])*k_low[28]/k_high[28],
                            ])

        falloff_center = self._pyro_make_array([
            1,
            self.usr_np.log10(0.43799999999999994*self.usr_np.exp((-1*temperature) / 91.0) + 0.562*self.usr_np.exp((-1*temperature) / 5836.0) + self.usr_np.exp(-8552.0 / temperature)),
            self.usr_np.log10(0.21699999999999997*self.usr_np.exp((-1*temperature) / 74.0) + 0.783*self.usr_np.exp((-1*temperature) / 2941.0) + self.usr_np.exp(-6964.0 / temperature)),
            self.usr_np.log10(0.21760000000000002*self.usr_np.exp((-1*temperature) / 271.0) + 0.7824*self.usr_np.exp((-1*temperature) / 2755.0) + self.usr_np.exp(-6570.0 / temperature)),
            self.usr_np.log10(0.2813*self.usr_np.exp((-1*temperature) / 103.00000000000001) + 0.7187*self.usr_np.exp((-1*temperature) / 1291.0) + self.usr_np.exp(-4160.0 / temperature)),
            self.usr_np.log10(0.242*self.usr_np.exp((-1*temperature) / 94.0) + 0.758*self.usr_np.exp((-1*temperature) / 1555.0) + self.usr_np.exp(-4200.0 / temperature)),
            self.usr_np.log10(0.4*self.usr_np.exp((-1*temperature) / 100.0) + 0.6*self.usr_np.exp((-1*temperature) / 90000.0) + self.usr_np.exp(-10000.0 / temperature)),
            self.usr_np.log10(0.30000000000000004*self.usr_np.exp((-1*temperature) / 100.0) + 0.7*self.usr_np.exp((-1*temperature) / 90000.0) + self.usr_np.exp(-10000.0 / temperature)),
            self.usr_np.log10(0.3536*self.usr_np.exp((-1*temperature) / 132.0) + 0.6464*self.usr_np.exp((-1*temperature) / 1315.0) + self.usr_np.exp(-5566.0 / temperature)),
            self.usr_np.log10(0.24929999999999997*self.usr_np.exp((-1*temperature) / 98.50000000000001) + 0.7507*self.usr_np.exp((-1*temperature) / 1302.0) + self.usr_np.exp(-4167.0 / temperature)),
            self.usr_np.log10(0.21799999999999997*self.usr_np.exp((-1*temperature) / 207.49999999999997) + 0.782*self.usr_np.exp((-1*temperature) / 2663.0) + self.usr_np.exp(-6095.0 / temperature)),
            self.usr_np.log10(0.024700000000000055*self.usr_np.exp((-1*temperature) / 209.99999999999997) + 0.9753*self.usr_np.exp((-1*temperature) / 983.9999999999999) + self.usr_np.exp(-4374.0 / temperature)),
            self.usr_np.log10(0.15780000000000005*self.usr_np.exp((-1*temperature) / 125.0) + 0.8422*self.usr_np.exp((-1*temperature) / 2219.0) + self.usr_np.exp(-6882.0 / temperature)),
            self.usr_np.log10(0.06799999999999995*self.usr_np.exp((-1*temperature) / 197.00000000000003) + 0.932*self.usr_np.exp((-1*temperature) / 1540.0) + self.usr_np.exp(-10300.0 / temperature)),
            self.usr_np.log10(0.26539999999999997*self.usr_np.exp((-1*temperature) / 94.0) + 0.7346*self.usr_np.exp((-1*temperature) / 1756.0) + self.usr_np.exp(-5182.0 / temperature)),
            self.usr_np.log10(0.5880000000000001*self.usr_np.exp((-1*temperature) / 195.0) + 0.412*self.usr_np.exp((-1*temperature) / 5900.0) + self.usr_np.exp(-6394.0 / temperature)),
            self.usr_np.log10(0.4243*self.usr_np.exp((-1*temperature) / 237.00000000000003) + 0.5757*self.usr_np.exp((-1*temperature) / 1652.0) + self.usr_np.exp(-5069.0 / temperature)),
            self.usr_np.log10(0.4093*self.usr_np.exp((-1*temperature) / 275.0) + 0.5907*self.usr_np.exp((-1*temperature) / 1226.0) + self.usr_np.exp(-5185.0 / temperature)),
            self.usr_np.log10(0.3973*self.usr_np.exp((-1*temperature) / 208.0) + 0.6027*self.usr_np.exp((-1*temperature) / 3921.9999999999995) + self.usr_np.exp(-10180.0 / temperature)),
            self.usr_np.log10(0.381*self.usr_np.exp((-1*temperature) / 73.2) + 0.619*self.usr_np.exp((-1*temperature) / 1180.0) + self.usr_np.exp(-9999.0 / temperature)),
            self.usr_np.log10(0.26549999999999996*self.usr_np.exp((-1*temperature) / 180.0) + 0.7345*self.usr_np.exp((-1*temperature) / 1035.0) + self.usr_np.exp(-5417.0 / temperature)),
            1,
            1,
            self.usr_np.log10(0.33299999999999996*self.usr_np.exp((-1*temperature) / 235.0) + 0.667*self.usr_np.exp((-1*temperature) / 2117.0) + self.usr_np.exp(-4536.0 / temperature)),
            self.usr_np.log10(0.42200000000000004*self.usr_np.exp((-1*temperature) / 122.0) + 0.578*self.usr_np.exp((-1*temperature) / 2535.0) + self.usr_np.exp(-9365.0 / temperature)),
            self.usr_np.log10(0.5349999999999999*self.usr_np.exp((-1*temperature) / 201.0) + 0.465*self.usr_np.exp((-1*temperature) / 1772.9999999999998) + self.usr_np.exp(-5333.0 / temperature)),
            self.usr_np.log10(0.8472999999999999*self.usr_np.exp((-1*temperature) / 291.0) + 0.1527*self.usr_np.exp((-1*temperature) / 2742.0) + self.usr_np.exp(-7748.0 / temperature)),
            self.usr_np.log10(0.8106*self.usr_np.exp((-1*temperature) / 277.0) + 0.1894*self.usr_np.exp((-1*temperature) / 8748.0) + self.usr_np.exp(-7891.0 / temperature)),
            self.usr_np.log10(0.685*self.usr_np.exp((-1*temperature) / 369.0) + 0.315*self.usr_np.exp((-1*temperature) / 3284.9999999999995) + self.usr_np.exp(-6667.0 / temperature)),
                        ])

        falloff_function = self._pyro_make_array([
            1,
            10**(falloff_center[1] / (1 + ((self.usr_np.log10(reduced_pressure[1]) + -0.4 + -1*0.67*falloff_center[1]) / (0.75 + -1*1.27*falloff_center[1] + -1*0.14*(self.usr_np.log10(reduced_pressure[1]) + -0.4 + -1*0.67*falloff_center[1])))**2)),
            10**(falloff_center[2] / (1 + ((self.usr_np.log10(reduced_pressure[2]) + -0.4 + -1*0.67*falloff_center[2]) / (0.75 + -1*1.27*falloff_center[2] + -1*0.14*(self.usr_np.log10(reduced_pressure[2]) + -0.4 + -1*0.67*falloff_center[2])))**2)),
            10**(falloff_center[3] / (1 + ((self.usr_np.log10(reduced_pressure[3]) + -0.4 + -1*0.67*falloff_center[3]) / (0.75 + -1*1.27*falloff_center[3] + -1*0.14*(self.usr_np.log10(reduced_pressure[3]) + -0.4 + -1*0.67*falloff_center[3])))**2)),
            10**(falloff_center[4] / (1 + ((self.usr_np.log10(reduced_pressure[4]) + -0.4 + -1*0.67*falloff_center[4]) / (0.75 + -1*1.27*falloff_center[4] + -1*0.14*(self.usr_np.log10(reduced_pressure[4]) + -0.4 + -1*0.67*falloff_center[4])))**2)),
            10**(falloff_center[5] / (1 + ((self.usr_np.log10(reduced_pressure[5]) + -0.4 + -1*0.67*falloff_center[5]) / (0.75 + -1*1.27*falloff_center[5] + -1*0.14*(self.usr_np.log10(reduced_pressure[5]) + -0.4 + -1*0.67*falloff_center[5])))**2)),
            10**(falloff_center[6] / (1 + ((self.usr_np.log10(reduced_pressure[6]) + -0.4 + -1*0.67*falloff_center[6]) / (0.75 + -1*1.27*falloff_center[6] + -1*0.14*(self.usr_np.log10(reduced_pressure[6]) + -0.4 + -1*0.67*falloff_center[6])))**2)),
            10**(falloff_center[7] / (1 + ((self.usr_np.log10(reduced_pressure[7]) + -0.4 + -1*0.67*falloff_center[7]) / (0.75 + -1*1.27*falloff_center[7] + -1*0.14*(self.usr_np.log10(reduced_pressure[7]) + -0.4 + -1*0.67*falloff_center[7])))**2)),
            10**(falloff_center[8] / (1 + ((self.usr_np.log10(reduced_pressure[8]) + -0.4 + -1*0.67*falloff_center[8]) / (0.75 + -1*1.27*falloff_center[8] + -1*0.14*(self.usr_np.log10(reduced_pressure[8]) + -0.4 + -1*0.67*falloff_center[8])))**2)),
            10**(falloff_center[9] / (1 + ((self.usr_np.log10(reduced_pressure[9]) + -0.4 + -1*0.67*falloff_center[9]) / (0.75 + -1*1.27*falloff_center[9] + -1*0.14*(self.usr_np.log10(reduced_pressure[9]) + -0.4 + -1*0.67*falloff_center[9])))**2)),
            10**(falloff_center[10] / (1 + ((self.usr_np.log10(reduced_pressure[10]) + -0.4 + -1*0.67*falloff_center[10]) / (0.75 + -1*1.27*falloff_center[10] + -1*0.14*(self.usr_np.log10(reduced_pressure[10]) + -0.4 + -1*0.67*falloff_center[10])))**2)),
            10**(falloff_center[11] / (1 + ((self.usr_np.log10(reduced_pressure[11]) + -0.4 + -1*0.67*falloff_center[11]) / (0.75 + -1*1.27*falloff_center[11] + -1*0.14*(self.usr_np.log10(reduced_pressure[11]) + -0.4 + -1*0.67*falloff_center[11])))**2)),
            10**(falloff_center[12] / (1 + ((self.usr_np.log10(reduced_pressure[12]) + -0.4 + -1*0.67*falloff_center[12]) / (0.75 + -1*1.27*falloff_center[12] + -1*0.14*(self.usr_np.log10(reduced_pressure[12]) + -0.4 + -1*0.67*falloff_center[12])))**2)),
            10**(falloff_center[13] / (1 + ((self.usr_np.log10(reduced_pressure[13]) + -0.4 + -1*0.67*falloff_center[13]) / (0.75 + -1*1.27*falloff_center[13] + -1*0.14*(self.usr_np.log10(reduced_pressure[13]) + -0.4 + -1*0.67*falloff_center[13])))**2)),
            10**(falloff_center[14] / (1 + ((self.usr_np.log10(reduced_pressure[14]) + -0.4 + -1*0.67*falloff_center[14]) / (0.75 + -1*1.27*falloff_center[14] + -1*0.14*(self.usr_np.log10(reduced_pressure[14]) + -0.4 + -1*0.67*falloff_center[14])))**2)),
            10**(falloff_center[15] / (1 + ((self.usr_np.log10(reduced_pressure[15]) + -0.4 + -1*0.67*falloff_center[15]) / (0.75 + -1*1.27*falloff_center[15] + -1*0.14*(self.usr_np.log10(reduced_pressure[15]) + -0.4 + -1*0.67*falloff_center[15])))**2)),
            10**(falloff_center[16] / (1 + ((self.usr_np.log10(reduced_pressure[16]) + -0.4 + -1*0.67*falloff_center[16]) / (0.75 + -1*1.27*falloff_center[16] + -1*0.14*(self.usr_np.log10(reduced_pressure[16]) + -0.4 + -1*0.67*falloff_center[16])))**2)),
            10**(falloff_center[17] / (1 + ((self.usr_np.log10(reduced_pressure[17]) + -0.4 + -1*0.67*falloff_center[17]) / (0.75 + -1*1.27*falloff_center[17] + -1*0.14*(self.usr_np.log10(reduced_pressure[17]) + -0.4 + -1*0.67*falloff_center[17])))**2)),
            10**(falloff_center[18] / (1 + ((self.usr_np.log10(reduced_pressure[18]) + -0.4 + -1*0.67*falloff_center[18]) / (0.75 + -1*1.27*falloff_center[18] + -1*0.14*(self.usr_np.log10(reduced_pressure[18]) + -0.4 + -1*0.67*falloff_center[18])))**2)),
            10**(falloff_center[19] / (1 + ((self.usr_np.log10(reduced_pressure[19]) + -0.4 + -1*0.67*falloff_center[19]) / (0.75 + -1*1.27*falloff_center[19] + -1*0.14*(self.usr_np.log10(reduced_pressure[19]) + -0.4 + -1*0.67*falloff_center[19])))**2)),
            10**(falloff_center[20] / (1 + ((self.usr_np.log10(reduced_pressure[20]) + -0.4 + -1*0.67*falloff_center[20]) / (0.75 + -1*1.27*falloff_center[20] + -1*0.14*(self.usr_np.log10(reduced_pressure[20]) + -0.4 + -1*0.67*falloff_center[20])))**2)),
            1,
            1,
            10**(falloff_center[23] / (1 + ((self.usr_np.log10(reduced_pressure[23]) + -0.4 + -1*0.67*falloff_center[23]) / (0.75 + -1*1.27*falloff_center[23] + -1*0.14*(self.usr_np.log10(reduced_pressure[23]) + -0.4 + -1*0.67*falloff_center[23])))**2)),
            10**(falloff_center[24] / (1 + ((self.usr_np.log10(reduced_pressure[24]) + -0.4 + -1*0.67*falloff_center[24]) / (0.75 + -1*1.27*falloff_center[24] + -1*0.14*(self.usr_np.log10(reduced_pressure[24]) + -0.4 + -1*0.67*falloff_center[24])))**2)),
            10**(falloff_center[25] / (1 + ((self.usr_np.log10(reduced_pressure[25]) + -0.4 + -1*0.67*falloff_center[25]) / (0.75 + -1*1.27*falloff_center[25] + -1*0.14*(self.usr_np.log10(reduced_pressure[25]) + -0.4 + -1*0.67*falloff_center[25])))**2)),
            10**(falloff_center[26] / (1 + ((self.usr_np.log10(reduced_pressure[26]) + -0.4 + -1*0.67*falloff_center[26]) / (0.75 + -1*1.27*falloff_center[26] + -1*0.14*(self.usr_np.log10(reduced_pressure[26]) + -0.4 + -1*0.67*falloff_center[26])))**2)),
            10**(falloff_center[27] / (1 + ((self.usr_np.log10(reduced_pressure[27]) + -0.4 + -1*0.67*falloff_center[27]) / (0.75 + -1*1.27*falloff_center[27] + -1*0.14*(self.usr_np.log10(reduced_pressure[27]) + -0.4 + -1*0.67*falloff_center[27])))**2)),
            10**(falloff_center[28] / (1 + ((self.usr_np.log10(reduced_pressure[28]) + -0.4 + -1*0.67*falloff_center[28]) / (0.75 + -1*1.27*falloff_center[28] + -1*0.14*(self.usr_np.log10(reduced_pressure[28]) + -0.4 + -1*0.67*falloff_center[28])))**2)),
                            ])*reduced_pressure/(1+reduced_pressure)

        k_fwd[11] = k_high[0]*falloff_function[0]*ones
        k_fwd[49] = k_high[1]*falloff_function[1]*ones
        k_fwd[51] = k_high[2]*falloff_function[2]*ones
        k_fwd[53] = k_high[3]*falloff_function[3]*ones
        k_fwd[55] = k_high[4]*falloff_function[4]*ones
        k_fwd[56] = k_high[5]*falloff_function[5]*ones
        k_fwd[58] = k_high[6]*falloff_function[6]*ones
        k_fwd[62] = k_high[7]*falloff_function[7]*ones
        k_fwd[69] = k_high[8]*falloff_function[8]*ones
        k_fwd[70] = k_high[9]*falloff_function[9]*ones
        k_fwd[71] = k_high[10]*falloff_function[10]*ones
        k_fwd[73] = k_high[11]*falloff_function[11]*ones
        k_fwd[75] = k_high[12]*falloff_function[12]*ones
        k_fwd[82] = k_high[13]*falloff_function[13]*ones
        k_fwd[84] = k_high[14]*falloff_function[14]*ones
        k_fwd[94] = k_high[15]*falloff_function[15]*ones
        k_fwd[130] = k_high[16]*falloff_function[16]*ones
        k_fwd[139] = k_high[17]*falloff_function[17]*ones
        k_fwd[146] = k_high[18]*falloff_function[18]*ones
        k_fwd[157] = k_high[19]*falloff_function[19]*ones
        k_fwd[173] = k_high[20]*falloff_function[20]*ones
        k_fwd[184] = k_high[21]*falloff_function[21]*ones
        k_fwd[236] = k_high[22]*falloff_function[22]*ones
        k_fwd[240] = k_high[23]*falloff_function[23]*ones
        k_fwd[288] = k_high[24]*falloff_function[24]*ones
        k_fwd[303] = k_high[25]*falloff_function[25]*ones
        k_fwd[311] = k_high[26]*falloff_function[26]*ones
        k_fwd[317] = k_high[27]*falloff_function[27]*ones
        k_fwd[319] = k_high[28]*falloff_function[28]*ones
        return

    def get_fwd_rate_coefficients(self, temperature, concentrations):
        ones = self._pyro_zeros_like(temperature) + 1.0
        k_fwd = [
            120000000000.00002*temperature**-1.0 * ones,
            500000000000.0001*temperature**-1.0 * ones,
            self.usr_np.exp(3.655839600035736 + 2.7*self.usr_np.log(temperature) + -1*(3150.1542797022735 / temperature)) * ones,
            20000000000.000004 * ones,
            self.usr_np.exp(9.172638504792172 + 2.0*self.usr_np.log(temperature) + -1*(2012.8781339950629 / temperature)) * ones,
            57000000000.00001 * ones,
            80000000000.00002 * ones,
            15000000000.000002 * ones,
            15000000000.000002 * ones,
            50600000000.00001 * ones,
            self.usr_np.exp(13.835313185260453 + 1.5*self.usr_np.log(temperature) + -1*(4327.687988089386 / temperature)) * ones,
            0*temperature,
            30000000000.000004 * ones,
            30000000000.000004 * ones,
            self.usr_np.exp(24.386827483076058 + -1*(1781.3971485856307 / temperature)) * ones,
            10000000000.000002 * ones,
            10000000000.000002 * ones,
            self.usr_np.exp(5.961005339623274 + 2.5*self.usr_np.log(temperature) + -1*(1559.9805538461737 / temperature)) * ones,
            self.usr_np.exp(4.867534450455582 + 2.5*self.usr_np.log(temperature) + -1*(2516.097667493829 / temperature)) * ones,
            50000000000.00001 * ones,
            self.usr_np.exp(9.51044496442652 + 2.0*self.usr_np.log(temperature) + -1*(956.117113647655 / temperature)) * ones,
            self.usr_np.exp(38.36741779139978 + -1.41*self.usr_np.log(temperature) + -1*(14568.205494789268 / temperature)) * ones,
            self.usr_np.exp(8.84505705350085 + 2.0*self.usr_np.log(temperature) + -1*(956.117113647655 / temperature)) * ones,
            30000000000.000004 * ones,
            self.usr_np.exp(9.433483923290392 + 1.83*self.usr_np.log(temperature) + -1*(110.70829736972846 / temperature)) * ones,
            22400000000.000004 * ones,
            self.usr_np.exp(11.40534025429029 + 1.92*self.usr_np.log(temperature) + -1*(2863.319145607977 / temperature)) * ones,
            100000000000.00002 * ones,
            self.usr_np.exp(23.025850929940457 + -1*(4025.7562679901257 / temperature)) * ones,
            self.usr_np.exp(21.282881624881835 + -1*(679.3463702233338 / temperature)) * ones,
            self.usr_np.exp(21.639556568820566 + -1*(24053.893701241002 / temperature)) * ones,
            self.usr_np.exp(25.328436022934504 + -1*(20128.78133995063 / temperature)) * ones,
            2800000000000.0005*temperature**-0.86 * ones,
            20800000000000.004*temperature**-1.24 * ones,
            11260000000000.002*temperature**-0.76 * ones,
            26000000000000.004*temperature**-1.24 * ones,
            700000000000.0001*temperature**-0.8 * ones,
            self.usr_np.exp(30.908165848920724 + -0.6707*self.usr_np.log(temperature) + -1*(8575.364070352467 / temperature)) * ones,
            1000000000000.0002*temperature**-1.0 * ones,
            90000000000.00002*temperature**-0.6 * ones,
            60000000000000.01*temperature**-1.25 * ones,
            550000000000000.1*temperature**-2.0 * ones,
            2.2000000000000004e+16*temperature**-2.0 * ones,
            self.usr_np.exp(22.10203193164551 + -1*(337.66030697767184 / temperature)) * ones,
            self.usr_np.exp(24.52547397636735 + -1*(537.4384617766818 / temperature)) * ones,
            self.usr_np.exp(25.154082635789724 + -1*(319.54440377171625 / temperature)) * ones,
            self.usr_np.exp(9.400960731584833 + 2.0*self.usr_np.log(temperature) + -1*(2616.741574193582 / temperature)) * ones,
            self.usr_np.exp(23.025850929940457 + -1*(1811.5903205955567 / temperature)) * ones,
            165000000000.00003 * ones,
            0*temperature,
            30000000000.000004 * ones,
            0*temperature,
            self.usr_np.exp(13.399995114002609 + 1.62*self.usr_np.log(temperature) + -1*(5454.899743126621 / temperature)) * ones,
            0*temperature,
            73400000000.00002 * ones,
            0*temperature,
            0*temperature,
            self.usr_np.exp(10.957799582307658 + 1.9*self.usr_np.log(temperature) + -1*(1379.8279608536157 / temperature)) * ones,
            0*temperature,
            20000000000.000004 * ones,
            self.usr_np.exp(18.921456031864857 + 0.65*self.usr_np.log(temperature) + -1*(-142.91434751364946 / temperature)) * ones,
            self.usr_np.exp(24.21369435233651 + -0.09*self.usr_np.log(temperature) + -1*(306.9639154342471 / temperature)) * ones,
            0*temperature,
            self.usr_np.exp(10.63344870621879 + 1.63*self.usr_np.log(temperature) + -1*(968.1943824516253 / temperature)) * ones,
            20000000000.000004 * ones,
            self.usr_np.exp(21.128730945054574 + 0.5*self.usr_np.log(temperature) + -1*(-55.35414868486423 / temperature)) * ones,
            self.usr_np.exp(26.291610340707507 + -0.23*self.usr_np.log(temperature) + -1*(538.4449008436793 / temperature)) * ones,
            self.usr_np.exp(9.740968623038354 + 2.1*self.usr_np.log(temperature) + -1*(2450.679128138989 / temperature)) * ones,
            self.usr_np.exp(8.34283980427146 + 2.1*self.usr_np.log(temperature) + -1*(2450.679128138989 / temperature)) * ones,
            0*temperature,
            0*temperature,
            0*temperature,
            30000000000.000004 * ones,
            0*temperature,
            self.usr_np.exp(7.1891677384203225 + 2.53*self.usr_np.log(temperature) + -1*(6159.407090024893 / temperature)) * ones,
            0*temperature,
            2000000000.0000002 * ones,
            self.usr_np.exp(11.652687407345388 + 1.9*self.usr_np.log(temperature) + -1*(3789.243087245706 / temperature)) * ones,
            100000000000.00002 * ones,
            self.usr_np.exp(24.635288842374557 + -1*(4025.7562679901257 / temperature)) * ones,
            self.usr_np.exp(23.148068562664704 + -1*(1725.036560833769 / temperature)) * ones,
            10000000000.000002 * ones,
            0*temperature,
            self.usr_np.exp(12.283033686666302 + 1.51*self.usr_np.log(temperature) + -1*(1726.0429999007665 / temperature)) * ones,
            0*temperature,
            self.usr_np.exp(3.5751506887855933 + 2.4*self.usr_np.log(temperature) + -1*(-1061.7932156823956 / temperature)) * ones,
            self.usr_np.exp(23.39741448637294 + -1*(-251.60976674938286 / temperature)) * ones,
            self.usr_np.exp(21.416413017506358 + -1*(214.87474080397297 / temperature)) * ones,
            self.usr_np.exp(35.06940464597285 + -1*(14799.686480198701 / temperature)) * ones,
            50000000000.00001 * ones,
            30000000000.000004 * ones,
            20000000000.000004 * ones,
            self.usr_np.exp(9.332558004700433 + 2.0*self.usr_np.log(temperature) + -1*(1509.6586004962971 / temperature)) * ones,
            30000000000.000004 * ones,
            0*temperature,
            self.usr_np.exp(10.933106969717286 + 1.6*self.usr_np.log(temperature) + -1*(2727.4498715633104 / temperature)) * ones,
            self.usr_np.exp(34.0987198420329 + -1.34*self.usr_np.log(temperature) + -1*(713.0620789677511 / temperature)) * ones,
            self.usr_np.exp(11.512925464970229 + 1.6*self.usr_np.log(temperature) + -1*(1570.0449445161491 / temperature)) * ones,
            self.usr_np.exp(10.770588040219511 + 1.228*self.usr_np.log(temperature) + -1*(35.2253673449136 / temperature)) * ones,
            50000000000.00001 * ones,
            self.usr_np.exp(15.048070819142122 + 1.18*self.usr_np.log(temperature) + -1*(-224.9391314739483 / temperature)) * ones,
            5000000000.000001 * ones,
            5000000000.000001 * ones,
            self.usr_np.exp(7.272398392570047 + 2.0*self.usr_np.log(temperature) + -1*(-422.70440813896323 / temperature)) * ones,
            self.usr_np.exp(8.748304912379623 + 2.0*self.usr_np.log(temperature) + -1*(754.8293002481486 / temperature)) * ones,
            20000000000.000004 * ones,
            self.usr_np.exp(-15.338770774157322 + 4.5*self.usr_np.log(temperature) + -1*(-503.2195334987657 / temperature)) * ones,
            self.usr_np.exp(6.222576268071369 + 2.3*self.usr_np.log(temperature) + -1*(6793.463702233337 / temperature)) * ones,
            self.usr_np.exp(10.425253116340453 + 2.0*self.usr_np.log(temperature) + -1*(7045.07346898272 / temperature)) * ones,
            self.usr_np.exp(-14.543249183293838 + 4.0*self.usr_np.log(temperature) + -1*(-1006.4390669975314 / temperature)) * ones,
            5000000000.000001 * ones,
            self.usr_np.exp(8.1886891244442 + 2.0*self.usr_np.log(temperature) + -1*(1258.0488337469144 / temperature)) * ones,
            self.usr_np.exp(8.17188200612782 + 2.12*self.usr_np.log(temperature) + -1*(437.8009941439262 / temperature)) * ones,
            self.usr_np.exp(22.738168857488677 + -1*(1006.4390669975314 / temperature)) * ones,
            self.usr_np.exp(18.683045008419857 + -1*(-820.2478396029882 / temperature)) * ones,
            self.usr_np.exp(26.763520548223827 + -1*(6038.634401985189 / temperature)) * ones,
            20000000000.000004 * ones,
            1000000000.0000001 * ones,
            37800000000.00001 * ones,
            self.usr_np.exp(25.733901131042668 + -1*(11875.980990570872 / temperature)) * ones,
            self.usr_np.exp(8.630521876723241 + 2.0*self.usr_np.log(temperature) + -1*(6038.634401985189 / temperature)) * ones,
            self.usr_np.exp(24.783708847492832 + -1*(289.85445129528904 / temperature)) * ones,
            50000000000.00001 * ones,
            50000000000.00001 * ones,
            67100000000.00001 * ones,
            self.usr_np.exp(25.40539706407063 + -1*(1565.0127491811616 / temperature)) * ones,
            self.usr_np.exp(22.465484860614332 + -1*(-379.93074779156814 / temperature)) * ones,
            40000000000.00001 * ones,
            30000000000.000004 * ones,
            60000000000.00001 * ones,
            0*temperature,
            self.usr_np.exp(25.970289909106896 + -1*(7946.842873012509 / temperature)) * ones,
            self.usr_np.exp(25.272923313004245 + -1*(-259.15805975186436 / temperature)) * ones,
            50000000000.00001 * ones,
            self.usr_np.exp(22.33270374938051 + -1*(754.8293002481486 / temperature)) * ones,
            self.usr_np.exp(6.214608098422192 + 2.0*self.usr_np.log(temperature) + -1*(3638.277227196076 / temperature)) * ones,
            self.usr_np.exp(28.101024745174286 + -1*(6010.454108109258 / temperature)) * ones,
            40000000000.00001 * ones,
            self.usr_np.exp(7.807916628926408 + 2.0*self.usr_np.log(temperature) + -1*(4161.6255420347925 / temperature)) * ones,
            0*temperature,
            30000000000.000004 * ones,
            self.usr_np.exp(23.43131603804862 + -1*(301.93172009925945 / temperature)) * ones,
            self.usr_np.exp(22.920490414282632 + -1*(301.93172009925945 / temperature)) * ones,
            28000000000.000004 * ones,
            12000000000.000002 * ones,
            70000000000.00002 * ones,
            0*temperature,
            30000000000.000004 * ones,
            self.usr_np.exp(23.208172486734412 + -1*(-286.83513409429645 / temperature)) * ones,
            self.usr_np.exp(23.495854559186192 + -1*(-286.83513409429645 / temperature)) * ones,
            9000000000.000002 * ones,
            7000000000.000001 * ones,
            14000000000.000002 * ones,
            self.usr_np.exp(24.412145291060348 + -1*(-276.77074342432115 / temperature)) * ones,
            self.usr_np.exp(24.295611474804396 + -1*(15338.13138104238 / temperature)) * ones,
            self.usr_np.exp(21.560513361480112 + -1*(10222.904823027426 / temperature)) * ones,
            self.usr_np.exp(3.1986731175506815 + 2.47*self.usr_np.log(temperature) + -1*(2606.6771835236063 / temperature)) * ones,
            0*temperature,
            self.usr_np.exp(22.64605356858087 + 0.1*self.usr_np.log(temperature) + -1*(5334.127055086917 / temperature)) * ones,
            26480000000.000004 * ones,
            self.usr_np.exp(1.1999647829283973 + 2.81*self.usr_np.log(temperature) + -1*(2948.866466302767 / temperature)) * ones,
            self.usr_np.exp(10.308952660644293 + 1.5*self.usr_np.log(temperature) + -1*(5002.002162977731 / temperature)) * ones,
            self.usr_np.exp(9.210340371976184 + 1.5*self.usr_np.log(temperature) + -1*(5002.002162977731 / temperature)) * ones,
            self.usr_np.exp(5.424950017481403 + 2.0*self.usr_np.log(temperature) + -1*(4629.619708188645 / temperature)) * ones,
            self.usr_np.exp(8.722580021141189 + 1.74*self.usr_np.log(temperature) + -1*(5258.644125062102 / temperature)) * ones,
            self.usr_np.exp(34.94424150301885 + -1.0*self.usr_np.log(temperature) + -1*(8554.732069479018 / temperature)) * ones,
            self.usr_np.exp(32.86212973278314 + -1.0*self.usr_np.log(temperature) + -1*(8554.732069479018 / temperature)) * ones,
            self.usr_np.exp(23.32224494299426 + -1*(201.2878133995063 / temperature)) * ones,
            self.usr_np.exp(23.613637594842576 + -1*(452.8975801488892 / temperature)) * ones,
            self.usr_np.exp(-35.38740847831102 + 7.6*self.usr_np.log(temperature) + -1*(-1776.364953250643 / temperature)) * ones,
            self.usr_np.exp(23.025850929940457 + -1*(-379.93074779156814 / temperature)) * ones,
            self.usr_np.exp(17.85504688369138 + 0.9*self.usr_np.log(temperature) + -1*(1002.9165302630402 / temperature)) * ones,
            self.usr_np.exp(31.45530520704869 + -1.39*self.usr_np.log(temperature) + -1*(510.7678265012472 / temperature)) * ones,
            0*temperature,
            self.usr_np.exp(20.548912449801634 + -1*(1949.9756923077173 / temperature)) * ones,
            self.usr_np.exp(21.886416646752092 + -1*(429.74948160794594 / temperature)) * ones,
            10000000000.000002 * ones,
            self.usr_np.exp(24.01910270295074 + -1*(178.64293439206185 / temperature)) * ones,
            self.usr_np.exp(16.012735135300495 + self.usr_np.log(temperature) + -1*(3270.9269677419775 / temperature)) * ones,
            self.usr_np.exp(24.23779190391557 + -1*(193.73952039702482 / temperature)) * ones,
            self.usr_np.exp(21.059738073567623 + -1*(5439.803157121658 / temperature)) * ones,
            self.usr_np.exp(24.090561666932885 + -1*(11649.532200496427 / temperature)) * ones,
            self.usr_np.exp(26.681690529976194 + -1*(9500.784792456698 / temperature)) * ones,
            self.usr_np.exp(21.416413017506358 + -1*(10597.803375484007 / temperature)) * ones,
            0*temperature,
            self.usr_np.exp(21.469953784434388 + -1*(-241.54537607940756 / temperature)) * ones,
            106000000000000.02*temperature**-1.41 * ones,
            self.usr_np.exp(22.08424239008201 + -1*(-120.77268803970378 / temperature)) * ones,
            self.usr_np.exp(25.60606775953278 + -1*(181.15903205955567 / temperature)) * ones,
            40000000000.00001 * ones,
            self.usr_np.exp(24.18900173974614 + -1*(166.0624460545927 / temperature)) * ones,
            20000000000.000004 * ones,
            2000000.0000000002*temperature**1.2 * ones,
            self.usr_np.exp(6.133398042996649 + 2.0*self.usr_np.log(temperature) + -1*(3270.9269677419775 / temperature)) * ones,
            self.usr_np.exp(7.154615356913663 + 1.5*self.usr_np.log(temperature) + -1*(50.32195334987657 / temperature)) * ones,
            15000000000.000002 * ones,
            self.usr_np.exp(23.7189981105004 + -1*(6969.590538957906 / temperature)) * ones,
            21600000000.000004*temperature**-0.23 * ones,
            365000000000.00006*temperature**-0.45 * ones,
            3000000000.0000005 * ones,
            39000000000.00001 * ones,
            self.usr_np.exp(24.412145291060348 + -1*(1836.751297270495 / temperature)) * ones,
            self.usr_np.exp(11.407564949312402 + 1.5*self.usr_np.log(temperature) + -1*(-231.48098540943224 / temperature)) * ones,
            330000000.0 * ones,
            self.usr_np.exp(25.590800287401994 + -0.11*self.usr_np.log(temperature) + -1*(2506.033276823853 / temperature)) * ones,
            5000000000.000001 * ones,
            25000000000.000004 * ones,
            70000000000.00002 * ones,
            50000000000.00001 * ones,
            20000000000.000004 * ones,
            25000000000.000004 * ones,
            self.usr_np.exp(31.433229255349488 + -1.32*self.usr_np.log(temperature) + -1*(372.38245478908664 / temperature)) * ones,
            25000000000.000004 * ones,
            self.usr_np.exp(20.617905321288585 + 0.72*self.usr_np.log(temperature) + -1*(332.1248921091854 / temperature)) * ones,
            self.usr_np.exp(9.472704636443673 + 1.9*self.usr_np.log(temperature) + -1*(-478.0585568238275 / temperature)) * ones,
            self.usr_np.exp(23.025850929940457 + -1*(6541.853935483955 / temperature)) * ones,
            77000000000.00002 * ones,
            40000000000.00001 * ones,
            self.usr_np.exp(22.80270737862625 + -1*(3754.0177199007926 / temperature)) * ones,
            self.usr_np.exp(22.53809057910546 + -1*(-221.4165947394569 / temperature)) * ones,
            self.usr_np.exp(5.68697535633982 + 2.45*self.usr_np.log(temperature) + -1*(1127.2117550372352 / temperature)) * ones,
            23500000000.000004 * ones,
            54000000000.00001 * ones,
            2500000000.0000005 * ones,
            20000000000.000004 * ones,
            self.usr_np.exp(21.416413017506358 + -1*(10064.390669975315 / temperature)) * ones,
            self.usr_np.exp(26.459838134425603 + -1*(27199.015785608288 / temperature)) * ones,
            self.usr_np.exp(32.87804518808903 + -1.52*self.usr_np.log(temperature) + -1*(372.38245478908664 / temperature)) * ones,
            self.usr_np.exp(35.87377746164302 + -2.0*self.usr_np.log(temperature) + -1*(402.5756267990126 / temperature)) * ones,
            self.usr_np.exp(59.90643313099847 + -3.3*self.usr_np.log(temperature) + -1*(63707.592940943745 / temperature)) * ones,
            self.usr_np.exp(3.0106208860477417 + 2.64*self.usr_np.log(temperature) + -1*(2506.033276823853 / temperature)) * ones,
            self.usr_np.exp(1.623340817603092 + 2.64*self.usr_np.log(temperature) + -1*(2506.033276823853 / temperature)) * ones,
            self.usr_np.exp(15.179047931961549 + 1.58*self.usr_np.log(temperature) + -1*(13385.639591067169 / temperature)) * ones,
            self.usr_np.exp(7.003065458786462 + 2.03*self.usr_np.log(temperature) + -1*(6728.045162878498 / temperature)) * ones,
            self.usr_np.exp(1.4816045409242158 + 2.26*self.usr_np.log(temperature) + -1*(3220.6050143921007 / temperature)) * ones,
            self.usr_np.exp(-1.83258146374831 + 2.56*self.usr_np.log(temperature) + -1*(4528.975801488891 / temperature)) * ones,
            0*temperature,
            self.usr_np.exp(24.81761039916851 + -1*(201.2878133995063 / temperature)) * ones,
            self.usr_np.exp(24.866400563337944 + -1*(23158.1629316132 / temperature)) * ones,
            self.usr_np.exp(14.953343559785665 + 0.88*self.usr_np.log(temperature) + -1*(10129.809209330155 / temperature)) * ones,
            0*temperature,
            self.usr_np.exp(23.025850929940457 + -1*(37238.24547890866 / temperature)) * ones,
            self.usr_np.exp(18.420680743952367 + -1*(32709.269677419772 / temperature)) * ones,
            19000000000.000004 * ones,
            29000000000.000004 * ones,
            41000000000.00001 * ones,
            16200000000.000002 * ones,
            24600000000.000004 * ones,
            self.usr_np.exp(33.36759341340774 + -1.38*self.usr_np.log(temperature) + -1*(639.0888075434325 / temperature)) * ones,
            self.usr_np.exp(26.393146759926932 + -0.69*self.usr_np.log(temperature) + -1*(382.44684545906193 / temperature)) * ones,
            self.usr_np.exp(24.360851996672796 + -0.36*self.usr_np.log(temperature) + -1*(291.86732942928415 / temperature)) * ones,
            self.usr_np.exp(33.36759341340774 + -1.38*self.usr_np.log(temperature) + -1*(639.0888075434325 / temperature)) * ones,
            self.usr_np.exp(26.393146759926932 + -0.69*self.usr_np.log(temperature) + -1*(382.44684545906193 / temperature)) * ones,
            self.usr_np.exp(24.360851996672796 + -0.36*self.usr_np.log(temperature) + -1*(291.86732942928415 / temperature)) * ones,
            self.usr_np.exp(25.287614028414247 + -1*(14492.722564764454 / temperature)) * ones,
            self.usr_np.exp(20.72326583694641 + -1*(10945.024853598155 / temperature)) * ones,
            22000000000.000004 * ones,
            2000000000.0000002 * ones,
            12000000000.000002 * ones,
            12000000000.000002 * ones,
            100000000000.00002 * ones,
            self.usr_np.exp(11.49272275765271 + 1.41*self.usr_np.log(temperature) + -1*(4277.366034739509 / temperature)) * ones,
            self.usr_np.exp(11.918390573078392 + 1.57*self.usr_np.log(temperature) + -1*(22141.659473945692 / temperature)) * ones,
            self.usr_np.exp(7.696212639346407 + 2.11*self.usr_np.log(temperature) + -1*(5736.702681885929 / temperature)) * ones,
            self.usr_np.exp(10.021270588192511 + 1.7*self.usr_np.log(temperature) + -1*(1912.23422729531 / temperature)) * ones,
            self.usr_np.exp(4.653960350157524 + 2.5*self.usr_np.log(temperature) + -1*(6692.8197955335845 / temperature)) * ones,
            self.usr_np.exp(10.404262840448618 + 1.5*self.usr_np.log(temperature) + -1*(1811.5903205955567 / temperature)) * ones,
            self.usr_np.exp(8.101677747454572 + 1.5*self.usr_np.log(temperature) + -1*(1811.5903205955567 / temperature)) * ones,
            self.usr_np.exp(30.099120647400166 + -1*(42632.75887801543 / temperature)) * ones,
            self.usr_np.exp(28.372958460657927 + -0.69*self.usr_np.log(temperature) + -1*(1434.1756704714824 / temperature)) * ones,
            self.usr_np.exp(19.41393251696265 + 0.18*self.usr_np.log(temperature) + -1*(1066.8254110173834 / temperature)) * ones,
            self.usr_np.exp(25.859064273996673 + -0.75*self.usr_np.log(temperature) + -1*(1454.304451811433 / temperature)) * ones,
            self.usr_np.exp(9.903487552536129 + 2.0*self.usr_np.log(temperature) + -1*(1006.4390669975314 / temperature)) * ones,
            9000000000.000002 * ones,
            self.usr_np.exp(27.136724794113768 + -0.31*self.usr_np.log(temperature) + -1*(145.93366471464208 / temperature)) * ones,
            self.usr_np.exp(22.03159865659659 + 0.15*self.usr_np.log(temperature) + -1*(-45.28975801488892 / temperature)) * ones,
            self.usr_np.exp(6.29156913955832 + 2.4*self.usr_np.log(temperature) + -1*(4989.421674640263 / temperature)) * ones,
            self.usr_np.exp(10.819778284410283 + 1.6*self.usr_np.log(temperature) + -1*(480.57465449132127 / temperature)) * ones,
            self.usr_np.exp(9.148464968258095 + 1.94*self.usr_np.log(temperature) + -1*(3250.7981864020267 / temperature)) * ones,
            self.usr_np.exp(23.025850929940457 + -1*(7221.200305707288 / temperature)) * ones,
            self.usr_np.exp(29.449097893473976 + -0.752*self.usr_np.log(temperature) + -1*(173.61073905707417 / temperature)) * ones,
            self.usr_np.exp(21.901920833288056 + -1*(-354.76977111662984 / temperature)) * ones,
            self.usr_np.exp(21.82187812561452 + -1*(5686.380728536053 / temperature)) * ones,
            33700000000.000008 * ones,
            self.usr_np.exp(8.809862805379058 + 1.83*self.usr_np.log(temperature) + -1*(110.70829736972846 / temperature)) * ones,
            109600000000.00002 * ones,
            self.usr_np.exp(29.24045902836265 + -1*(8720.794515533611 / temperature)) * ones,
            self.usr_np.exp(15.89495209964411 + 0.5*self.usr_np.log(temperature) + -1*(-883.1502812903339 / temperature)) * ones,
            0*temperature,
            self.usr_np.exp(22.481123754498785 + -1*(754.8293002481486 / temperature)) * ones,
            self.usr_np.exp(21.598734574300313 + -1*(754.8293002481486 / temperature)) * ones,
            self.usr_np.exp(26.021583203494448 + -1*(5529.879453617937 / temperature)) * ones,
            self.usr_np.exp(18.037955122813692 + 0.25*self.usr_np.log(temperature) + -1*(-470.510263821346 / temperature)) * ones,
            self.usr_np.exp(19.529243363473643 + 0.29*self.usr_np.log(temperature) + -1*(5.5354148684864235 / temperature)) * ones,
            self.usr_np.exp(7.198183577101943 + 1.61*self.usr_np.log(temperature) + -1*(-193.23630086352605 / temperature)) * ones,
            self.usr_np.exp(21.7948494532266 + -1*(909.8209165657685 / temperature)) * ones,
            self.usr_np.exp(21.7948494532266 + -1*(909.8209165657685 / temperature)) * ones,
            self.usr_np.exp(24.12779100870124 + -1*(19701.044736476677 / temperature)) * ones,
            self.usr_np.exp(14.53335035111459 + 1.16*self.usr_np.log(temperature) + -1*(1210.2429780645316 / temperature)) * ones,
            self.usr_np.exp(14.53335035111459 + 1.16*self.usr_np.log(temperature) + -1*(1210.2429780645316 / temperature)) * ones,
            self.usr_np.exp(16.969527810483978 + 0.73*self.usr_np.log(temperature) + -1*(-560.0833407841262 / temperature)) * ones,
            self.usr_np.exp(21.825205915707194 + -1*(5999.8864979057835 / temperature)) * ones,
            self.usr_np.exp(7.908387159290043 + 1.77*self.usr_np.log(temperature) + -1*(2979.059638312693 / temperature)) * ones,
            0*temperature,
            150000000000.00003 * ones,
            18100000.000000004 * ones,
            23500000.000000004 * ones,
            22000000000.000004 * ones,
            11000000000.000002 * ones,
            12000000000.000002 * ones,
            30100000000.000004 * ones,
            0*temperature,
            self.usr_np.exp(5.262690188904886 + 2.68*self.usr_np.log(temperature) + -1*(1869.9637864814135 / temperature)) * ones,
            self.usr_np.exp(7.1853870155804165 + 2.54*self.usr_np.log(temperature) + -1*(3399.7511683176613 / temperature)) * ones,
            self.usr_np.exp(10.360912399575003 + 1.8*self.usr_np.log(temperature) + -1*(470.0070442878472 / temperature)) * ones,
            self.usr_np.exp(-0.9728610833625492 + 2.72*self.usr_np.log(temperature) + -1*(754.8293002481486 / temperature)) * ones,
            self.usr_np.exp(-7.009788004547288 + 3.65*self.usr_np.log(temperature) + -1*(3600.03254265017 / temperature)) * ones,
            0*temperature,
            96400000000.00002 * ones,
            0*temperature,
            self.usr_np.exp(8.308938252595778 + 2.19*self.usr_np.log(temperature) + -1*(447.8653848139015 / temperature)) * ones,
            24100000000.000004 * ones,
            self.usr_np.exp(17.054189010128656 + 0.255*self.usr_np.log(temperature) + -1*(-474.5360200893361 / temperature)) * ones,
            24100000000.000004 * ones,
            19270000000.000004*temperature**-0.32 * ones,
                ]
        self.get_falloff_rates(temperature, concentrations, k_fwd)

        k_fwd[0] *= (0.83*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.75*concentrations[14] + 3.6*concentrations[15] + 2.4*concentrations[0] + 15.4*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])
        k_fwd[1] *= (0.7*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])
        k_fwd[32] *= (1.5*concentrations[26] + 0.75*concentrations[14] + 1.5*concentrations[15] + concentrations[0] + concentrations[1] + concentrations[2] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[13] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])
        k_fwd[33] *= (concentrations[3])
        k_fwd[34] *= (concentrations[5])
        k_fwd[35] *= (concentrations[47])
        k_fwd[36] *= (concentrations[48])
        k_fwd[38] *= (0.63*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[14] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])
        k_fwd[39] *= (concentrations[0])
        k_fwd[40] *= (concentrations[5])
        k_fwd[41] *= (concentrations[15])
        k_fwd[42] *= (0.38*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 0.73*concentrations[0] + 3.65*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[14] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])
        k_fwd[165] *= (concentrations[5])
        k_fwd[166] *= (3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[48] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])
        k_fwd[186] *= (0.7*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])
        k_fwd[204] *= (0.7*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])
        k_fwd[211] *= (0.7*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])
        k_fwd[226] *= (0.7*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])
        k_fwd[229] *= (0.7*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])
        k_fwd[268] *= (0.7*concentrations[48] + 3.0*concentrations[26] + 2.0*concentrations[13] + 1.5*concentrations[14] + 2.0*concentrations[15] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31] + concentrations[32] + concentrations[33] + concentrations[34] + concentrations[35] + concentrations[36] + concentrations[37] + concentrations[38] + concentrations[39] + concentrations[40] + concentrations[41] + concentrations[42] + concentrations[43] + concentrations[44] + concentrations[45] + concentrations[46] + concentrations[47] + concentrations[49] + concentrations[50] + concentrations[51] + concentrations[52])
        k_fwd[302] *= (concentrations[12])
        return self._pyro_make_array(k_fwd)

    def get_net_rates_of_progress(self, temperature, concentrations):
        k_fwd = self.get_fwd_rate_coefficients(temperature, concentrations)
        log_k_eq = self.get_equilibrium_constants(temperature)
        return self._pyro_make_array([
                    k_fwd[0]*(concentrations[2]**2.0 + -1*self.usr_np.exp(log_k_eq[0])*concentrations[3]),
                    k_fwd[1]*(concentrations[1]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[1])*concentrations[4]),
                    k_fwd[2]*(concentrations[0]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[2])*concentrations[1]*concentrations[4]),
                    k_fwd[3]*(concentrations[6]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[3])*concentrations[3]*concentrations[4]),
                    k_fwd[4]*(concentrations[7]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[4])*concentrations[6]*concentrations[4]),
                    k_fwd[5]*(concentrations[9]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[5])*concentrations[14]*concentrations[1]),
                    k_fwd[6]*(concentrations[10]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[6])*concentrations[1]*concentrations[16]),
                    k_fwd[7]*(concentrations[11]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[7])*concentrations[14]*concentrations[0]),
                    k_fwd[8]*(concentrations[11]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[8])*concentrations[1]*concentrations[16]),
                    k_fwd[9]*(concentrations[12]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[9])*concentrations[17]*concentrations[1]),
                    k_fwd[10]*(concentrations[13]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[10])*concentrations[12]*concentrations[4]),
                    k_fwd[11]*(concentrations[14]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[11])*concentrations[15]),
                    k_fwd[12]*(concentrations[16]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[12])*concentrations[14]*concentrations[4]),
                    k_fwd[13]*(concentrations[16]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[13])*concentrations[15]*concentrations[1]),
                    k_fwd[14]*(concentrations[17]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[14])*concentrations[16]*concentrations[4]),
                    k_fwd[15]*(concentrations[18]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[15])*concentrations[17]*concentrations[4]),
                    k_fwd[16]*(concentrations[19]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[16])*concentrations[17]*concentrations[4]),
                    k_fwd[17]*(concentrations[20]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[17])*concentrations[18]*concentrations[4]),
                    k_fwd[18]*(concentrations[20]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[18])*concentrations[19]*concentrations[4]),
                    k_fwd[19]*(concentrations[21]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[19])*concentrations[9]*concentrations[14]),
                    k_fwd[20]*(concentrations[22]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[20])*concentrations[1]*concentrations[27]),
                    k_fwd[21]*(concentrations[22]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[21])*concentrations[21]*concentrations[4]),
                    k_fwd[22]*(concentrations[22]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[22])*concentrations[10]*concentrations[14]),
                    k_fwd[23]*(concentrations[23]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[23])*concentrations[28]*concentrations[1]),
                    k_fwd[24]*(concentrations[24]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[24])*concentrations[12]*concentrations[16]),
                    k_fwd[25]*(concentrations[25]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[25])*concentrations[17]*concentrations[12]),
                    k_fwd[26]*(concentrations[26]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[26])*concentrations[25]*concentrations[4]),
                    k_fwd[27]*(concentrations[27]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[27])*concentrations[14]**2.0*concentrations[1]),
                    k_fwd[28]*(concentrations[28]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[28])*concentrations[27]*concentrations[4]),
                    k_fwd[29]*(concentrations[28]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[29])*concentrations[10]*concentrations[15]),
                    k_fwd[30]*(concentrations[14]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[30])*concentrations[15]*concentrations[2]),
                    k_fwd[31]*(concentrations[17]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[31])*concentrations[16]*concentrations[6]),
                    k_fwd[32]*(concentrations[1]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[32])*concentrations[6]),
                    k_fwd[33]*(concentrations[1]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[33])*concentrations[6]),
                    k_fwd[34]*(concentrations[1]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[34])*concentrations[6]),
                    k_fwd[35]*(concentrations[1]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[35])*concentrations[6]),
                    k_fwd[36]*(concentrations[1]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[36])*concentrations[6]),
                    k_fwd[37]*(concentrations[1]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[37])*concentrations[2]*concentrations[4]),
                    k_fwd[38]*(concentrations[1]**2.0 + -1*self.usr_np.exp(log_k_eq[38])*concentrations[0]),
                    k_fwd[39]*(concentrations[1]**2.0 + -1*self.usr_np.exp(log_k_eq[39])*concentrations[0]),
                    k_fwd[40]*(concentrations[1]**2.0 + -1*self.usr_np.exp(log_k_eq[40])*concentrations[0]),
                    k_fwd[41]*(concentrations[1]**2.0 + -1*self.usr_np.exp(log_k_eq[41])*concentrations[0]),
                    k_fwd[42]*(concentrations[1]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[42])*concentrations[5]),
                    k_fwd[43]*(concentrations[1]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[43])*concentrations[5]*concentrations[2]),
                    k_fwd[44]*(concentrations[1]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[44])*concentrations[0]*concentrations[3]),
                    k_fwd[45]*(concentrations[1]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[45])*concentrations[4]**2.0),
                    k_fwd[46]*(concentrations[1]*concentrations[7] + -1*self.usr_np.exp(log_k_eq[46])*concentrations[0]*concentrations[6]),
                    k_fwd[47]*(concentrations[1]*concentrations[7] + -1*self.usr_np.exp(log_k_eq[47])*concentrations[5]*concentrations[4]),
                    k_fwd[48]*(concentrations[9]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[48])*concentrations[8]*concentrations[0]),
                    k_fwd[49]*(concentrations[10]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[49])*concentrations[12]),
                    k_fwd[50]*(concentrations[11]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[50])*concentrations[9]*concentrations[0]),
                    k_fwd[51]*(concentrations[12]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[51])*concentrations[13]),
                    k_fwd[52]*(concentrations[13]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[52])*concentrations[12]*concentrations[0]),
                    k_fwd[53]*(concentrations[1]*concentrations[16] + -1*self.usr_np.exp(log_k_eq[53])*concentrations[17]),
                    k_fwd[54]*(concentrations[1]*concentrations[16] + -1*self.usr_np.exp(log_k_eq[54])*concentrations[14]*concentrations[0]),
                    k_fwd[55]*(concentrations[17]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[55])*concentrations[18]),
                    k_fwd[56]*(concentrations[17]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[56])*concentrations[19]),
                    k_fwd[57]*(concentrations[17]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[57])*concentrations[0]*concentrations[16]),
                    k_fwd[58]*(concentrations[18]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[58])*concentrations[20]),
                    k_fwd[59]*(concentrations[18]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[59])*concentrations[17]*concentrations[0]),
                    k_fwd[60]*(concentrations[18]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[60])*concentrations[12]*concentrations[4]),
                    k_fwd[61]*(concentrations[18]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[61])*concentrations[11]*concentrations[5]),
                    k_fwd[62]*(concentrations[19]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[62])*concentrations[20]),
                    k_fwd[63]*(concentrations[19]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[63])*concentrations[18]*concentrations[1]),
                    k_fwd[64]*(concentrations[19]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[64])*concentrations[17]*concentrations[0]),
                    k_fwd[65]*(concentrations[19]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[65])*concentrations[12]*concentrations[4]),
                    k_fwd[66]*(concentrations[19]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[66])*concentrations[11]*concentrations[5]),
                    k_fwd[67]*(concentrations[20]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[67])*concentrations[18]*concentrations[0]),
                    k_fwd[68]*(concentrations[20]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[68])*concentrations[19]*concentrations[0]),
                    k_fwd[69]*(concentrations[21]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[69])*concentrations[22]),
                    k_fwd[70]*(concentrations[22]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[70])*concentrations[23]),
                    k_fwd[71]*(concentrations[23]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[71])*concentrations[24]),
                    k_fwd[72]*(concentrations[23]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[72])*concentrations[22]*concentrations[0]),
                    k_fwd[73]*(concentrations[24]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[73])*concentrations[25]),
                    k_fwd[74]*(concentrations[24]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[74])*concentrations[23]*concentrations[0]),
                    k_fwd[75]*(concentrations[25]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[75])*concentrations[26]),
                    k_fwd[76]*(concentrations[25]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[76])*concentrations[24]*concentrations[0]),
                    k_fwd[77]*(concentrations[26]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[77])*concentrations[25]*concentrations[0]),
                    k_fwd[78]*(concentrations[1]*concentrations[27] + -1*self.usr_np.exp(log_k_eq[78])*concentrations[11]*concentrations[14]),
                    k_fwd[79]*(concentrations[28]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[79])*concentrations[0]*concentrations[27]),
                    k_fwd[80]*(concentrations[28]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[80])*concentrations[12]*concentrations[14]),
                    k_fwd[81]*(concentrations[1]*concentrations[29] + -1*self.usr_np.exp(log_k_eq[81])*concentrations[28]*concentrations[1]),
                    k_fwd[82]*(concentrations[14]*concentrations[0] + -1*self.usr_np.exp(log_k_eq[82])*concentrations[17]),
                    k_fwd[83]*(concentrations[0]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[83])*concentrations[1]*concentrations[5]),
                    k_fwd[84]*(concentrations[4]**2.0 + -1*self.usr_np.exp(log_k_eq[84])*concentrations[7]),
                    k_fwd[85]*(concentrations[4]**2.0 + -1*self.usr_np.exp(log_k_eq[85])*concentrations[5]*concentrations[2]),
                    k_fwd[86]*(concentrations[6]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[86])*concentrations[5]*concentrations[3]),
                    k_fwd[87]*(concentrations[7]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[87])*concentrations[5]*concentrations[6]),
                    k_fwd[88]*(concentrations[7]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[88])*concentrations[5]*concentrations[6]),
                    k_fwd[89]*(concentrations[8]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[89])*concentrations[14]*concentrations[1]),
                    k_fwd[90]*(concentrations[9]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[90])*concentrations[1]*concentrations[16]),
                    k_fwd[91]*(concentrations[10]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[91])*concentrations[17]*concentrations[1]),
                    k_fwd[92]*(concentrations[10]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[92])*concentrations[9]*concentrations[5]),
                    k_fwd[93]*(concentrations[11]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[93])*concentrations[17]*concentrations[1]),
                    k_fwd[94]*(concentrations[12]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[94])*concentrations[20]),
                    k_fwd[95]*(concentrations[12]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[95])*concentrations[10]*concentrations[5]),
                    k_fwd[96]*(concentrations[12]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[96])*concentrations[11]*concentrations[5]),
                    k_fwd[97]*(concentrations[13]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[97])*concentrations[12]*concentrations[5]),
                    k_fwd[98]*(concentrations[14]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[98])*concentrations[15]*concentrations[1]),
                    k_fwd[99]*(concentrations[16]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[99])*concentrations[14]*concentrations[5]),
                    k_fwd[100]*(concentrations[17]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[100])*concentrations[5]*concentrations[16]),
                    k_fwd[101]*(concentrations[18]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[101])*concentrations[17]*concentrations[5]),
                    k_fwd[102]*(concentrations[19]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[102])*concentrations[17]*concentrations[5]),
                    k_fwd[103]*(concentrations[20]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[103])*concentrations[18]*concentrations[5]),
                    k_fwd[104]*(concentrations[20]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[104])*concentrations[19]*concentrations[5]),
                    k_fwd[105]*(concentrations[21]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[105])*concentrations[1]*concentrations[27]),
                    k_fwd[106]*(concentrations[22]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[106])*concentrations[28]*concentrations[1]),
                    k_fwd[107]*(concentrations[22]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[107])*concentrations[1]*concentrations[29]),
                    k_fwd[108]*(concentrations[22]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[108])*concentrations[21]*concentrations[5]),
                    k_fwd[109]*(concentrations[22]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[109])*concentrations[12]*concentrations[14]),
                    k_fwd[110]*(concentrations[23]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[110])*concentrations[22]*concentrations[5]),
                    k_fwd[111]*(concentrations[24]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[111])*concentrations[23]*concentrations[5]),
                    k_fwd[112]*(concentrations[26]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[112])*concentrations[25]*concentrations[5]),
                    k_fwd[113]*(concentrations[28]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[113])*concentrations[5]*concentrations[27]),
                    k_fwd[114]*(concentrations[6]**2.0 + -1*self.usr_np.exp(log_k_eq[114])*concentrations[7]*concentrations[3]),
                    k_fwd[115]*(concentrations[6]**2.0 + -1*self.usr_np.exp(log_k_eq[115])*concentrations[7]*concentrations[3]),
                    k_fwd[116]*(concentrations[10]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[116])*concentrations[17]*concentrations[4]),
                    k_fwd[117]*(concentrations[12]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[117])*concentrations[13]*concentrations[3]),
                    k_fwd[118]*(concentrations[12]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[118])*concentrations[19]*concentrations[4]),
                    k_fwd[119]*(concentrations[14]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[119])*concentrations[15]*concentrations[4]),
                    k_fwd[120]*(concentrations[17]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[120])*concentrations[7]*concentrations[16]),
                    k_fwd[121]*(concentrations[8]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[121])*concentrations[14]*concentrations[2]),
                    k_fwd[122]*(concentrations[8]*concentrations[10] + -1*self.usr_np.exp(log_k_eq[122])*concentrations[21]*concentrations[1]),
                    k_fwd[123]*(concentrations[8]*concentrations[12] + -1*self.usr_np.exp(log_k_eq[123])*concentrations[22]*concentrations[1]),
                    k_fwd[124]*(concentrations[9]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[124])*concentrations[16]*concentrations[2]),
                    k_fwd[125]*(concentrations[9]*concentrations[0] + -1*self.usr_np.exp(log_k_eq[125])*concentrations[10]*concentrations[1]),
                    k_fwd[126]*(concentrations[9]*concentrations[5] + -1*self.usr_np.exp(log_k_eq[126])*concentrations[17]*concentrations[1]),
                    k_fwd[127]*(concentrations[9]*concentrations[10] + -1*self.usr_np.exp(log_k_eq[127])*concentrations[22]*concentrations[1]),
                    k_fwd[128]*(concentrations[9]*concentrations[12] + -1*self.usr_np.exp(log_k_eq[128])*concentrations[23]*concentrations[1]),
                    k_fwd[129]*(concentrations[9]*concentrations[13] + -1*self.usr_np.exp(log_k_eq[129])*concentrations[24]*concentrations[1]),
                    k_fwd[130]*(concentrations[9]*concentrations[14] + -1*self.usr_np.exp(log_k_eq[130])*concentrations[27]),
                    k_fwd[131]*(concentrations[9]*concentrations[15] + -1*self.usr_np.exp(log_k_eq[131])*concentrations[14]*concentrations[16]),
                    k_fwd[132]*(concentrations[9]*concentrations[17] + -1*self.usr_np.exp(log_k_eq[132])*concentrations[28]*concentrations[1]),
                    k_fwd[133]*(concentrations[9]*concentrations[27] + -1*self.usr_np.exp(log_k_eq[133])*concentrations[22]*concentrations[14]),
                    k_fwd[134]*concentrations[10]*concentrations[3],
                    k_fwd[135]*(concentrations[10]*concentrations[0] + -1*self.usr_np.exp(log_k_eq[135])*concentrations[12]*concentrations[1]),
                    k_fwd[136]*(concentrations[10]**2.0 + -1*self.usr_np.exp(log_k_eq[136])*concentrations[22]*concentrations[0]),
                    k_fwd[137]*(concentrations[10]*concentrations[12] + -1*self.usr_np.exp(log_k_eq[137])*concentrations[24]*concentrations[1]),
                    k_fwd[138]*(concentrations[10]*concentrations[13] + -1*self.usr_np.exp(log_k_eq[138])*concentrations[12]**2.0),
                    k_fwd[139]*(concentrations[10]*concentrations[14] + -1*self.usr_np.exp(log_k_eq[139])*concentrations[28]),
                    k_fwd[140]*(concentrations[10]*concentrations[27] + -1*self.usr_np.exp(log_k_eq[140])*concentrations[23]*concentrations[14]),
                    k_fwd[141]*(concentrations[11]*concentrations[47] + -1*self.usr_np.exp(log_k_eq[141])*concentrations[10]*concentrations[47]),
                    k_fwd[142]*(concentrations[48]*concentrations[11] + -1*self.usr_np.exp(log_k_eq[142])*concentrations[48]*concentrations[10]),
                    k_fwd[143]*(concentrations[11]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[143])*concentrations[14]*concentrations[1]*concentrations[4]),
                    k_fwd[144]*(concentrations[11]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[144])*concentrations[14]*concentrations[5]),
                    k_fwd[145]*(concentrations[11]*concentrations[0] + -1*self.usr_np.exp(log_k_eq[145])*concentrations[12]*concentrations[1]),
                    k_fwd[146]*(concentrations[11]*concentrations[5] + -1*self.usr_np.exp(log_k_eq[146])*concentrations[20]),
                    k_fwd[147]*(concentrations[11]*concentrations[5] + -1*self.usr_np.exp(log_k_eq[147])*concentrations[10]*concentrations[5]),
                    k_fwd[148]*(concentrations[11]*concentrations[12] + -1*self.usr_np.exp(log_k_eq[148])*concentrations[24]*concentrations[1]),
                    k_fwd[149]*(concentrations[11]*concentrations[13] + -1*self.usr_np.exp(log_k_eq[149])*concentrations[12]**2.0),
                    k_fwd[150]*(concentrations[11]*concentrations[14] + -1*self.usr_np.exp(log_k_eq[150])*concentrations[10]*concentrations[14]),
                    k_fwd[151]*(concentrations[11]*concentrations[15] + -1*self.usr_np.exp(log_k_eq[151])*concentrations[10]*concentrations[15]),
                    k_fwd[152]*(concentrations[11]*concentrations[15] + -1*self.usr_np.exp(log_k_eq[152])*concentrations[17]*concentrations[14]),
                    k_fwd[153]*(concentrations[26]*concentrations[11] + -1*self.usr_np.exp(log_k_eq[153])*concentrations[25]*concentrations[12]),
                    k_fwd[154]*(concentrations[12]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[154])*concentrations[19]*concentrations[2]),
                    k_fwd[155]*(concentrations[12]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[155])*concentrations[17]*concentrations[4]),
                    k_fwd[156]*(concentrations[12]*concentrations[7] + -1*self.usr_np.exp(log_k_eq[156])*concentrations[13]*concentrations[6]),
                    k_fwd[157]*(concentrations[12]**2.0 + -1*self.usr_np.exp(log_k_eq[157])*concentrations[26]),
                    k_fwd[158]*(concentrations[12]**2.0 + -1*self.usr_np.exp(log_k_eq[158])*concentrations[25]*concentrations[1]),
                    k_fwd[159]*(concentrations[12]*concentrations[16] + -1*self.usr_np.exp(log_k_eq[159])*concentrations[13]*concentrations[14]),
                    k_fwd[160]*(concentrations[17]*concentrations[12] + -1*self.usr_np.exp(log_k_eq[160])*concentrations[13]*concentrations[16]),
                    k_fwd[161]*(concentrations[12]*concentrations[20] + -1*self.usr_np.exp(log_k_eq[161])*concentrations[18]*concentrations[13]),
                    k_fwd[162]*(concentrations[12]*concentrations[20] + -1*self.usr_np.exp(log_k_eq[162])*concentrations[19]*concentrations[13]),
                    k_fwd[163]*(concentrations[24]*concentrations[12] + -1*self.usr_np.exp(log_k_eq[163])*concentrations[23]*concentrations[13]),
                    k_fwd[164]*(concentrations[26]*concentrations[12] + -1*self.usr_np.exp(log_k_eq[164])*concentrations[25]*concentrations[13]),
                    k_fwd[165]*(concentrations[16] + -1*self.usr_np.exp(log_k_eq[165])*concentrations[14]*concentrations[1]),
                    k_fwd[166]*(concentrations[16] + -1*self.usr_np.exp(log_k_eq[166])*concentrations[14]*concentrations[1]),
                    k_fwd[167]*(concentrations[16]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[167])*concentrations[14]*concentrations[6]),
                    k_fwd[168]*(concentrations[18]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[168])*concentrations[17]*concentrations[6]),
                    k_fwd[169]*(concentrations[19]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[169])*concentrations[17]*concentrations[6]),
                    k_fwd[170]*(concentrations[21]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[170])*concentrations[14]*concentrations[16]),
                    k_fwd[171]*(concentrations[21]*concentrations[0] + -1*self.usr_np.exp(log_k_eq[171])*concentrations[22]*concentrations[1]),
                    k_fwd[172]*(concentrations[23]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[172])*concentrations[17]*concentrations[16]),
                    k_fwd[173]*(concentrations[24] + -1*self.usr_np.exp(log_k_eq[173])*concentrations[22]*concentrations[0]),
                    k_fwd[174]*(concentrations[25]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[174])*concentrations[24]*concentrations[6]),
                    k_fwd[175]*(concentrations[27]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[175])*concentrations[14]**2.0*concentrations[4]),
                    k_fwd[176]*(concentrations[27]**2.0 + -1*self.usr_np.exp(log_k_eq[176])*concentrations[22]*concentrations[14]**2.0),
                    k_fwd[177]*(concentrations[30]*concentrations[35] + -1*self.usr_np.exp(log_k_eq[177])*concentrations[47]*concentrations[2]),
                    k_fwd[178]*(concentrations[30]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[178])*concentrations[35]*concentrations[2]),
                    k_fwd[179]*(concentrations[30]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[179])*concentrations[1]*concentrations[35]),
                    k_fwd[180]*(concentrations[37]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[180])*concentrations[47]*concentrations[3]),
                    k_fwd[181]*(concentrations[37]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[181])*concentrations[35]**2.0),
                    k_fwd[182]*(concentrations[1]*concentrations[37] + -1*self.usr_np.exp(log_k_eq[182])*concentrations[47]*concentrations[4]),
                    k_fwd[183]*(concentrations[37]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[183])*concentrations[6]*concentrations[47]),
                    k_fwd[184]*(concentrations[37] + -1*self.usr_np.exp(log_k_eq[184])*concentrations[47]*concentrations[2]),
                    k_fwd[185]*(concentrations[6]*concentrations[35] + -1*self.usr_np.exp(log_k_eq[185])*concentrations[36]*concentrations[4]),
                    k_fwd[186]*(concentrations[35]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[186])*concentrations[36]),
                    k_fwd[187]*(concentrations[36]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[187])*concentrations[35]*concentrations[3]),
                    k_fwd[188]*(concentrations[1]*concentrations[36] + -1*self.usr_np.exp(log_k_eq[188])*concentrations[35]*concentrations[4]),
                    k_fwd[189]*(concentrations[31]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[189])*concentrations[1]*concentrations[35]),
                    k_fwd[190]*(concentrations[1]*concentrations[31] + -1*self.usr_np.exp(log_k_eq[190])*concentrations[0]*concentrations[30]),
                    k_fwd[191]*(concentrations[31]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[191])*concentrations[1]*concentrations[38]),
                    k_fwd[192]*(concentrations[31]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[192])*concentrations[5]*concentrations[30]),
                    k_fwd[193]*(concentrations[31]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[193])*concentrations[38]*concentrations[2]),
                    k_fwd[194]*(concentrations[31]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[194])*concentrations[35]*concentrations[4]),
                    k_fwd[195]*(concentrations[30]*concentrations[31] + -1*self.usr_np.exp(log_k_eq[195])*concentrations[1]*concentrations[47]),
                    k_fwd[196]*(concentrations[5]*concentrations[31] + -1*self.usr_np.exp(log_k_eq[196])*concentrations[0]*concentrations[38]),
                    k_fwd[197]*(concentrations[31]*concentrations[35] + -1*self.usr_np.exp(log_k_eq[197])*concentrations[47]*concentrations[4]),
                    k_fwd[198]*(concentrations[31]*concentrations[35] + -1*self.usr_np.exp(log_k_eq[198])*concentrations[1]*concentrations[37]),
                    k_fwd[199]*(concentrations[32]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[199])*concentrations[31]*concentrations[4]),
                    k_fwd[200]*(concentrations[32]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[200])*concentrations[1]*concentrations[38]),
                    k_fwd[201]*(concentrations[1]*concentrations[32] + -1*self.usr_np.exp(log_k_eq[201])*concentrations[0]*concentrations[31]),
                    k_fwd[202]*(concentrations[32]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[202])*concentrations[5]*concentrations[31]),
                    k_fwd[203]*(concentrations[34] + -1*self.usr_np.exp(log_k_eq[203])*concentrations[1]*concentrations[47]),
                    k_fwd[204]*(concentrations[34] + -1*self.usr_np.exp(log_k_eq[204])*concentrations[1]*concentrations[47]),
                    k_fwd[205]*(concentrations[34]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[205])*concentrations[6]*concentrations[47]),
                    k_fwd[206]*(concentrations[34]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[206])*concentrations[47]*concentrations[4]),
                    k_fwd[207]*(concentrations[34]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[207])*concentrations[31]*concentrations[35]),
                    k_fwd[208]*(concentrations[1]*concentrations[34] + -1*self.usr_np.exp(log_k_eq[208])*concentrations[0]*concentrations[47]),
                    k_fwd[209]*(concentrations[34]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[209])*concentrations[5]*concentrations[47]),
                    k_fwd[210]*(concentrations[12]*concentrations[34] + -1*self.usr_np.exp(log_k_eq[210])*concentrations[13]*concentrations[47]),
                    k_fwd[211]*(concentrations[1]*concentrations[35] + -1*self.usr_np.exp(log_k_eq[211])*concentrations[38]),
                    k_fwd[212]*(concentrations[38]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[212])*concentrations[35]*concentrations[4]),
                    k_fwd[213]*(concentrations[1]*concentrations[38] + -1*self.usr_np.exp(log_k_eq[213])*concentrations[0]*concentrations[35]),
                    k_fwd[214]*(concentrations[38]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[214])*concentrations[5]*concentrations[35]),
                    k_fwd[215]*(concentrations[38]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[215])*concentrations[6]*concentrations[35]),
                    k_fwd[216]*(concentrations[39]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[216])*concentrations[14]*concentrations[30]),
                    k_fwd[217]*(concentrations[39]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[217])*concentrations[1]*concentrations[46]),
                    k_fwd[218]*(concentrations[39]*concentrations[5] + -1*self.usr_np.exp(log_k_eq[218])*concentrations[40]*concentrations[4]),
                    k_fwd[219]*(concentrations[39]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[219])*concentrations[46]*concentrations[2]),
                    k_fwd[220]*(concentrations[39]*concentrations[0] + -1*self.usr_np.exp(log_k_eq[220])*concentrations[1]*concentrations[40]),
                    k_fwd[221]*(concentrations[46]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[221])*concentrations[14]*concentrations[35]),
                    k_fwd[222]*(concentrations[1]*concentrations[46] + -1*self.usr_np.exp(log_k_eq[222])*concentrations[14]*concentrations[31]),
                    k_fwd[223]*(concentrations[46]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[223])*concentrations[14]*concentrations[1]*concentrations[35]),
                    k_fwd[224]*(concentrations[30]*concentrations[46] + -1*self.usr_np.exp(log_k_eq[224])*concentrations[14]*concentrations[47]),
                    k_fwd[225]*(concentrations[46]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[225])*concentrations[15]*concentrations[35]),
                    k_fwd[226]*(concentrations[46] + -1*self.usr_np.exp(log_k_eq[226])*concentrations[14]*concentrations[30]),
                    k_fwd[227]*(concentrations[46]*concentrations[35] + -1*self.usr_np.exp(log_k_eq[227])*concentrations[14]*concentrations[37]),
                    k_fwd[228]*(concentrations[46]*concentrations[35] + -1*self.usr_np.exp(log_k_eq[228])*concentrations[15]*concentrations[47]),
                    k_fwd[229]*(concentrations[40] + -1*self.usr_np.exp(log_k_eq[229])*concentrations[39]*concentrations[1]),
                    k_fwd[230]*(concentrations[40]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[230])*concentrations[1]*concentrations[46]),
                    k_fwd[231]*(concentrations[40]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[231])*concentrations[14]*concentrations[31]),
                    k_fwd[232]*(concentrations[40]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[232])*concentrations[39]*concentrations[4]),
                    k_fwd[233]*(concentrations[40]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[233])*concentrations[1]*concentrations[44]),
                    k_fwd[234]*(concentrations[40]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[234])*concentrations[1]*concentrations[45]),
                    k_fwd[235]*(concentrations[40]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[235])*concentrations[14]*concentrations[32]),
                    k_fwd[236]*(concentrations[1]*concentrations[40] + -1*self.usr_np.exp(log_k_eq[236])*concentrations[41]),
                    k_fwd[237]*(concentrations[41]*concentrations[30] + -1*self.usr_np.exp(log_k_eq[237])*concentrations[10]*concentrations[47]),
                    k_fwd[238]*(concentrations[8]*concentrations[47] + -1*self.usr_np.exp(log_k_eq[238])*concentrations[39]*concentrations[30]),
                    k_fwd[239]*(concentrations[9]*concentrations[47] + -1*self.usr_np.exp(log_k_eq[239])*concentrations[40]*concentrations[30]),
                    k_fwd[240]*(concentrations[9]*concentrations[47] + -1*self.usr_np.exp(log_k_eq[240])*concentrations[42]),
                    k_fwd[241]*(concentrations[10]*concentrations[47] + -1*self.usr_np.exp(log_k_eq[241])*concentrations[40]*concentrations[31]),
                    k_fwd[242]*(concentrations[11]*concentrations[47] + -1*self.usr_np.exp(log_k_eq[242])*concentrations[40]*concentrations[31]),
                    k_fwd[243]*(concentrations[8]*concentrations[35] + -1*self.usr_np.exp(log_k_eq[243])*concentrations[39]*concentrations[2]),
                    k_fwd[244]*(concentrations[8]*concentrations[35] + -1*self.usr_np.exp(log_k_eq[244])*concentrations[14]*concentrations[30]),
                    k_fwd[245]*(concentrations[9]*concentrations[35] + -1*self.usr_np.exp(log_k_eq[245])*concentrations[40]*concentrations[2]),
                    k_fwd[246]*(concentrations[9]*concentrations[35] + -1*self.usr_np.exp(log_k_eq[246])*concentrations[1]*concentrations[46]),
                    k_fwd[247]*(concentrations[9]*concentrations[35] + -1*self.usr_np.exp(log_k_eq[247])*concentrations[16]*concentrations[30]),
                    k_fwd[248]*(concentrations[10]*concentrations[35] + -1*self.usr_np.exp(log_k_eq[248])*concentrations[1]*concentrations[45]),
                    k_fwd[249]*(concentrations[10]*concentrations[35] + -1*self.usr_np.exp(log_k_eq[249])*concentrations[40]*concentrations[4]),
                    k_fwd[250]*(concentrations[10]*concentrations[35] + -1*self.usr_np.exp(log_k_eq[250])*concentrations[1]*concentrations[43]),
                    k_fwd[251]*(concentrations[11]*concentrations[35] + -1*self.usr_np.exp(log_k_eq[251])*concentrations[1]*concentrations[45]),
                    k_fwd[252]*(concentrations[11]*concentrations[35] + -1*self.usr_np.exp(log_k_eq[252])*concentrations[40]*concentrations[4]),
                    k_fwd[253]*(concentrations[11]*concentrations[35] + -1*self.usr_np.exp(log_k_eq[253])*concentrations[1]*concentrations[43]),
                    k_fwd[254]*(concentrations[12]*concentrations[35] + -1*self.usr_np.exp(log_k_eq[254])*concentrations[5]*concentrations[40]),
                    k_fwd[255]*(concentrations[12]*concentrations[35] + -1*self.usr_np.exp(log_k_eq[255])*concentrations[41]*concentrations[4]),
                    k_fwd[256]*(concentrations[42]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[256])*concentrations[14]*concentrations[1]*concentrations[47]),
                    k_fwd[257]*(concentrations[42]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[257])*concentrations[40]*concentrations[35]),
                    k_fwd[258]*(concentrations[42]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[258])*concentrations[16]*concentrations[47]*concentrations[2]),
                    k_fwd[259]*(concentrations[42]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[259])*concentrations[1]*concentrations[16]*concentrations[47]),
                    k_fwd[260]*(concentrations[1]*concentrations[42] + -1*self.usr_np.exp(log_k_eq[260])*concentrations[10]*concentrations[47]),
                    k_fwd[261]*(concentrations[45]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[261])*concentrations[15]*concentrations[31]),
                    k_fwd[262]*(concentrations[45]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[262])*concentrations[14]*concentrations[38]),
                    k_fwd[263]*(concentrations[45]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[263])*concentrations[46]*concentrations[4]),
                    k_fwd[264]*(concentrations[1]*concentrations[45] + -1*self.usr_np.exp(log_k_eq[264])*concentrations[14]*concentrations[32]),
                    k_fwd[265]*(concentrations[1]*concentrations[45] + -1*self.usr_np.exp(log_k_eq[265])*concentrations[0]*concentrations[46]),
                    k_fwd[266]*(concentrations[45]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[266])*concentrations[5]*concentrations[46]),
                    k_fwd[267]*(concentrations[45]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[267])*concentrations[15]*concentrations[32]),
                    k_fwd[268]*(concentrations[45] + -1*self.usr_np.exp(log_k_eq[268])*concentrations[14]*concentrations[31]),
                    k_fwd[269]*(concentrations[1]*concentrations[43] + -1*self.usr_np.exp(log_k_eq[269])*concentrations[1]*concentrations[45]),
                    k_fwd[270]*(concentrations[1]*concentrations[43] + -1*self.usr_np.exp(log_k_eq[270])*concentrations[40]*concentrations[4]),
                    k_fwd[271]*(concentrations[1]*concentrations[43] + -1*self.usr_np.exp(log_k_eq[271])*concentrations[14]*concentrations[32]),
                    k_fwd[272]*(concentrations[1]*concentrations[44] + -1*self.usr_np.exp(log_k_eq[272])*concentrations[1]*concentrations[45]),
                    k_fwd[273]*(concentrations[27]*concentrations[35] + -1*self.usr_np.exp(log_k_eq[273])*concentrations[14]*concentrations[43]),
                    k_fwd[274]*(concentrations[12]*concentrations[30] + -1*self.usr_np.exp(log_k_eq[274])*concentrations[1]*concentrations[41]),
                    k_fwd[275]*(concentrations[12]*concentrations[30] + -1*self.usr_np.exp(log_k_eq[275])*concentrations[0]*concentrations[40]),
                    k_fwd[276]*(concentrations[1]*concentrations[33] + -1*self.usr_np.exp(log_k_eq[276])*concentrations[0]*concentrations[32]),
                    k_fwd[277]*(concentrations[33]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[277])*concentrations[5]*concentrations[32]),
                    k_fwd[278]*(concentrations[33]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[278])*concentrations[32]*concentrations[4]),
                    k_fwd[279]*(concentrations[15]*concentrations[31] + -1*self.usr_np.exp(log_k_eq[279])*concentrations[14]*concentrations[38]),
                    k_fwd[280]*(concentrations[39]*concentrations[36] + -1*self.usr_np.exp(log_k_eq[280])*concentrations[46]*concentrations[35]),
                    k_fwd[281]*(concentrations[46]*concentrations[36] + -1*self.usr_np.exp(log_k_eq[281])*concentrations[15]*concentrations[37]),
                    k_fwd[282]*(concentrations[15]*concentrations[30] + -1*self.usr_np.exp(log_k_eq[282])*concentrations[14]*concentrations[35]),
                    k_fwd[283]*concentrations[12]*concentrations[2],
                    k_fwd[284]*(concentrations[24]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[284])*concentrations[51]*concentrations[1]),
                    k_fwd[285]*(concentrations[25]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[285])*concentrations[52]*concentrations[1]),
                    k_fwd[286]*(concentrations[6]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[286])*concentrations[5]*concentrations[3]),
                    k_fwd[287]*concentrations[12]*concentrations[4],
                    k_fwd[288]*(concentrations[9]*concentrations[0] + -1*self.usr_np.exp(log_k_eq[288])*concentrations[12]),
                    k_fwd[289]*concentrations[10]*concentrations[3],
                    k_fwd[290]*(concentrations[10]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[290])*concentrations[17]*concentrations[2]),
                    k_fwd[291]*concentrations[10]**2.0,
                    k_fwd[292]*concentrations[11]*concentrations[5],
                    k_fwd[293]*(concentrations[23]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[293])*concentrations[51]*concentrations[2]),
                    k_fwd[294]*(concentrations[23]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[294])*concentrations[22]*concentrations[6]),
                    k_fwd[295]*(concentrations[52]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[295])*concentrations[51]*concentrations[4]),
                    k_fwd[296]*concentrations[52]*concentrations[2],
                    k_fwd[297]*concentrations[52]*concentrations[3],
                    k_fwd[298]*(concentrations[52]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[298])*concentrations[51]*concentrations[0]),
                    k_fwd[299]*concentrations[52]*concentrations[1],
                    k_fwd[300]*concentrations[52]*concentrations[4],
                    k_fwd[301]*concentrations[52]*concentrations[6],
                    k_fwd[302]*concentrations[52],
                    k_fwd[303]*(concentrations[28]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[303])*concentrations[51]),
                    k_fwd[304]*concentrations[51]*concentrations[2],
                    k_fwd[305]*concentrations[51]*concentrations[3],
                    k_fwd[306]*concentrations[51]*concentrations[3],
                    k_fwd[307]*(concentrations[51]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[307])*concentrations[12]*concentrations[16]),
                    k_fwd[308]*(concentrations[51]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[308])*concentrations[28]*concentrations[0]),
                    k_fwd[309]*(concentrations[51]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[309])*concentrations[28]*concentrations[5]),
                    k_fwd[310]*(concentrations[51]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[310])*concentrations[18]*concentrations[16]),
                    k_fwd[311]*(concentrations[25]*concentrations[12] + -1*self.usr_np.exp(log_k_eq[311])*concentrations[50]),
                    k_fwd[312]*(concentrations[50]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[312])*concentrations[49]*concentrations[4]),
                    k_fwd[313]*(concentrations[50]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[313])*concentrations[49]*concentrations[0]),
                    k_fwd[314]*(concentrations[50]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[314])*concentrations[49]*concentrations[5]),
                    k_fwd[315]*(concentrations[49]*concentrations[7] + -1*self.usr_np.exp(log_k_eq[315])*concentrations[50]*concentrations[6]),
                    k_fwd[316]*(concentrations[50]*concentrations[12] + -1*self.usr_np.exp(log_k_eq[316])*concentrations[49]*concentrations[13]),
                    k_fwd[317]*(concentrations[24]*concentrations[12] + -1*self.usr_np.exp(log_k_eq[317])*concentrations[49]),
                    k_fwd[318]*(concentrations[49]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[318])*concentrations[25]*concentrations[17]),
                    k_fwd[319]*(concentrations[49]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[319])*concentrations[50]),
                    k_fwd[320]*(concentrations[49]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[320])*concentrations[25]*concentrations[12]),
                    k_fwd[321]*(concentrations[49]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[321])*concentrations[25]*concentrations[18]),
                    k_fwd[322]*(concentrations[49]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[322])*concentrations[50]*concentrations[3]),
                    k_fwd[323]*concentrations[49]*concentrations[6],
                    k_fwd[324]*(concentrations[49]*concentrations[12] + -1*self.usr_np.exp(log_k_eq[324])*concentrations[25]**2.0),
               ])

    def get_net_production_rates(self, rho, temperature, mass_fractions):
        c = self.get_concentrations(rho, mass_fractions)
        r_net = self.get_net_rates_of_progress(temperature, c)
        ones = self._pyro_zeros_like(r_net[0]) + 1.0
        return self._pyro_make_array([
                r_net[7] + r_net[38] + r_net[39] + r_net[40] + r_net[41] + r_net[44] + r_net[46] + r_net[48] + r_net[50] + r_net[52] + r_net[54] + r_net[57] + r_net[59] + r_net[64] + r_net[67] + r_net[68] + r_net[72] + r_net[74] + r_net[76] + r_net[77] + r_net[79] + r_net[136] + r_net[173] + r_net[190] + r_net[196] + r_net[201] + r_net[208] + r_net[213] + r_net[265] + r_net[275] + r_net[276] + r_net[283] + r_net[287] + r_net[292] + r_net[298] + r_net[299] + r_net[308] + r_net[313] + -1*(r_net[2] + r_net[82] + r_net[83] + r_net[125] + r_net[135] + r_net[145] + r_net[171] + r_net[220] + r_net[288]) * ones,
                r_net[2] + r_net[5] + r_net[6] + r_net[8] + r_net[9] + r_net[13] + r_net[20] + r_net[23] + r_net[27] + r_net[63] + r_net[81] + r_net[83] + r_net[89] + r_net[90] + r_net[91] + r_net[93] + r_net[98] + r_net[105] + r_net[106] + r_net[107] + r_net[122] + r_net[123] + r_net[125] + r_net[126] + r_net[127] + r_net[128] + r_net[129] + r_net[132] + r_net[134] + r_net[135] + r_net[137] + r_net[143] + r_net[145] + r_net[148] + r_net[158] + r_net[165] + r_net[166] + r_net[171] + r_net[179] + r_net[189] + r_net[191] + r_net[195] + r_net[198] + r_net[200] + r_net[203] + r_net[204] + r_net[217] + r_net[220] + r_net[223] + r_net[229] + r_net[230] + r_net[233] + r_net[234] + r_net[246] + r_net[248] + r_net[250] + r_net[251] + r_net[253] + r_net[256] + r_net[259] + r_net[269] + r_net[272] + r_net[274] + r_net[283] + r_net[284] + r_net[285] + 2.0*r_net[289] + 2.0*r_net[291] + r_net[304] + -1*(r_net[1] + r_net[32] + r_net[33] + r_net[34] + r_net[35] + r_net[36] + r_net[37] + 2.0*r_net[38] + 2.0*r_net[39] + 2.0*r_net[40] + 2.0*r_net[41] + r_net[42] + r_net[43] + r_net[44] + r_net[45] + r_net[46] + r_net[47] + r_net[48] + r_net[49] + r_net[50] + r_net[51] + r_net[52] + r_net[53] + r_net[54] + r_net[55] + r_net[56] + r_net[57] + r_net[58] + r_net[59] + r_net[60] + r_net[61] + r_net[62] + r_net[63] + r_net[64] + r_net[65] + r_net[66] + r_net[67] + r_net[68] + r_net[69] + r_net[70] + r_net[71] + r_net[72] + r_net[73] + r_net[74] + r_net[75] + r_net[76] + r_net[77] + r_net[78] + r_net[79] + r_net[80] + r_net[81] + r_net[182] + r_net[188] + r_net[190] + r_net[201] + r_net[208] + r_net[211] + r_net[213] + r_net[222] + r_net[236] + r_net[260] + r_net[264] + r_net[265] + r_net[269] + r_net[270] + r_net[271] + r_net[272] + r_net[276] + r_net[298] + r_net[299] + r_net[303] + r_net[307] + r_net[308] + r_net[313] + r_net[319] + r_net[320]) * ones,
                r_net[30] + r_net[37] + r_net[43] + r_net[85] + r_net[121] + r_net[124] + r_net[154] + r_net[177] + r_net[178] + r_net[184] + r_net[193] + r_net[219] + r_net[243] + r_net[245] + r_net[258] + r_net[290] + r_net[293] + -1*(2.0*r_net[0] + r_net[1] + r_net[2] + r_net[3] + r_net[4] + r_net[5] + r_net[6] + r_net[7] + r_net[8] + r_net[9] + r_net[10] + r_net[11] + r_net[12] + r_net[13] + r_net[14] + r_net[15] + r_net[16] + r_net[17] + r_net[18] + r_net[19] + r_net[20] + r_net[21] + r_net[22] + r_net[23] + r_net[24] + r_net[25] + r_net[26] + r_net[27] + r_net[28] + r_net[29] + r_net[180] + r_net[181] + r_net[186] + r_net[187] + r_net[189] + r_net[199] + r_net[200] + r_net[206] + r_net[207] + r_net[212] + r_net[216] + r_net[221] + r_net[230] + r_net[231] + r_net[232] + r_net[256] + r_net[257] + r_net[261] + r_net[262] + r_net[263] + r_net[278] + r_net[283] + r_net[284] + r_net[285] + r_net[295] + r_net[296] + r_net[304] + r_net[312] + r_net[318]) * ones,
                r_net[0] + r_net[3] + r_net[44] + r_net[86] + r_net[114] + r_net[115] + r_net[117] + r_net[180] + r_net[187] + r_net[286] + r_net[322] + -1*(r_net[30] + r_net[31] + r_net[32] + r_net[33] + r_net[34] + r_net[35] + r_net[36] + r_net[37] + r_net[121] + r_net[124] + r_net[134] + r_net[143] + r_net[144] + r_net[154] + r_net[155] + r_net[167] + r_net[168] + r_net[169] + r_net[170] + r_net[172] + r_net[174] + r_net[175] + r_net[178] + r_net[193] + r_net[194] + r_net[205] + r_net[215] + r_net[219] + r_net[225] + r_net[258] + r_net[289] + r_net[290] + r_net[293] + r_net[294] + r_net[297] + r_net[305] + r_net[306]) * ones,
                r_net[1] + r_net[2] + r_net[3] + r_net[4] + r_net[10] + r_net[12] + r_net[14] + r_net[15] + r_net[16] + r_net[17] + r_net[18] + r_net[21] + r_net[26] + r_net[28] + r_net[37] + 2.0*r_net[45] + r_net[47] + r_net[60] + r_net[65] + r_net[116] + r_net[118] + r_net[119] + r_net[134] + r_net[143] + r_net[155] + r_net[175] + r_net[182] + r_net[185] + r_net[188] + r_net[194] + r_net[197] + r_net[199] + r_net[206] + r_net[212] + r_net[218] + r_net[232] + r_net[249] + r_net[252] + r_net[255] + r_net[263] + r_net[270] + r_net[278] + r_net[295] + r_net[296] + r_net[305] + r_net[306] + r_net[312] + r_net[323] + -1*(r_net[42] + r_net[83] + 2.0*r_net[84] + 2.0*r_net[85] + r_net[86] + r_net[87] + r_net[88] + r_net[89] + r_net[90] + r_net[91] + r_net[92] + r_net[93] + r_net[94] + r_net[95] + r_net[96] + r_net[97] + r_net[98] + r_net[99] + r_net[100] + r_net[101] + r_net[102] + r_net[103] + r_net[104] + r_net[105] + r_net[106] + r_net[107] + r_net[108] + r_net[109] + r_net[110] + r_net[111] + r_net[112] + r_net[113] + r_net[179] + r_net[183] + r_net[191] + r_net[192] + r_net[202] + r_net[209] + r_net[214] + r_net[217] + r_net[223] + r_net[233] + r_net[234] + r_net[235] + r_net[259] + r_net[266] + r_net[267] + r_net[277] + r_net[286] + r_net[287] + r_net[300] + r_net[309] + r_net[310] + r_net[314] + r_net[321]) * ones,
                r_net[42] + r_net[43] + r_net[47] + r_net[61] + r_net[66] + r_net[83] + r_net[85] + r_net[86] + r_net[87] + r_net[88] + r_net[92] + r_net[95] + r_net[96] + r_net[97] + r_net[99] + r_net[100] + r_net[101] + r_net[102] + r_net[103] + r_net[104] + r_net[108] + r_net[110] + r_net[111] + r_net[112] + r_net[113] + r_net[144] + r_net[147] + r_net[192] + r_net[202] + r_net[209] + r_net[214] + r_net[254] + r_net[266] + r_net[277] + r_net[286] + r_net[300] + r_net[309] + r_net[314] + -1*(r_net[126] + r_net[146] + r_net[147] + r_net[196] + r_net[218] + r_net[292]) * ones,
                r_net[4] + r_net[31] + r_net[32] + r_net[33] + r_net[34] + r_net[35] + r_net[36] + r_net[46] + r_net[87] + r_net[88] + r_net[156] + r_net[167] + r_net[168] + r_net[169] + r_net[174] + r_net[183] + r_net[205] + r_net[215] + r_net[294] + r_net[297] + r_net[315] + -1*(r_net[3] + r_net[43] + r_net[44] + r_net[45] + r_net[86] + 2.0*r_net[114] + 2.0*r_net[115] + r_net[116] + r_net[117] + r_net[118] + r_net[119] + r_net[120] + r_net[185] + r_net[286] + r_net[301] + r_net[322] + r_net[323]) * ones,
                r_net[84] + r_net[114] + r_net[115] + r_net[120] + r_net[301] + -1*(r_net[4] + r_net[46] + r_net[47] + r_net[87] + r_net[88] + r_net[156] + r_net[315]) * ones,
                r_net[48] + -1*(r_net[89] + r_net[121] + r_net[122] + r_net[123] + r_net[238] + r_net[243] + r_net[244]) * ones,
                r_net[19] + r_net[50] + r_net[92] + -1*(r_net[5] + r_net[48] + r_net[90] + r_net[124] + r_net[125] + r_net[126] + r_net[127] + r_net[128] + r_net[129] + r_net[130] + r_net[131] + r_net[132] + r_net[133] + r_net[239] + r_net[240] + r_net[245] + r_net[246] + r_net[247] + r_net[288]) * ones,
                r_net[22] + r_net[29] + r_net[95] + r_net[125] + r_net[141] + r_net[142] + r_net[147] + r_net[150] + r_net[151] + r_net[237] + r_net[260] + r_net[304] + -1*(r_net[6] + r_net[49] + r_net[91] + r_net[92] + r_net[116] + r_net[122] + r_net[127] + r_net[134] + r_net[135] + 2.0*r_net[136] + r_net[137] + r_net[138] + r_net[139] + r_net[140] + r_net[241] + r_net[248] + r_net[249] + r_net[250] + r_net[289] + r_net[290] + 2.0*r_net[291]) * ones,
                r_net[61] + r_net[66] + r_net[78] + r_net[96] + -1*(r_net[7] + r_net[8] + r_net[50] + r_net[93] + r_net[141] + r_net[142] + r_net[143] + r_net[144] + r_net[145] + r_net[146] + r_net[147] + r_net[148] + r_net[149] + r_net[150] + r_net[151] + r_net[152] + r_net[153] + r_net[242] + r_net[251] + r_net[252] + r_net[253] + r_net[292]) * ones,
                r_net[10] + r_net[24] + r_net[25] + r_net[49] + r_net[52] + r_net[60] + r_net[65] + r_net[80] + r_net[97] + r_net[109] + r_net[135] + 2.0*r_net[138] + r_net[145] + 2.0*r_net[149] + r_net[153] + r_net[288] + r_net[296] + r_net[297] + r_net[299] + r_net[300] + r_net[301] + r_net[307] + r_net[320] + -1*(r_net[9] + r_net[51] + r_net[94] + r_net[95] + r_net[96] + r_net[117] + r_net[118] + r_net[123] + r_net[128] + r_net[137] + r_net[148] + r_net[154] + r_net[155] + r_net[156] + 2.0*r_net[157] + 2.0*r_net[158] + r_net[159] + r_net[160] + r_net[161] + r_net[162] + r_net[163] + r_net[164] + r_net[210] + r_net[254] + r_net[255] + r_net[274] + r_net[275] + r_net[283] + r_net[287] + r_net[311] + r_net[316] + r_net[317] + r_net[324]) * ones,
                r_net[51] + r_net[117] + r_net[156] + r_net[159] + r_net[160] + r_net[161] + r_net[162] + r_net[163] + r_net[164] + r_net[210] + r_net[302] + r_net[316] + -1*(r_net[10] + r_net[52] + r_net[97] + r_net[129] + r_net[138] + r_net[149]) * ones,
                r_net[5] + r_net[7] + r_net[12] + r_net[19] + r_net[22] + 2.0*r_net[27] + r_net[54] + r_net[78] + r_net[80] + r_net[89] + r_net[99] + r_net[109] + r_net[121] + r_net[131] + r_net[133] + r_net[134] + r_net[140] + r_net[143] + r_net[144] + r_net[150] + r_net[152] + r_net[159] + r_net[165] + r_net[166] + r_net[167] + r_net[170] + 2.0*r_net[175] + 2.0*r_net[176] + r_net[216] + r_net[221] + r_net[222] + r_net[223] + r_net[224] + r_net[226] + r_net[227] + r_net[231] + r_net[235] + r_net[244] + r_net[256] + r_net[262] + r_net[264] + r_net[268] + r_net[271] + r_net[273] + r_net[279] + r_net[282] + r_net[283] + r_net[296] + r_net[297] + r_net[299] + r_net[300] + r_net[301] + r_net[302] + r_net[305] + -1*(r_net[11] + r_net[30] + r_net[82] + r_net[98] + r_net[119] + r_net[130] + r_net[139] + r_net[150]) * ones,
                r_net[11] + r_net[13] + r_net[29] + r_net[30] + r_net[98] + r_net[119] + r_net[151] + r_net[225] + r_net[228] + r_net[261] + r_net[267] + r_net[281] + r_net[289] + r_net[304] + -1*(r_net[131] + r_net[151] + r_net[152] + r_net[279] + r_net[282]) * ones,
                r_net[6] + r_net[8] + r_net[14] + r_net[24] + r_net[31] + r_net[57] + r_net[90] + r_net[100] + r_net[120] + r_net[124] + r_net[131] + r_net[160] + r_net[170] + r_net[172] + r_net[247] + r_net[258] + r_net[259] + 2.0*r_net[306] + r_net[307] + r_net[310] + -1*(r_net[12] + r_net[13] + r_net[53] + r_net[54] + r_net[99] + r_net[159] + r_net[165] + r_net[166] + r_net[167]) * ones,
                r_net[9] + r_net[15] + r_net[16] + r_net[25] + r_net[53] + r_net[59] + r_net[64] + r_net[82] + r_net[91] + r_net[93] + r_net[101] + r_net[102] + r_net[116] + r_net[126] + r_net[152] + r_net[155] + r_net[168] + r_net[169] + r_net[172] + r_net[287] + r_net[290] + r_net[292] + r_net[305] + r_net[318] + r_net[323] + -1*(r_net[14] + r_net[31] + r_net[55] + r_net[56] + r_net[57] + r_net[100] + r_net[120] + r_net[132] + r_net[160]) * ones,
                r_net[17] + r_net[55] + r_net[63] + r_net[67] + r_net[103] + r_net[161] + r_net[310] + r_net[321] + -1*(r_net[15] + r_net[58] + r_net[59] + r_net[60] + r_net[61] + r_net[101] + r_net[168]) * ones,
                r_net[18] + r_net[56] + r_net[68] + r_net[104] + r_net[118] + r_net[154] + r_net[162] + -1*(r_net[16] + r_net[62] + r_net[63] + r_net[64] + r_net[65] + r_net[66] + r_net[102] + r_net[169]) * ones,
                r_net[58] + r_net[62] + r_net[94] + r_net[146] + -1*(r_net[17] + r_net[18] + r_net[67] + r_net[68] + r_net[103] + r_net[104] + r_net[161] + r_net[162]) * ones,
                r_net[21] + r_net[108] + r_net[122] + -1*(r_net[19] + r_net[69] + r_net[105] + r_net[170] + r_net[171]) * ones,
                r_net[69] + r_net[72] + r_net[110] + r_net[123] + r_net[127] + r_net[133] + r_net[136] + r_net[171] + r_net[173] + r_net[176] + r_net[291] + r_net[294] + -1*(r_net[20] + r_net[21] + r_net[22] + r_net[70] + r_net[106] + r_net[107] + r_net[108] + r_net[109]) * ones,
                r_net[70] + r_net[74] + r_net[111] + r_net[128] + r_net[140] + r_net[163] + -1*(r_net[23] + r_net[71] + r_net[72] + r_net[110] + r_net[172] + r_net[293] + r_net[294]) * ones,
                r_net[71] + r_net[76] + r_net[129] + r_net[137] + r_net[148] + r_net[174] + -1*(r_net[24] + r_net[73] + r_net[74] + r_net[111] + r_net[163] + r_net[173] + r_net[284] + r_net[317]) * ones,
                r_net[26] + r_net[73] + r_net[77] + r_net[112] + r_net[153] + r_net[158] + r_net[164] + r_net[318] + r_net[320] + r_net[321] + r_net[323] + 2.0*r_net[324] + -1*(r_net[25] + r_net[75] + r_net[76] + r_net[174] + r_net[285] + r_net[311]) * ones,
                r_net[75] + r_net[157] + -1*(r_net[26] + r_net[77] + r_net[112] + r_net[153] + r_net[164]) * ones,
                r_net[20] + r_net[28] + r_net[79] + r_net[105] + r_net[113] + r_net[130] + -1*(r_net[27] + r_net[78] + r_net[133] + r_net[140] + r_net[175] + 2.0*r_net[176] + r_net[273]) * ones,
                r_net[23] + r_net[81] + r_net[106] + r_net[132] + r_net[139] + r_net[308] + r_net[309] + -1*(r_net[28] + r_net[29] + r_net[79] + r_net[80] + r_net[113] + r_net[303]) * ones,
                r_net[107] + -1*r_net[81] * ones,
                r_net[190] + r_net[192] + r_net[216] + r_net[226] + r_net[238] + r_net[239] + r_net[244] + r_net[247] + -1*(r_net[177] + r_net[178] + r_net[179] + r_net[195] + r_net[224] + r_net[237] + r_net[274] + r_net[275] + r_net[282]) * ones,
                r_net[199] + r_net[201] + r_net[202] + r_net[207] + r_net[222] + r_net[231] + r_net[241] + r_net[242] + r_net[261] + r_net[268] + -1*(r_net[189] + r_net[190] + r_net[191] + r_net[192] + r_net[193] + r_net[194] + r_net[195] + r_net[196] + r_net[197] + r_net[198] + r_net[279]) * ones,
                r_net[235] + r_net[264] + r_net[267] + r_net[271] + r_net[276] + r_net[277] + r_net[278] + -1*(r_net[199] + r_net[200] + r_net[201] + r_net[202]) * ones,
                -1*(r_net[276] + r_net[277] + r_net[278]) * ones,
                -1*(r_net[203] + r_net[204] + r_net[205] + r_net[206] + r_net[207] + r_net[208] + r_net[209] + r_net[210]) * ones,
                r_net[178] + r_net[179] + 2.0*r_net[181] + r_net[187] + r_net[188] + r_net[189] + r_net[194] + r_net[207] + r_net[212] + r_net[213] + r_net[214] + r_net[215] + r_net[221] + r_net[223] + r_net[225] + r_net[257] + r_net[280] + r_net[282] + -1*(r_net[177] + r_net[185] + r_net[186] + r_net[197] + r_net[198] + r_net[211] + r_net[227] + r_net[228] + r_net[243] + r_net[244] + r_net[245] + r_net[246] + r_net[247] + r_net[248] + r_net[249] + r_net[250] + r_net[251] + r_net[252] + r_net[253] + r_net[254] + r_net[255] + r_net[273]) * ones,
                r_net[185] + r_net[186] + -1*(r_net[187] + r_net[188] + r_net[280] + r_net[281]) * ones,
                r_net[198] + r_net[227] + r_net[281] + -1*(r_net[180] + r_net[181] + r_net[182] + r_net[183] + r_net[184]) * ones,
                r_net[191] + r_net[193] + r_net[196] + r_net[200] + r_net[211] + r_net[262] + r_net[279] + -1*(r_net[212] + r_net[213] + r_net[214] + r_net[215]) * ones,
                r_net[229] + r_net[232] + r_net[238] + r_net[243] + -1*(r_net[216] + r_net[217] + r_net[218] + r_net[219] + r_net[220] + r_net[280]) * ones,
                r_net[218] + r_net[220] + r_net[239] + r_net[241] + r_net[242] + r_net[245] + r_net[249] + r_net[252] + r_net[254] + r_net[257] + r_net[270] + r_net[275] + -1*(r_net[229] + r_net[230] + r_net[231] + r_net[232] + r_net[233] + r_net[234] + r_net[235] + r_net[236]) * ones,
                r_net[236] + r_net[255] + r_net[274] + -1*r_net[237] * ones,
                r_net[240] + -1*(r_net[256] + r_net[257] + r_net[258] + r_net[259] + r_net[260]) * ones,
                r_net[250] + r_net[253] + r_net[273] + -1*(r_net[269] + r_net[270] + r_net[271]) * ones,
                r_net[233] + -1*r_net[272] * ones,
                r_net[234] + r_net[248] + r_net[251] + r_net[269] + r_net[272] + -1*(r_net[261] + r_net[262] + r_net[263] + r_net[264] + r_net[265] + r_net[266] + r_net[267] + r_net[268]) * ones,
                r_net[217] + r_net[219] + r_net[230] + r_net[246] + r_net[263] + r_net[265] + r_net[266] + r_net[280] + -1*(r_net[221] + r_net[222] + r_net[223] + r_net[224] + r_net[225] + r_net[226] + r_net[227] + r_net[228] + r_net[281]) * ones,
                r_net[141] + r_net[177] + r_net[180] + r_net[182] + r_net[183] + r_net[184] + r_net[195] + r_net[197] + r_net[203] + r_net[204] + r_net[205] + r_net[206] + r_net[208] + r_net[209] + r_net[210] + r_net[224] + r_net[228] + r_net[237] + r_net[256] + r_net[258] + r_net[259] + r_net[260] + -1*(r_net[141] + r_net[238] + r_net[239] + r_net[240] + r_net[241] + r_net[242]) * ones,
                r_net[142] + -1*r_net[142] * ones,
                r_net[312] + r_net[313] + r_net[314] + r_net[316] + r_net[317] + -1*(r_net[315] + r_net[318] + r_net[319] + r_net[320] + r_net[321] + r_net[322] + r_net[323] + r_net[324]) * ones,
                r_net[311] + r_net[315] + r_net[319] + r_net[322] + -1*(r_net[312] + r_net[313] + r_net[314] + r_net[316]) * ones,
                r_net[284] + r_net[293] + r_net[295] + r_net[298] + r_net[303] + -1*(r_net[304] + r_net[305] + r_net[306] + r_net[307] + r_net[308] + r_net[309] + r_net[310]) * ones,
                r_net[285] + -1*(r_net[295] + r_net[296] + r_net[297] + r_net[298] + r_net[299] + r_net[300] + r_net[301] + r_net[302]) * ones,
               ])
