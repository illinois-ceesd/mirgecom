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
        self.model_name = 'mechs/uconn32.yaml'
        self.num_elements = 5
        self.num_species = 32
        self.num_reactions = 206
        self.num_falloff = 21

        self.one_atm = 101325.0
        self.gas_constant = 8314.46261815324
        self.big_number = 1.0e300

        self.species_names = ['H2', 'H', 'O', 'O2', 'OH', 'H2O', 'HO2', 'H2O2', 'CH', 'CH2', 'CH2*', 'CH3', 'CH4', 'CO', 'CO2', 'HCO', 'CH2O', 'CH3O', 'C2H2', 'H2CC', 'C2H3', 'C2H4', 'C2H5', 'C2H6', 'HCCO', 'CH2CO', 'CH2CHO', 'CH3CHO', 'aC3H5', 'C3H6', 'nC3H7', 'N2']
        self.species_indices = {'H2': 0, 'H': 1, 'O': 2, 'O2': 3, 'OH': 4, 'H2O': 5, 'HO2': 6, 'H2O2': 7, 'CH': 8, 'CH2': 9, 'CH2*': 10, 'CH3': 11, 'CH4': 12, 'CO': 13, 'CO2': 14, 'HCO': 15, 'CH2O': 16, 'CH3O': 17, 'C2H2': 18, 'H2CC': 19, 'C2H3': 20, 'C2H4': 21, 'C2H5': 22, 'C2H6': 23, 'HCCO': 24, 'CH2CO': 25, 'CH2CHO': 26, 'CH3CHO': 27, 'aC3H5': 28, 'C3H6': 29, 'nC3H7': 30, 'N2': 31}

        self.wts = np.array([2.016, 1.008, 15.999, 31.998, 17.007, 18.015, 33.006, 34.014, 13.018999999999998, 14.027, 14.027, 15.035, 16.043, 28.009999999999998, 44.009, 29.018, 30.026, 31.034, 26.037999999999997, 26.037999999999997, 27.046, 28.054, 29.061999999999998, 30.07, 41.028999999999996, 42.037, 43.045, 44.053, 41.073, 42.081, 43.089, 28.014])
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
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.87846473 + 0.000970913681*temperature + 1.44445655e-07*temperature**2 + -1.30687849e-10*temperature**3 + 1.76079383e-14*temperature**4, 3.48981665 + 0.000323835541*temperature + -1.68899065e-06*temperature**2 + 3.16217327e-09*temperature**3 + -1.40609067e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.87410113 + 0.00365639292*temperature + -1.40894597e-06*temperature**2 + 2.60179549e-10*temperature**3 + -1.87727567e-14*temperature**4, 3.76267867 + 0.000968872143*temperature + 2.79489841e-06*temperature**2 + -3.85091153e-09*temperature**3 + 1.68741719e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.29203842 + 0.00465588637*temperature + -2.01191947e-06*temperature**2 + 4.17906e-10*temperature**3 + -3.39716365e-14*temperature**4, 4.19860411 + -0.00236661419*temperature + 8.2329622e-06*temperature**2 + -6.68815981e-09*temperature**3 + 1.94314737e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.28571772 + 0.00723990037*temperature + -2.98714348e-06*temperature**2 + 5.95684644e-10*temperature**3 + -4.67154394e-14*temperature**4, 3.6735904 + 0.00201095175*temperature + 5.73021856e-06*temperature**2 + -6.87117425e-09*temperature**3 + 2.54385734e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 0.074851495 + 0.0133909467*temperature + -5.73285809e-06*temperature**2 + 1.22292535e-09*temperature**3 + -1.0181523e-13*temperature**4, 5.14987613 + -0.0136709788*temperature + 4.91800599e-05*temperature**2 + -4.84743026e-08*temperature**3 + 1.66693956e-11*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.71518561 + 0.00206252743*temperature + -9.98825771e-07*temperature**2 + 2.30053008e-10*temperature**3 + -2.03647716e-14*temperature**4, 3.57953347 + -0.00061035368*temperature + 1.01681433e-06*temperature**2 + 9.07005884e-10*temperature**3 + -9.04424499e-13*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.85746029 + 0.00441437026*temperature + -2.21481404e-06*temperature**2 + 5.23490188e-10*temperature**3 + -4.72084164e-14*temperature**4, 2.35677352 + 0.00898459677*temperature + -7.12356269e-06*temperature**2 + 2.45919022e-09*temperature**3 + -1.43699548e-13*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.77217438 + 0.00495695526*temperature + -2.48445613e-06*temperature**2 + 5.89161778e-10*temperature**3 + -5.33508711e-14*temperature**4, 4.22118584 + -0.00324392532*temperature + 1.37799446e-05*temperature**2 + -1.33144093e-08*temperature**3 + 4.33768865e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 1.76069008 + 0.00920000082*temperature + -4.42258813e-06*temperature**2 + 1.00641212e-09*temperature**3 + -8.8385564e-14*temperature**4, 4.79372315 + -0.00990833369*temperature + 3.73220008e-05*temperature**2 + -3.79285261e-08*temperature**3 + 1.31772652e-11*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.770799 + 0.007871497*temperature + -2.656384e-06*temperature**2 + 3.944431e-10*temperature**3 + -2.112616e-14*temperature**4, 2.106204 + 0.007216595*temperature + 5.338472e-06*temperature**2 + -7.377636e-09*temperature**3 + 2.07561e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.14756964 + 0.00596166664*temperature + -2.37294852e-06*temperature**2 + 4.67412171e-10*temperature**3 + -3.61235213e-14*temperature**4, 0.808681094 + 0.0233615629*temperature + -3.55171815e-05*temperature**2 + 2.80152437e-08*temperature**3 + -8.50072974e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.278034 + 0.0047562804*temperature + -1.6301009e-06*temperature**2 + 2.5462806e-10*temperature**3 + -1.4886379e-14*temperature**4, 3.2815483 + 0.0069764791*temperature + -2.3855244e-06*temperature**2 + -1.2104432e-09*temperature**3 + 9.8189545e-13*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.016724 + 0.0103302292*temperature + -4.68082349e-06*temperature**2 + 1.01763288e-09*temperature**3 + -8.62607041e-14*temperature**4, 3.21246645 + 0.00151479162*temperature + 2.59209412e-05*temperature**2 + -3.57657847e-08*temperature**3 + 1.47150873e-11*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.03611116 + 0.0146454151*temperature + -6.71077915e-06*temperature**2 + 1.47222923e-09*temperature**3 + -1.25706061e-13*temperature**4, 3.95920148 + -0.00757052247*temperature + 5.70990292e-05*temperature**2 + -6.91588753e-08*temperature**3 + 2.69884373e-11*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 1.95465642 + 0.0173972722*temperature + -7.98206668e-06*temperature**2 + 1.75217689e-09*temperature**3 + -1.49641576e-13*temperature**4, 4.30646568 + -0.00418658892*temperature + 4.97142807e-05*temperature**2 + -5.99126606e-08*temperature**3 + 2.30509004e-11*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 1.0718815 + 0.0216852677*temperature + -1.00256067e-05*temperature**2 + 2.21412001e-09*temperature**3 + -1.9000289e-13*temperature**4, 4.29142492 + -0.0055015427*temperature + 5.99438288e-05*temperature**2 + -7.08466285e-08*temperature**3 + 2.68685771e-11*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.6282058 + 0.0040853401*temperature + -1.5934547e-06*temperature**2 + 2.8626052e-10*temperature**3 + -1.9407832e-14*temperature**4, 2.2517214 + 0.017655021*temperature + -2.3729101e-05*temperature**2 + 1.7275759e-08*temperature**3 + -5.0664811e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.51129732 + 0.00900359745*temperature + -4.16939635e-06*temperature**2 + 9.23345882e-10*temperature**3 + -7.94838201e-14*temperature**4, 2.1358363 + 0.0181188721*temperature + -1.73947474e-05*temperature**2 + 9.34397568e-09*temperature**3 + -2.01457615e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.9756699 + 0.0081305914*temperature + -2.7436245e-06*temperature**2 + 4.0703041e-10*temperature**3 + -2.1760171e-14*temperature**4, 3.4090624 + 0.010738574*temperature + 1.8914925e-06*temperature**2 + 7.1585831e-09*temperature**3 + 2.8673851e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.4041108 + 0.011723059*temperature + -4.2263137e-06*temperature**2 + 6.8372451e-10*temperature**3 + -4.0984863e-14*temperature**4, 4.7294595 + -0.0031932858*temperature + 4.7534921e-05*temperature**2 + -5.7458611e-08*temperature**3 + 2.1931112e-11*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 6.5007877 + 0.014324731*temperature + -5.6781632e-06*temperature**2 + 1.1080801e-09*temperature**3 + -9.0363887e-14*temperature**4, 1.3631835 + 0.019813821*temperature + 1.249706e-05*temperature**2 + -3.3355555e-08*temperature**3 + 1.5846571e-11*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 6.732257 + 0.01490834*temperature + -4.949899e-06*temperature**2 + 7.212022e-10*temperature**3 + -3.766204e-14*temperature**4, 1.493307 + 0.02092518*temperature + 4.486794e-06*temperature**2 + -1.668912e-08*temperature**3 + 7.158146e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 7.7097479 + 0.016031485*temperature + -5.2720238e-06*temperature**2 + 7.5888352e-10*temperature**3 + -3.8862719e-14*temperature**4, 1.0491173 + 0.026008973*temperature + 2.3542516e-06*temperature**2 + -1.9595132e-08*temperature**3 + 9.3720207e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.92664 + 0.0014879768*temperature + -5.68476e-07*temperature**2 + 1.0097038e-10*temperature**3 + -6.753351e-15*temperature**4, 3.298677 + 0.0014082404*temperature + -3.963222e-06*temperature**2 + 5.641515e-09*temperature**3 + -2.444854e-12*temperature**4),
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
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.87846473 + 0.0004854568405*temperature + 4.814855166666666e-08*temperature**2 + -3.267196225e-11*temperature**3 + 3.5215876599999997e-15*temperature**4 + 71012.4364 / temperature, 3.48981665 + 0.0001619177705*temperature + -5.629968833333334e-07*temperature**2 + 7.905433175e-10*temperature**3 + -2.8121813400000003e-13*temperature**4 + 70797.2934 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.87410113 + 0.00182819646*temperature + -4.6964865666666667e-07*temperature**2 + 6.504488725e-11*temperature**3 + -3.75455134e-15*temperature**4 + 46263.604 / temperature, 3.76267867 + 0.0004844360715*temperature + 9.316328033333334e-07*temperature**2 + -9.627278825e-10*temperature**3 + 3.37483438e-13*temperature**4 + 46004.0401 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.29203842 + 0.002327943185*temperature + -6.706398233333333e-07*temperature**2 + 1.044765e-10*temperature**3 + -6.7943273e-15*temperature**4 + 50925.9997 / temperature, 4.19860411 + -0.001183307095*temperature + 2.744320733333333e-06*temperature**2 + -1.6720399525e-09*temperature**3 + 3.88629474e-13*temperature**4 + 50496.8163 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.28571772 + 0.003619950185*temperature + -9.957144933333333e-07*temperature**2 + 1.48921161e-10*temperature**3 + -9.34308788e-15*temperature**4 + 16775.5843 / temperature, 3.6735904 + 0.001005475875*temperature + 1.9100728533333335e-06*temperature**2 + -1.7177935625e-09*temperature**3 + 5.08771468e-13*temperature**4 + 16444.9988 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 0.074851495 + 0.00669547335*temperature + -1.9109526966666665e-06*temperature**2 + 3.057313375e-10*temperature**3 + -2.0363046000000002e-14*temperature**4 + -9468.34459 / temperature, 5.14987613 + -0.0068354894*temperature + 1.63933533e-05*temperature**2 + -1.211857565e-08*temperature**3 + 3.33387912e-12*temperature**4 + -10246.6476 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.71518561 + 0.001031263715*temperature + -3.329419236666667e-07*temperature**2 + 5.7513252e-11*temperature**3 + -4.07295432e-15*temperature**4 + -14151.8724 / temperature, 3.57953347 + -0.00030517684*temperature + 3.3893811e-07*temperature**2 + 2.26751471e-10*temperature**3 + -1.808848998e-13*temperature**4 + -14344.086 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.85746029 + 0.00220718513*temperature + -7.382713466666667e-07*temperature**2 + 1.30872547e-10*temperature**3 + -9.44168328e-15*temperature**4 + -48759.166 / temperature, 2.35677352 + 0.004492298385*temperature + -2.3745208966666665e-06*temperature**2 + 6.14797555e-10*temperature**3 + -2.8739909599999997e-14*temperature**4 + -48371.9697 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.77217438 + 0.00247847763*temperature + -8.281520433333334e-07*temperature**2 + 1.472904445e-10*temperature**3 + -1.067017422e-14*temperature**4 + 4011.91815 / temperature, 4.22118584 + -0.00162196266*temperature + 4.593314866666667e-06*temperature**2 + -3.328602325e-09*temperature**3 + 8.6753773e-13*temperature**4 + 3839.56496 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 1.76069008 + 0.00460000041*temperature + -1.4741960433333332e-06*temperature**2 + 2.5160303e-10*temperature**3 + -1.7677112800000002e-14*temperature**4 + -13995.8323 / temperature, 4.79372315 + -0.004954166845*temperature + 1.2440666933333332e-05*temperature**2 + -9.482131525e-09*temperature**3 + 2.63545304e-12*temperature**4 + -14308.9567 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.770799 + 0.0039357485*temperature + -8.854613333333334e-07*temperature**2 + 9.8610775e-11*temperature**3 + -4.225232e-15*temperature**4 + 127.83252 / temperature, 2.106204 + 0.0036082975*temperature + 1.7794906666666667e-06*temperature**2 + -1.844409e-09*temperature**3 + 4.15122e-13*temperature**4 + 978.6011 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.14756964 + 0.00298083332*temperature + -7.9098284e-07*temperature**2 + 1.1685304275e-10*temperature**3 + -7.22470426e-15*temperature**4 + 25935.9992 / temperature, 0.808681094 + 0.01168078145*temperature + -1.18390605e-05*temperature**2 + 7.003810925e-09*temperature**3 + -1.700145948e-12*temperature**4 + 26428.9807 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.278034 + 0.0023781402*temperature + -5.433669666666667e-07*temperature**2 + 6.3657015e-11*temperature**3 + -2.9772757999999997e-15*temperature**4 + 48316.688 / temperature, 3.2815483 + 0.00348823955*temperature + -7.951748e-07*temperature**2 + -3.026108e-10*temperature**3 + 1.9637909000000002e-13*temperature**4 + 48621.794 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.016724 + 0.0051651146*temperature + -1.5602744966666665e-06*temperature**2 + 2.5440822e-10*temperature**3 + -1.725214082e-14*temperature**4 + 34612.8739 / temperature, 3.21246645 + 0.00075739581*temperature + 8.640313733333333e-06*temperature**2 + -8.941446175e-09*temperature**3 + 2.94301746e-12*temperature**4 + 34859.8468 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.03611116 + 0.00732270755*temperature + -2.2369263833333335e-06*temperature**2 + 3.680573075e-10*temperature**3 + -2.51412122e-14*temperature**4 + 4939.88614 / temperature, 3.95920148 + -0.003785261235*temperature + 1.9033009733333333e-05*temperature**2 + -1.7289718825e-08*temperature**3 + 5.3976874600000004e-12*temperature**4 + 5089.77593 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 1.95465642 + 0.0086986361*temperature + -2.6606888933333333e-06*temperature**2 + 4.380442225e-10*temperature**3 + -2.99283152e-14*temperature**4 + 12857.52 / temperature, 4.30646568 + -0.00209329446*temperature + 1.65714269e-05*temperature**2 + -1.497816515e-08*temperature**3 + 4.61018008e-12*temperature**4 + 12841.6265 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 1.0718815 + 0.01084263385*temperature + -3.3418689e-06*temperature**2 + 5.535300025e-10*temperature**3 + -3.8000578e-14*temperature**4 + -11426.3932 / temperature, 4.29142492 + -0.00275077135*temperature + 1.998127626666667e-05*temperature**2 + -1.7711657125e-08*temperature**3 + 5.37371542e-12*temperature**4 + -11522.2055 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.6282058 + 0.00204267005*temperature + -5.311515666666667e-07*temperature**2 + 7.156513e-11*temperature**3 + -3.8815663999999995e-15*temperature**4 + 19327.215 / temperature, 2.2517214 + 0.0088275105*temperature + -7.909700333333333e-06*temperature**2 + 4.31893975e-09*temperature**3 + -1.0132962200000001e-12*temperature**4 + 20059.449 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.51129732 + 0.004501798725*temperature + -1.3897987833333332e-06*temperature**2 + 2.308364705e-10*temperature**3 + -1.589676402e-14*temperature**4 + -7551.05311 / temperature, 2.1358363 + 0.00905943605*temperature + -5.798249133333333e-06*temperature**2 + 2.33599392e-09*temperature**3 + -4.0291523e-13*temperature**4 + -7042.91804 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.9756699 + 0.0040652957*temperature + -9.145415e-07*temperature**2 + 1.017576025e-10*temperature**3 + -4.3520342e-15*temperature**4 + 490.32178 / temperature, 3.4090624 + 0.005369287*temperature + 6.304975e-07*temperature**2 + 1.789645775e-09*temperature**3 + 5.7347702e-13*temperature**4 + 1521.4766 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.4041108 + 0.0058615295*temperature + -1.4087712333333332e-06*temperature**2 + 1.709311275e-10*temperature**3 + -8.1969726e-15*temperature**4 + -22593.122 / temperature, 4.7294595 + -0.0015966429*temperature + 1.5844973666666667e-05*temperature**2 + -1.436465275e-08*temperature**3 + 4.3862224e-12*temperature**4 + -21572.878 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 6.5007877 + 0.0071623655*temperature + -1.8927210666666665e-06*temperature**2 + 2.77020025e-10*temperature**3 + -1.8072777399999997e-14*temperature**4 + 17482.449 / temperature, 1.3631835 + 0.0099069105*temperature + 4.165686666666667e-06*temperature**2 + -8.33888875e-09*temperature**3 + 3.1693142e-12*temperature**4 + 19245.629 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 6.732257 + 0.00745417*temperature + -1.6499663333333331e-06*temperature**2 + 1.8030055e-10*temperature**3 + -7.532408e-15*temperature**4 + -923.5703 / temperature, 1.493307 + 0.01046259*temperature + 1.495598e-06*temperature**2 + -4.17228e-09*temperature**3 + 1.4316292e-12*temperature**4 + 1074.826 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 7.7097479 + 0.0080157425*temperature + -1.7573412666666665e-06*temperature**2 + 1.8972088e-10*temperature**3 + -7.7725438e-15*temperature**4 + 7976.2236 / temperature, 1.0491173 + 0.0130044865*temperature + 7.847505333333333e-07*temperature**2 + -4.898783e-09*temperature**3 + 1.87440414e-12*temperature**4 + 10312.346 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.92664 + 0.0007439884*temperature + -1.8949200000000001e-07*temperature**2 + 2.5242595e-11*temperature**3 + -1.3506701999999999e-15*temperature**4 + -922.7977 / temperature, 3.298677 + 0.0007041202*temperature + -1.3210739999999999e-06*temperature**2 + 1.41037875e-09*temperature**3 + -4.889707999999999e-13*temperature**4 + -1020.8999 / temperature),
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
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.87846473*self.usr_np.log(temperature) + 0.000970913681*temperature + 7.22228275e-08*temperature**2 + -4.3562616333333334e-11*temperature**3 + 4.401984575e-15*temperature**4 + 5.48497999, 3.48981665*self.usr_np.log(temperature) + 0.000323835541*temperature + -8.44495325e-07*temperature**2 + 1.0540577566666666e-09*temperature**3 + -3.515226675e-13*temperature**4 + 2.08401108),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.87410113*self.usr_np.log(temperature) + 0.00365639292*temperature + -7.04472985e-07*temperature**2 + 8.672651633333333e-11*temperature**3 + -4.693189175e-15*temperature**4 + 6.17119324, 3.76267867*self.usr_np.log(temperature) + 0.000968872143*temperature + 1.397449205e-06*temperature**2 + -1.2836371766666668e-09*temperature**3 + 4.218542975e-13*temperature**4 + 1.56253185),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.29203842*self.usr_np.log(temperature) + 0.00465588637*temperature + -1.005959735e-06*temperature**2 + 1.3930200000000002e-10*temperature**3 + -8.492909125e-15*temperature**4 + 8.62650169, 4.19860411*self.usr_np.log(temperature) + -0.00236661419*temperature + 4.1164811e-06*temperature**2 + -2.2293866033333336e-09*temperature**3 + 4.857868425e-13*temperature**4 + -0.769118967),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.28571772*self.usr_np.log(temperature) + 0.00723990037*temperature + -1.49357174e-06*temperature**2 + 1.98561548e-10*temperature**3 + -1.167885985e-14*temperature**4 + 8.48007179, 3.6735904*self.usr_np.log(temperature) + 0.00201095175*temperature + 2.86510928e-06*temperature**2 + -2.2903914166666666e-09*temperature**3 + 6.35964335e-13*temperature**4 + 1.60456433),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 0.074851495*self.usr_np.log(temperature) + 0.0133909467*temperature + -2.866429045e-06*temperature**2 + 4.076417833333333e-10*temperature**3 + -2.54538075e-14*temperature**4 + 18.437318, 5.14987613*self.usr_np.log(temperature) + -0.0136709788*temperature + 2.459002995e-05*temperature**2 + -1.6158100866666668e-08*temperature**3 + 4.1673489e-12*temperature**4 + -4.64130376),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.71518561*self.usr_np.log(temperature) + 0.00206252743*temperature + -4.994128855e-07*temperature**2 + 7.6684336e-11*temperature**3 + -5.0911929e-15*temperature**4 + 7.81868772, 3.57953347*self.usr_np.log(temperature) + -0.00061035368*temperature + 5.08407165e-07*temperature**2 + 3.023352946666667e-10*temperature**3 + -2.2610612475e-13*temperature**4 + 3.50840928),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.85746029*self.usr_np.log(temperature) + 0.00441437026*temperature + -1.10740702e-06*temperature**2 + 1.7449672933333335e-10*temperature**3 + -1.18021041e-14*temperature**4 + 2.27163806, 2.35677352*self.usr_np.log(temperature) + 0.00898459677*temperature + -3.561781345e-06*temperature**2 + 8.197300733333333e-10*temperature**3 + -3.5924887e-14*temperature**4 + 9.90105222),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.77217438*self.usr_np.log(temperature) + 0.00495695526*temperature + -1.242228065e-06*temperature**2 + 1.9638725933333335e-10*temperature**3 + -1.3337717775e-14*temperature**4 + 9.79834492, 4.22118584*self.usr_np.log(temperature) + -0.00324392532*temperature + 6.8899723e-06*temperature**2 + -4.438136433333333e-09*temperature**3 + 1.0844221625e-12*temperature**4 + 3.39437243),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 1.76069008*self.usr_np.log(temperature) + 0.00920000082*temperature + -2.211294065e-06*temperature**2 + 3.3547070666666664e-10*temperature**3 + -2.2096391e-14*temperature**4 + 13.656323, 4.79372315*self.usr_np.log(temperature) + -0.00990833369*temperature + 1.86610004e-05*temperature**2 + -1.2642842033333333e-08*temperature**3 + 3.2943163e-12*temperature**4 + 0.6028129),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.770799*self.usr_np.log(temperature) + 0.007871497*temperature + -1.328192e-06*temperature**2 + 1.3148103333333333e-10*temperature**3 + -5.28154e-15*temperature**4 + 2.929575, 2.106204*self.usr_np.log(temperature) + 0.007216595*temperature + 2.669236e-06*temperature**2 + -2.459212e-09*temperature**3 + 5.189025e-13*temperature**4 + 13.152177),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.14756964*self.usr_np.log(temperature) + 0.00596166664*temperature + -1.18647426e-06*temperature**2 + 1.55804057e-10*temperature**3 + -9.030880325e-15*temperature**4 + -1.23028121, 0.808681094*self.usr_np.log(temperature) + 0.0233615629*temperature + -1.775859075e-05*temperature**2 + 9.338414566666667e-09*temperature**3 + -2.125182435e-12*temperature**4 + 13.9397051),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.278034*self.usr_np.log(temperature) + 0.0047562804*temperature + -8.1505045e-07*temperature**2 + 8.487602000000001e-11*temperature**3 + -3.72159475e-15*temperature**4 + 0.64023701, 3.2815483*self.usr_np.log(temperature) + 0.0069764791*temperature + -1.1927622e-06*temperature**2 + -4.0348106666666665e-10*temperature**3 + 2.454738625e-13*temperature**4 + 5.920391),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.016724*self.usr_np.log(temperature) + 0.0103302292*temperature + -2.340411745e-06*temperature**2 + 3.3921096e-10*temperature**3 + -2.1565176025e-14*temperature**4 + 7.78732378, 3.21246645*self.usr_np.log(temperature) + 0.00151479162*temperature + 1.29604706e-05*temperature**2 + -1.1921928233333333e-08*temperature**3 + 3.678771825e-12*temperature**4 + 8.51054025),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.03611116*self.usr_np.log(temperature) + 0.0146454151*temperature + -3.355389575e-06*temperature**2 + 4.907430766666667e-10*temperature**3 + -3.142651525e-14*temperature**4 + 10.3053693, 3.95920148*self.usr_np.log(temperature) + -0.00757052247*temperature + 2.85495146e-05*temperature**2 + -2.3052958433333332e-08*temperature**3 + 6.747109325e-12*temperature**4 + 4.09733096),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 1.95465642*self.usr_np.log(temperature) + 0.0173972722*temperature + -3.99103334e-06*temperature**2 + 5.840589633333333e-10*temperature**3 + -3.7410394e-14*temperature**4 + 13.4624343, 4.30646568*self.usr_np.log(temperature) + -0.00418658892*temperature + 2.485714035e-05*temperature**2 + -1.9970886866666665e-08*temperature**3 + 5.7627251e-12*temperature**4 + 4.70720924),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 1.0718815*self.usr_np.log(temperature) + 0.0216852677*temperature + -5.01280335e-06*temperature**2 + 7.380400033333333e-10*temperature**3 + -4.75007225e-14*temperature**4 + 15.1156107, 4.29142492*self.usr_np.log(temperature) + -0.0055015427*temperature + 2.99719144e-05*temperature**2 + -2.3615542833333335e-08*temperature**3 + 6.717144275e-12*temperature**4 + 2.66682316),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.6282058*self.usr_np.log(temperature) + 0.0040853401*temperature + -7.9672735e-07*temperature**2 + 9.542017333333333e-11*temperature**3 + -4.851958e-15*temperature**4 + -3.9302595, 2.2517214*self.usr_np.log(temperature) + 0.017655021*temperature + -1.18645505e-05*temperature**2 + 5.758586333333334e-09*temperature**3 + -1.266620275e-12*temperature**4 + 12.490417),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 4.51129732*self.usr_np.log(temperature) + 0.00900359745*temperature + -2.084698175e-06*temperature**2 + 3.0778196066666667e-10*temperature**3 + -1.9870955025e-14*temperature**4 + 0.632247205, 2.1358363*self.usr_np.log(temperature) + 0.0181188721*temperature + -8.6973737e-06*temperature**2 + 3.11465856e-09*temperature**3 + -5.036440375e-13*temperature**4 + 12.215648),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.9756699*self.usr_np.log(temperature) + 0.0081305914*temperature + -1.37181225e-06*temperature**2 + 1.3567680333333333e-10*temperature**3 + -5.44004275e-15*temperature**4 + -5.0320879, 3.4090624*self.usr_np.log(temperature) + 0.010738574*temperature + 9.4574625e-07*temperature**2 + 2.3861943666666667e-09*temperature**3 + 7.16846275e-13*temperature**4 + 9.5714535),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 5.4041108*self.usr_np.log(temperature) + 0.011723059*temperature + -2.11315685e-06*temperature**2 + 2.2790817e-10*temperature**3 + -1.024621575e-14*temperature**4 + -3.4807917, 4.7294595*self.usr_np.log(temperature) + -0.0031932858*temperature + 2.37674605e-05*temperature**2 + -1.9152870333333333e-08*temperature**3 + 5.482778e-12*temperature**4 + 4.1030159),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 6.5007877*self.usr_np.log(temperature) + 0.014324731*temperature + -2.8390816e-06*temperature**2 + 3.6936003333333333e-10*temperature**3 + -2.259097175e-14*temperature**4 + -11.24305, 1.3631835*self.usr_np.log(temperature) + 0.019813821*temperature + 6.24853e-06*temperature**2 + -1.1118518333333333e-08*temperature**3 + 3.96164275e-12*temperature**4 + 17.173214),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 6.732257*self.usr_np.log(temperature) + 0.01490834*temperature + -2.4749495e-06*temperature**2 + 2.4040073333333336e-10*temperature**3 + -9.41551e-15*temperature**4 + -13.31335, 1.493307*self.usr_np.log(temperature) + 0.02092518*temperature + 2.243397e-06*temperature**2 + -5.56304e-09*temperature**3 + 1.7895365e-12*temperature**4 + 16.14534),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 7.7097479*self.usr_np.log(temperature) + 0.016031485*temperature + -2.6360119e-06*temperature**2 + 2.5296117333333336e-10*temperature**3 + -9.71567975e-15*temperature**4 + -15.515297, 1.0491173*self.usr_np.log(temperature) + 0.026008973*temperature + 1.1771258e-06*temperature**2 + -6.531710666666667e-09*temperature**3 + 2.343005175e-12*temperature**4 + 21.136034),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.92664*self.usr_np.log(temperature) + 0.0014879768*temperature + -2.84238e-07*temperature**2 + 3.3656793333333334e-11*temperature**3 + -1.68833775e-15*temperature**4 + 5.980528, 3.298677*self.usr_np.log(temperature) + 0.0014082404*temperature + -1.981611e-06*temperature**2 + 1.8805050000000002e-09*temperature**3 + -6.112135e-13*temperature**4 + 3.950372),
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
                    g0_rt[2] + g0_rt[4] + -1*(g0_rt[1] + g0_rt[3]),
                    g0_rt[1] + g0_rt[4] + -1*(g0_rt[0] + g0_rt[2]),
                    g0_rt[1] + g0_rt[5] + -1*(g0_rt[0] + g0_rt[4]),
                    g0_rt[5] + g0_rt[2] + -1*2.0*g0_rt[4],
                    g0_rt[0] + -1*2.0*g0_rt[1] + -1*-1.0*c0,
                    g0_rt[0] + -1*2.0*g0_rt[1] + -1*-1.0*c0,
                    g0_rt[0] + -1*2.0*g0_rt[1] + -1*-1.0*c0,
                    g0_rt[0] + -1*2.0*g0_rt[1] + -1*-1.0*c0,
                    g0_rt[5] + -1*(g0_rt[1] + g0_rt[4]) + -1*-1.0*c0,
                    g0_rt[4] + -1*(g0_rt[1] + g0_rt[2]) + -1*-1.0*c0,
                    g0_rt[3] + -1*2.0*g0_rt[2] + -1*-1.0*c0,
                    g0_rt[6] + -1*(g0_rt[1] + g0_rt[3]) + -1*-1.0*c0,
                    g0_rt[6] + -1*(g0_rt[1] + g0_rt[3]) + -1*-1.0*c0,
                    g0_rt[6] + -1*(g0_rt[1] + g0_rt[3]) + -1*-1.0*c0,
                    g0_rt[6] + -1*(g0_rt[1] + g0_rt[3]) + -1*-1.0*c0,
                    g0_rt[7] + -1*2.0*g0_rt[4] + -1*-1.0*c0,
                    g0_rt[5] + g0_rt[2] + -1*(g0_rt[1] + g0_rt[6]),
                    g0_rt[0] + g0_rt[3] + -1*(g0_rt[1] + g0_rt[6]),
                    2.0*g0_rt[4] + -1*(g0_rt[1] + g0_rt[6]),
                    g0_rt[3] + g0_rt[4] + -1*(g0_rt[6] + g0_rt[2]),
                    g0_rt[5] + g0_rt[3] + -1*(g0_rt[6] + g0_rt[4]),
                    g0_rt[7] + g0_rt[3] + -1*2.0*g0_rt[6],
                    g0_rt[7] + g0_rt[3] + -1*2.0*g0_rt[6],
                    g0_rt[0] + g0_rt[6] + -1*(g0_rt[1] + g0_rt[7]),
                    g0_rt[5] + g0_rt[4] + -1*(g0_rt[1] + g0_rt[7]),
                    g0_rt[6] + g0_rt[4] + -1*(g0_rt[7] + g0_rt[2]),
                    g0_rt[5] + g0_rt[6] + -1*(g0_rt[7] + g0_rt[4]),
                    g0_rt[5] + g0_rt[6] + -1*(g0_rt[7] + g0_rt[4]),
                    g0_rt[14] + -1*(g0_rt[13] + g0_rt[2]) + -1*-1.0*c0,
                    g0_rt[14] + g0_rt[1] + -1*(g0_rt[13] + g0_rt[4]),
                    g0_rt[16] + -1*(g0_rt[13] + g0_rt[0]) + -1*-1.0*c0,
                    g0_rt[14] + g0_rt[2] + -1*(g0_rt[13] + g0_rt[3]),
                    g0_rt[14] + g0_rt[4] + -1*(g0_rt[13] + g0_rt[6]),
                    g0_rt[13] + g0_rt[1] + -1*(g0_rt[8] + g0_rt[2]),
                    g0_rt[1] + g0_rt[15] + -1*(g0_rt[8] + g0_rt[4]),
                    g0_rt[9] + g0_rt[1] + -1*(g0_rt[8] + g0_rt[0]),
                    g0_rt[16] + g0_rt[1] + -1*(g0_rt[8] + g0_rt[5]),
                    g0_rt[15] + g0_rt[2] + -1*(g0_rt[8] + g0_rt[3]),
                    g0_rt[24] + -1*(g0_rt[8] + g0_rt[13]) + -1*-1.0*c0,
                    g0_rt[13] + g0_rt[15] + -1*(g0_rt[8] + g0_rt[14]),
                    g0_rt[16] + -1*(g0_rt[1] + g0_rt[15]) + -1*-1.0*c0,
                    g0_rt[13] + g0_rt[0] + -1*(g0_rt[1] + g0_rt[15]),
                    g0_rt[13] + g0_rt[4] + -1*(g0_rt[15] + g0_rt[2]),
                    g0_rt[14] + g0_rt[1] + -1*(g0_rt[15] + g0_rt[2]),
                    g0_rt[13] + g0_rt[5] + -1*(g0_rt[15] + g0_rt[4]),
                    g0_rt[13] + g0_rt[1] + -1*g0_rt[15] + -1*c0,
                    g0_rt[13] + g0_rt[6] + -1*(g0_rt[15] + g0_rt[3]),
                    g0_rt[11] + -1*(g0_rt[9] + g0_rt[1]) + -1*-1.0*c0,
                    g0_rt[11] + g0_rt[1] + -1*(g0_rt[9] + g0_rt[0]),
                    g0_rt[1] + g0_rt[15] + -1*(g0_rt[9] + g0_rt[2]),
                    g0_rt[15] + g0_rt[4] + -1*(g0_rt[9] + g0_rt[3]),
                    g0_rt[14] + 2.0*g0_rt[1] + -1*(g0_rt[9] + g0_rt[3]) + -1*c0,
                    g0_rt[16] + g0_rt[1] + -1*(g0_rt[9] + g0_rt[4]),
                    g0_rt[8] + g0_rt[5] + -1*(g0_rt[9] + g0_rt[4]),
                    g0_rt[16] + g0_rt[4] + -1*(g0_rt[9] + g0_rt[6]),
                    g0_rt[25] + -1*(g0_rt[9] + g0_rt[13]) + -1*-1.0*c0,
                    g0_rt[18] + g0_rt[1] + -1*(g0_rt[8] + g0_rt[9]),
                    g0_rt[18] + g0_rt[0] + -1*2.0*g0_rt[9],
                    g0_rt[9] + g0_rt[31] + -1*(g0_rt[10] + g0_rt[31]),
                    g0_rt[8] + g0_rt[0] + -1*(g0_rt[10] + g0_rt[1]),
                    g0_rt[13] + g0_rt[0] + -1*(g0_rt[10] + g0_rt[2]),
                    g0_rt[1] + g0_rt[15] + -1*(g0_rt[10] + g0_rt[2]),
                    g0_rt[16] + g0_rt[1] + -1*(g0_rt[10] + g0_rt[4]),
                    g0_rt[11] + g0_rt[1] + -1*(g0_rt[10] + g0_rt[0]),
                    g0_rt[13] + g0_rt[1] + g0_rt[4] + -1*(g0_rt[10] + g0_rt[3]) + -1*c0,
                    g0_rt[13] + g0_rt[5] + -1*(g0_rt[10] + g0_rt[3]),
                    g0_rt[9] + g0_rt[5] + -1*(g0_rt[10] + g0_rt[5]),
                    g0_rt[9] + g0_rt[13] + -1*(g0_rt[10] + g0_rt[13]),
                    g0_rt[9] + g0_rt[14] + -1*(g0_rt[10] + g0_rt[14]),
                    g0_rt[16] + g0_rt[13] + -1*(g0_rt[10] + g0_rt[14]),
                    g0_rt[17] + -1*(g0_rt[16] + g0_rt[1]) + -1*-1.0*c0,
                    g0_rt[0] + g0_rt[15] + -1*(g0_rt[16] + g0_rt[1]),
                    g0_rt[15] + g0_rt[4] + -1*(g0_rt[16] + g0_rt[2]),
                    g0_rt[5] + g0_rt[15] + -1*(g0_rt[16] + g0_rt[4]),
                    g0_rt[15] + g0_rt[6] + -1*(g0_rt[16] + g0_rt[3]),
                    g0_rt[7] + g0_rt[15] + -1*(g0_rt[16] + g0_rt[6]),
                    g0_rt[25] + g0_rt[1] + -1*(g0_rt[8] + g0_rt[16]),
                    g0_rt[12] + -1*(g0_rt[11] + g0_rt[1]) + -1*-1.0*c0,
                    g0_rt[16] + g0_rt[1] + -1*(g0_rt[11] + g0_rt[2]),
                    g0_rt[9] + g0_rt[5] + -1*(g0_rt[11] + g0_rt[4]),
                    g0_rt[10] + g0_rt[5] + -1*(g0_rt[11] + g0_rt[4]),
                    g0_rt[17] + g0_rt[2] + -1*(g0_rt[11] + g0_rt[3]),
                    g0_rt[16] + g0_rt[4] + -1*(g0_rt[11] + g0_rt[3]),
                    g0_rt[12] + g0_rt[3] + -1*(g0_rt[11] + g0_rt[6]),
                    g0_rt[17] + g0_rt[4] + -1*(g0_rt[11] + g0_rt[6]),
                    g0_rt[12] + g0_rt[6] + -1*(g0_rt[11] + g0_rt[7]),
                    g0_rt[20] + g0_rt[1] + -1*(g0_rt[8] + g0_rt[11]),
                    g0_rt[12] + g0_rt[13] + -1*(g0_rt[11] + g0_rt[15]),
                    g0_rt[27] + -1*(g0_rt[11] + g0_rt[15]) + -1*-1.0*c0,
                    g0_rt[12] + g0_rt[15] + -1*(g0_rt[16] + g0_rt[11]),
                    g0_rt[21] + g0_rt[1] + -1*(g0_rt[9] + g0_rt[11]),
                    g0_rt[21] + g0_rt[1] + -1*(g0_rt[10] + g0_rt[11]),
                    g0_rt[23] + -1*2.0*g0_rt[11] + -1*-1.0*c0,
                    g0_rt[22] + g0_rt[1] + -1*2.0*g0_rt[11],
                    g0_rt[21] + g0_rt[13] + -1*(g0_rt[11] + g0_rt[24]),
                    g0_rt[16] + g0_rt[0] + -1*(g0_rt[17] + g0_rt[1]),
                    g0_rt[11] + g0_rt[4] + -1*(g0_rt[17] + g0_rt[1]),
                    g0_rt[10] + g0_rt[5] + -1*(g0_rt[17] + g0_rt[1]),
                    g0_rt[16] + g0_rt[4] + -1*(g0_rt[17] + g0_rt[2]),
                    g0_rt[16] + g0_rt[5] + -1*(g0_rt[17] + g0_rt[4]),
                    g0_rt[16] + g0_rt[6] + -1*(g0_rt[17] + g0_rt[3]),
                    g0_rt[11] + g0_rt[0] + -1*(g0_rt[12] + g0_rt[1]),
                    g0_rt[11] + g0_rt[4] + -1*(g0_rt[12] + g0_rt[2]),
                    g0_rt[11] + g0_rt[5] + -1*(g0_rt[12] + g0_rt[4]),
                    g0_rt[21] + g0_rt[1] + -1*(g0_rt[8] + g0_rt[12]),
                    2.0*g0_rt[11] + -1*(g0_rt[9] + g0_rt[12]),
                    2.0*g0_rt[11] + -1*(g0_rt[10] + g0_rt[12]),
                    g0_rt[10] + g0_rt[13] + -1*(g0_rt[1] + g0_rt[24]),
                    2.0*g0_rt[13] + g0_rt[1] + -1*(g0_rt[24] + g0_rt[2]) + -1*c0,
                    2.0*g0_rt[13] + g0_rt[4] + -1*(g0_rt[24] + g0_rt[3]) + -1*c0,
                    g0_rt[18] + g0_rt[13] + -1*(g0_rt[8] + g0_rt[24]),
                    g0_rt[20] + g0_rt[13] + -1*(g0_rt[9] + g0_rt[24]),
                    g0_rt[18] + 2.0*g0_rt[13] + -1*2.0*g0_rt[24] + -1*c0,
                    g0_rt[19] + -1*g0_rt[18],
                    g0_rt[18] + g0_rt[1] + -1*g0_rt[20] + -1*c0,
                    g0_rt[1] + g0_rt[24] + -1*(g0_rt[18] + g0_rt[2]),
                    g0_rt[9] + g0_rt[13] + -1*(g0_rt[18] + g0_rt[2]),
                    g0_rt[25] + g0_rt[1] + -1*(g0_rt[18] + g0_rt[4]),
                    g0_rt[11] + g0_rt[13] + -1*(g0_rt[18] + g0_rt[4]),
                    g0_rt[20] + g0_rt[13] + -1*(g0_rt[18] + g0_rt[15]),
                    g0_rt[28] + -1*(g0_rt[18] + g0_rt[11]) + -1*-1.0*c0,
                    g0_rt[18] + g0_rt[1] + -1*(g0_rt[1] + g0_rt[19]),
                    g0_rt[9] + g0_rt[13] + -1*(g0_rt[19] + g0_rt[2]),
                    g0_rt[25] + g0_rt[1] + -1*(g0_rt[19] + g0_rt[4]),
                    g0_rt[9] + g0_rt[14] + -1*(g0_rt[19] + g0_rt[3]),
                    g0_rt[26] + -1*(g0_rt[25] + g0_rt[1]) + -1*-1.0*c0,
                    g0_rt[0] + g0_rt[24] + -1*(g0_rt[25] + g0_rt[1]),
                    g0_rt[11] + g0_rt[13] + -1*(g0_rt[25] + g0_rt[1]),
                    g0_rt[24] + g0_rt[4] + -1*(g0_rt[25] + g0_rt[2]),
                    g0_rt[9] + g0_rt[14] + -1*(g0_rt[25] + g0_rt[2]),
                    g0_rt[5] + g0_rt[24] + -1*(g0_rt[25] + g0_rt[4]),
                    g0_rt[21] + -1*(g0_rt[20] + g0_rt[1]) + -1*-1.0*c0,
                    g0_rt[18] + g0_rt[0] + -1*(g0_rt[20] + g0_rt[1]),
                    g0_rt[0] + g0_rt[19] + -1*(g0_rt[20] + g0_rt[1]),
                    g0_rt[25] + g0_rt[1] + -1*(g0_rt[20] + g0_rt[2]),
                    g0_rt[11] + g0_rt[13] + -1*(g0_rt[20] + g0_rt[2]),
                    g0_rt[18] + g0_rt[5] + -1*(g0_rt[20] + g0_rt[4]),
                    g0_rt[18] + g0_rt[6] + -1*(g0_rt[20] + g0_rt[3]),
                    g0_rt[26] + g0_rt[2] + -1*(g0_rt[20] + g0_rt[3]),
                    g0_rt[16] + g0_rt[15] + -1*(g0_rt[20] + g0_rt[3]),
                    g0_rt[26] + g0_rt[4] + -1*(g0_rt[20] + g0_rt[6]),
                    g0_rt[21] + g0_rt[6] + -1*(g0_rt[20] + g0_rt[7]),
                    g0_rt[21] + g0_rt[13] + -1*(g0_rt[20] + g0_rt[15]),
                    g0_rt[18] + g0_rt[12] + -1*(g0_rt[20] + g0_rt[11]),
                    g0_rt[29] + -1*(g0_rt[20] + g0_rt[11]) + -1*-1.0*c0,
                    g0_rt[1] + g0_rt[28] + -1*(g0_rt[20] + g0_rt[11]),
                    g0_rt[11] + g0_rt[13] + -1*g0_rt[26] + -1*c0,
                    g0_rt[27] + -1*(g0_rt[26] + g0_rt[1]) + -1*-1.0*c0,
                    g0_rt[11] + g0_rt[15] + -1*(g0_rt[26] + g0_rt[1]),
                    g0_rt[25] + g0_rt[0] + -1*(g0_rt[26] + g0_rt[1]),
                    g0_rt[25] + g0_rt[4] + -1*(g0_rt[26] + g0_rt[2]),
                    g0_rt[25] + g0_rt[5] + -1*(g0_rt[26] + g0_rt[4]),
                    g0_rt[25] + g0_rt[6] + -1*(g0_rt[26] + g0_rt[3]),
                    g0_rt[16] + g0_rt[13] + g0_rt[4] + -1*(g0_rt[26] + g0_rt[3]) + -1*c0,
                    g0_rt[0] + g0_rt[19] + -1*g0_rt[21] + -1*c0,
                    g0_rt[22] + -1*(g0_rt[21] + g0_rt[1]) + -1*-1.0*c0,
                    g0_rt[20] + g0_rt[0] + -1*(g0_rt[21] + g0_rt[1]),
                    g0_rt[20] + g0_rt[4] + -1*(g0_rt[21] + g0_rt[2]),
                    g0_rt[11] + g0_rt[15] + -1*(g0_rt[21] + g0_rt[2]),
                    g0_rt[9] + g0_rt[16] + -1*(g0_rt[21] + g0_rt[2]),
                    g0_rt[20] + g0_rt[5] + -1*(g0_rt[21] + g0_rt[4]),
                    g0_rt[20] + g0_rt[6] + -1*(g0_rt[21] + g0_rt[3]),
                    g0_rt[27] + g0_rt[4] + -1*(g0_rt[21] + g0_rt[6]),
                    g0_rt[22] + g0_rt[13] + -1*(g0_rt[21] + g0_rt[15]),
                    g0_rt[1] + g0_rt[28] + -1*(g0_rt[21] + g0_rt[9]),
                    g0_rt[12] + g0_rt[19] + -1*(g0_rt[21] + g0_rt[10]),
                    g0_rt[1] + g0_rt[28] + -1*(g0_rt[21] + g0_rt[10]),
                    g0_rt[20] + g0_rt[12] + -1*(g0_rt[21] + g0_rt[11]),
                    g0_rt[30] + -1*(g0_rt[21] + g0_rt[11]) + -1*-1.0*c0,
                    g0_rt[23] + -1*(g0_rt[22] + g0_rt[1]) + -1*-1.0*c0,
                    g0_rt[21] + g0_rt[0] + -1*(g0_rt[22] + g0_rt[1]),
                    g0_rt[16] + g0_rt[11] + -1*(g0_rt[22] + g0_rt[2]),
                    g0_rt[27] + g0_rt[1] + -1*(g0_rt[22] + g0_rt[2]),
                    g0_rt[21] + g0_rt[6] + -1*(g0_rt[22] + g0_rt[3]),
                    g0_rt[23] + g0_rt[3] + -1*(g0_rt[22] + g0_rt[6]),
                    g0_rt[21] + g0_rt[7] + -1*(g0_rt[22] + g0_rt[6]),
                    g0_rt[16] + g0_rt[11] + g0_rt[4] + -1*(g0_rt[22] + g0_rt[6]) + -1*c0,
                    g0_rt[23] + g0_rt[6] + -1*(g0_rt[22] + g0_rt[7]),
                    g0_rt[23] + g0_rt[13] + -1*(g0_rt[22] + g0_rt[15]),
                    g0_rt[22] + g0_rt[0] + -1*(g0_rt[23] + g0_rt[1]),
                    g0_rt[22] + g0_rt[4] + -1*(g0_rt[23] + g0_rt[2]),
                    g0_rt[22] + g0_rt[5] + -1*(g0_rt[23] + g0_rt[4]),
                    g0_rt[22] + g0_rt[11] + -1*(g0_rt[23] + g0_rt[10]),
                    g0_rt[22] + g0_rt[12] + -1*(g0_rt[23] + g0_rt[11]),
                    g0_rt[29] + -1*(g0_rt[1] + g0_rt[28]) + -1*-1.0*c0,
                    g0_rt[12] + g0_rt[19] + -1*(g0_rt[1] + g0_rt[28]),
                    g0_rt[29] + g0_rt[3] + -1*(g0_rt[6] + g0_rt[28]),
                    g0_rt[20] + g0_rt[16] + g0_rt[4] + -1*(g0_rt[6] + g0_rt[28]) + -1*c0,
                    g0_rt[29] + g0_rt[13] + -1*(g0_rt[15] + g0_rt[28]),
                    g0_rt[30] + -1*(g0_rt[29] + g0_rt[1]) + -1*-1.0*c0,
                    g0_rt[21] + g0_rt[11] + -1*(g0_rt[29] + g0_rt[1]),
                    g0_rt[0] + g0_rt[28] + -1*(g0_rt[29] + g0_rt[1]),
                    g0_rt[25] + g0_rt[11] + g0_rt[1] + -1*(g0_rt[29] + g0_rt[2]) + -1*c0,
                    g0_rt[22] + g0_rt[15] + -1*(g0_rt[29] + g0_rt[2]),
                    g0_rt[4] + g0_rt[28] + -1*(g0_rt[29] + g0_rt[2]),
                    g0_rt[5] + g0_rt[28] + -1*(g0_rt[29] + g0_rt[4]),
                    g0_rt[7] + g0_rt[28] + -1*(g0_rt[29] + g0_rt[6]),
                    g0_rt[12] + g0_rt[28] + -1*(g0_rt[29] + g0_rt[11]),
                    g0_rt[22] + g0_rt[11] + -1*(g0_rt[1] + g0_rt[30]),
                    g0_rt[29] + g0_rt[0] + -1*(g0_rt[1] + g0_rt[30]),
                    g0_rt[22] + g0_rt[16] + -1*(g0_rt[2] + g0_rt[30]),
                    g0_rt[29] + g0_rt[5] + -1*(g0_rt[4] + g0_rt[30]),
                    g0_rt[29] + g0_rt[6] + -1*(g0_rt[3] + g0_rt[30]),
                    g0_rt[22] + g0_rt[16] + g0_rt[4] + -1*(g0_rt[6] + g0_rt[30]) + -1*c0,
                    g0_rt[29] + g0_rt[12] + -1*(g0_rt[11] + g0_rt[30]),
                    g0_rt[11] + g0_rt[28] + -1*(g0_rt[20] + g0_rt[22]),
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
            74000000000.00002*temperature**-0.37,
            self.usr_np.exp(10.6689553946757 + 1.5*self.usr_np.log(temperature) + -1*(40056.27486650175 / temperature)),
            50000000000.00001,
            self.usr_np.exp(20.809443533187462 + 0.48*self.usr_np.log(temperature) + -1*(-130.8370787096791 / temperature)),
            25000000000000.004*temperature**-0.8,
            self.usr_np.exp(20.51254480563076 + 0.5*self.usr_np.log(temperature) + -1*(2269.5200960794336 / temperature)),
            self.usr_np.exp(20.107079697522593 + 0.454*self.usr_np.log(temperature) + -1*(1308.370787096791 / temperature)),
            self.usr_np.exp(30.172623109393093 + -0.63*self.usr_np.log(temperature) + -1*(192.7330813300273 / temperature)),
            18000000000.000004,
            self.usr_np.exp(30.685022297606515 + -0.97*self.usr_np.log(temperature) + -1*(311.99611076923475 / temperature)),
            self.usr_np.exp(34.315632843596475 + -0.52*self.usr_np.log(temperature) + -1*(25538.391325062363 / temperature)),
            self.usr_np.exp(19.771347927429105 + 1.62*self.usr_np.log(temperature) + -1*(18643.387985359645 / temperature)),
            self.usr_np.exp(26.522358491406937 + -0.06*self.usr_np.log(temperature) + -1*(4277.366034739509 / temperature)),
            self.usr_np.exp(22.528270532924488 + 0.27*self.usr_np.log(temperature) + -1*(140.9014693796544 / temperature)),
            25000000000.000004,
            100000000000.00002,
            self.usr_np.exp(29.710462657608385 + 0.44*self.usr_np.log(temperature) + -1*(44670.79798868544 / temperature)),
            self.usr_np.exp(20.80022687808254 + 0.454*self.usr_np.log(temperature) + -1*(915.8595509677536 / temperature)),
            self.usr_np.exp(33.88677115768191 + -0.99*self.usr_np.log(temperature) + -1*(795.0868629280499 / temperature)),
            200000000000.00003,
            self.usr_np.exp(23.31102987217412 + -1*(1640.8479328794253 / temperature)),
                ])

        k_low = self._pyro_make_array([
            self.usr_np.exp(28.463930238863654 + -0.9*self.usr_np.log(temperature) + -1*(-855.4732069479018 / temperature)),
            self.usr_np.exp(49.97762777047805 + -3.42*self.usr_np.log(temperature) + -1*(42446.56765062089 / temperature)),
            self.usr_np.exp(51.646413239482754 + -3.74*self.usr_np.log(temperature) + -1*(974.2330168536105 / temperature)),
            self.usr_np.exp(41.74663626634316 + -2.57*self.usr_np.log(temperature) + -1*(717.0878352357412 / temperature)),
            self.usr_np.exp(49.51743776268064 + -3.14*self.usr_np.log(temperature) + -1*(618.9600262034819 / temperature)),
            self.usr_np.exp(63.159338704452985 + -5.11*self.usr_np.log(temperature) + -1*(3570.342590173743 / temperature)),
            self.usr_np.exp(56.050499592221364 + -4.8*self.usr_np.log(temperature) + -1*(2797.9006062531375 / temperature)),
            self.usr_np.exp(63.076845661346454 + -4.76*self.usr_np.log(temperature) + -1*(1227.8556617369884 / temperature)),
            self.usr_np.exp(97.49703126611419 + -9.588*self.usr_np.log(temperature) + -1*(2566.419620843705 / temperature)),
            self.usr_np.exp(101.88472363832375 + -9.67*self.usr_np.log(temperature) + -1*(3130.025498362323 / temperature)),
            self.usr_np.exp(28.527109140485184 + -0.64*self.usr_np.log(temperature) + -1*(25010.010814888657 / temperature)),
            self.usr_np.exp(56.20400071047983 + -3.4*self.usr_np.log(temperature) + -1*(18014.615178252938 / temperature)),
            self.usr_np.exp(81.92547932152394 + -7.64*self.usr_np.log(temperature) + -1*(5988.312448635313 / temperature)),
            self.usr_np.exp(55.59851446847831 + -3.86*self.usr_np.log(temperature) + -1*(1670.6888512159023 / temperature)),
            self.usr_np.exp(121.18603866293091 + -11.94*self.usr_np.log(temperature) + -1*(4916.354198376241 / temperature)),
            self.usr_np.exp(77.63396669439089 + -7.297*self.usr_np.log(temperature) + -1*(2365.131807444199 / temperature)),
            self.usr_np.exp(110.16740951977546 + -9.31*self.usr_np.log(temperature) + -1*(50251.50261518675 / temperature)),
            self.usr_np.exp(83.0753849045796 + -7.62*self.usr_np.log(temperature) + -1*(3507.440148486397 / temperature)),
            self.usr_np.exp(81.278612893528 + -7.08*self.usr_np.log(temperature) + -1*(3364.022581439249 / temperature)),
            self.usr_np.exp(124.62477396391213 + -12.0*self.usr_np.log(temperature) + -1*(3003.113532013934 / temperature)),
            self.usr_np.exp(75.51690316092147 + -6.66*self.usr_np.log(temperature) + -1*(3522.53673449136 / temperature)),
                ])

        reduced_pressure = self._pyro_make_array([
            (3.0*concentrations[18] + 3.0*concentrations[21] + 3.0*concentrations[23] + 2.0*concentrations[12] + 1.5*concentrations[13] + 2.0*concentrations[14] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])*k_low[0]/k_high[0],
            (3.0*concentrations[18] + 3.0*concentrations[21] + 3.0*concentrations[23] + 2.0*concentrations[12] + 1.5*concentrations[13] + 2.0*concentrations[14] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])*k_low[1]/k_high[1],
            (3.0*concentrations[18] + 3.0*concentrations[21] + 3.0*concentrations[23] + 2.0*concentrations[12] + 1.5*concentrations[13] + 2.0*concentrations[14] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])*k_low[2]/k_high[2],
            (3.0*concentrations[18] + 3.0*concentrations[21] + 3.0*concentrations[23] + 2.0*concentrations[12] + 1.5*concentrations[13] + 2.0*concentrations[14] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])*k_low[3]/k_high[3],
            (3.0*concentrations[18] + 3.0*concentrations[21] + 3.0*concentrations[23] + 2.0*concentrations[12] + 1.5*concentrations[13] + 2.0*concentrations[14] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])*k_low[4]/k_high[4],
            (3.0*concentrations[18] + 3.0*concentrations[21] + 3.0*concentrations[23] + 2.0*concentrations[12] + 1.5*concentrations[13] + 2.0*concentrations[14] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])*k_low[5]/k_high[5],
            (3.0*concentrations[18] + 3.0*concentrations[21] + 3.0*concentrations[23] + 2.0*concentrations[12] + 1.5*concentrations[13] + 2.0*concentrations[14] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])*k_low[6]/k_high[6],
            (3.0*concentrations[18] + 3.0*concentrations[21] + 3.0*concentrations[23] + 2.0*concentrations[12] + 1.5*concentrations[13] + 2.0*concentrations[14] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])*k_low[7]/k_high[7],
            (3.0*concentrations[18] + 3.0*concentrations[21] + 3.0*concentrations[23] + 2.0*concentrations[12] + 1.5*concentrations[13] + 2.0*concentrations[14] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])*k_low[8]/k_high[8],
            (3.0*concentrations[18] + 3.0*concentrations[21] + 3.0*concentrations[23] + 2.0*concentrations[12] + 1.5*concentrations[13] + 2.0*concentrations[14] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])*k_low[9]/k_high[9],
            (2.5*concentrations[18] + 2.5*concentrations[21] + 3.0*concentrations[23] + 2.0*concentrations[12] + 1.5*concentrations[13] + 2.0*concentrations[14] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])*k_low[10]/k_high[10],
            (3.0*concentrations[18] + 3.0*concentrations[21] + 3.0*concentrations[23] + 2.0*concentrations[12] + 1.5*concentrations[13] + 2.0*concentrations[14] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])*k_low[11]/k_high[11],
            (3.0*concentrations[18] + 3.0*concentrations[21] + 3.0*concentrations[23] + 2.0*concentrations[12] + 1.5*concentrations[13] + 2.0*concentrations[14] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])*k_low[12]/k_high[12],
            (3.0*concentrations[18] + 3.0*concentrations[21] + 3.0*concentrations[23] + 2.0*concentrations[12] + 1.5*concentrations[13] + 2.0*concentrations[14] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])*k_low[13]/k_high[13],
            (3.0*concentrations[18] + 3.0*concentrations[21] + 3.0*concentrations[23] + 2.0*concentrations[12] + 1.5*concentrations[13] + 2.0*concentrations[14] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])*k_low[14]/k_high[14],
            (3.0*concentrations[18] + 3.0*concentrations[21] + 3.0*concentrations[23] + 2.0*concentrations[12] + 1.5*concentrations[13] + 2.0*concentrations[14] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])*k_low[15]/k_high[15],
            (3.0*concentrations[18] + 3.0*concentrations[21] + 3.0*concentrations[23] + 2.0*concentrations[12] + 1.5*concentrations[13] + 2.0*concentrations[14] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])*k_low[16]/k_high[16],
            (3.0*concentrations[18] + 3.0*concentrations[21] + 3.0*concentrations[23] + 2.0*concentrations[12] + 1.5*concentrations[13] + 2.0*concentrations[14] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])*k_low[17]/k_high[17],
            (3.0*concentrations[18] + 3.0*concentrations[21] + 3.0*concentrations[23] + 2.0*concentrations[12] + 1.5*concentrations[13] + 2.0*concentrations[14] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])*k_low[18]/k_high[18],
            (3.0*concentrations[18] + 3.0*concentrations[21] + 3.0*concentrations[23] + 2.0*concentrations[12] + 1.5*concentrations[13] + 2.0*concentrations[14] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])*k_low[19]/k_high[19],
            (3.0*concentrations[23] + 2.0*concentrations[12] + 1.5*concentrations[13] + 2.0*concentrations[14] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])*k_low[20]/k_high[20],
                            ])

        falloff_center = self._pyro_make_array([
            self.usr_np.log10(0.26539999999999997*self.usr_np.exp((-1*temperature) / 94.0) + 0.7346*self.usr_np.exp((-1*temperature) / 1756.0) + self.usr_np.exp(-5182.0 / temperature)),
            self.usr_np.log10(0.06799999999999995*self.usr_np.exp((-1*temperature) / 197.00000000000003) + 0.932*self.usr_np.exp((-1*temperature) / 1540.0) + self.usr_np.exp(-10300.0 / temperature)),
            self.usr_np.log10(0.4243*self.usr_np.exp((-1*temperature) / 237.00000000000003) + 0.5757*self.usr_np.exp((-1*temperature) / 1652.0) + self.usr_np.exp(-5069.0 / temperature)),
            self.usr_np.log10(0.21760000000000002*self.usr_np.exp((-1*temperature) / 271.0) + 0.7824*self.usr_np.exp((-1*temperature) / 2755.0) + self.usr_np.exp(-6570.0 / temperature)),
            self.usr_np.log10(0.31999999999999995*self.usr_np.exp((-1*temperature) / 78.0) + 0.68*self.usr_np.exp((-1*temperature) / 1995.0) + self.usr_np.exp(-5590.0 / temperature)),
            self.usr_np.log10(0.4093*self.usr_np.exp((-1*temperature) / 275.0) + 0.5907*self.usr_np.exp((-1*temperature) / 1226.0) + self.usr_np.exp(-5185.0 / temperature)),
            self.usr_np.log10(0.242*self.usr_np.exp((-1*temperature) / 94.0) + 0.758*self.usr_np.exp((-1*temperature) / 1555.0) + self.usr_np.exp(-4200.0 / temperature)),
            self.usr_np.log10(0.21699999999999997*self.usr_np.exp((-1*temperature) / 74.0) + 0.783*self.usr_np.exp((-1*temperature) / 2941.0) + self.usr_np.exp(-6964.0 / temperature)),
            self.usr_np.log10(0.38270000000000004*self.usr_np.exp((-1*temperature) / 13.076000000000002) + 0.6173*self.usr_np.exp((-1*temperature) / 2078.0) + self.usr_np.exp(-5093.0 / temperature)),
            self.usr_np.log10(0.4675*self.usr_np.exp((-1*temperature) / 151.0) + 0.5325*self.usr_np.exp((-1*temperature) / 1038.0) + self.usr_np.exp(-4970.0 / temperature)),
            1,
            self.usr_np.log10(-0.9816*self.usr_np.exp((-1*temperature) / 5383.7) + 1.9816*self.usr_np.exp((-1*temperature) / 4.2932) + self.usr_np.exp(0.0795 / temperature)),
            self.usr_np.log10(0.663*self.usr_np.exp((-1*temperature) / 1707.0) + 0.337*self.usr_np.exp((-1*temperature) / 3200.0) + self.usr_np.exp(-4131.0 / temperature)),
            self.usr_np.log10(0.21799999999999997*self.usr_np.exp((-1*temperature) / 207.49999999999997) + 0.782*self.usr_np.exp((-1*temperature) / 2663.0) + self.usr_np.exp(-6095.0 / temperature)),
            self.usr_np.log10(0.825*self.usr_np.exp((-1*temperature) / 1340.6) + 0.175*self.usr_np.exp((-1*temperature) / 60000.0) + self.usr_np.exp(-10139.8 / temperature)),
            self.usr_np.log10(0.44999999999999996*self.usr_np.exp((-1*temperature) / 8900.0) + 0.55*self.usr_np.exp((-1*temperature) / 4350.0) + self.usr_np.exp(-7244.0 / temperature)),
            self.usr_np.log10(0.26549999999999996*self.usr_np.exp((-1*temperature) / 180.0) + 0.7345*self.usr_np.exp((-1*temperature) / 1035.0) + self.usr_np.exp(-5417.0 / temperature)),
            self.usr_np.log10(0.024700000000000055*self.usr_np.exp((-1*temperature) / 209.99999999999997) + 0.9753*self.usr_np.exp((-1*temperature) / 983.9999999999999) + self.usr_np.exp(-4374.0 / temperature)),
            self.usr_np.log10(0.15780000000000005*self.usr_np.exp((-1*temperature) / 125.0) + 0.8422*self.usr_np.exp((-1*temperature) / 2219.0) + self.usr_np.exp(-6882.0 / temperature)),
            self.usr_np.log10(0.98*self.usr_np.exp((-1*temperature) / 1096.6) + 0.02*self.usr_np.exp((-1*temperature) / 1096.6) + self.usr_np.exp(-6859.5 / temperature)),
            self.usr_np.log10(self.usr_np.exp((-1*temperature) / 1310.0) + self.usr_np.exp(-48097.0 / temperature)),
                        ])

        falloff_function = self._pyro_make_array([
            10**(falloff_center[0] / (1 + ((self.usr_np.log10(reduced_pressure[0]) + -0.4 + -1*0.67*falloff_center[0]) / (0.75 + -1*1.27*falloff_center[0] + -1*0.14*(self.usr_np.log10(reduced_pressure[0]) + -0.4 + -1*0.67*falloff_center[0])))**2)),
            10**(falloff_center[1] / (1 + ((self.usr_np.log10(reduced_pressure[1]) + -0.4 + -1*0.67*falloff_center[1]) / (0.75 + -1*1.27*falloff_center[1] + -1*0.14*(self.usr_np.log10(reduced_pressure[1]) + -0.4 + -1*0.67*falloff_center[1])))**2)),
            10**(falloff_center[2] / (1 + ((self.usr_np.log10(reduced_pressure[2]) + -0.4 + -1*0.67*falloff_center[2]) / (0.75 + -1*1.27*falloff_center[2] + -1*0.14*(self.usr_np.log10(reduced_pressure[2]) + -0.4 + -1*0.67*falloff_center[2])))**2)),
            10**(falloff_center[3] / (1 + ((self.usr_np.log10(reduced_pressure[3]) + -0.4 + -1*0.67*falloff_center[3]) / (0.75 + -1*1.27*falloff_center[3] + -1*0.14*(self.usr_np.log10(reduced_pressure[3]) + -0.4 + -1*0.67*falloff_center[3])))**2)),
            10**(falloff_center[4] / (1 + ((self.usr_np.log10(reduced_pressure[4]) + -0.4 + -1*0.67*falloff_center[4]) / (0.75 + -1*1.27*falloff_center[4] + -1*0.14*(self.usr_np.log10(reduced_pressure[4]) + -0.4 + -1*0.67*falloff_center[4])))**2)),
            10**(falloff_center[5] / (1 + ((self.usr_np.log10(reduced_pressure[5]) + -0.4 + -1*0.67*falloff_center[5]) / (0.75 + -1*1.27*falloff_center[5] + -1*0.14*(self.usr_np.log10(reduced_pressure[5]) + -0.4 + -1*0.67*falloff_center[5])))**2)),
            10**(falloff_center[6] / (1 + ((self.usr_np.log10(reduced_pressure[6]) + -0.4 + -1*0.67*falloff_center[6]) / (0.75 + -1*1.27*falloff_center[6] + -1*0.14*(self.usr_np.log10(reduced_pressure[6]) + -0.4 + -1*0.67*falloff_center[6])))**2)),
            10**(falloff_center[7] / (1 + ((self.usr_np.log10(reduced_pressure[7]) + -0.4 + -1*0.67*falloff_center[7]) / (0.75 + -1*1.27*falloff_center[7] + -1*0.14*(self.usr_np.log10(reduced_pressure[7]) + -0.4 + -1*0.67*falloff_center[7])))**2)),
            10**(falloff_center[8] / (1 + ((self.usr_np.log10(reduced_pressure[8]) + -0.4 + -1*0.67*falloff_center[8]) / (0.75 + -1*1.27*falloff_center[8] + -1*0.14*(self.usr_np.log10(reduced_pressure[8]) + -0.4 + -1*0.67*falloff_center[8])))**2)),
            10**(falloff_center[9] / (1 + ((self.usr_np.log10(reduced_pressure[9]) + -0.4 + -1*0.67*falloff_center[9]) / (0.75 + -1*1.27*falloff_center[9] + -1*0.14*(self.usr_np.log10(reduced_pressure[9]) + -0.4 + -1*0.67*falloff_center[9])))**2)),
            1,
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
                            ])*reduced_pressure/(1+reduced_pressure)

        k_fwd[15] = k_high[0]*falloff_function[0]*ones
        k_fwd[30] = k_high[1]*falloff_function[1]*ones
        k_fwd[38] = k_high[2]*falloff_function[2]*ones
        k_fwd[40] = k_high[3]*falloff_function[3]*ones
        k_fwd[47] = k_high[4]*falloff_function[4]*ones
        k_fwd[55] = k_high[5]*falloff_function[5]*ones
        k_fwd[70] = k_high[6]*falloff_function[6]*ones
        k_fwd[77] = k_high[7]*falloff_function[7]*ones
        k_fwd[88] = k_high[8]*falloff_function[8]*ones
        k_fwd[92] = k_high[9]*falloff_function[9]*ones
        k_fwd[113] = k_high[10]*falloff_function[10]*ones
        k_fwd[114] = k_high[11]*falloff_function[11]*ones
        k_fwd[125] = k_high[12]*falloff_function[12]*ones
        k_fwd[131] = k_high[13]*falloff_function[13]*ones
        k_fwd[144] = k_high[14]*falloff_function[14]*ones
        k_fwd[147] = k_high[15]*falloff_function[15]*ones
        k_fwd[154] = k_high[16]*falloff_function[16]*ones
        k_fwd[155] = k_high[17]*falloff_function[17]*ones
        k_fwd[169] = k_high[18]*falloff_function[18]*ones
        k_fwd[184] = k_high[19]*falloff_function[19]*ones
        k_fwd[189] = k_high[20]*falloff_function[20]*ones
        return

    def get_fwd_rate_coefficients(self, temperature, concentrations):
        ones = self._pyro_zeros_like(temperature) + 1.0
        k_fwd = [
            self.usr_np.exp(25.14210644474301 + -1*(7252.903136317711 / temperature)) * ones,
            self.usr_np.exp(3.9120230054281464 + 2.67*self.usr_np.log(temperature) + -1*(3165.2508657072367 / temperature)) * ones,
            self.usr_np.exp(12.283033686666302 + 1.51*self.usr_np.log(temperature) + -1*(1726.0429999007665 / temperature)) * ones,
            self.usr_np.exp(3.5751506887855933 + 2.4*self.usr_np.log(temperature) + -1*(-1061.7932156823956 / temperature)) * ones,
            1000000000000.0002*temperature**-1.0 * ones,
            90000000000.00002*temperature**-0.6 * ones,
            60000000000000.01*temperature**-1.25 * ones,
            550000000000000.1*temperature**-2.0 * ones,
            2.2000000000000004e+16*temperature**-2.0 * ones,
            500000000000.0001*temperature**-1.0 * ones,
            120000000000.00002*temperature**-1.0 * ones,
            2800000000000.0005*temperature**-0.86 * ones,
            300000000000000.06*temperature**-1.72 * ones,
            16520000000000.004*temperature**-0.76 * ones,
            26000000000000.004*temperature**-1.24 * ones,
            0*temperature,
            self.usr_np.exp(22.10203193164551 + -1*(337.66030697767184 / temperature)) * ones,
            self.usr_np.exp(23.532668532308907 + -1*(412.6400174689879 / temperature)) * ones,
            self.usr_np.exp(24.983124837646084 + -1*(150.96586004962973 / temperature)) * ones,
            20000000000.000004 * ones,
            self.usr_np.exp(24.56056529617862 + -1*(-251.60976674938286 / temperature)) * ones,
            self.usr_np.exp(18.683045008419857 + -1*(-820.2478396029882 / temperature)) * ones,
            self.usr_np.exp(26.763520548223827 + -1*(6038.634401985189 / temperature)) * ones,
            self.usr_np.exp(9.400960731584833 + 2.0*self.usr_np.log(temperature) + -1*(2616.741574193582 / temperature)) * ones,
            self.usr_np.exp(23.025850929940457 + -1*(1811.5903205955567 / temperature)) * ones,
            self.usr_np.exp(9.172638504792172 + 2.0*self.usr_np.log(temperature) + -1*(2012.8781339950629 / temperature)) * ones,
            self.usr_np.exp(21.282881624881835 + -1*(161.03025071960505 / temperature)) * ones,
            self.usr_np.exp(27.086293940486875 + -1*(4810.778740248201 / temperature)) * ones,
            self.usr_np.exp(20.215768003273094 + -1*(1509.6586004962971 / temperature)) * ones,
            self.usr_np.exp(10.770588040219511 + 1.228*self.usr_np.log(temperature) + -1*(35.2253673449136 / temperature)) * ones,
            0*temperature,
            self.usr_np.exp(21.639556568820566 + -1*(24053.893701241002 / temperature)) * ones,
            self.usr_np.exp(25.733901131042668 + -1*(11875.980990570872 / temperature)) * ones,
            57000000000.00001 * ones,
            30000000000.000004 * ones,
            self.usr_np.exp(11.614579118696728 + 1.79*self.usr_np.log(temperature) + -1*(840.3766209429388 / temperature)) * ones,
            self.usr_np.exp(22.465484860614332 + -1*(-379.93074779156814 / temperature)) * ones,
            33000000000.000004 * ones,
            0*temperature,
            self.usr_np.exp(21.947041268568526 + -1*(347.22147811414834 / temperature)) * ones,
            0*temperature,
            73400000000.00002 * ones,
            30000000000.000004 * ones,
            30000000000.000004 * ones,
            50000000000.00001 * ones,
            self.usr_np.exp(32.86212973278314 + -1.0*self.usr_np.log(temperature) + -1*(8554.732069479018 / temperature)) * ones,
            self.usr_np.exp(22.751414084238696 + -1*(201.2878133995063 / temperature)) * ones,
            0*temperature,
            self.usr_np.exp(6.214608098422192 + 2.0*self.usr_np.log(temperature) + -1*(3638.277227196076 / temperature)) * ones,
            80000000000.00002 * ones,
            self.usr_np.exp(23.080339115224525 + -1*(754.8293002481486 / temperature)) * ones,
            self.usr_np.exp(21.694044754104635 + -1*(754.8293002481486 / temperature)) * ones,
            20000000000.000004 * ones,
            self.usr_np.exp(9.332558004700433 + 2.0*self.usr_np.log(temperature) + -1*(1509.6586004962971 / temperature)) * ones,
            20000000000.000004 * ones,
            0*temperature,
            40000000000.00001 * ones,
            32000000000.000004 * ones,
            self.usr_np.exp(23.43131603804862 + -1*(301.93172009925945 / temperature)) * ones,
            30000000000.000004 * ones,
            15000000000.000002 * ones,
            15000000000.000002 * ones,
            30000000000.000004 * ones,
            70000000000.00002 * ones,
            28000000000.000004 * ones,
            12000000000.000002 * ones,
            30000000000.000004 * ones,
            9000000000.000002 * ones,
            7000000000.000001 * ones,
            14000000000.000002 * ones,
            0*temperature,
            self.usr_np.exp(16.951004773893423 + 1.05*self.usr_np.log(temperature) + -1*(1648.0439722084577 / temperature)) * ones,
            self.usr_np.exp(24.386827483076058 + -1*(1781.3971485856307 / temperature)) * ones,
            self.usr_np.exp(15.048070819142122 + 1.18*self.usr_np.log(temperature) + -1*(-224.9391314739483 / temperature)) * ones,
            self.usr_np.exp(25.328436022934504 + -1*(20128.78133995063 / temperature)) * ones,
            self.usr_np.exp(20.72326583694641 + -1*(4025.7562679901257 / temperature)) * ones,
            self.usr_np.exp(25.272923313004245 + -1*(-259.15805975186436 / temperature)) * ones,
            0*temperature,
            84300000000.00002 * ones,
            self.usr_np.exp(10.933106969717286 + 1.6*self.usr_np.log(temperature) + -1*(2727.4498715633104 / temperature)) * ones,
            25010000000.000004 * ones,
            self.usr_np.exp(24.15175407884447 + -1*(14492.722564764454 / temperature)) * ones,
            self.usr_np.exp(17.399029496420383 + -1*(4498.782629478966 / temperature)) * ones,
            1000000000.0000001 * ones,
            13400000000.000002 * ones,
            self.usr_np.exp(3.1986731175506815 + 2.47*self.usr_np.log(temperature) + -1*(2606.6771835236063 / temperature)) * ones,
            30000000000.000004 * ones,
            8480000000.000002 * ones,
            0*temperature,
            self.usr_np.exp(1.1999647829283973 + 2.81*self.usr_np.log(temperature) + -1*(2948.866466302767 / temperature)) * ones,
            40000000000.00001 * ones,
            self.usr_np.exp(23.208172486734412 + -1*(-286.83513409429645 / temperature)) * ones,
            0*temperature,
            self.usr_np.exp(22.33070174670984 + 0.1*self.usr_np.log(temperature) + -1*(5334.127055086917 / temperature)) * ones,
            50000000000.00001 * ones,
            20000000000.000004 * ones,
            32000000000.000004 * ones,
            16000000000.000002 * ones,
            10000000000.000002 * ones,
            5000000000.000001 * ones,
            self.usr_np.exp(-35.38740847831102 + 7.6*self.usr_np.log(temperature) + -1*(-1776.364953250643 / temperature)) * ones,
            self.usr_np.exp(13.399995114002609 + 1.62*self.usr_np.log(temperature) + -1*(5454.899743126621 / temperature)) * ones,
            self.usr_np.exp(13.835313185260453 + 1.5*self.usr_np.log(temperature) + -1*(4327.687988089386 / temperature)) * ones,
            self.usr_np.exp(11.512925464970229 + 1.6*self.usr_np.log(temperature) + -1*(1570.0449445161491 / temperature)) * ones,
            60000000000.00001 * ones,
            self.usr_np.exp(7.807916628926408 + 2.0*self.usr_np.log(temperature) + -1*(4161.6255420347925 / temperature)) * ones,
            self.usr_np.exp(23.495854559186192 + -1*(-286.83513409429645 / temperature)) * ones,
            100000000000.00002 * ones,
            100000000000.00002 * ones,
            self.usr_np.exp(21.193269466192145 + -1*(429.74948160794594 / temperature)) * ones,
            50000000000.00001 * ones,
            30000000000.000004 * ones,
            10000000000.000002 * ones,
            0*temperature,
            0*temperature,
            self.usr_np.exp(9.700146628518098 + 2.0*self.usr_np.log(temperature) + -1*(956.117113647655 / temperature)) * ones,
            self.usr_np.exp(8.313852267398207 + 2.0*self.usr_np.log(temperature) + -1*(956.117113647655 / temperature)) * ones,
            self.usr_np.exp(-15.338770774157322 + 4.5*self.usr_np.log(temperature) + -1*(-503.2195334987657 / temperature)) * ones,
            self.usr_np.exp(-14.543249183293838 + 4.0*self.usr_np.log(temperature) + -1*(-1006.4390669975314 / temperature)) * ones,
            self.usr_np.exp(9.210340371976184 + 2.0*self.usr_np.log(temperature) + -1*(3019.3172009925943 / temperature)) * ones,
            self.usr_np.exp(113.61512691707252 + -11.82*self.usr_np.log(temperature) + -1*(17980.0339319109 / temperature)) * ones,
            100000000000.00002 * ones,
            100000000000.00002 * ones,
            20000000000.000004 * ones,
            10000000000.000002 * ones,
            0*temperature,
            self.usr_np.exp(24.635288842374557 + -1*(4025.7562679901257 / temperature)) * ones,
            self.usr_np.exp(14.22097566607244 + 1.43*self.usr_np.log(temperature) + -1*(1353.6605451116798 / temperature)) * ones,
            self.usr_np.exp(23.025850929940457 + -1*(4025.7562679901257 / temperature)) * ones,
            self.usr_np.exp(21.282881624881835 + -1*(679.3463702233338 / temperature)) * ones,
            self.usr_np.exp(22.738168857488677 + -1*(1006.4390669975314 / temperature)) * ones,
            0*temperature,
            30000000000.000004 * ones,
            60000000000.00001 * ones,
            48000000000.00001 * ones,
            48000000000.00001 * ones,
            30110000000.000004 * ones,
            self.usr_np.exp(7.200424892944957 + 1.61*self.usr_np.log(temperature) + -1*(-192.93436914342678 / temperature)) * ones,
            self.usr_np.exp(19.519293032620475 + 0.29*self.usr_np.log(temperature) + -1*(5.5354148684864235 / temperature)) * ones,
            self.usr_np.exp(31.459662512417644 + -1.39*self.usr_np.log(temperature) + -1*(508.2517288337534 / temperature)) * ones,
            10000000000.000002 * ones,
            self.usr_np.exp(16.308716010566968 + -1*(-299.9188419652644 / temperature)) * ones,
            90330000000.00002 * ones,
            392000000.00000006 * ones,
            0*temperature,
            self.usr_np.exp(48.759752060983125 + -2.83*self.usr_np.log(temperature) + -1*(9368.94127468002 / temperature)) * ones,
            self.usr_np.exp(96.46011254645141 + -9.147*self.usr_np.log(temperature) + -1*(23600.996121092114 / temperature)) * ones,
            0*temperature,
            90000000000.00002 * ones,
            self.usr_np.exp(23.7189981105004 + -1*(2012.8781339950629 / temperature)) * ones,
            self.usr_np.exp(23.7189981105004 + -1*(2012.8781339950629 / temperature)) * ones,
            self.usr_np.exp(23.025850929940457 + -1*(1006.4390669975314 / temperature)) * ones,
            140000000.00000003 * ones,
            18000000.000000004 * ones,
            0*temperature,
            0*temperature,
            self.usr_np.exp(10.833681189579275 + 1.93*self.usr_np.log(temperature) + -1*(6516.692958809016 / temperature)) * ones,
            self.usr_np.exp(9.622450022803015 + 1.91*self.usr_np.log(temperature) + -1*(1882.041055285384 / temperature)) * ones,
            self.usr_np.exp(9.862665558015873 + 1.83*self.usr_np.log(temperature) + -1*(110.70829736972846 / temperature)) * ones,
            self.usr_np.exp(5.950642552587727 + 1.83*self.usr_np.log(temperature) + -1*(110.70829736972846 / temperature)) * ones,
            self.usr_np.exp(8.1886891244442 + 2.0*self.usr_np.log(temperature) + -1*(1258.0488337469144 / temperature)) * ones,
            self.usr_np.exp(24.46568605798838 + -1*(30595.74763672496 / temperature)) * ones,
            self.usr_np.exp(21.416413017506358 + -1*(7045.07346898272 / temperature)) * ones,
            self.usr_np.exp(9.210340371976184 + 2.0*self.usr_np.log(temperature) + -1*(4025.7562679901257 / temperature)) * ones,
            self.usr_np.exp(23.7189981105004 + -1*(3019.3172009925943 / temperature)) * ones,
            50000000000.00001 * ones,
            50000000000.00001 * ones,
            self.usr_np.exp(5.424950017481403 + 2.0*self.usr_np.log(temperature) + -1*(4629.619708188645 / temperature)) * ones,
            self.usr_np.exp(19.6146032124248 + -1*(3874.790407940496 / temperature)) * ones,
            0*temperature,
            2000000000.0000002 * ones,
            16040000000.000002 * ones,
            80200000000.00002 * ones,
            20000000.000000004 * ones,
            300000000.00000006 * ones,
            300000000.00000006 * ones,
            24000000000.000004 * ones,
            self.usr_np.exp(15.978833583624812 + -1*(490.13582562779783 / temperature)) * ones,
            120000000000.00002 * ones,
            self.usr_np.exp(11.652687407345388 + 1.9*self.usr_np.log(temperature) + -1*(3789.243087245706 / temperature)) * ones,
            self.usr_np.exp(11.40534025429029 + 1.92*self.usr_np.log(temperature) + -1*(2863.319145607977 / temperature)) * ones,
            self.usr_np.exp(8.17188200612782 + 2.12*self.usr_np.log(temperature) + -1*(437.8009941439262 / temperature)) * ones,
            self.usr_np.exp(24.412145291060348 + -1*(-276.77074342432115 / temperature)) * ones,
            self.usr_np.exp(8.722580021141189 + 1.74*self.usr_np.log(temperature) + -1*(5258.644125062102 / temperature)) * ones,
            0*temperature,
            self.usr_np.exp(23.7189981105004 + -1*(1006.4390669975314 / temperature)) * ones,
            2660000000.0000005 * ones,
            6600000000.000001 * ones,
            60000000000.00001 * ones,
            0*temperature,
            self.usr_np.exp(44.2191203961326 + -2.39*self.usr_np.log(temperature) + -1*(5625.994384516201 / temperature)) * ones,
            self.usr_np.exp(5.135798437050262 + 2.5*self.usr_np.log(temperature) + -1*(1253.0166384119266 / temperature)) * ones,
            self.usr_np.exp(11.695247021764184 + 1.65*self.usr_np.log(temperature) + -1*(164.5527874540964 / temperature)) * ones,
            self.usr_np.exp(10.463103340471552 + 1.65*self.usr_np.log(temperature) + -1*(-489.1293865608003 / temperature)) * ones,
            self.usr_np.exp(19.008467408854486 + 0.7*self.usr_np.log(temperature) + -1*(2958.9308569727427 / temperature)) * ones,
            self.usr_np.exp(8.039157390473237 + 2.0*self.usr_np.log(temperature) + -1*(-149.9594209826322 / temperature)) * ones,
            self.usr_np.exp(2.2617630984737906 + 2.6*self.usr_np.log(temperature) + -1*(6999.783710967831 / temperature)) * ones,
            self.usr_np.exp(-6.119297918617867 + 3.5*self.usr_np.log(temperature) + -1*(2855.7708526054957 / temperature)) * ones,
            self.usr_np.exp(49.66261977252514 + -2.92*self.usr_np.log(temperature) + -1*(6292.760266402065 / temperature)) * ones,
            1800000000.0000002 * ones,
            96000000000.00002 * ones,
            24000000000.000004 * ones,
            90000000.00000001 * ones,
            24000000000.000004 * ones,
            11000000000.000002 * ones,
            self.usr_np.exp(68.13594424996293 + -5.22*self.usr_np.log(temperature) + -1*(9937.076128000126 / temperature)) * ones,
                ]
        self.get_falloff_rates(temperature, concentrations, k_fwd)

        k_fwd[4] *= (3.0*concentrations[18] + 3.0*concentrations[21] + 3.0*concentrations[23] + 2.0*concentrations[12] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[13] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])
        k_fwd[5] *= (concentrations[0])
        k_fwd[6] *= (concentrations[5])
        k_fwd[7] *= (concentrations[14])
        k_fwd[8] *= (3.0*concentrations[18] + 3.0*concentrations[21] + 3.0*concentrations[23] + 2.0*concentrations[12] + 0.73*concentrations[0] + 3.65*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[13] + concentrations[14] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])
        k_fwd[9] *= (3.0*concentrations[18] + 3.0*concentrations[21] + 3.0*concentrations[23] + 2.0*concentrations[12] + 1.5*concentrations[13] + 2.0*concentrations[14] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])
        k_fwd[10] *= (3.0*concentrations[18] + 3.0*concentrations[21] + 3.0*concentrations[23] + 2.0*concentrations[12] + 1.75*concentrations[13] + 3.6*concentrations[14] + 2.4*concentrations[0] + 15.4*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])
        k_fwd[11] *= (3.0*concentrations[18] + 3.0*concentrations[21] + 1.5*concentrations[23] + 0.75*concentrations[13] + 1.5*concentrations[14] + concentrations[0] + concentrations[1] + concentrations[2] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30])
        k_fwd[12] *= (concentrations[3])
        k_fwd[13] *= (concentrations[5])
        k_fwd[14] *= (concentrations[31])
        k_fwd[28] *= (3.0*concentrations[18] + 3.0*concentrations[21] + 3.0*concentrations[23] + 2.0*concentrations[12] + 1.5*concentrations[13] + 3.5*concentrations[14] + 2.0*concentrations[0] + 6.0*concentrations[5] + 6.0*concentrations[3] + concentrations[1] + concentrations[2] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])
        k_fwd[45] *= (3.0*concentrations[18] + 3.0*concentrations[21] + 3.0*concentrations[23] + 2.0*concentrations[12] + 1.5*concentrations[13] + 2.0*concentrations[14] + 2.0*concentrations[0] + 6.0*concentrations[5] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[19] + concentrations[20] + concentrations[22] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])
        k_fwd[120] *= (concentrations[0] + concentrations[1] + concentrations[2] + concentrations[3] + concentrations[4] + concentrations[5] + concentrations[6] + concentrations[7] + concentrations[8] + concentrations[9] + concentrations[10] + concentrations[11] + concentrations[12] + concentrations[13] + concentrations[14] + concentrations[15] + concentrations[16] + concentrations[17] + concentrations[18] + concentrations[19] + concentrations[20] + concentrations[21] + concentrations[22] + concentrations[23] + concentrations[24] + concentrations[25] + concentrations[26] + concentrations[27] + concentrations[28] + concentrations[29] + concentrations[30] + concentrations[31])
        return self._pyro_make_array(k_fwd)

    def get_net_rates_of_progress(self, temperature, concentrations):
        k_fwd = self.get_fwd_rate_coefficients(temperature, concentrations)
        log_k_eq = self.get_equilibrium_constants(temperature)
        return self._pyro_make_array([
                    k_fwd[0]*(concentrations[1]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[0])*concentrations[2]*concentrations[4]),
                    k_fwd[1]*(concentrations[0]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[1])*concentrations[1]*concentrations[4]),
                    k_fwd[2]*(concentrations[0]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[2])*concentrations[1]*concentrations[5]),
                    k_fwd[3]*(concentrations[4]**2.0 + -1*self.usr_np.exp(log_k_eq[3])*concentrations[5]*concentrations[2]),
                    k_fwd[4]*(concentrations[1]**2.0 + -1*self.usr_np.exp(log_k_eq[4])*concentrations[0]),
                    k_fwd[5]*(concentrations[1]**2.0 + -1*self.usr_np.exp(log_k_eq[5])*concentrations[0]),
                    k_fwd[6]*(concentrations[1]**2.0 + -1*self.usr_np.exp(log_k_eq[6])*concentrations[0]),
                    k_fwd[7]*(concentrations[1]**2.0 + -1*self.usr_np.exp(log_k_eq[7])*concentrations[0]),
                    k_fwd[8]*(concentrations[1]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[8])*concentrations[5]),
                    k_fwd[9]*(concentrations[1]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[9])*concentrations[4]),
                    k_fwd[10]*(concentrations[2]**2.0 + -1*self.usr_np.exp(log_k_eq[10])*concentrations[3]),
                    k_fwd[11]*(concentrations[1]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[11])*concentrations[6]),
                    k_fwd[12]*(concentrations[1]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[12])*concentrations[6]),
                    k_fwd[13]*(concentrations[1]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[13])*concentrations[6]),
                    k_fwd[14]*(concentrations[1]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[14])*concentrations[6]),
                    k_fwd[15]*(concentrations[4]**2.0 + -1*self.usr_np.exp(log_k_eq[15])*concentrations[7]),
                    k_fwd[16]*(concentrations[1]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[16])*concentrations[5]*concentrations[2]),
                    k_fwd[17]*(concentrations[1]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[17])*concentrations[0]*concentrations[3]),
                    k_fwd[18]*(concentrations[1]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[18])*concentrations[4]**2.0),
                    k_fwd[19]*(concentrations[6]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[19])*concentrations[3]*concentrations[4]),
                    k_fwd[20]*(concentrations[6]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[20])*concentrations[5]*concentrations[3]),
                    k_fwd[21]*(concentrations[6]**2.0 + -1*self.usr_np.exp(log_k_eq[21])*concentrations[7]*concentrations[3]),
                    k_fwd[22]*(concentrations[6]**2.0 + -1*self.usr_np.exp(log_k_eq[22])*concentrations[7]*concentrations[3]),
                    k_fwd[23]*(concentrations[1]*concentrations[7] + -1*self.usr_np.exp(log_k_eq[23])*concentrations[0]*concentrations[6]),
                    k_fwd[24]*(concentrations[1]*concentrations[7] + -1*self.usr_np.exp(log_k_eq[24])*concentrations[5]*concentrations[4]),
                    k_fwd[25]*(concentrations[7]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[25])*concentrations[6]*concentrations[4]),
                    k_fwd[26]*(concentrations[7]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[26])*concentrations[5]*concentrations[6]),
                    k_fwd[27]*(concentrations[7]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[27])*concentrations[5]*concentrations[6]),
                    k_fwd[28]*(concentrations[13]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[28])*concentrations[14]),
                    k_fwd[29]*(concentrations[13]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[29])*concentrations[14]*concentrations[1]),
                    k_fwd[30]*(concentrations[13]*concentrations[0] + -1*self.usr_np.exp(log_k_eq[30])*concentrations[16]),
                    k_fwd[31]*(concentrations[13]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[31])*concentrations[14]*concentrations[2]),
                    k_fwd[32]*(concentrations[13]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[32])*concentrations[14]*concentrations[4]),
                    k_fwd[33]*(concentrations[8]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[33])*concentrations[13]*concentrations[1]),
                    k_fwd[34]*(concentrations[8]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[34])*concentrations[1]*concentrations[15]),
                    k_fwd[35]*(concentrations[8]*concentrations[0] + -1*self.usr_np.exp(log_k_eq[35])*concentrations[9]*concentrations[1]),
                    k_fwd[36]*(concentrations[8]*concentrations[5] + -1*self.usr_np.exp(log_k_eq[36])*concentrations[16]*concentrations[1]),
                    k_fwd[37]*(concentrations[8]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[37])*concentrations[15]*concentrations[2]),
                    k_fwd[38]*(concentrations[8]*concentrations[13] + -1*self.usr_np.exp(log_k_eq[38])*concentrations[24]),
                    k_fwd[39]*(concentrations[8]*concentrations[14] + -1*self.usr_np.exp(log_k_eq[39])*concentrations[13]*concentrations[15]),
                    k_fwd[40]*(concentrations[1]*concentrations[15] + -1*self.usr_np.exp(log_k_eq[40])*concentrations[16]),
                    k_fwd[41]*(concentrations[1]*concentrations[15] + -1*self.usr_np.exp(log_k_eq[41])*concentrations[13]*concentrations[0]),
                    k_fwd[42]*(concentrations[15]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[42])*concentrations[13]*concentrations[4]),
                    k_fwd[43]*(concentrations[15]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[43])*concentrations[14]*concentrations[1]),
                    k_fwd[44]*(concentrations[15]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[44])*concentrations[13]*concentrations[5]),
                    k_fwd[45]*(concentrations[15] + -1*self.usr_np.exp(log_k_eq[45])*concentrations[13]*concentrations[1]),
                    k_fwd[46]*(concentrations[15]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[46])*concentrations[13]*concentrations[6]),
                    k_fwd[47]*(concentrations[9]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[47])*concentrations[11]),
                    k_fwd[48]*(concentrations[9]*concentrations[0] + -1*self.usr_np.exp(log_k_eq[48])*concentrations[11]*concentrations[1]),
                    k_fwd[49]*(concentrations[9]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[49])*concentrations[1]*concentrations[15]),
                    k_fwd[50]*(concentrations[9]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[50])*concentrations[15]*concentrations[4]),
                    k_fwd[51]*(concentrations[9]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[51])*concentrations[14]*concentrations[1]**2.0),
                    k_fwd[52]*(concentrations[9]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[52])*concentrations[16]*concentrations[1]),
                    k_fwd[53]*(concentrations[9]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[53])*concentrations[8]*concentrations[5]),
                    k_fwd[54]*(concentrations[9]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[54])*concentrations[16]*concentrations[4]),
                    k_fwd[55]*(concentrations[9]*concentrations[13] + -1*self.usr_np.exp(log_k_eq[55])*concentrations[25]),
                    k_fwd[56]*(concentrations[8]*concentrations[9] + -1*self.usr_np.exp(log_k_eq[56])*concentrations[18]*concentrations[1]),
                    k_fwd[57]*(concentrations[9]**2.0 + -1*self.usr_np.exp(log_k_eq[57])*concentrations[18]*concentrations[0]),
                    k_fwd[58]*(concentrations[10]*concentrations[31] + -1*self.usr_np.exp(log_k_eq[58])*concentrations[9]*concentrations[31]),
                    k_fwd[59]*(concentrations[10]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[59])*concentrations[8]*concentrations[0]),
                    k_fwd[60]*(concentrations[10]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[60])*concentrations[13]*concentrations[0]),
                    k_fwd[61]*(concentrations[10]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[61])*concentrations[1]*concentrations[15]),
                    k_fwd[62]*(concentrations[10]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[62])*concentrations[16]*concentrations[1]),
                    k_fwd[63]*(concentrations[10]*concentrations[0] + -1*self.usr_np.exp(log_k_eq[63])*concentrations[11]*concentrations[1]),
                    k_fwd[64]*(concentrations[10]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[64])*concentrations[13]*concentrations[1]*concentrations[4]),
                    k_fwd[65]*(concentrations[10]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[65])*concentrations[13]*concentrations[5]),
                    k_fwd[66]*(concentrations[10]*concentrations[5] + -1*self.usr_np.exp(log_k_eq[66])*concentrations[9]*concentrations[5]),
                    k_fwd[67]*(concentrations[10]*concentrations[13] + -1*self.usr_np.exp(log_k_eq[67])*concentrations[9]*concentrations[13]),
                    k_fwd[68]*(concentrations[10]*concentrations[14] + -1*self.usr_np.exp(log_k_eq[68])*concentrations[9]*concentrations[14]),
                    k_fwd[69]*(concentrations[10]*concentrations[14] + -1*self.usr_np.exp(log_k_eq[69])*concentrations[16]*concentrations[13]),
                    k_fwd[70]*(concentrations[16]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[70])*concentrations[17]),
                    k_fwd[71]*(concentrations[16]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[71])*concentrations[0]*concentrations[15]),
                    k_fwd[72]*(concentrations[16]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[72])*concentrations[15]*concentrations[4]),
                    k_fwd[73]*(concentrations[16]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[73])*concentrations[5]*concentrations[15]),
                    k_fwd[74]*(concentrations[16]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[74])*concentrations[15]*concentrations[6]),
                    k_fwd[75]*(concentrations[16]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[75])*concentrations[7]*concentrations[15]),
                    k_fwd[76]*(concentrations[8]*concentrations[16] + -1*self.usr_np.exp(log_k_eq[76])*concentrations[25]*concentrations[1]),
                    k_fwd[77]*(concentrations[11]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[77])*concentrations[12]),
                    k_fwd[78]*(concentrations[11]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[78])*concentrations[16]*concentrations[1]),
                    k_fwd[79]*(concentrations[11]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[79])*concentrations[9]*concentrations[5]),
                    k_fwd[80]*(concentrations[11]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[80])*concentrations[10]*concentrations[5]),
                    k_fwd[81]*(concentrations[11]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[81])*concentrations[17]*concentrations[2]),
                    k_fwd[82]*(concentrations[11]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[82])*concentrations[16]*concentrations[4]),
                    k_fwd[83]*(concentrations[11]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[83])*concentrations[12]*concentrations[3]),
                    k_fwd[84]*(concentrations[11]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[84])*concentrations[17]*concentrations[4]),
                    k_fwd[85]*(concentrations[11]*concentrations[7] + -1*self.usr_np.exp(log_k_eq[85])*concentrations[12]*concentrations[6]),
                    k_fwd[86]*(concentrations[8]*concentrations[11] + -1*self.usr_np.exp(log_k_eq[86])*concentrations[20]*concentrations[1]),
                    k_fwd[87]*(concentrations[11]*concentrations[15] + -1*self.usr_np.exp(log_k_eq[87])*concentrations[12]*concentrations[13]),
                    k_fwd[88]*(concentrations[11]*concentrations[15] + -1*self.usr_np.exp(log_k_eq[88])*concentrations[27]),
                    k_fwd[89]*(concentrations[16]*concentrations[11] + -1*self.usr_np.exp(log_k_eq[89])*concentrations[12]*concentrations[15]),
                    k_fwd[90]*(concentrations[9]*concentrations[11] + -1*self.usr_np.exp(log_k_eq[90])*concentrations[21]*concentrations[1]),
                    k_fwd[91]*(concentrations[10]*concentrations[11] + -1*self.usr_np.exp(log_k_eq[91])*concentrations[21]*concentrations[1]),
                    k_fwd[92]*(concentrations[11]**2.0 + -1*self.usr_np.exp(log_k_eq[92])*concentrations[23]),
                    k_fwd[93]*(concentrations[11]**2.0 + -1*self.usr_np.exp(log_k_eq[93])*concentrations[22]*concentrations[1]),
                    k_fwd[94]*(concentrations[11]*concentrations[24] + -1*self.usr_np.exp(log_k_eq[94])*concentrations[21]*concentrations[13]),
                    k_fwd[95]*(concentrations[17]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[95])*concentrations[16]*concentrations[0]),
                    k_fwd[96]*(concentrations[17]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[96])*concentrations[11]*concentrations[4]),
                    k_fwd[97]*(concentrations[17]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[97])*concentrations[10]*concentrations[5]),
                    k_fwd[98]*(concentrations[17]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[98])*concentrations[16]*concentrations[4]),
                    k_fwd[99]*(concentrations[17]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[99])*concentrations[16]*concentrations[5]),
                    k_fwd[100]*(concentrations[17]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[100])*concentrations[16]*concentrations[6]),
                    k_fwd[101]*(concentrations[12]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[101])*concentrations[11]*concentrations[0]),
                    k_fwd[102]*(concentrations[12]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[102])*concentrations[11]*concentrations[4]),
                    k_fwd[103]*(concentrations[12]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[103])*concentrations[11]*concentrations[5]),
                    k_fwd[104]*(concentrations[8]*concentrations[12] + -1*self.usr_np.exp(log_k_eq[104])*concentrations[21]*concentrations[1]),
                    k_fwd[105]*(concentrations[9]*concentrations[12] + -1*self.usr_np.exp(log_k_eq[105])*concentrations[11]**2.0),
                    k_fwd[106]*(concentrations[10]*concentrations[12] + -1*self.usr_np.exp(log_k_eq[106])*concentrations[11]**2.0),
                    k_fwd[107]*(concentrations[1]*concentrations[24] + -1*self.usr_np.exp(log_k_eq[107])*concentrations[10]*concentrations[13]),
                    k_fwd[108]*(concentrations[24]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[108])*concentrations[13]**2.0*concentrations[1]),
                    k_fwd[109]*(concentrations[24]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[109])*concentrations[13]**2.0*concentrations[4]),
                    k_fwd[110]*(concentrations[8]*concentrations[24] + -1*self.usr_np.exp(log_k_eq[110])*concentrations[18]*concentrations[13]),
                    k_fwd[111]*(concentrations[9]*concentrations[24] + -1*self.usr_np.exp(log_k_eq[111])*concentrations[20]*concentrations[13]),
                    k_fwd[112]*(concentrations[24]**2.0 + -1*self.usr_np.exp(log_k_eq[112])*concentrations[18]*concentrations[13]**2.0),
                    k_fwd[113]*(concentrations[18] + -1*self.usr_np.exp(log_k_eq[113])*concentrations[19]),
                    k_fwd[114]*(concentrations[20] + -1*self.usr_np.exp(log_k_eq[114])*concentrations[18]*concentrations[1]),
                    k_fwd[115]*(concentrations[18]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[115])*concentrations[1]*concentrations[24]),
                    k_fwd[116]*(concentrations[18]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[116])*concentrations[9]*concentrations[13]),
                    k_fwd[117]*(concentrations[18]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[117])*concentrations[25]*concentrations[1]),
                    k_fwd[118]*(concentrations[18]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[118])*concentrations[11]*concentrations[13]),
                    k_fwd[119]*(concentrations[18]*concentrations[15] + -1*self.usr_np.exp(log_k_eq[119])*concentrations[20]*concentrations[13]),
                    k_fwd[120]*(concentrations[18]*concentrations[11] + -1*self.usr_np.exp(log_k_eq[120])*concentrations[28]),
                    k_fwd[121]*(concentrations[1]*concentrations[19] + -1*self.usr_np.exp(log_k_eq[121])*concentrations[18]*concentrations[1]),
                    k_fwd[122]*(concentrations[19]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[122])*concentrations[9]*concentrations[13]),
                    k_fwd[123]*(concentrations[19]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[123])*concentrations[25]*concentrations[1]),
                    k_fwd[124]*(concentrations[19]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[124])*concentrations[9]*concentrations[14]),
                    k_fwd[125]*(concentrations[25]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[125])*concentrations[26]),
                    k_fwd[126]*(concentrations[25]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[126])*concentrations[0]*concentrations[24]),
                    k_fwd[127]*(concentrations[25]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[127])*concentrations[11]*concentrations[13]),
                    k_fwd[128]*(concentrations[25]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[128])*concentrations[24]*concentrations[4]),
                    k_fwd[129]*(concentrations[25]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[129])*concentrations[9]*concentrations[14]),
                    k_fwd[130]*(concentrations[25]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[130])*concentrations[5]*concentrations[24]),
                    k_fwd[131]*(concentrations[20]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[131])*concentrations[21]),
                    k_fwd[132]*(concentrations[20]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[132])*concentrations[18]*concentrations[0]),
                    k_fwd[133]*(concentrations[20]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[133])*concentrations[0]*concentrations[19]),
                    k_fwd[134]*(concentrations[20]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[134])*concentrations[25]*concentrations[1]),
                    k_fwd[135]*(concentrations[20]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[135])*concentrations[11]*concentrations[13]),
                    k_fwd[136]*(concentrations[20]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[136])*concentrations[18]*concentrations[5]),
                    k_fwd[137]*(concentrations[20]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[137])*concentrations[18]*concentrations[6]),
                    k_fwd[138]*(concentrations[20]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[138])*concentrations[26]*concentrations[2]),
                    k_fwd[139]*(concentrations[20]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[139])*concentrations[16]*concentrations[15]),
                    k_fwd[140]*(concentrations[20]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[140])*concentrations[26]*concentrations[4]),
                    k_fwd[141]*(concentrations[20]*concentrations[7] + -1*self.usr_np.exp(log_k_eq[141])*concentrations[21]*concentrations[6]),
                    k_fwd[142]*(concentrations[20]*concentrations[15] + -1*self.usr_np.exp(log_k_eq[142])*concentrations[21]*concentrations[13]),
                    k_fwd[143]*(concentrations[20]*concentrations[11] + -1*self.usr_np.exp(log_k_eq[143])*concentrations[18]*concentrations[12]),
                    k_fwd[144]*(concentrations[20]*concentrations[11] + -1*self.usr_np.exp(log_k_eq[144])*concentrations[29]),
                    k_fwd[145]*(concentrations[20]*concentrations[11] + -1*self.usr_np.exp(log_k_eq[145])*concentrations[1]*concentrations[28]),
                    k_fwd[146]*(concentrations[26] + -1*self.usr_np.exp(log_k_eq[146])*concentrations[11]*concentrations[13]),
                    k_fwd[147]*(concentrations[26]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[147])*concentrations[27]),
                    k_fwd[148]*(concentrations[26]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[148])*concentrations[11]*concentrations[15]),
                    k_fwd[149]*(concentrations[26]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[149])*concentrations[25]*concentrations[0]),
                    k_fwd[150]*(concentrations[26]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[150])*concentrations[25]*concentrations[4]),
                    k_fwd[151]*(concentrations[26]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[151])*concentrations[25]*concentrations[5]),
                    k_fwd[152]*(concentrations[26]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[152])*concentrations[25]*concentrations[6]),
                    k_fwd[153]*(concentrations[26]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[153])*concentrations[16]*concentrations[13]*concentrations[4]),
                    k_fwd[154]*(concentrations[21] + -1*self.usr_np.exp(log_k_eq[154])*concentrations[0]*concentrations[19]),
                    k_fwd[155]*(concentrations[21]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[155])*concentrations[22]),
                    k_fwd[156]*(concentrations[21]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[156])*concentrations[20]*concentrations[0]),
                    k_fwd[157]*(concentrations[21]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[157])*concentrations[20]*concentrations[4]),
                    k_fwd[158]*(concentrations[21]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[158])*concentrations[11]*concentrations[15]),
                    k_fwd[159]*(concentrations[21]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[159])*concentrations[9]*concentrations[16]),
                    k_fwd[160]*(concentrations[21]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[160])*concentrations[20]*concentrations[5]),
                    k_fwd[161]*(concentrations[21]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[161])*concentrations[20]*concentrations[6]),
                    k_fwd[162]*(concentrations[21]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[162])*concentrations[27]*concentrations[4]),
                    k_fwd[163]*(concentrations[21]*concentrations[15] + -1*self.usr_np.exp(log_k_eq[163])*concentrations[22]*concentrations[13]),
                    k_fwd[164]*(concentrations[21]*concentrations[9] + -1*self.usr_np.exp(log_k_eq[164])*concentrations[1]*concentrations[28]),
                    k_fwd[165]*(concentrations[21]*concentrations[10] + -1*self.usr_np.exp(log_k_eq[165])*concentrations[12]*concentrations[19]),
                    k_fwd[166]*(concentrations[21]*concentrations[10] + -1*self.usr_np.exp(log_k_eq[166])*concentrations[1]*concentrations[28]),
                    k_fwd[167]*(concentrations[21]*concentrations[11] + -1*self.usr_np.exp(log_k_eq[167])*concentrations[20]*concentrations[12]),
                    k_fwd[168]*(concentrations[21]*concentrations[11] + -1*self.usr_np.exp(log_k_eq[168])*concentrations[30]),
                    k_fwd[169]*(concentrations[22]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[169])*concentrations[23]),
                    k_fwd[170]*(concentrations[22]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[170])*concentrations[21]*concentrations[0]),
                    k_fwd[171]*(concentrations[22]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[171])*concentrations[16]*concentrations[11]),
                    k_fwd[172]*(concentrations[22]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[172])*concentrations[27]*concentrations[1]),
                    k_fwd[173]*(concentrations[22]*concentrations[3] + -1*self.usr_np.exp(log_k_eq[173])*concentrations[21]*concentrations[6]),
                    k_fwd[174]*(concentrations[22]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[174])*concentrations[23]*concentrations[3]),
                    k_fwd[175]*(concentrations[22]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[175])*concentrations[21]*concentrations[7]),
                    k_fwd[176]*(concentrations[22]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[176])*concentrations[16]*concentrations[11]*concentrations[4]),
                    k_fwd[177]*(concentrations[22]*concentrations[7] + -1*self.usr_np.exp(log_k_eq[177])*concentrations[23]*concentrations[6]),
                    k_fwd[178]*(concentrations[22]*concentrations[15] + -1*self.usr_np.exp(log_k_eq[178])*concentrations[23]*concentrations[13]),
                    k_fwd[179]*(concentrations[23]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[179])*concentrations[22]*concentrations[0]),
                    k_fwd[180]*(concentrations[23]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[180])*concentrations[22]*concentrations[4]),
                    k_fwd[181]*(concentrations[23]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[181])*concentrations[22]*concentrations[5]),
                    k_fwd[182]*(concentrations[23]*concentrations[10] + -1*self.usr_np.exp(log_k_eq[182])*concentrations[22]*concentrations[11]),
                    k_fwd[183]*(concentrations[23]*concentrations[11] + -1*self.usr_np.exp(log_k_eq[183])*concentrations[22]*concentrations[12]),
                    k_fwd[184]*(concentrations[1]*concentrations[28] + -1*self.usr_np.exp(log_k_eq[184])*concentrations[29]),
                    k_fwd[185]*(concentrations[1]*concentrations[28] + -1*self.usr_np.exp(log_k_eq[185])*concentrations[12]*concentrations[19]),
                    k_fwd[186]*(concentrations[6]*concentrations[28] + -1*self.usr_np.exp(log_k_eq[186])*concentrations[29]*concentrations[3]),
                    k_fwd[187]*(concentrations[6]*concentrations[28] + -1*self.usr_np.exp(log_k_eq[187])*concentrations[20]*concentrations[16]*concentrations[4]),
                    k_fwd[188]*(concentrations[15]*concentrations[28] + -1*self.usr_np.exp(log_k_eq[188])*concentrations[29]*concentrations[13]),
                    k_fwd[189]*(concentrations[29]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[189])*concentrations[30]),
                    k_fwd[190]*(concentrations[29]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[190])*concentrations[21]*concentrations[11]),
                    k_fwd[191]*(concentrations[29]*concentrations[1] + -1*self.usr_np.exp(log_k_eq[191])*concentrations[0]*concentrations[28]),
                    k_fwd[192]*(concentrations[29]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[192])*concentrations[25]*concentrations[11]*concentrations[1]),
                    k_fwd[193]*(concentrations[29]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[193])*concentrations[22]*concentrations[15]),
                    k_fwd[194]*(concentrations[29]*concentrations[2] + -1*self.usr_np.exp(log_k_eq[194])*concentrations[4]*concentrations[28]),
                    k_fwd[195]*(concentrations[29]*concentrations[4] + -1*self.usr_np.exp(log_k_eq[195])*concentrations[5]*concentrations[28]),
                    k_fwd[196]*(concentrations[29]*concentrations[6] + -1*self.usr_np.exp(log_k_eq[196])*concentrations[7]*concentrations[28]),
                    k_fwd[197]*(concentrations[29]*concentrations[11] + -1*self.usr_np.exp(log_k_eq[197])*concentrations[12]*concentrations[28]),
                    k_fwd[198]*(concentrations[1]*concentrations[30] + -1*self.usr_np.exp(log_k_eq[198])*concentrations[22]*concentrations[11]),
                    k_fwd[199]*(concentrations[1]*concentrations[30] + -1*self.usr_np.exp(log_k_eq[199])*concentrations[29]*concentrations[0]),
                    k_fwd[200]*(concentrations[2]*concentrations[30] + -1*self.usr_np.exp(log_k_eq[200])*concentrations[22]*concentrations[16]),
                    k_fwd[201]*(concentrations[4]*concentrations[30] + -1*self.usr_np.exp(log_k_eq[201])*concentrations[29]*concentrations[5]),
                    k_fwd[202]*(concentrations[3]*concentrations[30] + -1*self.usr_np.exp(log_k_eq[202])*concentrations[29]*concentrations[6]),
                    k_fwd[203]*(concentrations[6]*concentrations[30] + -1*self.usr_np.exp(log_k_eq[203])*concentrations[22]*concentrations[16]*concentrations[4]),
                    k_fwd[204]*(concentrations[11]*concentrations[30] + -1*self.usr_np.exp(log_k_eq[204])*concentrations[29]*concentrations[12]),
                    k_fwd[205]*(concentrations[20]*concentrations[22] + -1*self.usr_np.exp(log_k_eq[205])*concentrations[11]*concentrations[28]),
               ])

    def get_net_production_rates(self, rho, temperature, mass_fractions):
        c = self.get_concentrations(rho, mass_fractions)
        r_net = self.get_net_rates_of_progress(temperature, c)
        ones = self._pyro_zeros_like(r_net[0]) + 1.0
        return self._pyro_make_array([
                r_net[4] + r_net[5] + r_net[6] + r_net[7] + r_net[17] + r_net[23] + r_net[41] + r_net[57] + r_net[59] + r_net[60] + r_net[71] + r_net[95] + r_net[101] + r_net[126] + r_net[132] + r_net[133] + r_net[149] + r_net[154] + r_net[156] + r_net[170] + r_net[179] + r_net[191] + r_net[199] + -1*(r_net[1] + r_net[2] + r_net[30] + r_net[35] + r_net[48] + r_net[63]) * ones,
                r_net[1] + r_net[2] + r_net[29] + r_net[33] + r_net[34] + r_net[35] + r_net[36] + r_net[43] + r_net[45] + r_net[48] + r_net[49] + 2.0*r_net[51] + r_net[52] + r_net[56] + r_net[61] + r_net[62] + r_net[63] + r_net[64] + r_net[76] + r_net[78] + r_net[86] + r_net[90] + r_net[91] + r_net[93] + r_net[104] + r_net[108] + r_net[114] + r_net[115] + r_net[117] + r_net[121] + r_net[123] + r_net[134] + r_net[145] + r_net[164] + r_net[166] + r_net[172] + r_net[192] + -1*(r_net[0] + 2.0*r_net[4] + 2.0*r_net[5] + 2.0*r_net[6] + 2.0*r_net[7] + r_net[8] + r_net[9] + r_net[11] + r_net[12] + r_net[13] + r_net[14] + r_net[16] + r_net[17] + r_net[18] + r_net[23] + r_net[24] + r_net[40] + r_net[41] + r_net[47] + r_net[59] + r_net[70] + r_net[71] + r_net[77] + r_net[95] + r_net[96] + r_net[97] + r_net[101] + r_net[107] + r_net[121] + r_net[125] + r_net[126] + r_net[127] + r_net[131] + r_net[132] + r_net[133] + r_net[147] + r_net[148] + r_net[149] + r_net[155] + r_net[156] + r_net[169] + r_net[170] + r_net[179] + r_net[184] + r_net[185] + r_net[189] + r_net[190] + r_net[191] + r_net[198] + r_net[199]) * ones,
                r_net[0] + r_net[3] + r_net[16] + r_net[31] + r_net[37] + r_net[81] + r_net[138] + -1*(r_net[1] + r_net[9] + 2.0*r_net[10] + r_net[19] + r_net[25] + r_net[28] + r_net[33] + r_net[42] + r_net[43] + r_net[49] + r_net[60] + r_net[61] + r_net[72] + r_net[78] + r_net[98] + r_net[102] + r_net[108] + r_net[115] + r_net[116] + r_net[122] + r_net[128] + r_net[129] + r_net[134] + r_net[135] + r_net[150] + r_net[157] + r_net[158] + r_net[159] + r_net[171] + r_net[172] + r_net[180] + r_net[192] + r_net[193] + r_net[194] + r_net[200]) * ones,
                r_net[10] + r_net[17] + r_net[19] + r_net[20] + r_net[21] + r_net[22] + r_net[83] + r_net[174] + r_net[186] + -1*(r_net[0] + r_net[11] + r_net[12] + r_net[13] + r_net[14] + r_net[31] + r_net[37] + r_net[46] + r_net[50] + r_net[51] + r_net[64] + r_net[65] + r_net[74] + r_net[81] + r_net[82] + r_net[100] + r_net[109] + r_net[124] + r_net[137] + r_net[138] + r_net[139] + r_net[152] + r_net[153] + r_net[161] + r_net[173] + r_net[202]) * ones,
                r_net[0] + r_net[1] + r_net[9] + 2.0*r_net[18] + r_net[19] + r_net[24] + r_net[25] + r_net[32] + r_net[42] + r_net[50] + r_net[54] + r_net[64] + r_net[72] + r_net[82] + r_net[84] + r_net[96] + r_net[98] + r_net[102] + r_net[109] + r_net[128] + r_net[140] + r_net[150] + r_net[153] + r_net[157] + r_net[162] + r_net[176] + r_net[180] + r_net[187] + r_net[194] + r_net[203] + -1*(r_net[2] + 2.0*r_net[3] + r_net[8] + 2.0*r_net[15] + r_net[20] + r_net[26] + r_net[27] + r_net[29] + r_net[34] + r_net[44] + r_net[52] + r_net[53] + r_net[62] + r_net[73] + r_net[79] + r_net[80] + r_net[99] + r_net[103] + r_net[117] + r_net[118] + r_net[123] + r_net[130] + r_net[136] + r_net[151] + r_net[160] + r_net[181] + r_net[195] + r_net[201]) * ones,
                r_net[2] + r_net[3] + r_net[8] + r_net[16] + r_net[20] + r_net[24] + r_net[26] + r_net[27] + r_net[44] + r_net[53] + r_net[65] + r_net[66] + r_net[73] + r_net[79] + r_net[80] + r_net[97] + r_net[99] + r_net[103] + r_net[130] + r_net[136] + r_net[151] + r_net[160] + r_net[181] + r_net[195] + r_net[201] + -1*(r_net[36] + r_net[66]) * ones,
                r_net[11] + r_net[12] + r_net[13] + r_net[14] + r_net[23] + r_net[25] + r_net[26] + r_net[27] + r_net[46] + r_net[74] + r_net[85] + r_net[100] + r_net[137] + r_net[141] + r_net[152] + r_net[161] + r_net[173] + r_net[177] + r_net[202] + -1*(r_net[16] + r_net[17] + r_net[18] + r_net[19] + r_net[20] + 2.0*r_net[21] + 2.0*r_net[22] + r_net[32] + r_net[54] + r_net[75] + r_net[83] + r_net[84] + r_net[140] + r_net[162] + r_net[174] + r_net[175] + r_net[176] + r_net[186] + r_net[187] + r_net[196] + r_net[203]) * ones,
                r_net[15] + r_net[21] + r_net[22] + r_net[75] + r_net[175] + r_net[196] + -1*(r_net[23] + r_net[24] + r_net[25] + r_net[26] + r_net[27] + r_net[85] + r_net[141] + r_net[177]) * ones,
                r_net[53] + r_net[59] + -1*(r_net[33] + r_net[34] + r_net[35] + r_net[36] + r_net[37] + r_net[38] + r_net[39] + r_net[56] + r_net[76] + r_net[86] + r_net[104] + r_net[110]) * ones,
                r_net[35] + r_net[58] + r_net[66] + r_net[67] + r_net[68] + r_net[79] + r_net[116] + r_net[122] + r_net[124] + r_net[129] + r_net[159] + -1*(r_net[47] + r_net[48] + r_net[49] + r_net[50] + r_net[51] + r_net[52] + r_net[53] + r_net[54] + r_net[55] + r_net[56] + 2.0*r_net[57] + r_net[90] + r_net[105] + r_net[111] + r_net[164]) * ones,
                r_net[80] + r_net[97] + r_net[107] + -1*(r_net[58] + r_net[59] + r_net[60] + r_net[61] + r_net[62] + r_net[63] + r_net[64] + r_net[65] + r_net[66] + r_net[67] + r_net[68] + r_net[69] + r_net[91] + r_net[106] + r_net[165] + r_net[166] + r_net[182]) * ones,
                r_net[47] + r_net[48] + r_net[63] + r_net[96] + r_net[101] + r_net[102] + r_net[103] + 2.0*r_net[105] + 2.0*r_net[106] + r_net[118] + r_net[127] + r_net[135] + r_net[146] + r_net[148] + r_net[158] + r_net[171] + r_net[176] + r_net[182] + r_net[190] + r_net[192] + r_net[198] + r_net[205] + -1*(r_net[77] + r_net[78] + r_net[79] + r_net[80] + r_net[81] + r_net[82] + r_net[83] + r_net[84] + r_net[85] + r_net[86] + r_net[87] + r_net[88] + r_net[89] + r_net[90] + r_net[91] + 2.0*r_net[92] + 2.0*r_net[93] + r_net[94] + r_net[120] + r_net[143] + r_net[144] + r_net[145] + r_net[167] + r_net[168] + r_net[183] + r_net[197] + r_net[204]) * ones,
                r_net[77] + r_net[83] + r_net[85] + r_net[87] + r_net[89] + r_net[143] + r_net[165] + r_net[167] + r_net[183] + r_net[185] + r_net[197] + r_net[204] + -1*(r_net[101] + r_net[102] + r_net[103] + r_net[104] + r_net[105] + r_net[106]) * ones,
                r_net[33] + r_net[39] + r_net[41] + r_net[42] + r_net[44] + r_net[45] + r_net[46] + r_net[60] + r_net[64] + r_net[65] + r_net[67] + r_net[69] + r_net[87] + r_net[94] + r_net[107] + 2.0*r_net[108] + 2.0*r_net[109] + r_net[110] + r_net[111] + 2.0*r_net[112] + r_net[116] + r_net[118] + r_net[119] + r_net[122] + r_net[127] + r_net[135] + r_net[142] + r_net[146] + r_net[153] + r_net[163] + r_net[178] + r_net[188] + -1*(r_net[28] + r_net[29] + r_net[30] + r_net[31] + r_net[32] + r_net[38] + r_net[55] + r_net[67]) * ones,
                r_net[28] + r_net[29] + r_net[31] + r_net[32] + r_net[43] + r_net[51] + r_net[68] + r_net[124] + r_net[129] + -1*(r_net[39] + r_net[68] + r_net[69]) * ones,
                r_net[34] + r_net[37] + r_net[39] + r_net[49] + r_net[50] + r_net[61] + r_net[71] + r_net[72] + r_net[73] + r_net[74] + r_net[75] + r_net[89] + r_net[139] + r_net[148] + r_net[158] + r_net[193] + -1*(r_net[40] + r_net[41] + r_net[42] + r_net[43] + r_net[44] + r_net[45] + r_net[46] + r_net[87] + r_net[88] + r_net[119] + r_net[142] + r_net[163] + r_net[178] + r_net[188]) * ones,
                r_net[30] + r_net[36] + r_net[40] + r_net[52] + r_net[54] + r_net[62] + r_net[69] + r_net[78] + r_net[82] + r_net[95] + r_net[98] + r_net[99] + r_net[100] + r_net[139] + r_net[153] + r_net[159] + r_net[171] + r_net[176] + r_net[187] + r_net[200] + r_net[203] + -1*(r_net[70] + r_net[71] + r_net[72] + r_net[73] + r_net[74] + r_net[75] + r_net[76] + r_net[89]) * ones,
                r_net[70] + r_net[81] + r_net[84] + -1*(r_net[95] + r_net[96] + r_net[97] + r_net[98] + r_net[99] + r_net[100]) * ones,
                r_net[56] + r_net[57] + r_net[110] + r_net[112] + r_net[114] + r_net[121] + r_net[132] + r_net[136] + r_net[137] + r_net[143] + -1*(r_net[113] + r_net[115] + r_net[116] + r_net[117] + r_net[118] + r_net[119] + r_net[120]) * ones,
                r_net[113] + r_net[133] + r_net[154] + r_net[165] + r_net[185] + -1*(r_net[121] + r_net[122] + r_net[123] + r_net[124]) * ones,
                r_net[86] + r_net[111] + r_net[119] + r_net[156] + r_net[157] + r_net[160] + r_net[161] + r_net[167] + r_net[187] + -1*(r_net[114] + r_net[131] + r_net[132] + r_net[133] + r_net[134] + r_net[135] + r_net[136] + r_net[137] + r_net[138] + r_net[139] + r_net[140] + r_net[141] + r_net[142] + r_net[143] + r_net[144] + r_net[145] + r_net[205]) * ones,
                r_net[90] + r_net[91] + r_net[94] + r_net[104] + r_net[131] + r_net[141] + r_net[142] + r_net[170] + r_net[173] + r_net[175] + r_net[190] + -1*(r_net[154] + r_net[155] + r_net[156] + r_net[157] + r_net[158] + r_net[159] + r_net[160] + r_net[161] + r_net[162] + r_net[163] + r_net[164] + r_net[165] + r_net[166] + r_net[167] + r_net[168]) * ones,
                r_net[93] + r_net[155] + r_net[163] + r_net[179] + r_net[180] + r_net[181] + r_net[182] + r_net[183] + r_net[193] + r_net[198] + r_net[200] + r_net[203] + -1*(r_net[169] + r_net[170] + r_net[171] + r_net[172] + r_net[173] + r_net[174] + r_net[175] + r_net[176] + r_net[177] + r_net[178] + r_net[205]) * ones,
                r_net[92] + r_net[169] + r_net[174] + r_net[177] + r_net[178] + -1*(r_net[179] + r_net[180] + r_net[181] + r_net[182] + r_net[183]) * ones,
                r_net[38] + r_net[115] + r_net[126] + r_net[128] + r_net[130] + -1*(r_net[94] + r_net[107] + r_net[108] + r_net[109] + r_net[110] + r_net[111] + 2.0*r_net[112]) * ones,
                r_net[55] + r_net[76] + r_net[117] + r_net[123] + r_net[134] + r_net[149] + r_net[150] + r_net[151] + r_net[152] + r_net[192] + -1*(r_net[125] + r_net[126] + r_net[127] + r_net[128] + r_net[129] + r_net[130]) * ones,
                r_net[125] + r_net[138] + r_net[140] + -1*(r_net[146] + r_net[147] + r_net[148] + r_net[149] + r_net[150] + r_net[151] + r_net[152] + r_net[153]) * ones,
                r_net[88] + r_net[147] + r_net[162] + r_net[172] * ones,
                r_net[120] + r_net[145] + r_net[164] + r_net[166] + r_net[191] + r_net[194] + r_net[195] + r_net[196] + r_net[197] + r_net[205] + -1*(r_net[184] + r_net[185] + r_net[186] + r_net[187] + r_net[188]) * ones,
                r_net[144] + r_net[184] + r_net[186] + r_net[188] + r_net[199] + r_net[201] + r_net[202] + r_net[204] + -1*(r_net[189] + r_net[190] + r_net[191] + r_net[192] + r_net[193] + r_net[194] + r_net[195] + r_net[196] + r_net[197]) * ones,
                r_net[168] + r_net[189] + -1*(r_net[198] + r_net[199] + r_net[200] + r_net[201] + r_net[202] + r_net[203] + r_net[204]) * ones,
                r_net[58] + -1*r_net[58] * ones,
               ])
