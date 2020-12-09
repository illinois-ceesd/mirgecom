from pytools.obj_array import make_obj_array
import numpy as np

class UIUCMechanism:
    def __init__(self, npctx=np):
        self.npctx = npctx
        self.model_name    = "uiuc.cti"
        self.num_elements  = 4
        self.num_species   = 7
        self.one_third = 1 / 3
        self.num_reactions = 3
        self.num_falloff   = 0
        
        self.one_atm = 101325.0
        self.gas_constant = 8314.46261815324
        self.big_number = 1.0e300

        self.species_names = ['C2H4', 'O2', 'CO2', 'CO', 'H2O', 'H2', 'N2']
        self.species_indices = {'C2H4': 0, 'O2': 1, 'CO2': 2, 'CO': 3, 'H2O': 4, 'H2': 5, 'N2': 6}

        self.wts = np.array([28.054, 31.998, 44.009, 28.009999999999998, 18.015, 2.016, 28.014])
        self.iwts = 1.0 / self.wts

    def species_name(self, species_index):
        return self.species_name[species_index]

    def species_index(self, species_name):
        return self.species_indices[species_name]

    def get_specific_gas_constant(self, Y):
        return self.gas_constant * np.dot( self.iwts, Y )

    def get_density(self, p, T, Y):
        mmw = self.get_mix_molecular_weight( Y )
        RT  = self.gas_constant * T
        return p * mmw / RT

    def get_pressure(self, rho, T, Y):
        mmw = self.get_mix_molecular_weight( Y )
        RT  = self.gas_constant * T
        return rho * RT / mmw

    def get_mix_molecular_weight(self, Y):
        return 1.0 / np.dot( self.iwts, Y )

    def get_concentrations(self, rho, Y):
        conctest = self.iwts * rho * Y
        zero = 0 * conctest[0]
        for i, conc in enumerate(conctest):
            conctest[i] = self.npctx.where(conctest[i] > 0, conctest[i], zero)
        return conctest

    def get_mixture_specific_heat_cp_mass(self, temperature, massfractions):
        cp0_r = self.get_species_specific_heats_R(temperature)
        cpsum = sum([massfractions[i] * cp0_r[i] * self.iwts[i]
                     for i in range(self.num_species)])
        return self.gas_constant * cpsum

    def get_mixture_specific_heat_cv_mass(self, temperature, massfractions):
        cp0_r = self.get_species_specific_heats_R(temperature) - 1.0
        cpsum = sum([massfractions[i] * cp0_r[i] * self.iwts[i]
                     for i in range(self.num_species)])
        return self.gas_constant * cpsum

    def get_mixture_enthalpy_mass(self, temperature, massfractions):
        h0_rt = self.get_species_enthalpies_RT(temperature)
        hsum = sum([massfractions[i] * h0_rt[i] * self.iwts[i]
                    for i in range(self.num_species)])
        return self.gas_constant * temperature * hsum

    def get_mixture_internal_energy_mass(self, temperature, massfractions):

        e0_rt = self.get_species_enthalpies_RT(temperature) - 1.0
        esum = sum([massfractions[i] * e0_rt[i] * self.iwts[i]
                    for i in range(self.num_species)])
        return self.gas_constant * temperature * esum

#    def get_mixture_internal_energy_mass(self, temperature, massfractions):
#        e0_rt = self.get_species_enthalpies_RT(temperature) - 1.0
#        mflist = sum([massfractions[i] * self.iwts[i] * e0_rt[i]
#                      for i in range(self.num_species)])
#        esum = sum([massfractions[i] * e0_rt[i] * self.iwts[i]
#                    for i in range(self.num_species)])
#        return self.gas_constant * temperature * esum

    def get_species_specific_heats_R(self, temperature):

        tt0 = temperature
        tt1 = temperature * tt0
        tt2 = temperature * tt1
        tt3 = temperature * tt2

        cp_high = (
            2.036111e00
            + 1.464542e-02 * tt0
            - 6.710779e-06 * tt1
            + 1.472229e-09 * tt2
            - 1.257061e-13 * tt3
        )
        cp_low = (
            3.959201e00
            - 7.570522e-03 * tt0
            + 5.709903e-05 * tt1
            - 6.915888e-08 * tt2
            + 2.698844e-11 * tt3
        )
        cpr0 = self.npctx.where(tt0 < 1.000000e03, cp_low, cp_high)

        cp_high = (
            3.282538e00
            + 1.483088e-03 * tt0
            - 7.579667e-07 * tt1
            + 2.094706e-10 * tt2
            - 2.167178e-14 * tt3
        )
        cp_low = (
            3.782456e00
            - 2.996734e-03 * tt0
            + 9.847302e-06 * tt1
            - 9.681295e-09 * tt2
            + 3.243728e-12 * tt3
        )
        cpr1 = self.npctx.where(tt0 < 1.000000e03, cp_low, cp_high)

        cp_high = (
            3.857460e00
            + 4.414370e-03 * tt0
            - 2.214814e-06 * tt1
            + 5.234902e-10 * tt2
            - 4.720842e-14 * tt3
        )
        cp_low = (
            2.356774e00
            + 8.984597e-03 * tt0
            - 7.123563e-06 * tt1
            + 2.459190e-09 * tt2
            - 1.436995e-13 * tt3
        )
        cpr2 = self.npctx.where(tt0 < 1.000000e03, cp_low, cp_high)

        cp_high = (
            2.715186e00
            + 2.062527e-03 * tt0
            - 9.988258e-07 * tt1
            + 2.300530e-10 * tt2
            - 2.036477e-14 * tt3
        )
        cp_low = (
            3.579533e00
            - 6.103537e-04 * tt0
            + 1.016814e-06 * tt1
            + 9.070059e-10 * tt2
            - 9.044245e-13 * tt3
        )
        cpr3 = self.npctx.where(tt0 < 1.000000e03, cp_low, cp_high)

        cp_high = (
            3.033992e00
            + 2.176918e-03 * tt0
            - 1.640725e-07 * tt1
            - 9.704199e-11 * tt2
            + 1.682010e-14 * tt3
        )
        cp_low = (
            4.198641e00
            - 2.036434e-03 * tt0
            + 6.520402e-06 * tt1
            - 5.487971e-09 * tt2
            + 1.771978e-12 * tt3
        )
        cpr4 = self.npctx.where(tt0 < 1.000000e03, cp_low, cp_high)

        cp_high = (
            3.337279e00
            - 4.940247e-05 * tt0
            + 4.994568e-07 * tt1
            - 1.795664e-10 * tt2
            + 2.002554e-14 * tt3
        )
        cp_low = (
            2.344331e00
            + 7.980521e-03 * tt0
            - 1.947815e-05 * tt1
            + 2.015721e-08 * tt2
            - 7.376118e-12 * tt3
        )
        cpr5 = self.npctx.where(tt0 < 1.000000e03, cp_low, cp_high)

        cp_high = (
            2.926640e00
            + 1.487977e-03 * tt0
            - 5.684760e-07 * tt1
            + 1.009704e-10 * tt2
            - 6.753351e-15 * tt3
        )
        cp_low = (
            3.298677e00
            + 1.408240e-03 * tt0
            - 3.963222e-06 * tt1
            + 5.641515e-09 * tt2
            - 2.444854e-12 * tt3
        )
        cpr6 = self.npctx.where(tt0 < 1.000000e03, cp_low, cp_high)

        return make_obj_array([cpr0, cpr1, cpr2, cpr3, cpr4, cpr5, cpr6])

    def get_species_enthalpies_RT(self, temperature):

        tt0 = temperature
        tt1 = temperature * tt0
        tt2 = temperature * tt1
        tt3 = temperature * tt2
        tt4 = 1.0 / temperature

        h_high = (
            2.036111e00
            + 1.464542e-02 * 0.50 * tt0
            - 6.710779e-06 * self.one_third * tt1
            + 1.472229e-09 * 0.25 * tt2
            - 1.257061e-13 * 0.20 * tt3
            + 4.939886e03 * tt4
        )
        h_low = (
            3.959201e00
            - 7.570522e-03 * 0.50 * tt0
            + 5.709903e-05 * self.one_third * tt1
            - 6.915888e-08 * 0.25 * tt2
            + 2.698844e-11 * 0.20 * tt3
            + 5.089776e03 * tt4
        )
        hrt0 = self.npctx.where(tt0 < 1.000000e03, h_low, h_high)

        h_high = (
            3.282538e00
            + 1.483088e-03 * 0.50 * tt0
            - 7.579667e-07 * self.one_third * tt1
            + 2.094706e-10 * 0.25 * tt2
            - 2.167178e-14 * 0.20 * tt3
            - 1.088458e03 * tt4
        )
        h_low = (
            3.782456e00
            - 2.996734e-03 * 0.50 * tt0
            + 9.847302e-06 * self.one_third * tt1
            - 9.681295e-09 * 0.25 * tt2
            + 3.243728e-12 * 0.20 * tt3
            - 1.063944e03 * tt4
        )
        hrt1 = self.npctx.where(tt0 < 1.000000e03, h_low, h_high)

        h_high = (
            3.857460e00
            + 4.414370e-03 * 0.50 * tt0
            - 2.214814e-06 * self.one_third * tt1
            + 5.234902e-10 * 0.25 * tt2
            - 4.720842e-14 * 0.20 * tt3
            - 4.875917e04 * tt4
        )
        h_low = (
            2.356774e00
            + 8.984597e-03 * 0.50 * tt0
            - 7.123563e-06 * self.one_third * tt1
            + 2.459190e-09 * 0.25 * tt2
            - 1.436995e-13 * 0.20 * tt3
            - 4.837197e04 * tt4
        )
        hrt2 = self.npctx.where(tt0 < 1.000000e03, h_low, h_high)

        h_high = (
            2.715186e00
            + 2.062527e-03 * 0.50 * tt0
            - 9.988258e-07 * self.one_third * tt1
            + 2.300530e-10 * 0.25 * tt2
            - 2.036477e-14 * 0.20 * tt3
            - 1.415187e04 * tt4
        )
        h_low = (
            3.579533e00
            - 6.103537e-04 * 0.50 * tt0
            + 1.016814e-06 * self.one_third * tt1
            + 9.070059e-10 * 0.25 * tt2
            - 9.044245e-13 * 0.20 * tt3
            - 1.434409e04 * tt4
        )
        hrt3 = self.npctx.where(tt0 < 1.000000e03, h_low, h_high)

        h_high = (
            3.033992e00
            + 2.176918e-03 * 0.50 * tt0
            - 1.640725e-07 * self.one_third * tt1
            - 9.704199e-11 * 0.25 * tt2
            + 1.682010e-14 * 0.20 * tt3
            - 3.000430e04 * tt4
        )
        h_low = (
            4.198641e00
            - 2.036434e-03 * 0.50 * tt0
            + 6.520402e-06 * self.one_third * tt1
            - 5.487971e-09 * 0.25 * tt2
            + 1.771978e-12 * 0.20 * tt3
            - 3.029373e04 * tt4
        )
        hrt4 = self.npctx.where(tt0 < 1.000000e03, h_low, h_high)

        h_high = (
            3.337279e00
            - 4.940247e-05 * 0.50 * tt0
            + 4.994568e-07 * self.one_third * tt1
            - 1.795664e-10 * 0.25 * tt2
            + 2.002554e-14 * 0.20 * tt3
            - 9.501589e02 * tt4
        )
        h_low = (
            2.344331e00
            + 7.980521e-03 * 0.50 * tt0
            - 1.947815e-05 * self.one_third * tt1
            + 2.015721e-08 * 0.25 * tt2
            - 7.376118e-12 * 0.20 * tt3
            - 9.179352e02 * tt4
        )
        hrt5 = self.npctx.where(tt0 < 1.000000e03, h_low, h_high)

        h_high = (
            2.926640e00
            + 1.487977e-03 * 0.50 * tt0
            - 5.684760e-07 * self.one_third * tt1
            + 1.009704e-10 * 0.25 * tt2
            - 6.753351e-15 * 0.20 * tt3
            - 9.227977e02 * tt4
        )
        h_low = (
            3.298677e00
            + 1.408240e-03 * 0.50 * tt0
            - 3.963222e-06 * self.one_third * tt1
            + 5.641515e-09 * 0.25 * tt2
            - 2.444854e-12 * 0.20 * tt3
            - 1.020900e03 * tt4
        )
        hrt6 = self.npctx.where(tt0 < 1.000000e03, h_low, h_high)

        return make_obj_array([hrt0, hrt1, hrt2, hrt3, hrt4, hrt5, hrt6])

    def get_species_entropies_R(self, temperature):

        tt0 = temperature
        tt1 = temperature * tt0
        tt2 = temperature * tt1
        tt3 = temperature * tt2
        tt6 = self.npctx.log(tt0)

        s_high = (
            2.036111e00 * tt6
            + 1.464542e-02 * tt0
            - 6.710779e-06 * 0.50 * tt1
            + 1.472229e-09 * self.one_third * tt2
            - 1.257061e-13 * 0.25 * tt3
            + 1.030537e01
        )
        s_low = (
            3.959201e00 * tt6
            - 7.570522e-03 * tt0
            + 5.709903e-05 * 0.50 * tt1
            - 6.915888e-08 * self.one_third * tt2
            + 2.698844e-11 * 0.25 * tt3
            + 4.097331e00
        )
        sr0 = self.npctx.where(tt0 < 1.000000e03, s_low, s_high)

        s_high = (
            3.282538e00 * tt6
            + 1.483088e-03 * tt0
            - 7.579667e-07 * 0.50 * tt1
            + 2.094706e-10 * self.one_third * tt2
            - 2.167178e-14 * 0.25 * tt3
            + 5.453231e00
        )
        s_low = (
            3.782456e00 * tt6
            - 2.996734e-03 * tt0
            + 9.847302e-06 * 0.50 * tt1
            - 9.681295e-09 * self.one_third * tt2
            + 3.243728e-12 * 0.25 * tt3
            + 3.657676e00
        )
        sr1 = self.npctx.where(tt0 < 1.000000e03, s_low, s_high)

        s_high = (
            3.857460e00 * tt6
            + 4.414370e-03 * tt0
            - 2.214814e-06 * 0.50 * tt1
            + 5.234902e-10 * self.one_third * tt2
            - 4.720842e-14 * 0.25 * tt3
            + 2.271638e00
        )
        s_low = (
            2.356774e00 * tt6
            + 8.984597e-03 * tt0
            - 7.123563e-06 * 0.50 * tt1
            + 2.459190e-09 * self.one_third * tt2
            - 1.436995e-13 * 0.25 * tt3
            + 9.901052e00
        )
        sr2 = self.npctx.where(tt0 < 1.000000e03, s_low, s_high)

        s_high = (
            2.715186e00 * tt6
            + 2.062527e-03 * tt0
            - 9.988258e-07 * 0.50 * tt1
            + 2.300530e-10 * self.one_third * tt2
            - 2.036477e-14 * 0.25 * tt3
            + 7.818688e00
        )
        s_low = (
            3.579533e00 * tt6
            - 6.103537e-04 * tt0
            + 1.016814e-06 * 0.50 * tt1
            + 9.070059e-10 * self.one_third * tt2
            - 9.044245e-13 * 0.25 * tt3
            + 3.508409e00
        )
        sr3 = self.npctx.where(tt0 < 1.000000e03, s_low, s_high)

        s_high = (
            3.033992e00 * tt6
            + 2.176918e-03 * tt0
            - 1.640725e-07 * 0.50 * tt1
            - 9.704199e-11 * self.one_third * tt2
            + 1.682010e-14 * 0.25 * tt3
            + 4.966770e00
        )
        s_low = (
            4.198641e00 * tt6
            - 2.036434e-03 * tt0
            + 6.520402e-06 * 0.50 * tt1
            - 5.487971e-09 * self.one_third * tt2
            + 1.771978e-12 * 0.25 * tt3
            - 8.490322e-01
        )
        sr4 = self.npctx.where(tt0 < 1.000000e03, s_low, s_high)

        s_high = (
            3.337279e00 * tt6
            - 4.940247e-05 * tt0
            + 4.994568e-07 * 0.50 * tt1
            - 1.795664e-10 * self.one_third * tt2
            + 2.002554e-14 * 0.25 * tt3
            - 3.205023e00
        )
        s_low = (
            2.344331e00 * tt6
            + 7.980521e-03 * tt0
            - 1.947815e-05 * 0.50 * tt1
            + 2.015721e-08 * self.one_third * tt2
            - 7.376118e-12 * 0.25 * tt3
            + 6.830102e-01
        )
        sr5 = self.npctx.where(tt0 < 1.000000e03, s_low, s_high)

        s_high = (
            2.926640e00 * tt6
            + 1.487977e-03 * tt0
            - 5.684760e-07 * 0.50 * tt1
            + 1.009704e-10 * self.one_third * tt2
            - 6.753351e-15 * 0.25 * tt3
            + 5.980528e00
        )
        s_low = (
            3.298677e00 * tt6
            + 1.408240e-03 * tt0
            - 3.963222e-06 * 0.50 * tt1
            + 5.641515e-09 * self.one_third * tt2
            - 2.444854e-12 * 0.25 * tt3
            + 3.950372e00
        )
        sr6 = self.npctx.where(tt0 < 1.000000e03, s_low, s_high)

        return make_obj_array([sr0, sr1, sr2, sr3, sr4, sr5, sr6])

    def get_temperature(self, enthalpy_or_energy, t_guess, y, do_energy=False):

        if do_energy is False:
            pv_fun = self.get_mixture_specific_heat_cp_mass
            he_fun = self.get_mixture_enthalpy_mass
        else:
            pv_fun = self.get_mixture_specific_heat_cv_mass
            he_fun = self.get_mixture_internal_energy_mass

        num_iter = 500
        tol = 1.0e-6
        ones = (1 + enthalpy_or_energy) - enthalpy_or_energy
        t_i = t_guess * ones

        for iter in range(num_iter):
            f = enthalpy_or_energy - he_fun(t_i, y)
            j = -pv_fun(t_i, y)
            dt = -f / j
            t_i += dt
            if self.npctx.linalg.norm(dt, np.inf) < tol:
                break

        return t_i

    def get_species_gibbs_RT(self, T):
        h0_RT = self.get_species_enthalpies_RT(T)
        s0_R  = self.get_species_entropies_R(T)
        return h0_RT - s0_R

    def get_equilibrium_constants(self, T):
        RT = self.gas_constant * T
        C0 = self.npctx.log( self.one_atm / RT )

        g0_RT = self.get_species_gibbs_RT( T )
        return make_obj_array([-86*T, g0_RT[2] + C0 / 2 - (g0_RT[3] + 0.5*g0_RT[1]),
                               g0_RT[4] + C0 / 2 - (g0_RT[5] + 0.5*g0_RT[1]), ])


    def get_falloff_rates(self, T, C, k_fwd):
        k_high = np.array([
        ])

        k_low = np.array([
        ])

        reduced_pressure = np.array([
        ])

        falloff_center = np.array([
        ])

        falloff_function = np.array([
        ])*reduced_pressure/(1+reduced_pressure)

        return

    def get_fwd_rate_coefficients(self, T, C):
        k_fwd = make_obj_array([
            self.npctx.exp(26.594857854425133 + -1*(17864.293439206183 / T)),
            self.npctx.exp(12.693776816787125
                           + 0.7*self.npctx.log(T) + -1*(6038.634401985189 / T)),
            self.npctx.exp(18.302572655472037 + -1*(17612.683672456802 / T)),
        ])
        return k_fwd

    def get_net_rates_of_progress(self, T, C):
        k_fwd = self.get_fwd_rate_coefficients(T, C)
        log_k_eq = self.get_equilibrium_constants(T)
        k_eq = self.npctx.exp(log_k_eq)
        return make_obj_array([k_fwd[0]*C[0]**0.5*C[1]**0.65,
                               k_fwd[1]*(C[3]*C[1]**0.5 + -1*k_eq[1]*C[2]),
                               k_fwd[2]*(C[5]*C[1]**0.5 + -1*k_eq[2]*C[4]), ])

    def get_net_production_rates(self, rho, T, Y):
        C = self.get_concentrations(rho, Y)
        r_net = self.get_net_rates_of_progress(T, C)

        return make_obj_array([
                -1*r_net[0],
                -1*(r_net[0] + 0.5*r_net[1] + 0.5*r_net[2]),
                r_net[1],
                2.0*r_net[0] + -1*r_net[1],
                r_net[2],
                2.0*r_net[0] + -1*r_net[2],
                0*r_net[0],
            ])


class UIUCMechanism2:
    def __init__(self, npctx=np):
        self.npctx = npctx
        self.model_name    = "uiuc.cti"
        self.num_elements  = 4
        self.num_species   = 7
        self.one_third = 1 / 3
        self.num_reactions = 3
        self.num_falloff   = 0
        
        self.one_atm = 101325.0
        self.gas_constant = 8314.46261815324
        self.big_number = 1.0e300

        self.species_names = ['C2H4', 'O2', 'CO2', 'CO', 'H2O', 'H2', 'N2']
        self.species_indices = {'C2H4': 0, 'O2': 1, 'CO2': 2, 'CO': 3, 'H2O': 4, 'H2': 5, 'N2': 6}

        self.wts = np.array([28.054, 31.998, 44.009, 28.009999999999998, 18.015, 2.016, 28.014])
        self.iwts = 1.0 / self.wts

    def species_name(self, species_index):
        return self.species_name[species_index]

    def species_index(self, species_name):
        return self.species_indices[species_name]

    def get_specific_gas_constant(self, Y):
        return self.gas_constant * np.dot( self.iwts, Y )

    def get_density(self, p, T, Y):
        mmw = self.get_mix_molecular_weight( Y )
        RT  = self.gas_constant * T
        return p * mmw / RT

    def get_pressure(self, rho, T, Y):
        mmw = self.get_mix_molecular_weight( Y )
        RT  = self.gas_constant * T
        return rho * RT / mmw

    def get_mix_molecular_weight(self, Y):
        return 1.0 / np.dot( self.iwts, Y )

    def get_concentrations(self, rho, Y):
        conctest = self.iwts * rho * Y
        zero = 0 * conctest[0]
        for i, conc in enumerate(conctest):
            conctest[i] = self.npctx.where(conctest[i] > 0, conctest[i], zero)
        return conctest

    def get_mixture_specific_heat_cp_mass(self, temperature, massfractions):
        cp0_r = self.get_species_specific_heats_R(temperature)
        cpsum = sum([massfractions[i] * cp0_r[i] * self.iwts[i]
                     for i in range(self.num_species)])
        return self.gas_constant * cpsum

    def get_mixture_specific_heat_cv_mass(self, temperature, massfractions):
        cp0_r = self.get_species_specific_heats_R(temperature) - 1.0
        cpsum = sum([massfractions[i] * cp0_r[i] * self.iwts[i]
                     for i in range(self.num_species)])
        return self.gas_constant * cpsum

    def get_mixture_enthalpy_mass(self, temperature, massfractions):
        h0_rt = self.get_species_enthalpies_RT(temperature)
        hsum = sum([massfractions[i] * h0_rt[i] * self.iwts[i]
                    for i in range(self.num_species)])
        return self.gas_constant * temperature * hsum

    def get_mixture_internal_energy_mass(self, temperature, massfractions):

        e0_rt = self.get_species_enthalpies_RT(temperature) - 1.0
        esum = sum([massfractions[i] * e0_rt[i] * self.iwts[i]
                    for i in range(self.num_species)])
        return self.gas_constant * temperature * esum

#    def get_mixture_internal_energy_mass(self, temperature, massfractions):
#        e0_rt = self.get_species_enthalpies_RT(temperature) - 1.0
#        mflist = sum([massfractions[i] * self.iwts[i] * e0_rt[i]
#                      for i in range(self.num_species)])
#        esum = sum([massfractions[i] * e0_rt[i] * self.iwts[i]
#                    for i in range(self.num_species)])
#        return self.gas_constant * temperature * esum

    def get_species_specific_heats_R(self, temperature):

        tt0 = temperature
        tt1 = temperature * tt0
        tt2 = temperature * tt1
        tt3 = temperature * tt2

        cp_high = (
            2.036111e00
            + 1.464542e-02 * tt0
            - 6.710779e-06 * tt1
            + 1.472229e-09 * tt2
            - 1.257061e-13 * tt3
        )
        cp_low = (
            3.959201e00
            - 7.570522e-03 * tt0
            + 5.709903e-05 * tt1
            - 6.915888e-08 * tt2
            + 2.698844e-11 * tt3
        )
        cpr0 = self.npctx.where(tt0 < 1.000000e03, cp_low, cp_high)

        cp_high = (
            3.282538e00
            + 1.483088e-03 * tt0
            - 7.579667e-07 * tt1
            + 2.094706e-10 * tt2
            - 2.167178e-14 * tt3
        )
        cp_low = (
            3.782456e00
            - 2.996734e-03 * tt0
            + 9.847302e-06 * tt1
            - 9.681295e-09 * tt2
            + 3.243728e-12 * tt3
        )
        cpr1 = self.npctx.where(tt0 < 1.000000e03, cp_low, cp_high)

        cp_high = (
            3.857460e00
            + 4.414370e-03 * tt0
            - 2.214814e-06 * tt1
            + 5.234902e-10 * tt2
            - 4.720842e-14 * tt3
        )
        cp_low = (
            2.356774e00
            + 8.984597e-03 * tt0
            - 7.123563e-06 * tt1
            + 2.459190e-09 * tt2
            - 1.436995e-13 * tt3
        )
        cpr2 = self.npctx.where(tt0 < 1.000000e03, cp_low, cp_high)

        cp_high = (
            2.715186e00
            + 2.062527e-03 * tt0
            - 9.988258e-07 * tt1
            + 2.300530e-10 * tt2
            - 2.036477e-14 * tt3
        )
        cp_low = (
            3.579533e00
            - 6.103537e-04 * tt0
            + 1.016814e-06 * tt1
            + 9.070059e-10 * tt2
            - 9.044245e-13 * tt3
        )
        cpr3 = self.npctx.where(tt0 < 1.000000e03, cp_low, cp_high)

        cp_high = (
            3.033992e00
            + 2.176918e-03 * tt0
            - 1.640725e-07 * tt1
            - 9.704199e-11 * tt2
            + 1.682010e-14 * tt3
        )
        cp_low = (
            4.198641e00
            - 2.036434e-03 * tt0
            + 6.520402e-06 * tt1
            - 5.487971e-09 * tt2
            + 1.771978e-12 * tt3
        )
        cpr4 = self.npctx.where(tt0 < 1.000000e03, cp_low, cp_high)

        cp_high = (
            3.337279e00
            - 4.940247e-05 * tt0
            + 4.994568e-07 * tt1
            - 1.795664e-10 * tt2
            + 2.002554e-14 * tt3
        )
        cp_low = (
            2.344331e00
            + 7.980521e-03 * tt0
            - 1.947815e-05 * tt1
            + 2.015721e-08 * tt2
            - 7.376118e-12 * tt3
        )
        cpr5 = self.npctx.where(tt0 < 1.000000e03, cp_low, cp_high)

        cp_high = (
            2.926640e00
            + 1.487977e-03 * tt0
            - 5.684760e-07 * tt1
            + 1.009704e-10 * tt2
            - 6.753351e-15 * tt3
        )
        cp_low = (
            3.298677e00
            + 1.408240e-03 * tt0
            - 3.963222e-06 * tt1
            + 5.641515e-09 * tt2
            - 2.444854e-12 * tt3
        )
        cpr6 = self.npctx.where(tt0 < 1.000000e03, cp_low, cp_high)

        return make_obj_array([cpr0, cpr1, cpr2, cpr3, cpr4, cpr5, cpr6])

    def get_species_enthalpies_RT(self, temperature):

        tt0 = temperature
        tt1 = temperature * tt0
        tt2 = temperature * tt1
        tt3 = temperature * tt2
        tt4 = 1.0 / temperature

        h_high = (
            2.036111e00
            + 1.464542e-02 * 0.50 * tt0
            - 6.710779e-06 * self.one_third * tt1
            + 1.472229e-09 * 0.25 * tt2
            - 1.257061e-13 * 0.20 * tt3
            + 4.939886e03 * tt4
        )
        h_low = (
            3.959201e00
            - 7.570522e-03 * 0.50 * tt0
            + 5.709903e-05 * self.one_third * tt1
            - 6.915888e-08 * 0.25 * tt2
            + 2.698844e-11 * 0.20 * tt3
            + 5.089776e03 * tt4
        )
        hrt0 = self.npctx.where(tt0 < 1.000000e03, h_low, h_high)

        h_high = (
            3.282538e00
            + 1.483088e-03 * 0.50 * tt0
            - 7.579667e-07 * self.one_third * tt1
            + 2.094706e-10 * 0.25 * tt2
            - 2.167178e-14 * 0.20 * tt3
            - 1.088458e03 * tt4
        )
        h_low = (
            3.782456e00
            - 2.996734e-03 * 0.50 * tt0
            + 9.847302e-06 * self.one_third * tt1
            - 9.681295e-09 * 0.25 * tt2
            + 3.243728e-12 * 0.20 * tt3
            - 1.063944e03 * tt4
        )
        hrt1 = self.npctx.where(tt0 < 1.000000e03, h_low, h_high)

        h_high = (
            3.857460e00
            + 4.414370e-03 * 0.50 * tt0
            - 2.214814e-06 * self.one_third * tt1
            + 5.234902e-10 * 0.25 * tt2
            - 4.720842e-14 * 0.20 * tt3
            - 4.875917e04 * tt4
        )
        h_low = (
            2.356774e00
            + 8.984597e-03 * 0.50 * tt0
            - 7.123563e-06 * self.one_third * tt1
            + 2.459190e-09 * 0.25 * tt2
            - 1.436995e-13 * 0.20 * tt3
            - 4.837197e04 * tt4
        )
        hrt2 = self.npctx.where(tt0 < 1.000000e03, h_low, h_high)

        h_high = (
            2.715186e00
            + 2.062527e-03 * 0.50 * tt0
            - 9.988258e-07 * self.one_third * tt1
            + 2.300530e-10 * 0.25 * tt2
            - 2.036477e-14 * 0.20 * tt3
            - 1.415187e04 * tt4
        )
        h_low = (
            3.579533e00
            - 6.103537e-04 * 0.50 * tt0
            + 1.016814e-06 * self.one_third * tt1
            + 9.070059e-10 * 0.25 * tt2
            - 9.044245e-13 * 0.20 * tt3
            - 1.434409e04 * tt4
        )
        hrt3 = self.npctx.where(tt0 < 1.000000e03, h_low, h_high)

        h_high = (
            3.033992e00
            + 2.176918e-03 * 0.50 * tt0
            - 1.640725e-07 * self.one_third * tt1
            - 9.704199e-11 * 0.25 * tt2
            + 1.682010e-14 * 0.20 * tt3
            - 3.000430e04 * tt4
        )
        h_low = (
            4.198641e00
            - 2.036434e-03 * 0.50 * tt0
            + 6.520402e-06 * self.one_third * tt1
            - 5.487971e-09 * 0.25 * tt2
            + 1.771978e-12 * 0.20 * tt3
            - 3.029373e04 * tt4
        )
        hrt4 = self.npctx.where(tt0 < 1.000000e03, h_low, h_high)

        h_high = (
            3.337279e00
            - 4.940247e-05 * 0.50 * tt0
            + 4.994568e-07 * self.one_third * tt1
            - 1.795664e-10 * 0.25 * tt2
            + 2.002554e-14 * 0.20 * tt3
            - 9.501589e02 * tt4
        )
        h_low = (
            2.344331e00
            + 7.980521e-03 * 0.50 * tt0
            - 1.947815e-05 * self.one_third * tt1
            + 2.015721e-08 * 0.25 * tt2
            - 7.376118e-12 * 0.20 * tt3
            - 9.179352e02 * tt4
        )
        hrt5 = self.npctx.where(tt0 < 1.000000e03, h_low, h_high)

        h_high = (
            2.926640e00
            + 1.487977e-03 * 0.50 * tt0
            - 5.684760e-07 * self.one_third * tt1
            + 1.009704e-10 * 0.25 * tt2
            - 6.753351e-15 * 0.20 * tt3
            - 9.227977e02 * tt4
        )
        h_low = (
            3.298677e00
            + 1.408240e-03 * 0.50 * tt0
            - 3.963222e-06 * self.one_third * tt1
            + 5.641515e-09 * 0.25 * tt2
            - 2.444854e-12 * 0.20 * tt3
            - 1.020900e03 * tt4
        )
        hrt6 = self.npctx.where(tt0 < 1.000000e03, h_low, h_high)

        return make_obj_array([hrt0, hrt1, hrt2, hrt3, hrt4, hrt5, hrt6])

    def get_species_entropies_R(self, temperature):

        tt0 = temperature
        tt1 = temperature * tt0
        tt2 = temperature * tt1
        tt3 = temperature * tt2
        tt6 = self.npctx.log(tt0)

        s_high = (
            2.036111e00 * tt6
            + 1.464542e-02 * tt0
            - 6.710779e-06 * 0.50 * tt1
            + 1.472229e-09 * self.one_third * tt2
            - 1.257061e-13 * 0.25 * tt3
            + 1.030537e01
        )
        s_low = (
            3.959201e00 * tt6
            - 7.570522e-03 * tt0
            + 5.709903e-05 * 0.50 * tt1
            - 6.915888e-08 * self.one_third * tt2
            + 2.698844e-11 * 0.25 * tt3
            + 4.097331e00
        )
        sr0 = self.npctx.where(tt0 < 1.000000e03, s_low, s_high)

        s_high = (
            3.282538e00 * tt6
            + 1.483088e-03 * tt0
            - 7.579667e-07 * 0.50 * tt1
            + 2.094706e-10 * self.one_third * tt2
            - 2.167178e-14 * 0.25 * tt3
            + 5.453231e00
        )
        s_low = (
            3.782456e00 * tt6
            - 2.996734e-03 * tt0
            + 9.847302e-06 * 0.50 * tt1
            - 9.681295e-09 * self.one_third * tt2
            + 3.243728e-12 * 0.25 * tt3
            + 3.657676e00
        )
        sr1 = self.npctx.where(tt0 < 1.000000e03, s_low, s_high)

        s_high = (
            3.857460e00 * tt6
            + 4.414370e-03 * tt0
            - 2.214814e-06 * 0.50 * tt1
            + 5.234902e-10 * self.one_third * tt2
            - 4.720842e-14 * 0.25 * tt3
            + 2.271638e00
        )
        s_low = (
            2.356774e00 * tt6
            + 8.984597e-03 * tt0
            - 7.123563e-06 * 0.50 * tt1
            + 2.459190e-09 * self.one_third * tt2
            - 1.436995e-13 * 0.25 * tt3
            + 9.901052e00
        )
        sr2 = self.npctx.where(tt0 < 1.000000e03, s_low, s_high)

        s_high = (
            2.715186e00 * tt6
            + 2.062527e-03 * tt0
            - 9.988258e-07 * 0.50 * tt1
            + 2.300530e-10 * self.one_third * tt2
            - 2.036477e-14 * 0.25 * tt3
            + 7.818688e00
        )
        s_low = (
            3.579533e00 * tt6
            - 6.103537e-04 * tt0
            + 1.016814e-06 * 0.50 * tt1
            + 9.070059e-10 * self.one_third * tt2
            - 9.044245e-13 * 0.25 * tt3
            + 3.508409e00
        )
        sr3 = self.npctx.where(tt0 < 1.000000e03, s_low, s_high)

        s_high = (
            3.033992e00 * tt6
            + 2.176918e-03 * tt0
            - 1.640725e-07 * 0.50 * tt1
            - 9.704199e-11 * self.one_third * tt2
            + 1.682010e-14 * 0.25 * tt3
            + 4.966770e00
        )
        s_low = (
            4.198641e00 * tt6
            - 2.036434e-03 * tt0
            + 6.520402e-06 * 0.50 * tt1
            - 5.487971e-09 * self.one_third * tt2
            + 1.771978e-12 * 0.25 * tt3
            - 8.490322e-01
        )
        sr4 = self.npctx.where(tt0 < 1.000000e03, s_low, s_high)

        s_high = (
            3.337279e00 * tt6
            - 4.940247e-05 * tt0
            + 4.994568e-07 * 0.50 * tt1
            - 1.795664e-10 * self.one_third * tt2
            + 2.002554e-14 * 0.25 * tt3
            - 3.205023e00
        )
        s_low = (
            2.344331e00 * tt6
            + 7.980521e-03 * tt0
            - 1.947815e-05 * 0.50 * tt1
            + 2.015721e-08 * self.one_third * tt2
            - 7.376118e-12 * 0.25 * tt3
            + 6.830102e-01
        )
        sr5 = self.npctx.where(tt0 < 1.000000e03, s_low, s_high)

        s_high = (
            2.926640e00 * tt6
            + 1.487977e-03 * tt0
            - 5.684760e-07 * 0.50 * tt1
            + 1.009704e-10 * self.one_third * tt2
            - 6.753351e-15 * 0.25 * tt3
            + 5.980528e00
        )
        s_low = (
            3.298677e00 * tt6
            + 1.408240e-03 * tt0
            - 3.963222e-06 * 0.50 * tt1
            + 5.641515e-09 * self.one_third * tt2
            - 2.444854e-12 * 0.25 * tt3
            + 3.950372e00
        )
        sr6 = self.npctx.where(tt0 < 1.000000e03, s_low, s_high)

        return make_obj_array([sr0, sr1, sr2, sr3, sr4, sr5, sr6])

    def get_temperature(self, enthalpy_or_energy, t_guess, y, do_energy=False):

        if do_energy is False:
            pv_fun = self.get_mixture_specific_heat_cp_mass
            he_fun = self.get_mixture_enthalpy_mass
        else:
            pv_fun = self.get_mixture_specific_heat_cv_mass
            he_fun = self.get_mixture_internal_energy_mass

        num_iter = 500
        tol = 1.0e-6
        ones = (1 + enthalpy_or_energy) - enthalpy_or_energy
        t_i = t_guess * ones

        for iter in range(num_iter):
            f = enthalpy_or_energy - he_fun(t_i, y)
            j = -pv_fun(t_i, y)
            dt = -f / j
            t_i += dt
            if self.npctx.linalg.norm(dt, np.inf) < tol:
                break

        return t_i

    def get_species_gibbs_RT(self, T):
        h0_RT = self.get_species_enthalpies_RT(T)
        s0_R  = self.get_species_entropies_R(T)
        return h0_RT - s0_R

    def get_equilibrium_constants(self, T):
        RT = self.gas_constant * T
        C0 = self.npctx.log( self.one_atm / RT )

        g0_RT = self.get_species_gibbs_RT( T )
        return make_obj_array([-86*T, g0_RT[2] + C0 / 2 - (g0_RT[3] + 0.5*g0_RT[1]),
                               g0_RT[4] + C0 / 2 - (g0_RT[5] + 0.5*g0_RT[1]), ])


    def get_falloff_rates(self, T, C, k_fwd):
        k_high = np.array([
        ])

        k_low = np.array([
        ])

        reduced_pressure = np.array([
        ])

        falloff_center = np.array([
        ])

        falloff_function = np.array([
        ])*reduced_pressure/(1+reduced_pressure)

        return

    def get_fwd_rate_coefficients(self, T, C):
        k_fwd = make_obj_array([
            self.npctx.exp(20.0 + -1*(17864.293439206183 / T)),
            self.npctx.exp(12.693776816787125
                           + 0.7*self.npctx.log(T) + -1*(6038.634401985189 / T)),
            self.npctx.exp(18.302572655472037 + -1*(17612.683672456802 / T)),
        ])
        return k_fwd

    def get_net_rates_of_progress(self, T, C):
        k_fwd = self.get_fwd_rate_coefficients(T, C)
        log_k_eq = self.get_equilibrium_constants(T)
        k_eq = self.npctx.exp(log_k_eq)
        return make_obj_array([k_fwd[0]*C[0]**0.5*C[1]**0.65,
                               k_fwd[1]*(C[3]*C[1]**0.5 + -1*k_eq[1]*C[2]),
                               k_fwd[2]*(C[5]*C[1]**0.5 + -1*k_eq[2]*C[4]), ])

    def get_net_production_rates(self, rho, T, Y):
        C = self.get_concentrations(rho, Y)
        r_net = self.get_net_rates_of_progress(T, C)

        return make_obj_array([
                -1*r_net[0],
                -1*(r_net[0] + 0.5*r_net[1] + 0.5*r_net[2]),
                r_net[1],
                2.0*r_net[0] + -1*r_net[1],
                r_net[2],
                2.0*r_net[0] + -1*r_net[2],
                0*r_net[0],
            ])
