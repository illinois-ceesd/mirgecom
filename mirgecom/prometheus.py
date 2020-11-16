import numpy as np
from pytools.obj_array import make_obj_array

class UIUCMechanism:
    def __init__(self, discr):

        self.model_name = "uiuc"
        self.num_elements = 4
        self.num_species = 7
        self.num_reactions = 3
        self.num_falloff = 0
        self.one_atm = 1.01325e5
        self.one_third = 1.0 / 3.0
        self.gas_constant = 8314.4621
        self.big_number = 1.0e300
        self.discr = discr

        self.wts = np.array(
            [
                2.805400e01,
                3.199800e01,
                4.400900e01,
                2.801000e01,
                1.801500e01,
                2.016000e00,
                2.801400e01,
            ]
        )
        self.iwts = np.reciprocal(self.wts)

        return

    def get_density(self, p, T, Y):  # noqa

        mmw = self.get_mix_molecular_weight(Y)
        RT = self.gas_constant * T  # noqa
        rho = p * mmw / RT
        return rho

    def get_pressure(self, rho, T, Y):  # noqa

        mmw = self.get_mix_molecular_weight(Y)
        RT = self.gas_constant * T  # noqa
        p = rho * RT / mmw
        return p

    def get_mix_molecular_weight(self, Y):  # noqa

        mmw = np.reciprocal(np.dot(self.iwts, Y))
        return mmw

    def get_concentrations(self, rho, Y):  # noqa

        C = self.iwts * rho * Y
        return C

    def get_mixture_specific_heat_cp_mass(self, T, Y):  # noqa

        cp0_R = self.get_species_specific_heats_R(T)  # noqa
        cp = 0.0
        cp += Y[0] * cp0_R[0] * self.iwts[0]
        cp += Y[1] * cp0_R[1] * self.iwts[1]
        cp += Y[2] * cp0_R[2] * self.iwts[2]
        cp += Y[3] * cp0_R[3] * self.iwts[3]
        cp += Y[4] * cp0_R[4] * self.iwts[4]
        cp += Y[5] * cp0_R[5] * self.iwts[5]
        cp += Y[6] * cp0_R[6] * self.iwts[6]
        cp *= self.gas_constant

        return cp

    def get_mixture_specific_heat_cv_mass(self, T, Y):  # noqa

        cp0_R = self.get_species_specific_heats_R(T) - 1.0  # noqa
        cp = 0.0
        cp += Y[0] * cp0_R[0] * self.iwts[0]
        cp += Y[1] * cp0_R[1] * self.iwts[1]
        cp += Y[2] * cp0_R[2] * self.iwts[2]
        cp += Y[3] * cp0_R[3] * self.iwts[3]
        cp += Y[4] * cp0_R[4] * self.iwts[4]
        cp += Y[5] * cp0_R[5] * self.iwts[5]
        cp += Y[6] * cp0_R[6] * self.iwts[6]
        cp *= self.gas_constant

        return cp

    def get_mixture_enthalpy_mass(self, T, Y):  # noqa

        h0_RT = self.get_species_enthalpies_RT(T)  # noqa
        h = 0.0
        h += Y[0] * h0_RT[0] * self.iwts[0]
        h += Y[1] * h0_RT[1] * self.iwts[1]
        h += Y[2] * h0_RT[2] * self.iwts[2]
        h += Y[3] * h0_RT[3] * self.iwts[3]
        h += Y[4] * h0_RT[4] * self.iwts[4]
        h += Y[5] * h0_RT[5] * self.iwts[5]
        h += Y[6] * h0_RT[6] * self.iwts[6]
        h *= self.gas_constant * T

        return h

    def get_mixture_internal_energy_mass(self, T, Y):  # noqa

        e0_RT = self.get_species_enthalpies_RT(T) - 1.0  # noqa
        e = 0.0
        e += Y[0] * e0_RT[0] * self.iwts[0]
        e += Y[1] * e0_RT[1] * self.iwts[1]
        e += Y[2] * e0_RT[2] * self.iwts[2]
        e += Y[3] * e0_RT[3] * self.iwts[3]
        e += Y[4] * e0_RT[4] * self.iwts[4]
        e += Y[5] * e0_RT[5] * self.iwts[5]
        e += Y[6] * e0_RT[6] * self.iwts[6]
        e *= self.gas_constant * T

        return e

    def get_species_specific_heats_R(self, T):  # noqa

        actx = T.array_context

        tt0 = T
        tt1 = T * tt0
        tt2 = T * tt1
        tt3 = T * tt2
        #        tt4 = np.power(T, -1.0)
        #        tt5 = tt4 * tt4

        #        cp0_R = np.zeros(self.num_species)

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
        cpr0 = actx.np.where(tt0 < 1.000000e03, cp_low, cp_high)

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
        cpr1 = actx.np.where(tt0 < 1.000000e03, cp_low, cp_high)

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
        cpr2 = actx.np.where(tt0 < 1.000000e03, cp_low, cp_high)

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
        cpr3 = actx.np.where(tt0 < 1.000000e03, cp_low, cp_high)

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
        cpr4 = actx.np.where(tt0 < 1.000000e03, cp_low, cp_high)

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
        cpr5 = actx.np.where(tt0 < 1.000000e03, cp_low, cp_high)

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
        cpr6 = actx.np.where(tt0 < 1.000000e03, cp_low, cp_high)

        return make_obj_array([cpr0, cpr1, cpr2, cpr3, cpr4, cpr5, cpr6])

    def get_species_enthalpies_RT(self, T):

        actx = T.array_context

        tt0 = T
        tt1 = T * tt0
        tt2 = T * tt1
        tt3 = T * tt2
        tt4 = np.power(T, -1.0)
        #        tt5 = tt4 * tt4
        #        tt6 = np.log(tt0) * tt4

        # h0_RT = np.zeros(self.num_species)

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
        hrt0 = actx.np.where(tt0 < 1.000000e03, h_low, h_high)

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
        hrt1 = actx.np.where(tt0 < 1.000000e03, h_low, h_high)

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
        hrt2 = actx.np.where(tt0 < 1.000000e03, h_low, h_high)

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
        hrt3 = actx.np.where(tt0 < 1.000000e03, h_low, h_high)

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
        hrt4 = actx.np.where(tt0 < 1.000000e03, h_low, h_high)

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
        hrt5 = actx.np.where(tt0 < 1.000000e03, h_low, h_high)

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
        hrt6 = actx.np.where(tt0 < 1.000000e03, h_low, h_high)

        return make_obj_array([hrt0, hrt1, hrt2, hrt3, hrt4, hrt5, hrt6])

    def get_species_entropies_R(self, T):

        actx = T.array_context

        tt0 = T
        tt1 = T * tt0
        tt2 = T * tt1
        tt3 = T * tt2
        tt4 = np.power(T, -1.0)
        tt5 = tt4 * tt4
        tt6 = np.log(tt0)

        #        s0_R = np.zeros(self.num_species)

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
        sr0 = actx.np.where(tt0 < 1.000000e03, s_low, s_high)

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
        sr1 = actx.np.where(tt0 < 1.000000e03, s_low, s_high)

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
        sr2 = actx.np.where(tt0 < 1.000000e03, s_low, s_high)

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
        sr3 = actx.np.where(tt0 < 1.000000e03, s_low, s_high)

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
        sr4 = actx.np.where(tt0 < 1.000000e03, s_low, s_high)

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
        sr5 = actx.np.where(tt0 < 1.000000e03, s_low, s_high)

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
        sr6 = actx.np.where(tt0 < 1.000000e03, s_low, s_high)

        return make_obj_array([sr0, sr1, sr2, sr3, sr4, sr5, sr6])

    def get_species_gibbs_RT(self, T):

        h0_RT = self.get_species_enthalpies_RT(T)
        s0_R = self.get_species_entropies_R(T)
        g0_RT = h0_RT - s0_R

        return g0_RT

    def get_equilibrium_constants(self, T):

        RT = self.gas_constant * T
        C0 = np.log(self.one_atm / RT)

        k_eq = np.zeros(self.num_reactions)

        g0_RT = self.get_species_gibbs_RT(T)

        k_eq[1] = C0 + (g0_RT[2]) - (g0_RT[3] + g0_RT[1])
        k_eq[2] = C0 + (g0_RT[4]) - (g0_RT[5] + g0_RT[1])

        return k_eq

    def get_temperature(self, H_or_E, T_guess, Y, do_energy=False):

        actx = H_or_E.array_context

        if do_energy is False:
            pv_fun = self.get_mixture_specific_heat_cp_mass
            he_fun = self.get_mixture_enthalpy_mass
        else:
            pv_fun = self.get_mixture_specific_heat_cv_mass
            he_fun = self.get_mixture_internal_energy_mass

        num_iter = 500
        tol = 1.0e-6
        T_i = T_guess * H_or_E * 1.0 / H_or_E

        for iter in range(num_iter):
            F = H_or_E - he_fun(T_i, Y)
            J = -pv_fun(T_i, Y)
            dT = -F / J
            T_i += dT
            tresid = actx.np.abs(dT)
            maxerr = self.discr.norm(tresid, np.inf)
            if maxerr < tol:
                break

        return T_i

    def get_rate_coefficients(self, T, C):

        log_T = np.log(T)
        inv_T = 1.0 / T
        k_eq = self.get_equilibrium_constants(T)
        k_fwd = np.zeros(self.num_reactions)
        k_rev = np.zeros(self.num_reactions)

        k_fwd[0] = np.exp(2.659486e01 - 1.786429e04 * inv_T)
        k_fwd[1] = np.exp(1.269378e01 + 7.000000e-01 * log_T - 6.038634e03 * inv_T)
        k_fwd[2] = np.exp(1.830257e01 - 1.761268e04 * inv_T)

        for i in range(0, self.num_reactions):
            if k_eq[i] > self.big_number:
                k_eq[i] = self.big_number
            k_rev[i] = k_fwd[i] * np.exp(k_eq[i])

        return k_fwd, k_rev

    def get_net_rates_of_progress(self, T, C):

        R_fwd = np.zeros(self.num_reactions)
        R_rev = np.zeros(self.num_reactions)
        R_net = np.zeros(self.num_reactions)
        k_fwd, k_rev = self.get_rate_coefficients(T, C)

        R_fwd[0] = k_fwd[0] * C[0] * C[1]

        R_fwd[1] = k_fwd[1] * C[3] * C[1]
        R_rev[1] = k_rev[1] * C[2]

        R_fwd[2] = k_fwd[2] * C[5] * C[1]
        R_rev[2] = k_rev[2] * C[4]

        for i in range(0, self.num_reactions):
            R_net[i] = R_fwd[i] - R_rev[i]

        return R_net

    def get_net_production_rates(self, rho, T, Y):

        C = self.get_concentrations(rho, Y)
        R_net = self.get_net_rates_of_progress(T, C)
        omega = np.zeros(self.num_species)

        omega[0] = -R_net[0]
        omega[1] = -R_net[0] - R_net[1] - R_net[2]
        omega[2] = +R_net[1]
        omega[3] = +R_net[0] + R_net[0] - R_net[1]
        omega[4] = +R_net[2]
        omega[5] = +R_net[0] + R_net[0] - R_net[2]

        return omega
