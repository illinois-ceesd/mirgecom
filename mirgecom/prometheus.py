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
        self.iwts = 1.0 / self.wts
        return

    def get_density(self, p, temperature, massfractions):

        mmw = self.get_mix_molecular_weight(massfractions)
        rt = self.gas_constant * temperature
        rho = p * mmw / rt
        return rho

    def get_pressure(self, rho, temperature, massfractions):

        mmw = self.get_mix_molecular_weight(massfractions)
        rt = self.gas_constant * temperature
        p = rho * rt / mmw
        return p

    def get_mix_molecular_weight(self, massfractions):

        return 1.0 / np.dot(self.iwts, massfractions)

    def get_concentrations(self, rho, massfractions):

        return self.iwts * rho * massfractions

    def get_mixture_specific_heat_cp_mass(self, temperature, massfractions):

        cp0_r = self.get_species_specific_heats(temperature)
        cpsum = sum([massfractions[i] * cp0_r[i] * self.iwts[i]
                     for i in range(self.num_species)])
        return self.gas_constant * cpsum

    def get_mixture_specific_heat_cv_mass(self, temperature, massfractions):

        cp0_r = self.get_species_specific_heats(temperature) - 1.0
        cpsum = sum([massfractions[i] * cp0_r[i] * self.iwts[i]
                     for i in range(self.num_species)])
        return self.gas_constant * cpsum

    def get_mixture_enthalpy_mass(self, temperature, massfractions):

        h0_rt = self.get_species_enthalpies(temperature)
        hsum = sum([massfractions[i] * h0_rt[i] * self.iwts[i]
                    for i in range(self.num_species)])
        return self.gas_constant * temperature * hsum

    def get_mixture_internal_energy_mass(self, temperature, massfractions):

        e0_rt = self.get_species_enthalpies(temperature) - 1.0
        esum = sum([massfractions[i] * e0_rt[i] * self.iwts[i]
                    for i in range(self.num_species)])
        return self.gas_constant * temperature * esum

    def get_species_specific_heats(self, temperature):

        actx = temperature.array_context

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

    def get_species_enthalpies(self, temperature):

        actx = temperature.array_context

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

    def get_species_entropies(self, temperature):

        actx = temperature.array_context

        tt0 = temperature
        tt1 = temperature * tt0
        tt2 = temperature * tt1
        tt3 = temperature * tt2
        #        tt4 = 1.0 / temperature
        #        tt5 = tt4 * tt4
        tt6 = actx.np.log(tt0)

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

    def get_species_gibbs(self, temperature):

        h0rt = self.get_species_enthalpies(temperature)
        s0r = self.get_species_entropies(temperature)
        return h0rt - s0r

    def get_equilibrium_constants(self, temperature):

        actx = temperature.array_context
        rt = self.gas_constant * temperature
        c0 = actx.np.log(self.one_atm / rt)

        g0rt = self.get_species_gibbs(temperature)

        k_eq1 = c0 + (g0rt[2]) - (g0rt[3] + g0rt[1])
        k_eq2 = c0 + (g0rt[4]) - (g0rt[5] + g0rt[1])

        return make_obj_array([k_eq1, k_eq2])

    def get_temperature(self, enthalpy_or_energy, t_guess, y, do_energy=False):

        if do_energy is False:
            pv_fun = self.get_mixture_specific_heat_cp_mass
            he_fun = self.get_mixture_enthalpy_mass
        else:
            pv_fun = self.get_mixture_specific_heat_cv_mass
            he_fun = self.get_mixture_internal_energy_mass

        num_iter = 500
        tol = 1.0e-6
        t_i = t_guess * enthalpy_or_energy * 1.0 / enthalpy_or_energy

        for iter in range(num_iter):
            f = enthalpy_or_energy - he_fun(t_i, y)
            j = -pv_fun(t_i, y)
            dt = -f / j
            t_i += dt
            if self.discr.norm(dt, np.inf) < tol:
                break

        return t_i

    def get_rate_coefficients(self, temperature, concentrations):

        actx = temperature.array_context
        logt = actx.np.log(temperature)
        invt = 1.0 / temperature
        k_eq = actx.np.min(self.big_number,
                           self.get_equilibrium_constants(temperature))

        k_fwd0 = actx.np.exp(2.659486e01 - 1.786429e04 * invt)
        k_fwd1 = actx.np.exp(1.269378e01 + 7.000000e-01 * logt - 6.038634e03 * invt)
        k_fwd2 = actx.np.exp(1.830257e01 - 1.761268e04 * invt)

        k_fwd = make_obj_array([k_fwd0, k_fwd1, k_fwd2])
        k_rev = k_fwd * actx.np.exp(k_eq)

        return k_fwd, k_rev

    def get_net_rates_of_progress(self, temperature, concentrations):

        k_fwd, k_rev = self.get_rate_coefficients(temperature, concentrations)

        r_fwd0 = k_fwd[0] * concentrations[0] * concentrations[1]
        r_rev0 = 0.0 * r_fwd0

        r_fwd1 = k_fwd[1] * concentrations[3] * concentrations[1]
        r_rev1 = k_rev[1] * concentrations[2]

        r_fwd2 = k_fwd[2] * concentrations[5] * concentrations[1]
        r_rev2 = k_rev[2] * concentrations[4]

        r_fwd = make_obj_array([r_fwd0, r_fwd1, r_fwd2])
        r_rev = make_obj_array([r_rev0, r_rev1, r_rev2])

        return r_fwd - r_rev

    def get_net_production_rates(self, rho, temperature, massfractions):

        concentrations = self.get_concentrations(rho, massfractions)
        r_net = self.get_net_rates_of_progress(temperature, concentrations)

        omega0 = -r_net[0]
        omega1 = sum([-r_net[i] for i in range(3)])
        omega2 = r_net[1]
        omega3 = 2 * r_net[0] - r_net[1]
        omega4 = r_net[2]
        omega5 = 2 * r_net[0] - r_net[2]

        return make_obj_array([omega0, omega1, omega2,
                               omega3, omega4, omega5])
