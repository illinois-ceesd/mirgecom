# NN model for Flamelet EOS
# Written by Esteban Cisneros for MIRGE
# TODO: Copyright
import h5py
from pytools.obj_array import make_obj_array
from arraycontext import get_container_context_recursively
import numpy as np


class FlameletNN:

    #    def __init__(self, w, b, state_mean, state_std, bell_curve_tab, actx,):
    #        self.w = w
    #        self.b = b
    #        self.state_mean = state_mean
    #        self.state_std = state_std
    #        self.bell_curve_tab = bell_curve_tab
    #        self.tab_dz = self.bell_curve_tab[0][1] - self.bell_curve_tab[0][0]
    #        self.actx = actx
    #        return
    def __init__(self, h5_init_data_filename):
        db = h5py.File(h5_init_data_filename, "r")
        self.state_mean = db["aux_params/state_mean"][:]
        self.state_std = db["aux_params/state_std"][:]
        self.bell_curve_tab = make_obj_array(
            db["aux_params/bell_curve_table"][:, :]
        )

        num_layers = len(db["nn_params/weights"].keys())

        nn_w = []
        nn_b = []
        for lyr in range(num_layers):
            nn_w += [db["nn_params/weights/w_" + str(lyr)][:, :]]
            nn_b += [db["nn_params/biases/b_" + str(lyr)][:]]

        self.w = make_obj_array(nn_w)
        self.b = make_obj_array(nn_b)
        self.tab_dz = self.bell_curve_tab[0][1] - self.bell_curve_tab[0][0]

    def _interpolation_weights(self, r):
        return make_obj_array([
            -r * (r-1) * (r-2) / 6,
            (r+1) * (r-1) * (r-2) / 2,
            -r * (r+1) * (r-2) / 2,
            r * (r+1) * (r-1) / 6
        ])

    def bell_curve(self, npctx, mixture_fraction):
        i = npctx.floor(mixture_fraction / self.tab_dz).astype(int)
        r = (mixture_fraction - self.bell_curve_tab[0][i]) / self.tab_dz
        w = self._interpolation_weights(r)
        return (
            w[0] * self.bell_curve_tab[1][i-1]
            + w[1] * self.bell_curve_tab[1][i]
            + w[2] * self.bell_curve_tab[1][i+1]
            + w[3] * self.bell_curve_tab[1][i+2]
        )

    def stoichiometric_diss_rate(self, npctx, mixture_fraction, diss_rate):
        return npctx.where(
            (diss_rate > 0) * (self.bell_curve(mixture_fraction) > 0),
            diss_rate / self.bell_curve(mixture_fraction),
            0
        )

    def linear_layer(self, npctx, w, b, x):
        return npctx.einsum(
            "ij,j...->i...", w, x
        ) + b

    def reconstruct(self, diss_rate, mixture_fraction, actx=None):
        if actx is None:
            actx = get_container_context_recursively(mixture_fraction)
            npctx = actx.np
        if actx is None:
            npctx = np

        num_dim = len(mixture_fraction.shape)
        diss_rate_st = self.stoichiometric_diss_rate(
            npctx, mixture_fraction, diss_rate
        )

        x = npctx.stack((diss_rate_st, mixture_fraction))
        x = npctx.tanh(
            self.linear_layer(
                npctx, self.w[0], self.b[0].reshape((-1,)
                                                    + num_dim * (1,)), x)
        )
        x = npctx.tanh(
            self.linear_layer(
                npctx, self.w[1], self.b[1].reshape((-1,)
                                                    + num_dim * (1,)), x)
        )
        x = npctx.tanh(
            self.linear_layer(
                npctx, self.w[2], self.b[2].reshape((-1,)
                                                    + num_dim * (1,)), x)
        )
        x = self.linear_layer(
            npctx, self.w[3], self.b[3].reshape((-1,)
                                                + num_dim * (1,)), x)

        temp = (
            self.state_std[0].reshape((-1,) + num_dim * (1,)) * x[0]
            + self.state_mean[0].reshape((-1,) + num_dim * (1,))
        )

        mass_frac = (
            self.state_std[1:].reshape((-1,) + num_dim * (1,)) * x[1:]
            + self.state_mean[1:].reshape((-1,) + num_dim * (1,))
        )

        return temp, mass_frac
