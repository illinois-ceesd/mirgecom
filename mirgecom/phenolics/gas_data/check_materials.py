
import numpy as np

import matplotlib.pyplot as plt
import sys

from scipy.interpolate import CubicSpline

from mirgecom.phenolics.gas import gas_properties


def eval_spline(x, x_bnds, coeffs):

    val = x*0.0
    nidx = len(x_bnds)
    for i in range(0,nidx-1):
        val = np.where(x < x_bnds[i+1],
                    np.where(x >= x_bnds[i],
                        coeffs[0,i]*(x-x_bnds[i])**3 + coeffs[1,i]*(x-x_bnds[i])**2 + coeffs[2,i]*(x-x_bnds[i]) + coeffs[3,i],
                        0.0),
                    0.0) + val

    return val

def eval_spline_derivative(x, x_bnds, coeffs):

    val = x*0.0
    nidx = len(x_bnds)
    for i in range(0,nidx-1):
        val = np.where(x < x_bnds[i+1],
                    np.where(x >= x_bnds[i],
                        3.0*coeffs[0,i]*(x-x_bnds[i])**2 + 2.0*coeffs[1,i]*(x-x_bnds[i]) + coeffs[2,i],
                        0.0),
                    0.0) + val

    return val



gas_py = gas_properties()
gas_data = gas_py._data

x = gas_data[:,0]
M = gas_data[:,1]
Cp = gas_data[:,2]*1000.0
gamma = gas_data[:,3]
h = gas_data[:,4]*1000.0
mu = gas_data[:,5]*1e-4
rho = gas_data[:,6]
T = np.linspace(200,3200.0,201)

N = T.shape[0]

#idx = [0, 6, 12, 18, 20, 22, 24, 26, 28, 31, 33, 36, 38, 40, 43,
#        48, 53, 55, 57, 60, 63, 69, 72]
#idx.extend(list(range(78,127,6)))
##print(np.array([x[idx],idx]).T)
##print(len(x[idx]))

"""."""
plt.plot(x, h, marker='o')
cs = CubicSpline(x, h)
gas_enthalpy = eval_spline(T, cs.x, cs.c)
plt.plot(T, gas_enthalpy, marker='x')
mirgecom = gas_py.gas_enthalpy(T)
plt.plot(T, mirgecom, marker='+')
plt.show()

plt.plot(x, Cp, marker='o')
heat_capacity = eval_spline_derivative(T, cs.x, cs.c)
plt.plot(T, heat_capacity, marker='x')
mirgecom = gas_py.gas_heat_capacity(T)
plt.plot(T, mirgecom, marker='+')
plt.show()

"""."""
plt.plot(x, mu, marker='o')
cs = CubicSpline(x, mu)
viscosity = eval_spline(T, cs.x, cs.c)
plt.plot(T, viscosity, marker='x')
mirgecom = gas_py.gas_viscosity(T)
plt.plot(T, mirgecom, marker='+')
plt.show()

"""."""
plt.plot(x, M, marker='o')
cs = CubicSpline(x, M)
molar_mass = eval_spline(T, cs.x, cs.c)
plt.plot(T, molar_mass, marker='x')
mirgecom = gas_py.gas_molar_mass(T)
plt.plot(T, mirgecom, marker='+')
plt.show()

dMdT = eval_spline_derivative(T, cs.x, cs.c)
plt.plot(T, dMdT, marker='x')
plt.show()

