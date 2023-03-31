
import numpy as np

import matplotlib.pyplot as plt
import sys

from scipy.interpolate import CubicSpline

from mirgecom.phenolics.gas import gas_properties

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

skip=6
print(x)
#idx = [0, 5, 10, 15, 16, 20, 25, 27, 30, 33, 36, 40, 43, 48, 53, 55, 57]
idx = [0, 6, 12, 18, 20, 22, 24, 26, 28, 31, 33, 36, 38, 40, 43,
        48, 53, 55, 57, 60, 63, 69, 72]
idx.extend(list(range(78,121,skip)))
print(np.array([x[idx],idx]).T)
print(len(x[idx]))

plt.plot(x, h, marker='o')
cs = CubicSpline(x[idx], h[idx])
gas_enthalpy = cs(T)
plt.plot(T, gas_enthalpy, marker='x')
plt.show()

plt.plot(x, Cp, marker='o')
cs = cs.derivative(nu=1)
heat_capacity = cs(T)
plt.plot(T, heat_capacity, marker='x')
plt.show()

plt.plot(x, mu, marker='o')
cs = CubicSpline(x[idx], mu[idx])
viscosity = cs(T)
plt.plot(T, viscosity, marker='x')
plt.show()

plt.plot(x, M, marker='o')
cs = CubicSpline(x[idx], M[idx])
molar_mass = cs(T)
plt.plot(T, molar_mass, marker='x')
plt.show()

cs = cs.derivative(nu=1)
dMdT = cs(T)
plt.plot(T, dMdT, marker='x')
plt.show()

