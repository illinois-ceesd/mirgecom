__copyright__ = """
Copyright (C) 2023 University of Illinois Board of Trustees
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

import numpy as np

#    # FIXME
#    class Bprime_table():

#        def __init__(self):

#            #bprime contains: B_g, B_c, Temperature T, Wall enthalpy H_W
#            bprime_table = (np.genfromtxt('Bprime_table/B_prime.dat', skip_header=1)[:,2:6]).reshape((25,151,4))

#            self._bounds_T = bprime_table[   0,:,2]
#            self._bounds_B = bprime_table[::-1,0,0]
#            self._Bc = bprime_table[::-1,:,1]
#            self._H  = bprime_table[::-1,:,3]
#            self._interp_Bc = scipy.interpolate.RegularGridInterpolator((bprime_table[::-1,0,0], bprime_table[0,:,2]), bprime_table[::-1,:,1])
#            self._interp_Hw = scipy.interpolate.RegularGridInterpolator((bprime_table[::-1,0,0], bprime_table[0,:,2]), bprime_table[::-1,:,3])


class pyrolysis():

    def __init__(self):
        return

    def get_sources(self, temperature, xi):
        return xi*0.0


def solid_enthalpy(temperature, tau):
    return 2e6 + 1500*temperature

def solid_heat_capacity(temperature, tau):       
    return 1500 + temperature*0.0

def solid_thermal_conductivity(temperature, tau):
    return 1.0 + temperature*0.0

def solid_permeability(temperature, tau):
    return 2.0e-11 + temperature*0.0

def solid_tortuosity(temperature, tau):
    return 1.2e-11 + temperature*0.0

def solid_volume_fraction(temperature, tau):
    return 0.2 + temperature*0.0 
 
def solid_emissivity(temperature, tau):
    return 0.85 + temperature*0.0 
