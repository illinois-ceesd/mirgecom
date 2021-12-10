"""Test lazy vs. eager pyrometheus."""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
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
import pyopencl as cl
import pyopencl.tools as cl_tools

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.array_context import (  # noqa
    PyOpenCLArrayContext,
    SingleGridWorkBalancingPytatoArrayContext as PytatoPyOpenCLArrayContext
)
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

import cantera


if __name__ == "__main__":
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    eager_actx = PyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    lazy_actx = PytatoPyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    from mirgecom.mechanisms import get_mechanism_cti
    mech_cti = get_mechanism_cti("uiuc")
    cantera_soln1 = cantera.Solution(phase_id="gas", source=mech_cti)
    cantera_soln2 = cantera.Solution(phase_id="gas", source=mech_cti)

    from mirgecom.thermochemistry import make_pyrometheus_mechanism
    eager_pyro = make_pyrometheus_mechanism(eager_actx, cantera_soln1)
    lazy_pyro = make_pyrometheus_mechanism(lazy_actx, cantera_soln2)

    from pytools.obj_array import make_obj_array

    def lazy_temperature(energy, y, tguess):
        return make_obj_array([lazy_pyro.get_temperature(energy, y, tguess,
                                                         do_energy=True)])

    lazy_temp = lazy_actx.compile(lazy_temperature)

    print(f"{type(lazy_temp)=}")
    input_file = "pyro_state_data.txt"
    input_data = np.loadtxt(input_file)
    #  num_samples = len(input_data)
    num_samples = 1

    print(f"{num_samples=}")

    # time = input_data[:num_samples, 0]
    mass_frac = input_data[:num_samples, 1:-1]
    temp = input_data[:num_samples, -1]
    initial_temp = temp[0]

    use_old_temperature = False

    lazy_eager_diff_tol = 1e-4

    for i, t, y in zip(range(num_samples), temp, mass_frac):

        t_old = initial_temp
        if use_old_temperature and i > 0:
            t_old = temp[i-1]

        #  rho = ptk.get_density(cantera.one_atm, t, y)
        e_int = eager_pyro.get_mixture_internal_energy_mass(t, y)
        print(f"{type(e_int)=}")
        print(f"{type(y)=}")
        t_eager = eager_pyro.get_temperature(e_int, t_old, y,
                                             do_energy=True)
        t_lazy = lazy_actx.to_numpy(lazy_actx.freeze(lazy_temp(e_int, t_old,
                           np.asarray(lazy_actx.from_numpy(y)))[0]))
        err = np.abs(t_eager - t_lazy)/t_eager
        print(f"{err=}")
        assert err < lazy_eager_diff_tol
        # w_pyro = ptk.get_net_production_rates(rho, t_pyro, y)
