"""Test the EOS interfaces."""

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

import logging
import numpy as np
import numpy.linalg as la  # noqa
import pyopencl as cl
import pyopencl.clrandom
import pyopencl.clmath
import pytest
from pytools.obj_array import make_obj_array

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.dof_array import thaw
from meshmode.array_context import PyOpenCLArrayContext
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

from mirgecom.prometheus import UIUCMechanism
from mirgecom.eos import IdealSingleGas, PrometheusMixture
from mirgecom.initializers import (
    Vortex2D, Lump,
    MixtureInitializer
)
from mirgecom.euler import split_conserved
from grudge.eager import EagerDGDiscretization
from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)
import cantera

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("y0", [0, 1])
def test_pyrometheus_uiuc(ctx_factory, y0):
    """Test pyrometheus uiuc mechanism.

    Tests that the pyrometheus uiuc mechanism
    gets the same thermo properties as Cantera.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    dim = 1
    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, n=(nel_1d,) * dim
    )

    order = 4

    logger.info(f"Number of elements {mesh.nelements}")

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())

    # Init soln with Vortex
    prometheus_mechanism = UIUCMechanism(actx.np)
    nspecies = prometheus_mechanism.num_species
    print(f"PrometheusMixture::NumSpecies = {nspecies}")

    press = 101500.0
    tempin = 300.0
    y0s = np.zeros(shape=(nspecies,))
    for i in range(1, nspecies):
        y0s[i] = y0 / (10.0 ** i)
    spec_sum = sum([y0s[i] for i in range(1, nspecies)])
    y0s[0] = 1.0 - spec_sum

    cantera_soln = cantera.Solution("uiuc.cti", "gas")
    cantera_soln.TPX = tempin, press, y0s
    cantera_soln.equilibrate("UV")
    can_t, can_rho, can_y = cantera_soln.TDY
    can_p = cantera_soln.P
    can_e = cantera_soln.int_energy_mass

    ones = (1.0 + nodes[0]) - nodes[0]
    tin = can_t * ones
    pin = can_p * ones
    yin = make_obj_array([can_y[i] * ones for i in range(nspecies)])

    prom_rho = prometheus_mechanism.get_density(pin, tin, yin)
    prom_e = prometheus_mechanism.get_mixture_internal_energy_mass(tin, yin)
    prom_t = prometheus_mechanism.get_temperature(prom_e, tin, yin, True)
    prom_p = prometheus_mechanism.get_pressure(prom_rho, tin, yin)

    print(f"can(rho, y, p, t) = ({can_rho}, {can_y}, {can_p}, {can_t}, {can_e})")
    print(f"prom(rho, y, p, t) = ({prom_rho}, {y0s}, {prom_p}, {prom_t}, {prom_e})")

    tol = 1e-6
    assert discr.norm((can_t - prom_t) / can_t, np.inf) < tol
    assert discr.norm((can_rho - prom_rho) / can_rho, np.inf) < tol
    assert discr.norm((can_p - prom_p) / can_p, np.inf) < tol
    assert discr.norm((can_e - prom_e) / can_e, np.inf) < tol


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("y0", [0, 1])
@pytest.mark.parametrize("vel", [0.0, 1.0])
def test_pyrometheus_eos_uiuc(ctx_factory, dim, y0, vel):
    """Test PyrometheusMixture EOS for uiuc mechanism.

    Tests that the pyrometheus uiuc mechanism
    gets the same thermo properties as Cantera.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, n=(nel_1d,) * dim
    )

    order = 4

    logger.info(f"Number of elements {mesh.nelements}")

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())

    # Init soln with Vortex
    prometheus_mechanism = UIUCMechanism(actx.np)
    nspecies = prometheus_mechanism.num_species
    print(f"PrometheusMixture::NumSpecies = {nspecies}")

    press = 101500.0
    tempin = 300.0
    y0s = np.zeros(shape=(nspecies,))
    for i in range(1, nspecies):
        y0s[i] = y0 / (10.0 ** i)
    spec_sum = sum([y0s[i] for i in range(1, nspecies)])
    y0s[0] = 1.0 - spec_sum
    velocity = vel * np.ones(shape=(dim,))

    cantera_soln = cantera.Solution("uiuc.cti", "gas")
    cantera_soln.TPX = tempin, press, y0s
    cantera_soln.equilibrate("UV")
    can_t, can_rho, can_y = cantera_soln.TDY
    can_p = cantera_soln.P
    can_e = cantera_soln.int_energy_mass

    ones = (1.0 + nodes[0]) - nodes[0]
    tin = can_t * ones
    pin = can_p * ones
    yin = make_obj_array([can_y[i] * ones for i in range(nspecies)])

    pyro_rho = prometheus_mechanism.get_density(pin, tin, yin)
    pyro_e = prometheus_mechanism.get_mixture_internal_energy_mass(tin, yin)
    pyro_t = prometheus_mechanism.get_temperature(pyro_e, tin, yin, True)
    pyro_p = prometheus_mechanism.get_pressure(pyro_rho, tin, yin)

    print(f"can(rho, y, p, t, e) = ({can_rho}, {can_y}, {can_p}, {can_t}, {can_e})")
    print(f"prom(rho, y, p, t, e) = ({pyro_rho}, {y0s},"
          f" {pyro_p}, {pyro_t}, {pyro_e})")

    eos = PrometheusMixture(prometheus_mechanism)
    initializer = MixtureInitializer(numdim=dim, nspecies=nspecies,
                                     pressure=can_p, temperature=can_t,
                                     massfractions=can_y, velocity=velocity)

    q = initializer(eos=eos, t=0, x_vec=nodes)
    cv = split_conserved(dim, q)

    p = eos.pressure(cv)
    temperature = eos.temperature(cv)
    internal_energy = eos.get_internal_energy(tin, yin)

    print(f"pyro_eos.p = {p}")
    print(f"pyro_eos.temp = {temperature}")
    print(f"pyro_eos.e = {internal_energy}")

    tol = 1e-6
    assert discr.norm((temperature - pyro_t) / pyro_t, np.inf) < tol
    assert discr.norm((internal_energy - pyro_e) / pyro_e, np.inf) < tol
    assert discr.norm((p - pyro_p) / pyro_p, np.inf) < tol


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_idealsingle_lump(ctx_factory, dim):
    """Test IdealSingle EOS with mass lump.

    Tests that the IdealSingleGas EOS returns
    the correct (uniform) pressure for the Lump
    solution field.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, n=(nel_1d,) * dim
    )

    order = 3
    logger.info(f"Number of elements {mesh.nelements}")

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())

    # Init soln with Vortex
    center = np.zeros(shape=(dim,))
    velocity = np.zeros(shape=(dim,))
    velocity[0] = 1
    lump = Lump(numdim=dim, center=center, velocity=velocity)
    eos = IdealSingleGas()
    lump_soln = lump(0, nodes)

    cv = split_conserved(dim, lump_soln)
    p = eos.pressure(cv)
    exp_p = 1.0
    errmax = discr.norm(p - exp_p, np.inf)

    exp_ke = 0.5 * cv.mass
    ke = eos.kinetic_energy(cv)
    kerr = discr.norm(ke - exp_ke, np.inf)

    te = eos.total_energy(cv, p)
    terr = discr.norm(te - cv.energy, np.inf)

    logger.info(f"lump_soln = {lump_soln}")
    logger.info(f"pressure = {p}")

    assert errmax < 1e-15
    assert kerr < 1e-15
    assert terr < 1e-15


def test_idealsingle_vortex(ctx_factory):
    r"""Test EOS with isentropic vortex.

    Tests that the IdealSingleGas EOS returns
    the correct pressure (p) for the Vortex2D solution
    field (i.e. $p = \rho^{\gamma}$).
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    dim = 2
    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=[(0.0,), (-5.0,)], b=[(10.0,), (5.0,)], n=(nel_1d,) * dim
    )

    order = 3
    logger.info(f"Number of elements {mesh.nelements}")

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())
    eos = IdealSingleGas()
    # Init soln with Vortex
    vortex = Vortex2D()
    vortex_soln = vortex(0, nodes)
    cv = split_conserved(dim, vortex_soln)
    gamma = eos.gamma()
    p = eos.pressure(cv)
    exp_p = cv.mass ** gamma
    errmax = discr.norm(p - exp_p, np.inf)

    exp_ke = 0.5 * np.dot(cv.momentum, cv.momentum) / cv.mass
    ke = eos.kinetic_energy(cv)
    kerr = discr.norm(ke - exp_ke, np.inf)

    te = eos.total_energy(cv, p)
    terr = discr.norm(te - cv.energy, np.inf)

    logger.info(f"vortex_soln = {vortex_soln}")
    logger.info(f"pressure = {p}")

    assert errmax < 1e-15
    assert kerr < 1e-15
    assert terr < 1e-15
