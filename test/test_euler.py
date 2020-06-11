from __future__ import division, absolute_import, print_function

__copyright__ = """Copyright (C) 2020 CEESD"""

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
import numpy.linalg as la  # noqa
import pyopencl as cl
import pyopencl.clmath
import pyopencl.clrandom
from pytools.obj_array import (
    join_fields, make_obj_array,
    with_object_array_or_scalar)
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
# TODO: Remove grudge dependence?
from grudge.eager import with_queue
from grudge.symbolic.primitives import TracePair
from mirgecom.euler import _inviscid_flux_2d
from mirgecom.euler import _inviscid_flux_3d
from mirgecom.euler import inviscid_operator
from mirgecom.euler import _facial_flux
from mirgecom.euler import _interior_trace_pair
from mirgecom.euler import Vortex
from meshmode.discretization import Discretization
from grudge.eager import EagerDGDiscretization
# Tests go here

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)


def test_inviscid_flux_2d():

    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    def scalevec(scalar, vec):
        # workaround for object array behavior
        return make_obj_array([ni * scalar for ni in vec])

    rho = cl.clrandom.rand(queue, (10,), dtype=np.float64)
    rhoE = cl.clrandom.rand(queue, (10,), dtype=np.float64)
    rhoV0 = cl.clrandom.rand(queue, (10,), dtype=np.float64)
    rhoV1 = cl.clrandom.rand(queue, (10,), dtype=np.float64)
    scal1 = cl.clrandom.rand(queue, (10,), dtype=np.float64)
    p = cl.clrandom.rand(queue, (10,), dtype=np.float64)

    rhoV = make_obj_array([rhoV0,rhoV1])
    ke = 0.5*(rhoV0*rhoV0 + rhoV1*rhoV1)/rho

    # ideal single spec
    gamma = 1.4
    p = (gamma-1.0)*(rhoE - ke)
    escale = (rhoE + p)/rho
    expected_mass_flux = rhoV
    expected_energy_flux = scalevec(escale,rhoV)
    expected_mom_flux1 = rhoV0*rhoV0/rho + p
    expected_mom_flux2 = rhoV0*rhoV1/rho # crossterms
    expected_mom_flux4 = rhoV1*rhoV1/rho + p

    q = join_fields(rho, rhoE, rhoV)

    flux = _inviscid_flux_2d(q)

    rhoflux = flux[0:2]
    rhoEflux = flux[2:4]
    rhoV0flux = flux[4:6]
    rhoV1flux = flux[6:]

    # these should be exact, right?
    assert(la.norm(rhoflux[0].get() - expected_mass_flux[0].get()) == 0.0)
    assert(la.norm(rhoflux[1].get() - expected_mass_flux[1].get()) == 0.0)
    assert(la.norm(rhoEflux[0].get() - expected_energy_flux[0].get()) == 0.0)
    assert(la.norm(rhoEflux[1].get() - expected_energy_flux[1].get()) == 0.0)
    assert(la.norm(rhoV0flux[0].get() - expected_mom_flux1.get()) == 0.0)
    assert(la.norm(rhoV0flux[1].get() - expected_mom_flux2.get()) == 0.0) # crossterms
    assert(la.norm(rhoV1flux[0].get() - expected_mom_flux2.get()) == 0.0) # crossterms
    assert(la.norm(rhoV1flux[1].get() - expected_mom_flux4.get()) == 0.0)


def test_inviscid_flux_3d():
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    gamma = 1.4

    def scalevec(scalar, vec):
        # workaround for object array behavior
        return make_obj_array([ni * scalar for ni in vec])

    rho = cl.clrandom.rand(queue, (10,), dtype=np.float64)
    rhoE = cl.clrandom.rand(queue, (10,), dtype=np.float64)
    rhoV0 = cl.clrandom.rand(queue, (10,), dtype=np.float64)
    rhoV1 = cl.clrandom.rand(queue, (10,), dtype=np.float64)
    rhoV2 = cl.clrandom.rand(queue, (10,), dtype=np.float64)
    scal1 = cl.clrandom.rand(queue, (10,), dtype=np.float64)
    p = cl.clrandom.rand(queue, (10,), dtype=np.float64)

    rhoV = make_obj_array([rhoV0,rhoV1,rhoV2])
    ke = 0.5*(rhoV0*rhoV0 + rhoV1*rhoV1 + rhoV2*rhoV2)/rho
    p = (gamma-1.0)*(rhoE - ke) # ideal single spec
    scal1 = (rhoE + p)/rho

    expected_mass_flux = rhoV
    expected_energy_flux = scalevec(scal1,rhoV)
    expected_mom_flux1 = rhoV0*rhoV0/rho + p
    expected_mom_flux2 = rhoV0*rhoV1/rho # crossterm1
    expected_mom_flux3 = rhoV0*rhoV2/rho # crossterm2
    expected_mom_flux5 = rhoV1*rhoV1/rho + p
    expected_mom_flux6 = rhoV1*rhoV2/rho # crossterm3
    expected_mom_flux9 = rhoV2*rhoV2/rho + p

    q = join_fields(rho, rhoE, rhoV)

    flux = _inviscid_flux_3d(q)

    rhoflux = flux[0:3]
    rhoEflux = flux[3:6]
    rhoV0flux = flux[6:9]
    rhoV1flux = flux[9:12]
    rhoV2flux = flux[12:]

    assert(la.norm(rhoflux[0].get() - expected_mass_flux[0].get()) == 0.0)
    assert(la.norm(rhoflux[1].get() - expected_mass_flux[1].get()) == 0.0)
    assert(la.norm(rhoflux[2].get() - expected_mass_flux[2].get()) == 0.0)
    assert(la.norm(rhoEflux[0].get() - expected_energy_flux[0].get()) == 0.0)
    assert(la.norm(rhoEflux[1].get() - expected_energy_flux[1].get()) == 0.0)
    assert(la.norm(rhoEflux[2].get() - expected_energy_flux[2].get()) == 0.0)
    assert(la.norm(rhoV0flux[0].get() - expected_mom_flux1.get()) == 0.0)
    assert(la.norm(rhoV0flux[1].get() - expected_mom_flux2.get()) == 0.0) # crossterm1
    assert(la.norm(rhoV0flux[2].get() - expected_mom_flux3.get()) == 0.0) # crossterm2
    assert(la.norm(rhoV1flux[0].get() - expected_mom_flux2.get()) == 0.0) # crossterm1
    assert(la.norm(rhoV1flux[1].get() - expected_mom_flux5.get()) == 0.0)
    assert(la.norm(rhoV1flux[2].get() - expected_mom_flux6.get()) == 0.0) # crossterm3
    assert(la.norm(rhoV2flux[0].get() - expected_mom_flux3.get()) == 0.0) # crossterm1
    assert(la.norm(rhoV2flux[1].get() - expected_mom_flux6.get()) == 0.0) # crossterm3
    assert(la.norm(rhoV2flux[2].get() - expected_mom_flux9.get()) == 0.0)


def test_facial_flux():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    dim = 2
    nel_1d = 16
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*dim,
            b=(0.5,)*dim,
            n=(nel_1d,)*dim)

    order = 3

    if dim == 2:
        # no deep meaning here, just a fudge factor
        dt = 0.75/(nel_1d*order**2)
    elif dim == 3:
        # no deep meaning here, just a fudge factor
        dt = 0.45/(nel_1d*order**2)
    else:
        raise ValueError("don't have a stable time step guesstimate")

    print("%d elements" % mesh.nelements)

    discr = EagerDGDiscretization(cl_ctx, mesh, order=order)

    mass_input = discr.zeros(queue)
    energy_input = discr.zeros(queue)
    mass_input[:] = 1.0
    energy_input[:] = 2.5
    mom_input = join_fields( [discr.zeros(queue) for i in range(discr.dim) ] )
    fields = join_fields( mass_input, energy_input, mom_input)

    interior_face_flux = _facial_flux(discr,w_tpair=_interior_trace_pair(discr,fields))

    err = np.max(np.array([ la.norm(interior_face_flux[i].get(),np.inf)
                           for i in range(0,2)]))
    assert(err < 1e-15)

    # mom flux should be p = 1
    for i in range(2,2+discr.dim):
        err = np.max(np.array([ la.norm(interior_face_flux[i].get(),np.inf) ]))
        assert((err - 1.0) < 1e-15)

    dir_rho = discr.interp("vol", BTAG_ALL, mass_input)
    dir_e = discr.interp("vol",BTAG_ALL, energy_input)
    dir_mom = discr.interp("vol",BTAG_ALL, mom_input)

    dir_bval = join_fields(dir_rho, dir_e, dir_mom)
    dir_bc = join_fields(dir_rho,dir_e,dir_mom)

    boundary_flux = _facial_flux(discr,w_tpair=TracePair(BTAG_ALL,dir_bval,dir_bc))

    err = np.max(np.array([ la.norm(boundary_flux[i].get(),np.inf)
                           for i in range(0,2)]))
    assert(err < 1e-15)

    # mom flux should be p = 1
    for i in range(2,2+discr.dim):
        err = np.max(np.array([ la.norm(boundary_flux[i].get(),np.inf) ]))
        assert((err - 1.0) < 1e-15)

def test_uniform_flow():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    iotag = 'test_uniform_flow: '

    dim = 2
    nel_1d = 16
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*dim,
            b=(0.5,)*dim,
            n=(nel_1d,)*dim)

    order = 3

    if dim == 2:
        # no deep meaning here, just a fudge factor
        dt = 0.75/(nel_1d*order**2)
    elif dim == 3:
        # no deep meaning here, just a fudge factor
        dt = 0.45/(nel_1d*order**2)
    else:
        raise ValueError("don't have a stable time step guesstimate")

    print(iotag+"%d elements" % mesh.nelements)

    discr = EagerDGDiscretization(cl_ctx, mesh, order=order)

    mass_input = discr.zeros(queue)
    energy_input = discr.zeros(queue)
    mass_input[:] = 1.0
    energy_input[:] = 2.5
    mom_input = join_fields( [discr.zeros(queue) for i in range(discr.dim) ] )
    fields = join_fields( mass_input, energy_input, mom_input)

    expected_rhs = join_fields( [discr.zeros(queue) for i in range(discr.dim+2) ] )

    inviscid_rhs = inviscid_operator(discr,fields)

#    print('inviscid_rhs shape = ',inviscid_rhs.shape)
#    print('inviscid_rhs rho = ',inviscid_rhs[0])
#    print('inviscid_rhs rhoE = ',inviscid_rhs[1])
#    print('inviscid_rhs mom1 = ',inviscid_rhs[2])
#    print('inviscid_rhs mom2 = ',inviscid_rhs[3])

    rhs_resid = inviscid_rhs - expected_rhs
    rho_resid = rhs_resid[0]
    rhoe_resid = rhs_resid[1]
    mom_resid = rhs_resid[2:]
    rho_rhs = inviscid_rhs[0]
    rhoe_rhs = inviscid_rhs[1]
    rhov_rhs = inviscid_rhs[2:]

    print(iotag+'rho_rhs = ',rho_rhs)
    print(iotag+'rhoe_rhs = ',rhoe_rhs)
    print(iotag+'rhov_rhs = ',rhov_rhs)

    assert(la.norm(rho_resid.get()) < 1e-9)
    assert(la.norm(rhoe_resid.get()) < 1e-9)
    assert(la.norm(mom_resid[0].get()) < 1e-9)
    assert(la.norm(mom_resid[1].get()) < 1e-9)

    # set a non-zero, but uniform velocity component
    i = 0
    for mom_component in mom_input:
        mom_component[:] = (-1.0)**i
        i = i + 1

    inviscid_rhs = inviscid_operator(discr,fields)
    rhs_resid = inviscid_rhs - expected_rhs
    rho_resid = rhs_resid[0]
    rhoe_resid = rhs_resid[1]
    mom_resid = rhs_resid[2:]

    assert(la.norm(rho_resid.get()) < 1e-9)
    assert(la.norm(rhoe_resid.get()) < 1e-9)
    assert(la.norm(mom_resid[0].get()) < 1e-9)
    assert(la.norm(mom_resid[1].get()) < 1e-9)

    # next test lump propagation


def test_isentropic_vortex(ctx_factory):

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    iotag = 'test_isentropic_vortex: '
    dim = 2
    nel_1d = 16
    from meshmode.mesh.generation import generate_regular_rect_mesh

    for nel_1d in [4, 8, 16]:
        mesh = generate_regular_rect_mesh(
            a=(-0.5,)*dim,
            b=(0.5,)*dim,
            n=(nel_1d,)*dim)

        order = 3

        if dim == 2:
            # no deep meaning here, just a fudge factor
            dt = 0.75/(nel_1d*order**2)
        elif dim == 3:
            # no deep meaning here, just a fudge factor
            dt = 0.45/(nel_1d*order**2)
        else:
            raise ValueError("don't have a stable time step guesstimate")

        print(iotag+"%d elements" % mesh.nelements)

        discr = EagerDGDiscretization(cl_ctx, mesh, order=order)
        nodes = discr.nodes().with_queue(queue)

        # Init soln with Vortex
        mass_input = discr.zeros(queue)
        energy_input = discr.zeros(queue)
        mom_input = join_fields( [discr.zeros(queue) for i in range(discr.dim) ] )
        vortex_soln = join_fields( [discr.zeros(queue) for i in range(discr.dim+2) ] )
        #from functools import partial
        #myvortex = partial(vortex, par1=stuff, par2=stuff, par3=stuff)
        vortex_soln = myvortex(t, nodes)
        vortex = Vortex()
        for i in range(discr.dim):
            vortex_soln = vortex_soln + vortex(0,nodes)

        fields = vortex_soln

        # Hardcode failure for now
        assert(False)


# PYOPENCL_TEST=port python -m pudb test_euler.py 'test_isentropic_vortex(cl._csc)'
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
