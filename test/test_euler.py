from __future__ import division, absolute_import, print_function

__copyright__ = (
    """Copyright (C) 2020 University of Illinois Board of Trustees"""
)

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
import pyopencl.clrandom
import pyopencl.clmath
from pytools.obj_array import (
    join_fields,
    make_obj_array,
    with_object_array_or_scalar,
)
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

# TODO: Remove grudge dependence?
from grudge.eager import with_queue
from grudge.symbolic.primitives import TracePair
from mirgecom.euler import inviscid_operator
from mirgecom.initializers import Vortex2D
from mirgecom.initializers import Lump
from mirgecom.eos import IdealSingleGas
from meshmode.discretization import Discretization
from grudge.eager import EagerDGDiscretization
from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)

def test_inviscid_flux():

    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    iotag = "test_inviscid_flux: "

    def scalevec(scalar, vec):
        # workaround for object array behavior
        return make_obj_array([ni * scalar for ni in vec])

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    dim = 2
    nel_1d = 16

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, n=(nel_1d,) * dim
    )

    order = 3

    dt = (1.0 - 0.25 * (dim - 1)) / (nel_1d * order ** 2)

    print(iotag + "%d elements." % mesh.nelements)
    print(iotag + "Timestep: ", dt)

    discr = EagerDGDiscretization(cl_ctx, mesh, order=order)
    eos = IdealSingleGas()

    rho = cl.clrandom.rand(queue, (mesh.nelements,), dtype=np.float64)
    rhoE = cl.clrandom.rand(queue, (mesh.nelements,), dtype=np.float64)
    rhoV = make_obj_array(
        [
            cl.clrandom.rand(queue, (mesh.nelements,), dtype=np.float64)
            for i in range(dim)
        ]
    )

    q = join_fields(rho, rhoE, rhoV)
    p = eos.Pressure(q)

    escale = (rhoE + p) / rho

    expected_mass_flux = rhoV
    expected_energy_flux = scalevec(escale, rhoV)
    expected_mom_flux = make_obj_array(
        [
            (rhoV[i] * rhoV[j] / rho + (p if i == j else 0))
            for i in range(dim)
            for j in range(dim)
        ]
    )
    expected_flux = join_fields(
        expected_mass_flux, expected_energy_flux, expected_mom_flux
    )

    from mirgecom.euler import _inviscid_flux

    flux = _inviscid_flux(discr, q, eos)
    flux_resid = flux - expected_flux

    for i in range((dim + 2) * dim):
        assert (la.norm(flux_resid[i].get())) == 0.0

    class mydiscr:
        def __init__(self, dim=1):
            self.dim = dim

    p0 = 1.0
    # === this next block tests 1,2,3 dimensions, single and multiple nodes
    # === with V = 0, fixed P = p0
    for dim in [1, 2, 3]:
        for ntestnodes in [1, 10, 100]:
            fake_dis = mydiscr(dim)
            rho = cl.clrandom.rand(
                queue, (ntestnodes,), dtype=np.float64
            )
            rhoE = cl.clrandom.rand(
                queue, (ntestnodes,), dtype=np.float64
            )
            rhoV = make_obj_array(
                [
                    cl.clrandom.rand(
                        queue, (ntestnodes,), dtype=np.float64
                    )
                    for i in range(dim)
                ]
            )
            p = cl.clrandom.rand(queue, (ntestnodes,), dtype=np.float64)

            for i in range(ntestnodes):
                rho[i] = 1.0 + i
                p[i] = p0
                for j in range(dim):
                    rhoV[j][i] = 0.0 * rho[i]

            v = scalevec(1 / rho, rhoV)
            rhoE = p / 0.4 + 0.5 * np.dot(rhoV, rhoV) / rho
            q = join_fields(rho, rhoE, rhoV)
            flux = _inviscid_flux(fake_dis, q, eos)

            print(iotag + str(dim) + "d flux = ", flux)

            # for velocity zero, these components should be == zero
            for i in range(2 * dim):
                for j in range(ntestnodes):
                    assert flux[i][j].get() == 0.0

            # The momentum diagonal should be p
            for i in range(dim):
                for j in range(dim):
                    if i != j:
                        pterm = np.zeros(shape=(ntestnodes,))
                    else:
                        pterm = p
                    for n in range(ntestnodes):
                        assert (
                            np.abs(
                                flux[(2 + i) * dim + j][n].get()
                                - pterm[n]
                            )
                            < 1e-15
                        )

    # === this next block tests 1,2,3 dimensions, single and multiple nodes
    # === with V = 1 (in some dimension), fixed P = p0
    for dim in [1, 2, 3]:
        for livedim in range(dim):
            for ntestnodes in [1, 10, 100]:
                fake_dis = mydiscr(dim)
                rho = cl.clrandom.rand(
                    queue, (ntestnodes,), dtype=np.float64
                )
                rhoE = cl.clrandom.rand(
                    queue, (ntestnodes,), dtype=np.float64
                )
                rhoV = make_obj_array(
                    [
                        cl.clrandom.rand(
                            queue, (ntestnodes,), dtype=np.float64
                        )
                        for i in range(dim)
                    ]
                )
                p = cl.clrandom.rand(
                    queue, (ntestnodes,), dtype=np.float64
                )

                for i in range(ntestnodes):
                    rho[i] = 1.0 + i
                    p[i] = p0
                    for j in range(dim):
                        rhoV[j][i] = 0.0 * rho[i]
                    rhoV[livedim][i] = rho[i]

                v = scalevec(1 / rho, rhoV)
                rhoE = p / 0.4 + 0.5 * np.dot(rhoV, rhoV) / rho
                q = join_fields(rho, rhoE, rhoV)
                flux = _inviscid_flux(fake_dis, q, eos)

                print(iotag + str(dim) + "d flux = ", flux)

                # first two components should be nonzero in livedim only
                expected_flux = scalevec(rho, v)
                for i in range(dim):
                    for j in range(ntestnodes):
                        assert (
                            np.abs(
                                flux[i][j].get()
                                - expected_flux[i][j].get()
                            )
                            < 1e-13
                        )

                expected_flux = scalevec((rhoE + p), v)
                for i in range(dim):
                    for j in range(ntestnodes):
                        assert (
                            np.abs(
                                flux[dim + i][j].get()
                                - expected_flux[i][j].get()
                            )
                            < 1e-13
                        )
                # The momentum diagonal should be p
                for i in range(dim):
                    expected_flux = scalevec(rho * v[i], v)
                    expected_flux[i] = expected_flux[i] + p
                    for j in range(dim):
                        for k in range(ntestnodes):
                            assert (
                                np.abs(
                                    flux[2 * dim + i * dim + j][k].get()
                                    - expected_flux[j][k].get()
                                )
                                < 1e-13
                            )


def test_facial_flux():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    iotag = "test_facial_flux: "

    dim = 2
    nel_1d = 16

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, n=(nel_1d,) * dim
    )

    order = 3

    dt = (1.0 - 0.25 * (dim - 1)) / (nel_1d * order ** 2)

    print(iotag + "%d elements." % mesh.nelements)
    print(iotag + "Timestep: ", dt)

    discr = EagerDGDiscretization(cl_ctx, mesh, order=order)

    mass_input = discr.zeros(queue)
    energy_input = discr.zeros(queue)
    mom_input = join_fields(
        [discr.zeros(queue) for i in range(discr.dim)]
    )

    # This sets p = 1
    mass_input[:] = 1.0
    energy_input[:] = 2.5

    fields = join_fields(mass_input, energy_input, mom_input)

    from mirgecom.euler import _facial_flux
    from mirgecom.euler import _interior_trace_pair

    interior_face_flux = _facial_flux(
        discr, w_tpair=_interior_trace_pair(discr, fields)
    )

    err = np.max(
        np.array(
            [
                la.norm(interior_face_flux[i].get(), np.inf)
                for i in range(0, 2)
            ]
        )
    )

    assert err < 1e-15

    # mom flux max should be p = 1 + (any interp error)
    for i in range(2, 2 + discr.dim):
        err = np.max(
            np.array([la.norm(interior_face_flux[i].get(), np.inf)])
        )
        assert (err - 1.0) < 1e-15

    # Check the boundary facial fluxes as called on a boundary
    dir_rho = discr.interp("vol", BTAG_ALL, mass_input)
    dir_e = discr.interp("vol", BTAG_ALL, energy_input)
    dir_mom = discr.interp("vol", BTAG_ALL, mom_input)

    dir_bval = join_fields(dir_rho, dir_e, dir_mom)
    dir_bc = join_fields(dir_rho, dir_e, dir_mom)

    from mirgecom.euler import _facial_flux

    boundary_flux = _facial_flux(
        discr, w_tpair=TracePair(BTAG_ALL, dir_bval, dir_bc)
    )

    err = np.max(
        np.array(
            [
                la.norm(boundary_flux[i].get(), np.inf)
                for i in range(0, 2)
            ]
        )
    )
    assert err < 1e-15

    # mom flux should be p = 1
    for i in range(2, 2 + discr.dim):
        err = np.max(
            np.array([la.norm(boundary_flux[i].get(), np.inf)])
        )
        assert (err - 1.0) < 1e-15

    # set random inputs


def test_uniform_rhs():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    iotag = "test_uniform_flow: "

    dim = 2
    nel_1d = 16
    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, n=(nel_1d,) * dim
    )

    order = 3

    if dim == 2:
        # no deep meaning here, just a fudge factor
        dt = 0.75 / (nel_1d * order ** 2)
    elif dim == 3:
        # no deep meaning here, just a fudge factor
        dt = 0.45 / (nel_1d * order ** 2)
    else:
        raise ValueError("don't have a stable time step guesstimate")

    print(iotag + "%d elements" % mesh.nelements)

    discr = EagerDGDiscretization(cl_ctx, mesh, order=order)

    mass_input = discr.zeros(queue)
    energy_input = discr.zeros(queue)
    mass_input[:] = 1.0
    energy_input[:] = 2.5
    mom_input = join_fields(
        [discr.zeros(queue) for i in range(discr.dim)]
    )
    fields = join_fields(mass_input, energy_input, mom_input)

    expected_rhs = join_fields(
        [discr.zeros(queue) for i in range(discr.dim + 2)]
    )

    inviscid_rhs = inviscid_operator(discr, fields)

    rhs_resid = inviscid_rhs - expected_rhs
    rho_resid = rhs_resid[0]
    rhoe_resid = rhs_resid[1]
    mom_resid = rhs_resid[2:]
    rho_rhs = inviscid_rhs[0]
    rhoe_rhs = inviscid_rhs[1]
    rhov_rhs = inviscid_rhs[2:]

    print(iotag + "rho_rhs = ", rho_rhs)
    print(iotag + "rhoe_rhs = ", rhoe_rhs)
    print(iotag + "rhov_rhs = ", rhov_rhs)

    assert la.norm(rho_resid.get()) < 1e-9
    assert la.norm(rhoe_resid.get()) < 1e-9
    assert la.norm(mom_resid[0].get()) < 1e-9
    assert la.norm(mom_resid[1].get()) < 1e-9

    # set a non-zero, but uniform velocity component
    i = 0
    for mom_component in mom_input:
        mom_component[:] = (-1.0) ** i
        i = i + 1

    inviscid_rhs = inviscid_operator(discr, fields)
    rhs_resid = inviscid_rhs - expected_rhs
    rho_resid = rhs_resid[0]
    rhoe_resid = rhs_resid[1]
    mom_resid = rhs_resid[2:]

    assert la.norm(rho_resid.get()) < 1e-9
    assert la.norm(rhoe_resid.get()) < 1e-9
    assert la.norm(mom_resid[0].get()) < 1e-9
    assert la.norm(mom_resid[1].get()) < 1e-9

    # next test lump propagation


# def test_isentropic_vortex(ctx_factory):
def test_vortex_rhs():

    #    cl_ctx = ctx_factory()
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    iotag = "test_vortex_rhs: "
    dim = 2
    nel_1d = 16
    order = 4

    for order in [1, 2, 3]:
        from pytools.convergence import EOCRecorder

        eoc_rec = EOCRecorder()

        from meshmode.mesh.generation import generate_regular_rect_mesh

        for nel_1d in [16, 32, 64]:
            mesh = generate_regular_rect_mesh(
                a=[(0.0,), (-5.0,)],
                b=[(10.0,), (5.0,)],
                n=(nel_1d,) * dim,
            )

            print(iotag + "%d elements" % mesh.nelements)

            discr = EagerDGDiscretization(cl_ctx, mesh, order=order)
            nodes = discr.nodes().with_queue(queue)

            # Init soln with Vortex and expected RHS = 0
            vortex = Vortex2D(center=[5, 0], velocity=[0, 0])
            #            vortex.SetBoundaryTag(BTAG_ALL)
            vortex_soln = vortex(0, nodes)
            boundaries = {BTAG_ALL: vortex}
            #            boundaryboss = BoundaryBoss()
            #            boundaryboss.AddBoundary(vortex)

            inviscid_rhs = inviscid_operator(
                discr, vortex_soln, t=0, boundaries=boundaries
            )

            # - these are only used in viz
            # gamma = 1.4
            # rho = vortex_soln[0]
            # rhoE = vortex_soln[1]
            # rhoV = vortex_soln[2:]
            # p = 0.4 * (rhoE - 0.5*np.dot(rhoV,rhoV)/rho)
            # exp_p = rho ** gamma

            err_max = np.max(
                np.array(
                    [
                        la.norm(inviscid_rhs[i].get(), np.inf)
                        for i in range(dim + 2)
                    ]
                )
            )

            #        eoc_rec.add_data_point(1./n, err)
            #            err_rhs = [ np.abs(inviscid_rhs[i].get()) for i in range(dim + 2) ]
            #            for i in range(dim + 2):
            #                err_rhs = inviscid_rhs[i]
            #                err_max = np.max(np.abs(err_rhs.get()))
            #                print(iotag + "err_max_" + str(nel_1d) + " = ", err_max)

            eoc_rec.add_data_point(1.0 / nel_1d, err_max)

        print(
            iotag
            + "Approxiation errors for order ("
            + str(order)
            + "):"
        )
        print(eoc_rec)
        assert (
            eoc_rec.order_estimate() >= order - 0.5
            or eoc_rec.max_error() < 1e-11
        )


def test_lump_rhs():
    #    cl_ctx = ctx_factory()
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    iotag = "test_lump_rhs: "
    dim = 2
    nel_1d = 16
    order = 4

    for order in [1, 2, 3]:

        from meshmode.mesh.generation import generate_regular_rect_mesh

        for nel_1d in [4, 8, 16]:
            mesh = generate_regular_rect_mesh(
                a=[(0.0,), (-5.0,)],
                b=[(10.0,), (5.0,)],
                n=(nel_1d,) * dim,
            )

            print(iotag + "%d elements" % mesh.nelements)

            discr = EagerDGDiscretization(cl_ctx, mesh, order=order)
            nodes = discr.nodes().with_queue(queue)

            # Init soln with Vortex and expected RHS = 0
            center = np.zeros(shape=(dim,))
            velocity = np.zeros(shape=(dim,))
            center[0] = 5
            lump = Lump(center=center, velocity=velocity)
            #            lump.SetBoundaryTag(BTAG_ALL)
            lump_soln = lump(0, nodes)
            #            boundaryboss = BoundaryBoss()
            #            boundaryboss.AddBoundary(lump)
            boundaries = {BTAG_ALL: lump}

            inviscid_rhs = inviscid_operator(
                discr, lump_soln, t=0.0, boundaries=boundaries
            )

            expected_rhs = lump.ExpectedRHS(discr, lump_soln, 0)
            # - these are only used in viz
            # gamma = 1.4
            # rho = lump_soln[0]
            # rhoE = lump_soln[1]
            # rhoV = lump_soln[2:]
            # p = 0.4 * (rhoE - 0.5*np.dot(rhoV,rhoV)/rho)
            # exp_p = 1.0
            # from grudge.shortcuts import make_visualizer
            # vis = make_visualizer(discr, discr.order)
            # vis.write_vtk_file(iotag+'_{n}.vtu'.format(n=nel_1d),
            #         [
            #             ("rho", lump_soln[0]),
            #             ("rhoE", lump_soln[1]),
            #             ("rhoV", lump_soln[2:]),
            #             ("rhorhs", inviscid_rhs[0]),
            #             ("rhoerhs", inviscid_rhs[1]),
            #             ("rhovrhs", inviscid_rhs[2:]),
            #             ("rhorhs_expected", expected_rhs[0]),
            #             ("rhoerhs_expected", expected_rhs[1]),
            #             ("rhovrhs_expected", expected_rhs[2:]),
            #             ("pressure",p),
            #             ])

            for i in range(dim + 2):
                err_rhs = la.norm(
                    (inviscid_rhs[i] - expected_rhs[i]).get(), np.inf
                )
                err_max = np.max(np.abs(err_rhs))
                print(iotag + "err_max_" + str(nel_1d) + " = ", err_max)
                assert err_max < 1e-10


# PYOPENCL_TEST=port python -m pudb test_euler.py 'test_isentropic_vortex(cl._csc)'
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main

        main([__file__])
