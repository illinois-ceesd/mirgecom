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
import numpy.linalg as la  # noqa
import pyopencl as cl
import pyopencl.clrandom
import pyopencl.clmath
import logging
from pytools.obj_array import (
    flat_obj_array,
    make_obj_array,
)
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.symbolic.primitives import TracePair
from mirgecom.euler import inviscid_operator
from mirgecom.euler import split_conserved
from mirgecom.steppers import euler_flow_stepper
from mirgecom.initializers import Vortex2D
from mirgecom.initializers import Lump
from mirgecom.boundary import PrescribedBoundary
from mirgecom.boundary import DummyBoundary
from mirgecom.eos import IdealSingleGas
from grudge.eager import EagerDGDiscretization
from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)


def test_inviscid_flux():
    """Checks that the Euler-internal inviscid flux
    routine (_inviscid_flux) returns exactly the
    expected result. The test checks flux for 1,
    2, and 3 spatial dimensions.

    The test is in 3 parts:

    - Exact expression: Checks the returned flux
      against the exact expressions:
      F(q) = <rhoV, (E+p)V, rho(V.x.V) + pI>
    - Prescribed p, V = 0:  Checks that only
      diagonal terms of the momentum flux
      [ rho(V.x.V) + pI ] are non-zero and return
      the correctly calculated p.
    - Prescribed p, V != 0: Checks that the
      flux terms are returned in the proper
      order by running only 1 non-zero velocity
      component at-a-time.
    """
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    logger = logging.getLogger(__name__)
    dim = 2
    nel_1d = 16

    from meshmode.mesh.generation import generate_regular_rect_mesh

    for dim in [1, 2, 3]:
        mesh = generate_regular_rect_mesh(
            a=(-0.5,) * dim, b=(0.5,) * dim, n=(nel_1d,) * dim
        )

        order = 3
        discr = EagerDGDiscretization(cl_ctx, mesh, order=order)
        eos = IdealSingleGas()

        logger.info(f"Number of {dim}d elems: {mesh.nelements}")

        mass = cl.clrandom.rand(
            queue, (mesh.nelements,), dtype=np.float64
        )
        energy = cl.clrandom.rand(
            queue, (mesh.nelements,), dtype=np.float64
        )
        mom = make_obj_array(
            [
                cl.clrandom.rand(
                    queue, (mesh.nelements,), dtype=np.float64
                )
                for i in range(dim)
            ]
        )

        q = flat_obj_array(mass, energy, mom)

        # Create the expected result
        p = eos.pressure(q)
        escale = (energy + p) / mass
        expected_mass_flux = mom
        expected_energy_flux = mom * make_obj_array([escale])
        expected_mom_flux = make_obj_array(
            [
                (mom[i] * mom[j] / mass + (p if i == j else 0))
                for i in range(dim)
                for j in range(dim)
            ]
        )
        expected_flux = flat_obj_array(
            expected_mass_flux, expected_energy_flux, expected_mom_flux
        )

        from mirgecom.euler import _inviscid_flux

        flux = _inviscid_flux(discr, q, eos)
        flux_resid = flux - expected_flux

        for i in range((dim + 2) * dim):
            assert (la.norm(flux_resid[i].get())) == 0.0

    class MyDiscr:
        def __init__(self, dim=1):
            self.dim = dim

    p0 = 1.0

    # === this next block tests 1,2,3 dimensions,
    # with single and multiple nodes/states. The
    # purpose of this block is to ensure that when
    # all components of V = 0, the flux recovers
    # the expected values (and p0 within tolerance)
    # === with V = 0, fixed P = p0
    tolerance = 1e-15
    for dim in [1, 2, 3]:
        for ntestnodes in [1, 10, 100]:
            fake_dis = MyDiscr(dim)
            mass = cl.clrandom.rand(
                queue, (ntestnodes,), dtype=np.float64
            )
            energy = cl.clrandom.rand(
                queue, (ntestnodes,), dtype=np.float64
            )
            mom = make_obj_array(
                [
                    cl.clrandom.rand(
                        queue, (ntestnodes,), dtype=np.float64
                    )
                    for i in range(dim)
                ]
            )
            p = cl.clrandom.rand(queue, (ntestnodes,), dtype=np.float64)

            for i in range(ntestnodes):
                mass[i] = 1.0 + i
                p[i] = p0
                for j in range(dim):
                    mom[j][i] = 0.0 * mass[i]
            energy = p / 0.4 + 0.5 * np.dot(mom, mom) / mass
            q = flat_obj_array(mass, energy, mom)
            p = eos.pressure(q)
            flux = _inviscid_flux(fake_dis, q, eos)

            logger.info(f"{dim}d flux = {flux}")

            # for velocity zero, these components should be == zero
            for i in range(2 * dim):
                for j in range(ntestnodes):
                    assert flux[i][j].get() == 0.0

            # The momentum diagonal should be p
            # Off-diagonal should be identically 0
            for i in range(dim):
                for j in range(dim):
                    print(f"(i,j) = ({i},{j})")
                    if i != j:
                        for n in range(ntestnodes):
                            assert (
                                flux[(2 + i) * dim + j][n].get() == 0.0
                            )
                    else:
                        for n in range(ntestnodes):
                            assert (
                                flux[(2 + i) * dim + j][n].get() == p[n]
                            )
                            assert (
                                np.abs(
                                    flux[(2 + i) * dim + j][n].get()
                                    - p0
                                )
                                < tolerance
                            )

    # === this next block tests 1,2,3 dimensions, single and multiple nodes
    # === with V = 1 (in some dimension), fixed P = p0
    # In this test, we verify that there is no spurious numerical noise
    # generated by non-zero momentum terms into unrelated flux components
    # by checking that the terms that should be are identically zero.
    # And we verify that the expected components have the expected values
    # by using an input state calculated using an input pressure (P0).
    # We expect calculated flux components to be floating point equal,
    # but not bitwise equal to the expected fluxes.
    for dim in [1, 2, 3]:
        for livedim in range(dim):
            for ntestnodes in [1, 10, 100]:
                fake_dis = MyDiscr(dim)
                mass = cl.clrandom.rand(
                    queue, (ntestnodes,), dtype=np.float64
                )
                energy = cl.clrandom.rand(
                    queue, (ntestnodes,), dtype=np.float64
                )
                mom = make_obj_array(
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
                    mass[i] = 1.0 + i
                    p[i] = p0
                    for j in range(dim):
                        mom[j][i] = 0.0 * mass[i]
                    mom[livedim][i] = mass[i]
                energy = (
                    p / (eos.gamma() - 1.0)
                    + 0.5 * np.dot(mom, mom) / mass
                )
                q = flat_obj_array(mass, energy, mom)
                p = eos.pressure(q)
                flux = _inviscid_flux(fake_dis, q, eos)

                logger.info(f"{dim}d flux = {flux}")

                # first two components should be nonzero in livedim only
                expected_flux = mom
                logger.info("Testing continuity")
                for i in range(dim):
                    assert (
                        la.norm((flux[i] - expected_flux[i]).get())
                        == 0.0
                    )
                    if i != livedim:
                        assert la.norm(flux[i].get()) == 0.0
                    else:
                        assert la.norm(flux[i].get()) > 0.0

                logger.info("Testing energy")
                expected_flux = mom * make_obj_array(
                    [(energy + p) / mass]
                )
                for i in range(dim):
                    assert (
                        la.norm(
                            (flux[dim + i] - expected_flux[i]).get()
                        )
                        == 0.0
                    )
                    if i != livedim:
                        assert la.norm(flux[dim + i].get()) == 0.0
                    else:
                        assert la.norm(flux[dim + i].get()) > 0.0

                logger.info("Testing momentum")
                xpmomflux = make_obj_array(
                    [
                        (mom[i] * mom[j] / mass + (p if i == j else 0))
                        for i in range(dim)
                        for j in range(dim)
                    ]
                )
                for i in range(dim):
                    expected_flux = xpmomflux[i * dim: (i + 1) * dim]
                    fluxindex = (2 + i) * dim
                    for j in range(dim):
                        assert (
                            la.norm(
                                (
                                    flux[fluxindex + j]
                                    - expected_flux[j]
                                ).get()
                            )
                            == 0
                        )
                        if i == j:
                            if i == livedim:
                                assert (
                                    la.norm(flux[fluxindex + j].get())
                                    > 0.0
                                )
                            else:
                                # just for sanity - make sure the flux recovered the
                                # prescribed value of p0 (within fp tol)
                                for k in range(ntestnodes):
                                    assert (
                                        np.abs(
                                            flux[fluxindex + j][k] - p0
                                        )
                                        < tolerance
                                    )
                        else:
                            assert (
                                la.norm(flux[fluxindex + j].get())
                                == 0.0
                            )


def test_facial_flux():
    """Check the flux across element faces by
    prescribing states (q) with known fluxes. Only
    uniform states are tested currently - ensuring
    that the Lax-Friedrichs flux terms which are
    proportional to jumps in state data vanish.

    Since the returned fluxes use state data which
    has been interpolated to-and-from the element
    faces, this test is grid-dependent.
    """
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    logger = logging.getLogger(__name__)

    tolerance = 1e-14
    p0 = 1.0

    from meshmode.mesh.generation import generate_regular_rect_mesh

    for order in [1, 2, 3]:
        for dim in [1, 2, 3]:
            from pytools.convergence import EOCRecorder

            eoc_rec0 = EOCRecorder()
            eoc_rec1 = EOCRecorder()
            for nel_1d in [4, 8, 12]:

                mesh = generate_regular_rect_mesh(
                    a=(-0.5,) * dim, b=(0.5,) * dim, n=(nel_1d,) * dim
                )

                #                order = 3

                logger.info(f"Number of elements: {mesh.nelements}")

                discr = EagerDGDiscretization(cl_ctx, mesh, order=order)

                mass_input = discr.zeros(queue)
                energy_input = discr.zeros(queue)
                mom_input = flat_obj_array(
                    [discr.zeros(queue) for i in range(discr.dim)]
                )

                # This sets p = 1
                mass_input[:] = 1.0
                energy_input[:] = 2.5

                fields = flat_obj_array(
                    mass_input, energy_input, mom_input
                )

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

                assert err < tolerance

                err1 = err
                # mom flux max should be p = 1 + (any interp error)
                maxmomerr = 0.0
                for i in range(2, 2 + discr.dim):
                    err = np.max(
                        np.array(
                            [
                                la.norm(
                                    interior_face_flux[i].get(), np.inf
                                )
                            ]
                        )
                    )
                    err = np.abs(err - p0)
                    if err > maxmomerr:
                        maxmomerr = err
                    assert err < tolerance
                errrec = np.maximum(err1, maxmomerr)
                eoc_rec0.add_data_point(1.0 / nel_1d, errrec)

                # Check the boundary facial fluxes as called on a boundary
                dir_mass = discr.interp("vol", BTAG_ALL, mass_input)
                dir_e = discr.interp("vol", BTAG_ALL, energy_input)
                dir_mom = discr.interp("vol", BTAG_ALL, mom_input)

                dir_bval = flat_obj_array(dir_mass, dir_e, dir_mom)
                dir_bc = flat_obj_array(dir_mass, dir_e, dir_mom)

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

                assert err < tolerance
                err1 = err

                # mom flux should be  == p ~= p0
                maxmomerr = 0.0
                for i in range(2, 2 + discr.dim):
                    err = np.max(
                        np.array(
                            [la.norm(boundary_flux[i].get(), np.inf)]
                        )
                    )
                    err = np.abs(err - p0)
                    assert err < tolerance
                    if err > maxmomerr:
                        maxmomerr = err
                errrec = np.maximum(err1, maxmomerr)
                eoc_rec1.add_data_point(1.0 / nel_1d, errrec)

            message = (
                f"standalone Errors:\n{eoc_rec0}"
                f"boundary Errors:\n{eoc_rec1}"
            )
            logger.info(message)
            assert (
                eoc_rec0.order_estimate() >= order - 0.5
                or eoc_rec0.max_error() < 1e-9
            )
            assert (
                eoc_rec1.order_estimate() >= order - 0.5
                or eoc_rec1.max_error() < 1e-9
            )


def test_uniform_rhs():
    """Tests the inviscid rhs using a trivial
    constant/uniform state which should
    yield rhs = 0 to FP.  The test is performed
    for 1, 2, and 3 dimensions.
    """
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    logger = logging.getLogger(__name__)

    tolerance = 1e-9
    maxxerr = 0.0
    from meshmode.mesh.generation import generate_regular_rect_mesh

    for dim in [1, 2, 3]:
        for order in [1, 2, 3]:
            #            from pytools.convergence import EOCRecorder
            #            eoc_rec0 = EOCRecorder()
            #            eoc_rec1 = EOCRecorder()
            for nel_1d in [4, 8, 12]:

                mesh = generate_regular_rect_mesh(
                    a=(-0.5,) * dim, b=(0.5,) * dim, n=(nel_1d,) * dim
                )

                logger.info(
                    f"Number of {dim}d elements: {mesh.nelements}"
                )

                discr = EagerDGDiscretization(cl_ctx, mesh, order=order)

                mass_input = discr.zeros(queue)
                energy_input = discr.zeros(queue)

                # this sets p = p0 = 1.0
                mass_input[:] = 1.0
                energy_input[:] = 2.5

                mom_input = make_obj_array(
                    [discr.zeros(queue) for i in range(discr.dim)]
                )
                fields = flat_obj_array(
                    mass_input, energy_input, mom_input
                )

                expected_rhs = make_obj_array(
                    [discr.zeros(queue) for i in range(discr.dim + 2)]
                )

                boundaries = {BTAG_ALL: DummyBoundary()}
                inviscid_rhs = inviscid_operator(discr, fields, boundaries)
                rhs_resid = inviscid_rhs - expected_rhs

                rho_resid = split_conserved(dim, rhs_resid).mass
                rhoe_resid = split_conserved(dim, rhs_resid).energy
                mom_resid = split_conserved(dim, rhs_resid).momentum

                rho_rhs = split_conserved(dim, inviscid_rhs).mass
                rhoe_rhs = split_conserved(dim, inviscid_rhs).energy
                rhov_rhs = split_conserved(dim, inviscid_rhs).momentum

                message = (
                    f"rho_rhs  = {rho_rhs}\n"
                    f"rhoe_rhs = {rhoe_rhs}\n"
                    f"rhov_rhs = {rhov_rhs}"
                )
                logger.info(message)

                assert np.max(np.abs(rho_resid.get())) < tolerance
                assert np.max(np.abs(rhoe_resid.get())) < tolerance
                for i in range(dim):
                    assert (
                        np.max(np.abs(mom_resid[i].get())) < tolerance
                    )

                err_max = np.max(np.abs(rhs_resid[i].get()))
                #                eoc_rec0.add_data_point(1.0 / nel_1d, err_max)
                assert(err_max < tolerance)
                if err_max > maxxerr:
                    maxxerr = err_max
                # set a non-zero, but uniform velocity component
                i = 0

                for mom_component in mom_input:
                    mom_component[:] = (-1.0) ** i
                    i = i + 1

                boundaries = {BTAG_ALL: DummyBoundary()}
                inviscid_rhs = inviscid_operator(discr, fields, boundaries)
                rhs_resid = inviscid_rhs - expected_rhs

                rho_resid = split_conserved(dim, rhs_resid).mass
                rhoe_resid = split_conserved(dim, rhs_resid).energy
                mom_resid = split_conserved(dim, rhs_resid).momentum

                assert np.max(np.abs(rho_resid.get())) < tolerance
                assert np.max(np.abs(rhoe_resid.get())) < tolerance
                for i in range(dim):
                    assert np.max(np.abs(mom_resid[i].get())) < tolerance

                err_max = np.max(np.abs(rhs_resid[i].get()))
                #                eoc_rec1.add_data_point(1.0 / nel_1d, err_max)
                assert(err_max < tolerance)
                if err_max > maxxerr:
                    maxxerr = err_max
            #            message = (
            #                f"{iotag}V == 0 Errors:\n{eoc_rec0}"
            #                f"{iotag}V != 0 Errors:\n{eoc_rec1}"
            #            )
            #            print(message)
            #            assert (
            #                eoc_rec0.order_estimate() >= order - 0.5
            #                or eoc_rec0.max_error() < 1e-9
            #            )
            #            assert (
            #                eoc_rec1.order_estimate() >= order - 0.5
            #                or eoc_rec1.max_error() < 1e-9
            #            )


def test_vortex_rhs():
    #def test_vortex_rhs(ctx_factory):
    """Tests the inviscid rhs using the non-trivial
    2D isentropic vortex case configured to yield
    rhs = 0. Checks several different orders
    and refinement levels to check error
    behavior.
    """
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    logger = logging.getLogger(__name__)

    dim = 2

    for order in [1, 2, 3]:

        from pytools.convergence import EOCRecorder

        eoc_rec = EOCRecorder()

        from meshmode.mesh.generation import generate_regular_rect_mesh

        for nel_1d in [16, 32, 64]:

            mesh = generate_regular_rect_mesh(
                a=(-5,) * dim, b=(5,) * dim, n=(nel_1d,) * dim,
            )

            logger.info(
                f"Number of {dim}d elements:  {mesh.nelements}"
            )

            discr = EagerDGDiscretization(cl_ctx, mesh, order=order)
            nodes = discr.nodes().with_queue(queue)

            # Init soln with Vortex and expected RHS = 0
            vortex = Vortex2D(center=[0, 0], velocity=[0, 0])
            vortex_soln = vortex(0, nodes)
            boundaries = {BTAG_ALL: PrescribedBoundary(vortex)}

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

            eoc_rec.add_data_point(1.0 / nel_1d, err_max)

        message = (
            f"Error for (dim,order) = ({dim},{order}):\n"
            f"{eoc_rec}"
        )
        logger.info(message)

        assert (
            eoc_rec.order_estimate() >= order - 0.5
            or eoc_rec.max_error() < 1e-11
        )


def test_lump_rhs():
    """Tests the inviscid rhs using the non-trivial
    1, 2, and 3D mass lump case against the analytic
    expressions of the RHS. Checks several different
    orders and refinement levels to check error behavior.
    """
    #    cl_ctx = ctx_factory()
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    logger = logging.getLogger(__name__)

    tolerance = 1e-10
    maxxerr = 0.0

    for dim in [1, 2, 3]:
        for order in [1, 2, 3]:
            from pytools.convergence import EOCRecorder

            eoc_rec = EOCRecorder()

            for nel_1d in [4, 8, 12]:
                from meshmode.mesh.generation import (
                    generate_regular_rect_mesh,
                )

                mesh = generate_regular_rect_mesh(
                    a=(-5,) * dim, b=(5,) * dim, n=(nel_1d,) * dim,
                )

                logger.info(f"Number of elements: {mesh.nelements}")

                discr = EagerDGDiscretization(cl_ctx, mesh, order=order)
                nodes = discr.nodes().with_queue(queue)

                # Init soln with Lump and expected RHS = 0
                center = np.zeros(shape=(dim,))
                velocity = np.zeros(shape=(dim,))
                lump = Lump(center=center, velocity=velocity)
                lump_soln = lump(0, nodes)
                boundaries = {BTAG_ALL: PrescribedBoundary(lump)}
                inviscid_rhs = inviscid_operator(
                    discr, lump_soln, t=0.0, boundaries=boundaries
                )
                expected_rhs = lump.exact_rhs(discr, lump_soln, 0)

                err_max = np.max(
                    np.array(
                        [
                            la.norm(
                                (
                                    inviscid_rhs[i] - expected_rhs[i]
                                ).get(),
                                np.inf,
                            )
                            for i in range(dim + 2)
                        ]
                    )
                )
                if err_max > maxxerr:
                    maxxerr = err_max

                eoc_rec.add_data_point(1.0 / nel_1d, err_max)
            logger.info(f"Max error: {maxxerr}")
            message = (
                f"Error for (dim,order) = ({dim},{order}):\n"
                f"{eoc_rec}"
            )
            logger.info(message)
            assert (
                eoc_rec.order_estimate() >= order - 0.5
                or eoc_rec.max_error() < tolerance
            )


def test_isentropic_vortex():
    """Advance the 2D isentropic vortex case in
    time with non-zero velocities using an RK4
    timestepping scheme. Check the advanced field
    values against the exact/analytic expressions.

    This tests all parts of the Euler module working
    together, with results converging at the expected
    rates vs. the order.
    """
    logger = logging.getLogger(__name__)

    dim = 2

    for order in [1, 2, 3]:
        from pytools.convergence import EOCRecorder

        eoc_rec = EOCRecorder()

        for nel_1d in [16, 32, 64]:
            from meshmode.mesh.generation import (
                generate_regular_rect_mesh,
            )

            mesh = generate_regular_rect_mesh(
                a=(-5.0,) * dim, b=(5.0,) * dim, n=(nel_1d,) * dim
            )

            exittol = 1.0
            t_final = 0.001
            cfl = 1.0
            vel = np.zeros(shape=(dim,))
            orig = np.zeros(shape=(dim,))
            vel[:dim] = 1.0
            dt = .0001
            initializer = Vortex2D(center=orig, velocity=vel)
            casename = 'Vortex'
            boundaries = {BTAG_ALL: PrescribedBoundary(initializer)}
            eos = IdealSingleGas()
            t = 0
            flowparams = {'dim': dim, 'dt': dt, 'order': order, 'time': t,
                          'boundaries': boundaries, 'initializer': initializer,
                          'eos': eos, 'casename': casename, 'mesh': mesh,
                          'tfinal': t_final, 'exittol': exittol, 'cfl': cfl,
                          'constantcfl': False, 'nstatus': 0}
            maxerr = euler_flow_stepper(flowparams)
            eoc_rec.add_data_point(1.0 / nel_1d, maxerr)

        message = (
            f"Error for (dim,order) = ({dim},{order}):\n"
            f"{eoc_rec}"
        )
        logger.info(message)
        assert (
            eoc_rec.order_estimate() >= order - 0.5
            or eoc_rec.max_error() < 1e-11
        )


#PYOPENCL_TEST="port python -m pudb test_euler.py 'test_isentropic_vortex(cl._csc)'"

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main

        main([__file__])
