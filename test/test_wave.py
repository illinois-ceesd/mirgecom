__copyright__ = """Copyright (C) 2020 CEESD Developers"""

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
import numpy.linalg as la
import pyopencl as cl
import pyopencl.array as cla  # noqa
import pyopencl.clmath as clmath
from pytools.obj_array import join_fields

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)

import pytest

import logging
logger = logging.getLogger(__name__)

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("order", [2, 3, 4])
def test_standing_wave(ctx_factory, dim, order):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for n in [4, 8, 16]:
        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(
                a=(-0.5*np.pi,)*dim,
                b=(0.5*np.pi,)*dim,
                n=(n,)*dim)

        from meshmode.discretization import Discretization

        from grudge.eager import EagerDGDiscretization
        discr = EagerDGDiscretization(cl_ctx, mesh, order=order)

        nodes = discr.nodes().with_queue(queue)

        # 2D: phi(x,y,t) = cos(omega*t-pi/4)*cos(x)*cos(y)
        # 3D: phi(x,y,z,t) = cos(omega*t-pi/4)*cos(x)*cos(y)*cos(z)
        # where omega = sqrt(#dims)*c

        c = 2.
        offset = np.pi/4.
        omega = np.sqrt(dim) * c

        # 2D: u = phi_t = -omega*sin(omega*t-pi/4)*cos(x)*cos(y)
        # 3D: u = phi_t = -omega*sin(omega*t-pi/4)*cos(x)*cos(y)*cos(z)
        u = discr.empty(queue, dtype=np.float64)
        u[:] = -omega * np.sin(-offset)
        for i in range(dim):
            u = u * clmath.cos(nodes[i])

        # 2D: v[0] = c*phi_x = -c*cos(omega*t-pi/4)*sin(x)*cos(y)
        #     v[1] = c*phi_y = -c*cos(omega*t-pi/4)*cos(x)*sin(y)
        # 3D: v[0] = c*phi_x = -c*cos(omega*t-pi/4)*sin(x)*cos(y)*cos(z)
        #     v[1] = c*phi_y = -c*cos(omega*t-pi/4)*cos(x)*sin(y)*cos(z)
        #     v[2] = c*phi_z = -c*cos(omega*t-pi/4)*cos(x)*cos(y)*sin(z)
        v = []
        for i in range(dim):
            v_i = discr.empty(queue, dtype=np.float64)
            v_i[:] = -c * np.cos(-offset)
            for j in range(dim):
                if i == j:
                    v_i = v_i * clmath.sin(nodes[j])
                else:
                    v_i = v_i * clmath.cos(nodes[j])
            v.append(v_i)
        fields = join_fields(u, v)

        from mirgecom.wave import wave_operator
        rhs = wave_operator(discr, c=c, w=fields)

        # rhs(u part) = c*div(v)
        # 2D: rhs(u part) = -2*c^2*cos(omega*t-pi/4)*cos(x)*cos(y)
        # 3D: rhs(u part) = -3*c^2*cos(omega*t-pi/4)*cos(x)*cos(y)*cos(z)
        expected_rhs_u = discr.empty(queue, dtype=np.float64)
        # Why no negative sign?
        # expected_rhs_u[:] = -omega**2 * np.cos(-offset)
        expected_rhs_u[:] = omega**2 * np.cos(-offset)
        for i in range(dim):
            expected_rhs_u = expected_rhs_u * clmath.cos(nodes[i])

        # rhs(v part) = c*grad(u)
        # 2D: rhs(v part)[0] = omega*c*sin(omega*t-pi/4)*sin(x)*cos(y)
        #     rhs(v part)[1] = omega*c*sin(omega*t-pi/4)*cos(x)*sin(y)
        # 3D: rhs(v part)[0] = omega*c*sin(omega*t-pi/4)*sin(x)*cos(y)*cos(z)
        #     rhs(v part)[1] = omega*c*sin(omega*t-pi/4)*cos(x)*sin(y)*cos(z)
        #     rhs(v part)[2] = omega*c*sin(omega*t-pi/4)*cos(x)*cos(y)*sin(z)
        expected_rhs_v = []
        for i in range(dim):
            rhs_v_i = discr.empty(queue, dtype=np.float64)
            # Why negative sign?
            # rhs_v_i[:] = omega*c * np.sin(-offset)
            rhs_v_i[:] = -omega*c * np.sin(-offset)
            for j in range(dim):
                if i == j:
                    rhs_v_i = rhs_v_i * clmath.sin(nodes[j])
                else:
                    rhs_v_i = rhs_v_i * clmath.cos(nodes[j])
            expected_rhs_v.append(rhs_v_i)
        expected_rhs = join_fields(expected_rhs_u, expected_rhs_v)

        err = np.max(np.array([la.norm((rhs[i] - expected_rhs[i]).get(), np.inf)
                for i in range(dim+1)]))
        eoc_rec.add_data_point(1./n, err)

        # from grudge.shortcuts import make_visualizer
        # vis = make_visualizer(discr, discr.order)
        # vis.write_vtk_file("result_{n}.vtu".format(n=n),
        #         [
        #             ("u", fields[0]),
        #             ("v", fields[1:]),
        #             ("rhs_u_actual", rhs[0]),
        #             ("rhs_v_actual", rhs[1:]),
        #             ("rhs_u_expected", expected_rhs[0]),
        #             ("rhs_v_expected", expected_rhs[1:]),
        #             ])

    print("Approximation error:")
    print(eoc_rec)
    assert(eoc_rec.order_estimate() >= order - 0.5 or eoc_rec.max_error() < 1e-11)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("order", [2, 3, 4])
def test_wave_manufactured(ctx_factory, dim, order):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for n in [4, 8, 16]:
        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(
                a=(-1.,)*dim,
                b=(1.,)*dim,
                n=(n,)*dim)

        from meshmode.discretization import Discretization

        from grudge.eager import EagerDGDiscretization
        discr = EagerDGDiscretization(cl_ctx, mesh, order=order)

        nodes = discr.nodes().with_queue(queue)

        # 2D: phi(x,y,t) = cos(omega*t-pi/4)*(x-1)^3*(x+1)^3*(y-1)^3*(y+1)^3
        # 3D: phi(x,y,z,t) = cos(omega*t-pi/4)*(x-1)^3*(x+1)^3*(y-1)^3*(y+1)^3
        #                      *(z-1)^3*(z+1)^3
        # Not actual solutions, but we can still apply wave operator to them

        c = 2.
        offset = np.pi/4.
        omega = 1.

        # 2D: u = phi_t = -omega*sin(omega*t-pi/4)*(x-1)^3*(x+1)^3*(y-1)^3*(y+1)^3
        # 3D: u = phi_t = -omega*sin(omega*t-pi/4)*(x-1)^3*(x+1)^3*(y-1)^3*(y+1)^3
        #                   *(z-1)^3*(z+1)^3
        u = discr.empty(queue, dtype=np.float64)
        u[:] = -omega * np.sin(-offset)
        for i in range(dim):
            u = u * (nodes[i]-1.)**3 * (nodes[i]+1.)**3

        # 2D: v[0] = c*phi_x = c*cos(omega*t-pi/4)*(3*(x-1)^2*(x+1)^3 + 3*(x-1)^3*(x+1)^2)
        #                        *(y-1)^3*(y+1)^3
        #     v[1] = c*phi_y = c*cos(omega*t-pi/4)*(x-1)^3*(x+1)^3*(3*(y-1)^2*(y+1)^3
        #                        + 3*(y-1)^3*(y+1)^2)
        # 3D: v[0] = c*phi_x = c*cos(omega*t-pi/4)*(3*(x-1)^2*(x+1)^3 + 3*(x-1)^3*(x+1)^2)
        #                        *(y-1)^3*(y+1)^3*(z-1)^3*(z+1)^3
        #     v[1] = c*phi_y = c*cos(omega*t-pi/4)*(x-1)^3*(x+1)^3*(3*(y-1)^2*(y+1)^3
        #                        + 3*(y-1)^3*(y+1)^2)*(z-1)^3*(z+1)^3
        #     v[2] = c*phi_z = c*cos(omega*t-pi/4)*(x-1)^3*(x+1)^3*(y-1)^3*(y+1)^3
        #                        *(3*(z-1)^2*(z+1)^3 + 3*(z-1)^3*(z+1)^2)
        v = []
        for i in range(dim):
            v_i = discr.empty(queue, dtype=np.float64)
            v_i[:] = c * np.cos(-offset)
            for j in range(dim):
                if i == j:
                    v_i = v_i * (3. * (nodes[j]-1.)**2 * (nodes[j]+1.)**3
                            + 3. * (nodes[j]-1.)**3 * (nodes[j]+1.)**2)
                else:
                    v_i = v_i * (nodes[j]-1.)**3 * (nodes[j]+1.)**3
            v.append(v_i)
        fields = join_fields(u, v)

        from mirgecom.wave import wave_operator
        rhs = wave_operator(discr, c=c, w=fields)

        # rhs(u part) = c*div(v)
        # 2D: rhs(u part) = c^2*cos(omega*t-pi/4)*((6*(x-1)*(x+1)^3 + 18*(x-1)^2*(x+1)^2
        #                     + 6*(x-1)^3*(x+1))*(y-1)^3*(y+1)^3 + (x-1)^3*(x+1)^3
        #                     * (6*(y-1)*(y+1)^3 + 18*(y-1)^2*(y+1)^2 + 6*(y-1)^3*(y+1)))
        # 3D: rhs(u part) = c^2*cos(omega*t-pi/4)*((6*(x-1)*(x+1)^3 + 18*(x-1)^2*(x+1)^2
        #                     + 6*(x-1)^3*(x+1))*(y-1)^3*(y+1)^3*(z-1)^3*(z+1)^3
        #                     + (x-1)^3*(x+1)^3*(6*(y-1)*(y+1)^3 + 18*(y-1)^2*(y+1)^2
        #                     + 6*(y-1)^3*(y+1))*(z-1)^3*(z+1)^3 + (x-1)^3*(x+1)^3*(y-1)^3
        #                     * (y+1)^3*(6*(z-1)*(z-1)^3 + 18*(z-1)^2*(z+1)^2 + 6*(z-1)^3*(z+1)))
        expected_rhs_u = discr.zeros(queue, dtype=np.float64)
        for i in range(dim):
            spatial_part_i = discr.empty(queue, dtype=np.float64)
            spatial_part_i[:] = 1.
            for j in range(dim):
                if i == j:
                    spatial_part_i = spatial_part_i * (
                               6. * (nodes[j]-1.) * (nodes[j]+1.)**3
                            + 18. * (nodes[j]-1.)**2 * (nodes[j]+1.)**2
                            +  6. * (nodes[j]-1.)**3 * (nodes[j]+1.))
                else:
                    spatial_part_i = spatial_part_i * ((nodes[j]-1.)**3
                            * (nodes[j]+1.)**3)
            expected_rhs_u = expected_rhs_u + spatial_part_i
        # Why negative sign?
        # expected_rhs_u = expected_rhs_u * c**2 * np.cos(-offset)
        expected_rhs_u = expected_rhs_u * -c**2 * np.cos(-offset)

        # rhs(v part) = c*grad(u)
        # 2D: rhs(v part)[0] = -omega*c*sin(omega*t-pi/4)*(3*(x-1)^2*(x+1)^3 + 3*(x-1)^3*(x+1)^2)
        #                        *(y-1)^3*(y+1)^3
        #     rhs(v part)[1] = -omega*c*sin(omega*t-pi/4)*(x-1)^3*(x+1)^3*(3*(y-1)^2*(y+1)^3
        #                        + 3*(y-1)^3*(y+1)^2)
        # 3D: rhs(v part)[0] = -omega*c*sin(omega*t-pi/4)*(3*(x-1)^2*(x+1)^3 + 3*(x-1)^3*(x+1)^2)
        #                        *(y-1)^3*(y+1)^3*(z-1)^3*(z+1)^3
        #     rhs(v part)[1] = -omega*c*sin(omega*t-pi/4)*(x-1)^3*(x+1)^3*(3*(y-1)^2*(y+1)^3
        #                        + 3*(y-1)^3*(y+1)^2)*(z-1)^3*(z+1)^3
        #     rhs(v part)[2] = -omega*c*sin(omega*t-pi/4)*(x-1)^3*(x+1)^3*(y-1)^3*(y+1)^3
        #                        *(3*(z-1)^2*(z+1)^3 + 3*(z-1)^3*(z+1)^2)
        expected_rhs_v = []
        for i in range(dim):
            rhs_v_i = discr.empty(queue, dtype=np.float64)
            # Why no negative sign?
            # rhs_v_i[:] = -omega*c * np.sin(-offset)
            rhs_v_i[:] = omega*c * np.sin(-offset)
            for j in range(dim):
                if i == j:
                    rhs_v_i = rhs_v_i * (3. * (nodes[j]-1.)**2 * (nodes[j]+1.)**3
                            + 3. * (nodes[j]-1.)**3 * (nodes[j]+1.)**2)
                else:
                    rhs_v_i = rhs_v_i * (nodes[j]-1.)**3 * (nodes[j]+1.)**3
            expected_rhs_v.append(rhs_v_i)
        expected_rhs = join_fields(expected_rhs_u, expected_rhs_v)

        err = np.max(np.array([la.norm((rhs[i] - expected_rhs[i]).get(), np.inf)
                for i in range(dim+1)]))
        eoc_rec.add_data_point(1./n, err)

        # from grudge.shortcuts import make_visualizer
        # vis = make_visualizer(discr, discr.order)
        # vis.write_vtk_file("result_{n}.vtu".format(n=n),
        #         [
        #             ("u", fields[0]),
        #             ("v", fields[1:]),
        #             ("rhs_u_actual", rhs[0]),
        #             ("rhs_v_actual", rhs[1:]),
        #             ("rhs_u_expected", expected_rhs[0]),
        #             ("rhs_v_expected", expected_rhs[1:]),
        #             ])

    print("Approximation error:")
    print(eoc_rec)
    assert(eoc_rec.order_estimate() >= order - 0.5 or eoc_rec.max_error() < 1e-11)
