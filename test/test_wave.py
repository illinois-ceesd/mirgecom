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

# Doesn't quite work at the moment -- using this definition of phi leads to nonzero
# boundary values for v. Try again when nonzero BCs are implemented?
# @pytest.mark.parametrize("dim", [2, 3])
# @pytest.mark.parametrize("order", [2, 3, 4])
# def test_wave(ctx_factory, dim, order):
#     cl_ctx = ctx_factory()
#     queue = cl.CommandQueue(cl_ctx)

#     from pytools.convergence import EOCRecorder
#     eoc_rec = EOCRecorder()

#     for n in [4, 8, 16]:
#         from meshmode.mesh.generation import generate_regular_rect_mesh
#         mesh = generate_regular_rect_mesh(
#                 a=(-0.5*np.pi,)*dim,
#                 b=(0.5*np.pi,)*dim,
#                 n=(n,)*dim)

#         from meshmode.discretization import Discretization

#         from grudge.eager import EagerDGDiscretization
#         discr = EagerDGDiscretization(cl_ctx, mesh, order=order)

#         nodes = discr.nodes().with_queue(queue)

#         # TODO: change t in cosine to c*t -- makes more sense
#         # 2D: phi(x,y,t) = cos(t-pi/4)*cos(x)*cos(2*y)
#         # 3D: phi(x,y,z,t) = cos(t-pi/4)*cos(x)*cos(2*y)*cos(3*z)

#         t_offset = np.pi/4.

#         c_invsq = 0.
#         for i in range(dim):
#             c_invsq = c_invsq + float(i+1)**2
#         c = 1./np.sqrt(c_invsq)

#         # u = phi_t
#         # v_i = c*phi_{x_i}
#         u = discr.empty(queue, dtype=np.float64)
#         u[:] = -np.sin(-t_offset)
#         for i in range(dim):
#             u = u * clmath.cos(float(i+1)*nodes[i])
#         v = []
#         for i in range(dim):
#             v_i = discr.empty(queue, dtype=np.float64)
#             v_i[:] = c * np.sin(-t_offset)
#             for j in range(dim):
#                 if i == j:
#                     v_i = v_i * -float(j+1) * clmath.sin(float(j+1)*nodes[j])
#                 else:
#                     v_i = v_i * clmath.cos(float(j+1)*nodes[j])
#             v.append(v_i)
#         fields = join_fields(u, v)

#         from mirgecom.wave import wave_operator
#         rhs = wave_operator(discr, c=c, w=fields)

#         expected_rhs_u = discr.empty(queue, dtype=np.float64)
#         expected_rhs_u[:] = -np.cos(-t_offset)
#         for i in range(dim):
#             expected_rhs_u = expected_rhs_u * clmath.cos(float(i+1)*nodes[i])
#         expected_rhs_v = []
#         for i in range(dim):
#             rhs_v_i = discr.empty(queue, dtype=np.float64)
#             rhs_v_i[:] = c * -np.sin(-t_offset)
#             for j in range(dim):
#                 if i == j:
#                     rhs_v_i = (rhs_v_i * -float(j+1) * clmath.sin(
#                             float(j+1)*nodes[j]))
#                 else:
#                     rhs_v_i = rhs_v_i * clmath.cos(float(j+1)*nodes[j])
#             expected_rhs_v.append(rhs_v_i)
#         expected_rhs = join_fields(expected_rhs_u, expected_rhs_v)

#         err = np.max(np.array([la.norm((rhs[i] - expected_rhs[i]).get(), np.inf)
#                 for i in range(dim+1)]))
#         eoc_rec.add_data_point(1./n, err)

#         from grudge.shortcuts import make_visualizer
#         vis = make_visualizer(discr, discr.order)
#         vis.write_vtk_file("result_{n}.vtu".format(n=n),
#                 [
#                     ("rhs_u_actual", rhs[0]),
#                     ("rhs_v_actual", rhs[1:]),
#                     ("rhs_u_expected", expected_rhs[0]),
#                     ("rhs_v_expected", expected_rhs[1:]),
#                     ])

#     print("Approximation error:")
#     print(eoc_rec)
#     assert(eoc_rec.order_estimate() >= order - 0.5 or eoc_rec.max_error() < 1e-11)

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("order", [2, 3, 4])
def test_wave(ctx_factory, dim, order):
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

        # 2D: phi(x,y,t) = cos(c*t-pi/4)*(x-1)^3*(x+1)^3*(y-1)^3*(y+1)^3
        # 3D: phi(x,y,z,t) = cos(c*t-pi/4)*(x-1)^3*(x+1)^3*(y-1)^3*(y+1)^3
        #                      *(z-1)^3*(z+1)^3
        # Not actual solutions, but we can still apply wave operator to them

        c = 2.
        t_offset = np.pi/4.

        # u = phi_t
        # v_i = c*phi_{x_i}
        u = discr.empty(queue, dtype=np.float64)
        u[:] = -c * np.sin(-t_offset)
        for i in range(dim):
            u = u * (nodes[i]-1.)**3 * (nodes[i]+1.)**3
        v = []
        for i in range(dim):
            v_i = discr.empty(queue, dtype=np.float64)
            v_i[:] = c * np.cos(-t_offset)
            for j in range(dim):
                if i == j:
                    v_i = v_i * 3. * ((nodes[j]-1.)**2 * (nodes[j]+1.)**3
                            + (nodes[j]-1.)**3 * (nodes[j]+1.)**2)
                else:
                    v_i = v_i * (nodes[j]-1.)**3 * (nodes[j]+1.)**3
            v.append(v_i)
        fields = join_fields(u, v)

        from mirgecom.wave import wave_operator
        rhs = wave_operator(discr, c=c, w=fields)

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
        # why negative sign?
        expected_rhs_u = expected_rhs_u * -c**2 * np.cos(-t_offset)
        expected_rhs_v = []
        for i in range(dim):
            rhs_v_i = discr.empty(queue, dtype=np.float64)
            # why no negative sign?
            rhs_v_i[:] = c**2 * np.sin(-t_offset)
            for j in range(dim):
                if i == j:
                    rhs_v_i = rhs_v_i * 3. * ((nodes[j]-1.)**2 * (nodes[j]+1.)**3
                            + (nodes[j]-1.)**3 * (nodes[j]+1.)**2)
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
        #             ("rhs_u_actual", rhs[0]),
        #             ("rhs_v_actual", rhs[1:]),
        #             ("rhs_u_expected", expected_rhs[0]),
        #             ("rhs_v_expected", expected_rhs[1:]),
        #             ])

    print("Approximation error:")
    print(eoc_rec)
    assert(eoc_rec.order_estimate() >= order - 0.5 or eoc_rec.max_error() < 1e-11)
