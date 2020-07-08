__copyright__ = """Copyright (C) 2020 University of Illinois Board of Trustees"""

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
import pyopencl.clmath as clmath # noqa
from pytools.obj_array import flat_obj_array, make_obj_array
import pymbolic as pmbl
import pymbolic.primitives as prim
import mirgecom.symbolic as sym
from mirgecom.wave import wave_operator

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)

import pytest

import logging
logger = logging.getLogger(__name__)


def get_standing_wave(dim):
    # 2D: phi(x,y,t) = cos(sqrt(2)*c*t-pi/4)*cos(x)*cos(y)
    # 3D: phi(x,y,z,t) = cos(sqrt(3)*c*t-pi/4)*cos(x)*cos(y)*cos(z)
    # on [-pi/2, pi/2]^{#dims}
    def mesh_factory(n):
        from meshmode.mesh.generation import generate_regular_rect_mesh
        return generate_regular_rect_mesh(
                a=(-0.5*np.pi,)*dim,
                b=(0.5*np.pi,)*dim,
                n=(n,)*dim)
    c = 2.
    sym_coords = prim.make_sym_vector('x', dim)
    sym_t = pmbl.var("t")
    sym_cos = pmbl.var("cos")
    sym_phi = sym_cos(np.sqrt(dim)*c*sym_t - np.pi/4)
    for i in range(dim):
        sym_phi *= sym_cos(sym_coords[i])
    return (dim, c, mesh_factory, sym_phi, 0.05)


def get_manufactured_cubic(dim):
    # 2D: phi(x,y,t) = cos(t-pi/4)*(x-1)^3*(x+1)^3*(y-1)^3*(y+1)^3
    # 3D: phi(x,y,z,t) = cos(t-pi/4)*(x-1)^3*(x+1)^3*(y-1)^3*(y+1)^3
    #                      *(z-1)^3*(z+1)^3
    # on [-1, 1]^{#dims}
    # (Manufactured solution)
    def mesh_factory(n):
        from meshmode.mesh.generation import generate_regular_rect_mesh
        return generate_regular_rect_mesh(
                a=(-1.,)*dim,
                b=(1.,)*dim,
                n=(n,)*dim)
    sym_coords = prim.make_sym_vector('x', dim)
    sym_t = pmbl.var("t")
    sym_cos = pmbl.var("cos")
    sym_phi = sym_cos(sym_t - np.pi/4)
    for i in range(dim):
        sym_phi *= (sym_coords[i]-1)**3 * (sym_coords[i]+1)**3
    return (dim, 2., mesh_factory, sym_phi, 0.025)


@pytest.mark.parametrize(("dim", "c", "mesh_factory", "sym_phi", "timestep_scale"),
    [
        get_standing_wave(2),
        get_standing_wave(3),
        get_manufactured_cubic(2),
        get_manufactured_cubic(3)
    ])
@pytest.mark.parametrize("order", [2, 3, 4])
def test_wave(ctx_factory, dim, c, mesh_factory, sym_phi, timestep_scale, order,
            visualize=False):
    """Checks accuracy and stability of the wave operator for a given problem setup.
    :arg dim: Problem dimension.
    :arg c: Sound speed.
    :arg mesh_factory: Creates a mesh given a characteristic size.
    :arg sym_phi: Symbolic expression for the solution.
    :arg timestep_scale: Scaling factor for the timestep in the stability test (tweak
        this to get close to stability limit).
    """

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    # Note: In order to support manufactured solutions, we modify the wave equation
    # to add a source term (f). If the solution is exact, this term should be 0.

    sym_coords = prim.make_sym_vector('x', dim)
    sym_t = pmbl.var("t")

    # f = phi_tt - c^2 * div(grad(phi))
    sym_f = sym.diff(sym_t)(sym.diff(sym_t)(sym_phi)) - c**2\
                * sym.div(sym.grad(dim, sym_phi))

    # u = phi_t
    sym_u = sym.diff(sym_t)(sym_phi)

    # v = c*grad(phi)
    sym_v = [c * sym.diff(sym_coords[i])(sym_phi) for i in range(dim)]

    # rhs(u part) = c*div(v) + f
    # rhs(v part) = c*grad(u)
    sym_rhs = flat_obj_array(
                c * sym.div(sym_v) + sym_f,
                make_obj_array([c]) * sym.grad(dim, sym_u))

    def max_inf_norm(w):
        return np.max(np.array([la.norm(field.get(), np.inf) for field in w]))

    # Check order of accuracy

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for n in [4, 8, 16]:
        mesh = mesh_factory(n)

        from grudge.eager import EagerDGDiscretization
        discr = EagerDGDiscretization(cl_ctx, mesh, order=order)

        nodes = discr.nodes().with_queue(queue)

        def sym_eval(expr, t):
            return sym.EvaluationMapper({"x": nodes, "t": t})(expr)

        t_check = 1.23456789

        u = sym_eval(sym_u, t_check)
        v = sym_eval(sym_v, t_check)

        fields = flat_obj_array(u, v)

        rhs = wave_operator(discr, c=c, w=fields)
        rhs[0] = rhs[0] + sym_eval(sym_f, t_check)

        expected_rhs = sym_eval(sym_rhs, t_check)

        eoc_rec.add_data_point(1./n, max_inf_norm(rhs - expected_rhs))

        if visualize:
            from grudge.shortcuts import make_visualizer
            vis = make_visualizer(discr, discr.order)
            vis.write_vtk_file("wave_accuracy_{order}_{n}.vtu".format(order=order,
                        n=n), [
                            ("u", fields[0]),
                            ("v", fields[1:]),
                            ("rhs_u_actual", rhs[0]),
                            ("rhs_v_actual", rhs[1:]),
                            ("rhs_u_expected", expected_rhs[0]),
                            ("rhs_v_expected", expected_rhs[1:]),
                        ])

    print("Approximation error:")
    print(eoc_rec)
    assert(eoc_rec.order_estimate() >= order - 0.5 or eoc_rec.max_error() < 1e-11)

    # Check stability

    mesh = mesh_factory(8)

    from grudge.eager import EagerDGDiscretization
    discr = EagerDGDiscretization(cl_ctx, mesh, order=order)

    nodes = discr.nodes().with_queue(queue)

    def sym_eval(expr, t):
        return sym.EvaluationMapper({"x": nodes, "t": t})(expr)

    def get_rhs(t, w):
        result = wave_operator(discr, c=c, w=w)
        result[0] += sym_eval(sym_f, t)
        return result

    t = 0.

    u = sym_eval(sym_u, t)
    v = sym_eval(sym_v, t)

    fields = flat_obj_array(u, v)

    from mirgecom.integrators import rk4_step
    dt = timestep_scale/order**2
    for istep in range(10):
        fields = rk4_step(fields, t, dt, get_rhs)
        t += dt

    expected_u = sym_eval(sym_u, 10*dt)
    expected_v = sym_eval(sym_v, 10*dt)
    expected_fields = flat_obj_array(expected_u, expected_v)

    if visualize:
        from grudge.shortcuts import make_visualizer
        vis = make_visualizer(discr, discr.order)
        vis.write_vtk_file("wave_stability.vtu",
                [
                    ("u", fields[0]),
                    ("v", fields[1:]),
                    ("u_expected", expected_fields[0]),
                    ("v_expected", expected_fields[1:]),
                    ])

    err = max_inf_norm(fields-expected_fields)
    max_err = max_inf_norm(expected_fields)

    assert(err < max_err)
