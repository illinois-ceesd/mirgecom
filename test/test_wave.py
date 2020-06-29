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
from pytools.obj_array import flat_obj_array, make_obj_array
import pymbolic as pmbl
import pymbolic.primitives as prim
import pymbolic.mapper.evaluator as ev

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)

import pytest

import logging
logger = logging.getLogger(__name__)


def sym_diff(var):
    from pymbolic.mapper.differentiator import DifferentiationMapper

    def func_map(arg_index, func, arg, allowed_nonsmoothness):
        if func == pmbl.var("sin"):
            return pmbl.var("cos")(*arg)
        elif func == pmbl.var("cos"):
            return -pmbl.var("sin")(*arg)
        else:
            raise ValueError("Unrecognized function")

    return DifferentiationMapper(var, func_map=func_map)


def sym_div(vector_func):
    coord_names = ["x", "y", "z"]
    div = 0
    for i in range(len(vector_func)):
        div = div + sym_diff(pmbl.var(coord_names[i]))(vector_func[i])
    return div


def sym_grad(dim, func):
    coord_names = ["x", "y", "z"]
    grad = []
    for i in range(dim):
        grad_i = sym_diff(pmbl.var(coord_names[i]))(func)
        grad.append(grad_i)
    return grad


class EvaluationMapper(ev.EvaluationMapper):
    def map_call(self, expr):
        assert isinstance(expr.function, prim.Variable)
        if expr.function.name == "sin":
            par, = expr.parameters
            return self._sin(self.rec(par))
        elif expr.function.name == "cos":
            par, = expr.parameters
            return self._cos(self.rec(par))
        else:
            raise ValueError("Unrecognized function '%s'" % expr.function)

    def _sin(self, val):
        from numbers import Number
        if isinstance(val, Number):
            return np.sin(val)
        else:
            return clmath.sin(val)

    def _cos(self, val):
        from numbers import Number
        if isinstance(val, Number):
            return np.cos(val)
        else:
            return clmath.cos(val)


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
    sym_coords = [pmbl.var(name) for name in ["x", "y", "z"]]
    sym_t = pmbl.var("t")
    sym_cos = pmbl.var("cos")
    sym_phi = sym_cos(np.sqrt(dim)*c*sym_t - np.pi/4)
    for i in range(dim):
        sym_phi = sym_phi * sym_cos(sym_coords[i])
    return (dim, c, mesh_factory, sym_phi)


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
    sym_coords = [pmbl.var(name) for name in ["x", "y", "z"]]
    sym_t = pmbl.var("t")
    sym_cos = pmbl.var("cos")
    sym_phi = sym_cos(sym_t - np.pi/4)
    for i in range(dim):
        sym_phi = sym_phi * (sym_coords[i]-1)**3 * (sym_coords[i]+1)**3
    return (dim, 2., mesh_factory, sym_phi)


@pytest.mark.parametrize(("dim", "c", "mesh_factory", "sym_phi"),
    [
        get_standing_wave(2),
        get_standing_wave(3),
        get_manufactured_cubic(2),
        get_manufactured_cubic(3)
    ])
@pytest.mark.parametrize("order", [2, 3, 4])
def test_wave(ctx_factory, dim, order, c, mesh_factory, sym_phi):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for n in [4, 8, 16]:
        mesh = mesh_factory(n)

        from grudge.eager import EagerDGDiscretization
        discr = EagerDGDiscretization(cl_ctx, mesh, order=order)

        nodes = discr.nodes().with_queue(queue)

        sym_coords = [pmbl.var(name) for name in ["x", "y", "z"]]
        sym_t = pmbl.var("t")

        # f = phi_tt - c^2 * div(grad(phi))
        sym_f = sym_diff(sym_t)(sym_diff(sym_t)(sym_phi)) - c**2\
                    * sym_div(sym_grad(dim, sym_phi))

        # u = phi_t
        sym_u = sym_diff(sym_t)(sym_phi)

        # v = c*grad(phi)
        sym_v = [c * sym_diff(sym_coords[i])(sym_phi) for i in range(dim)]

        # rhs(u part) = c*div(v) + f
        # rhs(v part) = c*grad(u)
        sym_rhs = flat_obj_array(
                    c * sym_div(sym_v) + sym_f,
                    make_obj_array([c]) * sym_grad(dim, sym_u))

        def sym_eval(expr, t):
            coord_names = ["x", "y", "z"]
            eval_values = {coord_names[i]: nodes[i] for i in range(dim)}
            eval_values["t"] = t
            return EvaluationMapper(eval_values)(expr)

        u = sym_eval(sym_u, 0.)
        v = sym_eval(sym_v, 0.)
        f = sym_eval(sym_f, 0.)

        from mirgecom.wave import wave_operator
        rhs = wave_operator(discr, c=c, w=flat_obj_array(u, v))
        rhs[0] = rhs[0] + f

        expected_rhs = sym_eval(sym_rhs, 0.)

        err = np.max(np.array([la.norm((rhs[i] - expected_rhs[i]).get(), np.inf)
                for i in range(dim+1)]))
        eoc_rec.add_data_point(1./n, err)

        # from grudge.shortcuts import make_visualizer
        # vis = make_visualizer(discr, discr.order)
        # vis.write_vtk_file("result_{n}.vtu".format(n=n),
        #         [
        #             ("u", u),
        #             ("v", v),
        #             ("rhs_u_actual", rhs[0]),
        #             ("rhs_v_actual", rhs[1:]),
        #             ("rhs_u_expected", expected_rhs[0]),
        #             ("rhs_v_expected", expected_rhs[1:]),
        #             ])

    print("Approximation error:")
    print(eoc_rec)
    assert(eoc_rec.order_estimate() >= order - 0.5 or eoc_rec.max_error() < 1e-11)
