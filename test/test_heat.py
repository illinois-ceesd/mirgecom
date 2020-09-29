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
import pyopencl.array as cla  # noqa
import pyopencl.clmath as clmath # noqa
from pytools.obj_array import make_obj_array
import pymbolic as pmbl
import pymbolic.primitives as prim
import mirgecom.symbolic as sym
from mirgecom.heat import heat_operator
from meshmode.dof_array import thaw, flat_norm

from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

import pytest
import os

import logging
logger = logging.getLogger(__name__)


# Tests below take a problem description as input, which is a tuple
#   (dim, alpha, mesh_factory, sym_u)
# where:
#   dim is the problem dimension
#   alpha is the diffusivity
#   mesh_factory is a factory that creates a mesh given a characteristic size
#   sym_u is a symbolic expression for the solution


def get_decaying_cosine(dim, alpha):
    # 1D: u(x,t) = exp(-alpha*t)*cos(x)
    # 2D: u(x,y,t) = exp(-2*alpha*t)*cos(x)*cos(y)
    # 3D: u(x,y,z,t) = exp(-3*alpha*t)*cos(x)*cos(y)*cos(z)
    # on [-pi/2, pi/2]^{#dims}
    def mesh_factory(n):
        from meshmode.mesh.generation import generate_regular_rect_mesh
        return generate_regular_rect_mesh(
            a=(-0.5*np.pi,)*dim,
            b=(0.5*np.pi,)*dim,
            n=(n,)*dim)
    sym_coords = prim.make_sym_vector("x", dim)
    sym_t = pmbl.var("t")
    sym_cos = pmbl.var("cos")
    sym_exp = pmbl.var("exp")
    sym_u = sym_exp(-dim*alpha*sym_t)
    for i in range(dim):
        sym_u *= sym_cos(sym_coords[i])
    return (dim, alpha, mesh_factory, sym_u)


def sym_heat(dim, sym_u):
    """Return symbolic expressions for the heat equation system given a desired
    solution. (Note: In order to support manufactured solutions, we modify the heat
    equation to add a source term (f). If the solution is exact, this term should
    be 0.)
    """

    sym_alpha = pmbl.var("alpha")
    sym_t = pmbl.var("t")

    # rhs = alpha * div(grad(u))
    sym_rhs = sym_alpha * sym.div(sym.grad(dim, sym_u))

    # f = u_t - rhs
    sym_f = sym.diff(sym_t)(sym_u) - sym_rhs

    return sym_f, sym_rhs


# Note: Must integrate in time for a while in order to achieve expected spatial
# accuracy. Checking the RHS alone will give lower numbers.
#
# Working hypothesis: RHS lives in lower order polynomial space and thus doesn't
# attain full-order convergence.
@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize(("problem", "nsteps", "dt"),
    [
        (get_decaying_cosine(1, 2.), 50, 5.e-5),
        (get_decaying_cosine(2, 2.), 50, 5.e-5),
        (get_decaying_cosine(3, 2.), 50, 5.e-5),
    ])
def test_heat_accuracy_and_stability(actx_factory, problem, nsteps, dt, order,
            visualize=False):
    """Checks accuracy/stability of the heat operator for a given problem setup.
    """
    actx = actx_factory()

    dim, alpha, mesh_factory, sym_u = problem

    sym_f, _ = sym_heat(dim, sym_u)

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for n in [8, 10, 12] if dim == 3 else [8, 16, 24]:
        mesh = mesh_factory(n)

        from grudge.eager import EagerDGDiscretization
        discr = EagerDGDiscretization(actx, mesh, order=order)

        nodes = thaw(actx, discr.nodes())

        def sym_eval(expr, t):
            return sym.EvaluationMapper({"alpha": alpha, "x": nodes, "t": t})(expr)

        def get_rhs(t, w):
            result = heat_operator(discr, alpha=alpha, w=w)
            result[0] += sym_eval(sym_f, t)
            return result

        t = 0.

        fields = make_obj_array([sym_eval(sym_u, t)])

        from mirgecom.integrators import rk4_step

        for istep in range(nsteps):
            fields = rk4_step(fields, t, dt, get_rhs)
            t += dt

        expected_fields = make_obj_array([sym_eval(sym_u, t)])

        rel_linf_err = (
            discr.norm(fields - expected_fields, np.inf)
            / discr.norm(expected_fields, np.inf))
        eoc_rec.add_data_point(1./n, rel_linf_err)

        if visualize:
            from grudge.shortcuts import make_visualizer
            vis = make_visualizer(discr, discr.order+3)
            vis.write_vtk_file("heat_accuracy_{order}_{n}.vtu".format(order=order,
                        n=n), [
                            ("u", fields[0]),
                            ("expected_u", expected_fields[0]),
                            ])

    print("Approximation error:")
    print(eoc_rec)
    # Expected convergence rates from Hesthaven/Warburton book
    expected_order = order+1 if order % 2 == 0 else order
    assert(eoc_rec.order_estimate() >= expected_order - 0.5
                or eoc_rec.max_error() < 1e-11)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("problem",
    [
        get_decaying_cosine(1, 1.)
    ])
def test_heat_compare_to_nodaldg(actx_factory, problem, order, print_err=False):
    """Compares heat operator to Hesthaven/Warburton Nodal-DG code."""
    if "NODALDG" in os.environ:
        nodaldg_path = os.environ["NODALDG"]
    else:
        pytest.skip("Nodal-DG code not detected. Set NODALDG environment variable "
                    "to Nodal-DG code's install path to enable.")

    actx = actx_factory()

    dim, alpha, mesh_factory, sym_u = problem

    assert dim == 1
    assert alpha == 1.

    sym_f, sym_rhs = sym_heat(dim, sym_u)

    for n in [4, 8, 16, 32, 64]:
        mesh = mesh_factory(n)

        from meshmode.interop.nodal_dg import NodalDGContext
        with NodalDGContext(os.path.join(nodaldg_path, "Codes1.1")) as ndgctx:
            ndgctx.set_mesh(mesh, order=order)

            t_check = 1.23456789

            from grudge.eager import EagerDGDiscretization
            discr_mirgecom = EagerDGDiscretization(actx, mesh, order=order)
            nodes_mirgecom = thaw(actx, discr_mirgecom.nodes())

            def sym_eval_mirgecom(expr, t):
                return sym.EvaluationMapper({"alpha": alpha, "x": nodes_mirgecom,
                            "t": t})(expr)

            fields_mirgecom = make_obj_array([sym_eval_mirgecom(sym_u, t_check)])

            rhs_mirgecom = heat_operator(discr_mirgecom, alpha=alpha,
                        w=fields_mirgecom)

            discr_ndg = ndgctx.get_discr(actx)
            nodes_ndg = thaw(actx, discr_ndg.nodes())

            def sym_eval_ndg(expr, t):
                return sym.EvaluationMapper({"alpha": alpha, "x": nodes_ndg,
                            "t": t})(expr)

            fields_ndg = make_obj_array([sym_eval_ndg(sym_u, t_check)])

            ndgctx.push_dof_array("u", fields_ndg[0])
            ndgctx.octave.push("t", t_check)
            ndgctx.octave.eval("[rhs] = HeatCRHS1D(u,t)", verbose=False)
            rhs_ndg = make_obj_array([ndgctx.pull_dof_array(actx, "rhs")])

            err = flat_norm(rhs_mirgecom[0]-rhs_ndg[0], np.inf)/flat_norm(
                rhs_ndg[0], np.inf)

            if print_err:
                rhs_exact = make_obj_array([sym_eval_mirgecom(sym_rhs, t_check)])
                err_exact = (flat_norm(rhs_mirgecom[0]-rhs_exact[0], np.inf)
                    / flat_norm(rhs_exact[0], np.inf))
                print(err, err_exact)

            assert err < 1.e-9


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
