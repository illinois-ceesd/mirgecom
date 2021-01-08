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
from mirgecom.diffusion import (
    diffusion_operator,
    DirichletDiffusionBoundary,
    NeumannDiffusionBoundary)
from meshmode.dof_array import thaw, flat_norm, DOFArray
from grudge import sym as grudge_sym

from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

import pytest

from dataclasses import dataclass
from typing import Callable

import logging
logger = logging.getLogger(__name__)


@dataclass
class HeatProblem:
    """Description of a heat equation problem.

    .. attribute:: dim

        The problem dimension.

    .. attribute:: alpha

        The diffusivity.

    .. attribute:: get_mesh

        A function that creates a mesh when given some characteristic size as input.

    .. attribute:: sym_u

        A symbolic expresssion for the solution.

    .. attribute:: get_boundaries

        A function that creates a btag-to-boundary dict given a discretization, an
        array context, and a time
    """

    dim: int
    alpha: float
    get_mesh: Callable
    sym_u: prim.Expression
    get_boundaries: Callable


def get_decaying_trig(dim, alpha):
    # 1D: u(x,t) = exp(-alpha*t)*cos(x)
    # 2D: u(x,y,t) = exp(-2*alpha*t)*sin(x)*cos(y)
    # 3D: u(x,y,z,t) = exp(-3*alpha*t)*sin(x)*sin(y)*cos(z)
    # on [-pi/2, pi/2]^{#dims}
    def get_mesh(n):
        dim_names = ["x", "y", "z"]
        neumann_boundaries = []
        for i in range(dim-1):
            neumann_boundaries += ["+"+dim_names[i], "-"+dim_names[i]]
        dirichlet_boundaries = ["+"+dim_names[dim-1], "-"+dim_names[dim-1]]
        from meshmode.mesh.generation import generate_regular_rect_mesh
        return generate_regular_rect_mesh(
            a=(-0.5*np.pi,)*dim,
            b=(0.5*np.pi,)*dim,
            n=(n,)*dim,
            boundary_tag_to_face={
                "dirichlet": dirichlet_boundaries,
                "neumann": neumann_boundaries
                })

    sym_coords = prim.make_sym_vector("x", dim)
    sym_t = pmbl.var("t")
    sym_cos = pmbl.var("cos")
    sym_sin = pmbl.var("sin")
    sym_exp = pmbl.var("exp")
    sym_u = sym_exp(-dim*alpha*sym_t)
    for i in range(dim-1):
        sym_u *= sym_sin(sym_coords[i])
    sym_u *= sym_cos(sym_coords[dim-1])

    def get_boundaries(discr, actx, t):
        return {
            grudge_sym.DTAG_BOUNDARY("dirichlet"): DirichletDiffusionBoundary(0.),
            grudge_sym.DTAG_BOUNDARY("neumann"): NeumannDiffusionBoundary(0.),
        }

    return HeatProblem(dim, alpha, get_mesh, sym_u, get_boundaries)


def get_decaying_trig_truncated_domain(dim, alpha):
    # 1D: u(x,t) = exp(-alpha*t)*cos(x)
    # 2D: u(x,y,t) = exp(-2*alpha*t)*sin(x)*cos(y)
    # 3D: u(x,y,z,t) = exp(-3*alpha*t)*sin(x)*sin(y)*cos(z)
    # on [-pi/2, pi/4]^{#dims}
    def get_mesh(n):
        dim_names = ["x", "y", "z"]
        neumann_lower_boundaries = ["-"+dim_names[i] for i in range(dim-1)]
        neumann_upper_boundaries = ["+"+dim_names[i] for i in range(dim-1)]
        dirichlet_lower_boundaries = ["-"+dim_names[dim-1]]
        dirichlet_upper_boundaries = ["+"+dim_names[dim-1]]
        from meshmode.mesh.generation import generate_regular_rect_mesh
        return generate_regular_rect_mesh(
            a=(-0.5*np.pi,)*dim,
            b=(0.25*np.pi,)*dim,
            n=(n,)*dim,
            boundary_tag_to_face={
                "dirichlet_lower": dirichlet_lower_boundaries,
                "dirichlet_upper": dirichlet_upper_boundaries,
                "neumann_lower": neumann_lower_boundaries,
                "neumann_upper": neumann_upper_boundaries
                })

    sym_coords = prim.make_sym_vector("x", dim)
    sym_t = pmbl.var("t")
    sym_cos = pmbl.var("cos")
    sym_sin = pmbl.var("sin")
    sym_exp = pmbl.var("exp")
    sym_u = sym_exp(-dim*alpha*sym_t)
    for i in range(dim-1):
        sym_u *= sym_sin(sym_coords[i])
    sym_u *= sym_cos(sym_coords[dim-1])

    def get_boundaries(discr, actx, t):
        nodes = thaw(actx, discr.nodes())

        def sym_eval(expr):
            return sym.EvaluationMapper({"x": nodes, "t": t})(expr)

        dirichlet_lower_btag = grudge_sym.DTAG_BOUNDARY("dirichlet_lower")
        dirichlet_upper_btag = grudge_sym.DTAG_BOUNDARY("dirichlet_upper")
        neumann_lower_btag = grudge_sym.DTAG_BOUNDARY("neumann_lower")
        neumann_upper_btag = grudge_sym.DTAG_BOUNDARY("neumann_upper")
        exact_u = sym_eval(sym_u)
        exact_grad_u = make_obj_array(sym_eval(sym.grad(dim, sym_u)))
        upper_u = discr.project("vol", dirichlet_upper_btag, exact_u)
        upper_grad_u = discr.project("vol", neumann_upper_btag, exact_grad_u)
        normal = thaw(actx, discr.normal(neumann_upper_btag))
        upper_grad_u_dot_n = np.dot(upper_grad_u, normal)

        return {
            dirichlet_lower_btag: DirichletDiffusionBoundary(0.),
            dirichlet_upper_btag: DirichletDiffusionBoundary(upper_u),
            neumann_lower_btag: NeumannDiffusionBoundary(0.),
            neumann_upper_btag: NeumannDiffusionBoundary(upper_grad_u_dot_n)
        }

    return HeatProblem(dim, alpha, get_mesh, sym_u, get_boundaries)


def sym_diffusion(dim, alpha, sym_u):
    """Return a symbolic expression for the diffusion operator applied to a function.
    """
    return alpha * sym.div(sym.grad(dim, sym_u))


# Note: Must integrate in time for a while in order to achieve expected spatial
# accuracy. Checking the RHS alone will give lower numbers.
#
# Working hypothesis: RHS lives in lower order polynomial space and thus doesn't
# attain full-order convergence.
@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize(("problem", "nsteps", "dt", "scales"),
    [
        (get_decaying_trig_truncated_domain(1, 2.), 50, 5.e-5, [8, 12, 16]),
        (get_decaying_trig_truncated_domain(2, 2.), 50, 5.e-5, [8, 12, 16]),
        (get_decaying_trig_truncated_domain(3, 2.), 50, 5.e-5, [8, 10, 12]),
    ])
def test_diffusion_accuracy(actx_factory, problem, nsteps, dt, scales, order,
            visualize=False):
    """
    Checks the accuracy of the diffusion operator by solving the heat equation for a
    given problem setup.
    """
    actx = actx_factory()

    p = problem

    sym_diffusion_u = sym_diffusion(p.dim, p.alpha, p.sym_u)

    # In order to support manufactured solutions, we modify the heat equation
    # to add a source term f. If the solution is exact, this term should be 0.
    sym_t = pmbl.var("t")
    sym_f = sym.diff(sym_t)(p.sym_u) - sym_diffusion_u

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for n in scales:
        mesh = p.get_mesh(n)

        from grudge.eager import EagerDGDiscretization
        discr = EagerDGDiscretization(actx, mesh, order=order)

        nodes = thaw(actx, discr.nodes())

        def sym_eval(expr, t):
            return sym.EvaluationMapper({"x": nodes, "t": t})(expr)

        def get_rhs(t, u):
            result = (diffusion_operator(discr, alpha=p.alpha,
                boundaries=p.get_boundaries(discr, actx, t), u=u)
                + sym_eval(sym_f, t))
            return result

        t = 0.

        u = sym_eval(p.sym_u, t)

        from mirgecom.integrators import rk4_step

        for istep in range(nsteps):
            u = rk4_step(u, t, dt, get_rhs)
            t += dt

        expected_u = sym_eval(p.sym_u, t)

        rel_linf_err = (
            discr.norm(u - expected_u, np.inf)
            / discr.norm(expected_u, np.inf))
        eoc_rec.add_data_point(1./n, rel_linf_err)

        if visualize:
            from grudge.shortcuts import make_visualizer
            vis = make_visualizer(discr, discr.order+3)
            vis.write_vtk_file("diffusion_accuracy_{order}_{n}.vtu".format(
                order=order, n=n), [
                    ("u", u),
                    ("expected_u", expected_u),
                    ])

    print("L^inf error:")
    print(eoc_rec)
    # Expected convergence rates from Hesthaven/Warburton book
    expected_order = order+1 if order % 2 == 0 else order
    assert(eoc_rec.order_estimate() >= expected_order - 0.5
                or eoc_rec.max_error() < 1e-11)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("problem",
    [
        get_decaying_trig(1, 1.)
    ])
def test_diffusion_compare_to_nodal_dg(actx_factory, problem, order,
            print_err=False):
    """Compares diffusion operator to Hesthaven/Warburton Nodal-DG code."""
    pytest.importorskip("oct2py")

    actx = actx_factory()

    p = problem

    assert p.dim == 1
    assert p.alpha == 1.

    from meshmode.interop.nodal_dg import download_nodal_dg_if_not_present
    download_nodal_dg_if_not_present()

    sym_diffusion_u = sym_diffusion(p.dim, p.alpha, p.sym_u)

    for n in [4, 8, 16, 32, 64]:
        mesh = p.get_mesh(n)

        from meshmode.interop.nodal_dg import NodalDGContext
        with NodalDGContext("./nodal-dg/Codes1.1") as ndgctx:
            ndgctx.set_mesh(mesh, order=order)

            t = 1.23456789

            from grudge.eager import EagerDGDiscretization
            discr_mirgecom = EagerDGDiscretization(actx, mesh, order=order)
            nodes_mirgecom = thaw(actx, discr_mirgecom.nodes())

            def sym_eval_mirgecom(expr):
                return sym.EvaluationMapper({"x": nodes_mirgecom, "t": t})(expr)

            u_mirgecom = sym_eval_mirgecom(p.sym_u)

            diffusion_u_mirgecom = diffusion_operator(discr_mirgecom, alpha=p.alpha,
                boundaries=p.get_boundaries(discr_mirgecom, actx, t), u=u_mirgecom)

            discr_ndg = ndgctx.get_discr(actx)
            nodes_ndg = thaw(actx, discr_ndg.nodes())

            def sym_eval_ndg(expr):
                return sym.EvaluationMapper({"x": nodes_ndg, "t": t})(expr)

            u_ndg = sym_eval_ndg(p.sym_u)

            ndgctx.push_dof_array("u", u_ndg)
            ndgctx.octave.push("t", t)
            ndgctx.octave.eval("[rhs] = HeatCRHS1D(u,t)", verbose=False)
            diffusion_u_ndg = ndgctx.pull_dof_array(actx, "rhs")

            err = (flat_norm(diffusion_u_mirgecom-diffusion_u_ndg, np.inf)
                        / flat_norm(diffusion_u_ndg, np.inf))

            if print_err:
                diffusion_u_exact = sym_eval_mirgecom(sym_diffusion_u)
                err_exact = (flat_norm(diffusion_u_mirgecom-diffusion_u_exact,
                            np.inf) / flat_norm(diffusion_u_exact, np.inf))
                print(err, err_exact)

            assert err < 1e-9


def test_diffusion_obj_array_vectorize(actx_factory):
    """
    Checks that the diffusion operator can be called with either scalars or object
    arrays for `u`.
    """
    actx = actx_factory()

    p = get_decaying_trig(1, 2.)

    sym_u1 = p.sym_u
    sym_u2 = 2*p.sym_u

    sym_diffusion_u1 = sym_diffusion(p.dim, p.alpha, sym_u1)
    sym_diffusion_u2 = sym_diffusion(p.dim, p.alpha, sym_u2)

    n = 128

    mesh = p.get_mesh(n)

    from grudge.eager import EagerDGDiscretization
    discr = EagerDGDiscretization(actx, mesh, order=4)

    nodes = thaw(actx, discr.nodes())

    t = 1.23456789

    def sym_eval(expr):
        return sym.EvaluationMapper({"x": nodes, "t": t})(expr)

    u1 = sym_eval(sym_u1)
    u2 = sym_eval(sym_u2)

    boundaries = p.get_boundaries(discr, actx, t)

    diffusion_u1 = diffusion_operator(discr, alpha=p.alpha, boundaries=boundaries,
                u=u1)

    assert isinstance(diffusion_u1, DOFArray)

    expected_diffusion_u1 = sym_eval(sym_diffusion_u1)
    rel_linf_err = (
        discr.norm(diffusion_u1 - expected_diffusion_u1, np.inf)
        / discr.norm(expected_diffusion_u1, np.inf))
    assert rel_linf_err < 1.e-5

    boundaries_vector = [boundaries, boundaries]
    u_vector = make_obj_array([u1, u2])

    diffusion_u_vector = diffusion_operator(discr, alpha=p.alpha,
        boundaries=boundaries_vector, u=u_vector)

    assert isinstance(diffusion_u_vector, np.ndarray)
    assert diffusion_u_vector.shape == (2,)

    expected_diffusion_u_vector = make_obj_array([
        sym_eval(sym_diffusion_u1),
        sym_eval(sym_diffusion_u2)
    ])
    rel_linf_err = (
        discr.norm(diffusion_u_vector - expected_diffusion_u_vector, np.inf)
        / discr.norm(expected_diffusion_u_vector, np.inf))
    assert rel_linf_err < 1.e-5


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
