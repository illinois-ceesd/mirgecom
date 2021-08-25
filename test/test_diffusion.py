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
from pymbolic.primitives import Expression
import mirgecom.symbolic as sym
from mirgecom.diffusion import (
    diffusion_operator,
    DirichletDiffusionBoundary,
    NeumannDiffusionBoundary)
from meshmode.dof_array import thaw, DOFArray
from grudge.dof_desc import DTAG_BOUNDARY, DISCR_TAG_BASE, DISCR_TAG_QUAD

from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

import pytest

from dataclasses import dataclass
from typing import Callable, Union
from numbers import Number

import logging
logger = logging.getLogger(__name__)


@dataclass
class HeatProblem:
    """Description of a heat equation problem.

    .. attribute:: dim

        The problem dimension.

    .. attribute:: get_mesh

        A function that creates a mesh when given some characteristic size as input.

    .. attribute:: sym_alpha

        A symbolic expression for the diffusivity.

    .. attribute:: sym_u

        A symbolic expression for the solution.

    .. attribute:: get_boundaries

        A function that creates a btag-to-boundary dict given a discretization, an
        array context, and a time
    """

    dim: int
    get_mesh: Callable
    sym_alpha: Union[Expression, Number]
    sym_u: Expression
    get_boundaries: Callable


def get_box_mesh(dim, a, b, n):
    dim_names = ["x", "y", "z"]
    boundary_tag_to_face = {}
    for i in range(dim):
        boundary_tag_to_face["-"+str(i)] = ["-"+dim_names[i]]
        boundary_tag_to_face["+"+str(i)] = ["+"+dim_names[i]]
    from meshmode.mesh.generation import generate_regular_rect_mesh
    return generate_regular_rect_mesh(a=(a,)*dim, b=(b,)*dim,
        nelements_per_axis=(n,)*dim, boundary_tag_to_face=boundary_tag_to_face)


# 1D: u(x,t) = exp(-alpha*t)*cos(x)
# 2D: u(x,y,t) = exp(-2*alpha*t)*sin(x)*cos(y)
# 3D: u(x,y,z,t) = exp(-3*alpha*t)*sin(x)*sin(y)*cos(z)
# on [-pi/2, pi/2]^{#dims}
def get_decaying_trig(dim, alpha):
    def get_mesh(n):
        return get_box_mesh(dim, -0.5*np.pi, 0.5*np.pi, n)

    sym_coords = pmbl.make_sym_vector("x", dim)
    sym_t = pmbl.var("t")
    sym_cos = pmbl.var("cos")
    sym_sin = pmbl.var("sin")
    sym_exp = pmbl.var("exp")

    sym_u = sym_exp(-dim*alpha*sym_t)
    for i in range(dim-1):
        sym_u *= sym_sin(sym_coords[i])
    sym_u *= sym_cos(sym_coords[dim-1])

    def get_boundaries(discr, actx, t):
        boundaries = {}

        for i in range(dim-1):
            boundaries[DTAG_BOUNDARY("-"+str(i))] = NeumannDiffusionBoundary(0.)
            boundaries[DTAG_BOUNDARY("+"+str(i))] = NeumannDiffusionBoundary(0.)

        boundaries[DTAG_BOUNDARY("-"+str(dim-1))] = DirichletDiffusionBoundary(0.)
        boundaries[DTAG_BOUNDARY("+"+str(dim-1))] = DirichletDiffusionBoundary(0.)

        return boundaries

    return HeatProblem(dim, get_mesh, alpha, sym_u, get_boundaries)


# 1D: u(x,t) = exp(-alpha*t)*cos(x)
# 2D: u(x,y,t) = exp(-2*alpha*t)*sin(x)*cos(y)
# 3D: u(x,y,z,t) = exp(-3*alpha*t)*sin(x)*sin(y)*cos(z)
# on [-pi/2, pi/4]^{#dims}
def get_decaying_trig_truncated_domain(dim, alpha):
    def get_mesh(n):
        return get_box_mesh(dim, -0.5*np.pi, 0.25*np.pi, n)

    sym_coords = pmbl.make_sym_vector("x", dim)
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

        exact_u = sym_eval(sym_u)
        exact_grad_u = sym_eval(sym.grad(dim, sym_u))

        boundaries = {}

        for i in range(dim-1):
            lower_btag = DTAG_BOUNDARY("-"+str(i))
            upper_btag = DTAG_BOUNDARY("+"+str(i))
            upper_grad_u = discr.project("vol", upper_btag, exact_grad_u)
            normal = thaw(actx, discr.normal(upper_btag))
            upper_grad_u_dot_n = np.dot(upper_grad_u, normal)
            boundaries[lower_btag] = NeumannDiffusionBoundary(0.)
            boundaries[upper_btag] = NeumannDiffusionBoundary(upper_grad_u_dot_n)

        lower_btag = DTAG_BOUNDARY("-"+str(dim-1))
        upper_btag = DTAG_BOUNDARY("+"+str(dim-1))
        upper_u = discr.project("vol", upper_btag, exact_u)
        boundaries[lower_btag] = DirichletDiffusionBoundary(0.)
        boundaries[upper_btag] = DirichletDiffusionBoundary(upper_u)

        return boundaries

    return HeatProblem(dim, get_mesh, alpha, sym_u, get_boundaries)


# 1D: alpha(x) = 1+0.2*cos(3*x)
#     u(x,t)   = cos(x)
# 2D: alpha(x,y) = 1+0.2*cos(3*x)*cos(3*y)
#     u(x,y,t)   = sin(x)*cos(y)
# 3D: alpha(x,y,z) = 1+0.2*cos(3*x)*cos(3*y)*cos(3*z)
#     u(x,y,z,t)   = sin(x)*sin(y)*cos(z)
# on [-pi/2, pi/2]^{#dims}
def get_static_trig_var_diff(dim):
    def get_mesh(n):
        return get_box_mesh(dim, -0.5*np.pi, 0.5*np.pi, n)

    sym_coords = pmbl.make_sym_vector("x", dim)
    sym_cos = pmbl.var("cos")
    sym_sin = pmbl.var("sin")

    sym_alpha = 1
    for i in range(dim):
        sym_alpha *= sym_cos(3.*sym_coords[i])
    sym_alpha = 1 + 0.2*sym_alpha

    sym_u = 1
    for i in range(dim-1):
        sym_u *= sym_sin(sym_coords[i])
    sym_u *= sym_cos(sym_coords[dim-1])

    def get_boundaries(discr, actx, t):
        boundaries = {}

        for i in range(dim-1):
            boundaries[DTAG_BOUNDARY("-"+str(i))] = NeumannDiffusionBoundary(0.)
            boundaries[DTAG_BOUNDARY("+"+str(i))] = NeumannDiffusionBoundary(0.)

        boundaries[DTAG_BOUNDARY("-"+str(dim-1))] = DirichletDiffusionBoundary(0.)
        boundaries[DTAG_BOUNDARY("+"+str(dim-1))] = DirichletDiffusionBoundary(0.)

        return boundaries

    return HeatProblem(dim, get_mesh, sym_alpha, sym_u, get_boundaries)


def sym_diffusion(dim, sym_alpha, sym_u):
    """Return a symbolic expression for the diffusion operator applied to a function.
    """
    return sym.div(sym_alpha * sym.grad(dim, sym_u))


# Note: Must integrate in time for a while in order to achieve expected spatial
# accuracy. Checking the RHS alone will give lower numbers.
#
# Working hypothesis: RHS lives in lower order polynomial space and thus doesn't
# attain full-order convergence.
@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize(("problem", "nsteps", "dt", "scales"),
    [
        (get_decaying_trig_truncated_domain(1, 2.), 50, 5.e-5, [8, 16, 24]),
        (get_decaying_trig_truncated_domain(2, 2.), 50, 5.e-5, [8, 12, 16]),
        (get_decaying_trig_truncated_domain(3, 2.), 50, 5.e-5, [8, 10, 12]),
        (get_static_trig_var_diff(1), 50, 5.e-5, [8, 16, 24]),
        (get_static_trig_var_diff(2), 50, 5.e-5, [12, 14, 16]),
        (get_static_trig_var_diff(3), 50, 5.e-5, [8, 10, 12]),
    ])
def test_diffusion_accuracy(actx_factory, problem, nsteps, dt, scales, order,
            visualize=False):
    """
    Checks the accuracy of the diffusion operator by solving the heat equation for a
    given problem setup.
    """
    actx = actx_factory()

    p = problem

    sym_diffusion_u = sym_diffusion(p.dim, p.sym_alpha, p.sym_u)

    # In order to support manufactured solutions, we modify the heat equation
    # to add a source term f. If the solution is exact, this term should be 0.
    sym_t = pmbl.var("t")
    sym_f = sym.diff(sym_t)(p.sym_u) - sym_diffusion_u

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for n in scales:
        mesh = p.get_mesh(n)

        from grudge.eager import EagerDGDiscretization
        from meshmode.discretization.poly_element import \
                QuadratureSimplexGroupFactory, \
                PolynomialWarpAndBlendGroupFactory
        discr = EagerDGDiscretization(
            actx, mesh,
            discr_tag_to_group_factory={
                DISCR_TAG_BASE: PolynomialWarpAndBlendGroupFactory(order),
                DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(3*order),
            }
        )

        nodes = thaw(actx, discr.nodes())

        def sym_eval(expr, t):
            return sym.EvaluationMapper({"x": nodes, "t": t})(expr)

        alpha = sym_eval(p.sym_alpha, 0.)

        if isinstance(alpha, DOFArray):
            discr_tag = DISCR_TAG_QUAD
        else:
            discr_tag = DISCR_TAG_BASE

        def get_rhs(t, u):
            return (diffusion_operator(discr, quad_tag=discr_tag, alpha=alpha,
                    boundaries=p.get_boundaries(discr, actx, t), u=u)
                + sym_eval(sym_f, t))

        t = 0.

        u = sym_eval(p.sym_u, t)

        from mirgecom.integrators import rk4_step

        for _ in range(nsteps):
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
def test_diffusion_discontinuous_alpha(actx_factory, order, visualize=False):
    """
    Checks the accuracy of the diffusion operator for an alpha field that has a
    jump across an element face.
    """
    actx = actx_factory()

    n = 8

    mesh = get_box_mesh(1, -1, 1, n)

    from grudge.eager import EagerDGDiscretization
    discr = EagerDGDiscretization(actx, mesh, order=order)

    nodes = thaw(actx, discr.nodes())

    # Set up a 1D heat equation interface problem, apply the diffusion operator to
    # the exact steady state solution, and check that it's zero

    lower_mask_np = np.empty((n, order+1), dtype=int)
    lower_mask_np[:, :] = 0
    lower_mask_np[:int(n/2), :] = 1
    lower_mask = DOFArray(actx, (actx.from_numpy(lower_mask_np),))

    upper_mask_np = np.empty((n, order+1), dtype=int)
    upper_mask_np[:, :] = 0
    upper_mask_np[int(n/2):, :] = 1
    upper_mask = DOFArray(actx, (actx.from_numpy(upper_mask_np),))

    alpha_lower = 0.5
    alpha_upper = 1

    alpha = alpha_lower * lower_mask + alpha_upper * upper_mask

    boundaries = {
        DTAG_BOUNDARY("-0"): DirichletDiffusionBoundary(0.),
        DTAG_BOUNDARY("+0"): DirichletDiffusionBoundary(1.),
    }

    flux = -alpha_lower*alpha_upper/(alpha_lower + alpha_upper)

    u_steady = (
              -flux/alpha_lower * (nodes[0] + 1)  * lower_mask  # noqa: E126, E221
        + (1 - flux/alpha_upper * (nodes[0] - 1)) * upper_mask)  # noqa: E131

    def get_rhs(t, u):
        return diffusion_operator(
            discr, quad_tag=DISCR_TAG_BASE, alpha=alpha, boundaries=boundaries, u=u)

    rhs = get_rhs(0, u_steady)

    if visualize:
        from grudge.shortcuts import make_visualizer
        vis = make_visualizer(discr, discr.order+3)
        vis.write_vtk_file("diffusion_discontinuous_alpha_rhs_{order}.vtu"
            .format(order=order), [
                ("alpha", alpha),
                ("u_steady", u_steady),
                ("rhs", rhs),
                ])

    linf_err = discr.norm(rhs, np.inf)
    assert(linf_err < 1e-11)

    # Now check stability

    from numpy.random import rand
    perturb_np = np.empty((n, order+1), dtype=float)
    for i in range(n):
        perturb_np[i, :] = 0.1*(rand()-0.5)
    perturb = DOFArray(actx, (actx.from_numpy(perturb_np),))

    u = u_steady + perturb

    dt = 1e-3 / order**2
    t = 0

    from mirgecom.integrators import rk4_step

    for _ in range(50):
        u = rk4_step(u, t, dt, get_rhs)
        t += dt

    if visualize:
        vis.write_vtk_file("diffusion_disc_alpha_stability_{order}.vtu"
            .format(order=order), [
                ("alpha", alpha),
                ("u", u),
                ("u_steady", u_steady),
                ])

    linf_diff = discr.norm(u - u_steady, np.inf)
    assert linf_diff < 0.1


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("problem",
    [
        get_decaying_trig(1, 1.)
    ])
@pytest.mark.octave
def test_diffusion_compare_to_nodal_dg(actx_factory, problem, order,
            print_err=False):
    """Compares diffusion operator to Hesthaven/Warburton Nodal-DG code."""
    pytest.importorskip("oct2py")

    actx = actx_factory()

    p = problem

    assert p.dim == 1
    assert p.sym_alpha == 1.

    from meshmode.interop.nodal_dg import download_nodal_dg_if_not_present
    download_nodal_dg_if_not_present()

    sym_diffusion_u = sym_diffusion(p.dim, p.sym_alpha, p.sym_u)

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

            diffusion_u_mirgecom = diffusion_operator(discr_mirgecom,
                quad_tag=DISCR_TAG_BASE, alpha=discr_mirgecom.zeros(actx)+1.,
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

            def inf_norm(f):
                return actx.np.linalg.norm(f, np.inf)

            err = (inf_norm(diffusion_u_mirgecom-diffusion_u_ndg)
                        / inf_norm(diffusion_u_ndg))

            if print_err:
                diffusion_u_exact = sym_eval_mirgecom(sym_diffusion_u)
                err_exact = (inf_norm(diffusion_u_mirgecom-diffusion_u_exact)
                            / inf_norm(diffusion_u_exact))
                print(err, err_exact)

            assert err < 1e-9


def test_diffusion_obj_array_vectorize(actx_factory):
    """
    Checks that the diffusion operator can be called with either scalars or object
    arrays for `u`.
    """
    actx = actx_factory()

    p = get_decaying_trig(1, 2.)

    assert isinstance(p.sym_alpha, Number)

    sym_u1 = p.sym_u
    sym_u2 = 2*p.sym_u

    sym_diffusion_u1 = sym_diffusion(p.dim, p.sym_alpha, sym_u1)
    sym_diffusion_u2 = sym_diffusion(p.dim, p.sym_alpha, sym_u2)

    n = 128

    mesh = p.get_mesh(n)

    from grudge.eager import EagerDGDiscretization
    discr = EagerDGDiscretization(actx, mesh, order=4)

    nodes = thaw(actx, discr.nodes())

    t = 1.23456789

    def sym_eval(expr):
        return sym.EvaluationMapper({"x": nodes, "t": t})(expr)

    alpha = sym_eval(p.sym_alpha)

    u1 = sym_eval(sym_u1)
    u2 = sym_eval(sym_u2)

    boundaries = p.get_boundaries(discr, actx, t)

    diffusion_u1 = diffusion_operator(discr, quad_tag=DISCR_TAG_BASE, alpha=alpha,
        boundaries=boundaries, u=u1)

    assert isinstance(diffusion_u1, DOFArray)

    expected_diffusion_u1 = sym_eval(sym_diffusion_u1)
    rel_linf_err = (
        discr.norm(diffusion_u1 - expected_diffusion_u1, np.inf)
        / discr.norm(expected_diffusion_u1, np.inf))
    assert rel_linf_err < 1.e-5

    boundaries_vector = [boundaries, boundaries]
    u_vector = make_obj_array([u1, u2])

    diffusion_u_vector = diffusion_operator(
        discr, quad_tag=DISCR_TAG_BASE, alpha=alpha,
        boundaries=boundaries_vector, u=u_vector
    )

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
