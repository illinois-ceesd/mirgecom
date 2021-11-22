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

from abc import ABCMeta, abstractmethod
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

from numbers import Number

import logging
logger = logging.getLogger(__name__)


def _np_like_for(val):
    from arraycontext import get_container_context_recursively
    actx = get_container_context_recursively(val)
    if actx is None:
        return np
    else:
        return actx.np


def _cos(val):
    if isinstance(val, Expression):
        return pmbl.var("cos")(val)
    else:
        return _np_like_for(val).cos(val)


def _sin(val):
    if isinstance(val, Expression):
        return pmbl.var("sin")(val)
    else:
        return _np_like_for(val).sin(val)


def _exp(val):
    if isinstance(val, Expression):
        return pmbl.var("exp")(val)
    else:
        return _np_like_for(val).exp(val)


def _sym_eval(expr, **kwargs):
    return sym.EvaluationMapper(kwargs)(expr)


class HeatProblem(metaclass=ABCMeta):
    """
    Description of a heat equation problem.

    .. autoproperty:: dim
    .. automethod:: get_mesh
    .. automethod:: get_alpha
    .. automethod:: get_solution
    .. automethod:: get_boundaries
    """

    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        """The problem dimension."""
        return self._dim

    @abstractmethod
    def get_mesh(self, n):
        """Generate and return a mesh of some given characteristic size *n*."""
        pass

    @abstractmethod
    def get_solution(self, x, t):
        """Return the solution for coordinates *x* and time *t*."""
        pass

    @abstractmethod
    def get_alpha(self, x, t, u):
        """Return the diffusivity for coordinates *x*, time *t*, and solution *u*."""
        pass

    @abstractmethod
    def get_boundaries(self, discr, actx, t):
        """
        Return a :class:`dict` that maps boundary tags to boundary conditions at
        time *t*.
        """
        pass


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
class DecayingTrig(HeatProblem):
    def __init__(self, dim, alpha):
        super().__init__(dim)
        self._alpha = alpha

    def get_mesh(self, n):
        return get_box_mesh(self.dim, -0.5*np.pi, 0.5*np.pi, n)

    def get_solution(self, x, t):
        u = _exp(-self.dim*self._alpha*t)
        for i in range(self.dim-1):
            u *= _sin(x[i])
        u *= _cos(x[self.dim-1])
        return u

    def get_alpha(self, x, t, u):
        return self._alpha

    def get_boundaries(self, discr, actx, t):
        boundaries = {}

        for i in range(self.dim-1):
            lower_btag = DTAG_BOUNDARY("-"+str(i))
            upper_btag = DTAG_BOUNDARY("+"+str(i))
            boundaries[lower_btag] = NeumannDiffusionBoundary(0.)
            boundaries[upper_btag] = NeumannDiffusionBoundary(0.)
        lower_btag = DTAG_BOUNDARY("-"+str(self.dim-1))
        upper_btag = DTAG_BOUNDARY("+"+str(self.dim-1))
        boundaries[lower_btag] = DirichletDiffusionBoundary(0.)
        boundaries[upper_btag] = DirichletDiffusionBoundary(0.)

        return boundaries


# 1D: u(x,t) = exp(-alpha*t)*cos(x)
# 2D: u(x,y,t) = exp(-2*alpha*t)*sin(x)*cos(y)
# 3D: u(x,y,z,t) = exp(-3*alpha*t)*sin(x)*sin(y)*cos(z)
# on [-pi/2, pi/4]^{#dims}
class DecayingTrigTruncatedDomain(HeatProblem):
    def __init__(self, dim, alpha):
        super().__init__(dim)
        self._alpha = alpha

    def get_mesh(self, n):
        return get_box_mesh(self.dim, -0.5*np.pi, 0.25*np.pi, n)

    def get_solution(self, x, t):
        u = _exp(-self.dim*self._alpha*t)
        for i in range(self.dim-1):
            u *= _sin(x[i])
        u *= _cos(x[self.dim-1])
        return u

    def get_alpha(self, x, t, u):
        return self._alpha

    def get_boundaries(self, discr, actx, t):
        nodes = thaw(actx, discr.nodes())

        sym_exact_u = self.get_solution(
            pmbl.make_sym_vector("x", self.dim), pmbl.var("t"))

        exact_u = _sym_eval(sym_exact_u, x=nodes, t=t)
        exact_grad_u = _sym_eval(sym.grad(self.dim, sym_exact_u), x=nodes, t=t)

        boundaries = {}

        for i in range(self.dim-1):
            lower_btag = DTAG_BOUNDARY("-"+str(i))
            upper_btag = DTAG_BOUNDARY("+"+str(i))
            upper_grad_u = discr.project("vol", upper_btag, exact_grad_u)
            normal = thaw(actx, discr.normal(upper_btag))
            upper_grad_u_dot_n = np.dot(upper_grad_u, normal)
            boundaries[lower_btag] = NeumannDiffusionBoundary(0.)
            boundaries[upper_btag] = NeumannDiffusionBoundary(upper_grad_u_dot_n)
        lower_btag = DTAG_BOUNDARY("-"+str(self.dim-1))
        upper_btag = DTAG_BOUNDARY("+"+str(self.dim-1))
        upper_u = discr.project("vol", upper_btag, exact_u)
        boundaries[lower_btag] = DirichletDiffusionBoundary(0.)
        boundaries[upper_btag] = DirichletDiffusionBoundary(upper_u)

        return boundaries


# 1D: alpha(x) = 1+0.2*cos(3*x)
#     u(x,t)   = cos(x) (manufactured)
# 2D: alpha(x,y) = 1+0.2*cos(3*x)*cos(3*y)
#     u(x,y,t)   = sin(x)*cos(y) (manufactured)
# 3D: alpha(x,y,z) = 1+0.2*cos(3*x)*cos(3*y)*cos(3*z)
#     u(x,y,z,t)   = sin(x)*sin(y)*cos(z) (manufactured)
# on [-pi/2, pi/2]^{#dims}
class StaticTrigVarDiff(HeatProblem):
    def __init__(self, dim):
        super().__init__(dim)

    def get_mesh(self, n):
        return get_box_mesh(self.dim, -0.5*np.pi, 0.5*np.pi, n)

    def get_solution(self, x, t):
        u = 1
        for i in range(self.dim-1):
            u *= _sin(x[i])
        u *= _cos(x[self.dim-1])
        return u

    def get_alpha(self, x, t, u):
        alpha = 1
        for i in range(self.dim):
            alpha *= _cos(3.*x[i])
        alpha = 1 + 0.2*alpha
        return alpha

    def get_boundaries(self, discr, actx, t):
        boundaries = {}

        for i in range(self.dim-1):
            lower_btag = DTAG_BOUNDARY("-"+str(i))
            upper_btag = DTAG_BOUNDARY("+"+str(i))
            boundaries[lower_btag] = NeumannDiffusionBoundary(0.)
            boundaries[upper_btag] = NeumannDiffusionBoundary(0.)
        lower_btag = DTAG_BOUNDARY("-"+str(self.dim-1))
        upper_btag = DTAG_BOUNDARY("+"+str(self.dim-1))
        boundaries[lower_btag] = DirichletDiffusionBoundary(0.)
        boundaries[upper_btag] = DirichletDiffusionBoundary(0.)

        return boundaries


# alpha(u) = 1 + u**3
# 1D: u(x,t) = cos(x) (manufactured)
# 2D: u(x,y,t) = sin(x)*cos(y) (manufactured)
# 3D: u(x,y,z,t) = sin(x)*sin(y)*cos(z) (manufactured)
# on [-pi/2, pi/2]^{#dims}
class StaticTrigNonlinearDiff(HeatProblem):
    def __init__(self, dim):
        super().__init__(dim)

    def get_mesh(self, n):
        return get_box_mesh(self.dim, -0.5*np.pi, 0.5*np.pi, n)

    def get_solution(self, x, t):
        u = 1
        for i in range(self.dim-1):
            u *= _sin(x[i])
        u *= _cos(x[self.dim-1])
        return u

    def get_alpha(self, x, t, u):
        return 1 + u**3

    def get_boundaries(self, discr, actx, t):
        boundaries = {}

        for i in range(self.dim-1):
            lower_btag = DTAG_BOUNDARY("-"+str(i))
            upper_btag = DTAG_BOUNDARY("+"+str(i))
            boundaries[lower_btag] = NeumannDiffusionBoundary(0.)
            boundaries[upper_btag] = NeumannDiffusionBoundary(0.)
        lower_btag = DTAG_BOUNDARY("-"+str(self.dim-1))
        upper_btag = DTAG_BOUNDARY("+"+str(self.dim-1))
        boundaries[lower_btag] = DirichletDiffusionBoundary(0.)
        boundaries[upper_btag] = DirichletDiffusionBoundary(0.)

        return boundaries


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
        (DecayingTrigTruncatedDomain(1, 2.), 50, 5.e-5, [8, 16, 24]),
        (DecayingTrigTruncatedDomain(2, 2.), 50, 5.e-5, [8, 12, 16]),
        (DecayingTrigTruncatedDomain(3, 2.), 50, 5.e-5, [8, 10, 12]),
        (StaticTrigVarDiff(1), 50, 5.e-5, [8, 16, 24]),
        (StaticTrigVarDiff(2), 50, 5.e-5, [12, 14, 16]),
        (StaticTrigNonlinearDiff(1), 50, 5.e-5, [8, 16, 24]),
        (StaticTrigNonlinearDiff(2), 50, 5.e-5, [12, 14, 16]),
    ])
def test_diffusion_accuracy(actx_factory, problem, nsteps, dt, scales, order,
            visualize=False):
    """
    Checks the accuracy of the diffusion operator by solving the heat equation for a
    given problem setup.
    """
    actx = actx_factory()

    p = problem

    sym_x = pmbl.make_sym_vector("x", p.dim)
    sym_t = pmbl.var("t")
    sym_u = p.get_solution(sym_x, sym_t)
    sym_alpha = p.get_alpha(sym_x, sym_t, sym_u)

    sym_diffusion_u = sym_diffusion(p.dim, sym_alpha, sym_u)

    # In order to support manufactured solutions, we modify the heat equation
    # to add a source term f. If the solution is exact, this term should be 0.
    sym_f = sym.diff(sym_t)(sym_u) - sym_diffusion_u

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

        def get_rhs(t, u):
            alpha = p.get_alpha(nodes, t, u)
            if isinstance(alpha, DOFArray):
                discr_tag = DISCR_TAG_QUAD
            else:
                discr_tag = DISCR_TAG_BASE
            return (diffusion_operator(discr, quad_tag=discr_tag, alpha=alpha,
                    boundaries=p.get_boundaries(discr, actx, t), u=u)
                + _sym_eval(sym_f, x=nodes, t=t))

        t = 0.

        u = p.get_solution(nodes, t)

        from mirgecom.integrators import rk4_step

        for _ in range(nsteps):
            u = rk4_step(u, t, dt, get_rhs)
            t += dt

        expected_u = p.get_solution(nodes, t)

        rel_linf_err = actx.to_numpy(
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

    linf_err = actx.to_numpy(discr.norm(rhs, np.inf))
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

    linf_diff = actx.to_numpy(discr.norm(u - u_steady, np.inf))
    assert linf_diff < 0.1


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("problem",
    [
        DecayingTrig(1, 1.)
    ])
@pytest.mark.octave
def test_diffusion_compare_to_nodal_dg(actx_factory, problem, order,
            print_err=False):
    """Compares diffusion operator to Hesthaven/Warburton Nodal-DG code."""
    pytest.importorskip("oct2py")

    actx = actx_factory()

    p = problem

    assert p.dim == 1

    sym_x = pmbl.make_sym_vector("x", p.dim)
    sym_t = pmbl.var("t")
    sym_u = p.get_solution(sym_x, sym_t)
    sym_alpha = p.get_alpha(sym_x, sym_t, sym_u)

    assert sym_alpha == 1

    sym_diffusion_u = sym_diffusion(p.dim, sym_alpha, sym_u)

    from meshmode.interop.nodal_dg import download_nodal_dg_if_not_present
    download_nodal_dg_if_not_present()

    for n in [4, 8, 16, 32, 64]:
        mesh = p.get_mesh(n)

        from meshmode.interop.nodal_dg import NodalDGContext
        with NodalDGContext("./nodal-dg/Codes1.1") as ndgctx:
            ndgctx.set_mesh(mesh, order=order)

            t = 1.23456789

            from grudge.eager import EagerDGDiscretization
            discr_mirgecom = EagerDGDiscretization(actx, mesh, order=order)
            nodes_mirgecom = thaw(actx, discr_mirgecom.nodes())

            u_mirgecom = p.get_solution(nodes_mirgecom, t)

            diffusion_u_mirgecom = diffusion_operator(discr_mirgecom,
                quad_tag=DISCR_TAG_BASE, alpha=discr_mirgecom.zeros(actx)+1.,
                boundaries=p.get_boundaries(discr_mirgecom, actx, t), u=u_mirgecom)

            discr_ndg = ndgctx.get_discr(actx)
            nodes_ndg = thaw(actx, discr_ndg.nodes())

            u_ndg = p.get_solution(nodes_ndg, t)

            ndgctx.push_dof_array("u", u_ndg)
            ndgctx.octave.push("t", t)
            ndgctx.octave.eval("[rhs] = HeatCRHS1D(u,t)", verbose=False)
            diffusion_u_ndg = ndgctx.pull_dof_array(actx, "rhs")

            def inf_norm(f):
                return actx.np.linalg.norm(f, np.inf)

            err = (inf_norm(diffusion_u_mirgecom-diffusion_u_ndg)
                        / inf_norm(diffusion_u_ndg))

            if print_err:
                diffusion_u_exact = _sym_eval(sym_diffusion_u, x=nodes_mirgecom, t=t)
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

    p = DecayingTrig(1, 2.)

    sym_x = pmbl.make_sym_vector("x", p.dim)
    sym_t = pmbl.var("t")

    def get_u1(x, t):
        return p.get_solution(x, t)

    def get_u2(x, t):
        return 2*p.get_solution(x, t)

    sym_u1 = get_u1(sym_x, sym_t)
    sym_u2 = get_u2(sym_x, sym_t)

    sym_alpha1 = p.get_alpha(sym_x, sym_t, sym_u1)
    sym_alpha2 = p.get_alpha(sym_x, sym_t, sym_u2)

    assert isinstance(sym_alpha1, Number)
    assert isinstance(sym_alpha2, Number)

    alpha = sym_alpha1

    sym_diffusion_u1 = sym_diffusion(p.dim, alpha, sym_u1)
    sym_diffusion_u2 = sym_diffusion(p.dim, alpha, sym_u2)

    n = 128

    mesh = p.get_mesh(n)

    from grudge.eager import EagerDGDiscretization
    discr = EagerDGDiscretization(actx, mesh, order=4)

    nodes = thaw(actx, discr.nodes())

    t = 1.23456789

    u1 = get_u1(nodes, t)
    u2 = get_u2(nodes, t)

    alpha = p.get_alpha(nodes, t, u1)

    boundaries = p.get_boundaries(discr, actx, t)

    diffusion_u1 = diffusion_operator(discr, quad_tag=DISCR_TAG_BASE, alpha=alpha,
        boundaries=boundaries, u=u1)

    assert isinstance(diffusion_u1, DOFArray)

    expected_diffusion_u1 = _sym_eval(sym_diffusion_u1, x=nodes, t=t)
    rel_linf_err = actx.to_numpy(
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
        _sym_eval(sym_diffusion_u1, x=nodes, t=t),
        _sym_eval(sym_diffusion_u2, x=nodes, t=t)
    ])
    rel_linf_err = actx.to_numpy(
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
