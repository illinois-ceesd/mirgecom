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
import grudge.op as op
from mirgecom.symbolic import (
    diff as sym_diff,
    grad as sym_grad,
    div as sym_div,
    evaluate)
import mirgecom.math as mm
from mirgecom.diffusion import (
    diffusion_operator,
    DirichletDiffusionBoundary,
    NeumannDiffusionBoundary)
from meshmode.dof_array import DOFArray
from grudge.dof_desc import BoundaryDomainTag, DISCR_TAG_BASE, DISCR_TAG_QUAD
from mirgecom.discretization import create_discretization_collection
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)
import pytest
from mirgecom.simutil import get_box_mesh
import logging
logger = logging.getLogger(__name__)


class HeatProblem(metaclass=ABCMeta):
    """
    Description of a heat equation problem.

    .. autoproperty:: dim
    .. automethod:: get_mesh
    .. automethod:: get_kappa
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
    def get_kappa(self, x, t, u):
        """
        Return the conductivity for coordinates *x*, time *t*, and solution *u*.
        """
        pass

    @abstractmethod
    def get_boundaries(self, dcoll, actx, t):
        """
        Return a :class:`dict` that maps boundary tags to boundary conditions at
        time *t*.
        """
        pass


# 1D: u(x,t) = exp(-kappa*t)*cos(x)
# 2D: u(x,y,t) = exp(-2*kappa*t)*sin(x)*cos(y)
# 3D: u(x,y,z,t) = exp(-3*kappa*t)*sin(x)*sin(y)*cos(z)
# on [-pi/2, pi/2]^{#dims}
class DecayingTrig(HeatProblem):
    def __init__(self, dim, kappa):
        super().__init__(dim)
        self._kappa = kappa

    def get_mesh(self, n):
        return get_box_mesh(self.dim, -0.5*np.pi, 0.5*np.pi, n)

    def get_solution(self, x, t):
        u = mm.exp(-self.dim*self._kappa*t)
        for i in range(self.dim-1):
            u = u * mm.sin(x[i])
        u = u * mm.cos(x[self.dim-1])
        return u

    def get_kappa(self, x, t, u):
        return self._kappa

    def get_boundaries(self, dcoll, actx, t):
        boundaries = {}

        for i in range(self.dim-1):
            lower_bdtag = BoundaryDomainTag("-"+str(i+1))
            upper_bdtag = BoundaryDomainTag("+"+str(i+1))
            boundaries[lower_bdtag] = NeumannDiffusionBoundary(0.)
            boundaries[upper_bdtag] = NeumannDiffusionBoundary(0.)
        lower_bdtag = BoundaryDomainTag("-"+str(self.dim))
        upper_bdtag = BoundaryDomainTag("+"+str(self.dim))
        boundaries[lower_bdtag] = DirichletDiffusionBoundary(0.)
        boundaries[upper_bdtag] = DirichletDiffusionBoundary(0.)

        return boundaries


# 1D: u(x,t) = exp(-kappa*t)*cos(x)
# 2D: u(x,y,t) = exp(-2*kappa*t)*sin(x)*cos(y)
# 3D: u(x,y,z,t) = exp(-3*kappa*t)*sin(x)*sin(y)*cos(z)
# on [-pi/2, pi/4]^{#dims}
class DecayingTrigTruncatedDomain(HeatProblem):
    def __init__(self, dim, kappa):
        super().__init__(dim)
        self._kappa = kappa

    def get_mesh(self, n):
        return get_box_mesh(self.dim, -0.5*np.pi, 0.25*np.pi, n)

    def get_solution(self, x, t):
        u = mm.exp(-self.dim*self._kappa*t)
        for i in range(self.dim-1):
            u = u * mm.sin(x[i])
        u = u * mm.cos(x[self.dim-1])
        return u

    def get_kappa(self, x, t, u):
        return self._kappa

    def get_boundaries(self, dcoll, actx, t):
        nodes = actx.thaw(dcoll.nodes())

        sym_exact_u = self.get_solution(
            pmbl.make_sym_vector("x", self.dim), pmbl.var("t"))

        exact_u = evaluate(sym_exact_u, x=nodes, t=t)
        exact_grad_u = evaluate(sym_grad(self.dim, sym_exact_u), x=nodes, t=t)

        boundaries = {}

        for i in range(self.dim-1):
            lower_bdtag = BoundaryDomainTag("-"+str(i+1))
            upper_bdtag = BoundaryDomainTag("+"+str(i+1))
            upper_grad_u = op.project(dcoll, "vol", upper_bdtag, exact_grad_u)
            normal = actx.thaw(dcoll.normal(upper_bdtag))
            upper_grad_u_dot_n = np.dot(upper_grad_u, normal)
            boundaries[lower_bdtag] = NeumannDiffusionBoundary(0.)
            boundaries[upper_bdtag] = NeumannDiffusionBoundary(upper_grad_u_dot_n)
        lower_bdtag = BoundaryDomainTag("-"+str(self.dim))
        upper_bdtag = BoundaryDomainTag("+"+str(self.dim))
        upper_u = op.project(dcoll, "vol", upper_bdtag, exact_u)
        boundaries[lower_bdtag] = DirichletDiffusionBoundary(0.)
        boundaries[upper_bdtag] = DirichletDiffusionBoundary(upper_u)

        return boundaries


# 1D: kappa(x) = 1+0.2*cos(3*x)
#     u(x,t)   = cos(t)*cos(x) (manufactured)
# 2D: kappa(x,y) = 1+0.2*cos(3*x)*cos(3*y)
#     u(x,y,t)   = cos(t)*sin(x)*cos(y) (manufactured)
# 3D: kappa(x,y,z) = 1+0.2*cos(3*x)*cos(3*y)*cos(3*z)
#     u(x,y,z,t)   = cos(t)*sin(x)*sin(y)*cos(z) (manufactured)
# on [-pi/2, pi/2]^{#dims}
class OscillatingTrigVarDiff(HeatProblem):
    def __init__(self, dim):
        super().__init__(dim)

    def get_mesh(self, n):
        return get_box_mesh(self.dim, -0.5*np.pi, 0.5*np.pi, n)

    def get_solution(self, x, t):
        u = mm.cos(t)
        for i in range(self.dim-1):
            u = u * mm.sin(x[i])
        u = u * mm.cos(x[self.dim-1])
        return u

    def get_kappa(self, x, t, u):
        kappa = 1
        for i in range(self.dim):
            kappa = kappa * mm.cos(3.*x[i])
        kappa = 1 + 0.2*kappa
        return kappa

    def get_boundaries(self, dcoll, actx, t):
        boundaries = {}

        for i in range(self.dim-1):
            lower_bdtag = BoundaryDomainTag("-"+str(i+1))
            upper_bdtag = BoundaryDomainTag("+"+str(i+1))
            boundaries[lower_bdtag] = NeumannDiffusionBoundary(0.)
            boundaries[upper_bdtag] = NeumannDiffusionBoundary(0.)
        lower_bdtag = BoundaryDomainTag("-"+str(self.dim))
        upper_bdtag = BoundaryDomainTag("+"+str(self.dim))
        boundaries[lower_bdtag] = DirichletDiffusionBoundary(0.)
        boundaries[upper_bdtag] = DirichletDiffusionBoundary(0.)

        return boundaries


# kappa(u) = 1 + u**3
# 1D: u(x,t) = cos(t)*cos(x) (manufactured)
# 2D: u(x,y,t) = cos(t)*sin(x)*cos(y) (manufactured)
# 3D: u(x,y,z,t) = cos(t)*sin(x)*sin(y)*cos(z) (manufactured)
# on [-pi/2, pi/2]^{#dims}
class OscillatingTrigNonlinearDiff(HeatProblem):
    def __init__(self, dim):
        super().__init__(dim)

    def get_mesh(self, n):
        return get_box_mesh(self.dim, -0.5*np.pi, 0.5*np.pi, n)

    def get_solution(self, x, t):
        u = mm.cos(t)
        for i in range(self.dim-1):
            u = u * mm.sin(x[i])
        u = u * mm.cos(x[self.dim-1])
        return u

    def get_kappa(self, x, t, u):
        return 1 + u**3

    def get_boundaries(self, dcoll, actx, t):
        boundaries = {}

        for i in range(self.dim-1):
            lower_bdtag = BoundaryDomainTag("-"+str(i+1))
            upper_bdtag = BoundaryDomainTag("+"+str(i+1))
            boundaries[lower_bdtag] = NeumannDiffusionBoundary(0.)
            boundaries[upper_bdtag] = NeumannDiffusionBoundary(0.)
        lower_bdtag = BoundaryDomainTag("-"+str(self.dim))
        upper_bdtag = BoundaryDomainTag("+"+str(self.dim))
        boundaries[lower_bdtag] = DirichletDiffusionBoundary(0.)
        boundaries[upper_bdtag] = DirichletDiffusionBoundary(0.)

        return boundaries


def sym_diffusion(dim, sym_kappa, sym_u):
    """Return a symbolic expression for the diffusion operator applied to a function.
    """
    return sym_div(dim, sym_kappa * sym_grad(dim, sym_u))


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
        (OscillatingTrigVarDiff(1), 50, 5.e-5, [8, 16, 24]),
        (OscillatingTrigVarDiff(2), 50, 5.e-5, [12, 14, 16]),
        (OscillatingTrigNonlinearDiff(1), 50, 5.e-5, [8, 16, 24]),
        (OscillatingTrigNonlinearDiff(2), 50, 5.e-5, [12, 14, 16]),
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
    sym_kappa = p.get_kappa(sym_x, sym_t, sym_u)

    sym_diffusion_u = sym_diffusion(p.dim, sym_kappa, sym_u)

    # In order to support manufactured solutions, we modify the heat equation
    # to add a source term f. If the solution is exact, this term should be 0.
    sym_f = sym_diff(sym_t)(sym_u) - sym_diffusion_u

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for n in scales:
        mesh = p.get_mesh(n)

        dcoll = create_discretization_collection(actx, mesh, order)

        nodes = actx.thaw(dcoll.nodes())

        def get_rhs(t, u):
            kappa = p.get_kappa(nodes, t, u)
            if isinstance(kappa, DOFArray):
                quadrature_tag = DISCR_TAG_QUAD
            else:
                quadrature_tag = DISCR_TAG_BASE
            return (
                diffusion_operator(
                    dcoll, kappa=kappa, boundaries=p.get_boundaries(dcoll, actx, t),
                    u=u, quadrature_tag=quadrature_tag)
                + evaluate(sym_f, x=nodes, t=t))

        t = 0.

        u = p.get_solution(nodes, t)

        from mirgecom.integrators import rk4_step

        for _ in range(nsteps):
            u = rk4_step(u, t, dt, get_rhs)
            t += dt

        expected_u = p.get_solution(nodes, t)

        rel_linf_err = actx.to_numpy(
            op.norm(dcoll, u - expected_u, np.inf)
            / op.norm(dcoll, expected_u, np.inf))
        eoc_rec.add_data_point(1./n, rel_linf_err)

        if visualize:
            from grudge.shortcuts import make_visualizer
            vis = make_visualizer(dcoll, order+3)
            vis.write_vtk_file("diffusion_accuracy_{order}_{n}.vtu".format(
                order=order, n=n), [
                    ("u", u),
                    ("expected_u", expected_u),
                    ])

    print("L^inf error:")
    print(eoc_rec)
    # Expected convergence rates from Hesthaven/Warburton book
    expected_order = order+1 if order % 2 == 0 else order
    assert (eoc_rec.order_estimate() >= expected_order - 0.5
            or eoc_rec.max_error() < 1e-11)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_diffusion_discontinuous_kappa(actx_factory, order, visualize=False):
    """
    Checks the accuracy of the diffusion operator for an kappa field that has a
    jump across an element face.
    """
    actx = actx_factory()

    n = 8

    mesh = get_box_mesh(1, -1, 1, n)

    dcoll = create_discretization_collection(actx, mesh, order)

    nodes = actx.thaw(dcoll.nodes())

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

    kappa_lower = 0.5
    kappa_upper = 1

    kappa = kappa_lower * lower_mask + kappa_upper * upper_mask

    boundaries = {
        BoundaryDomainTag("-1"): DirichletDiffusionBoundary(0.),
        BoundaryDomainTag("+1"): DirichletDiffusionBoundary(1.),
    }

    flux = -kappa_lower*kappa_upper/(kappa_lower + kappa_upper)

    u_steady = (
              -flux/kappa_lower * (nodes[0] + 1)  * lower_mask  # noqa: E126, E221
        + (1 - flux/kappa_upper * (nodes[0] - 1)) * upper_mask)  # noqa: E131

    def get_rhs(t, u, return_grad_u=False):
        return diffusion_operator(
            dcoll, kappa=kappa, boundaries=boundaries, u=u,
            return_grad_u=return_grad_u)

    rhs, grad_u_steady = get_rhs(0, u_steady, return_grad_u=True)

    grad_u_steady_exact = -flux/kappa

    if visualize:
        from grudge.shortcuts import make_visualizer
        vis = make_visualizer(dcoll, order+3)
        vis.write_vtk_file("diffusion_discontinuous_kappa_rhs_{order}.vtu"
            .format(order=order), [
                ("kappa", kappa),
                ("u_steady", u_steady),
                ("grad_u_steady", grad_u_steady),
                ("grad_u_steady_exact", grad_u_steady_exact),
                ("rhs", rhs),
                ])

    linf_err = actx.to_numpy(
        op.norm(dcoll, grad_u_steady - grad_u_steady_exact, np.inf))
    assert linf_err < 1e-11

    linf_err = actx.to_numpy(op.norm(dcoll, rhs, np.inf))
    assert linf_err < 1e-11

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
        vis.write_vtk_file("diffusion_disc_kappa_stability_{order}.vtu"
            .format(order=order), [
                ("kappa", kappa),
                ("u", u),
                ("u_steady", u_steady),
                ])

    linf_diff = actx.to_numpy(op.norm(dcoll, u - u_steady, np.inf))
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
    sym_kappa = p.get_kappa(sym_x, sym_t, sym_u)

    assert sym_kappa == 1

    sym_diffusion_u = sym_diffusion(p.dim, sym_kappa, sym_u)

    from meshmode.interop.nodal_dg import download_nodal_dg_if_not_present
    download_nodal_dg_if_not_present()

    for n in [4, 8, 16, 32, 64]:
        mesh = p.get_mesh(n)

        from meshmode.interop.nodal_dg import NodalDGContext
        with NodalDGContext("./nodal-dg/Codes1.1") as ndgctx:
            ndgctx.set_mesh(mesh, order=order)

            t = 1.23456789

            dcoll_mirgecom = create_discretization_collection(actx, mesh,
                                                              order=order)
            nodes_mirgecom = actx.thaw(dcoll_mirgecom.nodes())

            u_mirgecom = p.get_solution(nodes_mirgecom, t)

            diffusion_u_mirgecom = diffusion_operator(
                dcoll_mirgecom, kappa=dcoll_mirgecom.zeros(actx)+1.,
                boundaries=p.get_boundaries(dcoll_mirgecom, actx, t),
                u=u_mirgecom)

            discr_ndg = ndgctx.get_discr(actx)
            nodes_ndg = actx.thaw(discr_ndg.nodes())

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
                diffusion_u_exact = evaluate(sym_diffusion_u, x=nodes_mirgecom, t=t)
                err_exact = (inf_norm(diffusion_u_mirgecom-diffusion_u_exact)
                            / inf_norm(diffusion_u_exact))
                print(err, err_exact)

            assert err < 1e-9


@pytest.mark.parametrize("vector_kappa", [False, True])
def test_diffusion_obj_array_vectorize(actx_factory, vector_kappa):
    """
    Checks that the diffusion operator can be called with either scalars or object
    arrays for `u`.
    """
    actx = actx_factory()

    if vector_kappa:
        kappa1 = 2
        kappa2 = 3
    else:
        kappa1 = 2
        kappa2 = kappa1

    p1 = DecayingTrig(1, kappa1)
    p2 = DecayingTrig(1, kappa2)

    sym_x = pmbl.make_sym_vector("x", 1)
    sym_t = pmbl.var("t")

    def get_u1(x, t):
        return p1.get_solution(x, t)

    def get_u2(x, t):
        return 2*p2.get_solution(x, t)

    sym_u1 = get_u1(sym_x, sym_t)
    sym_u2 = get_u2(sym_x, sym_t)

    sym_diffusion_u1 = sym_diffusion(1, kappa1, sym_u1)
    sym_diffusion_u2 = sym_diffusion(1, kappa2, sym_u2)

    n = 128

    mesh = p1.get_mesh(n)

    dcoll = create_discretization_collection(actx, mesh, order=4)

    nodes = actx.thaw(dcoll.nodes())

    t = 1.23456789

    u1 = get_u1(nodes, t)
    u2 = get_u2(nodes, t)

    boundaries = p1.get_boundaries(dcoll, actx, t)

    diffusion_u1 = diffusion_operator(
        dcoll, kappa=kappa1, boundaries=boundaries, u=u1)

    assert isinstance(diffusion_u1, DOFArray)

    expected_diffusion_u1 = evaluate(sym_diffusion_u1, x=nodes, t=t)
    rel_linf_err = actx.to_numpy(
        op.norm(dcoll, diffusion_u1 - expected_diffusion_u1, np.inf)
        / op.norm(dcoll, expected_diffusion_u1, np.inf))
    assert rel_linf_err < 1.e-5

    boundaries_vector = [boundaries, boundaries]
    u_vector = make_obj_array([u1, u2])

    if vector_kappa:
        kappa_vector = make_obj_array([kappa1, kappa2])
        diffusion_u_vector = diffusion_operator(
            dcoll, kappa=kappa_vector, boundaries=boundaries_vector, u=u_vector)
    else:
        diffusion_u_vector = diffusion_operator(
            dcoll, kappa=kappa1, boundaries=boundaries_vector, u=u_vector)

    assert isinstance(diffusion_u_vector, np.ndarray)
    assert diffusion_u_vector.shape == (2,)

    expected_diffusion_u_vector = make_obj_array([
        evaluate(sym_diffusion_u1, x=nodes, t=t),
        evaluate(sym_diffusion_u2, x=nodes, t=t)
    ])
    rel_linf_err = actx.to_numpy(
        op.norm(dcoll, diffusion_u_vector - expected_diffusion_u_vector, np.inf)
        / op.norm(dcoll, expected_diffusion_u_vector, np.inf))
    assert rel_linf_err < 1.e-5


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
