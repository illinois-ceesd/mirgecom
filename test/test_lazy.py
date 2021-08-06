__copyright__ = """Copyright (C) 2021 University of Illinois Board of Trustees"""

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
from functools import partial
from pytools.obj_array import make_obj_array, obj_array_vectorize
import pyopencl as cl
import pyopencl.tools as cl_tools
import pyopencl.array as cla  # noqa
import pyopencl.clmath as clmath  # noqa
from meshmode.array_context import (  # noqa
    PyOpenCLArrayContext,
    PytatoPyOpenCLArrayContext
)
from arraycontext.container.traversal import freeze, thaw

from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

import pytest  # noqa

import logging
logger = logging.getLogger(__name__)


def op_test_data(ctx_factory):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    eager_actx = PyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    lazy_actx = PytatoPyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    def get_discr(order):
        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(
            a=(-0.5,)*2,
            b=(0.5,)*2,
            nelements_per_axis=(4,)*2,
            boundary_tag_to_face={
                "-x": ["-x"],
                "+x": ["+x"],
                "-y": ["-y"],
                "+y": ["+y"],
            })

        from grudge.eager import EagerDGDiscretization
        return EagerDGDiscretization(eager_actx, mesh, order=order)

    return eager_actx, lazy_actx, get_discr


# Mimics math.isclose for state arrays
def _isclose(discr, x, y, rel_tol=1e-9, abs_tol=0, return_operands=False):
    def componentwise_norm(a):
        from mirgecom.fluid import ConservedVars
        if isinstance(a, ConservedVars):
            return componentwise_norm(a.join())
        return obj_array_vectorize(lambda b: discr.norm(b, np.inf), a)

    lhs = componentwise_norm(x - y)
    rhs = np.maximum(
        rel_tol * np.maximum(
            componentwise_norm(x),
            componentwise_norm(y)),
        abs_tol)

    is_close = np.all(lhs <= rhs)

    if return_operands:
        return is_close, lhs, rhs
    else:
        return is_close


# FIXME: Re-enable and fix up if/when standalone gradient operator exists
# def test_lazy_op_gradient(ctx_factory):
#     cl_ctx = ctx_factory()
#     actx, discr = _op_test_fixture(cl_ctx)
#
#     from grudge.dof_desc import DTAG_BOUNDARY, DISCR_TAG_BASE
#     from mirgecom.diffusion import (
#         _gradient_operator,
#         DirichletDiffusionBoundary,
#         NeumannDiffusionBoundary)
#
#     boundaries = {
#         DTAG_BOUNDARY("x"): DirichletDiffusionBoundary(0),
#         DTAG_BOUNDARY("y"): NeumannDiffusionBoundary(0)
#     }
#
#     def op(alpha, u):
#         return _gradient_operator(
#             discr, DISCR_TAG_BASE, alpha, boundaries, u)

#     compiled_op = actx.compile(op)
#     alpha = discr.zeros(actx) + 1
#     u = discr.zeros(actx)
#     compiled_op(alpha, u)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_lazy_op_divergence(ctx_factory, order):
    eager_actx, lazy_actx, get_discr = op_test_data(ctx_factory)
    discr = get_discr(order)

    from grudge.trace_pair import interior_trace_pair
    from mirgecom.operators import div_operator

    def get_flux(u_tpair):
        dd = u_tpair.dd
        dd_allfaces = dd.with_dtag("all_faces")
        normal = thaw(discr.normal(dd), u_tpair.int[0].array_context)
        flux = u_tpair.avg @ normal
        return discr.project(dd, dd_allfaces, flux)

    def op(u):
        return div_operator(discr, u, get_flux(interior_trace_pair(discr, u)))

    lazy_op = lazy_actx.compile(op)

    def get_inputs(actx):
        nodes = thaw(discr.nodes(), actx)
        u = make_obj_array([actx.np.sin(np.pi*nodes[i]) for i in range(2)])
        return u,

    tol = 1e-12
    isclose = partial(
        _isclose, discr, rel_tol=tol, abs_tol=tol, return_operands=True)

    def lazy_to_eager(u):
        return thaw(freeze(u, lazy_actx), eager_actx)

    eager_result = op(*get_inputs(eager_actx))
    lazy_result = lazy_to_eager(lazy_op(*get_inputs(lazy_actx)))
    is_close, lhs, rhs = isclose(lazy_result, eager_result)
    assert is_close, f"{lhs} not <= {rhs}"


@pytest.mark.parametrize("order", [1, 2, 3])
def test_lazy_op_diffusion(ctx_factory, order):
    eager_actx, lazy_actx, get_discr = op_test_data(ctx_factory)
    discr = get_discr(order)

    from grudge.dof_desc import DTAG_BOUNDARY, DISCR_TAG_BASE
    from mirgecom.diffusion import (
        diffusion_operator,
        DirichletDiffusionBoundary,
        NeumannDiffusionBoundary)

    boundaries = {
        DTAG_BOUNDARY("-x"): DirichletDiffusionBoundary(0),
        DTAG_BOUNDARY("+x"): DirichletDiffusionBoundary(0),
        DTAG_BOUNDARY("-y"): NeumannDiffusionBoundary(0),
        DTAG_BOUNDARY("+y"): NeumannDiffusionBoundary(0),
    }

    def op(alpha, u):
        return diffusion_operator(
            discr, DISCR_TAG_BASE, alpha, boundaries, u)

    lazy_op = lazy_actx.compile(op)

    def get_inputs(actx):
        nodes = thaw(discr.nodes(), actx)
        alpha = discr.zeros(actx) + 1
        u = actx.np.cos(np.pi*nodes[0])
        return alpha, u

    tol = 1e-12
    isclose = partial(
        _isclose, discr, rel_tol=tol, abs_tol=tol, return_operands=True)

    def lazy_to_eager(u):
        return thaw(freeze(u, lazy_actx), eager_actx)

    eager_result = op(*get_inputs(eager_actx))
    lazy_result = lazy_to_eager(lazy_op(*get_inputs(lazy_actx)))
    is_close, lhs, rhs = isclose(lazy_result, eager_result)
    assert is_close, f"{lhs} not <= {rhs}"


def _get_pulse():
    from mirgecom.eos import IdealSingleGas
    eos = IdealSingleGas()

    from mirgecom.initializers import Uniform, AcousticPulse
    uniform_init = Uniform(dim=2)
    pulse_init = AcousticPulse(dim=2, center=np.zeros(2), amplitude=1.0, width=.1)

    def init(nodes):
        return pulse_init(x_vec=nodes, cv=uniform_init(nodes), eos=eos)

    from meshmode.mesh import BTAG_ALL
    from mirgecom.boundary import AdiabaticSlipBoundary
    boundaries = {
        BTAG_ALL: AdiabaticSlipBoundary()
    }

    return eos, init, boundaries, 1e-12


def _get_scalar_lump():
    from mirgecom.eos import IdealSingleGas
    eos = IdealSingleGas()

    from mirgecom.initializers import MulticomponentLump
    init = MulticomponentLump(
        dim=2, nspecies=3, velocity=np.ones(2), spec_y0s=np.ones(3),
        spec_amplitudes=np.ones(3))

    from meshmode.mesh import BTAG_ALL
    from mirgecom.boundary import PrescribedInviscidBoundary
    boundaries = {
        BTAG_ALL: PrescribedInviscidBoundary(fluid_solution_func=init)
    }

    return eos, init, boundaries, 1e-12


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("problem", [
    _get_pulse(),
    _get_scalar_lump(),
])
def test_lazy_op_euler(ctx_factory, problem, order):
    eager_actx, lazy_actx, get_discr = op_test_data(ctx_factory)
    discr = get_discr(order)

    eos, init, boundaries, tol = problem

    from mirgecom.euler import euler_operator

    def op(state):
        return euler_operator(discr, eos, boundaries, state)

    lazy_op = lazy_actx.compile(op)

    def get_inputs(actx):
        nodes = thaw(discr.nodes(), actx)
        state = init(nodes)
        return state,

    isclose = partial(
        _isclose, discr, rel_tol=tol, abs_tol=tol, return_operands=True)

    def lazy_to_eager(u):
        return thaw(freeze(u, lazy_actx), eager_actx)

    eager_result = op(*get_inputs(eager_actx))
    lazy_result = lazy_to_eager(lazy_op(*get_inputs(lazy_actx)))
    is_close, lhs, rhs = isclose(lazy_result, eager_result)
    assert is_close, f"{lhs} not <= {rhs}"


def _get_poiseuille():
    p_low = 1
    p_hi = 1.001
    mu = 1

    from mirgecom.eos import IdealSingleGas
    from mirgecom.transport import SimpleTransport
    eos = IdealSingleGas(transport_model=SimpleTransport(viscosity=mu))

    def poiseuille_init(nodes, eos, cv=None, **kwargs):
        x = nodes[0] + 0.5
        y = nodes[1] + 0.5
        u_x = (p_hi - p_low)*y*(1 - y)/(2*mu)
        p_x = p_hi - (p_hi - p_low)*x
        rho = 1.0
        mass = 0*x + rho
        u_y = 0*x
        velocity = make_obj_array([u_x, u_y])
        ke = .5*np.dot(velocity, velocity)*mass
        gamma = eos.gamma()
        if cv is not None:
            mass = cv.mass
            vel = cv.velocity
            ke = .5*np.dot(vel, vel)*mass

        rho_e = p_x/(gamma-1) + ke
        from mirgecom.fluid import make_conserved
        return make_conserved(
            2, mass=mass, energy=rho_e, momentum=mass*velocity)

    init = partial(poiseuille_init, eos=eos)

    from grudge.dof_desc import DTAG_BOUNDARY
    from mirgecom.boundary import PrescribedViscousBoundary, IsothermalNoSlipBoundary
    boundaries = {
        DTAG_BOUNDARY("-x"): PrescribedViscousBoundary(q_func=poiseuille_init),
        DTAG_BOUNDARY("+x"): PrescribedViscousBoundary(q_func=poiseuille_init),
        DTAG_BOUNDARY("-y"): IsothermalNoSlipBoundary(),
        DTAG_BOUNDARY("+y"): IsothermalNoSlipBoundary(),
    }

    return eos, init, boundaries, 1e-12


def _get_viscous_scalar_lump():
    from mirgecom.eos import IdealSingleGas
    from mirgecom.transport import SimpleTransport
    transport = SimpleTransport(
        viscosity=1,
        species_diffusivity=np.ones(3))
    eos = IdealSingleGas(transport_model=transport)

    from mirgecom.initializers import MulticomponentLump
    init = MulticomponentLump(
        dim=2, nspecies=3, velocity=np.ones(2), spec_y0s=np.ones(3),
        spec_amplitudes=np.ones(3))

    from meshmode.mesh import BTAG_ALL
    from mirgecom.boundary import PrescribedViscousBoundary
    boundaries = {
        BTAG_ALL: PrescribedViscousBoundary(q_func=init)
    }

    return eos, init, boundaries, 1e-12


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("problem", [
    _get_poiseuille(),
    _get_viscous_scalar_lump(),
])
def test_lazy_op_ns(ctx_factory, problem, order):
    eager_actx, lazy_actx, get_discr = op_test_data(ctx_factory)
    discr = get_discr(order)

    eos, init, boundaries, tol = problem

    from mirgecom.navierstokes import ns_operator

    def op(state):
        return ns_operator(discr, eos, boundaries, state)

    lazy_op = lazy_actx.compile(op)

    def get_inputs(actx):
        nodes = thaw(discr.nodes(), actx)
        state = init(nodes)
        return state,

    isclose = partial(
        _isclose, discr, rel_tol=tol, abs_tol=tol, return_operands=True)

    def lazy_to_eager(u):
        return thaw(freeze(u, lazy_actx), eager_actx)

    eager_result = op(*get_inputs(eager_actx))
    lazy_result = lazy_to_eager(lazy_op(*get_inputs(lazy_actx)))
    is_close, lhs, rhs = isclose(lazy_result, eager_result)
    assert is_close, f"{lhs} not <= {rhs}"


@pytest.mark.parametrize("order", [1, 2, 3])
def test_lazy_op_projection(ctx_factory, order):
    eager_actx, lazy_actx, get_discr = op_test_data(ctx_factory)
    discr = get_discr(order)

    from grudge.dof_desc import DTAG_BOUNDARY, as_dofdesc

    lower_y_btag = DTAG_BOUNDARY("-y")
    upper_y_btag = DTAG_BOUNDARY("+y")
    btags = [lower_y_btag, upper_y_btag]

    def op(u):
        return make_obj_array([
            discr.project("vol", as_dofdesc(btag), u)
            for btag in btags])

    lazy_op = lazy_actx.compile(op)

    def get_inputs(actx):
        nodes = thaw(discr.nodes(), actx)
        u = actx.np.cos(np.pi*nodes[0])
        return u,

    tol = 1e-12
    isclose = partial(
        _isclose, discr, rel_tol=tol, abs_tol=tol, return_operands=True)

    def lazy_to_eager(u):
        return thaw(freeze(u, lazy_actx), eager_actx)

    from pytools.obj_array import obj_array_vectorized

    @obj_array_vectorized
    def dof_array_to_numpy(u):
        actx = u.array_context
        return make_obj_array([
            actx.to_numpy(grp_array)
            for grp_array in u])

    eager_result = dof_array_to_numpy(op(*get_inputs(eager_actx)))
    lazy_result = dof_array_to_numpy(lazy_op(*get_inputs(lazy_actx)))
    lazy_result_2 = dof_array_to_numpy(lazy_op(*get_inputs(lazy_actx)))

    print(f"{lazy_result=}")
    print(f"{lazy_result_2=}")
    print("")

    is_close, lhs, rhs = isclose(lazy_result, eager_result)
    assert is_close, f"{lhs} not <= {rhs}"


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
