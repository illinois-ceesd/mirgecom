"""Test the Euler gas dynamics module."""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

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
import numpy.random
import logging
import pytest

from functools import partial

from pytools.obj_array import make_obj_array

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
import grudge.op as op
from mirgecom.discretization import create_discretization_collection

from meshmode.array_context import PytestPyOpenCLArrayContextFactory
from arraycontext import pytest_generate_tests_for_array_contexts

from mirgecom.integrators import rk4_step
from meshmode.mesh import TensorProductElementGroup
from meshmode.mesh.generation import generate_regular_rect_mesh
from grudge.dof_desc import (
    DISCR_TAG_QUAD,
    DISCR_TAG_BASE,
)
from grudge.dof_desc import DD_VOLUME_ALL
from meshmode.discretization.connection import FACE_RESTR_ALL
from mirgecom.steppers import advance_state
import grudge.geometry as geo
from mirgecom.operators import div_operator

logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts(
    [PytestPyOpenCLArrayContextFactory])


class _AgitatorCommTag:
    pass


def _advection_flux_interior(dcoll, state_tpair, velocity):
    r"""Compute the numerical flux for divergence of advective flux."""
    actx = state_tpair.int.array_context
    dd = state_tpair.dd
    normal = geo.normal(actx, dcoll, dd)
    v_n = np.dot(velocity, normal)
    # v2 = np.dot(velocity, velocity)
    # vmag = np.sqrt(v2)

    # Lax-Friedrichs type
    # return state_tpair.avg * v_dot_n \
    # + 0.5 * vmag * (state_tpair.int - state_tpair.ext)
    # Simple upwind flux
    # state_upwind = actx.np.where(v_n > 0, state_tpair.int, state_tpair.ext)
    # return state_upwind * v_n
    # Central flux
    return state_tpair.avg * v_n


def run_agitator(
        actx, use_overintegration=False,
        use_leap=False, casename=None, rst_filename=None,
        init_type=None, order=None, quad_order=None,
        tpe=None, p_adv=1.0):
    """Drive the example."""
    if order is None:
        order = 1
    if quad_order is None:
        quad_order = order if tpe else order + 3
    if tpe is None:
        tpe = False
    if use_overintegration is None:
        use_overintegration = False

    # timestepping control
    current_step = 0
    timestepper = rk4_step

    # nsteps = 5000
    current_dt = 5e-3
    # t_final = nsteps * current_dt
    current_t = 0
    nsteps_period = 1000
    period = nsteps_period*current_dt
    t_final = 4*period

    # Mengaldo test case setup stuff
    alpha = 41.
    xc = -.3
    yc = 0

    dim = 2
    nel_1d = 8

    def g(t, actx=None):
        if actx is None:
            return np.cos(np.pi*t/period)
        return actx.np.cos(np.pi*t/period)

    # Note that *r* here is used only to get the array_context
    # The actual velocity is returned at points on the discretization
    # that comes from the DD.
    def get_velocity(r, t=0, dd=None):
        if dd is None:
            dd = DD_VOLUME_ALL
        actx = r[0].array_context
        discr_local = dcoll.discr_from_dd(dd)
        r_local = actx.thaw(discr_local.nodes())
        x = r_local[0]
        y = r_local[1]
        vx = y**p_adv
        vy = -1*x**p_adv
        return np.pi * g(t, actx) * make_obj_array([vx, vy])

    def poly_vel_initializer(xyz_vec, t=0):
        x = xyz_vec[0]
        y = xyz_vec[1]
        actx = x.array_context
        return actx.np.exp(-alpha*((x-xc)**2 + (y-yc)**2))

    nel_axes = (nel_1d,)*dim
    box_ll = (-1,)*dim
    box_ur = (1,)*dim
    print(f"{box_ll=}, {box_ur=}, {nel_axes=}")

    grp_cls = TensorProductElementGroup if tpe else None
    mesh_type = None if tpe else "X"
    generate_mesh = partial(
        generate_regular_rect_mesh,
        a=box_ll, b=box_ur, nelements_per_axis=nel_axes,
        periodic=(True,)*dim, mesh_type=mesh_type,
        group_cls=grp_cls
    )
    local_mesh = generate_mesh()
    dcoll = create_discretization_collection(actx, local_mesh, order=order,
                                             quadrature_order=quad_order)

    nodes_base = actx.thaw(dcoll.nodes())
    quadrature_tag = DISCR_TAG_QUAD if use_overintegration else DISCR_TAG_BASE

    # transfer trace pairs to quad grid, update pair dd
    interp_to_surf_quad = partial(op.tracepair_with_discr_tag,
                                  dcoll, quadrature_tag)

    exact_state_func = poly_vel_initializer
    init_state_base = exact_state_func(nodes_base)

    # Set the current state from time 0
    current_state = init_state_base

    dd_vol = DD_VOLUME_ALL
    dd_allfaces = dd_vol.trace(FACE_RESTR_ALL)
    dd_vol_quad = dd_vol.with_discr_tag(quadrature_tag)
    dd_allfaces_quad = dd_vol_quad.trace(FACE_RESTR_ALL)

    def my_pre_step(step, t, dt, state):

        time_left = max(0, t_final - t)
        dt = min(time_left, dt)
        return state, dt

    def my_post_step(step, t, dt, state):
        return state, dt

    def my_advection_rhs(t, state, velocity,
                         state_interior_trace_pairs=None,
                         flux_only=False):

        if state_interior_trace_pairs is None:
            state_interior_trace_pairs = \
                op.interior_trace_pairs(dcoll, state, comm_tag=_AgitatorCommTag)
        v_quad = op.project(dcoll, dd_vol, dd_vol_quad, velocity)

        # This "flux" function returns the *numerical flux* that will
        # be used in the divergence operation given a trace pair,
        # i.e. the soln on the -/+ sides of the face.
        def flux(state_tpair):
            # why project to all_faces? to "size" the array correctly
            # for all faces rather than just the selected "tpair.dd"
            v_dd = op.project(dcoll, dd_vol, state_tpair.dd, velocity)
            return op.project(
                dcoll, state_tpair.dd, dd_allfaces_quad,
                _advection_flux_interior(
                    dcoll, state_tpair, v_dd))

        vol_flux = state * v_quad

        # sums up the fluxes for each element boundary
        surf_flux = sum(flux(tpair)
                        for tpair in state_interior_trace_pairs)

        if flux_only:
            return vol_flux, surf_flux

        return -div_operator(dcoll, dd_vol, dd_allfaces, vol_flux, surf_flux)

    def my_rhs(t, state):
        state_quad = op.project(dcoll, dd_vol, dd_vol_quad, state)
        velocity = get_velocity(nodes_base, t, dd_vol)

        state_interior_trace_pairs = [
            interp_to_surf_quad(tpair=tpair)
            for tpair in op.interior_trace_pairs(dcoll, state,
                                                 comm_tag=_AgitatorCommTag)
        ]
        vol_fluxes, surf_fluxes = \
            my_advection_rhs(t, state_quad, velocity,
                             state_interior_trace_pairs, flux_only=True)

        return -1.0*div_operator(dcoll, dd_vol_quad, dd_allfaces_quad,
                                 vol_fluxes, surf_fluxes)

    current_step, current_t, current_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step, dt=current_dt,
                      state=current_state, t=current_t, t_final=t_final)

    finish_tol = 1e-12
    print(f"{current_t=}, {t_final=}, {(current_t - t_final)=}")
    assert np.abs(current_t - t_final) < finish_tol

    exact_state = exact_state_func(nodes_base, current_t)
    state_resid = current_state - exact_state
    return actx.to_numpy(op.norm(dcoll, state_resid, 2))


@pytest.mark.parametrize("warp", [1, 2, 3, 4])
def test_dealiasing_with_overintegration(actx_factory, warp):

    actx = actx_factory()
    poly_degree = 4
    tol = 2e-16
    q_expected = int(poly_degree + warp/2. + 3./2. + 1./2.)
    print(f"{q_expected=}")
    l2_error_p = run_agitator(actx, use_overintegration=False,
                              p_adv=warp, order=poly_degree, tpe=True)
    l2_error_n = run_agitator(actx, use_overintegration=True,
                              p_adv=warp, order=poly_degree, tpe=True)
    q_n = poly_degree
    diff = abs(l2_error_p - l2_error_n)
    while diff > tol:
        q_n = q_n + 1
        l2_error_m = l2_error_n
        l2_error_n = run_agitator(actx, use_overintegration=True,
                                  p_adv=warp, order=poly_degree, tpe=True,
                                  quad_order=q_n)
        diff = abs(l2_error_m - l2_error_n)

    print(f"{diff=}, {q_n=}")
    q = q_n + 1
    assert q == q_expected
