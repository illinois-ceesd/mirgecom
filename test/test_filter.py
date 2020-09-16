"""Test filter-related functions and constructs."""

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

import logging
import math
import pytest
import numpy as np
from functools import partial

from meshmode.dof_array import thaw
from grudge.eager import EagerDGDiscretization
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)
from pytools.obj_array import (
    make_obj_array
)
from meshmode.dof_array import thaw  # noqa
from mirgecom.filter import make_spectral_filter
from pytools.obj_array import obj_array_vectorized_n_args
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from mirgecom.eos import IdealSingleGas
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)
from mirgecom.euler import get_inviscid_timestep
from mirgecom.integrators import rk4_step
from mirgecom.initializers import Vortex2D
from mirgecom.boundary import PrescribedBoundary
from mirgecom.euler import inviscid_operator

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("filter_order", [1, 2, 3])
def test_filter_coeff(actx_factory, filter_order, order, dim):
    """
    Test the construction of filter coefficients.

    Tests that the filter coefficients have the right values
    at the imposed band limits of the filter.  Also tests that
    the created filter operator has the expected shape:
    (nummodes x nummodes) matrix, and the filter coefficients
    in the expected positions corresponding to mode ids.
    """
    actx = actx_factory()

    nel_1d = 16

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, n=(nel_1d,) * dim
    )

    discr = EagerDGDiscretization(actx, mesh, order=order)
    vol_discr = discr.discr_from_dd("vol")

    eta = .5  # just filter half the modes
    # counting modes (see
    # JSH/TW Nodal DG Methods, Section 10.1
    # DOI: 10.1007/978-0-387-72067-8)
    nmodes = 1
    for d in range(1, dim+1):
        nmodes *= (order + d)
    nmodes /= math.factorial(int(dim))
    nmodes = int(nmodes)

    cutoff = int(eta * order)

    # number of filtered modes
    nfilt = order - cutoff
    # alpha = f(machine eps)
    # Alpha value suggested by:
    # JSH/TW Nodal DG Methods, Section 5.3
    # DOI: 10.1007/978-0-387-72067-8
    alpha = -1.0*np.log(np.finfo(float).eps)

    # expected values @ filter band limits
    expected_high_coeff = np.exp(-1.0*alpha)
    expected_cutoff_coeff = 1.0
    if dim == 1:
        cutoff_indices = [cutoff]
        high_indices = [order]
    elif dim == 2:
        sk = 0
        cutoff_indices = []
        high_indices = []
        for i in range(order + 1):
            for j in range(order - i + 1):
                if (i + j) == cutoff:
                    cutoff_indices.append(sk)
                if (i + j) == order:
                    high_indices.append(sk)
                sk += 1
    elif dim == 3:
        sk = 0
        cutoff_indices = []
        high_indices = []
        for i in range(order + 1):
            for j in range(order - i + 1):
                for k in range(order - (i + j) + 1):
                    if (i + j + k) == cutoff:
                        cutoff_indices.append(sk)
                    if (i + j + k) == order:
                        high_indices.append(sk)
                    sk += 1

    if nfilt <= 0:
        expected_high_coeff = 1.0

    from mirgecom.filter import exponential_mode_response_function as xmrfunc
    frfunc = partial(xmrfunc, alpha=alpha, filter_order=filter_order)

    from modepy import vandermonde
    for group in vol_discr.groups:
        mode_ids = group.mode_ids()
        vander = vandermonde(group.basis(), group.unit_nodes)
        vanderm1 = np.linalg.inv(vander)
        filter_coeff = make_spectral_filter(group, cutoff=cutoff,
                                            mode_response_function=frfunc)
        assert(filter_coeff.shape == vanderm1.shape)
        for mode_index, mode_id in enumerate(mode_ids):
            mode = mode_id
            if dim > 1:
                mode = sum(mode_id)
            if mode == cutoff:
                assert(filter_coeff[mode_index][mode_index] == expected_cutoff_coeff)
            if mode == order:
                assert(filter_coeff[mode_index][mode_index] == expected_high_coeff)


@obj_array_vectorized_n_args
def _apply_linear_operator(discr, operator, fields):
    """Apply *operator* matrix to *fields*.

    This is a utility used in testing only. It assumes
    a one-group discretization.
    """
    assert(len(discr.groups) == 1)
    from mirgecom.filter import linear_operator_kernel
    actx = fields.array_context
    result = discr.empty(actx, dtype=fields.entry_dtype)
    for group in discr.groups:
        actx.call_loopy(
            linear_operator_kernel(),
            mat=actx.from_numpy(operator),
            result=result[group.index],
            vec=fields[group.index])
    return result


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("order", [2, 3, 4])
def test_filter_function(actx_factory, dim, order, do_viz=False):
    """
    Test the stand-alone procedural interface to spectral filtering.

    Tests that filtered fields have expected attenuated higher modes.
    """
    actx = actx_factory()

    logger = logging.getLogger(__name__)
    filter_order = 1
    nel_1d = 2
    eta = .5   # filter half the modes
    # Alpha value suggested by:
    # JSH/TW Nodal DG Methods, Seciton 5.3
    # DOI: 10.1007/978-0-387-72067-8
    alpha = -1.0*np.log(np.finfo(float).eps)

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(0.0,) * dim, b=(1.0,) * dim, n=(nel_1d,) * dim
    )

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())

    # number of modes (see
    # JSH/TW Nodal DG Methods, Section 10.1
    # DOI: 10.1007/978-0-387-72067-8)
    nummodes = int(1)
    for i in range(dim):
        nummodes *= int(order + i + 1)
    nummodes /= math.factorial(int(dim))
    cutoff = int(eta * order)

    from mirgecom.filter import exponential_mode_response_function as xmrfunc
    frfunc = partial(xmrfunc, alpha=alpha, filter_order=filter_order)

    vol_discr = discr.discr_from_dd("vol")
    groups = vol_discr.groups
    group = groups[0]

    # First test a uniform field, which should pass through
    # the filter unharmed.
    from mirgecom.initializers import Uniform
    initr = Uniform(numdim=dim)
    uniform_soln = initr(t=0, x_vec=nodes)

    from mirgecom.filter import filter_modally
    filtered_soln = filter_modally(discr, "vol", cutoff=cutoff,
                                   mode_resp_func=frfunc, field=uniform_soln)
    soln_resid = uniform_soln - filtered_soln
    max_errors = [discr.norm(v, np.inf) for v in soln_resid]

    tol = 1e-14

    logger.info(f"Max Errors (uniform field) = {max_errors}")
    assert(np.max(max_errors) < tol)

    # construct polynomial field:
    # a0 + a1*x + a2*x*x + ....
    def polyfn(coeff):  # , x_vec):
        # r = actx.np.sqrt(np.dot(nodes, nodes))
        r = nodes[0]
        result = 0
        for n, a in enumerate(coeff):
            result += a * r ** n
        return make_obj_array([result])

    # Any order {cutoff} and below fields should be unharmed
    tol = 1e-14
    field_order = int(cutoff)
    coeff = [1.0 / (i + 1) for i in range(field_order + 1)]
    field = polyfn(coeff=coeff)
    filtered_field = filter_modally(discr, "vol", cutoff=cutoff,
                                    mode_resp_func=frfunc, field=field)
    soln_resid = field - filtered_field
    max_errors = [discr.norm(v, np.inf) for v in soln_resid]
    logger.info(f"Field = {field}")
    logger.info(f"Filtered = {filtered_field}")
    logger.info(f"Max Errors (poly) = {max_errors}")
    assert(np.max(max_errors) < tol)

    # Any order > cutoff fields should have higher modes attenuated
    threshold = 1e-3
    tol = 1e-1
    if do_viz is True:
        from grudge.shortcuts import make_visualizer
        vis = make_visualizer(discr, discr.order)

    from modepy import vandermonde
    for field_order in range(cutoff+1, cutoff+4):
        coeff = [1.0 / (i + 1) for i in range(field_order+1)]
        field = polyfn(coeff=coeff)
        filtered_field = filter_modally(discr, "vol", cutoff=cutoff,
                                        mode_resp_func=frfunc, field=field)
        for group in vol_discr.groups:
            vander = vandermonde(group.basis(), group.unit_nodes)
            vanderm1 = np.linalg.inv(vander)
            unfiltered_spectrum = _apply_linear_operator(vol_discr, vanderm1, field)
            filtered_spectrum = _apply_linear_operator(vol_discr, vanderm1,
                                                      filtered_field)
            if do_viz is True:
                spectrum_resid = unfiltered_spectrum - filtered_spectrum
                io_fields = [
                    ("unfiltered", field),
                    ("filtered", filtered_field),
                    ("unfiltered_spectrum", unfiltered_spectrum),
                    ("filtered_spectrum", filtered_spectrum),
                    ("residual", spectrum_resid)
                ]
                vis.write_vtk_file(f"filter_test_{field_order}.vtu", io_fields)
            field_resid = unfiltered_spectrum - filtered_spectrum
            max_errors = [discr.norm(v, np.inf) for v in field_resid]
            # fields should be different, but not too different
            assert(tol > np.max(max_errors) > threshold)


def _euler_flow_stepper(actx, parameters):
    """Implement a generic time stepping loop for testing an inviscid flow."""
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    mesh = parameters["mesh"]
    t = parameters["time"]
    order = parameters["order"]
    t_final = parameters["tfinal"]
    initializer = parameters["initializer"]
    boundaries = parameters["boundaries"]
    eos = parameters["eos"]
    cfl = parameters["cfl"]
    dt = parameters["dt"]
    constantcfl = parameters["constantcfl"]
    filteron = parameters["filteron"]

    if t_final <= t:
        return(0.0)

    dim = mesh.dim
    istep = 0

    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(actx, discr.nodes())
    fields = initializer(0, nodes)
    sdt = get_inviscid_timestep(discr, eos=eos, cfl=cfl, q=fields)

    def rhs(t, q):
        return inviscid_operator(discr, eos=eos, boundaries=boundaries, q=q, t=t)

    filter_order = 4
    # alpha = -1.0*np.log(np.finfo(float).eps)
    alpha = .01
    nummodes = int(1)
    for i in range(dim):
        nummodes *= int(order + i + 1)
    nummodes /= math.factorial(int(dim))
    cutoff = nummodes / 2 + 1
    print(f"nummodes={nummodes}, cutoff={cutoff}")

    from mirgecom.filter import (
        exponential_mode_response_function as xmrfunc,
        filter_modally
    )
    frfunc = partial(xmrfunc, alpha=alpha, filter_order=filter_order)

    while t < t_final:

        if constantcfl is True:
            dt = sdt
        else:
            cfl = dt / sdt

        fields = rk4_step(fields, t, dt, rhs)
        if filteron > 0:
            fields = filter_modally(discr, "vol", cutoff=cutoff,
                                    mode_resp_func=frfunc, field=fields)

        t += dt
        istep += 1

        sdt = get_inviscid_timestep(discr, eos=eos, cfl=cfl, q=fields)

    expected_result = initializer(t, nodes)
    resid = fields - expected_result
    err2 = [discr.norm(resid[i], 2) for i in range(dim+2)]

    return(err2)


def test_filter_eoc(actx_factory):
    """Test the effect of filtering on Vortex2D EOC.

    Advance the 2D isentropic vortex case in
    time with zero velocities using an RK4
    timestepping scheme. Check the advanced field
    values against the exact/analytic expressions and
    estimate EOC.  Repeat with filtering enabled, to
    see that design order is still achieved - but at
    slightly lower rate.
    """
    actx = actx_factory()

    dim = 2
    order = 5

    from pytools.convergence import EOCRecorder

    for filteron in range(2):
        eoc_rec = EOCRecorder()
        for nel_1d in [16, 24, 32]:
            from meshmode.mesh.generation import (
                generate_regular_rect_mesh,
            )

            mesh = generate_regular_rect_mesh(
                a=(-5.0,) * dim, b=(5.0,) * dim, n=(nel_1d,) * dim
            )

            exittol = 2.0
            t_final = 0.000001
            cfl = 1.0
            vel = np.zeros(shape=(dim,))
            orig = np.zeros(shape=(dim,))
            # vel[:dim] = 1.0
            dt = .000001
            initializer = Vortex2D(center=orig, velocity=vel)
            casename = "Vortex"
            boundaries = {BTAG_ALL: PrescribedBoundary(initializer)}
            eos = IdealSingleGas()
            t = 0
            flowparams = {"dim": dim, "dt": dt, "order": order, "time": t,
                        "boundaries": boundaries, "initializer": initializer,
                          "eos": eos, "casename": casename, "mesh": mesh,
                          "tfinal": t_final, "exittol": exittol, "cfl": cfl,
                          "constantcfl": False, "nstatus": 0, "filteron": filteron}
            err2 = _euler_flow_stepper(actx, flowparams)
            h = 10.0 / (nel_1d - 1)
            eoc_rec.add_data_point(h, err2[0])
            print(f"h={h}, err={err2}, filter={filteron}")
        message = (
            f"Error for (dim,order) = ({dim},{order}):\n"
            f"{eoc_rec}"
        )
        logger.info(message)
        order_estimate = eoc_rec.order_estimate()
        #        assert (
        #            eoc_rec.order_estimate() >= order - 0.5
        #            or eoc_rec.max_error() < 1e-11
        #        )
        print(f"order_estimate{filteron} = {order_estimate}")
    assert(False)
