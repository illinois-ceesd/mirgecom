__copyright__ = (
    "Copyright (C) 2020 University of Illinois Board of Trustees"
)

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
import numpy.linalg as la  # noqa
from pytools.obj_array import (
    join_fields,
    make_obj_array,
    with_object_array_or_scalar,
)
import pyopencl.clmath as clmath
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from mirgecom.boundary import BoundaryBoss

# TODO: Remove grudge dependence?
from grudge.eager import with_queue
from grudge.symbolic.primitives import TracePair


__doc__ = """
.. autofunction:: inviscid_operator
"""


#
# Euler flow eqns:
# d_t(q) + nabla .dot. f = 0 (no sources atm)
# state vector q: [rho rhoE rhoV]
# flux tensor f: [rhoV (rhoE + p)V (rhoV.x.V + delta_ij*p)]
#

def _interior_trace_pair(discr, vec):
    i = discr.interp("vol", "int_faces", vec)
    e = with_object_array_or_scalar(
        lambda el: discr.opposite_face_connection()(el.queue, el), i
    )
    return TracePair("int_faces", i, e)


def _inviscid_flux(discr, q):

    # q = [ rho rhoE rhoV ]
    ndim = discr.dim

    rho = q[0]
    rhoE = q[1]
    rhoV = q[2:]

    # --- EOS stuff TBD ---
    # gamma (ideal monatomic) = 1.4
    gamma = 1.4
    # p(ideal single gas) =
    p = (gamma - 1.0) * (rhoE - 0.5 * (np.dot(rhoV, rhoV)) / rho)

    def scalevec(scalar, vec):
        # workaround for object array behavior
        return make_obj_array([ni * scalar for ni in vec])

    momFlux = make_obj_array(
        [
            (rhoV[i] * rhoV[j] / rho + (p if i == j else 0))
            for i in range(ndim)
            for j in range(ndim)
        ]
    )

    # physical flux =
    # [ rhoV (rhoE + p)V (rhoV.x.V + delta_ij*p) ]
    flux = join_fields(
        scalevec(1.0, rhoV), scalevec((rhoE + p) / rho, rhoV), momFlux,
    )

    return flux


def _get_wavespeeds(discr, w_tpair):

    dim = discr.dim

    rho = w_tpair[0]
    rhoE = w_tpair[1]
    rhoV = w_tpair[2]

    def scalevec(scalar, vec):
        # workaround for object array behavior
        return make_obj_array([ni * scalar for ni in vec])

    gamma = 1.4

    pint, pext = [
        ((gamma - 1.0) * (rhoe - 0.5 * (np.dot(rhov, rhov) / lrho)))
        for rhoe, rhov, lrho in [
            (rhoE.int, rhoV.int, rho.int),
            (rhoE.ext, rhoV.ext, rho.ext),
        ]
    ]

    v_int, v_ext = [
        scalevec(1.0 / lrho, lrhov)
        for lrho, lrhov in [(rho.int, rhoV.int), (rho.ext, rhoV.ext)]
    ]

    c_int, c_ext = [
        clmath.sqrt(gamma * lp / lrho)
        for lp, lrho in [(pint, rho.int), (pext, rho.ext)]
    ]

    fspeed_int, fspeed_ext = [
        (clmath.sqrt(np.dot(lv, lv)) + lc)
        for lv, lc in [(v_int, c_int), (v_ext, c_ext)]
    ]

    return join_fields(fspeed_int, fspeed_ext)


def _facial_flux(discr, w_tpair):

    dim = discr.dim

    rho = w_tpair[0]
    rhoE = w_tpair[1]
    rhoV = w_tpair[2:]

    def scalevec(scalar, vec):
        # workaround for object array behavior
        return make_obj_array([ni * scalar for ni in vec])

    normal = with_queue(rho.int.queue, discr.normal(w_tpair.dd))

    # Get inviscid fluxes [rhoV (rhoE + p)V (rhoV.x.V + delta_ij*p) ]
    qint = join_fields(rho.int, rhoE.int, rhoV.int)
    qext = join_fields(rho.ext, rhoE.ext, rhoV.ext)
    # - Figure out how to manage grudge branch dependencies
    #    qjump = join_fields(rho.jump, rhoE.jump, rhoV.jump)
    qjump = qext - qint
    flux_int = _inviscid_flux(discr, qint)
    flux_ext = _inviscid_flux(discr, qext)

    # Lax/Friedrichs/Rusunov after JSH/TW Nodal DG Methods, p. 209
    #    flux_jump = scalevec(1.0,(flux_int - flux_ext))
    flux_jump = scalevec(0.5, (flux_int + flux_ext))
    gamma = 1.4
    # p(ideal single gas) = (gamma - 1)*(rhoE - .5(rhoV*V))
    pint, pext = [
        ((gamma - 1.0) * (rhoe - 0.5 * (np.dot(rhov, rhov) / lrho)))
        for rhoe, rhov, lrho in [
            (rhoE.int, rhoV.int, rho.int),
            (rhoE.ext, rhoV.ext, rho.ext),
        ]
    ]
    v_int, v_ext = [
        scalevec(1.0 / lrho, lrhov)
        for lrho, lrhov in [(rho.int, rhoV.int), (rho.ext, rhoV.ext)]
    ]

    c_int, c_ext = [
        clmath.sqrt(gamma * lp / lrho)
        for lp, lrho in [(pint, rho.int), (pext, rho.ext)]
    ]

    fspeed_int, fspeed_ext = [
        (clmath.sqrt(np.dot(lv, lv)) + lc)
        for lv, lc in [(v_int, c_int), (v_ext, c_ext)]
    ]
    # fspeed_int, fspeed_ext = _get_wavespeeds(discr, w_tpair)

    # - Gak!  What's the matter with this next line?   (CL issues?)
    #    lam = np.max(fspeed_int.get(),fspeed_ext.get())
    lam = fspeed_int
    #    print('lam shape = ',lam.shape)
    #    print('qjump[0] shape = ',qjump[0].shape)
    lfr = scalevec(0.5 * lam, qjump)

    # Surface fluxes should be inviscid flux .dot. normal
    # rhoV .dot. normal
    # (rhoE + p)V  .dot. normal
    # (rhoV.x.V)_1 .dot. normal
    # (rhoV.x.V)_2 .dot. normal
    nflux = join_fields(
        [
            np.dot(flux_jump[i * dim : (i + 1) * dim], normal)
            for i in range(dim + 2)
        ]
    )

    # add Lax/Friedrichs term
    flux_weak = nflux + lfr

    return discr.interp(w_tpair.dd, "all_faces", flux_weak)


def inviscid_operator(discr, w, t=0.0, boundaries=BoundaryBoss()):
    """
    Returns the RHS of the Euler flow equations:
    :math: \partial_t Q = - \\nabla \\cdot F
    where Q = [ rho rhoE rhoV ]
          F = [ rhoV (rhoE + p)V (rho(V.x.V) + p*delta_ij) ]
    """

    ndim = discr.dim

    rho = w[0]
    rhoE = w[1]
    rhoV = w[2:]

    # We'll use exact soln of isentropic vortex for boundary/BC
    # Spiegel (https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20150018403.pdf)
    # AK has this coded in "hedge" code: gas_dynamics_initials.py

    # Quick fix for no BCs yet - OK for time-independent flow tests
    #    dir_rho = discr.interp("vol", BTAG_ALL, rho)
    #    dir_e = discr.interp("vol", BTAG_ALL, rhoE)
    #    dir_mom = discr.interp("vol", BTAG_ALL, rhoV)
    #    dir_bval = join_fields(dir_rho, dir_e, dir_mom)
    #    dir_bc = join_fields(dir_rho, dir_e, dir_mom)

    vol_flux = _inviscid_flux(discr, w)
    dflux = join_fields(
        [
            discr.weak_div(vol_flux[i * ndim : (i + 1) * ndim])
            for i in range(ndim + 2)
        ]
    )

    interior_face_flux = _facial_flux(
        discr, w_tpair=_interior_trace_pair(discr, w)
    )

    boundary_flux = boundaries.GetBoundaryFlux(discr, w, t)

    #    boundary_flux = _facial_flux(
    #        discr, w_tpair=TracePair(BTAG_ALL, dir_bval, dir_bc)
    #    )

    return discr.inverse_mass(
        dflux - discr.face_mass(interior_face_flux + boundary_flux)
    )
