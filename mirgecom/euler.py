__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__author__ = """
Center for Exascale-Enabled Scramjet Design
University of Illinois, Urbana, IL 61801
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
import numpy.linalg as la  # noqa
from pytools.obj_array import (
    join_fields,
    make_obj_array,
    with_object_array_or_scalar,
)
import pyopencl.clmath as clmath
import pyopencl.array as clarray
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from mirgecom.boundary import DummyBoundary
from mirgecom.eos import IdealSingleGas

from grudge.eager import with_queue
from grudge.symbolic.primitives import TracePair
from grudge.dt_finding import (
    dt_geometric_factor,
    dt_non_geometric_factor,
)

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


def _inviscid_flux(discr, q, eos=IdealSingleGas()):

    ndim = discr.dim

    # q = [ rho rhoE rhoV ]
    rho = q[0]
    rhoE = q[1]
    rhoV = q[2:]

    p = eos.Pressure(q)

    def scalevec(scalar, vec):
        # workaround for object array behavior
        return make_obj_array([ni * scalar for ni in vec])

    # physical flux =
    # [ rhoV (rhoE + p)V (rhoV.x.V + delta_ij*p) ]

    momFlux = make_obj_array(
        [
            (rhoV[i] * rhoV[j] / rho + (p if i == j else 0))
            for i in range(ndim)
            for j in range(ndim)
        ]
    )

    flux = join_fields(
        scalevec(1.0, rhoV), scalevec((rhoE + p) / rho, rhoV), momFlux,
    )

    return flux


def _get_wavespeed(w, eos=IdealSingleGas()):

    rho = w[0]
    rhoV = w[2:]

    def scalevec(scalar, vec):
        # workaround for object array behavior
        return make_obj_array([ni * scalar for ni in vec])

    v = scalevec(1.0/rho,rhoV)
    sos = eos.SpeedOfSound(w)
    wavespeed = clmath.sqrt(np.dot(v,v)) + sos
    return wavespeed

def _facial_flux(discr, w_tpair, eos=IdealSingleGas()):

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
    flux_int = _inviscid_flux(discr, qint, eos)
    flux_ext = _inviscid_flux(discr, qext, eos)

    # Lax/Friedrichs/Rusonov after JSH/TW Nodal DG Methods, p. 209
    #    flux_jump = scalevec(1.0,(flux_int - flux_ext))
    flux_jump = scalevec(0.5, (flux_int + flux_ext))

    # wavespeeds = [ wavespeed_int, wavespeed_ext ]
    wavespeeds = [ _get_wavespeed(qint), _get_wavespeed(qext) ]

    lam = clarray.maximum(wavespeeds[0], wavespeeds[1])
    lfr = scalevec(0.5 * lam, qjump)

    # Surface fluxes should be inviscid flux .dot. normal
    # rhoV .dot. normal
    # (rhoE + p)V  .dot. normal
    # (rhoV.x.V)_1 .dot. normal
    # (rhoV.x.V)_2 .dot. normal
    nflux = join_fields(
        [
            np.dot(flux_jump[(i * dim): ((i + 1) * dim)], normal)
            for i in range(dim + 2)
        ]
    )

    # add Lax/Friedrichs term
    flux_weak = nflux + lfr

    return discr.interp(w_tpair.dd, "all_faces", flux_weak)


def inviscid_operator(discr, w, t=0.0, eos=IdealSingleGas(),
                      boundaries={BTAG_ALL: DummyBoundary()}):
    """
    Returns the RHS of the Euler flow equations:
    :math: \partial_t Q = - \\nabla \\cdot F
    where Q = [ rho rhoE rhoV ]
          F = [ rhoV (rhoE + p)V (rho(V.x.V) + p*delta_ij) ]
    """

    ndim = discr.dim

    vol_flux = _inviscid_flux(discr, w, eos)
    dflux = join_fields(
        [
            discr.weak_div(vol_flux[(i * ndim): (i + 1) * ndim])
            for i in range(ndim + 2)
        ]
    )

    interior_face_flux = _facial_flux(
        discr, w_tpair=_interior_trace_pair(discr, w), eos=eos
    )

    def scalevec(scalar, vec):
        # workaround for object array behavior
        return make_obj_array([ni * scalar for ni in vec])
    # Ack! how to avoid this?
    # boundary_flux = join_fields( [discr.zeros(queue) for i in range(numsoln)] )
    boundary_flux = discr.interp("vol", "all_faces", w)
    boundary_flux = scalevec(0.0, boundary_flux)
    for btag in boundaries:
        bndhnd = boundaries[btag]
        boundary_flux += bndhnd.get_boundary_flux(discr, w, t=t,
                                                  btag=btag,eos=eos)

    return discr.inverse_mass(
        dflux - discr.face_mass(interior_face_flux + boundary_flux)
    )

def get_inviscid_timestep(discr,w,c=1.0,eos=IdealSingleGas()):

    dim = discr.dim
    mesh = discr.mesh
    order = discr.order

    nelements = mesh.nelements
    nel_1d = nelements**(1.0/(1.0*dim))
    
    dt = (1.0 - 0.25 * (dim - 1)) / (nel_1d * order ** 2)
    return(c*dt)
#    dt_ngf = dt_non_geometric_factor(discr.mesh)
#    dt_gf  = dt_geometric_factor(discr.mesh)
#    wavespeeds = _get_wavespeed(w,eos=eos)
#    max_v = clmath.max(wavespeeds)
#    return c*dt_ngf*dt_gf/max_v
