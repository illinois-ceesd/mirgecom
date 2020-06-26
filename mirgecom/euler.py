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
    flat_obj_array,
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

    # Fluxes:
    # [ rhoV (rhoE + p)V (rhoV.x.V + delta_ij*p) ]
    momFlux = make_obj_array(
        [
            (rhoV[i] * rhoV[j] / rho + (p if i == j else 0))
            for i in range(ndim)
            for j in range(ndim)
        ]
    )
    massFlux = rhoV * make_obj_array([ 1.0 ])
    energyFlux = rhoV * make_obj_array([ (rhoE + p) / rho ])
    
    flux = flat_obj_array(
        massFlux, energyFlux, momFlux,
    )

    return flux


def _get_wavespeed(w, eos=IdealSingleGas()):

    rho = w[0]
    rhoV = w[2:]

    v = rhoV * make_obj_array( [ 1.0/rho ] )
    
    sos = eos.SpeedOfSound(w)
    wavespeed = clmath.sqrt(np.dot(v, v)) + sos
    return wavespeed


def _facial_flux(discr, w_tpair, eos=IdealSingleGas()):

    dim = discr.dim

    rho = w_tpair[0]
    rhoE = w_tpair[1]
    rhoV = w_tpair[2:]

    normal = with_queue(rho.int.queue, discr.normal(w_tpair.dd))

    # Get inviscid fluxes [rhoV (rhoE + p)V (rhoV.x.V + delta_ij*p) ]
    qint = flat_obj_array(rho.int, rhoE.int, rhoV.int)
    qext = flat_obj_array(rho.ext, rhoE.ext, rhoV.ext)

    # - Figure out how to manage grudge branch dependencies
    #    qjump = flat_obj_array(rho.jump, rhoE.jump, rhoV.jump)
    qjump = qext - qint
    flux_int = _inviscid_flux(discr, qint, eos)
    flux_ext = _inviscid_flux(discr, qext, eos)

    # Lax/Friedrichs/Rusonov after JSH/TW Nodal DG Methods, p. 209
    flux_jump = (flux_int + flux_ext) * make_obj_array([ 0.5 ])

    # wavespeeds = [ wavespeed_int, wavespeed_ext ]
    wavespeeds = [_get_wavespeed(qint), _get_wavespeed(qext)]

    lam = clarray.maximum(wavespeeds[0], wavespeeds[1])
    lfr = qjump * make_obj_array( [ 0.5 * lam ] )

    # Surface fluxes should be inviscid flux .dot. normal
    # rhoV .dot. normal
    # (rhoE + p)V  .dot. normal
    # (rhoV.x.V)_1 .dot. normal
    # (rhoV.x.V)_2 .dot. normal
    nflux = flat_obj_array(
        [
            np.dot(flux_jump[(i * dim) : ((i + 1) * dim)], normal)
            for i in range(dim + 2)
        ]
    )

    # add Lax/Friedrichs term
    flux_weak = nflux + lfr

    return discr.interp(w_tpair.dd, "all_faces", flux_weak)


def inviscid_operator(
    discr,
    w,
    t=0.0,
    eos=IdealSingleGas(),
    boundaries={BTAG_ALL: DummyBoundary()},
):
    """
    Returns the RHS of the Euler flow equations:
    :math: \partial_t Q = - \\nabla \\cdot F
    where Q = [ rho rhoE rhoV ]
          F = [ rhoV (rhoE + p)V (rho(V.x.V) + p*delta_ij) ]
    """

    ndim = discr.dim

    vol_flux = _inviscid_flux(discr, w, eos)
    dflux = flat_obj_array(
        [
            discr.weak_div(vol_flux[(i * ndim) : (i + 1) * ndim])
            for i in range(ndim + 2)
        ]
    )

    interior_face_flux = _facial_flux(
        discr, w_tpair=_interior_trace_pair(discr, w), eos=eos
    )

    # Domain boundaries
    domain_boundary_flux = sum(
        boundaries[btag].get_boundary_flux(
            discr,w,t=t,btag=btag,eos=eos
            ) for btag in boundaries
        )

    # Partition boundaries here
    
    return discr.inverse_mass(
        dflux - discr.face_mass(interior_face_flux + domain_boundary_flux)
    )


def get_inviscid_timestep(discr, w, c=1.0, eos=IdealSingleGas()):

    dim = discr.dim
    mesh = discr.mesh
    order = discr.order

    nelements = mesh.nelements
    nel_1d = nelements ** (1.0 / (1.0 * dim))

    dt = (1.0 - 0.25 * (dim - 1)) / (nel_1d * order ** 2)
    return c * dt
#    dt_ngf = dt_non_geometric_factor(discr.mesh)
#    dt_gf  = dt_geometric_factor(discr.mesh)
#    wavespeeds = _get_wavespeed(w,eos=eos)
#    max_v = clmath.max(wavespeeds)
#    return c*dt_ngf*dt_gf/max_v
