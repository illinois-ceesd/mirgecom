__copyright__ = "Copyright (C) 2020 CEESD Developers"

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
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

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


def _inviscid_flux_2d(q):

    # q = [ rho rhoE rhoV ]
    #    ndim = discr.dim

    rho = q[0]
    rhoE = q[1]
    rhoV = q[2:]

    #    print('rhoV shape = ',rhoV.shape)
    
    # --- EOS stuff TBD ---
    # gamma (ideal monatomic) = 1.4
    gamma = 1.4
    # p(ideal single gas) =
    p = (gamma - 1.0) * (rhoE - 0.5 * (np.dot(rhoV, rhoV)) / rho)

    def scalevec(scalar, vec):
        # workaround for object array behavior
        return make_obj_array([ni * scalar for ni in vec])

    momFlux1 = make_obj_array(
        [(rhoV[0] * rhoV[0] / rho + p), (rhoV[0] * rhoV[1] / rho)]
    )
    #    print('momFlux1.shape = ',momFlux1.shape)
    
    momFlux2 = make_obj_array(
        [(rhoV[0] * rhoV[1] / rho), (rhoV[1] * rhoV[1] / rho + p)]
    )
    # physical flux =
    # [ rhoV (rhoE + p)V (rhoV.x.V + delta_ij*p) ]
    flux = join_fields(scalevec(1.0,rhoV), scalevec((rhoE + p) / rho, rhoV), momFlux1, momFlux2,)
    #    print("flux.shape = ",flux.shape)
    return flux


def _facial_flux(discr, w_tpair):

    dim = discr.dim
    
    rho = w_tpair[0]
    rhoE = w_tpair[1]
    rhoV = w_tpair[2:]

    normal = with_queue(rho.int.queue, discr.normal(w_tpair.dd))
    #    print ("normal shape = ",normal.shape)
    #    print("normal = ",normal)
    # Get inviscid fluxes [rhoV (rhoE + p)V (rhoV.x.V + delta_ij*p) ]
    qint = join_fields(rho.int, rhoE.int, rhoV.int)
    qext = join_fields(rho.ext, rhoE.ext, rhoV.ext)

    if dim == 2:
        flux_int = _inviscid_flux_2d(qint)
        flux_ext = _inviscid_flux_2d(qext)
    if dim == 3:
        flux_int = _inviscid_flux_3d(qint)
        flux_ext = _inviscid_flux_3d(qext)
        
    flux_avg = 0.5 * (flux_int + flux_ext)
    
    rhofluxavg = flux_avg[0:dim]
    rhoefluxavg = flux_avg[dim:2*dim]
    momfluxavg = flux_avg[2*dim:]

#    print('rhofluxavg shape = ',rhofluxavg.shape)
#    print('rhoefluxavg shape = ',rhoefluxavg.shape)
#    print('momfluxavg shape = ',momfluxavg.shape)
    
    # Surface fluxes should be inviscid flux .dot. normal
    # rhoV .dot. normal
    # (rhoE + p)V  .dot. normal
    # (rhoV.x.V)_1 .dot. normal
    # (rhoV.x.V)_2 .dot. normal
    # What about upwinding?

    if dim == 2:
        mom_flux_weak = join_fields(np.dot(momfluxavg[0:2],normal),
                                    np.dot(momfluxavg[2:4],normal))
    if dim == 3:
        mom_flux_weak = join_fields(np.dot(momfluxavg[0:2],normal),
                                    np.dot(momfluxavg[2:4],normal),
                                    np.dot(momfluxavg[4:6],normal))
        
    flux_weak = join_fields(
        np.dot(rhofluxavg, normal),
        np.dot(rhoefluxavg, normal),
        mom_flux_weak
    )

    #  HOWTO: --- upwinding?
    #    v_jump = np.dot(normal, v.int-v.ext)
    #    flux_weak -= join_fields(
    #            0.5*(u.int-u.ext),
    #            0.5*normal_times(v_jump),
    #            )

    return discr.interp(w_tpair.dd, "all_faces", flux_weak)


def inviscid_operator(discr, w):
    """
    Returns the RHS of the Euler flow equations:
    :math: \partial_t Q = - \\nabla \\cdot F
    """

    ndim = discr.dim

    rho = w[0]
    rhoE = w[1]
    rhoV = w[2:]

    # We'll use exact soln of isentropic vortex for boundary/BC
    # Spiegel (https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20150018403.pdf)
    # AK has this coded in "hedge" code: gas_dynamics_initials.py
    dir_rho = discr.interp("vol", BTAG_ALL, rho)
    dir_e = discr.interp("vol",BTAG_ALL, rhoE)
    dir_mom = discr.interp("vol",BTAG_ALL, rhoV)

    dir_bval = join_fields(dir_rho, dir_e, dir_mom)
    dir_bc = join_fields(dir_rho,dir_e,dir_mom)
    #    dir_bval = join_fields(dir_u, dir_v)
    #    dir_bc = join_fields(-dir_u, dir_v)

    # vol_flux = [ rhoV, (rhoE + p)V, ((rhoV.x.V) + p*delta_ij) ]
    #        = [ (rho*u, rho*v), ( (rhoE+p)*u, (rhoE+p)*v ),
    #            ( (rhouu + p), rhouv ), ( (rhovu, (rhovv + p)) )
    #          ]
    vol_flux = _inviscid_flux_2d(w)
    #    print('vol_flux shape = ',vol_flux.shape)
    dflux = join_fields( discr.weak_div(vol_flux[0:ndim]),
                         discr.weak_div(vol_flux[ndim:2*ndim]),
                         discr.weak_div(vol_flux[2*ndim:3*ndim]),
                         discr.weak_div(vol_flux[3*ndim:4*ndim]) )
    #    print('dflux shape = ',dflux.shape)
    #    print('dflux = ',dflux)
    interior_face_flux = _facial_flux(discr,w_tpair=_interior_trace_pair(discr,w))
    #    print('interior_face_flux shape = ',interior_face_flux.shape)
    
    boundary_flux = _facial_flux(discr,w_tpair=TracePair(BTAG_ALL,dir_bval,dir_bc))
    #    print('boundary_flux = shape',boundary_flux.shape)
    
    # vol_flux is already "joined"
    return discr.inverse_mass(
        dflux -
        discr.face_mass( interior_face_flux + boundary_flux ))

def _inviscid_flux_3d(q):

    # q = [ rho rhoE rhoV ]
    #    ndim = discr.dim

    rho = q[0]
    rhoE = q[1]
    rhoV = q[2:]

    #    print('rhoV shape = ',rhoV.shape)
    
    # --- EOS stuff TBD ---
    # gamma (ideal monatomic) = 1.4
    gamma = 1.4
    # p(ideal single gas) =
    p = (gamma - 1.0) * (rhoE - 0.5 * (np.dot(rhoV, rhoV)) / rho)

    def scalevec(scalar, vec):
        # workaround for object array behavior
        return make_obj_array([ni * scalar for ni in vec])

    momFlux1 = make_obj_array(
        [(rhoV[0] * rhoV[0] / rho + p), (rhoV[0] * rhoV[1] / rho),
         (rhoV[0] * rhoV[2] / rho)]
    )
    #    print('momFlux1.shape = ',momFlux1.shape)
    
    momFlux2 = make_obj_array(
        [(rhoV[0] * rhoV[1] / rho), (rhoV[1] * rhoV[1] / rho + p),
         (rhoV[1] * rhoV[2] / rho)]
    )

    momFlux3 = make_obj_array(
        [(rhoV[0] * rhoV[2] / rho ), (rhoV[2] * rhoV[1]/rho),
         (rhoV[2] * rhoV[2] / rho + p)]
    )
        
    # physical flux =
    # [ rhoV (rhoE + p)V (rhoV.x.V + delta_ij*p) ]
    flux = join_fields(scalevec(1.0,rhoV),
                       scalevec((rhoE + p) / rho, rhoV),
                       momFlux1, momFlux2, momFlux3)
    #    print("flux.shape = ",flux.shape)
    return flux

def _flux_3d(discr, w_tpair):

    rho = w_tpair[0]
    rhoE = w_tpair[1]
    rhoV = w_tpair[2:]

    normal = with_queue(rho.int.queue, discr.normal(w_tpair.dd))

    # Get inviscid fluxes [rhoV (rhoE + p)V (rhoV.x.V + delta_ij*p) ]
    qint = join_fields(rho.int, rhoE.int, rhoV.int)
    qext = join_fields(rho.ext, rhoE.ext, rhoV.ext)
    flux_int = _inviscid_flux_3d(qint)
    flux_ext = _inviscid_flux_3d(qext)

    flux_avg = 0.5 * (flux_int + flux_ext)
    
    rhofluxavg = flux_avg[0:1]
    rhoefluxavg = flux_avg[2:3]
    rhov0fluxavg = flux_avg[4:5]
    rhov1fluxavg = flux_avg[6:7]
    
    # Surface fluxes should be inviscid flux .dot. normal
    # rhoV .dot. normal
    # (rhoE + p)V  .dot. normal
    # (rhoV.x.V)_1 .dot. normal
    # (rhoV.x.V)_2 .dot. normal
    # What about upwinding?

    flux_weak = join_fields(
        np.dot(rhofluxavg, normal),
        np.dot(rhoefluxavg, normal),
        np.dot(rhov0fluxavg, normal),
        np.dot(rhov1fluxavg, normal),
    )

    #  HOWTO: --- upwinding?
    #    v_jump = np.dot(normal, v.int-v.ext)
    #    flux_weak -= join_fields(
    #            0.5*(u.int-u.ext),
    #            0.5*normal_times(v_jump),
    #            )

    return discr.interp(w_tpair.dd, "all_faces", flux_weak)


def euler_operator(discr, w):
    """
    Returns the RHS of the Euler flow equations:
    :math: \partial_t Q = - \\nabla \\cdot F
    """

    ndim = discr.dim

    rho = w[0]
    rhoE = w[1]
    rhoV = w[2:]

    # We'll use exact soln of isentropic vortex for boundary/BC
    # Spiegel (https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20150018403.pdf)
    # AK has this coded in "hedge" code: gas_dynamics_initials.py
    dir_rho = discr.interp("vol", BTAG_ALL, rho)
    dir_e = discr.interp("vol",BTAG_ALL, rhoE)
    dir_mom = discr.interp("vol",BTAG_ALL, rhoV)
    
    dir_bval = join_fields(dir_rho, dir_e, dir_mom)
    dir_bc = join_fields(dir_rho,dir_e,dir_mom)
    #    dir_bval = join_fields(dir_u, dir_v)
    #    dir_bc = join_fields(-dir_u, dir_v)

    # vol_flux = [ rhoV, (rhoE + p)V, ((rhoV.x.V) + p*delta_ij) ]
    #        = [ (rho*u, rho*v), ( (rhoE+p)*u, (rhoE+p)*v ),
    #            ( (rhouu + p), rhouv ), ( (rhovu, (rhovv + p)) )
    #          ]
    vol_flux = _inviscid_flux_2d(discr, w)
    # vol_flux is already "joined"
    return discr.inverse_mass(
        vol_flux
        - discr.face_mass(  # noqa: W504
            _flux_2d(discr, w_tpair=_interior_trace_pair(discr, w))
            + _flux_2d(discr, w_tpair=TracePair(BTAG_ALL, dir_bval,
                                                dir_bc))
        )
    )


