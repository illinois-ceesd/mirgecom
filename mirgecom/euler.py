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

class Vortex:
    def __init__(self):
        self.beta = 5
        self.gamma = 1.4
        self.center = np.array([5, 0])
        self.velocity = np.array([1, 0])
        self.final_time = 0.5

        self.mu = 0
        self.prandtl = 0.72
        self.spec_gas_const = 287.1

    def __call__(self, t, x_vec):
        vortex_loc = self.center + t*self.velocity

        # coordinates relative to vortex center
        x_rel = x_vec[0] - vortex_loc[0]
        y_rel = x_vec[1] - vortex_loc[1]

        # Y.C. Zhou, G.W. Wei / Journal of Computational Physics 189 (2003) 159
        # also JSH/TW Nodal DG Methods, p. 209

        from math import pi
        r = np.sqrt(x_rel**2+y_rel**2)
        expterm = self.beta*numpy.exp(1-r**2)
        u = self.velocity[0] - expterm*y_rel/(2*pi)
        v = self.velocity[1] + expterm*x_rel/(2*pi)
        rho = (1-(self.gamma-1)/(16*self.gamma*pi**2)*expterm**2)**(1/(self.gamma-1))
        p = rho**self.gamma

        e = p/(self.gamma-1) + rho/2*(u**2+v**2)

        return join_fields(rho, e, rho*u, rho*v)


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
    
    momFlux2 = make_obj_array(
        [(rhoV[0] * rhoV[1] / rho), (rhoV[1] * rhoV[1] / rho + p)]
    )

    # physical flux =
    # [ rhoV (rhoE + p)V (rhoV.x.V + delta_ij*p) ]
    flux = join_fields(scalevec(1.0,rhoV), scalevec((rhoE + p) / rho, rhoV), momFlux1, momFlux2,)

    return flux


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
    qjump = join_fields(rho.jump, rhoE.jump, rhoV.jump)
    
    if dim == 2:
        flux_int = _inviscid_flux_2d(qint)
        flux_ext = _inviscid_flux_2d(qext)
    if dim == 3:
        flux_int = _inviscid_flux_3d(qint)
        flux_ext = _inviscid_flux_3d(qext)

#    flux_jump = scalevec(1.0,(flux_int - flux_ext))
    flux_jump = scalevec(0.5,(flux_int + flux_ext))

    # Lax/Friedrichs/Rusunov after JSH/TW Nodal DG Methods, p. 209
    gamma = 1.4
    # p(ideal single gas) = (gamma - 1)*(rhoE - .5(rhoV*V))
    pint = (gamma - 1.0) * (rhoE.int - 0.5 * (np.dot(rhoV.int, rhoV.int)) / rho.int)
    pext = (gamma - 1.0) * (rhoE.ext - 0.5 * (np.dot(rhoV.ext, rhoV.ext)) / rho.ext)
    rhoV_int = rhoV.int
    rhoV_ext = rhoV.ext
    v_int = join_fields([ rhoV_int[i]/rho.int for i in range(dim) ])
    v_ext = join_fields([ rhoV_ext[i]/rho.ext for i in range(dim) ])
    c_int = join_fields( np.sqrt(np.abs((gamma*pint/rho.int).get())) )
    c_ext = join_fields( np.sqrt(np.abs((gamma*pext/rho.ext).get())) )

    fspeed_int = join_fields( np.sqrt((np.dot(v_int,v_int)).get()) + c_int) 
    fspeed_ext = join_fields( np.sqrt((np.dot(v_ext,v_ext)).get()) + c_ext)    
# - Gak!  What's the matter with this block?    
#    lam = np.max(fspeed_int.get(),fspeed_ext.get())
#    lam = fspeed_int
#    print('lam shape = ',lam.shape)
#    print('qjump[0] shape = ',qjump[0].shape)
#    lfr = lam*qjump[0]
#    nflux = flux_jump + lfr
    nflux = flux_jump # skip lax/friedrichs for now
    
    rhofluxjump = nflux[0:dim]
    rhoefluxjump = nflux[dim:2*dim]
    momfluxjump = nflux[2*dim:]
    
    # Surface fluxes should be inviscid flux .dot. normal
    # rhoV .dot. normal
    # (rhoE + p)V  .dot. normal
    # (rhoV.x.V)_1 .dot. normal
    # (rhoV.x.V)_2 .dot. normal
    rho_flux_weak = np.dot(rhofluxjump,normal)
    rhoe_flux_weak = np.dot(rhoefluxjump,normal)
    
    if dim == 2:
        mom_flux_weak = join_fields(np.dot(momfluxjump[0:2],normal),
                                    np.dot(momfluxjump[2:4],normal))
    if dim == 3:
        mom_flux_weak = join_fields(np.dot(momfluxjump[0:2],normal),
                                    np.dot(momfluxjump[2:4],normal),
                                    np.dot(momfluxjump[4:6],normal))
        
    flux_weak = join_fields(
        rho_flux_weak,
        rhoe_flux_weak,
        mom_flux_weak
    )

    return discr.interp(w_tpair.dd, "all_faces", flux_weak)


def _inviscid_flux_3d(q):

    # q = [ rho rhoE rhoV ]
    #    ndim = discr.dim

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

    momFlux1 = make_obj_array(
        [(rhoV[0] * rhoV[0] / rho + p), (rhoV[0] * rhoV[1] / rho),
         (rhoV[0] * rhoV[2] / rho)]
    )
    
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
    return flux


def _inviscid_flux(discr,w):
    if discr.dim == 2:
        return _inviscid_flux_2d(w)

    if discr.dim == 3:
        return _inviscid_flux_3d(w)

def inviscid_operator(discr, w):
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

    # Quick fix for no BCs yet - OK for uniform flow tests
    dir_rho = discr.interp("vol", BTAG_ALL, rho)
    dir_e = discr.interp("vol",BTAG_ALL, rhoE)
    dir_mom = discr.interp("vol",BTAG_ALL, rhoV)
    dir_bval = join_fields(dir_rho, dir_e, dir_mom)
    dir_bc = join_fields(dir_rho,dir_e,dir_mom)


    # vol_flux = [ rhoV, (rhoE + p)V, ((rhoV.x.V) + p*delta_ij) ]
    #        = [ (rho*u, rho*v), ( (rhoE+p)*u, (rhoE+p)*v ),
    #            ( (rhouu + p), rhouv ), ( (rhovu, (rhovv + p)) )
    #          ]
    vol_flux = _inviscid_flux(discr,w)

    dflux = join_fields( discr.weak_div(vol_flux[0:ndim]),
                         discr.weak_div(vol_flux[ndim:2*ndim]),
                         discr.weak_div(vol_flux[2*ndim:3*ndim]),
                         discr.weak_div(vol_flux[3*ndim:4*ndim]) )

    interior_face_flux = _facial_flux(discr,w_tpair=_interior_trace_pair(discr,w))
    
    boundary_flux = _facial_flux(discr,w_tpair=TracePair(BTAG_ALL,dir_bval,dir_bc))
    
    return discr.inverse_mass(
        dflux -
        discr.face_mass( interior_face_flux + boundary_flux ))

    
