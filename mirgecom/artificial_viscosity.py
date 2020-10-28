r""":mod:`mirgecom.artificial viscosity` applys and artifical viscosity to the euler equations
"""

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

#from dataclasses import dataclass

import numpy as np
from pytools.obj_array import make_obj_array
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import (
    interior_trace_pair,
    cross_rank_trace_pairs
)
from grudge.symbolic.primitives import TracePair
from mirgecom.euler import split_conserved, join_conserved


def scalar(s):
    """Create an object array for a scalar."""
    return make_obj_array([s])


def dissapative_flux(discr, q):

    dim = discr.dim
    cv = split_conserved(dim, q)

    return join_conserved(dim,
            mass=np.ones(dim)*scalar(cv.mass),
            energy=np.ones(dim)*scalar(cv.energy),
            momentum= np.ones((dim,dim))*cv.momentum )

def _facial_flux(discr, q_tpair):

    dim = discr.dim

    actx = q_tpair[0].int.array_context

    flux_int = dissapative_flux(discr,q_tpair.int);
    flux_ext = dissapative_flux(discr,q_tpair.ext);
    flux_dis = 0.5*(flux_ext + flux_int); 

    normal = thaw(actx, discr.normal(q_tpair.dd))

    flux_out = flux_dis @ normal

    return discr.project(q_tpair.dd, "all_faces", flux_out)

def artificial_viscosity(discr, r):
    r"""Compute artifical viscosity for the euler equations

    """

    #compute dissapation flux
    vol_flux_r = dissapative_flux(discr, r)
    dflux_r = discr.weak_div(vol_flux_r)

    #interior face flux
    iff_r = _facial_flux(discr, q_tpair=interior_trace_pair(discr,r))

    #partition boundaries flux
    pbf_r = sum(
        _facial_flux(discr, q_tpair=part_pair)
        for part_pair in cross_rank_trace_pairs(discr, r)
    )

    #domain boundary flux
    dir_r = discr.project("vol", BTAG_ALL,r)
    dbf_r = _facial_flux(
            discr, 
            q_tpair=TracePair(BTAG_ALL,interior=dir_r,exterior=dir_r)
        )

    q = discr.inverse_mass( 1.0e-2 * (dflux_r - discr.face_mass(iff_r + pbf_r + dbf_r)))

    #flux of q
    vol_flux_q = dissapative_flux(discr,  q)
    dflux_q =  discr.weak_div(vol_flux_q)

    #interior face flux of q
    iff_q =  _facial_flux(discr, q_tpair=interior_trace_pair(discr,q))

    #flux across partition boundaries
    pbf_q = sum(
        _facial_flux(discr, q_tpair=part_pair)
        for part_pair in cross_rank_trace_pairs(discr, q)
    )

    #dombain boundary flux
    dir_q = discr.project("vol",BTAG_ALL, q);
    dbf_q = _facial_flux(
            discr,
            q_tpair=TracePair(BTAG_ALL,interior=dir_q,exterior=dir_q)
        )
    
    
    return discr.inverse_mass(
        dflux_q - discr.face_mass(iff_q + pbf_q + dbf_q)
    )

