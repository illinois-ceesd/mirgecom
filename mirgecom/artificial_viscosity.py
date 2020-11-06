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
from pytools.obj_array import make_obj_array, obj_array_vectorize, flat_obj_array,obj_array_vectorize_n_args
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import (
    interior_trace_pair,
    cross_rank_trace_pairs
)
from grudge.symbolic.primitives import TracePair
from mirgecom.euler import split_conserved, join_conserved


def _facial_flux_r(discr, q_tpair):

    dim = discr.dim
    actx = q_tpair[0].int.array_context

    flux_dis = q_tpair.avg

    normal = thaw(actx, discr.normal(q_tpair.dd))

    flux_out = flux_dis * normal
    
    # Can't do it here... "obj arrays not allowed on compute device"
    #def flux_calc(flux):
    #    return (flux * normal)
    #flux_out = obj_array_vectorize(flux_calc,flux_dis)

    return discr.project(q_tpair.dd, "all_faces", flux_out)

def _facial_flux_q(discr, q_tpair):

    dim = discr.dim

    actx = q_tpair[0].int.array_context

    normal = thaw(actx, discr.normal(q_tpair.dd))

    flux_out = np.dot(q_tpair.avg,normal)

    return discr.project(q_tpair.dd, "all_faces", flux_out)

def artificial_viscosity(discr, t, eos, boundaries, r, alpha):
    r"""Compute artifical viscosity for the euler equations

    """
    #compute dissapation flux

    #Cannot call weak_grad on obj of nd arrays, use obj_array_vectorize as work around
    dflux_r = obj_array_vectorize(discr.weak_grad,r)
    
    #interior face flux

    #Doesn't work: something related to obj on compute device
    #Not 100% on reason
    #qin = interior_trace_pair(discr,r)
    #iff_r = _facial_flux(discr,q_tpair=qin)

    #Work around?
    def my_facialflux_r_interior(q):
        qin = interior_trace_pair(discr,make_obj_array([q]))
        return _facial_flux_r(discr,q_tpair=qin)

    iff_r = obj_array_vectorize(my_facialflux_r_interior,r)
    

    #partition boundaries flux
    #flux across partition boundaries
    def my_facialflux_r_partition(q):
        qin = cross_rank_trace_pairs(discr,q)
        return  sum(_facial_flux_r(discr,q_tpair=part_pair) for part_pair in cross_rank_trace_pairs(discr,make_obj_array([q])) )
    
    pbf_r = obj_array_vectorize(my_facialflux_r_partition,r)
    
    #pbf_r = sum(
    #    _facial_flux_r(discr, q_tpair=part_pair)
    #    for part_pair in cross_rank_trace_pairs(discr, r)
    #)

    #domain boundary flux basic boundary implementation
    #def my_facialflux2(r):
    #    dir_r = discr.project("vol", BTAG_ALL,make_obj_array([r]))
    #    dbf_r = _facial_flux(
    #         discr, 
    #         q_tpair=TracePair(BTAG_ALL,interior=dir_r,exterior=dir_r)
    #     )
    #    return (dbf_r)
    #dbf_r = obj_array_vectorize(my_facialflux2,r)


    #True boundary implementation
    #Okay, not sure about this...
    #What I am attempting:
    #       1. Loop through all the boundaries
    #       2. Define a function my_TP that performes the trace pair for the given boundary
    #            given a solution variable
    #       3. Get the external solution from the boundary routine
    #       4. Get hte projected internal solution
    #       5. Compute the boundary flux as a sum over boundaries, using the obj_array_vectorize to
    #           pass each solution variable one at a time
    # DO I really need to do this like this?
    dbf_r = 0.0*iff_r
    for btag in boundaries:
            def my_facialflux_r_boundary(sol_ext,sol_int):
                q_tpair = TracePair(btag,interior=make_obj_array([sol_int]),exterior=make_obj_array([sol_ext]))
                return _facial_flux_r(discr,q_tpair=q_tpair)
            r_ext=boundaries[btag].interior_sol(discr,eos=eos,btag=btag,t=t,q=r)
            r_int=discr.project("vol",btag,r)
            dbf_r = dbf_r + obj_array_vectorize_n_args(my_facialflux_r_boundary,r_ext,r_int)
            
        
    #Compute q, half way done!
    q = discr.inverse_mass( -alpha * (dflux_r - discr.face_mass(iff_r + pbf_r + dbf_r)))

    #flux of q
    
    #Again we need to vectorize
    #q is a object array of object arrays (dim,) of DOFArrays (?)
    dflux_q =  obj_array_vectorize(discr.weak_div,q)

    #interior face flux of q
    def my_facialflux_q_interior(q):
        qin = interior_trace_pair(discr,q)
        iff_q =  _facial_flux_q(discr, q_tpair=qin)
        return (iff_q)

    iff_q = obj_array_vectorize(my_facialflux_q_interior,q)
    

    #flux across partition boundaries
    def my_facialflux_q_partition(q):
        qin = cross_rank_trace_pairs(discr,q)
        return  sum(_facial_flux_q(discr,q_tpair=part_pair) for part_pair in cross_rank_trace_pairs(discr,make_obj_array([q])) )
    
    pbf_q = obj_array_vectorize(my_facialflux_q_partition,q)
    #pbf_q = sum(
    #    _facial_flux_q(discr, q_tpair=part_pair)
    #    for part_pair in cross_rank_trace_pairs(discr, q)
    #)

    def my_facialflux_q_boundary(q):
        #dombain boundary flux
        dir_q = discr.project("vol",BTAG_ALL, q);
        dbf_q = _facial_flux_q(
             discr,
             q_tpair=TracePair(BTAG_ALL,interior=dir_q,exterior=dir_q)
            )
        return(dbf_q)

    dbf_q = obj_array_vectorize(my_facialflux_q_boundary,q)

    #Return the rhs contribution
    return ( discr.inverse_mass( -dflux_q + discr.face_mass(iff_q + pbf_q + dbf_q) ) )
   
