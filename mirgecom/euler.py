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
import numpy.linalg as la  # noqa
from pytools.obj_array import (
    flat_obj_array,
    make_obj_array,
    with_object_array_or_scalar,
)
import pyopencl.clmath as clmath
import pyopencl.array as clarray
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from mirgecom.eos import IdealSingleGas

from grudge.eager import with_queue
from grudge.symbolic.primitives import TracePair
from dataclasses import dataclass


r"""
This module is designed provide functions and utilities
useful for solving the Euler flow equations.

The Euler flow equations are:

.. :math::

    \partial_t \mathbf{Q} = -\nabla\cdot{\mathbf{F}} +
    (\mathbf{F}\cdot\hat{n})_\partial_{\Omega} + \mathbf{S}

where:
    state :math:`\mathbf{Q} = [\rho, \rho{E}, \rho\vec{V} ]`
    flux :math:`\mathbf{F} = [\rho\vec{V},(\rho{E} + p)\vec{V},
                (\rho(\vec{V}\otimes\vec{V}) + p*\mathbf{I})]`,
    domain boundary :math:`\partial_{\Omega}`,
    sources :math:`mathbf{S} =
                   [{(\partial_t{\rho})}_s, {(\partial_t{\rho{E}})}_s,
                    {(\partial_t{\rho\vec{V}})}_s]`

"""

# from grudge.dt_finding import (
#    dt_geometric_factor,
#    dt_non_geometric_factor,
# )

__doc__ = """
.. autofunction:: inviscid_operator
.. autofunction:: number_of_scalars
.. autofunction:: split_conserved
.. autofunction:: split_species
.. autofunction:: split_fields
.. autofunction:: get_inviscid_timestep
"""


#
# Euler flow eqns:
# d_t(q) + nabla .dot. f = 0 (no sources atm)
# state vector q: [rho rhoE rhoV]
# flux tensor f: [rhoV (rhoE + p)V (rhoV.x.V + p*I)]
#


@dataclass
class ConservedVars:
    r"""
    Class to resolve the canonical conserved quantities,
    (mass, energy, momentum) per unit volume =
    :math:`(\rho,\rhoE,\rho\vec{V})` from an agglomerated
    object array.
    """
    mass: np.ndarray
    """mass per unit volume"""
    energy: np.ndarray
    """energy per unit volume"""
    momentum: np.ndarray
    """momentum vector per unit volume"""


@dataclass
class MassFractions:
    r"""
    Class to pick off the species mass fractions
    (mass fractions) per unit volume =
    :math:`(\rhoY_{\alpha}) | 1 \le \alpha \le N_{species}`,
    from an agglomerated object array. :math:`N_{species}` is
    the number of mixture species.
    """
    massfractions: np.ndarray
    """mass fraction per unit volume for each mixture species"""


def number_of_scalars(ndim, w):
    """
    Return the number of scalars or mixture species in a flow solution.
    """
    return len(w) - (ndim + 2)


def split_conserved(dim, w):
    """
    Return a 'ConservedVars' object that splits out conserved quantities
    by name. Useful for expressive coding.
    """
    return ConservedVars(mass=w[0], energy=w[1], momentum=w[2:2+dim])


def split_species(dim, w):
    """
    Return a 'MassFractions' object that splits out mixture species
    conserved quantities by name. Useful for expressive coding.
    """
    numscalar = number_of_scalars(dim, w)
    sindex = dim + 2
    return MassFractions(massfractions=w[sindex:sindex+numscalar])


def _interior_trace_pair(discr, vec):
    i = discr.interp("vol", "int_faces", vec)
    e = with_object_array_or_scalar(
        lambda el: discr.opposite_face_connection()(el.queue, el), i
    )
    return TracePair("int_faces", i, e)


def _inviscid_flux(discr, q, eos=IdealSingleGas()):
    r"""Computes the inviscid flux vectors from flow solution *q*

    The inviscid fluxes are
    :math:`(\rho\vec{V},(\rhoE+p)\vec{V},\rho(\vec{V}\otimes\vec{V})+p\mathbf{I})
    """
    ndim = discr.dim

    # q = [ rho rhoE rhoV ]
    mass = split_conserved(ndim, q).mass
    energy = split_conserved(ndim, q).energy
    mom = split_conserved(ndim, q).momentum

    p = eos.pressure(q)

    # Fluxes:
    # [ rhoV (rhoE + p)V (rhoV.x.V + p*I) ]
    momflux = make_obj_array(
        [
            (mom[i] * mom[j] / mass + (p if i == j else 0))
            for i in range(ndim)
            for j in range(ndim)
        ]
    )
    massflux = mom * make_obj_array([1.0])
    energyflux = mom * make_obj_array([(energy + p) / mass])

    return flat_obj_array(massflux, energyflux, momflux,)


def _get_wavespeed(dim, w, eos=IdealSingleGas()):
    """Returns the maximum wavespeed in for flow solution *w*"""
    mass = split_conserved(dim, w).mass
    mom = split_conserved(dim, w).momentum

    v = mom * make_obj_array([1.0 / mass])

    sos = eos.sound_speed(w)
    return clmath.sqrt(np.dot(v, v)) + sos


def _facial_flux(discr, w_tpair, eos=IdealSingleGas()):
    """Returns the flux across a face given the solution on both sides *w_tpair*"""
    dim = discr.dim

    mass = split_conserved(dim, w_tpair).mass
    energy = split_conserved(dim, w_tpair).energy
    mom = split_conserved(dim, w_tpair).momentum

    normal = with_queue(mass.int.queue, discr.normal(w_tpair.dd))

    # Get inviscid fluxes [rhoV (rhoE + p)V (rhoV.x.V + p*I) ]
    qint = flat_obj_array(mass.int, energy.int, mom.int)
    qext = flat_obj_array(mass.ext, energy.ext, mom.ext)

    # - Figure out how to manage grudge branch dependencies
    #    qjump = flat_obj_array(rho.jump, rhoE.jump, rhoV.jump)
    qjump = qext - qint
    flux_int = _inviscid_flux(discr, qint, eos)
    flux_ext = _inviscid_flux(discr, qext, eos)

    # Lax/Friedrichs/Rusanov after JSH/TW Nodal DG Methods, p. 209
    flux_jump = (flux_int + flux_ext) * 0.5

    # wavespeeds = [ wavespeed_int, wavespeed_ext ]
    wavespeeds = [_get_wavespeed(dim, qint), _get_wavespeed(dim, qext)]

    lam = clarray.maximum(wavespeeds[0], wavespeeds[1])
    lfr = qjump * make_obj_array([0.5 * lam])

    # Surface fluxes should be inviscid flux .dot. normal
    # rhoV .dot. normal
    # (rhoE + p)V  .dot. normal
    # (rhoV.x.V)_1 .dot. normal
    # (rhoV.x.V)_2 .dot. normal
    nflux = flat_obj_array(
        [
            np.dot(flux_jump[(i * dim): ((i + 1) * dim)], normal)
            for i in range(dim + 2)
        ]
    )

    # add Lax/Friedrichs term
    flux_weak = nflux + lfr

    return discr.interp(w_tpair.dd, "all_faces", flux_weak)


def inviscid_operator(
        discr, w, boundaries, t=0.0, eos=IdealSingleGas(),
):
    """
    RHS of the Euler flow equations
    """

    ndim = discr.dim

    vol_flux = _inviscid_flux(discr, w, eos)
    dflux = flat_obj_array(
        [
            discr.weak_div(vol_flux[(i * ndim): (i + 1) * ndim])
            for i in range(ndim + 2)
        ]
    )

    interior_face_flux = _facial_flux(
        discr, w_tpair=_interior_trace_pair(discr, w), eos=eos
    )

    # Domain boundaries
    domain_boundary_flux = sum(
        #        _facial_flux(
        #            discr,
        #            w_tpair=boundaries[btag].boundary_pair(discr,
        #                                                   w,
        #                                                   t=t,
        #                                                   btag=btag,
        #                                                   eos=eos),
        #            eos=eos
        #        )
        boundaries[btag].get_boundary_flux(discr, w, t=t, btag=btag, eos=eos)
        for btag in boundaries
    )

    return discr.inverse_mass(
        dflux - discr.face_mass(interior_face_flux + domain_boundary_flux)
    )


def get_inviscid_timestep(discr, w, c=1.0, eos=IdealSingleGas()):
    """
    Routine (will) return the maximum stable inviscid timestep. Currently,
    it's a hack.
    """
    dim = discr.dim
    mesh = discr.mesh
    order = max([grp.order for grp in discr.discr_from_dd("vol").groups])
    nelements = mesh.nelements
    nel_1d = nelements ** (1.0 / (1.0 * dim))

    # This roughly reproduces the timestep AK used in wave toy
    dt = (1.0 - 0.25 * (dim - 1)) / (nel_1d * order ** 2)
    return c * dt


#    dt_ngf = dt_non_geometric_factor(discr.mesh)
#    dt_gf  = dt_geometric_factor(discr.mesh)
#    wavespeeds = _get_wavespeed(w,eos=eos)
#    max_v = clmath.max(wavespeeds)
#    return c*dt_ngf*dt_gf/max_v


def split_fields(ndim, w):
    """
    Method to spit out a list of named flow variables in
    an agglomerated flow solution. Useful for specifying
    named data arrays to helper functions (e.g. I/O).
    """
    mass = split_conserved(ndim, w).mass
    energy = split_conserved(ndim, w).energy
    mom = split_conserved(ndim, w).momentum

    retlist = [
        ("mass", mass),
        ("energy", energy),
        ("momentum", mom),
    ]
    nscalar = number_of_scalars(ndim, w)
    if nscalar > 0:
        massfrac = split_species(ndim, w).massfraction
        retlist.append(("massfraction", massfrac))

    return retlist
