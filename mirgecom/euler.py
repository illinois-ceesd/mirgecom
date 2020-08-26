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

from dataclasses import dataclass

import numpy as np
from pytools.obj_array import (
    flat_obj_array,
    make_obj_array,
)
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import (
    interior_trace_pair,
    cross_rank_trace_pairs
)


__doc__ = r"""
:mod:`mirgecom.euler` helps solve Euler's equations of gas dynamics:

.. math::

    \partial_t \mathbf{Q} = -\nabla\cdot{\mathbf{F}} +
    (\mathbf{F}\cdot\hat{n})_{\partial_{\Omega}} + \mathbf{S}

where:

-   state :math:`\mathbf{Q} = [\rho, \rho{E}, \rho\vec{V} ]`
-   flux :math:`\mathbf{F} = [\rho\vec{V},(\rho{E} + p)\vec{V},
    (\rho(\vec{V}\otimes\vec{V}) + p*\mathbf{I})]`,
-   domain boundary :math:`\partial_{\Omega}`,
-   sources :math:`\mathbf{S} = [{(\partial_t{\rho})}_s, {(\partial_t{\rho{E}})}_s,
    {(\partial_t{\rho\vec{V}})}_s]`


State Vector Handling
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ConservedVars
.. autofunction:: split_conserved
.. autofunction:: split_fields

RHS Evaluation
^^^^^^^^^^^^^^

.. autofunction:: inviscid_flux
.. autofunction:: inviscid_operator

Time Step Computation
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: get_inviscid_timestep
.. autofunction:: get_inviscid_cfl
"""


@dataclass
class ConservedVars:
    r"""
    Resolve the canonical conserved quantities,
    (mass, energy, momentum) per unit volume =
    :math:`(\rho,\rho E,\rho\vec{V})` from an agglomerated
    object array.

    .. attribute:: mass

        Mass per unit volume

    .. attribute:: energy

        Energy per unit volume

    .. attribute:: momentum

        Momentum vector per unit volume
    """
    mass: np.ndarray
    energy: np.ndarray
    momentum: np.ndarray


def split_fields(ndim, q):
    """
    Create list of named flow variables in
    an agglomerated flow solution. Useful for specifying
    named data arrays to helper functions (e.g. I/O).
    """
    qs = split_conserved(ndim, q)
    mass = qs.mass
    energy = qs.energy
    mom = qs.momentum

    return [
        ("mass", mass),
        ("energy", energy),
        ("momentum", mom),
    ]


def number_of_equations(ndim, q):
    """
    Return the number of equations (i.e. number of dofs) in the soln
    """
    return len(q)


def split_conserved(dim, q):
    """
    Return a :class:`ConservedVars` that is the canonical conserved quantities,
    mass, energy, and momentum from the agglomerated object array representing
    the state, q.
    """
    return ConservedVars(mass=q[0], energy=q[1], momentum=q[2:2+dim])


def inviscid_flux(discr, eos, q):
    r"""Compute the inviscid flux vectors from flow solution *q*

    The inviscid fluxes are
    :math:`(\rho\vec{V},(\rhoE+p)\vec{V},\rho(\vec{V}\otimes\vec{V})+p\mathbf{I})
    """
    ndim = discr.dim

    # q = [ rho rhoE rhoV ]
    qs = split_conserved(ndim, q)
    mass = qs.mass
    energy = qs.energy
    mom = qs.momentum

    p = eos.get_pressure(q)

    flux = np.zeros((ndim + 2, ndim), dtype=object)
    flux[0] = mom * make_obj_array([1.0])
    flux[1] = mom * make_obj_array([(energy + p) / mass])
    for i in range(ndim):
        for j in range(ndim):
            flux[i+2, j] = (mom[i] * mom[j] / mass + (p if i == j else 0))

    return flux


def _get_wavespeed(dim, eos, q):
    """Return the maximum wavespeed in for flow solution *q*"""
    qs = split_conserved(dim, q)
    mass = qs.mass
    mom = qs.momentum
    actx = mass.array_context

    v = mom * make_obj_array([1.0 / mass])

    sos = eos.get_sound_speed(q)
    return actx.np.sqrt(np.dot(v, v)) + sos


def _facial_flux(discr, eos, q_tpair):
    """Return the flux across a face given the solution on both sides *q_tpair*"""
    dim = discr.dim

    qs = split_conserved(dim, q_tpair)
    mass = qs.mass
    energy = qs.energy
    mom = qs.momentum
    actx = qs.mass.int.array_context

    normal = thaw(actx, discr.normal(q_tpair.dd))

    qint = flat_obj_array(mass.int, energy.int, mom.int)
    qext = flat_obj_array(mass.ext, energy.ext, mom.ext)

    flux_int = inviscid_flux(discr, eos, qint)
    flux_ext = inviscid_flux(discr, eos, qext)

    # Lax-Friedrichs/Rusanov after JSH/TW Nodal DG Methods, p. 209
    # DOI: 10.1007/978-0-387-72067-8
    flux_avg = 0.5*(flux_int + flux_ext)

    lam = actx.np.maximum(
        _get_wavespeed(dim, eos=eos, q=qint),
        _get_wavespeed(dim, eos=eos, q=qext))

    flux_weak = flux_avg @ normal + make_obj_array([0.5 * lam]) * (qext - qint)

    return discr.project(q_tpair.dd, "all_faces", flux_weak)


def inviscid_operator(discr, eos, boundaries, q, t=0.0):
    r"""
    Compute RHS of the Euler flow equations

    Returns
    -------
    The right-hand-side of the Euler flow equations:

    :math:`\dot\mathbf{q} = \mathbf{S} - \nabla\cdot\mathbf{F} +
          (\mathbf{F}\cdot\hat{n})_\partial_{\Omega}`

    Parameters
    ----------
    q
        State array which expects at least the canonical conserved quantities
        (mass, energy, momentum) for the fluid at each point.

    boundaries
        Dictionary of boundary functions, one for each valid btag

    t
        Time

    eos : mirgecom.eos.GasEOS
        Implementing the pressure and temperature functions for
        returning pressure and temperature as a function of the state q.
    """

    vol_flux = inviscid_flux(discr, eos, q)
    dflux = discr.weak_div(vol_flux)

    interior_face_flux = _facial_flux(
        discr, eos=eos, q_tpair=interior_trace_pair(discr, q))

    # Domain boundaries
    domain_boundary_flux = sum(
        _facial_flux(
            discr,
            q_tpair=boundaries[btag].boundary_pair(discr,
                                                   eos=eos,
                                                   btag=btag,
                                                   t=t,
                                                   q=q),
            eos=eos
        )
        for btag in boundaries
    )

    # Flux across partition boundaries
    partition_boundary_flux = sum(
        _facial_flux(discr, eos=eos, q_tpair=part_pair)
        for part_pair in cross_rank_trace_pairs(discr, q)
    )

    return discr.inverse_mass(
        dflux - discr.face_mass(interior_face_flux + domain_boundary_flux
                                + partition_boundary_flux)
    )


def get_inviscid_cfl(discr, eos, dt, q):
    """
    Calculate and return CFL based on current state and timestep
    """
    wanted_dt = get_inviscid_timestep(discr, eos=eos, cfl=1.0, q=q)
    return dt / wanted_dt


def get_inviscid_timestep(discr, eos, cfl, q):
    """
    Routine (will) return the (local) maximum stable inviscid timestep.
    Currently, it's a hack waiting for the geometric_factor helpers port
    from grudge.
    """
    dim = discr.dim
    mesh = discr.mesh
    order = max([grp.order for grp in discr.discr_from_dd("vol").groups])
    nelements = mesh.nelements
    nel_1d = nelements ** (1.0 / (1.0 * dim))

    # This roughly reproduces the timestep AK used in wave toy
    dt = (1.0 - 0.25 * (dim - 1)) / (nel_1d * order ** 2)
    return cfl * dt

#    dt_ngf = dt_non_geometric_factor(discr.mesh)
#    dt_gf  = dt_geometric_factor(discr.mesh)
#    wavespeeds = _get_wavespeed(w,eos=eos)
#    max_v = clmath.max(wavespeeds)
#    return c*dt_ngf*dt_gf/max_v
