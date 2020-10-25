r""":mod:`mirgecom.euler` helps solve Euler's equations of gas dynamics.

Euler's equations of gas dynamics:

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

RHS Evaluation
^^^^^^^^^^^^^^

.. autofunction:: inviscid_flux
.. autofunction:: inviscid_operator

Time Step Computation
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: get_inviscid_timestep
.. autofunction:: get_inviscid_cfl
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

from dataclasses import dataclass

import numpy as np
from pytools.obj_array import make_obj_array
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import (
    interior_trace_pair,
    cross_rank_trace_pairs
)


@dataclass
class ConservedVars:  # FIXME: Name?
    r"""Resolve the canonical conserved quantities.

    Get the canonical conserved quantities (mass, energy, momentum)
    per unit volume = :math:`(\rho,\rho E,\rho\vec{V})` from an agglomerated
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


def _aux_shape(ary, leading_shape):
    """:arg leading_shape: a tuple with which ``ary.shape`` is expected to begin."""
    from meshmode.dof_array import DOFArray
    if (isinstance(ary, np.ndarray) and ary.dtype == np.object
            and not isinstance(ary, DOFArray)):
        naxes = len(leading_shape)
        if ary.shape[:naxes] != leading_shape:
            raise ValueError("array shape does not start with expected leading "
                    "dimensions")
        return ary.shape[naxes:]
    else:
        if leading_shape != ():
            raise ValueError("array shape does not start with expected leading "
                    "dimensions")
        return ()


def split_conserved(dim, q):
    """Get the canonical conserved quantities.

    Return a :class:`ConservedVars` that is the canonical conserved quantities,
    mass, energy, and momentum from the agglomerated object array extracted
    from the state vector *q*.
    """
    assert len(q) == 2+dim
    return ConservedVars(mass=q[0], energy=q[1], momentum=q[2:2+dim])


def join_conserved(dim, mass, energy, momentum):
    """Create an agglomerated solution array from the conserved quantities."""
    from pytools import single_valued
    aux_shape = single_valued([
        _aux_shape(mass, ()),
        _aux_shape(energy, ()),
        _aux_shape(momentum, (dim,))])

    result = np.zeros((2+dim,) + aux_shape, dtype=object)
    result[0] = mass
    result[1] = energy
    result[2:] = momentum
    return result


def scalar(s):
    """Create an object array for a scalar."""
    return make_obj_array([s])


def inviscid_flux(discr, eos, q):
    r"""Compute the inviscid flux vectors from flow solution *q*.

    The inviscid fluxes are
    :math:`(\rho\vec{V},(\rhoE+p)\vec{V},\rho(\vec{V}\otimes\vec{V})+p\mathbf{I})`
    """
    dim = discr.dim
    cv = split_conserved(dim, q)
    p = eos.pressure(cv)

    mom = cv.momentum
    return join_conserved(dim,
            mass=mom,
            energy=mom * scalar((cv.energy + p) / cv.mass),
            momentum=np.outer(mom, mom)/scalar(cv.mass) + np.eye(dim)*scalar(p))


def _get_wavespeed(dim, eos, cv: ConservedVars):
    """Return the maximum wavespeed in for flow solution *q*."""
    actx = cv.mass.array_context

    v = cv.momentum / scalar(cv.mass)
    return actx.np.sqrt(np.dot(v, v)) + eos.sound_speed(cv)


def _facial_flux(discr, eos, q_tpair, local=False):
    """Return the flux across a face given the solution on both sides *q_tpair*.

    Parameters
    ----------
    eos: mirgecom.eos.GasEOS
        Implementing the pressure and temperature functions for
        returning pressure and temperature as a function of the state q.

    q_tpair
        Trace pair for the face upon which flux calculation is to be performed

    local: boolean
        Indicates whether to skip projection of fluxes to "all_faces" or not.
    """
#    dd
#        Discretization restriction indicating the target discretization to which
#        the fluxes should be projected.
    dim = discr.dim

    actx = q_tpair[0].int.array_context

    flux_int = inviscid_flux(discr, eos, q_tpair.int)
    flux_ext = inviscid_flux(discr, eos, q_tpair.ext)

    # Lax-Friedrichs/Rusanov after JSH/TW Nodal DG Methods, Section 6.6
    # DOI: 10.1007/978-0-387-72067-8
    flux_avg = 0.5*(flux_int + flux_ext)

    lam = actx.np.maximum(
        _get_wavespeed(dim, eos=eos, cv=split_conserved(dim, q_tpair.int)),
        _get_wavespeed(dim, eos=eos, cv=split_conserved(dim, q_tpair.ext))
    )

    normal = thaw(actx, discr.normal(q_tpair.dd))
    flux_weak = (
        flux_avg @ normal
        - scalar(0.5 * lam) * (q_tpair.ext - q_tpair.int))

    if local is False:
        return discr.project(q_tpair.dd, "all_faces", flux_weak)
    return flux_weak


def inviscid_operator(discr, eos, boundaries, q, t=0.0):
    r"""Compute RHS of the Euler flow equations.

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
    """Calculate and return CFL based on current state and timestep."""
    wanted_dt = get_inviscid_timestep(discr, eos=eos, cfl=1.0, q=q)
    return dt / wanted_dt


def get_inviscid_timestep(discr, eos, cfl, q):
    """Routine (will) return the (local) maximum stable inviscid timestep.

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
