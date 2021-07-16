r""":mod:`mirgecom.inviscid` provides helper functions for inviscid flow.

Flux Calculation
^^^^^^^^^^^^^^^^

.. autofunction:: inviscid_flux
.. autofunction:: inviscid_facial_flux

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

import numpy as np
from mirgecom.fluid import (
    split_conserved,
    join_conserved
)
from meshmode.dof_array import thaw
from grudge.symbolic.primitives import TracePair
from mirgecom.flux import lfr_flux


def inviscid_flux(discr, eos, q):
    r"""Compute the inviscid flux vectors from flow solution *q*.

    The inviscid fluxes are
    $(\rho\vec{V},(\rho{E}+p)\vec{V},\rho(\vec{V}\otimes\vec{V})
    +p\mathbf{I}, \rho{Y_s}\vec{V})$

    .. note::

        The fluxes are returned as a 2D object array with shape:
        ``(num_equations, ndim)``.  Each entry in the
        flux array is a :class:`~meshmode.dof_array.DOFArray`.  This
        form and shape for the flux data is required by the built-in
        state data handling mechanism in :mod:`mirgecom.fluid`. That
        mechanism is used by at least
        :class:`mirgecom.fluid.ConservedVars`, and
        :func:`mirgecom.fluid.join_conserved`, and
        :func:`mirgecom.fluid.split_conserved`.
    """
    dim = discr.dim
    cv = split_conserved(dim, q)
    p = eos.pressure(cv)

    mom = cv.momentum

    return join_conserved(dim,
            mass=mom,
            energy=mom * (cv.energy + p) / cv.mass,
            momentum=np.outer(mom, mom) / cv.mass + np.eye(dim)*p,
            species_mass=(  # reshaped: (nspecies, dim)
                (mom / cv.mass) * cv.species_mass.reshape(-1, 1)))


def inviscid_facial_flux(discr, eos, q_tpair, local=False):
    """Return the flux across a face given the solution on both sides *q_tpair*.

    Parameters
    ----------
    eos: mirgecom.eos.GasEOS
        Implementing the pressure and temperature functions for
        returning pressure and temperature as a function of the state q.

    q_tpair: :class:`grudge.sym.TracePair`
        Trace pair for the face upon which flux calculation is to be performed

    local: bool
        Indicates whether to skip projection of fluxes to "all_faces" or not. If
        set to *False* (the default), the returned fluxes are projected to
        "all_faces."  If set to *True*, the returned fluxes are not projected to
        "all_faces"; remaining instead on the boundary restriction.
    """
    actx = q_tpair[0].int.array_context
    dim = discr.dim
    flux_tpair = TracePair(q_tpair.dd,
                           interior=inviscid_flux(discr, eos, q_tpair.int),
                           exterior=inviscid_flux(discr, eos, q_tpair.ext))

    from mirgecom.fluid import compute_wavespeed
    lam = actx.np.maximum(
        compute_wavespeed(dim, eos=eos, cv=split_conserved(dim, q_tpair.int)),
        compute_wavespeed(dim, eos=eos, cv=split_conserved(dim, q_tpair.ext))
    )
    normal = thaw(actx, discr.normal(q_tpair.dd))

    # todo: user-supplied flux routine
    flux_weak = lfr_flux(q_tpair, flux_tpair, normal=normal, lam=lam)

    if local is False:
        return discr.project(q_tpair.dd, "all_faces", flux_weak)
    return flux_weak


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
#    wavespeeds = compute_wavespeed(w,eos=eos)
#    max_v = clmath.max(wavespeeds)
#    return c*dt_ngf*dt_gf/max_v


def get_inviscid_cfl(discr, eos, dt, q):
    """Calculate and return CFL based on current state and timestep."""
    wanted_dt = get_inviscid_timestep(discr, eos=eos, cfl=1.0, q=q)
    return dt / wanted_dt
