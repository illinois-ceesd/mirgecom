r""":mod:`mirgecom.inviscid` provides helper functions for inviscid flow.

Flux Calculation
^^^^^^^^^^^^^^^^

.. autofunction:: inviscid_flux

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
from mirgecom.fluid import make_conserved


def inviscid_flux(discr, eos, cv):
    r"""Compute the inviscid flux vectors from fluid conserved vars *cv*.

    The inviscid fluxes are
    $(\rho\vec{V},(\rho{E}+p)\vec{V},\rho(\vec{V}\otimes\vec{V})
    +p\mathbf{I}, \rho{Y_s}\vec{V})$

    .. note::

        The fluxes are returned as a :class:`mirgecom.fluid.ConservedVars`
        object with a *dim-vector* for each conservation equation. See
        :class:`mirgecom.fluid.ConservedVars` for more information about
        how the fluxes are represented.
    """
    dim = cv.dim
    p = eos.pressure(cv)

    mom = cv.momentum

    return make_conserved(
        dim, mass=mom, energy=mom * (cv.energy + p) / cv.mass,
        momentum=np.outer(mom, mom) / cv.mass + np.eye(dim)*p,
        species_mass=(  # reshaped: (nspecies, dim)
            (mom / cv.mass) * cv.species_mass.reshape(-1, 1)))


def get_inviscid_timestep(discr, eos, cfl, cv):
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


def get_inviscid_cfl(discr, eos, dt, cv):
    """Calculate and return CFL based on current state and timestep."""
    wanted_dt = get_inviscid_timestep(discr, eos=eos, cfl=1.0, cv=cv)
    return dt / wanted_dt
