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
)
from meshmode.dof_array import thaw
from mirgecom.eos import IdealSingleGas


class Vortex2D:
    r"""Implements the isentropic vortex after
        - Y.C. Zhou, G.W. Wei / Journal of Computational Physics 189 (2003) 159
          (https://doi.org/10.1016/S0021-9991(03)00206-7)
        - JSH/TW Nodal DG Methods, p. 209
          DOI: 10.1007/978-0-387-72067-8

    The isentropic vortex is defined by:

    .. math::

         u = u_0 - \beta\exp^{(1-r^2)}\frac{y - y_0}{2\pi}\\
         v = v_0 + \beta\exp^{(1-r^2)}\frac{x - x_0}{2\pi}\\
         \rho =
         ( 1 - (\frac{\gamma - 1}{16\gamma\pi^2})\beta^2
         \exp^{2(1-r^2)})^{\frac{1}{\gamma-1}}\\
         p = \rho^{\gamma}

    A call to this object after creation/init creates
    the isentropic vortex solution at a given time (t)
    relative to the configured origin (center) and
    background flow velocity (velocity).
    """

    def __init__(
        self, beta=5, center=[0, 0], velocity=[0, 0],
    ):
        """Initialize vortex parameters

        Parameters
        ----------
        beta : float
            vortex amplitude
        center : 2-dimensional float array
            center of vortex
        velocity : 2-dimensional float array
            fixed flow velocity used for exact solution at t != 0
        """

        self._beta = beta
        self._center = np.array(center)
        self._velocity = np.array(velocity)

    def __call__(self, t, x_vec, eos=IdealSingleGas()):
        vortex_loc = self._center + t * self._velocity

        # coordinates relative to vortex center
        x_rel = x_vec[0] - vortex_loc[0]
        y_rel = x_vec[1] - vortex_loc[1]
        actx = x_vec[0].array_context
        gamma = eos.gamma()
        r = actx.np.sqrt(x_rel ** 2 + y_rel ** 2)
        expterm = self._beta * actx.np.exp(1 - r ** 2)
        u = self._velocity[0] - expterm * y_rel / (2 * np.pi)
        v = self._velocity[1] + expterm * x_rel / (2 * np.pi)
        mass = (1 - (gamma - 1) / (16 * gamma * np.pi ** 2)
                * expterm ** 2) ** (1 / (gamma - 1))
        p = mass ** gamma

        e = p / (gamma - 1) + mass / 2 * (u ** 2 + v ** 2)

        return flat_obj_array(mass, e, mass * u, mass * v)


class SodShock1D:
    r"""Implements a 1D Sod Shock
        - JSH/TW Nodal DG Methods, p. 209

    The Sod Shock setup is defined by:

    .. math::

         {\rho}(x < x_0, 0) = \rho_l\\
         {\rho}(x > x_0, 0) = \rho_r\\
         {\rho}{V_x}(x, 0) = 0
         {\rho}E(x < x_0, 0) = \frac{1}{\gamma - 1}
         {\rho}E(x > x_0, 0) = \frac{.1}{\gamma - 1}

    A call to this object after creation/init creates
    Sod's shock solution at a given time (t)
    relative to the configured origin (center) and
    background flow velocity (velocity).
    """

    def __init__(
            self, dim=2, xdir=0, x0=0.5, rhol=1.0, rhor=0.125, pleft=1.0, pright=0.1,
    ):
        """Initialize shock parameters

        Parameters
        ----------
        dim: integer
           dimension of domain
        x0: float
           location of shock
        rhol: float
           density to left of shock
        rhor: float
           density to right of shock
        pleft: float
           pressure to left of shock
        pright: float
           pressure to right of shock
        """

        self._x0 = x0
        self._rhol = rhol
        self._rhor = rhor
        self._energyl = pleft
        self._energyr = pright
        self._dim = dim
        self._xdir = xdir
        if self._xdir >= self._dim:
            self._xdir = self._dim - 1

    def __call__(self, t, x_vec, eos=IdealSingleGas()):
        gm1 = eos.gamma() - 1.0
        gmn1 = 1.0 / gm1
        x_rel = x_vec[self._xdir]
        actx = x_rel.array_context

        zeros = 0*x_rel

        rhor = zeros + self._rhor
        rhol = zeros + self._rhol
        x0 = zeros + self._x0
        energyl = zeros + gmn1 * self._energyl
        energyr = zeros + gmn1 * self._energyr
        yesno = x_rel > x0
        mass = actx.np.where(yesno, rhor, rhol)
        energy = actx.np.where(yesno, energyr, energyl)
        mom = make_obj_array(
            [
                0*x_rel
                for i in range(self._dim)
            ]
        )

        return flat_obj_array(mass, energy, mom)


class Lump:
    r"""Implements an N-dimensional Gaussian lump of mass.

    The Gaussian lump is defined by:

    .. math::

         {\rho}(r) = {\rho}_{0} + {\rho}_{a}\exp^{(1-r^{2})}\\
         {\rho}\vec{V} = {\rho}(r)\vec{V_0}\\
         {\rho}E = (\frac{p_0}{(\gamma - 1)} + \frac{1}{2}\rho{|V_0|}^2

    Where :math:`V_0` is the fixed velocity specified
    by the user at init time, and :math:`\gamma` is taken
    from the equation-of-state object (eos).

    A call to this object after creation/init creates
    the lump solution at a given time (t)
    relative to the configured origin (center) and
    background flow velocity (velocity).

    This object also functions as a boundary condition
    by providing the "get_boundary_flux" method to
    prescribe exact field values on the given boundary.

    This object also supplies the exact expected RHS
    terms from the analytic expression in the
    "expected_rhs" method.
    """

    def __init__(
            self, numdim=1, rho0=1.0, rhoamp=1.0,
            p0=1.0, center=[0], velocity=[0]
    ):
        r"""Initialize Lump parameters

        Parameters
        ----------
        numdim : integer
            specify the number of dimensions for the lump
        rho0 : float
            specifies the value of :math:`\rho_0`
        rhoamp : float
            specifies the value of :math:`\rho_a`
        p0 : float
            specifies the value of :math:`p_0`
        center : 2-dimensional float array
            center of lump
        velocity : 2-dimensional float array
            fixed flow velocity used for exact solution at t != 0
        """

        if len(center) == numdim:
            self._center = center
        elif len(center) > numdim:
            numdim = len(center)
            self._center = center
        else:
            self._center = np.zeros(shape=(numdim,))

        if len(velocity) == numdim:
            self._velocity = velocity
        elif len(velocity) > numdim:
            numdim = len(velocity)
            self._velocity = velocity
            new_center = np.zeros(shape=(numdim,))
            for i in range(len(self._center)):
                new_center[i] = self._center[i]
            self._center = new_center
        else:
            self._velocity = np.zeros(shape=(numdim,))

        assert len(self._velocity) == numdim
        assert len(self._center) == numdim

        self._p0 = p0
        self._rho0 = rho0
        self._rhoamp = rhoamp
        self._dim = numdim

    def __call__(self, t, x_vec, eos=IdealSingleGas()):
        lump_loc = self._center + t * self._velocity
        assert len(x_vec) == self._dim
        # coordinates relative to lump center
        rel_center = make_obj_array(
            [x_vec[i] - lump_loc[i] for i in range(self._dim)]
        )
        actx = x_vec[0].array_context
        r = actx.np.sqrt(np.dot(rel_center, rel_center))

        gamma = eos.gamma()
        expterm = self._rhoamp * actx.np.exp(1 - r ** 2)
        mass = expterm + self._rho0
        mom = self._velocity * make_obj_array([mass])
        energy = (self._p0 / (gamma - 1.0)) + np.dot(mom, mom) / (2.0 * mass)

        return flat_obj_array(mass, energy, mom)

    def exact_rhs(self, discr, q, t=0.0):
        actx = q[0].array_context
        nodes = thaw(actx, discr.nodes())
        lump_loc = self._center + t * self._velocity
        # coordinates relative to lump center
        rel_center = make_obj_array(
            [nodes[i] - lump_loc[i] for i in range(self._dim)]
        )
        r = actx.np.sqrt(np.dot(rel_center, rel_center))

        # The expected rhs is:
        # rhorhs  = -2*rho*(r.dot.v)
        # rhoerhs = -rho*v^2*(r.dot.v)
        # rhovrhs = -2*rho*(r.dot.v)*v
        expterm = self._rhoamp * actx.np.exp(1 - r ** 2)
        mass = expterm + self._rho0

        v = self._velocity * make_obj_array([1.0 / mass])
        v2 = np.dot(v, v)
        rdotv = np.dot(rel_center, v)
        massrhs = -2 * rdotv * mass
        energyrhs = -v2 * rdotv * mass
        momrhs = v * make_obj_array([-2 * mass * rdotv])

        return flat_obj_array(massrhs, energyrhs, momrhs)


class Uniform:
    r"""Implements initialization to a uniform flow

    A uniform flow is the same everywhere and should have
    a zero RHS.
    """

    def __init__(
            self, numdim=1, rho=1.0, p=1.0, e=2.5, velocity=[0],
    ):
        r"""Initialize uniform flow parameters

        Parameters
        ----------
        numdim : integer
            specify the number of dimensions for the lump
        rho : float
            specifies the density
        p : float
            specifies the pressure
        e : float
            specifies the internal energy
        velocity : float array
            specifies the flow velocity
        """

        if len(velocity) == numdim:
            self._velocity = velocity
        elif len(velocity) > numdim:
            numdim = len(velocity)
            self._velocity = velocity
        else:
            self._velocity = np.zeros(shape=(numdim,))

        assert len(self._velocity) == numdim

        self._p = p
        self._rho = rho
        self._e = e
        self._dim = numdim

    def __call__(self, t, x_vec, eos=IdealSingleGas()):
        gamma = eos.gamma()
        mass = x_vec[0].copy()
        mom = self._velocity * make_obj_array([mass])
        energy = (self._p / (gamma - 1.0)) + np.dot(mom, mom) / (2.0 * mass)

        return flat_obj_array(mass, energy, mom)

    def exact_rhs(self, discr, q, t=0.0):
        actx = q[0].array_context
        nodes = thaw(actx, discr.nodes())
        mass = nodes[0].copy()
        mass[:] = 1.0
        massrhs = 0.0 * mass
        energyrhs = 0.0 * mass
        momrhs = make_obj_array([0 * mass])

        return flat_obj_array(massrhs, energyrhs, momrhs)
