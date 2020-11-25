"""
:mod:`mirgecom.initializers` helps intialize and compute flow solution fields.

Solution Initializers
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Vortex2D
.. autoclass:: SodShock1D
.. autoclass:: Lump
.. autoclass:: Uniform
.. autoclass:: AcousticPulse
.. automethod: _make_pulse
.. automethod: _make_uniform_flow
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
from pytools.obj_array import (
    flat_obj_array,
    make_obj_array,
)
from meshmode.dof_array import thaw
from mirgecom.eos import IdealSingleGas
from mirgecom.euler import split_conserved, join_conserved


def _make_uniform_flow(x_vec, mass=1.0, energy=2.5, pressure=1.0,
                       velocity=None, eos=IdealSingleGas()):
    r"""Construct uniform, constant flow.

    Construct a uniform, constant flow from mass, energy, pressure, and
    an EOS object.

    Parameters
    ----------
    x_vec: np.ndarray
        Nodal positions
    mass: float
        Value to set $\rho$
    energy: float
        Optional value to set $\rho{E}$
    pressure: float
        Value to use for calculating $\rho{E}$
    velocity: np.ndarray
        Optional constant velocity to set $\rho\vec{V}$

    Returns
    -------
    q: Object array of DOFArrays
        Agglomerated object array with the uniform flow
    """
    dim = len(x_vec)
    if velocity is None:
        velocity = np.zeros(shape=(dim,))

    _rho = mass
    _p = pressure
    _velocity = velocity
    _gamma = eos.gamma()

    mom0 = _velocity * _rho
    e0 = _p / (_gamma - 1.0)
    ke0 = _rho * 0.5 * np.dot(_velocity, _velocity)

    x_rel = x_vec[0]
    zeros = 0.0*x_rel
    ones = zeros + 1.0

    mass = zeros + _rho
    mom = mom0 * ones
    energy = e0 + ke0 + zeros

    return join_conserved(dim=dim, mass=mass, energy=energy, momentum=mom)


def _make_pulse(amp, r0, w, r):
    r"""Create a Gaussian pulse.

    The Gaussian pulse is defined by:

    .. math::

        G(\vec{r}) = a_0*\exp^{-(\frac{(\vec{r}-\vec{r_0})}{\sqrt{2}w})^{2}}\\

    Where $\vec{r}$ is the position, and the parameters are
    the pulse amplitude $a_0$, the pulse location $\vec{r_0}$, and the
    RMS width of the pulse, $w$.

    Parameters
    ----------
    amp: float
        specifies the value of $\a_0$, the pulse amplitude
    r0: float array
        specifies the value of $\r_0$, the pulse location
    w: float
        specifies the value of $w$, the rms pulse width
    r: Object array of DOFArrays
        specifies the nodal coordinates

    Returns
    -------
    G: float array
        The values of the exponential function
    """
    dim = len(r)
    r_0 = np.zeros(dim)
    r_0 = r_0 + r0
    # coordinates relative to pulse center
    rel_center = make_obj_array(
        [r[i] - r_0[i] for i in range(dim)]
    )
    actx = r[0].array_context
    rms2 = w * w
    r2 = np.dot(rel_center, rel_center) / rms2
    return amp * actx.np.exp(-.5 * r2)


class Vortex2D:
    r"""Initializer for the isentropic vortex solution.

    Implements the isentropic vortex after
        - [Zhou_2003]_
        - [Hesthaven_2008]_, Section 6.6

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

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(
        self, beta=5, center=[0, 0], velocity=[0, 0],
    ):
        """Initialize vortex parameters.

        Parameters
        ----------
        beta: float
            vortex amplitude
        center: numpy.ndarray
            center of vortex, shape ``(2,)``
        velocity: numpy.ndarray
            fixed flow velocity used for exact solution at t != 0, shape ``(2,)``
        """
        self._beta = beta
        self._center = np.array(center)
        self._velocity = np.array(velocity)

    def __call__(self, t, x_vec, eos=IdealSingleGas()):
        """
        Create the isentropic vortex solution at time *t* at locations *x_vec*.

        Parameters
        ----------
        t: float
            Current time at which the solution is desired.
        x_vec: numpy.ndarray
            Nodal coordinates
        eos: :class:`mirgecom.eos.GasEOS`
            Equation of state class to be used in construction of soln (if needed)
        """
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
    r"""Solution initializer for the 1D Sod Shock.

    This is Sod's 1D shock solution as explained in [Hesthaven_2008]_, Section 5.9
    The Sod Shock setup is defined by:

    .. math::

         {\rho}(x < x_0, 0) = \rho_l\\
         {\rho}(x > x_0, 0) = \rho_r\\
         {\rho}{V_x}(x, 0) = 0\\
         {\rho}E(x < x_0, 0) = \frac{1}{\gamma - 1}\\
         {\rho}E(x > x_0, 0) = \frac{.1}{\gamma - 1}

    A call to this object after creation/init creates Sod's shock solution at a
    given time (t) relative to the configured origin (center) and background
    flow velocity (velocity).

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(
            self, dim=2, xdir=0, x0=0.5, rhol=1.0, rhor=0.125, pleft=1.0, pright=0.1,
    ):
        """Initialize shock parameters.

        Parameters
        ----------
        dim: int
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
        """
        Create the 1D Sod's shock solution at locations *x_vec*.

        Parameters
        ----------
        t: float
            Current time at which the solution is desired (unused)
        x_vec: numpy.ndarray
            Nodal coordinates
        eos: :class:`mirgecom.eos.GasEOS`
            Equation of state class to be used in construction of soln (if needed)
        """
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
    r"""Solution initializer for N-dimensional Gaussian lump of mass.

    The Gaussian lump is defined by:

    .. math::

         {\rho}(r) = {\rho}_{0} + {\rho}_{a}\exp^{(1-r^{2})}\\
         {\rho}\vec{V} = {\rho}(r)\vec{V_0}\\
         {\rho}E = (\frac{p_0}{(\gamma - 1)} + \frac{1}{2}\rho{|V_0|}^2

    Where $V_0$ is the fixed velocity specified
    by the user at init time, and $\gamma$ is taken
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

    .. automethod:: __init__
    .. automethod:: __call__
    .. automethod:: exact_rhs
    """

    def __init__(
            self, numdim=1, rho0=1.0, rhoamp=1.0,
            p0=1.0, center=[0], velocity=[0]
    ):
        r"""Initialize Lump parameters.

        Parameters
        ----------
        numdim: int
            specify the number of dimensions for the lump
        rho0: float
            specifies the value of $\rho_0$
        rhoamp: float
            specifies the value of $\rho_a$
        p0: float
            specifies the value of $p_0$
        center: numpy.ndarray
            center of lump, shape ``(2,)``
        velocity: numpy.ndarray
            fixed flow velocity used for exact solution at t != 0,
            shape ``(2,)``
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
        """
        Create the lump-of-mass solution at time *t* and locations *x_vec*.

        Note that *t* is used to advect the mass lump under the assumption of
        constant, and uniform velocity.

        Parameters
        ----------
        t: float
            Current time at which the solution is desired
        x_vec: numpy.ndarray
            Nodal coordinates
        eos: :class:`mirgecom.eos.GasEOS`
            Equation of state class to be used in construction of soln (if needed)
        """
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
        mom = self._velocity * mass
        energy = (self._p0 / (gamma - 1.0)) + np.dot(mom, mom) / (2.0 * mass)

        return flat_obj_array(mass, energy, mom)

    def exact_rhs(self, discr, q, t=0.0):
        """
        Create the RHS for the lump-of-mass solution at time *t*, locations *x_vec*.

        Note that this routine is only useful for testing under the condition of
        uniform, and constant velocity field.

        Parameters
        ----------
        q
            State array which expects at least the canonical conserved quantities
            (mass, energy, momentum) for the fluid at each point.
        t: float
            Time at which RHS is desired
        """
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

        v = self._velocity / mass
        v2 = np.dot(v, v)
        rdotv = np.dot(rel_center, v)
        massrhs = -2 * rdotv * mass
        energyrhs = -v2 * rdotv * mass
        momrhs = v * (-2 * mass * rdotv)

        return flat_obj_array(massrhs, energyrhs, momrhs)


class AcousticPulse:
    r"""Solution initializer for N-dimensional Gaussian acoustic pulse.

    The Gaussian pulse is defined by:

    .. math::

        {\rho}E(\vec{r}) = {\rho}E + a_0 * G(\vec{r})\\
        G(\vec{r}) = \exp^{-(\frac{(\vec{r}-\vec{r_0})}{\sqrt{2}w})^{2}},

    where $\vec{r}$ are the nodal coordinates, and $\vec{r_0}$,
    $amplitude$, and $w$, are the the user-specified pulse location,
    amplitude, and width, respectively.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, numdim=1, amplitude=1,
                 center=None, width=1):
        r"""
        Initialize acoustic pulse parameters.

        Parameters
        ----------
        numdim: int
            specify the number of dimensions for the pulse
        amplitude: float
            specifies the value of $amplitude$
        width: float
            specifies the rms width of the pulse
        center: numpy.ndarray
            pulse location, shape ``(numdim,)``
        """
        if len(center) == numdim:
            self._center = center
        elif len(center) > numdim:
            numdim = len(center)
            self._center = center
        else:
            self._center = np.zeros(shape=(numdim,))

        assert len(self._center) == numdim

        self._amp = amplitude
        self._width = width
        self._dim = numdim

    def __call__(self, x_vec, q, eos=IdealSingleGas()):
        """
        Create the acoustic pulse at locations *x_vec*.

        Parameters
        ----------
        t: float
            Current time at which the solution is desired (unused)
        x_vec: numpy.ndarray
            Nodal coordinates
        eos: :class:`mirgecom.eos.GasEOS`
            Equation of state class to be used in construction of soln (unused)
        """
        assert len(x_vec) == self._dim
        cv = split_conserved(self._dim, q)
        return cv.replace(
            energy=cv.energy + _make_pulse(
                amp=self._amp, w=self._width, r0=self._center, r=x_vec)
            ).join()


class Uniform:
    r"""Solution initializer for a uniform flow.

    A uniform flow is the same everywhere and should have
    a zero RHS.

    .. automethod:: __init__
    .. automethod:: __call__
    .. automethod:: exact_rhs
    """

    def __init__(
            self, numdim=1, rho=1.0, p=1.0, e=2.5, velocity=np.zeros((1,)),
    ):
        r"""Initialize uniform flow parameters.

        Parameters
        ----------
        numdim: int
            specify the number of dimensions for the lump
        rho: float
            specifies the density
        p: float
            specifies the pressure
        e: float
            specifies the internal energy
        velocity: numpy.ndarray
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
        """
        Create a uniform flow solution at locations *x_vec*.

        Parameters
        ----------
        t: float
            Current time at which the solution is desired (unused)
        x_vec: numpy.ndarray
            Nodal coordinates
        eos: :class:`mirgecom.eos.GasEOS`
            Equation of state class to be used in construction of soln (unused)
        """
        return _make_uniform_flow(x_vec=x_vec, mass=self._rho,
                                  pressure=self._p, energy=self._e,
                                  velocity=self._velocity, eos=eos)

    def exact_rhs(self, discr, q, t=0.0):
        """
        Create the RHS for the uniform solution. (Hint - it should be all zero).

        Parameters
        ----------
        q
            State array which expects at least the canonical conserved quantities
            (mass, energy, momentum) for the fluid at each point. (unused)
        t: float
            Time at which RHS is desired (unused)
        """
        actx = q[0].array_context
        nodes = thaw(actx, discr.nodes())
        mass = nodes[0].copy()
        mass[:] = 1.0
        massrhs = 0.0 * mass
        energyrhs = 0.0 * mass
        momrhs = make_obj_array([0 * mass])

        return flat_obj_array(massrhs, energyrhs, momrhs)
