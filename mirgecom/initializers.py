"""
:mod:`mirgecom.initializers` helps intialize and compute flow solution fields.

Solution Initializers
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Vortex2D
.. autoclass:: SodShock1D
.. autoclass:: Lump
.. autoclass:: MulticomponentLump
.. autoclass:: Uniform
.. autoclass:: AcousticPulse
.. automethod: make_pulse
.. autoclass:: MixtureInitializer
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
from pytools.obj_array import make_obj_array
from meshmode.dof_array import thaw
from mirgecom.eos import IdealSingleGas
from mirgecom.euler import split_conserved, join_conserved


def make_pulse(amp, r0, w, r):
    r"""Create a Gaussian pulse.

    The Gaussian pulse is defined by:

    .. math::

        G(\mathbf{r}) =
        a_0*\exp^{-(\frac{(\mathbf{r}-\mathbf{r}_0)}{\sqrt{2}w})^{2}},

    where $\mathbf{r}$ is the position, and the parameters are the pulse amplitude
    $a_0$, the pulse location $\mathbf{r}_0$, and the rms width of the pulse, $w$.

    Parameters
    ----------
    amp: float
        specifies the value of $a_0$, the pulse amplitude
    r0: float array
        specifies the value of $\mathbf{r}_0$, the pulse location
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

         u &= u_0 - \beta\exp^{(1-r^2)}\frac{y - y_0}{2\pi}\\
         v &= v_0 + \beta\exp^{(1-r^2)}\frac{x - x_0}{2\pi}\\
         \rho &=
         ( 1 - (\frac{\gamma - 1}{16\gamma\pi^2})\beta^2
         \exp^{2(1-r^2)})^{\frac{1}{\gamma-1}}\\
         p &= \rho^{\gamma}

    A call to this object after creation/init creates the isentropic
    vortex solution at a given time (t) relative to the configured
    origin (center) and background flow velocity (velocity).

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(
        self, *, beta=5, center=[0, 0], velocity=[0, 0],
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

    def __call__(self, x_vec, *, t=0, eos=IdealSingleGas()):
        """
        Create the isentropic vortex solution at time *t* at locations *x_vec*.

        The solution at time *t* is created by advecting the vortex under the
        assumption of user-supplied constant, uniform velocity
        (``Vortex2D._velocity``).

        Parameters
        ----------
        t: float
            Current time at which the solution is desired.
        x_vec: numpy.ndarray
            Nodal coordinates
        eos: mirgecom.eos.IdealSingleGas
            Equation of state class to supply method for gas *gamma*.
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
        velocity = make_obj_array([u, v])
        mass = (1 - (gamma - 1) / (16 * gamma * np.pi ** 2)
                * expterm ** 2) ** (1 / (gamma - 1))
        momentum = mass * velocity
        p = mass ** gamma

        energy = p / (gamma - 1) + mass / 2 * (u ** 2 + v ** 2)

        return join_conserved(dim=2, mass=mass, energy=energy,
                              momentum=momentum)


class SodShock1D:
    r"""Solution initializer for the 1D Sod Shock.

    This is Sod's 1D shock solution as explained in [Hesthaven_2008]_, Section 5.9
    The Sod Shock setup is defined by:

    .. math::

         {\rho}(x < x_0, 0) &= \rho_l\\
         {\rho}(x > x_0, 0) &= \rho_r\\
         {\rho}{V_x}(x, 0) &= 0\\
         {\rho}E(x < x_0, 0) &= \frac{1}{\gamma - 1}\\
         {\rho}E(x > x_0, 0) &= \frac{.1}{\gamma - 1}

    A call to this object after creation/init creates Sod's shock solution at a
    given time (t) relative to the configured origin (center) and background
    flow velocity (velocity).

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(
            self, *, dim=2, xdir=0, x0=0.5, rhol=1.0,
            rhor=0.125, pleft=1.0, pright=0.1,
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

    def __call__(self, x_vec, *, t=0, eos=IdealSingleGas()):
        """
        Create the 1D Sod's shock solution at locations *x_vec*.

        Parameters
        ----------
        t: float
            Current time at which the solution is desired (unused)
        x_vec: numpy.ndarray
            Nodal coordinates
        eos: :class:`mirgecom.eos.IdealSingleGas`
            Equation of state class with method to supply gas *gamma*.
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

        return join_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=mom)


class Lump:
    r"""Solution initializer for N-dimensional Gaussian lump of mass.

    The Gaussian lump is defined by:

    .. math::

         {\rho} &= {\rho}_{0} + {\rho}_{a}\exp^{(1-r^{2})}\\
         {\rho}\mathbf{V} &= {\rho}\mathbf{V}_0\\
         {\rho}E &= \frac{p_0}{(\gamma - 1)} + \frac{1}{2}\rho{|V_0|}^2,

    where $\mathbf{V}_0$ is the fixed velocity specified by the user at init
    time, and $\gamma$ is taken from the equation-of-state object (eos).

    A call to this object after creation/init creates the lump solution
    at a given time (t) relative to the configured origin (center) and
    background flow velocity (velocity).

    This object also supplies the exact expected RHS terms from the
    analytic expression through :meth:`exact_rhs`.

    .. automethod:: __init__
    .. automethod:: __call__
    .. automethod:: exact_rhs
    """

    def __init__(
            self, *, dim=1, nspecies=0,
            rho0=1.0, rhoamp=1.0, p0=1.0,
            center=None, velocity=None,
    ):
        r"""Initialize Lump parameters.

        Parameters
        ----------
        dim: int
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
        if center is None:
            center = np.zeros(shape=(dim,))
        if velocity is None:
            velocity = np.zeros(shape=(dim,))
        dimmsg = f"is expected to be {dim}-dimensional"
        if center.shape != (dim,):
            raise ValueError(f"Lump center {dimmsg}.")
        if velocity.shape != (dim,):
            raise ValueError(f"Lump velocity {dimmsg}.")

        self._dim = dim
        self._velocity = velocity
        self._center = center
        self._p0 = p0
        self._rho0 = rho0
        self._rhoamp = rhoamp

    def __call__(self, x_vec, *, t=0, eos=IdealSingleGas()):
        """
        Create the lump-of-mass solution at time *t* and locations *x_vec*.

        The solution at time *t* is created by advecting the mass lump under the
        assumption of constant, uniform velocity (``Lump._velocity``).

        Parameters
        ----------
        t: float
            Current time at which the solution is desired
        x_vec: numpy.ndarray
            Nodal coordinates
        eos: :class:`mirgecom.eos.IdealSingleGas`
            Equation of state class with method to supply gas *gamma*.
        """
        if x_vec.shape != (self._dim,):
            raise ValueError(f"Position vector has unexpected dimensionality,"
                             f" expected {self._dim}.")

        amplitude = self._rhoamp
        lump_loc = self._center + t * self._velocity

        # coordinates relative to lump center
        rel_center = make_obj_array(
            [x_vec[i] - lump_loc[i] for i in range(self._dim)]
        )
        actx = x_vec[0].array_context
        r = actx.np.sqrt(np.dot(rel_center, rel_center))
        expterm = amplitude * actx.np.exp(1 - r ** 2)

        mass = expterm + self._rho0
        gamma = eos.gamma()
        mom = self._velocity * mass
        energy = (self._p0 / (gamma - 1.0)) + np.dot(mom, mom) / (2.0 * mass)

        return join_conserved(dim=self._dim, mass=mass, energy=energy, momentum=mom)

    def exact_rhs(self, discr, q, t=0.0):
        """
        Create the RHS for the lump-of-mass solution at time *t*, locations *x_vec*.

        The RHS at time *t* is created by advecting the mass lump under the
        assumption of constant, uniform velocity (``Lump._velocity``).

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

        return join_conserved(dim=self._dim, mass=massrhs, energy=energyrhs,
                              momentum=momrhs)


class MulticomponentLump:
    r"""Solution initializer for multi-component N-dimensional Gaussian lump of mass.

    The Gaussian lump is defined by:

    .. math::

         \rho &= 1.0\\
         {\rho}\mathbf{V} &= {\rho}\mathbf{V}_0\\
         {\rho}E &= \frac{p_0}{(\gamma - 1)} + \frac{1}{2}\rho{|V_0|}^{2}\\
         {\rho~Y_\alpha} &= {\rho~Y_\alpha}_{0}
         + {a_\alpha}{e}^{({c_\alpha}-{r_\alpha})^2},

    where $\mathbf{V}_0$ is the fixed velocity specified by the user at init time,
    and $\gamma$ is taken from the equation-of-state object (eos).

    The user-specified vector of initial values (${{Y}_\alpha}_0$)
    for the mass fraction of each species, *spec_y0s*, and $a_\alpha$ is the
    user-specified vector of amplitudes for each species, *spec_amplitudes*, and
    $c_\alpha$ is the user-specified origin for each species, *spec_centers*.

    A call to this object after creation/init creates the lump solution at a given
    time (*t*) relative to the configured origin (*center*) and background flow
    velocity (*velocity*).

    This object also supplies the exact expected RHS terms from the analytic
    expression via :meth:`exact_rhs`.

    .. automethod:: __init__
    .. automethod:: __call__
    .. automethod:: exact_rhs
    """

    def __init__(
            self, *, dim=1, nspecies=0,
            rho0=1.0, p0=1.0,
            center=None, velocity=None,
            spec_y0s=None, spec_amplitudes=None,
            spec_centers=None
    ):
        r"""Initialize MulticomponentLump parameters.

        Parameters
        ----------
        dim: int
            specify the number of dimensions for the lump
        rho0: float
            specifies the value of $\rho_0$
        p0: float
            specifies the value of $p_0$
        center: numpy.ndarray
            center of lump, shape ``(dim,)``
        velocity: numpy.ndarray
            fixed flow velocity used for exact solution at t != 0,
            shape ``(dim,)``
        """
        if center is None:
            center = np.zeros(shape=(dim,))
        if velocity is None:
            velocity = np.zeros(shape=(dim,))
        if center.shape != (dim,) or velocity.shape != (dim,):
            raise ValueError(f"Expected {dim}-dimensional vector inputs.")

        if nspecies > 0:
            if spec_y0s is None:
                spec_y0s = np.ones(shape=(nspecies,))
            if spec_centers is None:
                spec_centers = make_obj_array([np.zeros(shape=dim,)
                                               for i in range(nspecies)])
            if spec_amplitudes is None:
                spec_amplitudes = np.ones(shape=(nspecies,))
            if len(spec_y0s) != nspecies or\
               len(spec_amplitudes) != nspecies or\
                   len(spec_centers) != nspecies:
                raise ValueError(f"Expected nspecies={nspecies} inputs.")
            for i in range(nspecies):
                if len(spec_centers[i]) != dim:
                    raise ValueError(f"Expected {dim}-dimensional "
                                     f"inputs for spec_centers.")

        self._nspecies = nspecies
        self._dim = dim
        self._velocity = velocity
        self._center = center
        self._p0 = p0
        self._rho0 = rho0
        self._spec_y0s = spec_y0s
        self._spec_centers = spec_centers
        self._spec_amplitudes = spec_amplitudes

    def __call__(self, x_vec, *, t=0, eos=IdealSingleGas()):
        """
        Create a multi-component lump solution at time *t* and locations *x_vec*.

        The solution at time *t* is created by advecting the component mass lump
        at the user-specified constant, uniform velocity
        (``MulticomponentLump._velocity``).

        Parameters
        ----------
        t: float
            Current time at which the solution is desired
        x_vec: numpy.ndarray
            Nodal coordinates
        eos: :class:`mirgecom.eos.IdealSingleGas`
            Equation of state class with method to supply gas *gamma*.
        """
        if x_vec.shape != (self._dim,):
            print(f"len(x_vec) = {len(x_vec)}")
            print(f"self._dim = {self._dim}")
            raise ValueError(f"Expected {self._dim}-dimensional inputs.")

        actx = x_vec[0].array_context

        loc_update = t * self._velocity

        gamma = eos.gamma()
        mass = 0 * x_vec[0] + self._rho0
        mom = self._velocity * mass
        energy = (self._p0 / (gamma - 1.0)) + np.dot(mom, mom) / (2.0 * mass)

        # process the species components independently
        species_mass = np.empty((self._nspecies,), dtype=object)
        for i in range(self._nspecies):
            lump_loc = self._spec_centers[i] + loc_update
            rel_pos = x_vec - lump_loc
            r2 = np.dot(rel_pos, rel_pos)
            expterm = self._spec_amplitudes[i] * actx.np.exp(-r2)
            species_mass[i] = self._rho0 * (self._spec_y0s[i] + expterm)

        return join_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=mom, species_mass=species_mass)

    def exact_rhs(self, discr, q, t=0.0):
        """
        Create a RHS for multi-component lump soln at time *t*, locations *x_vec*.

        The RHS at time *t* is created by advecting the species mass lump at the
        user-specified constant, uniform velocity (``MulticomponentLump._velocity``).

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
        loc_update = t * self._velocity

        mass = 0 * nodes[0] + self._rho0
        mom = self._velocity * mass

        v = mom / mass
        massrhs = 0 * mass
        energyrhs = 0 * mass
        momrhs = 0 * mom

        # process the species components independently
        specrhs = np.empty((self._nspecies,), dtype=object)
        for i in range(self._nspecies):
            lump_loc = self._spec_centers[i] + loc_update
            rel_pos = nodes - lump_loc
            r2 = np.dot(rel_pos, rel_pos)
            expterm = self._spec_amplitudes[i] * actx.np.exp(-r2)
            specrhs[i] = 2 * self._rho0 * expterm * np.dot(rel_pos, v)

        return join_conserved(dim=self._dim, mass=massrhs, energy=energyrhs,
                              momentum=momrhs, species_mass=specrhs)


class AcousticPulse:
    r"""Solution initializer for N-dimensional Gaussian acoustic pulse.

    The Gaussian pulse is defined by:

    .. math::

        {\rho}E(\mathbf{r}) = {\rho}E + a_0 * G(\mathbf{r})\\
        G(\mathbf{r}) = \exp^{-(\frac{(\mathbf{r}-\mathbf{r}_0)}{\sqrt{2}w})^{2}},

    where $\mathbf{r}$ are the nodal coordinates, and $\mathbf{r}_0$,
    $amplitude$, and $w$, are the the user-specified pulse location,
    amplitude, and width, respectively.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, *, dim=1, amplitude=1,
                 center=None, width=1):
        r"""
        Initialize acoustic pulse parameters.

        Parameters
        ----------
        dim: int
            specify the number of dimensions for the pulse
        amplitude: float
            specifies the value of $amplitude$
        width: float
            specifies the rms width of the pulse
        center: numpy.ndarray
            pulse location, shape ``(dim,)``
        """
        if len(center) == dim:
            self._center = center
        elif len(center) > dim:
            dim = len(center)
            self._center = center
        else:
            self._center = np.zeros(shape=(dim,))
        if self._center.shape != (dim,):
            raise ValueError(f"Expected {dim}-dimensional inputs.")

        self._amp = amplitude
        self._width = width
        self._dim = dim

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
        if x_vec.shape != (self._dim,):
            raise ValueError(f"Expected {self._dim}-dimensional inputs.")

        cv = split_conserved(self._dim, q)
        return cv.replace(
            energy=cv.energy + make_pulse(
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
            self, *, dim=1, nspecies=0, rho=1.0, p=1.0, e=2.5,
            velocity=None, mass_fracs=None
    ):
        r"""Initialize uniform flow parameters.

        Parameters
        ----------
        dim: int
            specify the number of dimensions for the flow
        nspecies: int
            specify the number of species in the flow
        rho: float
            specifies the density
        p: float
            specifies the pressure
        e: float
            specifies the internal energy
        velocity: numpy.ndarray
            specifies the flow velocity
        """
        if velocity is not None:
            numvel = len(velocity)
            myvel = velocity
            if numvel > dim:
                dim = numvel
            elif numvel < dim:
                myvel = np.zeros(shape=(dim,))
                for i in range(numvel):
                    myvel[i] = velocity[i]
            self._velocity = myvel
        else:
            self._velocity = np.zeros(shape=(dim,))

        if mass_fracs is not None:
            self._nspecies = len(mass_fracs)
            self._mass_fracs = mass_fracs
        else:
            self._nspecies = nspecies
            self._mass_fracs = np.zeros(shape=(nspecies,))

        if self._velocity.shape != (dim,):
            raise ValueError(f"Expected {dim}-dimensional inputs.")

        self._p = p
        self._rho = rho
        self._e = e
        self._dim = dim

    def __call__(self, x_vec, *, t=0, eos=IdealSingleGas()):
        """
        Create a uniform flow solution at locations *x_vec*.

        Parameters
        ----------
        t: float
            Current time at which the solution is desired (unused)
        x_vec: numpy.ndarray
            Nodal coordinates
        eos: :class:`mirgecom.eos.IdealSingleGas`
            Equation of state class with method to supply gas *gamma*.
        """
        gamma = eos.gamma()
        mass = 0.0 * x_vec[0] + self._rho
        mom = self._velocity * mass
        energy = (self._p / (gamma - 1.0)) + np.dot(mom, mom) / (2.0 * mass)
        species_mass = self._mass_fracs * mass

        from mirgecom.euler import join_conserved
        return join_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=mom, species_mass=species_mass)

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
        momrhs = make_obj_array([0 * mass for i in range(self._dim)])
        yrhs = make_obj_array([0 * mass for i in range(self._nspecies)])

        return join_conserved(dim=self._dim, mass=massrhs, energy=energyrhs,
                              momentum=momrhs, species_mass=yrhs)


class MixtureInitializer:
    r"""Solution initializer for multi-species mixture.

    This initializer creates a physics-consistent mixture solution
    given an initial thermal state (pressure, temperature) and a
    mixture-compatible EOS.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(
            self, *, dim=3, nspecies=0,
            pressure=101500.0, temperature=300.0,
            massfractions=None, velocity=None,
    ):
        r"""Initialize mixture parameters.

        Parameters
        ----------
        dim: int
            specifies the number of dimensions for the solution
        nspeces: int
            specifies the number of mixture species
        pressure: float
            specifies the value of :math:`p_0`
        temperature: float
            specifies the  value of :math:`T_0`
        massfractions: numpy.ndarray
            specifies the mass fraction for each species
        velocity: numpy.ndarray
            fixed uniform flow velocity used for kinetic energy
        """
        if velocity is None:
            velocity = np.zeros(shape=(dim,))
        if massfractions is None:
            if nspecies > 0:
                massfractions = np.zeros(shape=(nspecies,))
        self._nspecies = nspecies
        self._dim = dim
        self._velocity = velocity
        self._pressure = pressure
        self._temperature = temperature
        self._massfracs = massfractions

    def __call__(self, x_vec, eos, *, t=0.0):
        """
        Create the mixture state at locations *x_vec* (t is ignored).

        Parameters
        ----------
        x_vec: numpy.ndarray
            Coordinates at which solution is desired
        eos:
            Mixture-compatible equation-of-state object must provide
            these functions:
            `eos.get_density`
            `eos.get_internal_energy`
        t: float
            Time is ignored by this solution intitializer
        """
        if x_vec.shape != (self._dim,):
            raise ValueError(f"Position vector has unexpected dimensionality,"
                             f" expected {self._dim}.")

        ones = (1.0 + x_vec[0]) - x_vec[0]
        pressure = self._pressure * ones
        temperature = self._temperature * ones
        velocity = make_obj_array([self._velocity[i] * ones
                                   for i in range(self._dim)])
        y = make_obj_array([self._massfracs[i] * ones
                            for i in range(self._nspecies)])
        mass = eos.get_density(pressure, temperature, y)
        specmass = mass * y
        mom = mass * velocity
        internal_energy = eos.get_internal_energy(temperature, y)
        kinetic_energy = 0.5 * np.dot(velocity, velocity)
        energy = mass * (internal_energy + kinetic_energy)

        return join_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=mom, species_mass=specmass)
