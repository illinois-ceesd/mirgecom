"""
:mod:`mirgecom.initializers` helps intialize and compute flow solution fields.

Solution Initializers
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Vortex2D
.. autoclass:: SodShock1D
.. autoclass:: DoubleMachReflection
.. autoclass:: Lump
.. autoclass:: MulticomponentLump
.. autoclass:: MulticomponentTrig
.. autoclass:: AcousticPulse
.. autoclass:: Uniform
.. autoclass:: MixtureInitializer
.. autoclass:: PlanarDiscontinuity
.. autoclass:: PlanarPoiseuille
.. autoclass:: ShearFlow
.. autoclass:: InviscidTaylorGreenVortex

State Initializers
^^^^^^^^^^^^^^^^^^
.. autofunction:: initialize_flow_solution

Initialization Utilities
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: make_pulse
"""

__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
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
from mirgecom.eos import IdealSingleGas
from mirgecom.fluid import make_conserved


def initialize_flow_solution(actx, coords, gas_model=None, eos=None,
        pressure=None, temperature=None, density=None, velocity=None,
        species_mass_fractions=None):
    """Create a fluid CV from a set of minimal input data."""
    state_spec = [pressure is None, temperature is None, density is None]
    if sum(state_spec) != 1:
        raise ValueError("Must provide 2 of (pressure, temperature, density).")

    dim = coords.shape[0]

    zeros = coords[0]*0.0

    if velocity is None:
        velocity = np.zeros(dim,)

    if pressure is None:
        pressure = eos.get_pressure(density, temperature, species_mass_fractions)

    if temperature is None:
        gas_const = eos.gas_const(species_mass_fractions=species_mass_fractions)
        temperature = pressure/(density*gas_const)

    if density is None:
        density = eos.get_density(pressure, temperature, species_mass_fractions)

    momentum = density*velocity
    energy = density*(eos.get_internal_energy(temperature, species_mass_fractions)
                      + 0.5*np.dot(velocity, velocity))

    if species_mass_fractions is None:
        species_mass = None
    else:
        nspecies = len(species_mass_fractions)
        species_mass = make_obj_array([density*species_mass_fractions[i] + zeros
                                       for i in range(nspecies)])

    return make_conserved(dim=dim, mass=density + zeros,
                          energy=energy + zeros,
                          momentum=momentum + zeros,
                          species_mass=species_mass)


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
    r0: numpy.ndarray
        specifies the value of $\mathbf{r}_0$, the pulse location
    w: float
        specifies the value of $w$, the rms pulse width
    r: numpy.ndarray
        specifies the nodal coordinates

    Returns
    -------
    G: numpy.ndarray
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

    def __init__(self, *, beta=5, center=(0, 0), velocity=(0, 0)):
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

    def __call__(self, x_vec, *, time=0, eos=None, **kwargs):
        """
        Create the isentropic vortex solution at time *t* at locations *x_vec*.

        The solution at time *t* is created by advecting the vortex under the
        assumption of user-supplied constant, uniform velocity
        (``Vortex2D._velocity``).

        Parameters
        ----------
        time: float
            Current time at which the solution is desired.
        x_vec: numpy.ndarray
            Nodal coordinates
        eos: mirgecom.eos.IdealSingleGas
            Equation of state class to supply method for gas *gamma*.
        """
        t = time
        if eos is None:
            eos = IdealSingleGas()
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

        return make_conserved(dim=2, mass=mass, energy=energy,
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

    def __init__(self, *, dim=1, xdir=0, x0=0.5, rhol=1.0,
                 rhor=0.125, pleft=1.0, pright=0.1):
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
        self._pl = pleft
        self._pr = pright
        self._dim = dim
        self._xdir = xdir
        if self._xdir >= self._dim:
            self._xdir = self._dim - 1

    def __call__(self, x_vec, *, eos=None, **kwargs):
        """Create the 1D Sod's shock solution at locations *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Nodal coordinates
        eos: :class:`mirgecom.eos.IdealSingleGas`
            Equation of state class with method to supply gas *gamma*.
        """
        if eos is None:
            eos = IdealSingleGas()

        gm1 = eos.gamma() - 1.0
        gmn1 = 1.0 / gm1
        x = x_vec[self._xdir]
        actx = x.array_context

        zeros = actx.np.zeros_like(x)

        rhor = zeros + self._rhor
        rhol = zeros + self._rhol
        x0 = zeros + self._x0
        energyl = zeros + gmn1 * self._pl
        energyr = zeros + gmn1 * self._pr

        sigma = 1e-13
        weight = 0.5 * (1.0 - actx.np.tanh(1.0/sigma * (x - x0)))

        mass = rhor + (rhol - rhor)*weight
        energy = energyr + (energyl - energyr)*weight
        momentum = make_obj_array([1.*zeros for _ in range(self._dim)])

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=momentum)


class DoubleMachReflection:
    r"""Implement the double shock reflection problem.

    The double shock reflection solution is crafted after [Woodward_1984]_
    and is defined by:

    .. math::

        {\rho}(x < x_s(y,t)) &= \gamma \rho_j\\
        {\rho}(x > x_s(y,t)) &= \gamma\\
        {\rho}{V_x}(x < x_s(y,t)) &= u_p \cos(\pi/6)\\
        {\rho}{V_x}(x > x_s(y,t)) &= 0\\
        {\rho}{V_y}(x > x_s(y,t)) &= u_p \sin(\pi/6)\\
        {\rho}{V_y}(x > x_s(y,t)) &= 0\\
        {\rho}E(x < x_s(y,t)) &= (\gamma-1)p_j\\
        {\rho}E(x > x_s(y,t)) &= (\gamma-1),

    where the shock position is given,

    .. math::

        x_s = x_0 + y/\sqrt{3} + 2 u_s t/\sqrt{3}

    and the normal shock jump relations are

    .. math::

        \rho_j &= \frac{(\gamma + 1) u_s^2}{(\gamma-1) u_s^2 + 2} \\
        p_j &= \frac{2 \gamma u_s^2 - (\gamma - 1)}{\gamma+1} \\
        u_p &= 2 \frac{u_s^2-1}{(\gamma+1) u_s}

    The initial shock location is given by $x_0$ and $u_s$ is the shock speed.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, shock_location=1.0/6.0, shock_speed=4.0, shock_sigma=0.05):
        """Initialize double shock reflection parameters.

        Parameters
        ----------
        shock_location: float
           initial location of shock
        shock_speed: float
           shock speed, Mach number
        shock_sigma: float
           initial condition sharpness
        """
        self._shock_location = shock_location
        self._shock_speed = shock_speed
        self._shock_sigma = shock_sigma

    def __call__(self, x_vec, *, eos=None, time=0, **kwargs):
        r"""Create double mach reflection solution at locations *x_vec*.

        At times $t > 0$, calls to this routine create an advanced solution
        under the assumption of constant normal shock speed *shock_speed*.
        The advanced solution *is not* the exact solution, but is appropriate
        for use as a boundary solution on the top and upstream (left)
        side of the domain.

        Parameters
        ----------
        time: float
            Time at which to compute the solution
        x_vec: numpy.ndarray
            Nodal coordinates
        eos: :class:`mirgecom.eos.GasEOS`
            Equation of state class to be used in construction of soln (if needed)
        """
        t = time
        # Fail if numdim is other than 2
        if len(x_vec) != 2:
            raise ValueError("Case only defined for 2 dimensions")
        if eos is None:
            eos = IdealSingleGas()

        gm1 = eos.gamma() - 1.0
        gp1 = eos.gamma() + 1.0
        gmn1 = 1.0 / gm1
        x_rel = x_vec[0]
        y_rel = x_vec[1]
        actx = x_rel.array_context

        # Normal Shock Relations
        shock_speed_2 = self._shock_speed * self._shock_speed
        rho_jump = gp1 * shock_speed_2 / (gm1 * shock_speed_2 + 2.)
        p_jump = (2. * eos.gamma() * shock_speed_2 - gm1) / gp1
        up = 2. * (shock_speed_2 - 1.) / (gp1 * self._shock_speed)

        rhol = eos.gamma() * rho_jump
        rhor = eos.gamma()
        ul = up * np.cos(np.pi/6.0)
        ur = 0.0
        vl = - up * np.sin(np.pi/6.0)
        vr = 0.0
        rhoel = gmn1 * p_jump
        rhoer = gmn1 * 1.0

        xinter = (self._shock_location + y_rel/np.sqrt(3.0)
                  + 2.0*self._shock_speed*t/np.sqrt(3.0))
        sigma = self._shock_sigma
        xtanh = 1.0/sigma*(x_rel-xinter)
        mass = rhol/2.0*(actx.np.tanh(-xtanh)+1.0)+rhor/2.0*(actx.np.tanh(xtanh)+1.0)
        rhoe = (rhoel/2.0*(actx.np.tanh(-xtanh)+1.0)
                + rhoer/2.0*(actx.np.tanh(xtanh)+1.0))
        u = ul/2.0*(actx.np.tanh(-xtanh)+1.0)+ur/2.0*(actx.np.tanh(xtanh)+1.0)
        v = vl/2.0*(actx.np.tanh(-xtanh)+1.0)+vr/2.0*(actx.np.tanh(xtanh)+1.0)

        vel = make_obj_array([u, v])
        mom = mass * vel
        energy = rhoe + .5*mass*np.dot(vel, vel)

        return make_conserved(dim=2, mass=mass, energy=energy, momentum=mom)


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

    def __init__(self, *, dim=1, rho0=1.0, rhoamp=1.0, p0=1.0,
                 center=None, velocity=None):
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

    def __call__(self, x_vec, *, eos=None, time=0, **kwargs):
        """Create the lump-of-mass solution at time *t* and locations *x_vec*.

        The solution at time *t* is created by advecting the mass lump under the
        assumption of constant, uniform velocity (``Lump._velocity``).

        Parameters
        ----------
        time: float
            Current time at which the solution is desired
        x_vec: numpy.ndarray
            Nodal coordinates
        eos: :class:`mirgecom.eos.IdealSingleGas`
            Equation of state class with method to supply gas *gamma*.
        """
        t = time
        if eos is None:
            eos = IdealSingleGas()
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

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=mom)

    def exact_rhs(self, dcoll, cv, time=0.0):
        """
        Create the RHS for the lump-of-mass solution at time *t*, locations *x_vec*.

        The RHS at time *t* is created by advecting the mass lump under the
        assumption of constant, uniform velocity (``Lump._velocity``).

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            Array with the conserved quantities
        time: float
            Time at which RHS is desired
        """
        t = time
        actx = cv.array_context
        nodes = actx.thaw(dcoll.nodes())
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

        return make_conserved(dim=self._dim, mass=massrhs, energy=energyrhs,
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
            self, *, dim=1, nspecies=0, rho0=1.0, p0=1.0, center=None, velocity=None,
            spec_y0s=None, spec_amplitudes=None, spec_centers=None, sigma=1.0):
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
        sigma: float
            std deviation of the gaussian
        """
        if center is None:
            center = np.zeros(shape=(dim,))
        if velocity is None:
            velocity = np.zeros(shape=(dim,))
        if center.shape != (dim,) or velocity.shape != (dim,):
            raise ValueError(f"Expected {dim}-dimensional vector inputs.")

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
        self._sigma = sigma

    def __call__(self, x_vec, *, eos=None, time=0, **kwargs):
        """
        Create a multi-component lump solution at time *t* and locations *x_vec*.

        The solution at time *t* is created by advecting the component mass lump
        at the user-specified constant, uniform velocity
        (``MulticomponentLump._velocity``).

        Parameters
        ----------
        time: float
            Current time at which the solution is desired
        x_vec: numpy.ndarray
            Nodal coordinates
        eos: :class:`mirgecom.eos.IdealSingleGas`
            Equation of state class with method to supply gas *gamma*.
        """
        t = time
        if eos is None:
            eos = IdealSingleGas()
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
            r2 = np.dot(rel_pos, rel_pos)/(self._sigma**2)
            expterm = self._spec_amplitudes[i] * actx.np.exp(-0.5*r2)
            species_mass[i] = self._rho0 * (self._spec_y0s[i] + expterm)

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=mom, species_mass=species_mass)

    def exact_rhs(self, dcoll, cv, time=0.0):
        """
        Create a RHS for multi-component lump soln at time *t*, locations *x_vec*.

        The RHS at time *t* is created by advecting the species mass lump at the
        user-specified constant, uniform velocity (``MulticomponentLump._velocity``).

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            Array with the conserved quantities
        time: float
            Time at which RHS is desired
        """
        t = time
        actx = cv.array_context
        nodes = actx.thaw(dcoll.nodes())
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
            r2 = np.dot(rel_pos, rel_pos)/self._sigma**2
            expterm = self._spec_amplitudes[i] * actx.np.exp(-0.5*r2)
            specrhs[i] = self._rho0 * expterm * np.dot(rel_pos, v) / self._sigma**2

        return make_conserved(dim=self._dim, mass=massrhs, energy=energyrhs,
                              momentum=momrhs, species_mass=specrhs)


class MulticomponentTrig:
    r"""Initializer for trig-distributed species fractions.

    The trig lump is defined by:

    .. math::

         \rho &= 1.0\\
         {\rho}\mathbf{V} &= {\rho}\mathbf{V}_0\\
         {\rho}E &= \frac{p_0}{(\gamma - 1)} + \frac{1}{2}\rho{|V_0|}^{2}\\
         {\rho~Y_\alpha} &= {\rho~Y_\alpha}_{0}
         + {a_\alpha}\sin{\omega(\mathbf{r} - \mathbf{v}t)},

    where $\mathbf{V}_0$ is the fixed velocity specified by the user at init time,
    and $\gamma$ is taken from the equation-of-state object (eos).

    The user-specified vector of initial values (${{Y}_\alpha}_0$)
    for the mass fraction of each species, *spec_y0s*, and $a_\alpha$ is the
    user-specified vector of amplitudes for each species, *spec_amplitudes*, and
    $c_\alpha$ is the user-specified origin for each species, *spec_centers*.

    A call to this object after creation/init creates the trig solution at a given
    time (*t*) relative to the configured origin (*center*), wave_vector k,  and
    background flow velocity (*velocity*).

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, *, dim=1, nspecies=0,
                 rho0=1.0, p0=1.0, gamma=1.4, center=None, velocity=None,
                 spec_y0s=None, spec_amplitudes=None, spec_centers=None,
                 spec_omegas=None, spec_diffusivities=None,
                 wave_vector=None, trig_function=None):
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
        wave_vector: numpy.ndarray
            optional fixed vector indicating normal direction of wave
        trig_function
            callable trig function
        """
        if center is None:
            center = np.zeros(shape=(dim,))
        if velocity is None:
            velocity = np.zeros(shape=(dim,))
        if center.shape != (dim,) or velocity.shape != (dim,):
            raise ValueError(f"Expected {dim}-dimensional vector inputs.")
        if spec_y0s is None:
            spec_y0s = 2.0*np.ones(shape=(nspecies,))
        if spec_centers is None:
            spec_centers = make_obj_array([np.zeros(shape=dim,)
                                           for i in range(nspecies)])
        if spec_omegas is None:
            spec_omegas = 2.*np.pi*np.ones(shape=(nspecies,))

        if spec_amplitudes is None:
            spec_amplitudes = np.ones(shape=(nspecies,))

        if spec_diffusivities is None:
            spec_diffusivities = np.ones(shape=(nspecies,))

        if wave_vector is None:
            wave_vector = np.zeros(shape=(dim,))
            wave_vector[0] = 1

        import mirgecom.math as mm
        if trig_function is None:
            trig_function = mm.sin

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
        self._spec_amps = spec_amplitudes
        self._gamma = gamma
        self._spec_omegas = spec_omegas
        self._d = spec_diffusivities
        self._wave_vector = wave_vector
        self._trig_func = trig_function

    def __call__(self, x_vec, *, eos=None, time=0, **kwargs):
        """
        Create a multi-component lump solution at time *t* and locations *x_vec*.

        The solution at time *t* is created by advecting the component mass lump
        at the user-specified constant, uniform velocity
        (``MulticomponentLump._velocity``).

        Parameters
        ----------
        time: float
            Current time at which the solution is desired
        x_vec: numpy.ndarray
            Nodal coordinates
        eos: :class:`mirgecom.eos.IdealSingleGas`
            Equation of state class with method to supply gas *gamma*.
        """
        t = time
        if x_vec.shape != (self._dim,):
            print(f"len(x_vec) = {len(x_vec)}")
            print(f"self._dim = {self._dim}")
            raise ValueError(f"Expected {self._dim}-dimensional inputs.")
        # actx = x_vec[0].array_context
        mass = 0 * x_vec[0] + self._rho0
        mom = self._velocity * mass
        energy = ((self._p0 / (self._gamma - 1.0))
                  + 0.5*mass*np.dot(self._velocity, self._velocity))

        vel_t = t * self._velocity
        import mirgecom.math as mm
        spec_mass = np.empty((self._nspecies,), dtype=object)
        for i in range(self._nspecies):
            spec_x = x_vec - self._spec_centers[i]
            wave_r = spec_x - vel_t
            wave_x = np.dot(wave_r, self._wave_vector)
            expterm = mm.exp(-t*self._d[i]*self._spec_omegas[i]**2)
            trigterm = self._trig_func(self._spec_omegas[i]*wave_x)
            spec_y = self._spec_y0s[i] + self._spec_amps[i]*expterm*trigterm
            spec_mass[i] = mass * spec_y

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=mom, species_mass=spec_mass)


class AcousticPulse:
    r"""Solution initializer for N-dimensional isentropic Gaussian acoustic pulse.

    The Gaussian pulse is defined by:

    .. math::

        q(\mathbf{r}) = q_0 + a_0 * G(\mathbf{r})\\
        G(\mathbf{r}) = \exp^{-(\frac{(\mathbf{r}-\mathbf{r}_0)}{\sqrt{2}w})^{2}},

    where $\mathbf{r}$ are the nodal coordinates, and $\mathbf{r}_0$,
    $amplitude$, and $w$, are the the user-specified pulse location,
    amplitude, and width, respectively.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, dim, *, amplitude=1, width=1, center=None):
        r"""Initialize acoustic pulse parameters.

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

    def __call__(self, x_vec, cv, eos=None, tseed=None, **kwargs):
        """Create the acoustic pulse at locations *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Nodal coordinates
        eos: :class:`mirgecom.eos.GasEOS`
            Equation of state class to be used in construction of soln
        """
        if eos is None:
            eos = IdealSingleGas()
        if x_vec.shape != (self._dim,):
            raise ValueError(f"Expected {self._dim}-dimensional inputs.")

        temperature = eos.temperature(cv, tseed)
        gamma = eos.gamma(cv, temperature)

        y = cv.species_mass_fractions

        ref_pressure = eos.pressure(cv, temperature)
        pressure = ref_pressure + \
            make_pulse(amp=self._amp, w=self._width, r0=self._center, r=x_vec)

        # isentropic relations
        mass = cv.mass*(pressure/ref_pressure)**(1.0/gamma)
        temperature = temperature*(pressure/ref_pressure)**(1.0 - 1.0/gamma)

        energy = mass*(
            eos.get_internal_energy(temperature, y)
            + 0.5*np.dot(cv.velocity, cv.velocity))

        return make_conserved(dim=self._dim,
                              mass=mass,
                              energy=energy,
                              momentum=cv.velocity*mass,
                              species_mass=y*mass)


class Uniform:
    r"""Solution initializer for a uniform flow.

    A uniform flow is the same everywhere and should have a zero RHS.

    .. automethod:: __init__
    .. automethod:: __call__
    .. automethod:: exact_rhs
    """

    def __init__(self, *, dim=1, nspecies=0, rho=None, pressure=None, energy=None,
                 velocity=None, temperature=None, species_mass_fractions=None):
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

        if species_mass_fractions is not None:
            self._nspecies = len(species_mass_fractions)
            self._mass_fracs = species_mass_fractions
        else:
            self._nspecies = nspecies
            self._mass_fracs = np.zeros(shape=(nspecies,))

        if self._velocity.shape != (dim,):
            raise ValueError(f"Expected {dim}-dimensional inputs.")

        self._temp = temperature
        self._p = pressure
        self._rho = rho
        self._e = energy  # XXX Unused. Deprecate this?
        self._dim = dim

    def __call__(self, x_vec, eos, **kwargs):
        """Create a uniform flow solution at locations *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Nodal coordinates
        eos: :class:`mirgecom.eos.IdealSingleGas`
            Equation of state class with method to supply gas *gamma*.

        Returns
        -------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            Fluid solution
        """
        actx = x_vec[0].array_context
        return initialize_flow_solution(
            actx, coords=x_vec, eos=eos, pressure=self._p, density=self._rho,
            velocity=self._velocity, temperature=self._temp,
            species_mass_fractions=self._mass_fracs)

    def exact_rhs(self, dcoll, cv, time=0.0):
        """Create the RHS for the uniform solution. (Hint - it should be all zero).

        Parameters
        ----------
        cv: :class:`~mirgecom.fluid.ConservedVars`
            Fluid solution
        t: float
            Time at which RHS is desired (unused)
        """
        actx = cv.array_context
        nodes = actx.thaw(dcoll.nodes())
        mass = nodes[0].copy()
        mass[:] = 1.0
        massrhs = 0.0 * mass
        energyrhs = 0.0 * mass
        momrhs = make_obj_array([0 * mass for i in range(self._dim)])
        yrhs = make_obj_array([0 * mass for i in range(self._nspecies)])

        return make_conserved(dim=self._dim, mass=massrhs, energy=energyrhs,
                              momentum=momrhs, species_mass=yrhs)


class MixtureInitializer:
    r"""Solution initializer for multi-species mixture.

    This initializer creates a physics-consistent mixture solution
    given an initial thermal state (pressure, temperature) and a
    mixture-compatible EOS.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, dim, *, pressure=101325.0, temperature=300.0,
                 species_mass_fractions=None, velocity=None):
        r"""Initialize mixture parameters.

        Parameters
        ----------
        dim: int
            specifies the number of dimensions for the solution
        pressure: float
            specifies the value of :math:`p_0`
        temperature: float
            specifies the  value of :math:`T_0`
        species_mass_fractions: numpy.ndarray
            specifies the mass fraction for each species
        velocity: numpy.ndarray
            fixed uniform flow velocity used for kinetic energy
        """
        if velocity is None:
            velocity = np.zeros(shape=(dim,))
        self._dim = dim
        self._velocity = velocity
        self._pressure = pressure
        self._temperature = temperature
        self._massfracs = species_mass_fractions

        from warnings import warn
        warn("MixtureInitializer is deprecated and will disappear in Q4 2023.",
             DeprecationWarning, stacklevel=2)

    def __call__(self, x_vec, eos, **kwargs):
        """Create the mixture state at locations *x_vec* (t is ignored).

        Parameters
        ----------
        x_vec: numpy.ndarray
            Coordinates at which solution is desired
        eos:
            Mixture-compatible equation-of-state object must provide
            these functions:
            `eos.get_density`
            `eos.get_internal_energy`
        """
        if x_vec.shape != (self._dim,):
            raise ValueError(f"Position vector has unexpected dimensionality,"
                             f" expected {self._dim}.")

        actx = x_vec[0].array_context
        return initialize_flow_solution(
            actx, coords=x_vec, eos=eos, pressure=self._pressure,
            temperature=self._temperature, velocity=self._velocity,
            species_mass_fractions=self._massfracs)


class PlanarDiscontinuity:
    r"""Solution initializer for flow with a discontinuity.

    This initializer creates a physics-consistent flow solution
    given an initial thermal state (pressure, temperature) and an EOS.

    The solution varies across a planar interface defined by a tanh function
    located at disc_location with normal normal_dir
    for pressure, temperature, velocity, and mass fraction

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, *, dim=3,
                 temperature_left, temperature_right,
                 pressure_left, pressure_right,
                 normal_dir=None, disc_location=None, nspecies=0,
                 velocity_left=None, velocity_right=None,
                 species_mass_left=None, species_mass_right=None,
                 convective_velocity=None, sigma=0.5):
        r"""Initialize mixture parameters.

        Parameters
        ----------
        dim: int
            specifies the number of dimensions for the solution
        normal_dir: numpy.ndarray
            specifies the direction (plane) the discontinuity is applied in
        disc_location: numpy.ndarray or Callable
            fixed location of discontinuity or optionally a function that
            returns the time-dependent location.
        nspecies: int
            specifies the number of mixture species
        pressure_left: float
            pressure to the left of the discontinuity
        temperature_left: float
            temperature to the left of the discontinuity
        velocity_left: numpy.ndarray
            velocity (vector) to the left of the discontinuity
        species_mass_left: numpy.ndarray
            species mass fractions to the left of the discontinuity
        pressure_right: float
            pressure to the right of the discontinuity
        temperature_right: float
            temperaure to the right of the discontinuity
        velocity_right: numpy.ndarray
            velocity (vector) to the right of the discontinuity
        species_mass_right: numpy.ndarray
            species mass fractions to the right of the discontinuity
        sigma: float
           sharpness parameter
        """
        if velocity_left is None:
            velocity_left = np.zeros(shape=(dim,))
        if velocity_right is None:
            velocity_right = np.zeros(shape=(dim,))

        if species_mass_left is None:
            species_mass_left = np.zeros(shape=(nspecies,))
        if species_mass_right is None:
            species_mass_right = np.zeros(shape=(nspecies,))

        if normal_dir is None:
            normal_dir = np.zeros(shape=(dim,))
            normal_dir[0] = 1.

        if disc_location is None:
            disc_location = np.zeros(shape=(dim,))

        self._nspecies = nspecies
        self._dim = dim
        self._disc_location = disc_location
        self._sigma = sigma
        self._ul = velocity_left
        self._ur = velocity_right
        self._uc = convective_velocity
        self._pl = pressure_left
        self._pr = pressure_right
        self._tl = temperature_left
        self._tr = temperature_right
        self._yl = species_mass_left
        self._yr = species_mass_right
        self._normal = normal_dir

    def __call__(self, x_vec, eos, *, time=0.0):
        """Create the mixture state at locations *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Coordinates at which solution is desired
        eos:
            Mixture-compatible equation-of-state object must provide
            these functions:
            `eos.get_density`
            `eos.get_internal_energy`
        time: float
            Time at which solution is desired. The location is (optionally)
            dependent on time
        """
        if x_vec.shape != (self._dim,):
            raise ValueError(f"Position vector has unexpected dimensionality,"
                             f" expected {self._dim}.")

        x = x_vec[0]
        actx = x.array_context
        if callable(self._disc_location):
            x0 = self._disc_location(time)
        else:
            x0 = self._disc_location

        dist = np.dot(x0 - x_vec, self._normal)
        xtanh = 1.0/self._sigma*dist
        weight = 0.5*(1.0 - actx.np.tanh(xtanh))
        pressure = self._pl + (self._pr - self._pl)*weight
        temperature = self._tl + (self._tr - self._tl)*weight
        velocity = self._ul + (self._ur - self._ul)*weight
        y = self._yl + (self._yr - self._yl)*weight

        if self._nspecies:
            mass = eos.get_density(pressure, temperature,
                                   species_mass_fractions=y)
        else:
            mass = pressure/(temperature*eos.gas_const())

        specmass = mass * y
        mom = mass * velocity
        internal_energy = eos.get_internal_energy(temperature,
                                                  species_mass_fractions=y)

        kinetic_energy = 0.5 * np.dot(velocity, velocity)
        energy = mass * (internal_energy + kinetic_energy)

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=mom, species_mass=specmass)


class PlanarPoiseuille:
    r"""Initializer for the planar Poiseuille case.

    The 2D planar Poiseuille case is defined as a viscous flow between two
    stationary parallel sides with a uniform pressure drop prescribed
    as *p_hi* at the inlet and *p_low* at the outlet. See the figure below:

    .. figure:: ../figures/poiseuille.png
        :scale: 50 %
        :alt: Poiseuille domain illustration

        Illustration of the Poiseuille case setup

    The exact Poiseuille solution is defined by the following:
    $$
    P(x) &= P_{\text{hi}} + P'x\\
    v_x &= \frac{-P'}{2\mu}y(h-y), v_y = 0\\
    \rho &= \rho_0\\
    \rho{E} &= \frac{P(x)}{(\gamma-1)} + \frac{\rho}{2}(\mathbf{v}\cdot\mathbf{v})
    $$

    Here, $P'$ is the constant slope of the linear pressure gradient from the inlet
    to the outlet and is calculated as:
    $$
    P' = \frac{(P_{\text{low}}-P_{\text{hi}})}{\text{length}}
    $$
    $v_x$, and $v_y$ are respectively the x and y components of the velocity,
    $\mathbf{v}$, and $\rho_0$ is the supplied constant density of the fluid.
    """

    def __init__(self, p_hi=100100., p_low=100000., mu=1.0, height=.02, length=.1,
                 density=1.0):
        """Initialize the Poiseuille solution initializer.

        Parameters
        ----------
        p_hi: float
            Pressure at the inlet (default=100100)
        p_low: float
            Pressure at the outlet (default=100000)
        mu: float
            Fluid viscosity, (default = 1.0)
        height: float
            Height of the domain, (default = .02)
        length: float
            Length of the domain, (default = .1)
        density: float
            Constant density of the fluid, (default=1.0)
        """
        self.length = length
        self.height = height
        self.dpdx = (p_low - p_hi)/length
        self.rho = density
        self.mu = mu
        self.p_hi = p_hi

    def __call__(self, x_vec, eos, cv=None, **kwargs):
        r"""Create the Poiseuille solution.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Array of :class:`~meshmode.dof_array.DOFArray` representing the 2D
            coordinates of the points at which the solution is desired
        eos: :class:`~mirgecom.eos.GasEOS`
            A gas equation of state
        cv: :class:`~mirgecom.fluid.ConservedVars`
            Optional fluid state to supply fluid density and velocity if needed.

        Returns
        -------
        :class:`~mirgecom.fluid.ConservedVars`
            The Poiseuille solution state
        """
        dim_mismatch = len(x_vec) != 2
        if cv is not None:
            dim_mismatch = dim_mismatch or cv.dim != 2
        if dim_mismatch:
            raise ValueError("PlanarPoiseuille initializer is 2D only.")

        x = x_vec[0]
        y = x_vec[1]

        actx = x.array_context
        zeros = actx.np.zeros_like(x)

        p_x = self.p_hi + self.dpdx*x

        if cv is not None:
            mass = cv.mass
            velocity = cv.velocity
        else:
            mass = self.rho + zeros
            u_x = -self.dpdx*y*(self.height - y)/(2*self.mu)
            velocity = make_obj_array([u_x, zeros])

        ke = .5*np.dot(velocity, velocity)*mass
        rho_e = p_x/(eos.gamma(cv)-1) + ke
        return make_conserved(2, mass=mass, energy=rho_e,
                              momentum=mass*velocity)

    def exact_grad(self, x_vec, eos, cv_exact):
        """Return the exact gradient of the Poiseuille state."""
        x = x_vec[0]
        y = x_vec[1]

        actx = x.array_context

        # FIXME: Symbolic infrastructure could perhaps do this better
        zeros = actx.np.zeros_like(x)
        ones = zeros + 1
        mass = cv_exact.mass
        velocity = cv_exact.velocity
        dvxdy = -self.dpdx*(self.height-2*y)/(2*self.mu)
        dvdy = make_obj_array([dvxdy, zeros])
        dedx = self.dpdx/(eos.gamma(cv_exact)-1)*ones
        dedy = mass*np.dot(velocity, dvdy)
        dmass = make_obj_array([zeros, zeros])
        denergy = make_obj_array([dedx, dedy])
        dvx = make_obj_array([zeros, dvxdy])
        dvy = make_obj_array([zeros, zeros])
        dv = np.stack((dvx, dvy))
        dmom = mass*dv
        species_mass = velocity*cv_exact.species_mass.reshape(-1, 1)
        return make_conserved(2, mass=dmass, energy=denergy,
                              momentum=dmom, species_mass=species_mass)


class ShearFlow:
    r"""Shear flow exact Navier-Stokes solution from [Hesthaven_2008]_.

    The shear flow solution is described in Section 7.5.3 of
    [Hesthaven_2008]_. It is generalized to major-axis-aligned
    3-dimensional cases here and defined as:

    .. math::

        \rho &= 1\\
        v_\parallel &= r_{t}^2\\
        \mathbf{v}_\bot &= 0\\
        E &= \frac{2\mu{r_\parallel} + 10}{\gamma-1}
        + \frac{r_{t}^4}{2}\\
        \gamma &= \frac{3}{2}, \mu=0.01, \kappa=0

    with fluid total energy $E$, viscosity $\mu$, and specific heat ratio
    $\gamma$. The flow velocity is $\mathbf{v}$ with flow speed and direction
    $v_\parallel$, and $r_\parallel$, respectively. The flow velocity in
    all directions other than $r_\parallel$, is denoted as $\mathbf{v}_\bot$.
    One major-axis-aligned flow-transverse direction, $r_t$ is set by the
    user. This shear flow solution is an exact solution to the fully
    compressible Navier-Stokes equations when neglecting thermal terms;
    i.e., when thermal conductivity $\kappa=0$.  This solution requires a 2d
    or 3d domain.
    """

    def __init__(self, dim=2, mu=.01, gamma=3./2., density=1.,
                 flow_dir=0, trans_dir=1):
        r"""Init the solution object.

        Parameters
        ----------
        dim
            Number of dimensions, 2 and 3 are valid
        mu: float
            Fluid viscosity
        gamma: float
            Ratio of specific heats for the fluid
        density: float
            Fluid mass density, $\rho$
        flow_dir
            Flow direction, $r_\parallel$, for the shear flow, 0=x, 1=y,
            2=z. Defaults to x.
        trans_dir
            Transverse direction, $r_t$, for setting up the shear flow,
            must be other than flow direction, $r_\parallel$, defaults to y.
        """
        if (flow_dir == trans_dir or trans_dir > (dim-1) or flow_dir > (dim-1)
                or flow_dir < 0 or trans_dir < 0):
            raise ValueError(f"Flow and transverse directions must be < {dim=}"
                             f" and > 0.")

        self._dim = dim
        self._mu = mu
        self._gamma = gamma
        self._rho = density
        self._flowdir = flow_dir
        self._transdir = trans_dir

    def __call__(self, x_vec, **kwargs):
        """Return shear flow solution at points *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Point coordinates at which the shear flow solution is desired.

        Returns
        -------
        :class:`~mirgecom.fluid.ConservedVars`
            A CV object with the shear flow solution
        """
        actx = x_vec[0].array_context
        zeros = actx.np.zeros_like(x_vec[0])

        vel = make_obj_array([zeros for i in range(self._dim)])

        flow_dir = self._flowdir
        trans_dir = self._transdir

        ones = zeros + 1.

        for idim in range(self._dim):
            if idim == flow_dir:
                vel[idim] = x_vec[trans_dir]**2
            else:
                vel[idim] = 1.*zeros

        density = self._rho * ones
        mom = self._rho * vel

        pressure = 2*self._mu*x_vec[flow_dir] + 10

        ie = pressure/(self._gamma - 1.)
        ke = self._rho * (x_vec[trans_dir]**4.)/2.
        total_energy = ie + ke

        return make_conserved(dim=self._dim, mass=density, momentum=mom,
                              energy=total_energy)


class InviscidTaylorGreenVortex:
    """Initialize Taylor-Green Vortex."""

    def __init__(
            self, *, dim=3, mach_number=0.05, domain_lengthscale=1, v0=1, p0=1,
            viscosity=1e-5
    ):
        """Initialize vortex parameters."""
        self._mach_number = mach_number
        self._domain_lengthscale = domain_lengthscale
        self._v0 = v0
        self._p0 = p0
        self._mu = viscosity
        self._dim = dim

    def __call__(self, x_vec, *, eos=None, time=0, **kwargs):
        """
        Create the 3D Taylor-Green initial profile at locations *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Nodal coordinates
        eos: :class:`mirgecom.eos.IdealSingleGas`
            Equation of state class with method to supply gas *gamma*.
        """
        if eos is None:
            eos = IdealSingleGas()

        length = self._domain_lengthscale
        gamma = eos.gamma()
        v0 = self._v0
        p0 = self._p0
        rho0 = gamma * self._mach_number ** 2
        dim = len(x_vec)
        x = x_vec[0]
        y = x_vec[1]
        actx = x_vec[0].array_context
        zeros = actx.np.zeros_like(x)
        ones = 1 + zeros
        nu = self._mu/rho0
        ft = actx.np.exp(-2*nu*time)

        if dim == 3:
            z = x_vec[0]

            p = p0 + rho0 * (v0 ** 2) / 16 * (
                actx.np.cos(2*x / length + actx.np.cos(2*y / length))
            ) * actx.np.cos(2*z / length + 2)
            u = (
                v0 * actx.np.sin(x / length) * actx.np.cos(y / length)
            ) * actx.np.cos(z / length)
            v = (
                -v0 * actx.np.cos(x / length) * actx.np.sin(y / length)
            ) * actx.np.cos(z / length)
            w = zeros
            velocity = make_obj_array([u, v, w])
        else:
            u = actx.np.sin(x)*actx.np.cos(y)*ft
            v = -actx.np.cos(x)*actx.np.sin(y)*ft
            p = rho0/4.0 * (actx.np.cos(2*x) + actx.np.sin(2*y)) * ft * ft
            velocity = make_obj_array([u, v])

        momentum = rho0 * velocity
        energy = p / (gamma - 1) + rho0 / 2 * np.dot(velocity, velocity)
        rho = rho0 * ones

        return make_conserved(dim=self._dim, mass=rho,
                              energy=energy, momentum=momentum)
