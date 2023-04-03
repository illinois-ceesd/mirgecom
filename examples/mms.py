"""Test the Navier-Stokes gas dynamics module with some manufactured solutions."""

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
import numpy.random
import numpy.linalg as la  # noqa
import pyopencl.clmath  # noqa
import logging

from pytools.obj_array import (  # noqa
    flat_obj_array,
    make_obj_array,
)

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from mirgecom.fluid import make_conserved

from mirgecom.boundary import (  # noqa
   IsothermalNoSlipBoundary,  # noqa
    AdiabaticSlipBoundary,  # noqa
    PrescribedFluidBoundary  # noqa
)
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)
from abc import ABCMeta, abstractmethod
import mirgecom.math as mm

from meshmode.array_context import (  # noqa
    PyOpenCLArrayContext,
    PytatoPyOpenCLArrayContext
)

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


# Box grid generator widget lifted from @majosm and slightly bent
def _get_box_mesh(dim, a, b, n, t=None, periodic=None):
    if periodic is None:
        periodic = (False,)*dim
    dim_names = ["x", "y", "z"]
    bttf = {}
    for i in range(dim):
        bttf["-"+str(i+1)] = ["-"+dim_names[i]]
        bttf["+"+str(i+1)] = ["+"+dim_names[i]]
    from meshmode.mesh.generation import generate_regular_rect_mesh as gen
    return gen(a=a, b=b, n=n, boundary_tag_to_face=bttf, mesh_type=t,
               periodic=periodic)


class FluidCase(metaclass=ABCMeta):
    """
    A manufactured fluid solution on a mesh.

    .. autoproperty:: dim
    .. automethod:: get_mesh
    .. automethod:: get_solution
    .. automethod:: get_boundaries
    """

    def __init__(self, dim):
        """Init it."""
        self._dim = dim

    @property
    def dim(self):
        """Return the solution ambient dimension."""
        return self._dim

    @abstractmethod
    def get_mesh(self, n):
        """Generate and return a mesh of some given characteristic size *n*."""
        pass

    @abstractmethod
    def get_solution(self, x, t):
        """Return the solution for coordinates *x* and time *t*."""
        pass

    @abstractmethod
    def get_boundaries(self, dcoll, actx, t):
        """Return :class:`dict` mapping boundary tags to bc at time *t*."""
        pass


class FluidManufacturedSolution(FluidCase):
    """Generic fluid manufactured solution for fluid."""

    def __init__(self, dim, lx=None, gamma=1.4, gas_const=287.):
        """Initialize it."""
        import warnings
        super().__init__(dim)
        if lx is None:
            lx = (2.*np.pi,)*self._dim
            warnings.warn(f"Set {lx=}", stacklevel=2)
        if len(lx) != self._dim:
            raise ValueError("Improper dimension for lx.")
        self._gamma = gamma
        self._gas_const = gas_const
        self._lx = lx

    def get_mesh(self, n=2, periodic=None):
        """Return the mesh: [-pi, pi] by default."""
        nx = (n,)*self._dim
        a = tuple(-lx_i/2 for lx_i in self._lx)
        b = tuple(lx_i/2 for lx_i in self._lx)
        return _get_box_mesh(self.dim, a, b, nx, periodic)

    @abstractmethod
    def get_solution(self, x, t):
        """Return the symbolically-compatible solution."""
        pass

    def get_boundaries(self):
        """Get the boundary condition dictionary: prescribed exact by default."""
        from mirgecom.gas_model import make_fluid_state

        def _boundary_state_func(dcoll, dd_bdry, gas_model, state_minus, time=0,
                                 **kwargs):
            actx = state_minus.array_context
            bnd_discr = dcoll.discr_from_dd(dd_bdry)
            nodes = actx.thaw(bnd_discr.nodes())
            return make_fluid_state(self.get_solution(x=nodes, t=time), gas_model)

        return {BTAG_ALL:
                PrescribedFluidBoundary(boundary_state_func=_boundary_state_func)}


# ==== Some trivial and known/exact solutions ===
class UniformSolution(FluidManufacturedSolution):
    """Trivial manufactured solution."""

    def __init__(self, dim=2, density=1, pressure=1, velocity=None,
                 nspecies=0):
        """Init the man soln."""
        super().__init__(dim)
        if velocity is None:
            velocity = make_obj_array([0. for _ in range(dim)])
        assert len(velocity) == dim
        self._vel = velocity
        self._rho = density
        self._pressure = pressure
        self._nspec = nspecies

    def get_mesh(self, n):
        """Get the mesh."""
        return super().get_mesh(n)

    def get_boundaries(self, dcoll, actx, t):
        """Get the boundaries."""
        return super().get_boundaries(dcoll, actx, t)

    def get_solution(self, x, t):
        """Return sym soln."""
        x_c = x[0]
        ones = mm.cos(x_c - x_c) * mm.cos(t - t)
        zeros = mm.sin(x_c - x_c) * mm.sin(t - t)
        for i in range(self._dim):
            if self._vel[i] == 0:
                self._vel[i] = zeros

        density = self._rho*ones
        mom = make_obj_array([self._rho*self._vel[i]*ones
                              for i in range(self._dim)])

        ie = self._pressure / (self._gamma - 1)
        pressure = self._pressure*ones
        ke = .5*density*np.dot(self._vel, self._vel)
        total_energy = (ie + ke)*ones
        temperature = ((pressure / (self._gas_const * density))
                       * ones)
        spec = make_obj_array([ones/self._nspec
                               for _ in range(self._nspec)]) \
            if self._nspec else None

        logger.info(f"{density=}"
                    f"{total_energy=}"
                    f"{mom=}"
                    f"{spec=}")

        return (make_conserved(dim=self._dim, mass=density, momentum=mom,
                               energy=total_energy, species_mass=spec),
                pressure, temperature)


class ShearFlow(FluidManufacturedSolution):
    """ShearFlow from [Hesthaven_2008]_."""

    def __init__(self, dim=2, density=1, pressure=1, gamma=3/2, velocity=None,
                 mu=.01):
        """Init the solution object."""
        super().__init__(dim, gamma=gamma)
        # if velocity is None:
        #     velocity = make_obj_array([0 for _ in range(dim)])
        # assert len(velocity) == dim
        # self._vel = velocity
        # self._rho = density
        # self._pressure = pressure
        self._mu = mu
        self._rho = density
        self._vel = [0, 0]
        self._pressure = pressure
        self._gas_const = 287.0
        self._nspec = 0

    def get_mesh(self, n):
        """Get the mesh."""
        nx = (n,)*self._dim
        a = (0,)*self._dim
        b = (1,)*self._dim
        periodic = (False,)*self._dim
        return _get_box_mesh(self._dim, a, b, nx, periodic=periodic)

    def get_boundaries(self, dcoll, actx, t):
        """Get the boundaries."""
        return super().get_boundaries()

    def get_solution(self, x, t):
        """Return sym soln."""
        x_c = x[0]
        y_c = x[1]

        zeros = mm.sin(x_c - x_c)*mm.sin(t - t)
        ones = mm.cos(x_c - x_c)*mm.cos(t - t)

        v_x = y_c**2
        v_y = 1.*zeros
        if self._dim > 2:
            v_z = 1.*zeros

        density = self._rho * ones

        if self._dim == 2:
            mom = make_obj_array([self._rho*v_x, self._rho*v_y])
        else:
            mom = make_obj_array([self._rho*v_z, self._rho*v_y, self._rho*v_z])

        pressure = 2*self._mu*x_c + 10

        ie = pressure/(self._gamma - 1)
        ke = (density*y_c**4)/2
        total_energy = (ie + ke)*ones

        temperature = (pressure*ones) / (self._gas_const * density)

        return make_conserved(dim=self._dim, mass=density, momentum=mom,
                              energy=total_energy), pressure, temperature


class IsentropicVortex(FluidManufacturedSolution):
    """Isentropic vortex from [Hesthaven_2008]_."""

    def __init__(
            self, dim=2, *, beta=5, center=(0, 0), velocity=(0, 0),
            gamma=1.4, gas_constant=287.
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
        super().__init__(dim)
        self._beta = beta
        self._center = np.array(center)
        self._velocity = np.array(velocity)
        self._gas_const = gas_constant
        self._gamma = gamma

    def get_mesh(self, n):
        """Get the mesh."""
        return super().get_mesh(n)

    def get_boundaries(self, dcoll, actx, t):
        """Get the boundaries."""
        return super().get_boundaries(dcoll, actx, t)

    def get_solution(self, x, t):
        """
        Create the isentropic vortex solution at time *t* at locations *x_vec*.

        The solution at time *t* is created by advecting the vortex under the
        assumption of user-supplied constant, uniform velocity
        (``Vortex2D._velocity``).

        Parameters
        ----------
        t: float
            Current time at which the solution is desired.
        x: numpy.ndarray
            Coordinates for points at which solution is desired.
        eos: mirgecom.eos.IdealSingleGas
            Equation of state class to supply method for gas *gamma*.
        """
        x_c = x[0]
        y_c = x[1]
        gamma = self._gamma

        # coordinates relative to vortex center
        vortex_loc = self._center + t * self._velocity
        x_rel = x_c - vortex_loc[0]
        y_rel = y_c - vortex_loc[1]

        r2 = (x_rel ** 2 + y_rel ** 2)
        expterm = self._beta * mm.exp(1 - r2)

        u = self._velocity[0] - expterm * y_rel / (2 * np.pi)
        v = self._velocity[1] + expterm * x_rel / (2 * np.pi)
        velocity = make_obj_array([u, v])

        mass = (1 - (gamma - 1) / (16 * gamma * np.pi ** 2)
                * expterm ** 2) ** (1 / (gamma - 1))
        momentum = mass * velocity
        p = mass ** gamma

        energy = p / (gamma - 1) + mass / 2 * (u ** 2 + v ** 2)
        temperature = p / (mass*self._gas_const)

        return make_conserved(dim=2, mass=mass, energy=energy,
                              momentum=momentum), p, temperature


class TrigSolution1(FluidManufacturedSolution):
    """CNS manufactured solution designed to vanish on the domain boundary."""

    def __init__(self, dim, q_coeff, x_coeff, lx=None,
                 gamma=1.4, gas_const=287.):
        """Initialize it."""
        super().__init__(dim, lx, gamma, gas_const)
        self._q_coeff = q_coeff

    def get_solution(self, x, t):
        """Return the symbolically-compatible solution."""
        velocity = 0
        press = 1
        density = 1

        mom = density*velocity
        temperature = press/(density*self._gas_const)
        energy = press/(self._gamma - 1) + density*np.dot(velocity, velocity)
        return make_conserved(dim=self._dim, mass=density, momentum=mom,
                              energy=energy), press, temperature

    def get_mesh(self, x, t):
        """Get the mesh."""
        return super().get_mesh(x, t)

    def get_boundaries(self):
        """Get the boundaries."""
        return super().get_boundaries()


class TestSolution(FluidManufacturedSolution):
    """Trivial manufactured solution."""

    def __init__(self, dim=2, density=1, pressure=1, velocity=None):
        """Init the man soln."""
        super().__init__(dim)
        if velocity is None:
            velocity = make_obj_array([0 for _ in range(dim)])
        assert len(velocity) == dim
        self._vel = velocity
        self._rho = density
        self._pressure = pressure

    def get_mesh(self, n):
        """Get the mesh."""
        return super().get_mesh(n)

    def get_boundaries(self, dcoll, actx, t):
        """Get the boundaries."""
        return super().get_boundaries(dcoll, actx, t)

    def get_solution(self, x, t):
        """Return sym soln."""
        density = 1*x[0]
        energy = 2*x[1]**2
        mom = make_obj_array([i*x[0]*x[1] for i in range(self._dim)])
        pressure = x[0]*x[0]*x[0]
        temperature = x[1]*x[1]*x[1]

        return make_conserved(dim=self._dim, mass=density, momentum=mom,
                              energy=energy), pressure, temperature


class RoySolution(FluidManufacturedSolution):
    """CNS manufactured solution from [Roy_2017]__."""

    def __init__(self, dim, q_coeff=None, x_coeff=None, lx=None, gamma=1.4,
                 gas_const=287., alpha=0.0, beta=1e-5, sigma=1e-1, n=.666,
                 nspecies=0, d_alpha=None, mu=0, kappa=0):
        """Initialize it."""
        super().__init__(dim=dim, lx=lx, gamma=gamma, gas_const=gas_const)
        if q_coeff is None:
            q_coeff = 0
        self._q_coeff = q_coeff
        self._x_coeff = x_coeff
        self._alpha = alpha
        self._beta = beta
        self._sigma = sigma
        self._n = n
        self._mu = mu
        self._kappa = kappa
        self._nspecies = nspecies
        self._d_alpha = d_alpha

    def get_solution(self, x, t):
        """Return the symbolically-compatible solution."""
        c = self._q_coeff[0]
        ax = self._x_coeff[0]
        lx = self._lx
        tone = mm.cos(t - t)
        xone = mm.cos(x[0] - x[0])
        omega_x = [np.pi*x[i]/lx[i] for i in range(self._dim)]

        funcs = [mm.sin, mm.cos, mm.sin]
        density = c[0] + sum(c[i+1]*funcs[i](ax[i]*omega_x[i])
                             for i in range(self._dim))*tone

        c = self._q_coeff[1]
        ax = self._x_coeff[1]
        funcs = [mm.cos, mm.sin, mm.cos]
        press = c[0] + sum(c[i+1]*funcs[i](ax[i]*omega_x[i])
                           for i in range(self._dim))*tone

        c = self._q_coeff[2]
        ax = self._x_coeff[2]
        funcs = [mm.sin, mm.cos, mm.cos]
        u = c[0] + sum(c[i+1]*funcs[i](ax[i]*omega_x[i])
                       for i in range(self._dim))*tone

        if self._dim > 1:
            c = self._q_coeff[3]
            ax = self._x_coeff[3]
            funcs = [mm.cos, mm.sin, mm.sin]
            v = c[0] + sum(c[i+1]*funcs[i](ax[i]*omega_x[i])
                           for i in range(self._dim))*tone

        if self._dim > 2:
            c = self._q_coeff[4]
            ax = self._x_coeff[4]
            funcs = [mm.sin, mm.sin, mm.cos]
            w = c[0] + sum(c[i+1]*funcs[i](ax[i]*omega_x[i])
                           for i in range(self._dim))*tone

        if self._dim == 1:
            velocity = make_obj_array([u])
        if self._dim == 2:
            velocity = make_obj_array([u, v])
        if self._dim == 3:
            velocity = make_obj_array([u, v, w])

        mom = density*velocity
        temperature = press/(density*self._gas_const)*tone
        energy = press/(self._gamma - 1) + .5*density*np.dot(velocity, velocity)*xone

        species_mass = None
        if self._nspecies > 0:
            spec = make_obj_array([xone/self._nspecies
                            for _ in range(self._nspecies)])
            species_mass = density*spec

        return (make_conserved(dim=self._dim, mass=density, momentum=mom,
                               energy=energy, species_mass=species_mass),
                press, temperature)

    def get_transport_properties(self, sym_soln, sym_pressure, sym_temperature):
        if self._n == 0:
            mu = self._mu
            kappa = self._kappa
        else:
            mu = self._alpha * sym_temperature*self._n
            kappa = self._sigma * mu * self._gas_const / (self._gamma - 1)

        return mu, kappa

    def get_mesh(self, n):
        """Get the mesh."""
        return super().get_mesh(n)

    def get_boundaries(self, dcoll, actx, t):
        """Get the boundaries."""
        return super().get_boundaries(dcoll, actx, t)


class VarMuSolution(FluidManufacturedSolution):
    """CNS manufactured solution from [Roy_2017]__ with modification to
    use powerlaw transport model."""

    def __init__(self, dim, q_coeff=None, x_coeff=None, lx=None, gamma=1.4,
                 gas_const=287., alpha=0.0, beta=1e-5, sigma=1e-1, n=.666):
        """Initialize it."""
        super().__init__(dim=dim, lx=lx, gamma=gamma, gas_const=gas_const)
        if q_coeff is None:
            q_coeff = 0
        self._q_coeff = q_coeff
        self._x_coeff = x_coeff
        self._alpha = alpha
        self._beta = beta
        self._sigma = sigma
        self._n = n

    def get_solution(self, x, t):
        """Return the symbolically-compatible solution."""
        c = self._q_coeff[0]
        ax = self._x_coeff[0]
        lx = self._lx
        tone = mm.cos(t - t)
        xone = mm.cos(x[0] - x[0])
        omega_x = [np.pi*x[i]/lx[i] for i in range(self._dim)]

        funcs = [mm.sin, mm.cos, mm.sin]
        density = c[0] + sum(c[i+1]*funcs[i](ax[i]*omega_x[i])
                             for i in range(self._dim))*tone

        c = self._q_coeff[1]
        ax = self._x_coeff[1]
        funcs = [mm.cos, mm.sin, mm.cos]
        press = c[0] + sum(c[i+1]*funcs[i](ax[i]*omega_x[i])
                           for i in range(self._dim))*tone

        c = self._q_coeff[2]
        ax = self._x_coeff[2]
        funcs = [mm.sin, mm.cos, mm.cos]
        u = c[0] + sum(c[i+1]*funcs[i](ax[i]*omega_x[i])
                       for i in range(self._dim))*tone

        if self._dim > 1:
            c = self._q_coeff[3]
            ax = self._x_coeff[3]
            funcs = [mm.cos, mm.sin, mm.sin]
            v = c[0] + sum(c[i+1]*funcs[i](ax[i]*omega_x[i])
                           for i in range(self._dim))*tone

        if self._dim > 2:
            c = self._q_coeff[4]
            ax = self._x_coeff[4]
            funcs = [mm.sin, mm.sin, mm.cos]
            w = c[0] + sum(c[i+1]*funcs[i](ax[i]*omega_x[i])
                           for i in range(self._dim))*tone

        if self._dim == 1:
            velocity = make_obj_array([u])
        if self._dim == 2:
            velocity = make_obj_array([u, v])
        if self._dim == 3:
            velocity = make_obj_array([u, v, w])

        mom = density*velocity
        temperature = press/(density*self._gas_const)*tone
        energy = press/(self._gamma - 1) + .5*density*np.dot(velocity, velocity)*xone
        return make_conserved(dim=self._dim, mass=density, momentum=mom,
                              energy=energy), press, temperature

    def get_transport_properties(self, sym_soln, sym_pressure, sym_temperature):
        mu = self._alpha * sym_temperature*self._n
        kappa = self._sigma * mu * self._gas_const / (self._gamma - 1)
        return mu, kappa

    def get_mesh(self, n):
        """Get the mesh."""
        return super().get_mesh(n)

    def get_boundaries(self, dcoll, actx, t):
        """Get the boundaries."""
        return super().get_boundaries(dcoll, actx, t)


class MoserSolution(FluidManufacturedSolution):
    """CNS manufactured solution from [Moser_2017]__."""

    def __init__(self, dim, q_coeff=None, lx=None, gamma=1.4,
                 gas_const=287., alpha=0.0, beta=1e-5, sigma=1e-1, n=.666,
                 nspecies=0, d_alpha=None, mu=0, kappa=0):
        """Initialize it."""
        super().__init__(dim=dim, lx=lx, gamma=gamma, gas_const=gas_const)
        self._alpha = alpha
        self._beta = beta
        self._sigma = sigma
        self._n = n
        self._mu = mu
        self._kappa = kappa
        self._nspecies = nspecies
        self._d_alpha = d_alpha
        self._lxx = self._lx[0]
        self._lxy = self._lx[1]
        self._lxz = self._lx[2]

    def _phi(self, a, b, c, d, e, f, g, r, t):
        x, y, z = r
        a_0, a_x, a_y, a_z, a_xy, a_xz, a_yz = a
        f_0, f_x, f_y, f_z, f_xy, f_xz, f_yz = f
        g_0, g_x, g_y, g_z, g_xy, g_xz, g_yz = g
        b_x, b_y, b_z, b_xy, b_xz, b_yz = b
        c_x, c_y, c_z, c_xy, c_xz, c_yz = c
        d_xy, d_xz, d_yz = d
        e_xy, e_xz, e_yz = e

        return \
            (a_0 * mm.cos(f_0 * t + g_0)
             + (a_x * mm.cos(b_x * 2*np.pi*x/self._lxx + c_x)
                * mm.cos(f_x * t + g_x))
             + (a_xy * mm.cos(b_xy * 2*np.pi*x/self._lxx + c_xy)
                * mm.cos(d_xy * 2*np.pi*y/self._lxy + e_xy)
                * mm.cos(f_xy * t + g_xy))
             + (a_xz * mm.cos(b_xz * 2*np.pi*x/self._lxx + c_xz)
                * mm.cos(d_xz * 2*np.pi*z/self._lxz + e_xz)
                * mm.cos(f_xz * t + g_xz))
             + (a_y * mm.cos(b_y * 2*np.pi*y/self._lxy + c_y)
                * mm.cos(f_y * t + g_y))
             + (a_yz * mm.cos(b_yz * 2*np.pi*y/self._lxy + c_yz)
                * mm.cos(d_yz * 2*np.pi*z/self._lxz + e_yz)
                * mm.cos(f_yz * t + g_yz))
             + (a_z * mm.cos(b_z * 2*np.pi*z/self._lxz + c_z)
                * mm.cos(f_z * t + g_z)))

    def _get_channel_solution(self, x, t):

        tone = mm.cos(t - t)
        xone = mm.cos(x[0] - x[0])

        # RHO soln parameters
        #        0,  x,   y,    z,    xy,   xz,   yz
        a_rho = [1., 0., 1./7., 0., 1./11., 0., 1./31.]
        f_rho = [0., 0., 1., 0., 3., 0., 2.]
        g_rho = [0., 0., np.pi/4.0 - 1./20., 0., np.pi/4., 0., np.pi/4. + 1./20.]

        #        x    y     z   xy   xz  yz
        b_rho = [0., 1./2., 0., 3.0, 0., 2.]
        c_rho = [0., 0., 0., 0., 0., 0.]

        #        xy    xz    yz
        d_rho = [3., 0., 2.]
        e_rho = [0., 0., 0.]

        rho = self._phi(a=a_rho, b=b_rho, c=c_rho, d=d_rho, e=e_rho, f=f_rho,
                        g=g_rho, r=x, t=t)*tone

        # vx=velocity[0] soln parameters
        #        0,  x,   y,    z,    xy,   xz,   yz
        a_vx = [0., 0., 53., 0., 53./37., 0., 53./41.]
        f_vx = [0., 0., 1., 0., 3., 0., 2.]
        g_vx = [0., 0., np.pi/4.0 - 1./20., 0., np.pi/4., 0., np.pi/4. + 1./20.]

        #        x    y     z   xy   xz  yz
        b_vx = [0., 1./2., 0., 3.0, 0., 2.]
        c_vx = [0., -np.pi/2., 0., -np.pi/2., 0., -np.pi/2.]

        #        xy    xz    yz
        d_vx = [3., 0., 2.]
        e_vx = [-np.pi/2., 0., -np.pi/2.]

        u = self._phi(a=a_vx, b=b_vx, c=c_vx, d=d_vx, e=e_vx, f=f_vx,
                      g=g_vx, r=x, t=t)*tone

        # vy=velocity[1] soln parameters
        #        0,  x,   y,    z,    xy,   xz,   yz
        a_vy = [0., 0., 2., 0., 3., 0., 5.]
        f_vy = [0., 0., 1., 0., 3., 0., 2.]
        g_vy = [0., 0., np.pi/4.0 - 1./20., 0., np.pi/4., 0., np.pi/4. + 1./20.]

        #        x    y     z   xy   xz  yz
        b_vy = [0., 1./2., 0., 3.0, 0., 2.]
        c_vy = [0., -np.pi/2., 0., -np.pi/2., 0., -np.pi/2.]

        #        xy    xz    yz
        d_vy = [3., 0., 2.]
        e_vy = [-np.pi/2., 0., -np.pi/2.]

        v = self._phi(a=a_vy, b=b_vy, c=c_vy, d=d_vy, e=e_vy, f=f_vy,
                      g=g_vy, r=x, t=t)*tone

        # vz=velocity[2] soln parameters
        #        0,  x,   y,    z,    xy,   xz,   yz
        a_vz = [0., 0., 7., 0., 11., 0., 13.]
        f_vz = [0., 0., 1., 0., 3., 0., 2.]
        g_vz = [0., 0., np.pi/4.0 - 1./20., 0., np.pi/4., 0., np.pi/4. + 1./20.]

        #        x    y     z   xy   xz  yz
        b_vz = [0., 1./2., 0., 3.0, 0., 2.]
        c_vz = [0., -np.pi/2., 0., -np.pi/2., 0., -np.pi/2.]

        #        xy    xz    yz
        d_vz = [3., 0., 2.]
        e_vz = [-np.pi/2., 0., -np.pi/2.]

        w = self._phi(a=a_vz, b=b_vz, c=c_vz, d=d_vz, e=e_vz, f=f_vz,
                      g=g_vz, r=x, t=t)*tone

        # Temperature soln parameters
        #        0,  x,   y,    z,    xy,   xz,   yz
        a_tmpr = [300., 0., 300./13., 0., 300./17., 0., 300./37.]
        f_tmpr = [0., 0., 1., 0., 3., 0., 2.]
        g_tmpr = [0., 0., np.pi/4.0 - 1./20., 0., np.pi/4., 0., np.pi/4. + 1./20.]

        #        x    y     z   xy   xz  yz
        b_tmpr = [0., 1./2., 0., 3.0, 0., 2.]
        c_tmpr = [0., -np.pi/2., 0., -np.pi/2., 0., -np.pi/2.]

        #        xy    xz    yz
        d_tmpr = [3., 0., 2.]
        e_tmpr = [-np.pi/2., 0., -np.pi/2.]

        tmptr = self._phi(a=a_tmpr, b=b_tmpr, c=c_tmpr, d=d_tmpr, e=e_tmpr, f=f_tmpr,
                          g=g_tmpr, r=x, t=t)*tone
        press = rho * self._gas_const * tmptr

        if self._dim == 1:
            velocity = make_obj_array([u])
        if self._dim == 2:
            velocity = make_obj_array([u, v])
        if self._dim == 3:
            velocity = make_obj_array([u, v, w])

        energy = press / (self._gamma - 1) + rho*np.dot(velocity, velocity)/2.
        mom = rho*velocity

        species_mass = None
        if self._nspecies > 0:
            spec = make_obj_array([xone/self._nspecies
                            for _ in range(self._nspecies)])
            species_mass = rho*spec

        return (make_conserved(dim=self._dim, mass=rho, momentum=mom,
                               energy=energy, species_mass=species_mass),
                press, tmptr)

    def get_transport_properties(self, sym_soln, sym_pressure, sym_temperature):
        if self._n == 0:
            mu = self._mu
            kappa = self._kappa
        else:
            mu = self._alpha * sym_temperature*self._n
            kappa = self._sigma * mu * self._gas_const / (self._gamma - 1)

        return mu, kappa

    def get_mesh(self, n=2, periodic=None):
        """Return the mesh: [-pi, pi] by default."""
        nx = (n,)*self._dim
        a = (0., 0., 0.)
        b = (4.*np.pi, 2., 4*np.pi/3.)
        return _get_box_mesh(self.dim, a, b, nx, periodic=periodic)

    def get_boundaries(self, dcoll, actx, t):
        """Get the boundaries."""
        return super().get_boundaries(dcoll, actx, t)

    def get_solution(self, x, t):
        """Return the symbolically-compatible solution."""
        return self._get_channel_solution(x, t)
