"""Test the Navier-Stokes gas dynamics module."""

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

from functools import partial
from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.random
import numpy.linalg as la  # noqa
import pyopencl.clmath  # noqa
import logging
import pytest

from pytools.obj_array import (
    flat_obj_array,
    make_obj_array,
)

from arraycontext import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from mirgecom.navierstokes import ns_operator
from mirgecom.fluid import make_conserved
from grudge.dof_desc import DTAG_BOUNDARY

import pymbolic as pmbl
from mirgecom.symbolic import (
    diff as sym_diff,
    grad as sym_grad,
    div as sym_div,
    EvaluationMapper,
    evaluate)
import mirgecom.math as mm

from mirgecom.boundary import (
    DummyBoundary,
    PrescribedFluidBoundary,
    AdiabaticNoslipMovingBoundary
)
from mirgecom.eos import IdealSingleGas
from mirgecom.transport import SimpleTransport
from grudge.eager import EagerDGDiscretization
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)


logger = logging.getLogger(__name__)


@pytest.mark.parametrize("nspecies", [0, 10])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_uniform_rhs(actx_factory, nspecies, dim, order):
    """Test the Navier-Stokes operator using a trivial constant/uniform state.

    This state should yield rhs = 0 to FP.  The test is performed for 1, 2,
    and 3 dimensions, with orders 1, 2, and 3, with and without passive species.
    """
    actx = actx_factory()

    tolerance = 1e-9

    from pytools.convergence import EOCRecorder
    eoc_rec0 = EOCRecorder()
    eoc_rec1 = EOCRecorder()
    # for nel_1d in [4, 8, 12]:
    for nel_1d in [4, 8]:
        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(
            a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
        )

        logger.info(
            f"Number of {dim}d elements: {mesh.nelements}"
        )

        discr = EagerDGDiscretization(actx, mesh, order=order)
        zeros = discr.zeros(actx)
        ones = zeros + 1.0

        mass_input = discr.zeros(actx) + 1
        energy_input = discr.zeros(actx) + 2.5

        mom_input = make_obj_array(
            [discr.zeros(actx) for i in range(discr.dim)]
        )

        mass_frac_input = flat_obj_array(
            [ones / ((i + 1) * 10) for i in range(nspecies)]
        )
        species_mass_input = mass_input * mass_frac_input
        num_equations = dim + 2 + len(species_mass_input)

        cv = make_conserved(
            dim, mass=mass_input, energy=energy_input, momentum=mom_input,
            species_mass=species_mass_input)

        expected_rhs = make_conserved(
            dim, q=make_obj_array([discr.zeros(actx)
                                   for i in range(num_equations)])
        )
        mu = 1.0
        kappa = 0.0
        spec_diffusivity = 0 * np.ones(nspecies)

        from mirgecom.gas_model import GasModel, make_fluid_state
        gas_model = GasModel(
            eos=IdealSingleGas(),
            transport=SimpleTransport(viscosity=mu, thermal_conductivity=kappa,
                                      species_diffusivity=spec_diffusivity))
        state = make_fluid_state(gas_model=gas_model, cv=cv)

        boundaries = {BTAG_ALL: DummyBoundary()}

        ns_rhs = ns_operator(discr, gas_model=gas_model, boundaries=boundaries,
                             state=state, time=0.0)

        rhs_resid = ns_rhs - expected_rhs
        rho_resid = rhs_resid.mass
        rhoe_resid = rhs_resid.energy
        mom_resid = rhs_resid.momentum
        rhoy_resid = rhs_resid.species_mass

        rho_rhs = ns_rhs.mass
        rhoe_rhs = ns_rhs.energy
        rhov_rhs = ns_rhs.momentum
        rhoy_rhs = ns_rhs.species_mass

        logger.info(
            f"rho_rhs  = {rho_rhs}\n"
            f"rhoe_rhs = {rhoe_rhs}\n"
            f"rhov_rhs = {rhov_rhs}\n"
            f"rhoy_rhs = {rhoy_rhs}\n"
        )

        assert actx.to_numpy(discr.norm(rho_resid, np.inf)) < tolerance
        assert actx.to_numpy(discr.norm(rhoe_resid, np.inf)) < tolerance
        for i in range(dim):
            assert actx.to_numpy(discr.norm(mom_resid[i], np.inf)) < tolerance
        for i in range(nspecies):
            assert actx.to_numpy(discr.norm(rhoy_resid[i], np.inf)) < tolerance

        err_max = actx.to_numpy(discr.norm(rho_resid, np.inf))
        eoc_rec0.add_data_point(1.0 / nel_1d, err_max)

        # set a non-zero, but uniform velocity component
        for i in range(len(mom_input)):
            mom_input[i] = discr.zeros(actx) + (-1.0) ** i

        cv = make_conserved(
            dim, mass=mass_input, energy=energy_input, momentum=mom_input,
            species_mass=species_mass_input)

        state = make_fluid_state(gas_model=gas_model, cv=cv)
        boundaries = {BTAG_ALL: DummyBoundary()}
        ns_rhs = ns_operator(discr, gas_model=gas_model, boundaries=boundaries,
                             state=state, time=0.0)

        rhs_resid = ns_rhs - expected_rhs

        rho_resid = rhs_resid.mass
        rhoe_resid = rhs_resid.energy
        mom_resid = rhs_resid.momentum
        rhoy_resid = rhs_resid.species_mass

        assert actx.to_numpy(discr.norm(rho_resid, np.inf)) < tolerance
        assert actx.to_numpy(discr.norm(rhoe_resid, np.inf)) < tolerance

        for i in range(dim):
            assert actx.to_numpy(discr.norm(mom_resid[i], np.inf)) < tolerance
        for i in range(nspecies):
            assert actx.to_numpy(discr.norm(rhoy_resid[i], np.inf)) < tolerance

        err_max = actx.to_numpy(discr.norm(rho_resid, np.inf))
        eoc_rec1.add_data_point(1.0 / nel_1d, err_max)

    logger.info(
        f"V == 0 Errors:\n{eoc_rec0}"
        f"V != 0 Errors:\n{eoc_rec1}"
    )

    assert (
        eoc_rec0.order_estimate() >= order - 0.5
        or eoc_rec0.max_error() < 1e-9
    )
    assert (
        eoc_rec1.order_estimate() >= order - 0.5
        or eoc_rec1.max_error() < 1e-9
    )


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


@pytest.mark.parametrize("order", [2, 3])
def test_poiseuille_rhs(actx_factory, order):
    """Test the Navier-Stokes operator using a Poiseuille state.

    This state should yield rhs = 0 to FP.  The test is performed for 1, 2,
    and 3 dimensions, with orders 1, 2, and 3, with and without passive species.
    """
    actx = actx_factory()
    dim = 2
    tolerance = 1e-9

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    base_pressure = 100000.0
    pressure_ratio = 1.001
    mu = 1.0
    left_boundary_location = 0
    right_boundary_location = 0.1
    ybottom = 0.
    ytop = .02
    nspecies = 0
    mu = 1.0
    kappa = 0.0
    spec_diffusivity = 0 * np.ones(nspecies)
    from mirgecom.gas_model import GasModel, make_fluid_state
    gas_model = GasModel(
        eos=IdealSingleGas(),
        transport=SimpleTransport(viscosity=mu, thermal_conductivity=kappa,
                                  species_diffusivity=spec_diffusivity))

    def poiseuille_2d(x_vec, eos, cv=None, **kwargs):
        y = x_vec[1]
        x = x_vec[0]
        x0 = left_boundary_location
        xmax = right_boundary_location
        xlen = xmax - x0
        p_low = base_pressure
        p_hi = pressure_ratio*base_pressure
        dp = p_hi - p_low
        dpdx = dp/xlen
        h = ytop - ybottom
        u_x = dpdx*y*(h - y)/(2*mu)
        p_x = p_hi - dpdx*x
        rho = 1.0
        mass = 0*x + rho
        u_y = 0*x
        velocity = make_obj_array([u_x, u_y])
        ke = .5*np.dot(velocity, velocity)*mass
        gamma = eos.gamma()
        if cv is not None:
            mass = cv.mass
            vel = cv.velocity
            ke = .5*np.dot(vel, vel)*mass

        rho_e = p_x/(gamma-1) + ke
        return make_conserved(2, mass=mass, energy=rho_e,
                              momentum=mass*velocity)

    initializer = poiseuille_2d

    # for nel_1d in [4, 8, 12]:
    for nfac in [1, 2, 4, 8]:

        npts_axis = nfac*(12, 20)
        box_ll = (left_boundary_location, ybottom)
        box_ur = (right_boundary_location, ytop)
        mesh = _get_box_mesh(2, a=box_ll, b=box_ur, n=npts_axis)

        logger.info(
            f"Number of {dim}d elements: {mesh.nelements}"
        )

        discr = EagerDGDiscretization(actx, mesh, order=order)
        nodes = thaw(discr.nodes(), actx)

        cv_input = initializer(x_vec=nodes, eos=gas_model.eos)
        num_eqns = dim + 2
        expected_rhs = make_conserved(
            dim, q=make_obj_array([discr.zeros(actx)
                                   for i in range(num_eqns)])
        )

        def boundary_func(discr, btag, gas_model, state_minus, **kwargs):
            actx = state_minus.array_context
            bnd_discr = discr.discr_from_dd(btag)
            nodes = thaw(bnd_discr.nodes(), actx)
            return make_fluid_state(initializer(x_vec=nodes, eos=gas_model.eos,
                                                **kwargs), gas_model)

        boundaries = {
            DTAG_BOUNDARY("-1"):
            PrescribedFluidBoundary(boundary_state_func=boundary_func),
            DTAG_BOUNDARY("+1"):
            PrescribedFluidBoundary(boundary_state_func=boundary_func),
            DTAG_BOUNDARY("-2"): AdiabaticNoslipMovingBoundary(),
            DTAG_BOUNDARY("+2"): AdiabaticNoslipMovingBoundary()}

        state = make_fluid_state(gas_model=gas_model, cv=cv_input)
        ns_rhs = ns_operator(discr, gas_model=gas_model, boundaries=boundaries,
                             state=state, time=0.0)

        rhs_resid = ns_rhs - expected_rhs
        rho_resid = rhs_resid.mass
        # rhoe_resid = rhs_resid.energy
        mom_resid = rhs_resid.momentum

        rho_rhs = ns_rhs.mass
        # rhoe_rhs = ns_rhs.energy
        rhov_rhs = ns_rhs.momentum
        # rhoy_rhs = ns_rhs.species_mass

        print(
            f"rho_rhs  = {rho_rhs}\n"
            # f"rhoe_rhs = {rhoe_rhs}\n"
            f"rhov_rhs = {rhov_rhs}\n"
            # f"rhoy_rhs = {rhoy_rhs}\n"
        )

        tol_fudge = 2e-4
        assert actx.to_numpy(discr.norm(rho_resid, np.inf)) < tolerance
        # assert actx.to_numpy(discr.norm(rhoe_resid, np.inf)) < tolerance
        mom_err = [actx.to_numpy(discr.norm(mom_resid[i], np.inf))
                   for i in range(dim)]
        err_max = max(mom_err)
        for i in range(dim):
            assert mom_err[i] < tol_fudge

        # err_max = actx.to_numpy(discr.norm(rho_resid, np.inf)
        eoc_rec.add_data_point(1.0 / nfac, err_max)

    logger.info(
        f"V != 0 Errors:\n{eoc_rec}"
    )

    assert (
        eoc_rec.order_estimate() >= order - 0.5
        or eoc_rec.max_error() < tol_fudge
    )


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
    def get_boundaries(self, discr, actx, t):
        """Return :class:`dict` mapping boundary tags to bc at time *t*."""
        pass


class FluidManufacturedSolution(FluidCase):
    """Generic fluid manufactured solution for fluid."""

    def __init__(self, dim, n=2, lx=None, nx=None, gamma=1.4, gas_const=287.):
        """Initialize it."""
        super().__init__(dim)
        if lx is None:
            lx = (2.*np.pi,)*self._dim
        if len(lx) != self._dim:
            raise ValueError("Improper dimension for lx.")
        self._gamma = gamma
        self._gas_const = gas_const
        self._lx = lx

    @abstractmethod
    def get_mesh(self, n=2, periodic=None):
        """Return the mesh: [-pi, pi] by default."""
        nx = (n,)*self._dim
        a = (-self._lx[0]/2,)
        b = (self._lx[0]/2,)
        if self._dim == 2:
            a = (a[0], -self._lx[1]/2)
            b = (b[0], self._lx[1]/2)
        if self._dim == 3:
            a = (a[0], -self._lx[1]/2, -self._lx[2]/2)
            b = (b[0], self._lx[1]/2, self._lx[2]/2)
        return _get_box_mesh(self.dim, a, b, nx, periodic)

    @abstractmethod
    def get_solution(self, x, t):
        """Return the symbolically-compatible solution."""
        pass

    @abstractmethod
    def get_boundaries(self, discr, actx, t):
        """Get the boundary condition dictionary: prescribed exact by default."""
        from mirgecom.gas_model import make_fluid_state

        def _boundary_state_func(discr, btag, gas_model, state_minus, **kwargs):
            actx = state_minus.array_context
            bnd_discr = discr.discr_from_dd(btag)
            nodes = thaw(bnd_discr.nodes(), actx)
            # need to evaluate symbolic soln for this part
            return make_fluid_state(self.get_solution(x=nodes, t=t), gas_model)

        return {BTAG_ALL:
                PrescribedFluidBoundary(boundary_state_func=_boundary_state_func)}


class RoyManufacturedSolution(FluidManufacturedSolution):
    """CNS manufactured solution from [Roy_2017]__."""

    def __init__(self, dim, q_coeff, x_coeff, n=2, lx=None, nx=None,
                 gamma=1.4, gas_const=287.):
        """Initialize it."""
        super().__init__(dim, lx, gamma, gas_const)
        self._q_coeff = q_coeff
        self._x_coeff = x_coeff

    def get_solution(self, x, t):
        """Return the symbolically-compatible solution."""
        c = self._q_coeff[0]
        ax = self._x_coeff[0]
        lx = self._lx
        omega_x = [np.pi*x[i]/lx[i] for i in range(self._dim)]

        funcs = [mm.sin, mm.cos, mm.sin]
        density = c[0] + sum(c[i+1]*funcs[i](ax[i]*omega_x[i])
                             for i in range(self._dim))

        c = self._q_coeff[1]
        ax = self._x_coeff[1]
        funcs = [mm.cos, mm.sin, mm.cos]
        press = c[0] + sum(c[i+1]*funcs[i](ax[i]*omega_x[i])
                           for i in range(self._dim))

        c = self._q_coeff[2]
        ax = self._x_coeff[2]
        funcs = [mm.sin, mm.cos, mm.cos]
        u = c[0] + sum(c[i+1]*funcs[i](ax[i]*omega_x[i])
                       for i in range(self._dim))

        if self._dim > 1:
            c = self._q_coeff[3]
            ax = self._x_coeff[3]
            funcs = [mm.cos, mm.sin, mm.sin]
            v = c[0] + sum(c[i+1]*funcs[i](ax[i]*omega_x[i])
                           for i in range(self._dim))

        if self._dim > 2:
            c = self._q_coeff[4]
            ax = self._x_coeff[4]
            funcs = [mm.sin, mm.sin, mm.cos]
            w = c[0] + sum(c[i+1]*funcs[i](ax[i]*omega_x[i])
                           for i in range(self._dim))

        if self._dim == 1:
            velocity = make_obj_array([u])
        if self._dim == 2:
            velocity = make_obj_array([u, v])
        if self._dim == 3:
            velocity = make_obj_array([u, v, w])

        mom = density*velocity
        temperature = press/(density*self._gas_const)
        energy = press/(self._gamma - 1) + density*np.dot(velocity, velocity)
        return make_conserved(dim=self._dim, mass=density, momentum=mom,
                              energy=energy), press, temperature


# ==== Some trivial and known/exact solutions ===
class UniformSolution(FluidManufacturedSolution):
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

    def get_boundaries(self, discr, actx, t):
        """Get the boundaries."""
        return super().get_boundaries(discr, actx, t)

    def get_solution(self, x, t):
        """Return sym soln."""
        zeros = 0*x[0]
        ones = zeros + 1.

        density = self._rho*ones
        velocity = make_obj_array([self._vel[i]*ones
                                   for i in range(self._dim)])

        mom = density*velocity
        ie = self._pressure / (self._gamma - 1)
        pressure = self._pressure*ones
        ke = .5*density*np.dot(velocity, velocity)
        total_energy = ie + ke
        temperature = pressure / (self._gas_const * density)

        return make_conserved(dim=self._dim, mass=density, momentum=mom,
                              energy=total_energy), pressure, temperature


class ShearFlow(FluidManufacturedSolution):
    """Trivial manufactured solution."""

    def __init__(self, dim=2, density=1, pressure=1, velocity=None):
        """Init the solution object."""
        super().__init__(dim)
        # if velocity is None:
        #     velocity = make_obj_array([0 for _ in range(dim)])
        # assert len(velocity) == dim
        # self._vel = velocity
        # self._rho = density
        # self._pressure = pressure

    def get_mesh(self, n):
        """Get the mesh."""
        return super().get_mesh(n)

    def get_boundaries(self, discr, actx, t):
        """Get the boundaries."""
        return super().get_boundaries(discr, actx, t)

    def get_solution(self, x, t):
        """Return sym soln."""
        x = x[0]
        y = x[1]
        zeros = 0*x
        ones = zeros + 1.
        v_x = y*y
        v_y = 0*ones
        density = 1*ones
        velocity = make_obj_array([v_x, v_y])
        mom = density*velocity
        mu = 1
        pressure = 2*mu*x+10
        ie = pressure/(self._gamma - 1)
        ke = y*y*y*y/2
        total_energy = ie + ke
        temperature = pressure / (self._gas_const * density)
        return make_conserved(dim=self._dim, mass=density, momentum=mom,
                              energy=total_energy), pressure, temperature


class IsentropicVortex(FluidManufacturedSolution):
    """Isentropic vortex from [Hesthaven_2008]_."""

    def __init__(
            self, dim, *, beta=5, center=(0, 0), velocity=(0, 0),
            gas_constant=287.
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
        # if eos is None:
        #    eos = IdealSingleGas()
        vortex_loc = self._center + t * self._velocity

        # coordinates relative to vortex center
        x_rel = x[0] - vortex_loc[0]
        y_rel = x[1] - vortex_loc[1]
        actx = x_rel.array_context

        gamma = 1.4  # eos.gamma()
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
        temperature = p / (mass*self._gas_const)
        return make_conserved(dim=2, mass=mass, energy=energy,
                              momentum=momentum), p, temperature


class TrigSolution1(FluidManufacturedSolution):
    """CNS manufactured solution designed to vanish on the domain boundary."""

    def __init__(self, dim, q_coeff, x_coeff, n=2, lx=None, nx=None,
                 gamma=1.4, gas_const=287.):
        """Initialize it."""
        super().__init__(dim, lx, gamma, gas_const)
        self._q_coeff = q_coeff
        # self._x_coeff = x_coeff

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


def _compute_mms_source(sym_operator, sym_soln, sym_t):
    return sym_diff(sym_soln)(sym_t) - sym_operator(sym_soln)


def sym_euler(sym_cv, sym_pressure, gamma=1.4, gas_constant=287.):
    """Return symbolic expression for the NS operator applied to a fluid state."""
    dim = sym_cv.dim
    rho = sym_cv.mass
    mom = sym_cv.momentum
    nrg = sym_cv.energy
    prs = sym_pressure
    vel = mom / rho

    # inviscid fluxes
    f_m_i = mom
    f_p_i = rho*(np.outer(vel, vel)) + prs*np.eye(dim)
    f_e_i = (nrg + prs)*vel
    f_i = make_obj_array([f_m_i, f_e_i, f_p_i])

    return -sym_div(f_i)


def sym_ns(sym_cv, sym_temperature, sym_pressure, mu=1, gamma=1.4, gas_constant=287.,
           prandtl=1.0):
    """Return symbolic expression for the NS operator applied to a fluid state."""
    dim = sym_cv.dim
    rho = sym_cv.mass
    mom = sym_cv.momentum
    nrg = sym_cv.energy
    prs = sym_pressure
    tmp = sym_temperature

    vel = mom/rho
    dvel = sym_grad(dim, vel)
    dtmp = sym_grad(dim, tmp)

    # pressure = (gamma-1)*(nrg - rho*np.dot(vel, vel)/2)

    # inviscid fluxes
    f_m_i = mom
    f_p_i = rho*(np.outer(vel, vel)) + prs*np.eye(dim)
    f_e_i = (nrg + prs)*vel
    f_i = make_obj_array([f_m_i, f_e_i, f_p_i])

    # viscous stress tensor
    tau = 2*mu/3*((dvel + dvel.T) - (dvel.trace))

    # heat flux
    kappa = gamma * gas_constant * mu / (prandtl * (gamma - 1))
    q_heat = -kappa * dtmp

    # viscous fluxes
    f_m_v = 0
    f_p_v = tau
    f_e_v = np.dot(tau, vel) - q_heat
    f_v = make_obj_array([f_m_v, f_e_v, f_p_v])

    return sym_div(f_v - f_i)


def test_ns_mms(actx_factory):
    """CNS manufactured solution tests."""
    actx = actx_factory()

    dim = 2
    sym_x = pmbl.make_sym_vector("x", dim)
    sym_t = pmbl.var("t")
    # q_coeff = ()
    # x_coeff = ()
    # man_soln = RoyManufacturedSolution(dim=dim, q_coeff=q_coeff, x_coeff=x_coeff)
    man_soln = UniformSolution()

    sym_cv, sym_prs, sym_tmp = man_soln.get_solution(sym_x, sym_t)
    print(f"{sym_cv=}")

    sym_source = -sym_euler(sym_cv, sym_prs)
    # sym_source = -sym_ns(sym_cv, sym_prs, sym_tmp)
    print(f"{sym_source=}")
    n = 2
    mesh = man_soln.get_mesh(n)

    order = 1
    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    from grudge.eager import EagerDGDiscretization
    from meshmode.discretization.poly_element import \
        QuadratureSimplexGroupFactory, \
        PolynomialWarpAndBlendGroupFactory
    discr = EagerDGDiscretization(
        actx, mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: PolynomialWarpAndBlendGroupFactory(order),
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(3*order),
        }
    )

    nodes = thaw(discr.nodes(), actx)
    print(f"{nodes=}")
    eval_mapper = EvaluationMapper(
            context=dict(x=nodes, t=0),
            zeros_factory=partial(discr.zeros, actx))

    source_eval = evaluate(sym_source, eval_mapper)
    print(f"{source_eval=}")

    pu.db
    cv_eval = evaluate(sym_cv, eval_mapper)
    print(f"{cv_eval=}")

    assert False

    # def get_rhs(t, u):
    #    # Hrm, need fluid state for this part....
    #    from mirgecom.gas_model import make_fluid_state
    #    fluid_state = make_fluid_state(...)   # uh
    #    return (ns_operator(
    #        discr, boundaries=man_soln.get_boundaries(discr, actx, t),
    #        state=fluid_state)
    #            + evaluate(sym_source, x=nodes, t=t))
    #
    # t = 0.
    # from mirgecom.integrators import rk4_step
    # dt = 1e-9
    # cv = man_soln.get_solution()
    # nsteps = 1

    # for _ in range(nsteps):
    #     cv = rk4_step(cv, t, dt, get_rhs)
    #     t += dt

    # expected_cv = sym_cv
