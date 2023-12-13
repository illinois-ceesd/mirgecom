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

import logging
from abc import ABCMeta, abstractmethod
import numpy as np
import pytest

import cantera
import pymbolic as pmbl
import grudge.op as op
from grudge.dof_desc import DISCR_TAG_QUAD, as_dofdesc
from pytools.convergence import EOCRecorder
from pytools.obj_array import flat_obj_array, make_obj_array
from meshmode.mesh.generation import generate_regular_rect_mesh
from meshmode.mesh import BTAG_ALL
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)
from meshmode.dof_array import DOFArray
import mirgecom.math as mm
from mirgecom.navierstokes import (
    ns_operator, grad_cv_operator, grad_t_operator
)
from mirgecom.fluid import (
    make_conserved, velocity_gradient, species_mass_fraction_gradient
)
from mirgecom.utils import force_evaluation
from mirgecom.boundary import DummyBoundary, PrescribedFluidBoundary
from mirgecom.eos import IdealSingleGas, PyrometheusMixture
from mirgecom.transport import SimpleTransport
from mirgecom.discretization import create_discretization_collection
from mirgecom.symbolic import (
    diff as sym_diff,
    evaluate)
from mirgecom.gas_model import GasModel, make_fluid_state
from mirgecom.simutil import (
    compare_fluid_solutions, componentwise_norms, get_box_mesh
)
from mirgecom.mechanisms import get_mechanism_input
from mirgecom.thermochemistry import get_pyrometheus_wrapper_class_from_cantera

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


logger = logging.getLogger(__name__)


@pytest.mark.parametrize("nspecies", [0, 10])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2, 5])
@pytest.mark.parametrize("use_overintegration", [False, True])
@pytest.mark.parametrize("periodic", [False, True])
def test_uniform_rhs(actx_factory, nspecies, dim, order, use_overintegration,
                     periodic):
    """Test the Navier-Stokes operator using a trivial constant/uniform state.

    This state should yield rhs = 0 to FP. Tests 1, 2, and 3 dimensions;
    with orders 1, 2, and 5; with and without passive species; with and without
    overintegration.
    """
    actx = actx_factory()

    tolerance = 4e-9 if dim < 3 else 3e-8

    def _conserved_vars(nodes, quiescent):
        zeros = actx.np.zeros_like(nodes[0])
        ones = zeros + 1.0

        mass_input = zeros + 1.0

        # set velocity profile
        mom_input = make_obj_array([zeros for i in range(dim)])
        if quiescent is False:
            for i in range(dim):
                mom_input[i] = zeros + (-1.0) ** i

        energy_input = 2.5 + 0.5*np.dot(mom_input, mom_input)/mass_input

        mass_frac_input = flat_obj_array(
            [ones / ((i + 1) * 10) for i in range(nspecies)]
        )
        species_mass_input = mass_input * mass_frac_input

        return make_conserved(
            dim, mass=mass_input, energy=energy_input, momentum=mom_input,
            species_mass=species_mass_input)

    eoc_rec0 = EOCRecorder()
    eoc_rec1 = EOCRecorder()
    for nel_1d in [2, 4, 8]:
        mesh = generate_regular_rect_mesh(
            a=(-0.5,)*dim, b=(0.5,)*dim, nelements_per_axis=(nel_1d,)*dim,
            periodic=(periodic,)*dim)

        boundaries = {} if periodic else {BTAG_ALL: DummyBoundary()}

        logger.info(f"Number of {dim}d elements: {mesh.nelements}")

        dcoll = create_discretization_collection(actx, mesh, order=order,
                                                 quadrature_order=2*order+1)
        quadrature_tag = DISCR_TAG_QUAD if use_overintegration else None
        nodes = force_evaluation(actx, dcoll.nodes())
        zeros = actx.np.zeros_like(nodes[0])

        cv = _conserved_vars(nodes=nodes, quiescent=True)

        expected_rhs = make_conserved(dim,
            mass=zeros,
            energy=zeros,
            momentum=make_obj_array([zeros for i in range(dim)]),
            species_mass=make_obj_array([zeros for i in range(nspecies)]))

        gas_model = GasModel(
            eos=IdealSingleGas(gamma=1.4, gas_const=1.0),
            transport=SimpleTransport(viscosity=1.0,
                                      thermal_conductivity=1.0,
                                      species_diffusivity=.5 * np.ones(nspecies))
        )
        state = make_fluid_state(gas_model=gas_model, cv=cv)
        assert actx.to_numpy(
            op.norm(dcoll, state.mass_density - cv.mass, np.inf)) < tolerance
        assert actx.to_numpy(
            op.norm(dcoll, state.energy_density - cv.energy, np.inf)) < tolerance

        ns_rhs, grad_cv, grad_t = ns_operator(
            dcoll, gas_model=gas_model, boundaries=boundaries, state=state,
            time=0.0, quadrature_tag=quadrature_tag, return_gradients=True)

        assert actx.to_numpy(op.norm(dcoll, ns_rhs.mass, np.inf)) < tolerance

        rhs_resid = ns_rhs - expected_rhs
        rho_resid = rhs_resid.mass/cv.mass
        rhoe_resid = rhs_resid.energy/cv.energy
        rhov_resid = rhs_resid.momentum  # don't normalize since it is zero
        rhoy_resid = rhs_resid.species_mass/cv.mass

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

        assert actx.to_numpy(op.norm(dcoll, rho_resid, np.inf)) < tolerance
        assert actx.to_numpy(op.norm(dcoll, rhoe_resid, np.inf)) < tolerance
        for i in range(dim):
            assert actx.to_numpy(op.norm(dcoll, rhov_resid[i], np.inf)) < tolerance
        for i in range(nspecies):
            assert actx.to_numpy(op.norm(dcoll, rhoy_resid[i], np.inf)) < tolerance

        err_max = actx.to_numpy(op.norm(dcoll, rho_resid, np.inf))
        eoc_rec0.add_data_point(1.0 / nel_1d, err_max)

        # set a non-zero, but uniform velocity component
        cv = _conserved_vars(nodes=nodes, quiescent=False)

        state = make_fluid_state(gas_model=gas_model, cv=cv)
        ns_rhs = ns_operator(dcoll, gas_model=gas_model, boundaries=boundaries,
                             state=state, time=0.0, quadrature_tag=quadrature_tag)

        rhs_resid = ns_rhs - expected_rhs
        rho_resid = rhs_resid.mass/cv.mass
        rhoe_resid = rhs_resid.energy/cv.energy
        rhov_resid = rhs_resid.momentum/cv.momentum
        rhoy_resid = rhs_resid.species_mass/cv.mass

        assert actx.to_numpy(op.norm(dcoll, rho_resid, np.inf)) < tolerance
        assert actx.to_numpy(op.norm(dcoll, rhoe_resid, np.inf)) < tolerance

        for i in range(dim):
            assert actx.to_numpy(op.norm(dcoll, rhov_resid[i], np.inf)) < tolerance
        for i in range(nspecies):
            assert actx.to_numpy(op.norm(dcoll, rhoy_resid[i], np.inf)) < tolerance

        err_max = actx.to_numpy(op.norm(dcoll, rho_resid, np.inf))
        eoc_rec1.add_data_point(1.0 / nel_1d, err_max)

    logger.info(
        f"V == 0 Errors:\n{eoc_rec0}"
        f"V != 0 Errors:\n{eoc_rec1}"
    )

    assert (eoc_rec0.order_estimate() >= order - 0.5
            or eoc_rec0.max_error() < tolerance)
    assert (eoc_rec1.order_estimate() >= order - 0.5
            or eoc_rec1.max_error() < tolerance)


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
        a = tuple(-lx_i/2 for lx_i in self._lx)
        b = tuple(lx_i/2 for lx_i in self._lx)
        return get_box_mesh(self.dim, a, b, n, periodic)

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
        periodic = (False,)*self._dim
        return get_box_mesh(self._dim, 0, 1, n, periodic=periodic)

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

        r2 = x_rel ** 2 + y_rel ** 2
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


# @pytest.mark.parametrize("nspecies", [0, 10])
@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize(("dim", "manufactured_soln", "mu"),
                         [(1, UniformSolution(dim=1), 0),
                          (2, UniformSolution(dim=2), 0),
                          (3, UniformSolution(dim=3), 0),
                          (2, UniformSolution(dim=2, velocity=[1, 1]), 0),
                          (3, UniformSolution(dim=3, velocity=[1, 1, 1]), 0),
                          (2, IsentropicVortex(), 0),
                          (2, IsentropicVortex(velocity=[1, 1]), 0),
                          (2, ShearFlow(mu=.01, gamma=3/2), .01)])
def test_exact_mms(actx_factory, order, dim, manufactured_soln, mu):
    """CNS manufactured solution tests."""
    # This is a test for the symbolic operators themselves.
    # It verifies that exact solutions to the Euler and NS systems actually
    # get RHS=0 when the symbolic operators are run on them.
    #
    # The solutions checked are:
    # - Uniform solution with and without momentum terms
    # - Isentropic vortex (explict time-dependence)
    # - ShearFlow exact soln for NS (not Euler)
    actx = actx_factory()

    sym_x = pmbl.make_sym_vector("x", dim)
    sym_t = pmbl.var("t")
    man_soln = manufactured_soln

    sym_cv, sym_prs, sym_tmp = man_soln.get_solution(sym_x, sym_t)

    logger.info(f"{sym_cv=}\n"
                f"{sym_cv.mass=}\n"
                f"{sym_cv.energy=}\n"
                f"{sym_cv.momentum=}\n"
                f"{sym_cv.species_mass=}")

    dcv_dt = sym_diff(sym_t)(sym_cv)
    print(f"{dcv_dt=}")

    from mirgecom.symbolic_fluid import sym_ns, sym_euler
    sym_ns_rhs = sym_ns(sym_cv, sym_prs, sym_tmp, mu)
    sym_euler_rhs = sym_euler(sym_cv, sym_prs)

    sym_ns_source = dcv_dt - sym_ns_rhs
    sym_euler_source = dcv_dt - sym_euler_rhs

    tol = 1e-15

    if mu == 0:
        assert actx.np.equal(sym_ns_source, sym_euler_source)
        sym_source = sym_euler_source
    else:
        sym_source = sym_ns_source

    logger.info(f"{sym_source=}")

    n = 1
    mesh = man_soln.get_mesh(n)

    dcoll = create_discretization_collection(actx, mesh, order)

    nodes = actx.thaw(dcoll.nodes())

    source_eval = evaluate(sym_source, t=0, x=nodes)

    source_norms = componentwise_norms(dcoll, source_eval)

    assert source_norms.mass < tol
    assert source_norms.energy < tol
    for i in range(dim):
        assert source_norms.momentum[i] < tol


@pytest.mark.parametrize(("dim", "flow_direction"),
                         [(2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("use_overintegration", [False, True])
def test_shear_flow(actx_factory, dim, flow_direction, order, use_overintegration):
    """Test the Navier-Stokes operator using an exact shear flow solution.

    The shear flow solution is defined in [Hesthaven_2008]_, Section 7.5.3
    and documented in :class:`~mirgecom.initializers.ShearFlow`.

    We expect convergence here to be *order* at best as we are checking
    the RHS directly, not a time-integrated solution, which takes far too
    long to perform in unit testing.
    """
    visualize = False  # set to True for debugging viz
    actx = actx_factory()
    transverse_direction = (flow_direction + 1) % dim

    mu = .01
    kappa = 0.

    eos = IdealSingleGas(gamma=3/2, gas_const=1.0)
    transport = SimpleTransport(viscosity=mu, thermal_conductivity=kappa)
    gas_model = GasModel(eos=eos, transport=transport)

    tol = 1e-8

    from mirgecom.initializers import ShearFlow as ExactShearFlow
    exact_soln = ExactShearFlow(dim=dim, mu=mu, flow_dir=flow_direction,
                                trans_dir=transverse_direction)

    # Only the energy eqn is non-trivial at all orders
    # Continuity eqn is solved exactly at any order
    eoc_energy = EOCRecorder()
    eoc_momentum = EOCRecorder()

    def _boundary_state_func(dcoll, dd_bdry, gas_model, state_minus, time=0,
                             **kwargs):
        actx = state_minus.array_context
        bnd_discr = dcoll.discr_from_dd(dd_bdry)
        nodes = actx.thaw(bnd_discr.nodes())
        boundary_cv = exact_soln(x_vec=nodes)
        return make_fluid_state(boundary_cv, gas_model)

    boundaries = {
        BTAG_ALL:
        PrescribedFluidBoundary(boundary_state_func=_boundary_state_func)
    }

    base_n = 4
    if dim > 2:
        base_n = 2

    for n in [1, 2, 4, 8]:
        nel_1d = n*base_n
        mesh = get_box_mesh(dim, (0,)*dim, (1,)*dim, n=(nel_1d,)*dim)

        dcoll = create_discretization_collection(actx, mesh, order=order,
                                                 quadrature_order=2*order+1)
        quadrature_tag = DISCR_TAG_QUAD if use_overintegration else None

        def inf_norm(x):
            return actx.to_numpy(op.norm(dcoll, x, np.inf))  # noqa

        nodes = actx.thaw(dcoll.nodes())
        # print(f"{nodes=}")

        cv_exact = exact_soln(x_vec=nodes)
        # print(f"{cv_exact=}")
        exact_fluid_state = make_fluid_state(cv=cv_exact, gas_model=gas_model)

        fluid_state = exact_fluid_state

        # Exact solution should have RHS=0, so the RHS itself is the residual
        ns_rhs, grad_cv, grad_t = \
            ns_operator(dcoll, gas_model, fluid_state, boundaries,
                        quadrature_tag=quadrature_tag, return_gradients=True)

        if visualize:
            from grudge.shortcuts import make_visualizer
            vis = make_visualizer(dcoll)
            if use_overintegration:
                viz_suffix = f"shear_flow_test_{dim}_{order}_{n}_over.vtu"
            else:
                viz_suffix = f"shear_flow_test_{dim}_{order}_{n}.vtu"
            vis.write_vtk_file(viz_suffix, [
                ("shear_flow", exact_fluid_state.cv),
                ("rhs", ns_rhs),
                ("grad(rhoU)", grad_cv.momentum[0]),
                ("grad(rhoV)", grad_cv.momentum[1]),
                ("grad(rhoE)", grad_cv.energy),
                ("P", fluid_state.pressure),
                ("T", fluid_state.temperature),
                ], overwrite=True)

        # print(f"{grad_cv=}")
        rhs_norms = componentwise_norms(dcoll, ns_rhs)

        assert inf_norm(grad_cv.momentum[flow_direction][transverse_direction]
                        - 2.0*nodes[transverse_direction]) < tol

        # these ones should be exact at all orders
        assert rhs_norms.mass < tol
        assert rhs_norms.momentum[transverse_direction] < tol

        # print(f"{rhs_norms=}")

        eoc_energy.add_data_point(1.0/nel_1d, actx.to_numpy(rhs_norms.energy))
        eoc_momentum.add_data_point(
            1.0/nel_1d, actx.to_numpy(rhs_norms.momentum[flow_direction]))

    assert (eoc_energy.order_estimate() >= order - 0.5
            or eoc_energy.max_error() < tol)

    assert (eoc_momentum.order_estimate() >= order - 0.5
            or eoc_momentum.max_error() < tol)


class RoySolution(FluidManufacturedSolution):
    """CNS manufactured solution from [Roy_2017]_."""

    def __init__(self, dim, q_coeff=None, x_coeff=None, lx=None, gamma=1.4,
                 gas_const=287.):
        """Initialize it."""
        super().__init__(dim=dim, lx=lx, gamma=gamma, gas_const=gas_const)
        if q_coeff is None:
            q_coeff = 0
        self._q_coeff = q_coeff
        self._x_coeff = x_coeff

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

    def get_mesh(self, n):
        """Get the mesh."""
        return super().get_mesh(n)

    def get_boundaries(self, dcoll, actx, t):
        """Get the boundaries."""
        return super().get_boundaries(dcoll, actx, t)


@pytest.mark.parametrize("order", [1])
@pytest.mark.parametrize(("dim", "u_0", "v_0", "w_0"),
                         [(1, 800, 0, 0),
                          (2, 800, 800, 0),
                          (1, 30, 0, 0),
                          (2, 40, 30, 0),
                          (2, 5, -20, 0)])
@pytest.mark.parametrize(("a_r", "a_p", "a_u", "a_v", "a_w"),
                         [(1.0, 2.0, .75, 2/3, 1/6)])
@pytest.mark.parametrize("use_overintegration", [False, True])
def test_roy_mms(actx_factory, order, dim, u_0, v_0, w_0, a_r, a_p, a_u,
                 a_v, a_w, use_overintegration):
    """CNS manufactured solution test from [Roy_2017]_."""
    actx = actx_factory()

    sym_x = pmbl.make_sym_vector("x", dim)
    sym_t = pmbl.var("t")
    q_coeff = [
        [2.0, .15, -.1, -.05],
        [1.e5, 2.e4, 5.e4, -1.e4],
        [u_0, u_0/20, -u_0/40, u_0/50],
        [v_0, -v_0/25, v_0/50, -v_0/100],
        [w_0, w_0/30, -w_0/25, -w_0/80]
    ]
    x_coeff = [
        [a_r, a_r/2, a_r/3],
        [a_p, a_p/10, a_p/5],
        [a_u, a_u/3, a_u/10],
        [a_v, a_v/6, a_v/3],
        [a_w, a_w/10, a_w/7]
    ]
    mu = 1.0
    gas_const = 287.
    prandtl = 1.0
    gamma = 1.4
    kappa = gamma * gas_const * mu / ((gamma - 1) * prandtl)

    eos = IdealSingleGas(gas_const=gas_const)
    transport_model = SimpleTransport(viscosity=mu,
                                      thermal_conductivity=kappa)
    gas_model = GasModel(eos=eos, transport=transport_model)

    man_soln = RoySolution(dim=dim, q_coeff=q_coeff, x_coeff=x_coeff, lx=None)

    sym_cv, sym_prs, sym_tmp = man_soln.get_solution(sym_x, sym_t)

    logger.info(f"{sym_cv=}\n"
                f"{sym_cv.mass=}\n"
                f"{sym_cv.energy=}\n"
                f"{sym_cv.momentum=}\n"
                f"{sym_cv.species_mass=}")

    dcv_dt = sym_diff(sym_t)(sym_cv)
    print(f"{dcv_dt=}")

    from mirgecom.symbolic_fluid import sym_ns
    sym_ns_rhs = sym_ns(sym_cv, sym_prs, sym_tmp, mu=mu, kappa=kappa)

    sym_ns_source = dcv_dt - sym_ns_rhs

    tol = 1e-12

    sym_source = sym_ns_source

    logger.info(f"{sym_source=}")

    eoc_rec = EOCRecorder()

    n0 = 4

    for n in [n0, 2*n0, 4*n0]:

        mesh = man_soln.get_mesh(n)

        quadrature_tag = DISCR_TAG_QUAD if use_overintegration else None
        dcoll = create_discretization_collection(actx, mesh, order=order,
                                                 quadrature_order=2*order+1)
        nodes = actx.thaw(dcoll.nodes())

        from grudge.dt_utils import characteristic_lengthscales
        char_len = actx.to_numpy(
            op.norm(dcoll, characteristic_lengthscales(actx, dcoll), np.inf)
        )

        source_eval = evaluate(sym_source, t=0, x=nodes)
        cv_exact = evaluate(sym_cv, t=0, x=nodes)

        # Sanity check the dependent quantities
        # tmp_exact = evaluate(sym_tmp, t=0, x=nodes)
        # tmp_eos = eos.temperature(cv=cv_exact)
        # prs_exact = evaluate(sym_prs, t=0, x=nodes)
        # prs_eos = eos.pressure(cv=cv_exact)
        # prs_resid = (prs_exact - prs_eos)/prs_exact
        # tmp_resid = (tmp_exact - tmp_eos)/tmp_exact
        # prs_err = actx.to_numpy(op.norm(dcoll, prs_resid, np.inf))
        # tmp_err = actx.to_numpy(op.norm(dcoll, tmp_resid, np.inf))

        # print(f"{prs_exact=}\n{prs_eos=}")
        # print(f"{tmp_exact=}\n{tmp_eos=}")

        # assert prs_err < tol
        # assert tmp_err < tol

        if isinstance(source_eval.mass, DOFArray):
            source_norms = componentwise_norms(dcoll, source_eval)
        else:
            source_norms = source_eval

        logger.info(f"{source_norms=}")
        logger.info(f"{source_eval=}")

        def _boundary_state_func(dcoll, dd_bdry, gas_model, state_minus, time=0,
                                 **kwargs):
            actx = state_minus.array_context
            bnd_discr = dcoll.discr_from_dd(dd_bdry)
            nodes = actx.thaw(bnd_discr.nodes())
            boundary_cv = evaluate(sym_cv, x=nodes, t=time)
            return make_fluid_state(boundary_cv, gas_model)

        boundaries = {
            BTAG_ALL:
            PrescribedFluidBoundary(boundary_state_func=_boundary_state_func)
        }

        from mirgecom.simutil import max_component_norm
        err_scale = max_component_norm(dcoll, cv_exact)

        def get_rhs(t, cv):
            fluid_state = make_fluid_state(cv=cv, gas_model=gas_model)
            rhs_val = ns_operator(dcoll, gas_model, fluid_state, boundaries,
                                  quadrature_tag=quadrature_tag) + source_eval
            print(f"{max_component_norm(dcoll, rhs_val/err_scale)=}")
            return rhs_val

        t = 0.
        from mirgecom.integrators import rk4_step
        dt = 1e-9
        nsteps = 10
        cv = cv_exact
        print(f"{cv.dim=}")
        print(f"{cv=}")

        for _ in range(nsteps):
            cv = rk4_step(cv, t, dt, get_rhs)
            t += dt

        soln_resid = compare_fluid_solutions(dcoll, cv, cv_exact)
        cv_err_scales = componentwise_norms(dcoll, cv_exact)

        max_err = soln_resid[0]/cv_err_scales.mass
        max_err = max(max_err, soln_resid[1]/cv_err_scales.energy)
        for i in range(dim):
            max_err = max(soln_resid[2+i]/cv_err_scales.momentum[i], max_err)
        max_err = actx.to_numpy(max_err)
        print(f"{max_err=}")
        eoc_rec.add_data_point(char_len, max_err)

    logger.info(
        f"{eoc_rec=}"
    )

    assert (
        eoc_rec.order_estimate() >= order - 0.5
        or eoc_rec.max_error() < tol
    )


@pytest.mark.parametrize("nspecies", [0, 3])
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_projection_to_quad_domain(actx_factory, nspecies, dim, order):
    """Test the projection for overintegration."""
    actx = actx_factory()

    tol = 1e-9

    def _conserved_vars(nodes):
        zeros = actx.np.zeros_like(nodes[0])
        ones = zeros + 1.0

        mass_input = 0.5 + 0.1*actx.np.cos(2.0*np.pi*nodes[0])

        temperature_input = 1.0 + 0.1*actx.np.sin(2.0*np.pi*nodes[0])

        mom_input = make_obj_array([zeros for i in range(dim)])
        mom_input[0] = 0.2 + 0.1*nodes[0]

        mass_frac_input = flat_obj_array(
            [ones / ((i + 1) * 10) for i in range(nspecies)]
        )
        species_mass_input = mass_input * mass_frac_input

        energy_input = mass_input*(
            gas_model.eos.get_internal_energy(temperature=temperature_input)) \
            + 0.5*np.dot(mom_input, mom_input)/mass_input

        return make_conserved(dim, mass=mass_input, momentum=mom_input,
            energy=energy_input, species_mass=species_mass_input)

    def error(x):
        return actx.to_numpy(op.norm(dcoll, x, np.inf))  # noqa

    boundaries = {BTAG_ALL: DummyBoundary()}

    for use_overintegration in [True]:

        eoc0 = EOCRecorder()
        eoc1 = EOCRecorder()
        eoc2 = EOCRecorder()
        eoc3 = EOCRecorder()
        eoc4 = EOCRecorder()
        for nel_1d in [16, 32, 64]:
            mesh = generate_regular_rect_mesh(
                a=(-1.0,)*dim, b=(1.0,)*dim, nelements_per_axis=(nel_1d,)*dim)

            logger.info(f"Number of {dim}d elements: {mesh.nelements}")

            dcoll = create_discretization_collection(actx, mesh, order=order,
                                                     quadrature_order=2*order+1)
            quadrature_tag = DISCR_TAG_QUAD if use_overintegration else None
            nodes = force_evaluation(actx, dcoll.nodes())

            dd_vol = as_dofdesc("vol")
            dd_vol_quad = dd_vol.with_discr_tag(quadrature_tag)

            gas_model = GasModel(eos=IdealSingleGas(gamma=1.4, gas_const=1.0))

            cv = _conserved_vars(nodes)
            state = make_fluid_state(gas_model=gas_model, cv=cv)

            # ~~~ project_everything
            project_full_cv = op.project(dcoll, dd_vol, dd_vol_quad, cv)
            state_from_proj_cv = make_fluid_state(gas_model=gas_model,
                                                  cv=project_full_cv)

            # ~~~ project cv + internal energy
            int_energy = gas_model.eos.internal_energy(cv)
            project_cv_and_e = op.project(dcoll, dd_vol, dd_vol_quad, cv)
            project_e = op.project(dcoll, dd_vol, dd_vol_quad, int_energy)
            kin_energy = 0.5/project_cv_and_e.mass*np.dot(project_cv_and_e.momentum,
                                                          project_cv_and_e.momentum)
            project_cv_and_e = project_cv_and_e.replace(energy=project_e+kin_energy)
            state_from_proj_cv_and_e = make_fluid_state(gas_model=gas_model,
                                                        cv=project_cv_and_e)

            # ~~~ project cv + temperature
            project_cv_and_t = op.project(dcoll, dd_vol, dd_vol_quad, cv)
            project_t = op.project(dcoll, dd_vol, dd_vol_quad, state.temperature)
            int_energy = \
                project_cv_and_t.mass*gas_model.eos.get_internal_energy(project_t)
            kin_energy = 0.5/project_cv_and_t.mass*np.dot(project_cv_and_t.momentum,
                                                          project_cv_and_t.momentum)
            project_cv_and_t = project_cv_and_t.replace(energy=int_energy+kin_energy)
            state_from_proj_cv_and_t = make_fluid_state(gas_model=gas_model,
                                                        cv=project_cv_and_t)

            # ~~~ exact solution
            overint_nodes = op.project(dcoll, dd_vol, dd_vol_quad, nodes)
            x = overint_nodes[0]

            # exact_mass = 0.5 + 0.1*actx.np.cos(2.0*np.pi*x)
            exact_temp = 1.0 + 0.1*actx.np.sin(2.0*np.pi*x)

            grad_mass_exact = -0.1*2.0*np.pi*actx.np.sin(2.0*np.pi*nodes[0])
            grad_temp_exact = +0.1*2.0*np.pi*actx.np.cos(2.0*np.pi*nodes[0])

            error_temp_1 = error(state_from_proj_cv.temperature - exact_temp)
            eoc0.add_data_point(1.0 / nel_1d, error_temp_1)
            error_temp_2 = error(state_from_proj_cv_and_e.temperature - exact_temp)
            eoc1.add_data_point(1.0 / nel_1d, error_temp_2)
            error_temp_3 = error(state_from_proj_cv_and_t.temperature - exact_temp)
            eoc2.add_data_point(1.0 / nel_1d, error_temp_3)

            grad_temp = grad_t_operator(dcoll, gas_model, boundaries, state,
                                        quadrature_tag=quadrature_tag)

            error_grad_temp = error(grad_temp[0] - grad_temp_exact)
            eoc3.add_data_point(1.0 / nel_1d, error_grad_temp)

            grad_cv = grad_cv_operator(dcoll, gas_model, boundaries, state,
                                       quadrature_tag=quadrature_tag)

            error_grad_mass = error(grad_cv.mass[0] - grad_mass_exact)
            eoc4.add_data_point(1.0 / nel_1d, error_grad_mass)

        assert (eoc0.order_estimate() >= order+1 - 0.5 or eoc0.max_error() < tol)
        assert (eoc1.order_estimate() >= order+1 - 0.5 or eoc1.max_error() < tol)
        assert (eoc2.order_estimate() >= order+1 - 0.5 or eoc2.max_error() < tol)
        assert (eoc3.order_estimate() >= order - 0.5 or eoc3.max_error() < tol)
        assert (eoc4.order_estimate() >= order - 0.5 or eoc4.max_error() < tol)


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("use_overintegration", [False, True])
def test_gradients_mixture(actx_factory, dim, order, use_overintegration):
    """Test the gradients from the Navier-Stokes operator with non-uniform profiles.

    This state should yield ...
    """
    actx = actx_factory()

    tol = 1e-7

    boundaries = {BTAG_ALL: DummyBoundary()}
    quadrature_tag = DISCR_TAG_QUAD if use_overintegration else None

    # Pyrometheus initialization
    mechname = "uiuc_7sp"
    mech_input = get_mechanism_input(mechname)
    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    pyro_obj = get_pyrometheus_wrapper_class_from_cantera(
        cantera_soln, temperature_niter=3)(actx.np)

    nspecies = pyro_obj.num_species
    print(f"PyrometheusMixture::NumSpecies = {nspecies}")

    tseed = 500.0

    i_fu = cantera_soln.species_index("H2")
    i_ox = cantera_soln.species_index("O2")
    i_di = cantera_soln.species_index("N2")

    # Transport data initilization
    eos = PyrometheusMixture(pyro_obj, temperature_guess=tseed)
    gas_model = GasModel(eos=eos)

    def _conserved_vars(nodes):
        zeros = actx.np.zeros_like(nodes[0])

        pressure = 101325.0 + 1000.0*nodes[0]
        temperature = tseed + 50.0*nodes[0]

        velocity = make_obj_array([zeros for i in range(dim)])
        velocity[0] = 100.0 + 30.0*nodes[0]

        mass_frac = make_obj_array([zeros for _ in range(nspecies)])
        mass_frac[i_fu] = 0.2 + 0.1*nodes[0]
        mass_frac[i_ox] = 0.2 - 0.1*nodes[0]
        mass_frac[i_di] = 1.0 - (mass_frac[i_fu] + mass_frac[i_ox])

        mass = eos.get_density(pressure, temperature, mass_frac)
        species_mass = mass * mass_frac
        momentum = mass*velocity

        energy_input = mass*(
            gas_model.eos.get_internal_energy(temperature, mass_frac)
            + 0.5*np.dot(velocity, velocity))

        return make_conserved(dim, mass=mass, momentum=momentum,
            energy=energy_input, species_mass=species_mass)

    def inf_norm(x):
        return actx.to_numpy(op.norm(dcoll, x, np.inf))

    eoc_rec0 = EOCRecorder()
    eoc_rec1 = EOCRecorder()
    eoc_rec2 = EOCRecorder()
    eoc_rec3 = EOCRecorder()
    eoc_rec4 = EOCRecorder()
    for nel_1d in [4, 8, 16, 32]:
        mesh = generate_regular_rect_mesh(
            a=(-1.0,)*dim, b=(1.0,)*dim, nelements_per_axis=(nel_1d,)*dim)

        logger.info(f"Number of {dim}d elements: {mesh.nelements}")

        dcoll = create_discretization_collection(actx, mesh, order=order,
                                                 quadrature_order=2*order+1)
        nodes = force_evaluation(actx, dcoll.nodes())

        cv = _conserved_vars(nodes)
        state = make_fluid_state(gas_model=gas_model, cv=cv,
                                 temperature_seed=tseed)

        grad_cv = grad_cv_operator(dcoll, gas_model, boundaries, state,
                                   quadrature_tag=quadrature_tag)

        grad_v = velocity_gradient(cv, grad_cv)
        grad_y = species_mass_fraction_gradient(cv, grad_cv)

        grad_temp = grad_t_operator(dcoll, gas_model, boundaries, state,
                                    quadrature_tag=quadrature_tag)

        grad_u = 30.0
        grad_t = 50.0

        err_grad_y0 = inf_norm(grad_y[i_fu][0] - 0.1)
        err_grad_y1 = inf_norm(grad_y[i_ox][0] + 0.1)
        err_grad_y2 = inf_norm(grad_y[i_di][0] - 0.0)
        err_grad_u = inf_norm(grad_v[0][0] - grad_u)
        err_grad_t = inf_norm(grad_temp[0] - grad_t)

        eoc_rec0.add_data_point(1.0 / nel_1d, err_grad_y0)
        eoc_rec1.add_data_point(1.0 / nel_1d, err_grad_y1)
        eoc_rec2.add_data_point(1.0 / nel_1d, err_grad_y2)
        eoc_rec3.add_data_point(1.0 / nel_1d, err_grad_u)
        eoc_rec4.add_data_point(1.0 / nel_1d, err_grad_t)

    print(order)
    print(eoc_rec0)
    print(eoc_rec1)
    print(eoc_rec2)
    print(eoc_rec3)
    print(eoc_rec4)

    assert eoc_rec0.order_estimate() >= order - 0.5 or eoc_rec0.max_error() < tol
    assert eoc_rec1.order_estimate() >= order - 0.5 or eoc_rec1.max_error() < tol
    assert eoc_rec2.order_estimate() >= order - 0.5 or eoc_rec2.max_error() < tol
    assert eoc_rec3.order_estimate() >= order - 0.5 or eoc_rec3.max_error() < tol
    assert eoc_rec4.order_estimate() >= order - 0.5 or eoc_rec4.max_error() < tol


@pytest.mark.parametrize("nspecies", [0, 3])
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("use_overintegration", [False, True])
def test_gradients(actx_factory, nspecies, dim, order, use_overintegration):
    """Test the gradients from the N-S operator with non-uniform profiles."""
    actx = actx_factory()

    tol = 2e-9

    boundaries = {BTAG_ALL: DummyBoundary()}

    quadrature_tag = DISCR_TAG_QUAD if use_overintegration else None

    eos = IdealSingleGas(gamma=1.4, gas_const=1.0)
    gas_model = GasModel(eos=eos)

    def _conserved_vars(nodes):
        zeros = actx.np.zeros_like(nodes[0])
        ones = zeros + 1.0

        mass_input = 0.5 + 0.1*nodes[0]
        temperature_input = 1.0 + 0.1*nodes[0]

        vel_input = make_obj_array([zeros for i in range(dim)])
        vel_input[0] = 0.2 + 0.1*nodes[0]
        mom_input = mass_input*vel_input

        mass_frac_input = flat_obj_array(
            [ones / ((i + 1) * 10) for i in range(nspecies)]
        )
        species_mass_input = mass_input * mass_frac_input

        energy_input = mass_input*(
            gas_model.eos.get_internal_energy(temperature=temperature_input)) \
            + 0.5*np.dot(mom_input, mom_input)/mass_input

        return make_conserved(dim, mass=mass_input, momentum=mom_input,
            energy=energy_input, species_mass=species_mass_input)

    def inf_norm(x):
        return actx.to_numpy(op.norm(dcoll, x, np.inf))

    eoc_rec0 = EOCRecorder()
    eoc_rec1 = EOCRecorder()
    eoc_rec2 = EOCRecorder()
    eoc_rec3 = EOCRecorder()
    eoc_rec4 = EOCRecorder()
    for nel_1d in [2, 4, 8, 16, 32, 64]:
        mesh = generate_regular_rect_mesh(
            a=(-1.0,)*dim, b=(1.0,)*dim, nelements_per_axis=(nel_1d,)*dim)

        logger.info(f"Number of {dim}d elements: {mesh.nelements}")

        dcoll = create_discretization_collection(actx, mesh, order=order,
                                                 quadrature_order=2*order+1)
        nodes = force_evaluation(actx, dcoll.nodes())

        cv = _conserved_vars(nodes)
        state = make_fluid_state(gas_model=gas_model, cv=cv)

        grad_cv = grad_cv_operator(dcoll, gas_model, boundaries, state,
                                   quadrature_tag=quadrature_tag)

        grad_v = velocity_gradient(cv, grad_cv)

        grad_temp = grad_t_operator(dcoll, gas_model, boundaries, state,
                                    quadrature_tag=quadrature_tag)

        x = nodes[0]
        grad_rho = 0.1
        grad_rhou = 0.07+0.02*x
        grad_rhoe = (1.0/0.4)*(0.02*x + 0.15) + 0.5*(0.003*x**2 + 0.018*x + 0.024)
        grad_u = 0.1
        grad_t = 0.1

        err_grad_rho = inf_norm(grad_cv.mass[0] - grad_rho)
        err_grad_rhoe = inf_norm(grad_cv.energy[0] - grad_rhoe)
        err_grad_rhou = inf_norm(grad_cv.momentum[0][0] - grad_rhou)
        err_grad_u = inf_norm(grad_v[0][0] - grad_u)
        err_grad_t = inf_norm(grad_temp[0] - grad_t)

        eoc_rec0.add_data_point(1.0 / nel_1d, err_grad_rho)
        eoc_rec1.add_data_point(1.0 / nel_1d, err_grad_rhoe)
        eoc_rec2.add_data_point(1.0 / nel_1d, err_grad_rhou)
        eoc_rec3.add_data_point(1.0 / nel_1d, err_grad_u)
        eoc_rec4.add_data_point(1.0 / nel_1d, err_grad_t)

    assert eoc_rec0.order_estimate() >= order - 0.5 or eoc_rec0.max_error() < tol
    assert eoc_rec1.order_estimate() >= order - 0.5 or eoc_rec1.max_error() < tol
    assert eoc_rec2.order_estimate() >= order - 0.5 or eoc_rec2.max_error() < tol
    assert eoc_rec3.order_estimate() >= order - 0.5 or eoc_rec3.max_error() < tol
    assert eoc_rec4.order_estimate() >= order - 0.5 or eoc_rec4.max_error() < tol
