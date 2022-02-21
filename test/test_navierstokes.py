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


class RoyManufacturedSolution(FluidCase):
    """CNS manufactured solution from [Roy_2017]__."""

    def __init__(self, dim, gas_model):
        """Initialize it."""
        super().__init__(dim)

    def get_mesh(self, n=2, nx=None, lx=None, periodic=None):
        """Return the mesh."""
        if lx is None:
            lx = (2*np.pi,)*self._dim
        if len(lx) != self._dim:
            raise ValueError("Improper dimension for lx.")
        if nx is None:
            nx = (n,)*self._dim
        if len(nx) != self._dim:
            raise ValueError("Improper dimension for nx.")
        self._lx = lx
        a = -self._lx/2
        b = self._lx/2
        return _get_box_mesh(self.dim, a, b, nx, periodic)

    def get_solution(self, x, t):
        """Return the symbolically-compatible solution."""
        c = self._q_coeff[0]
        ar = self._x_coeff[0]
        lx = self._lx
        omega_x = [np.pi*x[i]/lx[i] for i in range(self._dim)]

        # rho = rho_0 + rho_x*sin(ar_x*pi*x/L_x) + rho_y*cos(ar_y*pi*y/L_y)
        #       + rho_z*sin(ar_z*pi*z/L_z)
        density = (c[0] + c[1]*mm.sin(ar[0]*omega_x[0]))
        if self._dim > 1:
            density = density + c[2]*mm.cos(ar[1]*omega_x[1])
        if self._dim > 2:
            density = density + c[3]*mm.sin(ar[2]*omega_x[2])

        # p = p_0 + p_x*cos(ap_x*pi*x/L_x) + p_y*sin(ap_y*pi*y/L_y)
        #     + p_z*cos(ap_z*pi*z/L_z)
        c = self._q_coeff[1]
        ap = self._x_coeff[1]
        press = c[0] + c[1]*mm.cos(ap[0]*omega_x[1])
        if self._dim > 1:
            press = press + c[2]*mm.sin(ap[1]*omega_x[2])
        if self._dim > 2:
            press = press + c[3]*mm.cos(ap[2]*omega_x[3])

        c = self._q_coeff[2]
        au = self._x_coeff[2]
        # u = u_0 + u_x*sin(au_x*pi*x/L_x) + u_y*cos(au_y*pi*y/L_y)
        #       + u_z*cos(au_z*pi*z/L_z)
        u = (c[0] + c[1]*mm.sin(au[0]*omega_x[0]))
        if self._dim > 1:
            u = u + c[2]*mm.cos(au[1]*omega_x[1])
        if self._dim > 2:
            u = u + c[3]*mm.cos(au[2]*omega_x[2])

        if self._dim > 1:
            c = self._q_coeff[3]
            av = self._x_coeff[3]
            # v = v_0 + v_x*cos(av_x*pi*x/L_x) + v_y*sin(av_y*pi*y/L_y)
            #       + v_z*sin(av_z*pi*z/L_z)
            v = (c[0] + c[1]*mm.cos(av[0]*omega_x[0])
                 + c[2]*mm.sin(av[1]*omega_x[1]))
            if self._dim > 2:
                v = v + c[3]*mm.sin(av[2]*omega_x[2])
            if self._dim > 2:
                c = self._q_coeff[4]
                aw = self._x_coeff[4]
                # w = w_0 + w_x*sin(aw_x*pi*x/L_x) + w_y*sin(aw_y*pi*y/L_y)
                #       + w_z*cos(aw_z*pi*z/L_z)
                w = (c[0] + c[1]*mm.sin(aw[0]*omega_x[0])
                     + c[2]*mm.sin(aw[1]*omega_x[1])
                     + c[3]*mm.cos(aw[2]*omega_x[2]))

        if self._dim == 1:
            velocity = make_obj_array([u])
        if self._dim == 2:
            velocity = make_obj_array([u, v])
        if self._dim == 3:
            velocity = make_obj_array([u, v, w])

        mom = density*velocity
        energy = press/(self._gamma - 1) + density*np.dot(velocity, velocity)
        return make_conserved(dim=self._dim, mass=density, momentum=mom,
                              energy=energy)

    def get_boundaries(self, discr, actx, t):

        def _boundary_state_func(discr, btag, gas_model, state_minus, **kwargs):
            actx = state_minus.array_context
            bnd_discr = discr.discr_from_dd(btag)
            nodes = thaw(bnd_discr.nodes(), actx)
            # need to evaluate symbolic soln for this part
            return make_fluid_state(self.get_solution(x=nodes, t=t), gas_model)

        return {BTAG_ALL :
                PrescribedFluidBoundary(boundary_state_func=_boundary_state_func)}


def sym_ns(sym_cv):
    """Return symbolic expression for the NS operator applied to a fluid state."""
    dim = sym_cv.dim
    rho = sym_cv.mass
    mom = sym_cv.momentum
    nrg = sym_cv.energy

    vel = mom/rho
    pressure = (gamma-1)*(nrg - rho*np.dot(vel, vel)/2)

    f_m_i = mom
    f_p_i = rho*(np.outer(vel, vel)) + pressure*np.eye(dim)
    f_e_i = (nrg + pressure)*vel

    gv = sim_grad(dim, vel)
    return sym_div(sym_alpha * sym_grad(dim, sym_u))


# Note: Must integrate in time for a while in order to achieve expected spatial
# accuracy. Checking the RHS alone will give lower numbers.
#
# Working hypothesis: RHS lives in lower order polynomial space and thus doesn't
# attain full-order convergence.
@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize(("problem", "nsteps", "dt", "scales"),
    [
        (DecayingTrigTruncatedDomain(1, 2.), 50, 5.e-5, [8, 16, 24]),
        (DecayingTrigTruncatedDomain(2, 2.), 50, 5.e-5, [8, 12, 16]),
        (DecayingTrigTruncatedDomain(3, 2.), 50, 5.e-5, [8, 10, 12]),
        (OscillatingTrigVarDiff(1), 50, 5.e-5, [8, 16, 24]),
        (OscillatingTrigVarDiff(2), 50, 5.e-5, [12, 14, 16]),
        (OscillatingTrigNonlinearDiff(1), 50, 5.e-5, [8, 16, 24]),
        (OscillatingTrigNonlinearDiff(2), 50, 5.e-5, [12, 14, 16]),
    ])
def test_diffusion_accuracy(actx_factory, problem, nsteps, dt, scales, order,
            visualize=False):
    """
    Checks the accuracy of the diffusion operator by solving the heat equation for a
    given problem setup.
    """
    actx = actx_factory()

    p = problem

    sym_x = pmbl.make_sym_vector("x", p.dim)
    sym_t = pmbl.var("t")
    sym_u = p.get_solution(sym_x, sym_t)
    sym_alpha = p.get_alpha(sym_x, sym_t, sym_u)

    sym_diffusion_u = sym_diffusion(p.dim, sym_alpha, sym_u)

    # In order to support manufactured solutions, we modify the heat equation
    # to add a source term f. If the solution is exact, this term should be 0.
    sym_f = sym_diff(sym_t)(sym_u) - sym_diffusion_u

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for n in scales:
        mesh = p.get_mesh(n)

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

        nodes = thaw(actx, discr.nodes())

        def get_rhs(t, u):
            alpha = p.get_alpha(nodes, t, u)
            if isinstance(alpha, DOFArray):
                discr_tag = DISCR_TAG_QUAD
            else:
                discr_tag = DISCR_TAG_BASE
            return (diffusion_operator(discr, quad_tag=discr_tag, alpha=alpha,
                    boundaries=p.get_boundaries(discr, actx, t), u=u)
                + evaluate(sym_f, x=nodes, t=t))

        t = 0.

        u = p.get_solution(nodes, t)

        from mirgecom.integrators import rk4_step

        for _ in range(nsteps):
            u = rk4_step(u, t, dt, get_rhs)
            t += dt

        expected_u = p.get_solution(nodes, t)

        rel_linf_err = actx.to_numpy(
            discr.norm(u - expected_u, np.inf)
            / discr.norm(expected_u, np.inf))
        eoc_rec.add_data_point(1./n, rel_linf_err)

        if visualize:
            from grudge.shortcuts import make_visualizer
            vis = make_visualizer(discr, discr.order+3)
            vis.write_vtk_file("diffusion_accuracy_{order}_{n}.vtu".format(
                order=order, n=n), [
                    ("u", u),
                    ("expected_u", expected_u),
                    ])

    print("L^inf error:")
    print(eoc_rec)
    # Expected convergence rates from Hesthaven/Warburton book
    expected_order = order+1 if order % 2 == 0 else order
    assert(eoc_rec.order_estimate() >= expected_order - 0.5
                or eoc_rec.max_error() < 1e-11)
