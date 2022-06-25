"""Test the inviscid fluid module."""

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
import pytest

from pytools.obj_array import (
    flat_obj_array,
    make_obj_array,
)

from arraycontext import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.trace_pair import TracePair
from mirgecom.fluid import make_conserved
from mirgecom.eos import IdealSingleGas
from mirgecom.discretization import create_discretization_collection
import grudge.op as op
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)
from mirgecom.inviscid import (
    inviscid_flux,
    inviscid_facial_flux_rusanov,
    inviscid_facial_flux_hll
)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("nspecies", [0, 1, 10])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_inviscid_flux(actx_factory, nspecies, dim):
    """Check inviscid flux against exact expected result: Identity test.

    Directly check inviscid flux routine, :func:`mirgecom.inviscid.inviscid_flux`,
    against the exact expected result. This test is designed to fail if the flux
    routine is broken.

    The expected inviscid flux is:
      F(q) = <rhoV, (E+p)V, rho(V.x.V) + pI, rhoY V>
    """
    actx = actx_factory()

    nel_1d = 16

    from meshmode.mesh.generation import generate_regular_rect_mesh

    #    for dim in [1, 2, 3]:
    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 3
    discr = create_discretization_collection(actx, mesh, order=order)
    eos = IdealSingleGas()

    logger.info(f"Number of {dim}d elems: {mesh.nelements}")

    def rand():
        from meshmode.dof_array import DOFArray
        return DOFArray(
            actx,
            tuple(actx.from_numpy(np.random.rand(grp.nelements, grp.nunit_dofs))
                  for grp in discr.discr_from_dd("vol").groups)
        )

    mass = rand()
    energy = rand()
    mom = make_obj_array([rand() for _ in range(dim)])

    mass_fractions = make_obj_array([rand() for _ in range(nspecies)])
    species_mass = mass * mass_fractions

    cv = make_conserved(dim, mass=mass, energy=energy, momentum=mom,
                        species_mass=species_mass)

    # {{{ create the expected result

    p = eos.pressure(cv)
    escale = (energy + p) / mass

    numeq = dim + 2 + nspecies

    expected_flux = np.zeros((numeq, dim), dtype=object)
    expected_flux[0] = mom
    expected_flux[1] = mom * escale

    for i in range(dim):
        for j in range(dim):
            expected_flux[2+i, j] = (mom[i] * mom[j] / mass + (p if i == j else 0))

    for i in range(nspecies):
        expected_flux[dim+2+i] = mom * mass_fractions[i]

    expected_flux = make_conserved(dim, q=expected_flux)

    # }}}

    from mirgecom.gas_model import GasModel, make_fluid_state
    gas_model = GasModel(eos=eos)
    state = make_fluid_state(cv, gas_model)

    flux = inviscid_flux(state)
    flux_resid = flux - expected_flux

    for i in range(numeq, dim):
        for j in range(dim):
            assert (la.norm(flux_resid[i, j].get())) == 0.0


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_inviscid_flux_components(actx_factory, dim):
    """Test uniform pressure case.

    Checks that the Euler-internal inviscid flux routine
    :func:`mirgecom.inviscid.inviscid_flux` returns exactly the expected result
    with a constant pressure and no flow.

    Expected inviscid flux is:
      F(q) = <rhoV, (E+p)V, rho(V.x.V) + pI>

    Checks that only diagonal terms of the momentum flux:
      [ rho(V.x.V) + pI ] are non-zero and return the correctly calculated p.
    """
    actx = actx_factory()

    eos = IdealSingleGas()

    p0 = 1.0

    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 3
    discr = create_discretization_collection(actx, mesh, order=order)
    eos = IdealSingleGas()

    logger.info(f"Number of {dim}d elems: {mesh.nelements}")
    # === this next block tests 1,2,3 dimensions,
    # with single and multiple nodes/states. The
    # purpose of this block is to ensure that when
    # all components of V = 0, the flux recovers
    # the expected values (and p0 within tolerance)
    # === with V = 0, fixed P = p0
    tolerance = 1e-15
    nodes = thaw(discr.nodes(), actx)
    mass = discr.zeros(actx) + np.dot(nodes, nodes) + 1.0
    mom = make_obj_array([discr.zeros(actx) for _ in range(dim)])
    p_exact = discr.zeros(actx) + p0
    energy = p_exact / 0.4 + 0.5 * np.dot(mom, mom) / mass
    cv = make_conserved(dim, mass=mass, energy=energy, momentum=mom)
    p = eos.pressure(cv)

    from mirgecom.gas_model import GasModel, make_fluid_state
    state = make_fluid_state(cv, GasModel(eos=eos))

    flux = inviscid_flux(state)

    def inf_norm(x):
        return actx.to_numpy(op.norm(discr, x, np.inf))

    assert inf_norm(p - p_exact) < tolerance
    logger.info(f"{dim}d flux = {flux}")

    # for velocity zero, these components should be == zero
    assert inf_norm(flux.mass) == 0.0
    assert inf_norm(flux.energy) == 0.0

    # The momentum diagonal should be p
    # Off-diagonal should be identically 0
    assert inf_norm(flux.momentum - p0*np.identity(dim)) < tolerance


@pytest.mark.parametrize(("dim", "livedim"), [
    (1, 0),
    (2, 0),
    (2, 1),
    (3, 0),
    (3, 1),
    (3, 2),
    ])
def test_inviscid_mom_flux_components(actx_factory, dim, livedim):
    r"""Test components of the momentum flux with constant pressure, V != 0.

    Checks that the flux terms are returned in the proper order by running
    only 1 non-zero velocity component at-a-time.
    """
    actx = actx_factory()

    eos = IdealSingleGas()

    p0 = 1.0

    nel_1d = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 3
    discr = create_discretization_collection(actx, mesh, order=order)
    nodes = thaw(discr.nodes(), actx)

    tolerance = 1e-15
    for livedim in range(dim):
        mass = discr.zeros(actx) + 1.0 + np.dot(nodes, nodes)
        mom = make_obj_array([discr.zeros(actx) for _ in range(dim)])
        mom[livedim] = mass
        p_exact = discr.zeros(actx) + p0
        energy = (
            p_exact / (eos.gamma() - 1.0)
            + 0.5 * np.dot(mom, mom) / mass
        )
        cv = make_conserved(dim, mass=mass, energy=energy, momentum=mom)
        p = eos.pressure(cv)
        from mirgecom.gas_model import GasModel, make_fluid_state
        state = make_fluid_state(cv, GasModel(eos=eos))

        def inf_norm(x):
            return actx.to_numpy(op.norm(discr, x, np.inf))

        assert inf_norm(p - p_exact) < tolerance
        flux = inviscid_flux(state)
        logger.info(f"{dim}d flux = {flux}")
        vel_exact = mom / mass

        # first two components should be nonzero in livedim only
        assert inf_norm(flux.mass - mom) == 0
        eflux_exact = (energy + p_exact)*vel_exact
        assert inf_norm(flux.energy - eflux_exact) == 0

        logger.info("Testing momentum")
        xpmomflux = mass*np.outer(vel_exact, vel_exact) + p_exact*np.identity(dim)
        assert inf_norm(flux.momentum - xpmomflux) < tolerance


@pytest.mark.parametrize("nspecies", [0, 10])
@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("num_flux", [inviscid_facial_flux_rusanov,
                                      inviscid_facial_flux_hll])
def test_facial_flux(actx_factory, nspecies, order, dim, num_flux):
    """Check the flux across element faces.

    The flux is checked by prescribing states (q) with known fluxes. Only uniform
    states are tested currently - ensuring that the Lax-Friedrichs flux terms which
    are proportional to jumps in state data vanish.

    Since the returned fluxes use state data which has been interpolated
    to-and-from the element faces, this test is grid-dependent.
    """
    actx = actx_factory()

    tolerance = 1e-14
    p0 = 1.0

    from meshmode.mesh.generation import generate_regular_rect_mesh
    from pytools.convergence import EOCRecorder

    eoc_rec0 = EOCRecorder()
    eoc_rec1 = EOCRecorder()
    for nel_1d in [4, 8, 12]:

        mesh = generate_regular_rect_mesh(
            a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
        )

        logger.info(f"Number of elements: {mesh.nelements}")

        discr = create_discretization_collection(actx, mesh, order=order)
        zeros = discr.zeros(actx)
        ones = zeros + 1.0

        mass_input = discr.zeros(actx) + 1.0
        energy_input = discr.zeros(actx) + 2.5
        mom_input = flat_obj_array(
            [discr.zeros(actx) for i in range(discr.dim)]
        )
        mass_frac_input = flat_obj_array(
            [ones / ((i + 1) * 10) for i in range(nspecies)]
        )
        species_mass_input = mass_input * mass_frac_input

        cv = make_conserved(
            dim, mass=mass_input, energy=energy_input, momentum=mom_input,
            species_mass=species_mass_input)
        from grudge.trace_pair import interior_trace_pairs
        cv_interior_pairs = interior_trace_pairs(discr, cv)
        # Check the boundary facial fluxes as called on an interior boundary
        # eos = IdealSingleGas()
        from mirgecom.gas_model import (
            GasModel,
            make_fluid_state
        )
        gas_model = GasModel(eos=IdealSingleGas())

        from mirgecom.gas_model import make_fluid_state_trace_pairs
        state_tpairs = make_fluid_state_trace_pairs(cv_interior_pairs, gas_model)
        interior_state_pair = state_tpairs[0]

        nhat = thaw(discr.normal(interior_state_pair.dd), actx)
        bnd_flux = num_flux(interior_state_pair, gas_model, nhat)
        dd = interior_state_pair.dd
        dd_allfaces = dd.with_dtag("all_faces")
        interior_face_flux = op.project(discr, dd, dd_allfaces, bnd_flux)

        def inf_norm(data):
            if len(data) > 0:
                return actx.to_numpy(op.norm(discr, data, np.inf, dd="all_faces"))
            else:
                return 0.0

        assert inf_norm(interior_face_flux.mass) < tolerance
        assert inf_norm(interior_face_flux.energy) < tolerance
        assert inf_norm(interior_face_flux.species_mass) < tolerance

        # The expected pressure is 1.0 (by design). And the flux diagonal is
        # [rhov_x*v_x + p] (etc) since we have zero velocities it's just p.
        #
        # The off-diagonals are zero. We get a {ndim}-vector for each
        # dimension, the flux for the x-component of momentum (for example) is:
        # f_momx = < 1.0, 0 , 0> , then we return f_momx .dot. normal, which
        # can introduce negative values.
        #
        # (Explanation courtesy of Mike Campbell,
        # https://github.com/illinois-ceesd/mirgecom/pull/44#discussion_r463304292)

        nhat = thaw(discr.normal("int_faces"), actx)
        mom_flux_exact = op.project(discr, "int_faces", "all_faces", p0*nhat)
        print(f"{mom_flux_exact=}")
        print(f"{interior_face_flux.momentum=}")
        momerr = inf_norm(interior_face_flux.momentum - mom_flux_exact)
        assert momerr < tolerance
        eoc_rec0.add_data_point(1.0 / nel_1d, momerr)

        # Check the boundary facial fluxes as called on a domain boundary
        dir_mass = op.project(discr, "vol", BTAG_ALL, mass_input)
        dir_e = op.project(discr, "vol", BTAG_ALL, energy_input)
        dir_mom = op.project(discr, "vol", BTAG_ALL, mom_input)
        dir_mf = op.project(discr, "vol", BTAG_ALL, species_mass_input)

        dir_bc = make_conserved(dim, mass=dir_mass, energy=dir_e,
                                momentum=dir_mom, species_mass=dir_mf)
        dir_bval = make_conserved(dim, mass=dir_mass, energy=dir_e,
                                  momentum=dir_mom, species_mass=dir_mf)
        state_tpair = TracePair(BTAG_ALL,
                                interior=make_fluid_state(dir_bval, gas_model),
                                exterior=make_fluid_state(dir_bc, gas_model))

        nhat = thaw(discr.normal(state_tpair.dd), actx)
        bnd_flux = num_flux(state_tpair, gas_model, nhat)
        dd = state_tpair.dd
        dd_allfaces = dd.with_dtag("all_faces")
        boundary_flux = op.project(discr, dd, dd_allfaces, bnd_flux)

        assert inf_norm(boundary_flux.mass) < tolerance
        assert inf_norm(boundary_flux.energy) < tolerance
        assert inf_norm(boundary_flux.species_mass) < tolerance

        nhat = thaw(discr.normal(BTAG_ALL), actx)
        mom_flux_exact = op.project(discr, BTAG_ALL, "all_faces", p0*nhat)
        momerr = inf_norm(boundary_flux.momentum - mom_flux_exact)
        assert momerr < tolerance

        eoc_rec1.add_data_point(1.0 / nel_1d, momerr)

    logger.info(
        f"standalone Errors:\n{eoc_rec0}"
        f"boundary Errors:\n{eoc_rec1}"
    )
    assert (
        eoc_rec0.order_estimate() >= order - 0.5
        or eoc_rec0.max_error() < 1e-9
    )
    assert (
        eoc_rec1.order_estimate() >= order - 0.5
        or eoc_rec1.max_error() < 1e-9
    )
