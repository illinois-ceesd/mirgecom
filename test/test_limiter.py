__copyright__ = """Copyright (C) 2020 University of Illinois Board of Trustees"""

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

from meshmode.array_context import PytestPyOpenCLArrayContextFactory
from arraycontext import pytest_generate_tests_for_array_contexts

from mirgecom.limiter import bound_preserving_limiter
from mirgecom.discretization import create_discretization_collection

from mirgecom.initializers import Uniform
from mirgecom.eos import IdealSingleGas
from mirgecom.gas_model import (  # noqa
    GasModel, make_fluid_state, make_operator_fluid_states
)
from mirgecom.fluid import make_conserved

import pytest

pytest_generate_tests = pytest_generate_tests_for_array_contexts(
    [PytestPyOpenCLArrayContextFactory])


def test_fluid_api(actx_factory):
    actx = actx_factory()
    dim = 2
    nel_1d = 4
    order = 1

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    dcoll = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())
    zeros = dcoll.zeros(actx)
    ones = zeros + 1.
    eos = IdealSingleGas(gas_const=1)
    gas_model = GasModel(eos=eos)

    eps = .001
    initializer = Uniform(dim=dim, pressure=1.0, rho=1.0)
    fluid_cv = initializer(nodes, eos=eos)
    fluid_cv = fluid_cv.replace(mass=ones-eps)

    # create a fluid CV limiting routine that preserves pressure and temperature
    def _limit_fluid_cv(cv, temperature_seed, gas_model, dd=None):

        density_lim = bound_preserving_limiter(dcoll, cv.mass, mmin=1.0)

        # make a new CV with the limited variables
        return make_conserved(dim=dim, mass=density_lim, energy=cv.energy,
                              momentum=density_lim*cv.velocity)

    fluid_state = make_fluid_state(cv=fluid_cv, gas_model=gas_model,
                                   limiter_func=_limit_fluid_cv)

    mass = fluid_state.mass_density
    assert min(actx.to_numpy(mass)).all() >= 1.0


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("dim", [2, 3])
def test_positivity_preserving_limiter(actx_factory, order, dim):
    """Testing positivity-preserving limiter."""
    actx = actx_factory()

    nel_1d = 2

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
        a=(0.0,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    dcoll = create_discretization_collection(actx, mesh, order=order)

    # create cells with negative values "eps"
    nodes = actx.thaw(actx.freeze(dcoll.nodes()))
    eps = -0.001
    field = nodes[0] + eps

    # apply positivity-preserving limiter
    limited_field = bound_preserving_limiter(dcoll, field, mmin=0.0)
    lmtd_fld_min = np.min(actx.to_numpy(limited_field[0]))
    assert lmtd_fld_min > -1e-13


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("dim", [2, 3])
def test_bound_preserving_limiter(actx_factory, order, dim):
    """Testing upper bound limiting."""
    actx = actx_factory()

    nel_1d = 2

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
        a=(0.0,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    dcoll = create_discretization_collection(actx, mesh, order=order)

    # create cells with values larger than 1.0
    nodes = actx.thaw(actx.freeze(dcoll.nodes()))
    eps = 0.01
    field = 0.5 + nodes[0] + eps

    # apply limiter
    limited_field = bound_preserving_limiter(dcoll, field, mmin=0.0, mmax=1.0)
    lmtd_fld_max = np.max(actx.to_numpy(limited_field[0]))
    assert lmtd_fld_max < 1.0 + 1.0e-13
