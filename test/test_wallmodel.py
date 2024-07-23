"""Test wall-model related functions."""

__copyright__ = """Copyright (C) 2023 University of Illinois Board of Trustees"""

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

import grudge.op as op
import numpy as np
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context as pytest_generate_tests,
)

from pytools.obj_array import make_obj_array

from mirgecom.discretization import create_discretization_collection
from mirgecom.simutil import get_box_mesh


def test_tacot_decomposition(actx_factory):
    """Check the wall degradation model."""
    actx = actx_factory()

    dim = 2
    nelems = 2
    order = 2
    mesh = get_box_mesh(dim, -0.1, 0.1, nelems)
    dcoll = create_discretization_collection(actx, mesh, order=order)

    nodes = actx.thaw(dcoll.nodes())
    zeros = actx.np.zeros_like(nodes[0])

    temperature = 900.0 + zeros

    from mirgecom.materials.tacot import Pyrolysis
    decomposition = Pyrolysis(virgin_mass=120.0, char_mass=60.0, fiber_mass=160.0,
                              pre_exponential=(12000.0, 4.48e9),
                              decomposition_temperature=(333.3, 555.6))
    chi = make_obj_array([30.0 + zeros, 90.0 + zeros, 160.0 + zeros])

    tol = 1e-8

    # ~~~ Test parameter setup
    tacot_decomp = decomposition.get_decomposition_parameters()

    assert tacot_decomp["virgin_mass"] - 120.0 < tol
    assert tacot_decomp["char_mass"] - 60.0 < tol
    assert tacot_decomp["fiber_mass"] - 160.0 < tol

    weights = tacot_decomp["reaction_weights"]
    assert weights[0] - 30.0 < tol
    assert weights[1] - 90.0 < tol

    pre_exp = tacot_decomp["pre_exponential"]
    assert pre_exp[0] - 12000.0 < tol
    assert pre_exp[1] - 4.48e9 < tol

    Tcrit = tacot_decomp["temperature"]  # noqa N806
    assert Tcrit[0] - 333.3 < tol
    assert Tcrit[1] - 555.6 < tol

    # ~~~ Test actual decomposition
    solid_mass_rhs = decomposition.get_source_terms(temperature, chi)

    sample_source_gas = -sum(solid_mass_rhs)

    assert actx.to_numpy(
        op.norm(dcoll, solid_mass_rhs[0] + 26.7676118539965, np.inf)) < tol
    assert actx.to_numpy(
        op.norm(dcoll, solid_mass_rhs[1] + 2.03565420370596, np.inf)) < tol
    assert actx.to_numpy(
        op.norm(dcoll, sample_source_gas - 28.8032660577024, np.inf)) < tol
