"""Test the restart module."""

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
import logging
import pytest
from pytools.obj_array import make_obj_array
from grudge.eager import EagerDGDiscretization
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)


logger = logging.getLogger(__name__)


@pytest.mark.parametrize("nspecies", [0, 10])
def test_restart_cv(actx_factory, nspecies):
    """Test that restart can read a CV array container."""
    actx = actx_factory()
    nel_1d = 4
    dim = 3
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )
    order = 3
    discr = EagerDGDiscretization(actx, mesh, order=order)
    from meshmode.dof_array import thaw
    nodes = thaw(actx, discr.nodes())

    mass = nodes[0]
    energy = nodes[1]
    mom = make_obj_array([nodes[2]*(i+3) for i in range(dim)])

    species_mass = None
    if nspecies > 0:
        mass_fractions = make_obj_array([i*nodes[0] for i in range(nspecies)])
        species_mass = mass * mass_fractions

    rst_filename = f"test_{nspecies}.pkl"

    from mirgecom.fluid import make_conserved
    test_state = make_conserved(dim, mass=mass, energy=energy, momentum=mom,
                                species_mass=species_mass)

    rst_data = {"state": test_state}
    from mirgecom.restart import write_restart_file
    write_restart_file(actx, rst_data, rst_filename)

    from mirgecom.restart import read_restart_data
    restart_data = read_restart_data(actx, rst_filename)

    resid = test_state - restart_data["state"]
    assert discr.norm(resid.join(), np.inf) == 0
