"""Test built-in callback routines."""

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
import pytest

from arraycontext import (  # noqa
    thaw,
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests
)

from mirgecom.fluid import join_conserved
from mirgecom.eos import IdealSingleGas

from grudge.eager import EagerDGDiscretization


def test_basic_cfd_healthcheck(actx_factory):
    from mirgecom.simutil import sim_cfd_healthcheck

    actx = actx_factory()
    nel_1d = 4
    dim = 2

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(1.0,) * dim, b=(2.0,) * dim, nelements_per_axis=(nel_1d,) * dim
    )

    order = 3
    discr = EagerDGDiscretization(actx, mesh, order=order)
    nodes = thaw(discr.nodes(), actx)
    zeros = discr.zeros(actx)
    ones = zeros + 1.0

    # Let's make a very bad state (negative mass)
    mass = -1*ones
    energy = zeros + 2.5
    velocity = 2 * nodes
    mom = mass * velocity

    eos = IdealSingleGas()
    q = join_conserved(dim, mass=mass, energy=energy, momentum=mom)

    from mirgecom.exceptions import SimulationHealthError

    with pytest.raises(SimulationHealthError):
        sim_cfd_healthcheck(discr, eos, q, ncheck=1)

    # Let's make another very bad state (nans)
    mass = 1*ones
    energy = zeros + 2.5
    velocity = np.nan * nodes
    mom = mass * velocity

    q = join_conserved(dim, mass=mass, energy=energy, momentum=mom)

    with pytest.raises(SimulationHealthError):
        sim_cfd_healthcheck(discr, eos, q, ncheck=1)

    # Let's make one last very bad state (inf)
    mass = 1*ones
    energy = np.inf * ones
    velocity = 2 * nodes
    mom = mass * velocity

    q = join_conserved(dim, mass=mass, energy=energy, momentum=mom)

    with pytest.raises(SimulationHealthError):
        sim_cfd_healthcheck(discr, eos, q, ncheck=1)
