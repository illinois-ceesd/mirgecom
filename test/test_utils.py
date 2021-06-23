__copyright__ = """Copyright (C) 2021 University of Illinois Board of Trustees"""

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
import pyopencl.array as cla  # noqa
import pyopencl.clmath as clmath # noqa
from pytools.obj_array import make_obj_array

from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

import pytest  # noqa

import logging
logger = logging.getLogger(__name__)


def test_outer(actx_factory):
    actx = actx_factory()

    dim = 2

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
        a=(-0.5,)*dim,
        b=(0.5,)*dim,
        nelements_per_axis=(4,)*dim)

    from grudge.eager import EagerDGDiscretization
    discr = EagerDGDiscretization(actx, mesh, order=3)

    from mirgecom.utils import outer

    zeros = discr.zeros(actx)

    # Two scalars
    a = zeros + 2
    b = zeros + 3
    assert outer(a, b) == a*b
    del a, b

    # Scalar and vector
    a = zeros + 2
    b = make_obj_array([zeros + 3, zeros + 4])
    assert np.all(outer(a, b) == a*b)
    assert np.all(outer(b, a) == b*a)
    del a, b

    # Two vectors
    a = make_obj_array([zeros + 2, zeros + 3])
    b = make_obj_array([zeros + 4, zeros + 5])
    assert np.all(outer(a, b) == np.outer(a, b))
    del a, b

    from mirgecom.fluid import ConservedVars

    # Scalar and array container
    a = zeros + 2
    b = ConservedVars(
        mass=zeros + 3,
        energy=zeros + 4,
        momentum=make_obj_array([zeros + 5, zeros + 6]),
        species_mass=make_obj_array([zeros + 7, zeros + 8, zeros + 9]))
    assert outer(a, b) == a*b
    assert outer(b, a) == b*a
    del a, b

    # Vector and array container
    a = make_obj_array([zeros + 2, zeros + 3])
    b = ConservedVars(
        mass=zeros + 4,
        energy=zeros + 5,
        momentum=make_obj_array([zeros + 6, zeros + 7]),
        species_mass=make_obj_array([zeros + 8, zeros + 9, zeros + 10]))
    assert np.all(outer(a, b) == make_obj_array([2*b, 3*b]))
    assert outer(b, a) == ConservedVars(
        mass=a*b.mass,
        energy=a*b.energy,
        momentum=np.outer(b.momentum, a),
        species_mass=np.outer(b.species_mass, a))
    del a, b

    # Two array containers
    a = ConservedVars(
        mass=zeros + 2,
        energy=zeros + 3,
        momentum=make_obj_array([zeros + 4, zeros + 5]),
        species_mass=make_obj_array([zeros + 6, zeros + 7, zeros + 8]))
    b = ConservedVars(
        mass=zeros + 9,
        energy=zeros + 10,
        momentum=make_obj_array([zeros + 11, zeros + 12]),
        species_mass=make_obj_array([zeros + 13, zeros + 14, zeros + 15]))
    assert outer(a, b) == ConservedVars(
        mass=a.mass*b.mass,
        energy=a.energy*b.energy,
        momentum=np.outer(a.momentum, b.momentum),
        species_mass=np.outer(a.species_mass, b.species_mass))
    del a, b


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
