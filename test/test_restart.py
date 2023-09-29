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
from mirgecom.discretization import create_discretization_collection
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
    dcoll = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())

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

    rst_state = restart_data["state"]

    resid = test_state - rst_state
    from mirgecom.simutil import max_component_norm
    assert max_component_norm(dcoll, resid, np.inf) == 0


@pytest.mark.parametrize("src_trg_np", [(1, 2),
                                        (1, 4),
                                        (2, 4),
                                        (4, 2),
                                        (4, 1)])
def test_interdecomp_overlap(src_trg_np):
    """Test that restart can read a CV array container."""
    import pickle
    from mirgecom.simutil import interdecomposition_overlap
    print(f"{src_trg_np=}")
    src_np, trg_np = src_trg_np

    trg_decomp_file = f"data/M24k_mesh_decomp_np{trg_np}_pkl_data"
    src_decomp_file = f"data/M24k_mesh_decomp_np{src_np}_pkl_data"

    with open(src_decomp_file, "rb") as file:
        src_dcmp = pickle.load(file)
    with open(trg_decomp_file, "rb") as file:
        trg_dcmp = pickle.load(file)

    from mirgecom.simutil import invert_decomp
    src_part_els = invert_decomp(src_dcmp)
    trg_part_els = invert_decomp(trg_dcmp)

    nsrc_parts = len(src_part_els)
    ntrg_parts = len(trg_part_els)

    print(f"Numver of source partitions: {nsrc_parts}.")
    print(f"Numver of target partitions: {ntrg_parts}.")

    idx = interdecomposition_overlap(trg_dcmp, src_dcmp)
    nsrc_els = len(src_dcmp)
    ntrg_els = len(trg_dcmp)
    assert nsrc_els == ntrg_els
    assert len(idx) == ntrg_parts

    nolap = 0
    for trank in range(ntrg_parts):
        for src_rank, olap in idx[trank].items():
            olen = len(olap)
            nolap = nolap + olen
            print(f"Rank({trank}) olap w/OGRank({src_rank}) is {olen} els.")
    assert nolap == nsrc_els


def test_dofarray_mapped_copy(actx_factory):
    """Test that restart can read a CV array container."""
    actx = actx_factory()
    nel_1d = 4
    dim = 3
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
        a=(-0.5,) * dim, b=(0.5,) * dim, nelements_per_axis=(nel_1d,) * dim
    )
    order = 3
    dcoll = create_discretization_collection(actx, mesh, order=order)
    nodes = actx.thaw(dcoll.nodes())

    test_data_1 = 1. * nodes[0]
    test_data_2 = 2. * nodes[1]
    nelems, nnodes = test_data_1[0].shape

    # Copy the whole thing
    el_map = {}
    for iel in range(nelems):
        el_map[iel] = iel

    from mirgecom.simutil import copy_mapped_dof_array_data
    test_data_1 = copy_mapped_dof_array_data(test_data_1, test_data_2, el_map)

    # print(f"{test_data_1=}")
    # print(f"{test_data_2=}")
    # raise AssertionError()

    resid = test_data_1 - test_data_2
    from mirgecom.simutil import max_component_norm
    assert max_component_norm(dcoll, resid, np.inf) == 0

    # Copy half the data
    test_data_1 = 3. * nodes[0]
    test_data_2 = 4. * nodes[1]

    el_map = {}
    ncopy = int(nelems/2)
    el_map_1 = {}
    el_map_2 = {}
    for iel in range(ncopy):
        el_map_1[iel] = iel
    for iel in range(ncopy, nelems):
        el_map_2[iel] = iel
    test_data_1 = copy_mapped_dof_array_data(test_data_1, test_data_2, el_map_1)
    test_data_2 = copy_mapped_dof_array_data(test_data_2, test_data_1, el_map_2)

    # print(f"{test_data_1=}")
    # print(f"{test_data_2=}")
    # raise AssertionError()

    resid = test_data_1 - test_data_2
    from mirgecom.simutil import max_component_norm
    assert max_component_norm(dcoll, resid, np.inf) == 0

    test_data_1 = 1. * nodes[0] + 1.0
    test_data_2 = 8. * nodes[1] + 2.0

    # Copy every other element
    el_map_odd = {}
    el_map_even = {}
    for iel in range(nelems):
        if iel % 2:
            el_map_odd[iel] = iel
        else:
            el_map_even[iel] = iel
    test_data_1 = copy_mapped_dof_array_data(test_data_1, test_data_2, el_map_odd)
    test_data_2 = copy_mapped_dof_array_data(test_data_2, test_data_1, el_map_even)

    # print(f"{test_data_1=}")
    # print(f"{test_data_2=}")
    # raise AssertionError()

    resid = test_data_1 - test_data_2
    from mirgecom.simutil import max_component_norm
    assert max_component_norm(dcoll, resid, np.inf) == 0
