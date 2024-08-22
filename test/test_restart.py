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
from collections import defaultdict
from pytools.obj_array import make_obj_array
from mirgecom.discretization import create_discretization_collection
from grudge.discretization import PartID
from meshmode.array_context import PytestPyOpenCLArrayContextFactory
from arraycontext import pytest_generate_tests_for_array_contexts
pytest_generate_tests = pytest_generate_tests_for_array_contexts(
    [PytestPyOpenCLArrayContextFactory])

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

    trg_decomp_file = f"data/M24k_mesh_idecomp_np{trg_np}_pkl_data"
    src_decomp_file = f"data/M24k_mesh_idecomp_np{src_np}_pkl_data"
    with open(src_decomp_file, "wb") as file:
        pickle.dump(src_part_els, file)
    with open(trg_decomp_file, "wb") as file:
        pickle.dump(trg_part_els, file)

    nsrc_parts = len(src_part_els)
    ntrg_parts = len(trg_part_els)

    print(f"Number of source partitions: {nsrc_parts}.")
    print(f"Number of target partitions: {ntrg_parts}.")

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


def test_multivolume_interdecomp_overlap_basic():
    """Test the multivolume_interdecomp_overlap."""
    import numpy as np
    # Total elements in the testing mesh
    total_elements = 100

    # Create test decomps
    src_np, trg_np = 3, 2
    elements_per_rank_src = total_elements // src_np
    src_decomp = {i: list(range(i * elements_per_rank_src,
                                (i + 1) * elements_per_rank_src))
                  for i in range(src_np)}

    elements_per_rank_trg = total_elements // trg_np
    trg_decomp = {i: list(range(i * elements_per_rank_trg,
                                (i + 1) * elements_per_rank_trg))
                  for i in range(trg_np)}

    # Adjust the last rank to include any remaining elements
    src_decomp[src_np - 1].extend(range(src_np * elements_per_rank_src,
                                        total_elements))
    trg_decomp[trg_np - 1].extend(range(trg_np * elements_per_rank_trg,
                                        total_elements))

    # Testing volume decomps
    # Vol1: even elements, Vol2: odd elements
    vol1_elements = np.array([i for i in range(total_elements) if i % 2 == 0])
    vol2_elements = np.array([i for i in range(total_elements) if i % 2 != 0])

    src_vol_decomp = {}
    trg_vol_decomp = {}

    # Test vol decomps
    for i in range(src_np):
        src_vol_decomp[PartID(volume_tag="vol1", rank=i)] = \
            np.array([el for el in src_decomp[i] if el in vol1_elements])
        src_vol_decomp[PartID(volume_tag="vol2", rank=i)] = \
            np.array([el for el in src_decomp[i] if el in vol2_elements])

    for i in range(trg_np):
        trg_vol_decomp[PartID(volume_tag="vol1", rank=i)] = \
            np.array([el for el in trg_decomp[i] if el in vol1_elements])
        trg_vol_decomp[PartID(volume_tag="vol2", rank=i)] = \
            np.array([el for el in trg_decomp[i] if el in vol2_elements])

    from mirgecom.simutil import multivolume_interdecomposition_overlap
    # Compute the multivolume interdecomp overlaps
    mv_idx = multivolume_interdecomposition_overlap(
        src_decomp, trg_decomp, src_vol_decomp, trg_vol_decomp
    )

    for trg_partid, src_partid_mappings in mv_idx.items():
        uncovered_elements = set(trg_vol_decomp[trg_partid])
        for src_partid, element_mapping in src_partid_mappings.items():
            for trg_local_idx, _ in element_mapping.items():
                uncovered_elements.discard(trg_vol_decomp[trg_partid][trg_local_idx])
            for trg_local_idx, src_local_idx in element_mapping.items():
                # match element ids for the trg and src at resp index
                assert trg_vol_decomp[trg_partid][trg_local_idx] == \
                    src_vol_decomp[src_partid][src_local_idx]
        assert not uncovered_elements

    for trg_partid, src_partid_mappings in mv_idx.items():
        mapped_trg_elems = set()
        # validate ranks in trg mapping
        assert 0 <= trg_partid.rank < trg_np,\
            f"Invalid target rank: {trg_partid.rank}"
        for src_partid, element_mapping in src_partid_mappings.items():
            # check for consistent volume_tags
            assert trg_partid.volume_tag == src_partid.volume_tag, \
                f"Volume tag mismatch: {trg_partid.volume_tag} "\
                f"vs {src_partid.volume_tag}"
            # validate ranks in src mapping
            assert 0 <= src_partid.rank < src_np,\
                f"Invalid source rank: {src_partid.rank}"
            for trg_local_idx, src_local_idx in element_mapping.items():
                # check that each trg el is mapped only once
                assert trg_local_idx not in mapped_trg_elems,\
                    f"Duplicate mapping for target element {trg_local_idx}"
                mapped_trg_elems.add(trg_local_idx)
                # check for valid src and trg indices
                assert 0 <= src_local_idx < len(src_vol_decomp[src_partid]),\
                    f"Invalid source index {src_local_idx} for {src_partid}"
                assert 0 <= trg_local_idx < len(trg_vol_decomp[trg_partid]),\
                    f"Invalid target index {trg_local_idx} for {trg_partid}"

    # Check that the mapping is 1-to-1, that each src element is covered and maps
    # to one and only one target element
    accumulated_mapped_src_elems = defaultdict(list)
    for _, src_partid_mappings in mv_idx.items():
        for src_partid, element_mapping in src_partid_mappings.items():
            mapped_src_elems = list(element_mapping.values())
            accumulated_mapped_src_elems[src_partid].extend(mapped_src_elems)

    for src_partid, mapped_src_elems in accumulated_mapped_src_elems.items():
        src_elem_set = set(mapped_src_elems)
        assert set(range(len(src_vol_decomp[src_partid]))) == src_elem_set, \
            f"Some elements in {src_partid} are not mapped to any target element"
        # do not map to more than one trg elem
        for elem in src_elem_set:
            assert mapped_src_elems.count(elem) == 1, \
                f"Element {elem} in {src_partid} is mapped more than once"


def _generate_decompositions(total_elements, num_ranks, pattern="chunked"):
    """Generate testing decomp."""
    elements_per_rank = total_elements // num_ranks
    if pattern == "chunked":
        decomp = {i: list(range(i * elements_per_rank,
                                (i + 1) * elements_per_rank))
                  for i in range(num_ranks)}
        # Toss remaining els into last rank
        decomp[num_ranks - 1].extend(range(num_ranks * elements_per_rank,
                                           total_elements))
    elif pattern == "strided":
        decomp = {i: list(range(i, total_elements, num_ranks))
                  for i in range(num_ranks)}
    elif pattern == "random":
        all_elements = list(range(total_elements))
        np.random.shuffle(all_elements)
        decomp = {i: all_elements[i * elements_per_rank: (i + 1) * elements_per_rank]
                  for i in range(num_ranks)}
        decomp[num_ranks - 1].extend(all_elements[num_ranks * elements_per_rank:])

    return {k: np.array(v) for k, v in decomp.items()}  # Convert to numpy arrays


@pytest.mark.parametrize("decomp_pattern", ["chunked", "strided", "random"])
@pytest.mark.parametrize("vol_pattern", ["front_back", "random_split"])
@pytest.mark.parametrize("src_trg_ranks", [(3, 4), (4, 4), (5, 4), (1, 4), (4, 1)])
def test_multivolume_interdecomp_overlap(decomp_pattern, vol_pattern, src_trg_ranks):
    """Test the multivolume_interdecomp_overlap."""
    total_elements = 100
    src_np, trg_np = src_trg_ranks

    # Generate global decomps
    src_decomp = _generate_decompositions(total_elements, src_np,
                                          pattern=decomp_pattern)
    trg_decomp = _generate_decompositions(total_elements, trg_np,
                                          pattern=decomp_pattern)

    # Volume splitting
    if vol_pattern == "front_back":
        mid_point = total_elements // 2
        vol1_elements = list(range(mid_point))
        vol2_elements = list(range(mid_point, total_elements))
    elif vol_pattern == "random_split":
        all_elements = list(range(total_elements))
        np.random.shuffle(all_elements)
        mid_point = total_elements // 2
        vol1_elements = all_elements[:mid_point]
        vol2_elements = all_elements[mid_point:]

    src_vol_decomp = {}
    trg_vol_decomp = {}

    # Form the multivol decomps
    for i in range(src_np):
        src_vol_decomp[PartID(volume_tag="vol1", rank=i)] = \
            [el for el in src_decomp[i] if el in vol1_elements]
        src_vol_decomp[PartID(volume_tag="vol2", rank=i)] = \
            [el for el in src_decomp[i] if el in vol2_elements]

    for i in range(trg_np):
        trg_vol_decomp[PartID(volume_tag="vol1", rank=i)] = \
            [el for el in trg_decomp[i] if el in vol1_elements]
        trg_vol_decomp[PartID(volume_tag="vol2", rank=i)] = \
            [el for el in trg_decomp[i] if el in vol2_elements]

    # Testing the overlap utility
    from mirgecom.simutil import multivolume_interdecomposition_overlap
    mv_idx = multivolume_interdecomposition_overlap(
        src_decomp, trg_decomp, src_vol_decomp, trg_vol_decomp
    )

    for trg_partid, src_partid_mappings in mv_idx.items():
        uncovered_elements = set(trg_vol_decomp[trg_partid])
        for src_partid, element_mapping in src_partid_mappings.items():
            for trg_local_idx, _ in element_mapping.items():
                uncovered_elements.discard(trg_vol_decomp[trg_partid][trg_local_idx])
            for trg_local_idx, src_local_idx in element_mapping.items():
                # match element ids for the trg and src at resp index
                assert trg_vol_decomp[trg_partid][trg_local_idx] == \
                    src_vol_decomp[src_partid][src_local_idx]
        assert not uncovered_elements

    for trg_partid, src_partid_mappings in mv_idx.items():
        mapped_trg_elems = set()
        # validate ranks in trg mapping
        assert 0 <= trg_partid.rank < trg_np,\
            f"Invalid target rank: {trg_partid.rank}"
        for src_partid, element_mapping in src_partid_mappings.items():
            # check for consistent volume_tags
            assert trg_partid.volume_tag == src_partid.volume_tag, \
                f"Volume tag mismatch: {trg_partid.volume_tag} "\
                f"vs {src_partid.volume_tag}"
            # validate ranks in src mapping
            assert 0 <= src_partid.rank < src_np,\
                f"Invalid source rank: {src_partid.rank}"
            for trg_local_idx, src_local_idx in element_mapping.items():
                # check that each trg el is mapped only once
                assert trg_local_idx not in mapped_trg_elems,\
                    f"Duplicate mapping for target element {trg_local_idx}"
                mapped_trg_elems.add(trg_local_idx)
                # check for valid src and trg indices
                assert 0 <= src_local_idx < len(src_vol_decomp[src_partid]),\
                    f"Invalid source index {src_local_idx} for {src_partid}"
                assert 0 <= trg_local_idx < len(trg_vol_decomp[trg_partid]),\
                    f"Invalid target index {trg_local_idx} for {trg_partid}"

    # Check that the mapping is 1-to-1, that each src element is covered and maps
    # to one and only one target element
    accumulated_mapped_src_elems = defaultdict(list)
    for _, src_partid_mappings in mv_idx.items():
        for src_partid, element_mapping in src_partid_mappings.items():
            mapped_src_elems = list(element_mapping.values())
            accumulated_mapped_src_elems[src_partid].extend(mapped_src_elems)

    for src_partid, mapped_src_elems in accumulated_mapped_src_elems.items():
        src_elem_set = set(mapped_src_elems)
        assert set(range(len(src_vol_decomp[src_partid]))) == src_elem_set, \
            f"Some elements in {src_partid} are not mapped to any target element"
        # do not map to more than one trg elem
        for elem in src_elem_set:
            assert mapped_src_elems.count(elem) == 1, \
                f"Element {elem} in {src_partid} is mapped more than once"
