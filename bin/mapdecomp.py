"""Read gmsh mesh, partition it, and create a pkl file per mesh partition."""

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
import argparse
import pickle
import os

from meshmode.distributed import get_connected_parts
from grudge.discretization import PartID
from meshmode.mesh import InterPartAdjacencyGroup
from collections import defaultdict


def print_connected_stuff(mesh):
    """For a local mesh part in *mesh*, determine the set of connected parts."""
    assert mesh.facial_adjacency_groups is not None

    for fagrp_list in mesh.facial_adjacency_groups:
        for grp in fagrp_list:
            if isinstance(grp, InterPartAdjacencyGroup):
                print(f"{grp.part_id=}")
                print(f"{len(grp.neighbor_faces)=}")


def analyze_mesh_interfaces(mesh, local_partid):
    """
    Analyze and print detailed summary of MPI and internal interfaces for each rank and domain.

    Parameters:
    - mesh: The local mesh data for this rank.
    - r: The rank being processed.
    - local_partid: The PartID of the current rank's domain.
    """
    r = -1
    try:
        r = local_partid.rank
    except:
        return

    print(f"\nProcessing volume: {local_partid.volume_tag} on rank {r}")

    mpi_interfaces = defaultdict(lambda: defaultdict(int))  # {remote_rank: {("domain1", "domain2"): count}}
    internal_interfaces = defaultdict(int)  # {("domain1", "domain2"): count}
    remote_ranks = set()

    for fagrp_list in mesh.facial_adjacency_groups:
        for grp in fagrp_list:
            if isinstance(grp, InterPartAdjacencyGroup):
                remote_rank = grp.part_id.rank
                local_rank = local_partid.rank
                remote_domain = grp.part_id.volume_tag
                local_domain = local_partid.volume_tag
                num_faces = len(grp.neighbor_faces)
                interface_key = (local_domain, remote_domain)

                if remote_rank == local_rank:
                    # Internal interface (must be symmetric)
                    internal_interfaces[interface_key] += num_faces
                else:
                    # MPI interface
                    remote_ranks.add(remote_rank)
                    mpi_interfaces[remote_rank][interface_key] += num_faces

    # Compute totals
    total_mpi_faces = sum(sum(iface_counts.values()) for iface_counts in mpi_interfaces.values())

    # Print summary
    print(f"  Unique remote MPI ranks: {len(remote_ranks)}")
    print(f"  Total MPI boundary faces: {total_mpi_faces}")

    # Print MPI details per remote rank
    for remote_rank, iface_counts in sorted(mpi_interfaces.items()):
        print(f"    - MPI interface with rank {remote_rank}:")
        for (dom1, dom2), face_count in iface_counts.items():
            print(f"      - {dom1} ↔ {dom2}: {face_count} faces")

    # Print internal interfaces
    print(f"  Internal domain-domain boundaries:")
    for (dom1, dom2), face_count in internal_interfaces.items():
        print(f"    - {dom1} ↔ {dom2}: {face_count} faces")

    # Sanity Checks
    print("\n  Running Sanity Checks...")
    all_checks_passed = True

    # Check local domain-domain interface symmetry
    for (dom1, dom2), count in internal_interfaces.items():
        reverse_key = (dom2, dom1)
        if reverse_key in internal_interfaces:
            if count != internal_interfaces[reverse_key]:
                print(f"    ❌ ERROR: Internal {dom1} ↔ {dom2} mismatch! {count} vs {internal_interfaces[reverse_key]}")
                all_checks_passed = False
        else:
            print(f"    ❌ ERROR: Missing internal {dom2} ↔ {dom1} interface!")
            all_checks_passed = False

    # Check MPI interface symmetry
    for remote_rank, iface_counts in mpi_interfaces.items():
        for (dom1, dom2), count in iface_counts.items():
            # If rank r sees dom1↔dom2 with N faces, rank remote_rank must see dom2↔dom1 with N faces
            expected_key = (dom2, dom1)
            if remote_rank in mpi_interfaces and expected_key in mpi_interfaces[remote_rank]:
                if count != mpi_interfaces[remote_rank][expected_key]:
                    print(f"    ❌ ERROR: MPI {r} ({dom1} ↔ {dom2}) ≠ {remote_rank} ({dom2} ↔ {dom1})!")
                    all_checks_passed = False
            else:
                print(f"    ❌ ERROR: Missing reciprocal MPI interface for {local_rank} ↔ {remote_rank} ({dom2} ↔ {dom1})!")
                all_checks_passed = False

    if all_checks_passed:
        print("  ✅ All sanity checks passed!\n")
    else:
        print("  ⚠️ Sanity checks found errors! Review the output above.\n")


def analyze_mesh_interfaces2(mesh, r):
    """Analyze and print detailed summary of MPI and internal interfaces for each rank."""

    mpi_interfaces = {}  # Track MPI interfaces (remote rank -> {'fluid-fluid': count, 'wall-wall': count})
    internal_fluid_wall = 0  # Count of internal fluid-wall boundaries
    remote_ranks = set()  # Unique remote ranks

    for fagrp_list in mesh.facial_adjacency_groups:
        for grp in fagrp_list:
            if isinstance(grp, InterPartAdjacencyGroup):
                remote_rank = grp.part_id.rank
                volume_tag = grp.part_id.volume_tag
                num_faces = len(grp.neighbor_faces)

                if remote_rank == r:
                    # Internal fluid-wall boundary
                    if volume_tag == "wall":
                        internal_fluid_wall += num_faces
                else:
                    # MPI boundary
                    remote_ranks.add(remote_rank)
                    if remote_rank not in mpi_interfaces:
                        mpi_interfaces[remote_rank] = {"fluid-fluid": 0, "wall-wall": 0}
                    
                    if volume_tag == "fluid":
                        mpi_interfaces[remote_rank]["fluid-fluid"] += num_faces
                    elif volume_tag == "wall":
                        mpi_interfaces[remote_rank]["wall-wall"] += num_faces

    # Compute totals
    total_mpi_faces = sum(
        sum(boundary.values()) for boundary in mpi_interfaces.values()
    )
    total_fluid_fluid_faces = sum(boundary["fluid-fluid"] for boundary in mpi_interfaces.values())
    total_wall_wall_faces = sum(boundary["wall-wall"] for boundary in mpi_interfaces.values())

    # Print summary for this rank
    print(f"  Unique remote MPI ranks: {len(remote_ranks)}")
    print(f"  Total MPI boundary faces: {total_mpi_faces}")
    print(f"    - Fluid-fluid MPI boundary faces: {total_fluid_fluid_faces}")
    print(f"    - Wall-wall MPI boundary faces: {total_wall_wall_faces}")

    for remote_rank, boundaries in sorted(mpi_interfaces.items()):
        print(f"    - MPI interface with rank {remote_rank}:")
        print(f"      - Fluid-fluid: {boundaries['fluid-fluid']} faces")
        print(f"      - Wall-wall: {boundaries['wall-wall']} faces")

    print(f"  Internal fluid-wall boundaries: {internal_fluid_wall} faces")


def main(mesh_filename=None, output_path=None):
    """Do it."""
    if output_path is None:
        output_path = "./"
    output_path.strip("'")

    if mesh_filename is None:
        # Try to detect the mesh filename
        raise AssertionError("No mesh filename.")

    intradecomp_map = {}
    nranks = 0
    nvolumes = 0
    volumes = set()
    for r in range(10000):
        print(f"Processing rank {r}")
        mesh_pkl_filename = mesh_filename + f"_rank{r}.pkl"
        if os.path.exists(mesh_pkl_filename):
            nranks = nranks + 1
            with open(mesh_pkl_filename, "rb") as pkl_file:
                global_nelements, volume_to_local_mesh_data = \
                    pickle.load(pkl_file)
            for vol, meshdat in volume_to_local_mesh_data.items():
                local_partid = PartID(volume_tag=vol, rank=r)
                print(f"Processing volume: {vol}")
                volumes.add(vol)
                # print_connected_stuff(meshdat[0])
                analyze_mesh_interfaces(meshdat[0], local_partid)
                connected_parts = get_connected_parts(meshdat[0])
                if connected_parts:
                    intradecomp_map[local_partid] = connected_parts
        else:
            break
    nvolumes = len(volumes)
    rank_rank_nbrs = {r: set() for r in range(nranks)}
    for part, nbrs in intradecomp_map.items():
        local_rank = part.rank
        for nbr in nbrs:
            if nbr.rank != local_rank:
                rank_rank_nbrs[local_rank].add(nbr.rank)
    min_rank_nbrs = nranks
    max_rank_nbrs = 0
    num_nbr_dist = {}
    total_nnbrs = 0
    for _, rank_nbrs in rank_rank_nbrs.items():
        nrank_nbrs = len(rank_nbrs)
        total_nnbrs += nrank_nbrs
        if nrank_nbrs not in num_nbr_dist:
            num_nbr_dist[nrank_nbrs] = 0
        num_nbr_dist[nrank_nbrs] += 1
        min_rank_nbrs = min(min_rank_nbrs, nrank_nbrs)
        max_rank_nbrs = max(max_rank_nbrs, nrank_nbrs)

    mean_nnbrs = (1.0*total_nnbrs) / (1.0*nranks)

    print(f"Number of ranks: {nranks}")
    print(f"Number of volumes: {nvolumes}")
    print(f"Volumes: {volumes}")
    print("Number of rank neighbors (min, max, mean): "
          f"({min_rank_nbrs}, {max_rank_nbrs}, {mean_nnbrs})")
    print(f"Distribution of num nbrs: {num_nbr_dist=}")
    print(f"Rank Nbrs: {rank_rank_nbrs}")

    # print(f"{intradecomp_map=}")
    with open(f"intradecomp_map_np{nranks}.pkl", "wb") as pkl_file:
        pickle.dump(intradecomp_map, pkl_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="MIRGE-Com Intradecomp mapper")
    parser.add_argument("-m", "--mesh", type=str, dest="mesh_filename",
                        nargs="?", action="store", help="root filename for mesh")

    args = parser.parse_args()

    main(mesh_filename=args.mesh_filename)
