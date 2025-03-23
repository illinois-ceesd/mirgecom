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


# Persistent storage for interface data
gid = {
    "domains": set(),
    "nranks": 0,
    # {rank: {remote_rank: {(dom1, dom2): count}}}
    "mpi_interfaces": defaultdict(lambda: defaultdict(lambda: defaultdict(int))),
    # {rank: {(dom1, dom2): count}}
    "internal_interfaces": defaultdict(lambda: defaultdict(int)),
    # {rank: {remote_rank1, remote_rank2, ...}}
    "remote_ranks": defaultdict(set),
}


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
    Store interface data for later analysis, instead of directly printing results.

    Parameters:
    - mesh: The local mesh data for this rank.
    - local_partid: The PartID of the current rank's domain.
    """
    r = local_partid.rank
    local_domain = local_partid.volume_tag

    for fagrp_list in mesh.facial_adjacency_groups:
        for grp in fagrp_list:
            if isinstance(grp, InterPartAdjacencyGroup):
                remote_rank = grp.part_id.rank
                remote_domain = grp.part_id.volume_tag
                num_faces = len(grp.neighbor_faces)
                interface_key = (local_domain, remote_domain)

                if remote_rank == r:
                    # Internal domain-domain interface
                    gid["internal_interfaces"][r][interface_key] += num_faces
                else:
                    # MPI interface
                    gid["remote_ranks"][r].add(remote_rank)
                    gid["mpi_interfaces"][r][remote_rank][interface_key] += num_faces


def interfaces_report():

    all_checks_passed = True
    total_partition_boundary_faces = 0
    total_local_domain_interface_faces = 0
    total_remote_domain_interface_faces = 0
    total_interface_faces = 0
    inter_domain_faces = {}
    domain_ranks = {}
    seen_interfaces = set()
    gid_mpi = sorted(gid["mpi_interfaces"].items())
    gid_idx = sorted(gid["internal_interfaces"].items())
    nranks = gid["nranks"]
    counted_mpi_pairs = set()
    domains = gid["domains"]

    print(f"Number of ranks: {nranks}")
    print(f"Domains: {domains}")

    print("--------- Rank-by-Rank Report -----------")
    for rank in range(nranks):
        total_interface_faces_rank = 0
        total_local_domain_interfaces_rank = 0
        total_remote_domain_interfaces_rank = 0
        remote_rank_data = gid["mpi_interfaces"][rank]
        nmpi_bnd = len(gid["remote_ranks"][rank])
        total_mpi_faces = sum(
            sum(iface_counts.values()) for iface_counts in remote_rank_data.values()
        )
        total_partition_boundary_faces += total_mpi_faces
        total_interface_faces_rank += total_mpi_faces
        print(f"- Rank {rank}: {total_mpi_faces} faces / {nmpi_bnd} MPI boundaries")
        print("-- Remote interfaces:")
        report = []
        for remote_rank, iface_counts in sorted(remote_rank_data.items()):
            # print(f"-- Remote rank {remote_rank}:")
            rri_report = []
            for (dom1, dom2), face_count in iface_counts.items():
                rri_report.append(f"{dom1}↔{dom2}({face_count})")
                # print(f"      - {dom1} ↔ {dom2}: {face_count} faces")
                if dom1 != dom2:
                    total_remote_domain_interfaces_rank += face_count
            report.append(f"--- Remote rank({remote_rank}): {' '.join(rri_report)}")
        print("\n".join(report))

        seen_interfaces_rank = set()
        if rank in gid["internal_interfaces"]:
            print("-- Internal interfaces")
            idr = gid["internal_interfaces"][rank]
            for (dom1, dom2), face_count in sorted(idr.items()):
                if (dom2, dom1) not in seen_interfaces_rank:
                    if (dom1, dom2) not in seen_interfaces:
                        if (dom2, dom1) not in seen_interfaces:
                            seen_interfaces.add((dom1, dom2))
                    print(f"    - {dom1} ↔ {dom2}: {face_count} faces")
                    total_local_domain_interface_faces += face_count
                    seen_interfaces_rank.add((dom1, dom2))


def validate_and_report_interfaces2():
    print("\n==================== Interface Analysis Report ====================\n")

    all_checks_passed = True
    total_partition_boundary_faces = 0
    total_local_domain_interface_faces = 0

    gid_mpi = sorted(gid["mpi_interfaces"].items())
    gid_idx = sorted(gid["internal_interfaces"].items())

    # print("MPI INTERFACES:")
    for rank, remote_rank_data in gid_mpi:
        total_mpi_faces = sum(
            sum(iface_counts.values()) for iface_counts in remote_rank_data.values()
        )
        total_partition_boundary_faces += total_mpi_faces
        nmpi_bnd = len(gid["remote_ranks"][rank])
        print(f"  Rank {rank}: {total_mpi_faces} faces / {nmpi_bnd} MPI boundaries")

        for remote_rank, iface_counts in sorted(remote_rank_data.items()):
            print(f"    - MPI interface with rank {remote_rank}:")
            for (dom1, dom2), face_count in iface_counts.items():
                print(f"      - {dom1} ↔ {dom2}: {face_count} faces")

    print("\nINTERNAL INTERFACES:")
    for rank, internal_data in gid_idx:
        print(f"  Rank {rank}:")
        seen_interfaces_rank = set()
        for (dom1, dom2), face_count in sorted(internal_data.items()):
            if (dom2, dom1) not in seen_interfaces_rank:
                print(f"    - {dom1} ↔ {dom2}: {face_count} faces")
                total_local_domain_interface_faces += face_count
                seen_interfaces_rank.add((dom1, dom2))

    for rank, internal_data in gid_idx:
        for (dom1, dom2), count in internal_data.items():
            reverse_key = (dom2, dom1)
            if reverse_key in internal_data:
                if count != internal_data[reverse_key]:
                    print(f"  ERROR: Internal {dom1} ↔ {dom2} mismatch on rank "
                          f"{rank}! {count} vs {internal_data[reverse_key]}")
                    all_checks_passed = False
            else:
                print(f"  ERROR: Missing internal {dom2} ↔ {dom1} interface on "
                      f"rank {rank}!")
                all_checks_passed = False

    for rank, remote_rank_data in gid["mpi_interfaces"].items():
        for remote_rank, iface_counts in remote_rank_data.items():
            for (dom1, dom2), count in iface_counts.items():
                expected_key = (dom2, dom1)
                all_keys = gid["mpi_interfaces"][remote_rank].get(rank, {})
                if remote_rank in gid["mpi_interfaces"] and expected_key in all_keys:
                    mfc = gid["mpi_interfaces"][remote_rank][rank][expected_key]
                    if count != mfc:
                        print(f"  ERROR: MPI {rank} ({dom1} ↔ {dom2}) ≠ "
                              f"{remote_rank} ({dom2} ↔ {dom1})!")
                        all_checks_passed = False
                else:
                    print(f"  ERROR: Missing reciprocal MPI interface for "
                          f"{rank} ↔ {remote_rank} ({dom2} ↔ {dom1})!")
                    all_checks_passed = False

    print("\n==================== Additional Metrics ====================")
    print(f"  Total MPI Partition Boundary Faces (without double-counting): "
          f"{total_partition_boundary_faces}")
    print("  Total Local Domain Interface Faces: "
          f"{total_local_domain_interface_faces}")

    if all_checks_passed:
        print("  All sanity checks passed!\n")
    else:
        print("  Sanity checks found errors! Review the output above.\n")


def count_matching_files(mesh_filename):
    count = 0
    while os.path.exists(f"{mesh_filename}_rank{count}.pkl"):
        count += 1
    return count


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
    nranks = count_matching_files(mesh_filename)
    print(f"Mapdecomp found {nranks} mesh partitions.")
    gid["nranks"] = nranks

    for r in range(nranks):
        mesh_pkl_filename = mesh_filename + f"_rank{r}.pkl"
        with open(mesh_pkl_filename, "rb") as pkl_file:
            global_nelements, volume_to_local_mesh_data = \
                pickle.load(pkl_file)
        for vol, meshdat in volume_to_local_mesh_data.items():
            local_partid = PartID(volume_tag=vol, rank=r)
            # print(f"Processing volume: {vol}")
            gid["domains"].add(vol)
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
    # print("===================== 1st version =============")
    # validate_and_report_interfaces()
    interfaces_report()
    print("===================== 2nd version =============")
    validate_and_report_interfaces2()
    print("===================== Old version =============")

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
