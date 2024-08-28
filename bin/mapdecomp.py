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
        mesh_pkl_filename = mesh_filename + f"_rank{r}.pkl"
        if os.path.exists(mesh_pkl_filename):
            nranks = nranks + 1
            with open(mesh_pkl_filename, "rb") as pkl_file:
                global_nelements, volume_to_local_mesh_data = \
                    pickle.load(pkl_file)
            for vol, meshdat in volume_to_local_mesh_data.items():
                local_partid = PartID(volume_tag=vol, rank=r)
                volumes.add(vol)
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
