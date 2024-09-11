M-to-N Restart Guide
=====================

This document describes how to perform an m-to-n restart, which involves restarting a parallel application on `N` processors from restart files that were generated on `M` processors. This process requires creating configuration files that provide mappings from the `M`-partitioning of the discrete geometry to the `N`-partitioning.

Overview
--------

In a parallel |mirgecom| simulation, the geometry and solution data are partitioned across multiple processors. When restarting a simulation with a different number of processors, it is necessary to re-partition the data. This guide covers the steps and tools needed to achieve this, including the use of `meshdist` and `redist` utilities.

**Key Utilities:**

- |mirgecom|: The simulation application generates restart files on `M` processors
- **meshdist** (`bin/meshdist.py`): Reads the `gmsh`-format mesh source file, and creates mapping files and mesh PKL files for both `M` and `N` partitions.
- **redist** (`bin/redist.py`): Reads the output of the `meshdist` runs and the |mirgecom| user's simulation application restart files generated on `M` processors, and generates a |mirgecom| restart dataset for `N` processors.

Workflow
--------

The general workflow for an m-to-n restart involves the following steps:

0. **Run `mirgecom` on `M` processors**: This step generates restart data specific for `M`-partitions/processors in `M` PKL files (one for each simulation rank).
   This step creates, for example, restart files with names formatted like `<casename>-<step-number>-<rank>.pkl` such as the following for step number `12` for a 100-rank simulation:

   .. code-block:: bash

      <casename>-000000012-00000.pkl
      ...
      <casename>-000000012-00099.pkl


1. **Run `meshdist` for `M` partitions**: Generate the initial mapping and PKL files for the `M`-partitioned geometry.

   .. code-block:: bash

      mpiexec -n P python -m mpi4py bin/meshdist.py [-w] [-1] -d <dimension> -s <mesh-source-file> -c <casename> -n M -o <m-output-directory> -z <imbalance-tolerance>

   The `-1` option will turn on 1d partitioning, and the `-z` option sets the 1d partitioning tolerance for partition imbalance.  The default is `1%` (i.e. the partitions sizes (number of elements) will be within `1%` of each other).  It is important that these partitioning parameters match those that were used by your |mirgecom| simulation run.  This step will create mesh files with filenames formatted by `<m-output-directory>/<casename>_mesh_np<M>_rank<rank>.pkl`, and mapping files with filenames formatted by `<m-output-directory>/<casename>_mesh_*decomp_np<M>.pkl`

   .. warning::

      The `meshdist` utility is specific to CEESD prediction, and would likely need customization or generalization for use in other cases.  Specifically, it will automatically look for the `fluid` volume, and the `-w` option sets multivolume to `ON` and looks for the `wall`, `wall-insert`, and `wall-surround` volumes inside the mesh.

   .. note::

      The `meshdist` utility will run on any number of processors `P`.  If `P > M` (or the number of ranks specified by the `-n` option), then the additional processors will sit idle. Larger meshes can benefit from using `P > 1`, and very large meshes will require processing on larger resource sets due to platorm memory constraints.

2. **Run `meshdist` for `N` partitions**: Generate the mapping and PKL files for the `N`-partitioned geometry.

   .. code-block:: bash

      mpiexec -n P python -m mpi4py bin/meshdist.py [-w] [-1] -d <dimension> -s <mesh-source-file> -c <casename> -n N -o <n-output-directory> -z <imbalance-tolerance>


3. **Run `redist` to create the restart dataset for `N` processors**: This step will use the outputs from the two `meshdist` runs and the user's restart files.

   .. code-block:: bash

      mpiexec -n P python -m mpi4py bin/redist.py -m M -n N -s <m-mesh-directory> -t <n-mesh-directory> -o <restart-output-directory> -i <input-path/root-filename>


   The `redist` utility will read in the files created in the `M`-specific `meshdist` run, and the `N`-specific `meshdist` run, which shall be found in the `<m-mesh-directory>` and `<n-mesh-directory>` directories, respectively. It will find the existing |mirgecom| restart dataset (for `M` ranks) in the  `<input-path>` directory with filenames formatted as `<root-filename>_<rank>.pkl`. (For the example 100-rank |mirgecom| dataset above, one should specify `-i <path-to-restart-files>/<casename>_<step-number>` to `redist`) Upon successful completion `redist` will write a new restart dataset to `<restart-output-directory>/<root-filename>_<rank>.pkl`.


   .. note::

      The m-to-n restart procedure should make no changes to the |mirgecom| solution. It should only "re-partition" the existing solution to a different partitioning. There should be no transient introduced into the simulation upon restart.

