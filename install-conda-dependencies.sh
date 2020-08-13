#!/bin/bash

#
# This script will install conda packages (in the current conda env) to satisfy
# mirgecom dependency requirements. It optionally accepts an argument that is
# the path to a file which should have a list of packages to install.
#
# Usage: install-conda-dependencies.sh [path/to/optional/packages.txt]
#

set -o errexit
pkg_file=""
if [[ ! -z "$1" ]]
then
    pkg_file="$1"
    if [[ ! -f "$pkg_file" ]]
    then
        echo "install-conda-dependencies.sh::Error: Package file ($pkg_file) not found."
        exit 1
    fi
fi

echo "==== Installing Conda packages with $(which conda)"

conda info --envs

if [[ ! -z "$pkg_file" ]]
then

    echo "  == Installing packages from file ($pkg_file)."
    # shellcheck disable=SC2013
    for package in $(cat "$pkg_file"); do
        echo "   -- Installing user-custom package ($package)."
        conda install --yes "$package"
    done

else

    if [[ $(uname) == "Darwin" ]]; then
        conda install --yes pocl libhwloc=1
    else
        conda install --yes pocl-cuda libhwloc=1
    fi

    conda install --yes pyvisfile pyopencl islpy clinfo

    # We need an MPI installation to build mpi4py.
    # Install OpenMPI if none is available.
    if ! command -v mpicc &> /dev/null ;then
        conda install --yes openmpi openmpi-mpicc
    fi
fi
