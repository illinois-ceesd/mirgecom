#!/bin/bash

set -o nounset -o errexit

myos=$(uname)
[[ $myos == "Darwin" ]] && myos="MacOSX"

myarch=$(uname -m)

# Use $HOME/miniforge3 by default
MY_CONDA_DIR=${MY_CONDA_DIR:-$HOME/miniforge3}

if [[ ! -d $MY_CONDA_DIR || ! -x $MY_CONDA_DIR/bin/conda ]]; then
    echo "==== Installing Miniforge in $MY_CONDA_DIR."
    wget -c --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$myos-"$myarch.sh"
    bash Miniforge3-$myos-"$myarch.sh" -u -b -p "$MY_CONDA_DIR"
else
    echo "==== Conda found in $MY_CONDA_DIR, skipping Miniforge installation."
fi
