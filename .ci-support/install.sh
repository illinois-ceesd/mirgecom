#!/bin/bash

set -ex

myos=$(uname)
myarch=$(uname -m)

if [[ "$(uname)" = "Darwin" ]]; then
    brew update
    brew install open-mpi
    brew install octave
else
    if ! command -v mpicc &> /dev/null ;then
        sudo apt-get update
        sudo apt-get -y install mpich
    fi
    if ! command -v octave &> /dev/null ;then
        sudo apt-get -y install octave
    fi
fi

MINIFORGE_INSTALL_DIR=.miniforge3
MINIFORGE_INSTALL_SH=Miniforge3-$myos-$myarch.sh
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/$MINIFORGE_INSTALL_SH"
rm -Rf "$MINIFORGE_INSTALL_DIR"
bash "$MINIFORGE_INSTALL_SH" -b -p "$MINIFORGE_INSTALL_DIR"

# Temporarily disabled to get CI unstuck
#PATH="$MINIFORGE_INSTALL_DIR/bin/:$PATH" conda update conda --yes --quiet
#PATH="$MINIFORGE_INSTALL_DIR/bin/:$PATH" conda update --all --yes --quiet

$MINIFORGE_INSTALL_DIR/bin/mamba env create --file conda-env.yml --name testing

. "$MINIFORGE_INSTALL_DIR/bin/activate" testing
conda list

pip install -r requirements.txt
python setup.py install
