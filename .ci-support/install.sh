#!/bin/bash

MINIFORGE_INSTALL_DIR=.miniforge3
MINIFORGE_INSTALL_SH=Miniforge3-$PLATFORM-x86_64.sh
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/$MINIFORGE_INSTALL_SH"
rm -Rf "$MINIFORGE_INSTALL_DIR"
bash "$MINIFORGE_INSTALL_SH" -b -p "$MINIFORGE_INSTALL_DIR"
PATH="$MINIFORGE_INSTALL_DIR/bin/:$PATH" conda update conda --yes --quiet
PATH="$MINIFORGE_INSTALL_DIR/bin/:$PATH" conda update --all --yes --quiet
PATH="$MINIFORGE_INSTALL_DIR/bin:$PATH" conda env create --file conda-env.yml --name testing --quiet

source "$MINIFORGE_INSTALL_DIR/bin/activate" testing
conda install mpi4py
conda list

MINIFORGE_INSTALL_DIR=.miniforge3
source "$MINIFORGE_INSTALL_DIR/bin/activate" testing

pip install -r requirements.txt
python setup.py install
