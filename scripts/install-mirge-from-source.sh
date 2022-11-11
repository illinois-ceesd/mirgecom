#!/bin/bash
set -x
#
# This script is intended to install mirgecom from an uninstalled
# mirgecom source gotten from a fresh clone.
#
EMIRGE_INSTALL_PATH=${1:-"."}

echo "EMIRGE_INSTALL_PATH=${EMIRGE_INSTALL_PATH}"

# Install the version of mirgecom we wish to test from source
./install.sh --skip-clone --install-prefix=${EMIRGE_INSTALL_PATH}/ --conda-env=${EMIRGE_INSTALL_PATH}/mirgecom/conda-env.yml --pip-pkgs=${EMIRGE_INSTALL_PATH}/mirgecom/requirements.txt

