#!/bin/bash
#
# Usage: install_driver.sh <driver_config_string> [driver_install_root]
# 
# Install (clone) a driver from its repository using an input
# configuration string with format "fork/repo@branch" like this example:
# "illinois-ceesd/drivers_y2-prediction@main"
#
# The example driver will be installed to a directory like:
# ${driver_install_root}_drivers_y2-prediction
#
driver_config_string=${1}
driver_install_root=${2:-""}

if [[ ! -z "${driver_install_root}" ]]; then
    driver_install_root="${driver_install_root}_"
fi

DRIVER_BRANCH=$(printf "$driver_config_string" | cut -d "@" -f 2)
DRIVER_REPO=$(printf "$driver_config_string" | cut -d "@" -f 1)
DRIVER_NAME=$(printf "$DRIVER_REPO" | cut -d "/" -f 2)
DRIVER_INSTALL_DIR="${driver_install_root}$DRIVER_NAME"

printf "Cloning ${DRIVER_REPO}:/${DRIVER_NAME}@${DRIVER_BRANCH} to ${DRIVER_INSTALL_DIR}.\n"

rm -rf $DRIVER_INSTALL_DIR
git clone -b "$DRIVER_BRANCH" https\://github.com/"$DRIVER_REPO" "$DRIVER_INSTALL_DIR"
