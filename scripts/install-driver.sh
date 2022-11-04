#!/bin/bash

driver_config_string=${1}
driver_install_root=${2:-"production_driver"}

printf "Installing from $driver_config_string.\n"

DRIVER_BRANCH=$(printf "$driver_config_string" | cut -d "@" -f 2)
DRIVER_REPO=$(printf "$driver_config_string" | cut -d "@" -f 1)
DRIVER_NAME=$(printf "$DRIVER_REPO" | cut -d "/" -f 2)
DRIVER_INSTALL_DIR="${driver_install_root}_$DRIVER_NAME"

rm -rf $DRIVER_INSTALL_DIR
git clone -b "$DRIVER_BRANCH" https\://github.com/"$DRIVER_REPO" "$DRIVER_INSTALL_DIR"
