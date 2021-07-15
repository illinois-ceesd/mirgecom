#!/bin/bash
#
# This script is designed to help customize the production environemtn
# under which CEESD production capability is tested under a proposed change
# to illinois-ceesd/mirgecom@main. If necessary, the script sets the
# following vars:
#
# The proposed changes to test may be in a fork, or a local branch. For
# forks, the environment config files should set:
#
# DEVELOPMENT_FORK=""   # the fork/home of the development
#
# The production capability to test against may be specified outright, or
# patched by the incoming development. The following vars control the
# production environment:
#
PRODUCTION_BRANCH="y1-production-testing"   # The base production branch to be installed by emirge
# PRODUCTION_CHANGE_FORK=""  # The fork/home of production changes (if any)
# PRODUCTION_CHANGE_BRANCH=""  # Branch from which to pull prod changes (if any)
#
# The production driver repo is specified by the following vars:
#
# PRODUCTION_DRIVER_FORK=""    # the fork/home of the driver
# PRODUCTION_DRIVER_NAME=""    # the repo for the driver
# PRODUCTION_DRIVER_BRANCH=""  # the branch for the driver
