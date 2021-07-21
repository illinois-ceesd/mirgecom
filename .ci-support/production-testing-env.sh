#!/bin/bash
set -x
#
# This script is designed to help customize the production environemtn
# under which CEESD production capability is tested under a proposed change
# to illinois-ceesd/mirgecom@main. If necessary, the script sets the
# following vars:
#
# The proposed changes to test may be in a fork, or a local branch. For
# forks, the environment config files should set:
#
# export DEVELOPMENT_FORK=""   # the fork/home of the development
#
# The production capability to test against may be specified outright, or
# patched by the incoming development. The following vars control the
# production environment:
#
# export PRODUCTION_BRANCH=""   # The base production branch to be installed by emirge
# export PRODUCTION_FORK=""  # The fork/home of production changes (if any)
#
# Multiple production drivers are supported. The user should provide a ':'-delimited
# list of driver locations, where each driver location is of the form:
# "fork/repo@branch". The defaults are provided below as an example. Provide custom
# production drivers in this variable:
#
# export PRODUCTION_DRIVERS=""
#
# Example:
# PRODUCTION_DRIVERS="illinois-ceesd/drivers_y1-nozzle@main:w-hagen/isolator@NS"
