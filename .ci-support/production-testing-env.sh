#!/bin/bash
set -x
#
# This script is designed to help customize the production environment
# under which CEESD production capability is tested under a proposed change
# to illinois-ceesd/mirgecom@main.
#
# The production capability may be in a CEESD-local mirgecom branch or in a
# fork, and is specified through the following settings:
#
# export PRODUCTION_BRANCH=""   # The base production branch to be installed by emirge
# export PRODUCTION_FORK=""  # The fork/home of production changes (if any)
#
# Multiple production drivers are supported. The user should provide a ':'-delimited
# list of driver locations, where each driver location is of the form:
# "fork/repo@branch". The defaults are provided below as an example. Provide custom
# production drivers in this variable:
#
export PRODUCTION_DRIVERS="illinois-ceesd/drivers_y1-nozzle@parallel-lazy-state-handling:illinois-ceesd/drivers_flame1d@state-handling:illinois-ceesd/drivers_y2-isolator@state-handling"
#
# Example:
# PRODUCTION_DRIVERS="illinois-ceesd/drivers_y1-nozzle@main:w-hagen/isolator@NS"
