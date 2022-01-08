#!/bin/bash
set -x

# Set this to the mirgecom branch you intend to land in main.  Overwrite
# whatever value is already here. This branch may live in the main repo or your
# personal fork, it does not matter.  If what is here does not match the branch
# name on which you are currently working, then none of this will have any
# effect.  As a consequence, this change will automatically deactivate itself
# once merged to main. The production branch must then manually be updated to
# track the PR production branch.
#
# The corresponding branch intended to land in production must be
# called "production-merge/$PRODUCTION_PR_MAIN_BRANCH". It may live in the main
# mirgecom repo or your personal fork, see below.
#
# This branch for production testing must be up-to-date with respect to the
# production branch in the main mirgecom repo as well as the PR branch. It will
# only be considered for production testing if the CI is currently testing a
# branch named $PRODUCTION_PR_MAIN_BRANCH, i.e.

export PRODUCTION_PR_MAIN_BRANCH=drop-vtk-from-env

# If your production branch lives in your personal mirgecom fork,
# set this to your Github user name. Otherwise, set this to an empty
# value.

export PRODUCTION_PR_FORK=

# Multiple production drivers are supported. The user should provide a ':'-delimited
# list of driver locations, where each driver location is of the form:
# "user/repo@branch". The defaults are provided below as an example. Provide custom
# production drivers in this variable:

export PRODUCTION_DRIVERS="illinois-ceesd/drivers_y1-nozzle@parallel-lazy-state-handling:illinois-ceesd/drivers_flame1d@state-handling:illinois-ceesd/drivers_y2-isolator@state-handling"

# Example:
# PRODUCTION_DRIVERS="illinois-ceesd/drivers_y1-nozzle@main:w-hagen/isolator@NS"
