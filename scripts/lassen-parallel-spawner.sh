#!/bin/bash
#
# Used to wrap the spawning of parallel mirgecom drivers on Lassen

export POCL_CACHE_DIR=/tmp/$USER/pocl_cache/$$

"$@"
