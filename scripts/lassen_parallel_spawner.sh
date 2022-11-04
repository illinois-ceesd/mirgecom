#!/bin/bash

export POCL_CACHE_DIR=/tmp/$USER/pocl_cache/$$

"$@"
