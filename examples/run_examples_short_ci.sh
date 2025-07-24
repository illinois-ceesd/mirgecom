#!/bin/bash

set -o nounset

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

$SCRIPT_DIR/run_examples.sh $SCRIPT_DIR/examples wave.py
