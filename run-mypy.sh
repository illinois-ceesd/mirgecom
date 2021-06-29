#!/bin/bash

python -m mypy --show-error-codes $(basename $PWD) examples test
