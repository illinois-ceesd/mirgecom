#!/bin/bash

set -o errexit -o nounset

zipfile=$PWD/modules.zip

rm -f "$zipfile"

MY_MODULES=""

cd src/

for name in *; do
    # Skip non-Python submodules
    [[ -f "$name/setup.py" ]] || continue

    MY_MODULES+="$name "

    cd "$name"

    [[ $name == "loo-py" ]] && name=loopy

    echo "=== Zipping $name"
    zip -r "$zipfile" "$name"
    cd ..
done

MY_PYTHON=$(command -v python)

echo "=== Preparing path file of '$MY_PYTHON'"
echo "=== for importing modules from '$zipfile'"
echo

sitefile="$($MY_PYTHON -c 'import site; print(site.getsitepackages()[0])')/emirge.pth"

echo "$zipfile" > "$sitefile"

echo "=== Done. Make sure to uninstall other copies of the emirge modules:"
echo "=== $MY_PYTHON -m pip uninstall $MY_MODULES"
echo "=== and verify that the correct modules can be loaded by running:"
echo "=== $MY_PYTHON -c 'import dagrt; print(dagrt.__path__)'"
