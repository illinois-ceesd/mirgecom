#!/bin/bash

source miniforge3/bin/activate dgfem
set -o nounset -o errexit



echo "*** Pip info"

python3 -m pip freeze


echo
echo "*** Conda info"

echo -n "Conda path: "
command -v conda || echo "No conda found."

conda info

conda info --envs

conda list


echo
echo "*** OS info"

which lsb_release >/dev/null && lsb_release -ds
which sw_vers >/dev/null && echo "MacOS $(sw_vers -productVersion)"
uname -a


echo
echo "*** Mirgecom dev packages"

res="Package|Branch|URL\n"
res+="=======|======|======\n"

cd src/
for name in *; do
    cd $name
    branch=$(git describe --always)
    url=$(git remote show origin| grep URL | head -1 | awk '{print $3}')

    branch=${branch:----}
    url=${url:----}

    res+="$name|$branch|$url\n"
    cd ..
done

echo -e $res | column -t -s '|'
