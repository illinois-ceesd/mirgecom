#!/bin/bash

set -o errexit -o nounset
# Conda does not like 'set -o nounset'

echo "#####################################################"
echo "# This script installs mirgecom, and dependencies.  #"
echo "#####################################################"
echo

usage()
{
  echo "Usage: $0 [--conda-prefix=DIR]"
  echo "                   [--env-name=NAME] [--modules] [--help]"
  echo "  --conda-prefix=DIR    Install conda in [DIR], (default=./miniforge3)"
  echo "  --env-name=NAME       Name of the conda environment to install to. (default=dgfem)"
  echo "  --modules             Create modules.zip and add to Python path."
  echo "  --conda-pkgs=FILE     Install these additional packages with conda."
  echo "  --pip-pkgs=FILE       Install these additional packages with pip."
  echo "  --help                Print this help text."
}

# {{{ Default conda location

# https://stackoverflow.com/q/39340169
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
conda_prefix=$SCRIPT_DIR/miniforge3
env_name="dgfem"
pip_pkg_file=""
conda_pkg_file=""

# }}}

# Build modules.zip? (via makezip.sh)
opt_modules=0

while [[ $# -gt 0 ]]; do
  arg=$1
  shift
  case $arg in
    --conda-prefix=*)
        # Install conda in non-default prefix
        conda_prefix=${arg#*=}
        ;;
    --env-name=*)
        # Use non-default environment name
        env_name=${arg#*=}
        ;;
    --conda-pkgs=*)
        # Install these additional packages with conda
        conda_pkg_file=${arg#*=}
        ;;
    --pip-pkgs=*)
        # Install these additional packages with pip
        pip_pkg_file=${arg#*=}
        ;;
    --modules)
        # Create modules.zip
        opt_modules=1
        ;;
    --help)
        usage
        exit 0
        ;;
    *)
        usage
        exit 1
        ;;
  esac
done

# Conda does not like ~
conda_prefix=${conda_prefix//\~/$HOME}
mcprefix=$SCRIPT_DIR
export MY_CONDA_DIR=$conda_prefix

./install-conda.sh

export PATH=$MY_CONDA_DIR/bin:$PATH

echo "==== Create $env_name conda environment"

# Make sure we get the just installed conda.
# See https://github.com/conda/conda/issues/10133 for details.
#shellcheck disable=SC1090
source "$MY_CONDA_DIR"/bin/activate

conda create --name "$env_name" --yes

#shellcheck disable=SC1090
source "$MY_CONDA_DIR"/bin/activate "$env_name"

mkdir -p "$mcprefix"

./install-conda-dependencies.sh
[[ -n "$conda_pkg_file" ]] && ./install-conda-dependencies.sh "$conda_pkg_file"

echo "==== Installing pip packages for general development"

# Semi-required for pyopencl
python -m pip install mako

# Semi-required for meshpy source install, avoids warning and wait
python -m pip install pybind11

# Some nice-to haves for development
python -m pip install pytest pudb flake8 pep8-naming pytest-pudb sphinx

echo "==== Installing packages from requirements.txt"

python -m pip install -r requirements.txt

[[ -n "$pip_pkg_file" ]] && python -m pip install -r "$pip_pkg_file"

# Install an environment activation script
rm -rf "$mcprefix"/config
mkdir -p "$mcprefix"/config
cat << EOF > "$mcprefix"/config/activate_env.sh
#!/bin/bash
#
# Automatically generated by mirgecom install
#
source $MY_CONDA_DIR/bin/activate $env_name

EOF
chmod +x "$mcprefix"/config/activate_env.sh

[[ $opt_modules -eq 1 ]] && ./makezip.sh

echo
echo "==================================================================="
echo "Mirgecom and its dependencies are now installed."
echo "Before using this installation, one should load the appropriate"
echo "conda environment (assuming bash shell):"
echo " $ source $mcprefix/config/activate_env.sh"
echo "To test the installation:"
echo " $ cd test && pytest *.py"
echo "To run the examples:"
echo " $ cd examples && ./run_examples.sh ./"
echo "==================================================================="
