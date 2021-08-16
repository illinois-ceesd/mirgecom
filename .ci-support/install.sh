if [ "$(uname)" = "Darwin" ]; then
PLATFORM=MacOSX
brew update
brew install open-mpi
brew install octave
else
PLATFORM=Linux
sudo apt-get update
sudo apt-get -y install libmpich-dev mpich
sudo apt-get -y install octave
fi
MINIFORGE_INSTALL_DIR=.miniforge3
MINIFORGE_INSTALL_SH=Miniforge3-$PLATFORM-x86_64.sh
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/$MINIFORGE_INSTALL_SH"
rm -Rf "$MINIFORGE_INSTALL_DIR"
bash "$MINIFORGE_INSTALL_SH" -b -p "$MINIFORGE_INSTALL_DIR"
PATH="$MINIFORGE_INSTALL_DIR/bin/:$PATH" conda update conda --yes --quiet
PATH="$MINIFORGE_INSTALL_DIR/bin/:$PATH" conda update --all --yes --quiet
PATH="$MINIFORGE_INSTALL_DIR/bin:$PATH" conda env create --file conda-env.yml --name testing --quiet

. "$MINIFORGE_INSTALL_DIR/bin/activate" testing
conda list

MINIFORGE_INSTALL_DIR=.miniforge3
. "$MINIFORGE_INSTALL_DIR/bin/activate" testing
pip install -r requirements.txt
python setup.py install
