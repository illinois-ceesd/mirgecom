if [ "$(uname)" = "Darwin" ]; then
PLATFORM=MacOSX
brew install open-mpi
brew install octave
else
PLATFORM=Linux
sudo apt-get -y install openmpi-bin libopenmpi-dev
sudo apt-get -y install octave
fi
MINIFORGE_INSTALL_DIR=.miniforge3
MINIFORGE_INSTALL_SH=Miniforge3-$PLATFORM-x86_64.sh
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/$MINIFORGE_INSTALL_SH"
rm -Rf "$MINIFORGE_INSTALL_DIR"
bash "$MINIFORGE_INSTALL_SH" -b -p "$MINIFORGE_INSTALL_DIR"
PATH="$MINIFORGE_INSTALL_DIR/bin/:$PATH" conda update conda --yes --quiet
PATH="$MINIFORGE_INSTALL_DIR/bin/:$PATH" conda update --all --yes --quiet
PATH="$MINIFORGE_INSTALL_DIR/bin:$PATH" conda env create --file .test-conda-env-py3.yml --name testing --quiet

. "$MINIFORGE_INSTALL_DIR/bin/activate" testing

MINIFORGE_INSTALL_DIR=.miniforge3
. "$MINIFORGE_INSTALL_DIR/bin/activate" testing
pip install -r requirements.txt
python setup.py install
