conda create --name ice_tide_test -y python=3.8
conda activate ice_tide_test

sleep 10
python -m pip install --upgrade pip
conda install -y conda-forge::libmamba
mamba install -y conda-forge::tomopy
pip3 install torch torchvision torchaudio
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
mamba install -y anaconda::ipython
mamba install -y -c jmcmurray os
mamba install -y -c conda-forge mrcfile
mamba install -y matplotlib
mamba install -y -c conda-forge ml-collections
mamba install -y -c conda-forge ipdb
mamba install -y conda-forge::glob2

##glob
#
#mamba install simpleitk::simpleitk
#mamba install -y anaconda::pandas
#pip install numexpr
#mamba install -y -c astra-toolbox astra-toolbox
#pip install bm3d
