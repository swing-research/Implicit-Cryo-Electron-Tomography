conda create --name ice_tide -y python=3.8
conda activate ice_tide

sleep 10
python -m pip install --upgrade pip
conda install -y conda-forge::tomopy
pip3 install torch torchvision torchaudio
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
conda install -y anaconda::ipython
conda install -y -c jmcmurray os
conda install -y -c conda-forge mrcfile
conda install -y matplotlib
conda install -y -c conda-forge ml-collections
conda install -y -c conda-forge ipdb
conda install -y conda-forge::glob2
