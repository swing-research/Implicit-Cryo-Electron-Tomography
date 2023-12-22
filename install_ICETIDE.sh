conda create --name ice-tide -y python=3.8
conda activate ice-tide
python -m pip install --upgrade pip
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install odl==0.7
conda install ipython
conda install scikit-image
conda install -c jmcmurray os
conda install -c conda-forge mrcfile
conda install matplotlib
pip install numexpr
conda install -c conda-forge ml-collections
conda install -c conda-forge ipdb