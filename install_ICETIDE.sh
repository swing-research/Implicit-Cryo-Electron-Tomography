conda create --name ice-tide3 -y python=3.8
conda activate ice-tide3
sleep 10
python -m pip install --upgrade pip
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
conda install -y -c "nvidia/label/cuda-11.7.1" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install odl==0.7
conda install -y numpy=1.23.4
pip install bm3d
conda install -y -c astra-toolbox astra-toolbox
conda install -y ipython
conda install -y scikit-image
conda install -y -c jmcmurray os
conda install -y -c conda-forge mrcfile
conda install -y matplotlib
pip install numexpr
conda install -y -c conda-forge ml-collections
conda install -y -c conda-forge ipdb
conda install simpleitk::simpleitk
conda install -y anaconda::pandas 
