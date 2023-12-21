# ICE-TIDE: Implicit Cryo-Electron Tomography: Inference and Deformation Estimation

Official repo for ICE-TIDE [(paper)](https://arxiv.org/abs/)

ICE-TIDE is a self-supervised machine learnign method to perform jointly alignment of tilt series in cryogenic electron tomography (cryo-ET) iand reconstruction.
ICE-TIDE relies on a fast and efficient implicit neural network [tiny cuda](https://github.com/NVlabs/tiny-cuda-nn). 
We show the benefits of using an implicit neural network to estimate a three-dimensional volume in a high-noise regime. 
We show that implicit neural networks provide a powerful prior for regularization and allows easily incorporating deformations in coordinate space.
We evaluate the performance of ICE-TIDE in comparison to existing approaches on simulations where the gain in resolution can be precisely evaluated.
ICE-TIDE's ability to perform on experimental data sets is also demonstrated.




## Get started
This repo contains the scripts used to produce the experiments in the associated paper.

### Python environment
You can then set up a conda environment with all dependencies like so:
```
conda env create -f ice-tide.yml
conda activate ice-tide
```

### Install [tiny cuda](https://github.com/NVlabs/tiny-cuda-nn)
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

### Download simulation dataset
Please download the [(Shrec 2021)](https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/XRTJMA) dataset and move the files in the folder datasests.
The structure of the folder should be 'datasets/model_X/grandmodel.mrc', where model_X is specified by the variable volume_name in the config files

## How to use
### Simulations
'''
python -m experiment_scripts.simulation_model0
'''

### Only data generation
'''
python -m experiment_scripts.simulation_model0 --no_train --no_aretomo --no_comparison
'''

### Only training ICE-TIDE
'''
python -m experiment_scripts.simulation_model0 --no_gen_data --no_aretomo --no_comparison
'''

### Only run AreTomo
Make sure to change the path of Aretomo in the config file
'''
python -m experiment_scripts.simulation_model0 --no_gen_data --no_train --no_comparison
'''

### Only comparisons and display of different methods
'''
python -m experiment_scripts.simulation_model0 --no_gen_data --no_train --no_aretomo
'''

### Use your own dataset
put your tilt-series here ... and run ...







Warning: it will only work on Linux and MacOS systems.

