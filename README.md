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

### Python environment installation
The Python environment can be installed using the following command. It will install all the required Python packgaes to properly run ICE-TIDE. 
Notice that the installation script should be run using 'source'.
```
bash install_ICETIDE.sh
```

Then, you can simply activate you conda environment using the follwoing command every time you start a new session
```
conda activate ice-tide
```

### Download simulation dataset
Please download the [(Shrec 2021)](https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/XRTJMA) dataset and move the files in the folder datasests.
The structure of the folder should be 'datasets/model_X/grandmodel.mrc', where model_X is specified by the variable volume_name in the config files

## How to use
### Simulations
Run the complete pipleine on model0:
```
python -m experiment_scripts.simulation_model0
```

### Only data generation
```
python -m experiment_scripts.simulation_model0 --no_train --no_aretomo --no_comparison
```

### Only training ICE-TIDE
```
python -m experiment_scripts.simulation_model0 --no_gen_data --no_aretomo --no_comparison
```

### Only runing AreTomo
Make sure to change the path of Aretomo in the config file
```
python -m experiment_scripts.simulation_model0 --no_gen_data --no_train --no_comparison
```

### Only comparisons and display of different methods
```
python -m experiment_scripts.simulation_model0 --no_gen_data --no_train --no_aretomo
```

### Use your own dataset
You can run ICE-TIDE on your own dataset, for that you can mimic the procedure that we used to process 
Assuming that your mrc file containing the projection is saved in 'datasets/tkiuv/tomo2_L1G1_ODD.mrc', you can run the following command to proccess this tilt-series
```
python -m experiment_scripts.real_data_tkiuv
```
Notice that the optional arguments '--no_gen_data', '--no_train', '--no_aretomo' and '--no_comparison' can be used.



### Reproduce the timing 
```
python -m experiment_scripts.simulation_shrec_timing 
```



## Warning
It will only work on Linux and MacOS systems.


## How to cite?
Please cite the following paper if you use this code into your research:
XXXXXXXXXXXXXXXXXXxx


