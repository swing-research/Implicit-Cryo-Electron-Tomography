# ICE-TIDE: Implicit Cryo-Electron Tomography: Inference and Deformation Estimation
Official repo for ICE-TIDE [(paper)](https://arxiv.org/abs/2403.02182)

This code is still under active development. A stable version should be available soon.

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
### Simulations on model 0 of Shrec 2021
Run the complete pipleine on model0:
```
python -m experiment_scripts.simulation_model0
```

#### Only data generation
```
python -m experiment_scripts.simulation_model0 --no_train --no_aretomo --no_comparison
```

#### Only training ICE-TIDE
```
python -m experiment_scripts.simulation_model0 --no_gen_data --no_aretomo --no_comparison
```

#### Only runing AreTomo
Make sure to change the path of Aretomo in the config file
```
python -m experiment_scripts.simulation_model0 --no_gen_data --no_train --no_comparison
```

#### Only comparisons and display of different methods
```
python -m experiment_scripts.simulation_model0 --no_gen_data --no_train --no_aretomo
```

### Reproduce experiments on real data
You can run ICE-TIDE on a T. kivui tomogram from the dataset [EMPIAR-11058](https://www.ebi.ac.uk/empiar/EMPIAR-11058/).
The raw projections were first drift corrected and dose-filtered using standard procedures with [IMOD](https://bio3d.colorado.edu/imod/).
The processed tilt series file can be downladed from [Zenodo](https://doi.org/10.5281/zenodo.10979053). Files `tomo2_L1G1-dose_filt.st` and `tomo2_L1G1-dose_filt.tlt` should be placed in 'datasets/tkiuv/'
Then, the following command proccess this tilt-series and will produce the results of the paper.
```
python -m experiment_scripts.real_data_tkiuv
```
Notice that the optional arguments '--no_gen_data', '--no_train', '--no_aretomo' and '--no_comparison' can be used.


### Reproduce the timing experiment
```
python -m experiment_scripts.simulation_shrec_timing 
```

### Reproduce the experiment on all Shrec 2021 models
```
python -m experiment_scripts.simulation_all_shrec_models
```

### Reproduce the experiment for various SNRs
```
python -m experiment_scripts.simulation_SNR_influence
```



## How to cite?
Please cite the following paper if you use this code into your research:

Ice-Tide: Implicit Cryo-ET Imaging and Deformation Estimation - [https://arxiv.org/abs/2403.02182](https://arxiv.org/abs/2403.02182)


If you have any furter questions or want to discuss, reach out to one of us!

* Valentin Debarnot: valentin.debarnot@unibas.ch
 
* Vinith Kishore: vinith.kishore@unibas.ch

* Ricardo D. Righetto: ricardo.righetto@unibas.ch
 
* Ivan Dokmanić: ivan.dokmanic@unibas.ch
