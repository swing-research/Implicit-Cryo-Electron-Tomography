# ICE-TIDE: Implicit Cryo-Electron Tomography: Inference and Deformation Estimation
Official repo for ICE-TIDE [(paper)](https://arxiv.org/abs/2403.02182).

This code is a fast and more efficient version of the original version used to generate the results of the paper.

With the current version, you can hope to align tilt-series containing 60 tilts of size 3096x3096 in about 5min. 
Default parameters should already provide a decent result.
This version should be used to obtain deformation parameters (shift, tilt-axis, in-plane rotation and local deformation) and 
are removed from the original tilt-series. The implicit neural network output can be computed but usually lack high resolution details
that are required to properly analyze the data. This is mainly the result of speeding up the estimation procedure. 
We believe that estimating the deformations is more stable and more accurate as it is unlikely to hallucinate erroneous structures. 


## Get started
### Python environment installation
The Python environment can be installed using the following command (using mamba). It will install all the required Python packages to properly run ICE-TIDE. 
```
source install_ICETIDE_mamba.sh
```


Alternatively, a more conventional way, but usually longer, of installing the required packages without using mamba is to run the following command 
```
source install_ICETIDE.sh
```

The challenging part can be to install [tiny cuda](https://github.com/NVlabs/tiny-cuda-nn). Once it is installed, any recent version of the standard packages in the instal file should work.

### Activate the environement
Then, you can simply activate you conda environment using the follwoing command every time you start a new session
```
conda activate ice-tide
```

## Run ICE-TIDE on your data
### Quick start
The quickest way to get started is to change the name of your tilt-series and place them in the folder 'dataset'.
Place your mrc file in 'dataset/my_data.mrc' and the angle file as 'dataset/my_data_tilts.tlt' and run
```
python -m run_icetide
```


### Stable procedure
The recommended approach   is to create your own config file, so that you can directly input the path of your data and chose the location where the results are saved. 
For that, create a config file in the folder 'configs'.
For example, after creating a file 'config_my_data.yaml', run the following command

```
python -m run_icetide --path_config real_data_fast.yaml
```

#### Using multiple tilt-series
There are multiple ways to include one or more files to process.
You can set _path_volume_ in the config file as:
- The full path to an mrc file: 'path/to/tilt_series.mrc'. Tilt-angles will be read directly from the path given in _path_angle_, if a valid one is given. 
- A folder, all the mrc files contained in this folder will be processed. Tilt-angles are expected to be in the same folder, with the same name as the tilt-series, with the extension _.tlt_. Argument _path_angle_ can be ignored.
- Using '\*' to refer to all files to be considered. Example: 'path/to/tilt_series_num_123_*.mrc'. Tilt angles are expected to be provided the same way in argument _path_angle_.
- List of tilt-series:
  ``` 
  path_volume:
   - 'path/to/tilt_series1.mrc'
   - 'path/to/other_tilt_series.mrc'
   - 'path/to/a_third/tilt_series.mrc'
    ```
  Tilt-angle files can be provided the same way.
### Optional arguments
Previously trained model can be used to only generate the output files, e.g. aligned tilt-series or FBP reconstruction.
This can be done using the argument 'no_training'.

```
python -m run_icetide --path_config 'real_data_fast.yaml' --no_training
```

The model can also be trained while ignoring its evaluation. This can be done using the argument 'no_evaluation'.
```
python -m run_icetide --path_config 'real_data_fast.yaml' --no_evaluation
```



## Original version
Previous version, the one corresponding to the result of the paper, can ba accessed on the Stable branch
```
git checkout Stable
```

The fast version is known as the 'faster' branch 
```
git checkout faster
```



## How to cite?
Please cite the following paper if you use this code into your research:

Debarnot, V., Kishore, V., Righetto, R. D., & Dokmanic, I. (2024). Ice-tide: Implicit cryo-et imaging and deformation estimation. IEEE Transactions on Computational Imaging.
[PDF](https://arxiv.org/abs/2403.02182).

If you have any furter questions or want to discuss, reach out to one of us!

* [Valentin Debarnot](https://sites.google.com/view/debarnot/home): valentin.debarnot@creatis.insa-lyon.fr
 
* Vinith Kishore: vinith.kishore@unibas.ch

* Ricardo D. Righetto: ricardo.righetto@unibas.ch
 
* Ivan Dokmanić: ivan.dokmanic@unibas.ch