Official repo for ICE-TIDE [(paper)](https://arxiv.org/abs/)

ICE-TIDE is a self-supervised machine learnign method to perform jointly alignment of tilt series in cryogenic electron tomography (cryo-ET) iand reconstruction.
ICE-TIDE relies on a fast and efficient implicit neural network [tiny cuda](https://github.com/NVlabs/tiny-cuda-nn). 
We show the benefits of using an implicit neural network to estimate a three-dimensional volume in a high-noise regime. 
We show that implicit neural networks provide a powerful prior for regularization and allows easily incorporating deformations in coordinate space.
We evaluate the performance of ICE-TIDE in comparison to existing approaches on simulations where the gain in resolution can be precisely evaluated.
ICE-TIDE's ability to perform on experimental data sets is also demonstrated.

This repo contains the scripts used to produce the experiments in the associated paper.





## Prepare for the ICE-TIDE

# Install everything
using requierments

# Install [tiny cuda](https://github.com/NVlabs/tiny-cuda-nn)
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Download simulation dataset
Please download the [(Shrec 2021)](https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/XRTJMA) dataset and move the files in the folder datasests.
The structure of the folder should be 'datasets/model_X/grandmodel.mrc', where model_X is specified by the variable volume_name in the config files

# Use your own dataset
put your tilt-series here ... and run ...


## How to use
python -m experiment_scripts.simulation_model0






```
usage: train.py [-h] [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE] [--dataset DATASET] [--lr LR]
                [--ml_threshold ML_THRESHOLD] [--model_depth MODEL_DEPTH] [--latent_depth LATENT_DEPTH] [--learntop LEARNTOP]
                [--gpu_num GPU_NUM] [--remove_all REMOVE_ALL] [--desc DESC] [--train] [--notrain] [--inv] [--noinv] [--posterior]
                [--noposterior] [--calc_logdet] [--nocalc_logdet] [--inv_prob INV_PROB] [--snr SNR]
                [--inv_conv_activation INV_CONV_ACTIVATION] [--T T]

optional arguments:
  -h, --help            show this help message and exit
  --num_epochs NUM_EPOCHS
                        number of epochs to train for
  --batch_size BATCH_SIZE
                        batch_size
  --dataset DATASET     which dataset to work with
  --lr LR               learning rate
  --ml_threshold ML_THRESHOLD
                        when should ml training begin
  --model_depth MODEL_DEPTH
                        revnet depth of model
  --latent_depth LATENT_DEPTH
                        revnet depth of latent model
  --learntop LEARNTOP   Trainable top
  --gpu_num GPU_NUM     GPU number
  --remove_all REMOVE_ALL
                        Remove the previous experiment
  --desc DESC           add a small descriptor to folder name
  --train
  --notrain
  --inv
  --noinv
  --posterior
  --noposterior
  --calc_logdet
  --nocalc_logdet
  --inv_prob INV_PROB   choose from denoising (default) | sr | randmask | randgauss
  --snr SNR             measurement SNR (dB)
  --inv_conv_activation INV_CONV_ACTIVATION
                        activation of invertible 1x1 conv layer
  --T T                 sampling tempreture

```


Warning: it will only work on Linux and MacOS systems.


Use 
'''
conda install --file requirements.txt
'''
