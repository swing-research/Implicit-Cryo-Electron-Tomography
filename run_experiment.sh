#!/bin/bash
# Bash script to run the experiments

# The configs are present to run the experiments you can choose one of them
# 1. default
#2. bare_bones
#3. local_implicit
#4. areTomoValidation
#5. volume_save
config='volume_save'



CUDA_VISIBLE_DEVICES=0 python3 main_data_generation.py --config $config
CUDA_VISIBLE_DEVICES=0 python3 main_training.py --config $config