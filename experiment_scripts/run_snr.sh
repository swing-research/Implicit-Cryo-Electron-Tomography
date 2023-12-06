#!/bin/bash
# This script is used to run the experiments on  different SNRs




for i in 0 1 2 3 4 5 6 
do
    # CUDA_VISIBLE_DEVICES=0 python3 main_data_generation.py --config 'snr' --snrIndex $i
    # CUDA_VISIBLE_DEVICES=0 python3 main_training.py --config 'snr' --snrIndex $i
    CUDA_VISIBLE_DEVICES=0 python3 main_evaluate_results.py --config 'snr' --snrIndex $i
done


