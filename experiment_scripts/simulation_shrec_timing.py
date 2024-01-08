"""
File to run the simulation with SHREC2021 model_0 volume.

It creates the dataset (projections, FBP), train ICE-TIDE and compare the results with AreTomo and Etomo.
"""

import argparse
import configs.shrec_timing as config_file
import subprocess
import time


def main():   
    parser = argparse.ArgumentParser(description='Run experiement for the SHREC dataset.')
    parser.add_argument('--no_gen_data', action='store_false', help='Generate the data, default is True.')
    parser.add_argument('--no_train', action='store_false', help='Train the model, default is True.')
    args = parser.parse_args()

    # Get config file
    config = config_file.get_config()

    # Make the data
    if args.no_gen_data:
        import data_generation
        data_generation.data_generation(config)

    # Train ICE-TIDE
    if args.no_train:
        import train
        train.train(config)



if __name__ == '__main__':
    main()