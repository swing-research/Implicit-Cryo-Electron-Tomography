"""
File to run the simulation with SHREC2021 model_0 volume.

It creates the dataset (projections, FBP), train ICE-TIDE and compare the results with AreTomo and Etomo.
"""

import train
import argparse
import data_generation
import configs.config_shrec_dataset as config_file

def main():   
    parser = argparse.ArgumentParser(description='Run experiement for the SHREC dataset.')
    parser.add_argument('--no_gen_data', action='store_false', help='Generate the data, default is True.')
    args = parser.parse_args()

    # Get config file
    config = config_file.get_config()

    # Make the data
    if args.no_gen_data:
        data_generation.data_generation(config)

    # Train ICE-TIDE
    train.train(config)

    # Compare the results and save the figures


if __name__ == '__main__':
    main()