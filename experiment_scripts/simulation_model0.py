"""
File to run the simulation with SHREC2021 model_0 volume.

It creates the dataset (projections, FBP), train ICE-TIDE and compare the results with AreTomo and Etomo.
"""

import train
import argparse
from compare_results import compare_results
import data_generation
import configs.config_shrec_dataset as config_file
import subprocess
import time


def main():   
    parser = argparse.ArgumentParser(description='Run experiement for the SHREC dataset.')
    parser.add_argument('--no_gen_data', action='store_false', help='Generate the data, default is True.')
    parser.add_argument('--no_train', action='store_false', help='Train the model, default is True.')
    parser.add_argument('--no_comparison', action='store_false', help='Compare the different methods, default is True.')
    args = parser.parse_args()

    # Get config file
    config = config_file.get_config()

    # Make the data
    if args.no_gen_data:
        data_generation.data_generation(config)

    # Train ICE-TIDE
    if args.no_train:
        train.train(config)

    # AreTomo
    if config.path_aretomo is not None:
        for npatch in config.nPatch:
            # TODO: keep track of GPU memory?
            t0 = time.time()
            try:
                delimiter = '|'
                combined_input = f"{config.path_aretomo}{delimiter}{config.path_save}{delimiter}{config.n3}{delimiter}{config.n3}{delimiter}{npatch}"
                subprocess.run(['bash', 'aretomo.sh'], input=combined_input.encode(), check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")
            t = time.time()-t0
            #TODO: save time and memory used in os.path.join(config.path_save,'AreTomo',f'time_memory_{npatch}by{npatch}.txt')

    # Etomo

    # Compare the results and save the figures
    if args.no_comparison:
        compare_results(config)


if __name__ == '__main__':
    main()