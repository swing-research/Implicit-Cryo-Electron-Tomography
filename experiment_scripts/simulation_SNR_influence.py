"""
File to run the simulation with SHREC2021 model_0 volume.

It creates the dataset (projections, FBP), train ICE-TIDE and compare the results with AreTomo and Etomo.
"""

import numpy as np
import argparse
import configs.shrec_all_SNR as config_file
import subprocess
import time


def main():   
    parser = argparse.ArgumentParser(description='Run experiement for the SHREC dataset.')
    parser.add_argument('--no_gen_data', action='store_false', help='Generate the data, default is True.')
    parser.add_argument('--no_train', action='store_false', help='Train the model, default is True.')
    parser.add_argument('--no_aretomo', action='store_false', help='Run AreTomo, default is True.')
    parser.add_argument('--no_comparison', action='store_false', help='Compare the different methods, default is True.')
    args = parser.parse_args()

    # Get config file
    config = config_file.get_config()

    v_SNR = np.arange(-20,31,10)
    for v_SNR in v_SNR_list:
        # Noise
        config.SNR_value = v_SNR
        # # Parameters for the data generation
        config.path_save_data = "./results/SNR_exp_"+str(config.volume_name)+"_SNR_"+str(config.SNR_value)+"_size_"+str(config.n1)+"_Nangles_"+str(config.Nangles)+"/"
        config.path_save = "./results/SNR_exp_"+str(config.volume_name)+"_SNR_"+str(config.SNR_value)+"_size_"+str(config.n1)+"_Nangles_"+str(config.Nangles)+"/"

        # Make the data
        if args.no_gen_data:
            import data_generation
            data_generation.data_generation(config)

        # Train ICE-TIDE
        if args.no_train:
            import train
            train.train(config)

        # AreTomo
        if args.no_aretomo:
            if config.path_aretomo is not None:
                for npatch in config.nPatch:
                    t0 = time.time()
                    try:
                        subprocess.run(['bash', 'aretomo.sh',config.path_aretomo,config.path_save,str(config.n3),str(config.n3),str(npatch)])
                    except subprocess.CalledProcessError as e:
                        print(f"Error: {e}")
                    t = time.time()-t0


        # Compare the results and save the figures
        if args.no_comparison:
            from compare_results import compare_results
            compare_results(config)

    # TODO load all FSCs and merge that into the bar plot



if __name__ == '__main__':
    main()