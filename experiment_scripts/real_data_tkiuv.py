"""
File to run the simulation with SHREC2021 model_0 volume.

It creates the dataset (projections, FBP), train ICE-TIDE and compare the results with AreTomo and Etomo.
"""

import argparse
import configs.real_data_tkiuv as config_file
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

    ############################
    ## generate data
    ############################
    if args.no_gen_data:
        import data_generation
        data_generation.data_generation_real_data(config)


    # Train ICE-TIDE
    if args.no_train:
        import train
        # Initial training
        config.multiresolution = True
        config.delay_deformations = 25 # Delay before learning deformations
        config.epochs = 1200
        config.load_existing_net = False

        config.encoding.otype = 'Grid'
        config.encoding.type = 'Hash'
        config.encoding.n_levels = 9
        config.encoding.base_resolution = 8

        train.train_without_ground_truth(config)

        # # refined training
        # config.multiresolution = False
        # config.delay_deformations = 0 # Delay before learning deformations
        # config.schedule_global = []
        # config.schedule_local = []
        # config.epochs = 4000
        # config.load_existing_net = True
        # config.delay_deformations = 0 # Delay before learning deformations

        # config.encoding.otype = 'Grid'
        # config.encoding.type = 'Hash'
        # config.encoding.log2_hashmap_size = 22
        # config.encoding.n_levels = 16
        # config.encoding.base_resolution = 16
        
        # train.train_without_ground_truth(config)


    # AreTomo
    if args.no_aretomo:
        if config.path_aretomo is not None:
            for npatch in config.nPatch:
                # TODO: keep track of GPU memory?
                t0 = time.time()
                try:
                    subprocess.run(['bash', 'aretomo.sh',config.path_aretomo,config.path_save,str(config.n3),str(config.n3),str(npatch)])
                except subprocess.CalledProcessError as e:
                    print(f"Error: {e}")
                t = time.time()-t0
                #TODO: save time and memory used in os.path.join(config.path_save,'AreTomo',f'time_memory_{npatch}by{npatch}.txt')

    # Etomo
    # TODO

    # Compare the results and save the figures
    if args.no_comparison:
        import compare_results
        compare_results.compare_results_real(config)


if __name__ == '__main__':
    main()