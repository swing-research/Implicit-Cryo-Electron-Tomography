"""
File to run the simulation with SHREC2021 model_0 volume.

It creates the dataset (projections, FBP), train ICE-TIDE and compare the results with AreTomo and Etomo.
"""
import os
import yaml
import glob
import torch
import argparse
import ml_collections

def main():   
    parser = argparse.ArgumentParser(description='Run experiement for the SHREC dataset.')
    parser.add_argument('--path_config', default='real_data_fast.yaml', action='store', help="Don't evaluate the results, default is False.")
    parser.add_argument('--no_evaluation', default=True, action='store_false', help="Don't evaluate the results, default is False.")
    parser.add_argument('--no_train', default=True, action='store_false', help="Don't train the model, default is False.")
    args = parser.parse_args()

    # Get config file
    with open(os.path.join('configs',args.path_config)) as cf_file:
        config_yaml = yaml.safe_load(cf_file.read())
    config = ml_collections.ConfigDict(config_yaml, type_safe=False)

    if config.torch_type == "float":
        config.torch_type = torch.float
    else:
        print("Only float type is implemented.")
        config.torch_type = torch.float
    if config.loss_data == 'l1':
        config.loss_data = torch.nn.L1Loss()
    elif config.loss_data == 'l2':
        config.loss_data = torch.nn.MSELoss()
    else:
        print("Only l1 and l2 loss is implemented. Default is l1.")
        config.loss_data = torch.nn.L1Loss()

    ######################################################################################################
    ## Setting the environment
    ######################################################################################################
    if not(hasattr(config, 'debug')):
        config.debug = False
    if not(hasattr(config, 'Nimplicit_volume')):
        config.Nimplicit_volume = config.Ntest
    if not(hasattr(config, 'Nalign')):
        config.Nalign = config.Ntest
    if not(hasattr(config, 'lr_rot_fixed')):
        config.lr_rot_fixed = config.lr_rot
    if not(hasattr(config, 'multiresolution')):
        config.multiresolution = False

    if args.no_train:
        import train_real_fast as train

    if args.no_evaluation:
        import generate_results_real

    if type(config.path_volume) is list:
        all_volumes = config.path_volume
        all_angles = config.path_angle
        for i, path_volume_ in enumerate(all_volumes):
            config.path_volume = path_volume_
            print("Running on ", path_volume_)
            if len(all_angles) <= i:
                print("No .tlt file found. Angles will be set uniformly between ", config.view_angle_min, " and ", config.view_angle_max)
                config.path_angle = ''
            else:
                config.path_angle = all_angles[i]
                print("With angle file ", config.path_angle)
            if args.no_train:
                train.train_without_ground_truth(config)
            if args.no_evaluation:
                import generate_results_real
                generate_results_real.generate_results(config)

    else:
        base_directory = config.path_volume[:config.path_volume.rfind('/')]
        name_file = os.path.basename(config.path_volume).split('/')[-1]
        # name_file = name_file.split('.')[0]
        # If path_volume is a folder
        if os.path.isdir(os.path.join(base_directory,name_file)):
            print(name_file, ' interpreted as a folder')
            print("Ice-Tide will be run on each of the mrc files in the folder")
            all_files = os.path.join(name_file, '*.mrc')
            all_angles = os.path.join(name_file, '*.tlt')
        # If path_volume contains '*'
        elif name_file.rfind('*')!=-1:
            print(name_file, ' interpreted as list of files')
            print("Ice-Tide will be run on each of the mrc files in ", glob.glob(name_file))
            all_files = name_file
            all_angles = config.path_angle
        else:
            all_files = name_file
            all_angles = config.path_angle


        all_angles = glob.glob(all_angles)
        for i, ts_file in enumerate(glob.glob(os.path.join(base_directory,all_files))):
            config.path_volume = ts_file
            print("Running on ", ts_file)
            if len(all_angles) <= i:
                print("No .tlt file found. Angles will be set uniformly between ", config.view_angle_min, " and ", config.view_angle_max)
                config.path_angle = ''
            else:
                config.path_angle = all_angles[i]
                print("With angle file ", config.path_angle)
            if args.no_train:
                train.train_without_ground_truth(config)
            if args.no_evaluation:
                import generate_results_real
                generate_results_real.generate_results(config)

        if len(glob.glob(os.path.join(base_directory,all_files))) == 0:
            print("No .tlt files found in ", all_files)

if __name__ == '__main__':
    main()
