import numpy as np
import torch
import ml_collections


def get_config():
    '''
    This config file uses the default parameters but saves the volume at NsaveNet epochs
    So that we can use it  to test how the volume is evolving and possible reduce the number of epochs
    '''
    config = ml_collections.ConfigDict()
    config.volume_name = 'model_0'

    # Parameters for the data generation
    # size of the volume to use to generate the tilt-series
    config.n1 = 512
    config.n2 = 512
    config.n3 = 180 # size of the effective volume
    # size of the patch to crop in the raw volume
    config.n1_patch = 512
    config.n2_patch = 512
    config.n3_patch = 180 # size of the effective volume
    config.Nangles = 61
    config.view_angle_min = -60
    config.view_angle_max = 60
    config.SNR_value = 10
    config.sigma_PSF = 0
    config.number_sub_projections = 1

    deformation_scale = 0.5
    config.scale_min = 1.0
    config.scale_max = 1.0
    config.shift_min = -0.03*deformation_scale # percentage of the field of view
    config.shift_max = 0.03*deformation_scale  # percentage of the field of view
    config.shear_min = -0.0
    config.shear_max = 0.0
    config.angle_min = -0.1/180*np.pi*deformation_scale # in degrees
    config.angle_max = 0.1/180*np.pi*deformation_scale # in degrees
    config.slowAngle = False 
    config.sigma_local_def = 2*deformation_scale
    config.N_ctrl_pts_local_def = (12,12)
    config.save_volume = True
    # # Parameters for the data generation
    config.path_save_data = "./results/"+str(config.volume_name)+"_SNR_"+str(config.SNR_value)+"_size_"+str(config.n1)+"_no_PSF_vol_save/"
    config.path_save = "./results/"+str(config.volume_name)+"_SNR_"+str(config.SNR_value)+"_size_"+str(config.n1)+"_no_PSF_vol_save/"

    config.seed = 42
    config.device_num = 0
    config.torch_type = torch.float
    config.track_memory = False

    # training
    # Estimate Volume from the deformed projections
    config.train_volume = True
    config.train_local_def = True
    config.train_global_def = True
    config.local_model = 'interp' #  'implicit' or 'interp'
    config.initialize_local_def = False
    config.initialize_volume = False
    config.volume_model = "MLP" # multi-resolution, Fourier-features, grid, MLP

    # When to start or stop optimizing over a variable
    config.schedule_local = []
    config.schedule_global = []
    config.schedule_volume = []

    config.batch_size = 4 # number of viewing direction per iteration
    config.nRays =  1500 # number of sampling rays per viewing direction
    config.z_max = 2*config.n3/max(config.n1,config.n2)/np.cos((90-np.max([config.view_angle_min,config.view_angle_max]))*np.pi/180)
    config.ray_length = 500 #int(np.floor(n1*z_max))
    config.rays_scaling = [1.,1.,1.] # scaling of the coordinatesalong each axis. To make sure that the input of implicit net stay in their range

    config.epochs = 1000
    config.Ntest = 25 # number of epoch before display
    config.NsaveNet = 100 # number of epoch before saving again the nets
    
    config.lr_volume = 1e-2
    config.lr_local_def = 1e-4
    config.lr_shift = 1e-3
    config.lr_rot = 1e-3

    config.lamb_volume = 1e-5 # regul parameters on volume regularization
    config.lamb_local_ampl = 1e0 # regul on amplitude of local def.
    config.lamb_rot = 1e-4 # regul parameters on inplane rotations
    config.lamb_shifts = 1e-4 # regul parameters on shifts
    config.wd = 5e-6 # weights decay
    config.scheduler_step_size = 200
    config.scheduler_gamma = 0.6
    config.delay_deformations = 25 # Delay before learning deformations

    # Params of implicit deformation
    config.deformationScale = 0.1
    config.inputRange = 1

    # paramsof volume implicit model
    config.local_deformation = ml_collections.ConfigDict()
    if config.local_model == 'implicit':
        config.local_deformation.input_size = 2
        config.local_deformation.output_size = 2
        config.local_deformation.num_layers = 3
        config.local_deformation.hidden_size = 32
        config.local_deformation.L = 10
    elif config.local_model == 'interp':
        config.local_deformation.N_ctrl_pts_net = 20

    # params of implicit volume
    config.input_size_volume = 3
    config.output_size_volume = 1
    config.num_layers_volume = 3
    config.hidden_size_volume = 64
    config.L_volume = 3

    config.loss_data = torch.nn.L1Loss()

    # params for the multi-resolution grids
    config.encoding = ml_collections.ConfigDict()
    config.encoding.otype = 'Grid'
    config.encoding.type = 'Hash'
    config.encoding.n_levels = 8#
    config.encoding.n_features_per_level = 4
    config.encoding.log2_hashmap_size = 22
    config.encoding.base_resolution = 8
    config.encoding.per_level_scale = 2#1.3
    config.encoding.interpolation = 'Smoothstep'
    config.network = ml_collections.ConfigDict()

    # params specific to Tiny cuda network
    config.network.otype = 'FullyFusedMLP'
    config.network.activation = 'ReLU'
    config.network.output_activation = 'None'

    return config
