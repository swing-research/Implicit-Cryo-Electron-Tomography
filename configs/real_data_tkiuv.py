import torch
import numpy as np
import ml_collections


def get_config():
    '''
    This config file uses the default parameters but saves the volume at NsaveNet epochs
    So that we can use it  to test how the volume is evolving and possible reduce the number of epochs
    '''
    #######################
    ## Device parameters ##
    #######################
    config = ml_collections.ConfigDict()
    config.seed = 42
    config.device_num = 2
    config.torch_type = torch.float
    config.track_memory = False

    ########################################
    ## Parameters for the data generation ##
    ########################################
    # Size of volume, if not none will be resize to that
    config.n1 = 2048
    config.n2 = 2048
    config.n3 = 1024
    config.n3 = 1024
    # Size of the patch to crop in the raw volume
    config.n1_patch = 1024
    config.n2_patch = 1024
    config.n3_patch = 512
    # Size of the patch to eval in the raw volume
    config.n1_eval = 1024
    config.n2_eval = 1024
    config.n3_eval = 512
    # Fixed angle that is approximately known
    config.fixed_angle = 5
    # Sampling operator
    config.view_angle_min = -60
    config.view_angle_max = 60
    config.number_sub_projections = 1
    
    # # Parameters for the data generation
    config.volume_name = 'tomo2_L1G1-dose_filt'
    config.angle_name = 'tomo2_L1G1-dose_filt.tlt'
    config.path_load = "./datasets/tkiuv/"
    config.path_save_data = "./results/tkiuv_"+str(config.volume_name)+"/"
    config.path_save = "./results/tkiuv_"+str(config.volume_name)+"/"
    config.name_best_volume = "emd_15056_additional_1.map"
    config.projections_raw = True # True to not resize the projections

    config.avg_XYZ = 40 # average on the z direction for better visualization, number of frame to average over

    #############################
    ## Parameters for training ##
    #############################
    # Estimate Volume from the deformed projections
    config.train_volume = True
    config.train_local_def = True
    config.train_global_def = True
    config.volume_model = "multi-resolution" # multi-resolution, Fourier-features, grid, MLP
    config.local_model = 'interp' #  'implicit' or 'interp'

    # Training schedule
    config.epochs = 5000
    config.Ntest = 500 # number of epoch before display
    config.save_volume = True # saving the volume or not during training
    config.scheduler_step_size = 300
    config.scheduler_gamma = 0.75 #0.75

    # Sampling strategy
    config.batch_size = 5 # number of viewing direction per iteration
    config.nRays = 800
    config.ray_length = 1500 #int(np.floor(n1*z_max))
    config.sampling_domain_lx = config.sampling_domain_ly = 1 # dimension of the sampling domain
    config.size_z_vol = 0.5 # size of the volume in the z direction, knowing that [-sampling_domain_lx,sampling_domain_lx] is the sampling domain
    config.std_noise_z = 1 # std of the noise perturbation to apply on the z direction of the rays. std_noise=1 means there is a perturbation of at most one pixel.
    
    # When to start or stop optimizing over a variable
    config.schedule_volume = []
    config.schedule_global = []
    config.schedule_local = []
    config.delay_deformations = 0 # Delay before learning deformations

    # Training learning rates for Adam optimizer
    config.loss_data = torch.nn.L1Loss()
    config.lr_volume = 1e-3
    config.lr_shift = 1e-3
    config.lr_rot = 0 
    config.lr_local_def = 1e-4

    # Training regularization
    config.lamb_volume = 0 # regul parameters on volume regularization
    config.lamb_rot = 0 # regul parameters on inplane rotations
    config.lamb_shifts = 1e-6 # regul parameters on shifts
    config.lamb_local_ampl = 5*1e-5 # regul on amplitude of local def.
    config.lamb_local_mean = 1e-6 # regul on mean of local def.
    config.wd = 1e-6 # weights decay

    # Params for implicit deformation
    config.deformationScale = 1

    # params of implicit volume
    config.input_size_volume = 3 # always 3 for 3d tomography
    config.output_size_volume = 1 # always 1 for 3d tomography
    config.num_layers_volume = 4
    config.hidden_size_volume = 64
    config.L_volume = 3
    # params for the multi-resolution grids encoding
    config.encoding = ml_collections.ConfigDict()
    config.encoding.otype = 'Grid'
    config.encoding.type = 'Hash'
    config.encoding.n_levels = 14
    config.encoding.n_features_per_level = 8
    config.encoding.log2_hashmap_size = 22
    config.encoding.base_resolution = 16
    config.encoding.per_level_scale = 2
    config.encoding.interpolation = 'Linear'
    # params specific to Tiny cuda network
    config.network = ml_collections.ConfigDict()
    config.network.otype = 'FullyFusedMLP'
    config.network.activation = 'ReLU'
    config.network.output_activation = 'None'

    # parameters of implicit deformations
    config.local_deformation = ml_collections.ConfigDict()
    if config.local_model == 'implicit':
        config.local_deformation.input_size = 2 # fixed
        config.local_deformation.output_size = 2 # fixed
        config.local_deformation.num_layers = 3
        config.local_deformation.hidden_size = 32
        config.local_deformation.L = 10
    elif config.local_model == 'interp':
        config.local_deformation.N_ctrl_pts_net = 10
        config.maskBoundary = 0

    #######################
    ## AreTomo ##
    #######################
    config.path_aretomo = "/scicore/home/dokman0000/debarn0000/Softwares/AreTomo_1.3.4_Cuda101_Feb22_2023" #None 
    config.nPatch = [0,4]



    return config
