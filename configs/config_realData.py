import ml_collections
import numpy as np
import torch

def get_default_realData():
    config = ml_collections.ConfigDict()
    # simulation
    # config = simulation = ml_collections.ConfigDict()

    config.volume_name = 'model_hiv'
    config.volume_file = 'b3tilt51.mrc'

    # Parameters for the data generation
    # size of the volume to use to generate the tilt-series
    config.n1 = 1024
    config.n2 = 1024
    config.n3 = 512 # size of the effective volume
    # size of the patch to crop in the raw volume
    config.n1_patch = 512
    config.n2_patch = 512
    config.n3_patch = 180 # size of the effective volume
    # nZ = 512 # size of the extended volume
    config.Nangles = 41
    config.sigma_PSF = 0
    config.number_sub_projections = 1
    config.downsample = True
    config.invert_projections = False
    config.angle_file = None # if None we use the view_angle and the view_angle_min and max
    config.view_angle_min = -60
    config.view_angle_max = 60
    config.transpose = True # if the axis of the projections are horizontal


    # # Parameters for the data generation
    config.path_save_data = "./results/"+str(config.volume_name)+"_size_"+str(config.n1)+"_"+str(config.n2)+"_no_PSF/"
    config.path_save = "./results/"+str(config.volume_name)+"_size_"+str(config.n1)+"_"+str(config.n2)+"_no_PSF/"

    config.seed = 42
    config.device_num = 0
    config.torch_type = torch.float

    config.isbare_bones = False

  # training
    # config.training = training = ml_collections.ConfigDict()


    # TODO: make this parameter inside a config file
    # Estimate Volume from the deformed projections
    config.train_volume = True
    config.train_local_def = False
    config.train_global_def = True
    config.local_model = 'interp' #  'implicit' or 'interp'
    config.initialize_local_def = False
    config.initialize_volume = False
    config.volume_model = "multi-resolution" # multi-resolution, Fourier-features, grid, MLP


    # When to start or stop optimizing over a variable
    config.schedule_local = []
    config.schedule_global = []
    config.schedule_volume = []

    config.batch_size = 2 # number of viewing direction per iteration
    config.nRays =  1000 # number of sampling rays per viewing direction
    # ray_length = 512 # number of points along one ray
    # TODO: try to change that
    config.z_max = 2*config.n3/max(config.n1,config.n2)/np.cos((90-np.max([config.view_angle_min,config.view_angle_max]))*np.pi/180)
    config.ray_length = 1500 #int(np.floor(n1*z_max))
    # TODO: try to chnage that with z_max
    config.rays_scaling = [0.5,0.5,0.5] # scaling of the coordinatesalong each axis. To make sure that the input of implicit net stay in their range

    ## Parameters
    config.epochs = 1000
    config.Ntest = 25 # number of epoch before display
    config.NsaveNet = 100 # number of epoch before saving again the nets
    config.save_volume = False
    config.lr_volume = 1e-3
    config.lr_local_def = 1e-4
    config.lr_shift = 1e-3
    config.lr_rot = 1e-3

    config.lamb_volume = 1e-5 # regul parameters on volume regularization
    config.lamb_local_ampl = 1e0 # regul on amplitude of local def.
    config.lamb_rot = 1e-4 # regul parameters on inplane rotations
    config.lamb_shifts = 1e-4 # regul parameters on shifts
    config.wd = 5e-6 # weights decay
    config.scheduler_step_size = 500
    config.scheduler_gamma = 0.1
    config.delay_deformations = 25 # Delay before learning deformations

    # Params of implicit deformation
    config.deformationScale = 0.1
    config.inputRange = 1
    config.loss_data = torch.nn.L1Loss()

    # if implicit model
    config.local_deformation = ml_collections.ConfigDict()
    if config.local_model == 'implicit':
        config.local_deformation.input_size = 2
        config.local_deformation.output_size = 2
        config.local_deformation.num_layers = 3
        config.local_deformation.hidden_size = 32
        config.local_deformation.L = 10
    elif config.local_model == 'interp':
        config.local_deformation.N_ctrl_pts_net = 20
    # params for the network
    # params of implicit volume
    config.input_size_volume = 3
    config.output_size_volume = 1
    config.num_layers_volume = 3
    config.hidden_size_volume = 64
    config.L_volume = 3

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
