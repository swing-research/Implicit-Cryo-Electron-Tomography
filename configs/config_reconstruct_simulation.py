import ml_collections
import numpy as np
import torch

def get_default_configs():
    config = ml_collections.ConfigDict()
    # simulation
    # config = simulation = ml_collections.ConfigDict()

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
    # nZ = 512 # size of the extended volume
    config.Nangles = 61
    config.view_angle_min = -60
    config.view_angle_max = 60
    config.SNR_value = 10
    config.sigma_PSF = 3.
    config.number_sub_projections = 1

    config.scale_min = 1.0
    config.scale_max = 1.0
    config.shift_min = -0.04
    config.shift_max = 0.04
    config.shear_min = -0.0
    config.shear_max = 0.0
    config.angle_min = -4/180*np.pi
    config.angle_max = 4/180*np.pi
    config.sigma_local_def = 4
    config.N_ctrl_pts_local_def = (12,12)


    # # Parameters for the data generation
    config.path_load = "./results/"+str(config.volume_name)+"_SNR_"+str(config.SNR_value)+"_size_"+str(config.n1)+"_no_PSF/"
    config.path_save = "./results/model0_SNR_"+str(config.SNR_value)+"_size_"+str(config.n1)+"_no_PSF/"

    config.seed = 42
    config.device_num = 0


  # training
    # config.training = training = ml_collections.ConfigDict()


    # TODO: make this parameter inside a config file
    # Estimate Volume from the deformed projections
    config.train_volume = True
    config.train_local_def = True
    config.train_global_def = True
    # train_all = True # train or load model
    # learn_volume = True # learn the volume
    # learn_global = True # learn global dn3/max(n1,n2)/np.cos((90-angle_bound)*np.pi/180)eformation
    # learn_local = True # learn local deformation
    config.local_model = 'interp' #  'implicit' or 'interp'
    config.initialize_local_def = False
    config.initialize_volume = False
    # use_deformation_estimation = True # estimate deformation, useful to see what happens when we don't
    config.volume_model = "multi-resolution" # multi-resolution, Fourier-features, grid, MLP
    # model_type = 2 #0 for Fourier feature, 1 for MLP

    # When to start or stop optimizing over a variable
    config.schedule_local = []
    config.schedule_global = []
    config.schedule_volume = []

    config.batch_size = 10 # number of viewing direction per iteration
    config.nRays =  1500 # number of sampling rays per viewing direction
    # ray_length = 512 # number of points along one ray
    # TODO: try to change that
    config.z_max = 2*config.n3/max(config.n1,config.n2)/np.cos((90-np.max([config.angle_min,config.angle_max]))*np.pi/180)
    config.ray_length = 500#int(np.floor(n1*z_max))
    # TODO: try to chnage that with z_max
    config.rays_scaling = [1.,1.,1.] # scaling of the coordinatesalong each axis. To make sure that the input of implicit net stay in their range

    ## Parameters
    config.epochs = 400
    config.Ntest = 25 # number of epoch before display
    config.NsaveNet = 100 # number of epoch before saving again the nets
    # iter_local = 1000000 # include training of local deformations after few epochs
    # frac = 1
    config.lr_volume = 1e-2
    # lr_global_def =1e-4
    config.lr_local_def = 1e-4
    config.lr_shift = 1e-3
    config.lr_rot = 1e-3

    config.lamb_volume = 0*1e-5 # regul parameters on volume regularization
    config.lamb_volume_out = 0*1e-0 # regul parameters on volume regularization to be 0 outside domain
    config.lamb_local = 0*1e-3 # regul parameters on local deformation
    # lamb_local_smooth = 0*1e-8 # regul parameters on local deformation to be smooth
    config.lamb_local_ampl = 1e2 # regul on amplitude of local def.
    config.lamb_rot = 1e-6 # regul parameters on inplane rotations
    config.lamb_shifts = 1e-6 # regul parameters on shifts
    config.wd = 5e-6 # weights decay
    # Nangles_ = Nangles
    config.scheduler_step_size = 200
    config.scheduler_gamma = 0.6
    config.delay_deformations = 25 # Delay before learning deformations

    # Params of implicit deformation
    config.deformationScale = 1
    config.inputRange = 1
    config.Npts_rd = 500 # number of random points to compute regul

    # if implicit model
    config.input_size = 2
    config.output_size = 2
    config.num_layers = 3
    config.hidden_size = 32
    config.L = 10
    # if interpolation model
    config.N_ctrl_pts_net = 20

    # params of implicit volume
    config.input_size_volume = 3
    config.output_size_volume = 1
    config.num_layers_volume = 4
    config.hidden_size_volume = 128
    config.L_volume = 3

    config.loss_data = torch.nn.L1Loss()



    return config