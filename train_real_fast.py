"""
Module to train the reconstruction network on the simulated data.
"""

import os
import time
import torch
import imageio
import mrcfile
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from skimage.transform import resize
from torch.utils.data import DataLoader, TensorDataset
from utils import utils_deformation, utils_display
from utils.utils_sampling import get_sampling_geometry, apply_deformations_to_locations, generate_rays_batch, sample_projections

from utils.utils_deformation import cropper
def aligned_projections(projections, rot_est, shift_est, implicit_deformation_est, fixed_rot, deformationScale=1, torch_type=torch.float, device=torch.device('cpu')):
    with torch.no_grad():
        Nangles, n1, n2 = projections.shape
        projections_undeformed = torch.zeros_like(projections)
        xx1 = torch.linspace(-1, 1, n1, dtype=torch_type, device=device)
        xx2 = torch.linspace(-1, 1, n2, dtype=torch_type, device=device)
        XX_t, YY_t = torch.meshgrid(xx1, xx2, indexing='ij')
        XX_t = torch.unsqueeze(XX_t, dim=2)
        YY_t = torch.unsqueeze(YY_t, dim=2)
        for i in range(Nangles):
            coordinates = torch.cat([XX_t, YY_t], 2).reshape(-1, 2)
            rot_deform = rot_est[i](dim=2).type(torch_type).to(device)
            fixed_rot_deform = fixed_rot(dim=2).type(torch_type).to(device)

            # thetas = torch.tensor(-rot_est[i].thetas.item()).to(device)
            coordinates = coordinates - deformationScale * (implicit_deformation_est[i](coordinates).type(torch_type).to(device))
            coordinates = coordinates - (shift_est[i].shifts_arr.type(torch_type).to(device))
            coordinates = torch.transpose(torch.matmul(torch.transpose(rot_deform,1,0), torch.transpose(coordinates, 0, 1)), 0,
                                          1)  ## do rotation
            coordinates = torch.transpose(torch.matmul(torch.transpose(fixed_rot_deform,1,0), torch.transpose(coordinates, 0, 1)), 0,
                                          1)  ## do rotation
            x = projections[i].clone().view(1, 1, n1, n2)
            x = x.expand(n1 * n2, -1, -1, -1)
            out = cropper(x, coordinates, output_size=1).reshape(n1, n2)
            projections_undeformed[i] = out
    return projections_undeformed

def globalDeformationValues(shift,rot):
    shiftValueList = []
    rotValueList = []
    for si, ri in  zip(shift,rot):
        shiftValue = si().clone().detach().cpu().numpy()
        rotValue = ri.thetas.clone().detach().cpu().numpy()
        shiftValueList.append(shiftValue)
        rotValueList.append(rotValue)
    shiftValueList = np.array(shiftValueList)
    rotValueList = np.array(rotValueList)
    return shiftValueList, rotValueList


from PIL import Image
import torchvision

import cProfile, pstats, io
from pstats import SortKey

def train_without_ground_truth(config):
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

    if config.debug:
        pr = cProfile.Profile()
        pr.enable()

    print("Runing training procedure.")
    # Choosing the seed and the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count()>1:
        torch.cuda.set_device(config.device_num)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    print("Device: {}".format(device))

    # prepare the folders    
    if not os.path.exists(config.path_save+"training/"):
        os.makedirs(config.path_save+"training/")
    if not os.path.exists(config.path_save+"training/volume/"):
        os.makedirs(config.path_save+"training/volume/")
    if not os.path.exists(config.path_save+"training/deformations/"):
        os.makedirs(config.path_save+"training/deformations/")
    if not os.path.exists(config.path_save+"training/deformations_x10/"):
        os.makedirs(config.path_save+"training/deformations_x10/")
    if not os.path.exists(config.path_save+"training/projections/"):
        os.makedirs(config.path_save+"training/projections/")
    if not os.path.exists(config.path_save + "training/projections/raw_aligned/"):
        os.makedirs(config.path_save + "training/projections/raw_aligned/")

    ######################################################################################################
    ## Loading the data
    ######################################################################################################
    # Load the data provided by the user
    print("Loading the tilt-series")
    t0 = time.time()
    projections_noisy = np.float32(mrcfile.open(config.path_volume,permissive=True).data)
    projections_noisy = projections_noisy/np.abs(projections_noisy).max()
    if config.projections_rotate:
        projections_noisy = np.rot90(np.flip(projections_noisy,axis=1), k=3, axes=((1, 2)))
    config.Nangles = projections_noisy.shape[0]
    projections_noisy = torch.Tensor(projections_noisy).type(config.torch_type).to(device)
    projections_noisy = projections_noisy/torch.abs(projections_noisy).max() # make sure that values to predict are between -1 and 1
    if config.debug:
        print("Elapsed time: {}".format(time.time() - t0))

    ######################################################################################################
    ## Define neural network architectures for the volume
    ######################################################################################################
    print("Defining volume neural network")
    t0 = time.time()
    # Define the neural networks
    if(config.volume_model=="Fourier-features"):
        from models.fourier_net import FourierNet,FourierNet_Features
        impl_volume = FourierNet_Features(
            in_features=config.input_size_volume,
            sub_features=config.sub_features,
            out_features=config.output_size_volume, 
            hidden_features=config.hidden_size_volume,
            hidden_blocks=config.num_layers_volume,
            L = config.L_volume).to(device)

    if(config.volume_model=="MLP"):
        from models.fourier_net import MLP
        impl_volume = MLP(in_features= 3, 
                            hidden_features=config.hidden_size_volume, hidden_blocks= config.num_layers_volume, out_features=config.output_size_volume).to(device)

    if(config.volume_model=="multi-resolution"):
        import tinycudann as tcnn
        config_network = {"encoding": {
                'otype': config.encoding.otype,
                'type': config.encoding.type,
                'n_levels': config.encoding.n_levels,
                'n_features_per_level': config.encoding.n_features_per_level,
                'log2_hashmap_size': config.encoding.log2_hashmap_size,
                'base_resolution': config.encoding.base_resolution,
                'per_level_scale': config.encoding.per_level_scale,
                'interpolation': config.encoding.interpolation
            },
            "network": {
                "otype": config.network.otype,   
                "activation": config.network.activation,       
                "output_activation": config.network.output_activation,
                "n_neurons": config.hidden_size_volume,           
                "n_hidden_layers": config.num_layers_volume,       
            }       
            }
        impl_volume = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=1, encoding_config=config_network["encoding"],
                                                    network_config=config_network["network"]).to(device)

    num_param = sum(p.numel() for p in impl_volume.parameters() if p.requires_grad) 
    print(f"---> Number of trainable parameters in volume net: {num_param}")

    if config.debug:
        print("Elapsed time: {}".format(time.time() - t0))

    ######################################################################################################
    ## Define neural network architectures for local deformations
    ######################################################################################################
    print("Defining local deformation neural network")
    t0 = time.time()
    if config.local_model=='implicit':
        from models.fourier_net import FourierNet,FourierNet_Features
        # Define Implicit representation of local deformations
        implicit_deformation_list = []
        for k in range(config.Nangles):
            implicit_deformation = FourierNet(
                in_features=config.local_deformation.input_size,
                out_features=config.local_deformation.output_size,
                hidden_features=config.local_deformation.hidden_size,
                hidden_blocks=config.local_deformation.num_layers,
                L = config.local_deformation.L,
                scale = config.deformationScale).to(device)
            implicit_deformation_list.append(implicit_deformation)

        num_param = sum(p.numel() for p in implicit_deformation_list[0].parameters() if p.requires_grad)
        print('---> Number of trainable parameters in implicit net: {}'.format(num_param))

    if config.local_model=='tcnn':
        import tinycudann as tcnn
        config_network = {"encoding": {
                'otype': config.local_deformation.encoding.otype,
                'type': config.local_deformation.encoding.type,
                'n_levels': config.local_deformation.encoding.n_levels,
                'n_features_per_level': config.local_deformation.encoding.n_features_per_level,
                'log2_hashmap_size': config.local_deformation.encoding.log2_hashmap_size,
                'base_resolution': config.local_deformation.encoding.base_resolution,
                'per_level_scale': config.local_deformation.encoding.per_level_scale,
                'interpolation': config.local_deformation.encoding.interpolation
            },
            "network": {
                "otype": config.local_deformation.network.otype,   
                "activation": config.local_deformation.network.activation,       
                "output_activation": config.local_deformation.network.output_activation,
                "n_neurons": config.local_deformation.hidden_size,           
                "n_hidden_layers": config.local_deformation.num_layers,       
            }       
            }
        implicit_deformation_list = []
        for k in range(config.Nangles):
            implicit_deformation = tcnn.NetworkWithInputEncoding(n_input_dims=config.local_deformation.input_size, 
                                                                n_output_dims=config.local_deformation.output_size,
                                                                encoding_config=config_network["encoding"],
                                                                network_config=config_network["network"]).to(device)
            implicit_deformation_list.append(implicit_deformation)

        num_param = sum(p.numel() for p in implicit_deformation_list[0].parameters() if p.requires_grad)
        print('---> Number of trainable parameters in implicit net: {}'.format(num_param))

    if config.local_model=='interp':
        depl_ctr_pts_net = torch.zeros((2,config.local_deformation.N_ctrl_pts_net,config.local_deformation.N_ctrl_pts_net)).to(device).type(config.torch_type)
        implicit_deformation_list = []
        for k in range(config.Nangles):
            field = utils_deformation.deformation_field(depl_ctr_pts_net.clone(),maskBoundary=config.maskBoundary)
            implicit_deformation_list.append(field)
        num_param = sum(p.numel() for p in implicit_deformation_list[0].parameters() if p.requires_grad)
        print('---> Number of trainable parameters in implicit net: {}'.format(num_param))

    if config.debug:
        print("Elapsed time: {}".format(time.time() - t0))

    ######################################################################################################
    ## Define global deformations
    ######################################################################################################
    fixedAngle = torch.FloatTensor([config.fixed_angle* np.pi/180]).to(device)[0]
    fixedAngle.requires_grad = True

    shift_est = []
    rot_est = []
    for k in range(config.Nangles):
        shift_est.append(utils_deformation.shiftNet(1).to(device))
        rot_est.append(utils_deformation.rotNet(1).to(device))
    fixed_rot = utils_deformation.rotNet(1,x0=fixedAngle).to(device)

    # TODO: check that it works
    # Load model if required and if exists
    if config.continue_training:
        if os.path.isfile(os.path.join(config.path_save, 'training', 'model_trained.pt')):
            checkpoint = torch.load(os.path.join(config.path_save, 'training', 'model_trained.pt'), map_location=device)
            impl_volume.load_state_dict(checkpoint['implicit_volume'])
            shift_est = checkpoint['shift_est']
            rot_est = checkpoint['rot_est']
            implicit_deformation_list = checkpoint['local_deformation_network']

    ######################################################################################################
    ## Define optimizers
    ######################################################################################################
    loss_data = config.loss_data
    train_global_def = config.train_global_def
    train_local_def = config.train_local_def
    list_params_deformations_glob = []
    list_params_deformations_loc = []
    if(train_global_def or train_local_def):
        for k in range(config.Nangles):
            if train_global_def:
                list_params_deformations_glob.append({"params": shift_est[k].parameters(), "lr": config.lr_shift})
                if config.lr_rot!=0:
                    list_params_deformations_glob.append({"params": rot_est[k].parameters(), "lr": config.lr_rot})
            if train_local_def:
                list_params_deformations_loc.append({"params": implicit_deformation_list[k].parameters(), "lr": config.lr_local_def})
    gains = Variable(torch.rand(config.Nangles).to(device)/5+1, requires_grad=True)
    optimizer_volume = torch.optim.Adam(list(impl_volume.parameters())+[gains], lr=config.lr_volume, weight_decay=config.wd)
    if len(list_params_deformations_glob)!=0:
        list_params_deformations_glob.append({"params": fixed_rot.parameters(), "lr": config.lr_rot_fixed})
        optimizer_deformations_glob = torch.optim.Adam(list_params_deformations_glob, weight_decay=config.wd)
        scheduler_deformation_glob = torch.optim.lr_scheduler.StepLR(
            optimizer_deformations_glob, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
    if len(list_params_deformations_loc)!=0:
        optimizer_deformations_loc = torch.optim.Adam(list_params_deformations_loc, weight_decay=config.wd)
        scheduler_deformation_loc = torch.optim.lr_scheduler.StepLR(
            optimizer_deformations_loc, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
    scheduler_volume = torch.optim.lr_scheduler.StepLR(optimizer_volume, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)

    ######################################################################################################
    ## Format data for batch training
    ######################################################################################################
    print("Preparing data for training and display")
    t0 = time.time()
    index = torch.arange(0, config.Nangles, dtype=torch.long) # index for the dataloader
    # Define dataset
    angles = np.linspace(config.view_angle_min,config.view_angle_max,config.Nangles)
    angles_t = torch.tensor(angles).type(config.torch_type).to(device)
    dataset = TensorDataset(angles_t,projections_noisy.detach(),index)
    trainLoader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    weights_tilt = torch.cos(angles_t/180*np.pi).to(device)

    if config.save_volume:
        ## grid for display
        x_lin1 = np.linspace(-1,1,config.n1_patch)
        x_lin2 = np.linspace(-1,1,config.n2_patch)
        XX, YY = np.meshgrid(x_lin1,x_lin2,indexing='ij')
        grid2d = np.concatenate([XX.reshape(-1,1),YY.reshape(-1,1)],1)
        grid2d_t = torch.tensor(grid2d).type(config.torch_type)

    # Define geometry of sampling
    size_xy_vol, z_max_value = get_sampling_geometry(config.size_z_vol, config.view_angle_min, config.view_angle_max, config.sampling_domain_lx, config.sampling_domain_ly)
    size_max_vol = 1.2*np.max([size_xy_vol,config.size_z_vol]) # increase by some small factor to account for deformations

    if config.debug:
        print("Elapsed time: {}".format(time.time() - t0))

    if hasattr(config, 'multiresolution'):
        if config.multiresolution:
            with torch.no_grad():
                print("Computing multiresolution volume")
                res_factor = config.multires_params.startResolution
                Nangles, n1_origin, n2_origin = projections_noisy.shape
                n1_resize = int(n1_origin/(2**res_factor))
                n2_resize = int(n2_origin/(2**res_factor))

                # print(projections_noisy.min(),projections_noisy.max())

                # t0 = time.time()
                # projections_noisy_resized = resize(projections_noisy.detach().cpu().numpy(),(Nangles,n1_resize,n2_resize), preserve_range=True)
                # projections_noisy_resized_t = torch.tensor(projections_noisy_resized).type(config.torch_type).to(device)
                # print(projections_noisy_resized_t.min(),projections_noisy_resized_t.max())
                # print("Elapsed time with skimage: ", time.time()-t0)
                #
                #
                # t0 = time.time()
                # projections_noisy_resized =  np.array([ resize(projections_noisy[ll].detach().cpu().numpy(),(Nangles,n1_resize,n2_resize), preserve_range=True)  for ll in range(Nangles) ])
                # projections_noisy_resized_t = torch.tensor(projections_noisy_resized).type(config.torch_type).to(device)
                # print(projections_noisy_resized_t.min(),projections_noisy_resized_t.max())
                # print("Elapsed time with skimage for loop: ", time.time()-t0)


                # t0 = time.time()
                # pr = projections_noisy.detach().cpu().numpy()
                # m1 = pr.min()
                # s1 = pr.max() - pr.min()
                # pr = (pr-m1)/(s1)
                # pr = np.uint8(pr*255)
                # proj_pil = [ Image.fromarray(pr[ll]) for ll in range(Nangles) ]
                # # proj_pil = torchvision.transforms.functional.to_pil_image(projections_noisy.view(Nangles,1,n1_origin,n2_origin), mode=None)
                # projections_noisy_resized = np.array([proj_pil[ll].resize((n1_resize,n2_resize)) for ll in range(Nangles) ])
                # projections_noisy_resized = projections_noisy_resized/255*s1 + m1
                # projections_noisy_resized_t = torch.tensor(projections_noisy_resized).type(config.torch_type).to(device)
                # print(projections_noisy_resized_t.min(),projections_noisy_resized_t.max())
                # print("Elapsed time with PIL: ", time.time()-t0)
                #
                # t0 = time.time()
                # projections_noisy_resized_t = projections_noisy[:,::2**res_factor,::2**res_factor]
                # print(projections_noisy_resized_t.min(),projections_noisy_resized_t.max())
                # print("Elapsed time spatial bining: ", time.time()-t0)


                t0 = time.time()
                factor = 2**res_factor
                h_crop = n1_origin // factor // 2
                w_crop = n2_origin // factor // 2
                center_h = n1_origin // 2
                center_w = n2_origin // 2
                fft_cropped = torch.fft.fftshift(torch.fft.fft2(projections_noisy.detach().cpu(), norm='forward'))[:,
                          center_h - h_crop:center_h + h_crop,
                          center_w - w_crop:center_w + w_crop]
                projections_noisy_resized_t = (torch.fft.ifft2(torch.fft.ifftshift(fft_cropped), norm='forward')).real
                projections_noisy_resized_t = projections_noisy_resized_t.type(config.torch_type).to(device)
                # projections_noisy_resized_t = projections_noisy_resized_t / torch.abs(
                #     projections_noisy_resized_t).max()  # make sure that values to predict are between -1 and 1
                print(projections_noisy_resized_t.min(),projections_noisy_resized_t.max(), projections_noisy_resized_t.shape)
                print("Elapsed time Fourier cropping: ", time.time()-t0)

                # from utils.utils_FSC import Resample
                # projections_noisy_resized = np.array([
                #     Resample(projections_noisy[ll].detach().cpu().numpy(), newsize=(n1_resize,n2_resize), apix=1.0, newapix=None)  for ll in range(Nangles) ])
                # projections_noisy_resized_t = torch.tensor(projections_noisy_resized).type(config.torch_type).to(device)
                # print(projections_noisy_resized_t.min(),projections_noisy_resized_t.max(), projections_noisy_resized_t.shape)
                # print("Elapsed time Ricardo Fourier cropping: ", time.time()-t0)
                # a = cevfe

            index = torch.arange(0, config.Nangles, dtype=torch.long) # index for the dataloader
            batch_set =  config.multires_params.batch_set
            # # proj_len = len(len_set)-1-config.multires_params.startResolution
            # proj_len = config.multires_params.startResolution
            # proj_set_Data = torch.FloatTensor(proj_pyramid_set[proj_len]).to(device)
            dataset = TensorDataset(angles_t,projections_noisy_resized_t.detach(),index)
            print('New resolution: ', projections_noisy_resized_t.shape)
            trainLoader = DataLoader(dataset, batch_size = batch_set[0], shuffle=True, drop_last=True)
            # ray_length_set = config.ray_length
            batch_set_index = 0
            ray_change_epoch = config.multires_params.ray_change_epoch
            multi_resolution_counter = 0
            # Save current projection used
            print("Saving current tilt-series used with resolution: ", projections_noisy_resized_t.shape)
            if not os.path.exists(config.path_save + "training/projections/multires_input_" + str(projections_noisy_resized_t.shape[1]) + '/'):
                os.makedirs(config.path_save + "training/projections/multires_input_" + str(projections_noisy_resized_t.shape[1]) + '/')
            for ll in range(projections_noisy_resized_t.shape[0]):
                tmp = projections_noisy_resized_t[ll].detach().cpu().numpy()
                tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
                tmp = np.floor(255 * tmp).astype(np.uint8)
                imageio.imwrite(
                    config.path_save_data + "training/projections/multires_input_" + str(projections_noisy_resized_t.shape[1]) + "/est_" + str(
                        ll).zfill(5) + ".png", tmp)
            print("Elapsed time: ", time.time()-t0)
    ######################################################################################################
    ## Iterative optimization
    loss_tot = []
    loss_data_fidelity = []
    loss_regul_local_ampl = []
    loss_regul_volume = []
    loss_regul_shifts = []
    loss_regul_rot = []
    shift_estimates = []
    rot_estimates = []
    train_volume = config.train_volume
    learn_deformations = False
    check_point_training = True
    if config.track_memory:
        memory_used = []
        check_point_training = False # Do not stop for display when keeping track of the memory

    t0_train = time.time()
    print("Training the network(s)...")
    for ep in range(config.epochs):
        loss_tmp = []
        # define what to estimate
        if(ep>=config.delay_deformations): 
            train_global_def = config.train_global_def
            train_local_def = config.train_local_def
            learn_deformations = True
            use_local_def = True if train_local_def else False
            use_global_def = True if train_global_def else False       
        else:
            use_local_def = False
            use_global_def = False
            train_local_def = False
            train_global_def = False

        if hasattr(config, 'multiresolution'):
            if config.multiresolution:
                if (ep in ray_change_epoch):
                    with torch.no_grad():
                        print("Updating the resolution of tilt-series")
                        t0 = time.time()
                        res_factor -= 1
                        _, n1_origin, n2_origin = projections_noisy.shape
                        n1_resize = int(n1_origin/(2**res_factor))
                        n2_resize = int(n2_origin/(2**res_factor))
                        projections_noisy_resized_t = projections_noisy[:, ::2 ** res_factor, ::2 ** res_factor]
                        # projections_noisy_resized = resize(projections_noisy.detach().cpu().numpy(),
                        #                                    (Nangles, n1_resize, n2_resize))
                        # projections_noisy_resized_t = torch.tensor(projections_noisy_resized).type(config.torch_type).to(
                        #     device)
                        # factor = 2**res_factor
                        # h_crop = n1_origin // factor // 2
                        # w_crop = n2_origin // factor // 2
                        # center_h = n1_origin // 2
                        # center_w = n2_origin // 2
                        # fft_cropped = torch.fft.fftshift(torch.fft.fft2(projections_noisy.detach().cpu()))[:,
                        #           center_h - h_crop:center_h + h_crop,
                        #           center_w - w_crop:center_w + w_crop]
                        # projections_noisy_resized_t = (torch.fft.ifft2(torch.fft.ifftshift(fft_cropped)) * (factor**2)).real
                        # projections_noisy_resized_t = projections_noisy_resized_t.type(config.torch_type).to(device)
                        #
                        # projections_noisy_resized_t = projections_noisy_resized_t / torch.abs(
                        #     projections_noisy_resized_t).max()  # make sure that values to predict are between -1 and 1

                        t0 = time.time()
                        factor = 2 ** res_factor
                        h_crop = n1_origin // factor // 2
                        w_crop = n2_origin // factor // 2
                        center_h = n1_origin // 2
                        center_w = n2_origin // 2
                        fft_cropped = torch.fft.fftshift(
                            torch.fft.fft2(projections_noisy.detach().cpu(), norm='forward'))[:,
                                      center_h - h_crop:center_h + h_crop,
                                      center_w - w_crop:center_w + w_crop]
                        projections_noisy_resized_t = (
                            torch.fft.ifft2(torch.fft.ifftshift(fft_cropped), norm='forward')).real
                        projections_noisy_resized_t = projections_noisy_resized_t.type(config.torch_type).to(device)
                        # projections_noisy_resized_t = projections_noisy_resized_t / torch.abs(
                        #     projections_noisy_resized_t).max()  # make sure that values to predict are between -1 and 1
                        print(projections_noisy_resized_t.min(), projections_noisy_resized_t.max(),
                              projections_noisy_resized_t.shape)
                        print("Elapsed time Fourier cropping: ", time.time() - t0)

                    # multi_resolution_counter += 1
                    batch_set_index = min(len(batch_set) - 1, batch_set_index + 1)
                    # proj_len = max(0, proj_len - 1)

                    index = torch.arange(0, config.Nangles, dtype=torch.long)  # index for the dataloader
                    # proj_set_Data = torch.FloatTensor(proj_pyramid_set[proj_len]).to(device)
                    print('New resolution: ', projections_noisy_resized_t.shape)
                    dataset = TensorDataset(angles_t, projections_noisy_resized_t.detach(), index)
                    trainLoader = DataLoader(dataset, batch_size=batch_set[batch_set_index], shuffle=True,
                                             drop_last=True)

                    print("Elapsed time: ", time.time() - t0)
                    # Save current projection used
                    print("Saving current tilt-series used with resolution: ", projections_noisy_resized_t.shape)
                    t0 = time.time()
                    if not os.path.exists(config.path_save + "training/projections/multires_input_"+str(projections_noisy_resized_t.shape[1])+'/'):
                        os.makedirs(config.path_save + "training/projections/multires_input_"+str(projections_noisy_resized_t.shape[1])+'/')
                    for ll in range(projections_noisy_resized_t.shape[0]):
                        tmp = projections_noisy_resized_t[ll].detach().cpu().numpy()
                        tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
                        tmp = np.floor(255*tmp).astype(np.uint8)
                        imageio.imwrite(config.path_save_data+"training/projections/multires_input_"+str(projections_noisy_resized_t.shape[1])+"/est_"+str(ll).zfill(5)+".png",tmp)
                    print("Elapsed time: ", time.time() - t0)
        for   angle, proj, idx_loader  in trainLoader:
            optimizer_volume.zero_grad()
            if learn_deformations:
                if train_global_def:
                    optimizer_deformations_glob.zero_grad()
                if train_local_def:
                    optimizer_deformations_loc.zero_grad()

                # Check if we stop or start learning something new
                if ep in config.schedule_local:
                    if train_local_def:
                        train_local_def = False
                    else:
                        train_local_def = True
                    if use_local_def is False:
                        use_local_def = True
                if ep in config.schedule_global:
                    if train_global_def:
                        train_global_def = False
                    else:
                        train_global_def = True
                    if use_global_def is False:
                        use_global_def = True
            if ep in config.schedule_volume:
                if train_volume:
                    train_volume = False
                else:
                    train_volume = True
            # Choosing the subset of the parameters
            if(use_local_def):
                local_deformSet= list(map(implicit_deformation_list.__getitem__, idx_loader))
            else:
                local_deformSet = None
            if use_global_def:
                rot_deformSet= list(map(rot_est.__getitem__, idx_loader))
                shift_deformSet= list(map(shift_est.__getitem__, idx_loader))
            else:
                rot_deformSet = None
                shift_deformSet = None
            # fixedRotSet = list(map(fixed_rot.__getitem__, idx_loader))

            # Define the detector locations
            detectorLocations = torch.rand(proj.shape[0],config.nRays,2).to(device)*2-1

            # Apply deformations in the 2D space
            detectorLocationsDeformed = apply_deformations_to_locations(detectorLocations,rot_deformSet,
                                                                    shift_deformSet,local_deformSet,fixed_rot,
                                                                    scale=config.deformationScale, cl=config.clip)

            # generate the rays in 3D
            rays_rotated = generate_rays_batch(detectorLocationsDeformed, angle, z_max_value, config.ray_length, std_noise=config.std_noise_z)

            # Scale the rays so that they are trully in [-1,1] 
            rays_rotated_scaled = rays_rotated/size_max_vol

            # Sample the implicit volume by making the input in [0,1]
            outputValues = impl_volume((rays_rotated_scaled/2+0.5).reshape(-1,3)).reshape(proj.shape[0],config.nRays,config.ray_length)

            support = (rays_rotated[:,:,:,2].abs()<config.size_z_vol)*1
            projEstimate = torch.sum(support*outputValues,2)/config.ray_length
            pixelValues = sample_projections(proj, detectorLocations, interp='bilinear')

            # Take the datafidelity loss
            loss = loss_data(projEstimate*gains[idx_loader,None]*weights_tilt[idx_loader,None],pixelValues.to(projEstimate.dtype)*weights_tilt[idx_loader,None])
            loss_data_fidelity.append(loss.item())

            ## Add regularizations
            if train_local_def and config.lamb_local_ampl!=0:
                # Using only the x and y coordinates
                for ii_ in idx_loader:
                    depl = torch.abs(implicit_deformation_list[ii_](detectorLocations.reshape(-1,2))*config.n1)
                    depl_mean = torch.abs(torch.mean(implicit_deformation_list[ii_](detectorLocations.reshape(-1,2))*config.n1))
                    loss += (config.lamb_local_ampl*depl.mean()+config.lamb_local_mean*depl_mean)
                    loss_regul_local_ampl.append((config.lamb_local_ampl*depl.mean()+config.lamb_local_mean*depl_mean).item())
            if train_global_def and (config.lamb_rot!=0 or config.lamb_shifts!=0):
                for ii in idx_loader:
                    loss += config.lamb_shifts*torch.abs(shift_est[ii]()*config.n1).mean()
                    loss += config.lamb_rot*torch.abs(rot_est[ii].thetas*180/np.pi).mean()
                    loss_regul_shifts.append((config.lamb_shifts*torch.abs(shift_est[ii]()*config.n1).mean()).item())
                    loss_regul_rot.append((config.lamb_rot*torch.abs(rot_est[ii].thetas*180/np.pi).mean()).item())
            if config.train_volume and config.lamb_volume!=0:
                loss += torch.linalg.norm(outputValues[outputValues<0])*config.lamb_volume
                loss_regul_volume.append((torch.linalg.norm(outputValues[outputValues<0])*config.lamb_volume).item())

            # Compute gradient and optimize
            loss.backward()
            if train_volume:
                optimizer_volume.step()
            if train_global_def:
                optimizer_deformations_glob.step()
            if train_local_def:
                optimizer_deformations_loc.step()
            loss_tmp.append(loss.item())

            # for jj in range(len(shift_est)):
            #     shift_est[jj].shifts_arr = torch.clip(shift_est[jj].shifts_arr, - 0.1, 0.1)

        loss_tot.append(np.mean(loss_tmp))
        scheduler_volume.step()
        if len(list_params_deformations_glob)!=0:
            scheduler_deformation_glob.step()
        if len(list_params_deformations_loc)!=0:
            scheduler_deformation_loc.step()

        shiftEstimate, rotEstimate = globalDeformationValues(shift_est,rot_est)
        shift_estimates.append(shiftEstimate)
        rot_estimates.append(rotEstimate)
        
        # Track loss and display values
        if ((ep%10)==0 and (ep%config.Ntest!=0)):
            loss_current_epoch = np.mean(loss_tot[-len(trainLoader):])
            l_fid = np.mean(loss_data_fidelity[-len(trainLoader):])
            l_v = np.mean(loss_regul_volume[-len(trainLoader):])
            l_sh = np.mean(loss_regul_shifts[-len(trainLoader)*trainLoader.batch_size:])
            l_rot = np.mean(loss_regul_rot[-len(trainLoader)*trainLoader.batch_size:])
            l_loc = np.mean(loss_regul_local_ampl[-len(trainLoader)*trainLoader.batch_size:])
            print("Epoch: {}, loss_avg: {:.3e} || Loss data fidelity: {:.3e}, regul volume: {:.2e}, regul shifts: {:.2e}, regul inplane: {:.2e}, regul local: {:.2e}, time: {:2.0f} s".format(
                ep,loss_current_epoch,l_fid,l_v,l_sh,l_rot,l_loc,time.time()-t0))
            # print("Fixed rotation: ", np.round(fixed_rot.thetas.detach().cpu().item()*180/np.pi,2))
        if config.track_memory:
            memory_used.append(torch.cuda.memory_allocated())

        # Save and display some results
        if (ep%config.Ntest==0) and check_point_training:
            with torch.no_grad():
                ## Avergae loss over until the last test
                loss_current_epoch = np.mean(loss_tot[-len(trainLoader)*config.Ntest:])
                l_fid = np.mean(loss_data_fidelity[-len(trainLoader)*config.Ntest:])
                l_v = np.mean(loss_regul_volume[-len(trainLoader)*config.Ntest:])
                l_sh = np.mean(loss_regul_shifts[-len(trainLoader)*config.Ntest:])
                l_rot = np.mean(loss_regul_rot[-len(trainLoader)*config.Ntest:])
                l_loc = np.mean(loss_regul_local_ampl[-len(trainLoader)*config.Ntest:])
                print("----Epoch: {}, loss_avg: {:.3e} || Loss data fidelity: {:.3e}, regul volume: {:.2e}, regul shifts: {:2.4f}, regul inplane: {:.2e}, regul local: {:2.4f}, time: {:2.0f} s".format(
                    ep,loss_current_epoch,l_fid,l_v,l_sh,l_rot,l_loc,time.time()-t0_train))
                
                print('Running and saving tests')  

                ## Save local deformation
                print("Saving local deformations")
                t0 = time.time()
                utils_display.display_local_movie(implicit_deformation_list,field_true=None,Npts=(20,20),
                                            img_path=config.path_save+"/training/deformations/local_deformations_",img_type='.png',
                                            scale=1,alpha=0.8,width=0.0015,weights_est=1)
                # # for index in range(len(implicit_deformation_list)):
                # for index in range(3):
                utils_display.display_local_movie(implicit_deformation_list,field_true=None,Npts=(20,20),
                                            img_path=config.path_save+"/training/deformations_x10/local_deformations_",img_type='.png',
                                            scale=0.1,alpha=0.8,width=0.0015,weights_est=1)
                    
                ## Save global deformation
                shiftEstimate, rotEstimate = globalDeformationValues(shift_est,rot_est)
                plt.figure(1)
                plt.clf()
                # plt.hist(shiftEstimate.reshape(-1)*config.n1,alpha=1)
                plt.scatter(angles,shiftEstimate[:,0,0]*config.n1, label='x')
                plt.scatter(angles,shiftEstimate[:,0,1]*config.n1, label='y')
                plt.axis([angles[0], angles[-1], -200, 200])
                plt.legend()
                plt.savefig(os.path.join(config.path_save+"/training/deformations/shifts"+str(ep)+".png"))

                plt.figure(1)
                plt.clf()
                # plt.hist(rotEstimate*180/np.pi,15)
                plt.scatter(angles, rotEstimate*180/np.pi)
                plt.scatter(angles, np.ones(len(angles))*fixed_rot.thetas.detach().cpu().numpy()*180/np.pi)
                plt.axis([angles[0], angles[-1], -20, 20])
                ang_ = np.round(fixed_rot.thetas.detach().cpu().item()*180/np.pi,2)
                # print("Fixed rotation: ", np.round(fixed_rot.thetas.detach().cpu().item()*180/np.pi,2))
                plt.legend(['est.','fixed rot. '+str(ang_)])
                plt.title('Angles in degrees')
                plt.savefig(os.path.join(config.path_save+"/training/deformations/rotations"+str(ep)+".png"))

                plt.figure(1)
                plt.clf()
                plt.plot(angles, gains.detach().cpu().numpy())
                plt.axis([angles[0], angles[-1], 0.1, 2])
                plt.savefig(os.path.join(config.path_save+"/training/deformations/gains"+str(ep)+".png"))
                print("Elapsed time: {:2.0f} s".format(time.time()-t0))

                if ep%config.Nalign == 0 and ep!=0:
                    # Save aligned projections
                    print("Aligning and saving projections")
                    t0 = time.time()
                    aligned_proj = aligned_projections(projections_noisy.detach().cpu(), rot_est, shift_est,
                                                       implicit_deformation_list, fixed_rot, deformationScale=config.deformationScale,
                                                       torch_type=config.torch_type, device=torch.device('cpu'))
                    if not os.path.exists(config.path_save + "training/projections/raw_aligned/"+str(ep).zfill(3)+"/"):
                        os.makedirs(config.path_save + "training/projections/raw_aligned/"+str(ep).zfill(3)+"/")
                    for ll in range(aligned_proj.shape[0]):
                        tmp = aligned_proj[ll].detach().cpu().numpy()/gains[ll].detach().cpu().numpy()
                        tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
                        tmp = np.floor(255 * tmp).astype(np.uint8)
                        imageio.imwrite(
                            config.path_save_data + "training/projections/raw_aligned/"+str(ep).zfill(3)+"/est_" + str(ll).zfill(5) + ".png", tmp)
                    os.system("ffmpeg -y -f image2 -framerate 10 -i "+ config.path_save_data + "training/projections/raw_aligned/"+str(ep).zfill(3)+"/est_%05d.png -crf 22 -vf scale=512x512  "+config.path_save_data + "training/projections/raw_aligned/"+"aligned_tilts_ep_{}.gif".format(ep))
                    print("Elapsed time: {:2.0f} s".format(time.time()-t0))

                if ep%config.Nimplicit_volume == 0 and ep!=0:
                    # Compute estimated projections
                    print("Compute estimated projections from the implicit net")
                    t0 = time.time()
                    projEstimate_tot = np.zeros((config.Nangles, config.n1_eval, config.n2_eval))
                    x_lin1 = np.linspace(-1, 1, config.n1_eval)
                    x_lin2 = np.linspace(-1, 1, config.n2_eval)
                    XX, YY = np.meshgrid(x_lin1, x_lin2, indexing='ij')
                    grid2d_ = np.concatenate([XX.reshape(-1, 1), YY.reshape(-1, 1)], 1)
                    grid2d_ = torch.tensor(grid2d_).type(config.torch_type).to(device)
                    for ll, angle in enumerate(angles_t[::4]):
                        for jj in range(config.n2_eval):
                            # Define the detector locations
                            detectorLocations = grid2d_.reshape(config.n1_eval,config.n2_eval,2)[:,jj].reshape(1,-1,2)

                            # Apply deformations in the 2D space
                            detectorLocationsDeformed = apply_deformations_to_locations(detectorLocations, rot_est[ll:ll+1],
                                                                                        shift_est[ll:ll+1], implicit_deformation_list[ll:ll+1],
                                                                                        fixed_rot, scale=config.deformationScale, cl=config.clip)
                            # generate the rays in 3D
                            rays_rotated = generate_rays_batch(detectorLocationsDeformed, angle[None], z_max_value, config.ray_length,
                                                               std_noise=config.std_noise_z)
                            # Scale the rays so that they are trully in [-1,1]
                            rays_rotated_scaled = rays_rotated / size_max_vol
                            # Sample the implicit volume by making the input in [0,1]
                            outputValues = impl_volume((rays_rotated_scaled / 2 + 0.5).reshape(-1, 3)).reshape(1,
                                                                                                               config.n1_eval,
                                                                                                               config.ray_length)
                            support = (rays_rotated[:, :, :, 2].abs() < config.size_z_vol) * 1
                            projEstimate = torch.sum(support * outputValues, 2) / config.ray_length
                            projEstimate_tot[ll,:,jj] = projEstimate.reshape(-1).detach().cpu().numpy()
                        tmp = projEstimate_tot[ll]
                        tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
                        tmp = np.floor(255*tmp).astype(np.uint8)
                        imageio.imwrite(config.path_save_data+"training/projections/est_"+str(ll).zfill(5)+".png",tmp)
                    print("Elapsed time: {:2.0f} s".format(time.time()-t0))


                                    
                if config.save_volume:
                    ## Save slice of the volume
                    print("Computing and saving the volume from the implict net")
                    t0 = time.time()
                    z_range = np.linspace(-1,1,config.n3_patch)*config.size_z_vol
                    V_icetide = np.zeros((config.n1_patch,config.n2_patch,config.n3_patch))
                    for zz, zval in enumerate(z_range):
                        grid3d = np.concatenate([grid2d_t, zval*torch.ones((grid2d_t.shape[0],1))],1)
                        grid3d_slice = torch.tensor(grid3d).type(config.torch_type).to(device)
                        estSlice = impl_volume(grid3d_slice/size_max_vol/2+0.5).detach().cpu().numpy().reshape(config.n1_patch,config.n2_patch)
                        pp = (estSlice)*1.
                        V_icetide[:,:,zz] = estSlice
                        plt.figure(1)
                        plt.clf()
                        plt.imshow(pp,cmap='gray')
                        plt.savefig(os.path.join(config.path_save+"/training/volume/volume_est_slice_{}.png".format(zz)))
                    if config.avg_XYZ>1:
                        padded_array = np.pad(V_icetide, ((0, 0), (0, 0), (0, config.avg_XYZ - 1)), mode='constant')
                        filt = np.zeros_like(padded_array)
                        filt[:,:,filt.shape[2]//2-config.avg_XYZ//2:filt.shape[2]//2+config.avg_XYZ//2] = 1/config.avg_XYZ
                        V_icetide = np.fft.fftshift(np.fft.ifft((np.fft.fft(filt) * np.fft.fft(padded_array))).real,axes=-1)[:,:,:config.n3_patch]
 
                    def display_XYZ(tmp,name="true"):
                        avg = 0
                        sl0 = tmp.shape[0]//2
                        sl1 = tmp.shape[1]//2
                        sl2 = tmp.shape[2]//2
                        f , aa = plt.subplots(2, 2, gridspec_kw={'height_ratios': [tmp.shape[2]/tmp.shape[0], 1], 'width_ratios': [1,tmp.shape[2]/tmp.shape[0]]})
                        aa[0,0].imshow(tmp[sl0-avg//2:sl0+avg//2+1,:,:].mean(0).T,cmap='gray',vmin=tmp.min(),vmax=tmp.max())
                        aa[0,0].axis('off')
                        aa[1,0].imshow(tmp[:,:,sl2-avg//2:sl2+avg//2+1].mean(2),cmap='gray',vmin=tmp.min(),vmax=tmp.max())
                        aa[1,0].axis('off')
                        aa[1,1].imshow(tmp[:,sl1-avg//2:sl1+avg//2+1,:].mean(1),cmap='gray',vmin=tmp.min(),vmax=tmp.max())
                        aa[1,1].axis('off')
                        aa[0,1].axis('off')
                        plt.tight_layout(pad=1, w_pad=-1, h_pad=1)
                        plt.savefig(os.path.join("tmp.png"))
                        plt.savefig(os.path.join(config.path_save_data,'training',"volume",name+"_XYZ_slice.png"))

                        f , aa = plt.subplots(2, 2, gridspec_kw={'height_ratios': [tmp.shape[2]/tmp.shape[0], 1], 'width_ratios': [1,tmp.shape[2]/tmp.shape[0]]})
                        aa[0,0].imshow(tmp.mean(0).T,cmap='gray',vmin=tmp.min(),vmax=tmp.max())
                        aa[0,0].axis('off')
                        aa[1,0].imshow(tmp.mean(2),cmap='gray',vmin=tmp.min(),vmax=tmp.max())
                        aa[1,0].axis('off')
                        aa[1,1].imshow(tmp.mean(1),cmap='gray',vmin=tmp.min(),vmax=tmp.max())
                        aa[1,1].axis('off')
                        aa[0,1].axis('off')
                        plt.tight_layout(pad=1, w_pad=-1, h_pad=1)
                        plt.savefig(os.path.join(config.path_save_data,'training',"volume",name+"_XYZ_proj.png"))

                    # ICETIDE
                    tmp = V_icetide
                    tmp = (tmp-tmp.min())/(tmp.max()-tmp.min())
                    tmp = np.clip(tmp,a_min=np.quantile(tmp,0.005),a_max=np.quantile(tmp,0.995))
                    display_XYZ(tmp,name="ICETIDE")
                    print("Elapsed time: {:2.0f} s".format(time.time()-t0))
                        
                print("Saving the model and parameters")
                torch.save({
                    'shift_est': shift_est,
                    'rot_est': rot_est,
                    'gains': gains,
                    'local_deformation_network': implicit_deformation_list,
                    'implicit_volume': impl_volume.state_dict(),
                    'optimizer_volume' : optimizer_volume.state_dict(),
                    'optimizer_deformations_glob' : optimizer_deformations_glob.state_dict(),
                    #'optimizer_deformations_loc' : optimizer_deformations_loc.state_dict(),
                    'scheduler_volume': scheduler_volume.state_dict(), 
                    'scheduler_deformation_glob': scheduler_deformation_glob.state_dict(), 
                    #'scheduler_deformation_loc': scheduler_deformation_loc.state_dict(),
                    'ep': ep,
                }, os.path.join(config.path_save,'training','model_trained.pt'))

                loss_tot_avg = np.array(loss_tot)
                step = (loss_tot_avg.max()-loss_tot_avg.min())*0.02
                plt.figure(figsize=(10,10))
                plt.semilogy(loss_tot_avg[10:])
                plt.xticks(np.arange(0, len(loss_tot_avg[1:]), 1+len(loss_tot_avg[1:])//10))
                # plt.yticks(np.linspace(loss_tot_avg.min()-step,loss_tot_avg.max()+step, 14))
                # plt.grid()
                plt.savefig(os.path.join(config.path_save,'training','loss.pdf'))
        plt.close('all')

        if ep == 50:
            pr.disable()
            s = io.StringIO()
            sortby = SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())

    print("Saving final state after training...")
    torch.save({
        'shift_est': shift_est,
        'rot_est': rot_est,
        'gains': gains,
        'local_deformation_network': implicit_deformation_list,
        'implicit_volume': impl_volume.state_dict(),
        'optimizer_volume' : optimizer_volume.state_dict(),
        'optimizer_deformations_glob' : optimizer_deformations_glob.state_dict(),
        #'optimizer_deformations_loc' : optimizer_deformations_loc.state_dict(),
        'scheduler_volume': scheduler_volume.state_dict(), 
        'scheduler_deformation_glob': scheduler_deformation_glob.state_dict(), 
        #'scheduler_deformation_loc': scheduler_deformation_loc.state_dict(),
        'ep': ep,
    }, os.path.join(config.path_save,'training','model_trained.pt'))

    training_time = time.time()-t0
    # Saving the training time and the memory used
    if config.track_memory:
        max_memory_allocated_bytes = torch.cuda.max_memory_allocated()
        # Convert bytes to gigabytes
        max_memory_allocated_gb = max_memory_allocated_bytes / (1024**3)
        np.save(os.path.join(config.path_save,'training','memory_used.npy'),memory_used)
        np.savetxt(os.path.join(config.path_save,'training','memory_used.txt'),np.array([np.max(memory_used)/ (1024**3),max_memory_allocated_gb])) # Conversion in Gb
    np.save(os.path.join(config.path_save,'training','training_time.npy'),training_time)
    np.savetxt(os.path.join(config.path_save,'training','training_time.txt'),np.array([training_time]))

    with torch.no_grad():
        ## Save slice of the volume
        z_range = np.linspace(-1,1,config.n3_patch)*config.size_z_vol
        V_icetide = np.zeros((config.n1_patch,config.n2_patch,config.n3_patch))
        for zz, zval in enumerate(z_range):
            grid3d = np.concatenate([grid2d_t, zval*torch.ones((grid2d_t.shape[0],1))],1)
            grid3d_slice = torch.tensor(grid3d).type(config.torch_type).to(device)
            estSlice = impl_volume(grid3d_slice/size_max_vol/2+0.5).detach().cpu().numpy().reshape(config.n1_patch,config.n2_patch)
            pp = (estSlice)*1.
            V_icetide[:,:,zz] = estSlice
        if config.avg_XYZ>1:
            padded_array = np.pad(V_icetide, ((0, 0), (0, 0), (0, config.avg_XYZ - 1)), mode='constant')
            filt = np.zeros_like(padded_array)
            filt[:,:,filt.shape[2]//2-config.avg_XYZ//2:filt.shape[2]//2+config.avg_XYZ//2] = 1/config.avg_XYZ
            V_icetide = np.fft.fftshift(np.fft.ifft((np.fft.fft(filt) * np.fft.fft(padded_array))).real,axes=-1)[:,:,:config.n3_patch]
        out = mrcfile.new(config.path_save+"/training/V_est_final.mrc",np.moveaxis(V_icetide.astype(np.float32),2,0),overwrite=True)
        out.close() 

    loss_tot_avg = np.array(loss_tot)
    step = (loss_tot_avg.max()-loss_tot_avg.min())*0.02
    plt.figure(figsize=(10,10))
    plt.plot(loss_tot_avg[10:])
    plt.xticks(np.arange(0, len(loss_tot_avg[1:]), 1+len(loss_tot_avg[1:])//10))
    plt.yticks(np.linspace(loss_tot_avg.min()-step,loss_tot_avg.max()+step, 14))
    # plt.grid()
    plt.savefig(os.path.join(config.path_save,'training','loss.png'))
    plt.savefig(os.path.join(config.path_save,'training','loss.pdf'))
    shift_estimates_np = np.array(shift_estimates)
    rot_estimates_np = np.array(rot_estimates)

    np.save(os.path.join(config.path_save,'training','shiftEstimates.npy'),shift_estimates_np)
    np.save(os.path.join(config.path_save,'training','rotestiamtes.npy'),rot_estimates_np)
    
    plt.figure(figsize=(10,10))
    plt.plot(shift_estimates_np[:,26,0,0])
    plt.plot(shift_estimates_np[:,26,0,1])
    plt.title('Shift Estimates')
    plt.savefig(os.path.join(config.path_save,'training','shiftEstimates.png'))

    print("Training is over.")

