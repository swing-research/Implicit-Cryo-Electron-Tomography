"""
Module to train the reconstruction network on the simulated data.
"""

import os
import time
import torch
import mrcfile
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from utils import utils_deformation, utils_display
from utils.utils_sampling import get_sampling_geometry, apply_deformations_to_locations, generate_rays_batch, sample_projections



def train(config):
    print("Runing training procedure.")
    # Choosing the seed and the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count()>1:
        torch.cuda.set_device(config.device_num)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # prepare the folders    
    if not os.path.exists(config.path_save+"training/"):
        os.makedirs(config.path_save+"training/")
    if not os.path.exists(config.path_save+"training/volume/"):
        os.makedirs(config.path_save+"training/volume/")
    if not os.path.exists(config.path_save+"training/deformations/"):
        os.makedirs(config.path_save+"training/deformations/")
    if not os.path.exists(config.path_save+"training/deformations_x10/"):
        os.makedirs(config.path_save+"training/deformations_x10/")

    ## Load data that was previously saved
    data = np.load(config.path_save_data+"volume_and_projections.npz")
    projections_noisy = torch.Tensor(data['projections_noisy']).type(config.torch_type).to(device)

    affine_tr = np.load(config.path_save_data+"global_deformations.npy",allow_pickle=True)
    local_tr = np.load(config.path_save_data+"local_deformations.npy", allow_pickle=True)
    shift_true = np.zeros((len(affine_tr),2))
    angle_true = np.zeros((len(affine_tr)))
    for k, affine_transformation in enumerate(affine_tr):
        shift_true[k,0] = affine_transformation.shiftX.item()
        shift_true[k,1] = affine_transformation.shiftY.item()
        angle_true[k] = affine_transformation.angle.item()

    ######################################################################################################
    ######################################################################################################
    ##
    ## TRAINING
    ##
    ######################################################################################################
    ######################################################################################################
    print("Loading models and setting parameters...")
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


    ######################################################################################################
    ## Define the implicit deformations
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
        depl_ctr_pts_net = torch.zeros((2,config.local_deformation.N_ctrl_pts_net,config.local_deformation.N_ctrl_pts_net)).to(device).type(config.torch_type)/max([config.n1,config.n2,config.n3])/10
        implicit_deformation_list = []
        for k in range(config.Nangles):
            field = utils_deformation.deformation_field(depl_ctr_pts_net.clone(),maskBoundary=config.maskBoundary)
            implicit_deformation_list.append(field)
        num_param = sum(p.numel() for p in implicit_deformation_list[0].parameters() if p.requires_grad)
        print('---> Number of trainable parameters in implicit net: {}'.format(num_param))

    ######################################################################################################
    ## Define the global deformations##############
    fixedAngle = torch.FloatTensor([config.fixed_angle* np.pi/180]).to(device)[0]

    shift_est = []
    rot_est = []
    fixed_rot =[ ]
    for k in range(config.Nangles):
        shift_est.append(utils_deformation.shiftNet(1).to(device))
        rot_est.append(utils_deformation.rotNet(1).to(device))
        fixed_rot.append(utils_deformation.rotNet(1,x0=fixedAngle).to(device))

    ######################################################################################################
    ## Optimizer
    loss_data = config.loss_data
    train_global_def = config.train_global_def
    train_local_def = config.train_local_def
    list_params_deformations_glob = []
    list_params_deformations_loc = []
    if(train_global_def or train_local_def):
        for k in range(config.Nangles):
            if train_global_def:
                list_params_deformations_glob.append({"params": shift_est[k].parameters(), "lr": config.lr_shift})
                list_params_deformations_glob.append({"params": rot_est[k].parameters(), "lr": config.lr_rot})
            if train_global_def:
                list_params_deformations_loc.append({"params": implicit_deformation_list[k].parameters(), "lr": config.lr_local_def})

    optimizer_volume = torch.optim.Adam(impl_volume.parameters(), lr=config.lr_volume, weight_decay=config.wd)
    optimizer_deformations_glob = torch.optim.Adam(list_params_deformations_glob, weight_decay=config.wd)
    optimizer_deformations_loc = torch.optim.Adam(list_params_deformations_loc, weight_decay=config.wd)

    scheduler_volume = torch.optim.lr_scheduler.StepLR(optimizer_volume, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
    scheduler_deformation_glob = torch.optim.lr_scheduler.StepLR(
        optimizer_deformations_glob, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
    scheduler_deformation_loc = torch.optim.lr_scheduler.StepLR(
        optimizer_deformations_loc, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)

    ######################################################################################################
    # Format data for batch training
    index = torch.arange(0, config.Nangles, dtype=torch.long) # index for the dataloader
    # Define dataset
    angles = np.linspace(config.view_angle_min,config.view_angle_max,config.Nangles)
    angles_t = torch.tensor(angles).type(config.torch_type).to(device)
    dataset = TensorDataset(angles_t,projections_noisy.detach(),index)
    trainLoader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    ######################################################################################################
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

    ## grid for display 
    x_lin1 = np.linspace(-1,1,config.n1)
    x_lin2 = np.linspace(-1,1,config.n2)
    XX, YY = np.meshgrid(x_lin1,x_lin2,indexing='ij')
    grid2d = np.concatenate([XX.reshape(-1,1),YY.reshape(-1,1)],1)
    grid2d_t = torch.tensor(grid2d).type(config.torch_type)

    # Define geometry of sampling
    size_xy_vol = np.max([config.sampling_domain_lx, config.sampling_domain_ly,config.size_z_vol])
    th = np.linspace(config.view_angle_min,config.view_angle_max,100)
    z_max_value1 = np.abs(size_xy_vol*np.sin(th*np.pi/180) + config.size_z_vol*np.cos(th*np.pi/180)).max()
    z_max_value2 = np.abs(size_xy_vol*np.sin(th*np.pi/180) - config.size_z_vol*np.cos(th*np.pi/180)).max()
    z_max_value = np.maximum(z_max_value1,z_max_value2)
    size_max_vol = 2*np.max([config.sampling_domain_lx, config.sampling_domain_ly,config.size_z_vol])


    ######################################################################################################
    ## Iterative optimization
    loss_tot = []
    loss_data_fidelity = []
    loss_regul_local_ampl = []
    loss_regul_volume = []
    loss_regul_shifts = []
    loss_regul_rot = []

    train_volume = config.train_volume
    learn_deformations = False
    check_point_training = True
    if config.track_memory:
        memory_used = []
        check_point_training = False # Do not stop for display when keeping track of the memory
    if config.compute_fsc:
        from utils import utils_FSC 
        import compare_results
        ep_tot = []
        resolution_icetide_tot = []
        resolution_FBP_tot = []
        resolution_FBP_no_deformed_tot = []
        CC_icetide_tot = []
        CC_FBP_tot = []
        CC_FBP_no_deformed_tot = []
        V = np.moveaxis(np.double(mrcfile.open(config.path_save_data+"V.mrc").data),0,2)
        V_FBP_no_deformed = np.moveaxis(np.double(mrcfile.open(config.path_save_data+"V_FBP_no_deformed.mrc").data),0,2)
        V_FBP =  np.moveaxis(np.double(mrcfile.open(config.path_save_data+"V_FBP.mrc").data),0,2)
        fsc_FBP = utils_FSC.FSC(V,V_FBP)
        fsc_FBP_no_deformed = utils_FSC.FSC(V,V_FBP_no_deformed)
        CC_FBP = compare_results.CC(V,V_FBP)
        CC_FBP_no_deformed = compare_results.CC(V,V_FBP_no_deformed)

        indeces = np.where(fsc_FBP<0.5)[0]
        choosenIndex = np.where(indeces>2)[0][0]
        resolution_FBP = indeces[choosenIndex]
        indeces = np.where(fsc_FBP_no_deformed<0.5)[0]
        choosenIndex = np.where(indeces>2)[0][0]
        resolution_FBP_no_deformed = indeces[choosenIndex]
    t0 = time.time()
    print("Training the network(s)...")
    for ep in range(config.epochs):
        # define what to estimate
        if(ep>=config.delay_deformations):
            learn_deformations = True
            use_local_def = True if train_local_def else False
            use_global_def = True if train_global_def else False        
            train_global_def = config.train_global_def
            train_local_def = config.train_local_def
        else:
            use_local_def = False
            use_global_def = False
            train_local_def = False
            train_global_def = False
        for angle, proj, idx_loader  in trainLoader:
            optimizer_volume.zero_grad()
            if learn_deformations:
                optimizer_deformations_glob.zero_grad()
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
            fixedRotSet = list(map(fixed_rot.__getitem__, idx_loader))

            # Define the detector locations
            detectorLocations = torch.rand(proj.shape[0],config.nRays,2).to(device)*2-1

            # Apply deformations in the 2D space
            detectorLocationsDeformed = apply_deformations_to_locations(detectorLocations,rot_deformSet,
                                                                    shift_deformSet,local_deformSet,fixedRotSet,scale=config.deformationScale)

            # generate the rays in 3D
            rays_rotated = generate_rays_batch(detectorLocationsDeformed, angle, z_max_value, config.ray_length, std_noise=config.std_noise_z)

            # Scale the rays so that they are trully in [-1,1] 
            rays_rotated_scaled = rays_rotated/size_max_vol

            # Sample the implicit volume by making the input in [0,1]
            outputValues = impl_volume((rays_rotated_scaled/2+0.5).reshape(-1,3)).reshape(proj.shape[0],config.nRays,config.ray_length)

            support = (rays_rotated[:,:,:,2].abs()<config.size_z_vol)*(rays_rotated[:,:,:,0].abs()<config.sampling_domain_lx)*(rays_rotated[:,:,:,1].abs()<config.sampling_domain_ly)
            projEstimate = torch.sum(support*outputValues,2)/config.ray_length
            pixelValues = sample_projections(proj, detectorLocations, interp='bilinear')

            # Take the datafidelity loss
            loss = loss_data(projEstimate,pixelValues.to(projEstimate.dtype))
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
                    loss += config.lamb_rot*torch.abs(rot_est[ii]()*180/np.pi).mean()
                    loss_regul_shifts.append((config.lamb_shifts*torch.abs(shift_est[ii]()*config.n1).mean()).item())
                    loss_regul_rot.append((config.lamb_rot*torch.abs(rot_est[ii]()*180/np.pi).mean()).item())
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
            loss_tot.append(loss.item())

        scheduler_volume.step()
        scheduler_deformation_glob.step()
        scheduler_deformation_loc.step()
        # Track loss and display values
        if ((ep%10)==0 and (ep%config.Ntest!=0)):
            loss_current_epoch = np.mean(loss_tot[-len(trainLoader):])
            l_fid = np.mean(loss_data_fidelity[-len(trainLoader):])
            l_v = np.mean(loss_regul_volume[-len(trainLoader):])
            l_sh = np.mean(loss_regul_shifts[-len(trainLoader)*trainLoader.batch_size:])
            l_rot = np.mean(loss_regul_rot[-len(trainLoader)*trainLoader.batch_size:])
            l_loc = np.mean(loss_regul_local_ampl[-len(trainLoader)*trainLoader.batch_size:])
            print("Epoch: {}, loss_avg: {:.3e} || Loss data fidelity: {:.3e}, regul volume: {:.3e}, regul shifts: {:.3e}, regul inplane: {:.3e}, regul local: {:.3e}, time: {:2.0f} s".format(
                ep,loss_current_epoch,l_fid,l_v,l_sh,l_rot,l_loc,time.time()-t0))
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
                    ep,loss_current_epoch,l_fid,l_v,l_sh,l_rot,l_loc,time.time()-t0))
                
                print('Running and saving tests')  

                ## Save local deformation
                utils_display.display_local_movie(implicit_deformation_list,field_true=local_tr,Npts=(20,20),
                                            img_path=config.path_save+"/training/deformations/local_deformations_",img_type='.png',
                                            scale=1,alpha=0.8,width=0.0015,weights_est=1)
                for index in range(len(implicit_deformation_list)):
                    utils_display.display_local_est_and_true(implicit_deformation_list[index],local_tr[index],Npts=(20,20),scale=0.1,
                                                img_path=config.path_save+"/training/deformations_x10/local_deformations_"+str(index),
                                                img_type='.png')

                    
                ## Save global deformation
                shiftEstimate, rotEstimate = globalDeformationValues(shift_est,rot_est)
                plt.figure(1)
                plt.clf()
                plt.hist(shiftEstimate.reshape(-1)*config.n1,alpha=1)
                plt.hist(shift_true.reshape(-1)*config.n1,alpha=0.5)
                plt.legend(['est.','true'])
                plt.savefig(os.path.join(config.path_save+"/training/deformations/shifts.png"))

                plt.figure(1)
                plt.clf()
                plt.hist(rotEstimate*180/np.pi,15)
                plt.hist(angle_true*180/np.pi,15,alpha=0.5)
                plt.legend(['est.','true'])
                plt.title('Angles in degrees')
                plt.savefig(os.path.join(config.path_save+"/training/deformations/rotations.png"))

                ## Save the loss    
                loss_tot_avg = np.array(loss_tot).reshape(config.Nangles//config.batch_size,-1).mean(0)
                step = (loss_tot_avg.max()-loss_tot_avg.min())*0.02
                plt.figure(figsize=(10,10))
                plt.plot(loss_tot_avg[10:])
                plt.xticks(np.arange(0, len(loss_tot_avg[1:]), 100))
                plt.yticks(np.linspace(loss_tot_avg.min()-step,loss_tot_avg.max()+step, 14))
                plt.grid()
                plt.savefig(os.path.join(config.path_save,'training','loss_iter.png'))
                                                    
                if config.save_volume:
                    ## Save slice of the volume
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
                            
                torch.save({
                    'shift_est': shift_est,
                    'rot_est': rot_est,
                    'local_deformation_network': implicit_deformation_list,
                    'implicit_volume': impl_volume.state_dict(),
                    'optimizer_volume' : optimizer_volume,
                    'optimizer_deformations_glob' : optimizer_deformations_glob,
                    'optimizer_deformations_loc' : optimizer_deformations_loc,
                    'scheduler_volume': scheduler_volume, 
                    'scheduler_deformation_glob': scheduler_deformation_glob, 
                    'scheduler_deformation_loc': scheduler_deformation_loc,
                    'ep': ep,
                }, os.path.join(config.path_save,'training','model_trained.pt'))

        if (ep%config.Ntest==0) and check_point_training and config.compute_fsc:
            with torch.no_grad():
                resolution_FBP_tot.append(resolution_FBP)
                resolution_FBP_no_deformed_tot.append(resolution_FBP_no_deformed)
                CC_FBP_tot.append(CC_FBP)
                CC_FBP_no_deformed_tot.append(CC_FBP_no_deformed)
                ## Compute our model at same resolution than other volume
                n1_eval, n2_eval, n3_eval = V.shape
                # Compute estimated volumex
                x_lin1 = np.linspace(-1,1,n1_eval)
                x_lin2 = np.linspace(-1,1,n2_eval)
                XX, YY = np.meshgrid(x_lin1,x_lin2,indexing='ij')
                grid2d_ = np.concatenate([XX.reshape(-1,1),YY.reshape(-1,1)],1)
                grid2d_t_ = torch.tensor(grid2d_).type(config.torch_type)
                z_range = np.linspace(-1,1,n3_eval)*config.size_z_vol
                V_icetide = np.zeros_like(V)
                for zz, zval in enumerate(z_range):
                    grid3d = np.concatenate([grid2d_t_, zval*torch.ones((grid2d_t_.shape[0],1))],1)
                    grid3d_slice = torch.tensor(grid3d).type(config.torch_type).to(device)
                    estSlice = impl_volume(grid3d_slice/size_max_vol/2+0.5).detach().cpu().numpy().reshape(n1_eval,n2_eval)
                    V_icetide[:,:,zz] = estSlice
                V_icetide = V_icetide[:,:,::-1]
                if config.avg_XYZ>1:
                    padded_array = np.pad(V_icetide, ((0, 0), (0, 0), (0, config.avg_XYZ - 1)), mode='constant')
                    filt = np.zeros_like(padded_array)
                    filt[:,:,filt.shape[2]//2-config.avg_XYZ//2:filt.shape[2]//2+config.avg_XYZ//2] = 1/config.avg_XYZ
                    V_icetide = np.fft.fftshift(np.fft.ifft((np.fft.fft(filt) * np.fft.fft(padded_array))).real,axes=-1)[:,:,:n3_eval]
                V_icetide_t = torch.tensor(V_icetide).type(config.torch_type).to(device)
                fsc_icetide = utils_FSC.FSC(V,V_icetide)
                CC_icetide = compare_results.CC(V,V_icetide)
                indeces = np.where(fsc_icetide<0.5)[0]
                choosenIndex = np.where(indeces>2)[0][0]
                resolution_icetide = indeces[choosenIndex]
                resolution_icetide_tot.append(resolution_icetide)
                CC_icetide_tot.append(CC_icetide)
                ep_tot.append(ep)
                np.save(os.path.join(config.path_save,'training','resolution05_iter.npy'),np.array(resolution_icetide_tot))
                header ='ep,icetide,FBP,FBP_no_deformed'
                np.savetxt(os.path.join(config.path_save,'training','resolution05_iter.csv'),np.array([ep_tot,resolution_icetide_tot,resolution_FBP_tot,resolution_FBP_no_deformed_tot]).T,header=header,delimiter=",",comments='')
                np.save(os.path.join(config.path_save,'training','CC_iter.npy'),np.array(CC_icetide_tot))
                header ='ep,icetide,FBP,FBP_no_deformed'
                np.savetxt(os.path.join(config.path_save,'training','CC_iter.csv'),np.array([ep_tot,CC_icetide_tot,CC_FBP_tot,CC_FBP_no_deformed_tot]).T,header=header,delimiter=",",comments='')


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
        plt.close('all')

    print("Saving final state after training...")
    torch.save({
        'shift_est': shift_est,
        'rot_est': rot_est,
        'local_deformation_network': implicit_deformation_list,
        'implicit_volume': impl_volume.state_dict(),
        'optimizer_volume' : optimizer_volume,
        'optimizer_deformations_glob' : optimizer_deformations_glob,
        'optimizer_deformations_loc' : optimizer_deformations_loc,
        'scheduler_volume': scheduler_volume, 
        'scheduler_deformation_glob': scheduler_deformation_glob, 
        'scheduler_deformation_loc': scheduler_deformation_loc,
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
        z_range = np.linspace(-1,1,config.n3_patch)*config.size_z_vol
        V_icetide = np.zeros((config.n1_patch,config.n2_patch,config.n3_patch))
        for zz, zval in enumerate(z_range):
            grid3d = np.concatenate([grid2d_t, zval*torch.ones((grid2d_t.shape[0],1))],1)
            grid3d_slice = torch.tensor(grid3d).type(config.torch_type).to(device)
            estSlice = impl_volume(grid3d_slice/size_max_vol/2+0.5).detach().cpu().numpy().reshape(config.n1_patch,config.n2_patch)
            pp = (estSlice)*1.
            V_icetide[:,:,zz] = estSlice
        V_icetide = V_icetide[:,:,::-1]
        if config.avg_XYZ>1:
            padded_array = np.pad(V_icetide, ((0, 0), (0, 0), (0, config.avg_XYZ - 1)), mode='constant')
            filt = np.zeros_like(padded_array)
            filt[:,:,filt.shape[2]//2-config.avg_XYZ//2:filt.shape[2]//2+config.avg_XYZ//2] = 1/config.avg_XYZ
            V_icetide = np.fft.fftshift(np.fft.ifft((np.fft.fft(filt) * np.fft.fft(padded_array))).real,axes=-1)[:,:,:n3_eval]
        V_icetide_t = torch.tensor(V_icetide).type(config.torch_type).to(device)
        out = mrcfile.new(config.path_save+"/training/V_est_final.mrc",np.moveaxis(V_icetide.astype(np.float32),2,0),overwrite=True)
        out.close() 

    loss_tot_avg = np.array(loss_tot).reshape(config.Nangles//config.batch_size,-1).mean(0)
    step = (loss_tot_avg.max()-loss_tot_avg.min())*0.02
    plt.figure(figsize=(10,10))
    plt.plot(loss_tot_avg[10:])
    plt.xticks(np.arange(0, len(loss_tot_avg[1:]), 100))
    plt.yticks(np.linspace(loss_tot_avg.min()-step,loss_tot_avg.max()+step, 14))
    plt.grid()
    plt.savefig(os.path.join(config.path_save,'training','loss.png'))
    plt.savefig(os.path.join(config.path_save,'training','loss.pdf'))
    print("Training is over.")

    if config.compute_fsc:
        plt.figure(figsize=(10,10))
        plt.plot(ep_tot,resolution_icetide_tot,label='ICETIDE')
        plt.plot(ep_tot,resolution_FBP_tot,label='FBP')
        plt.plot(ep_tot,resolution_FBP_no_deformed_tot,label='FBP no deform')
        plt.legend()
        plt.xticks(ep_tot)
        plt.ylabel('Resolution (1/pixel size)')
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(config.path_save,'training','resolution05_iter.png'))
        plt.savefig(os.path.join(config.path_save,'training','resolution05_iter.pdf'))














def train_without_ground_truth(config):
    print("Runing training procedure.")
    # Choosing the seed and the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count()>1:
        torch.cuda.set_device(config.device_num)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # prepare the folders    
    if not os.path.exists(config.path_save+"training/"):
        os.makedirs(config.path_save+"training/")
    if not os.path.exists(config.path_save+"training/volume/"):
        os.makedirs(config.path_save+"training/volume/")
    if not os.path.exists(config.path_save+"training/deformations/"):
        os.makedirs(config.path_save+"training/deformations/")
    if not os.path.exists(config.path_save+"training/deformations_x10/"):
        os.makedirs(config.path_save+"training/deformations_x10/")

    ## Load data that was previously saved
    data = np.load(config.path_save_data+"volume_and_projections.npz")
    if config.projections_raw:
        projections_noisy = torch.Tensor(np.float32(mrcfile.open(os.path.join(config.path_load,config.volume_name+".mrc"),permissive=True).data)).type(config.torch_type).to(device)
    else:
        projections_noisy = torch.Tensor(data['projections']).type(config.torch_type).to(device)
    config.Nangles = projections_noisy.shape[0]
    projections_noisy = projections_noisy/torch.abs(projections_noisy).max() # make sure that values to predict are between -1 and 1

    ######################################################################################################
    ######################################################################################################
    ##
    ## TRAINING
    ##
    ######################################################################################################
    ######################################################################################################
    print("Loading models and setting parameters...")
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

    ######################################################################################################
    ## Define the implicit deformations
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
        depl_ctr_pts_net = torch.zeros((2,config.local_deformation.N_ctrl_pts_net,config.local_deformation.N_ctrl_pts_net)).to(device).type(config.torch_type)/max([config.n1,config.n2,config.n3])/10
        implicit_deformation_list = []
        for k in range(config.Nangles):
            field = utils_deformation.deformation_field(depl_ctr_pts_net.clone(),maskBoundary=config.maskBoundary)
            implicit_deformation_list.append(field)
        num_param = sum(p.numel() for p in implicit_deformation_list[0].parameters() if p.requires_grad)
        print('---> Number of trainable parameters in implicit net: {}'.format(num_param))

    ######################################################################################################
    ## Define the global deformations
    fixedAngle = torch.FloatTensor([config.fixed_angle* np.pi/180]).to(device)[0]

    shift_est = []
    rot_est = []
    fixed_rot =[ ]
    for k in range(config.Nangles):
        shift_est.append(utils_deformation.shiftNet(1).to(device))
        rot_est.append(utils_deformation.rotNet(1).to(device))
        fixed_rot.append(utils_deformation.rotNet(1,x0=fixedAngle).to(device))

    ######################################################################################################
    ## Optimizer
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
        optimizer_deformations_glob = torch.optim.Adam(list_params_deformations_glob, weight_decay=config.wd)
        scheduler_deformation_glob = torch.optim.lr_scheduler.StepLR(
            optimizer_deformations_glob, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
    if len(list_params_deformations_loc)!=0:
        optimizer_deformations_loc = torch.optim.Adam(list_params_deformations_loc, weight_decay=config.wd)
        scheduler_deformation_loc = torch.optim.lr_scheduler.StepLR(
            optimizer_deformations_loc, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
    scheduler_volume = torch.optim.lr_scheduler.StepLR(optimizer_volume, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)

    ######################################################################################################
    # Format data for batch training
    index = torch.arange(0, config.Nangles, dtype=torch.long) # index for the dataloader
    # Define dataset
    angles = np.linspace(config.view_angle_min,config.view_angle_max,config.Nangles)
    angles_t = torch.tensor(angles).type(config.torch_type).to(device)
    dataset = TensorDataset(angles_t,projections_noisy.detach(),index)
    trainLoader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    weights_tilt = torch.cos(angles_t/180*np.pi).to(device)

    # ######################################################################################################
    # ## Track sampling
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

    ## grid for display 
    x_lin1 = np.linspace(-1,1,config.n1_patch)
    x_lin2 = np.linspace(-1,1,config.n2_patch)
    XX, YY = np.meshgrid(x_lin1,x_lin2,indexing='ij')
    grid2d = np.concatenate([XX.reshape(-1,1),YY.reshape(-1,1)],1)
    grid2d_t = torch.tensor(grid2d).type(config.torch_type)

    # Define geometry of sampling
    size_xy_vol, z_max_value = get_sampling_geometry(config.size_z_vol, config.view_angle_min, config.view_angle_max, config.sampling_domain_lx, config.sampling_domain_ly)
    size_max_vol = 1.2*np.max([size_xy_vol,config.size_z_vol]) # increase by some small factor to account for deformations

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
    t0 = time.time()
    print("Training the network(s)...")
    for ep in range(config.epochs):
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
            fixedRotSet = list(map(fixed_rot.__getitem__, idx_loader))

            # Define the detector locations
            detectorLocations = torch.rand(proj.shape[0],config.nRays,2).to(device)*2-1

            # Apply deformations in the 2D space
            detectorLocationsDeformed = apply_deformations_to_locations(detectorLocations,rot_deformSet,
                                                                    shift_deformSet,local_deformSet,fixedRotSet,scale=config.deformationScale)

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
                    loss += config.lamb_rot*torch.abs(rot_est[ii]()*180/np.pi).mean()
                    loss_regul_shifts.append((config.lamb_shifts*torch.abs(shift_est[ii]()*config.n1).mean()).item())
                    loss_regul_rot.append((config.lamb_rot*torch.abs(rot_est[ii]()*180/np.pi).mean()).item())
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
            loss_tot.append(loss.item())

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
                    ep,loss_current_epoch,l_fid,l_v,l_sh,l_rot,l_loc,time.time()-t0))
                
                print('Running and saving tests')  

                ## Save local deformation
                utils_display.display_local_movie(implicit_deformation_list,field_true=None,Npts=(20,20),
                                            img_path=config.path_save+"/training/deformations/local_deformations_",img_type='.png',
                                            scale=1,alpha=0.8,width=0.0015,weights_est=1)
                for index in range(len(implicit_deformation_list)):
                    utils_display.display_local_movie(implicit_deformation_list,field_true=None,Npts=(20,20),
                                                img_path=config.path_save+"/training/deformations_x10/local_deformations_",img_type='.png',
                                                scale=0.1,alpha=0.8,width=0.0015,weights_est=1)
                    
                ## Save global deformation
                shiftEstimate, rotEstimate = globalDeformationValues(shift_est,rot_est)
                plt.figure(1)
                plt.clf()
                plt.hist(shiftEstimate.reshape(-1)*config.n1,alpha=1)
                plt.legend(['est.'])
                plt.savefig(os.path.join(config.path_save+"/training/deformations/shifts.png"))

                plt.figure(1)
                plt.clf()
                plt.hist(rotEstimate*180/np.pi,15)
                plt.legend(['est.'])
                plt.title('Angles in degrees')
                plt.savefig(os.path.join(config.path_save+"/training/deformations/rotations.png"))

                plt.figure(1)
                plt.clf()
                plt.plot(gains.detach().cpu().numpy())
                plt.savefig(os.path.join(config.path_save+"/training/deformations/gains.png"))
                                    
                if config.save_volume:
                    ## Save slice of the volume
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
                        

                torch.save({
                    'shift_est': shift_est,
                    'rot_est': rot_est,
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
                plt.plot(loss_tot_avg[10:])
                plt.xticks(np.arange(0, len(loss_tot_avg[1:]), 1+len(loss_tot_avg[1:])//10))
                plt.yticks(np.linspace(loss_tot_avg.min()-step,loss_tot_avg.max()+step, 14))
                # plt.grid()
                plt.savefig(os.path.join(config.path_save,'training','loss.pdf'))
        plt.close('all')

    print("Saving final state after training...")
    torch.save({
        'shift_est': shift_est,
        'rot_est': rot_est,
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
