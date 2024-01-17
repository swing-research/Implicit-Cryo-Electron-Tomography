"""
Module to train the reconstruction network on the simulated data.
"""

import os
import time
import torch
import mrcfile
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from utils import utils_deformation, utils_display
from utils.utils_sampling import sample_implicit_batch_lowComp, generate_rays_batch_bilinear


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
    # Some processing
    rays_scaling = torch.tensor(np.array(config.rays_scaling))[None,None,None].type(config.torch_type).to(device)
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
            field = utils_deformation.deformation_field(depl_ctr_pts_net.clone(),maskBoundary=2)
            implicit_deformation_list.append(field)
        num_param = sum(p.numel() for p in implicit_deformation_list[0].parameters() if p.requires_grad)
        print('---> Number of trainable parameters in implicit net: {}'.format(num_param))

    ######################################################################################################
    ## Define the global deformations
    shift_est = []
    rot_est = []
    for k in range(config.Nangles):
        shift_est.append(utils_deformation.shiftNet(1).to(device))
        rot_est.append(utils_deformation.rotNet(1).to(device))

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
    ## Track sampling
    choosenLocations_all = {}
    for ii, _ in enumerate(angles):
        choosenLocations_all[ii] = []
    current_sampling = np.ones_like(projections_noisy.detach().cpu().numpy())

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
    x_lin1 = np.linspace(-1,1,config.n1)*rays_scaling[0,0,0,0].item()/2+0.5
    x_lin2 = np.linspace(-1,1,config.n2)*rays_scaling[0,0,0,1].item()/2+0.5
    XX, YY = np.meshgrid(x_lin1,x_lin2,indexing='ij')
    grid2d = np.concatenate([XX.reshape(-1,1),YY.reshape(-1,1)],1)
    grid2d_t = torch.tensor(grid2d).type(config.torch_type)

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
        fsc_icetide_tot = []
        resolution_icetide_tot = []
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
        for   angle,proj, idx_loader  in trainLoader:
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

            ## Sample the rays
            raysSet,raysRot, isOutsideSet, pixelValues = generate_rays_batch_bilinear(proj,angle,config.nRays,config.ray_length,
                                                                                                randomZ=2,zmax=config.z_max,
                                                                                                choosenLocations_all=choosenLocations_all,
                                                                                                density_sampling=None,idx_loader=idx_loader)

            # Compute the projections
            raysSet = raysSet*rays_scaling
            outputValues,support = sample_implicit_batch_lowComp(impl_volume,raysSet,angle,
                rot_deformSet=rot_deformSet,shift_deformSet=shift_deformSet,local_deformSet=local_deformSet,
                scale=1,grid_positive=config.grid_positive,zlimit=config.n3/max(config.n1,config.n2))
            outputValues = outputValues.type(config.torch_type)
            support = support.reshape(outputValues.shape[0],outputValues.shape[1],-1)
            projEstimate = torch.sum(support*outputValues,2)/config.n3

            # Take the datafidelity loss
            loss = loss_data(projEstimate,pixelValues.to(projEstimate.dtype))
            loss_data_fidelity.append(loss.item())

            # update sampling
            with torch.no_grad():
                for ii_ in idx_loader:
                    ii = ii_.item()
                    idx = np.floor((choosenLocations_all[ii][-1]+1)/2*max(config.n1,config.n2)).astype(np.int)
                    current_sampling[ii,idx[:,0],idx[:,1]] += 1

            ## Add regularizations
            if train_local_def and config.lamb_local_ampl!=0:
                # Using only the x and y coordinates
                for ii_ in idx_loader:
                    depl = torch.abs(implicit_deformation_list[ii](raysSet[:,:,0,:2].reshape(-1,2))*config.n1)
                    depl_mean = torch.abs(torch.mean(implicit_deformation_list[ii](raysSet[:,:,0,:2].reshape(-1,2))*config.n1))
                    loss += (config.lamb_local_ampl*depl.mean()+config.lamb_local_mean*depl_mean)
                    loss_regul_local_ampl.append((config.lamb_local_ampl*depl.mean()+config.lamb_local_mean*depl_mean).item())
            if train_global_def and (config.lamb_rot!=0 or config.lamb_shifts!=0):
                for ii in idx_loader:
                    loss += config.lamb_shifts*torch.abs(shift_est[ii]()*config.n1).sum()
                    loss += config.lamb_rot*torch.abs(rot_est[ii]()*180/np.pi).sum()
                    loss_regul_shifts.append((config.lamb_shifts*torch.abs(shift_est[ii]()*config.n1).sum()).item())
                    loss_regul_rot.append((config.lamb_rot*torch.abs(rot_est[ii]()*180/np.pi).sum()).item())
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
            print("Epoch: {}, loss_avg: {:2.3f} || Loss data fidelity: {:2.3f}, regul volume: {:2.2f}, regul shifts: {:2.2f}, regul inplane: {:2.2f}, regul local: {:2.2f}, time: {:2.0f} s".format(
                ep,loss_current_epoch,l_fid,l_v,l_sh,l_rot,l_loc,time.time()-t0))
        if config.track_memory:
            memory_used.append(torch.cuda.memory_allocated())

        # Save and display some results
        if (ep%config.Ntest==0) and check_point_training:
            print('Running and saving tests')

            with torch.no_grad():
                ## Avergae loss over until the last test
                loss_current_epoch = np.mean(loss_tot[-len(trainLoader)*config.Ntest:])
                l_fid = np.mean(loss_data_fidelity[-len(trainLoader)*config.Ntest:])
                l_v = np.mean(loss_regul_volume[-len(trainLoader)*config.Ntest:])
                l_sh = np.mean(loss_regul_shifts[-len(trainLoader)*config.Ntest:])
                l_rot = np.mean(loss_regul_rot[-len(trainLoader)*config.Ntest:])
                l_loc = np.mean(loss_regul_local_ampl[-len(trainLoader)*config.Ntest:])
                print("----Epoch: {}, loss_avg: {:2.3f} || Loss data fidelity: {:2.3f}, regul volume: {:2.2f}, regul shifts: {:2.4f}, regul inplane: {:2.2f}, regul local: {:2.4f}, time: {:2.0f} s".format(
                    ep,loss_current_epoch,l_fid,l_v,l_sh,l_rot,l_loc,time.time()-t0))

                ## Save local deformation
                utils_display.display_local_movie(implicit_deformation_list,field_true=local_tr,Npts=(20,20),
                                            img_path=config.path_save+"/training/deformations/local_deformations_",img_type='.png',
                                            scale=1,alpha=0.8,width=0.0015,weights_est=1)
                for index in range(len(implicit_deformation_list)):
                    utils_display.display_local_est_and_true(implicit_deformation_list[index],local_tr[index],Npts=(20,20),scale=0.1,
                                                img_path=config.path_save+"/training/deformations_x10/local_deformations_"+str(index),img_type='.png')

                    
                ## Save global deformation
                shiftEstimate, rotEstimate = globalDeformationValues(shift_est,rot_est)
                plt.figure(1)
                plt.clf()
                plt.hist(shiftEstimate.reshape(-1)*config.n1,alpha=1)
                plt.hist(shift_true.reshape(-1)*config.n1,alpha=0.5)
                plt.legend(['est.','true'])
                plt.savefig(os.path.join(config.path_save+"/training/deformations/shitfs.png"))

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
                
                ## Save slice of the volume
                z_range = np.linspace(-1,1,config.n3//5)*rays_scaling[0,0,0,2].item()*(config.n3/config.n1)/2+0.5
                for zz, zval in enumerate(z_range):
                    grid3d = np.concatenate([grid2d_t, zval*torch.ones((grid2d_t.shape[0],1))],1)
                    grid3d_slice = torch.tensor(grid3d).type(config.torch_type).to(device)
                    estSlice = impl_volume(grid3d_slice).detach().cpu().numpy().reshape(config.n1,config.n2)
                    pp = (estSlice)*1.
                    plt.figure(1)
                    plt.clf()
                    plt.imshow(pp,cmap='gray')
                    plt.savefig(os.path.join(config.path_save+"/training/volume/volume_est_slice_{}.png".format(zz)))
                                    
                if config.save_volume:
                    z_range = np.linspace(-1,1,config.n3)*rays_scaling[0,0,0,2].item()*(config.n3/config.n1)/2+0.5
                    V_ours = np.zeros((config.n1,config.n2,config.n3))
                    for zz, zval in enumerate(z_range):
                        grid3d = np.concatenate([grid2d_t, zval*torch.ones((grid2d_t.shape[0],1))],1)
                        grid3d_slice = torch.tensor(grid3d).type(config.torch_type).to(device)
                        estSlice = impl_volume(grid3d_slice).detach().cpu().numpy().reshape(config.n1,config.n2)
                        V_ours[:,:,zz] = estSlice
                    out = mrcfile.new(config.path_save+"/training/V_est"+".mrc",np.moveaxis(V_ours.astype(np.float32),2,0),overwrite=True)
                    out.close() 
                            
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
            from utils import utils_FSC 
            with torch.no_grad():
                V = np.moveaxis(np.double(mrcfile.open(config.path_save_data+"V.mrc").data),0,2)
                ## Compute our model at same resolution than other volume
                rays_scaling = torch.tensor(np.array(config.rays_scaling))[None,None,None].type(config.torch_type).to(device)
                n1_eval, n2_eval, n3_eval = V.shape
                # Compute estimated volumex
                x_lin1 = np.linspace(-1,1,n1_eval)*rays_scaling[0,0,0,0].item()/2+0.5
                x_lin2 = np.linspace(-1,1,n2_eval)*rays_scaling[0,0,0,1].item()/2+0.5
                XX, YY = np.meshgrid(x_lin1,x_lin2,indexing='ij')
                grid2d = np.concatenate([XX.reshape(-1,1),YY.reshape(-1,1)],1)
                grid2d_t = torch.tensor(grid2d).type(config.torch_type)
                z_range = np.linspace(-1,1,n3_eval)*rays_scaling[0,0,0,2].item()*(n3_eval/n1_eval)/2+0.5
                V_icetide = np.zeros_like(V)
                for zz, zval in enumerate(z_range):
                    grid3d = np.concatenate([grid2d_t, zval*torch.ones((grid2d_t.shape[0],1))],1)
                    grid3d_slice = torch.tensor(grid3d).type(config.torch_type).to(device)
                    estSlice = impl_volume(grid3d_slice).detach().cpu().numpy().reshape(config.n1,config.n2)
                    V_icetide[:,:,zz] = estSlice
                fsc_icetide = utils_FSC.FSC(V,V_icetide)
                fsc_icetide_tot.append(fsc_icetide)
                import ipdb; ipdb.set_trace()
                # x_fsc = np.arange(fsc_icetide.shape[0])
                indeces = np.where(fsc<0.5)[0]
                choosenIndex = np.where(indeces>2)[0][0]
                resolution_icetide = indeces[choosenIndex]
                resolution_icetide_tot.append(resolution_icetide)
                np.save(os.path.join(config.path_save,'training','fsc_iter.npy'),np.array(resolution_icetide_tot))

                header ='x,icetide,FBP,FBP_no_deformed,AreTomo_patch0,ETOMO,FBP_est_deformed,AreTomo_patch1'
                np.savetxt(os.path.join(config.path_save,'evaluation','FSC.csv'),fsc_arr,header=header,delimiter=",",comments='')





                # # TODO: save and load fsc along iteration, in npy and txt and check that it works
                # dat = np.load(os.path.join(config.path_save,'training','fsc_iter.npy'))
                # fsc_iter = dat['fsc_iter']

            


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
        np.savetxt(os.path.join(config.path_save,'training','memory_used.txt'),np.array([memory_used.max()/ (1024**3),max_memory_allocated_gb])) # Conversion in Gb
    np.save(os.path.join(config.path_save,'training','training_time.npy'),training_time)
    np.savetxt(os.path.join(config.path_save,'training','training_time.txt'),np.array([training_time]))

    with torch.no_grad():
        z_range = np.linspace(-1,1,config.n3)*rays_scaling[0,0,0,2].item()*(config.n3/config.n1)/2+0.5
        V_ours = np.zeros((config.n1,config.n2,config.n3))
        for zz, zval in enumerate(z_range):
            grid3d = np.concatenate([grid2d_t, zval*torch.ones((grid2d_t.shape[0],1))],1)
            grid3d_slice = torch.tensor(grid3d).type(config.torch_type).to(device)
            estSlice = impl_volume(grid3d_slice).detach().cpu().numpy().reshape(config.n1,config.n2)
            V_ours[:,:,zz] = estSlice
        out = mrcfile.new(config.path_save+"/training/V_est_final.mrc",np.moveaxis(V_ours.astype(np.float32),2,0),overwrite=True)
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