import os
import time
import torch
import imageio
import mrcfile
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from skimage.transform import resize, radon
from torch.utils.data import DataLoader, TensorDataset

from utils import utils_deformation, utils_display
from utils.utils_sampling import (get_sampling_geometry, apply_deformations_to_locations,
                                  generate_rays_batch, sample_projections_from_location, sample_volume,
                                  sample_projection_from_implicit_net, aligned_projections)
from utils.utils_data_processing import (load_projections, load_angles, select_volume_neural_network,
                                         select_local_deformation_neural_network)

def train_without_ground_truth(config):
    ######################################################################################################
    ## Setting the environment
    ######################################################################################################
    if config.debug:
        import cProfile, pstats, io
        from pstats import SortKey
        pr = cProfile.Profile()
        pr.enable()

    print("Runing training procedure.")
    # Choosing the seed and the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device
    if torch.cuda.device_count()>1:
        torch.cuda.set_device(config.device_num)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    print("Device: {}".format(device))

    # prepare the folders
    if not os.path.exists(os.path.join(config.path_save,"training")):
        os.makedirs(os.path.join(config.path_save,"training"))
    if not os.path.exists(os.path.join(config.path_save,"training","deformations")):
        os.makedirs(os.path.join(config.path_save,"training","deformations"))
    if not os.path.exists(os.path.join(config.path_save,"training","deformations_x10")):
        os.makedirs(os.path.join(config.path_save,"training","deformations_x10"))

    ######################################################################################################
    ## Loading the data
    ######################################################################################################
    # Load the data provided by the user
    print("Loading the tilt-series")
    projections_noisy, name_file = load_projections(config)
    (angles, Nangles_origin, n1_origin, n2_origin,
     view_angle_min, view_angle_max) = load_angles(config, projections_noisy)

    ######################################################################################################
    ## Define neural network architectures for the volume
    ######################################################################################################
    print("Defining volume neural network")
    impl_volume = select_volume_neural_network(config)

    ######################################################################################################
    ## Define neural network architectures for local deformations
    ######################################################################################################
    print("Defining local deformation neural network")
    implicit_deformation_list = select_local_deformation_neural_network(config)

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
    angles_t = torch.tensor(angles).type(config.torch_type).to(device)
    dataset = TensorDataset(angles_t,projections_noisy.detach(),index)
    trainLoader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    weights_tilt = torch.cos(angles_t/180*np.pi).to(device)

    # Define geometry of sampling
    size_xy_vol, z_max_value = get_sampling_geometry(config.size_z_vol, view_angle_min, view_angle_max, config.sampling_domain_lx, config.sampling_domain_ly)
    size_max_vol = 1.2*np.max([size_xy_vol,config.size_z_vol]) # increase by some small factor to account for deformations

    if config.debug:
        print("Elapsed time: {}".format(time.time() - t0))

    if config.multiresolution:
        with torch.no_grad():
            print("Computing multiresolution volume")
            t0 = time.time()
            res_factor = config.multires_params.startResolution
            Nangles, n1_origin, n2_origin = projections_noisy.shape
            n1_resize = n1_origin // (2 ** res_factor)
            n2_resize = n2_origin // (2 ** res_factor)
            projections_noisy_resized = resize(projections_noisy.detach().cpu().numpy(),
                                               (Nangles, n1_resize, n2_resize))
            projections_noisy_resized_t = torch.tensor(projections_noisy_resized).type(config.torch_type).to(device)
            if config.debug:
                print("Elapsed time: {}".format(time.time() - t0))

        index = torch.arange(0, config.Nangles, dtype=torch.long) # index for the dataloader
        batch_set =  config.multires_params.batch_set
        dataset = TensorDataset(angles_t,projections_noisy_resized_t.detach(),index)
        print('New resolution: ', projections_noisy_resized_t.shape)
        trainLoader = DataLoader(dataset, batch_size = batch_set[0], shuffle=True, drop_last=True)
        batch_set_index = 0
        ray_change_epoch = config.multires_params.ray_change_epoch
        if config.debug:
            # Save current projection used
            print("Saving current tilt-series used with resolution: ", projections_noisy_resized_t.shape)
            path_ = os.path.join(config.path_save, "projections", "multires_input_" + str(projections_noisy_resized_t.shape[1]))
            if not os.path.exists(path_):
                os.makedirs(path_)
            for ll in range(projections_noisy_resized_t.shape[0]):
                tmp = projections_noisy_resized_t[ll].detach().cpu().numpy()
                tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
                tmp = np.floor(255 * tmp).astype(np.uint8)
                imageio.imwrite( os.path.join(path_,"est_"+str(ll).zfill(5)+".png"), tmp)
            print("Elapsed time: ", time.time()-t0)

    ######################################################################################################
    ## Iterative training
    ######################################################################################################
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

    print("Training the network(s)...")
    t0_train = time.time()
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

        if config.multiresolution:
            if (ep in ray_change_epoch):
                with torch.no_grad():
                    print("Updating the resolution of tilt-series")
                    t0 = time.time()

                    res_factor -= 1
                    _, n1_origin, n2_origin = projections_noisy.shape
                    n1_resize = n1_origin // (2 ** res_factor)
                    n2_resize = n2_origin // (2 ** res_factor)
                    projections_noisy_resized = resize(projections_noisy.detach().cpu().numpy(),
                                                       (Nangles, n1_resize, n2_resize))
                    projections_noisy_resized_t = torch.tensor(projections_noisy_resized).type(config.torch_type).to(
                        device)
                    if config.debug:
                        print("Elapsed time: ", time.time() - t0)

                batch_set_index = min(len(batch_set) - 1, batch_set_index + 1)
                index = torch.arange(0, config.Nangles, dtype=torch.long)  # index for the dataloader
                print('New resolution: ', projections_noisy_resized_t.shape)
                dataset = TensorDataset(angles_t, projections_noisy_resized_t.detach(), index)
                trainLoader = DataLoader(dataset, batch_size=batch_set[batch_set_index], shuffle=True, drop_last=True)

                if config.debug:
                    # Save current projection used
                    print("Saving current tilt-series used with resolution: ", projections_noisy_resized_t.shape)
                    t0 = time.time()
                    path_ = os.path.join(config.path_save, "training", "projections",
                                         "multires_input_" + str(projections_noisy_resized_t.shape[1]))
                    if not os.path.exists(path_):
                        os.makedirs(path_)
                    for ll in range(projections_noisy_resized_t.shape[0]):
                        tmp = projections_noisy_resized_t[ll].detach().cpu().numpy()
                        tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
                        tmp = np.floor(255*tmp).astype(np.uint8)
                        imageio.imwrite(os.path.join(path_, "est_"+str(ll).zfill(5)+".png"),tmp)
                    print("Elapsed time: ", time.time() - t0)

        for angle, proj, idx_loader  in trainLoader:
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
            # Choosing the right subset of the parameters
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

            # Define the detector locations
            detectorLocations = torch.rand(proj.shape[0],config.nRays,2).to(device)*2-1
            # Apply deformations in the 2D space
            detectorLocationsDeformed = apply_deformations_to_locations(detectorLocations,rot_deformSet,
                                                                    shift_deformSet,local_deformSet,fixed_rot,
                                                                    scale=config.deformationScale, cl=config.clip)
            # generate the rays in 3D
            rays_rotated = generate_rays_batch(detectorLocationsDeformed, angle, z_max_value, config.ray_length, std_noise=config.std_noise_z)
            # Scale the final rays so that they are always in [-1,1]
            rays_rotated_scaled = rays_rotated/size_max_vol
            # Sample the implicit volume by converting the input from [-1,1] to [0,1]
            outputValues = impl_volume((rays_rotated_scaled/2+0.5).reshape(-1,3)).reshape(proj.shape[0],config.nRays,config.ray_length)
            # Remove values that are outside of the admissible box. Can happen as we learn deformations
            support = (rays_rotated[:,:,:,2].abs()<config.size_z_vol)*1
            projEstimate = torch.sum(support*outputValues,2)/config.ray_length
            pixelValues = sample_projections_from_location(proj, detectorLocations, interp='bilinear')
            # Take the datafidelity loss
            loss = loss_data(projEstimate*gains[idx_loader,None]*weights_tilt[idx_loader,None],pixelValues.to(projEstimate.dtype)*weights_tilt[idx_loader,None])
            loss_data_fidelity.append(loss.item())

            ## Add regularizations
            if train_local_def and config.lamb_local_ampl!=0:
                # For current tilts, penalize 1/n \sum^n |def_loc(detectorLocations)|
                for ii_ in idx_loader:
                    depl = torch.abs(implicit_deformation_list[ii_](detectorLocations.reshape(-1,2))*n1_origin)
                    depl_mean = torch.abs(torch.mean(implicit_deformation_list[ii_](detectorLocations.reshape(-1,2))*n1_origin))
                    loss += (config.lamb_local_ampl*depl.mean()+config.lamb_local_mean*depl_mean)
                    loss_regul_local_ampl.append((config.lamb_local_ampl*depl.mean()+config.lamb_local_mean*depl_mean).item())
            if train_global_def and (config.lamb_rot!=0 or config.lamb_shifts!=0):
                # For current tilts, penalize 1/I \sum^I |shift(i)| + |rot(i)|
                for ii in idx_loader:
                    loss += config.lamb_shifts*torch.abs(shift_est[ii]()*n1_origin).mean()
                    loss += config.lamb_rot*torch.abs(rot_est[ii].thetas*180/np.pi).mean()
                    loss_regul_shifts.append((config.lamb_shifts*torch.abs(shift_est[ii]()*n1_origin).mean()).item())
                    loss_regul_rot.append((config.lamb_rot*torch.abs(rot_est[ii].thetas*180/np.pi).mean()).item())
            if config.train_volume and config.lamb_volume!=0:
                # Penalize negative values in the volume estimation
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

        loss_tot.append(np.mean(loss_tmp))
        if train_volume:
            scheduler_volume.step()
        if len(list_params_deformations_glob)!=0 and train_global_def:
            scheduler_deformation_glob.step()
        if len(list_params_deformations_loc)!=0 and train_local_def:
            scheduler_deformation_loc.step()

        shiftEstimate, rotEstimate = utils_deformation.globalDeformationValues(shift_est,rot_est)
        shift_estimates.append(shiftEstimate)
        rot_estimates.append(rotEstimate)
        
        # Track loss and display values
        if ((ep%10)==0 and (ep%config.Ntest!=0)):
            loss_current_epoch = np.mean(loss_tot[-len(trainLoader):])
            l_fid = np.mean(loss_data_fidelity[-len(trainLoader):])
            if config.train_volume and config.lamb_volume != 0:
                l_v = np.mean(loss_regul_volume[-len(trainLoader):])
            else:
                l_v = 0
            if train_global_def and (config.lamb_rot != 0 or config.lamb_shifts != 0):
                l_sh = np.mean(loss_regul_shifts[-len(trainLoader)*trainLoader.batch_size:])
                l_rot = np.mean(loss_regul_rot[-len(trainLoader)*trainLoader.batch_size:])
            else:
                l_sh = 0
                l_rot = 0
            if config.train_volume and config.lamb_volume != 0:
                l_loc = np.mean(loss_regul_local_ampl[-len(trainLoader)*trainLoader.batch_size:])
            else:
                l_loc = 0
            print("Epoch: {}, loss_avg: {:.3e} || Loss data fidelity: {:.3e}, regul volume: {:.2e}, regul shifts: {:.2e}, regul inplane: {:.2e}, regul local: {:.2e}, time: {:2.0f} s".format(
                ep,loss_current_epoch,l_fid,l_v,l_sh,l_rot,l_loc,time.time()-t0_train))
        if config.track_memory:
            memory_used.append(torch.cuda.memory_allocated())

        # Save and display some results
        if (ep%config.Ntest==0) and check_point_training:
            with torch.no_grad():
                ## Avergae loss over until the last test
                loss_current_epoch = np.mean(loss_tot[-len(trainLoader)*config.Ntest:])
                l_fid = np.mean(loss_data_fidelity[-len(trainLoader)*config.Ntest:])
                if config.train_volume and config.lamb_volume != 0:
                    l_v = np.mean(loss_regul_volume[-len(trainLoader)*config.Ntest:])
                else:
                    l_v = 0
                if train_global_def and (config.lamb_rot != 0 or config.lamb_shifts != 0):
                    l_sh = np.mean(loss_regul_shifts[-len(trainLoader)*config.Ntest:])
                    l_rot = np.mean(loss_regul_rot[-len(trainLoader)*config.Ntest:])
                else:
                    l_sh = 0
                    l_rot = 0
                if config.train_volume and config.lamb_volume != 0:
                    l_loc = np.mean(loss_regul_local_ampl[-len(trainLoader)*config.Ntest:])
                else:
                    l_loc = 0
                print("----Epoch: {}, loss_avg: {:.3e} || Loss data fidelity: {:.3e}, regul volume: {:.2e}, regul shifts: {:2.4f}, regul inplane: {:.2e}, regul local: {:2.4f}, time: {:2.0f} s".format(
                    ep,loss_current_epoch,l_fid,l_v,l_sh,l_rot,l_loc,time.time()-t0_train))

                print("Saving the model and parameters")
                if len(list_params_deformations_loc)!=0:
                    opt_loc = optimizer_deformations_loc.state_dict()
                    sch_loc = scheduler_deformation_loc.state_dict()
                else:
                    opt_loc = []
                    sch_loc = []
                torch.save({
                    'shift_est': shift_est,
                    'rot_est': rot_est,
                    'fixed_rot_est': fixed_rot,
                    'gains': gains,
                    'local_deformation_network': implicit_deformation_list,
                    'implicit_volume': impl_volume.state_dict(),
                    'optimizer_volume': optimizer_volume.state_dict(),
                    'optimizer_deformations_glob': optimizer_deformations_glob.state_dict(),
                    'optimizer_deformations_loc' : opt_loc,
                    'scheduler_volume': scheduler_volume.state_dict(),
                    'scheduler_deformation_glob': scheduler_deformation_glob.state_dict(),
                    'scheduler_deformation_loc': sch_loc,
                    'ep': ep,
                }, os.path.join(config.path_save, 'training', 'model_trained.pt'))

                loss_tot_avg = np.array(loss_tot)
                step = (loss_tot_avg.max() - loss_tot_avg.min()) * 0.02
                plt.figure(figsize=(10, 10))
                plt.semilogy(loss_tot_avg[10:])
                plt.xticks(np.arange(0, len(loss_tot_avg[1:]), 1 + len(loss_tot_avg[1:]) // 10))
                # plt.grid()
                plt.savefig(os.path.join(config.path_save, 'training', 'loss.pdf'))

                print('Running and saving tests')
                if config.save_local_deformations:
                    ## Save local deformation
                    print("Saving local deformations")
                    t0 = time.time()
                    # if config.local_model=='interpolation':
                    utils_display.display_local_movie(implicit_deformation_list,field_true=None,Npts=(20,20),
                                                img_path=os.path.join(config.path_save,"training","deformations","local_deformations_"),img_type='.png',
                                                scale=1,alpha=0.8,width=0.0015,weights_est=1,device=config.device)
                    utils_display.display_local_movie(implicit_deformation_list,field_true=None,Npts=(20,20),
                                                img_path=os.path.join(config.path_save,"training","deformations_x10","local_deformations_"),img_type='.png',
                                                scale=0.1,alpha=0.8,width=0.0015,weights_est=1,device=config.device)
                    # else:
                    #     utils_display.display_local(implicit_deformation_list,field_true=None,Npts=(20,20),
                    #                                 img_path=os.path.join(config.path_save,"training","deformations","local_deformations_"),img_type='.png',
                    #                                 scale=1,alpha=0.8,width=0.0015)
                    #     utils_display.display_local(implicit_deformation_list,field_true=None,Npts=(20,20),
                    #                                 img_path=os.path.join(config.path_save,"training","deformations_x10","local_deformations_"),img_type='.png',
                    #                                 scale=0.1,alpha=0.8,width=0.0015)
                    if config.debug:
                        print("Elapsed time: {:2.0f} s".format(time.time() - t0))

                if config.save_global_deformations:
                    ## Save global deformation
                    print("Saving global deformations")
                    shiftEstimate, rotEstimate = utils_deformation.globalDeformationValues(shift_est,rot_est)
                    plt.figure(1)
                    plt.clf()
                    plt.scatter(angles,shiftEstimate[:,0,0]*n1_origin, label='x')
                    plt.scatter(angles,shiftEstimate[:,0,1]*n1_origin, label='y')
                    plt.axis([angles[0], angles[-1], -n1_origin//4, n2_origin//4])
                    plt.legend()
                    plt.savefig(os.path.join(config.path_save,"training","deformations","shifts_ep"+str(ep)+".png"))

                    plt.figure(1)
                    plt.clf()
                    plt.scatter(angles, rotEstimate*180/np.pi)
                    plt.scatter(angles, np.ones(len(angles))*fixed_rot.thetas.detach().cpu().numpy()*180/np.pi)
                    plt.axis([angles[0], angles[-1], -20, 20])
                    ang_ = np.round(fixed_rot.thetas.detach().cpu().item()*180/np.pi,2)
                    plt.legend(['est.','fixed rot. '+str(ang_)])
                    plt.title('Angles in degrees')
                    plt.savefig(os.path.join(config.path_save,"training","deformations","rotations_ep"+str(ep)+".png"))

                    plt.figure(1)
                    plt.clf()
                    plt.plot(angles, gains.detach().cpu().numpy())
                    plt.axis([angles[0], angles[-1], 0.1, 2])
                    plt.savefig(os.path.join(config.path_save,"training","deformations","gains_ep"+str(ep)+".png"))
                    if config.debug:
                        print("Elapsed time: {:2.0f} s".format(time.time()-t0))

                if config.save_volume and ep != 0:
                    ## Save slice of the volume
                    print("Computing and saving the volume from the implict net")
                    t0 = time.time()
                    V_icetide = sample_volume(impl_volume, config.n1, config.n2, config.n3, config.size_z_vol,
                                              size_max_vol, config.avg_XYZ,
                                              config.torch_type, device)
                    if config.debug:
                        path_ = os.path.join(config.path_save, "volume", "volume_slice_ep" + str(ep).zfill((5)))
                        if not os.path.exists(path_):
                            os.makedirs(path_)
                        for zz in range(config.n3_patch):
                            tmp = V_icetide[:, :, zz]
                            tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
                            tmp = np.floor(255 * tmp).astype(np.uint8)
                            imageio.imwrite(os.path.join(path_, "est_" + str(zz).zfill(5) + ".png"), tmp)
                    print("Saving implicit volume at location ", config.path_save, " under the name ", name_file,
                          "_icetide.mrc")
                    out = mrcfile.new(os.path.join(config.path_save, name_file + "_icetide.mrc"),
                                      np.moveaxis(V_icetide.astype(np.float32), 2, 0), overwrite=True)
                    out.close()

                    if config.debug:
                        # ICETIDE
                        tmp = V_icetide
                        tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
                        tmp = np.clip(tmp, a_min=np.quantile(tmp, 0.005), a_max=np.quantile(tmp, 0.995))
                        utils_display.display_XYZ(tmp, name="ICETIDE", path_save=os.path.join(config.path_save,'training'))
                        print("Elapsed time: {:2.0f} s".format(time.time() - t0))

        if (ep%config.Nalign == 0 or ep==config.epochs-1) and check_point_training and ep!=0 and config.save_aligned:
            with torch.no_grad():
                # Save aligned projections
                print("Aligning and saving projections")
                t0 = time.time()
                aligned_proj = aligned_projections(projections_noisy.detach().cpu(), rot_est, shift_est,
                                                   implicit_deformation_list, fixed_rot, deformationScale=config.deformationScale,
                                                   torch_type=config.torch_type, device=torch.device('cpu'))
                if config.debug:
                    path_ = os.path.join(config.path_save + "projections","raw_aligned_ep"+str(ep).zfill((5)))
                    if not os.path.exists(path_):
                        os.makedirs(path_)
                    for ll in range(aligned_proj.shape[0]):
                        tmp = aligned_proj[ll].detach().cpu().numpy()/gains[ll].detach().cpu().numpy()
                        tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
                        tmp = np.floor(255 * tmp).astype(np.uint8)
                        imageio.imwrite(os.path.join(path_,"est_"+str(ll).zfill(5))+".png", tmp)
                    try:
                        os.system(
                            "ffmpeg -y -f image2 -framerate 10 -i " + path_ + "/est_%05d.png -crf 22 -vf scale=512x512  " + path_+".gif")
                    except:
                        print("Didn't manage to run ffmpeg to save movie of the aligned projections.")
                tmp = aligned_proj.detach().cpu().numpy()/gains[:,None,None].detach().cpu().numpy()
                print("Saving aligned projections at location ", config.path_save, " under the name ", name_file,"_aligned.mrc")
                out = mrcfile.new(os.path.join(config.path_save, name_file+"_aligned.mrc"),
                                  tmp.astype(np.float32), overwrite=True)
                out.close()
                if config.debug:
                    print("Elapsed time: {:2.0f} s".format(time.time()-t0))

        if (ep%config.Nimplicit_volume==0 or ep==config.epochs-1) and check_point_training and (ep!=0) and config.save_aligned_implicitnet:
            with torch.no_grad():
                # Compute estimated projections
                print("Compute estimated projections from the implicit net")
                projEstimate_tot = sample_projection_from_implicit_net(config, impl_volume, rot_est, shift_est, implicit_deformation_list,
                                                     fixed_rot, angles_t, z_max_value, size_max_vol)
                tmp = projEstimate_tot/gains[:,None,None].detach().cpu().numpy()
                print("Saving projections from the implicit net at location ", config.path_save, " under the name ", name_file,"_aligned_implicit_net.mrc")
                out = mrcfile.new(os.path.join(config.path_save, name_file+"_aligned_implicit_net.mrc"),
                                  tmp.astype(np.float32), overwrite=True)
                out.close()
                if config.debug:
                    try:
                        os.system("ffmpeg -y -f image2 -framerate 10 -i "+ path_ +"/est_%05d.png -crf 22 -vf scale=512x512  "+path_ + ".gif")
                    except:
                        print("Didn't manage to run ffmpeg to save movie of the aligned projections.")

        plt.close('all')

        if ep == 50 and config.debug:
            print("##############################")
            print("Profiling of the code up to epoch 50.")
            print("##############################")
            pr.disable()
            s = io.StringIO()
            sortby = SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())

    print("Saving final state after training...")
    if len(list_params_deformations_loc)!=0:
        opt_loc = optimizer_deformations_loc.state_dict()
        sch_loc = scheduler_deformation_loc.state_dict()
    else:
        opt_loc = []
        sch_loc = []
    torch.save({
        'shift_est': shift_est,
        'rot_est': rot_est,
        'fixed_rot_est': fixed_rot,
        'gains': gains,
        'local_deformation_network': implicit_deformation_list,
        'implicit_volume': impl_volume.state_dict(),
        'optimizer_volume' : optimizer_volume.state_dict(),
        'optimizer_deformations_glob' : optimizer_deformations_glob.state_dict(),
        'optimizer_deformations_loc' : opt_loc,
        'scheduler_volume': scheduler_volume.state_dict(), 
        'scheduler_deformation_glob': scheduler_deformation_glob.state_dict(), 
        'scheduler_deformation_loc': sch_loc,
        'ep': ep,
    }, os.path.join(config.path_save,'training','model_trained.pt'))

    training_time = time.time()-t0_train
    # Saving the training time and the memory used
    if config.track_memory:
        max_memory_allocated_bytes = torch.cuda.max_memory_allocated()
        # Convert bytes to gigabytes
        max_memory_allocated_gb = max_memory_allocated_bytes / (1024**3)
        np.save(os.path.join(config.path_save,'training','memory_used.npy'),memory_used)
        np.savetxt(os.path.join(config.path_save,'training','memory_used.txt'),np.array([np.max(memory_used)/ (1024**3),max_memory_allocated_gb])) # Conversion in Gb
    np.save(os.path.join(config.path_save,'training','training_time.npy'),training_time)
    np.savetxt(os.path.join(config.path_save,'training','training_time.txt'),np.array([training_time]))

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

    print("Training is over.")
    if config.debug:
        plt.figure(figsize=(10,10))
        plt.plot(shift_estimates_np[:,shift_estimates_np.shape[1]//2,0,0]*n1_origin)
        plt.plot(shift_estimates_np[:,shift_estimates_np.shape[1]//2,0,1]*n2_origin)
        plt.title('Shift Estimates')
        plt.savefig(os.path.join(config.path_save,'training','shiftEstimates.png'))
