import os
import time
import torch
import mrcfile
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from utils import utils_deformation, utils_display

from utils.utils_sampling import get_sampling_geometry, sample_volume, aligned_projections
from utils.utils_data_processing import (load_projections, load_angles, select_volume_neural_network,
                                         select_local_deformation_neural_network)
from utils.reconstruction import TV_tomopy, FBP_tomopy, SIRT_tomopy


def generate_results(config):
    ######################################################################################################
    ## Setting the environment
    ######################################################################################################
    print("Evaluating the results.")
    # Choosing the seed and the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device
    if torch.cuda.device_count()>1:
        torch.cuda.set_device(config.device_num)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    print("Device: {}".format(device))

    # Parent Directories
    if not os.path.exists(config.path_save):
        os.makedirs(config.path_save)
    if not os.path.exists(os.path.join(config.path_save,"evaluation")):
        os.makedirs(os.path.join(config.path_save,"evaluation"))

    ######################################################################################################
    ## Loading the data
    ######################################################################################################
    # Load the data provided by the user
    print("Loading the tilt-series")
    projections_noisy, name_file = load_projections(config)
    (angles, Nangles_origin, n1_origin, n2_origin,
     view_angle_min, view_angle_max) = load_angles(config, projections_noisy)

    # Define geometry of sampling
    size_xy_vol, z_max_value = get_sampling_geometry(config.size_z_vol, view_angle_min, view_angle_max,
                                                     config.sampling_domain_lx, config.sampling_domain_ly)
    size_max_vol = 1.2*np.max([size_xy_vol,config.size_z_vol]) # increase by some small factor to account for deformations

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
    # Load model if required and if exists
    if os.path.isfile(os.path.join(config.path_save, 'training', 'model_trained.pt')):
        checkpoint = torch.load(os.path.join(config.path_save, 'training', 'model_trained.pt'), map_location=device, weights_only=False)
        impl_volume.load_state_dict(checkpoint['implicit_volume'])
        shift_est = checkpoint['shift_est']
        rot_est = checkpoint['rot_est']
        fixed_rot = checkpoint['fixed_rot_est']
        gains = checkpoint['gains']
        implicit_deformation_list = checkpoint['local_deformation_network']

    if config.downsample_tilt_series:
        print("Binning the data")
        projections_noisy = torch.tensor(resize(projections_noisy.detach().cpu().numpy(),
                                       (Nangles_origin, config.n1, config.n1))).to(device).type(config.torch_type)

    print("Aligning and saving projections")
    aligned_proj = aligned_projections(projections_noisy.detach().cpu(), rot_est, shift_est,
                                       implicit_deformation_list, fixed_rot, deformationScale=config.deformationScale,
                                       torch_type=config.torch_type, device=torch.device('cpu'))
    tmp = aligned_proj.detach().cpu().numpy() / gains[:, None, None].detach().cpu().numpy()
    print("Saving aligned projections at location ", os.path.join(config.path_save, 'evaluation', name_file + "_aligned.mrc"))
    out = mrcfile.new(os.path.join(config.path_save, 'evaluation', name_file + "_aligned.mrc"),
                      tmp.astype(np.float32), overwrite=True)
    out.close()

    print("Volume reconstruction")
    if config.FBP_reconstruction:
        print("Saving FBP volume at location ", os.path.join(config.path_save, 'evaluation', name_file + "_FBP_volume.mrc"))
        t0 = time.time()
        V_FBP = FBP_tomopy(aligned_proj.detach().cpu().numpy(), angles, config.n3)
        if config.debug:
            print("Elapsed time: {}".format(time.time() - t0))
        out = mrcfile.new(os.path.join(config.path_save, 'evaluation', name_file + "_FBP_volume.mrc"),
                          np.moveaxis(V_FBP.astype(np.float32), 2, 0), overwrite=True)
        out.close()
    if config.SIRT_reconstruction:
        print("Saving SIRT volume at location ", os.path.join(config.path_save, 'evaluation', name_file + "_SIRT_volume.mrc"))
        t0 = time.time()
        V_SIRT = SIRT_tomopy(aligned_proj.detach().cpu().numpy(), angles, config.nit_sirt, config.n3)
        if config.debug:
            print("Elapsed time: {}".format(time.time() - t0))
        out = mrcfile.new(os.path.join(config.path_save, 'evaluation', name_file + "_SIRT_volume.mrc"),
                          np.moveaxis(V_SIRT.astype(np.float32), 2, 0), overwrite=True)
        out.close()
    if config.TV_reconstruction:
        print("Saving TV volume at location ", os.path.join(config.path_save, 'evaluation', name_file + "_TV_volume.mrc"))
        t0 = time.time()
        V_TV = TV_tomopy(aligned_proj.detach().cpu().numpy(), angles, config.reg_tv, config.nit_tv, config.n3)
        if config.debug:
            print("Elapsed time: {}".format(time.time() - t0))
        out = mrcfile.new(os.path.join(config.path_save, 'evaluation', name_file + "_TV_volume.mrc"),
                          np.moveaxis(V_TV.astype(np.float32), 2, 0), overwrite=True)
        out.close()

    print("Saving the volume from the implict net")
    t0 = time.time()
    V_icetide = sample_volume(impl_volume, config.n1, config.n2, config.n3, config.size_z_vol,
                              size_max_vol, config.avg_XYZ,
                              config.torch_type, device)
    if config.debug:
        print("Elapsed time: {}".format(time.time() - t0))
    out = mrcfile.new(os.path.join(config.path_save, 'evaluation', name_file + "_icetide.mrc"),
                      np.moveaxis(V_icetide.astype(np.float32), 2, 0), overwrite=True)
    out.close()
    tmp = V_icetide
    tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
    tmp = np.clip(tmp, a_min=np.quantile(tmp, 0.005), a_max=np.quantile(tmp, 0.995))
    utils_display.display_XYZ(tmp, name=name_file+"_ICETIDE", path_save=os.path.join(config.path_save,'evaluation'))

    ## Save global deformation
    print("Saving global deformations")
    shiftEstimate, rotEstimate = utils_deformation.globalDeformationValues(shift_est, rot_est)
    plt.figure(1)
    plt.clf()
    plt.scatter(angles, shiftEstimate[:, 0, 0] * n1_origin, label='x')
    plt.scatter(angles, shiftEstimate[:, 0, 1] * n1_origin, label='y')
    plt.axis([angles[0], angles[-1], -n1_origin // 10, n2_origin // 10])
    plt.legend()
    plt.savefig(os.path.join(config.path_save, "evaluation",  name_file+"_shifts.png"))

    plt.figure(1)
    plt.clf()
    plt.scatter(angles, rotEstimate * 180 / np.pi)
    plt.scatter(angles, np.ones(len(angles)) * fixed_rot.thetas.detach().cpu().numpy() * 180 / np.pi)
    plt.axis([angles[0], angles[-1], -20, 20])
    ang_ = np.round(fixed_rot.thetas.detach().cpu().item() * 180 / np.pi, 2)
    plt.legend(['est.', 'fixed rot. ' + str(ang_)])
    plt.title('Angles in degrees')
    plt.savefig(os.path.join(config.path_save, "evaluation", name_file+"_rotations.png"))

    plt.figure(1)
    plt.clf()
    plt.plot(angles, gains.detach().cpu().numpy())
    plt.axis([angles[0], angles[-1], 0.1, 2])
    plt.savefig(os.path.join(config.path_save, "evaluation", name_file+"_gains.png"))
