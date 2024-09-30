from skimage.transform import resize

import numpy as np
import torch
import matplotlib.pyplot as plt
plt.ion()
import mrcfile
from ops.radon_3d_lib import ParallelBeamGeometry3DOpAngles_rectangular
import os
import imageio
from utils import utils_deformation, utils_display, utils_FSC ,utils_sampling
import shutil

import astra

from matplotlib import gridspec
from scipy.interpolate import griddata

import pandas as pd
# from reconstruct_FBP_volume import reconstruct_FBP_volume
from utils.utils_deformation import cropper


# import configs.real_10643 as config_file
import configs.real_11070 as config_file
config = config_file.get_config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count()>1:
    torch.cuda.set_device(config.device_num)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
config.device = device

# Parent Dircetorys 
if not os.path.exists(config.path_save):
    os.makedirs(config.path_save)
if not os.path.exists(config.path_save+"/evaluation/"):
    os.makedirs(config.path_save+"/evaluation/")
if not os.path.exists(config.path_save+"/evaluation/projections/"):
    os.makedirs(config.path_save+"/evaluation/projections/")
if not os.path.exists(config.path_save+"/evaluation/volumes/"):
    os.makedirs(config.path_save+"/evaluation/volumes/")
if not os.path.exists(config.path_save+"/evaluation/deformations/"):
    os.makedirs(config.path_save+"/evaluation/deformations/")
if not os.path.exists(config.path_save+"/evaluation/volume_slices/"):
    os.makedirs(config.path_save+"/evaluation/volume_slices/")

# Our method
if not os.path.exists(config.path_save+"/evaluation/projections/ICETIDE/"):
    os.makedirs(config.path_save+"/evaluation/projections/ICETIDE/")
if not os.path.exists(config.path_save+"/evaluation/volumes/ICETIDE/"):
    os.makedirs(config.path_save+"/evaluation/volumes/ICETIDE/")
if not os.path.exists(config.path_save+"/evaluation/deformations/ICETIDE/"):
    os.makedirs(config.path_save+"/evaluation/deformations/ICETIDE/")
if not os.path.exists(config.path_save+"/evaluation/volume_slices/ICETIDE/"):
    os.makedirs(config.path_save+"/evaluation/volume_slices/ICETIDE/")

# AreTomos method
if not os.path.exists(config.path_save+"/evaluation/projections/AreTomo/"):
    os.makedirs(config.path_save+"/evaluation/projections/AreTomo/")
if not os.path.exists(config.path_save+"/evaluation/volumes/AreTomo/"):
    os.makedirs(config.path_save+"/evaluation/volumes/AreTomo/")
if not os.path.exists(config.path_save+"/evaluation/deformations/AreTomo/"):
    os.makedirs(config.path_save+"/evaluation/deformations/AreTomo/")
if not os.path.exists(config.path_save+"/evaluation/volume_slices/AreTomo/"):
    os.makedirs(config.path_save+"/evaluation/volume_slices/AreTomo/")

# Etomo method
if not os.path.exists(config.path_save+"/evaluation/projections/Etomo/"):
    os.makedirs(config.path_save+"/evaluation/projections/Etomo/")   
if not os.path.exists(config.path_save+"/evaluation/volumes/Etomo/"):
    os.makedirs(config.path_save+"/evaluation/volumes/Etomo/")
if not os.path.exists(config.path_save+"/evaluation/deformations/Etomo/"):
    os.makedirs(config.path_save+"/evaluation/deformations/Etomo/")
if not os.path.exists(config.path_save+"/evaluation/volume_slices/Etomo/"):
    os.makedirs(config.path_save+"/evaluation/volume_slices/Etomo/")

# True volume
if not os.path.exists(config.path_save+"/evaluation/deformations/true/"):
    os.makedirs(config.path_save+"/evaluation/deformations/true/")
if not os.path.exists(config.path_save+"/evaluation/volume_slices/true/"):
    os.makedirs(config.path_save+"/evaluation/volume_slices/true/")

# FBP on undistorted projections
if not os.path.exists(config.path_save+"/evaluation/projections/FBP_no_deformed/"):
    os.makedirs(config.path_save+"/evaluation/projections/FBP_no_deformed/")
if not os.path.exists(config.path_save+"/evaluation/volumes/FBP_no_deformed/"):
    os.makedirs(config.path_save+"/evaluation/volumes/FBP_no_deformed/")
if not os.path.exists(config.path_save+"/evaluation/volume_slices/FBP_no_deformed/"):
    os.makedirs(config.path_save+"/evaluation/volume_slices/FBP_no_deformed/")

# FBP
if not os.path.exists(config.path_save+"/evaluation/projections/FBP/"):
    os.makedirs(config.path_save+"/evaluation/projections/FBP/")
if not os.path.exists(config.path_save+"/evaluation/volumes/FBP/"):
    os.makedirs(config.path_save+"/evaluation/volumes/FBP/")
if not os.path.exists(config.path_save+"/evaluation/volume_slices/FBP/"):
    os.makedirs(config.path_save+"/evaluation/volume_slices/FBP/")

#FBP ICETIDE deformation estimations
if not os.path.exists(config.path_save+"/evaluation/projections/FBP_ICETIDE/"):
    os.makedirs(config.path_save+"/evaluation/projections/FBP_ICETIDE/")
if not os.path.exists(config.path_save+"/evaluation/volumes/FBP_ICETIDE/"):
    os.makedirs(config.path_save+"/evaluation/volumes/FBP_ICETIDE/")
if not os.path.exists(config.path_save+"/evaluation/volume_slices/FBP_ICETIDE/"):
    os.makedirs(config.path_save+"/evaluation/volume_slices/FBP_ICETIDE/")


# SART_TV
if not os.path.exists(config.path_save+"/evaluation/projections/SART_TV/"):
    os.makedirs(config.path_save+"/evaluation/projections/SART_TV/")
if not os.path.exists(config.path_save+"/evaluation/volumes/SART_TV/"):
    os.makedirs(config.path_save+"/evaluation/volumes/SART_TV/")
if not os.path.exists(config.path_save+"/evaluation/volume_slices/SART_TV/"):
    os.makedirs(config.path_save+"/evaluation/volume_slices/SART_TV/")

# print('data generation')
# import data_generation
# data_generation.data_generation_real_data(config)
#
# print('Training...')
# import train as train
# train.train_without_ground_truth(config)
#
#
#
# aa = cd






import tomopy
def TV_tomopy(projections, angles, reg_tv, nit_tv, n3):
    recon = np.swapaxes(tomopy.recon(projections, angles/180*np.pi,
                                     algorithm='tv', sinogram_order=False, reg_par=reg_tv, num_iter=nit_tv), 1,2)
    recon = recon[:, :, recon.shape[2] // 2 - n3 // 2:recon.shape[2] // 2 + n3 // 2]
    return recon[:, :, ::-1]

import os
import torch
import shutil
import mrcfile
import imageio
import numpy as np
import pandas as pd
from matplotlib import gridspec
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.interpolate import griddata
from ops.radon_3d_lib import ParallelBeamGeometry3DOpAngles_rectangular

from utils.utils_deformation import cropper
from utils.utils_sampling import get_sampling_geometry
from utils import utils_deformation, utils_display, utils_FSC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    torch.cuda.set_device(config.device_num)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
config.device = device

# Parent Dircetorys
if not os.path.exists(config.path_save):
    os.makedirs(config.path_save)
if not os.path.exists(config.path_save + "/evaluation/"):
    os.makedirs(config.path_save + "/evaluation/")
if not os.path.exists(config.path_save + "/evaluation/projections/"):
    os.makedirs(config.path_save + "/evaluation/projections/")
if not os.path.exists(config.path_save + "/evaluation/volumes/"):
    os.makedirs(config.path_save + "/evaluation/volumes/")
if not os.path.exists(config.path_save + "/evaluation/deformations/"):
    os.makedirs(config.path_save + "/evaluation/deformations/")
if not os.path.exists(config.path_save + "/evaluation/volume_slices/"):
    os.makedirs(config.path_save + "/evaluation/volume_slices/")

# Our method
if not os.path.exists(config.path_save + "/evaluation/projections/ICETIDE/"):
    os.makedirs(config.path_save + "/evaluation/projections/ICETIDE/")
if not os.path.exists(config.path_save + "/evaluation/deformations/ICETIDE/"):
    os.makedirs(config.path_save + "/evaluation/deformations/ICETIDE/")
if not os.path.exists(config.path_save + "/evaluation/volume_slices/ICETIDE/"):
    os.makedirs(config.path_save + "/evaluation/volume_slices/ICETIDE/")

# FBP
if not os.path.exists(config.path_save + "/evaluation/projections/Best/"):
    os.makedirs(config.path_save + "/evaluation/projections/Best/")
if not os.path.exists(config.path_save + "/evaluation/volume_slices/Best/"):
    os.makedirs(config.path_save + "/evaluation/volume_slices/Best/")

# FBP ICETIDE deformation estimations
if not os.path.exists(config.path_save + "/evaluation/projections/FBP_ICETIDE/"):
    os.makedirs(config.path_save + "/evaluation/projections/FBP_ICETIDE/")
if not os.path.exists(config.path_save + "/evaluation/volume_slices/FBP_ICETIDE/"):
    os.makedirs(config.path_save + "/evaluation/volume_slices/FBP_ICETIDE/")

######################################################################################################
## Load data
######################################################################################################
data = np.load(config.path_save_data + "volume_and_projections.npz")
# projections_noisy = torch.tensor(data['projections_noisy']).type(config.torch_type).to(device)
if config.name_best_volume is not None:
    if config.name_best_volume != "":
        V_best_t = torch.tensor(
            np.moveaxis(np.double(mrcfile.open(config.path_load + config.name_best_volume).data), 0, 2)).type(
            config.torch_type).to(device)
        V_best_t = torch.rot90(V_best_t, k=2, dims=[0, 1])
    else:
        V_best_t = torch.zeros((config.n1, config.n2, config.n3))
else:
    V_best_t = torch.zeros((config.n1, config.n2, config.n3))
# numpy
V_best = V_best_t.detach().cpu().numpy()

data = np.load(config.path_save_data + "volume_and_projections.npz")
projections_noisy = torch.Tensor(data['projections_noisy']).type(config.torch_type).to(device)
config.Nangles = projections_noisy.shape[0]
projections_noisy_resize = torch.Tensor(
    resize(projections_noisy.detach().cpu().numpy(), (config.Nangles, config.n1, config.n2))).type(
    config.torch_type).to(device)


######################################################################################################
## Load and estimate our volume
######################################################################################################
## Load implicit network
if (config.volume_model == "Fourier-features"):
    from models.fourier_net import FourierNet, FourierNet_Features

    impl_volume = FourierNet_Features(
        in_features=config.input_size_volume,
        sub_features=config.sub_features,
        out_features=config.output_size_volume,
        hidden_features=config.hidden_size_volume,
        hidden_blocks=config.num_layers_volume,
        L=config.L_volume).to(device)

if (config.volume_model == "MLP"):
    from models.fourier_net import MLP

    impl_volume = MLP(in_features=1,
                      hidden_features=config.hidden_size_volume, hidden_blocks=config.num_layers_volume,
                      out_features=config.output_size_volume).to(device)

if (config.volume_model == "multi-resolution"):
    import tinycudann as tcnn

    config_network = {"encoding": {
        'otype': config.encoding.otype,
        'type': config.encoding.type,
        'n_levels': config.encoding.n_levels,
        'n_features_per_level': config.encoding.n_features_per_level,
        'log2_hashmap_size': config.encoding.log2_hashmap_size,
        'base_resolution': config.encoding.base_resolution,
        'per_level_scale': config.encoding.per_level_scale,
        'interpolation': config.encoding.interpolation,
    },
        "network": {
            "otype": config.network.otype,
            "activation": config.network.activation,
            "output_activation": config.network.output_activation,
            "n_neurons": config.hidden_size_volume,
            "n_hidden_layers": config.num_layers_volume
        }
    }
    impl_volume = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=1,
                                                encoding_config=config_network["encoding"],
                                                network_config=config_network["network"]).to(device)
num_param = sum(p.numel() for p in impl_volume.parameters() if p.requires_grad)
print('---> Number of trainable parameters in volume net: {}'.format(num_param))
checkpoint = torch.load(os.path.join(config.path_save, 'training', 'model_trained.pt'), map_location=device)
impl_volume.load_state_dict(checkpoint['implicit_volume'])
shift_icetide = checkpoint['shift_est']
rot_icetide = checkpoint['rot_est']
gains = checkpoint['gains']
implicit_deformation_icetide = checkpoint['local_deformation_network']
size_xy_vol, z_max_value = get_sampling_geometry(config.size_z_vol, config.view_angle_min, config.view_angle_max,
                                                 config.sampling_domain_lx, config.sampling_domain_ly)
size_max_vol = 1.2 * np.max(
    [size_xy_vol, config.size_z_vol])  # increase by some small factor to account for deformations
# Compute estimated volume
with torch.no_grad():
    x_lin1 = np.linspace(-1, 1, config.n1_eval)
    x_lin2 = np.linspace(-1, 1, config.n2_eval)
    XX, YY = np.meshgrid(x_lin1, x_lin2, indexing='ij')
    grid2d = np.concatenate([XX.reshape(-1, 1), YY.reshape(-1, 1)], 1)
    grid2d_t = torch.tensor(grid2d).type(config.torch_type)
    z_range = np.linspace(-1, 1, config.n3_eval) * config.size_z_vol
    V_icetide = np.zeros((config.n1_eval, config.n2_eval, config.n3_eval))
    for zz, zval in enumerate(z_range):
        grid3d = np.concatenate([grid2d_t, zval * torch.ones((grid2d_t.shape[0], 1))], 1)
        grid3d_slice = torch.tensor(grid3d).type(config.torch_type).to(device)
        estSlice = impl_volume(grid3d_slice / size_max_vol / 2 + 0.5).detach().cpu().numpy().reshape(config.n1_eval,
                                                                                                     config.n2_eval)
        V_icetide[:, :, zz] = estSlice
    if config.avg_XYZ > 1:
        padded_array = np.pad(V_icetide, ((0, 0), (0, 0), (0, config.avg_XYZ - 1)), mode='constant')
        filt = np.zeros_like(padded_array)
        filt[:, :,
        filt.shape[2] // 2 - config.avg_XYZ // 2:filt.shape[2] // 2 + config.avg_XYZ // 2] = 1 / config.avg_XYZ
        V_icetide = np.fft.fftshift(np.fft.ifft((np.fft.fft(filt) * np.fft.fft(padded_array))).real, axes=-1)[:, :,
                    :config.n3_eval]
    V_icetide_t = torch.tensor(V_icetide).type(config.torch_type).to(device)

# # Get the local deformation error plots
# for index in range(config.Nangles):
#     savepath = os.path.join(config.path_save, 'evaluation', 'deformations', 'ICETIDE',
#                             'local_deformation_factor10_{}'.format(index))
#     utils_display.display_local_est_and_true(implicit_deformation_icetide[index], None, Npts=(20, 20), scale=0.1,
#                                              img_path=savepath)


def reconstruct_FBP_volume(config, tiltseries):
    """
    Args:
        config :
        tiltseries (torch tensor): volume
    """
    # Define the forward operator
    angles = np.linspace(config.view_angle_min,config.view_angle_max,config.Nangles)
    operator_ET = ParallelBeamGeometry3DOpAngles_rectangular((config.n1,config.n2,config.n3), angles/180*np.pi, fact=1)

    # Reconstruct the volume
    V_FBP = operator_ET.pinv(tiltseries).detach().requires_grad_(False)

    return V_FBP
######################################################################################################
# Using only the deformation estimates
######################################################################################################
projections_noisy_undeformed = torch.zeros_like(projections_noisy_resize)
xx1 = torch.linspace(-1, 1, config.n1, dtype=config.torch_type, device=device)
xx2 = torch.linspace(-1, 1, config.n2, dtype=config.torch_type, device=device)
XX_t, YY_t = torch.meshgrid(xx1, xx2, indexing='ij')
XX_t = torch.unsqueeze(XX_t, dim=2)
YY_t = torch.unsqueeze(YY_t, dim=2)
for i in range(config.Nangles):
    coordinates = torch.cat([XX_t, YY_t], 2).reshape(-1, 2)
    thetas = torch.tensor(-rot_icetide[i].thetas.item()).to(device)

    rot_deform = torch.stack(
        [torch.stack([torch.cos(thetas), torch.sin(thetas)], 0),
         torch.stack([-torch.sin(thetas), torch.cos(thetas)], 0)]
        , 0)
    coordinates = coordinates - shift_icetide[i].shifts_arr
    coordinates = coordinates - config.deformationScale * implicit_deformation_icetide[i](coordinates)
    coordinates = torch.transpose(torch.matmul(rot_deform, torch.transpose(coordinates, 0, 1)), 0, 1)  ## do rotation
    x = projections_noisy_resize[i].clone().view(1, 1, config.n1, config.n2)
    x = x.expand(config.n1 * config.n2, -1, -1, -1)
    out = cropper(x, coordinates, output_size=1).reshape(config.n1, config.n2)
    projections_noisy_undeformed[i] = out
V_FBP_icetide = reconstruct_FBP_volume(config, projections_noisy_undeformed).detach().cpu().numpy()

projections_FBP_icetide = projections_noisy_undeformed.detach().cpu().numpy()
out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","FBP_icetide_projections.mrc"),projections_FBP_icetide.astype(np.float32),overwrite=True)
out.close()

# Compute estimated projections
from utils.utils_sampling import get_sampling_geometry, apply_deformations_to_locations, generate_rays_batch, sample_projections
angles = np.linspace(config.view_angle_min,config.view_angle_max,config.Nangles)
angles_t = torch.tensor(angles).type(config.torch_type).to(device)
projEstimate_tot = np.zeros((config.Nangles, config.n1_eval, config.n2_eval))
x_lin1 = np.linspace(-1, 1, config.n1_eval)
x_lin2 = np.linspace(-1, 1, config.n2_eval)
XX, YY = np.meshgrid(x_lin1, x_lin2, indexing='ij')
grid2d_ = np.concatenate([XX.reshape(-1, 1), YY.reshape(-1, 1)], 1)
grid2d_ = torch.tensor(grid2d_).type(config.torch_type).to(device)
fixed_rot = []
fixedAngle = torch.FloatTensor([config.fixed_angle * np.pi / 180]).to(device)[0]
for k in range(config.Nangles):
    fixed_rot.append(utils_deformation.rotNet(1, x0=fixedAngle).to(device))
for ll, angle in enumerate(angles_t):
    print(ll)
    for jj in range(config.n2_eval):
        # Define the detector locations
        detectorLocations = grid2d_.reshape(config.n1_eval, config.n2_eval, 2)[:, jj].reshape(1, -1, 2)

        # Apply deformations in the 2D space
        # detectorLocationsDeformed = apply_deformations_to_locations(detectorLocations, rot_icetide[ll:ll + 1],
        #                                                             shift_icetide[ll:ll + 1],
        #                                                             implicit_deformation_icetide[ll:ll + 1],
        #                                                             fixed_rot[ll:ll + 1], scale=config.deformationScale,
        #                                                             cl=config.clip)


        # generate the rays in 3D
        rays_rotated = generate_rays_batch(detectorLocations, angle[None], z_max_value, config.ray_length,
                                           std_noise=config.std_noise_z)
        # Scale the rays so that they are trully in [-1,1]
        rays_rotated_scaled = rays_rotated / size_max_vol
        # Sample the implicit volume by making the input in [0,1]
        outputValues = impl_volume((rays_rotated_scaled / 2 + 0.5).reshape(-1, 3)).reshape(1,
                                                                                           config.n1_eval,
                                                                                           config.ray_length)
        support = (rays_rotated[:, :, :, 2].abs() < config.size_z_vol) * 1
        projEstimate = torch.sum(support * outputValues, 2) / config.ray_length
        projEstimate_tot[ll, :, jj] = (gains[ll]*projEstimate).reshape(-1).detach().cpu().numpy()
    tmp = projEstimate_tot[ll]
    tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
    tmp = np.floor(255 * tmp).astype(np.uint8)
    imageio.imwrite(config.path_save_data + "evaluation/projections/ICETIDE/est_" + str(ll) + ".png", tmp)
out = mrcfile.new(os.path.join(config.path_save_data, 'evaluation',
                               "projections", "ICETIDE_proj.mrc"), projEstimate_tot.astype(np.float32),overwrite=True)
out.close()

def display_XYZ(tmp, name="true"):
    avg = 0
    sl0 = tmp.shape[0] // 2
    sl1 = tmp.shape[1] // 2
    sl2 = tmp.shape[2] // 2
    f, aa = plt.subplots(2, 2, gridspec_kw={'height_ratios': [tmp.shape[2] / tmp.shape[0], 1],
                                            'width_ratios': [1, tmp.shape[2] / tmp.shape[0]]})
    aa[0, 0].imshow(tmp[sl0 - avg // 2:sl0 + avg // 2 + 1, :, :].mean(0).T, cmap='gray', vmin=tmp.min(), vmax=tmp.max())
    aa[0, 0].axis('off')
    aa[1, 0].imshow(tmp[:, :, sl2 - avg // 2:sl2 + avg // 2 + 1].mean(2), cmap='gray', vmin=tmp.min(), vmax=tmp.max())
    aa[1, 0].axis('off')
    aa[1, 1].imshow(tmp[:, sl1 - avg // 2:sl1 + avg // 2 + 1, :].mean(1), cmap='gray', vmin=tmp.min(), vmax=tmp.max())
    aa[1, 1].axis('off')
    aa[0, 1].axis('off')
    plt.tight_layout(pad=1, w_pad=-1, h_pad=1)
    plt.savefig(os.path.join("tmp.png"))
    plt.savefig(os.path.join(config.path_save_data, 'evaluation', "volumes", name + "_XYZ_slice.png"))

    f, aa = plt.subplots(2, 2, gridspec_kw={'height_ratios': [tmp.shape[2] / tmp.shape[0], 1],
                                            'width_ratios': [1, tmp.shape[2] / tmp.shape[0]]})
    aa[0, 0].imshow(tmp.mean(0).T, cmap='gray', vmin=tmp.min(), vmax=tmp.max())
    aa[0, 0].axis('off')
    aa[1, 0].imshow(tmp.mean(2), cmap='gray', vmin=tmp.min(), vmax=tmp.max())
    aa[1, 0].axis('off')
    aa[1, 1].imshow(tmp.mean(1), cmap='gray', vmin=tmp.min(), vmax=tmp.max())
    aa[1, 1].axis('off')
    aa[0, 1].axis('off')
    plt.tight_layout(pad=1, w_pad=-1, h_pad=1)
    plt.savefig(os.path.join(config.path_save_data, 'evaluation', "volumes", name + "_XYZ_proj.png"))

# FBP_ICETIDE volume
tmp = V_FBP_icetide[40:-40,40:-40,60:-60]
tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
tmp = np.clip(tmp**0.8, a_min=np.quantile(tmp, 0.005), a_max=np.quantile(tmp, 0.995))
display_XYZ(tmp, name="FBP_ICETIDE")


# ICETIDE volume
tmp = V_icetide
tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
tmp = np.clip(tmp, a_min=np.quantile(tmp, 0.005), a_max=np.quantile(tmp, 0.995))
display_XYZ(tmp, name="ICETIDE")

out = mrcfile.new(os.path.join(config.path_save_data, 'evaluation',
                               "volumes", "ICETIDE_volume.mrc"), np.moveaxis(V_icetide.astype(np.float32), 2, 0),
                  overwrite=True)
out.close()
out = mrcfile.new(os.path.join(config.path_save_data, 'evaluation', "volumes",
                               "FBP_icetide_volume.mrc"), np.moveaxis(V_FBP_icetide.astype(np.float32), 2, 0),
                  overwrite=True)
out.close()
plt.close('all')
print("volumes saved")

if config.name_best_proj != "":
    P_best = np.double(mrcfile.open(os.path.join(config.path_load, config.name_best_proj)).data)
    if config.projections_rotate:
        P_best = np.rot90(np.flip(P_best,axis=1), k=3, axes=((1, 2)))
    P_best_t = torch.tensor(P_best).type(config.torch_type).to(device)
    angles = np.linspace(config.view_angle_min, config.view_angle_max, config.Nangles)
    import ml_collections
    config_best = ml_collections.ConfigDict()
    config_best.n1 = P_best.shape[1]
    config_best.n2 = P_best.shape[2]
    config_best.n3 = P_best.shape[1]//2
    config_best.view_angle_min = -57
    config_best.view_angle_max = 57
    config_best.Nangles = P_best.shape[0]
    V_no_deformed_FBP = reconstruct_FBP_volume(config_best, P_best_t).detach().cpu().numpy()
    display_XYZ(V_no_deformed_FBP, name="FBP_no_deformed")
    out = mrcfile.new(os.path.join(config.path_save_data, 'evaluation', "volumes", "FBP_no_deformed.mrc"),
                      np.moveaxis(V_no_deformed_FBP,2,0).astype(np.float32), overwrite=True)
    out.close()
    V_no_deformed_tv = TV_tomopy(P_best, angles, config.lamb_tv, config.nit_tv, config_best.n3)
    display_XYZ(V_no_deformed_tv, name="Tv_no_deformed")
    out = mrcfile.new(os.path.join(config.path_save_data, 'evaluation', "volumes", "TV_no_deformed.mrc"),
                      np.moveaxis(V_no_deformed_tv,2,0).astype(np.float32), overwrite=True)
    out.close()














