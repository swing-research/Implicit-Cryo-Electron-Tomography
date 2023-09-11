import numpy as np
import torch
# import torch.nn as nn
import matplotlib.pyplot as plt
plt.ion()
from skimage.transform import resize
import mrcfile
# import time
# from torchsummary import summary
# from ops.radon_3d_lib import ParallelBeamGeometry3DOpAngles, ParallelBeamGeometry3DOpAngles_rectangular
from ops.radon_3d_lib import ParallelBeamGeometry3DOpAngles_rectangular
import os
import imageio
# import torch.nn.functional as F
import torch.nn as nn

# from utils import data_generation, reconstruction, utils_deformation, utils_sampling, utils_interpolation, utils_display
from utils import data_generation, utils_deformation, utils_display
# from utils.reconstruction import getfsc

# from torch.utils.data import DataLoader, TensorDataset
# from utils.utils_sampling import sample_implicit_batch,sample_rays, sample_implicit, sample_implicit_batch_v2

import warnings
warnings.filterwarnings('ignore') 

# Introduction
'''
This script is used to generate data from a clean tomogram. 
The goal is to compare different approach on this dataset that is suppose to mimic the CryoET image formation model.
'''

"""
## TODO: parameters to put inot args
- device
- config.torch_type
- seed
- config.volume_name
- config.n1, config.n2, config.n3
- config.Nangles
- angle_min, angle_max
- SNR_value
- config.sigma_PSF
- scale_min
- scale_max
- shift_min
- shift_max
- shear_min
- shear_max
- angle_min
- angle_max
- sigma_local_def
- N_ctrl_pts_local_def
"""
from configs.config_reconstruct_simulation import get_default_configs
config = get_default_configs()


use_cuda=torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
if torch.cuda.device_count()>1:
    torch.cuda.set_device(config.device_num)
np.random.seed(config.seed)
torch.manual_seed(config.seed)

# config.volume_name = 'model_0'

# # Parameters for the data generation
# # size of the volume to use to generate the tilt-series
# config.n1 = 512
# config.n2 = 512
# config.n3 = 180 # size of the effective volume
# # size of the patch to crop in the raw volume
# config.n1_patch = 512
# config.n2_patch = 512
# config.n3_patch = 180 # size of the effective volume
# # nZ = 512 # size of the extended volume
# config.Nangles = 61
# view_angle_min = -60
# view_angle_max = 60
# SNR_value = 10
# config.sigma_PSF = 3.
# config.number_sub_projections = 1

# scale_min = 1.0
# scale_max = 1.0
# shift_min = -0.04
# shift_max = 0.04
# shear_min = -0.0
# shear_max = 0.0
# angle_min = -4/180*np.pi
# angle_max = 4/180*np.pi
# sigma_local_def = 4
# N_ctrl_pts_local_def = (12,12)


# grid_class = utils_sampling.grid_class(config.n1,config.n2,config.n3,config.torch_type,device)

# # define undersampling grid
# us = 4 # under sample factor
# grid_class_us = utils_sampling.grid_class(config.n1//us,config.n2//us,config.n3//us,config.torch_type,device)


if not os.path.exists("results/"):
    os.makedirs("results/")
if not os.path.exists(config.path_save_data):
    os.makedirs(config.path_save_data)
if not os.path.exists(config.path_save_data+"projections/"):
    os.makedirs(config.path_save_data+"projections/")
if not os.path.exists(config.path_save_data+"deformations/"):
    os.makedirs(config.path_save_data+"deformations/")


#######################################################################################
## Load data
#######################################################################################
# Parameters
name_volume="grandmodel.mrc" # here: https://www.shrec.net/cryo-et/

# make sure it works on different computers, add your path bellow
if os.path.exists("/raid/Valentin/"+str(config.volume_name)+"/"+name_volume):
    path_volume = "/raid/Valentin/"+str(config.volume_name)+"/"+name_volume
elif os.path.exists("/users/staff/dmi-dmi/debarn0000/data_nobackup/"+str(config.volume_name)+"/"+name_volume):
    path_volume = "/users/staff/dmi-dmi/debarn0000/data_nobackup/"+str(config.volume_name)+"/"+name_volume
elif os.path.exists("/home/debarn0000/Documents/Data/shrec2021_full_dataset/"+str(config.volume_name)+"/"+name_volume):
    path_volume = "/home/debarn0000/Documents/Data/shrec2021_full_dataset/"+str(config.volume_name)+"/"+name_volume
else:
    path_volume = " " # "./datasets/volume_reconstruction.mrc" # download here: https://drive.google.com/file/d/1o9w5EX8YgSlW78nfaKaT9ZN8UnfxZ6Be/view?usp=sharing

# Loading and shaping the volume
# TODO: save full middle slice and the one taken, just to validate what we are doing
V = np.double(mrcfile.open(path_volume).data)
nv = V.shape # size of the loaded volume 
V = V[nv[0]//2-config.n3_patch//2:nv[0]//2+config.n3_patch//2,nv[1]//2-config.n1_patch//2:nv[1]//2+config.n1_patch//2,nv[2]//2-config.n2_patch//2:nv[2]//2+config.n2_patch//2]
V = resize(V,(config.n3,config.n1,config.n2))
V = np.swapaxes(V,0,1)
V = np.swapaxes(V,1,2)
V = (V - V.min())/(V.max()-V.min())
V /= V.max()

# # define molifier to ensure support and avoid padding artifacts
# mollifier = utils_sampling.mollifier_class(-1,config.torch_type,device)

# V = mollifier.mollify3d_np()*V
# w = (V.sum()/(config.n1*config.n2*config.n3)).item()
# V /= w
V_t = torch.tensor(V).to(device).type(config.torch_type)
# barycenter_true = (V_t.reshape(-1,1)*grid_class.grid3d_t).mean(0)


#######################################################################################
## Generate projections
#######################################################################################
# Define angles and X-ray transform
angles = np.linspace(config.view_angle_min,config.view_angle_max,config.Nangles)
angles_t = torch.tensor(angles).type(config.torch_type).to(device)
operator_ET = ParallelBeamGeometry3DOpAngles_rectangular((config.n1,config.n2,config.n3), angles/180*np.pi, op_snr=np.inf, fact=1)


# TODO: implement more realistic PSF (and spatially varying?)
def PSF_generator(E,F,c,x):
    return E[0]


# Define the PSF
if config.sigma_PSF!=0:
    xx1 = np.linspace(-config.n1//2,config.n1//2,config.n1)
    xx2 = np.linspace(-config.n2//2,config.n2//2,config.n2)
    XX, YY = np.meshgrid(xx1,xx2)
    G = np.exp(-(XX**2+YY**2)/(2*config.sigma_PSF**2))
    supp = int(np.round(4*config.sigma_PSF))
    PSF = G[config.n1//2-supp//2:config.n1//2+supp//2,config.n2//2-supp//2:config.n2//2+supp//2]
    PSF /= PSF.sum()
    PSF_ext = np.zeros_like(G)
    PSF_ext[config.n1//2-supp//2:config.n1//2+supp//2,config.n2//2-supp//2:config.n2//2+supp//2] = G[config.n1//2-supp//2:config.n1//2+supp//2,config.n2//2-supp//2:config.n2//2+supp//2]
    PSF_ext_t = torch.tensor(PSF_ext).type(config.torch_type).to(device)
    PSF_ext_fft_t = torch.fft.fft2(PSF_ext_t).view(1,config.n1,config.n2)
else: 
    PSF = 0


# Define global and local deformations
affine_tr = []
local_tr = []
# TODO: add a model of local deformation as Gaussian blob
for i in range(config.Nangles*config.number_sub_projections):
    scaleX, scaleY, shiftX, shiftY, shearX, shearY, angle  = utils_deformation.generate_params_deformation(config.scale_min,
                config.scale_max,config.shift_min,config.shift_max,config.shear_min,config.shear_max,config.angle_min,config.angle_max)
    affine_tr.append(utils_deformation.AffineTransform(scaleX, scaleY, shiftX, shiftY, shearX, shearY, angle ).cuda())

    depl_ctr_pts = torch.randn(2,config.N_ctrl_pts_local_def[0],config.N_ctrl_pts_local_def[1]).to(device).type(config.torch_type)
    depl_ctr_pts[0] = depl_ctr_pts[0]/config.n1*config.sigma_local_def
    depl_ctr_pts[1] = depl_ctr_pts[1]/config.n2*config.sigma_local_def
    field = utils_deformation.deformation_field(depl_ctr_pts)
    local_tr.append(field)

    # Display the transformations
    if i < 10:
        nsr = (config.n1*4,config.n2*4)
        Nsp = (config.n1//20,config.n2//20) # number of Diracs in each direction
        supp = config.n1//70

        # Display local deformations
        utils_display.display_local(field,field_true=None,Npts=Nsp,img_path=config.path_save_data+"deformations/local_quiver_"+str(i),
                                    img_type='.png',scale=1,alpha=0.8,width=0.0015,wx=config.n1//2,wy=config.n2//2)

        # Display global deformations
        sp1 = np.array(np.floor(np.linspace(0,nsr[0],Nsp[0]+2)),dtype=int)[1:-1]
        sp2 = np.array(np.floor(np.linspace(0,nsr[1],Nsp[1]+2)),dtype=int)[1:-1]
        spx, spy = np.meshgrid(sp1,sp2)  
        xx1 = np.linspace(-nsr[0]/2,nsr[0]/2,nsr[0])
        xx2 = np.linspace(-nsr[1]/2,nsr[1]/2,nsr[1])
        XX, YY = np.meshgrid(xx1,xx2, indexing='ij')
        G = np.exp(-(XX**2+YY**2)/(2*(supp/3)**2))
        G[:nsr[0]//2-supp,:]=0
        G[nsr[0]//2+supp:,:]=0
        G[:,:nsr[1]//2-supp]=0
        G[:,nsr[1]//2+supp:]=0

        G /= G.sum()
        im_grid = np.zeros(nsr)
        im_grid[spx,spy] = 1
        im_grid = np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(G)*np.fft.fft2(im_grid))).real
        im_grid_t = torch.tensor(im_grid).to(device).type(config.torch_type)

        img_deform_global = utils_deformation.apply_deformation([affine_tr[-1]],im_grid_t.reshape(1,nsr[0],nsr[1]))

        tmp = img_deform_global.detach().cpu().numpy()[0].reshape(nsr)
        tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
        tmp = np.floor(255*tmp).astype(np.uint8)
        imageio.imwrite(config.path_save_data+"deformations/global_deformation_indiviudal_"+str(i)+".png",tmp)
    

projections_clean = operator_ET(V_t)
projections_clean = projections_clean[:,None].repeat(1,config.number_sub_projections,1,1).reshape(-1,config.n1,config.n2)


# add deformations
projections_deformed_global = utils_deformation.apply_deformation(affine_tr,projections_clean)
projections_deformed = utils_deformation.apply_local_deformation(local_tr,projections_deformed_global)

# add PSF
# TODO: implement spatial PSF
if config.sigma_PSF!=0:
    projections_deformed = torch.fft.fftshift(torch.fft.ifft2(PSF_ext_fft_t * torch.fft.fft2(projections_deformed,dim=(1,2))),(1,2)).real
    
# add noise
sigma_noise = data_generation.find_sigma_noise_t(config.SNR_value,projections_deformed)
projections_noisy = projections_deformed.clone() + torch.randn_like(projections_deformed)*sigma_noise
projections_noisy_no_deformed = projections_clean.clone() + torch.randn_like(projections_clean)*sigma_noise


np.save(config.path_save_data+"global_deformations.npy",affine_tr)
np.save(config.path_save_data+"local_deformations.npy",local_tr)
np.savez(config.path_save_data+"volume_and_projections.npz",projections_noisy=projections_noisy.detach().cpu().numpy(),projections_deformed=projections_deformed.detach().cpu().numpy(),projections_deformed_global=projections_deformed_global.detach().cpu().numpy(),projections_clean=projections_clean.detach().cpu().numpy(),PSF=PSF)
# np.savez(config.path_save_data+"parameters.npz",Nangles=config.Nangles,angle_min=config.angle_min,angle_max=config.angle_max,
#          n1=config.n1,n2=config.n2,n3=config.n3,SNR_value=config.SNR_value,sigma_PSF=config.sigma_PSF)

# save projections
for k in range(config.Nangles):
    tmp = projections_clean[k].detach().cpu().numpy()
    tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
    tmp = np.floor(255*tmp).astype(np.uint8)
    imageio.imwrite(config.path_save_data+"projections/clean_"+str(k)+".png",tmp)

    tmp = projections_deformed[k].detach().cpu().numpy()
    tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
    tmp = np.floor(255*tmp).astype(np.uint8)
    imageio.imwrite(config.path_save_data+"projections/deformed_"+str(k)+".png",tmp)

    tmp = projections_noisy[k].detach().cpu().numpy()
    tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
    tmp = np.floor(255*tmp).astype(np.uint8)
    imageio.imwrite(config.path_save_data+"projections/noisy_"+str(k)+".png",tmp)


projections_noisy_avg = projections_noisy.reshape(config.Nangles,-1,config.n1,config.n2).mean(1)
projections_noisy_no_deformed_avg =  projections_noisy_no_deformed.reshape(config.Nangles,-1,config.n1,config.n2).mean(1)
V_FBP = operator_ET.pinv(projections_noisy_avg).detach().requires_grad_(False)
V_FBP_no_deformed = operator_ET.pinv(projections_noisy_no_deformed_avg).detach().requires_grad_(False)
out = mrcfile.new(config.path_save_data+"V_FBP.mrc",np.moveaxis(V_FBP.detach().cpu().numpy().reshape(config.n1,config.n2,config.n3),2,0),overwrite=True)
out.close() 
out = mrcfile.new(config.path_save_data+"V_FBP_no_deformed.mrc",np.moveaxis(V_FBP_no_deformed.detach().cpu().numpy().reshape(config.n1,config.n2,config.n3),2,0),overwrite=True)
out.close() 
out = mrcfile.new(config.path_save_data+"V.mrc",np.moveaxis(V_t.detach().cpu().numpy().reshape(config.n1,config.n2,config.n3),2,0),overwrite=True)
out.close() 
out = mrcfile.new(config.path_save_data+"projections.mrc",projections_noisy.detach().cpu().numpy(),overwrite=True)
out.close() 


projections_noisy_ = projections_noisy.detach().cpu().numpy()*1
projections_noisy_no_deformed_ = projections_noisy_no_deformed.detach().cpu().numpy()*1
projections_noisy_reversed = projections_noisy_.max() - projections_noisy_ + 0.0001
projections_noisy_no_deformed_reversed = projections_noisy_no_deformed_.max() - projections_noisy_no_deformed_ + 0.0001
out = mrcfile.new(config.path_save_data+"projections_reversed.mrc",projections_noisy_reversed,overwrite=True)
out.close()
out = mrcfile.new(config.path_save_data+"projections_noisy_no_deformed.mrc",projections_noisy_no_deformed_,overwrite=True)
out.close()
out = mrcfile.new(config.path_save_data+"projections_noisy_no_deformed_reversed.mrc",projections_noisy_no_deformed_reversed,overwrite=True)
out.close()