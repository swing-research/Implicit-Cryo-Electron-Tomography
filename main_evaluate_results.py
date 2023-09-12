import numpy as np
import torch
import matplotlib.pyplot as plt
plt.ion()
import mrcfile
from ops.radon_3d_lib import ParallelBeamGeometry3DOpAngles_rectangular
import os
import imageio
from utils import utils_deformation, utils_display, utils_ricardo
import shutil

from configs.config_reconstruct_simulation import get_default_configs
config = get_default_configs()

import warnings
warnings.filterwarnings('ignore') 

# Introduction
'''
This script is used to compare our reconstruction with AreTomo and other standard methods
'''

use_cuda=torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
if torch.cuda.device_count()>1:
    torch.cuda.set_device(config.device_num)
np.random.seed(config.seed)
torch.manual_seed(config.seed)




if not os.path.exists(config.path_save):
    os.makedirs(config.path_save)
if not os.path.exists(config.path_save+"/evaluation/"):
    os.makedirs(config.path_save+"/evaluation/")
if not os.path.exists(config.path_save+"/evaluation/projections/"):
    os.makedirs(config.path_save+"/evaluation/projections/")
if not os.path.exists(config.path_save+"/evaluation/projections/ours/"):
    os.makedirs(config.path_save+"/evaluation/projections/ours/")
if not os.path.exists(config.path_save+"/evaluation/projections/AreTomo/"):
    os.makedirs(config.path_save+"/evaluation/projections/AreTomo/")
if not os.path.exists(config.path_save+"/evaluation/projections/FBP_no_deformed/"):
    os.makedirs(config.path_save+"/evaluation/projections/FBP_no_deformed/")
if not os.path.exists(config.path_save+"/evaluation/projections/FBP/"):
    os.makedirs(config.path_save+"/evaluation/projections/FBP/")
if not os.path.exists(config.path_save+"/evaluation/volumes/"):
    os.makedirs(config.path_save+"/evaluation/volumes/")
if not os.path.exists(config.path_save+"/evaluation/volumes/ours/"):
    os.makedirs(config.path_save+"/evaluation/volumes/ours/")
if not os.path.exists(config.path_save+"/evaluation/volumes/AreTomo/"):
    os.makedirs(config.path_save+"/evaluation/volumes/AreTomo/")
if not os.path.exists(config.path_save+"/evaluation/volumes/FBP_no_deformed/"):
    os.makedirs(config.path_save+"/evaluation/volumes/FBP_no_deformed/")
if not os.path.exists(config.path_save+"/evaluation/volumes/FBP/"):
    os.makedirs(config.path_save+"/evaluation/volumes/FBP/")
if not os.path.exists(config.path_save+"/evaluation/deformations/"):
    os.makedirs(config.path_save+"/evaluation/deformations/")
if not os.path.exists(config.path_save+"/evaluation/deformations/ours/"):
    os.makedirs(config.path_save+"/evaluation/deformations/ours/")
if not os.path.exists(config.path_save+"/evaluation/deformations/AreTomo/"):
    os.makedirs(config.path_save+"/evaluation/deformations/AreTomo/")
if not os.path.exists(config.path_save+"/evaluation/deformations/true/"):
    os.makedirs(config.path_save+"/evaluation/deformations/true/")


######################################################################################################
## Load data
######################################################################################################

data = np.load(config.path_save_data+"volume_and_projections.npz")
projections_noisy = torch.tensor(data['projections_noisy']).type(config.torch_type).to(device)
projections_deformed = torch.tensor(data['projections_deformed']).type(config.torch_type).to(device)
projections_deformed_global = torch.tensor(data['projections_deformed_global']).type(config.torch_type).to(device)
projections_clean = torch.tensor(data['projections_clean']).type(config.torch_type).to(device)

affine_tr = np.load(config.path_save_data+"global_deformations.npy",allow_pickle=True)
local_tr = np.load(config.path_save_data+"local_deformations.npy", allow_pickle=True)

V_t = torch.tensor(np.moveaxis(np.double(mrcfile.open(config.path_save_data+"V.mrc").data),0,2)).type(config.torch_type).to(device)
V_FBP_no_deformed_t = torch.tensor(np.moveaxis(np.double(mrcfile.open(config.path_save_data+"V_FBP_no_deformed.mrc").data),0,2)).type(config.torch_type).to(device)
V_FBP_t =  torch.tensor(np.moveaxis(np.double(mrcfile.open(config.path_save_data+"V_FBP.mrc").data),0,2)).type(config.torch_type).to(device)

# TODO: make script on how to use AreTomo and Isonet (and cryocare?) and call them here is files don't exist
# V_AreTomo = np.moveaxis(np.double(mrcfile.open(config.path_save_data+"V_AreTomo.mrc").data),0,2)
# V_AreTomo_t = torch.tensor(V_AreTomo).type(config.torch_type).to(device)
# V_ours_isonet = np.moveaxis(np.double(mrcfile.open(config.path_save_data+"V_ours+Isonet.mrc").data),0,2)
# V_AreTomo_isonet = np.moveaxis(np.double(mrcfile.open(config.path_save_data+"V_AreTomo+Isonet.mrc").data),0,2)

V = V_t.detach().cpu().numpy()
V_FBP = V_FBP_t.detach().cpu().numpy()
V_FBP_no_deformed = V_FBP_no_deformed_t.detach().cpu().numpy()

######################################################################################################
## Load and estimate our volume
######################################################################################################
## Load implicit network
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
    impl_volume = MLP(in_features= 1, 
                          hidden_features=config.hidden_size_volume, hidden_blocks= config.num_layers_volume, out_features=config.output_size_volume).to(device)

if(config.volume_model=="multi-resolution"):  
    import tinycudann as tcnn
    config_network = {"encoding": {
            'otype': 'Grid',
            'type': 'Hash',
            'n_levels': 9,
            'n_features_per_level': 2,
            'log2_hashmap_size': 20,
            'base_resolution': 8,
            'per_level_scale': 2,
            'interpolation': 'Smoothstep'
        },
        "network": {
            "otype": "FullyFusedMLP",   
            "activation": "ReLU",       
            "output_activation": "None",
            "n_neurons": config.hidden_size_volume,           
            "n_hidden_layers": config.num_layers_volume,       
        }       
        }
    impl_volume = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=1, encoding_config=config_network["encoding"], network_config=config_network["network"]).to(device)

num_param = sum(p.numel() for p in impl_volume.parameters() if p.requires_grad) 
print('---> Number of trainable parameters in volume net: {}'.format(num_param))

checkpoint = torch.load(os.path.join(config.path_save,'training','model_everything_joint_batch.pt'),map_location=device)
impl_volume.load_state_dict(checkpoint['implicit_volume'])
shift_ours = checkpoint['shift_est']
rot_ours = checkpoint['rot_est']
implicit_deformation_ours = checkpoint['local_deformation_network']

## Compute our model at same resolution than other volume
rays_scaling = torch.tensor(np.array(config.rays_scaling))[None,None,None].type(config.torch_type).to(device)
n1_eval, n2_eval, n3_eval = V.shape
x_lin1 = np.linspace(-1,1,n1_eval)*rays_scaling[0,0,0,0].item()/2+0.5
x_lin2 = np.linspace(-1,1,n2_eval)*rays_scaling[0,0,0,1].item()/2+0.5
XX, YY = np.meshgrid(x_lin1,x_lin2,indexing='ij')
grid2d = np.concatenate([XX.reshape(-1,1),YY.reshape(-1,1)],1)
grid2d_t = torch.tensor(grid2d).type(config.torch_type)
z_range = np.linspace(-1,1,n3_eval)*rays_scaling[0,0,0,2].item()*(n3_eval/n1_eval)/2+0.5
V_ours = np.zeros_like(V)
for zz, zval in enumerate(z_range):
    grid3d = np.concatenate([grid2d_t, zval*torch.ones((grid2d_t.shape[0],1))],1)
    grid3d_slice = torch.tensor(grid3d).type(config.torch_type).to(device)
    estSlice = impl_volume(grid3d_slice).detach().cpu().numpy().reshape(config.n1,config.n2)
    V_ours[:,:,zz] = estSlice
V_ours_t = torch.tensor(V_ours).type(config.torch_type).to(device)


## Compute FSCs
fsc_ours = utils_ricardo.FSC(V,V_ours)
fsc_FBP = utils_ricardo.FSC(V,V_FBP)
fsc_FBP_no_deformed = utils_ricardo.FSC(V,V_FBP_no_deformed)
# fsc_AreTomo = utils_ricardo.FSC(V,V_AreTomo)
# TODO: add Isonet etc
x_fsc = np.arange(fsc_FBP.shape[0])


plt.figure(1)
plt.clf()
plt.plot(x_fsc,fsc_ours,'b',label="ours")
# plt.plot(x_fsc,fsc_ours_post_process,'--b',label="ours post process")
# plt.plot(x_fsc,fsc_ours_isonet,'--b',label="ours + Isonet")
# plt.plot(x_fsc,fsc_AreTomo,'r',label="AreTomo")
plt.plot(x_fsc,fsc_FBP,'k',label="FBP")
plt.plot(x_fsc,fsc_FBP_no_deformed,'g',label="FBP no def.")
plt.legend()
plt.savefig(os.path.join(config.path_save,'evaluation','FSC.png'))
plt.savefig(os.path.join(config.path_save,'evaluation','FSC.pdf'))


fsc_arr = np.zeros((x_fsc.shape[0],7))
fsc_arr[:,0] = x_fsc
fsc_arr[:,1] = fsc_ours[:,0]
fsc_arr[:,2] = fsc_FBP[:,0]
fsc_arr[:,3] = fsc_FBP_no_deformed[:,0]
# fsc_arr[:,4] = fsc_AreTomo[:,0]
# fsc_arr[:,5] = fsc_FBP_est_deformed[:,0]
# fsc_arr[:,6] = fsc_ours_isonet[:,0]
header ='x,ours,FBP,FBP_no_deformed,AreTomo,FBP_est_deformed,ours_isonet'
np.savetxt(os.path.join(config.path_save,'evaluation','FSC.csv'),fsc_arr,header=header,delimiter=",",comments='')




#######################################################################################
## Generate projections
#######################################################################################
# Define angles and X-ray transform
angles = np.linspace(config.view_angle_min,config.view_angle_max,config.Nangles)
angles_t = torch.tensor(angles).type(config.torch_type).to(device)
operator_ET = ParallelBeamGeometry3DOpAngles_rectangular((config.n1,config.n2,config.n3), angles/180*np.pi, op_snr=np.inf, fact=1)

projections_ours = operator_ET(V_ours_t).detach().cpu().numpy()
projections_FBP = operator_ET(V_FBP_t).detach().cpu().numpy()
projections_FBP_no_deformed = operator_ET(V_FBP_no_deformed_t).detach().cpu().numpy()
# projections_AreTomo = operator_ET(V_AreTomo_t).detach().cpu().numpy()

out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","ours","projections.mrc"),projections_ours.astype(np.float32),overwrite=True)
out.close()
# out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","AreTomo","projections.mrc"),projections_AreTomo.astype(np.float32),overwrite=True)
# out.close()
out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","FBP","projections.mrc"),projections_FBP.astype(np.float32),overwrite=True)
out.close()
out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","FBP_no_deformed","projections.mrc"),projections_FBP_no_deformed.astype(np.float32),overwrite=True)
out.close()


for k in range(config.Nangles):
    tmp = projections_ours[k]
    tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
    tmp = np.floor(255*tmp).astype(np.uint8)
    imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"projections","ours","snapshot_{}.mrc".format(k)),tmp)
    # tmp = projections_AreTomo[k]
    # tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
    # tmp = np.floor(255*tmp).astype(np.uint8)
    # imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"projections","AreTomo","snapshot_{}.mrc".format(k)),tmp)
    tmp = projections_FBP[k]
    tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
    tmp = np.floor(255*tmp).astype(np.uint8)
    imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"projections","FBP","snapshot_{}.mrc".format(k)),tmp)
    tmp = projections_FBP_no_deformed[k]
    tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
    tmp = np.floor(255*tmp).astype(np.uint8)
    imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"projections","FBP_no_deformed","snapshot_{}.mrc".format(k)),tmp)

out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"volumes","ours","projections.mrc"),np.moveaxis(V_ours,2,0),overwrite=True)
out.close()
# out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"volumes","AreTomo","projections.mrc"),np.moveaxis(V_AreTomo,2,0),overwrite=True)
# out.close()
out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"volumes","FBP","projections.mrc"),np.moveaxis(V_FBP,2,0),overwrite=True)
out.close()
out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"volumes","FBP_no_deformed","projections.mrc"),np.moveaxis(V_FBP_no_deformed,2,0),overwrite=True)
out.close()







# # V_ours_post_process =  np.moveaxis(np.double(mrcfile.open("./results/Isonet_prior/V_est_post_process.mrc").data),0,2)
# # V_ours = np.load(path_save+"Vinith/volume_est.npy")
# # V_ours = np.load(path_save+"Vinith/vol_noTV_config_6_epcs_5k.npy")

# # AreTomo
# V_AreTomo =  np.moveaxis(np.double(mrcfile.open(path_save+"AreTomo"+"/areTomo_reconstruction.mrc").data),1,2)
# nv = V_AreTomo.shape # size of the loaded volume 
# V_AreTomo = V_AreTomo[nv[0]//2-n1//2:nv[0]//2+n1//2,nv[1]//2-n2//2:nv[1]//2+n2//2,nv[2]//2-n3//2:nv[2]//2+n3//2]
# V_AreTomo = V_AreTomo[:,:,::-1]

# projections_undeformed_est_from_obs =  np.double(mrcfile.open(path_save+"projections_undeformed_est_from_obs.mrc").data)
# projections_undeformed_est_from_obs_t = torch.tensor(projections_undeformed_est_from_obs).to(device)
# angles = np.linspace(-angle_bound,angle_bound,Nangles)
# operator_ET = ParallelBeamGeometry3DOpAngles_rectangular((n1,n2,n3), angles/180*np.pi, op_snr=np.inf, fact=1)
# V_FBP_est_deformed = operator_ET.pinv(projections_undeformed_est_from_obs_t)
# V_FBP_est_deformed = V_FBP_est_deformed.detach().cpu().numpy()

# # To remove?
# V_FBP /= V_FBP.sum()
# V_FBP_no_deformed_t /= V_FBP_no_deformed_t.sum()
# V_ours /= V_ours.sum()
# V_ours_isonet = (V_ours_isonet-V_ours_isonet.min())/(V_ours_isonet.max()-V_ours_isonet.min())
# V_ours_isonet /= V_ours_isonet.sum()
# V_AreTomo /= V_AreTomo.sum()
# V_t /= V_t.sum()
# V_FBP_est_deformed /= V_FBP_est_deformed.sum()


# ######################################################################################################
# ## FSC
# ######################################################################################################

# fsc_AreTomo = utils_ricardo.FSC(V_t,V_AreTomo)
# fsc_ours = utils_ricardo.FSC(V_t,V_ours)
# fsc_ours_isonet = utils_ricardo.FSC(V_t,V_ours_isonet)
# fsc_FBP = utils_ricardo.FSC(V_t,V_FBP)
# fsc_FBP_no_deformed = utils_ricardo.FSC(V_t,V_FBP_no_deformed_t)
# fsc_FBP_est_deformed = utils_ricardo.FSC(V_t,V_FBP_est_deformed)
# x_fsc = np.arange(fsc_FBP.shape[0])



# plt.figure(1)
# plt.clf()
# plt.plot(x_fsc,fsc_ours,'b',label="ours")
# # plt.plot(x_fsc,fsc_ours_post_process,'--b',label="ours post process")
# plt.plot(x_fsc,fsc_ours_isonet,'--b',label="ours + Isonet")
# plt.plot(x_fsc,fsc_AreTomo,'r',label="AreTomo")
# plt.plot(x_fsc,fsc_FBP,'k',label="FBP")
# plt.plot(x_fsc,fsc_FBP_no_deformed,'g',label="FBP no def.")
# plt.plot(x_fsc,fsc_FBP_est_deformed,'c',label="FBP est. def.")
# plt.legend()
# plt.savefig(os.path.join(path_save,'evaluation','FSC.png'))


# fsc_arr = np.zeros((x_fsc.shape[0],7))
# fsc_arr[:,0] = x_fsc
# fsc_arr[:,1] = fsc_ours[:,0]
# fsc_arr[:,2] = fsc_FBP[:,0]
# fsc_arr[:,3] = fsc_FBP_no_deformed[:,0]
# fsc_arr[:,4] = fsc_AreTomo[:,0]
# fsc_arr[:,5] = fsc_FBP_est_deformed[:,0]
# fsc_arr[:,6] = fsc_ours_isonet[:,0]

# header ='x,ours,FBP,FBP_no_deformed,AreTomo,FBP_est_deformed,ours_isonet'
# np.savetxt(os.path.join(path_save,'evaluation','FSC.csv'),fsc_arr,header=header,delimiter=",",comments='')




# from scipy.ndimage import gaussian_filter
# V_ours_post_process =  np.double(mrcfile.open("./results/Isonet_repro/test/tomo_est2.mrc").data)
# # V_ours_post_process =  np.moveaxis(np.double(mrcfile.open("./results/Isonet_prior/V_est_post_process.mrc").data),0,2)
# # V_ours_post_process =  np.moveaxis(np.double(mrcfile.open("./IsoNet/bin/corrected_tomos/V_est_corrected.mrc").data),0,2)
# # V_ours_post_process = gaussian_filter(V_ours_post_process,sigma=0)
# V_ours_post_process = (V_ours_post_process-V_ours_post_process.min())/(V_ours_post_process.max()-V_ours_post_process.min())
# V_ours_post_process /= V_ours_post_process.sum()
# fsc_ours_post_process = utils_ricardo.FSC(V_t,V_ours_post_process)
# plt.figure(1)
# plt.clf()
# plt.plot(x_fsc,fsc_ours,'b',label="ours")
# plt.plot(x_fsc,fsc_ours_post_process,'--b',label="ours post process")
# # plt.plot(x_fsc,fsc_ours_isonet,'--b',label="ours + Isonet")
# plt.plot(x_fsc,fsc_AreTomo,'r',label="AreTomo")
# plt.plot(x_fsc,fsc_FBP,'k',label="FBP")
# plt.plot(x_fsc,fsc_FBP_no_deformed,'g',label="FBP no def.")
# plt.plot(x_fsc,fsc_FBP_est_deformed,'c',label="FBP est. def.")
# plt.legend()




# fsc_5_ours = 1/x_fsc[np.where(fsc_ours[:,0]>0.5)][-1]
# fsc_5_FBP = 1/x_fsc[np.where(fsc_FBP[:,0]>0.5)][-1]
# fsc_5_FBP_no_deformed = 1/x_fsc[np.where(fsc_FBP_no_deformed[:,0]>0.5)][-1]
# fsc_5_AreTomo = 1/x_fsc[np.where(fsc_AreTomo[:,0]>0.5)][-1]
# fsc_5_est_def = 1/x_fsc[np.where(fsc_FBP_est_deformed[:,0]>0.5)][-1]

# fsc_143_ours = 1/x_fsc[np.where(fsc_ours[:,0]>0.143)][-1]
# fsc_143_FBP = 1/x_fsc[np.where(fsc_FBP[:,0]>0.143)][-1]
# fsc_143_FBP_no_deformed = 1/x_fsc[np.where(fsc_FBP_no_deformed[:,0]>0.143)][-1]
# fsc_143_AreTomo = 1/x_fsc[np.where(fsc_AreTomo[:,0]>0.143)][-1]
# fsc_143_est_def = 1/x_fsc[np.where(fsc_FBP_est_deformed[:,0]>0.143)][-1]

# print("FSC>0.5: Ours: {:2.2} -- FBP: {:2.2} -- FBP no deform: {:2.2} -- AreTomo: {:2.2} -- Est; def.: {:2.2}".format(fsc_5_ours,fsc_5_FBP,fsc_5_FBP_no_deformed,fsc_5_AreTomo,fsc_5_est_def))
# print("FSC>0.143: Ours: {:2.2} -- FBP: {:2.2} -- FBP no deform: {:2.2} -- AreTomo: {:2.2} -- Est. def.: {:2.2}".format(fsc_143_ours,fsc_143_FBP,fsc_143_FBP_no_deformed,fsc_143_AreTomo,fsc_143_est_def))


# ######################################################################################################
# ## Save slice of the volume
# ######################################################################################################

# kz_list = np.arange(0,180,10)
# for kz in kz_list:
#     plt.figure(2)
#     plt.subplot(1,5,1)
#     plt.imshow(V_t[:,:,kz])
#     plt.title('True')
#     plt.subplot(1,5,2)
#     plt.imshow(V_ours[:,:,kz])
#     plt.title('Ours')
#     plt.subplot(1,5,3)
#     plt.imshow(V_AreTomo[:,:,kz])
#     plt.title('AreTomo')
#     plt.subplot(1,5,4)
#     plt.imshow(V_FBP[:,:,kz])
#     plt.title('FBP')
#     plt.subplot(1,5,5)
#     plt.imshow(V_FBP_no_deformed_t[:,:,kz])
#     plt.title('FBP no def.')
#     plt.savefig(os.path.join(path_save,'evaluation','slice_{}.png'.format(kz)))




# if not os.path.exists(path_save+"evaluation/tmp_vol_true/"):
#     os.makedirs(path_save+"evaluation/tmp_vol_true/")
# else:
#     shutil.rmtree(path_save+"evaluation/tmp_vol_true/")
#     os.makedirs(path_save+"evaluation/tmp_vol_true/")
# for k in range(n3):
#     tmp = V_t[:,:,k]
#     tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
#     tmp = np.floor(255*tmp).astype(np.uint8)
#     imageio.imwrite(os.path.join(path_save,'evaluation','tmp_vol_true','est_ep_'+str(k).zfill(5)+'.png'),tmp)
# file_png = os.path.join(path_save,'evaluation','tmp_vol_true','*.png')
# file_save = os.path.join(path_save,'evaluation','annimation_vol_true')
# os.system("convert -delay 10 -loop 0 "+ file_png +" "+ file_save+".gif")                                                                                                             
# os.system("ffmpeg -i " + file_save+ '.gif' +" -movflags faststart -pix_fmt yuv420p -y "+file_save+".mp4")



# if not os.path.exists(path_save+"evaluation/tmp_vol_ours/"):
#     os.makedirs(path_save+"evaluation/tmp_vol_ours/")
# else:
#     shutil.rmtree(path_save+"evaluation/tmp_vol_ours/")
#     os.makedirs(path_save+"evaluation/tmp_vol_ours/")
# for k in range(n3):
#     tmp = V_ours[:,:,k]
#     tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
#     # tmp = (V_ours[:,:,k].max()/V_ours.max())*tmp
#     tmp = np.floor(255*tmp).astype(np.uint8)
#     imageio.imwrite(os.path.join(path_save,'evaluation','tmp_vol_ours','est_ep_'+str(k).zfill(5)+'.png'),tmp)
# file_png = os.path.join(path_save,'evaluation','tmp_vol_ours','*.png')
# file_save = os.path.join(path_save,'evaluation','annimation_vol_ours')
# os.system("convert -delay 10 -loop 0 "+ file_png +" "+ file_save+".gif")                                                                                                             
# os.system("ffmpeg -i " + file_save+ '.gif' +" -movflags faststart -pix_fmt yuv420p -y "+file_save+".mp4")

# if not os.path.exists(path_save+"evaluation/tmp_vol_AreTomo/"):
#     os.makedirs(path_save+"evaluation/tmp_vol_AreTomo/")
# else:
#     shutil.rmtree(path_save+"evaluation/tmp_vol_AreTomo/")
#     os.makedirs(path_save+"evaluation/tmp_vol_AreTomo/")
# for k in range(n3):
#     tmp = V_AreTomo[:,:,k]
#     tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
#     tmp = np.floor(255*tmp).astype(np.uint8)
#     imageio.imwrite(os.path.join(path_save,'evaluation','tmp_vol_AreTomo','est_ep_'+str(k).zfill(5)+'.png'),tmp)
# file_png = os.path.join(path_save,'evaluation','tmp_vol_AreTomo','*.png')
# file_save = os.path.join(path_save,'evaluation','annimation_vol_AreTomo')
# os.system("convert -delay 10 -loop 0 "+ file_png +" "+ file_save+".gif")                                                                                                             
# os.system("ffmpeg -i " + file_save+ '.gif' +" -movflags faststart -pix_fmt yuv420p -y "+file_save+".mp4")


# if not os.path.exists(path_save+"evaluation/tmp_vol_FBP/"):
#     os.makedirs(path_save+"evaluation/tmp_vol_FBP/")
# else:
#     shutil.rmtree(path_save+"evaluation/tmp_vol_FBP/")
#     os.makedirs(path_save+"evaluation/tmp_vol_FBP/")
# for k in range(n3):
#     tmp = V_FBP[:,:,k]
#     tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
#     tmp = np.floor(255*tmp).astype(np.uint8)
#     imageio.imwrite(os.path.join(path_save,'evaluation','tmp_vol_FBP','est_ep_'+str(k).zfill(5)+'.png'),tmp)
# file_png = os.path.join(path_save,'evaluation','tmp_vol_FBP','*.png')
# file_save = os.path.join(path_save,'evaluation','annimation_vol_FBP')
# os.system("convert -delay 10 -loop 0 "+ file_png +" "+ file_save+".gif")                                                                                                             
# os.system("ffmpeg -i " + file_save+ '.gif' +" -movflags faststart -pix_fmt yuv420p -y "+file_save+".mp4")


# if not os.path.exists(path_save+"evaluation/tmp_vol_FBP_no_deformed/"):
#     os.makedirs(path_save+"evaluation/tmp_vol_FBP_no_deformed/")
# else:
#     shutil.rmtree(path_save+"evaluation/tmp_vol_FBP_no_deformed/")
#     os.makedirs(path_save+"evaluation/tmp_vol_FBP_no_deformed/")
# for k in range(n3):
#     tmp = V_FBP_no_deformed_t[:,:,k]
#     tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
#     tmp = np.floor(255*tmp).astype(np.uint8)
#     imageio.imwrite(os.path.join(path_save,'evaluation','tmp_vol_FBP_no_deformed','est_ep_'+str(k).zfill(5)+'.png'),tmp)
# file_png = os.path.join(path_save,'evaluation','tmp_vol_FBP_no_deformed','*.png')
# file_save = os.path.join(path_save,'evaluation','annimation_vol_FBP_no_deformed')
# os.system("convert -delay 10 -loop 0 "+ file_png +" "+ file_save+".gif")                                                                                                             
# os.system("ffmpeg -i " + file_save+ '.gif' +" -movflags faststart -pix_fmt yuv420p -y "+file_save+".mp4")


# if not os.path.exists(path_save+"evaluation/tmp_vol_FBP_est_deformed/"):
#     os.makedirs(path_save+"evaluation/tmp_vol_FBP_est_deformed/")
# else:
#     shutil.rmtree(path_save+"evaluation/tmp_vol_FBP_est_deformed/")
#     os.makedirs(path_save+"evaluation/tmp_vol_FBP_est_deformed/")
# for k in range(n3):
#     tmp = V_FBP_est_deformed[:,:,k]
#     tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
#     tmp = np.floor(255*tmp).astype(np.uint8)
#     imageio.imwrite(os.path.join(path_save,'evaluation','tmp_vol_FBP_est_deformed','est_ep_'+str(k).zfill(5)+'.png'),tmp)
# file_png = os.path.join(path_save,'evaluation','tmp_vol_FBP_est_deformed','*.png')
# file_save = os.path.join(path_save,'evaluation','annimation_vol_FBP_est_deformed')
# os.system("convert -delay 10 -loop 0 "+ file_png +" "+ file_save+".gif")                                                                                                             
# os.system("ffmpeg -i " + file_save+ '.gif' +" -movflags faststart -pix_fmt yuv420p -y "+file_save+".mp4")




# ######################################################################################################
# ## Movie of reconstructed projections
# #####################################################################################################
# proj_ours =  np.double(mrcfile.open(path_save+"projections_undeformed_est.mrc").data)
# proj_AreTomo =  np.double(mrcfile.open(path_save+"AreTomo/areTomo_alligned.mrc").data)



# if not os.path.exists(path_save+"evaluation/tmp_proj_ours/"):
#     os.makedirs(path_save+"evaluation/tmp_proj_ours/")
# else:
#     shutil.rmtree(path_save+"evaluation/tmp_proj_ours/")
#     os.makedirs(path_save+"evaluation/tmp_proj_ours/")
# for k in range(Nangles):
#     tmp = proj_ours[k]
#     tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
#     tmp = np.floor(255*tmp).astype(np.uint8)
#     imageio.imwrite(os.path.join(path_save,'evaluation','tmp_proj_ours','est_ep_'+str(k).zfill(5)+'.png'),tmp)
# file_png = os.path.join(path_save,'evaluation','tmp_proj_ours','*.png')
# file_save = os.path.join(path_save,'evaluation','annimation_proj_ours')
# os.system("convert -delay 10 -loop 0 "+ file_png +" "+ file_save+".gif")                                                                                                             
# os.system("ffmpeg -i " + file_save+ '.gif' +" -movflags faststart -pix_fmt yuv420p -y "+file_save+".mp4")


# if not os.path.exists(path_save+"evaluation/tmp_proj_AreTomo/"):
#     os.makedirs(path_save+"evaluation/tmp_proj_AreTomo/")
# else:
#     shutil.rmtree(path_save+"evaluation/tmp_proj_AreTomo/")
#     os.makedirs(path_save+"evaluation/tmp_proj_AreTomo/")
# for k in range(Nangles):
#     tmp = proj_AreTomo[k]
#     tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
#     tmp = np.floor(255*tmp).astype(np.uint8)
#     imageio.imwrite(os.path.join(path_save,'evaluation','tmp_proj_AreTomo','est_ep_'+str(k).zfill(5)+'.png'),tmp)
# file_png = os.path.join(path_save,'evaluation','tmp_proj_AreTomo','*.png')
# file_save = os.path.join(path_save,'evaluation','annimation_proj_AreTomo')
# os.system("convert -delay 10 -loop 0 "+ file_png +" "+ file_save+".gif")                                                                                                             
# os.system("ffmpeg -i " + file_save+ '.gif' +" -movflags faststart -pix_fmt yuv420p -y "+file_save+".mp4")


# if not os.path.exists(path_save+"evaluation/tmp_proj_clean/"):
#     os.makedirs(path_save+"evaluation/tmp_proj_clean/")
# else:
#     shutil.rmtree(path_save+"evaluation/tmp_proj_clean/")
#     os.makedirs(path_save+"evaluation/tmp_proj_clean/")
# for k in range(Nangles):
#     tmp = projections_clean[k].detach().cpu().numpy()
#     tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
#     tmp = np.floor(255*tmp).astype(np.uint8)
#     imageio.imwrite(os.path.join(path_save,'evaluation','tmp_proj_clean','est_ep_'+str(k).zfill(5)+'.png'),tmp)
# file_png = os.path.join(path_save,'evaluation','tmp_proj_clean','*.png')
# file_save = os.path.join(path_save,'evaluation','annimation_proj_clean')
# os.system("convert -delay 10 -loop 0 "+ file_png +" "+ file_save+".gif")                                                                                                             
# os.system("ffmpeg -i " + file_save+ '.gif' +" -movflags faststart -pix_fmt yuv420p -y "+file_save+".mp4")

# if not os.path.exists(path_save+"evaluation/tmp_proj_deformed/"):
#     os.makedirs(path_save+"evaluation/tmp_proj_deformed/")
# else:
#     shutil.rmtree(path_save+"evaluation/tmp_proj_deformed/")
#     os.makedirs(path_save+"evaluation/tmp_proj_deformed/")
# for k in range(Nangles):
#     tmp = projections_deformed[k].detach().cpu().numpy()
#     tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
#     tmp = np.floor(255*tmp).astype(np.uint8)
#     imageio.imwrite(os.path.join(path_save,'evaluation','tmp_proj_deformed','est_ep_'+str(k).zfill(5)+'.png'),tmp)
# file_png = os.path.join(path_save,'evaluation','tmp_proj_deformed','*.png')
# file_save = os.path.join(path_save,'evaluation','annimation_proj_deformed')
# os.system("convert -delay 10 -loop 0 "+ file_png +" "+ file_save+".gif")                                                                                                             
# os.system("ffmpeg -i " + file_save+ '.gif' +" -movflags faststart -pix_fmt yuv420p -y "+file_save+".mp4")

# if not os.path.exists(path_save+"evaluation/tmp_proj_noisy/"):
#     os.makedirs(path_save+"evaluation/tmp_proj_noisy/")
# else:
#     shutil.rmtree(path_save+"evaluation/tmp_proj_noisy/")
#     os.makedirs(path_save+"evaluation/tmp_proj_noisy/")
# for k in range(Nangles):
#     tmp = projections_noisy[k].detach().cpu().numpy()
#     tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
#     tmp = np.floor(255*tmp).astype(np.uint8)
#     imageio.imwrite(os.path.join(path_save,'evaluation','tmp_proj_noisy','est_ep_'+str(k).zfill(5)+'.png'),tmp)
# file_png = os.path.join(path_save,'evaluation','tmp_proj_noisy','*.png')
# file_save = os.path.join(path_save,'evaluation','annimation_proj_noisy')
# os.system("convert -delay 10 -loop 0 "+ file_png +" "+ file_save+".gif")                                                                                                             
# os.system("ffmpeg -i " + file_save+ '.gif' +" -movflags faststart -pix_fmt yuv420p -y "+file_save+".mp4")


# if not os.path.exists(path_save+"evaluation/tmp_proj_undeformed_est_from_obs/"):
#     os.makedirs(path_save+"evaluation/tmp_proj_undeformed_est_from_obs/")
# else:
#     shutil.rmtree(path_save+"evaluation/tmp_proj_undeformed_est_from_obs/")
#     os.makedirs(path_save+"evaluation/tmp_proj_undeformed_est_from_obs/")
# for k in range(Nangles):
#     tmp = projections_undeformed_est_from_obs[k]
#     tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
#     tmp = np.floor(255*tmp).astype(np.uint8)
#     imageio.imwrite(os.path.join(path_save,'evaluation','tmp_proj_undeformed_est_from_obs','est_ep_'+str(k).zfill(5)+'.png'),tmp)
# file_png = os.path.join(path_save,'evaluation','tmp_proj_undeformed_est_from_obs','*.png')
# file_save = os.path.join(path_save,'evaluation','annimation_proj_undeformed_est_from_obs')
# os.system("convert -delay 10 -loop 0 "+ file_png +" "+ file_save+".gif")                                                                                                             
# os.system("ffmpeg -i " + file_save+ '.gif' +" -movflags faststart -pix_fmt yuv420p -y "+file_save+".mp4")




# ######################################################################################################
# ## Load deformations
# ######################################################################################################

# # True
# affine_tr
# local_tr

# # Ours
# checkpoint = torch.load(os.path.join(path_save, 'model_everything_joint_batch.pt'),map_location=device)
# shift_ours = checkpoint['shift_est']
# rot_ours = checkpoint['rot_est']
# implicit_deformation_ours = checkpoint['local_deformation_network']

# # AreTomo
# import csv
# path_angles = os.path.join(path_save, 'AreTomo','areTomo_alligned.aln')
# file = open(path_angles)
# csvreader = csv.reader(file)
# rot_AreTomo = []
# shift_AreTomo = []
# local_AreTomo = np.zeros((Nangles,10**2,4))
# for k, r in enumerate(csvreader):
#     if k>2 and k<Nangles+3:
#         rot_AreTomo.append(np.float(r[0].split()[1]))
#         shift_AreTomo.append([np.float(r[0].split()[3]),np.float(r[0].split()[4])])
#     if k>Nangles+3:
#         i1 = np.int8(r[0].split()[0])
#         i2 = np.int8(r[0].split()[1])
#         local_AreTomo[i1,i2,0] = np.float(r[0].split()[2])
#         local_AreTomo[i1,i2,1] = np.float(r[0].split()[3])
#         local_AreTomo[i1,i2,2] = np.float(r[0].split()[4])
#         local_AreTomo[i1,i2,3] = np.float(r[0].split()[5])

# from scipy.interpolate import griddata
# N_ctrl_pts = 10
# xx1 = np.linspace(-n1//2,n1//2,N_ctrl_pts)
# xx2 = np.linspace(-n2//2,n2//2,N_ctrl_pts)
# XX_t, YY_t = np.meshgrid(xx1,xx2,indexing='ij')
# XX_t = XX_t[:,:,None]
# YY_t = YY_t[:,:,None]
# grid_interp = np.concatenate([XX_t,YY_t],2).reshape(-1,2)

# # Possibly ignoring rotations or other (2 first parameters)
# implicit_deformation_AreTomo = []
# for k in range(Nangles):
#     values_x = griddata(local_AreTomo[k][:,:2],local_AreTomo[k][:,2],grid_interp,method='cubic',fill_value=0,rescale=True)
#     values_y = griddata(local_AreTomo[k][:,:2],local_AreTomo[k][:,3],grid_interp,method='cubic',fill_value=0,rescale=True)
#     depl_ctr_pts_net = np.concatenate([values_x[None],values_y[None]],0).reshape(2,N_ctrl_pts,N_ctrl_pts)
#     depl_ctr_pts_net = torch.tensor(depl_ctr_pts_net/n1).to(device).type(torch_type)


#     # depl_ctr_pts_net = local_AreTomo[k].reshape(10,10,4)/n1
#     # depl_ctr_pts_net = torch.tensor(depl_ctr_pts_net[:,:,2:]).to(device).type(torch_type)
#     # depl_ctr_pts_net = depl_ctr_pts_net.permute(2,0,1)
#     field = utils_deformation.deformation_field(depl_ctr_pts_net.clone())
#     implicit_deformation_AreTomo.append(field)

# shift_est_AreTomo = np.array(shift_AreTomo)/n1
# rot_est_AreTomo = np.array(rot_AreTomo)*np.pi/180


# ## Compute errors
# shift_true = np.zeros((Nangles,2))
# rot_true = np.zeros((Nangles))
# shift_est_ours = np.zeros((Nangles,2))
# rot_est_ours = np.zeros((Nangles))
# for k in range(Nangles):
#     shift_true[k,0] = affine_tr[k].shiftX.detach().cpu().numpy()
#     shift_true[k,1] = affine_tr[k].shiftY.detach().cpu().numpy()
#     rot_true[k] = affine_tr[k].angle.detach().cpu().numpy()
#     shift_est_ours[k] = shift_ours[k].shifts_arr[0].detach().cpu().numpy()
#     rot_est_ours[k] = rot_ours[k].thetas.detach().cpu().numpy()

# err_local_ours = np.zeros(Nangles)
# err_local_AreTomo = np.zeros(Nangles)
# err_local_init = np.zeros(Nangles)
# for k in range(Nangles):
#     grid_correction_true = local_tr[k](grid_class.grid2d_t).detach().cpu().numpy()
#     grid_correction_est_ours = implicit_deformation_ours[k](grid_class.grid2d_t).detach().cpu().numpy()
#     tmp = np.abs(grid_correction_true-grid_correction_est_ours)
#     err_local_ours[k] = (0.5*n1*tmp[:,0]+0.5*n2*tmp[:,1]).mean()
#     grid_correction_est_AreTomo = implicit_deformation_AreTomo[k](grid_class.grid2d_t).detach().cpu().numpy()
#     tmp = np.abs(grid_correction_true-grid_correction_est_AreTomo)
#     err_local_AreTomo[k] = (0.5*n1*tmp[:,0]+0.5*n2*tmp[:,1]).mean()
#     tmp = np.abs(grid_correction_true)
#     err_local_init[k] = (0.5*n1*tmp[:,0]+0.5*n2*tmp[:,1]).mean()


# err_shift_ours = np.abs(shift_est_ours-shift_true)
# err_shift_ours = 0.5*n1*err_shift_ours[:,0] + 0.5*n2*err_shift_ours[:,1]
# err_shift_AreTomo = np.abs(shift_est_AreTomo-shift_true)
# err_shift_AreTomo = 0.5*n1*err_shift_AreTomo[:,0] + 0.5*n2*err_shift_AreTomo[:,1]
# err_shift_init = np.abs(shift_true)
# err_shift_init = 0.5*n1*err_shift_init[:,0] + 0.5*n2*err_shift_init[:,1]

# err_rot_ours = 180/np.pi*np.abs(rot_est_ours - rot_true)
# err_rot_AreTomo = 180/np.pi*np.abs(rot_est_AreTomo - rot_true)
# err_rot_init = 180/np.pi*np.abs(rot_true)


# print("AreTomo        || Err shift: {:2.3}+/-{:2.3} -- Err rot: {:2.3}+/-{:2.3} -- Err local: {:2.3}+/-{:2.3}".format(err_shift_AreTomo.mean(),
#                             err_shift_AreTomo.std(),err_rot_AreTomo.mean(),err_rot_AreTomo.std(),err_local_AreTomo.mean(),err_local_AreTomo.std()))
# print("Ours           || Err shift: {:2.3}+/-{:2.3} -- Err rot: {:2.3}+/-{:2.3} -- Err local: {:2.3}+/-{:2.3}".format(err_shift_ours.mean(),
#                             err_shift_ours.std(),err_rot_ours.mean(),err_rot_ours.std(),err_local_ours.mean(),err_local_ours.std()))
# print("initialization || Err shift: {:2.3}+/-{:2.3} -- Err rot: {:2.3}+/-{:2.3} -- Err local: {:2.3}+/-{:2.3}".format(err_shift_init.mean(),
#                             err_shift_init.std(),err_rot_init.mean(),err_rot_init.std(),err_local_init.mean(),err_local_init.std()))


# ## Display deformations
# utils_display.display_local_movie(implicit_deformation_ours,field_true=local_tr,Npts=(20,20),
#                                                       img_path=path_save+"evaluation/deformations/ours/",img_type='.png',
#                                                       scale=1/10,alpha=0.8,width=0.002,legend1='Ours',legend2='True')
# utils_display.display_local_movie(implicit_deformation_AreTomo,field_true=local_tr,Npts=(20,20),
#                                                       img_path=path_save+"evaluation/deformations/AreTomo/",img_type='.png',
#                                                       scale=1/10,alpha=0.8,width=0.002,legend1='AreTomo',legend2='True')






# # V_AreTomo2 = np.zeros_like(V_AreTomo)
# # # V_AreTomo2 = np.copy(V_AreTomo)

# # # First compensate shift
# # s = -np.flip(np.floor(np.mean(shift_AreTomo,0)).astype(np.int8))
# # if s[0]<0:
# #     V_AreTomo2[:s[0],:,:] = V_AreTomo[-s[0]:,:,:]
# # elif s[0]>0:
# #     V_AreTomo2[s[0]:,:,:] = V_AreTomo[:-s[0],:,:]
# # if s[1]<0:
# #     V_AreTomo2[:,:s[1],:] = V_AreTomo[:,-s[1]:,:]
# # elif s[1]>0:
# #     V_AreTomo2[:,s[1]:,:] = V_AreTomo[:,:-s[1],:]

# # # second compensate rotation
# # from skimage.transform import rotate
# # for kk in range(n3):
# #     V_AreTomo2[:,:,kk] = rotate(V_AreTomo2[:,:,kk],+rot_AreTomo.mean())


# # TODO: report errors


# kz=70
# plt.figure(4)
# plt.subplot(2,2,1)
# plt.imshow((V_t)[:,:,kz])
# plt.colorbar()
# plt.subplot(2,2,2)
# plt.imshow((V_AreTomo2-V_t)[:,:,kz])
# plt.subplot(2,2,3)
# plt.imshow((V_AreTomo)[:,:,kz])
# plt.colorbar()
# plt.subplot(2,2,4)
# plt.imshow((V_AreTomo2)[:,:,kz])
# plt.colorbar()



# fsc_AreTomo2 = utils_ricardo.FSC(V_t,V_AreTomo2)

# plt.figure(1)
# plt.clf()
# plt.plot(x_fsc,fsc_ours,'b',label="ours")
# plt.plot(x_fsc,fsc_AreTomo,'r',label="AreTomo")
# plt.plot(x_fsc,fsc_AreTomo2,'--r',label="AreTomo2")
# plt.plot(x_fsc,fsc_FBP,'k',label="FBP")
# plt.plot(x_fsc,fsc_FBP_no_deformed,'g',label="FBP no def.")
# plt.legend()
# plt.savefig(os.path.join(path_save,'evaluation','FSC.png'))








