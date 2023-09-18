from skimage.transform import resize

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
import pandas as pd
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
if not os.path.exists(config.path_save+"/evaluation/projections/Etomo/"):
    os.makedirs(config.path_save+"/evaluation/projections/Etomo/")   
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
if not os.path.exists(config.path_save+"/evaluation/volumes/Etomo/"):
    os.makedirs(config.path_save+"/evaluation/volumes/Etomo/")
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
if not os.path.exists(config.path_save+"/evaluation/deformations/Etomo/"):
    os.makedirs(config.path_save+"/evaluation/deformations/Etomo/")

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

# If the file exist evaluate the reconstruction
eval_AreTomo = False
eval_ETOMO = False
if os.path.exists(config.path_save_data+"areTomo_reconstruction.mrc"):
    eval_AreTomo = True
if(os.path.exists(config.path_save_data+"etomo_reconstruction.mrc")):
    eval_ETOMO = True

if(eval_AreTomo):
    V_AreTomo = np.flip(np.moveaxis(np.double(mrcfile.open(config.path_save_data+"areTomo_reconstruction.mrc").data),1,2),2).copy()
    V_AreTomo_t = torch.tensor(V_AreTomo).type(config.torch_type).to(device)

if(eval_ETOMO):
    V_Etomo = np.flip(np.moveaxis(np.double(mrcfile.open(config.path_save_data+"etomo_reconstruction.mrc").data),1,2),2).copy()
    V_Etomo_t = torch.tensor(V_Etomo).type(config.torch_type).to(device)

# V_ours_isonet = np.moveaxis(np.double(mrcfile.open(config.path_save_data+"V_ours+Isonet.mrc").data),0,2)
# V_AreTomo_isonet = np.moveaxis(np.double(mrcfile.open(config.path_save_data+"V_AreTomo+Isonet.mrc").data),0,2)

V = V_t.detach().cpu().numpy()
V_FBP = V_FBP_t.detach().cpu().numpy()
V_FBP_no_deformed = V_FBP_no_deformed_t.detach().cpu().numpy()
if(eval_AreTomo):
    V_AreTomo = V_AreTomo_t.detach().cpu().numpy()
if(eval_ETOMO):
    V_Etomo = V_Etomo_t.detach().cpu().numpy()

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
#######################################################################################
## Compare the error between the true and estimated deformation
#######################################################################################

x_shifts = np.zeros(config.Nangles)
y_shifts = np.zeros(config.Nangles)
inplane_rotation = np.zeros(config.Nangles)

# Extract the true deformations
for index ,affine_transform in enumerate(affine_tr):

    x_shifts[index] = affine_transform.shiftX.item()
    y_shifts[index] = affine_transform.shiftY.item()
    inplane_rotation[index] = affine_transform.angle.item()*180/np.pi

#Extract the estimated deformations for the network
x_shifts_ours = np.zeros(config.Nangles)
y_shifts_ours = np.zeros(config.Nangles)
inplane_rotation_ours = np.zeros(config.Nangles)

for index, (shift_net, rot_net) in enumerate(zip(shift_ours,rot_ours)):
    x_shifts_ours[index] = shift_net.shifts_arr[0,0].item()
    y_shifts_ours[index] = shift_net.shifts_arr[0,1].item()
    inplane_rotation_ours[index] = rot_net.thetas.item()*180/np.pi


# Compute the error between the true and estimated deformation

error_x_shifts_ours = np.around(np.abs(x_shifts-x_shifts_ours).mean()*n1_eval,decimals=4)
error_y_shifts_ours = np.around(np.abs(y_shifts-y_shifts_ours).mean()*n1_eval,decimals=4)
error_inplane_rotation_ours =np.around( np.abs(inplane_rotation-inplane_rotation_ours
                                               ).mean(),decimals=4)


# Save the avg errors in a csv file with rownames: ours, AreTomo, Etomo
error_arr = pd.DataFrame(columns=['Method','x_shifts','y_shifts','inplane_rotation'])
# Include the avg absolute error in pixels in the table


x_mean_shift = np.around(np.abs(x_shifts).mean()*n1_eval,decimals=4)
y_mean_shift = np.around(np.abs(y_shifts).mean()*n1_eval,decimals=4)
inplane_mean_rotation = np.around(np.abs(inplane_rotation).mean(),decimals=4)
error_arr.loc[0] = ['Observation',x_mean_shift,y_mean_shift,inplane_mean_rotation]
error_arr.loc[1] = ['ours',error_x_shifts_ours,error_y_shifts_ours,error_inplane_rotation_ours]
# TODO: move this to a correct location?
def extract_angle(rot_matrix):
    """
    Extract the angle from a rotation matrix in degrees
    """
    if rot_matrix[0,0] == 1:
        angle = 0
    elif rot_matrix[0,0] == -1:
        angle = 180
    else:
        angle = np.arctan2(rot_matrix[1,0], rot_matrix[0,0])*180/np.pi
    return angle
if eval_ETOMO:
    # Extract the estimated deformations for etomo
    ETOMO_FILENAME = 'projections.xf'
    etomodata = pd.read_csv(config.path_save+ETOMO_FILENAME, sep='\s+', header=None)
    etomodata.columns = ['a11', 'a12', 'a21', 'a22','y','x']
    x_shifts_etomo = etomodata['x'].values
    y_shifts_etomo = etomodata['y'].values
    inplane_rotation_etomo = np.zeros(config.Nangles)

    for index in range(len(etomodata)):
        etomo_rotation_matrix = np.array([[etomodata['a11'][index],etomodata['a12'][index]],
                               [etomodata['a21'][index],etomodata['a22'][index]]])
        inplane_rotation_etomo[index] = extract_angle(etomo_rotation_matrix)

    # The estimates are already in pixelsb
    error_x_shifts_etomo = np.around(np.abs(x_shifts*n1_eval-x_shifts_etomo).mean(),decimals=4)
    error_y_shifts_etomo = np.around(np.abs(y_shifts*n1_eval-y_shifts_etomo).mean(),decimals=4)
    error_inplane_rotation_etomo = np.around(np.abs(inplane_rotation-
                                                    inplane_rotation_etomo).mean(),decimals=4)
    error_arr.loc[2] = ['Etomo',error_x_shifts_etomo,error_y_shifts_etomo,
                        error_inplane_rotation_etomo]
    
if eval_AreTomo:
    # Extract the estimated deformations for AreTomo

    ARETOMO_FILENAME = 'projections.aln'
    with open(config.path_save+ARETOMO_FILENAME, 'r',encoding="utf-8") as file:
        lines = file.readlines()

    comments = []
    affine_transforms_aretomo = []
    local_transforms_aretomo = []

    affine_flag = True
    for line in lines:
        if line.startswith('#'):
            comments.append(line.strip())  # Strip removes leading/trailing whitespace
            if line.startswith('# Local Alignment'):
                affine_flag = False
        else:
            if affine_flag:
                affine_transforms_aretomo.append(line.strip().split())
            else:
                local_transforms_aretomo.append(line.strip().split())  

    aretomo_data = pd.DataFrame(affine_transforms_aretomo, 
                                columns=["SEC", "ROT", "GMAG", "TX", 
                                                          "TY", "SMEAN", "SFIT", "SCALE",
                                                            "BASE", "TILT"])
    
    x_shifts_aretomo = aretomo_data['TX'].values.astype(np.float32)
    y_shifts_aretomo = aretomo_data['TY'].values.astype(np.float32)
    inplane_rotation_aretomo = aretomo_data['ROT'].values.astype(np.float32)

    # The estimates are already in pixels
    error_x_shifts_aretomo = np.around(np.abs(x_shifts*n1_eval-x_shifts_aretomo).mean(),decimals=4)
    error_y_shifts_aretomo = np.around(np.abs(y_shifts*n1_eval-y_shifts_aretomo).mean(),decimals=4)
    error_inplane_rotation_aretomo = np.around(np.abs(inplane_rotation-
                                                    inplane_rotation_aretomo).mean(),decimals=4)
    
    error_arr.loc[3] = ['AreTomo',error_x_shifts_aretomo,error_y_shifts_aretomo,
                        error_inplane_rotation_aretomo]


# save the error in a csv file
error_arr.to_csv(os.path.join(config.path_save,'evaluation'+'/affine_error.csv'),index=False)

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
if(eval_AreTomo):
    fsc_AreTomo = utils_ricardo.FSC(V,V_AreTomo)
if(eval_ETOMO):
    fsc_Etomo = utils_ricardo.FSC(V,V_Etomo)

# TODO: add Isonet etc
x_fsc = np.arange(fsc_FBP.shape[0])


plt.figure(1)
plt.clf()
plt.plot(x_fsc,fsc_ours,'b',label="ours")
# plt.plot(x_fsc,fsc_ours_post_process,'--b',label="ours post process")
# plt.plot(x_fsc,fsc_ours_isonet,'--b',label="ours + Isonet")
if(eval_AreTomo):
    plt.plot(x_fsc,fsc_AreTomo,'r',label="AreTomo")
if(eval_ETOMO):
    plt.plot(x_fsc,fsc_Etomo,'c',label="Etomo")
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
if(eval_AreTomo):
    fsc_arr[:,4] = fsc_AreTomo[:,0]
if(eval_ETOMO):
    fsc_arr[:,5] = fsc_Etomo[:,0]
# fsc_arr[:,5] = fsc_FBP_est_deformed[:,0]
# fsc_arr[:,6] = fsc_ours_isonet[:,0]
header ='x,ours,FBP,FBP_no_deformed,AreTomo,ETOMO,ours_isonet'
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
if(eval_AreTomo):
    projections_AreTomo = operator_ET(V_AreTomo_t).detach().cpu().numpy()
if(eval_ETOMO):
    projections_Etomo = operator_ET(V_Etomo_t).detach().cpu().numpy()

out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","ours","projections.mrc"),projections_ours.astype(np.float32),overwrite=True)
out.close()
if(eval_AreTomo):
    out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","AreTomo","projections.mrc"),projections_AreTomo.astype(np.float32),overwrite=True)
    out.close()
if(eval_ETOMO):
    out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","Etomo","projections.mrc"),projections_Etomo.astype(np.float32),overwrite=True)
    out.close()

out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","FBP","projections.mrc"),projections_FBP.astype(np.float32),overwrite=True)
out.close()
out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","FBP_no_deformed","projections.mrc"),projections_FBP_no_deformed.astype(np.float32),overwrite=True)
out.close()


for k in range(config.Nangles):
    tmp = projections_ours[k]
    tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
    tmp = np.floor(255*tmp).astype(np.uint8)
    imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"projections","ours","snapshot_{}.png".format(k)),tmp)
    if(eval_AreTomo):
        tmp = projections_AreTomo[k]
        tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
        tmp = np.floor(255*tmp).astype(np.uint8)
        imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"projections","AreTomo","snapshot_{}.png".format(k)),tmp)
    if(eval_ETOMO):
        tmp = projections_Etomo[k]
        tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
        tmp = np.floor(255*tmp).astype(np.uint8)
        imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"projections","Etomo","snapshot_{}.png".format(k)),tmp)
    tmp = projections_FBP[k]
    tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
    tmp = np.floor(255*tmp).astype(np.uint8)
    imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"projections","FBP","snapshot_{}.png".format(k)),tmp)
    tmp = projections_FBP_no_deformed[k]
    tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
    tmp = np.floor(255*tmp).astype(np.uint8)
    imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"projections","FBP_no_deformed","snapshot_{}.png".format(k)),tmp)

out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"volumes","ours","projections.mrc"),np.moveaxis(V_ours,2,0),overwrite=True)
out.close()
if(eval_AreTomo):
    out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"volumes","AreTomo","projections.mrc"),np.moveaxis(V_AreTomo,2,0),overwrite=True)
    out.close()
if(eval_ETOMO):
    out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"volumes","Etomo","projections.mrc"),np.moveaxis(V_Etomo,2,0),overwrite=True)
    out.close()
out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"volumes","FBP","projections.mrc"),np.moveaxis(V_FBP,2,0),overwrite=True)
out.close()
out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"volumes","FBP_no_deformed","projections.mrc"),np.moveaxis(V_FBP_no_deformed,2,0),overwrite=True)
out.close()












# TODO: comparing local deformation 












