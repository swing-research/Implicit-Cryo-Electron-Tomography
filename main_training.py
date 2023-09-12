import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
plt.ion()
import mrcfile
import time
import os
import imageio
import torch.nn.functional as F
import torch.nn as nn

from utils import utils_deformation, utils_ricardo, utils_display

from torch.utils.data import DataLoader, TensorDataset
from utils.utils_sampling import sample_implicit_batch_lowComp, generate_rays_batch_bilinear

from configs.config_reconstruct_simulation import get_default_configs
config = get_default_configs()


import warnings
warnings.filterwarnings('ignore') 

# Introduction
'''
This script...
'''

use_cuda=torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
if torch.cuda.device_count()>1:
    torch.cuda.set_device(config.device_num)
np.random.seed(config.seed)
torch.manual_seed(config.seed)

if not os.path.exists(config.path_save+"training/"):
    os.makedirs(config.path_save+"training/")
if not os.path.exists(config.path_save+"training/volume/"):
    os.makedirs(config.path_save+"training/volume/")
if not os.path.exists(config.path_save+"training/deformations/"):
    os.makedirs(config.path_save+"training/deformations/")

## TODO: do we need all the following for training?
data = np.load(config.path_save_data+"volume_and_projections.npz")
projections_noisy = torch.tensor(data['projections_noisy']).type(config.torch_type).to(device)
projections_deformed = torch.tensor(data['projections_deformed']).type(config.torch_type).to(device)
projections_deformed_global = torch.tensor(data['projections_deformed_global']).type(config.torch_type).to(device)
projections_clean = torch.tensor(data['projections_clean']).type(config.torch_type).to(device)
# PSF = torch.tensor(data['PSF']).type(config.torch_type).to(device)
if config.sigma_PSF!=0:
    supp_PSF = max(PSF.shape)
    xx1 = np.linspace(-config.n1//2,config.n1//2,config.n1)
    xx2 = np.linspace(-config.n2//2,config.n2//2,config.n2)
    XX, YY = np.meshgrid(xx1,xx2)
    G = np.exp(-(XX**2+YY**2)/(2*config.sigma_PSF**2))
    supp = int(np.round(4*config.sigma_PSF))
    PSF = G[config.n1//2-supp//2:config.n1//2+supp//2,config.n2//2-supp//2:config.n2//2+supp//2]
    PSF /= PSF.sum()
    PSF_t = torch.tensor(PSF.reshape(1,1,-1,1)).type(config.torch_type).to(device)
else: 
    PSF = 0


affine_tr = np.load(config.path_save_data+"global_deformations.npy",allow_pickle=True)
local_tr = np.load(config.path_save_data+"local_deformations.npy", allow_pickle=True)


shift_true = np.zeros((len(affine_tr),2))
angle_true = np.zeros((len(affine_tr)))
for k in range(len(affine_tr)):
    shift_true[k,0] = affine_tr[k].shiftX.item()
    shift_true[k,1] = affine_tr[k].shiftY.item()
    angle_true[k] = affine_tr[k].angle.item()

V_t = torch.tensor(np.moveaxis(np.double(mrcfile.open(config.path_save_data+"V.mrc").data),0,2)).type(config.torch_type).to(device)
V_FBP_no_deformed_t = torch.tensor(np.moveaxis(np.double(mrcfile.open(config.path_save_data+"V_FBP_no_deformed.mrc").data),0,2)).type(config.torch_type).to(device)
V_FBP =  torch.tensor(np.moveaxis(np.double(mrcfile.open(config.path_save_data+"V_FBP.mrc").data),0,2)).type(config.torch_type).to(device)

V_ = V_t.detach().cpu().numpy()
V_FBP_ = V_FBP.detach().cpu().numpy()
V_FBP_no_deformed = V_FBP_no_deformed_t.detach().cpu().numpy()

V_ /= V_.sum()
V_FBP_ /= V_FBP_.sum()
V_FBP_no_deformed /= V_FBP_no_deformed.sum()
fsc_FBP = utils_ricardo.FSC(V_,V_FBP_)
fsc_FBP_no_deformed = utils_ricardo.FSC(V_,V_FBP_no_deformed)
x_fsc = np.arange(fsc_FBP.shape[0])


######################################################################################################
######################################################################################################
##
## TRAINING
##
######################################################################################################
######################################################################################################

# Some processing
if config.sigma_PSF!=0:
    config.nRays = config.nRays//(supp_PSF**2)
    psf_shift = torch.zeros((supp_PSF,supp_PSF)).type(config.torch_type).to(device)
    xx_ = torch.tensor(np.arange(-supp_PSF//2,supp_PSF//2)/config.n1).type(config.torch_type).to(device)
    yy_ = torch.tensor(np.arange(-supp_PSF//2,supp_PSF//2)/config.n2).type(config.torch_type).to(device)
    psf_shift_x,psf_shift_y = torch.meshgrid(xx_,yy_)
    psf_shift_x = psf_shift_x.reshape(1,1,-1,1)
    psf_shift_y = psf_shift_y.reshape(1,1,-1,1)
rays_scaling = torch.tensor(np.array(config.rays_scaling))[None,None,None].type(config.torch_type).to(device)

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
# TODO: make these parameters in the config files
    import tinycudann as tcnn
    config_network = {"encoding": {
            'otype': 'Grid',
            'type': 'Hash',
            'n_levels': 16,
            'n_features_per_level': 4,
            'log2_hashmap_size': 20,
            'base_resolution': 8,
            'per_level_scale': 1.3,
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


# TODO: add Gaussian blob with trainable position and directions
######################################################################################################
## Define the implicit deformations
if config.local_model=='implicit':
    from models.fourier_net import FourierNet,FourierNet_Features
    # Define Implicit representation of local deformations
    implicit_deformation_list = []
    for k in range(config.Nangles):
        implicit_deformation = FourierNet(
            in_features=config.input_size,
            out_features=config.output_size,
            hidden_features=config.hidden_size,
            hidden_blocks=config.num_layers,
            L = config.L).to(device)
        implicit_deformation_list.append(implicit_deformation)

    num_param = sum(p.numel() for p in implicit_deformation_list[0].parameters() if p.requires_grad) 
    print('---> Number of trainable parameters in implicit net: {}'.format(num_param))
if config.local_model=='interp':
    depl_ctr_pts_net = torch.zeros((2,config.N_ctrl_pts_net,config.N_ctrl_pts_net)).to(device).type(config.torch_type)/max([config.n1,config.n2,config.n3])/10
    implicit_deformation_list = []
    for k in range(config.Nangles):
        # depl_ctr_pts_net = local_tr[k].depl_ctr_pts.clone().detach()[0].cuda()/deformationScale
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
# Optimizer
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
if train_global_def:
    scheduler_deformation_glob = torch.optim.lr_scheduler.StepLR(
        optimizer_deformations_glob, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
if train_local_def:
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
for ii, a in enumerate(angles):
    choosenLocations_all[ii] = []
current_sampling = np.ones_like(projections_noisy.detach().cpu().numpy())

def  globalDeformationValues(shift,rot):
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



######################################################################################################
## Iterative optimization
loss_tot = []
loss_data_fidelity = []
loss_regul_local_smooth = []
loss_regul_local_ampl = []
loss_regul_volume = []
loss_regul_shifts = []
loss_regul_rot = []
SNR_tot = []
t_test = []

train_volume = config.train_volume
learn_deformations = False
t0 = time.time()
for ep in range(config.epochs):
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
        ## TODO: make sure that every parameter can be changed in config file
        ## TODO: add an option for density_sampling
        raysSet,raysRot, isOutsideSet, pixelValues = generate_rays_batch_bilinear(proj,angle,config.nRays,config.ray_length,
                                                                                            randomZ=2,zmax=config.z_max,
                                                                                            choosenLocations_all=choosenLocations_all,density_sampling=None,idx_loader=idx_loader)

        if config.sigma_PSF!=0:
            raysSet_ = raysSet.reshape(config.batch_size,config.nRays,1,config.ray_length,3).repeat(1,1,supp_PSF**2,1,1)
            raysSet_[:,:,:,:,0] = raysSet_[:,:,:,:,0]+psf_shift_x
            raysSet_[:,:,:,:,1] = raysSet_[:,:,:,:,1]+psf_shift_y
            raysSet = raysSet_.reshape(config.batch_size,config.nRays*supp_PSF**2,config.ray_length,3)

        raysSet = raysSet*rays_scaling
        outputValues,support = sample_implicit_batch_lowComp(impl_volume,raysSet,angle,
            rot_deformSet=rot_deformSet,shift_deformSet=shift_deformSet,local_deformSet=local_deformSet,
            scale=config.deformationScale,range=config.inputRange,zlimit=config.n3/max(config.n1,config.n2))
        outputValues = outputValues.type(config.torch_type)

        if config.sigma_PSF!=0:
            outputValues = (outputValues.reshape(config.batch_size,config.nRays,supp_PSF**2,config.ray_length)*PSF_t).sum(2)
            support = support.reshape(outputValues.shape[0],outputValues.shape[1],supp_PSF**2,-1)
            support = support[:,:,supp_PSF//2+supp_PSF//2,:] # take only the central elements
        else:
            support = support.reshape(outputValues.shape[0],outputValues.shape[1],-1)
            
        # Compute the projections
        projEstimate = torch.sum(support*outputValues,2)/config.n3

        # Take the datafidelity loss
        loss = loss_data(projEstimate,pixelValues.to(projEstimate.dtype))
        loss_data_fidelity.append(loss.item())

        # update sampling
        with torch.no_grad():
            for jj, ii_ in enumerate(idx_loader):
                ii = ii_.item()
                idx = np.floor((choosenLocations_all[ii][-1]+1)/2*max(config.n1,config.n2)).astype(np.int)
                current_sampling[ii,idx[:,0],idx[:,1]] += 1

        ## Add regularizations
        if train_local_def and config.lamb_local_ampl!=0:
            depl = torch.abs(implicit_deformation_list[ii](raysSet.reshape(-1,3)))
            loss += config.lamb_local_ampl*(depl.mean())
            loss_regul_local_ampl.append(config.lamb_local_ampl*depl.mean().item())
        if train_global_def and (config.lamb_rot!=0 or config.lamb_shifts!=0):
            for ii in idx_loader:
                loss += config.lamb_shifts*torch.abs(shift_est[ii]()*config.n1).sum()
                loss += config.lamb_rot*torch.abs(rot_est[ii]()*180/np.pi).sum()
                loss_regul_shifts.append((config.lamb_shifts*torch.abs(shift_est[ii]()*config.n1).sum()).item())
                loss_regul_rot.append((config.lamb_rot*torch.abs(rot_est[ii]()*180/np.pi).sum()).item())
        
        if config.train_volume and config.lamb_volume!=0:
            V_est = impl_volume(raysSet)
            loss += torch.linalg.norm(outputValues[outputValues<0])*config.lamb_volume
            loss_regul_volume.append((torch.linalg.norm(outputValues[outputValues<0])*config.lamb_volume).item())

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

    loss_current_epoch = np.mean(loss_tot[-len(trainLoader):])
    l_fid = np.mean(loss_data_fidelity[-len(trainLoader):])
    l_v = np.mean(loss_regul_volume[-len(trainLoader):])
    l_sh = np.mean(loss_regul_shifts[-len(trainLoader):])
    l_rot = np.mean(loss_regul_rot[-len(trainLoader):])
    l_loc = np.mean(loss_regul_local_ampl[-len(trainLoader):])
    print("Epoch: {}, loss_avg: {:2.3} || Loss data fidelity: {:2.3}, regul volume: {:2.3}, regul shifts: {:2.3}, regul inplane: {:2.3}, regul local: {:2.3}, time: {:2.3}".format(
        ep,loss_current_epoch,l_fid,l_v,l_sh,l_rot,l_loc,time.time()-t0))

    if ep%config.Ntest==0 :#and ep!=0:
        x_lin1 = np.linspace(-1,1,config.n1)*rays_scaling[0,0,0,0].item()/2+0.5
        x_lin2 = np.linspace(-1,1,config.n2)*rays_scaling[0,0,0,1].item()/2+0.5
        XX, YY = np.meshgrid(x_lin1,x_lin2,indexing='ij')
        grid2d = np.concatenate([XX.reshape(-1,1),YY.reshape(-1,1)],1)
        grid2d_t = torch.tensor(grid2d).type(config.torch_type)
        z_range = np.linspace(-1,1,15)*rays_scaling[0,0,0,2].item()*(config.n3/config.n1)/2+0.5
        for zz, zval in enumerate(z_range):
            grid3d = np.concatenate([grid2d_t, zval*torch.ones((grid2d_t.shape[0],1))],1)
            grid3d_slice = torch.tensor(grid3d).type(config.torch_type).to(device)
            estSlice = impl_volume(grid3d_slice).detach().cpu().numpy().reshape(config.n1,config.n2)
            pp = (estSlice)*1.
            plt.figure(1)
            plt.clf()
            plt.imshow(pp,cmap='gray')
            plt.savefig(os.path.join(config.path_save+"/training/volume/volume_slice_{}.png".format(zz)))


        loss_current_epoch = np.mean(loss_tot[-len(trainLoader)*config.Ntest:])
        l_fid = np.mean(loss_data_fidelity[-len(trainLoader)*config.Ntest:])
        l_v = np.mean(loss_regul_volume[-len(trainLoader)*config.Ntest:])
        l_sh = np.mean(loss_regul_shifts[-len(trainLoader)*config.Ntest:])
        l_rot = np.mean(loss_regul_rot[-len(trainLoader)*config.Ntest:])
        l_loc = np.mean(loss_regul_local_ampl[-len(trainLoader)*config.Ntest:])
        print("###### Epoch: {}, loss_avg: {:2.3} || Loss data fidelity: {:2.3}, regul volume: {:2.3}, regul shifts: {:2.3}, regul inplane: {:2.3}, regul local: {:2.3}, time: {:2.3}".format(
            ep,loss_current_epoch,l_fid,l_v,l_sh,l_rot,l_loc,time.time()-t0))


        
        # TODO: Display local deformation
        utils_display.display_local_movie(implicit_deformation_list,field_true=local_tr,Npts=(20,20),
                                    img_path=config.path_save+"/training/deformations/def_",img_type='.png',
                                    scale=1/10,alpha=0.8,width=0.002)
            

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

        # TODO: display the sampling

        
    if ep%config.NsaveNet ==0 and ep!=0:                    
        torch.save({
        'shift_est': shift_est,
        'rot_est': rot_est,
        'local_deformation_network': implicit_deformation_list,
        'implicit_volume': impl_volume.state_dict(),
        }, os.path.join(config.path_save,'training','model_everything_joint_batch.pt'))


torch.save({
'shift_est': shift_est,
'rot_est': rot_est,
'local_deformation_network': implicit_deformation_list,
'implicit_volume': impl_volume.state_dict(),
}, os.path.join(config.path_save,'training','model_everything_joint_batch.pt'))


## save or volume in full resolution
x_lin1 = np.linspace(-1,1,config.n1)*rays_scaling[0,0,0,0].item()/2+0.5
x_lin2 = np.linspace(-1,1,config.n2)*rays_scaling[0,0,0,1].item()/2+0.5
XX, YY = np.meshgrid(x_lin1,x_lin2,indexing='ij')
grid2d = np.concatenate([XX.reshape(-1,1),YY.reshape(-1,1)],1)
grid2d_t = torch.tensor(grid2d).type(config.torch_type)
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