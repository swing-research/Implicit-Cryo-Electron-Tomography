"""
This script is used to reconstruct the volume form alligend tilt-series.
It uses the FBP algorithm.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import mrcfile
from ops.radon_3d_lib import ParallelBeamGeometry3DOpAngles_rectangular

from configs.config_reconstruct_simulation import get_default_configs,get_areTomoValidation_configs

config = get_default_configs()
device = 0


ARE_TOMO_FILE = 'areTomo_alligned.mrc'
ETOMO_FILE = 'projections_ali.mrc'

# Initialize the forward operator
angles = np.linspace(config.view_angle_min,config.view_angle_max,config.Nangles)
angles_t = torch.tensor(angles).type(config.torch_type).to(device)
operator_ET = ParallelBeamGeometry3DOpAngles_rectangular((config.n1,config.n2,config.n3), angles/180*np.pi, op_snr=np.inf, fact=1)


aretomo_projections = np.double(mrcfile.open(os.path.join(config.path_save,ARE_TOMO_FILE)).data)
aretomo_projections_t = torch.tensor(aretomo_projections).type(config.torch_type).to(device)


etomo_projections = np.double(mrcfile.open(os.path.join(config.path_save,ETOMO_FILE)).data)
etomo_projections_t = torch.tensor(etomo_projections).type(config.torch_type).to(device)
# Reconstruct the volume    

V_aretomo = operator_ET.pinv(aretomo_projections_t).detach().requires_grad_(False)
V_etomo = operator_ET.pinv(etomo_projections_t).detach().requires_grad_(False)


print('Saving the volume')
out = mrcfile.new(config.path_save_data+"V_aretomo.mrc",np.moveaxis(V_aretomo.detach().cpu().numpy().reshape(config.n1,config.n2,config.n3),2,0),overwrite=True)
out.close() 

out = mrcfile.new(config.path_save_data+"V_etomo.mrc",np.moveaxis(V_etomo.detach().cpu().numpy().reshape(config.n1,config.n2,config.n3),2,0),overwrite=True)
out.close()


