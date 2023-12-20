"""
This script is used to reconstruct the volume from alligend tilt-series.
It uses the FBP algorithm.
"""
import os
import torch
import mrcfile
import numpy as np
from ops.radon_3d_lib import ParallelBeamGeometry3DOpAngles_rectangular

def reconstruct_aretomo(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count()>1:
        torch.cuda.set_device(config.device_num)

    ARE_TOMO_FILE = 'areTomo_alligned.mrc'
    path_file = os.path.join(config.path_save,'Aretomo',ARE_TOMO_FILE)
    if os.path.isfile(path_file):
        aretomo_projections = np.double(mrcfile.open(path_file).data)
        aretomo_projections_t = torch.tensor(aretomo_projections).type(config.torch_type).to(device)
        V_FBP = reconstruct_FBP_volume(config, aretomo_projections_t)
        out = mrcfile.new(config.path_save_data+"V_aretomo.mrc",np.moveaxis(V_FBP.detach().cpu().numpy().reshape(config.n1,config.n2,config.n3),2,0),overwrite=True)
        out.close() 

def reconstruct_etomo(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count()>1:
        torch.cuda.set_device(config.device_num)

    ETOMO_FILE = 'projections_ali.mrc'
    path_file = os.path.join(config.path_save,'Etomo',ETOMO_FILE)
    if os.path.isfile(path_file):
        etomo_projections = np.double(mrcfile.open(path_file).data)
        etomo_projections_t = torch.tensor(etomo_projections).type(config.torch_type).to(device)
        V_FBP = reconstruct_FBP_volume(config, etomo_projections_t)
        out = mrcfile.new(config.path_save_data+"V_etomo.mrc",np.moveaxis(V_FBP.detach().cpu().numpy().reshape(config.n1,config.n2,config.n3),2,0),overwrite=True)
        out.close() 

def reconstruct_FBP_volume(config, tiltseries):
    """

    Args:
        config : 
        tiltseries (torch tensor): volume
    """
    ARE_TOMO_FILE = 'areTomo_alligned.mrc'
    ETOMO_FILE = 'projections_ali.mrc'

    # Define the forward operator
    angles = np.linspace(config.view_angle_min,config.view_angle_max,config.Nangles)
    operator_ET = ParallelBeamGeometry3DOpAngles_rectangular((config.n1,config.n2,config.n3), angles/180*np.pi, fact=1)

    # Reconstruct the volume    
    V_FBP = operator_ET.pinv(tiltseries).detach().requires_grad_(False)

    return V_FBP



