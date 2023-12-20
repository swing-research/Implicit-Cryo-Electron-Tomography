import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation as scipy_rot



"""
angles are in radian.
"""
def getRotationMatrix(angles,order='ZYZ',degrees=True):
    rr = scipy_rot.from_euler(order,angles,degrees)
    mat = rr.as_matrix()
    rotationMatrix = mat
    return rotationMatrix

# TODO: replace by pytorch3d https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html
def rotate_t(input_tensor, rotation_matrix):
    device_ = input_tensor.device
    d, h, w  = input_tensor.shape
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    R_ = torch.zeros((1,3,4))
    R_[0,0,2] = 1
    R_[0,1,1] = 1
    R_[0,2,0] = 1

    grid = F.affine_grid(R_,size=(1,1,d,h,w),align_corners=False).to(device_)
    rotated_3d_positions = torch.matmul(rotation_matrix,grid.view(-1,3).T).T.view(1,d,h,w,3)
    # somehow we need to invert axis 0 and axis 2, and swap back
    # tmp = torch.cat((rotated_3d_positions[:,:,:,:,0].view(1,d,h,w,1),rotated_3d_positions[:,:,:,:,1].view(1,d,h,w,1),rotated_3d_positions[:,:,:,:,2].view(1,d,h,w,1)),dim=4)
    tmp = torch.cat((rotated_3d_positions[:,:,:,:,2].view(1,d,h,w,1),rotated_3d_positions[:,:,:,:,1].view(1,d,h,w,1),rotated_3d_positions[:,:,:,:,0].view(1,d,h,w,1)),dim=4)
    rotated_signal = F.grid_sample(input=input_tensor, grid=tmp, mode='bilinear',  align_corners=False).squeeze(0).squeeze(0)
    return rotated_signal.to(device_)

from scipy.interpolate import interpn
def rotate_np(input, rotation_matrix):
    d, h, w  = input.shape
    linx = np.linspace(-1, 1, d)
    liny = np.linspace(-1, 1, h)
    linz = np.linspace(-1, 1, w)
    XX, YY, ZZ = np.meshgrid(linx,liny,linz,indexing='ij')
    grid = np.concatenate((XX.reshape(1,-1),YY.reshape(1,-1),ZZ.reshape(1,-1)),axis=0)
    rotated_3d_positions = np.matmul(rotation_matrix,grid)
    out = interpn((linx,liny,linz),input,rotated_3d_positions.T,bounds_error=False,fill_value=0.)
    VV=out.reshape(d,h,w)
    return VV

## Define observation model
def find_sigma_noise(SNR_value,x_ref):
    nref = np.mean(x_ref**2)
    out = (10**(-SNR_value/10)) * nref
    return np.sqrt(out)

def find_sigma_noise_comparison(SNR_value,x_ref):
    nref = np.mean(x_ref**2)
    res=lambda sigma: (10*np.log10((nref+1e-16)/(sigma**2+1e-16))-SNR_value)**2
    out = minimize_scalar(res)
    return out.x
    
## Define observation model
def find_sigma_noise_t(SNR_value,x_ref):
    nref = torch.mean(x_ref**2)
    sigma_noise = (10**(-SNR_value/10)) * nref
    return torch.sqrt(sigma_noise)
