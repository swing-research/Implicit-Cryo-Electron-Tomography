import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils_deformation

# Express the true volume continuously using bicubic interpolation
def cropper3d(image, coordinate , output_size):
    # Coordinate shape: b X 3
    # image shape: b X c X d X h X w
    d_coordinate = coordinate
    b, c , d, h , w = image.shape
    n = max(d, h, w)
    crop_size = output_size/n
    x_m_x = crop_size
    x_p_x = d_coordinate[:,2]
    y_m_y = crop_size
    y_p_y = d_coordinate[:,1]
    z_m_z = crop_size
    z_p_z = d_coordinate[:,0]
    # Y, X,Z
    theta = torch.zeros(b,3,4).to(image.device)
    theta[:,0,0] = x_m_x
    theta[:,0,3] = x_p_x
    theta[:,1,1] = y_m_y
    theta[:,1,3] = y_p_y
    theta[:,2,2] = z_m_z
    theta[:,2,3] = z_p_z
    f = F.affine_grid(theta, size=(b, c, output_size, output_size, output_size), align_corners=True)
    # image_cropped = grid_sample_customized_bilinear(image, f, align_corners = True)
    # We might need to change with the above if we want to pass gradient more than once to the parameters of the deformation
    image_cropped = F.grid_sample(image, f, mode='bilinear', align_corners = True)
    return image_cropped

def interp_volume(vol,coordinates=None, n1=None, n2=None, n3=None):
    n = vol.shape[-1]
    if n1 is None:
        n1, n2, n3 = vol.shape[-3:]
    if coordinates is None:
        xx = torch.linspace(-1,1,n,dtype=vol.dtype,device=vol.device)
        XX_t, YY_t, ZZ_t = torch.meshgrid(xx,xx,xx,indexing='ij')
        XX_t = torch.unsqueeze(XX_t, dim = 3)
        YY_t = torch.unsqueeze(YY_t, dim = 3)
        ZZ_t = torch.unsqueeze(ZZ_t, dim = 3)
        coordinates = torch.cat([XX_t,YY_t,ZZ_t],3).reshape(-1,3)
    vol = vol.expand(coordinates.shape[0], 1, n1, n2, n3)
    return cropper3d(vol,coordinates,output_size = 1).reshape(n1,n2,n3)

"""
Define new function to avoid conflicts.
"""
def interp_volume_rectangular(vol,coordinates=None, n1=None, n2=None, n3=None):
    n = vol.shape[-1]
    if n1 is None:
        n1, n2, n3 = vol.shape[-3:]
    if coordinates is None:
        xx = torch.linspace(-1,1,n,dtype=vol.dtype,device=vol.device)
        XX_t, YY_t, ZZ_t = torch.meshgrid(xx,xx,xx,indexing='ij')
        XX_t = torch.unsqueeze(XX_t, dim = 3)
        YY_t = torch.unsqueeze(YY_t, dim = 3)
        ZZ_t = torch.unsqueeze(ZZ_t, dim = 3)
        coordinates = torch.cat([XX_t,YY_t,ZZ_t],3).reshape(-1,3)
    vol = vol.expand(coordinates.shape[0], 1, n1, n2, n3)
    n = max(n1, n2, n3)
    w = torch.tensor([n/n1,n/n2,n/n3]).type_as(vol).to(coordinates.device).view(1,-1)
    return cropper3d(vol,coordinates*w,output_size = 1).reshape(n1,n2,n3)

def interp_img(img,coordinates=None):
    n = img.shape[-1]
    if coordinates is None:
        xx = torch.linspace(-1,1,n,dtype=img.dtype,device=img.device)
        XX_t, YY_t = torch.meshgrid(xx,xx,indexing='ij')
        XX_t = torch.unsqueeze(XX_t, dim = 2)
        YY_t = torch.unsqueeze(YY_t, dim = 2)
        coordinates = torch.cat([XX_t,YY_t],2).reshape(-1,2)
    img = img.expand(coordinates.shape[0], 1, n, n)
    return utils_deformation.cropper(img,coordinates,output_size = 1).reshape(-1,n,n)


