import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import shift
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


# replace by rotate_np.
# ## Define the rotations
# def rotate_volume(V,alpha,beta,gamma):
#     V = rotate(V, alpha, mode='constant', cval=0, order=3, axes=(2, 1), reshape=False)
#     V = rotate(V, beta, mode='constant', cval=0, order=3, axes=(2, 0), reshape=False)
#     V = rotate(V, gamma, mode='constant', cval=0, order=3, axes=(1,0), reshape=False)
#     return V




# """
# Given points on the sphere, return azimuthal angle phi (around axis z), and polar angle (theta).
# Physics notations based on wikipedia.
# """
# def cart2sph(x,y,z):
#     # phi_ = np.arctan2(y,x)
#     # theta_ = np.arccos(z)
#     r=np.sqrt(x**2+y**2)+1e-16
#     theta_=np.arctan(z/r)
#     phi_=np.arctan2(y,x)
#     return phi_, theta_
# def sph2cart(phi,theta):
#     #takes list rthetaphi (single coord)
#     x = np.cos( theta ) * np.cos( phi )
#     y = np.cos( theta ) * np.sin( phi )
#     z = np.sin( theta )
#     return x,y,z

# """
# Return Ntheta points uniformly distributed on the sphere.
# """
# def sampling_sphere(Ntheta):
#     # th = np.random.random(Ntheta)*np.pi*2
#     # x = np.random.random(Ntheta)*2-1
#     # out = np.array([np.cos(th)*np.sqrt(1-x**2),np.sin(th)*np.sqrt(1-x**2),x]).T

#     out = np.random.randn(Ntheta,3)
#     out /= np.linalg.norm(out,axis=1,keepdims=True)

#     return out

# """
# Return 3 rotation angles, around axis x, y and z.
# dofAngle: 3 is full 3D rotation, 2 there is no inplane rotation, 1 only rotation around x is nonzero
# """
# def generate_rotation(Ntheta,dofAngle=3,order='ZYX'):
#     if dofAngle==3:
#         # rot_z = np.random.random(Ntheta)*2*np.pi
#         # rot_x, rot_y = cart2sph(pts[:,0],pts[:,1],pts[:,2])
#         rot_ = scipy_rot.random(Ntheta)
#         rot = rot_.as_euler(order,degrees=True)
#     elif dofAngle==4:
#         from icosphere import icosphere
#         nu = 6  # or any other integer
#         vertices, faces = icosphere(nu)
#         ax1,ax2,ax3=vertices[:,0],vertices[:,1],vertices[:,2]
#         rot_x, rot_y = cart2sph(ax1,ax2,ax3)
#         # r=np.sqrt(ax1**2+ax2**2)+1e-16
#         # rot_y=np.arctan(ax3/r)
#         # rot_x=np.arctan2(ax2,ax1)

#         rot_z=np.linspace(0,2*np.pi,10)

#         rot = np.zeros((rot_x.shape[0]*rot_z.shape[0],3))
#         k = 0
#         for i in range(rot_x.shape[0]):
#             for j in range(rot_z.shape[0]):
#                 rot[k] = np.array([rot_x[i],rot_y[i],rot_z[j]])
#                 k += 1

#         rot = rot/(np.pi)*180

#         # psi_list=np.arctan(ax3/r)/np.pi*180
#         # phi_list=np.arctan2(ax2,ax1)/np.pi*180
#     elif dofAngle == 1:
#         rot_x = np.random.random(Ntheta)*2*np.pi
#         rot_y = np.zeros(Ntheta)
#         rot_z = np.zeros(Ntheta)
#         rot = np.array([rot_x, rot_y, rot_z]).T
#         rot = rot/(np.pi)*180
#     else:
#         print("Error: dofAngle should be in {1,2,3}")
#         # TODO: raise a proper error
#     return rot


# """
# Compute 2D convolution using 0 padding.
# Image are expected with 0 frenquency at the center of the spatial domain.
# """
# def conv2D(im1,im2,ext=0):
#     n1,n2=im1.shape
#     if ((im2.shape[0]!=n1) or (im2.shape[1]!=n2)):
#         return -1
#     if ext>0:
#         im1_=np.zeros((n1+2*ext,n2+2*ext))
#         im1_[ext:n1+ext,ext:n2+ext]=im1
#         im2_=np.zeros((n1+2*ext,n2+2*ext))
#         im2_[ext:n1+ext,ext:n2+ext]=im2
#         im1=im1_
#         im2=im2_
#     im2=np.fft.fftshift(im2)
#     out=np.real(np.fft.ifft2(np.fft.fft2(im1)*np.fft.fft2(im2)))
#     if ext>0:
#         out=out[ext:n1+ext,ext:n2+ext]
#     return out



# """
# Torch function to get rotations matrix. 
# First dim of input is batch size.
# angles are given in order xyz.
# angles in radian.
# """
# def getRotationMatrix_t(angles,order='ZYX'):
#     rr = scipy_rot.from_euler(order,angles.detach().cpu().numpy(),degrees=True)
#     mat = rr.as_matrix()
#     rotationMatrix = torch.tensor(mat).type(torch.float).to(angles.device)
#     # device_ = angles.device
#     # c1 = torch.cos(angles[:,0])
#     # s1 = torch.sin(angles[:,0])
#     # c2 = torch.cos(angles[:,1])
#     # s2 = torch.sin(angles[:,1])
#     # c3 = torch.cos(angles[:,2])
#     # s3= torch.sin(angles[:,2])
    
#     # # Tait-Bryan angles, i.e. extrinsic Euler rotationa ngles, from 
#     # # https://en.wikipedia.org/wiki/Euler_angles#Definition_by_intrinsic_rotations
#     # # using "X1Y2Z3" convention
#     # rotationMatrix = torch.zeros(angles.shape[0],3,3).to(angles.device)
#     # if order=='XYZ':
#     #     rotationMatrix[:,0,0] = c2*c3
#     #     rotationMatrix[:,1,0] = c1*s3+c3*s1*s2
#     #     rotationMatrix[:,2,0] = s1*s3-c1*c3*s2
#     #     rotationMatrix[:,0,1] = -c2*s3
#     #     rotationMatrix[:,1,1] = c1*c3-s1*s2*s3
#     #     rotationMatrix[:,2,1] = c3*s1+c1*s2*s3
#     #     rotationMatrix[:,0,2] = s2
#     #     rotationMatrix[:,1,2] = -c2*s1
#     #     rotationMatrix[:,2,2] = c1*c2
#     # # ZYX
#     # if order=='ZYX':
#     #     rotationMatrix[:,0,0] = c1*c2
#     #     rotationMatrix[:,1,0] = c2*s1
#     #     rotationMatrix[:,2,0] = -s2
#     #     rotationMatrix[:,0,1] = c1*s2*s3-c3*s1
#     #     rotationMatrix[:,1,1] = c1*c3+s1*s2*s3
#     #     rotationMatrix[:,2,1] = c2*s3
#     #     rotationMatrix[:,0,2] = s1*s3+c1*c3*s2
#     #     rotationMatrix[:,1,2] = c3*s1*s2-c1*s3
#     #     rotationMatrix[:,2,2] = c2*c3

#     # # # zxz
#     # if order=='zxz':
#     #     rotationMatrix[:,0,0] = c1*c3-c2*s1*s3
#     #     rotationMatrix[:,1,0] = c3*s1+c1*c2*s3
#     #     rotationMatrix[:,2,0] = s2*s3
#     #     rotationMatrix[:,0,1] = -c1*s3-c2*c3*s1
#     #     rotationMatrix[:,1,1] = c1*c2*c3-s1*s3
#     #     rotationMatrix[:,2,1] = c3*s2
#     #     rotationMatrix[:,0,2] = s1*s2
#     #     rotationMatrix[:,1,2] = -c1*s2
#     #     rotationMatrix[:,2,2] = c2

#     # # # yzy
#     # if order=='yzy':
#     #     rotationMatrix[:,0,0] = c1*c2*c3-s1*s3
#     #     rotationMatrix[:,1,0] = c3*s2
#     #     rotationMatrix[:,2,0] = -c1*s3-c2*c3*s1
#     #     rotationMatrix[:,0,1] = -c1*s2
#     #     rotationMatrix[:,1,1] = c2
#     #     rotationMatrix[:,2,1] = s1*s2
#     #     rotationMatrix[:,0,2] = c3*s1+c1*c2*s3
#     #     rotationMatrix[:,1,2] = s2*s3
#     #     rotationMatrix[:,2,2] = c1*c3-c2*s1*s3

#     # # # ZXY
#     # if order=='ZXY':
#     #     rotationMatrix[:,0,0] = c1*c3-s1*s2*s3
#     #     rotationMatrix[:,1,0] = c3*s1+c1*s2*s3
#     #     rotationMatrix[:,2,0] = -c2*s3
#     #     rotationMatrix[:,0,1] = -c2*s1
#     #     rotationMatrix[:,1,1] = c1*c2
#     #     rotationMatrix[:,2,1] = s2
#     #     rotationMatrix[:,0,2] = c1*s3+c3*s1*s2
#     #     rotationMatrix[:,1,2] = s1*s3-c1*c3*s2
#     #     rotationMatrix[:,2,2] = c2*c3

#     # # # ZXY
#     # if order=='YXZ':
#     #     rotationMatrix[:,0,0] = c1*c3-s1*s2*s3
#     #     rotationMatrix[:,1,0] = c3*s1+c1*s2*s3
#     #     rotationMatrix[:,2,0] = -c2*s3
#     #     rotationMatrix[:,0,1] = -c2*s1
#     #     rotationMatrix[:,1,1] = c1*c2
#     #     rotationMatrix[:,2,1] = s2
#     #     rotationMatrix[:,0,2] = c1*s3+c3*s1*s2
#     #     rotationMatrix[:,1,2] = s1*s3-c1*c3*s2
#     #     rotationMatrix[:,2,2] = c2*c3
#     return rotationMatrix


# def get_3d_locations(d,h,w,device_):
#     locations_x = torch.linspace(-1, 1, w).view(1, 1, 1, w).to(device_).expand(1, d, h, w)
#     locations_y = torch.linspace(-1, 1, h).view(1, 1, h, 1).to(device_).expand(1, d, h, w)
#     locations_z = torch.linspace(-1, 1,d).view(1, d, 1, 1).to(device_).expand(1, d, h, w)
#     # stack locations
#     locations_3d = torch.stack([locations_x, locations_y, locations_z], dim=4).view(-1, 3, 1)
#     return locations_3d

# ## Define the translation
# def translate(V,tx,ty):
#     if (tx>0 or ty>0):
#         V = shift(V,(tx,ty,0))
#     return V
# # Torch equivalent function
# # TODO: use torch.transform to get a non-integer shifts
# def translate_t(V,tx,ty):
#     if (tx>0 or ty>0):
#         print("Warning: only intger shifts are done in Pytorch (cuda).")
#         V = torch.roll(V,shifts=(int(tx),int(ty)),dim=(0,1))
#     return V


# def SNR(x_ref,x):
#     dif = np.sum((x_ref-x)**2)
#     nref = np.sum(x_ref**2)
#     res=10*np.log10((nref+1e-16)/(dif+1e-16))
#     return res



# # def find_sigma_noise(snr, x_ref):
# #     nref = np.std(x_ref)
# #     sigma_noise = (10**(-snr/20)) * nref
# #     return sigma_noise

# def getPSF(sigma,n):
#     if sigma==0:
#         return 0, 0, 0
#     else:
#         ext = 3*sigma # padding size
#         lin = np.linspace(-n/2,n/2,n)
#         XX,YY = np.meshgrid(lin,lin)
#         if sigma==0:
#             g = np.zeros((n,n))
#             g[n//2,n//2]=1
#         else:
#             g = np.exp(-(XX**2+YY**2)/(2*sigma**2))
#             g /= np.sum(g)

#         g_hat = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(g)))
#         g = np.repeat(g[:,:,None],n,axis=2)
#         g_hat = np.repeat(g_hat[:,:,None],n,axis=2)
#         return g, g_hat, ext

# """
# Mtheta: rotation matrix

# """
# def forward(V, Mtheta, shifts,g, paddingSize,idx=0):
#     #Theta: 3 dim vector
#     #shift: 2 dim vector
#     V_ = rotate_np(V, Mtheta)
#     # V_ = rotate_volume(V, theta[0],theta[1],theta[2])
#     V_ = translate(V_,shifts[0],shifts[1])
#     Vproj = V_.sum(2)
#     if np.linalg.norm(g)!=0:
#         Vout = conv2D(g[:,:,idx],Vproj,ext=paddingSize)
#     else:
#         Vout = Vproj
#     return Vout

# """
# Mtheta: rotation matrix
# """
# def forward_t(V, Mtheta, shifts,g, paddingSize,idx=0):
#     # V_ = rotate_volume(V, Mtheta[0],theta[1],theta[2])
#     V_ = rotate_t(V,Mtheta).squeeze(0)
#     Vproj = torch.sum(V_,2)
#     V_ = translate_t(Vproj,shifts[0],shifts[1])
#     if torch.linalg.norm(g)!=0:
#         Vout = conv2D(g[:,:,idx],Vproj,ext=paddingSize)
#         Vout = F.conv2d(Vproj[None,None], g[:,:,idx][None,None], padding = 'valid')
#     else:
#         Vout = Vproj
#     return Vout

# # def forward(V,idx=0,seed=None):
# #     np.random.seed(seed)
# #     alpha = np.random.random()*360
# #     beta = np.random.random()*360
# #     gamma = np.random.random()*360
# #     tx = np.random.random()*shift_max
# #     ty = np.random.random()*shift_max
# #     V_ = rotate_volume(V,alpha,beta,gamma)
# #     V_ = translate(V_,tx,ty)
# #     Vproj = V_.sum(2)
# #     Vout = conv2D(g[:,:,idx],Vproj,ext=ext)
# #     return Vout, alpha, beta, gamma, tx, ty