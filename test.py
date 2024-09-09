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


import configs.shrec_model0 as config_file
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



######################################################################################################
## Load data
######################################################################################################
data = np.load(config.path_save_data+"volume_and_projections.npz")
projections_noisy = torch.tensor(data['projections_noisy']).type(config.torch_type).to(device)
affine_tr = np.load(config.path_save_data+"global_deformations.npy",allow_pickle=True)
local_tr = np.load(config.path_save_data+"local_deformations.npy", allow_pickle=True)
V_t = torch.tensor(np.moveaxis(np.double(mrcfile.open(config.path_save_data+"V.mrc",permissive=True).data),0,2)).type(config.torch_type).to(device)
V_FBP_no_deformed_t = torch.tensor(np.moveaxis(np.double(mrcfile.open(config.path_save_data+"V_FBP_no_deformed.mrc",permissive=True).data),0,2)).type(config.torch_type).to(device)
V_FBP_t =  torch.tensor(np.moveaxis(np.double(mrcfile.open(config.path_save_data+"V_FBP.mrc",permissive=True).data),0,2)).type(config.torch_type).to(device)
# numpy
V = V_t.detach().cpu().numpy()
V_FBP = V_FBP_t.detach().cpu().numpy()
V_FBP_no_deformed = V_FBP_no_deformed_t.detach().cpu().numpy()

projections_clean = torch.tensor(data['projections_clean']).type(config.torch_type).to(device)

proj = projections_clean
angles = np.linspace(config.view_angle_min,config.view_angle_max,config.Nangles)/180*np.pi

# # Implent TV
# # Implement ADMM
# import astra
# def SIRT(proj,angles,num_iter=100):
#     # Create an ASTRA projection geometry and reconstruction
#     proj_geom = astra.create_proj_geom('parallel3d', 1, 1, config.n1, config.n2, angles)
#     vol_geom = astra.create_vol_geom(config.n3, config.n1, config.n2)
#
#     # Create an ASTRA sinogram and initialize the reconstruction
#     proj_id = astra.data3d.create('-sino', proj_geom, proj.swapaxes(0,1))
#     rec_id = astra.data3d.create('-vol', vol_geom, np.zeros((config.n1, config.n3, config.n2)))
#
#     # Create a SART algorithm and run it
#     cfg = astra.astra_dict('SIRT3D_CUDA')
#     cfg['ProjectionDataId'] = proj_id
#     cfg['ReconstructionDataId'] = rec_id
#     alg_id = astra.algorithm.create(cfg)
#     astra.algorithm.run(alg_id, num_iter)  # Run 100 iterations
#
#     # Retrieve and process the result
#     result = astra.data3d.get(rec_id).swapaxes(1,2)
#     return result
#
# result = SIRT(proj,angles,num_iter=100)
# plt.imshow(result[:,:,90])


# def sart_proximal_operator(operator, a, m, P, vol, nit):
#     """
#     SART Proximal Operator
#     Parameters:
#     - W: numpy array of shape (M, N) representing the weights matrix
#     - a: scalar step size
#     - m: scalar parameter
#     - P: list of projection images, each being a numpy array of shape (M, N)
#     - vol: initial vol
#     - nit: number of iterations
#
#     Returns:
#     - v(T): numpy array of shape (N,) representing the final updated vector
#     """
#     # Initialize variables
#     v = np.copy(vol)
#     y = np.zeros_like(P)  # Initialize y with the same shape as projection images
#     A = operator(np.ones_like(v))
#     # Iterate over T iterations
#     for t in range(nit):
#         Av = operator(v)
#         for j, proj in enumerate(P):
#             # Update y
#             term1 = np.sqrt(2 * m) * (proj[j] - np.sqrt(2 * m) * Av[j]-y[j]) / (1 + np.sqrt(2 * m) * A[j])
#             y[j] += a * np.sum(term1)
#             # Update v
#             term2 = (term1*np.sqrt(2 * m) * A[j]).sum()/ (np.sqrt(2 * m) * A[j].sum())
#             v += a * term2
#         # Proximal operator (soft thresholding)
#         v = np.maximum(0, v)
#     return v
#
#
#
# proj_geom = astra.create_proj_geom('parallel3d', 1, 1, config.n1, config.n2, angles)
# vol_geom = astra.create_vol_geom(config.n3, config.n1, config.n2)
# operator = lambda vol: astra.create_sino3d_gpu(vol.swapaxes(1,2), proj_geom, vol_geom, returnData=True)[1].swapaxes(0,1)
#
# # Create a SART algorithm and run it
# proj_id = astra.data3d.create('-sino', proj_geom, proj.detach().cpu().numpy().swapaxes(0, 1))
# rec_id = astra.data3d.create('-vol', vol_geom, np.zeros((config.n1, config.n3, config.n2)))
# cfg = astra.astra_dict('SIRT3D_CUDA')
# cfg['ProjectionDataId'] = proj_id
# cfg['ReconstructionDataId'] = rec_id
# alg_id = astra.algorithm.create(cfg)
# astra.algorithm.run(alg_id, 10)
# vol_sirt = astra.data3d.get(rec_id).swapaxes(1, 2)
#
# nit = 10
# a = 1
# m = 1
# P = proj.detach().cpu().numpy()
# vol = vol_sirt
# v_sart = sart_proximal_operator(operator, a, m, P, vol, nit)


def d1(u):
    d = np.zeros_like(u)
    d[:-1] = u[1:]-u[:-1]
    return d
def d2(u):
    d = np.zeros_like(u)
    d[:,:-1] = u[:,1:]-u[:,:-1]
    return d
def d3(u):
    d = np.zeros_like(u)
    d[:,:,:-1] = u[:,:,1:]-u[:,:,:-1]
    return d
def d1T(u):
    d = np.zeros_like(u)
    d[1:-1] = -u[1:-1]-u[:-2]
    d[0] = -u[0]
    d[-1] = d[-2]
    return d
def d2T(u):
    d = np.zeros_like(u)
    d[:,1:-1] = -u[:,1:-1]-u[:,:-2]
    d[:,0] = -u[:,0]
    d[:,-1] = d[:,-2]
    return d
def d3T(u):
    d = np.zeros_like(u)
    d[:,:,1:-1] = -u[:,:,1:-1]-u[:,:,:-2]
    d[:,:,0] = -u[:,:,0]
    d[:,:,-1] = d[:,:,-2]
    return d

def prox_l1(u, rho):
    return np.sign(u) * np.maximum(np.abs(u)- rho, 0)

#
# x, cf = CG_inverse(P, np.zeros_like(V), angles, mu, 10, 1e-5)
#
# plt.figure(1)
# plt.clf()
# plt.plot(cf)
#
# plt.figure(2)
# plt.clf()
# plt.imshow(x[:,:,90])

def sirt_prox(P, vol, m, angles, nit):
    n1, n2, n3 = vol.shape
    proj_geom = astra.create_proj_geom('parallel3d', 1, 1, n1, n2, angles)
    vol_geom = astra.create_vol_geom(n3, n1, n2)
    operator = lambda vv: astra.create_sino3d_gpu(vv.swapaxes(1, 2), proj_geom, vol_geom, returnData=True)[
        1].swapaxes(0, 1)

    if m == 0:
        inp = P
    else:
        inp = P + operator(vol)/m

    # Create a SART algorithm and run it
    proj_id = astra.data3d.create('-sino', proj_geom, inp.swapaxes(0, 1))
    rec_id = astra.data3d.create('-vol', vol_geom, np.zeros((n1, n3, n2)))
    cfg = astra.astra_dict('SIRT3D_CUDA')
    cfg['ProjectionDataId'] = proj_id
    cfg['ReconstructionDataId'] = rec_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, nit)
    vol_sirt = astra.data3d.get(rec_id).swapaxes(1, 2)
    return vol_sirt



# nit = 10
# m = 1
# P = proj.detach().cpu().numpy()
# vol = np.zeros_like(V)
# v_sirt = sirt_prox(P, vol, m, nit)
# plt.imshow(v_sirt[:,:,90])

def FP(vol, proj_geom, vol_geom, Nangles, n1, n2, n3, nit=10, device_num=0):
    proj_id = astra.data3d.create('-sino', proj_geom, np.zeros((n1, Nangles, n2)))
    rec_id = astra.data3d.create('-vol', vol_geom, vol.swapaxes(1,2))
    # rec_id = astra.data3d.create('-vol', vol_geom, np.zeros((config.n1, config.n3, config.n2)))
    cfg = astra.astra_dict('FP3D_CUDA')
    cfg['ProjectionDataId'] = proj_id
    cfg['VolumeDataId'] = rec_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, nit)
    proj_fp = astra.data3d.get(proj_id).swapaxes(0, 1)
    return proj_fp

def BP(P, proj_geom, vol_geom, n1, n2, n3, nit=10, device_num=0):
    proj_id = astra.data3d.create('-sino', proj_geom, P.swapaxes(0, 1))
    rec_id = astra.data3d.create('-vol', vol_geom, np.zeros((n1, n3, n2)))
    cfg = astra.astra_dict('BP3D_CUDA')
    cfg['ProjectionDataId'] = proj_id
    cfg['ReconstructionDataId'] = rec_id
    cfg['GPUindex'] = device_num
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, nit)
    vol_bp = astra.data3d.get(rec_id).swapaxes(1, 2)
    return vol_bp

def CG_inverse(P, u, angles, mu, max_iter, tol):
    # solve (mu*AtA+I)x = mu*At(P) + u for x
    At_sirt = lambda pp : sirt_prox(pp, vol, 0, angles, 5)
    x0 = At_sirt(P)
    n1, n2, n3 = x0.shape

    proj_geom = astra.create_proj_geom('parallel3d', 1, 1, n1, n2, angles)
    vol_geom = astra.create_vol_geom(n3, n1, n2)
    def A(vv):
        vv = vv/np.sqrt((vv**2).sum())
        out = FP(vol, proj_geom, vol_geom, P.shape[0], n1, n2, n3, nit=10, device_num=0)
        # out = astra.create_sino3d_gpu(vv.swapaxes(1, 2), proj_geom, vol_geom, returnData=True, gpuIndex=config.device.index)[
        # 1].swapaxes(0, 1)
        return out
    # At = lambda P: BP(P, proj_geom, vol_geom, n1, n2, n3, device_num=config.device.index)
    def At(P):
        out = BP(P, proj_geom, vol_geom, n1, n2, n3, device_num=config.device.index)
        out = out/np.abs(out).max()*np.abs(P).max()
        return out
    H = lambda vv: mu*At(A(vv)) + vv
    b = mu*At(P) + u

    # check with initialization the number of iterations needed
    x = x0
    r = b - H(x)
    p = r
    rsold = np.sum(r * r)
    cf = []
    cf.append(((H(x)-b)**2).sum())
    for i in range(max_iter):
        Hp = H(p)
        alpha = rsold / np.sum(p*Hp)
        x = x + alpha * p
        r = r - alpha * Hp
        rsnew = np.sum(r*r)
        print(rsnew)
        if np.sqrt(rsnew) < tol:
            return x,i
        p = r + (rsnew / rsold) * p
        rsold = rsnew
        cf.append(((H(x) - b) ** 2).sum())
    return x, cf





from skimage.restoration import denoise_tv_chambolle, denoise_tv_bregman, denoise_wavelet
def sart_update(vol, P, nit, nit_tv, lamb, tau):
    n1, n2, n3 = vol.shape
    v = vol
    proj_geom = astra.create_proj_geom('parallel3d', 1, 1, n1, n2, angles)
    vol_geom = astra.create_vol_geom(n3, n1, n2)
    def A(vv):
        vv = vv/np.sqrt((vv**2).sum())
        out = FP(vv, proj_geom, vol_geom, P.shape[0], n1, n2, n3, nit=10, device_num=0)
        # out = astra.create_sino3d_gpu(vv.swapaxes(1, 2), proj_geom, vol_geom, returnData=True, gpuIndex=config.device.index)[
        # 1].swapaxes(0, 1)
        return out
    # At = lambda P: BP(P, proj_geom, vol_geom, n1, n2, n3, device_num=config.device.index)
    def At(P):
        out = BP(P, proj_geom, vol_geom, n1, n2, n3, device_num=config.device.index)
        # out = out/np.abs(out).max()*np.abs(P).max()
        out = out / np.sqrt((out ** 2).sum())
        return out

    cf = []
    # mask = np.zeros_like(v)
    # mask[20:-20,20:-20,20:-20] = 1
    for i in range(nit):
        v = v - tau*(At(A(v) - P))#*mask
        cf.append(((A(v) - P) ** 2).sum())
        # Total Variation regularization
        # v = denoise_tv_chambolle(v, weight=lamb, max_num_iter=nit_tv)
        # v = denoise_tv_bregman(v, weight=lamb, max_num_iter=nit_tv)
        v = denoise_wavelet(v, sigma=lamb)
    return v, cf

n1, n2, n3 = config.n1, config.n2, config.n3
proj_geom = astra.create_proj_geom('parallel3d', 1, 1, n1, n2, angles)
vol_geom = astra.create_vol_geom(n3, n1, n2)
def A(vv):
    vv = vv / np.sqrt((vv ** 2).sum())
    out = FP(vv, proj_geom, vol_geom, len(angles), n1, n2, n3, nit=10, device_num=0)
    # out = astra.create_sino3d_gpu(vv.swapaxes(1, 2), proj_geom, vol_geom, returnData=True, gpuIndex=config.device.index)[
    # 1].swapaxes(0, 1)
    return out
# At = lambda P: BP(P, proj_geom, vol_geom, n1, n2, n3, device_num=config.device.index)
def At(P):
    out = BP(P, proj_geom, vol_geom, n1, n2, n3, device_num=config.device.index)
    # out = out/np.abs(out).max()*np.abs(P).max()
    out = out / np.sqrt((out ** 2).sum())
    return out
s = 3*1e-2
P = A(V) + s*np.random.randn(config.Nangles, n1, n2)

v_sart, cf = sart_update(np.zeros_like(V)+1e-1, P, lamb=2*1e-3, tau=1e2, nit=10, nit_tv=10)
plt.figure(1)
plt.clf()
plt.plot(cf)
plt.figure(2)
plt.clf()
plt.imshow(v_sart[50:-50,50:-50,90])

vol = np.zeros_like(V)
v_sirt = sirt_prox(P, vol, 1, angles, nit=10)
plt.figure(3)
plt.clf()
plt.imshow(v_sirt[50:-50,50:-50,90])


# Denoising
vv = denoise_wavelet(v_sart, sigma=1e-1, mode='soft')
plt.figure(4)
plt.clf()
plt.imshow(vv[50:-50,50:-50,90])
plt.title('Wavelet')

vv, cf = FISTA_prox_tv(v_sart, lamb=1e1, tau=1e-2, nit=10)
plt.figure(5)
plt.clf()
plt.imshow(vv[50:-50,50:-50,90])
plt.title('prox tv')
plt.figure(1)
plt.clf()
plt.plot(cf)

# Try my own TV? first on one image then iterate
# why tv prox diverges ?

# x, cf = CG_inverse(P, np.zeros_like(V), angles, 1, 10, 1e-5)
#
# plt.figure(1)
# plt.clf()
# plt.plot(cf)
#
# plt.figure(2)
# plt.clf()
# plt.imshow(x[:,:,90])
from skimage.restoration import denoise_nl_means, estimate_sigma

# ADMM
mu = 1e-3
rho = 1e-2
lamb = 1e1
nit = 5
nit_sirt = 3
nit_nlm = 0
p_dist = 20
p_sze = 4
P = proj.detach().cpu().numpy()
vol = np.zeros_like(V)
v_sirt = sirt_prox(P, vol, mu, angles, nit_sirt)
vol = v_sirt

def ADMM(vol, P, rho, mu, lamb, nit, nit_sirt, nit_nlm, p_dist=10, p_sze=5):
    n1, n2, n3 = vol.shape
    v = vol
    z = np.zeros((3,n1,n2,n3))
    y = np.zeros((3,n1,n2,n3))

    cf = []
    proj_geom = astra.create_proj_geom('parallel3d', 1, 1, n1, n2, angles)
    vol_geom = astra.create_vol_geom(n3, n1, n2)
    operator = lambda vv: astra.create_sino3d_gpu(vv.swapaxes(1, 2), proj_geom, vol_geom, returnData=True, gpuIndex=config.device.index)[
        1].swapaxes(0, 1)

    for i in range(nit):
        tmp1 = lamb*d1(v) - z[0] + y[0]
        tmp2 = lamb*d2(v) - z[1] + y[1]
        tmp3 = lamb*d3(v) - z[2] + y[2]
        inp = v - lamb*mu/rho*(d1T(tmp1) + d2T(tmp2) + d3T(tmp3))
        # v = sirt_prox(P, inp, mu, nit_sirt)
        v, cf_cg = CG_inverse(P, inp, angles, mu, 4, 1e-5)

        z[0] = prox_l1(lamb*d1(v)+y[0],rho)
        z[1] = prox_l1(lamb*d2(v)+y[1],rho)
        z[2] = prox_l1(lamb*d3(v)+y[2],rho)

        y[0] += d1(v) - z[0]
        y[1] += d2(v) - z[1]
        y[2] += d3(v) - z[2]

        cf.append(((operator(v) - P)**2).mean() + lamb * np.mean( np.abs(d1(v)) + np.abs(d2(v)) + np.abs(d3(v))) )

    for i in range(nit_nlm):
        tmp1 = lamb*d1(v) - z[0] + y[0]
        tmp2 = lamb*d2(v) - z[1] + y[1]
        tmp3 = lamb*d3(v) - z[2] + y[2]
        inp = v - lamb*mu/rho*(d1T(tmp1) + d2T(tmp2) + d3T(tmp3))
        # v = sirt_prox(P, inp, mu, nit_sirt)
        v, _ = CG_inverse(P, inp, angles, mu, 4, 1e-5)
        sigma_est = np.mean(estimate_sigma(v, channel_axis=-1))
        sigma_est = np.mean(estimate_sigma(v))
        patch_kw = dict(
            patch_size=p_sze, patch_distance=p_dist  # 5x5 patches  # 13x13 search area
        )
        v = denoise_nl_means(v, h=0.8 * sigma_est, fast_mode=True, **patch_kw)

        z[0] = prox_l1(lamb*d1(v)+y[0],rho)
        z[1] = prox_l1(lamb*d2(v)+y[1],rho)
        z[2] = prox_l1(lamb*d3(v)+y[2],rho)

        y[0] += d1(v) - z[0]
        y[1] += d2(v) - z[1]
        y[2] += d3(v) - z[2]

        cf.append(((operator(v) - P)**2).sum() + lamb * np.sum( np.abs(d1(v)) + np.abs(d2(v)) + np.abs(d3(v))) )
    return v, cf

# v_admm, cf = ADMM(vol, P, rho, mu, lamb, nit, nit_sirt, nit_nlm, p_dist, p_sze)
#
# plt.figure(1)
# plt.clf()
# plt.plot(cf)
#
# plt.figure(2)
# plt.clf()
# plt.imshow(np.clip(v_admm[:,:,90],-1,1e0))





def FISTA_prox_tv(z, lamb, tau, nit, eps=1e-5):
    # solve min_u ||u-z||_2^2 + lamb*||\nabla u||_2
    n1, n2, n3 = z.shape
    v = z.copy()
    u = v.copy()
    cf = []
    for i in range(nit):
        v_ = v.copy()
        norm = np.sqrt(d1(u)**2 + d2(u)**2 + d3(u)**2 + eps**2).sum()
        grad = (u - z) + lamb*(d1T(d1(u))+d2T(d2(u))+d3T(d3(u)))/norm
        for j in range(10):
            vtmp = v - tau*grad
            cost = 0.5*((vtmp - z)**2).sum() + lamb * np.sum(np.sqrt(np.abs(d1(vtmp))**2 + np.abs(d2(vtmp))**2 + np.abs(d3(vtmp))**2 ))
            if i==0:
                break
            if cost>cf[-1]:
                tau = tau*0.5
            # if cost<cf[-1]:
            #     tau = tau/0.5
        print(tau)
        v = v - tau*grad
        # u = v + 0.99*(v-v_)
        u = v
        cf.append((0.5*(v - z)**2).sum() + lamb *  np.sum(np.sqrt(np.abs(d1(vtmp))**2 + np.abs(d2(vtmp))**2 + np.abs(d3(vtmp))**2 )) )
    return v, cf





def FISTA(vol, P, lamb, tau, nit, eps=1e-5):
    n1, n2, n3 = vol.shape
    v = vol
    u = v.copy()

    proj_geom = astra.create_proj_geom('parallel3d', 1, 1, n1, n2, angles)
    vol_geom = astra.create_vol_geom(n3, n1, n2)
    def A(vv):
        vv = vv/np.sqrt((vv**2).sum())
        out = astra.create_sino3d_gpu(vv.swapaxes(1, 2), proj_geom, vol_geom, returnData=True, gpuIndex=config.device.index)[
        1].swapaxes(0, 1)
        return out
    # At = lambda P: BP(P, proj_geom, vol_geom, n1, n2, n3, device_num=config.device.index)
    def At(P):
        out = BP(P, proj_geom, vol_geom, n1, n2, n3, device_num=config.device.index)
        # out = out/np.abs(out).max()*np.abs(P).max()
        out = out / np.sqrt((out ** 2).sum())
        return out

    cf = []
    for i in range(nit):
        v_ = v.copy()
        norm = np.sqrt(d1(u)**2 + d2(u)**2 + d3(u)**2 + eps**2).sum()
        grad = At(A(u) - P) + 0.5*lamb*(d1T(u)+d2T(u)+d3T(u))/norm
        for j in range(10):
            vtmp = v - tau*grad
            cost = ((A(vtmp) - P)**2).sum() + lamb * np.sum( np.abs(d1(vtmp)) + np.abs(d2(vtmp)) + np.abs(d3(vtmp)))
            if i==0:
                break
            if cost>cf[-1]:
                tau = tau*0.5
            # if cost<cf[-1]:
            #     tau = tau/0.5
        print(tau)
        v = vtmp
        u = v + 0.99*(v-v_)
        cf.append(((A(v) - P)**2).sum() + lamb * np.sum( np.abs(d1(v)) + np.abs(d2(v)) + np.abs(d3(v))) )
    return v, cf


# n1, n2, n3 = vol.shape
# proj_geom = astra.create_proj_geom('parallel3d', 1, 1, n1, n2, angles)
# vol_geom = astra.create_vol_geom(n3, n1, n2)
# def A(vv):
#     vv = vv/np.sqrt((vv**2).sum())
#     out = astra.create_sino3d_gpu(vv.swapaxes(1, 2), proj_geom, vol_geom, returnData=True, gpuIndex=config.device.index)[
#     1].swapaxes(0, 1)
#     return out
# # At = lambda P: BP(P, proj_geom, vol_geom, n1, n2, n3, device_num=config.device.index)
# def At(P):
#     out = BP(P, proj_geom, vol_geom, n1, n2, n3, device_num=config.device.index)
#     # out = out/np.abs(out).max()*np.abs(P).max()
#     out = out / np.sqrt((out ** 2).sum())
#     return out
# P = A(V)
#
# v_fista, cf = FISTA(vol, P, lamb=5*1e1, tau=1e-1, nit=10, eps=1e-5)
#
# plt.figure(1)
# plt.clf()
# plt.plot(cf)
#
# plt.figure(2)
# plt.clf()
# plt.imshow(v_fista[:,:,90])




# sigma_est = np.mean(estimate_sigma(v_admm))
# patch_kw = dict(
#     patch_size=p_sze, patch_distance=p_dist  # 5x5 patches  # 13x13 search area
# )
# denoise_fast = denoise_nl_means(v_admm, h=0.8 * sigma_est, fast_mode=True, **patch_kw)



def denoise_TV_L2_bounds(z, alpha, a, b, nit, x1=None, x2=None, x3=None):
# This function solves:
# min_{a <= x <= b} alpha ||Nabla x ||_1 + 0.5 || x - z ||_2^2
# with an accelerated gradient descent on the dual
    if x1 is None:
        x1 = np.zeros_like(z)
    if x2 is None:
        x2 = np.zeros_like(z)
    if x3 is None:
        x3 = np.zeros_like(z)
    y1 = x1
    y2 = x2
    y3 = x3

    tau = 1/64
    cf = []
    for i in range(nit):
        tmp = z - d1T(y1) - d2T(y2) - d3T(y3)
        grad1 = d1(tmp)
        grad2 = d2(tmp)
        grad3 = d3(tmp)

        xp1 = x1
        xp2 = x2
        xp3 = x3

        x1 = y1 + tau*grad1
        x2 = y2 + tau*grad2
        x3 = y3 + tau*grad3
        nx = np.sqrt(x1**2+x2**2+x3**2)+1e-5
        x1 = (x1/nx)*np.minimum(nx,alpha)
        x2 = (x2/nx)*np.minimum(nx,alpha)
        x3 = (x3/nx)*np.minimum(nx,alpha)

        y1 = x1 + 0.99 * (x1 - xp1)
        y2 = x2 + 0.99 * (x2 - xp2)
        y3 = x3 + 0.99 * (x3 - xp3)

        u = z - d1T(x1) - d2T(x2) - d3T(x3)
        loss = alpha*(np.sqrt(d1(u)**2+d2(u)**2+d3(u)**2)).sum() + 0.5*((u-z)**2).sum()
        cf.append(loss)

    # return np.maximum(np.minimum(u,b),a), cf
    return u, cf




# def apply_ATA (x):
#     yy=diffraction_Fourier_domain_discrete_volume(x,anglesSet,lamb, Cs, dF, eps, sigma_aliasing)
#     V_FBP= fbp_diffraction(yy,anglesSet,n1,n2,n3,lamb,Cs,dF,device_num,filter_=False,filter_type=ramp_diff)
#     V_FBP /= n1 * n2 * n3
#     #V_FBP*=0.0005
#     return V_FBP
#


#
#
# operator_ET = ParallelBeamGeometry3DOpAngles_rectangular((config.n1,config.n2,config.n3), angles/180*np.pi, fact=1)
# lamb = 1 # regul
# rho = 1 # admm param
# nit = 10
# nit_sirt = 10
# def ADMM(vol, proj, rho, mu, lamb, nit, nit_sirt):
#     n1, n2, n3 = vol.shape
#     v = vol
#     y = np.zeros((3,n1,n2,n3))
#     u1 = d1(vol)
#     u2 = d2(vol)
#     u3 = d3(vol)
#     ut1 = u1
#     ut2 = u2
#     ut3 = u3
#     BP_proj
#     for i in range(nit):
#         u1 = prox_l1(u1, lamb/rho)
#         u2 = prox_l1(u2, lamb/rho)
#         u3 = prox_l1(u3, lamb/rho)
#         b = BP_proj + rho*( d1T(u1-ut1) + d2T(u2-ut2) + d3T(u3-ut3) )
#         c =
#         v = SIRT(proj, angles, nit_sirt)
#
#
#
#
# alpha = 1e-2
# a = -1
# b = 1
# nit = 10
# V_est_denoise, cf = denoise_TV_L2_bounds(result, alpha, a, b, nit, x1=None, x2=None)
#
# plt.figure(1)
# plt.clf()
# plt.plot(cf)
#
# plt.figure(2)
# plt.clf()
# plt.subplot(1,2,1)
# plt.imshow(V_est_denoise[:,:,config.n3//2])
# plt.subplot(1,2,2)
# plt.imshow(result[:,:,config.n3//2])




# ######################################################################################################
# ## Load and estimate our volume
# ######################################################################################################
# ## Load implicit network
# if(config.volume_model=="Fourier-features"):
#     from models.fourier_net import FourierNet,FourierNet_Features
#     impl_volume = FourierNet_Features(
#         in_features=config.input_size_volume,
#         sub_features=config.sub_features,
#         out_features=config.output_size_volume,
#         hidden_features=config.hidden_size_volume,
#         hidden_blocks=config.num_layers_volume,
#         L = config.L_volume).to(device)
#
# if(config.volume_model=="MLP"):
#     from models.fourier_net import MLP
#     impl_volume = MLP(in_features= 1,
#                         hidden_features=config.hidden_size_volume, hidden_blocks= config.num_layers_volume, out_features=config.output_size_volume).to(device)
#
# if(config.volume_model=="multi-resolution"):
#     import tinycudann as tcnn
#     config_network = {"encoding": {
#             'otype': config.encoding.otype,
#             'type': config.encoding.type,
#             'n_levels': config.encoding.n_levels,
#             'n_features_per_level': config.encoding.n_features_per_level,
#             'log2_hashmap_size': config.encoding.log2_hashmap_size,
#             'base_resolution': config.encoding.base_resolution,
#             'per_level_scale': config.encoding.per_level_scale,
#             'interpolation': config.encoding.interpolation,
#         },
#         "network": {
#             "otype": config.network.otype,
#             "activation": config.network.activation,
#             "output_activation": config.network.output_activation,
#             "n_neurons": config.hidden_size_volume,
#             "n_hidden_layers": config.num_layers_volume
#         }
#         }
#     impl_volume = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=1, encoding_config=config_network["encoding"],
#                                                 network_config=config_network["network"]).to(device)
# num_param = sum(p.numel() for p in impl_volume.parameters() if p.requires_grad)
# print('---> Number of trainable parameters in volume net: {}'.format(num_param))
# checkpoint = torch.load(os.path.join(config.path_save,'training','model_trained.pt'),map_location=device)
# impl_volume.load_state_dict(checkpoint['implicit_volume'])
# shift_icetide = checkpoint['shift_est']
# rot_icetide = checkpoint['rot_est']
# implicit_deformation_icetide = checkpoint['local_deformation_network']
# ## Compute our model at same resolution than other volume
# rays_scaling = torch.tensor(np.array(config.rays_scaling))[None,None,None].type(config.torch_type).to(device)
# n1_eval, n2_eval, n3_eval = V.shape
#
# # Compute estimated volume
# with torch.no_grad():
#     x_lin1 = np.linspace(-1,1,n1_eval)*rays_scaling[0,0,0,0].item()/2+0.5
#     x_lin2 = np.linspace(-1,1,n2_eval)*rays_scaling[0,0,0,1].item()/2+0.5
#     XX, YY = np.meshgrid(x_lin1,x_lin2,indexing='ij')
#     grid2d = np.concatenate([XX.reshape(-1,1),YY.reshape(-1,1)],1)
#     grid2d_t = torch.tensor(grid2d).type(config.torch_type)
#     z_range = np.linspace(-1,1,n3_eval)*rays_scaling[0,0,0,2].item()*(n3_eval/n1_eval)/2+0.5
#     V_icetide = np.zeros_like(V)
#     for zz, zval in enumerate(z_range):
#         grid3d = np.concatenate([grid2d_t, zval*torch.ones((grid2d_t.shape[0],1))],1)
#         grid3d_slice = torch.tensor(grid3d).type(config.torch_type).to(device)
#         estSlice = impl_volume(grid3d_slice).detach().cpu().numpy().reshape(config.n1,config.n2)
#         V_icetide[:,:,zz] = estSlice
#     V_icetide_t = torch.tensor(V_icetide).type(config.torch_type).to(device)

# ######################################################################################################
# # Using only the deformation estimates
# ######################################################################################################
# projections_noisy_undeformed = torch.zeros_like(projections_noisy)
# xx1 = torch.linspace(-1,1,config.n1,dtype=config.torch_type,device=device)
# xx2 = torch.linspace(-1,1,config.n2,dtype=config.torch_type,device=device)
# XX_t, YY_t = torch.meshgrid(xx1,xx2,indexing='ij')
# XX_t = torch.unsqueeze(XX_t, dim = 2)
# YY_t = torch.unsqueeze(YY_t, dim = 2)
# for i in range(config.Nangles):
#     coordinates = torch.cat([XX_t,YY_t],2).reshape(-1,2)
#     #field = utils_deformation.deformation_field(-implicit_deformation_icetide[i].depl_ctr_pts[0].detach().clone())
#     thetas = torch.tensor(-rot_icetide[i].thetas.item()).to(device)
#
#     rot_deform = torch.stack(
#                     [torch.stack([torch.cos(thetas),torch.sin(thetas)],0),
#                     torch.stack([-torch.sin(thetas),torch.cos(thetas)],0)]
#                     ,0)
#     coordinates = coordinates - config.deformationScale*implicit_deformation_icetide[i](coordinates)
#     coordinates = coordinates - shift_icetide[i].shifts_arr
#     coordinates = torch.transpose(torch.matmul(rot_deform,torch.transpose(coordinates,0,1)),0,1) ## do rotation
#     x = projections_noisy[i].clone().view(1,1,config.n1,config.n2)
#     x = x.expand(config.n1*config.n2, -1, -1, -1)
#     out = cropper(x,coordinates,output_size = 1).reshape(config.n1,config.n2)
#     projections_noisy_undeformed[i] = out
# V_FBP_icetide = reconstruct_FBP_volume(config, projections_noisy_undeformed).detach().cpu().numpy()


#######################################################################################
## Compute FSC
#######################################################################################
fsc_icetide = utils_FSC.FSC(V,V_icetide)
fsc_FBP_icetide = utils_FSC.FSC(V,V_FBP_icetide)
fsc_FBP = utils_FSC.FSC(V,V_FBP)
fsc_FBP_no_deformed = utils_FSC.FSC(V,V_FBP_no_deformed)
if(eval_AreTomo):
    fsc_AreTomo = utils_FSC.FSC(V,V_FBP_aretomo)
if(eval_Etomo):
    fsc_Etomo = utils_FSC.FSC(V,V_FBP_etomo)

x_fsc = np.arange(fsc_FBP.shape[0])


plt.figure(1)
plt.clf()
plt.plot(x_fsc,fsc_icetide,'b',label="icetide")
plt.plot(x_fsc,fsc_FBP_icetide,'--b',label="FBP with our deform. est. ")
if(eval_AreTomo):
    plt.plot(x_fsc,fsc_AreTomo,'r',label="AreTomo")
if(eval_Etomo):
    plt.plot(x_fsc,fsc_Etomo,'c',label="Etomo")
plt.plot(x_fsc,fsc_FBP,'k',label="FBP")
plt.plot(x_fsc,fsc_FBP_no_deformed,'g',label="FBP no def.")
plt.legend()
plt.savefig(os.path.join(config.path_save,'evaluation','FSC.png'))
plt.savefig(os.path.join(config.path_save,'evaluation','FSC.pdf'))


fsc_arr = np.zeros((x_fsc.shape[0],7))
fsc_arr[:,0] = x_fsc
fsc_arr[:,1] = fsc_icetide[:,0]
fsc_arr[:,2] = fsc_FBP[:,0]
fsc_arr[:,3] = fsc_FBP_no_deformed[:,0]
if(eval_AreTomo):
    fsc_arr[:,4] = fsc_AreTomo[:,0]
if(eval_Etomo):
    fsc_arr[:,5] = fsc_Etomo[:,0]
fsc_arr[:,6] = fsc_FBP_icetide[:,0]
# fsc_arr[:,6] = fsc_icetide_isonet[:,0]
header ='x,icetide,FBP,FBP_no_deformed,AreTomo,ETOMO,FBP_est_deformed'
np.savetxt(os.path.join(config.path_save,'evaluation','FSC.csv'),fsc_arr,header=header,delimiter=",",comments='')


# -compute error between local deformations


# if eval_AreTomo:
#     # Extract the estimated deformations for AreTomo

#     ARETOMO_FILENAME = 'projections.aln'
#     with open(config.path_save+ARETOMO_FILENAME, 'r',encoding="utf-8") as file:
#         lines = file.readlines()

#     comments = []
#     affine_transforms_aretomo = []
#     local_transforms_aretomo = []

#     affine_flag = True
#     num_patches = 0
#     for line in lines:
#         if line.startswith('#'):
#             comments.append(line.strip())  # Strip removes leading/trailing whitespace
#             if line.startswith('# Local Alignment'):
#                 affine_flag = False
#             if line.startswith('# NumPatches'):
#                 num_patches = int(line.split('=')[-1])
#         else:
#             if affine_flag:
#                 affine_transforms_aretomo.append(line.strip().split())
#             else:
#                 local_transforms_aretomo.append(line.strip().split())  

#     aretomo_data = pd.DataFrame(affine_transforms_aretomo, 
#                                 columns=["SEC", "ROT", "GMAG", "TX", 
#                                                         "TY", "SMEAN", "SFIT", "SCALE",
#                                                             "BASE", "TILT"])
    
#     x_shifts_aretomo = aretomo_data['TY'].values.astype(np.float32)
#     y_shifts_aretomo = aretomo_data['TX'].values.astype(np.float32)
#     inplane_rotation_aretomo = aretomo_data['ROT'].values.astype(np.float32)

#     # The estimates are already in pixels
#     error_x_shifts_aretomo = np.around(np.abs(x_shifts*n1_eval/2-x_shifts_aretomo).mean(),decimals=4)
#     error_y_shifts_aretomo = np.around(np.abs(y_shifts*n1_eval/2-y_shifts_aretomo).mean(),decimals=4)
#     error_inplane_rotation_aretomo = np.around(np.abs(inplane_rotation-
#                                                     inplane_rotation_aretomo).mean(),decimals=4)
    
#     error_arr.loc[3] = ['AreTomo',error_x_shifts_aretomo,error_y_shifts_aretomo,
#                         error_inplane_rotation_aretomo]
    

    
#     local_AreTomo = np.zeros((config.Nangles,num_patches,4))

#     for local_est in local_transforms_aretomo:
#         angle_index =int(local_est[0])
#         patch_index = int(local_est[1])

#         local_AreTomo[angle_index,patch_index,0] = float(local_est[2])
#         local_AreTomo[angle_index,patch_index,1] = float(local_est[3])
#         local_AreTomo[angle_index,patch_index,2] = float(local_est[4])
#         local_AreTomo[angle_index,patch_index,3] = float(local_est[5])
        





# # save the error in a csv file
# error_arr.to_csv(os.path.join(config.path_save,'evaluation'+'/affine_error.csv'),index=False)


#######################################################################################
## Save slices of volumes
#######################################################################################
saveIndex = [n3_eval//4,n3_eval//2,int(3*n3_eval//4)] # The slices to save taken from previous plots
for index in saveIndex:
    # True volume
    tmp = V[:,:,index]
    tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
    tmp = np.floor(255*tmp).astype(np.uint8)
    imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","true","slice_{}.png".format(index)),tmp)

    # ICETIDE
    tmp = V_icetide[:,:,index]
    tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
    tmp = np.floor(255*tmp).astype(np.uint8)
    imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","ICETIDE","slice_{}.png".format(index)),tmp)

    # FBP
    tmp = V_FBP[:,:,index]
    tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
    tmp = np.floor(255*tmp).astype(np.uint8)
    imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","FBP","slice_{}.png".format(index)),tmp)

    # FBP no deformed
    tmp = V_FBP_no_deformed[:,:,index]
    tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
    tmp = np.floor(255*tmp).astype(np.uint8)
    imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","FBP_no_deformed","slice_{}.png".format(index)),tmp)

    if(eval_AreTomo):
        tmp = V_FBP_aretomo[:,:,index]
        tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
        tmp = np.floor(255*tmp).astype(np.uint8)
        imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","AreTomo","slice_{}.png".format(index)),tmp)

    if(eval_Etomo):
        tmp = V_FBP_etomo[:,:,index]
        tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
        tmp = np.floor(255*tmp).astype(np.uint8)
        imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","Etomo","slice_{}.png".format(index)),tmp)

    # FBP icetide
    tmp = V_FBP_icetide[:,:,index]
    tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
    tmp = np.floor(255*tmp).astype(np.uint8)
    imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","FBP_ICETIDE","slice_{}.png".format(index)),tmp)


#######################################################################################
## Generate projections
#######################################################################################
# Define angles and X-ray transform
angles = np.linspace(config.view_angle_min,config.view_angle_max,config.Nangles)
operator_ET = ParallelBeamGeometry3DOpAngles_rectangular((config.n1,config.n2,config.n3), angles/180*np.pi, fact=1)

projections_icetide = operator_ET(V_icetide_t).detach().cpu().numpy()
projections_FBP = operator_ET(V_FBP_t).detach().cpu().numpy()
projections_FBP_no_deformed = operator_ET(V_FBP_no_deformed_t).detach().cpu().numpy()
projections_FBP_icetide = projections_noisy_undeformed.detach().cpu().numpy()
if(eval_AreTomo):
    V_FBP_aretomo_t = torch.tensor(V_FBP_aretomo).to(device)
    projections_AreTomo = operator_ET(V_FBP_aretomo_t).detach().cpu().numpy()
if(eval_Etomo):
    V_FBP_etomo_t = torch.tensor(V_FBP_etomo).to(device)
    projections_Etomo = operator_ET(V_FBP_etomo_t).detach().cpu().numpy()

out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","icetide_projections.mrc"),projections_icetide.astype(np.float32),overwrite=True)
out.close()
if(eval_AreTomo):
    out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","AreTomo_projections.mrc"),projections_AreTomo.astype(np.float32),overwrite=True)
    out.close()
if(eval_Etomo):
    out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","Etomo_projections.mrc"),projections_Etomo.astype(np.float32),overwrite=True)
    out.close()

out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","FBP_projections.mrc"),projections_FBP.astype(np.float32),overwrite=True)
out.close()
out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","FBP_no_deformed_projections.mrc"),projections_FBP_no_deformed.astype(np.float32),overwrite=True)
out.close()
out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","FBP_icetide_projections.mrc"),projections_FBP_icetide.astype(np.float32),overwrite=True)

for k in range(config.Nangles):
    tmp = projections_icetide[k]
    tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
    tmp = np.floor(255*tmp).astype(np.uint8)
    imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"projections","ICETIDE","snapshot_{}.png".format(k)),tmp)
    if(eval_AreTomo):
        tmp = projections_AreTomo[k]
        tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
        tmp = np.floor(255*tmp).astype(np.uint8)
        imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"projections","AreTomo","snapshot_{}.png".format(k)),tmp)
    if(eval_Etomo):
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
    tmp = projections_FBP_icetide[k]
    tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
    tmp = np.floor(255*tmp).astype(np.uint8)
    imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"projections","FBP_ICETIDE","snapshot_{}.png".format(k)),tmp)

out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',
                            "volumes","ICETIDE_volume.mrc"),np.moveaxis(V_icetide,2,0),overwrite=True)
out.close()
if eval_AreTomo:
    out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',
                                "volumes","AreTomo_volume.mrc"),np.moveaxis(V_FBP_aretomo,2,0),overwrite=True)
    out.close()
if eval_Etomo:
    out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',
                                "volumes","Etomo_volume.mrc"),np.moveaxis(V_FBP_etomo,2,0),overwrite=True)
    out.close()
out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"volumes",
                            "FBP_volume.mrc"),np.moveaxis(V_FBP,2,0),overwrite=True)
out.close()
out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"volumes",
                            "FBP_no_deformed_volume.mrc"),np.moveaxis(V_FBP_no_deformed,2,0),overwrite=True)
out.close()
out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"volumes",
                            "FBP_icetide_volume.mrc"),np.moveaxis(V_FBP_icetide,2,0),overwrite=True)

# ## Saving the inplance angles 
# inplaneAngles = np.zeros((config.Nangles,5))
# inplaneAngles[:,0] = angles
# inplaneAngles[:,1] = inplane_rotation
# inplaneAngles[:,2] = inplane_rotation_icetide
# if eval_AreTomo:
#     inplaneAngles[:,3] = inplane_rotation_aretomo
# if eval_ETOMO:
#     inplaneAngles[:,4] = inplane_rotation_etomo


# # save as a csv file
# header ='angles,true,icetide,AreTomo,Etomo'
# np.savetxt(os.path.join(config.path_save,'evaluation','inplane_angles.csv'),inplaneAngles,header=header,delimiter=",",comments='')


# TODO: compute error of local deformations and display
#######################################################################################
## Local deformation errror Estimation
#######################################################################################
x_lin1 = np.linspace(-1,1,n1_eval)*rays_scaling[0,0,0,0].item()/2+0.5
x_lin2 = np.linspace(-1,1,n2_eval)*rays_scaling[0,0,0,1].item()/2+0.5
XX, YY = np.meshgrid(x_lin1,x_lin2,indexing='ij')
grid2d = np.concatenate([XX.reshape(-1,1),YY.reshape(-1,1)],1)
grid2d_t = torch.tensor(grid2d).type(config.torch_type)
err_local_icetide = np.zeros(config.Nangles)
err_local_init = np.zeros(config.Nangles)
err_local_AreTomo = np.zeros(config.Nangles)

for k in range(config.Nangles):
    # Error in icetide
    grid_correction_true = local_tr[k](grid2d_t).detach().cpu().numpy()
    grid_correction_est_icetide = config.deformationScale*implicit_deformation_icetide[k](
        grid2d_t).detach().cpu().numpy()
    tmp = np.abs(grid_correction_true-grid_correction_est_icetide)
    err_local_icetide[k] = (0.5*config.n1*tmp[:,0]+0.5*config.n2*tmp[:,1]).mean()
    # Finidng the magnitude for init
    tmp = np.abs(grid_correction_true)
    err_local_init[k] = (0.5*config.n1*tmp[:,0]+0.5*config.n2*tmp[:,1]).mean()
    # Finding the error for AreTomo
    if eval_AreTomo:
        grid_correction_est_AreTomo = implicit_deformation_AreTomo[k](
            grid2d_t).detach().cpu().numpy()
        tmp = np.abs(grid_correction_true-grid_correction_est_AreTomo)
        err_local_AreTomo[k] = (0.5*config.n1*tmp[:,0]+0.5*config.n2*tmp[:,1]).mean()
    else: 
        err_local_AreTomo[k] = np.nan


# Save the error in a csv file
err_local_arr = np.zeros((config.Nangles,4))
err_local_arr[:,0] = angles
err_local_arr[:,1] = err_local_icetide
err_local_arr[:,2] = err_local_init
err_local_arr[:,3] = err_local_AreTomo

err_mean = np.nanmean(err_local_arr[:,1:],0)
err_std = np.nanstd(err_local_arr[:,1:],0)

err_local_arr = np.concatenate([np.array([err_mean,err_std])],0)

HEADER ='icetide,init,AreTomo'
np.savetxt(os.path.join(config.path_save,'evaluation','local_deformation_error.csv'),err_local_arr,header=HEADER,delimiter=",",comments='')


# Get the local deformation error plots 
for index in range(config.Nangles):
    # icetide
    savepath = os.path.join(config.path_save,'evaluation','deformations','ICETIDE','local_deformation_error_{}'.format(index))
    utils_display.display_local(implicit_deformation_icetide[index],local_tr[index],Npts=(20,20),scale=0.1, img_path=savepath)
    # Aretomo

    if eval_AreTomo:
        savepath = os.path.join(config.path_save,'evaluation','deformations','AreTomo','local_deformation_error_{}'.format(index))
        utils_display.display_local(implicit_deformation_AreTomo[index],local_tr[index],Npts=(20,20),scale=0.1, img_path=savepath )
