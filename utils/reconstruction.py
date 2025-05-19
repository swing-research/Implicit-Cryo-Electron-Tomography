import tomopy
import numpy as np

def TV_tomopy(projections, angles, reg_tv, nit_tv, n3):
    recon = tomopy.recon(projections, angles/180*np.pi,
                                     algorithm='tv', sinogram_order=False, reg_par=reg_tv, num_iter=nit_tv)
    recon = np.swapaxes(recon, 1,2)
    recon = recon[:, :, recon.shape[2] // 2 - n3 // 2:recon.shape[2] // 2 + n3 // 2]
    return recon[:, :, ::-1]

def FBP_tomopy(projections, angles, n3):
    recon = tomopy.recon(projections, angles/180*np.pi,
                                     algorithm='fbp', sinogram_order=False)
    recon = np.swapaxes(recon, 1,2)
    recon = recon[:, :, recon.shape[2] // 2 - n3 // 2:recon.shape[2] // 2 + n3 // 2]
    return recon[:, :, ::-1]

def SIRT_tomopy(projections, angles, nit, n3):
    recon = tomopy.recon(projections, angles/180*np.pi,
                                     algorithm='sirt', sinogram_order=False, num_iter=nit)
    recon = np.swapaxes(recon, 1,2)
    recon = recon[:, :, recon.shape[2] // 2 - n3 // 2:recon.shape[2] // 2 + n3 // 2]
    return recon[:, :, ::-1]