
from scipy.spatial import SphericalVoronoi
import  torch
import numpy as np
import torch.nn.functional as F
# import raster_geometry as mrt
import matplotlib.pyplot as plt

# from scipy.spatial.transform import Rotation as rot
import sys
# sys.path.insert(0,'./../')
from utils.data_generation import getRotationMatrix,rotate_t
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation as R

from utils import utils_sampling


def SNR(x, xhat):
  """Returns SNR of xhat wrt to gt image x."""

  # TODO(konik): assert x and xhat have same image.
  diff = x - xhat
  return -20*np.log10(np.linalg.norm(diff)/ np.linalg.norm(x))



def backprojection(projections,angles,weightType = 0,ignoreAxis=2,polynomialOrder = 2,order='X',degrees=True):
    #projections: samplesxnxn, angles samplesx3
    n = projections.shape[1]
    angleMatrixTorch = getRotationMatrix(angles,order=order,degrees=degrees)
    angleMatrixTorch = torch.tensor(angleMatrixTorch).to(projections.device).type_as(projections)
     
    projectionTorch = projections #torch.FloatTensor(projections)
    reconVolume = torch.zeros(n,n,n).to(projections.device).type_as(projections)

    if(weightType==0):
        weights = torch.ones((angleMatrixTorch.shape[0],1)).to(projections.device).type_as(projections)
    if(weightType==1):
        weights = computeWeigths(angles,ignoreAnlges=ignoreAxis)
    if(weightType==2):
        #print(angles)
        weights = computeWeigthsApprox(angles,ignoreAnlges=ignoreAxis)

    if(weightType==3):
        #n is the order of the polynomial
        #print(n)
        weights = computeWigner(angles,n = polynomialOrder)
    #print(weigths)

    weights = weights/torch.max(abs(weights))
    for angleMatrix,Ptorch,weigth in zip(angleMatrixTorch,projectionTorch,weights):
        Ptorch = Ptorch.type(torch.FloatTensor).to(projectionTorch.device).type_as(projections)
        backProjectedVolume = Ptorch.unsqueeze(2).repeat(1,1,Ptorch.shape[0])
        backProjectedVolume = rotate_t(backProjectedVolume,angleMatrix.T)
        reconVolume = reconVolume + weigth*backProjectedVolume
        #if(useweight):
        #    reconVolume = reconVolume + weigth*backProjectedVolume[0]#*sphere
        #else:
        #    reconVolume = reconVolume + backProjectedVolume[0]
    return reconVolume


def computeWeigths(angles, ignoreAnlges = 0 ):
    #print(angles)
    anglesNumpy  = angles.cpu().numpy()
    anglesSub = np.delete(anglesNumpy,ignoreAnlges,axis=1)
    anglesSpherical = np.zeros_like(anglesNumpy)

    anglesSpherical[:,0] = np.cos(anglesSub[:,0]*np.pi/180)*np.cos(anglesSub[:,1]*np.pi/180)
    anglesSpherical[:,1] = np.sin(anglesSub[:,0]*np.pi/180)*np.cos(anglesSub[:,1]*np.pi/180)
    anglesSpherical[:,2] = np.sin(-anglesSub[:,1]*np.pi/180)
    # anglesSpherical[:,0] = np.cos(anglesSub[:,0])*np.sin(anglesSub[:,1])
    # anglesSpherical[:,1] = np.sin(anglesSub[:,0])*np.sin(anglesSub[:,1])
    # anglesSpherical[:,2] = np.cos(anglesSub[:,1])
    #print(anglesSpherical)
    yVor = SphericalVoronoi(anglesSpherical)
    return torch.FloatTensor(yVor.calculate_areas()).to(angles.device)
    #anglesSpherical[:,0] = torch.


def computeWeigthsApprox(angles, ignoreAnlges = 0, scale =0.75):
    #print(scale)
    anglesNumpy  = angles.cpu().numpy()
    anglesSub = np.delete(anglesNumpy,ignoreAnlges,axis=1)
    anglesSpherical = np.zeros_like(anglesNumpy)
    anglesSpherical[:,0] = np.cos(anglesSub[:,0]*np.pi/180)*np.cos(anglesSub[:,1]*np.pi/180)
    anglesSpherical[:,1] = np.sin(anglesSub[:,0]*np.pi/180)*np.cos(anglesSub[:,1]*np.pi/180)
    anglesSpherical[:,2] = np.sin(-anglesSub[:,1]*np.pi/180)
    # anglesSpherical[:,0] = np.cos(anglesSub[:,0])*np.sin(anglesSub[:,1])
    # anglesSpherical[:,1] = np.sin(anglesSub[:,0])*np.sin(anglesSub[:,1])
    # anglesSpherical[:,2] = np.cos(anglesSub[:,1])
    #print(anglesSpherical)
    distances = np.arccos(1 - 0.5*euclidean_distances(anglesSpherical,anglesSpherical, squared=True))
    distances = distances + np.eye(len(distances))*np.max(distances)
    minDistance = np.min(distances,axis=1)

    areas = 2*np.pi*(1- np.cos(minDistance*scale))
    return torch.FloatTensor(areas).to(angles.device)
    #anglesSpherical[:,0] = torch

#def computeWeightS1(angles):

def computeWigner(angles,n=5):
    #angles : nsamplesx3 
    # n: is the order of the polynomial to be approximated
    print('N value',str(n))
    anglesXYZ = angles.cpu().numpy()
    angleScipy = R.from_euler('XYZ',anglesXYZ, degrees=True)
    anglesRad = angleScipy.as_euler('ZYZ',degrees=False)
    wigner = WignerWeigths()
    return torch.FloatTensor(wigner.getWeigths(anglesRad,n=n)).to(angles.device)



def compute_fsc(
        image_1: np.ndarray,
        image_2: np.ndarray,
        bin_width: int = 2.0, epsilon=1e-10
):
# Code modified from https://tttrlib.readthedocs.io/en/latest/auto_examples/imaging/plot_imaging_frc.html
    """ Computes the Fourier Ring/Shell Correlation of two 3-D volume

    :param image_1:
    :param image_2:
    :param bin_width:
    :return:
    """
    image_1 = image_1 / np.sum(image_1)
    image_2 = image_2 / np.sum(image_2)
    f1, f2 = np.fft.fftn(image_1), np.fft.fftn(image_2)
    af1f2 = np.real(f1 * np.conj(f2))
    af1_2, af2_2 = np.abs(f1)**2, np.abs(f2)**2
    nx, ny,nz = af1f2.shape
    x = np.arange(-np.floor(nx / 2.0), np.ceil(nx / 2.0))
    y = np.arange(-np.floor(ny / 2.0), np.ceil(ny / 2.0))
    z = np.arange(-np.floor(nz / 2.0), np.ceil(nz / 2.0))
    distances = list()
    wf1f2 = list()
    wf1 = list()
    wf2 = list()
    for xi, yi in np.array(np.meshgrid(x,y)).T.reshape(-1, 2):
        #print(xi)
        for zi in z:
            distances.append(np.sqrt(xi**2 + xi**2+zi**2))
            xi = int(xi)
            yi = int(yi)
            zi = int(zi)
            wf1f2.append(af1f2[xi, yi,zi])
            wf1.append(af1_2[xi, yi,zi])
            wf2.append(af2_2[xi, yi,zi])

    bins = np.arange(0, np.sqrt((nx//2)**2 + (ny//2)**2+(nz//2)**2 ), bin_width)
    f1f2_r, bin_edges = np.histogram(
        distances,
        bins=bins,
        weights=wf1f2
    )
    f12_r, bin_edges = np.histogram(
        distances,
        bins=bins,
        weights=wf1
    )
    f22_r, bin_edges = np.histogram(
        distances,
        bins=bins,
        weights=wf2
    )
    density = f1f2_r / np.sqrt(f12_r * f22_r + epsilon)
    return density, bin_edges

# Not sure which is correct
# def getfsc(vol1,vol2,mask=None):
#     #Works for even n
#     #This is the function used in cryodrgn    
#     D = vol1.shape[0]
#     x = np.arange(-D//2, D//2)
#     x2, x1, x0 = np.meshgrid(x,x,x, indexing='ij')
#     coords = np.stack((x0,x1,x2), -1)
#     r = (coords**2).sum(-1)**.5
    
#     #print(r[D//2, D//2, D//2])

#     #assert r[D//2, D//2, D//2] == 0.0

#     vol1 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(vol1)))
#     vol2 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(vol2)))

#     #log(r[D//2, D//2, D//2:])
#     prev_mask = np.zeros((D,D,D), dtype=bool)
#     fsc = [1.0]
#     for i in range(1,D//2):
#         mask = r < i
#         shell = np.where(mask & np.logical_not(prev_mask))
#         v1 = vol1[shell]
#         v2 = vol2[shell]
#         p = np.vdot(v1,v2) / (np.vdot(v1,v1)*np.vdot(v2,v2))**.5
#         #print(p)
#         fsc.append(p.real)
#         prev_mask = mask
#     #print(fsc)
#     fsc = np.asarray(fsc)
#     x = np.arange(D//2)/D
    
#     return fsc,x


def getfsc(vol1,vol2,mask=None, DIndex = None):
    #Works for even n
    #This is the function used in cryodrgn   
    # ) 
    # DIndex  is the index of the dimension of the volume that is used to compute the fsc
    nx,ny,nz = vol1.shape

    if(DIndex is None):
        D = max(nx,ny,nz)
    else:
        D = vol1.shape[DIndex]

    x = np.linspace(-0.5, 0.5,nx,endpoint=False)
    y = np.linspace(-0.5, 0.5,ny,endpoint=False)
    z = np.linspace(-0.5, 0.5,nz,endpoint=False)

    x,y,z = np.meshgrid(x,y,z, indexing='ij')
    
    coords = np.stack((x,y,z), -1)
    #print(coords)
    r = (coords**2).sum(-1)**.5


    #print(r)
    
    #print(r[D//2, D//2, D//2])

    #assert r[D//2, D//2, D//2] == 0.0

    vol1 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(vol1)))
    vol2 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(vol2)))

    #log(r[D//2, D//2, D//2:])
    prev_mask = np.zeros((nx,ny,nz), dtype=bool)
    fsc = [1.0]
    for i in range(1,D//2):
        mask = r*D < i
        shell = np.where(mask & np.logical_not(prev_mask))
        v1 = vol1[shell]
        v2 = vol2[shell]
        #print(v1.dtype)
        p = np.vdot(v1,v2) / (np.vdot(v1,v1)*np.vdot(v2,v2))**.5
        #print(p)
        fsc.append(p.real)
        prev_mask = mask
    #print(fsc)
    fsc = np.asarray(fsc)
    x = np.arange(D//2)/D
    
    return fsc,x





def factorial(number):
    return np.math.factorial(number)

class WignerWeigths: 
    def wigner_d(self,j,mprime,m, beta):
        # j = 0,1,2,,... for SO(3)
        # m, mprime = -j, -j+1, ....j
        smin = max(0,m-mprime)
        smax = min(j+m,j-mprime)


        #print(factorial(j+mprime)*factorial(j-mprime)*factorial(j+m)*factorial(j-m))
        factorialVar =  np.sqrt(factorial(j+mprime))*np.sqrt(factorial(
            j-mprime))*np.sqrt(factorial(j+m))*np.sqrt(factorial(j-m))
        sumValue = 0
        for s in range(smin,smax+1):
            sumValue = sumValue +((-1)**(mprime-m+s))*(np.cos(beta/2)**(
                2*j+m-mprime+2*s))*(np.sin(beta/2)**(mprime-m+2*s))/(factorial(j+m-s)*factorial(
                s)*factorial(mprime-m+s)*factorial(j-mprime-s))

        return factorialVar*sumValue

    def wigner_DValue(self,j,m,mprime,alpha,beta,gamma):
        return np.exp(-(1j)*alpha*mprime)*self.wigner_d(j,m,mprime,beta)*np.exp(-(1j)*gamma*m)


    def wigner_DMatrx(self,j,alpha,beta,gamma):
        DjMatrix = np.zeros((2*j+1,2*j+1),dtype=np.complex64)
        for i, m in enumerate(range(-j,j+1)):
            for k, mprime in enumerate(range(-j,j+1)):
                #print(m,mprime)
                #ll = wigner_DValue(j,m,mprime,alpha,beta,gamma)
                #print(ll)  
                DjMatrix[i,k] = self.wigner_DValue(j,m,mprime,alpha,beta,gamma)

        return DjMatrix


    def wigner_DmatrixVectorize(self,alpha,beta,gamma,n):
        wdlist = []
        for i in range(0,n+1):
            wdmatrix = self.wigner_DMatrx(i,alpha,beta,gamma).flatten()
            wdlist.append(wdmatrix)

        return np.concatenate(wdlist)

    def wignerBasis(self,angles,n=2):
        #angles nx3 in radians
        print(n)
        dn = int((2*n+1)*(n+1)*(2*n+3)/3)
        wignerBasisMatrix = np.zeros((len(angles),dn),dtype=np.complex64)
        #return wignerBasisMatrix
        for index,angle in enumerate(angles):
            #print(n)
            wignerBasisMatrix[index] = self.wigner_DmatrixVectorize(angle[0],angle[1],angle[2],n=n)
        return wignerBasisMatrix

    def getRealMatrix(self,D,n=2):
        #Brute force method
        #TODO: As defined in the paper
        DR = D.real.T
        DI = - D.imag.T

        return np.vstack((DR,DI))


    def weigthSolver(self,A,xInit= None, L =10000,Iterations = 1000):
        #Solves the  problem \min \|Aw -e_0 \| s.t. w \geq 0
        # Where e_0 is a vector with first element 1 rest zeros
        e = np.zeros(A.shape[0])
        e[0] = 1
#         lb = np.zeros(A.shape[1])
#         res = lsq_linear(A, e, bounds=(lb, np.inf), verbose=1,tol=1e-10,max_iter=100)#,lsq_solver='exact')
#         return res.x
        
        #Solve using FISTA
        #if(xInit == None):
        #    xInit = np.zeros(A.shape[1])
        #xSol,cost =fista(A,e,xInit,L,Iterations)
        #plt.plot(cost)
        #plt.show()

        xSol = np.matmul(np.linalg.pinv(A),e)
        return xSol
        
       


    def getWeigths(self,angles,n = 2):
        Dmatrix = self.wignerBasis(angles,n=n)
        DReal = self.getRealMatrix(Dmatrix)
        wEstimate = self.weigthSolver(DReal)
        return wEstimate
    
def fista(A,b,xInit,L,iterations):
    ATAMatrix = np.matmul(A.T,A)
    ATb = np.matmul(A.T,b)/L
    W = (np.eye(xInit.shape[0]) - ATAMatrix/L)
    cost = []
    xk = np.copy(xInit)
    yk = np.copy(xInit)
    tk= 1 
    tnew = 0
    for i in range(iterations):
        xprev = np.copy(xk)
        xk = np.clip(np.matmul(W,yk) + ATb,0,None)
        tnew = (1+ np.sqrt(1+ 4*tk**2))/2
        yk = xk + ((tk-1)/tnew)*(xk - xprev)
        cost.append(np.linalg.norm(np.matmul(A,xk)-b))
    return xk, cost
        
        

"""
Function that compute different useful scores.
It computes the SNR of projections, 3D volume, FSC, and error between deformations

-shift_est: list of shift functions.
-rot_est: list of inplane rotation functions.
-local_est, local_true: list of local deformation functions.
-affine_tr_true: true affine transformation, contains shift and inplane rotations as an instance of AffineTransform.


-fsc_5 and fsc_143: resolution by thresholding the FSC between estimated volume and true volume for different thresold (0.5 and 0.143).
-SNR_est_vol: SNR between reconstruction and true volume.
-SNR_proj: SNR between clean projections
-err_shift, err_shift_init: average shift error in pixel. Init. is the error when assuming shifts are 0.
-err_rot, err_rot_init: average inplane rotation error in degrees. Init. is the error when assuming inplane angles are 0.
-err_local, err_local_init: average local deformation mismatch error in pixel. Init. is the error when assuming there are no local deformations.
"""
def compute_score_reconstruction(
        impl_volume, V_t, projections_clean, mollifier, grid,
        angles_t,
        shift_est,rot_est,local_est,affine_tr_true,local_true,
        DIndex=1):

    n1, n2, n3 = V_t.shape

    ## Compute scores of 3D reconstruction
    V_est = impl_volume(grid.grid_ball3d_t)
    V_est = utils_sampling.resample_ball(V_est,grid.ball3d)
    # compute SNr of volume
    if mollifier is not None:
        SNR_est_vol = SNR((mollifier.mollify3d_np()*V_t.detach().cpu().numpy()).reshape(-1) , (mollifier.mollify3d()*V_est).detach().cpu().numpy().reshape(-1))
    else:
        SNR_est_vol = SNR((V_t.detach().cpu().numpy()).reshape(-1) , (V_est).detach().cpu().numpy().reshape(-1))
    # Compute the fsc resolution
    if mollifier is not None:
        fsc,x_fsc = getfsc((mollifier.mollify3d()*V_est).detach().cpu().numpy().reshape(n1,n2,n3),V_t.detach().cpu().numpy().reshape(n1,n2,n3),DIndex=DIndex)
    else:
        fsc,x_fsc = getfsc((V_est).detach().cpu().numpy().reshape(n1,n2,n3),V_t.detach().cpu().numpy().reshape(n1,n2,n3),DIndex=DIndex)
    fsc_5 = 1/x_fsc[np.where(fsc>0.5)][-1]
    fsc_143 = 1/x_fsc[np.where(fsc>0.143)][-1]


    ## Compute scores on 2D projections
    SNR_proj = np.zeros(projections_clean.shape[0])
    for k in range(projections_clean.shape[0]):
        out = utils_sampling.sample_implicit(impl_volume,grid.grid_ball3d_t,angles_t[k],rot_deform=None,shift_deform=None,local_deform = None,scale=1)
        out = utils_sampling.resample_ball(out,grid.ball3d)
        proj_est = utils_sampling.projection(out)
        if mollifier is not None:
            SNR_proj[k] = SNR((projections_clean[k]*mollifier.mollify2d()).detach().cpu().numpy() , proj_est.detach().cpu().numpy())
        else:
            SNR_proj[k] = SNR((projections_clean[k]).detach().cpu().numpy() , proj_est.detach().cpu().numpy())


    if affine_tr_true is not None:
        Nangles = angles_t.shape[0]
        shift_true = np.zeros((Nangles,2))
        rot_true = np.zeros((Nangles))
        shift_est_ = np.zeros((Nangles,2))
        rot_est_ = np.zeros((Nangles))
        for k in range(Nangles):
            shift_true[k,0] = affine_tr_true[k].shiftX.detach().cpu().numpy()
            shift_true[k,1] = affine_tr_true[k].shiftY.detach().cpu().numpy()
            rot_true[k] = affine_tr_true[k].angle.detach().cpu().numpy()
            shift_est_[k] = shift_est[k].shifts_arr[0].detach().cpu().numpy()
            rot_est_[k] = rot_est[k].thetas.detach().cpu().numpy()

        err_shift = np.zeros(Nangles)
        err_rot = np.zeros(Nangles)
        err_local = np.zeros(Nangles)
        err_shift_init = np.zeros(Nangles)
        err_rot_init = np.zeros(Nangles)
        err_local_init = np.zeros(Nangles)
        if local_true is not None:
            for k in range(Nangles):
                grid_correction_true = local_true[k](grid.grid2d_t).detach().cpu().numpy()
                grid_correction_est = local_est[k](grid.grid2d_t).detach().cpu().numpy()
                tmp = np.abs(grid_correction_true-grid_correction_est)
                err_local[k] = (0.5*n1*tmp[:,0]+0.5*n2*tmp[:,1]).mean()
                tmp = np.abs(grid_correction_true)
                err_local_init[k] = (0.5*n1*tmp[:,0]+0.5*n2*tmp[:,1]).mean()


        err_shift = np.abs(shift_est_-shift_true)
        err_shift = 0.5*n1*err_shift[:,0] + 0.5*n2*err_shift[:,1]
        err_shift_init = np.abs(shift_true)
        err_shift_init = 0.5*n1*err_shift_init[:,0] + 0.5*n2*err_shift_init[:,1]

        err_rot = 180/np.pi*np.abs(rot_est_ - rot_true)
        err_rot_init = 180/np.pi*np.abs(rot_true)


    return fsc_5, fsc_143, fsc, x_fsc, SNR_est_vol, SNR_proj, err_shift, err_shift_init, err_rot, err_rot_init, err_local, err_local_init

def computeDeformationScore(grid2D,angles_t,shift_est,rot_est,local_est,affine_tr_true,local_true,n,scale=1):
   n1 = n
   n2 = n 
   if affine_tr_true is not None:
        Nangles = angles_t.shape[0]
        shift_true = np.zeros((Nangles,2))
        rot_true = np.zeros((Nangles))
        shift_est_ = np.zeros((Nangles,2))
        rot_est_ = np.zeros((Nangles))
        for k in range(Nangles):
            shift_true[k,0] = affine_tr_true[k].shiftX.detach().cpu().numpy()
            shift_true[k,1] = affine_tr_true[k].shiftY.detach().cpu().numpy()
            rot_true[k] = affine_tr_true[k].angle.detach().cpu().numpy()
            shift_est_[k] = shift_est[k].shifts_arr[0].detach().cpu().numpy()
            rot_est_[k] = rot_est[k].thetas.detach().cpu().numpy()
        # print('shift Estimation')
        # print(shift_est_[0:5])
        # print('Rotation Estimation')
        # print(rot_est_[0:5])

        err_shift = np.zeros(Nangles)
        err_rot = np.zeros(Nangles)
        err_local = np.zeros(Nangles)
        err_shift_init = np.zeros(Nangles)
        err_rot_init = np.zeros(Nangles)
        err_local_init = np.zeros(Nangles)

        #localEstimation = np.zeros((Nangles,2))
        if local_true is not None:
            for k in range(Nangles):
                grid_correction_true = local_true[k](grid2D).detach().cpu().numpy()
                grid_correction_est = scale*local_est[k](grid2D).detach().cpu().numpy()
                tmp = np.abs(grid_correction_true-grid_correction_est)
                err_local[k] = (0.5*n1*tmp[:,0]+0.5*n2*tmp[:,1]).mean()
                tmp = np.abs(grid_correction_true)
                err_local_init[k] = (0.5*n1*tmp[:,0]+0.5*n2*tmp[:,1]).mean()


        err_shift = np.abs(shift_est_-shift_true)
        err_shift = 0.5*n1*err_shift[:,0] + 0.5*n2*err_shift[:,1]
        err_shift_init = np.abs(shift_true)
        err_shift_init = 0.5*n1*err_shift_init[:,0] + 0.5*n2*err_shift_init[:,1]

        err_rot = 180/np.pi*np.abs(rot_est_ - rot_true)
        err_rot_init = 180/np.pi*np.abs(rot_true)

        # print('local Estimates')

        # print('shift error:'+ str(np.mean(err_shift)))
        # print('Rotation error:'+ str(np.mean(err_rot)))
        # print('local error:'+ str(np.mean(err_local)))

        return  err_shift, err_shift_init, err_rot, err_rot_init, err_local, err_local_init
