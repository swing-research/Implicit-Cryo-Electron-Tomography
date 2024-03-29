import torch
import numpy as np
from skimage.transform import resize

# import sys
# sys.path.insert(0, '..')
from ops.radon_3d_lib import ParallelBeamGeometry3DOpAngles_rectangular


"""
Here we define the functions used to sample the volume and apply the deformation.
We define forward model that may differ from the models used during training.
We make sure that everything is consistent between both approaches.
"""


## Define functions to compute forward operator 
# Given a viewing direction (degrees and torch) return the 3d rotation matrix
def viewing_direction_to_rotation(view_dir,torch_type,device):
    mat = torch.zeros((3,3)).to(device).type(torch_type)
    mat[0,0] = 1.
    mat[1,1] = torch.cos(view_dir/180*np.pi)
    mat[1,2] = -torch.sin(view_dir/180*np.pi)
    mat[2,1] = torch.sin(view_dir/180*np.pi)
    mat[2,2] = torch.cos(view_dir/180*np.pi)
    return mat


"""
For loop over the 'generate_ray' function

INPUT:
    - detectorLocations, (nBatch,nRays,2): 2D positions in the detector space.
    - anglesSet, (list of torch float): angle of the rays to generate, in degree.
    - z_max_value, (float): half-length of the ray in the adimensional axes. 
            Should be large enough to always pass through the entire sample when it's rotated.
            Can be obtained using the function 'get_sampling_geometry'.
    - ray_n, int: number of discrete point in the ray.
    - std_noise, float >=0: std of the noise perturbation to apply on the z direction of the rays.
            Each perturbation is different. std_noise=x means that the perturbation will shift by at most x pixel.
            std_noise=0 means there is no perturbation.

OUTPUT:
    - rays_rotated, (nBatch, nRays, ray_n, 3): 3D coordinates of the rays to sample.
"""


def generate_rays_batch(detectorLocations, anglesSet, z_max_value=1, ray_n=100, std_noise=0):
    nBatch, nRays, _ = detectorLocations.shape
    rays_rotated = torch.zeros((nBatch,nRays, ray_n,3)).to(detectorLocations.device)
    for i in range(nBatch):
        rays_rotated[i] = generate_ray(detectorLocations[i], anglesSet[i], z_max_value, ray_n, std_noise)
    return rays_rotated



"""
Given the dimension of the microscope and the height of the volume,
compute the sampling dimension to ensure consistent domain.
By using this function, we make sure to never assume any zero-padding on the x-y dimension.
The volume is assumed to be 0 outside its height.

INPUT:
    - size_z_vol, float in [0,1]: height of the volume (size along the z dimension) int he same dimension than
            sampling_domain_lx and sampling_domain_ly. 
    - angle_min, angle_max, float: minimum and maximum viewing angle in degree. 
    - sampling_domain_lx, sampling_domain_ly, float in [0,1]: adimensional half-length of the sampling domain. The 
            default values are 1. 
            
OUTPUT:
    - size_xy_vol, float: minum size of the volume along the x-y direction to correctly define the projections.
            The volume should be estimated on the domain (-size_xy_vol:size_xy_vol,-size_xy_vol:size_xy_vol,-size_z_vol:size_z_vol)
    - z_max_value, float: maximum value of the ray in order to always sample all the volume. After this value, the ray will necessarily 
            goes over the height of the volume (given by size_z_vol).

"""
def get_sampling_geometry(size_z_vol, angle_min=-60, angle_max=60, sampling_domain_lx=1, sampling_domain_ly=1):
    th = np.linspace(angle_min,angle_max,100) # the worse case could happen in the middle of the interval
    size_xy_vol = size_z_vol

    # If we want the volume, after any rotation, to always be in the sampling domain, 
    # then sampling_domain_lx should be < than x_sampling_min
    # x_sampling_lim is the value in the detector space untill which we can sample without the need
    # to define padding in the x-y dimension of the sample
    x_sampling_min1 = np.abs(size_xy_vol*np.cos(th*np.pi/180)- size_z_vol*np.sin(th*np.pi/180)).min()
    x_sampling_min2 = np.abs(size_xy_vol*np.cos(th*np.pi/180)+ size_z_vol*np.sin(th*np.pi/180)).min()
    x_sampling_lim = np.minimum(x_sampling_min1,x_sampling_min2)

    sampling_domain_lxy = np.maximum(sampling_domain_lx,sampling_domain_ly)
    while sampling_domain_lxy >= x_sampling_lim:
        size_xy_vol += 0.5
        x_sampling_min1 = np.abs(size_xy_vol*np.cos(th*np.pi/180)- size_z_vol*np.sin(th*np.pi/180)).min()
        x_sampling_min2 = np.abs(size_xy_vol*np.cos(th*np.pi/180)+ size_z_vol*np.sin(th*np.pi/180)).min()
        x_sampling_lim = np.minimum(x_sampling_min1,x_sampling_min2)

    z_max_value1 = np.abs(size_xy_vol*np.sin(th*np.pi/180) + size_z_vol*np.cos(th*np.pi/180)).max()
    z_max_value2 = np.abs(size_xy_vol*np.sin(th*np.pi/180) - size_z_vol*np.cos(th*np.pi/180)).max()
    z_max_value = np.maximum(z_max_value1,z_max_value2)
    return size_xy_vol, z_max_value


"""
Given a set of locations in the detector space, one viewing direction and the geometry 
of the sample and the microscope, returns the set of rays where to sample a 3D volume to simulate 
the projection.

INPUT: 
    - detectorLocations, (nBatch,2): 2D positions in the detector space.
    - angle, (torch float): angle of the rays to generate, in degree.
    - z_max_value, (float): half-length of the ray in the adimensional axes. 
            Should be large enough to always pass through the entire sample when it's rotated.
            Can be obtained using the function 'get_sampling_geometry'.
    - ray_n, int: number of discrete point in the ray.
    - std_noise, float >=0: std of the noise perturbation to apply on the z direction of the rays.
            Each perturbation is different. std_noise=x means that the perturbation will shift by at most x pixel.
            std_noise=0 means there is no perturbation.

OUTPUT:
    - rays_rotated, (nBatch, ray_n,3): 3D coordinates of the rays to sample.

"""
def generate_ray(detectorLocations, angle, z_max_value=1, ray_n=100, std_noise=0):
    nBatch, _ = detectorLocations.shape
    device = detectorLocations.device
    torch_type = detectorLocations.dtype
    # Define the ray geometry (length, discretization, position)
    zlin = torch.linspace(-1,1,ray_n+2)[1:-1].reshape(1,-1).to(device)
    dxz = torch.mean(zlin[0,1:]-zlin[0,:-1])
    noise = torch.rand(size=(nBatch,1),device=device)-0.5
    # perturbe slighlty this line in the z-direction to not sample always the same points
    zlin = z_max_value*(zlin.repeat(nBatch,1) + std_noise*noise*dxz)

    # Rotate all the lines
    raysSet = torch.concat([torch.unsqueeze(detectorLocations[:,0:1],dim=2).repeat(1,ray_n,1),torch.unsqueeze(detectorLocations[:,1:2],dim=2).repeat(1,ray_n,1),torch.unsqueeze(zlin,dim=2)],dim=2)
    # get the rotation matrix taking into account the transpose in the next lines
    mat_view = viewing_direction_to_rotation(-angle,torch_type,device) 
    rays_rotated = torch.transpose(torch.matmul(mat_view,torch.transpose(raysSet.reshape(-1,3),0,1)),0,1).reshape(-1,ray_n,3)

    return rays_rotated

"""
Apply the deformation in a differentiable manner to a set of 2D coordinates.

INPUT:
    - detectorLocations, (nBatch,nRays,2): 2D positions in the detector space.
    - rot_deformSet: inplane rotation deformation. Instance of utils_deformation.rotNet.
    - shift_deformSet: global shift deformations. Instance of utils_deformation.shiftNet.
    - local_deformSet: local deformations. Instance of utils_deformation.deformation_field.
    - fixedRotSet: fixed inplane rotation deformation. Instance of utils_deformation.rotNet.
    - scale, float: scale the local deformation amplitudes.

OUTPUT:
    - raysSet, (nBatch,nRays,2): locations of the sampling points after apply the deformations.
"""
def apply_deformations_to_locations(detectorLocations,rot_deformSet=None,shift_deformSet=None,
                        local_deformSet=None,fixedRotSet=None,scale=1):
    nBatch, nRays, _ = detectorLocations.shape
    device = detectorLocations.device
    raysSet = torch.zeros((nBatch,nRays,2)).to(device)

    for i in range(nBatch):
        pixelPositions_ = torch.unsqueeze(detectorLocations[i],dim=2)
        # Get deformations on the 2D detector grid
        if(fixedRotSet!=None):
            fixed_rot_deform = fixedRotSet[i](dim=2)
            pixelPositions_ = torch.matmul(fixed_rot_deform,pixelPositions_)
        if rot_deformSet != None:
            rot_deform = rot_deformSet[i](dim=2)
            pixelPositions_ = torch.matmul(rot_deform,pixelPositions_)
        # Apply shift deformation
        if shift_deformSet!=None:
            shift_deform = torch.unsqueeze(shift_deformSet[i](),dim=2)
            pixelPositions_ = pixelPositions_+shift_deform
        # Apply local deformation
        if local_deformSet!=None:
            local_deform = local_deformSet[i]
            pixelPositions_ = pixelPositions_ + scale*torch.unsqueeze(local_deform(torch.squeeze(pixelPositions_,2)),dim=2)
        raysSet[i] = pixelPositions_.squeeze(2)
    
    return raysSet


"""
Sample the projection at given 2D location in the detector space.

INPUT: 
    - projectionSet, (batch,n1,n2): torch tensor of the observed projections. Pixel of these projections
            are sampled at position given by 'sampleLocations'. Can be set to None.
    - sampleLocations, (nbatch,nRays,2): 2D position in the detector space where to sample the rays of the forwrd model. 

OUTPUT:
    - pixelValues, (nBatch,nRays)
"""
def sample_projections(projectionSet, sampleLocations, interp='bilinear'):
    nBatch, nRays, _ = sampleLocations.shape
    pixelValues = torch.zeros(nBatch,nRays).to(projectionSet.device)
    for i in range(nBatch):
        # Get the pixel in the observed titl-series
        pixelValues[i] = torch.nn.functional.grid_sample(projectionSet[i].T.unsqueeze(0).unsqueeze(0),
                                                        sampleLocations[i].unsqueeze(0).unsqueeze(0),mode=interp,align_corners=False).squeeze(0).squeeze(0)
    return pixelValues


"""
Sample the volume in coordinate given by raysSet.

INPUT:
    - impl_volume: implicit volume, it is a function that maps tensor of size (B,3) to a B-dimensional output. The input coordinates
            are rescale from [-1,1] to [0,1].
    - raysSet, (nBatc,nRays,ray_n,3): position of the 3D points to sample. Coordinates should be in [-1,1]

OUTPUT:
    - outputValues: values of the volume at the position given by raysSet.
"""
def sample_volume(impl_volume,raysSet):
    outputValues = impl_volume(raysSet.reshape(-1,3)/2 + 0.5).reshape(raysSet.shape[0],raysSet.shape[1],raysSet.shape[2])
    return outputValues