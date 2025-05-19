import torch
import torch.nn.functional as F
import numpy as np

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
                        local_deformSet=None,fixedRotSet=None,scale=1, cl=0):
    nBatch, nRays, _ = detectorLocations.shape
    device = detectorLocations.device
    raysSet = torch.zeros((nBatch,nRays,2)).to(device)

    for i in range(nBatch):
        pixelPositions_ = torch.unsqueeze(detectorLocations[i],dim=2)
        # Get deformations on the 2D detector grid
        if(fixedRotSet!=None):
            fixed_rot_deform = fixedRotSet(dim=2)
            pixelPositions_ = torch.matmul(fixed_rot_deform,pixelPositions_)
        if rot_deformSet != None:
            rot_deform = rot_deformSet[i](dim=2)
            pixelPositions_ = torch.matmul(rot_deform,pixelPositions_)
        # Apply shift deformation
        if shift_deformSet!=None:
            if cl !=0:
                shift_deform = torch.unsqueeze(torch.clip(shift_deformSet[i](), -cl,cl),dim=2)
            else:
                shift_deform = torch.unsqueeze(shift_deformSet[i](), dim=2)
            pixelPositions_ = pixelPositions_+ shift_deform
        # Apply local deformation
        if local_deformSet!=None:
            local_deform = local_deformSet[i]
            if cl !=0:
                pixelPositions_ = pixelPositions_ + torch.clip(scale*torch.unsqueeze(local_deform(torch.squeeze(pixelPositions_,2)),dim=2), -cl,cl)
            else:
                pixelPositions_ = pixelPositions_ + scale * torch.unsqueeze(local_deform(torch.squeeze(pixelPositions_, 2)), dim=2)
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
def sample_projections_from_location(projectionSet, sampleLocations, interp='bilinear'):
    nBatch, nRays, _ = sampleLocations.shape
    pixelValues = torch.zeros(nBatch,nRays).to(projectionSet.device)
    for i in range(nBatch):
        # Get the pixel in the observed titl-series
        pixelValues[i] = torch.nn.functional.grid_sample(projectionSet[i].T.unsqueeze(0).unsqueeze(0),
                                                        sampleLocations[i].unsqueeze(0).unsqueeze(0),mode=interp,align_corners=False).squeeze(0).squeeze(0)
    return pixelValues


def sample_volume(impl_volume, n1, n2, n3, size_z_vol, size_max_vol, avg_XYZ=1,
                  torch_type=torch.float, device=torch.device('cpu')):
    """
    input:
        n1, n2, n3: dimension of the volume

    return:
        volume: numpy array of shape (n1, n2, n3)
    """
    z_range = np.linspace(-1, 1, n3) * size_z_vol
    V_icetide = np.zeros((n1, n2, n3))
    ## grid for display
    x_lin1 = np.linspace(-1, 1, n1)
    x_lin2 = np.linspace(-1, 1, n2)
    XX, YY = np.meshgrid(x_lin1, x_lin2, indexing='ij')
    grid2d = np.concatenate([XX.reshape(-1, 1), YY.reshape(-1, 1)], 1)
    grid2d_t = torch.tensor(grid2d).type(torch_type)
    for zz, zval in enumerate(z_range):
        grid3d = np.concatenate([grid2d_t, zval * torch.ones((grid2d_t.shape[0], 1))], 1)
        grid3d_slice = torch.tensor(grid3d).type(torch_type).to(device)
        estSlice = impl_volume(grid3d_slice / size_max_vol / 2 + 0.5).detach().cpu().numpy().reshape(n1,n2)
        V_icetide[:, :, zz] = estSlice
    if avg_XYZ > 1:
        padded_array = np.pad(V_icetide, ((0, 0), (0, 0), (0, avg_XYZ - 1)), mode='constant')
        filt = np.zeros_like(padded_array)
        filt[:, :,
        filt.shape[2] // 2 - avg_XYZ // 2:filt.shape[2] // 2 + avg_XYZ // 2] = 1 / avg_XYZ
        V_icetide = np.fft.fftshift(np.fft.ifft((np.fft.fft(filt) * np.fft.fft(padded_array))).real, axes=-1)[:,:,:n3]
    return V_icetide

import os
import time
import imageio
def sample_projection_from_implicit_net(config, impl_volume, rot_est, shift_est, implicit_deformation_list, fixed_rot, angles_t, z_max_value, size_max_vol):
    t0 = time.time()
    projEstimate_tot = np.zeros((config.Nangles, config.n1, config.n2))
    x_lin1 = np.linspace(-1, 1, config.n1)
    x_lin2 = np.linspace(-1, 1, config.n2)
    XX, YY = np.meshgrid(x_lin1, x_lin2, indexing='ij')
    grid2d_ = np.concatenate([XX.reshape(-1, 1), YY.reshape(-1, 1)], 1)
    grid2d_ = torch.tensor(grid2d_).type(config.torch_type).to(config.device)
    for ll, angle in enumerate(angles_t):
        path_ = os.path.join(config.path_save + "projections", "implicit_net_ep" + str(ep).zfill((5)))
        if not os.path.exists(path_):
            os.makedirs(path_)
        for jj in range(config.n2):
            # Define the detector locations
            detectorLocations = grid2d_.reshape(config.n1, config.n2, 2)[:, jj].reshape(1, -1, 2)
            # Apply deformations in the 2D space
            detectorLocationsDeformed = apply_deformations_to_locations(detectorLocations, rot_est[ll:ll + 1],
                                                                        shift_est[ll:ll + 1],
                                                                        implicit_deformation_list[ll:ll + 1],
                                                                        fixed_rot, scale=config.deformationScale,
                                                                        cl=config.clip)
            # generate the rays in 3D
            rays_rotated = generate_rays_batch(detectorLocationsDeformed, angle[None], z_max_value, config.ray_length,
                                               std_noise=config.std_noise_z)
            # Scale the rays so that they are trully in [-1,1]
            rays_rotated_scaled = rays_rotated / size_max_vol
            # Sample the implicit volume by making the input in [0,1]
            outputValues = impl_volume((rays_rotated_scaled / 2 + 0.5).reshape(-1, 3)).reshape(1,
                                                                                               config.n1,
                                                                                               config.ray_length)
            support = (rays_rotated[:, :, :, 2].abs() < config.size_z_vol) * 1
            projEstimate = torch.sum(support * outputValues, 2) / config.ray_length
            projEstimate_tot[ll, :, jj] = projEstimate.reshape(-1).detach().cpu().numpy()
        if config.debug:
            tmp = projEstimate_tot[ll]
            tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
            tmp = np.floor(255 * tmp).astype(np.uint8)
            imageio.imwrite(os.path.join(path_, "est_" + str(ll).zfill(5) + ".png"), tmp)
    if config.debug:
        print("Elapsed time: {:2.0f} s".format(time.time() - t0))
    return projEstimate_tot


def cropper(image, coordinate , output_size, padding_mode="zeros"):
    # Coordinate shape: b X 2
    # image shape: b X c X h X w
    d_coordinate = coordinate
    b, c , h , w = image.shape
    crop_size = output_size/h
    x_m_x = crop_size
    x_p_x = d_coordinate[:,1]
    y_m_y = crop_size
    y_p_y = d_coordinate[:,0]
    theta = torch.zeros(b, 2,3).to(image.device)
    theta[:,0,0] = x_m_x
    theta[:,0,2] = x_p_x
    theta[:,1,1] = y_m_y
    theta[:,1,2] = y_p_y
    image = image.reshape(b, c , h , w)
    theta = theta.reshape(b , 2 , 3)
    f = F.affine_grid(theta, size=(b, c, output_size, output_size), align_corners=False)
    image_cropped = F.grid_sample(image, f, mode='bicubic', align_corners = True, padding_mode=padding_mode)
    return image_cropped

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

def aligned_projections(projections, rot_est, shift_est, implicit_deformation_est, fixed_rot, deformationScale=1, torch_type=torch.float, device=torch.device('cpu')):
    with torch.no_grad():
        Nangles, n1, n2 = projections.shape
        projections_undeformed = torch.zeros_like(projections)
        xx1 = torch.linspace(-1, 1, n1, dtype=torch_type, device=device)
        xx2 = torch.linspace(-1, 1, n2, dtype=torch_type, device=device)
        XX_t, YY_t = torch.meshgrid(xx1, xx2, indexing='ij')
        XX_t = torch.unsqueeze(XX_t, dim=2)
        YY_t = torch.unsqueeze(YY_t, dim=2)
        for i in range(Nangles):
            coordinates = torch.cat([XX_t, YY_t], 2).reshape(-1, 2)
            rot_deform = rot_est[i](dim=2).type(torch_type).to(device)
            fixed_rot_deform = fixed_rot(dim=2).type(torch_type).to(device)

            coordinates = coordinates - deformationScale * (implicit_deformation_est[i](coordinates).type(torch_type).to(device))
            coordinates = coordinates - (shift_est[i].shifts_arr.type(torch_type).to(device))
            coordinates = torch.transpose(torch.matmul(torch.transpose(rot_deform,1,0), torch.transpose(coordinates, 0, 1)), 0,
                                          1)  ## do rotation
            coordinates = torch.transpose(torch.matmul(torch.transpose(fixed_rot_deform,1,0), torch.transpose(coordinates, 0, 1)), 0,
                                          1)  ## do rotation
            x = projections[i].clone().view(1, 1, n1, n2)
            x = x.expand(n1 * n2, -1, -1, -1)
            out = cropper(x, coordinates, output_size=1).reshape(n1, n2)
            projections_undeformed[i] = out
    return projections_undeformed
