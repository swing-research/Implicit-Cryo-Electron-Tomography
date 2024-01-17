import torch
import numpy as np
from skimage.transform import resize

# import sys
# sys.path.insert(0, '..')
from ops.radon_3d_lib import ParallelBeamGeometry3DOpAngles_rectangular


def test():
    t= ParallelBeamGeometry3DOpAngles_rectangular
    return t

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
    # mat[1,1] = 1.
    # mat[0,0] = torch.cos(view_dir/180*np.pi)
    # mat[0,2] = torch.sin(view_dir/180*np.pi)
    # mat[2,0] = -torch.sin(view_dir/180*np.pi)
    # mat[2,2] = torch.cos(view_dir/180*np.pi)
    return mat

# Given an implicit model, the grid where to evaluate the volume, and the viewing direction, it 
# returns the rotated volume sampled on this grid. If a rotation of shift deformation is given,
# it applies these 2d deformations parallel to the z-axis. 
# INPUT:
#   - implt_repr: implicit function that maps coordinates to values
#   - grid: 3D grid of shape (N,3)
#   - view_dir: viewing direction in degrees
#   - rot_deform: 2D rotation matrix defining the in-plane rotation deformation
#   - shift_deform: torch array of 2 elements defining the shift deformation
#   - local_deform: function defining the displacement for any given position
def sample_implicit(implt_repr,grid,view_dir,rot_deform=None,shift_deform=None,local_deform=None,scale=1.0):
    grid_deform = torch.transpose(grid.clone(),0,1)
    # Apply in-plane rotation deformation
    if rot_deform != None:
        grid_deform = torch.matmul(rot_deform,grid_deform)
    # Apply shift deformation
    if shift_deform!=None:
        # idx = torch.LongTensor([1,0]).to(shift_deform.device)
        ss = torch.zeros_like(grid_deform)
        ss[:2,:] = shift_deform.view(shift_deform.shape[-1],-1)#.index_select(0,idx)
        grid_deform = grid_deform + ss
    # Apply local deformation
    if local_deform!=None:
        grid_deform[:2]  = grid_deform[:2] + scale*torch.transpose(local_deform(grid[:,:2]),0,1)
    # Apply 3D rotation 
    mat_view = viewing_direction_to_rotation(view_dir,grid.dtype,grid.device)
    grid_deform = torch.transpose(torch.matmul(mat_view,grid_deform),0,1).contiguous()
    # import ipdb; ipdb.set_trace()
    return implt_repr(grid_deform)



# compute the rays in a batch of projections
# Note: rays here should be in the form of batch of rays (Nbatch, nrays, n ,3)
def sample_implicit_batch(implt_repr,rays,view_dirSet,rot_deformSet=None,shift_deformSet=None,local_deformSet=None,scale=1.0):
    grid_deformSet = torch.zeros_like(rays).to(rays.device).reshape(rays.shape[0],-1,3)
    for i, view_dir in enumerate(view_dirSet):
        #grid = rays[i].reshape(-1,3)
        grid_deform = torch.transpose(rays[i].reshape(-1,3).clone(),0,1) 
        # Apply in-plane rotation deformation
        if rot_deformSet != None:
            rot_deform = rot_deformSet[i]()
            grid_deform = torch.matmul(rot_deform,grid_deform)
        # Apply shift deformation
        if shift_deformSet!=None:
            shift_deform = shift_deformSet[i]()
            ss = torch.zeros_like(grid_deform)
            ss[:2,:] = shift_deform.view(shift_deform.shape[-1],-1)#.index_select(0,idx)
            grid_deform = grid_deform + ss
        # Apply local deformation
        if local_deformSet!=None:
            local_deform = local_deformSet[i]
            grid_deform[:2]  = grid_deform[:2] + scale*torch.transpose(local_deform(rays[i].reshape(-1,3)[:,:2]),0,1)
        # Apply 3D rotation 
        mat_view = viewing_direction_to_rotation(view_dir,rays.dtype,rays.device)
        #ll = torch.transpose(torch.matmul(mat_view,grid_deform),0,1).contiguous()
        
        grid_deformSet[i] = torch.transpose(torch.matmul(mat_view,grid_deform),0,1).contiguous()
        #print(ll.shape)
        #return implt_repr(rays.reshape(-1,3)).reshape(1,rays.shape[1],rays.shape[2])
    grid_supp = ((grid_deformSet[:,:,0]<=1)&(grid_deformSet[:,:,0]>=-1)&(grid_deformSet[:,:,1]<=1)&(grid_deformSet[:,:,1]>=-1)&(grid_deformSet[:,:,2]<=1)&(grid_deformSet[:,:,2]>=-1))

    # import ipdb; ipdb.set_trace()
    return implt_repr(grid_deformSet.reshape(-1,3)).reshape(rays.shape[0],rays.shape[1],rays.shape[2]),grid_supp

## TODO: is volume support really accurate when z dimension is originally smaller?
def sample_implicit_batch_lowComp(implt_repr,rays,view_dirSet,rot_deformSet=None,shift_deformSet=None,
                                    local_deformSet=None,scale=1.0,grid_positive=False,zlimit=1.0,fixedRotSet=None,
                                    yBoundary=True,xBoundary=True,local_range=0):
    # Range is to shift the grid in to the positive space to avoid negative indices in the grid
    # grid_positive = False: keep the grid in [-1,1], otherwise shift to [0,1]
    # scale is the scale factor for the local deformation
    # zlimit: limit in the grid where volume is present (default is 1.0) should be greater than 0.0
    # yBoundary: if true, the y values are also limited to -1,1 (usually for simulated data), 
    #           for real data this need not bounded, its the cases when you crop the projections as well.
    #xBoundary: if true, the x values are also limited to -1,1 (usually for simulated data),
    #           for real data this need not bounded depening on the use case
    grid_deformSet = torch.zeros_like(rays).to(rays.device).reshape(rays.shape[0],-1,3)
    for i, view_dir in enumerate(view_dirSet):
        #grid = rays[i].reshape(-1,3)
        grid_deform = torch.transpose(rays[i].reshape(-1,3).clone(),0,1) 
        # Apply in-plane rotation deformation
        if(fixedRotSet!=None):
            fixed_rot_deform = fixedRotSet[i]()
            grid_deform = torch.matmul(fixed_rot_deform,grid_deform)
        if rot_deformSet != None:
            rot_deform = rot_deformSet[i]()
            grid_deform = torch.matmul(rot_deform,grid_deform)
        # Apply shift deformation
        if shift_deformSet!=None:
            shift_deform = shift_deformSet[i]()
            ss = torch.zeros_like(grid_deform)
            ss[:2,:] = shift_deform.view(shift_deform.shape[-1],-1)#.index_select(0,idx)
            grid_deform = grid_deform + ss
        # Apply local deformation
        if local_deformSet!=None:
            local_deform = local_deformSet[i]
            if local_range==0:
                grid_deform[:2]  = grid_deform[:2] + scale*torch.transpose(local_deform(rays[i][:,0,:2]).unsqueeze(1).repeat(1,rays.shape[2],1).reshape(-1,2),0,1)
            else:
                grid_deform[:2]  = grid_deform[:2] + scale*torch.transpose(local_deform(rays[i][:,0,:2]/2+0.5).unsqueeze(1).repeat(1,rays.shape[2],1).reshape(-1,2),0,1)
        # Apply 3D rotation 
        mat_view = viewing_direction_to_rotation(view_dir,rays.dtype,rays.device)
        #ll = torch.transpose(torch.matmul(mat_view,grid_deform),0,1).contiguous()
        
        grid_deformSet[i] = torch.transpose(torch.matmul(mat_view,grid_deform),0,1).contiguous()
        #print(ll.shape)
        #return implt_repr(rays.reshape(-1,3)).reshape(1,rays.shape[1],rays.shape[2])
    if(type(zlimit)==float):
        zlimit = abs(zlimit)
        if(yBoundary and xBoundary):
            grid_supp = ((grid_deformSet[:,:,0]<=1)&(grid_deformSet[:,:,0]>=-1)&(grid_deformSet[:,:,1]<=1)&(grid_deformSet[:,:,1]>=-1)&(grid_deformSet[:,:,2]<=zlimit)&(grid_deformSet[:,:,2]>=-zlimit))
        if(yBoundary and not xBoundary):
            grid_supp = ((grid_deformSet[:,:,1]<=1)&(grid_deformSet[:,:,1]>=-1)&(grid_deformSet[:,:,2]<=zlimit)&(grid_deformSet[:,:,2]>=-zlimit))
        if(not yBoundary and xBoundary):
            grid_supp = ((grid_deformSet[:,:,0]<=1)&(grid_deformSet[:,:,0]>=-1)&(grid_deformSet[:,:,2]<=zlimit)&(grid_deformSet[:,:,2]>=-zlimit))
        if(not yBoundary and not xBoundary):
            grid_supp = ((grid_deformSet[:,:,2]<=zlimit)&(grid_deformSet[:,:,2]>=-zlimit))
    # the limit is applied to each of the z-axis separately for each projection
    elif(type(zlimit)==np.ndarray):
        grid_supp = torch.zeros_like(grid_deformSet[:,:,0]).to(grid_deformSet.device)
        for i, view_dir in enumerate(view_dirSet):
            if(yBoundary and xBoundary):
                grid_supp[i] = ((grid_deformSet[i,:,0]<=1)&(grid_deformSet[i,:,0]>=-1)&(grid_deformSet[i,:,1]<=1)&(grid_deformSet[i,:,1]>=-1)&(grid_deformSet[i,:,2]<=zlimit[i])&(grid_deformSet[i,:,2]>=-zlimit[i]))
            if(yBoundary and not xBoundary):
                grid_supp[i] = ((grid_deformSet[i,:,1]<=1)&(grid_deformSet[i,:,1]>=-1)&(grid_deformSet[i,:,2]<=zlimit[i])&(grid_deformSet[i,:,2]>=-zlimit[i]))
            if(not yBoundary and xBoundary):
                grid_supp[i] = ((grid_deformSet[i,:,0]<=1)&(grid_deformSet[i,:,0]>=-1)&(grid_deformSet[i,:,2]<=zlimit[i])&(grid_deformSet[i,:,2]>=-zlimit[i]))
            if(not yBoundary and not xBoundary):
                grid_supp[i] = ((grid_deformSet[i,:,2]<=zlimit[i])&(grid_deformSet[i,:,2]>=-zlimit[i]))
    
    if grid_positive:
        return implt_repr(grid_deformSet.reshape(-1,3)/2 + 0.5).reshape(rays.shape[0],rays.shape[1],rays.shape[2]),grid_supp
    else:
        return implt_repr(grid_deformSet.reshape(-1,3)).reshape(rays.shape[0],rays.shape[1],rays.shape[2]),grid_supp


def sample_implicit_batch_Double_lowComp(volModel,implt_repr,rays,view_dirSet,rot_deformSet=None,shift_deformSet=None,local_deformSet=None,scale=1.0,range=0):
    grid_deformSet = torch.zeros_like(rays).to(rays.device).reshape(rays.shape[0],-1,3)
    for i, view_dir in enumerate(view_dirSet):
        #grid = rays[i].reshape(-1,3)
        grid_deform = torch.transpose(rays[i].reshape(-1,3).clone(),0,1) 
        # Apply in-plane rotation deformation
        if rot_deformSet != None:
            rot_deform = rot_deformSet[i]()
            grid_deform = torch.matmul(rot_deform,grid_deform)
        # Apply shift deformation
        if shift_deformSet!=None:
            shift_deform = shift_deformSet[i]()
            ss = torch.zeros_like(grid_deform)
            ss[:2,:] = shift_deform.view(shift_deform.shape[-1],-1)#.index_select(0,idx)
            grid_deform = grid_deform + ss
        # Apply local deformation
        if local_deformSet!=None:
            local_deform = local_deformSet[i]
            grid_deform[:2]  = grid_deform[:2] + scale*torch.transpose(local_deform(rays[i][:,0,:2]).unsqueeze(1).repeat(1,rays.shape[2],1).reshape(-1,2),0,1)
        # Apply 3D rotation 
        mat_view = viewing_direction_to_rotation(view_dir,rays.dtype,rays.device)
        #ll = torch.transpose(torch.matmul(mat_view,grid_deform),0,1).contiguous()
        
        grid_deformSet[i] = torch.transpose(torch.matmul(mat_view,grid_deform),0,1).contiguous()
        #print(ll.shape)
        #return implt_repr(rays.reshape(-1,3)).reshape(1,rays.shape[1],rays.shape[2])
    grid_supp = ((grid_deformSet[:,:,0]<=1)&(grid_deformSet[:,:,0]>=-1)&(grid_deformSet[:,:,1]<=1)&(grid_deformSet[:,:,1]>=-1)&(grid_deformSet[:,:,2]<=1)&(grid_deformSet[:,:,2]>=-1))

    # import ipdb; ipdb.set_trace()

    if(range==0):
        return implt_repr(grid_deformSet.reshape(-1,3)).reshape(rays.shape[0],rays.shape[1],rays.shape[2]), volModel(grid_deformSet.reshape(-1,3)).reshape(rays.shape[0],rays.shape[1],rays.shape[2]), grid_supp
    if(range==1):
        return implt_repr(grid_deformSet.reshape(-1,3)/2 + 0.5).reshape(rays.shape[0],rays.shape[1],rays.shape[2]), volModel(grid_deformSet.reshape(-1,3)).reshape(rays.shape[0],rays.shape[1],rays.shape[2]), grid_supp



# compute the rays in a batch of projections
# Note: rays here should be in the form of batch of rays (Nbatch, nrays, n ,3)
def sample_implicit_batch_v2(implt_repr,frac,rays,view_dirSet,rot_deformSet=None,shift_deformSet=None,local_deformSet=None,scale=1.0):
    grid_deformSet = torch.zeros_like(rays).to(rays.device).reshape(rays.shape[0],-1,3)
    for i, view_dir in enumerate(view_dirSet):


        # # Get support
        mat_view = viewing_direction_to_rotation(view_dir,rays.dtype,rays.device)
        # grid_supp = torch.matmul(rays[i].clone(), torch.transpose(mat_view.clone(),0,1)).contiguous()
        # grid_supp = ((grid_supp[:,:,0]<=1)&(grid_supp[:,:,0]>=-1)&(grid_supp[:,:,1]<=1)&(grid_supp[:,:,1]>=-1)&(grid_supp[:,:,2]<=1)&(grid_supp[:,:,2]>=-1))


        #grid = rays[i].reshape(-1,3)
        grid_deform = torch.transpose(rays[i].reshape(-1,3).clone(),0,1) 
        # Apply in-plane rotation deformation
        if rot_deformSet != None:
            rot_deform = rot_deformSet[i]()
            grid_deform = torch.matmul(rot_deform,grid_deform)
        # Apply shift deformation
        if shift_deformSet!=None:
            shift_deform = shift_deformSet[i]()
            ss = torch.zeros_like(grid_deform)
            ss[:2,:] = shift_deform.view(shift_deform.shape[-1],-1)#.index_select(0,idx)
            grid_deform = grid_deform + ss
        # Apply local deformation
        if local_deformSet!=None:
            local_deform = local_deformSet[i]
            grid_deform[:2]  = grid_deform[:2] + scale*torch.transpose(local_deform(rays[i].reshape(-1,3)[:,:2]),0,1)
        # Apply 3D rotation 
        # mat_view = viewing_direction_to_rotation(view_dir,rays.dtype,rays.device)
        #ll = torch.transpose(torch.matmul(mat_view,grid_deform),0,1).contiguous()
        
        tmp = torch.transpose(torch.matmul(mat_view,grid_deform),0,1).contiguous()
        grid_deformSet[i] = tmp
        #print(ll.shape)
        #return implt_repr(rays.reshape(-1,3)).reshape(1,rays.shape[1],rays.shape[2])

    grid_supp = ((grid_deformSet[:,:,0]<=1)&(grid_deformSet[:,:,0]>=-1)&(grid_deformSet[:,:,1]<=1)&(grid_deformSet[:,:,1]>=-1)&(grid_deformSet[:,:,2]<=1)&(grid_deformSet[:,:,2]>=-1))


    # import ipdb; ipdb.set_trace()
    return implt_repr(grid_deformSet.reshape(-1,3),frac).reshape(rays.shape[0],rays.shape[1],rays.shape[2]), grid_supp




def sample_implicit_batch_list(implt_repr,frac,rays,view_dirSet,rot_deformSet=None,shift_deformSet=None,local_deformSet=None,scale=1.0):
    grid_deformSet = torch.zeros_like(rays).to(rays.device).reshape(rays.shape[0],-1,3)
    for i, view_dir in enumerate(view_dirSet):


        # # Get support
        mat_view = viewing_direction_to_rotation(view_dir,rays.dtype,rays.device)
        # grid_supp = torch.matmul(rays[i].clone(), torch.transpose(mat_view.clone(),0,1)).contiguous()
        # grid_supp = ((grid_supp[:,:,0]<=1)&(grid_supp[:,:,0]>=-1)&(grid_supp[:,:,1]<=1)&(grid_supp[:,:,1]>=-1)&(grid_supp[:,:,2]<=1)&(grid_supp[:,:,2]>=-1))


        #grid = rays[i].reshape(-1,3)
        grid_deform = torch.transpose(rays[i].reshape(-1,3).clone(),0,1) 
        # Apply in-plane rotation deformation
        if rot_deformSet != None:
            rot_deform = rot_deformSet[i]()
            grid_deform = torch.matmul(rot_deform,grid_deform)
        # Apply shift deformation
        if shift_deformSet!=None:
            shift_deform = shift_deformSet[i]()
            ss = torch.zeros_like(grid_deform)
            ss[:2,:] = shift_deform.view(shift_deform.shape[-1],-1)#.index_select(0,idx)
            grid_deform = grid_deform + ss
        # Apply local deformation
        if local_deformSet!=None:
            local_deform = local_deformSet[i]
            grid_deform[:2]  = grid_deform[:2] + scale*torch.transpose(local_deform(rays[i].reshape(-1,3)[:,:2]),0,1)
        # Apply 3D rotation 
        # mat_view = viewing_direction_to_rotation(view_dir,rays.dtype,rays.device)
        #ll = torch.transpose(torch.matmul(mat_view,grid_deform),0,1).contiguous()
        
        tmp = torch.transpose(torch.matmul(mat_view,grid_deform),0,1).contiguous()
        grid_deformSet[i] = tmp
        #print(ll.shape)
        #return implt_repr(rays.reshape(-1,3)).reshape(1,rays.shape[1],rays.shape[2])

    grid_supp = ((grid_deformSet[:,:,0]<=1)&(grid_deformSet[:,:,0]>=-1)&(grid_deformSet[:,:,1]<=1)&(grid_deformSet[:,:,1]>=-1)&(grid_deformSet[:,:,2]<=1)&(grid_deformSet[:,:,2]>=-1))


    # import ipdb; ipdb.set_trace()
    return implt_repr(grid_deformSet.reshape(-1,3),frac), grid_supp




def repeatSample(valid_locs,n ):
    # Function repeates the indecdes if the number of rays is larger than the number of valid locations
    Nsize = valid_locs.shape[0]

    if n < Nsize:
        return torch.randperm(valid_locs.shape[0])[:n]
    else:
        count = 0
        while(count < n):
            sampleSize = min(Nsize,n-count)
            if count == 0:
                sample = torch.randperm(valid_locs.shape[0])[:sampleSize]
            else:
                sample = torch.cat((sample,torch.randperm(valid_locs.shape[0])[:sampleSize]))
            count += sampleSize
        return sample

def sample_rays(projection,grid_ball,ball,n_rays,valid_locs):
    # Projection is the image from which we sample the intensities
    # grib_ball is the grid of points on the ball
    # n_rays is the number of rays to sample
    # valid_locs is the indeces of the valid locations in the image (Essentially locations of the pixels inside the disk of a projection)
    # ball is the boolean mask of the ball 


    # sample pixel location indeces
    Nbatch, n, m = projection.shape
    n_z = grid_ball.shape[2]

    chooseIndeces = repeatSample(valid_locs,n_rays*Nbatch)
    locs = valid_locs[chooseIndeces]
    x = locs[:,0]
    y = locs[:,1]
    # sample random angles
    rays = grid_ball[x,y]
    raysBool = ball[x,y] 

    x = x.reshape(Nbatch,n_rays)
    y = y.reshape(Nbatch,n_rays)

    # sample pixel values

    pixelValues = torch.zeros(Nbatch,n_rays).to(projection.device)

    rays = rays.reshape(Nbatch,n_rays,n_z,3)
    raysBool = raysBool.reshape(Nbatch,n_rays,n_z)

    for i, (xi,yi) in enumerate(zip(x,y)):
        pixelValues[i] = projection[i,xi,yi]


    return rays, raysBool, pixelValues, locs



# This is same as the above function which is used to generate the rotation matrix for the 3D rotation with added parameter of choosing the axis of rotation
def angles_to_rotation_matrix(angle,index=0):
    # angles: [alpha,beta,gamma]
    if(index==0):
        alpha = angle # torch.zeros_like(angle)
        beta = torch.zeros_like(angle)
        gamma = torch.zeros_like(angle)
    elif(index==1):
        alpha = torch.zeros_like(angle)
        beta = angle
        gamma = torch.zeros_like(angle)
    elif(index==2):
        alpha = torch.zeros_like(angle)
        beta = torch.zeros_like(angle)
        gamma = angle
    else:
        print("index should be 0,1 or 2")
        return 0
    if(len(angle.shape)==0):
        R_alpha = torch.tensor([[1,0,0],[0,torch.cos(alpha),-torch.sin(alpha)],[0,torch.sin(alpha),torch.cos(alpha)]]).to(angle.device)
        R_beta = torch.tensor([[torch.cos(beta),0,torch.sin(beta)],[0,1,0],[-torch.sin(beta),0,torch.cos(beta)]]).to(angle.device)
        R_gamma = torch.tensor([[torch.cos(gamma),-torch.sin(gamma),0],[torch.sin(gamma),torch.cos(gamma),0],[0,0,1]]).to(angle.device)
        R = torch.matmul(R_alpha,torch.matmul(R_beta,R_gamma))
        return R
    else:
        R_alpha = torch.zeros((len(angle),3,3)).to(angle.device)
        R_beta = torch.zeros((len(angle),3,3)).to(angle.device)
        R_gamma = torch.zeros((len(angle),3,3)).to(angle.device)
        R = torch.zeros((len(angle),3,3)).to(angle.device)
        for i in range(len(angle)):
            R_alpha[i] = torch.tensor([[1,0,0],[0,torch.cos(alpha[i]),-torch.sin(alpha[i])],[0,torch.sin(alpha[i]),torch.cos(alpha[i])]])
            R_beta[i] = torch.tensor([[torch.cos(beta[i]),0,torch.sin(beta[i])],[0,1,0],[-torch.sin(beta[i]),0,torch.cos(beta[i])]])
            R_gamma[i] = torch.tensor([[torch.cos(gamma[i]),-torch.sin(gamma[i]),0],[torch.sin(gamma[i]),torch.cos(gamma[i]),0],[0,0,1]])
            R[i] = torch.matmul(R_alpha[i],torch.matmul(R_beta[i],R_gamma[i]))
            return R
        
# Generate rays in batches with additon of the z values being randomly sampled in equal intervals     
def generate_rays_batch(projectionSet,angleSet,nRays,rayLength,validLocations,randomZ = False, zmax=1.5):
    # Projection set of the form [Nbatch,n,n
    # Angle set of the form [Nbatch]
    # nRays is the number of rays per projection
    # rayLength is the length of the rays

    nBatch = projectionSet.shape[0]
    n = projectionSet.shape[1]


    pixelValues = torch.zeros(nBatch,nRays).to(projectionSet.device)
    raysSet = torch.zeros(nBatch,nRays,rayLength,3).to(projectionSet.device)
    raysRot = torch.zeros_like(raysSet).to(projectionSet.device)
    isOutsideSet = torch.zeros(nBatch,nRays,rayLength,dtype=torch.bool).to(projectionSet.device)

    for i in range(nBatch):
        indeces = torch.randperm(validLocations.shape[0])[:nRays]
        choosenLocations = validLocations[indeces]
        pixelValues[i] = projectionSet[i,choosenLocations[:,0],choosenLocations[:,1]]


        # Convert to [-1,1]

        choosenLocations = (choosenLocations - n/2)/(n/2)

            # Generate the rays
        # locations are sampleed from the valid locations


        rays,rayRot, isOutside = generate_ray(choosenLocations,angleSet[i],rayLength,randomZ = randomZ,zmax=zmax)
        raysSet[i] = rays
        raysRot[i] = rayRot
        isOutsideSet[i] = isOutside

    return raysSet,raysRot, isOutsideSet, pixelValues

              
    
# idx_loader is needed if choosenLocations_all is not None
def generate_rays_batch_bilinear(projectionSet,angleSet,nRays,rayLength,randomZ = 0,type=1,zmax=1.5,
                                 choosenLocations_all=None,density_sampling=None,idx_loader=None):
    # Projection set of the form [Nbatch,n,n]
    # Angle set of the form [Nbatch]
    # nRays is the number of rays per projection
    # rayLength is the length of the rays
    # randomZ says whether to sample the z values randomly or not (0,1,2 depending on the type of sampling)
    # type = 1: bilinear interpolation
    # type = 2:  bicubic interpolation
    # zmax is the maximum absolute value of the z to be sampled
    # TODO: valid location is not used here

    nBatch = projectionSet.shape[0]
    n = projectionSet.shape[1]

    pixelValues = torch.zeros(nBatch,nRays).to(projectionSet.device)
    raysSet = torch.zeros(nBatch,nRays,rayLength,3).to(projectionSet.device)
    raysRot = torch.zeros_like(raysSet).to(projectionSet.device)
    isOutsideSet = torch.zeros(nBatch,nRays,rayLength,dtype=torch.bool).to(projectionSet.device)

    for i in range(nBatch):

        if density_sampling is not None:
            choosenLocations = np.random.choice(np.arange(np.prod(density_sampling[i].shape)),size=(nRays),replace=False,p=density_sampling[i].reshape(-1))

            choosenLocations_x, choosenLocations_y = np.unravel_index(choosenLocations, density_sampling[i].shape, order='C')
            choosenLocations = np.concatenate([choosenLocations_x[:,None],choosenLocations_y[:,None]],1)*1.

            choosenLocations += np.random.random(size=choosenLocations.shape)-0.5
            choosenLocations[:,0] = choosenLocations[:,0]/density_sampling[i].shape[0]*2-1
            choosenLocations[:,1] = choosenLocations[:,1]/density_sampling[i].shape[1]*2-1
            choosenLocations = np.maximum(choosenLocations,-1)
            choosenLocations = np.minimum(choosenLocations,1)
            choosenLocations = torch.tensor(choosenLocations).to(projectionSet.device).type_as(projectionSet)

        else:
            choosenLocations = torch.rand(nRays,2).to(projectionSet.device)*2-1

        if choosenLocations_all is not None:
            choosenLocations_all[idx_loader[i].item()].append(choosenLocations.detach().cpu().numpy())

        if(type==1):
            pixelValues[i] = torch.nn.functional.grid_sample(projectionSet[i].T.unsqueeze(0).unsqueeze(0),
                                                         choosenLocations.unsqueeze(0).unsqueeze(0),mode='bilinear',align_corners=False).squeeze(0).squeeze(0)
        if(type==2):
            pixelValues[i] = torch.nn.functional.grid_sample(projectionSet[i].T.unsqueeze(0).unsqueeze(0),
                                                         choosenLocations.unsqueeze(0).unsqueeze(0),mode='bicubic',align_corners=False).squeeze(0).squeeze(0)

        rays,rayRot, isOutside = generate_ray(choosenLocations,angleSet[i],rayLength,randomZ = randomZ,zmax=zmax)
        raysSet[i] = rays
        raysRot[i] = rayRot
        isOutsideSet[i] = isOutside

    return raysSet,raysRot, isOutsideSet, pixelValues

# Does bilinear interpolation with support for which locations to samples. Does this using rejection sampling
# Used when the data has obstacles in the projections which we want to avoid
def generate_rays_batch_bilinear_support(projectionSet,support_set,angleSet,nRays,rayLength,randomZ = False,type=1,zmax=1.5):
    # Projection set of the form [Nbatch,n,n
    # Angle set of the form [Nbatch]
    # nRays is the number of rays per projection
    # rayLength is the length of the rays
    # type = 1: bilinear interpolation
    # type = 2:  bicubic interpolation
    # TODO: valid location is not used here

    nBatch = projectionSet.shape[0]
    n = projectionSet.shape[1]


    pixelValues = torch.zeros(nBatch,nRays).to(projectionSet.device)
    raysSet = torch.zeros(nBatch,nRays,rayLength,3).to(projectionSet.device)
    raysRot = torch.zeros_like(raysSet).to(projectionSet.device)
    isOutsideSet = torch.zeros(nBatch,nRays,rayLength,dtype=torch.bool).to(projectionSet.device)

    for i in range(nBatch):


        choosenLocations = generateSamplesSupport(nRays,support_set[i])

        if(type==1):
            pixelValues[i] = torch.nn.functional.grid_sample(projectionSet[i].T.unsqueeze(0).unsqueeze(0),
                                                         choosenLocations.unsqueeze(0).unsqueeze(0),mode='bilinear',align_corners=False).squeeze(0).squeeze(0)
        if(type==2):
            pixelValues[i] = torch.nn.functional.grid_sample(projectionSet[i].T.unsqueeze(0).unsqueeze(0),
                                                         choosenLocations.unsqueeze(0).unsqueeze(0),mode='bicubic',align_corners=False).squeeze(0).squeeze(0)

        rays,rayRot, isOutside = generate_ray(choosenLocations,angleSet[i],rayLength,randomZ = randomZ,zmax=zmax)
        raysSet[i] = rays
        raysRot[i] = rayRot
        isOutsideSet[i] = isOutside

    return raysSet,raysRot, isOutsideSet, pixelValues



def generateSamplesSupport(npoints,support):
    # Generate npoint samples from the support of the function
    # support is a nxn matrix of ones and zeros

    n1 = support.shape[0]
    n2 = support.shape[1]

    choosenPointSet = torch.zeros((npoints,2)).to(support.device)
    currentPointIndex = 0

    while(currentPointIndex<npoints):
        points = torch.rand(npoints,2).to(support.device) * 2 - 1

        # sample points from the support,
        # if the support is 1 at the point, then add it to the list

        supportVals = torch.nn.functional.grid_sample(support.T.unsqueeze(0).unsqueeze(0),points.unsqueeze(0).unsqueeze(0),mode='bilinear',align_corners=False).squeeze(0).squeeze(0).squeeze(0)
        #print(supportVals.shape)
        choosenPoints = points[supportVals==1]
        if(choosenPoints.shape[0]+ currentPointIndex>npoints):
            choosenPoints = choosenPoints[0:npoints-currentPointIndex,:]
        choosenPointSet[currentPointIndex:min(npoints,currentPointIndex+choosenPoints.shape[0])] = choosenPoints
        currentPointIndex += choosenPoints.shape[0]
    
    return choosenPointSet
    



def generate_rays_batch_patch(projectionSet,angleSet,nPatchs,patchSize,rayLength,validLocations,randomZ = False):
    # Projection set of the form [Nbatch,n,n
    # Angle set of the form [Nbatch]
    # nRays is the number of rays per projection
    # rayLength is the length of the rays

    nBatch = projectionSet.shape[0]
    n = projectionSet.shape[1]


    pixelValues = torch.zeros(nBatch,nPatchs,patchSize,patchSize).to(projectionSet.device)
    raysSet = torch.zeros(nBatch,nPatchs,patchSize,patchSize,rayLength,3).to(projectionSet.device)
    raysRot = torch.zeros_like(raysSet).to(projectionSet.device)
    isOutsideSet = torch.zeros(nBatch,nPatchs,patchSize,patchSize,rayLength,dtype=torch.bool).to(projectionSet.device)

    for i in range(nBatch):

        patchCenter = torch.randperm(validLocations.shape[0])[:nPatchs]

        indeces = torch.randperm(validLocations.shape[0])[:nRays]
        choosenLocations = validLocations[indeces]
        pixelValues[i] = projectionSet[i,choosenLocations[:,0],choosenLocations[:,1]]


        # Convert to [-1,1]

        choosenLocations = (choosenLocations - n/2)/(n/2)

            # Generate the rays
        # locations are sampleed from the valid locations


        rays,rayRot, isOutside = generate_ray(choosenLocations,angleSet[i],rayLength,randomZ = randomZ)
        raysSet[i] = rays
        raysRot[i] = rayRot
        isOutsideSet[i] = isOutside

    return raysSet,raysRot, isOutsideSet, pixelValues

"""
randomZ: 0 means not random, 1 random, 2 random but fixed space
"""
def generate_ray(locations,angle,Nsamples,randomZ = 0,zmax=1.5):
    #Locations of the form [Nbatch,2]
    Nbatch = locations.shape[0]
    # ray starts at z=1 and goes to z=0 
    # Rays between -1 and 1 are generated
    x_lim_l = -1
    x_lim_u = 1
    y_lim_l = -1
    y_lim_u = 1
    z_lim_l = -1
    z_lim_u = 1


    rays = torch.zeros(Nbatch, Nsamples,3).to(locations.device)
    #Samples z-values from 1 to 0 uniformly  with randomness    

    rMat = angles_to_rotation_matrix(angle*torch.pi/180)
    z = torch.linspace(-zmax,zmax,Nsamples+2)[1:-1].to(locations.device)
    dz = torch.mean(z[1:]-z[:-1])
    if randomZ==1:
        # Obtain distance between the consecutive z values
        noise= torch.rand_like(z)-0.5
        z = z + noise*dz
    if randomZ==2:
        noise= np.random.random()-0.5
        z = z + noise*dz
    
    #Generate rays
    rays[:, :, 0] = locations[:, 0].unsqueeze(1).repeat(1,Nsamples)
    rays[:, :, 1] = locations[:, 1].unsqueeze(1).repeat(1,Nsamples)
    rays[:, :, 2] = z.unsqueeze(0).repeat(Nbatch,1)  # + torch.rand(Nbatch,Nsamples).to(locations.device)*(1/Nsamples)


    # Rotate the rays

    raysRot = torch.transpose(torch.matmul(rMat,torch.transpose(rays,1,2)),1,2)

    # Find the rays outside the cube
    isOutside = torch.zeros_like(rays[:,:,0],dtype = torch.bool).to(locations.device)
    isOutside = isOutside | (raysRot[:,:,0] < x_lim_l) | (raysRot[:,:,0] > x_lim_u) | (raysRot[:,:,1] < y_lim_l) | (
        raysRot[:,:,1] > y_lim_u) | (raysRot[:,:,2] < z_lim_l) | (raysRot[:,:,2] > z_lim_u)
    
    return rays,raysRot, isOutside


"""
If the points to samples have different limits than xlim, ylim, then you need to pass them as input

Example of how to use during training:
    # to ensure that we use rays that continue outside the domain
    xlim = 1.5
    ylim = 1.5
    pixelPositions = torch.rand(proj.shape[0],nRays,2).to(device)*2-1
    pixelPositions[:,:,0] = pixelPositions[:,:,0]
    pixelPositions[:,:,1] = pixelPositions[:,:,1]
    outputValues_vol, outputValues_net, support, pixelValues, raysSet, pixelPositions = sample_rays_custom(init_volume, impl_volume, proj, angle,
                            nRays, ray_length, pixelPositions=pixelPositions,
                            rot_deformSet=rot_deformSet,shift_deformSet=shift_deformSet,
                    local_deformSet=local_deformSet,z_lim=z_lim, scale=deformationScale, fixedRotSet=fixedRotSet, mode='bilinear',
                    xlim=xlim,ylim=ylim)



Example of how to use for display:
    pixelPositions = torch.unsqueeze(grid2d_t.to(device),dim=0).repeat(1,1,1)
    outputValues_vol, outputValues_net, support, pixelValues, raysSet, pixelPositions = sample_rays_custom(init_volume, None, None,
                            list_angles, nRays, ray_length_test, pixelPositions,
                            rot_deformSet=rot_deformSet,shift_deformSet=shift_deformSet,
                    local_deformSet=local_deformSet,z_lim=z_lim, scale=deformationScale, fixedRotSet=fixedRotSet, mode='bilinear',xlim=xlim,ylim=ylim)

    outputValues = (outputValues_vol+outputValues_net).type(torch_type)
    sum_normalization = torch.sum(support,2)
    sum_normalization[sum_normalization==0] = 1
    projection_est = torch.sum(support*outputValues,2)/sum_normalization
    projection_est = projection_est.detach().cpu().numpy().reshape(1,n1_test,n1_test)    

"""
"""
Here we make the volume rotates around a fixed parallel comb. If the ratio between x/y axis and z is too important, 
then for large angle some rays (on the x/y corners) will be outside of the volume. 
Another defnition could be to define the rays of a comb (angle 0), and then rotate each ray at position (x,y) with respect to (x,y)

"""
def sample_rays_custom(init_volume, impl_volume, proj, angle, nRays=1, ray_length=10, pixelPositions=None,
                        rot_deformSet=None,shift_deformSet=None,
                        local_deformSet=None,scale=1.0,z_lim=1.0,fixedRotSet=None,mode='bilinear',
                        xlim=1,ylim=1):
    if proj is not None:
        device = proj.device
        nBatch = proj.shape[0]
    else:
        device = angle[0].device
        nBatch = len(angle)
    if pixelPositions is None:
        pixelPositions = torch.rand(nBatch,nRays,2).to(device)*2-1
        pixelPositions[:,:,0] = pixelPositions[:,:,0]*xlim
        pixelPositions[:,:,1] = pixelPositions[:,:,1]*ylim
    else:
        nRays = pixelPositions.shape[1]
    pixelValues = torch.zeros(nBatch,nRays).to(device)
    raysSet = torch.zeros((nBatch,nRays,ray_length,3)).to(device)
    support = torch.zeros((nBatch,nRays,ray_length)).to(device)
    for i in range(nBatch):
        # get true values
        if proj is not None:
            pixelValues[i] = torch.nn.functional.grid_sample(proj[i].T.unsqueeze(0).unsqueeze(0),
                                                        pixelPositions[i].unsqueeze(0).unsqueeze(0),mode=mode,align_corners=False).squeeze(0).squeeze(0)
        pixelPositions_def = torch.unsqueeze(pixelPositions[i],dim=2)
        # Get deformations on the 2D detector grid
        if(fixedRotSet!=None):
            fixed_rot_deform = fixedRotSet[i](dim=2)
            pixelPositions_def = torch.matmul(fixed_rot_deform,pixelPositions_def)
        if rot_deformSet != None:
            rot_deform = rot_deformSet[i](dim=2)
            pixelPositions_def = torch.matmul(rot_deform,pixelPositions_def)
        # Apply shift deformation
        if shift_deformSet!=None:
            shift_deform = torch.unsqueeze(shift_deformSet[i](),dim=2)
            pixelPositions_def = pixelPositions_def+shift_deform
        # Apply local deformation
        if local_deformSet!=None:
            local_deform = local_deformSet[i]
            pixelPositions_def = pixelPositions_def + scale*torch.unsqueeze(local_deform(torch.squeeze(pixelPositions_def,2)),dim=2)
    
        xzlin = torch.linspace(-1,1,ray_length+2)[1:-1].to(device).reshape(1,-1).repeat(nRays,1)
        dxz = torch.mean(xzlin[0,1:]-xzlin[0,:-1])
        noise = torch.rand(size=(nRays,1),device=device)-0.5
        # perturbe slighlty this line in the z-direction to not sample always the same points
        xzlin = xzlin + noise*dxz
        if angle[i]>=0:
            x_max = (pixelPositions_def[:,1]+ z_lim*torch.sin((angle/180*np.pi)))/torch.cos((angle/180*np.pi))
            x_min = (pixelPositions_def[:,1]- z_lim*torch.sin((angle/180*np.pi)))/torch.cos((angle/180*np.pi))
        if angle[i]<0:
            x_min = (pixelPositions_def[:,1]+ z_lim*torch.sin((angle/180*np.pi)))/torch.cos((angle/180*np.pi))
            x_max = (pixelPositions_def[:,1]- z_lim*torch.sin((angle/180*np.pi)))/torch.cos((angle/180*np.pi))
        z_min = -z_lim
        z_max = z_lim

        # check when they are out of the domain
        x_min_ = torch.maximum(x_min,-ylim*torch.ones_like(pixelPositions_def[:,1]))
        x_max_ = torch.minimum(x_max, ylim*torch.ones_like(pixelPositions_def[:,1]))

        # update 
        x_step = 2*torch.sin(angle[i]/180*torch.pi)
        z_step = 2*torch.cos(angle[i]/180*torch.pi)
        # modifying x_min will impact z_min
        z_incr_min = (x_min_-x_min)*z_step/(x_step+1e-8)
        # x_incr_min = (z_min_-z_min)*x_step/z_step 

        # modifying x_max will impact z_max
        z_incr_max = (x_max_-x_max)*z_step/(x_step+1e-8)
        # x_incr_max = (z_max_-z_max)*x_step/z_step 

        x_min = x_min_
        z_min = z_min + z_incr_min
        x_max = x_max_
        z_max = z_max + z_incr_max

        ## Compute 
        rays_x = pixelPositions_def[:,0].repeat(1,ray_length)
        rays_y = ((xzlin+1)/2*(x_max-x_min)+x_min)
        rays_z = ((xzlin+1)/2*(z_max-z_min)+z_min)

        raysSet[i] = torch.concat([torch.unsqueeze(rays_x,dim=2),torch.unsqueeze(rays_y,dim=2),torch.unsqueeze(rays_z,dim=2)],dim=2)
        support[i] = (raysSet[i,:,:,0].abs()<=xlim) & (raysSet[i,:,:,1].abs()<=ylim) & (raysSet[i,:,:,2].abs()<=z_lim)

    if init_volume is not None:
        outputValues_vol = init_volume(raysSet.reshape(-1,3)).reshape(raysSet.shape[0],raysSet.shape[1],raysSet.shape[2])
    else:
        outputValues_vol = torch.zeros((raysSet.shape[0],raysSet.shape[1],raysSet.shape[2])).to(device)
    outputValues_net = impl_volume(raysSet.reshape(-1,3)/2 + 0.5).reshape(raysSet.shape[0],raysSet.shape[1],raysSet.shape[2])
    return outputValues_vol, outputValues_net, support, pixelValues, raysSet, pixelPositions






# Projection if sampled on a square grid
# Ensure that the amplitude is independent of the grid size
def projection(volume_grid):
    # return volume_grid.sum(2)/ball.sum(2)
    n = volume_grid.shape[2]
    return volume_grid.sum(2)/n

"""
Projection if sampled on a discretization of a ball


xx = np.linspace(-1,1,n)
XX, YY, ZZ = np.meshgrid(xx,xx,xx)
grid = np.concatenate([YY.reshape(-1,1),XX.reshape(-1,1),ZZ.reshape(-1,1)],1)
grid_t = torch.tensor(grid).to(device).type(torch_type)
ball = (XX**2+YY**2+ZZ**2) <= 1
"""
def resample_ball(inp,ball):
    n1,n2,n3 = ball.shape
    out = torch.zeros((n1*n2*n3)).type(inp.dtype).to(inp.device)
    out[ball.reshape(-1)] = inp.reshape(-1)
    return out.reshape(n1,n2,n3)

# given a volume array, returns the values of this array in a ball of radius 1
def sample_ball(inp,ball):
    return inp[ball.reshape(-1)]


class mollifier_class():
    # define molifier to ensure support and avoid padding artifacts
    def __init__(self, n, torch_type=torch.float, device='cpu'):
        super(mollifier_class, self).__init__()

        # Ensure to keep first version working
        if type(n) is int:
            if n==-1:
                self.mollifier = 1.
                self.mollifier_t = 1.
                self.mollifier2d_t = 1.
                self.mollifier2d = 1.

            else:
                xx = np.linspace(-1,1,n)
                XX, YY, ZZ = np.meshgrid(xx,xx,xx,indexing='ij')
                mollifier = (XX**2+YY**2+ZZ**2) <= 1
                s = 1/(n/2)
                G = np.exp(-(XX**2+YY**2+ZZ**2)/2/s**2)
                G = G/G.sum()
                self.mollifier = np.real(np.fft.fftshift(np.fft.ifftn(np.fft.fftn(G)*np.fft.fftn(mollifier))))
                self.mollifier_t = torch.tensor(self.mollifier).type(torch_type).to(device)
                self.mollifier2d_t = self.mollifier_t[:,:,n//2]
                self.mollifier2d = mollifier[:,:,n//2]

        else: 
            self.n1, self.n2, self.n3 = n
            xx1 = np.linspace(-1,1,self.n1)
            xx2 = np.linspace(-1,1,self.n2)
            xx3 = np.linspace(-1,1,self.n3)
            XX, YY, ZZ = np.meshgrid(xx1,xx2,xx3,indexing='ij')
            mollifier = (XX**2+YY**2+ZZ**2) <= 1            
            s = 1/(np.min(n)/2)
            G = np.exp(-(XX**2+YY**2+ZZ**2)/2/s**2)
            G = G/G.sum()
            self.mollifier = np.real(np.fft.fftshift(np.fft.ifftn(np.fft.fftn(G)*np.fft.fftn(mollifier))))
            self.mollifier_t = torch.tensor(self.mollifier).type(torch_type).to(device)
            self.mollifier2d_t = self.mollifier_t[:,:,self.n3//2]
            self.mollifier2d = mollifier[:,:,self.n3//2]

    def mollify2d(self):
        return self.mollifier2d_t
    
    def mollify3d(self):
        return self.mollifier_t
    
    def mollify3d_np(self):
        return self.mollifier

    def mollify2d_np(self):
        return self.mollifier2d
    

def generate_ball3d(n):
    xx = np.linspace(-1,1,n)
    XX, YY, ZZ = np.meshgrid(xx,xx,xx,indexing='ij')
    grid = np.concatenate([XX.reshape(-1,1),YY.reshape(-1,1),ZZ.reshape(-1,1)],1)
    ball = (XX**2+YY**2+ZZ**2) <= 1
    return ball

def generate_grid3d(n):
    xx = np.linspace(-1,1,n)
    XX, YY, ZZ = np.meshgrid(xx,xx,xx,indexing='ij')
    grid = np.concatenate([XX.reshape(-1,1),YY.reshape(-1,1),ZZ.reshape(-1,1)],1)
    return grid

def generate_grid2d(n):
    xx = np.linspace(-1,1,n)
    XX, YY = np.meshgrid(xx,xx,indexing='ij')
    grid2d = np.concatenate([XX.reshape(-1,1),YY.reshape(-1,1)],1)
    return grid2d


class grid_class():
    def __init__(self, n1, n2, n3, torch_type=torch.float32, device='cpu'):
        super(grid_class, self).__init__()
        self.n1 = n1 
        self.n2 = n2 
        self.n3 = n3 

        self.device = device
        self.torch_type = torch_type

        self.x_lin1 = np.linspace(-1,1,self.n1)
        self.x_lin2 = np.linspace(-1,1,self.n2)
        self.x_lin3 = np.linspace(-1,1,self.n3)

        self.XX, self.YY, self.ZZ = np.meshgrid(self.x_lin1,self.x_lin2,self.x_lin3,indexing='ij')
        self.XX2d, self.YY2d  = np.meshgrid(self.x_lin1,self.x_lin2,indexing='ij')

        self.grid3d = np.concatenate([self.XX.reshape(-1,1),self.YY.reshape(-1,1),self.ZZ.reshape(-1,1)],1)
        self.grid2d = np.concatenate([self.XX2d.reshape(-1,1),self.YY2d.reshape(-1,1)],1)

        self.ball3d = (self.XX**2+self.YY**2+self.ZZ**2) <= 1
        self.ball2d = (self.XX2d**2+self.YY2d**2) <= 1

        self.grid2d_t = torch.tensor(self.grid2d).to(self.device).type(self.torch_type)
        self.grid3d_t = torch.tensor(self.grid3d).to(self.device).type(self.torch_type)
        self.ball2d_t = torch.tensor(self.ball2d).to(self.device).type(self.torch_type)
        self.ball3d_t = torch.tensor(self.ball3d).to(self.device).type(self.torch_type)


        self.grid_ball3d = self.grid3d[self.ball3d.reshape(-1),:]
        self.grid_ball3d_t = torch.tensor(self.grid_ball3d).type(self.torch_type).to(self.device)
        self.grid_ball2d = self.grid2d[self.ball2d.reshape(-1),:]
        self.grid_ball2d_t = torch.tensor(self.grid_ball2d).type(self.torch_type).to(self.device)

    def grid2d_t(self):
        return self.grid2d_t
    def grid3d_t(self):
        return self.grid3d_t
    def ball2d_t(self):
        return self.ball2d_t
    def ball3d_t(self):
        return self.ball3d_t
    def grid_ball2d_t(self):
        return self.grid_ball2d_t
    def grid_ball3d_t(self):
        return self.grid_ball3d_t
    def grid2d(self):
        return self.grid2d
    def grid3d(self):
        return self.grid3d
    def ball2d(self):
        return self.ball2d
    def ball3d(self):
        return self.ball3d
    def grid_ball2d(self):
        return self.grid_ball2d
    def grid_ball3d(self):
        return self.grid_ball3d

"""
Define a new function to avoid conflicts.
TODO: need to adapt the mollifiers to ellipse and not ball.
"""
class grid_class_rectangular():
    def __init__(self, n1, n2, n3, torch_type=torch.float32, device='cpu'):
        super(grid_class_rectangular, self).__init__()
        self.n1 = n1 
        self.n2 = n2 
        self.n3 = n3 
        n = max(self.n1, self.n2, self.n3)

        self.device = device
        self.torch_type = torch_type

        self.x_lin1 = np.linspace(-self.n1/(n),self.n1/(n),self.n1)
        self.x_lin2 = np.linspace(-self.n2/(n),self.n2/(n),self.n2)
        self.x_lin3 = np.linspace(-self.n3/(n),self.n3/(n),self.n3)

        self.XX, self.YY, self.ZZ = np.meshgrid(self.x_lin1,self.x_lin2,self.x_lin3,indexing='ij')
        self.XX2d, self.YY2d  = np.meshgrid(self.x_lin1,self.x_lin2,indexing='ij')

        self.grid3d = np.concatenate([self.XX.reshape(-1,1),self.YY.reshape(-1,1),self.ZZ.reshape(-1,1)],1)
        self.grid2d = np.concatenate([self.XX2d.reshape(-1,1),self.YY2d.reshape(-1,1)],1)

        self.ball3d = (self.XX**2+self.YY**2+self.ZZ**2) <= 1
        self.ball2d = (self.XX2d**2+self.YY2d**2) <= 1

        self.grid2d_t = torch.tensor(self.grid2d).to(self.device).type(self.torch_type)
        self.grid3d_t = torch.tensor(self.grid3d).to(self.device).type(self.torch_type)
        self.ball2d_t = torch.tensor(self.ball2d).to(self.device).type(self.torch_type)
        self.ball3d_t = torch.tensor(self.ball3d).to(self.device).type(self.torch_type)


        self.grid_ball3d = self.grid3d[self.ball3d.reshape(-1),:]
        self.grid_ball3d_t = torch.tensor(self.grid_ball3d).type(self.torch_type).to(self.device)
        self.grid_ball2d = self.grid2d[self.ball2d.reshape(-1),:]
        self.grid_ball2d_t = torch.tensor(self.grid_ball2d).type(self.torch_type).to(self.device)

    def grid2d_t(self):
        return self.grid2d_t
    def grid3d_t(self):
        return self.grid3d_t
    def ball2d_t(self):
        return self.ball2d_t
    def ball3d_t(self):
        return self.ball3d_t
    def grid_ball2d_t(self):
        return self.grid_ball2d_t
    def grid_ball3d_t(self):
        return self.grid_ball3d_t
    def grid2d(self):
        return self.grid2d
    def grid3d(self):
        return self.grid3d
    def ball2d(self):
        return self.ball2d
    def ball3d(self):
        return self.ball3d
    def grid_ball2d(self):
        return self.grid_ball2d
    def grid_ball3d(self):
        return self.grid_ball3d


