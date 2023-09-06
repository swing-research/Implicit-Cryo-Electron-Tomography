import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import utils_sampling

#######################################################################################
## Define local deformations
#######################################################################################

## Function to perform bilinear interpolation. These functions are not use but can be usefull if we need to differentiate 
# through the deformation parameters.
def reflect_coords(ix, min_val, max_val):
    pos_delta = ix[ix>max_val] - max_val
    neg_delta = min_val - ix[ix < min_val]
    ix[ix>max_val] = ix[ix>max_val] - 2*pos_delta
    ix[ix<min_val] = ix[ix<min_val] + 2*neg_delta
    return ix

def grid_sample_customized_bilinear(image, grid, pad = 'reflect', align_corners = False):
    '''Differentiable grid_sample:
    equivalent performance with torch.nn.functional.grid_sample can be obtained by setting
    align_corners = True,
    pad: 'border': use border pixels,
    'reflect': create reflect pad manually.
    image is a tensor of shape (N, C, IH, IW)
    grid is a tensor of shape (N, H, W, 2)'''

    N, C, IH, IW = image.shape
    _, H, W, _ = grid.shape

    ix = grid[..., 0]
    iy = grid[..., 1]


    if align_corners == True:
        ix = ((ix + 1) / 2) * (IW-1);
        iy = ((iy + 1) / 2) * (IH-1);
        
        boundary_x = (0, IW-1)
        boundary_y = (0, IH-1)
        
    
    elif align_corners == False:
        ix = ((1+ix)*IW/2) - 1/2
        iy = ((1+iy)*IH/2) - 1/2
        
        boundary_x = (-1/2, IW-1/2)
        boundary_y = (-1/2, IH-1/2)
    

    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)

    if pad == 'reflect' or 'reflection':
        
        ix_nw = reflect_coords(ix_nw, boundary_x[0], boundary_x[1])
        iy_nw = reflect_coords(iy_nw, boundary_y[0], boundary_y[1])

        ix_ne = reflect_coords(ix_ne, boundary_x[0], boundary_x[1])
        iy_ne = reflect_coords(iy_ne, boundary_y[0], boundary_y[1])

        ix_sw = reflect_coords(ix_sw, boundary_x[0], boundary_x[1])
        iy_sw = reflect_coords(iy_sw, boundary_y[0], boundary_y[1])

        ix_se = reflect_coords(ix_se, boundary_x[0], boundary_x[1])
        iy_se = reflect_coords(iy_se, boundary_y[0], boundary_y[1])


    elif pad == 'border':

        with torch.no_grad():
            torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
            torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

            torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
            torch.clamp(iy_ne, 0, IH-1, out=iy_ne)

            torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
            torch.clamp(iy_sw, 0, IH-1, out=iy_sw)

            torch.clamp(ix_se, 0, IW-1, out=ix_se)
            torch.clamp(iy_se, 0, IH-1, out=iy_se)


    image = image.reshape(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
            ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
            sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
            se_val.view(N, C, H, W) * se.view(N, 1, H, W))
    return out_val


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
    f = F.affine_grid(theta, size=(b, c, output_size, output_size), align_corners=True)
    # image_cropped = grid_sample_customized_bilinear(image, f, align_corners = True)
    # We might need to change with the above if we want to pass gradient more than once to the parameters of the deformation
    image_cropped = F.grid_sample(image, f, mode='bicubic', align_corners = True, padding_mode=padding_mode)
    return image_cropped


"""
Only implement in 2d at the moment.
INPUT:
  -depl_ctr_pts: size (dim,N_pts,N_pts)
  -scaling: scalar to control the amplitude of the generated deformation (only used if depl_ctr_pts is None)
  N_ctrl_pts: number of control points (only used if depl_ctr_pts is None)
  -torch_type: type of torch tensors
  -device: device of torch tensors

"""
class deformation_field(nn.Module):
    def __init__(self, depl_ctr_pts = None, scaling=(1,1), N_ctrl_pts = (-1,-1), torch_type=torch.float, device='cpu',maskBoundary = 1):
        super(deformation_field, self).__init__()
        if depl_ctr_pts is None:
            if type(N_ctrl_pts) is int:
                max_ampl = 1/(N_ctrl_pts-1)
                depl_ctr_pts = (torch.rand(2,N_ctrl_pts+2,N_ctrl_pts+2, dtype=torch_type, device=device)-0.5)*2*max_ampl*scaling
            else:
                max_ampl = 1/(N_ctrl_pts-1)
                depl_ctr_pts = (torch.rand(2,N_ctrl_pts+2,N_ctrl_pts+2, dtype=torch_type, device=device)-0.5)
                depl_ctr_pts[0] *= 2*max_ampl*scaling[0]
                depl_ctr_pts[1] *= 2*max_ampl*scaling[1]
        mask = torch.ones_like(depl_ctr_pts)
        if(maskBoundary>=1):
            depl_ctr_pts[:,:maskBoundary,:] = 0.
            depl_ctr_pts[:,-maskBoundary:,:] = 0.
            depl_ctr_pts[:,:,:maskBoundary] = 0.
            depl_ctr_pts[:,:,-maskBoundary:] = 0.
            mask[:,:maskBoundary,:] = 0.
            mask[:,-maskBoundary:,:] = 0.
            mask[:,:,:maskBoundary] = 0.
            mask[:,:,-maskBoundary:] = 0.

        self.mask = nn.Parameter(mask,requires_grad=False)
        self.depl_ctr_pts = depl_ctr_pts
        self.N_ctrl_pts = depl_ctr_pts.shape[-2:]
        self.dim_out = depl_ctr_pts.shape[0]      
        self.depl_ctr_pts = nn.Parameter(torch.unsqueeze(self.depl_ctr_pts, dim = 0)).type(torch.float).to(depl_ctr_pts.device)


        n1, n2 = self.N_ctrl_pts
        xx1 = torch.linspace(-1,1,n1,dtype=self.depl_ctr_pts.dtype,device=self.depl_ctr_pts.device)
        xx2 = torch.linspace(-1,1,n2,dtype=self.depl_ctr_pts.dtype,device=self.depl_ctr_pts.device)
        XX_t, YY_t = torch.meshgrid(xx1,xx2,indexing='ij')
        XX_t = torch.unsqueeze(XX_t, dim = 0)
        YY_t = torch.unsqueeze(YY_t, dim = 0)
        self.coordinates = torch.cat([XX_t,YY_t],0)

        self.test_invertible()

    def test_invertible(self):
        n1, n2 = self.N_ctrl_pts
        dep = self.depl_ctr_pts[0]+self.coordinates
        _, iix = torch.sort(dep,1)
        _, iiy = torch.sort(dep,2)
        indx_true_x = torch.arange(n1).reshape(-1,1).repeat(1,n2).to(iix.device)
        indx_true_y = torch.arange(n2).reshape(1,-1).repeat(n1,1).to(iix.device)
        if (indx_true_x == iix[0]).sum()<n1*n2 or (indx_true_y == iiy[1]).sum()<n1*n2:
            print("###########################################################################")
            print("#### Deformation is not invertible, choose another displacement field! ####")
            print("###########################################################################")


    # img of size (batch,N1,N2)
    def transform_image(self,img,coordinates=None):
        n1, n2 = img.shape[-2:]
        if coordinates is None:
            # n1, n2 = self.N_ctrl_pts
            xx1 = torch.linspace(-1,1,n1,dtype=self.depl_ctr_pts.dtype,device=self.depl_ctr_pts.device)
            xx2 = torch.linspace(-1,1,n2,dtype=self.depl_ctr_pts.dtype,device=self.depl_ctr_pts.device)
            XX_t, YY_t = torch.meshgrid(xx1,xx2,indexing='ij')
            XX_t = torch.unsqueeze(XX_t, dim = 2)
            YY_t = torch.unsqueeze(YY_t, dim = 2)
            coordinates = torch.cat([XX_t,YY_t],2).reshape(-1,2)
        deformations = self(coordinates)
        x = img.expand(n1*n2, -1, -1, -1)
        return cropper(x,coordinates+deformations,output_size = 1).reshape(-1,n1,n2)

    # coordinate is a batch of coordinate, size (batch, dim_out)
    # return the displacements at the coordinates
    def forward(self, coordinate):
        b , _ = coordinate.shape
        x = (self.depl_ctr_pts*self.mask).expand(b, -1, -1, -1)
        x = cropper(x, coordinate, output_size = 1, padding_mode="reflection")
        return torch.squeeze(x)
    
# def interp_img(img,coordinates=None):
#     if coordinates is None:
#         n = img.shape[-1]
#         xx = torch.linspace(-1,1,n,dtype=img.dtype,device=img.device)
#         XX_t, YY_t = torch.meshgrid(xx,xx)
#         XX_t = torch.unsqueeze(XX_t, dim = 2)
#         YY_t = torch.unsqueeze(YY_t, dim = 2)
#         coordinates = torch.cat([XX_t,YY_t],2).reshape(-1,2)
#     return cropper(img,coordinates,output_size = 1).reshape(-1,n,n)

def apply_local_deformation(field,img):
    N= img.shape[0]
    yout = torch.zeros_like(img)
    for i in range(N):
        yout[i] = field[i].transform_image(img[i].unsqueeze(0))
    return yout

#######################################################################################
## Define global deformations (shift and rotation)
#######################################################################################
T = torch.tensor
# Warning, this is not exactly the same implementation than in the ICASSP paper.
# Here, the transformation matrix is manually apply to the grid.
class AffineTransform(nn.Module):
    def __init__(self, scaleX=1, scaleY=1, shiftX=0, shiftY=0, shearX=0, shearY=0, angle=0, pad=20):
        super().__init__()
        self.pad = pad
        self.scaleX = nn.Parameter(T(scaleX, dtype=torch.float))
        self.scaleY = nn.Parameter(T(scaleY, dtype=torch.float))
        self.shiftX = nn.Parameter(T(shiftX, dtype=torch.float))
        self.shiftY = nn.Parameter(T(shiftY, dtype=torch.float))
        self.shearX = nn.Parameter(T(shearX, dtype=torch.float))
        self.shearY = nn.Parameter(T(shearY, dtype=torch.float))
        self.angle = nn.Parameter(T(angle, dtype=torch.float))

    def theta(self,device):
        return torch.stack(
            [torch.stack([self.scaleX*torch.cos(self.angle)+self.shearX*torch.sin(self.angle)*self.scaleX,-torch.sin(self.angle)*self.scaleY+self.shearX*self.scaleY*torch.cos(self.angle), self.shiftX]),
            torch.stack([self.shearY*self.scaleX*torch.cos(self.angle)+torch.sin(self.angle)*self.scaleX,-self.shearY*torch.sin(self.angle)*self.scaleY+self.scaleY*torch.cos(self.angle), self.shiftY])]
            ,0)[None].cuda(device)

    def theta_inv(self):
        return torch.stack(
            [torch.stack([(1./self.scaleX)*torch.cos(-self.angle)-self.shearX*torch.sin(-self.angle)*(1./self.scaleX),-torch.sin(-self.angle)*(1./self.scaleY)-self.shearX*(1./self.scaleY)*torch.cos(-self.angle), -self.shiftX]),
            torch.stack([-self.shearY*(1./self.scaleX)*torch.cos(-self.angle)+torch.sin(-self.angle)*(1./self.scaleX),+self.shearY*torch.sin(-self.angle)*(1./self.scaleY)+(1./self.scaleY)*torch.cos(-self.angle), -self.shiftY])]
            ,0)[None].cuda()

    def inverse_transform(self, x):
        # grid = F.affine_grid(self.theta_inv(), x.size(), align_corners=False,padding_mode='reflection')
        m = nn.ReflectionPad2d(self.pad)
        x_pad = m(x)
        n1,n2 = x_pad.shape[-2:]
        xx1 = np.linspace(-1,1,n1)
        xx2 = np.linspace(-1,1,n2)
        XX, YY = np.meshgrid(xx1,xx2,indexing='ij')
        grid2d = np.concatenate([YY.reshape(-1,1),XX.reshape(-1,1)],1) # same as in cropper, for grid_sampler
        grid2d_t = torch.tensor(grid2d).to(x.device).type(x.dtype)

        mat = self.theta_inv()
        mat_view = mat[0,:2,:2]
        idx = torch.LongTensor([1,0]).to(mat.device)
        ss = (mat[0,:,2][None]*x.shape[-1]/x_pad.shape[-1]).index_select(1,idx)
        grid_tmp = torch.matmul(grid2d_t,torch.transpose(mat_view,0,1))+ss
        grid_tmp = grid_tmp.reshape(1,n1,n2,2)
        out_pad = F.grid_sample(x_pad, grid_tmp, mode='bilinear', align_corners=True)
        return out_pad[:,:,self.pad:-self.pad,self.pad:-self.pad]

    def forward(self, x):
        m = nn.ReflectionPad2d(self.pad)
        x_pad = m(x)
        n1,n2 = x_pad.shape[-2:]
        xx1 = np.linspace(-1,1,n1)
        xx2 = np.linspace(-1,1,n2)
        XX, YY = np.meshgrid(xx1,xx2,indexing='ij')
        grid2d = np.concatenate([YY.reshape(-1,1),XX.reshape(-1,1)],1) # same as in cropper, for grid_sampler
        grid2d_t = torch.tensor(grid2d).to(x.device).type(x.dtype)

        mat = self.theta(x.device)
        mat_view = mat[0,:2,:2]
        idx = torch.LongTensor([1,0]).to(mat.device)
        ss = (mat[0,:,2][None]*x.shape[-1]/x_pad.shape[-1]).index_select(1,idx)
        # Apply in-plane rotation
        grid_tmp = torch.matmul(grid2d_t,torch.transpose(mat_view,0,1))
        # Apply shift
        grid_tmp = grid_tmp+ss
        grid_tmp = grid_tmp.reshape(1,n1,n2,2)
        out_pad = F.grid_sample(x_pad, grid_tmp, mode='bilinear', align_corners=True)
        return out_pad[:,:,self.pad:-self.pad,self.pad:-self.pad]
    
    def __str__(self):
        return 'Deformation(scaleX={:.2f}, scaleY={:.2f}, shiftX={:.2f}, shiftY={:.2f}, shearX={:.2f}, shearY={:.2f}, angle={:.2f})'.format(
            self.scaleX.item(), self.scaleY.item(), self.shiftX.item(), self.shiftY.item(), self.shearX.item(), self.shearY.item(), self.angle.item())

def generate_params_deformation(scale_min=1, scale_max=1, shift_min=0., shift_max=0., shear_min=0, shear_max=0, angle_min=0, angle_max=0.):
    scaleX = np.random.random()*(scale_max-scale_min)+scale_min
    scaleY = np.random.random()*(scale_max-scale_min)+scale_min
    shiftX = np.random.random()*(shift_max-shift_min)+shift_min
    shiftY = np.random.random()*(shift_max-shift_min)+shift_min
    shearX = np.random.random()*(shear_max-shear_min)+shear_min
    shearY = np.random.random()*(shear_max-shear_min)+shear_min
    angle = np.random.random()*(angle_max-angle_min)+angle_min
    return scaleX, scaleY, shiftX, shiftY, shearX, shearY, angle

# img of size (N_batch,Nx,Ny)
# affine_tr is a list of AffineTransform
def apply_deformation(affine_tr,img):
    N= img.shape[0]
    yout = torch.zeros_like(img)
    for i in range(N):
        yout[i] = affine_tr[i](img[i].unsqueeze(0).unsqueeze(0))[0,0]
    return yout

def apply_deformation_inverse(affine_tr,img):
    N= img.shape[0]
    yout = torch.zeros_like(img)
    for i in range(N):
        yout[i] = affine_tr[i].inverse_transform(img[i].unsqueeze(0).unsqueeze(0))[0,0]
    return yout


## Define learnable deformations
class shiftNet(nn.Module):
    def __init__(self, Nproj, x0=None):
        super().__init__()
        if x0 != None:
            shifts_arr = x0.clone()
        else:
            shifts_arr = torch.zeros((Nproj,2), dtype=torch.float)
        self.shifts_arr = nn.Parameter(shifts_arr)
    def forward(self,k=-1):
        if k == -1:
            return self.shifts_arr
        else:
            return self.shifts_arr[k]

class rotNet(nn.Module):
    def __init__(self, Nproj, x0=None):
        super().__init__()
        self.Nproj = Nproj
        if x0 != None:
            thetas = x0.clone()
            self.thetas = thetas
        else:
            if Nproj!=1:
                thetas = torch.zeros((Nproj), dtype=torch.float)
            else:
                thetas = torch.tensor(0.).type(torch.float)
            self.thetas = nn.Parameter(thetas)

        if Nproj!=1:
            self.e3 = torch.tensor([[0, 0, 1]]).repeat(Nproj,1).type_as(thetas)
        else:
            self.e3 = torch.tensor([0, 0, 1]).type_as(thetas)
    
    def forward(self,k=-1,dim=3):
        if dim == 3:
            if k ==-1:
                if self.Nproj!=1:
                    return torch.stack(
                        [torch.stack([torch.cos(self.thetas),-torch.sin(self.thetas), torch.zeros_like(self.thetas)],1),
                        torch.stack([torch.sin(self.thetas),torch.cos(self.thetas), torch.zeros_like(self.thetas)],1),
                        self.e3.to(self.thetas.device)]
                        ,1)
                else:
                    return torch.stack(
                        [torch.stack([torch.cos(self.thetas),torch.sin(self.thetas), torch.zeros_like(self.thetas)],0),
                        torch.stack([-torch.sin(self.thetas),torch.cos(self.thetas), torch.zeros_like(self.thetas)],0),
                        self.e3.to(self.thetas.device)]
                        ,0)
            else:
                return torch.stack(
                    [torch.stack([torch.cos(self.thetas[k]),-torch.sin(self.thetas[k]), torch.zeros_like(self.thetas[k])],0),
                    torch.stack([torch.sin(self.thetas[k]),torch.cos(self.thetas[k]), torch.zeros_like(self.thetas[k])],0),
                    self.e3[k].to(self.thetas.device)]
                    ,1)
        else:
            if k ==-1:
                if self.Nproj!=1:
                    return torch.stack(
                        [torch.stack([torch.cos(self.thetas),-torch.sin(self.thetas)],1),
                        torch.stack([torch.sin(self.thetas),torch.cos(self.thetas)],1)]
                        ,1)
                else:
                    return torch.stack(
                        [torch.stack([torch.cos(self.thetas),torch.sin(self.thetas)],0),
                        torch.stack([-torch.sin(self.thetas),torch.cos(self.thetas)],0)]
                        ,0)
            else:
                return torch.stack(
                    [torch.stack([torch.cos(self.thetas[k]),-torch.sin(self.thetas[k])],0),
                    torch.stack([torch.sin(self.thetas[k]),torch.cos(self.thetas[k])],0)]
                    ,1)


# load deformation parameters from the dataset folder
# type = 0 when the input to affine_tr and local_tr are saved as numpy arrays
# type = 1 when affine_tr and local_tr are saved
def loadDeformationParameters(dataPath, device, type=0):
    # Loading the deformation paramters
    if(type==0):
        angle_set = np.load('datasets/areTomo/'+dataPath+'/deformation_parameters/angle_set.npy')
        scaleX_set = np.load('datasets/areTomo/'+dataPath+'/deformation_parameters/scaleX_set.npy')
        scaleY_set = np.load('datasets/areTomo/'+dataPath+'/deformation_parameters/scaleY_set.npy')
        shearX_set = np.load('datasets/areTomo/'+dataPath+'/deformation_parameters/shearX_set.npy')
        shearY_set = np.load('datasets/areTomo/'+dataPath+'/deformation_parameters/shearY_set.npy')
        shiftX_set = np.load('datasets/areTomo/'+dataPath+'/deformation_parameters/shiftX_set.npy')
        shiftY_set = np.load('datasets/areTomo/'+dataPath+'/deformation_parameters/shiftY_set.npy')
        deep_ctr_pts_set = np.load('datasets/areTomo/'+dataPath+'/deformation_parameters/deep_ctr_pts_set.npy')

        affine_tr = []
        local_tr = []

        for i in range(len(angle_set)):
            affine_tr.append(AffineTransform(scaleX_set[i], scaleY_set[i], shiftX_set[i], shiftY_set[i], shearX_set[i], shearY_set[i], angle_set[i]).to(device))
            local_tr.append(deformation_field(torch.from_numpy(deep_ctr_pts_set[i])).to(device))

    if(type==1):
        affine_tr = np.load('datasets/areTomo/'+dataPath+'/global_deformations.npy', allow_pickle=True)
        local_tr = np.load('datasets/areTomo/'+dataPath+'/local_deformations.npy', allow_pickle=True)

    return affine_tr, local_tr



if __name__ == '__main__':
    torch_type=torch.float
    use_cuda=torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    if torch.cuda.device_count()>1:
        torch.cuda.set_device(0)
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed) 


    ## Parameters
    n = 65

    ## Define grid to visualize displacements
    grid = utils_sampling.generate_grid2d(n)
    grid_t = torch.tensor(grid).to(device).type(torch_type)
    im_grid = np.zeros((n,n))
    im_grid[::10,::10] = 1
    im_grid_t = torch.tensor(im_grid).to(device).type(torch_type)


    # Define custom displacement grid
    sigma = 10
    N_ctrl_pts = 5
    depl_ctr_pts = torch.ones((2,N_ctrl_pts,N_ctrl_pts)).to(device).type(torch_type)*2/n*sigma
    depl_ctr_pts[0] *= 0.25
    # XX_t = torch.unsqueeze(torch.tensor(XX).to(device).type(torch_type), dim = 2)
    # YY_t = torch.unsqueeze(torch.tensor(YY).to(device).type(torch_type), dim = 2)
    # coordinates = torch.cat([XX_t,YY_t],2)
    coordinates = grid_t.reshape(n,n,2)

    # Define dispalcement model
    field = deformation_field(depl_ctr_pts)
    # Apply the displacements on some coordinates
    out = field(coordinates.reshape(-1,2))
    # Transform an image according to the displacement model
    img_deform = field.transform_image(im_grid_t.reshape(1,1,n,n))

    plt.figure(1)
    plt.subplot(2,2,1)
    plt.imshow(field.depl_ctr_pts.detach().cpu().numpy()[0,0])
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.imshow(out.detach().cpu().numpy()[:,0].reshape(n,n))
    plt.colorbar()
    plt.subplot(2,2,2)
    plt.imshow(im_grid_t.detach().cpu().numpy())
    plt.subplot(2,2,4)
    plt.imshow(img_deform.detach().cpu().numpy()[0].reshape(n,n))

    # Let the model define a random deformation field that is guarentee to be admissible
    N_ctrl_pts = 5
    scaling = 0.5
    # XX_t = torch.unsqueeze(torch.tensor(XX).to(device).type(torch_type), dim = 2)
    # YY_t = torch.unsqueeze(torch.tensor(YY).to(device).type(torch_type), dim = 2)
    # coordinates = torch.cat([XX_t,YY_t],2)

    # Define dispalcement model
    field = deformation_field(depl_ctr_pts=None,scaling=scaling,N_ctrl_pts=N_ctrl_pts,torch_type=torch_type,device=device)
    # Apply the displacements on some coordinates
    out = field(coordinates.reshape(-1,2))
    # Transform an image according to the displacement model
    img_deform = field.transform_image(im_grid_t.reshape(1,1,n,n))

    plt.figure(2)
    plt.clf()
    plt.subplot(2,2,1)
    plt.imshow(0.5*n*field.depl_ctr_pts.detach().cpu().numpy()[0,0])
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.imshow(0.5*n*out.detach().cpu().numpy()[:,0].reshape(n,n))
    plt.colorbar()
    plt.subplot(2,2,2)
    plt.imshow(im_grid_t.detach().cpu().numpy())
    plt.subplot(2,2,4)
    plt.imshow(img_deform.detach().cpu().numpy()[0].reshape(n,n))


    #################################################################################
    ## Test global deformation
    #################################################################################
    # xx = np.linspace(-1,1,n)
    # XX, YY, ZZ = np.meshgrid(xx,xx,xx)
    # grid = np.concatenate([YY.reshape(-1,1),XX.reshape(-1,1),ZZ.reshape(-1,1)],1)
    # grid_t = torch.tensor(grid).to(device).type(torch_type)
    # ball = (XX**2+YY**2+ZZ**2) <= 1
    # grid_ball = grid[ball.reshape(-1),:]
    # grid_ball_t = torch.tensor(grid_ball).type(torch_type).to(device)
    # XX, YY = np.meshgrid(xx,xx)
    # grid2d = np.concatenate([XX.reshape(-1,1),YY.reshape(-1,1)],1)
    # grid2d_t = torch.tensor(grid2d).to(device).type(torch_type)


    # Deformation
    deformation_scale = 2
    scale_min = 1.0
    scale_max = 1.0
    shift_min = -0.05*deformation_scale
    shift_max = 0.05*deformation_scale
    shear_min = -0.0*deformation_scale
    shear_max = 0.0*deformation_scale
    angle_min = -5/180*np.pi*deformation_scale
    angle_max = 5/180*np.pi*deformation_scale

    scaleX, scaleY, shiftX, shiftY, shearX, shearY, angle  = generate_params_deformation(scale_min,scale_max,shift_min,shift_max,shear_min,shear_max,angle_min,angle_max)
    rigid_transform = AffineTransform(scaleX, scaleY, shiftX, shiftY, shearX, shearY, angle ).cuda()

    im_deformed = apply_deformation([rigid_transform],im_grid_t.reshape(1,n,n))
    im_original_est = apply_deformation_inverse([rigid_transform],im_deformed)


    plt.figure(3)
    plt.clf()
    plt.subplot(1,3,1)
    plt.imshow(im_grid_t.detach().cpu().numpy())
    plt.subplot(1,3,2)
    plt.imshow(im_original_est[0].detach().cpu().numpy())
    plt.subplot(1,3,3)
    plt.imshow(im_deformed[0].detach().cpu().numpy())





    plt.show()

