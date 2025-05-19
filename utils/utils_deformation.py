
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from . import utils_sampling

#######################################################################################
## Define local deformations
#######################################################################################
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
        return utils_sampling.cropper(x,coordinates+deformations,output_size = 1).reshape(-1,n1,n2)

    # coordinate is a batch of coordinate, size (batch, dim_out)
    # return the displacements at the coordinates
    def forward(self, coordinate):
        b , _ = coordinate.shape
        x = (self.depl_ctr_pts*self.mask).expand(b, -1, -1, -1)
        x = utils_sampling.cropper(x, coordinate, output_size = 1, padding_mode="reflection")
        return torch.squeeze(x)

#######################################################################################
## Define global deformations (shift and rotation)
#######################################################################################
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



def globalDeformationValues(shift,rot):
    """
    Return the global deformations (shifts and rotations) as np array.
    Input are lists of shiftNet and rotNet.
    """
    shiftValueList = []
    rotValueList = []
    for si, ri in  zip(shift,rot):
        shiftValue = si().clone().detach().cpu().numpy()
        rotValue = ri.thetas.clone().detach().cpu().numpy()
        shiftValueList.append(shiftValue)
        rotValueList.append(rotValue)
    shiftValueList = np.array(shiftValueList)
    rotValueList = np.array(rotValueList)
    return shiftValueList, rotValueList

