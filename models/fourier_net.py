"""Holds all siren and related utilities."""
import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from utils import utils_interpolation
from utils import utils_deformation





"""
Create multi-resolution features similarly to Tiny CUDA Neural Networks

d: dimension of the implicit representation, only 2 or 3 implemented
res: list of pixel size grids to use to define the features
nFeature: number of feature per grid
L: number of frequency to encode position using Fourier features


How to use:
batch = 5

d = 3
nFeature = 10
res = [16,32,64]
L = 1

torch_type=torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

x = (torch.rand((batch,d)).type(torch_type).to(device)-0.5)*2

net = MultiResImplicitFeature(d, nFeature, res, L).cuda()

out = net(x)


"""
class MultiResImplicitFeature(nn.Module):
  def __init__(self, d, nFeature, res, L=1):
    super().__init__()
    self.res = res
    self.nFeature = nFeature
    self.d = d

    self.L = L
    self.coefs = nn.Parameter(torch.arange(start=1, end=L+1e-12) * math.pi * 0.5,requires_grad=False)

    self.features = []
    for r in res:
      if self.d==2:
        self.features.append(nn.Parameter( torch.randn((1,self.nFeature,r,r)) ,requires_grad=True))
      elif self.d==3:
        self.features.append(nn.Parameter( torch.randn((1,self.nFeature,r,r,r)) ,requires_grad=True))
      
    self.features = nn.ParameterList(self.features)

  def forward(self, x):
    # Create encoding
    argument = torch.kron(self.coefs, x)
    out1 = torch.sin(argument)
    out2 = torch.cos(argument)
    fx = torch.hstack((out1, out2 ))
    
    if self.d == 3:
      for f in self.features:
        fx_ = torch.squeeze(torch.squeeze(torch.squeeze(utils_interpolation.cropper3d(f.expand(x.shape[0],-1,-1,-1,-1),x,1),4),3),2)
        fx = torch.hstack((fx, fx_ ))
    elif self.d == 2:
      for f in self.features:
        fx_ = utils_deformation.cropper(f.expand(x.shape[0],-1,-1,-1),x,1,1,1)
        fx = torch.hstack((fx, fx_ ))
    return fx

"""
Combine feature extraction using MultiResImplicitFeature and the processing
using FourierNet

hidden_features: number features for the MLP
hidden_blocks:  number of hidden blocks for the MLP
out_features; number of output features

How to use: 

batch = 5

d = 3
nFeature = 10
res = [16,32,64]
L = 1
hidden_features = 128
hidden_blocks = 3
out_features = 2
torch_type=torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
x = (torch.rand((batch,d)).type(torch_type).to(device)-0.5)*2
net = MultiResImplicitRepresentation(d, nFeature, res, L, hidden_features, hidden_blocks, out_features).cuda()
out = net(x)


"""
class MultiResImplicitRepresentation(nn.Module):
  def __init__(self, d, nFeature, res, L, hidden_features, hidden_blocks, out_features):
    super().__init__()
    in_features = nFeature*len(res)+L*d*2
    self.net_features = MultiResImplicitFeature(d, nFeature, res, L)
    self.net_MLP = MLP(in_features, hidden_features, hidden_blocks, out_features)

  def forward(self,x):
    features = self.net_features(x)
    out_value = self.net_MLP(features)
    return out_value
  
  def forward_and_features(self,x):
    features = self.net_features(x)
    out_value = self.net_MLP(features)
    return out_value, features
    




class FFMLayer(nn.Module):
  def __init__(self, rep_dim, L=10):
    super().__init__()
    self.L = L
    self.rep_dim = rep_dim
    self.coefs = nn.Parameter(torch.arange(start=1, end=L+1e-12) * math.pi * 0.5,requires_grad=False)

  def forward(self, x, frac=1.):
    # coefs[int(np.round(self.L*frac)):] = 0.
    argument = torch.kron(self.coefs, x)
    out1 = torch.sin(argument)
    out2 = torch.cos(argument)
    out1[:,int(np.round(self.L*frac)):] = 0
    out2[:,int(np.round(self.L*frac)):] = 0
    out = torch.hstack((out1, out2 ))
    return out
  
class FFMLayer_v2(nn.Module):
  def __init__(self, rep_dim, L=10):
    super().__init__()
    self.L = L
    # self.rep_dim = rep_dim
    self.coefs = nn.Parameter(2.**torch.arange(start=0, end=L) * math.pi ,requires_grad=False)

  def forward(self, x, frac=1.):
    # coefs[int(np.round(self.L*frac)):] = 0.
    argument = torch.kron(self.coefs, x)
    out1 = torch.sin(argument)
    out2 = torch.cos(argument)
    out1[:,int(np.round(self.L*frac)):] = 0
    out2[:,int(np.round(self.L*frac)):] = 0
    out = torch.hstack((out1, out2 ))
    return out

class FourierNet(nn.Module):
  def __init__(self,
         in_features,
         hidden_features,
         hidden_blocks,
         out_features,
         L = 10):
    super().__init__()

    self.ffm = FFMLayer_v2(in_features, L)
    ffm_expansion_size = 2*in_features*L

    self.blocks = []

    ### First block
    self.blocks.append(nn.ModuleList([
      nn.Linear(ffm_expansion_size, hidden_features),
      nn.Linear(hidden_features, hidden_features)
      ]))

    ### Hidden block
    for i in range(hidden_blocks-1):
      self.blocks.append(nn.ModuleList([
        nn.Linear(hidden_features + ffm_expansion_size, hidden_features),
        nn.Linear(hidden_features, hidden_features)
      ]))

    ### Final
    self.final_block = [
      nn.Linear(hidden_features + ffm_expansion_size, hidden_features),
      nn.Linear(hidden_features, int(hidden_features / 2)),
      nn.Linear(int(hidden_features / 2), out_features)
      ]

    self.blocks = nn.ModuleList(self.blocks)
    self.final_block = nn.ModuleList(self.final_block)

  def forward(self, coords, frac=1):
    ffm_out = self.ffm(coords, frac)
    x = ffm_out

    for b in range(len(self.blocks)):
      fcs = self.blocks[b]
      x = fcs[0](x)
      x = F.relu(x)
      x = fcs[1](x)
      x = F.relu(x)
      x = torch.cat((x, ffm_out), dim=1)
    
    x = self.final_block[0](x)
    x = F.relu(x)
    x = self.final_block[1](x)
    x = F.relu(x)
    x = self.final_block[2](x)

    return x
  
class FourierNet_Features(nn.Module):
  def __init__(self,
         in_features,
         sub_features,
         hidden_features,
         hidden_blocks,
         out_features,
         L = 10):
    super().__init__()

    self.ffm = FFMLayer_v2(in_features, L)
    ffm_expansion_size = 2*in_features*L
    self.in_features = in_features

    self.blocks = []

    ### First block
    self.blocks.append(nn.ModuleList([
      nn.Linear(ffm_expansion_size+sub_features, hidden_features),
      nn.Linear(hidden_features, hidden_features)
      ]))

    ### Hidden block
    for i in range(hidden_blocks-1):
      self.blocks.append(nn.ModuleList([
        nn.Linear(hidden_features + ffm_expansion_size+sub_features, hidden_features),
        nn.Linear(hidden_features, hidden_features)
      ]))

    ### Final
    self.final_block = [
      nn.Linear(hidden_features + ffm_expansion_size+sub_features, hidden_features),
      nn.Linear(hidden_features, int(hidden_features / 2)),
      nn.Linear(int(hidden_features / 2), out_features)
      ]

    self.blocks = nn.ModuleList(self.blocks)
    self.final_block = nn.ModuleList(self.final_block)

  def forward(self, inputVal, frac=1):
    coords = inputVal[:,:self.in_features]
    sub = inputVal[:,self.in_features:]

    ffm_out = self.ffm(coords, frac)
    x = torch.concat((ffm_out, sub), dim=1)


    for b in range(len(self.blocks)):
      fcs = self.blocks[b]
      x = fcs[0](x)
      x = F.relu(x)
      x = fcs[1](x)
      x = F.relu(x)
      x = torch.cat((x, torch.concat((ffm_out, sub), dim=1)), dim=1)
    
    x = self.final_block[0](x)
    x = F.relu(x)
    x = self.final_block[1](x)
    x = F.relu(x)
    x = self.final_block[2](x)

    return x

class FourierNetBlock(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    
    self.fc1 = nn.Linear(in_features, out_features)
    self.fc2 = nn.Linear(out_features, out_features)
    
  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    return x
    
class FourierNetFinalBlock(nn.Module):
  def __init__(self, in_features, hidden_features, out_features):
    super().__init__()
    
    self.fc1 = nn.Linear(in_features, hidden_features)
    self.fc2 = nn.Linear(hidden_features, int(hidden_features / 2))
    self.fc3 = nn.Linear(int(hidden_features / 2), out_features)
      
  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    x = self.fc3(x)
    return x

class FourierNetCheckpointed(nn.Module):
  def __init__(self, in_features, hidden_features, out_features, L = 10):
    super().__init__()

    self.ffm = FFMLayer(in_features, L)
    ffm_expansion_size = 2*in_features*L
    
    self.block1 = FourierNetBlock(ffm_expansion_size,
                                       hidden_features)
    self.block2 = FourierNetBlock(hidden_features + ffm_expansion_size,
                                       hidden_features)
    self.block3 = FourierNetBlock(hidden_features + ffm_expansion_size,
                                       hidden_features)
    self.block4 = FourierNetBlock(hidden_features + ffm_expansion_size,
                                       hidden_features)
    self.block5 = FourierNetBlock(hidden_features + ffm_expansion_size,
                                       hidden_features)
    self.block6 = FourierNetBlock(hidden_features + ffm_expansion_size,
                                       hidden_features)
    
    self.final = FourierNetFinalBlock(hidden_features + ffm_expansion_size,
                                           hidden_features, 
                                           out_features)
      
  def custom(self, module):
    def custom_forward(*inputs):
        inputs = module(inputs[0])
        return inputs
    return custom_forward
    
  def forward(self, x):
      
    ffm_out = self.ffm(x)
    x = ffm_out
    
    x = self.block1(x)
    x = torch.cat((x, ffm_out), dim=1)
    
#         x = self.block2(x)
    x = checkpoint.checkpoint(self.custom(self.block2), x)
    x = torch.cat((x, ffm_out), dim=1)
    
#         x = self.block3(x)
    x = checkpoint.checkpoint(self.custom(self.block3), x)
    x = torch.cat((x, ffm_out), dim=1)
    
#         x = self.block4(x)
    x = checkpoint.checkpoint(self.custom(self.block4), x)
    x = torch.cat((x, ffm_out), dim=1)
    
#         x = self.block5(x)
    x = checkpoint.checkpoint(self.custom(self.block5), x)
    x = torch.cat((x, ffm_out), dim=1)
    
#         x = self.block6(x)
    x = checkpoint.checkpoint(self.custom(self.block6), x)
    x = torch.cat((x, ffm_out), dim=1)
    
#         x = self.final(x)
    x = checkpoint.checkpoint(self.custom(self.final), x)

    return x    

def unit_test_ffm_dimension():
  import ops, utils

  operator = ops.ParallelBeamGeometryOp(64, 60, 500)
  grid_params = {
    'angles': torch.cos(torch.tensor(operator.angles, dtype=torch.float32)),
    'num_detectors': operator.num_detectors
    }
  grid = utils.get_sino_mgrid(**grid_params)
  print (grid.shape)

  ffm = FFMLayer(rep_dim = grid.shape[1], L = 10)
  ffm = ffm(grid)
  print (ffm.shape) ### should be [grid.shape[0], 2*L*rep_dim]

def unit_test_fourier_net():
  import ops, utils

  operator = ops.ParallelBeamGeometryOp(64, 60, 500)
  grid_params = {
    'angles': torch.cos(torch.tensor(operator.angles, dtype=torch.float32)),
    'num_detectors': operator.num_detectors
    }
  grid = utils.get_sino_mgrid(**grid_params)

  fn = FourierNet(in_features=2, hidden_features=256, hidden_blocks=2, out_features=1)
  fn(grid)

# if __name__=='__main__':
  # unit_test_ffm_dimension()
  # unit_test_fourier_net()



        #  in_features,
        #  hidden_features,
        #  hidden_blocks,
        #  out_features,
        #  L = 10):
"""
in_features: dimension of input position
num_pos_enc: number of frequencies to use
large: boolean, use large or not large network
out_features: dimension of the output
"""
class EM_Simulator(torch.nn.Module):
    def __init__(self, in_features, num_pos_enc, large, out_features, features = 256):
        super(EM_Simulator, self).__init__()
        self.large = large
        
        channels = features
        input_pos_enc = num_pos_enc*2*in_features

        self.ffm = FFMLayer_v2(num_pos_enc)
        
        if(self.large):
            self.layers_1 = torch.nn.Sequential(
                *[
                    torch.nn.Linear(input_pos_enc, channels), 
                    torch.nn.ReLU(),
                    torch.nn.Linear(channels, channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(channels, channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(channels, channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(channels, channels),
                    torch.nn.ReLU(),
                ])    
            self.layers_2 = torch.nn.Sequential(
                *[
                    torch.nn.Linear(channels + input_pos_enc, channels), 
                    torch.nn.ReLU(),
                    torch.nn.Linear(channels, channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(channels, channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(channels, channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(channels, out_features),
                    # torch.nn.Sigmoid(),
                ])        
        else: 
            self.layers_1 = torch.nn.Sequential(
                *[
                    torch.nn.Linear(input_pos_enc, channels), 
                    torch.nn.ReLU(),
                    torch.nn.Linear(channels, channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(channels, channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(channels, out_features),
                    # torch.nn.Sigmoid(),
                ])       

    def forward(self, coords, frac=1):
        ffm_out = self.ffm(coords, frac)
        x = ffm_out

        # x = x[:,3:]
        y = self.layers_1(x)
        if(self.large):
            return self.layers_2(torch.cat((x, y), axis=-1))
        else:
            return y
        

# standard MLP with n hidden layers
class MLP(torch.nn.Module):
  def __init__(self, in_features, hidden_features, hidden_blocks, out_features):
    super(MLP, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.hidden_blocks = hidden_blocks
    self.hidden_features = hidden_features

    self.input_layer = torch.nn.Linear(in_features, hidden_features)
    self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_features, hidden_features) for i in range(hidden_blocks)])
    self.output_layer = torch.nn.Linear(hidden_features, out_features)

  def forward(self, x):
    x = self.input_layer(x)
    x = torch.nn.functional.relu(x)

    for i in range(self.hidden_blocks):
      x = self.hidden_layers[i](x)
      x = torch.nn.functional.relu(x)

    x = self.output_layer(x)

    return x



# MLP with n hidden layers with two possible outputs
class MLP_two_outputs(torch.nn.Module):
  def __init__(self, in_features, hidden_features, hidden_blocks, out_features,
                      hidden_features2, hidden_blocks2, out_features2):
    super(MLP, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.hidden_blocks = hidden_blocks
    self.hidden_features = hidden_features
    self.hidden_blocks2 = hidden_blocks2
    self.hidden_features2 = hidden_features2

    self.input_layer = torch.nn.Linear(in_features, hidden_features)
    self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_features, hidden_features) for i in range(hidden_blocks)])
    self.output_layer = torch.nn.Linear(hidden_features, out_features)

    self.hidden_layers2 = torch.nn.ModuleList([torch.nn.Linear(hidden_features2, hidden_features2) for i in range(hidden_blocks2)])
    self.output_layer2 = torch.nn.Linear(hidden_features2, out_features2)


  def forward(self, x):
    x = self.input_layer(x)
    x = torch.nn.functional.relu(x)

    for i in range(self.hidden_blocks):
      x = self.hidden_layers[i](x)
      x = torch.nn.functional.relu(x)

    x = self.output_layer(x)

    return x
  
  def forward2(self,x):
    x = self.input_layer(x)
    x = torch.nn.functional.relu(x)

    for i in range(self.hidden_blocks):
      x = self.hidden_layers[i](x)
      x = torch.nn.functional.relu(x)

    for i in range(self.hidden_blocks2):
      x = self.hidden_layers2[i](x)
      x = torch.nn.functional.relu(x)

    x = self.output_layer2(x)

    return x
