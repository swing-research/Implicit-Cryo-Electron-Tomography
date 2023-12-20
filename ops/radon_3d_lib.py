import odl
import torch
import numpy as np

from .odl_lib import apply_angle_noise
from .ODLHelper import OperatorFunction

class ParallelBeamGeometry3DOp(object):
  def __init__(self, img_size, num_angles, op_snr, angle_max=np.pi/3,fact=2):
    self.img_size = img_size
    self.num_angles = num_angles
    self.reco_space = odl.uniform_discr(
      min_pt=[-0.5, -0.5, -0.5],
      max_pt=[0.5, 0.5, 0.5],
      shape=[img_size, img_size, img_size],
      dtype='float32'
      )
      
    # Make a 3d single-axis parallel beam geometry with flat detector
    self.angle_partition = odl.uniform_partition(-angle_max, angle_max, num_angles)
    self.detector_partition = odl.uniform_partition([-0.5, -0.5], [0.5, 0.5], [fact*img_size,fact*img_size])
    self.geometry = odl.tomo.Parallel3dAxisGeometry(self.angle_partition, self.detector_partition)
    self.num_detectors_x, self.num_detectors_y = self.geometry.detector.shape
    self.angles = apply_angle_noise(self.geometry.angles, op_snr)
    self.optimizable_params = torch.tensor(self.angles, dtype=torch.float32)  # Convert to torch.Tensor.     
    self.op = odl.tomo.RayTransform(self.reco_space, self.geometry, impl='astra_cuda')
    self.fbp = odl.tomo.analytic.filtered_back_projection.fbp_op(self.op)

  def __call__(self, x):
    return OperatorFunction.apply(self.op, x)

  def pinv(self, y):
    return OperatorFunction.apply(self.fbp, y)

class ParallelBeamGeometry3DOpAngles_rectangular(ParallelBeamGeometry3DOp):
  def __init__(self, img_size, angles, fact=2):
    self.n1, self.n2, self.n3 = img_size
    self.num_angles = angles.shape[0]
    n = max(self.n1, self.n2, self.n3)
    self.reco_space = odl.uniform_discr(
      min_pt=[-self.n1/(n), -self.n2/(n), -self.n3/(n)],
      max_pt=[self.n1/(n), self.n2/(n), self.n3/(n)],
      shape=[self.n1, self.n2, self.n3],
      dtype='float32'
      )
    self.angles = angles
    # angle partition is changed to not be uniform
    self.angle_partition = odl.discr.nonuniform_partition(np.sort(self.angles))
    self.detector_partition = odl.uniform_partition([-self.n1/(n), -self.n2/(n)], [self.n1/(n), self.n2/(n)], [fact*self.n1,fact*self.n2])
    self.geometry = odl.tomo.Parallel3dAxisGeometry(self.angle_partition, self.detector_partition,
    axis=(1,0,0),det_axes_init=[(1, 0, 0), (0, 1, 0)],det_pos_init=(0,0,1))
    self.op = odl.tomo.RayTransform(self.reco_space, self.geometry, impl='astra_cuda')
    self.fbp = odl.tomo.analytic.filtered_back_projection.fbp_op(self.op)

  def __call__(self, x):
    return OperatorFunction.apply(self.op, x)

  def pinv(self, y):
    return OperatorFunction.apply(self.fbp, y)

def unit_test():
  import matplotlib.pyplot as plt

  img_size = 64
  num_angles = 60
  A = ParallelBeamGeometry3DOp(img_size, num_angles, np.inf)

  x = torch.rand([img_size, img_size, img_size])
  y = A(x)
  x_hat = A.pinv(y)
  print (x.shape)
  print (y.shape)
  print(x_hat.shape)

  # try non-square data
  from utils.utils_sampling import sample_implicit, grid_class_rectangular
  from utils import utils_interpolation
  torch_type=torch.float
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
  n1 = 256
  n2 = 256
  n3 = 64
  n = max(n1, n2, n3)
  x = torch.ones([n1,n2,n3])
  x /= x.sum()
  x_ = torch.zeros((n,n,n))
  x_[n//2-n1//2:n//2+n1//2,n//2-n2//2:n//2+n2//2,n//2-n3//2:n//2+n3//2] = x
  angles = np.linspace(-90,90,91)
  angles_t = torch.tensor(angles).type(torch_type).to(device)
  A = ParallelBeamGeometry3DOpAngles_rectangular((n1,n2,n3), angles/180*np.pi, fact=1)

  proj_clean = A(x)
  proj_clean /= proj_clean.sum((1,2)).mean()

  kk = 15# np.random.randint(num_angles)
  # impl_volume = lambda coord: utils_interpolation.interp_volume_rectangular(x,coord,n1,n2,n3)
  impl_volume = lambda coord: utils_interpolation.interp_volume_rectangular(x_,coord,n,n,n)
  grid_class_ = grid_class_rectangular(n,n,n,torch_type,device)
  proj_est = sample_implicit(impl_volume,grid_class_.grid3d_t,angles_t[kk],rot_deform=None,shift_deform=None,local_deform=None,scale=1.0).reshape(n,n,n)
  proj_est = proj_est.sum(2)
  proj_est = proj_est[n//2-n1//2:n//2+n1//2,n//2-n2//2:n//2+n2//2]

  plt.figure(1)
  plt.subplot(1,3,1)
  plt.imshow(proj_clean[kk].detach().cpu())
  plt.colorbar()
  plt.subplot(1,3,2)
  plt.imshow(proj_est.detach().cpu())
  plt.colorbar()
  plt.subplot(1,3,3)
  plt.imshow(torch.abs(proj_est.detach().cpu()-proj_clean[kk].detach().cpu()))
  plt.colorbar()
  plt.show()


if __name__ == "__main__":
  unit_test()