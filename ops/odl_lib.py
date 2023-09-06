"""Holds all the odl related utilities."""
import numpy as np
import matplotlib.pyplot as plt
import torch

import odl

# from odl.contrib.torch import OperatorFunction
from ops.ODLHelper import OperatorFunction

class OperatorAsAutogradFunction(object):
  """Dummy class around OperatorFunction.

  Check https://github.com/swing-research/dual_implicit_rep/issues/1 .
  """
  def __init__(self, odl_op):
    self.op = odl_op
  
  def __call__(self, x):
    return OperatorFunction.apply(self.op, x)
    

def apply_angle_noise(angles, noise):
  """Applies operator noise in the angles of the operator.

  SNR = 20 log (P_angles/P_noise)
  Args:
    angles (np.ndarray): 1D array of angles used in the Radon operator.
    noise (float): SNR of the noise.
  Returns:
    noisy_angles (np.ndarray): Noisy angles at `noise` dB.
  """
  if noise > 200:
    return angles

  noise_std_dev = 10**(-noise/20)

  noisy_angles = angles * (1 + np.random.randn(*angles.shape)*noise_std_dev)

  return noisy_angles


class ParallelBeamGeometryOp(object):
  """Creates an `img_size` mesh parallel geometry tomography operator."""

  def __init__(self, img_size, num_angles, op_snr=500):
    self.img_size = img_size
    self.num_angles = num_angles
    self.reco_space = odl.uniform_discr(
      min_pt=[-1, -1], 
      max_pt=[1, 1], 
      shape=[img_size, img_size],
      dtype='float32'
    )

    # self.detector_partition = odl.uniform_partition([-1], [1], [img_size])
    self.geometry = odl.tomo.parallel_beam_geometry(
      self.reco_space, num_angles, img_size)

    self.num_detectors = self.geometry.detector.size
    self.op_snr = op_snr
    self.angles = apply_angle_noise(self.geometry.angles, op_snr)

    self.optimizable_params = torch.tensor(
      self.angles, dtype=torch.float32)  # Convert to torch.Tensor.  

    self.op = odl.tomo.RayTransform(
      self.reco_space,
      self.geometry,
      impl='astra_cuda')

    self.fbp = odl.tomo.analytic.filtered_back_projection.fbp_op(self.op)

  def __call__(self, x):
    return OperatorFunction.apply(self.op, x)

  def pinv(self, y):
    return OperatorFunction.apply(self.fbp, y)

class ParallelBeamGeometryOpBroken(ParallelBeamGeometryOp):
  """Creates a noisy angle instance of ParallelBeamGeometryOp

    # Steps taken from implementation of odl.tomo.parallel_beam_geometry
    # https://github.com/odlgroup/odl/blob/master/odl/tomo/geometry/parallel.py#L1471

  Notes
  -----
  According to [NW2001]_, pages 72--74, a function
  :math:`f : \mathbb{R}^2 \to \mathbb{R}` that has compact support
  .. math::
      | x | > rho  implies f(x) = 0,
  and is essentially bandlimited
  .. math::
     | xi | > Omega implies hat{f}(xi) approx 0,
  can be fully reconstructed from a parallel beam ray transform
  if (1) the projection angles are sampled with a spacing of
  :math:`Delta psi` such that
  .. math::
      Delta psi leq frac{pi}{rho Omega},
  and (2) the detector is sampled with an interval :math:`Delta s`
  that satisfies
  .. math::
      Delta s leq frac{pi}{Omega}.
  The geometry returned by this function satisfies these conditions exactly.
  If the domain is 3-dimensional, the geometry is "separable", in that each
  slice along the z-dimension of the data is treated as independed 2d data.
  References
  ----------
  .. [NW2001] Natterer, F and Wuebbeling, F.
     *Mathematical Methods in Image Reconstruction*.
     SIAM, 2001.
     https://dx.doi.org/10.1137/1.9780898718324
  """
  def __init__(self, clean_operator, op_snr):
    super().__init__(clean_operator.img_size, clean_operator.num_angles, op_snr)

    space = self.reco_space

    # Find maximum distance from rotation axis
    corners = space.domain.corners()[:, :2]
    rho = np.max(np.linalg.norm(corners, axis=1))

    # Find default values according to Nyquist criterion.

    # We assume that the function is bandlimited by a wave along the x or y
    # axis. The highest frequency we can measure is then a standing wave with
    # period of twice the inter-node distance.
    min_side = min(space.partition.cell_sides[:2])
    omega = np.pi / min_side
    num_px_horiz = 2 * int(np.ceil(rho * omega / np.pi)) + 1
    det_min_pt = -rho
    det_max_pt = rho
    det_shape = num_px_horiz
    det_partition = odl.discr.uniform_partition(det_min_pt, det_max_pt, det_shape)

    self.angles = apply_angle_noise(clean_operator.geometry.angles, op_snr)

    self.optimizable_params = torch.tensor(clean_operator.geometry.angles, dtype=torch.float32)

    # angle partition is changed to not be uniform
    angle_partition = odl.discr.nonuniform_partition(np.sort(self.angles))

    self.geometry = odl.tomo.Parallel2dGeometry(angle_partition, det_partition)

    self.num_detectors = self.geometry.detector.size

    self.op = odl.tomo.RayTransform(
      self.reco_space,
      self.geometry,
      impl='astra_cuda')

    self.fbp = odl.tomo.analytic.filtered_back_projection.fbp_op(self.op)


class ParallelBeamGeometryOpNonUniform(ParallelBeamGeometryOp):
  """Creates a noisy angle instance of ParallelBeamGeometryOp

    # Steps taken from implementation of odl.tomo.parallel_beam_geometry
    # https://github.com/odlgroup/odl/blob/master/odl/tomo/geometry/parallel.py#L1471

  Notes
  -----
  According to [NW2001]_, pages 72--74, a function
  :math:`f : \mathbb{R}^2 \to \mathbb{R}` that has compact support
  .. math::
      | x | > rho  implies f(x) = 0,
  and is essentially bandlimited
  .. math::
     | xi | > Omega implies hat{f}(xi) approx 0,
  can be fully reconstructed from a parallel beam ray transform
  if (1) the projection angles are sampled with a spacing of
  :math:`Delta psi` such that
  .. math::
      Delta psi leq frac{pi}{rho Omega},
  and (2) the detector is sampled with an interval :math:`Delta s`
  that satisfies
  .. math::
      Delta s leq frac{pi}{Omega}.
  The geometry returned by this function satisfies these conditions exactly.
  If the domain is 3-dimensional, the geometry is "separable", in that each
  slice along the z-dimension of the data is treated as independed 2d data.
  References
  ----------
  .. [NW2001] Natterer, F and Wuebbeling, F.
     *Mathematical Methods in Image Reconstruction*.
     SIAM, 2001.
     https://dx.doi.org/10.1137/1.9780898718324
  """
  def __init__(self, clean_operator, nonuniform_angles):
    super().__init__(clean_operator.img_size, clean_operator.num_angles, clean_operator.op_snr)

    space = self.reco_space

    # Find maximum distance from rotation axis
    corners = space.domain.corners()[:, :2]
    rho = np.max(np.linalg.norm(corners, axis=1))

    # Find default values according to Nyquist criterion.

    # We assume that the function is bandlimited by a wave along the x or y
    # axis. The highest frequency we can measure is then a standing wave with
    # period of twice the inter-node distance.
    min_side = min(space.partition.cell_sides[:2])
    omega = np.pi / min_side
    num_px_horiz = 2 * int(np.ceil(rho * omega / np.pi)) + 1
    det_min_pt = -rho
    det_max_pt = rho
    det_shape = num_px_horiz
    det_partition = odl.discr.uniform_partition(det_min_pt, det_max_pt, self.geometry.det_grid.shape[0])

    del self.angles

    # angle partition is changed to not be uniform
    angle_partition = odl.discr.nonuniform_partition(np.sort(nonuniform_angles))

    self.geometry = odl.tomo.Parallel2dGeometry(angle_partition, det_partition)

    self.num_detectors = self.geometry.detector.size

    self.op = odl.tomo.RayTransform(
      self.reco_space,
      self.geometry,
      impl='astra_cuda')

    self.fbp = odl.tomo.analytic.filtered_back_projection.fbp_op(self.op)


def unit_test():
  op_64_32 = ParallelBeamGeometryOp(1024, 32)
  phantom = odl.phantom.shepp_logan(
    op_64_32.reco_space, modified=True)

  x = torch.from_numpy(phantom.data)
  y = op_64_32(x)

  print(y.shape)
  print(x.shape)

def unit_test_nonuniform():

  operator1 = ParallelBeamGeometryOp(64, 45)
  x = torch.from_numpy(odl.phantom.shepp_logan(operator1.reco_space, modified=True).data)
  y1 = operator1(x)

  operator2 = ParallelBeamGeometryOpNonUniform(operator1, operator1.geometry.angles)
  y2 = operator2(x)

  print(y1 - y2)

if __name__ == '__main__':
  unit_test()
  unit_test_nonuniform()
  