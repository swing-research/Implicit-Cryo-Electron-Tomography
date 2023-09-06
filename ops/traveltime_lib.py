import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-12

def sense_2(sensors, img):

  img_size = img.shape[0]

  # make a line
  x1 = sensors[0, 0]
  y1 = sensors[0, 1]

  x2 = sensors[1, 0]
  y2 = sensors[1, 1]

  slope = (y2 - y1) / (x2 - x1)

  per_pixel_width = 1.0/img_size
  n_pts_x = int(np.abs(x2-x1)/per_pixel_width)
  n_pts_y = int(np.abs(y2-y1)/per_pixel_width)


  intersect_vert = None
  intersect_horz = None

  if n_pts_x > 0:
    xs = x1 + np.arange(1, n_pts_x + 1) * per_pixel_width * np.sign(x2-x1)
    ys = y2 - slope * (x2 - xs)
    intersect_vert = np.stack((xs, ys), axis=-1)

  if n_pts_y > 0:
    ys = y1 + np.arange(1, n_pts_y + 1) * per_pixel_width * np.sign(y2-y1)
    xs = x2 - (y2 - ys) / slope
    intersect_horz = np.stack((xs, ys), axis=-1)

  all_pts = np.concatenate((sensors, intersect_horz, intersect_vert), axis=0)
  idx= np.argsort(all_pts, axis=0)[:, 0]
  all_pts = all_pts[idx]  # Sorts acc. to x coordinate.

  # find line segment length x pixels
  midpoints = (all_pts[:-1] + all_pts[1:])/2
  lengths = np.linalg.norm(all_pts[:-1] - all_pts[1:], axis=-1)

  # pick correct pixels x line segment length
  midpoints = np.clip((midpoints + 1) / 2.0, 0, 1-EPS)
  pixel_idx = np.floor(midpoints/per_pixel_width).astype(np.int32)
  pixel_intensities = img[pixel_idx[:, 0], pixel_idx[:, 1]]


  return np.sum(lengths * pixel_intensities)


class tt_2sensors(nn.Module):
  """Traveltime for 2 sensors"""
  def __init__(self, sensors, img_size):
    super(tt_2sensors, self).__init__()
    self.lengths, self.idx = self.__build(sensors, img_size)

  @staticmethod
  def __build(sensors, img_size):
    if isinstance(sensors, np.ndarray):
      sensors = torch.from_numpy(sensors.astype(np.float32)).cuda()
    
    # make a line
    x1 = sensors[0, 0]
    y1 = sensors[0, 1]

    x2 = sensors[1, 0]
    y2 = sensors[1, 1]

    slope = (y2 - y1) / (x2 - x1)

    per_pixel_width = 1.0/img_size
    n_pts_x = torch.abs(x2-x1)/per_pixel_width
    n_pts_x = n_pts_x.type(torch.int)
    n_pts_y = torch.abs(y2-y1)/per_pixel_width
    n_pts_y = n_pts_y.type(torch.int)


    intersect_vert = None
    intersect_horz = None

    if n_pts_x > 0:
      xs = x1 + torch.arange(
        1, n_pts_x + 1, device='cuda') * per_pixel_width * torch.sign(x2-x1)
      ys = y2 - slope * (x2 - xs)
      intersect_vert = torch.stack((xs, ys), dim=-1)

    if n_pts_y > 0:
      ys = y1 + torch.arange(
        1, n_pts_y + 1, device='cuda') * per_pixel_width * torch.sign(y2-y1)
      xs = x2 - (y2 - ys) / slope
      intersect_horz = torch.stack((xs, ys), dim=-1)

    all_pts = sensors.clone().cuda()
    if intersect_horz is not None:
      all_pts = torch.cat((sensors, intersect_horz), dim=0)
    if intersect_vert is not None:
      all_pts = torch.cat((all_pts, intersect_vert), dim=0)

    idx = torch.argsort(all_pts, dim=0)[:, 0]
    all_pts = all_pts[idx]  # Sorts acc. to x coordinate.

    # find line segment length x pixels
    midpoints = (all_pts[:-1] + all_pts[1:])/2
    lengths = torch.norm(all_pts[:-1] - all_pts[1:], dim=-1)

    # pick correct pixels x line segment length
    midpoints = torch.clip((midpoints + 1) / 2.0, 0, 1-EPS)
    pixel_idx = torch.floor(midpoints/per_pixel_width).type(
      torch.cuda.LongTensor)

    return lengths, pixel_idx


  def forward(self, img):
    pixel_intensities = img[self.idx[:,0], self.idx[:, 1]]
    return torch.sum(pixel_intensities * self.lengths)


class TravelTimeOperator(nn.Module):
  """Builds the linear traveltime tomography operator."""

  def __init__(self, sensors, img_size):
    """Initializes a linear TT operator.
    Args:
      sensors (torch.Tensor): Locations of sensors [-1, 1]
      img_size (int): Size of the image
    """
    super(TravelTimeOperator, self).__init__()
    self.sensors = sensors
    self.lengths, self.idx, self.nelems = self._build(sensors, img_size)
    self.optimizable_params = sensors


  @staticmethod
  def _build(sensors, img_size):
    N = len(sensors)
    lengths = []
    nelems = []
    idx = []

    for i in range(N-1):
      for j in range(i+1, N):
        sense2 = torch.stack((sensors[i], sensors[j]), axis=0)
        row = tt_2sensors(sense2, img_size)
        lengths.append(row.lengths)
        idx.append(row.idx)
        nelems.append(len(row.lengths))

    return torch.cat(lengths), torch.cat(idx, dim=0), nelems



  def forward(self, img):
    pixel_intensities = img[self.idx[:,0], self.idx[:, 1]]
    y_measured = pixel_intensities * self.lengths
    measured_chunks = torch.split(y_measured, self.nelems)
    y_measured = torch.stack([
      torch.sum(c) for c in measured_chunks]
      )

    return y_measured


def time_op():
  print('\n Timing the op')
  from time import time
  IMG_SIZE = 64
  NS = 50
  CVAL = 0.3
  img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float32) * CVAL

  sensors = np.random.rand(NS, 2)
  sensors = torch.tensor(sensors, requires_grad=True).cuda()

  A = TravelTimeOperator(sensors, IMG_SIZE).cuda()

  print('Operator built.')

  img = torch.from_numpy(img).cuda()
  t = time()
  for _ in range(500):  
    y = A(img)

  print(f'Time per forward: {(time()-t)/500}s')

def unit_test():
  print('\n Unit testing the op')
  from time import time
  IMG_SIZE = 1024
  NS = 50
  CVAL = 0.3
  img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float32) * CVAL

  sensors = np.random.rand(NS, 2)
  sensors = torch.tensor(sensors).cuda()
  sensors.requires_grad_(True)

  A = TravelTimeOperator(sensors, IMG_SIZE)
  print(f'A length is on {A.lengths.device}')
  print(f'A idx is on {A.idx.device}')
  
  y = A(torch.from_numpy(img).cuda())

  obj = torch.sum(y)
  obj.backward()

  y_t = y.cpu().detach().numpy()
  print(sensors.grad)

  
  gt = np.zeros(int(NS * (NS-1) / 2))
  c = 0
  for i in range(NS-1):
    for j in range(i+1, NS):
      gt[c] = torch.norm(sensors[i] - sensors[j]) * CVAL
      c += 1

  assert np.allclose(y_t, gt, atol=1e-5), f'Max abs error = {np.abs(y_t - gt).max()}.'


def unit_test_2sensor():
  """Unit testing the 2 sensor row.

  KNOWN BUG:
  - In the numpy version of this code, there is a chance 
  that two sensors are so close in either x or y that when computing
  the intersection points along x or y output can be None. This causes
  failure but since this happens rarely it is not fixed.
  """

  print('\n Unit testing each row of the op')

  IMG_SIZE = 64
  img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float32) * 0.3
  # img = np.random.rand(IMG_SIZE, IMG_SIZE)

  # sensors = np.array([[-1.0, -1.0], [1.0, 1.0]])
  sensors = np.random.rand(2, 2).astype(np.float32)
  sensors_t = torch.from_numpy(sensors).cuda()
  sensors_t.requires_grad_(True)
  A = tt_2sensors(sensors_t, IMG_SIZE).cuda()

  print(f'tt_2sensors length is on {A.lengths.device}')
  print(f'tt_2sensors idx is on {A.idx.device}')

  y = A(torch.from_numpy(img).cuda())

  y.backward()
  print(sensors_t.grad)

  y_t = y.cpu().detach().numpy()
 
  y_np = sense_2(sensors, img).astype(np.float32)
  # gt = np.diag(img).sum()/32.0*np.sqrt(2)
  gt = np.linalg.norm(sensors[0] - sensors[1]) * 0.3

  assert np.abs(y_t - y_np) < 1e-6, f'Got {y_t}, expected {y_np}.'

  assert np.isclose(y_t, gt), f'Got {y_t}, expected {gt}.'


if __name__ == '__main__':
  # unit_test_2sensor()
  # unit_test()
  time_op()


