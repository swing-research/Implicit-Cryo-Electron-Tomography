import numpy as np
import torch

from absl import logging

from .odl_lib import ParallelBeamGeometryOp, ParallelBeamGeometryOpBroken
from .radon_3d_lib import ParallelBeamGeometry3DOp, ParallelBeamGeometry3DOpBroken
from .traveltime_lib import TravelTimeOperator

def get_operator_dict(config):
  if config.problem == 'radon':
    operator = ParallelBeamGeometryOp(
      config.img_size,
      config.op_param,
      op_snr=np.inf)

    operator_dict = {}

    if config.opt_strat == 'dip_noisy':
      broken_operator = ParallelBeamGeometryOpBroken(operator, config.op_snr)
      operator_dict.update({'original_operator': broken_operator})
      operator_dict.update({'noisy_operator': operator})
    elif config.opt_strat in ['dip', 'joint', 'joint_input']:
      dense_operator = ParallelBeamGeometryOp(
        config.img_size, config.dense_op_param, op_snr=np.inf)
      operator_dict.update({'dense_operator': dense_operator})
      broken_operator = ParallelBeamGeometryOpBroken(operator, config.op_snr)
      operator_dict.update({'original_operator': broken_operator})
    elif config.opt_strat == 'broken_machine':
      dense_operator = ParallelBeamGeometryOp(
        config.img_size,
        config.op_param,
        op_snr=np.inf)
      operator_dict = {'original_operator_clean': dense_operator}
      
      broken_operator = ParallelBeamGeometryOpBroken(dense_operator, config.op_snr)
      operator_dict.update({'original_operator': broken_operator})
  
      dense_operator = ParallelBeamGeometryOp(
        config.img_size,
        config.dense_op_param,
        op_snr=np.inf)
      operator_dict['dense_operator'] = dense_operator
      
      logging.info(f'operator_dict: {operator_dict}')

    else:
      raise ValueError(f'Did not recognize opt.strat={config.opt_strat}.')

  elif config.problem == 'radon_3d':
    operator_dict = {}

    if config.opt_strat == 'broken_machine':
      dense_operator = ParallelBeamGeometry3DOp(config.img_size, config.op_param, op_snr=np.inf)
      operator_dict = {'original_operator_clean': dense_operator}
      
      broken_operator = ParallelBeamGeometry3DOpBroken(dense_operator, config.op_snr)
      operator_dict.update({'original_operator': broken_operator})
  
      dense_operator = ParallelBeamGeometry3DOp(config.img_size, config.dense_op_param, op_snr=np.inf)
      operator_dict['dense_operator'] = dense_operator
      
      logging.info(f'operator_dict: {operator_dict}')

    else:
      raise ValueError(f'Did not recognize opt.strat={config.opt_strat}.')


  elif config.problem == 'traveltime':
    original_sensors = torch.rand(
      config.op_param, 2, dtype=torch.float32)

    operator = TravelTimeOperator(
      original_sensors,
      config.img_size)
    operator_dict = {'original_operator': operator}

    # Add new sensors for dense operator.
    new_sensors = torch.rand(
      config.dense_op_param - config.op_param, 2, dtype=torch.float32)
    sensors = torch.cat((original_sensors, new_sensors), dim=0)

    dense_operator = TravelTimeOperator(
      sensors, config.img_size)
    operator_dict.update({'dense_operator': dense_operator})

  else:
    raise ValueError('Inverse problem unrecognized.')

  return operator_dict