import torch
import numpy as np

from .siren_lib import Siren
from .fourier_net import FourierNet, FourierNetCheckpointed
# from NURBSDiff.nurbs_eval import SurfEval
from .nurbs_eval import SurfEval
from .nurbs_eval_3d import SurfEval3D

import sys
sys.path.append('..')
# import .unet_3d as Unet3D

def get_model_dict(config, y_siren_inputs, guess_recon, spline_info):
  """Initializes the appropriate model params given config.

  Args:
    config (ml_collections.ConfigDict): Trainer config.
    y_siren_inputs (torch.Tensor): Init values of y_siren_inputs
  """
  # Perhaps these params will also be part of config.

  if config.img_inn.network_type == 'siren':
    img_siren = Siren(
      in_features=config.img_inn.input_size,
      out_features=config.img_inn.output_size,
      hidden_features=config.img_inn.hidden_size,
      hidden_layers=config.img_inn.num_layers,
      first_omega_0=config.img_inn.omega_0,
      outermost_linear=True)

    model_dict={'img_siren': img_siren.cuda()}
  # elif config.img_inn.network_type == 'unet':
  #   if config.problem == 'radon_3d':
  #     unet = Unet3D.UNet(in_channels=1, 
  #       out_channels=1, 
  #       n_blocks=4, 
  #       start_filts=16,
  #       activation='relu',
  #       normalization='batch',
  #       conv_mode='same',
  #       dim=3)
  #     if config.real_sample:
  #       print ('Using CryoET model')
  #       # checkpoint = torch.load(f'unet_3d/models/cryoet_model_{config.unet_angles}angles_measurementsnr{int(config.measurement_snr)}dB_dynamic.pt')
  #       # checkpoint = torch.load(f'unet_3d/models/cryoet_correctpermute_model_{config.unet_angles}angles_measurementsnr{int(config.measurement_snr)}dB_dynamic.pt')
  #       # checkpoint = torch.load(f'unet_3d/models/cryoet_mixed_model_{config.unet_angles}angles_measurementsnr{int(config.measurement_snr)}dB_dynamic.pt')
  #       # checkpoint = torch.load(f'unet_3d/models/cryoem_model_{config.unet_angles}angles_measurementsnr{int(config.measurement_snr)}dB_dynamic.pt')
  #       checkpoint = torch.load(f'unet_3d/models/cryoem_correctnoise_model_{config.unet_angles}angles_measurementsnr{int(config.measurement_snr)}dB_dynamic.pt')
  #       # checkpoint = torch.load(f'unet_3d/models/model_{config.unet_angles}angles_measurementsnr{int(config.measurement_snr)}dB_dynamic.pt')
  #     else:
  #       print ('Using synthtetic phantom model')
  #       checkpoint = torch.load(f'unet_3d/models/model_{config.unet_angles}angles_measurementsnr{int(config.measurement_snr)}dB_dynamic.pt')
  #   else:
  #     unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
  #       in_channels=1, out_channels=1, init_features=16, pretrained=False)
  #     # checkpoint = torch.load(f'unet_training/models/model_{config.dense_op_param}angles.pt')
  #     # checkpoint = torch.load(f'/home/sid/implicit_reps/dual_implicit_rep/unet_training/models/model_{config.dense_op_param}angles_measurementsnr35dB_dynamic.pt')
  #     checkpoint = torch.load(f'unet_training/models/model_{config.unet_angles}angles_measurementsnr{int(config.measurement_snr)}dB_dynamic.pt')
  #   unet.load_state_dict(checkpoint['model_state_dict'])
  #   unet.eval()
  #   model_dict={'img_siren': unet.cuda()}
  else:
    raise ValueError('Invalid img network_type')

  if config.opt_strat == 'joint' or config.opt_strat == 'joint_input' or config.opt_strat == 'broken_machine':
    if config.measurement_inn.network_type == 'siren':
      measurement_siren = Siren(
        in_features=config.measurement_inn.input_size,
        out_features=config.measurement_inn.output_size,
        hidden_features=config.measurement_inn.hidden_size,
        hidden_layers=config.measurement_inn.num_layers,
        first_omega_0=config.measurement_inn.omega_0,
        outermost_linear=True)
    elif config.measurement_inn.network_type == 'fourier':
      measurement_siren = FourierNet(
        in_features=config.measurement_inn.input_size,
        out_features=config.measurement_inn.output_size,
        hidden_features=config.measurement_inn.hidden_size,
        hidden_blocks=config.measurement_inn.num_layers,
        L = config.measurement_inn.L)
    elif config.measurement_inn.network_type == 'fourier_checkpointed':
      measurement_siren = FourierNetCheckpointed(
        in_features=config.measurement_inn.input_size,
        out_features=config.measurement_inn.output_size,
        hidden_features=config.measurement_inn.hidden_size,
        L = config.measurement_inn.L)
    elif config.measurement_inn.network_type == 'nurbs':
      if config.problem == 'radon':
        _, num_ctrl_pts_v = spline_info['y_measured'].shape
        angles_spline_space = spline_info['angles_spline_space']
        angles_spline_space = np.concatenate(
          (angles_spline_space[-config.measurement_inn.deg_x:] - np.pi, 
            angles_spline_space, 
            angles_spline_space[:config.measurement_inn.deg_x] + np.pi), axis=-1)
        angles_spline_space += config.angles_shift
        angles_spline_space /= (np.pi + 2*config.angles_shift)
        detectors_spline_space = np.linspace(0, 1, num_ctrl_pts_v)
        X, Y = np.meshgrid(angles_spline_space, detectors_spline_space, indexing='ij')
        Z = spline_info['y_measured'].detach().cpu().numpy()
        Z = np.concatenate(
          (np.flip(Z[-config.measurement_inn.deg_x:], axis=-1), 
            Z, 
            np.flip(Z[:config.measurement_inn.deg_x], axis=-1)), axis=0)
        inp_ctrl_pts = torch.from_numpy(np.array([X,Y,Z])).permute(1,2,0).unsqueeze(0).contiguous()
        weights = torch.ones(1, len(angles_spline_space), num_ctrl_pts_v, 1)
        measurement_siren = SurfEval(
          inp_ctrl_pts,
          weights, 
          angles_spline_space,
          detectors_spline_space,
          config.measurement_inn.deg_x, 
          config.measurement_inn.deg_y)

        # Used in loss. Only measurements (Z) are actually used. Check if other dims are required.
        target = torch.FloatTensor(np.array([X,Y,Z])).permute(1,2,0).unsqueeze(0).cuda()
        spline_info.update({'spline_target': target})
      
      elif config.problem == 'radon_3d':
        num_ctrl_pts_u, num_ctrl_pts_v, num_ctrl_pts_w = spline_info['y_measured'].shape
        angles_spline_space = (spline_info['angles_spline_space'] + np.pi/3) / (2*np.pi / 3)
        detectors_v_spline_space = np.linspace(0, 1, num_ctrl_pts_v)
        detectors_w_spline_space = np.linspace(0, 1, num_ctrl_pts_w)
        W, X, Y = np.meshgrid(angles_spline_space, detectors_v_spline_space, detectors_w_spline_space, indexing='ij')
        Z = spline_info['y_measured'].detach().cpu().numpy()
        inp_ctrl_pts = torch.from_numpy(np.array([W,X,Y,Z])).permute(1,2,3,0).unsqueeze(0).contiguous()
        weights = torch.ones(1, num_ctrl_pts_u, num_ctrl_pts_v, num_ctrl_pts_w, 1)
        measurement_siren = SurfEval3D(
          inp_ctrl_pts,
          weights,
          angles_spline_space,
          detectors_v_spline_space,
          detectors_w_spline_space,
          config.measurement_inn.deg_x,
          config.measurement_inn.deg_y,
          config.measurement_inn.deg_z)

        # Used in loss. Only measurements (Z) are actually used. Check if other dims are required.
        target = torch.FloatTensor(np.array([W,X,Y,Z])).permute(1,2,3,0).unsqueeze(0).cuda()
        spline_info.update({'spline_target': target})

      else:
        raise ValueError('Invalid inverse problem')
    else:
      raise ValueError('Invalid measurement network_type')

    model_dict.update({'measurement_siren': measurement_siren.cuda()})

  if 'input' in config.opt_strat or config.opt_strat == 'broken_machine':
    y_siren_inputs.requires_grad_(True)
    if config.measurement_inn.network_type == 'nurbs':
      y_siren_inputs = model_dict['measurement_siren'].u_spline_space * np.pi
  else:
    y_siren_inputs.requires_grad_(False)

  model_dict.update({'input_parameters': y_siren_inputs})
  model_dict.update({'guess_recon': guess_recon})

  return model_dict