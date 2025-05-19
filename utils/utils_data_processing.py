import os
import time
import torch
import mrcfile
import numpy as np
from . import utils_deformation


def load_projections(config):
    t0 = time.time()
    base_directory = os.path.basename(config.path_volume)
    name_file = os.path.basename(config.path_volume).split('/')[-1]
    name_file = name_file.split('.')[0]
    projections_noisy = np.float32(mrcfile.open(config.path_volume,permissive=True).data)
    projections_noisy = projections_noisy/np.abs(projections_noisy).max()
    if config.projections_rotate:
        projections_noisy = np.rot90(np.flip(projections_noisy,axis=1), k=3, axes=((1, 2)))
    config.Nangles = projections_noisy.shape[0]
    projections_noisy = torch.Tensor(projections_noisy).type(config.torch_type).to(config.device)
    projections_noisy = projections_noisy/torch.abs(projections_noisy).max() # make sure that values to predict are between -1 and 1
    if config.debug:
        print("Elapsed time: {}".format(time.time() - t0))
    return projections_noisy, name_file


def load_angles(config, projections_noisy):
    Nangles_origin, n1_origin, n2_origin = projections_noisy.shape
    if os.path.isfile(config.path_angle):
        try:
            angles = []
            with open(config.path_angle) as f:
                for x in f:
                    angles.append(np.float32(x.split('\n')[0]))
            angles = np.array(angles)
        except:
            angles = np.linspace(config.view_angle_min, config.view_angle_max, config.Nangles)
    else:
        angles = np.linspace(config.view_angle_min, config.view_angle_max, config.Nangles)
    view_angle_min = angles.min()
    view_angle_max = angles.max()

    return angles, Nangles_origin, n1_origin, n2_origin, view_angle_min, view_angle_max

def select_volume_neural_network(config):
    t0 = time.time()
    if(config.volume_model=="multi-resolution"):
        import tinycudann as tcnn
        config_network = {"encoding": {
                'otype': config.implicit_volume.encoding.otype,
                'type': config.implicit_volume.encoding.type,
                'n_levels': config.implicit_volume.encoding.n_levels,
                'n_features_per_level': config.implicit_volume.encoding.n_features_per_level,
                'log2_hashmap_size': config.implicit_volume.encoding.log2_hashmap_size,
                'base_resolution': config.implicit_volume.encoding.base_resolution,
                'per_level_scale': config.implicit_volume.encoding.per_level_scale,
                'interpolation': config.implicit_volume.encoding.interpolation
            },
            "network": {
                "otype": config.implicit_volume.network.otype,
                "activation": config.implicit_volume.network.activation,
                "output_activation": config.implicit_volume.network.output_activation,
                "n_neurons": config.implicit_volume.hidden_size_volume,
                "n_hidden_layers": config.implicit_volume.num_layers_volume,
            }
            }
        impl_volume = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=1, encoding_config=config_network["encoding"],
                                                    network_config=config_network["network"]).to(config.device)

    num_param = sum(p.numel() for p in impl_volume.parameters() if p.requires_grad)
    print(f"---> Number of trainable parameters in volume net: {num_param}")

    if config.debug:
        print("Elapsed time: {}".format(time.time() - t0))
    return impl_volume

def select_local_deformation_neural_network(config):
    t0 = time.time()
    if config.local_model=='multi-resolution':
        import tinycudann as tcnn
        config_network = {"encoding": {
                'otype': config.local_deformation.encoding.otype,
                'type': config.local_deformation.encoding.type,
                'n_levels': config.local_deformation.encoding.n_levels,
                'n_features_per_level': config.local_deformation.encoding.n_features_per_level,
                'log2_hashmap_size': config.local_deformation.encoding.log2_hashmap_size,
                'base_resolution': config.local_deformation.encoding.base_resolution,
                'per_level_scale': config.local_deformation.encoding.per_level_scale,
                'interpolation': config.local_deformation.encoding.interpolation
            },
            "network": {
                "otype": config.local_deformation.network.otype,
                "activation": config.local_deformation.network.activation,
                "output_activation": config.local_deformation.network.output_activation,
                "n_neurons": config.local_deformation.hidden_size,
                "n_hidden_layers": config.local_deformation.num_layers,
            }
            }
        implicit_deformation_list = []
        for k in range(config.Nangles):
            implicit_deformation = tcnn.NetworkWithInputEncoding(n_input_dims=2,
                                                                n_output_dims=2,
                                                                encoding_config=config_network["encoding"],
                                                                network_config=config_network["network"]).to(config.device)
            implicit_deformation_list.append(implicit_deformation)

        num_param = sum(p.numel() for p in implicit_deformation_list[0].parameters() if p.requires_grad)
        print('---> Number of trainable parameters in implicit net: {}'.format(num_param))

    if config.local_model=='interpolation':
        depl_ctr_pts_net = torch.zeros(
            (2, config.local_deformation.N_ctrl_pts_net, config.local_deformation.N_ctrl_pts_net)).to(
            config.device).type(config.torch_type)
        implicit_deformation_list = []
        for k in range(config.Nangles):
            field = utils_deformation.deformation_field(depl_ctr_pts_net.clone(),maskBoundary=config.maskBoundary)
            implicit_deformation_list.append(field)
        num_param = sum(p.numel() for p in implicit_deformation_list[0].parameters() if p.requires_grad)
        print('---> Number of trainable parameters in implicit net: {}'.format(num_param))

    if config.debug:
        print("Elapsed time: {}".format(time.time() - t0))
    return implicit_deformation_list

