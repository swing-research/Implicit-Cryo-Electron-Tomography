from skimage.transform import resize

import numpy as np
import torch
import matplotlib.pyplot as plt
plt.ion()
import mrcfile
from ops.radon_3d_lib import ParallelBeamGeometry3DOpAngles_rectangular
import os
import imageio
from utils import utils_deformation, utils_display, utils_FSC ,utils_sampling
import shutil
import pandas as pd
from configs.config_reconstruct_simulation import get_default_configs, get_areTomoValidation_configs,get_config_local_implicit
from configs.config_reconstruct_simulation import get_volume_save_configs


from matplotlib import gridspec
from scipy.interpolate import griddata


import pandas as pd
from utils.utils_deformation import cropper



# # TODO: Remove
# import configs.config_shrec_dataset as config_file
# config = config_file.get_config()


""" 
Remove deformation from a tilt-series.
"""
def correct_deformations(projections, shifts, inplane_rotations, config):
    Nangles, n1, n2 = projections.shape
    projections_corrected = torch.zeros_like(projections)
    xx1 = torch.linspace(-1,1,n1,dtype=config.torch_type,device=config.device)
    xx2 = torch.linspace(-1,1,n2,dtype=config.torch_type,device=config.device)
    XX_t, YY_t = torch.meshgrid(xx1,xx2,indexing='ij')
    XX_t = torch.unsqueeze(XX_t, dim = 2)
    YY_t = torch.unsqueeze(YY_t, dim = 2)
    for i in range(Nangles):
        coordinates = torch.cat([XX_t,YY_t],2).reshape(-1,2)
        thetas = inplane_rotations[i]
        rot_deform = torch.stack(
                        [torch.stack([torch.cos(thetas),torch.sin(thetas)],0),
                        torch.stack([-torch.sin(thetas),torch.cos(thetas)],0)]
                        ,0)
        coordinates = coordinates -shifts[i]
        coordinates = torch.transpose(torch.matmul(rot_deform,torch.transpose(coordinates,0,1)),0,1) 
        x = projections[i].clone().view(1,1,n1,n2)
        x = x.expand(n1*n2, -1, -1, -1)
        out = cropper(x,coordinates,output_size = 1).reshape(n1,n2)
        projections_corrected[i] = out
    return projections_corrected

def extract_angle(rot_matrix):
    """
    Extract the angle from a rotation matrix in degrees
    """
    if rot_matrix[0,0] == 1:
        angle = 0
    elif rot_matrix[0,0] == -1:
        angle = 180
    else:
        angle = np.arctan2(rot_matrix[1,0], rot_matrix[0,0])*180/np.pi
    return angle


import SimpleITK as sitk
def perform_3d_registration(fixed_image, moving_image):
    # Create an instance of the ImageRegistrationMethod class
    registration_method = sitk.ImageRegistrationMethod()

    # Set the metric (e.g., mutual information)
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    # Set the optimizer (e.g., gradient descent)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)

    # Set the interpolator (e.g., linear interpolation)
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Create an affine transformation (rotation and scaling)
    affine_transform = sitk.AffineTransform(fixed_image.GetDimension())

    # Create a translation transform
    translation_transform = sitk.TranslationTransform(fixed_image.GetDimension())

    # Create a composite transform
    composite_transform = sitk.CompositeTransform([affine_transform, translation_transform])

    # Set the initial transform
    registration_method.SetInitialTransform(composite_transform, inPlace=False)

    # Set the scale parameters for the optimizer (translation, rotation, scaling)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Perform the registration
    final_transform = registration_method.Execute(fixed_image, moving_image)

    return final_transform


def reconstruct_FBP_volume(config, tiltseries):
    """
    Args:
        config : 
        tiltseries (torch tensor): volume
    """
    # Define the forward operator
    angles = np.linspace(config.view_angle_min,config.view_angle_max,config.Nangles)
    operator_ET = ParallelBeamGeometry3DOpAngles_rectangular((config.n1,config.n2,config.n3), angles/180*np.pi, fact=1)

    # Reconstruct the volume    
    V_FBP = operator_ET.pinv(tiltseries).detach().requires_grad_(False)

    return V_FBP

def CC(V1,V2):
    V1_norm = np.sqrt(np.sum(((V1-V1.mean()))**2))
    V2_norm = np.sqrt(np.sum(((V2-V2.mean()))**2))
    return np.sum((V1-V1.mean())*(V2-V2.mean()))/(V1_norm*V2_norm)

def compare_results(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    if torch.cuda.device_count()>1:
        torch.cuda.set_device(config.device_num)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    config.device = device

    # Parent Dircetorys 
    if not os.path.exists(config.path_save):
        os.makedirs(config.path_save)
    if not os.path.exists(config.path_save+"/evaluation/"):
        os.makedirs(config.path_save+"/evaluation/")
    if not os.path.exists(config.path_save+"/evaluation/projections/"):
        os.makedirs(config.path_save+"/evaluation/projections/")
    if not os.path.exists(config.path_save+"/evaluation/volumes/"):
        os.makedirs(config.path_save+"/evaluation/volumes/")
    if not os.path.exists(config.path_save+"/evaluation/deformations/"):
        os.makedirs(config.path_save+"/evaluation/deformations/")
    if not os.path.exists(config.path_save+"/evaluation/volume_slices/"):
        os.makedirs(config.path_save+"/evaluation/volume_slices/")

    # Our method
    if not os.path.exists(config.path_save+"/evaluation/projections/ICETIDE/"):
        os.makedirs(config.path_save+"/evaluation/projections/ICETIDE/")
    if not os.path.exists(config.path_save+"/evaluation/volumes/ICETIDE/"):
        os.makedirs(config.path_save+"/evaluation/volumes/ICETIDE/")
    if not os.path.exists(config.path_save+"/evaluation/deformations/ICETIDE/"):
        os.makedirs(config.path_save+"/evaluation/deformations/ICETIDE/")
    if not os.path.exists(config.path_save+"/evaluation/volume_slices/ICETIDE/"):
        os.makedirs(config.path_save+"/evaluation/volume_slices/ICETIDE/")

    # AreTomos method
    if not os.path.exists(config.path_save+"/evaluation/projections/AreTomo/"):
        os.makedirs(config.path_save+"/evaluation/projections/AreTomo/")
    if not os.path.exists(config.path_save+"/evaluation/volumes/AreTomo/"):
        os.makedirs(config.path_save+"/evaluation/volumes/AreTomo/")
    if not os.path.exists(config.path_save+"/evaluation/deformations/AreTomo/"):
        os.makedirs(config.path_save+"/evaluation/deformations/AreTomo/")
    if not os.path.exists(config.path_save+"/evaluation/volume_slices/AreTomo/"):
        os.makedirs(config.path_save+"/evaluation/volume_slices/AreTomo/")

    # Etomo method
    if not os.path.exists(config.path_save+"/evaluation/projections/Etomo/"):
        os.makedirs(config.path_save+"/evaluation/projections/Etomo/")   
    if not os.path.exists(config.path_save+"/evaluation/volumes/Etomo/"):
        os.makedirs(config.path_save+"/evaluation/volumes/Etomo/")
    if not os.path.exists(config.path_save+"/evaluation/deformations/Etomo/"):
        os.makedirs(config.path_save+"/evaluation/deformations/Etomo/")
    if not os.path.exists(config.path_save+"/evaluation/volume_slices/Etomo/"):
        os.makedirs(config.path_save+"/evaluation/volume_slices/Etomo/")

    # True volume
    if not os.path.exists(config.path_save+"/evaluation/deformations/true/"):
        os.makedirs(config.path_save+"/evaluation/deformations/true/")
    if not os.path.exists(config.path_save+"/evaluation/volume_slices/true/"):
        os.makedirs(config.path_save+"/evaluation/volume_slices/true/")
    if not os.path.exists(config.path_save+"/evaluation/volumes/true/"):
        os.makedirs(config.path_save+"/evaluation/volumes/true/")

    # FBP on undistorted projections
    if not os.path.exists(config.path_save+"/evaluation/projections/FBP_no_deformed/"):
        os.makedirs(config.path_save+"/evaluation/projections/FBP_no_deformed/")
    if not os.path.exists(config.path_save+"/evaluation/volumes/FBP_no_deformed/"):
        os.makedirs(config.path_save+"/evaluation/volumes/FBP_no_deformed/")
    if not os.path.exists(config.path_save+"/evaluation/volume_slices/FBP_no_deformed/"):
        os.makedirs(config.path_save+"/evaluation/volume_slices/FBP_no_deformed/")

    # FBP
    if not os.path.exists(config.path_save+"/evaluation/projections/FBP/"):
        os.makedirs(config.path_save+"/evaluation/projections/FBP/")
    if not os.path.exists(config.path_save+"/evaluation/volumes/FBP/"):
        os.makedirs(config.path_save+"/evaluation/volumes/FBP/")
    if not os.path.exists(config.path_save+"/evaluation/volume_slices/FBP/"):
        os.makedirs(config.path_save+"/evaluation/volume_slices/FBP/")

    #FBP ICETIDE deformation estimations
    if not os.path.exists(config.path_save+"/evaluation/projections/FBP_ICETIDE/"):
        os.makedirs(config.path_save+"/evaluation/projections/FBP_ICETIDE/")
    if not os.path.exists(config.path_save+"/evaluation/volumes/FBP_ICETIDE/"):
        os.makedirs(config.path_save+"/evaluation/volumes/FBP_ICETIDE/")
    if not os.path.exists(config.path_save+"/evaluation/volume_slices/FBP_ICETIDE/"):
        os.makedirs(config.path_save+"/evaluation/volume_slices/FBP_ICETIDE/")



    ######################################################################################################
    ## Load data
    ######################################################################################################
    data = np.load(config.path_save_data+"volume_and_projections.npz")
    projections_noisy = torch.tensor(data['projections_noisy']).type(config.torch_type).to(device)
    affine_tr = np.load(config.path_save_data+"global_deformations.npy",allow_pickle=True)
    local_tr = np.load(config.path_save_data+"local_deformations.npy", allow_pickle=True)
    V_t = torch.tensor(np.moveaxis(np.double(mrcfile.open(config.path_save_data+"V.mrc").data),0,2)).type(config.torch_type).to(device)
    V_FBP_no_deformed_t = torch.tensor(np.moveaxis(np.double(mrcfile.open(config.path_save_data+"V_FBP_no_deformed.mrc").data),0,2)).type(config.torch_type).to(device)
    V_FBP_t =  torch.tensor(np.moveaxis(np.double(mrcfile.open(config.path_save_data+"V_FBP.mrc").data),0,2)).type(config.torch_type).to(device)
    # numpy
    V = V_t.detach().cpu().numpy()
    V_FBP = V_FBP_t.detach().cpu().numpy()
    V_FBP_no_deformed = V_FBP_no_deformed_t.detach().cpu().numpy()    

    ## Aretomo
    # get the files
    eval_AreTomo = False
    fsc_AreTomo_list = []
    fsc_AreTomo_centered_list = []
    CC_AreTomo_list = []
    CC_AreTomo_centered_list = []
    for npatch in config.nPatch:
        ARE_TOMO_FILE = f'projections_aligned_aretomo_{npatch}by{npatch}.mrc'
        path_file = os.path.join(config.path_save,'AreTomo',ARE_TOMO_FILE)
        shift_aretomo = np.zeros((config.Nangles,2))
        translation_matrix_np = np.zeros((2))       
        inplane_rotation_aretomo = np.zeros(config.Nangles) 
        if os.path.isfile(path_file):
            eval_AreTomo = True
            # load projections # need to be reorder and there seem to be a shift in intensity, so we match the mean on the noisy one 
            proj_aligned_aretomo = np.moveaxis(np.float32(mrcfile.open(path_file,permissive=True).data),1,0)
            proj_aligned_aretomo = np.swapaxes(proj_aligned_aretomo,1,2)
            proj_aligned_aretomo = np.rot90(proj_aligned_aretomo,axes=(1,2))*1
            proj_aligned_aretomo = proj_aligned_aretomo + (projections_noisy.detach().cpu().numpy().mean()-proj_aligned_aretomo.mean())

            # Reconstruct with accurate FBP operator
            V_FBP_aretomo = reconstruct_FBP_volume(config, torch.tensor(proj_aligned_aretomo).to(device)).detach().cpu().numpy()
            V_FBP_aretomo /= np.linalg.norm(V_FBP_aretomo)

            # Find best affine transformation between volumes
            V_sk = sitk.GetImageFromArray(V/np.linalg.norm(V))
            V_aretomo_sk = sitk.GetImageFromArray(V_FBP_aretomo)
            final_transform = perform_3d_registration(V_sk, V_aretomo_sk)
            # Apply the final transform to the moving image
            registered_image = sitk.Resample(V_aretomo_sk, V_sk, final_transform, sitk.sitkLinear, 0.0, V_aretomo_sk.GetPixelID())
            V_aretomo_centered = sitk.GetArrayFromImage(registered_image)

            # Get deformation matrix
            num_transforms = final_transform.GetNumberOfTransforms()
            composite_transform = final_transform.GetNthTransform(num_transforms - 1)
            affine_transform = composite_transform.GetNthTransform(0)
            translation_transform = composite_transform.GetNthTransform(1)
            affine_matrix = affine_transform.GetMatrix()
            affine_matrix_np = np.array(affine_matrix).reshape((V_sk.GetDimension(), V_sk.GetDimension()))
            translation_matrix = translation_transform.GetParameters()
            translation_matrix_np = np.array(translation_matrix)
            cos_theta = (np.trace(affine_matrix_np[:3, :3]) - 1) / 2
            rotation_angles_est = np.arccos(cos_theta) * 180 / np.pi  # Convert to degrees

            # Save volumes
            out = mrcfile.new(config.path_save_data+f"V_aretomo_centered{npatch}by{npatch}.mrc",np.moveaxis(V_aretomo_centered.astype(np.float32),2,0),overwrite=True)
            out.close() 
            out = mrcfile.new(config.path_save_data+f"V_aretomo_{npatch}by{npatch}_corrected.mrc",np.moveaxis(V_FBP_aretomo.astype(np.float32),2,0),overwrite=True)
            out.close() 

            # Compute fsc and CC
            fsc_AreTomo = utils_FSC.FSC(V,V_FBP_aretomo)
            fsc_AreTomo_centered = utils_FSC.FSC(V,V_aretomo_centered)
            fsc_AreTomo_list.append(fsc_AreTomo)
            fsc_AreTomo_centered_list.append(fsc_AreTomo_centered)

            CC_AreTomo = CC(V,V_FBP_aretomo)
            CC_AreTomo_centered = CC(V,V_aretomo_centered)
            CC_AreTomo_list.append(CC_AreTomo)
            CC_AreTomo_centered_list.append(CC_AreTomo_centered)

            # load estimated deformations
            ARETOMO_FILENAME = f'projections_{npatch}by{npatch}.aln'
            with open(os.path.join(config.path_save,'AreTomo',ARETOMO_FILENAME), 'r',encoding="utf-8") as file:
                lines = file.readlines()
            comments = []
            affine_transforms_aretomo = []
            local_transforms_aretomo = []
            affine_flag = True
            num_patches = 0
            for line in lines:
                if line.startswith('#'):
                    comments.append(line.strip())  # Strip removes leading/trailing whitespace
                    if line.startswith('# Local Alignment'):
                        affine_flag = False
                    if line.startswith('# NumPatches'):
                        num_patches = int(line.split('=')[-1])
                else:
                    if affine_flag:
                        affine_transforms_aretomo.append(line.strip().split())
                    else:
                        local_transforms_aretomo.append(line.strip().split())  
            aretomo_data = pd.DataFrame(affine_transforms_aretomo, 
                                        columns=["SEC", "ROT", "GMAG", "TX", 
                                                                    "TY", "SMEAN", "SFIT", "SCALE",
                                                                    "BASE", "TILT"])
            x_shifts_aretomo = aretomo_data['TX'].values.astype(np.float32)
            y_shifts_aretomo = aretomo_data['TY'].values.astype(np.float32)
            inplane_rotation_aretomo = aretomo_data['ROT'].values.astype(np.float32)
            # The estimates are already in pixels
            local_AreTomo = np.zeros((config.Nangles,num_patches,4))
            for local_est in local_transforms_aretomo:
                angle_index =int(local_est[0])
                patch_index = int(local_est[1])
                local_AreTomo[angle_index,patch_index,0] = float(local_est[2])
                local_AreTomo[angle_index,patch_index,1] = float(local_est[3])
                local_AreTomo[angle_index,patch_index,2] = float(local_est[4])
            shift_aretomo[:,1] = -x_shifts_aretomo
            shift_aretomo[:,0] = -y_shifts_aretomo
            # # correct the mean shitfs
            # shift_aretomo[:,0] = shift_aretomo[:,0] - np.mean(shift_aretomo[:,0])
            # shift_aretomo[:,1] = shift_aretomo[:,1] - np.mean(shift_aretomo[:,1])
            shift_aretomo_t = torch.from_numpy(shift_aretomo).to(device).type(config.torch_type)/config.n1*2
            inplane_rotation_aretomo_t = -torch.from_numpy(inplane_rotation_aretomo).to(device).type(config.torch_type)*np.pi/180
            projections_aretomo_corrected_python = correct_deformations(projections_noisy, shift_aretomo_t, inplane_rotation_aretomo_t, config)

            # Local def
            x_deformation = np.linspace(-config.n1//2,config.n1//2,config.N_ctrl_pts_local_def[0])
            y_deformation = np.linspace(-config.n2//2,config.n2//2,config.N_ctrl_pts_local_def[1])
            xx_deformation, yy_deformation = np.meshgrid(x_deformation,y_deformation,indexing='ij')
            xx_deformation =xx_deformation[:,:,None]
            yy_deformation = yy_deformation[:,:,None]
            grid_interp = np.concatenate([xx_deformation,yy_deformation],2).reshape(-1,2)

            implicit_deformation_AreTomo = []
            for k in range(config.Nangles):
                if num_patches != 0:
                    values_x = griddata(local_AreTomo[k][:,:2],local_AreTomo[k][:,2],grid_interp,method='cubic',fill_value=0,rescale=True)
                    values_y = griddata(local_AreTomo[k][:,:2],local_AreTomo[k][:,3],grid_interp,method='cubic',fill_value=0,rescale=True)
                    depl_ctr_pts_net = np.concatenate([values_x[None],values_y[None]],0).reshape(2,config.N_ctrl_pts_local_def[0],config.N_ctrl_pts_local_def[1])
                    depl_ctr_pts_net = torch.tensor(depl_ctr_pts_net/config.n1).to(device).type(config.torch_type)
                else:
                    depl_ctr_pts_net = torch.zeros(2,config.N_ctrl_pts_local_def[0],config.N_ctrl_pts_local_def[1]).to(device).type(config.torch_type)
                field = utils_deformation.deformation_field(depl_ctr_pts_net.clone())
                implicit_deformation_AreTomo.append(field)



    ETOMO_FILE = 'projections_etomo_ali.mrc'
    path_file = os.path.join(config.path_save,'projections_etomo',ETOMO_FILE)
    shift_etomo = np.zeros((config.Nangles,2))
    inplane_rotation_etomo = np.zeros(config.Nangles)
    eval_Etomo = False
    if os.path.isfile(path_file):
        eval_Etomo = True
        etomo_projections = np.double(mrcfile.open(path_file).data)
        etomo_projections_t = torch.tensor(etomo_projections).type(config.torch_type).to(device)
        V_FBP_etomo = reconstruct_FBP_volume(config, etomo_projections_t).detach().cpu().numpy()
        out = mrcfile.new(config.path_save_data+"V_etomo.mrc",np.moveaxis(V_FBP_etomo.astype(np.float32),2,0),overwrite=True)
        out.close() 

        # Find best affine transformation between volumes
        V_sk = sitk.GetImageFromArray(V/np.linalg.norm(V))
        V_etomo_sk = sitk.GetImageFromArray(V_FBP_etomo)
        final_transform = perform_3d_registration(V_sk, V_etomo_sk)
        # Apply the final transform to the moving image
        registered_image = sitk.Resample(V_etomo_sk, V_sk, final_transform, sitk.sitkLinear, 0.0, V_etomo_sk.GetPixelID())
        V_etomo_centered = sitk.GetArrayFromImage(registered_image)

        # Get deformation matrix
        num_transforms = final_transform.GetNumberOfTransforms()
        composite_transform = final_transform.GetNthTransform(num_transforms - 1)
        affine_transform = composite_transform.GetNthTransform(0)
        translation_transform = composite_transform.GetNthTransform(1)
        affine_matrix = affine_transform.GetMatrix()
        affine_matrix_np = np.array(affine_matrix).reshape((V_sk.GetDimension(), V_sk.GetDimension()))
        translation_matrix = translation_transform.GetParameters()
        translation_matrix_np = np.array(translation_matrix)
        cos_theta = (np.trace(affine_matrix_np[:3, :3]) - 1) / 2
        rotation_angles_est = np.arccos(cos_theta) * 180 / np.pi  # Convert to degrees


        # Extract the estimated deformations for etomo
        ETOMO_FILENAME = 'projections_etomo.xf'
        path_file_etomo = os.path.join(config.path_save,'projections_etomo',ETOMO_FILENAME)
        etomodata = pd.read_csv(path_file_etomo, sep='\s+', header=None)
        etomodata.columns = ['a11', 'a12', 'a21', 'a22','y','x']
        x_shifts_etomo = etomodata['x'].values
        y_shifts_etomo = etomodata['y'].values

        shift_etomo[:,1] = x_shifts_etomo
        shift_etomo[:,0] = y_shifts_etomo
        shift_etomo[:,0] = shift_etomo[:,0] - np.mean(shift_etomo[:,0])
        shift_etomo[:,1] = shift_etomo[:,1] - np.mean(shift_etomo[:,1])

        for index in range(len(etomodata)):
            etomo_rotation_matrix = np.array([[etomodata['a11'][index],etomodata['a12'][index]],
                                [etomodata['a21'][index],etomodata['a22'][index]]])
            inplane_rotation_etomo[index] = extract_angle(etomo_rotation_matrix)
        print("Loaded Etomo")


    ######################################################################################################
    ## Load and estimate our volume
    ######################################################################################################
    ## Load implicit network
    if(config.volume_model=="Fourier-features"):
        from models.fourier_net import FourierNet,FourierNet_Features
        impl_volume = FourierNet_Features(
            in_features=config.input_size_volume,
            sub_features=config.sub_features,
            out_features=config.output_size_volume, 
            hidden_features=config.hidden_size_volume,
            hidden_blocks=config.num_layers_volume,
            L = config.L_volume).to(device)

    if(config.volume_model=="MLP"):
        from models.fourier_net import MLP
        impl_volume = MLP(in_features= 1, 
                            hidden_features=config.hidden_size_volume, hidden_blocks= config.num_layers_volume, out_features=config.output_size_volume).to(device)

    if(config.volume_model=="multi-resolution"):  
        import tinycudann as tcnn
        config_network = {"encoding": {
                'otype': config.encoding.otype,
                'type': config.encoding.type,
                'n_levels': config.encoding.n_levels,
                'n_features_per_level': config.encoding.n_features_per_level,
                'log2_hashmap_size': config.encoding.log2_hashmap_size,
                'base_resolution': config.encoding.base_resolution,
                'per_level_scale': config.encoding.per_level_scale,
                'interpolation': config.encoding.interpolation,
            },
            "network": {
                "otype": config.network.otype,   
                "activation": config.network.activation,       
                "output_activation": config.network.output_activation,
                "n_neurons": config.hidden_size_volume,           
                "n_hidden_layers": config.num_layers_volume    
            }       
            }
        impl_volume = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=1, encoding_config=config_network["encoding"],
                                                    network_config=config_network["network"]).to(device)
    num_param = sum(p.numel() for p in impl_volume.parameters() if p.requires_grad) 
    print('---> Number of trainable parameters in volume net: {}'.format(num_param))
    checkpoint = torch.load(os.path.join(config.path_save,'training','model_trained.pt'),map_location=device)
    impl_volume.load_state_dict(checkpoint['implicit_volume'])
    shift_icetide = checkpoint['shift_est']
    rot_icetide = checkpoint['rot_est']
    implicit_deformation_icetide = checkpoint['local_deformation_network']
    ## Compute our model at same resolution than other volume
    rays_scaling = torch.tensor(np.array(config.rays_scaling))[None,None,None].type(config.torch_type).to(device)
    n1_eval, n2_eval, n3_eval = V.shape

    # Compute estimated volume
    with torch.no_grad():
        x_lin1 = np.linspace(-1,1,n1_eval)*rays_scaling[0,0,0,0].item()/2+0.5
        x_lin2 = np.linspace(-1,1,n2_eval)*rays_scaling[0,0,0,1].item()/2+0.5
        XX, YY = np.meshgrid(x_lin1,x_lin2,indexing='ij')
        grid2d = np.concatenate([XX.reshape(-1,1),YY.reshape(-1,1)],1)
        grid2d_t = torch.tensor(grid2d).type(config.torch_type)
        z_range = np.linspace(-1,1,n3_eval)*rays_scaling[0,0,0,2].item()*(n3_eval/n1_eval)/2+0.5
        V_icetide = np.zeros_like(V)
        for zz, zval in enumerate(z_range):
            grid3d = np.concatenate([grid2d_t, zval*torch.ones((grid2d_t.shape[0],1))],1)
            grid3d_slice = torch.tensor(grid3d).type(config.torch_type).to(device)
            estSlice = impl_volume(grid3d_slice).detach().cpu().numpy().reshape(config.n1,config.n2)
            V_icetide[:,:,zz] = estSlice
        V_icetide_t = torch.tensor(V_icetide).type(config.torch_type).to(device)

    ######################################################################################################
    # Using only the deformation estimates
    ######################################################################################################
    projections_noisy_undeformed = torch.zeros_like(projections_noisy)
    xx1 = torch.linspace(-1,1,config.n1,dtype=config.torch_type,device=device)
    xx2 = torch.linspace(-1,1,config.n2,dtype=config.torch_type,device=device)
    XX_t, YY_t = torch.meshgrid(xx1,xx2,indexing='ij')
    XX_t = torch.unsqueeze(XX_t, dim = 2)
    YY_t = torch.unsqueeze(YY_t, dim = 2)
    for i in range(config.Nangles):
        coordinates = torch.cat([XX_t,YY_t],2).reshape(-1,2)
        #field = utils_deformation.deformation_field(-implicit_deformation_icetide[i].depl_ctr_pts[0].detach().clone())
        thetas = torch.tensor(-rot_icetide[i].thetas.item()).to(device)
    
        rot_deform = torch.stack(
                        [torch.stack([torch.cos(thetas),torch.sin(thetas)],0),
                        torch.stack([-torch.sin(thetas),torch.cos(thetas)],0)]
                        ,0)
        coordinates = coordinates - config.deformationScale*implicit_deformation_icetide[i](coordinates)
        coordinates = coordinates - shift_icetide[i].shifts_arr
        coordinates = torch.transpose(torch.matmul(rot_deform,torch.transpose(coordinates,0,1)),0,1) ## do rotation
        x = projections_noisy[i].clone().view(1,1,config.n1,config.n2)
        x = x.expand(config.n1*config.n2, -1, -1, -1)
        out = cropper(x,coordinates,output_size = 1).reshape(config.n1,config.n2)
        projections_noisy_undeformed[i] = out
    V_FBP_icetide = reconstruct_FBP_volume(config, projections_noisy_undeformed).detach().cpu().numpy()



    #######################################################################################
    ## Save Fourier of volumes
    #######################################################################################
    index = n2_eval//2
    # True volume
    tmp = np.abs(np.fft.fft3(V))[:,index,:]
    tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
    tmp = np.floor(255*tmp).astype(np.uint8)
    imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","true_Fourier_XZ.png"),tmp)

    # ICETIDE
    tmp = np.abs(np.fft.fft3(V_icetide))[:,index,:]
    tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
    tmp = np.floor(255*tmp).astype(np.uint8)
    imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","ICETIDE_Fourier_XZ.png"),tmp)

    # FBP
    tmp = np.abs(np.fft.fft3(V_FBP))[:,index,:]
    tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
    tmp = np.floor(255*tmp).astype(np.uint8)
    imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","FBP_Fourier_XZ.png"),tmp)

    # FBP no deformed
    tmp = np.abs(np.fft.fft3(V_FBP_no_deformed))[:,index,:]
    tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
    tmp = np.floor(255*tmp).astype(np.uint8)
    imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","FBP_no_deformed_Fourier_XZ.png"),tmp)

    if(eval_AreTomo):
        tmp = np.abs(np.fft.fft3(V_aretomo_centered))[:,index,:]
        tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
        tmp = np.floor(255*tmp).astype(np.uint8)
        imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","AreTomo_Fourier_XZ.png"),tmp)

    if(eval_Etomo):
        tmp = np.abs(np.fft.fft3(V_etomo_centered))[:,index,:]
        tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
        tmp = np.floor(255*tmp).astype(np.uint8)
        imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","Etomo_Fourier_XZ.png"),tmp)

    # FBP icetide
    tmp = np.abs(np.fft.fft3(V_FBP_icetide))[:,index,:]
    tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
    tmp = np.floor(255*tmp).astype(np.uint8)
    imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","FBP_ICETIDE_Fourier_XZ.png"),tmp) 

def tmp():



    #######################################################################################
    ## Compare the error between the true and estimated deformation
    #######################################################################################
    x_shifts = np.zeros(config.Nangles)
    y_shifts = np.zeros(config.Nangles)
    inplane_rotation = np.zeros(config.Nangles)

    # Extract the true deformations
    for index ,affine_transform in enumerate(affine_tr):
        x_shifts[index] = affine_transform.shiftX.item()
        y_shifts[index] = affine_transform.shiftY.item()
        inplane_rotation[index] = affine_transform.angle.item()*180/np.pi

    #Extract the estimated deformations for the network
    x_shifts_icetide = np.zeros(config.Nangles)
    y_shifts_icetide = np.zeros(config.Nangles)
    inplane_rotation_icetide = np.zeros(config.Nangles)
    for index, (shift_net, rot_net) in enumerate(zip(shift_icetide,rot_icetide)):
        x_shifts_icetide[index] = shift_net.shifts_arr[0,0].item()
        y_shifts_icetide[index] = shift_net.shifts_arr[0,1].item()
        inplane_rotation_icetide[index] = rot_net.thetas.item()*180/np.pi

    # Compute the error between the true and estimated deformation
    error_x_shifts_icetide = np.around(np.abs(x_shifts-x_shifts_icetide).mean()*n1_eval,decimals=4)
    error_y_shifts_icetide = np.around(np.abs(y_shifts-y_shifts_icetide).mean()*n1_eval,decimals=4)
    error_inplane_rotation_icetide =np.around( np.abs(inplane_rotation-inplane_rotation_icetide
                                                ).mean(),decimals=4)

    # Compute the error between the true and AreTomo estimated deformation
    error_x_shifts_aretomo = np.around(np.abs(x_shifts*n1_eval-shift_aretomo[:,0]).mean(),decimals=4)
    error_y_shifts_aretomo = np.around(np.abs(y_shifts*n1_eval-shift_aretomo[:,1]).mean(),decimals=4)
    error_inplane_rotation_aretomo =np.around( np.abs(inplane_rotation-inplane_rotation_aretomo
                                                ).mean(),decimals=4)

    # Compute the error between the true and Etomo estimated deformation
    error_x_shifts_etomo = np.around(np.abs(x_shifts-shift_etomo[:,0]).mean()*n1_eval,decimals=4)
    error_y_shifts_etomo = np.around(np.abs(y_shifts-shift_etomo[:,1]).mean()*n1_eval,decimals=4)
    error_inplane_rotation_etomo = np.around(np.abs(inplane_rotation-inplane_rotation_etomo
                                                ).mean(),decimals=4)


    # Save the avg errors in a csv file with rownames: icetide, AreTomo, Etomo
    error_arr = pd.DataFrame(columns=['Method','x_shifts','y_shifts','inplane_rotation'])
    # Include the avg absolute error in pixels in the table

    x_mean_shift = np.around(np.abs(x_shifts).mean()*n1_eval,decimals=4)
    y_mean_shift = np.around(np.abs(y_shifts).mean()*n1_eval,decimals=4)
    inplane_mean_rotation = np.around(np.abs(inplane_rotation).mean(),decimals=4)
    error_arr.loc[0] = ['Observation',x_mean_shift,y_mean_shift,inplane_mean_rotation]
    error_arr.loc[1] = ['icetide',error_x_shifts_icetide,error_y_shifts_icetide,error_inplane_rotation_icetide]
    error_arr.loc[2] = ['aretomo',error_x_shifts_aretomo,error_y_shifts_aretomo,error_inplane_rotation_aretomo]
    error_arr.loc[3] = ['etomo',error_x_shifts_etomo,error_y_shifts_etomo,error_inplane_rotation_etomo]
    
    error_arr.to_csv(os.path.join(config.path_save,'evaluation'+'/error_rigid_deformations.csv'),index=False)


    #######################################################################################
    ## Compute FSC
    #######################################################################################
    fsc_icetide = utils_FSC.FSC(V,V_icetide)
    fsc_FBP_icetide = utils_FSC.FSC(V,V_FBP_icetide)
    fsc_FBP = utils_FSC.FSC(V,V_FBP)
    fsc_FBP_no_deformed = utils_FSC.FSC(V,V_FBP_no_deformed)
    if(eval_Etomo):
        fsc_Etomo = utils_FSC.FSC(V,V_etomo_centered)
    x_fsc = np.arange(fsc_FBP.shape[0])

    plt.figure(1)
    plt.clf()
    plt.plot(x_fsc,fsc_icetide,'b',label="icetide")
    plt.plot(x_fsc,fsc_FBP_icetide,'--b',label="FBP with our deform. est. ")
    if(eval_AreTomo):
        for i, npatch in enumerate(config.nPatch):
            col = ['r','m']
            plt.plot(x_fsc,fsc_AreTomo_list[i],col[i],label=f"AreTomo patch {npatch}")
            plt.plot(x_fsc,fsc_AreTomo_centered_list[i],col[i],linestyle='--',label=f"AreTomo centered patch {npatch}")
    if(eval_Etomo):
        plt.plot(x_fsc,fsc_Etomo,'c',label="Etomo")
    plt.plot(x_fsc,fsc_FBP,'k',label="FBP")
    plt.plot(x_fsc,fsc_FBP_no_deformed,'g',label="FBP no def.")
    plt.legend()
    plt.savefig(os.path.join(config.path_save,'evaluation','FSC.png'))
    plt.savefig(os.path.join(config.path_save,'evaluation','FSC.pdf'))


    fsc_arr = np.zeros((x_fsc.shape[0],8))
    fsc_arr[:,0] = x_fsc
    fsc_arr[:,1] = fsc_icetide[:,0]
    fsc_arr[:,2] = fsc_FBP[:,0]
    fsc_arr[:,3] = fsc_FBP_no_deformed[:,0]
    if(eval_AreTomo):
        for i, npatch in enumerate(config.nPatch):
            if i==0:
                fsc_arr[:,4] = fsc_AreTomo_centered_list[i][:,0] 
            if i==1:
                fsc_arr[:,7] = fsc_AreTomo_centered_list[i][:,0] 
    if(eval_Etomo):
        fsc_arr[:,5] = fsc_Etomo[:,0]
    fsc_arr[:,6] = fsc_FBP_icetide[:,0]
    # fsc_arr[:,6] = fsc_icetide_isonet[:,0]
    header ='x,icetide,FBP,FBP_no_deformed,AreTomo_patch0,ETOMO,FBP_est_deformed,AreTomo_patch1'
    np.savetxt(os.path.join(config.path_save,'evaluation','FSC.csv'),fsc_arr,header=header,delimiter=",",comments='')


    #######################################################################################
    ## Compute Correlation Coefficient
    #######################################################################################
    CC_icetide = CC(V,V_icetide)
    CC_FBP_icetide = CC(V,V_FBP_icetide)
    CC_FBP = CC(V,V_FBP)
    CC_FBP_no_deformed = CC(V,V_FBP_no_deformed)
    if(eval_Etomo):
        CC_Etomo = CC(V,V_etomo_centered)

    # plt.figure(1)
    # plt.clf()
    # plt.plot(x_fsc,CC_icetide,'b',label="icetide")
    # plt.plot(x_fsc,fsc_FBP_icetide,'--b',label="FBP with our deform. est. ")
    # if(eval_AreTomo):
    #     for i, npatch in enumerate(config.nPatch):
    #         col = ['r','m']
    #         plt.plot(x_fsc,fsc_AreTomo_list[i],col[i],label=f"AreTomo patch {npatch}")
    #         plt.plot(x_fsc,fsc_AreTomo_centered_list[i],col[i],linestyle='--',label=f"AreTomo centered patch {npatch}")
    # if(eval_Etomo):
    #     plt.plot(x_fsc,fsc_Etomo,'c',label="Etomo")
    # plt.plot(x_fsc,fsc_FBP,'k',label="FBP")
    # plt.plot(x_fsc,fsc_FBP_no_deformed,'g',label="FBP no def.")
    # plt.legend()
    # plt.savefig(os.path.join(config.path_save,'evaluation','FSC.png'))
    # plt.savefig(os.path.join(config.path_save,'evaluation','FSC.pdf'))

    CC_arr = np.zeros((1,8))
    CC_arr[:,1] = CC_icetide
    CC_arr[:,2] = CC_FBP
    CC_arr[:,3] = CC_FBP_no_deformed
    if(eval_AreTomo):
        for i, npatch in enumerate(config.nPatch):
            if i==0:
                CC_arr[:,4] = CC_AreTomo_centered_list[i]
            if i==1:
                CC_arr[:,7] = CC_AreTomo_centered_list[i]
    if(eval_Etomo):
        CC_arr[:,5] = CC_Etomo
    CC_arr[:,6] = CC_FBP_icetide
    # CC_arr[:,6] = CC_icetide_isonet
    header ='x,icetide,FBP,FBP_no_deformed,AreTomo_patch0,ETOMO,FBP_est_deformed,AreTomo_patch1'
    np.savetxt(os.path.join(config.path_save,'evaluation','CC.csv'),CC_arr,header=header,delimiter=",",comments='')



    ## Compute error between local deformations


    # if eval_AreTomo:
    #     # Extract the estimated deformations for AreTomo

    #     ARETOMO_FILENAME = 'projections.aln'
    #     with open(config.path_save+ARETOMO_FILENAME, 'r',encoding="utf-8") as file:
    #         lines = file.readlines()

    #     comments = []
    #     affine_transforms_aretomo = []
    #     local_transforms_aretomo = []

    #     affine_flag = True
    #     num_patches = 0
    #     for line in lines:
    #         if line.startswith('#'):
    #             comments.append(line.strip())  # Strip removes leading/trailing whitespace
    #             if line.startswith('# Local Alignment'):
    #                 affine_flag = False
    #             if line.startswith('# NumPatches'):
    #                 num_patches = int(line.split('=')[-1])
    #         else:
    #             if affine_flag:
    #                 affine_transforms_aretomo.append(line.strip().split())
    #             else:
    #                 local_transforms_aretomo.append(line.strip().split())  

    #     aretomo_data = pd.DataFrame(affine_transforms_aretomo, 
    #                                 columns=["SEC", "ROT", "GMAG", "TX", 
    #                                                         "TY", "SMEAN", "SFIT", "SCALE",
    #                                                             "BASE", "TILT"])
        
    #     x_shifts_aretomo = aretomo_data['TY'].values.astype(np.float32)
    #     y_shifts_aretomo = aretomo_data['TX'].values.astype(np.float32)
    #     inplane_rotation_aretomo = aretomo_data['ROT'].values.astype(np.float32)

    #     # The estimates are already in pixels
    #     error_x_shifts_aretomo = np.around(np.abs(x_shifts*n1_eval/2-x_shifts_aretomo).mean(),decimals=4)
    #     error_y_shifts_aretomo = np.around(np.abs(y_shifts*n1_eval/2-y_shifts_aretomo).mean(),decimals=4)
    #     error_inplane_rotation_aretomo = np.around(np.abs(inplane_rotation-
    #                                                     inplane_rotation_aretomo).mean(),decimals=4)
        
    #     error_arr.loc[3] = ['AreTomo',error_x_shifts_aretomo,error_y_shifts_aretomo,
    #                         error_inplane_rotation_aretomo]
        

        
    #     local_AreTomo = np.zeros((config.Nangles,num_patches,4))

    #     for local_est in local_transforms_aretomo:
    #         angle_index =int(local_est[0])
    #         patch_index = int(local_est[1])

    #         local_AreTomo[angle_index,patch_index,0] = float(local_est[2])
    #         local_AreTomo[angle_index,patch_index,1] = float(local_est[3])
    #         local_AreTomo[angle_index,patch_index,2] = float(local_est[4])
    #         local_AreTomo[angle_index,patch_index,3] = float(local_est[5])

    # # save the error in a csv file
    # error_arr.to_csv(os.path.join(config.path_save,'evaluation'+'/affine_error.csv'),index=False)

    #######################################################################################
    ## Save Fourier of volumes
    #######################################################################################
    index = n2_eval//2
    # True volume
    tmp = np.abs(np.fft.fft3(V))[:,index,:]
    tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
    tmp = np.floor(255*tmp).astype(np.uint8)
    imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","true_Fourier_XZ.png"),tmp)

    # ICETIDE
    tmp = np.abs(np.fft.fft3(V_icetide))[:,index,:]
    tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
    tmp = np.floor(255*tmp).astype(np.uint8)
    imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","ICETIDE_Fourier_XZ.png"),tmp)

    # FBP
    tmp = np.abs(np.fft.fft3(V_FBP))[:,index,:]
    tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
    tmp = np.floor(255*tmp).astype(np.uint8)
    imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","FBP_Fourier_XZ.png"),tmp)

    # FBP no deformed
    tmp = np.abs(np.fft.fft3(V_FBP_no_deformed))[:,index,:]
    tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
    tmp = np.floor(255*tmp).astype(np.uint8)
    imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","FBP_no_deformed_Fourier_XZ.png"),tmp)

    if(eval_AreTomo):
        tmp = np.abs(np.fft.fft3(V_aretomo_centered))[:,index,:]
        tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
        tmp = np.floor(255*tmp).astype(np.uint8)
        imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","AreTomo_Fourier_XZ.png"),tmp)

    if(eval_Etomo):
        tmp = np.abs(np.fft.fft3(V_etomo_centered))[:,index,:]
        tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
        tmp = np.floor(255*tmp).astype(np.uint8)
        imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","Etomo_Fourier_XZ.png"),tmp)

    # FBP icetide
    tmp = np.abs(np.fft.fft3(V_FBP_icetide))[:,index,:]
    tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
    tmp = np.floor(255*tmp).astype(np.uint8)
    imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","FBP_ICETIDE_Fourier_XZ.png"),tmp) 




    #######################################################################################
    ## Save slices of volumes
    #######################################################################################

    saveIndex = [n3_eval//4,n3_eval//2,int(3*n3_eval//4)] # The slices to save taken from previous plots
    for index in saveIndex:
        # True volume
        tmp = V[:,:,index]
        tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
        tmp = np.floor(255*tmp).astype(np.uint8)
        imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","true","slice_{}.png".format(index)),tmp)

        # ICETIDE
        tmp = V_icetide[:,:,index]
        tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
        tmp = np.floor(255*tmp).astype(np.uint8)
        imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","ICETIDE","slice_{}.png".format(index)),tmp)

        # FBP
        tmp = V_FBP[:,:,index]
        tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
        tmp = np.floor(255*tmp).astype(np.uint8)
        imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","FBP","slice_{}.png".format(index)),tmp)

        # FBP no deformed
        tmp = V_FBP_no_deformed[:,:,index]
        tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
        tmp = np.floor(255*tmp).astype(np.uint8)
        imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","FBP_no_deformed","slice_{}.png".format(index)),tmp)

        if(eval_AreTomo):
            tmp = V_aretomo_centered[:,:,index]
            tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
            tmp = np.floor(255*tmp).astype(np.uint8)
            imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","AreTomo","slice_{}.png".format(index)),tmp)

        if(eval_Etomo):
            tmp = V_etomo_centered[:,:,index]
            tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
            tmp = np.floor(255*tmp).astype(np.uint8)
            imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","Etomo","slice_{}.png".format(index)),tmp)

        # FBP icetide
        tmp = V_FBP_icetide[:,:,index]
        tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
        tmp = np.floor(255*tmp).astype(np.uint8)
        imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","FBP_ICETIDE","slice_{}.png".format(index)),tmp)


    #######################################################################################
    ## Save volumes
    #######################################################################################

    def display_XYZ(tmp,name="true"):
        f , aa = plt.subplots(2, 2, gridspec_kw={'height_ratios': [tmp.shape[2]/tmp.shape[0], 1], 'width_ratios': [1,tmp.shape[2]/tmp.shape[0]]})
        aa[0,0].imshow(tmp.mean(0).T,cmap='gray')
        aa[0,0].axis('off')
        aa[1,0].imshow(tmp.mean(2),cmap='gray')
        aa[1,0].axis('off')
        aa[1,1].imshow(tmp.mean(1),cmap='gray')
        aa[1,1].axis('off')
        aa[0,1].axis('off')
        plt.tight_layout(pad=1, w_pad=-1, h_pad=1)
        plt.savefig(os.path.join(config.path_save_data,'evaluation',"volumes",name,"XYZ.png"))

    # True volume
    tmp = V
    display_XYZ(tmp,name="true")

    # ICETIDE
    tmp = V_icetide
    display_XYZ(tmp,name="ICETIDE")

    # FBP volume
    tmp = V_FBP
    display_XYZ(tmp,name="FBP")

    # FBP_no_deformed volume
    tmp = V_FBP_no_deformed
    display_XYZ(tmp,name="FBP_no_deformed")

    if(eval_AreTomo):
        # AreTomo volume
        tmp = V_aretomo_centered
        display_XYZ(tmp,name="AreTomo")

    if(eval_Etomo):
        # Etomo volume
        tmp = V_etomo_centered
        display_XYZ(tmp,name="Etomo")

    # FBP_ICETIDE volume
    tmp = V_FBP_icetide
    display_XYZ(tmp,name="FBP_ICETIDE")


    #######################################################################################
    ## Generate projections
    #######################################################################################
    # Define angles and X-ray transform
    angles = np.linspace(config.view_angle_min,config.view_angle_max,config.Nangles)
    operator_ET = ParallelBeamGeometry3DOpAngles_rectangular((config.n1,config.n2,config.n3), angles/180*np.pi, fact=1)

    projections_icetide = operator_ET(V_icetide_t).detach().cpu().numpy()
    projections_FBP = operator_ET(V_FBP_t).detach().cpu().numpy()
    projections_FBP_no_deformed = operator_ET(V_FBP_no_deformed_t).detach().cpu().numpy()
    projections_FBP_icetide = projections_noisy_undeformed.detach().cpu().numpy()
    if(eval_AreTomo):
        V_FBP_aretomo_t = torch.tensor(V_aretomo_centered).to(device)
        projections_AreTomo = operator_ET(V_FBP_aretomo_t).detach().cpu().numpy()
    if(eval_Etomo):
        V_FBP_etomo_t = torch.tensor(V_etomo_centered).to(device)
        projections_Etomo = etomo_projections

    out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","icetide_projections.mrc"),projections_icetide.astype(np.float32),overwrite=True)
    out.close()
    if(eval_AreTomo):
        out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","AreTomo_projections.mrc"),projections_AreTomo.astype(np.float32),overwrite=True)
        out.close()
    if(eval_Etomo):
        out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","Etomo_projections.mrc"),projections_Etomo.astype(np.float32),overwrite=True)
        out.close()

    out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","FBP_projections.mrc"),projections_FBP.astype(np.float32),overwrite=True)
    out.close()
    out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","FBP_no_deformed_projections.mrc"),projections_FBP_no_deformed.astype(np.float32),overwrite=True)
    out.close()
    out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","FBP_icetide_projections.mrc"),projections_FBP_icetide.astype(np.float32),overwrite=True)
    out.close()
    
    for k in range(config.Nangles):
        tmp = projections_icetide[k]
        tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
        tmp = np.floor(255*tmp).astype(np.uint8)
        imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"projections","ICETIDE","snapshot_{}.png".format(k)),tmp)
        if(eval_AreTomo):
            tmp = projections_AreTomo[k]
            tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
            tmp = np.floor(255*tmp).astype(np.uint8)
            imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"projections","AreTomo","snapshot_{}.png".format(k)),tmp)
        if(eval_Etomo):
            tmp = projections_Etomo[k]
            tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
            tmp = np.floor(255*tmp).astype(np.uint8)
            imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"projections","Etomo","snapshot_{}.png".format(k)),tmp)
        tmp = projections_FBP[k]
        tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
        tmp = np.floor(255*tmp).astype(np.uint8)
        imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"projections","FBP","snapshot_{}.png".format(k)),tmp)
        tmp = projections_FBP_no_deformed[k]
        tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
        tmp = np.floor(255*tmp).astype(np.uint8)
        imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"projections","FBP_no_deformed","snapshot_{}.png".format(k)),tmp)
        tmp = projections_FBP_icetide[k]
        tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
        tmp = np.floor(255*tmp).astype(np.uint8)
        imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"projections","FBP_ICETIDE","snapshot_{}.png".format(k)),tmp)

    out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',
                                "volumes","ICETIDE_volume.mrc"),np.moveaxis(V_icetide,2,0),overwrite=True)
    out.close()
    if eval_AreTomo:
        out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',
                                    "volumes","AreTomo_volume.mrc"),np.moveaxis(V_FBP_aretomo,2,0),overwrite=True)
        out.close()
    if eval_Etomo:
        out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',
                                    "volumes","Etomo_volume.mrc"),np.moveaxis(V_etomo_centered,2,0),overwrite=True)
        out.close()
    out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"volumes",
                                "FBP_volume.mrc"),np.moveaxis(V_FBP,2,0),overwrite=True)
    out.close()
    out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"volumes",
                                "FBP_no_deformed_volume.mrc"),np.moveaxis(V_FBP_no_deformed,2,0),overwrite=True)
    out.close()
    out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"volumes",
                                "FBP_icetide_volume.mrc"),np.moveaxis(V_FBP_icetide,2,0),overwrite=True)

    # ## Saving the inplance angles 
    # inplaneAngles = np.zeros((config.Nangles,5))
    # inplaneAngles[:,0] = angles
    # inplaneAngles[:,1] = inplane_rotation
    # inplaneAngles[:,2] = inplane_rotation_icetide
    # if eval_AreTomo:
    #     inplaneAngles[:,3] = inplane_rotation_aretomo
    # if eval_ETOMO:
    #     inplaneAngles[:,4] = inplane_rotation_etomo


    # # save as a csv file
    # header ='angles,true,icetide,AreTomo,Etomo'
    # np.savetxt(os.path.join(config.path_save,'evaluation','inplane_angles.csv'),inplaneAngles,header=header,delimiter=",",comments='')


    #######################################################################################
    ## Local deformation errror Estimation
    #######################################################################################
    x_lin1 = np.linspace(-1,1,n1_eval)*rays_scaling[0,0,0,0].item()/2+0.5
    x_lin2 = np.linspace(-1,1,n2_eval)*rays_scaling[0,0,0,1].item()/2+0.5
    XX, YY = np.meshgrid(x_lin1,x_lin2,indexing='ij')
    grid2d = np.concatenate([XX.reshape(-1,1),YY.reshape(-1,1)],1)
    grid2d_t = torch.tensor(grid2d).type(config.torch_type)
    err_local_icetide = np.zeros(config.Nangles)
    err_local_init = np.zeros(config.Nangles)
    err_local_AreTomo = np.zeros(config.Nangles)

    for k in range(config.Nangles):
        # Error in icetide
        grid_correction_true = local_tr[k](grid2d_t).detach().cpu().numpy()
        grid_correction_est_icetide = config.deformationScale*implicit_deformation_icetide[k](
            grid2d_t).detach().cpu().numpy()
        tmp = np.abs(grid_correction_true-grid_correction_est_icetide)
        err_local_icetide[k] = (0.5*config.n1*tmp[:,0]+0.5*config.n2*tmp[:,1]).mean()
        # Finidng the magnitude for init
        tmp = np.abs(grid_correction_true)
        err_local_init[k] = (0.5*config.n1*tmp[:,0]+0.5*config.n2*tmp[:,1]).mean()
        # Finding the error for AreTomo
        if eval_AreTomo:
            grid_correction_est_AreTomo = implicit_deformation_AreTomo[k](
                grid2d_t).detach().cpu().numpy()
            tmp = np.abs(grid_correction_true-grid_correction_est_AreTomo)
            err_local_AreTomo[k] = (0.5*config.n1*tmp[:,0]+0.5*config.n2*tmp[:,1]).mean()
        else: 
            err_local_AreTomo[k] = np.nan


    # Save the error in a csv file
    err_local_arr = np.zeros((config.Nangles,4))
    err_local_arr[:,0] = angles
    err_local_arr[:,1] = err_local_icetide
    err_local_arr[:,2] = err_local_init
    err_local_arr[:,3] = err_local_AreTomo

    err_mean = np.nanmean(err_local_arr[:,1:],0)
    err_std = np.nanstd(err_local_arr[:,1:],0)
    err_local_arr = np.concatenate([np.array([err_mean,err_std])],0)

    HEADER ='icetide,init,AreTomo'
    np.savetxt(os.path.join(config.path_save,'evaluation','local_deformation_error.csv'),err_local_arr,header=HEADER,delimiter=",",comments='')

    # Get the local deformation error plots 
    for index in range(config.Nangles):
        # icetide
        savepath = os.path.join(config.path_save,'evaluation','deformations','ICETIDE','local_deformation_factor10_error_{}'.format(index))
        utils_display.display_local_est_and_true(implicit_deformation_icetide[index],local_tr[index],Npts=(20,20),scale=0.1, img_path=savepath)
        # Aretomo
        if eval_AreTomo:
            savepath = os.path.join(config.path_save,'evaluation','deformations','AreTomo','local_deformation_factor10_error_{}'.format(index))
            utils_display.display_local_est_and_true(implicit_deformation_AreTomo[index],local_tr[index],Npts=(20,20),scale=0.1, img_path=savepath )

    plt.close('all')



def compare_results_real(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    if torch.cuda.device_count()>1:
        torch.cuda.set_device(config.device_num)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    config.device = device

    # Parent Dircetorys 
    if not os.path.exists(config.path_save):
        os.makedirs(config.path_save)
    if not os.path.exists(config.path_save+"/evaluation/"):
        os.makedirs(config.path_save+"/evaluation/")
    if not os.path.exists(config.path_save+"/evaluation/projections/"):
        os.makedirs(config.path_save+"/evaluation/projections/")
    if not os.path.exists(config.path_save+"/evaluation/volumes/"):
        os.makedirs(config.path_save+"/evaluation/volumes/")
    if not os.path.exists(config.path_save+"/evaluation/deformations/"):
        os.makedirs(config.path_save+"/evaluation/deformations/")
    if not os.path.exists(config.path_save+"/evaluation/volume_slices/"):
        os.makedirs(config.path_save+"/evaluation/volume_slices/")

    # Our method
    if not os.path.exists(config.path_save+"/evaluation/projections/ICETIDE/"):
        os.makedirs(config.path_save+"/evaluation/projections/ICETIDE/")
    if not os.path.exists(config.path_save+"/evaluation/volumes/ICETIDE/"):
        os.makedirs(config.path_save+"/evaluation/volumes/ICETIDE/")
    if not os.path.exists(config.path_save+"/evaluation/deformations/ICETIDE/"):
        os.makedirs(config.path_save+"/evaluation/deformations/ICETIDE/")
    if not os.path.exists(config.path_save+"/evaluation/volume_slices/ICETIDE/"):
        os.makedirs(config.path_save+"/evaluation/volume_slices/ICETIDE/")

    # AreTomos method
    if not os.path.exists(config.path_save+"/evaluation/projections/AreTomo/"):
        os.makedirs(config.path_save+"/evaluation/projections/AreTomo/")
    if not os.path.exists(config.path_save+"/evaluation/volumes/AreTomo/"):
        os.makedirs(config.path_save+"/evaluation/volumes/AreTomo/")
    if not os.path.exists(config.path_save+"/evaluation/deformations/AreTomo/"):
        os.makedirs(config.path_save+"/evaluation/deformations/AreTomo/")
    if not os.path.exists(config.path_save+"/evaluation/volume_slices/AreTomo/"):
        os.makedirs(config.path_save+"/evaluation/volume_slices/AreTomo/")

    # Etomo method
    if not os.path.exists(config.path_save+"/evaluation/projections/Etomo/"):
        os.makedirs(config.path_save+"/evaluation/projections/Etomo/")   
    if not os.path.exists(config.path_save+"/evaluation/volumes/Etomo/"):
        os.makedirs(config.path_save+"/evaluation/volumes/Etomo/")
    if not os.path.exists(config.path_save+"/evaluation/deformations/Etomo/"):
        os.makedirs(config.path_save+"/evaluation/deformations/Etomo/")
    if not os.path.exists(config.path_save+"/evaluation/volume_slices/Etomo/"):
        os.makedirs(config.path_save+"/evaluation/volume_slices/Etomo/")

    # True volume
    if not os.path.exists(config.path_save+"/evaluation/deformations/true/"):
        os.makedirs(config.path_save+"/evaluation/deformations/true/")
    if not os.path.exists(config.path_save+"/evaluation/volume_slices/true/"):
        os.makedirs(config.path_save+"/evaluation/volume_slices/true/")
    if not os.path.exists(config.path_save+"/evaluation/volumes/true/"):
        os.makedirs(config.path_save+"/evaluation/volumes/true/")

    # FBP on undistorted projections
    if not os.path.exists(config.path_save+"/evaluation/projections/FBP_no_deformed/"):
        os.makedirs(config.path_save+"/evaluation/projections/FBP_no_deformed/")
    if not os.path.exists(config.path_save+"/evaluation/volumes/FBP_no_deformed/"):
        os.makedirs(config.path_save+"/evaluation/volumes/FBP_no_deformed/")
    if not os.path.exists(config.path_save+"/evaluation/volume_slices/FBP_no_deformed/"):
        os.makedirs(config.path_save+"/evaluation/volume_slices/FBP_no_deformed/")

    # FBP
    if not os.path.exists(config.path_save+"/evaluation/projections/FBP/"):
        os.makedirs(config.path_save+"/evaluation/projections/FBP/")
    if not os.path.exists(config.path_save+"/evaluation/volumes/FBP/"):
        os.makedirs(config.path_save+"/evaluation/volumes/FBP/")
    if not os.path.exists(config.path_save+"/evaluation/volume_slices/FBP/"):
        os.makedirs(config.path_save+"/evaluation/volume_slices/FBP/")

    #FBP ICETIDE deformation estimations
    if not os.path.exists(config.path_save+"/evaluation/projections/FBP_ICETIDE/"):
        os.makedirs(config.path_save+"/evaluation/projections/FBP_ICETIDE/")
    if not os.path.exists(config.path_save+"/evaluation/volumes/FBP_ICETIDE/"):
        os.makedirs(config.path_save+"/evaluation/volumes/FBP_ICETIDE/")
    if not os.path.exists(config.path_save+"/evaluation/volume_slices/FBP_ICETIDE/"):
        os.makedirs(config.path_save+"/evaluation/volume_slices/FBP_ICETIDE/")



    ######################################################################################################
    ## Load data
    ######################################################################################################
    data = np.load(config.path_save_data+"volume_and_projections.npz")
    # projections_noisy = torch.tensor(data['projections_noisy']).type(config.torch_type).to(device)
    V_FBP_t =  torch.tensor(np.moveaxis(np.double(mrcfile.open(config.path_save_data+"V_FBP.mrc").data),0,2)).type(config.torch_type).to(device)
    # numpy
    V_FBP = V_FBP_t.detach().cpu().numpy()

    data = np.load(config.path_save_data+"volume_and_projections.npz")
    projections_noisy = torch.Tensor(data['projections_noisy']).type(config.torch_type).to(device)
    config.Nangles = projections_noisy.shape[0]

    config.n1 = config.n1_patch
    config.n2 = config.n2_patch
    config.n3 = config.n3_patch

    projections_noisy_resize = torch.Tensor(resize(projections_noisy.detach().cpu().numpy(),(config.Nangles,config.n1,config.n2))).type(config.torch_type).to(device)

    ## Aretomo
    # get the files
    eval_AreTomo = False
    fsc_AreTomo_list = []
    fsc_AreTomo_centered_list = []
    for npatch in config.nPatch:
        ARE_TOMO_FILE = f'projections_aligned_aretomo_{npatch}by{npatch}.mrc'
        # ARE_TOMO_FILE_FBP = f'projections_rec_aretomo_{npatch}by{npatch}.mrc'
        path_file = os.path.join(config.path_save,'AreTomo',ARE_TOMO_FILE)
        # path_file_FBP = os.path.join(config.path_save,'AreTomo',ARE_TOMO_FILE_FBP)
        if os.path.isfile(path_file):
            eval_AreTomo = True
            # load projections # need to be reorder and there seem to be a shift in intensity, so we match the mean on the noisy one 
            proj_aligned_aretomo = np.moveaxis(np.float32(mrcfile.open(path_file,permissive=True).data),1,0)
            proj_aligned_aretomo = np.swapaxes(proj_aligned_aretomo,1,2)
            proj_aligned_aretomo = np.rot90(proj_aligned_aretomo,axes=(1,2))*1
            proj_aligned_aretomo = proj_aligned_aretomo + (projections_noisy.detach().cpu().numpy().mean()-proj_aligned_aretomo.mean())

            # Reconstruct with accurate FBP operator
            V_FBP_aretomo = reconstruct_FBP_volume(config, torch.tensor(proj_aligned_aretomo).to(device)).detach().cpu().numpy()
            V_FBP_aretomo /= np.linalg.norm(V_FBP_aretomo)



    ETOMO_FILE = 'projections_ali.mrc'
    path_file = os.path.join(config.path_save,'Etomo',ETOMO_FILE)
    eval_Etomo = False
    if os.path.isfile(path_file):
        eval_Etomo = True
        etomo_projections = np.double(mrcfile.open(path_file).data)
        etomo_projections_t = torch.tensor(etomo_projections).type(config.torch_type).to(device)
        V_FBP_etomo = reconstruct_FBP_volume(config, etomo_projections_t).detach().cpu().numpy()
        out = mrcfile.new(config.path_save_data+"V_etomo.mrc",np.moveaxis(V_FBP_etomo.astype(np.float32),2,0),overwrite=True)
        out.close()
        out = mrcfile.new(config.path_save_data+"V_etomo_centered.mrc",np.moveaxis(V_etomo_centered.astype(np.float32),2,0),overwrite=True)
        out.close() 



    ######################################################################################################
    ## Load and estimate our volume
    ######################################################################################################
    ## Load implicit network
    if(config.volume_model=="Fourier-features"):
        from models.fourier_net import FourierNet,FourierNet_Features
        impl_volume = FourierNet_Features(
            in_features=config.input_size_volume,
            sub_features=config.sub_features,
            out_features=config.output_size_volume, 
            hidden_features=config.hidden_size_volume,
            hidden_blocks=config.num_layers_volume,
            L = config.L_volume).to(device)

    if(config.volume_model=="MLP"):
        from models.fourier_net import MLP
        impl_volume = MLP(in_features= 1, 
                            hidden_features=config.hidden_size_volume, hidden_blocks= config.num_layers_volume, out_features=config.output_size_volume).to(device)

    if(config.volume_model=="multi-resolution"):  
        import tinycudann as tcnn
        config_network = {"encoding": {
                'otype': config.encoding.otype,
                'type': config.encoding.type,
                'n_levels': config.encoding.n_levels,
                'n_features_per_level': config.encoding.n_features_per_level,
                'log2_hashmap_size': config.encoding.log2_hashmap_size,
                'base_resolution': config.encoding.base_resolution,
                'per_level_scale': config.encoding.per_level_scale,
                'interpolation': config.encoding.interpolation,
            },
            "network": {
                "otype": config.network.otype,   
                "activation": config.network.activation,       
                "output_activation": config.network.output_activation,
                "n_neurons": config.hidden_size_volume,           
                "n_hidden_layers": config.num_layers_volume    
            }       
            }
        impl_volume = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=1, encoding_config=config_network["encoding"],
                                                    network_config=config_network["network"]).to(device)
    num_param = sum(p.numel() for p in impl_volume.parameters() if p.requires_grad) 
    print('---> Number of trainable parameters in volume net: {}'.format(num_param))
    checkpoint = torch.load(os.path.join(config.path_save,'training','model_trained.pt'),map_location=device)
    impl_volume.load_state_dict(checkpoint['implicit_volume'])
    shift_icetide = checkpoint['shift_est']
    rot_icetide = checkpoint['rot_est']
    implicit_deformation_icetide = checkpoint['local_deformation_network']
    ## Compute our model at same resolution than other volume
    rays_scaling = torch.tensor(np.array(config.rays_scaling))[None,None,None].type(config.torch_type).to(device)
    n1_eval, n2_eval, n3_eval = V_FBP.shape

    # Compute estimated volume
    with torch.no_grad():
        x_lin1 = np.linspace(-1,1,n1_eval)*rays_scaling[0,0,0,0].item()/2+0.5
        x_lin2 = np.linspace(-1,1,n2_eval)*rays_scaling[0,0,0,1].item()/2+0.5
        XX, YY = np.meshgrid(x_lin1,x_lin2,indexing='ij')
        grid2d = np.concatenate([XX.reshape(-1,1),YY.reshape(-1,1)],1)
        grid2d_t = torch.tensor(grid2d).type(config.torch_type)
        z_range = np.linspace(-1,1,n3_eval)*rays_scaling[0,0,0,2].item()*(n3_eval/n1_eval)/2+0.5
        V_icetide = np.zeros_like(V_FBP)
        for zz, zval in enumerate(z_range):
            grid3d = np.concatenate([grid2d_t, zval*torch.ones((grid2d_t.shape[0],1))],1)
            grid3d_slice = torch.tensor(grid3d).type(config.torch_type).to(device)
            estSlice = impl_volume(grid3d_slice).detach().cpu().numpy().reshape(n1_eval,n2_eval)
            V_icetide[:,:,zz] = estSlice
        V_icetide_t = torch.tensor(V_icetide).type(config.torch_type).to(device)

    ######################################################################################################
    # Using only the deformation estimates
    ######################################################################################################
    projections_noisy_undeformed = torch.zeros_like(projections_noisy_resize)
    xx1 = torch.linspace(-1,1,config.n1,dtype=config.torch_type,device=device)
    xx2 = torch.linspace(-1,1,config.n2,dtype=config.torch_type,device=device)
    XX_t, YY_t = torch.meshgrid(xx1,xx2,indexing='ij')
    XX_t = torch.unsqueeze(XX_t, dim = 2)
    YY_t = torch.unsqueeze(YY_t, dim = 2)
    for i in range(config.Nangles):
        coordinates = torch.cat([XX_t,YY_t],2).reshape(-1,2)
        #field = utils_deformation.deformation_field(-implicit_deformation_icetide[i].depl_ctr_pts[0].detach().clone())
        thetas = torch.tensor(-rot_icetide[i].thetas.item()).to(device)
    
        rot_deform = torch.stack(
                        [torch.stack([torch.cos(thetas),torch.sin(thetas)],0),
                        torch.stack([-torch.sin(thetas),torch.cos(thetas)],0)]
                        ,0)
        coordinates = coordinates - config.deformationScale*implicit_deformation_icetide[i](coordinates)
        coordinates = coordinates - shift_icetide[i].shifts_arr
        coordinates = torch.transpose(torch.matmul(rot_deform,torch.transpose(coordinates,0,1)),0,1) ## do rotation
        x = projections_noisy_resize[i].clone().view(1,1,config.n1,config.n2)
        x = x.expand(config.n1*config.n2, -1, -1, -1)
        out = cropper(x,coordinates,output_size = 1).reshape(config.n1,config.n2)
        projections_noisy_undeformed[i] = out
    V_FBP_icetide = reconstruct_FBP_volume(config, projections_noisy_undeformed).detach().cpu().numpy()


    #######################################################################################
    ## Save slices of volumes
    #######################################################################################
    saveIndex = [n3_eval//4,n3_eval//2,int(3*n3_eval//4)] # The slices to save taken from previous plots
    for index in saveIndex:
        # ICETIDE
        tmp = V_icetide[:,:,index]
        tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
        tmp = np.floor(255*tmp).astype(np.uint8)
        imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","ICETIDE","slice_{}.png".format(index)),tmp)

        # FBP
        tmp = V_FBP[:,:,index]
        tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
        tmp = np.floor(255*tmp).astype(np.uint8)
        imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","FBP","slice_{}.png".format(index)),tmp)

        if(eval_AreTomo):
            if index < V_FBP_aretomo.shape[2]:
                tmp = V_FBP_aretomo[:,:,index]
                tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
                tmp = np.floor(255*tmp).astype(np.uint8)
                imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","AreTomo","slice_{}.png".format(index)),tmp)

        if(eval_Etomo):
            if index < V_FBP_etomo.shape[2]:
                tmp = V_FBP_etomo[:,:,index]
                tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
                tmp = np.floor(255*tmp).astype(np.uint8)
                imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","Etomo","slice_{}.png".format(index)),tmp)

        # FBP icetide
        if index < V_FBP_icetide.shape[2]:
            tmp = V_FBP_icetide[:,:,index]
            tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
            tmp = np.floor(255*tmp).astype(np.uint8)
            imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"volume_slices","FBP_ICETIDE","slice_{}.png".format(index)),tmp)


    #######################################################################################
    ## Save volumes
    #######################################################################################

    def display_XYZ(tmp,name="true"):
        f , aa = plt.subplots(2, 2, gridspec_kw={'height_ratios': [tmp.shape[2]/tmp.shape[0], 1], 'width_ratios': [1,tmp.shape[2]/tmp.shape[0]]})
        aa[0,0].imshow(tmp.mean(0).T,cmap='gray')
        aa[0,0].axis('off')
        aa[1,0].imshow(tmp.mean(2),cmap='gray')
        aa[1,0].axis('off')
        aa[1,1].imshow(tmp.mean(1),cmap='gray')
        aa[1,1].axis('off')
        aa[0,1].axis('off')
        plt.tight_layout(pad=1, w_pad=-1, h_pad=1)
        plt.savefig(os.path.join(config.path_save_data,'evaluation',"volumes",name,"XYZ.png"))

    # ICETIDE
    tmp = V_icetide
    display_XYZ(tmp,name="ICETIDE")

    # FBP volume
    tmp = V_FBP
    display_XYZ(tmp,name="FBP")

    if(eval_AreTomo):
        # AreTomo volume
        tmp = V_FBP_aretomo
        display_XYZ(tmp,name="AreTomo")

    if(eval_Etomo):
        # Etomo volume
        tmp = V_FBP_etomo
        display_XYZ(tmp,name="Etomo")

    # FBP_ICETIDE volume
    tmp = V_FBP_icetide
    display_XYZ(tmp,name="FBP_ICETIDE")


    out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',
                                "volumes","ICETIDE_volume.mrc"),np.moveaxis(V_icetide,2,0),overwrite=True)
    out.close()
    if eval_AreTomo:
        out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',
                                    "volumes","AreTomo_volume.mrc"),np.moveaxis(V_FBP_aretomo,2,0),overwrite=True)
        out.close()
    if eval_Etomo:
        out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',
                                    "volumes","Etomo_volume.mrc"),np.moveaxis(V_FBP_etomo,2,0),overwrite=True)
        out.close()
    out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"volumes",
                                "FBP_volume.mrc"),np.moveaxis(V_FBP,2,0),overwrite=True)
    out.close()
    out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"volumes",
                                "FBP_icetide_volume.mrc"),np.moveaxis(V_FBP_icetide,2,0),overwrite=True)
    out.close()


    #######################################################################################
    ## Generate projections
    #######################################################################################
    # Define angles and X-ray transform
    angles = np.linspace(config.view_angle_min,config.view_angle_max,config.Nangles)
    operator_ET = ParallelBeamGeometry3DOpAngles_rectangular((config.n1,config.n2,config.n3), angles/180*np.pi, fact=1)

    projections_icetide = operator_ET(V_icetide_t).detach().cpu().numpy()
    projections_FBP = operator_ET(V_FBP_t).detach().cpu().numpy()
    projections_FBP_icetide = projections_noisy_undeformed.detach().cpu().numpy()
    if(eval_AreTomo):
        V_FBP_aretomo_t = torch.tensor(V_FBP_aretomo).to(device)
        projections_AreTomo = operator_ET(V_FBP_aretomo_t).detach().cpu().numpy()
    if(eval_Etomo):
        V_FBP_etomo_t = torch.tensor(V_FBP_etomo).to(device)
        projections_Etomo = operator_ET(V_FBP_etomo_t).detach().cpu().numpy()

    out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","icetide_projections.mrc"),projections_icetide.astype(np.float32),overwrite=True)
    out.close()
    if(eval_AreTomo):
        out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","AreTomo_projections.mrc"),projections_AreTomo.astype(np.float32),overwrite=True)
        out.close()
    if(eval_Etomo):
        out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","Etomo_projections.mrc"),projections_Etomo.astype(np.float32),overwrite=True)
        out.close()

    out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","FBP_projections.mrc"),projections_FBP.astype(np.float32),overwrite=True)
    out.close()
    out = mrcfile.new(os.path.join(config.path_save_data,'evaluation',"projections","FBP_icetide_projections.mrc"),projections_FBP_icetide.astype(np.float32),overwrite=True)
    out.close()

    for k in range(config.Nangles):
        tmp = projections_icetide[k]
        tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
        tmp = np.floor(255*tmp).astype(np.uint8)
        imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"projections","ICETIDE","snapshot_{}.png".format(k)),tmp)
        if(eval_AreTomo):
            tmp = projections_AreTomo[k]
            tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
            tmp = np.floor(255*tmp).astype(np.uint8)
            imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"projections","AreTomo","snapshot_{}.png".format(k)),tmp)
        if(eval_Etomo):
            tmp = projections_Etomo[k]
            tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
            tmp = np.floor(255*tmp).astype(np.uint8)
            imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"projections","Etomo","snapshot_{}.png".format(k)),tmp)
        tmp = projections_FBP[k]
        tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
        tmp = np.floor(255*tmp).astype(np.uint8)
        imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"projections","FBP","snapshot_{}.png".format(k)),tmp)
        tmp = projections_FBP_icetide[k]
        tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
        tmp = np.floor(255*tmp).astype(np.uint8)
        imageio.imwrite(os.path.join(config.path_save_data,'evaluation',"projections","FBP_ICETIDE","snapshot_{}.png".format(k)),tmp)

    # ## Saving the inplance angles 
    # inplaneAngles = np.zeros((config.Nangles,5))
    # inplaneAngles[:,0] = angles
    # inplaneAngles[:,1] = inplane_rotation
    # inplaneAngles[:,2] = inplane_rotation_icetide
    # if eval_AreTomo:
    #     inplaneAngles[:,3] = inplane_rotation_aretomo
    # if eval_ETOMO:
    #     inplaneAngles[:,4] = inplane_rotation_etomo


    # # save as a csv file
    # header ='angles,true,icetide,AreTomo,Etomo'
    # np.savetxt(os.path.join(config.path_save,'evaluation','inplane_angles.csv'),inplaneAngles,header=header,delimiter=",",comments='')

    plt.close('all')












# Computing the resolution
def resolution(fsc,cutt_off=0.5,minIndex=2):
    """
    The function returns the resolution of the volume
    """
    resolution_Set = np.zeros(fsc.shape[0])         
    for i in range(fsc.shape[0]):
        indeces = np.where(fsc[i]<cutt_off)[0]
        choosenIndex = np.where(indeces>minIndex)[0][0]
        resolution_Set[i] = indeces[choosenIndex]
    return resolution_Set

def getReolution(dataframe,cutoffs=[0.5,0.143]):
    """
    Uses the pandas dataframe extracts the fsc for each method and outputs the resolution for two values
    """
    fsc_ours = dataframe['icetide'].values
    if('ETOMO' in dataframe.columns):
        fsc_etomo = dataframe['ETOMO'].values
    else:
        fsc_etomo = np.zeros(len(fsc_ours))
    if('AreTomo_patch1' in dataframe.columns):
        fsc_areTomo = dataframe['AreTomo_patch1'].values
    else:
        fsc_areTomo = np.zeros(len(fsc_ours))
    fsc_FBP = dataframe['FBP'].values
    fsc_FBP_undeformed = dataframe['FBP_no_deformed'].values
    fsc_FBP_est_deformed = dataframe['FBP_est_deformed'].values

    fscSet = np.zeros((6, len(fsc_ours)))
    fscSet[0] = fsc_ours
    fscSet[1] = fsc_etomo
    fscSet[2] = fsc_areTomo
    fscSet[3] = fsc_FBP
    fscSet[4] = fsc_FBP_undeformed
    fscSet[5] = fsc_FBP_est_deformed

    res_set = np.zeros((len(cutoffs),6))
    for i,cutoff in enumerate(cutoffs):
        res_set[i] = resolution(fscSet,cutoff)

    return res_set


def getCorrelation(dataframe):
    """
    Uses the pandas dataframe extracts the CC for each method
    """
    CC_ours = dataframe['icetide'].values
    if('ETOMO' in dataframe.columns):
        CC_etomo = dataframe['ETOMO'].values
    else:
        CC_etomo = 0
    if('AreTomo_patch1' in dataframe.columns):
        CC_areTomo = dataframe['AreTomo_patch1'].values
    else:
        CC_areTomo = 0
    CC_FBP = dataframe['FBP'].values
    CC_FBP_undeformed = dataframe['FBP_no_deformed'].values
    CC_FBP_est_deformed = dataframe['FBP_est_deformed'].values

    CCSet = np.zeros((6, 1))
    CCSet[0] = CC_ours
    CCSet[1] = CC_etomo
    CCSet[2] = CC_areTomo
    CCSet[3] = CC_FBP
    CCSet[4] = CC_FBP_undeformed
    CCSet[5] = CC_FBP_est_deformed

    return CCSet



