'''
Contain the function used to generate simulated data from a clean tomogram. 
The goal is to compare different approaches on this dataset that is suppose
to mimic the CryoET image formation model.
'''
import os
import torch
import mrcfile
import imageio
import numpy as np
from skimage.transform import resize

from ops.radon_3d_lib import ParallelBeamGeometry3DOpAngles_rectangular
from utils import utils_data_generation, utils_deformation, utils_display


def data_generation(config):
    print("Runing data generation to generate simulated projections.")

    SNR_value = config.SNR_value

    # Choosing the seed and the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    if torch.cuda.device_count()>1:
        torch.cuda.set_device(config.device_num)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # prepare the folders
    if not os.path.exists("results/"):
        os.makedirs("results/")
    if not os.path.exists("datasests/"):
        os.makedirs("datasests/")
    if not os.path.exists(config.path_save_data):
        os.makedirs(config.path_save_data)
    if not os.path.exists(config.path_save_data+"projections/"):
        os.makedirs(config.path_save_data+"projections/")
    if not os.path.exists(config.path_save_data+"projections/noisy/"):
        os.makedirs(config.path_save_data+"projections/noisy/")
    if not os.path.exists(config.path_save_data+"projections/clean/"):
        os.makedirs(config.path_save_data+"projections/clean/")
    if not os.path.exists(config.path_save_data+"projections/deformed/"):
        os.makedirs(config.path_save_data+"projections/deformed/")
    if not os.path.exists(config.path_save_data+"volumes/"):
        os.makedirs(config.path_save_data+"volumes/")
    if not os.path.exists(config.path_save_data+"volumes/clean/"):
        os.makedirs(config.path_save_data+"volumes/clean/")
    if not os.path.exists(config.path_save_data+"deformations/"):
        os.makedirs(config.path_save_data+"deformations/")

    #######################################################################################
    ## Load data
    #######################################################################################
    # Parameters
    print("Loading volume...")
    name_volume="grandmodel.mrc" # here: https://www.shrec.net/cryo-et/
    path_volume = "./datasets/"+str(config.volume_name)+"/"+name_volume

    if not(os.path.isfile(path_volume)):
        print("Please download the Shrec 2021 dataset and move the files in the folder datasests.")
        print("The structure of the folder should be 'datasets/model_X/grandmodel.mrc', where model_X is specified by config.volume_name")
        print("Here is the link to download the dataset: https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/XRTJMA")
        
    # Loading and shaping the volume
    V = -np.double(mrcfile.open(path_volume).data)
    nv = V.shape # size of the loaded volume 
    V = V[nv[0]//2-config.n3_patch//2:nv[0]//2+config.n3_patch//2,nv[1]//2-config.n1_patch//2:nv[1]//2+config.n1_patch//2,nv[2]//2-config.n2_patch//2:nv[2]//2+config.n2_patch//2]
    V = resize(V,(config.n3,config.n1,config.n2))
    V = np.swapaxes(V,0,1)
    V = np.swapaxes(V,1,2)
    V_t = torch.tensor(V).to(device).type(config.torch_type)

    print("Volume loaded.")

    #######################################################################################
    ## Generate projections
    #######################################################################################
    # Define angles and X-ray transform
    print("Making tilt-series...")
    angles = np.linspace(config.view_angle_min,config.view_angle_max,config.Nangles)
    angles_t = torch.tensor(angles).type(config.torch_type).to(device)
    operator_ET = ParallelBeamGeometry3DOpAngles_rectangular((config.n1,config.n2,config.n3), angles/180*np.pi, fact=1)

    # Define global and local deformations
    affine_tr = []
    local_tr = []

    # Using polynomials to generate angle deformation, this is to comply with aretomo
    if(config.slowAngle):
        print("Use polynomial with roots "+str(config.n_roots)+" to generate angle deformation")
        angle_def = utils_deformation.polynomial_angle_deformation(**config)
    for i in range(config.Nangles*config.number_sub_projections):
        scaleX, scaleY, shiftX, shiftY, shearX, shearY, angle  = utils_deformation.generate_params_deformation(config.scale_min,
                    config.scale_max,config.shift_min,config.shift_max,config.shear_min,config.shear_max,config.angle_min,config.angle_max)  
        if(config.slowAngle):
            angle = angle_def[i]
        affine_tr.append(utils_deformation.AffineTransform(scaleX, scaleY, shiftX, shiftY, shearX, shearY, angle ).cuda())
        depl_ctr_pts = torch.randn(2,config.N_ctrl_pts_local_def[0],config.N_ctrl_pts_local_def[1]).to(device).type(config.torch_type)
        depl_ctr_pts[0] = depl_ctr_pts[0]/config.n1*config.sigma_local_def
        depl_ctr_pts[1] = depl_ctr_pts[1]/config.n2*config.sigma_local_def
        field = utils_deformation.deformation_field(depl_ctr_pts)
        local_tr.append(field)
        # Some display
        nsr = (config.n1*4,config.n2*4)
        Nsp = (config.n1//20,config.n2//20) # number of Diracs in each direction
        supp = config.n1//70
        # Display local deformations
        utils_display.display_local(field,field_true=None,Npts=Nsp,img_path=config.path_save_data+"deformations/local_deformations_view_"+str(i),
                                    img_type='.png',scale=1,alpha=0.8,width=0.0015,wx=config.n1//2,wy=config.n2//2)
        # Display global deformations
        sp1 = np.array(np.floor(np.linspace(0,nsr[0],Nsp[0]+2)),dtype=int)[1:-1]
        sp2 = np.array(np.floor(np.linspace(0,nsr[1],Nsp[1]+2)),dtype=int)[1:-1]
        spx, spy = np.meshgrid(sp1,sp2)  
        xx1 = np.linspace(-nsr[0]/2,nsr[0]/2,nsr[0])
        xx2 = np.linspace(-nsr[1]/2,nsr[1]/2,nsr[1])
        XX, YY = np.meshgrid(xx1,xx2, indexing='ij')
        G = np.exp(-(XX**2+YY**2)/(2*(supp/3)**2))
        G[:nsr[0]//2-supp,:]=0
        G[nsr[0]//2+supp:,:]=0
        G[:,:nsr[1]//2-supp]=0
        G[:,nsr[1]//2+supp:]=0
        G /= G.sum()
        im_grid = np.zeros(nsr)
        im_grid[spx,spy] = 1
        im_grid = np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(G)*np.fft.fft2(im_grid))).real
        im_grid_t = torch.tensor(im_grid).to(device).type(config.torch_type)
        img_deform_global = utils_deformation.apply_deformation([affine_tr[-1]],im_grid_t.reshape(1,nsr[0],nsr[1]))
        tmp = img_deform_global.detach().cpu().numpy()[0].reshape(nsr)
        tmp = (tmp - tmp.max())/(tmp.max()-tmp.min())
        tmp = np.floor(255*tmp).astype(np.uint8)
        imageio.imwrite(config.path_save_data+"deformations/global_deformations_view_"+str(i)+".png",tmp)  

    with torch.no_grad():
        projections_clean = operator_ET(V_t)
        projections_clean = projections_clean[:,None].repeat(1,config.number_sub_projections,1,1).reshape(-1,config.n1,config.n2)
        print("Tilt-series made.")

        print("Saving volumes, tilt-series, FBP, ...")
        # add deformations
        projections_deformed_global = utils_deformation.apply_deformation(affine_tr,projections_clean)
        projections_deformed = utils_deformation.apply_local_deformation(local_tr,projections_deformed_global)

        # add noise
        sigma_noise = utils_data_generation.find_sigma_noise_t(SNR_value,projections_deformed)
        projections_noisy = projections_deformed.clone() + torch.randn_like(projections_deformed)*sigma_noise
        projections_noisy_no_deformed = projections_clean.clone() + torch.randn_like(projections_clean)*sigma_noise

        # Save deformations and projections
        np.save(config.path_save_data+"global_deformations.npy",affine_tr)
        np.save(config.path_save_data+"local_deformations.npy",local_tr)
        np.savez(config.path_save_data+"volume_and_projections.npz",projections_noisy=projections_noisy.detach().cpu().numpy(),projections_deformed=projections_deformed.detach().cpu().numpy(),projections_deformed_global=projections_deformed_global.detach().cpu().numpy(),projections_clean=projections_clean.detach().cpu().numpy())

        # save projections
        for k in range(config.Nangles):
            tmp = projections_clean[k].detach().cpu().numpy()
            tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
            tmp = np.floor(255*tmp).astype(np.uint8)
            imageio.imwrite(config.path_save_data+"projections/clean/clean_"+str(k)+".png",tmp)

            tmp = projections_deformed[k].detach().cpu().numpy()
            tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
            tmp = np.floor(255*tmp).astype(np.uint8)
            imageio.imwrite(config.path_save_data+"projections/deformed/deformed_"+str(k)+".png",tmp)

            tmp = projections_noisy[k].detach().cpu().numpy()
            tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
            tmp = np.floor(255*tmp).astype(np.uint8)
            imageio.imwrite(config.path_save_data+"projections/noisy/noisy_"+str(k)+".png",tmp)


        ## Save volumes and other interesting qunatities
        for k in range(V_t.shape[2]):
            tmp = V_t[:,:,k].detach().cpu().numpy()
            tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
            tmp = np.floor(255*tmp).astype(np.uint8)
            imageio.imwrite(os.path.join(config.path_save,'volumes','clean','obs_{}.png'.format(k)),tmp)

        projections_noisy_avg = projections_noisy.reshape(config.Nangles,-1,config.n1,config.n2).mean(1).contiguous().type(torch.float32)
        projections_noisy_no_deformed_avg =  projections_noisy_no_deformed.reshape(config.Nangles,-1,config.n1,config.n2).mean(1)
        V_FBP = operator_ET.pinv(projections_noisy_avg.detach().requires_grad_(False))
        V_FBP_no_deformed = operator_ET.pinv(projections_noisy_no_deformed_avg).detach().requires_grad_(False)
        out = mrcfile.new(config.path_save_data+"V_FBP.mrc",np.moveaxis(V_FBP.detach().cpu().numpy().reshape(config.n1,config.n2,config.n3),2,0),overwrite=True)
        out.close() 
        out = mrcfile.new(config.path_save_data+"V_FBP_no_deformed.mrc",np.moveaxis(V_FBP_no_deformed.detach().cpu().numpy().reshape(config.n1,config.n2,config.n3),2,0),overwrite=True)
        out.close() 
        out = mrcfile.new(config.path_save_data+"V.mrc",np.moveaxis(V_t.detach().cpu().numpy().reshape(config.n1,config.n2,config.n3),2,0),overwrite=True)
        out.close() 
        out = mrcfile.new(config.path_save_data+"projections.mrc",projections_noisy.detach().cpu().numpy(),overwrite=True)
        out.close() 

        projections_noisy_ = projections_noisy.detach().cpu().numpy()*1
        projections_noisy_no_deformed_ = projections_noisy_no_deformed.detach().cpu().numpy()*1
        out = mrcfile.new(config.path_save_data+"projections_noisy_no_deformed.mrc",projections_noisy_no_deformed_,overwrite=True)
        out.close()

        # Save angle files
        np.save(config.path_save_data+"angles.npy",angles)
        np.savetxt(config.path_save_data+"angles.txt",angles)
        print("Saving done.")


    
def data_generation_real_data(config):
    # Choosing the seed and the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count()>1:
        torch.cuda.set_device(config.device_num)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # prepare the folders
    if not os.path.exists(config.path_save_data):
        os.makedirs(config.path_save_data)
    if not os.path.exists(config.path_save_data+"projections/"):
        os.makedirs(config.path_save_data+"projections/")
    if not os.path.exists(config.path_save_data+"projections/noisy/"):
        os.makedirs(config.path_save_data+"projections/noisy/")

    #######################################################################################
    ## Load data
    #######################################################################################
    projections_noisy = np.float32(mrcfile.open(os.path.join(config.path_load,config.volume_name+".mrc"),permissive=True).data)
    projections_noisy = projections_noisy/np.abs(projections_noisy).max()
    if config.n1 is not None:
        config.Nangles = projections_noisy.shape[0]
        projections_noisy = resize(projections_noisy,(config.Nangles,config.n1,config.n2))
    else:
        config.Nangles, config.n1, config.n2 = projections_noisy.shape
        config.n3 = config.n1*2

    np.savez(config.path_save_data+"volume_and_projections.npz",projections_noisy=projections_noisy)

    # save projections
    for k in range(config.Nangles):
        tmp = projections_noisy[k]
        tmp = (tmp - tmp.min())/(tmp.max()-tmp.min())
        tmp = np.floor(255*tmp).astype(np.uint8)
        imageio.imwrite(config.path_save_data+"projections/noisy/noisy_"+str(k)+".png",tmp)


    angles = np.linspace(config.view_angle_min,config.view_angle_max,config.Nangles)
    angles_t = torch.tensor(angles).type(config.torch_type).to(device)
    operator_ET = ParallelBeamGeometry3DOpAngles_rectangular((config.n1,config.n2,config.n3), angles/180*np.pi, fact=1)
    V_FBP = operator_ET.pinv(torch.tensor(projections_noisy).to(device).detach().requires_grad_(False))
    out = mrcfile.new(config.path_save_data+"V_FBP.mrc",np.moveaxis(V_FBP.detach().cpu().numpy().reshape(config.n1,config.n2,config.n3),2,0),overwrite=True)
    out.close() 
    out = mrcfile.new(config.path_save_data+"projections.mrc",projections_noisy,overwrite=True)
    out.close() 

    # Save angle files
    np.save(config.path_save_data+"angles.npy",angles)
    np.savetxt(config.path_save_data+"angles.txt",angles)
    print("Saving done.")
