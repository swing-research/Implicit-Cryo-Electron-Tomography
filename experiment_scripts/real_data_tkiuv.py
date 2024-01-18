"""
File to run the simulation with SHREC2021 model_0 volume.

It creates the dataset (projections, FBP), train ICE-TIDE and compare the results with AreTomo and Etomo.
"""

import argparse
import configs.real_data_tkiuv as config_file
import subprocess
from skimage.transform import resize
import time


def main():   
    parser = argparse.ArgumentParser(description='Run experiement for the SHREC dataset.')
    parser.add_argument('--no_gen_data', action='store_false', help='Generate the data, default is True.')
    parser.add_argument('--no_train', action='store_false', help='Train the model, default is True.')
    parser.add_argument('--no_aretomo', action='store_false', help='Run AreTomo, default is True.')
    parser.add_argument('--no_comparison', action='store_false', help='Compare the different methods, default is True.')
    args = parser.parse_args()

    # Get config file
    config = config_file.get_config()

    # # Make the data
    # if args.no_gen_data:
    #     import data_generation
    #     data_generation.data_generation(config)


    ############################
    ## generate data
    ############################
    import os
    import numpy as np
    import torch
    import mrcfile
    import imageio

    from ops.radon_3d_lib import ParallelBeamGeometry3DOpAngles_rectangular
    from skimage.transform import pyramid_gaussian


    if args.no_gen_data:
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



    # Train ICE-TIDE
    if args.no_train:
        import train
        train.train_without_ground_truth(config)

    # AreTomo
    if args.no_aretomo:
        if config.path_aretomo is not None:
            for npatch in config.nPatch:
                # TODO: keep track of GPU memory?
                t0 = time.time()
                try:
                    subprocess.run(['bash', 'aretomo.sh',config.path_aretomo,config.path_save,str(config.n3),str(config.n3),str(npatch)])
                except subprocess.CalledProcessError as e:
                    print(f"Error: {e}")
                t = time.time()-t0
                #TODO: save time and memory used in os.path.join(config.path_save,'AreTomo',f'time_memory_{npatch}by{npatch}.txt')

    # Etomo
    # TODO

    # Compare the results and save the figures
    if args.no_comparison:
        from compare_results import compare_results
        compare_results(config)


if __name__ == '__main__':
    main()