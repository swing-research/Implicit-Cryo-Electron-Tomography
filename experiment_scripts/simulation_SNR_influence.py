"""
File to run the simulation with SHREC2021 model_0 volume.

It creates the dataset (projections, FBP), train ICE-TIDE and compare the results with AreTomo and Etomo.
"""

import numpy as np
import argparse
import configs.shrec_all_SNR as config_file
import matplotlib.pyplot as plt
import subprocess
import time
import os


def main():   
    parser = argparse.ArgumentParser(description='Run experiement for the SHREC dataset.')
    parser.add_argument('--no_gen_data', action='store_false', help='Generate the data, default is True.')
    parser.add_argument('--no_train', action='store_false', help='Train the model, default is True.')
    parser.add_argument('--no_aretomo', action='store_false', help='Run AreTomo, default is True.')
    parser.add_argument('--no_comparison', action='store_false', help='Compare the different methods, default is True.')
    args = parser.parse_args()

    # Get config file
    config = config_file.get_config()

    v_SNR_list = np.arange(-20,31,10)
    for v_SNR in v_SNR_list:
        # Noise
        config.SNR_value = v_SNR
        # # Parameters for the data generation
        config.path_save_data = "./results/SNR_exp_"+str(config.volume_name)+"_SNR_"+str(config.SNR_value)+"_size_"+str(config.n1)+"_Nangles_"+str(config.Nangles)+"/"
        config.path_save = "./results/SNR_exp_"+str(config.volume_name)+"_SNR_"+str(config.SNR_value)+"_size_"+str(config.n1)+"_Nangles_"+str(config.Nangles)+"/"

        # Make the data
        if args.no_gen_data:
            import data_generation
            data_generation.data_generation(config)

        # Train ICE-TIDE
        if args.no_train:
            import train
            train.train(config)

        # AreTomo
        if args.no_aretomo:
            if config.path_aretomo is not None:
                for npatch in config.nPatch:
                    t0 = time.time()
                    try:
                        subprocess.run(['bash', 'aretomo.sh',config.path_aretomo,config.path_save,str(config.n3),str(config.n3),str(npatch)])
                    except subprocess.CalledProcessError as e:
                        print(f"Error: {e}")
                    t = time.time()-t0


    # Compare the results and save the figures
    if args.no_comparison:
        import pandas as pd
        from compare_results import getReolution, getCorrelation
        from compare_results import compare_results

        resolution05 = np.zeros((6,len(v_SNR_list)))
        resolution0143 = np.zeros((6,len(v_SNR_list)))
        CC_all = np.zeros((6,len(v_SNR_list)))
        x_val = np.zeros(len(v_SNR_list))
        model_name = []
        for i, v_SNR in enumerate(v_SNR_list):
            config.SNR_value = v_SNR
            config.path_save_data = "./results/SNR_exp_"+str(config.volume_name)+"_SNR_"+str(config.SNR_value)+"_size_"+str(config.n1)+"_Nangles_"+str(config.Nangles)+"/"
            config.path_save = "./results/SNR_exp_"+str(config.volume_name)+"_SNR_"+str(config.SNR_value)+"_size_"+str(config.n1)+"_Nangles_"+str(config.Nangles)+"/"
            
            # compare_results(config)

            data_path = os.path.join(config.path_save,'evaluation','FSC.csv')
            fsc = pd.read_csv(data_path)
            resolutions = getReolution(fsc)
            resolution05[:,i] = resolutions[0]
            resolution0143[:,i] = resolutions[1]
            x_val[i] = i
            model_name.append('SNR '+str(v_SNR))

            data_path = os.path.join(config.path_save,'evaluation','CC.csv')
            cor = pd.read_csv(data_path)
            correlations = getCorrelation(cor)
            CC_all[:,i] = correlations[:,0]

        config.path_save = "./results/SNR_exp_"+str(config.volume_name)+"_size_"+str(config.n1)+"_Nangles_"+str(config.Nangles)+"/"
        if not os.path.exists(config.path_save):
            os.makedirs(config.path_save)
        if not os.path.exists(config.path_save+"evaluation/"):
            os.makedirs(config.path_save+"evaluation/")

        plt.figure()
        plt.plot(v_SNR_list,resolution05[3,:], label='FBP')
        plt.plot(v_SNR_list,resolution05[4,:], label='FBP undeformed')
        plt.plot(v_SNR_list,resolution05[0,:],label='ICETIDE')
        plt.plot(v_SNR_list,resolution05[5,:],label='FBP est deformation')
        # plt.xticks(x_val+0.3, model_name,rotation=45)
        plt.ylabel('Resolution (1/pixel size)')
        plt.ylabel('SNR (dB)')
        plt.legend()
        plt.savefig(os.path.join(config.path_save,'evaluation','resolution05.pdf'))
        plt.close() 

        plt.figure()
        plt.plot(v_SNR_list,resolution0143[3,:], label='FBP')
        plt.plot(v_SNR_list,resolution0143[4,:], label='FBP undeformed')
        plt.plot(v_SNR_list,resolution0143[0,:],label='ICETIDE')
        plt.plot(v_SNR_list,resolution0143[5,:],label='FBP est deformation')
        # plt.xticks(x_val+0.3, model_name,rotation=45)
        plt.ylabel('Resolution (1/pixel size)')
        plt.ylabel('SNR (dB)')
        plt.legend()
        plt.savefig(os.path.join(config.path_save,'evaluation','resolution0143.pdf'))
        plt.close() 

        #save as csv file with header and SNR values as columns
        resolution05 = np.vstack((model_name,resolution05))
        resolution0143 = np.vstack((model_name,resolution0143))
        header = ['icetide','ETOMO','AreTomo','FBP','FBP_no_deformed','FBP_est_deformed']
        header= 'MODEL_NAME'+','+','.join(header)
        pd_resoluton05 = pd.DataFrame(resolution05.T,columns=header.split(','))
        pd_resoluton0143 = pd.DataFrame(resolution0143.T,columns=header.split(','))
        pd_resoluton05.to_csv(os.path.join(config.path_save,'evaluation','resolution05.csv'),index=False)
        pd_resoluton0143.to_csv(os.path.join(config.path_save,'evaluation','resolution0143.csv'),index=False)


        plt.figure()
        plt.plot(CC_all[3,:], label='FBP')
        plt.plot(CC_all[4,:], label='FBP undeformed')
        plt.plot(CC_all[0,:],label='ICETIDE')
        plt.plot(CC_all[5,:],label='FBP est deformation')
        plt.ylabel('Resolution (1/pixel size)')
        plt.ylabel('SNR (dB)')
        plt.legend()
        plt.savefig(os.path.join(config.path_save,'evaluation','CC.pdf'))
        plt.close() 

        #save as csv file with header and SNR values as columns
        CC_all = np.vstack((model_name,CC_all))
        header = ['icetide','ETOMO','AreTomo','FBP','FBP_no_deformed','FBP_est_deformed']
        header= 'MODEL_NAME'+','+','.join(header)
        pd_CC = pd.DataFrame(CC_all.T,columns=header.split(','))
        pd_CC.to_csv(os.path.join(config.path_save,'evaluation','CC.csv'),index=False)





if __name__ == '__main__':
    main()