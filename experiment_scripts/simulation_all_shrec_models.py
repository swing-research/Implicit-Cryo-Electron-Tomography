"""
File to run the simulation with SHREC2021 model_0 volume.

It creates the dataset (projections, FBP), train ICE-TIDE and compare the results with AreTomo and Etomo.
"""

import numpy as np
import argparse
import configs.shrec_all_model as config_file
import subprocess
import time
import os



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
    fsc_ours = dataframe['ours'].values
    if('ETOMO' in dataframe.columns):
        fsc_etomo = dataframe['ETOMO'].values
    else:
        fsc_etomo = np.zeros(len(fsc_ours))
    if('AreTomo' in dataframe.columns):
        fsc_areTomo = dataframe['AreTomo'].values
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

def main():   
    parser = argparse.ArgumentParser(description='Run experiement for the SHREC dataset.')
    parser.add_argument('--no_gen_data', action='store_false', help='Generate the data, default is True.')
    parser.add_argument('--no_train', action='store_false', help='Train the model, default is True.')
    parser.add_argument('--no_aretomo', action='store_false', help='Run AreTomo, default is True.')
    parser.add_argument('--no_comparison', action='store_false', help='Compare the different methods, default is True.')
    args = parser.parse_args()

    # Get config file
    config = config_file.get_config()

    volume_name_list = ['model_0',
                        'model_1',
                        'model_2',
                        'model_3',
                        'model_4',
                        'model_5',
                        'model_6',
                        'model_7',
                        'model_8',
                        'model_9']
    for v_name in volume_name_list:
        config.volume_name = v_name
        # # Parameters for the data generation
        config.path_save_data = "./results/all_models_"+str(config.volume_name)+"_SNR_"+str(config.SNR_value)+"_size_"+str(config.n1)+"_Nangles_"+str(config.Nangles)+"/"
        config.path_save = "./results/all_models_"+str(config.volume_name)+"_SNR_"+str(config.SNR_value)+"_size_"+str(config.n1)+"_Nangles_"+str(config.Nangles)+"/"

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
            from compare_results import compare_results
            compare_results(config)


    if args.no_comparison:
        resolution05 = np.zeros((6,len(volume_name_list)))
        resolution0143 = np.zeros((6,len(volume_name_list)))
        x_val = np.zeros(len(volume_name_list))
        for i, v_name in enumerate(volume_name_list):
            config.path_save_data = "./results/all_models_"+str(config.volume_name)+"_SNR_"+str(config.SNR_value)+"_size_"+str(config.n1)+"_Nangles_"+str(config.Nangles)+"/"
            config.path_save = "./results/all_models_"+str(config.volume_name)+"_SNR_"+str(config.SNR_value)+"_size_"+str(config.n1)+"_Nangles_"+str(config.Nangles)+"/"

            data_path = os.path.join(config.path_save,'evaluation','FSC.csv')
            fsc = pd.read_csv(data_path+'FSC.csv')
            resolutions = getReolution(fsc)
            resolution05[:,i] = resolutions[0]
            resolution0143[:,i] = resolutions[1]
            x_val[i] = int(f.split('_')[1])
            model_name.append('model'+str(i))

        config.path_save = "./results/all_models"+"_SNR_"+str(config.SNR_value)+"_size_"+str(config.n1)+"_Nangles_"+str(config.Nangles)+"/"
        if not os.path.exists(config.path_save):
            os.makedirs(config.path_save)
        if not os.path.exists(config.path_save+"evaluation/"):
            os.makedirs(config.path_save+"evaluation/")

        plt.figure()
        plt.bar(x_val,resolution05[3,:],width=0.2, label='FBP')
        plt.bar(x_val+0.2,resolution05[4,:],width=0.2, label='FBP undeformed')
        plt.bar(x_val+0.4,resolution05[0,:],width=0.2,label='ours')
        plt.bar(x_val+0.6,resolution05[5,:],width=0.2,label='FBP est deformation')
        plt.xticks(x_val+0.3, model_name)
        plt.ylabel('Resolution (1/pixel size)')
        plt.legend()
        plt.savefig(os.path.join(config.path_save,'evaluation','resolution05.pdf'))
        plt.close()

        plt.figure()
        plt.bar(x_val,resolution0143[3,:],width=0.2, label='FBP')
        plt.bar(x_val+0.2,resolution0143[4,:],width=0.2, label='FBP undeformed')
        plt.bar(x_val+0.4,resolution0143[0,:],width=0.2,label='ours')
        plt.bar(x_val+0.6,resolution0143[5,:],width=0.2,label='FBP est deformation')
        plt.xticks(x_val+0.3, model_name)
        plt.ylabel('Resolution (1/pixel size)')
        plt.legend()
        plt.savefig(os.path.join(config.path_save,'evaluation','resolution0143.pdf'))

        #save as csv file with header and SNR values as columns
        resolution05 = np.vstack((model_name,resolution05))
        resolution0143 = np.vstack((model_name,resolution0143))
        header = ['ours','ETOMO','AreTomo','FBP','FBP_no_deformed','FBP_est_deformed']
        header= 'MODEL_NAME'+','+','.join(header)
        pd_resoluton05 = pd.DataFrame(resolution05.T,columns=header.split(','))
        pd_resoluton0143 = pd.DataFrame(resolution0143.T,columns=header.split(','))
        pd_resoluton05.to_csv(os.path.join(config.path_save,'evaluation','resolution05.csv'),index=False)
        pd_resoluton0143.to_csv(os.path.join(config.path_save,'evaluation','resolution0143.csv'),index=False)



if __name__ == '__main__':
    main()