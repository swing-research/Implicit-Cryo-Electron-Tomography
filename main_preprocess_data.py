'''
Preprocess data for training and validation
TODO: each data set is different and might require to make a class? 
'''
from skimage.transform import resize
import os
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
import mrcfile
import imageio
from utils import data_generation, utils_deformation, utils_display
warnings.filterwarnings('ignore') 
from ops.radon_3d_lib import ParallelBeamGeometry3DOpAngles_rectangular

from configs.config_realData import get_default_realData


config = get_default_realData()


if not os.path.exists("results/"):
    os.makedirs("results/")
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


n1 = config.n1
n2 = config.n2


path_volume = "./datasets/"+str(config.volume_name)+"/"+config.volume_file #TODO: maybe requires a redisign

projection = np.double(mrcfile.open(path_volume).data)

projection_n1 = projection.shape[1]
projection_n2 = projection.shape[2]

DOWNSAMPLE = config.downsample

projection_downsamples = np.zeros((projection.shape[0], n1, n2))
if DOWNSAMPLE:
    print("Downsampling")
    for index, proj in  enumerate(projection):

        proj = resize(proj, (n1, n2), anti_aliasing=True)
        if config.transpose:
            projection_downsamples[index,:,:] = proj.T
        else:
            projection_downsamples[index,:,:] = proj
else:
    for i in range(projection.shape[0]):
        projection_downsamples[i,:,:] = projection[i,projection_n1//2-n1//2:projection_n1//2+n1//2,projection_n2//2-n2//2:projection_n2//2+n2//2]


#TODO : Add denoising here


INVERT_PROJECTIONS = config.invert_projections


if INVERT_PROJECTIONS:
    projection_downsamples = np.max(projection_downsamples)-projection_downsamples



angle_file = config.angle_file

if angle_file is None:
    angles = np.linspace(config.view_angle_min, config.view_angle_max, config.Nangles)
else:
    angles = "./datasets/"+str(config.volume_name)+"/"+config.angle_file



np.save(config.path_save_data+"projections.npy",projection_downsamples)
np.save(config.path_save_data+"angles.npy",angles)


out = mrcfile.new(config.path_save_data+"projections.mrc",projection_downsamples.astype(np.float32),overwrite=True)
out.close()