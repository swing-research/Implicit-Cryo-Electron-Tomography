'''
The base class for loading the data and preprocessing it.
'''

import os
import numpy as np
import mrcfile
from skimage.transform import resize



class DataPreprocessor:
    """
    This class is used to load the data and preprocess it
    Used as a base class for the different datasets.
    """
    def __init__(self, config):
        self.config = config
        self.angles = None
        self.projections = None
        self.setup_directories()
        self.process_data()
        self.save_data()

    def setup_directories(self):
        """
        This function creates the directories for saving the data.
        """
        for dir_name in [self.config.path_save_data, self.config.path_save_data + "projections/", self.config.path_save_data + "projections/noisy/",
                            self.config.path_save_data + "projections/clean/", self.config.path_save_data + "projections/deformed/",
                            self.config.path_save_data + "volumes/", self.config.path_save_data + "volumes/clean/",
                            self.config.path_save_data + "deformations/"]:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)


    def downsample(self,projection, n1, n2):
        """
        Function to downsample the projection
        """

        projection_downsamples = np.zeros((projection.shape[0], n1, n2))
        for index ,proj in enumerate(projection):
            proj = resize(proj, (n1, n2), anti_aliasing=True)
            if self.config.transpose:
                projection_downsamples[index, :, :] = proj.T
            else:
                projection_downsamples[index, :, :] = proj
        
        return projection_downsamples
    
    def crop(self, projection, n1, n2):
        """
        Function to crop the projection, we keep the center of the projection
        """
        
        projection_crop = np.zeros((projection.shape[0], n1, n2))

        for i in range(projection.shape[0]):
            projection_crop[i, :, :] = projection[i, projection.shape[1] // 2 - n1 // 2:projection.shape[1] // 2 + n1 // 2,
                                                 projection.shape[2] // 2 - n2 // 2:projection.shape[2] // 2 + n2 // 2]
            
        return projection_crop
    
    def get_angles(self, angle_file):
        """
        Function to generate the angles from the angle file or from a range. Prioritize the angle file.
        """
        

        if angle_file is None:
            angles = np.linspace(self.config.view_angle_min, self.config.view_angle_max, self.config.Nangles)
        else:
            angles = np.load(angle_file)

        return angles
    
    def save_data(self):
        np.save(self.config.path_save_data + "projections.npy", self.projections)
        np.save(self.config.path_save_data + "angles.npy", self.angles)
        np.savetxt(self.config.path_save_data + "angles.txt", self.angles)

        out = mrcfile.new(self.config.path_save_data + "projections.mrc", self.projections.astype(np.float32),
                            overwrite=True)
        out.close()


    def process_data(self):
        """
        Function use to process the real data, and store it in angles and projections variables
        """
        n1, n2 = self.config.n1, self.config.n2

        path_volume = f"./datasets/{self.config.volume_name}/{self.config.volume_file}"
        projection = np.double(mrcfile.open(path_volume).data)



        if self.config.downsample:
            print("Downsampling")
            projection_downsamples = self.downsample(projection, n1, n2)
        else:
            projection_downsamples = self.crop(projection, n1, n2)

        # TODO: Add denoising here
        if self.config.invert_projections:
            projection_downsamples = np.max(projection_downsamples) - projection_downsamples

        self.projections = projection_downsamples
        self.angles = self.get_angles(self.config.angle_file)