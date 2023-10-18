"""
This is data preprocessing package for EMPIAR 10364 dataset.
where we need to load multiple mrc files and with multiple stacks
"""

import mrcfile
from .data_preprocessor import DataPreprocessor
import numpy as np
import os



class DataPreprocessorEMP10364(DataPreprocessor):
    """
    The class to preprocess the EMPIAR 10364 dataset. 
    """
    def process_data(self):
        """
        Function use to process the real data,
        """
        n1, n2 = self.config.n1, self.config.n2
        

        path_projections = f"./datasets/{self.config.volume_name}"


        angle_values = np.zeros(61*5)
        projections = np.zeros((61*5,3710,3838))

        for index ,f in enumerate(os.listdir(path_projections)):
            print(f)
            if f.endswith(".mrc"):
                mrc_file = mrcfile.open(path_projections+'/'+f).data
                projections[index*5: (index+1)*5] = mrc_file
                angle_values[index*5: (index+1)*5] += float(f.split('_')[-2])



        if self.config.downsample:
            print("Downsampling")
            projections = super().downsample(projections, n1, n2)

        if self.config.invert_projections:
            projections = np.max(projections) - projections



        index_sort = np.argsort(angle_values)
        angles = angle_values[index_sort]
        projection_sorted = projections[index_sort]

        self.angles = angles
        self.projections = projection_sorted


        
