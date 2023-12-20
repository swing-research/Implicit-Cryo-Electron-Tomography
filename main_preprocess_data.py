'''
Preprocess data for training and validation
TODO: each data set is different and might require to make a class? 
'''


from data_preprocessing.data_preprocessor import DataPreprocessor
from data_preprocessing.data_preprocessor_emp_10364 import DataPreprocessorEMP10364


from configs.config_realData import get_default_realData_multiresolution
from configs.config_realData_emp_10364 import get_default_realData_10364



if __name__ == "__main__":

    config = get_default_realData_10364()
    data_preprocessor = DataPreprocessorEMP10364(config)


    #config = get_default_realData_multiresolution()
    #data_preprocessor = DataPreprocessor(config)
