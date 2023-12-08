"""
File to run the simulation with SHREC2021 model_0 volume.

It creates the dataset (projections, FBP), train ICE-TIDE and compare the results with AreTomo and Etomo.
"""

import train
import data_generation
import configs.config_shrec_dataset as config_file


config = config_file.get_config()

# Make the data
# data_generation.data_generation(config)

# Train ICE-TIDE
train.train(config)


# Compare the results and save the figures

