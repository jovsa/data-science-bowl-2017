# main

import helpers.helpers as helpers
import numpy as np
import pandas as pd


# Constants
train_data_folder = '/kaggle/dev/data-science-bowl-2017-data/stage1/'
labels_data_folder = '/kaggle/dev/data-science-bowl-2017-data/stage1_labels.csv'

# Pre-processing
train_loc = helpers.verify_location(train_data_folder)
labels_loc = helpers.verify_location(labels_data_folder)

patient_data = helpers.folder_explorer(train_loc)
patient_data = pd.DataFrame(list(patient_data.items()), columns=["id", "scans"])

labels = pd.read_csv(labels_loc)
train = pd.merge(patient_data, labels, how="inner", on=['id'])

print(train.head())

# Model Building and Traning

# Predicting on CV

# Predicting on Test

# Post-test Analysis

# Submission
