# main

import helpers.helpers as helpers

import numpy as np
import pandas as pd

import os
import xgboost as xgb
import mxnet as mx
#from sklearn import cross_validation
#import glob
#from matplotlib import pyplot as plt



######################
def pre_process():
    # Pre-processing

    stage1 = '/kaggle/dev/data-science-bowl-2017-data/stage1/'
    labels = '/kaggle/dev/data-science-bowl-2017-data/stage1_labels.csv'
    stage1_processed = '/kaggle/dev/data-science-bowl-2017-data/stage1_processed/'

    stage1_loc = helpers.verify_location(stage1)
    labels_loc = helpers.verify_location(labels)

    patient_data = helpers.folder_explorer(stage1_loc)
    patient_data = pd.DataFrame(list(patient_data.items()), columns=["id", "scans"])
    labels = pd.read_csv(labels_loc)

    data = pd.merge(patient_data, labels, how="inner", on=['id'])
    return

######################

def get_extractor():
    model = mx.model.FeedForward.load('models/resnet-50', 0, ctx=mx.cpu(), numpy_batch_size=1)
    fea_symbol = model.symbol.get_internals()["flatten0_output"]
    feature_extractor = mx.model.FeedForward(ctx=mx.cpu(), symbol=fea_symbol, numpy_batch_size=64,
                                            arg_params=model.arg_params, aux_params=model.aux_params,
                                            allow_extra_params=True)
    return feature_extractor


def calc_features():
    net = get_extractor()







# Model Building and Traning

# Predicting on CV

# Predicting on Test

# Post-test Analysis

# Submission


if __name__ == '__main__':
    calc_features()
    print("done")
