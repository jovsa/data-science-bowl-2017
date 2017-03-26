import glob
import os
import re
import pandas as pd
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from datetime import timedelta
import sys
import datetime
import tensorflow as tf
import math
from sklearn import cross_validation
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier as RF
import scipy as sp
from sklearn.decomposition import PCA
import sklearn.metrics

def perform_PCA(input_image):
    n_components = 1000
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(input_image)
    patient_data_pca = pca.transform(input_image)
    return patient_data_pca

def get_inputs():
    labels = pd.read_csv(LABELS)
    input_features = {}

    for features in glob.glob(DATA_PATH + '*_transfer_values.npy'):
        n = re.match('([a-f0-9].*)_transfer_values.npy', os.path.basename(features))
        patient_id = n.group(1)
        predictions = np.array([np.mean(np.load(DATA_PATH + patient_id + '_predictions.npy'), axis=0)])
        transfer_values = np.array(np.load(DATA_PATH + patient_id + '_transfer_values.npy'))
        transfer_values = sp.misc.imresize(transfer_values, (1150, 1150))
        transfer_values = transfer_values.flatten()
        feature_val = transfer_values
        try:
            label_val = int(labels.loc[labels['id'] == patient_id, 'cancer'])
        except TypeError:
            continue
        input_features[patient_id] = [feature_val, label_val]
        print('Patient {} predictions {} transfer_values {}'.format(patient_id, predictions.shape, transfer_values.shape))

    return input_features

def train_xgboost(trn_x, val_x, trn_y, val_y):

    clf = xgb.XGBRegressor(max_depth=10,
                           gamma=0.5,
                           objective="binary:logistic",
                           n_estimators=1500,
                           min_child_weight=6,
                           learning_rate=0.005,
                           nthread=8,
                           subsample=0.80,
                           colsample_bytree=0.80,
                           seed=79,
                           max_delta_step=1,
                           reg_alpha=0.1,
                           reg_lambda=0.5)
    clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=50)
    return clf


def make_submission():
    inputs = get_inputs()

    x = np.array([inputs[keys][0]for keys in inputs.keys()])
    y = np.array([inputs[keys][1] for keys in inputs.keys()])

    trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=42, stratify=y, test_size=0.20)
    clf = train_xgboost(trn_x, val_x, trn_y, val_y)
    val_y_pred = clf.predict(val_x)

    for i in range(val_y.shape[0]):
        print("val_y:", val_y[i], "val_y_pred:",val_y_pred[i])



if __name__ == '__main__':
    start_time = time.time()

    OUTPUT_PATH = '/kaggle/dev/data-science-bowl-2017-data/submissions/'
    DATA_PATH = '/kaggle/dev/data-science-bowl-2017-data/stage1_features_v2/'
    LABELS = '/kaggle/dev/data-science-bowl-2017-data/stage1_labels.csv'
    STAGE1_SUBMISSION = '/kaggle/dev/data-science-bowl-2017-data/stage1_sample_submission.csv'

    #globals initializing
    FLAGS = tf.app.flags.FLAGS

    ## Prediction problem specific
    tf.app.flags.DEFINE_integer('num_classes', 2,
                                """Number of classes to predict.""")
    tf.app.flags.DEFINE_integer('batch_size', 32,
                                """Number of items in a batch.""")
    tf.app.flags.DEFINE_float('require_improvement_percentage', 0.20,
                                """Percent of max_iterations after which optimization will be halted if no improvement found""")
    tf.app.flags.DEFINE_float('iteration_analysis_percentage', 0.10,
                                """Percent of max_iterations after which analysis will be done""")

    ## Tensorflow specific
    tf.app.flags.DEFINE_integer('num_gpus', 2,
                                """How many GPUs to use.""")
    tf.app.flags.DEFINE_boolean('log_device_placement', False,
                                """Whether to log device placement.""")
    tf.app.flags.DEFINE_boolean('allow_soft_placement', True,
                                """Whether to allow soft placement of calculations by tf.""")
    tf.app.flags.DEFINE_boolean('allow_growth', True,
                                """Whether to allow GPU growth by tf.""")

    make_submission()
    end_time = time.time()
    print("Total Time usage: " + str(timedelta(seconds=int(round(end_time - start_time)))))
