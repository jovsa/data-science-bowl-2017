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



def get_inputs():
    num_chunks = 0

    inputs = {}
    labels = pd.read_csv(LABELS)
    input_features = {}

    for features in glob.glob(DATA_PATH + '*_transfer_values.npy')[0:10]:
        n = re.match('([a-f0-9].*)_transfer_values.npy', os.path.basename(features))
        patient_id = n.group(1)
        predictions = np.load(DATA_PATH + patient_id + '_predictions.npy')
        transfer_values = np.load(DATA_PATH + patient_id + '_transfer_values.npy')
        feature_val = np.append(predictions, transfer_values)
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
                           reg_lambda=0.05)

    clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=50)
    return clf


def make_submission():
    inputs = get_inputs()


    print(x.dtype)
    print(y.dtype)
    trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=42, stratify=y, test_size=0.20)

    print(type(trn_x), type(val_x), type(trn_y), type(val_y))
    print((trn_x).shape, (val_x).shape, (trn_y).shape, (val_y).shape)
    clf = train_xgboost(trn_x, val_x, trn_y, val_y)


if __name__ == '__main__':
    start_time = time.time()

    OUTPUT_PATH = '/kaggle/dev/data-science-bowl-2017-data/submissions/'
    DATA_PATH = '/kaggle/dev/data-science-bowl-2017-data/stage1_features/'
    TENSORBOARD_SUMMARIES = '/kaggle/dev/data-science-bowl-2017-data/tensorboard_summaries/'
    MODELS = '/kaggle_2/luna/luna16/models/'
    MODEL_CHECKPOINTS = '/kaggle/dev/data-science-bowl-2017-data/models/checkpoints/'
    MODEL_PATH = '/kaggle_2/luna/luna16/models/e03f0475-091e-4821-862e-ae5303d670c8/'
    STAGE1 = '/kaggle/dev/data-science-bowl-2017-data/stage1/'
    LABELS = '/kaggle/dev/data-science-bowl-2017-data/stage1_labels.csv'
    STAGE1_SUBMISSION = '/kaggle/dev/data-science-bowl-2017-data/stage1_sample_submission.csv'
    NAIVE_SUBMISSION = '/kaggle/dev/jovan/data-science-bowl-2017/data-science-bowl-2017/submissions/naive_submission.csv'

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
