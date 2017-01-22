# main

import helpers.helpers as helpers

import numpy as np
import pandas as pd
import re
import sys

import cv2
import dicom
import os
import xgboost as xgb
import mxnet as mx
from sklearn import cross_validation
import glob
from matplotlib import pyplot as plt

######################
def pre_process():
    # Pre-processing
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


def get_3d_data(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return np.stack([s.pixel_array for s in slices])


def get_data_id(path):
    sample_image = np.load(path)
    sample_image[sample_image == -2000] = 0

    batch = []
    cnt = 0
    dx = int((40.0/512.0) * sample_image.shape[1])
    ds = sample_image.shape[1]
    for i in range(0, sample_image.shape[0] - 3, 3):
        tmp = []
        for j in range(3):
            img = sample_image[i + j]
            img = 255.0 / np.amax(img) * img
            img = cv2.equalizeHist(img.astype(np.uint8))
            img = img[dx: ds - dx, dx: ds - dx]
            img = cv2.resize(img, (224, 224))
            tmp.append(img)

        tmp = np.array(tmp)
        batch.append(np.array(tmp))

    batch = np.array(batch)
    return batch

def calc_features():
    net = get_extractor()
    count = 0
    for folder in glob.glob(stage1_processed + 'segment_lungs_fill_*'):
        p_id = re.match(r'segment_lungs_fill_([a-f0-9].*).npy', os.path.basename(folder)).group(1)
        print('Processing patient ' + str(count) + ' id: ' + p_id)
        batch = get_data_id(folder)
        feats = net.predict(batch)
        np.save(stage1_features + p_id, feats)
        count = count + 1

def train_xgboost():
    df = pd.read_csv(labels)
    #print(df.head())
    x = np.array([np.mean(np.load(stage1_features + '/segment_lungs_fill_%s.npy' % str(id)), axis=0) for id in df['id'][:2].tolist()])
    print(x.shape)
    # y = df['cancer'].as_matrix()
    # trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=42, stratify=y,
    #                                                                test_size=0.20)
    #
    # clf = xgb.XGBRegressor(max_depth=10,
    #                        n_estimators=1500,
    #                        min_child_weight=9,
    #                        learning_rate=0.05,
    #                        nthread=8,
    #                        subsample=0.80,
    #                        colsample_bytree=0.80,
    #                        seed=4242)
    #
    # clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=50)
    # return clf


def make_submit():
    clf = train_xgboost()

    # df = pd.read_csv('/kaggle/dev/data-science-bowl-2017-data/stage1_sample_submission.csv')
    #
    # x = np.array([np.mean(np.load('/kaggle/dev/data-science-bowl-2017-data/stage1_processed/segmented_lungs_fill%s.npy' % str(id)), axis=0) for id in df['id'].tolist()])
    #
    # pred = clf.predict(x)
    #
    # df['cancer'] = pred
    # df.to_csv('subm1.csv', index=False)
    # print(df.head())


if __name__ == '__main__':
    data = '/kaggle/dev/data-science-bowl-2017-data/'
    stage1 = '/kaggle/dev/data-science-bowl-2017-data/stage1/'
    labels = '/kaggle/dev/data-science-bowl-2017-data/stage1_labels.csv'
    stage1_processed = '/kaggle/dev/data-science-bowl-2017-data/stage1_processed/'
    stage1_features = '/kaggle/dev/data-science-bowl-2017-data/stage1_features_mx/'

    #calc_features()
    make_submit()
    print("done")







# Model Building and Traning

# Predicting on CV

# Predicting on Test

# Post-test Analysis

# Submission
