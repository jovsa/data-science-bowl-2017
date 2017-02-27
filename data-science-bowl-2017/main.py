# main

import helpers.helpers as helpers
import helpers.cache as cache
import helpers.download as download
import helpers.inception as inception
from helpers.inception import transfer_values_cache


import numpy as np
import pandas as pd
import re
import sys
import datetime


import matplotlib
# Force matplotlib to not use any Xwindows backend, so that you can output graphs
matplotlib.use('Agg')

import cv2
import dicom
import os
import xgboost as xgb
#import mxnet as mx
from sklearn import cross_validation
import glob
from matplotlib import pyplot as plt
import math
from sklearn.decomposition import PCA
from time import time

import tensorflow as tf

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

def normalize_scans(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def normalize_general(image):
    MIN_BOUND = np.min(image)
    MAX_BOUND = np.max(image)
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    return image

def zero_center(image):
    PIXEL_MEAN = 0.25
    image = image - PIXEL_MEAN
    return image

def train_xgboost():
    ids = list()
    for s in glob.glob(stage1_features + "*"):
        id = os.path.basename(s)
        id = re.match(r'([a-f0-9].*).npy' , id).group(1)
        ids.append(id)
    ids = pd.DataFrame(ids,  columns=["id"])

    df = pd.read_csv(labels)
    df = pd.merge(df, ids, how='inner', on=['id'])

    x = np.array([np.mean(np.load(stage1_features + s + ".npy"), axis=0) for s in df['id'].tolist()])
    for s in range(0, len(x)):
        x[s] = normalize(x[s])
        x[s] = zero_center(x[s])

    y = df['cancer'].as_matrix()
    trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=42, stratify=y,
                                                                    test_size=0.20)

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

def make_submit():
    clf = train_xgboost()

    ids = list()
    for s in glob.glob(stage1_features + "*"):
        id = os.path.basename(s)
        id = re.match(r'([a-f0-9].*).npy' , id).group(1)
        ids.append(id)
    ids = pd.DataFrame(ids,  columns=["id"])

    submission_sample = pd.read_csv(stage1_submission)
    df = pd.merge(submission_sample, ids, how='inner', on=['id'])
    x = np.array([np.mean(np.load(stage1_features + s + ".npy"), axis=0) for s in df['id'].tolist()])

    for s in range(0, len(x)):
        x[s] = normalize(x[s])
        x[s] = zero_center(x[s])

    pred = clf.predict(x)
    df['cancer'] = pred

    #Submission preparation
    submission = pd.merge(submission_sample, df, how='left', on=['id'])
    submission = submission.iloc[:,(0,2)]
    submission = submission.rename(index=str, columns={"cancer_y": "cancer"})

    # Outputting submission file
    timestamp = datetime.datetime.now().isoformat()
    filename = 'submissions/submission-' + str(timestamp) + ".csv"
    submission.to_csv(filename, index=False)

    # Submission file analysis
    print("----submission file analysis----")
    patient_count = submission['id'].count()
    predecited = submission['cancer'].count()
    print("Total number of patients: " + str(patient_count))
    print("Number of predictions: " + str(predecited))
    print("submission file stored at: " + filename)

def file_exists(id):
    returnVal = True
    for folder in glob.glob(stage1_processed_pca + 'lungs_pca_*'):
        filename = re.match(r'lungs_pca_([a-f0-9].*).npy', os.path.basename(folder))
        file_id = filename.group(1)
        if(file_id == id):
            returnVal = False
    return returnVal

def PCA_transform(patient_data, components):
    if(components >= patient_data.shape[0]):
        n_components = patient_data.shape[0]
    else:
        n_components = components
    h = int(math.sqrt(patient_data.shape[1]))
    pca = PCA(n_components=n_components, svd_solver='randomized',
              whiten=True).fit(patient_data)
    patient_data_pca = pca.transform(patient_data)
    eigenvectors = pca.components_.reshape((n_components, h, h))
    explained_variance_ratio = pca.explained_variance_ratio_
    return patient_data_pca, eigenvectors, explained_variance_ratio

def process_pca():
    t0 = time()
    index = 1
    pca_n_components = 10000 # want to have n_componets == dim[0]
    for folder in glob.glob(stage1_processed + 'segment_lungs_fill_*'):
        t0 = time()
        filename = re.match(r'segment_lungs_fill_([a-f0-9].*).npy', os.path.basename(folder))
        p_id = filename.group(1)
        if(file_exists(p_id)):
            segment_lungs_fill_ = np.load(stage1_processed + filename.group(0))
            segment_lungs_ = np.load(stage1_processed + "segmented_lungs_" + str(filename.group(1)) + ".npy" )
            lungs = segment_lungs_fill_ -  segment_lungs_
            lungs = lungs.reshape(lungs.shape[0], lungs.shape[1]* lungs.shape[2])
            lungs_pca, eigenvectors, _ = PCA_transform(lungs, pca_n_components)
            np.save(stage1_processed_pca + "lungs_pca_" + p_id, lungs_pca)
            print("id: " + p_id + " -> (" + str(index) + "/1595)" + " done in %0.3fs" % (time() - t0))
        else:
            print("already exists, skipping: " + p_id)
        index += 1
    print("total PCA done in %0.3fs" % (time() - t0))


# Helper function for scans to
def img_to_rgb(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret

# Convert grayscale scans to rgb
# (num_scans, w, h) -> (num_scans, w, h, 3)
def scans_to_rgb(scans):
    num_scans, w, h = scans.shape
    reshaped_scans = np.empty((num_scans, w, h, 3), dtype=np.uint8)
    for scn in enumerate(scans):
        reshaped_scans[scn[0]] = img_to_rgb(scn[1])
    return reshaped_scans


def calc_features_inception():
    inception.maybe_download()
    download.maybe_download_and_extract(cifar10_url, cifar_data)
    model = inception.Inception()
    count = 0

    for folder in glob.glob(stage1_processed + 'scan_segmented_lungs_fill_*'):
        p_id = re.match(r'scan_segmented_lungs_fill_([a-f0-9].*).npy', os.path.basename(folder))
        print('Processing patient ' + str(count) + ' id: ' + p_id.group(1))
        data = np.load(stage1_processed + p_id.group(0))
        # print("original: " + str(data.shape))
        data = scans_to_rgb(data)
        data = normalize_scans(data)
        data = zero_center(data)
        data = normalize_general(data)
        # print("after: " + str(data.shape))



        # Scale images because Inception needs pixels to be between 0 and 255,
        data = data * 255.0
        filepath_cache = cifar_data + "cache/inception_cifar10_" + p_id.group(1) + ".pkl"
        # print(np.min(data))
        # print(np.max(data))
        # print("after scalling: " + str(data.shape))
        transfer_values_train = transfer_values_cache(cache_path=filepath_cache, images=data, model=model)
        count = count + 1




if __name__ == '__main__':
    data = '/kaggle/dev/data-science-bowl-2017-data/'
    stage1 = '/kaggle/dev/data-science-bowl-2017-data/stage1/'
    labels = '/kaggle/dev/data-science-bowl-2017-data/stage1_labels.csv'
    stage1_processed = '/kaggle/dev/data-science-bowl-2017-data/stage1_processed/'
    stage1_features = '/kaggle/dev/data-science-bowl-2017-data/stage1_features_mx/'
    stage1_submission = '/kaggle/dev/data-science-bowl-2017-data/stage1_sample_submission.csv'
    naive_submission = '/kaggle/dev/jovan/data-science-bowl-2017/data-science-bowl-2017/submissions/naive_submission.csv'
    stage1_processed_pca = '/kaggle/dev/data-science-bowl-2017-data/stage1_processed_pca/'

    cifar10_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    cifar_data = "/kaggle/dev/data-science-bowl-2017-data/CIFAR-10/"


    #process_pca()
    #calc_features()
    calc_features_inception()
    #make_submit()
    print("done")

# Model Building and Traning

# Predicting on CV

# Predicting on Test

# Post-test Analysis

# Submission
