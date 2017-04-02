import re
import sys
import datetime
import scipy
import numpy as np
import pandas as pd
import cv2
import dicom
import os
import glob
import math
import time
import uuid
from datetime import timedelta
import matplotlib
# Force matplotlib to not use any Xwindows backend, so that you can output graphs
matplotlib.use('Agg')
from sklearn import model_selection, metrics
from matplotlib import pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import pickle

DATA_PATH_PREPROCESS = '/kaggle/dev/data-science-bowl-2017-data/stage1_features_v3/'
DATA_PATH_POSTPROCESS = '/kaggle_3/stage1_features_v3_1_chunked/'
LABELS = '/kaggle/dev/data-science-bowl-2017-data/stage1_labels.csv'

NUM_CLASSES_LUNA = 4

def get_ids(PATH):
    patient_ids = []
    for path in glob.glob(DATA_PATH_PREPROCESS + "*_transfer_values.npy"):
        patient_id = re.match(r'([a-f0-9].*)_transfer_values.npy', os.path.basename(path)).group(1)
        patient_ids.append(patient_id)
    return patient_ids

def chunk_map():
    patient_ids = get_ids(DATA_PATH_PREPROCESS)
    labels = pd.read_csv(LABELS)
    train_patient_ids = set(list(labels['id']))

    patient_count = 1
    chunk_count = 1
    Y_list = []
    X_list = []
    X_dict = {}

    for patient_id in tqdm(patient_ids):
        predictions = np.array(np.load(DATA_PATH_PREPROCESS + patient_id + '_predictions.npy'))
        transfer_values = np.array(np.load(DATA_PATH_PREPROCESS + patient_id + '_transfer_values.npy'))
        features_shape = (transfer_values.shape[0], transfer_values.shape[1] + NUM_CLASSES_LUNA + 1)
        features = np.zeros(shape=features_shape, dtype=np.float32)
        features[:, 0:transfer_values.shape[1]] = transfer_values
        features[:, transfer_values.shape[1]:transfer_values.shape[1] + NUM_CLASSES_LUNA] = predictions
        if patient_id in train_patient_ids:
            label = int(labels.loc[labels['id'] == patient_id, 'cancer'])
        else:
            label = -1

        for chunk_number in range(features.shape[0]):
            features[chunk_number, transfer_values.shape[1] + NUM_CLASSES_LUNA] = chunk_number
            X = features[chunk_number]
            key = patient_id + '/{}_X.npy'.format(chunk_number)
            X_dict[key] = X
            X_list.append(key)
            Y_list.append(label)


        if (patient_count % 100 == 0) or (patient_count == (len(patient_ids))):
            filename = ('X_dict_{}.pkl').format(patient_count)
            output = open(os.path.join(DATA_PATH_POSTPROCESS, 'X_dict/',filename ), 'wb')
            pickle.dump(X_dict, output )
            output.close()
            print('Processed {} chunks'.format(str(chunk_count)))
            del X_dict
            X_dict = {}
        chunk_count = chunk_count + features.shape[0]
        patient_count = patient_count + 1



    pickle.dump(X_list, open(os.path.join(DATA_PATH_POSTPROCESS, 'X_list.pkl'), 'wb'))
    pickle.dump(Y_list, open(os.path.join(DATA_PATH_POSTPROCESS, 'Y_list.pkl'), 'wb'))
    print('Processed {} chunks'.format(str(chunk_count)))



chunk_map()
