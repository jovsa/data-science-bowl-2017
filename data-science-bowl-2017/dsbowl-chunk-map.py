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

DATA_PATH_PREPROCESS = '/kaggle/dev/data-science-bowl-2017-data/stage1_features_v3/'
DATA_PATH_POSTPROCESS = '/kaggle_3/stage1_features_v3_chunked/'
LABELS = '/kaggle/dev/data-science-bowl-2017-data/stage1_labels.csv'

NUM_CLASSES_LUNA = 4

def shuffle_in_unison(arrays, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
    n = len(arrays[0])
    for a in arrays:
        assert(len(a) == n)
    idx = np.random.permutation(n)
    return [a[idx] for a in arrays]

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

    processed_patient_ids = set()
    for folder in os.listdir(DATA_PATH_POSTPROCESS):
        processed_patient_ids.add(folder)

    patient_count = 0
    chunk_count = 0

    for patient_id in patient_ids:
        if patient_id in processed_patient_ids:
            print('Skipping already processed patient {}'.format(patient_id))
            patient_count = patient_count + 1
            continue

        print('Processing patient {}'.format(patient_id))
        patient_folder = os.path.join(DATA_PATH_POSTPROCESS, patient_id)
        if not os.path.exists(patient_folder):
            os.makedirs(patient_folder)

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
            
        for i in range(features.shape[0]):
            features[i, transfer_values.shape[1] + NUM_CLASSES_LUNA] = i
            X = features[i]
            np.save(patient_folder + '/' + str(i) + '_X.npy', X)
            if label != -1:
                np.save(patient_folder + '/' + str(i) + '_Y.npy', label)

        chunk_count = chunk_count + features.shape[0]
        patient_count = patient_count + 1
        print('Processed {}/{} patients'.format(patient_count, len(patient_ids)))
    print('Processed {} chunks'.format(str(chunk_count)))

chunk_map()
