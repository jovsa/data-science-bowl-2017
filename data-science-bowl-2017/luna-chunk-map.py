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

DATA_PATH_PREPROCESS = '/kaggle_2/luna/luna16/data/pre_processed_chunks_augmented_v4/'
DATA_PATH_POSTPROCESS = '/kaggle_2/luna/luna16/data/pre_processed_chunks_augmented_v4_single/'

def shuffle_in_unison(arrays, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
    n = len(arrays[0])
    for a in arrays:
        assert(len(a) == n)
    idx = np.random.permutation(n)
    return [a[idx] for a in arrays]

def get_ids(PATH):
    ids = []
    for path in glob.glob(PATH + '[0-9\.]*_X.npy'):
        patient_id = re.match(r'([0-9\.]*)_X.npy', os.path.basename(path)).group(1)
        ids.append(patient_id)
    return ids

def chunk_map():
    patient_ids = get_ids(DATA_PATH_PREPROCESS)

    processed_patient_ids = set()
    patient_count = 0
    chunk_count = 0

    for patient_id in patient_ids:
        if patient_id in processed_patient_ids:
            print('Skipping already processed patient {}'.format(patient_id))
            patient_count = patient_count + 1
            continue

        x = np.load(DATA_PATH_PREPROCESS + patient_id + '_X.npy')
        y = np.load(DATA_PATH_PREPROCESS + patient_id + '_Y.npy')

        print(patient_id + " " + str(x.shape[0]))
        for idx in range(x.shape[0]):
            chunk_id = str(uuid.uuid4())
            np.save(DATA_PATH_POSTPROCESS + chunk_id + '_X.npy', x[idx])
            np.save(DATA_PATH_POSTPROCESS + chunk_id + '_Y.npy', y[idx])
        chunk_count = chunk_count + x.shape[0]

    print('Processed {} chunks'.format(str(chunk_count)))
chunk_map()
