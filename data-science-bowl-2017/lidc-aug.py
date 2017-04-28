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
from sklearn import model_selection
from matplotlib import pyplot as plt
import tensorflow as tf
from tqdm import tqdm

# Fixes "SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame"
pd.options.mode.chained_assignment = None

DATA_PATH_PREPROCESS_NODULES = '/kaggle_2/lidc_idri/data/nodules_chunked/'
DATA_PATH_PREPROCESS_NON_NODULES = '/kaggle_2/lidc_idri/data/non_nodules_chunked/'
DATA_PATH_POSTPROCESS = '/kaggle_2/lidc_idri/data/nodules_nz_aug/'
CHUNK_SIZE = 32

def get_ids(PATH):
    ids = []
    for path in glob.glob(PATH + '[0-9\.]*_X.npy'):
        patient_id = re.match(r'([0-9\.]*)_X.npy', os.path.basename(path)).group(1)
        ids.append(patient_id)
    return ids

def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def zero_center(image):
    PIXEL_MEAN = 0.25
    image = image - PIXEL_MEAN
    return image

def process_data(patient_ids, PATH):
    graph = tf.Graph()
    with graph.as_default():
        img_chunk = tf.placeholder(tf.float32, shape=[CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE], name='img_chunk')
        img_flipped_up_down = tf.image.flip_up_down(img_chunk)
        img_flipped_left_right = tf.image.flip_left_right(img_chunk)
        img_rot_90 = tf.image.rot90(img_chunk)
        img_trans = tf.image.transpose_image(img_chunk)

    chunk_count = 0
    patients_count = len(patient_ids)
    for count in tqdm(range(patients_count)):
        patient_id = patient_ids[count]
        x = np.load(PATH + patient_id + '_X.npy').astype(np.float32, copy=False)
        y = np.load(PATH + patient_id + '_Y.npy')

        x = normalize(x)
        x = zero_center(x)

        assert x.shape[0] == y.shape[0]

        for idx in range(x.shape[0]):
            chunk_id = str(uuid.uuid4())
            np.save(DATA_PATH_POSTPROCESS + chunk_id + '_X.npy', x[idx])
            np.save(DATA_PATH_POSTPROCESS + chunk_id + '_Y.npy', y[idx])

            with tf.Session(graph=graph) as session:
                session.run(tf.global_variables_initializer())
                chunk_flipped_up_down, chunk_flipped_left_right, chunk_rot_90, chunk_trans = session.run([img_flipped_up_down,
                                                                                                          img_flipped_left_right,
                                                                                                          img_rot_90,
                                                                                                          img_trans], feed_dict={img_chunk: x[idx]})
                chunk_id = str(uuid.uuid4())
                np.save(DATA_PATH_POSTPROCESS + chunk_id + '_X.npy', chunk_flipped_up_down)
                np.save(DATA_PATH_POSTPROCESS + chunk_id + '_Y.npy', y[idx])

                chunk_id = str(uuid.uuid4())
                np.save(DATA_PATH_POSTPROCESS + chunk_id + '_X.npy', chunk_flipped_left_right)
                np.save(DATA_PATH_POSTPROCESS + chunk_id + '_Y.npy', y[idx])

                chunk_id = str(uuid.uuid4())
                np.save(DATA_PATH_POSTPROCESS + chunk_id + '_X.npy', chunk_rot_90)
                np.save(DATA_PATH_POSTPROCESS + chunk_id + '_Y.npy', y[idx])

                chunk_id = str(uuid.uuid4())
                np.save(DATA_PATH_POSTPROCESS + chunk_id + '_X.npy', chunk_trans)
                np.save(DATA_PATH_POSTPROCESS + chunk_id + '_Y.npy', y[idx])

        chunk_count = chunk_count + (x.shape[0] * 5)

    print('Processed {} chunks'.format(chunk_count))
patient_ids = get_ids(DATA_PATH_PREPROCESS_NODULES)
process_data(patient_ids, DATA_PATH_PREPROCESS_NODULES)
# patient_ids = get_ids(DATA_PATH_PREPROCESS_NON_NODULES)
# process_data(patient_ids, DATA_PATH_PREPROCESS_NON_NODULES)
