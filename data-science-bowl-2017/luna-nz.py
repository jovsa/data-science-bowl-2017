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

DATA_PATH_PREPROCESS = '/kaggle_2/luna/luna16/data/pre_processed_chunks_segmented/'
DATA_PATH_POSTPROCESS = '/kaggle_2/luna/luna16/data/pre_processed_chunks_nz_segmented/'

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
    
    patients_processed_files = glob.glob(DATA_PATH_POSTPROCESS + '[0-9\.]*_X.npy')
    patients_processed = set()
    for filename in patients_processed_files:
        m = re.match(r'([0-9\.]*)_X.npy', os.path.basename(filename))
        patients_processed.add(m.group(1))
    
    count = 0
    num_chunks = 0
    for patient_id in patient_ids:
        if patient_id in patients_processed:
            print('Skipping already processed patient {}'.format(patient_id))
            count = count + 1
            continue
        print('Processing patient {}'.format(patient_id))
        
        x = np.load(PATH + patient_id + '_X.npy').astype(np.float32, copy=False)
        
        x = normalize(x)
        x = zero_center(x)
    
        np.save(DATA_PATH_POSTPROCESS + patient_id + '_X.npy', x)
        
        count = count + 1
        num_chunks = num_chunks + x.shape[0]
        print('Processed {}/{} patients/chunks'.format(count, num_chunks))
        
patient_ids = get_ids(DATA_PATH_PREPROCESS)
process_data(patient_ids, DATA_PATH_PREPROCESS)
