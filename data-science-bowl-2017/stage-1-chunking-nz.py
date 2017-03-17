import glob
import os
import re
import pandas as pd
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

import skimage.transform
import scipy.ndimage
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from tqdm import tqdm
import time
from datetime import timedelta

import scipy.misc
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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

def chunk_nz():

    DATA_PATH = '/kaggle/dev/data-science-bowl-2017-data/stage1_processed/'
    OUTPUT_FOLDER_ORIGINAL = '/kaggle_2/stage1_processed_chunks/'
    OUTPUT_FOLDER_NZ = '/kaggle_2/stage1_processed_chunks_nz/'
    PATIENT_SCANS = 'scan_segmented_lungs_fill_'
    CHUNK_SIZE = 64
    NUM_CLASSES = 7
    OVERLAP_PERCENTAGE = 0.3


    completed_patients = []
    for patients in glob.glob(OUTPUT_FOLDER_ORIGINAL + '*_X.npy'):
        n = re.match('([a-f0-9].*)_X.npy', os.path.basename(patients))
        completed_patients.append(n.group(1))

    for folder in tqdm(glob.glob(DATA_PATH + PATIENT_SCANS + '*')):
        m = re.match(PATIENT_SCANS +'([a-f0-9].*).npy', os.path.basename(folder))
        scans = np.load(DATA_PATH + m.group(0))
        patient_uid = m.group(1)

        if patient_uid in completed_patients:
            print('Skipping already processed patient {}'.format(patient_uid))
            continue


        chunk_counter = 1
        step_size = int((CHUNK_SIZE*(1-OVERLAP_PERCENTAGE)))
        num_chunks_0 = int((scans.shape[0])/(step_size)) + 1
        num_chunks_1 = int((scans.shape[1])/(step_size)) + 1
        num_chunks_2 = int((scans.shape[2])/(step_size)) + 1
        chunk_list = []

        start_index_0 = 0
        end_index_0 = 0
        for i in range(0, num_chunks_0):
            end_index_0 = start_index_0 + CHUNK_SIZE

            start_index_1 = 0
            end_index_1 = 0
            for j in range(0, num_chunks_1):
                end_index_1 = start_index_1 + CHUNK_SIZE

                start_index_2 = 0
                end_index_2 = 0
                for k in range(0, num_chunks_2):
                    end_index_2 = start_index_2 + CHUNK_SIZE

                    end_index_0 = scans.shape[0] if  (end_index_0 > scans.shape[0]) else end_index_0
                    end_index_1 = scans.shape[1] if  (end_index_1 > scans.shape[1]) else end_index_1
                    end_index_2 = scans.shape[2] if  (end_index_2 > scans.shape[2]) else end_index_2

                    chunk = np.full((CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE), -1000.0)

                    end_index_0_chunks = end_index_0 - start_index_0
                    end_index_1_chunks = end_index_1 - start_index_1
                    end_index_2_chunks = end_index_2 - start_index_2

                    chunk[0:end_index_0_chunks, 0:end_index_1_chunks, 0:end_index_2_chunks] = scans[start_index_0:end_index_0, start_index_1:end_index_1, start_index_2:end_index_2]
                    chunk_list.append(chunk)

                    chunk_counter += 1
                    start_index_2 += step_size
                start_index_1 += step_size
            start_index_0 += step_size

        X = np.ndarray([len(chunk_list), CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE], dtype=np.int16)
        Y = np.zeros([len(chunk_list), NUM_CLASSES], dtype=np.int16)
        for m in range(0,len(chunk_list)):
            X[m,:,:] = chunk_list[m]

        np.save(OUTPUT_FOLDER_ORIGINAL + patient_uid + '_X.npy', X)
        np.save(OUTPUT_FOLDER_ORIGINAL + patient_uid + '_Y.npy', Y)

        print('processed patient:', patient_uid  , '_original shape:', scans.shape )
        print('_num_chunks:', len(chunk_list), '_X.shape:', X.shape, '_Y.shape:', Y.shape)

        # Normalizing and Zero Centering
        X_nz = normalize(X)
        X_nz = zero_center(X_nz)
        np.save(OUTPUT_FOLDER_NZ + patient_uid + '_X.npy', X_nz)
        np.save(OUTPUT_FOLDER_NZ + patient_uid + '_Y.npy', Y)

        # Clearning memory
        del X,Y,X_nz






if __name__ == '__main__':
    start_time = time.time()
    DATA_PATH = '/kaggle/dev/data-science-bowl-2017-data/stage1_processed/'
    OUTPUT_FOLDER_ORIGINAL = '/kaggle_2/stage1_processed_chunks/'
    OUTPUT_FOLDER_NZ = '/kaggle_2/stage1_processed_chunks_nz/'
    PATIENT_SCANS = 'scan_segmented_lungs_fill_'
    CHUNK_SIZE = 64
    NUM_CLASSES = 7
    OVERLAP_PERCENTAGE = 0.6

    chunk_nz()
    end_time = time.time()
    print("Total Time usage: " + str(timedelta(seconds=int(round(end_time - start_time)))))
