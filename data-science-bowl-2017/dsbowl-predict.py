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

def predict_features():
    completed_patients = []
    for patients in glob.glob(OUTPUT_PATH + '*_X.npy'):
        n = re.match('([a-f0-9].*)_X.npy', os.path.basename(patients))
        completed_patients.append(n.group(1))

    for folder in tqdm(glob.glob(DATA_PATH + PATIENT_SCANS + '*')[0:1]):
        m = re.match(PATIENT_SCANS +'([a-f0-9].*).npy', os.path.basename(folder))
        patient_uid = m.group(1)

        if patient_uid in completed_patients:
            print('Skipping already processed patient {}'.format(patient_uid))
            continue

        scans = np.load(DATA_PATH + m.group(0))
        chunk_counter = 1
        step_size = int(CHUNK_SIZE * (1 - OVERLAP_PERCENTAGE))
        num_chunks_0 = int(scans.shape[0] / step_size) + 1
        num_chunks_1 = int(scans.shape[1] / step_size) + 1
        num_chunks_2 = int(scans.shape[2] / step_size) + 1
        chunk_list = []

        start_index_0 = 0
        end_index_0 = 0
        for i in range(0, num_chunks_0):
            coordZ1 = i * step_size
            coordZ2 = coordZ1 + CHUNK_SIZE

            for j in range(0, num_chunks_1):
                coordY1 = j * step_size
                coordY2 = coordY1 + CHUNK_SIZE

                for k in range(0, num_chunks_2):
                    coordX1 = k * step_size
                    coordX2 = coordX1 + CHUNK_SIZE
                    
                    coordZ2 = scans.shape[0] if  (coordZ2 > scans.shape[0]) else coordZ2
                    coordY2 = scans.shape[1] if  (coordY2 > scans.shape[1]) else coordY2
                    coordX2 = scans.shape[2] if  (coordX2 > scans.shape[2]) else coordX2

                    # print("Chunk {}, coords ({}, {}) ({}, {}) ({}, {})".format(chunk_counter, coordX1, coordX2, coordY1, coordY2, coordZ1, coordZ2))

                    chunk = np.full((CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE), -1000.0)
                    chunk[0:coordZ2-coordZ1, 0:coordY2-coordY1, 0:coordX2-coordX1] = scans[coordZ1:coordZ2, coordY1:coordY2, coordX1:coordX2]
                    chunk_list.append(chunk)

                    chunk_counter += 1

        X = np.ndarray([len(chunk_list), CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE], dtype=np.float32)

        for m in range(0, len(chunk_list)):
            X[m, :, :] = chunk_list[m]

        # Normalizing and Zero Centering
        X = X.astype(np.float32, copy=False)
        X = normalize(X)
        X = zero_center(X)
        np.save(OUTPUT_PATH + patient_uid + '_X.npy', X)
      
        
        del X

if __name__ == '__main__':
    start_time = time.time()
   
    DATA_PATH = '/kaggle/dev/data-science-bowl-2017-data/stage1_processed/'
    OUTPUT_PATH = '/kaggle/dev/data-science-bowl-2017-data/stage1_features/'
    PATIENT_SCANS = 'scan_segmented_lungs_fill_'
    CHUNK_SIZE = 64
    NUM_CLASSES = 7
    OVERLAP_PERCENTAGE = 0.3

    predict_features()
    end_time = time.time()
    print("Total Time usage: " + str(timedelta(seconds=int(round(end_time - start_time)))))
