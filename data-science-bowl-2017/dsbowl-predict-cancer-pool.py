import glob
import os
import csv
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
import math
from sklearn import model_selection
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier as RF
import scipy as sp
from sklearn.decomposition import PCA
import sklearn.metrics

def get_patient_labels(patient_ids):
    labels = pd.read_csv(LABELS)
    input_labels = {}
    for patient_id in patient_ids:
        input_labels[patient_id] = int(labels.loc[labels['id'] == patient_id, 'cancer'])
    return input_labels

def get_patient_features(patient_ids):
    input_features = {}

    num_patients = len(patient_ids)
    patient_count = 0
    for patient_id in patient_ids:
        scans = np.load(DATA_SCANS_PATH + PATIENT_SCANS + patient_id + '.npy')
        step_size = int(CHUNK_SIZE * (1 - OVERLAP_PERCENTAGE))
        num_chunks_0 = int(scans.shape[0] / step_size) + 1
        num_chunks_1 = int(scans.shape[1] / step_size) + 1
        num_chunks_2 = int(scans.shape[2] / step_size) + 1

        predictions = np.array(np.load(DATA_PATH + patient_id + '_predictions.npy'))
        transfer_values = np.array(np.load(DATA_PATH + patient_id + '_transfer_values.npy'))
        transfer_values_cube = np.ndarray(shape=(num_chunks_0, num_chunks_1, num_chunks_2, TRANSFER_VALUES_SHAPE + NUM_CLASSES), dtype=np.float32)

        chunk_count = 0
        for i in range(0, num_chunks_0):
            for j in range(0, num_chunks_1):
                for k in range(0, num_chunks_2):
                    feature = np.ndarray(shape=(TRANSFER_VALUES_SHAPE + NUM_CLASSES), dtype=np.float32)
                    feature[0: TRANSFER_VALUES_SHAPE] = transfer_values[chunk_count]
                    feature[TRANSFER_VALUES_SHAPE : TRANSFER_VALUES_SHAPE + NUM_CLASSES] = predictions[chunk_count]
                    transfer_values_cube[i, j, k, :] = feature
                    chunk_count = chunk_count + 1
        # print('Num chunks calculated: ', (num_chunks_2 * num_chunks_1 * num_chunks_0))
        # print('Num chunks predictions: ', predictions.shape[0])
        # print('transfer_values_cube.shape', transfer_values_cube.shape)

        PIECE_SIZE_Z = math.ceil(transfer_values_cube.shape[0] / NUM_PIECES_Z)
        PIECE_SIZE_Y = math.ceil(transfer_values_cube.shape[1] / NUM_PIECES_Y)
        PIECE_SIZE_X = math.ceil(transfer_values_cube.shape[2] / NUM_PIECES_X)

        patient_features = np.zeros(shape=(NUM_PIECES_Z, NUM_PIECES_Y, NUM_PIECES_X, TRANSFER_VALUES_SHAPE + NUM_CLASSES), dtype=np.float32)

        for i in range(NUM_PIECES_Z):
            for j in range(NUM_PIECES_Y):
                for k in range(NUM_PIECES_X):
                    z1 = i * PIECE_SIZE_Z
                    z2 = z1 + PIECE_SIZE_Z

                    y1 = j * PIECE_SIZE_Y
                    y2 = y1 + PIECE_SIZE_Y

                    x1 = k * PIECE_SIZE_X
                    x2 = x1 + PIECE_SIZE_X

                    z2 = transfer_values_cube.shape[0] if z2 > transfer_values_cube.shape[0] else z2
                    y2 = transfer_values_cube.shape[1] if y2 > transfer_values_cube.shape[1] else y2
                    x2 = transfer_values_cube.shape[2] if x2 > transfer_values_cube.shape[2] else x2

                    patient_features[i, j, k] = np.mean(transfer_values_cube[z1:z2, y1:y2, x1:x2], axis=(2, 1, 0))

        input_features[patient_id] = patient_features.flatten()
        patient_count = patient_count + 1

        print('Loaded data for patient {}/{}'.format(patient_count, num_patients))
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
                           reg_lambda=0.5)
    clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=50)
    return clf

def make_submission():
    print('Loading data..')
    time0 = time.time()
    patient_ids = set()
    for file_path in glob.glob(DATA_PATH + "*_transfer_values.npy"):
        filename = os.path.basename(file_path)
        patient_id = re.match(r'([a-f0-9].*)_transfer_values.npy', filename).group(1)
        patient_ids.add(patient_id)

    sample_submission = pd.read_csv(STAGE1_SUBMISSION)
    #df = pd.merge(sample_submission, patient_ids_df, how='inner', on=['id'])
    test_patient_ids = set(sample_submission['id'].tolist())
    train_patient_ids = patient_ids.difference(test_patient_ids)
    train_inputs = get_patient_features(train_patient_ids)
    train_labels = get_patient_labels(train_patient_ids)

    num_patients = len(train_patient_ids)
    X = np.ndarray(shape=(num_patients, (NUM_PIECES_Z * NUM_PIECES_Y * NUM_PIECES_X) * (TRANSFER_VALUES_SHAPE + NUM_CLASSES)), dtype=np.float32)
    Y = np.ndarray(shape=(num_patients), dtype=np.float32)

    count = 0
    for key in train_inputs.keys():
        X[count] = train_inputs[key]
        Y[count] = train_labels[key]
        count = count + 1

    print('Loaded train data for {} patients'.format(count))
    print("Total time to load data: " + str(timedelta(seconds=int(round(time.time() - time0)))))
    print('\nSplitting data into train, validation')
    train_x, validation_x, train_y, validation_y = model_selection.train_test_split(X, Y, random_state=42, stratify=Y, test_size=0.20)

    del X
    del Y

    print('train_x: {}'.format(train_x.shape))
    print('validation_x: {}'.format(validation_x.shape))
    print('train_y: {}'.format(train_y.shape))
    print('validation_y: {}'.format(validation_y.shape))

    print('\nTraining..')
    clf = train_xgboost(train_x, validation_x, train_y, validation_y)

    timestamp = str(int(time.time()))

    print('\nCreating feature importance plot')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xgb.plot_importance(clf, ax=ax, max_num_features=1000)
    ax.plot()
    savefig('feature_importance_plot_' + timestamp +  '.png')

    print('\nPredicting on validation set')
    validation_y_predicted = clf.predict(validation_x)
    validation_log_loss = sklearn.metrics.log_loss(validation_y, validation_y_predicted, eps=1e-15)
    print('Post-trian validation log loss: {}'.format(validation_log_loss))

    del train_x, train_y, validation_x, validation_y
    #print(validation_y)
    #print(validation_y_predicted)

    num_patients = len(test_patient_ids)
    test_inputs = get_patient_features(test_patient_ids)
    X = np.ndarray(shape=(num_patients, FEATURES_SHAPE * FEATURES_SHAPE), dtype=np.float32)

    #timestamp = str(int(time.time()))
    filename = OUTPUT_PATH + 'submission-' + timestamp + ".csv"

    with open(filename, 'w') as csvfile:
        submission_writer = csv.writer(csvfile, delimiter=',')
        submission_writer.writerow(['id', 'cancer'])

        print('\nPredicting on test set')
        for key in test_inputs.keys():
            x = test_inputs[key]
            y = clf.predict([x])
            submission_writer.writerow([key, y[0]])

    print('Generated submission file: {}'.format(filename))

if __name__ == '__main__':
    start_time = time.time()
    OUTPUT_PATH = '/kaggle/dev/data-science-bowl-2017-data/submissions/'
    DATA_PATH = '/kaggle_3/stage1_features_v5/'
    DATA_SCANS_PATH = '/kaggle_3/stage1_processed_unseg/'
    PATIENT_SCANS = 'scan_lungs_'
    LABELS = '/kaggle/dev/data-science-bowl-2017-data/stage1_labels.csv'
    STAGE1_SUBMISSION = '/kaggle/dev/data-science-bowl-2017-data/stage1_sample_submission.csv'

    NUM_CLASSES = 4
    TRANSFER_VALUES_SHAPE = 1000
    CHUNK_SIZE = 48
    OVERLAP_PERCENTAGE = 0.7

    NUM_PIECES_Z = 10
    NUM_PIECES_Y = 10
    NUM_PIECES_X = 10

    make_submission()
    end_time = time.time()
    print("Total Time usage: " + str(timedelta(seconds=int(round(end_time - start_time)))))
