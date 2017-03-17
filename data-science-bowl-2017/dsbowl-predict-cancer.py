import glob
import os
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
import tensorflow as tf
import math

def variable_summaries(var):
    # Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def train_nn():
    num_chunks = 0
    
    for features in glob.glob(DATA_PATH + '*_transfer_values.npy'):
        n = re.match('([a-f0-9].*)_transfer_values.npy', os.path.basename(features))
        patient_id = n.group(1)
        predictions = np.load(DATA_PATH + patient_id + '_predictions.npy')
        transfer_values = np.load(DATA_PATH + patient_id + '_transfer_values.npy')
        print('Patient {} predictions {} transfer_values {}'.format(patient_id, predictions.shape, transfer_values.shape))

               
def make_submission():
    train_nn()
    
if __name__ == '__main__':
    start_time = time.time()
   
    OUTPUT_PATH = '/kaggle/dev/data-science-bowl-2017-data/submissions/'
    DATA_PATH = '/kaggle/dev/data-science-bowl-2017-data/stage1_features/'
    TENSORBOARD_SUMMARIES = '/kaggle/dev/data-science-bowl-2017-data/tensorboard_summaries/'
    MODELS = '/kaggle_2/luna/luna16/models/'
    MODEL_CHECKPOINTS = '/kaggle/dev/data-science-bowl-2017-data/models/checkpoints/'
    MODEL_PATH = '/kaggle_2/luna/luna16/models/e03f0475-091e-4821-862e-ae5303d670c8/'
    STAGE1 = '/kaggle/dev/data-science-bowl-2017-data/stage1/'
    LABELS = '/kaggle/dev/data-science-bowl-2017-data/stage1_labels.csv'
    STAGE1_SUBMISSION = '/kaggle/dev/data-science-bowl-2017-data/stage1_sample_submission.csv'
    NAIVE_SUBMISSION = '/kaggle/dev/jovan/data-science-bowl-2017/data-science-bowl-2017/submissions/naive_submission.csv'  
    
    #globals initializing
    FLAGS = tf.app.flags.FLAGS

    ## Prediction problem specific
    tf.app.flags.DEFINE_integer('num_classes', 2,
                                """Number of classes to predict.""")
    tf.app.flags.DEFINE_integer('batch_size', 32,
                                """Number of items in a batch.""")
    tf.app.flags.DEFINE_float('require_improvement_percentage', 0.20,
                                """Percent of max_iterations after which optimization will be halted if no improvement found""")
    tf.app.flags.DEFINE_float('iteration_analysis_percentage', 0.10,
                                """Percent of max_iterations after which analysis will be done""")

    ## Tensorflow specific
    tf.app.flags.DEFINE_integer('num_gpus', 2,
                                """How many GPUs to use.""")
    tf.app.flags.DEFINE_boolean('log_device_placement', False,
                                """Whether to log device placement.""")
    tf.app.flags.DEFINE_boolean('allow_soft_placement', True,
                                """Whether to allow soft placement of calculations by tf.""")
    tf.app.flags.DEFINE_boolean('allow_growth', True,
                                """Whether to allow GPU growth by tf.""")
    
    make_submission()
    end_time = time.time()
    print("Total Time usage: " + str(timedelta(seconds=int(round(end_time - start_time)))))