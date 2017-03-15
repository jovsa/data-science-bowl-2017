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

def img_to_rgb(im):
        n, x, y, z = im.shape
        ret = np.empty((n, x, y, z, 1), dtype=np.float32)
        ret[:, :, :, :, 0] = im
        return ret
        
def get_ids():
    ids = set()
    for path in glob.glob(DATA_PATH + '[0-9\.]*_X.npy'):
        patient_id = re.match(r'([0-9\.]*)_X.npy', os.path.basename(path)).group(1)
        ids.add(patient_id)
    return ids

def get_data(patient_ids):
    num_chunks = 0
    
    for patient_id in patient_ids:
        x = np.load(DATA_PATH + patient_id + '_X.npy')
        num_chunks = num_chunks + x.shape[0]
       
    X = np.ndarray([num_chunks, 64, 64, 64, 1], dtype=np.float32)
    Y = np.ndarray([num_chunks, 7], dtype=np.float32)
    
    count = 0
    for patient_id in patient_ids:
        x = np.load(DATA_PATH + patient_id + '_X.npy').astype(numpy.float32, copy=False)
        y = np.load(DATA_PATH + patient_id + '_Y.npy').astype(numpy.float32, copy=False)
        
        X[count : count + x.shape[0], :, :, :, :] = img_to_rgb(x)
        Y[count : count + y.shape[0], :] = y
        
        count = count + x.shape[0]
    
    return X, Y

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def conv3d(inputs,             # The previous layer.
           filter_size,        # Width and height of each filter.
           num_filters,        # Number of filters.
           num_channels,       # 1
           strides,            # [1,1,1,1,1]
           name):
    filters = tf.Variable(tf.truncated_normal([filter_size, filter_size, filter_size, num_channels, num_filters],
                                              dtype=tf.float32, stddev=1e-1), name= name + '_weights')
    conv = tf.nn.conv3d(inputs, filters, strides, padding='SAME', name)
    biases = tf.Variable(tf.constant(0.0, shape=[num_filters], dtype=tf.float32), name= name + '_biases')
    out = tf.nn.bias_add(conv, biases)

    out = tf.nn.relu(out)
    return out, filters

def max_pool_3d(inputs,
                filter_size,  # [1, 2, 2, 2, 1]
                strides,      # [1, 2, 2, 2, 1]
                name):
    return tf.nn.max_pool3d(inputs,
                               ksize=filter_size,
                               strides,
                               padding='SAME',
                               name= name)

def dropout_3d(inputs,
               keep_prob,
               name):
    return tf.nn.dropout(inputs, keep_prob, name=name)

def flatten_3d(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:5].num_elements()   
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def dense_3d(inputs,
             num_inputs,
             num_outputs,
             name):
    weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], dtype=tf.float32, stddev=1e-1), name= name + '_weights')
    biases = tf.Variable(tf.constant(0.0, shape=[num_outputs], dtype=tf.float32), name= name + '_biases')
    layer = tf.matmul(inputs, weights) + biases
    layer = tf.nn.relu(layer)
    return layer

def train_3d_nn():
    patient_ids = get_ids()
    X, Y = get_data(patient_ids)
    
    print(X.shape)
    print(Y.shape)
    ##################################
    # TODO: Normalize, zero-center X #
    ##################################
        
    train_x, validation_x, train_y, validation_y = model_selection.train_test_split(X, Y, random_state=42, stratify=Y,
                                                                    test_size=0.20)
    # Graph construction
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, shape=[None, 64, 64, 64, 1], name='x')
        y = tf.placeholder(tf.float32, shape=[None, FLAGS.num_classes], name='y')
        y_labels = tf.placeholder(tf.float32, shape=[None, FLAGS.num_classes], name='y_labels')
        
        layer1_conv3d_out, layer1_conv3d_weights = conv3d(inputs = x, filter_size = 3, num_filters = 16,
                                                          num_channels = 1, strides = [1, 3, 3, 3, 1],
                                                          'layer1_conv3d')
        
        layer1_maxpool3d_out = max_pool_3d(inputs = layer1_conv3d_out, filter_size = [1, 2, 2, 2, 1],
                                           strides = [1, 2, 2, 2, 1], 'layer1_maxpool3d')
        
        layer2_conv3d_out, layer2_conv3d_weights = conv3d(inputs = layer1_maxpool3d_out, filter_size = 3,
                                                          num_filters = 32, num_channels = 1, strides = [1, 3, 3, 3, 1],
                                                          'layer2_conv3d')
        
        layer2_maxpool3d_out = max_pool_3d(inputs = layer2_conv3d_out, filter_size = [1, 2, 2, 2, 1],
                                           strides = [1, 2, 2, 2, 1], 'layer2_maxpool3d')
        
        layer3_conv3d_out, layer3_conv3d_weights = conv3d(inputs = layer2_maxpool3d_out, filter_size = 3,
                                                          num_filters = 64, num_channels = 1, strides = [1, 3, 3, 3, 1],
                                                          'layer3_conv3d')
        
        layer3_maxpool3d_out = max_pool_3d(inputs = layer3_conv3d_out, filter_size = [1, 2, 2, 2, 1],
                                           strides = [1, 2, 2, 2, 1], 'layer3_maxpool3d')
        
        layer3_dropout3d_out = dropout_3d(layer3_maxpool3d_out, 0.25, 'layer3_dropout3d')

        layer3_flatten3d_out, layer3_flatten3d_features = flatten_3d(layer3_dropout3d_out)
        
        layer4_dense3d_out = dense3d(inputs=layer3_flatten3d_out, num_inputs=layer3_flatten3d_out.shape[1],
                                     num_outputs=512, name='layer4_dense3d')
 
        layer4_dropout3d_out = dropout_3d(layer4_dense3d_out, 0.5, 'layer4_dropout3d')
        
        layer5_dense3d_out = dense3d(inputs=layer4_dropout3d_out, num_inputs=layer4_dropout3d_out.shape[1],
                                     num_outputs=128, name='layer5_dense3d')
        
        layer5_dropout3d_out = dropout_3d(layer5_dense3d_out, 0.5, 'layer5_dropout3d')

        layer6_dense3d_out = dense3d(inputs=layer5_dropout3d_out, num_inputs=layer5_dropout3d_out.shape[1],
                                     num_outputs=2, name='layer6_dense3d')
        
        y_pred = tf.nn.softmax(layer6_dense3d_out)        
        
        
    # Setting up config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = FLAGS.allow_growth
    config.log_device_placement=FLAGS.log_device_placement
    config.allow_soft_placement=FLAGS.allow_soft_placement
    
    
    with tf.Session(graph=graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        
        ## TODO
        
        sess.close()
        
        
if __name__ == '__main__':
    start_time = time.time()
    DATA_PATH = '/kaggle_2/luna/luna16/data/pre_processed_chunks/'
    TENSORBOARD_SUMMARIES = '/kaggle/dev/data-science-bowl-2017-data/tensorboard_summaries'
    
    #globals initializing
    FLAGS = tf.app.flags.FLAGS

    ## Prediction problem specific
    tf.app.flags.DEFINE_integer('num_classes', 7,
                                """Number of classes to predict.""")
    tf.app.flags.DEFINE_integer('batch_size', 10,
                                """Number of items in a batch.""")
    tf.app.flags.DEFINE_integer('max_iterations', 100000,
                                """Number of batches to run.""")
    tf.app.flags.DEFINE_float('require_improvement', 0.20,
                                """Percent of max_iterations after which optimization will be halted if no improvement found""")
    tf.app.flags.DEFINE_float('iteration_analysis', 0.10,
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

    train_3d_nn()
    end_time = time.time()
    print("Total Time usage: " + str(timedelta(seconds=int(round(end_time - start_time)))))
