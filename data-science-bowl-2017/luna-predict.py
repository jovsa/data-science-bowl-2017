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
    conv = tf.nn.conv3d(inputs, filters, strides, padding='SAME', name=name)
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
                               strides=strides,
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

def get_batch(x, y, batch_size):
        num_images = len(x)
        idx = np.random.choice(num_images,
                               size=batch_size,
                               replace=False)
        x_batch = x[idx]
        y_batch = y[idx]

        return x_batch, y_batch

def get_ids(PATH):
    ids = []
    for path in glob.glob(PATH + '[0-9\.]*_X.npy'):
        patient_id = re.match(r'([0-9\.]*)_X.npy', os.path.basename(path)).group(1)
        ids.append(patient_id)
    return ids

def img_to_rgb(im):
    n, x, y, z = im.shape
    ret = np.empty((n, x, y, z, 1), dtype=np.float32)
    ret[:, :, :, :, 0] = im
    return ret

def predict_prob_test(X, sess):
    num_chunks = len(X)
    prob_pred = np.zeros(shape=[num_chunks, FLAGS.num_classes], dtype=np.float32)
    transfer_values = np.zeros(shape=[num_chunks, 512], dtype=np.float32)
    
    i = 0
    while i < num_chunks:
        j = 1
        feed_dict = {'x': X[i:j],
                     'y_labels': np.zeros([j, FLAGS.num_classes], dtype=np.float32)}

        y_calc, trans_value = sess.run([y, layer4_dense3d_out], feed_dict=feed_dict)
        prob_pred[i:j] = y_calc
        transfer_values[i:j] = trans_value
        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j
    
    return prob_pred, transfer_values
    
def predict_3d_nn():
    time0 = time.time()

    patient_ids = get_ids(DATA_PATH)

    # Graph construction
    graph = tf.Graph()
    with graph.as_default():
        model_name = 'convnet3D_v0.1'
        with tf.name_scope(model_name):
            x = tf.placeholder(tf.float32, shape=[None, 64, 64, 64, 1], name = 'x')
            y = tf.placeholder(tf.float32, shape=[None, FLAGS.num_classes], name = 'y')
            y_labels = tf.placeholder(tf.float32, shape=[None, FLAGS.num_classes], name ='y_labels')

            layer1_conv3d_out, layer1_conv3d_weights = conv3d(inputs = x, filter_size = 3, num_filters = 16,
                                                              num_channels = 1, strides = [1, 3, 3, 3, 1],
                                                              name ='layer1_conv3d')
            print(layer1_conv3d_out)
            layer1_maxpool3d_out = max_pool_3d(inputs = layer1_conv3d_out, filter_size = [1, 2, 2, 2, 1],
                                               strides = [1, 2, 2, 2, 1], name ='layer1_maxpool3d')


            print(layer1_maxpool3d_out)
            layer2_conv3d_out, layer2_conv3d_weights = conv3d(inputs = layer1_maxpool3d_out, filter_size = 3,
                                                              num_filters = 32, num_channels = 16, strides = [1, 3, 3, 3, 1],
                                                              name ='layer2_conv3d')

            print(layer2_conv3d_out)
            layer2_maxpool3d_out = max_pool_3d(inputs = layer2_conv3d_out, filter_size = [1, 2, 2, 2, 1],
                                               strides = [1, 2, 2, 2, 1], name ='layer2_maxpool3d')

            print(layer2_maxpool3d_out)
            layer3_conv3d_out, layer3_conv3d_weights = conv3d(inputs = layer2_maxpool3d_out, filter_size = 3,
                                                              num_filters = 64, num_channels = 32, strides = [1, 3, 3, 3, 1],
                                                              name = 'layer3_conv3d')
            print(layer3_conv3d_out)

            layer3_maxpool3d_out = max_pool_3d(inputs = layer3_conv3d_out, filter_size = [1, 2, 2, 2, 1],
                                               strides = [1, 2, 2, 2, 1], name = 'layer3_maxpool3d')
            print(layer3_maxpool3d_out)

            layer3_dropout3d_out = dropout_3d(layer3_maxpool3d_out, 0.25, 'layer3_dropout3d')
            print(layer3_dropout3d_out)

            layer3_flatten3d_out, layer3_flatten3d_features = flatten_3d(layer3_dropout3d_out)
            print(layer3_flatten3d_out)

            layer4_dense3d_out = dense_3d(inputs=layer3_flatten3d_out, num_inputs=int(layer3_flatten3d_out.shape[1]),
                                         num_outputs=512, name ='layer4_dense3d')
            print(layer4_dense3d_out)

            # Save transfer_values = layer4_dense3d_out on prediction
            layer4_dropout3d_out = dropout_3d(layer4_dense3d_out, 0.5, 'layer4_dropout3d')
            print(layer4_dropout3d_out)

            layer5_dense3d_out = dense_3d(inputs=layer4_dropout3d_out, num_inputs=int(layer4_dropout3d_out.shape[1]),
                                         num_outputs=128, name ='layer5_dense3d')
            print(layer5_dense3d_out)

            layer5_dropout3d_out = dropout_3d(layer5_dense3d_out, 0.5, 'layer5_dropout3d')
            print(layer5_dropout3d_out)

            layer6_dense3d_out = dense_3d(inputs=layer5_dropout3d_out, num_inputs=int(layer5_dropout3d_out.shape[1]),
                                         num_outputs=7, name ='layer6_dense3d')
            print(layer6_dense3d_out)

            y = tf.nn.softmax(layer6_dense3d_out)
            print(y)

            with tf.name_scope('log_loss'):
                log_loss = tf.losses.log_loss(y_labels, y, epsilon=10e-15)
                tf.summary.scalar('log_loss', log_loss)

            optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(log_loss)

        merged = tf.summary.merge_all()
        saver = tf.train.Saver()
    
    # Setting up config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = FLAGS.allow_growth
    config.log_device_placement=FLAGS.log_device_placement
    config.allow_soft_placement=FLAGS.allow_soft_placement
    
    with tf.Session(graph=graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(MODEL_PATH + '/model.meta')
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))
        
        for patient_id in patient_ids[0:50]:
            x_in = np.load(DATA_PATH + patient_id + '_X.npy').astype(np.float32, copy=False)

            X = np.ndarray([x_in.shape[0], 64, 64, 64, 1], dtype=np.float32)
            X[0: x_in.shape[0], :, :, :, :] = img_to_rgb(x_in)
            
            print('\nPatient id: ' + patient_id)
            print('X: {}'.format(X.shape))
            print(x.shape)
            print(y_labels)
            
            feed_dict = {x: X,
                         y_labels: np.zeros([x_in.shape[0], FLAGS.num_classes], dtype=np.float32)}

            predictions, transfer_values = sess.run([y, layer4_dense3d_out], feed_dict=feed_dict)

            print('predictions: ' + str(predictions.shape))
            print('transfer_values: ' + str(transfer_values.shape))

            np.save(OUTPUT_FOLDER + patient_id + '_predictions.npy', predictions)
            np.save(OUTPUT_FOLDER + patient_id + '_transfer_values.npy', transfer_values)

    sess.close()
    
if __name__ == '__main__':
    start_time = time.time()
    DATA_PATH = '/kaggle_2/luna/luna16/data/pre_processed_chunks_nz/'
    OUTPUT_FOLDER = '/kaggle_2/luna/luna16/data/features/'
    TENSORBOARD_SUMMARIES = '/kaggle_2/luna/luna16/data/tensorboard_summaries/'
    MODELS = '/kaggle_2/luna/luna16/models/'
    MODEL_PATH = '/kaggle_2/luna/luna16/models/99309d2f-9916-4fab-8e2e-453880e7a061/'

    #globals initializing
    FLAGS = tf.app.flags.FLAGS

    ## Prediction problem specific
    tf.app.flags.DEFINE_integer('num_classes', 7,
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

    predict_3d_nn()
    end_time = time.time()
    print("Total Time usage: " + str(timedelta(seconds=int(round(end_time - start_time)))))