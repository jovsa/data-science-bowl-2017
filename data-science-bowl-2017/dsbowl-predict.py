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
import multiprocessing as mp
import random

# Fixes "SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame"
pd.options.mode.chained_assignment = None

import scipy.misc
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def img_to_rgb(im):
    n, x, y, z = im.shape
    ret = np.empty((n, x, y, z, 1), dtype=np.float32)
    ret[:, :, :, :, 0] = im
    return ret

def conv3d(inputs,             # The previous layer.
           filter_size,        # Width and height of each filter.
           num_filters,        # Number of filters.
           num_channels,       # 1
           strides,            # [1,1,1,1,1]
           layer_name):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            filters = tf.get_variable(layer_name + 'weights', shape = [filter_size, filter_size, filter_size, num_channels, num_filters],
                                      initializer = tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32),
                                      regularizer = tf.contrib.layers.l2_regularizer(FLAGS.reg_constant))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0, shape=[num_filters], dtype=tf.float32))
        with tf.name_scope('conv'):
            conv = tf.nn.conv3d(inputs, filters, strides, padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        return out, filters

def max_pool_3d(inputs,
                filter_size,
                strides,
                layer_name):
    with tf.name_scope(layer_name):
        return tf.nn.max_pool3d(inputs,
                                ksize=filter_size,
                                strides=strides,
                                padding='SAME',
                                name='max_pool')

def dropout_3d(inputs,
               keep_prob,
               layer_name):
    with tf.name_scope(layer_name):
        return tf.nn.dropout(inputs, keep_prob, name='dropout')

def flatten_3d(layer, layer_name):
    with tf.name_scope(layer_name):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:5].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
        return layer_flat, num_features

def relu_3d(inputs,
            layer_name):
    with tf.name_scope(layer_name):
        return tf.nn.relu(inputs, name='relu')

def dense_3d(inputs,
             num_inputs,
             num_outputs,
             layer_name):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.get_variable(layer_name + 'weights', shape = [num_inputs, num_outputs],
                              initializer = tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32),
                              regularizer = tf.contrib.layers.l2_regularizer(FLAGS.reg_constant))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0, shape=[num_outputs], dtype=tf.float32))
        with tf.name_scope('Wx_plus_b'):
            layer = tf.matmul(inputs, weights) + biases
        return layer

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

def get_patient_data_chunks(patient_id):
    if os.path.isfile(DATA_PATH + PATIENT_SCANS + patient_id + '.npy'):
        scans = np.load(DATA_PATH + PATIENT_SCANS + patient_id + '.npy')
    elif os.path.isfile(DATA_PATH2 + PATIENT_SCANS + patient_id + '.npy'):
        scans = np.load(DATA_PATH2 + PATIENT_SCANS + patient_id + '.npy')
    else:
        with open("error.out", "a") as myfile:
            myfile.write('Couldnt find scan for patient {}'.format(patient_id))

    chunk_counter = 1
    step_size = int(FLAGS.chunk_size * (1 - OVERLAP_PERCENTAGE))
    num_chunks_0 = int(scans.shape[0] / step_size) + 1
    num_chunks_1 = int(scans.shape[1] / step_size) + 1
    num_chunks_2 = int(scans.shape[2] / step_size) + 1
    chunk_list = []

    for i in range(0, num_chunks_0):
        coordZ1 = i * step_size
        coordZ2 = coordZ1 + FLAGS.chunk_size

        for j in range(0, num_chunks_1):
            coordY1 = j * step_size
            coordY2 = coordY1 + FLAGS.chunk_size

            for k in range(0, num_chunks_2):
                coordX1 = k * step_size
                coordX2 = coordX1 + FLAGS.chunk_size

                coordZ2 = scans.shape[0] if  (coordZ2 > scans.shape[0]) else coordZ2
                coordY2 = scans.shape[1] if  (coordY2 > scans.shape[1]) else coordY2
                coordX2 = scans.shape[2] if  (coordX2 > scans.shape[2]) else coordX2

                chunk = np.full((FLAGS.chunk_size, FLAGS.chunk_size, FLAGS.chunk_size), -1000.0)
                chunk[0:coordZ2-coordZ1, 0:coordY2-coordY1, 0:coordX2-coordX1] = scans[coordZ1:coordZ2, coordY1:coordY2, coordX1:coordX2]
                chunk_list.append(chunk)

                chunk_counter += 1

    X = np.ndarray([len(chunk_list), FLAGS.chunk_size, FLAGS.chunk_size, FLAGS.chunk_size], dtype=np.float32)

    for m in range(0, len(chunk_list)):
        X[m, :, :, :] = chunk_list[m]

    del scans, chunk_list
    # Normalizing and Zero Centering and Adding extra channel
    X = X.astype(np.float32, copy=False)
    X = normalize(X)
    X = zero_center(X)
    X = img_to_rgb(X)

    return X

def worker(patient_uids):
    # Graph construction
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, shape=[None, FLAGS.chunk_size, FLAGS.chunk_size, FLAGS.chunk_size, 1], name = 'x')
        y = tf.placeholder(tf.float32, shape=[None, FLAGS.num_classes], name = 'y')
        y_labels = tf.placeholder(tf.float32, shape=[None, FLAGS.num_classes], name ='y_labels')
        cross_entropy_weights = tf.placeholder(tf.float32, shape=[None], name ='cross_entropy_weights')
        keep_prob = tf.placeholder(tf.float32)

        class_weights_base = tf.ones_like(y_labels)
        class_weights = tf.multiply(class_weights_base , [69838.0/40513.0, 69838.0/29325.0])

        # layer1
        conv1_1_out, conv1_1_weights = conv3d(inputs = x, filter_size = 3, num_filters = 16, num_channels = 1, strides = [1, 3, 3, 3, 1], layer_name ='conv1_1')
        relu1_1_out = relu_3d(inputs = conv1_1_out, layer_name='relu1_1')

        conv1_2_out, conv1_2_weights = conv3d(inputs = relu1_1_out, filter_size = 3, num_filters = 16, num_channels = 16, strides = [1, 3, 3, 3, 1], layer_name ='conv1_2')
        relu1_2_out = relu_3d(inputs = conv1_2_out, layer_name='relu1_2')

        pool1_out = max_pool_3d(inputs = relu1_2_out, filter_size = [1, 2, 2, 2, 1], strides = [1, 2, 2, 2, 1], layer_name ='pool1')

        # layer2
        conv2_1_out, conv2_1_weights = conv3d(inputs = pool1_out, filter_size = 3, num_filters = 32, num_channels = 16, strides = [1, 3, 3, 3, 1], layer_name ='conv2_1')
        relu2_1_out = relu_3d(inputs = conv2_1_out, layer_name='relu2_1')

        conv2_2_out, conv2_2_weights = conv3d(inputs = relu2_1_out, filter_size = 3, num_filters = 32, num_channels = 32, strides = [1, 3, 3, 3, 1], layer_name ='conv2_2')
        relu2_2_out = relu_3d(inputs = conv2_2_out, layer_name='relu2_2')

        pool2_out = max_pool_3d(inputs = relu2_2_out, filter_size = [1, 2, 2, 2, 1], strides = [1, 2, 2, 2, 1], layer_name ='pool2')

        # layer3
        conv3_1_out, conv3_1_weights = conv3d(inputs = pool2_out, filter_size = 3, num_filters = 64, num_channels = 32, strides = [1, 3, 3, 3, 1], layer_name ='conv3_1')
        relu3_1_out = relu_3d(inputs = conv3_1_out, layer_name='relu3_1')

        conv3_2_out, conv3_2_weights = conv3d(inputs = relu3_1_out, filter_size = 3, num_filters = 64, num_channels = 64, strides = [1, 3, 3, 3, 1], layer_name ='conv3_2')
        relu3_2_out = relu_3d(inputs = conv3_2_out, layer_name='relu3_2')

        conv3_3_out, conv3_3_weights = conv3d(inputs = relu3_2_out, filter_size = 3, num_filters = 64, num_channels = 64, strides = [1, 3, 3, 3, 1], layer_name ='conv3_3')
        relu3_3_out = relu_3d(inputs = conv3_3_out, layer_name='relu3_3')

        pool3_out = max_pool_3d(inputs = relu3_3_out, filter_size = [1, 2, 2, 2, 1], strides = [1, 2, 2, 2, 1], layer_name ='pool3')

        # layer4
        conv4_1_out, conv4_1_weights = conv3d(inputs = pool3_out, filter_size = 3, num_filters = 128, num_channels = 64, strides = [1, 3, 3, 3, 1], layer_name ='conv4_1')
        relu4_1_out = relu_3d(inputs = conv4_1_out, layer_name='relu4_1')

        conv4_2_out, conv4_2_weights = conv3d(inputs = relu4_1_out, filter_size = 3, num_filters = 128, num_channels = 128, strides = [1, 3, 3, 3, 1], layer_name ='conv4_2')
        relu4_2_out = relu_3d(inputs = conv4_2_out, layer_name='relu4_2')

        conv4_3_out, conv4_3_weights = conv3d(inputs = relu4_2_out, filter_size = 3, num_filters = 128, num_channels = 128, strides = [1, 3, 3, 3, 1], layer_name ='conv4_3')
        relu4_3_out = relu_3d(inputs = conv4_3_out, layer_name='relu4_3')

        pool4_out = max_pool_3d(inputs = relu4_3_out, filter_size = [1, 2, 2, 2, 1], strides = [1, 2, 2, 2, 1], layer_name ='pool4')

        # layer5
        conv5_1_out, conv5_1_weights = conv3d(inputs = pool4_out, filter_size = 3, num_filters = 256, num_channels = 128, strides = [1, 3, 3, 3, 1], layer_name ='conv5_1')
        relu5_1_out = relu_3d(inputs = conv5_1_out, layer_name='relu5_1')

        conv5_2_out, conv5_2_weights = conv3d(inputs = relu5_1_out, filter_size = 3, num_filters = 256, num_channels = 256, strides = [1, 3, 3, 3, 1], layer_name ='conv5_2')
        relu5_2_out = relu_3d(inputs = conv5_2_out, layer_name='relu5_2')

        conv5_3_out, conv5_3_weights = conv3d(inputs = relu5_2_out, filter_size = 3, num_filters = 256, num_channels = 256, strides = [1, 3, 3, 3, 1], layer_name ='conv5_3')
        relu5_3_out = relu_3d(inputs = conv5_3_out, layer_name='relu5_3')

        pool5_out = max_pool_3d(inputs = relu5_3_out, filter_size = [1, 2, 2, 2, 1], strides = [1, 2, 2, 2, 1], layer_name ='pool5')
        flatten5_out, flatten5_features = flatten_3d(pool5_out, layer_name='flatten5')

        # layer6
        dense6_out = dense_3d(inputs=flatten5_out, num_inputs=int(flatten5_out.shape[1]), num_outputs=4096, layer_name ='fc6')
        relu6_out = relu_3d(inputs = dense6_out, layer_name='relu6')
        dropout6_out = dropout_3d(inputs = relu6_out, keep_prob = 0.5, layer_name='drop6')

        # layer7
        dense7_out = dense_3d(inputs=dropout6_out, num_inputs=int(dropout6_out.shape[1]), num_outputs=4096, layer_name ='fc7')
        relu7_out = relu_3d(inputs = dense7_out, layer_name='relu7')
        dropout7_out = dropout_3d(inputs = relu7_out, keep_prob = 0.5, layer_name='drop7')

        # layer8
        dense8_out = dense_3d(inputs=dropout7_out, num_inputs=int(dropout7_out.shape[1]), num_outputs=1000, layer_name ='fc8')

        # layer9
        dense9_out = dense_3d(inputs=dense8_out, num_inputs=int(dense8_out.shape[1]), num_outputs=FLAGS.num_classes, layer_name ='fc9')

        # Final softmax
        y = tf.nn.softmax(dense9_out)

        # Overall Metrics Calculations
        with tf.name_scope('log_loss'):
            log_loss = tf.losses.log_loss(y_labels, y, epsilon=10e-15)
            tf.summary.scalar('log_loss', log_loss)

        with tf.name_scope('softmax_cross_entropy'):
            softmax_cross_entropy = tf.losses.softmax_cross_entropy(y_labels, dense9_out)
            tf.summary.scalar('softmax_cross_entropy', softmax_cross_entropy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            tf.summary.scalar('accuracy', accuracy)

        with tf.name_scope('weighted_log_loss'):
            weighted_log_loss = tf.losses.log_loss(y_labels, y, weights=class_weights, epsilon=10e-15) + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            tf.summary.scalar('weighted_log_loss', weighted_log_loss)

        # with tf.name_scope('weighted_softmax_cross_entropy'):
        #     weighted_softmax_cross_entropy = tf.losses.softmax_cross_entropy(y_labels, dense9_out, weights=cross_entropy_weights)
        #     tf.summary.scalar('weighted_softmax_cross_entropy', weighted_softmax_cross_entropy)

        with tf.name_scope('sparse_softmax_cross_entropy'):
            y_labels_argmax_int = tf.to_int32(tf.argmax(y_labels, axis=1))
            sparse_softmax_cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_labels_argmax_int, logits=dense9_out)
            tf.summary.scalar('sparse_softmax_cross_entropy', sparse_softmax_cross_entropy)

        # with tf.name_scope('weighted_sparse_softmax_cross_entropy'):
        #     y_labels_argmax_int = tf.to_int32(tf.argmax(y_labels, axis=1))
        #     weighted_sparse_softmax_cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_labels_argmax_int, logits=dense9_out, weights=cross_entropy_weights)
        #     tf.summary.scalar('weighted_sparse_softmax_cross_entropy', weighted_sparse_softmax_cross_entropy)

        # Class Based Metrics calculations
        y_pred_class = tf.argmax(y, 1)
        y_labels_class = tf.argmax(y_labels, 1)

        confusion_matrix = tf.confusion_matrix(y_labels_class, y_pred_class, num_classes=FLAGS.num_classes)

        sum_row_0 = tf.reduce_sum(confusion_matrix[0, :])
        sum_row_1 = tf.reduce_sum(confusion_matrix[1, :])
        sum_col_0 = tf.reduce_sum(confusion_matrix[:, 0])
        sum_col_1 = tf.reduce_sum(confusion_matrix[:, 1])

        sum_all = tf.reduce_sum(confusion_matrix[:, :])

        with tf.name_scope('precision'):
            precision_0 = confusion_matrix[0,0] / sum_col_0
            precision_1 = confusion_matrix[1,1] / sum_col_1

            tf.summary.scalar('precision_0', precision_0)
            tf.summary.scalar('precision_1', precision_1)

        with tf.name_scope('recall'):
            recall_0 = confusion_matrix[0,0] / sum_row_0
            recall_1 = confusion_matrix[1,1] / sum_row_1
            # recall_2 = confusion_matrix[2,2] / sum_row_2
            # recall_3 = confusion_matrix[3,3] / sum_row_3

            tf.summary.scalar('recall_0', recall_0)
            tf.summary.scalar('recall_1', recall_1)
            # tf.summary.scalar('recall_2', recall_2)
            # tf.summary.scalar('recall_3', recall_3)

        with tf.name_scope('specificity'):
            tn_0 = sum_all - (sum_row_0 + sum_col_0 - confusion_matrix[0,0])
            fp_0 = sum_col_0 - confusion_matrix[0,0]
            specificity_0 = tn_0 / (tn_0 + fp_0)

            tn_1 = sum_all - (sum_row_1 + sum_col_1 - confusion_matrix[1,1])
            fp_1 = sum_col_1 - confusion_matrix[1,1]
            specificity_1 = tn_1 / (tn_1 + fp_1)

            # tn_2 = sum_all - (sum_row_2 + sum_col_2 - confusion_matrix[2,2])
            # fp_2 = sum_col_2 - confusion_matrix[2,2]
            # specificity_2 = tn_2 / (tn_2 + fp_2)
            #
            # tn_3 = sum_all - (sum_row_3 + sum_col_3 - confusion_matrix[3,3])
            # fp_3 = sum_col_3 - confusion_matrix[3,3]
            # specificity_3 = tn_3 / (tn_3 + fp_3)

            tf.summary.scalar('specificity_0', specificity_0)
            tf.summary.scalar('specificity_1', specificity_1)
            # tf.summary.scalar('specificity_2', specificity_2)
            # tf.summary.scalar('specificity_3', specificity_3)

        with tf.name_scope('true_positives'):
            tp_0 = confusion_matrix[0,0]
            tp_1 = confusion_matrix[1,1]
            # tp_2 = confusion_matrix[2,2]
            # tp_3 = confusion_matrix[3,3]

            tf.summary.scalar('true_positives_0', tp_0)
            tf.summary.scalar('true_positives_1', tp_1)
            # tf.summary.scalar('true_positives_2', tp_2)
            # tf.summary.scalar('true_positives_3', tp_3)

        with tf.name_scope('true_negatives'):
            tf.summary.scalar('true_negatives_0', tn_0)
            tf.summary.scalar('true_negatives_1', tn_1)
            # tf.summary.scalar('true_negatives_2', tn_2)
            # tf.summary.scalar('true_negatives_3', tn_3)

        with tf.name_scope('false_positives'):
            tf.summary.scalar('false_positives_0', fp_0)
            tf.summary.scalar('false_positives_1', fp_1)
            # tf.summary.scalar('false_positives_2', fp_2)
            # tf.summary.scalar('false_positives_3', fp_3)

        with tf.name_scope('false_negatives'):
            fn_0 = sum_row_0 - tp_0
            fn_1 = sum_row_1 - tp_1
            # fn_2 = sum_row_2 - tp_2
            # fn_3 = sum_row_3 - tp_3

            tf.summary.scalar('false_negatives_0', fn_0)
            tf.summary.scalar('false_negatives_1', fn_1)
            # tf.summary.scalar('false_negatives_2', fn_2)
            # tf.summary.scalar('false_negatives_3', fn_3)

        with tf.name_scope('log_loss_by_class'):
            log_loss_0 = tf.losses.log_loss(y_labels[0], y[0], epsilon=10e-15)
            log_loss_1 = tf.losses.log_loss(y_labels[1], y[1], epsilon=10e-15)
            # log_loss_2 = tf.losses.log_loss(y_labels[2], y[2], epsilon=10e-15)
            # log_loss_3 = tf.losses.log_loss(y_labels[3], y[3], epsilon=10e-15)

            #added extra '_' to avoid tenosorboard name collision with the main log_loss metric
            tf.summary.scalar('log_loss__0', log_loss_0)
            tf.summary.scalar('log_loss__1', log_loss_1)
            # tf.summary.scalar('log_loss__2', log_loss_2)
            # tf.summary.scalar('log_loss__3', log_loss_3)

        with tf.name_scope('softmax_cross_entropy_by_class'):
            softmax_cross_entropy_0 = tf.losses.softmax_cross_entropy(y_labels[0], dense9_out[0])
            softmax_cross_entropy_1 = tf.losses.softmax_cross_entropy(y_labels[1], dense9_out[1])
            # softmax_cross_entropy_2 = tf.losses.softmax_cross_entropy(y_labels[2], dense9_out[2])
            # softmax_cross_entropy_3 = tf.losses.softmax_cross_entropy(y_labels[3], dense9_out[3])

            tf.summary.scalar('softmax_cross_entropy_0', softmax_cross_entropy_0)
            tf.summary.scalar('softmax_cross_entropy_1', softmax_cross_entropy_1)
            # tf.summary.scalar('softmax_cross_entropy_2', softmax_cross_entropy_2)
            # tf.summary.scalar('softmax_cross_entropy_3', softmax_cross_entropy_3)

        with tf.name_scope('accuracy_by_class'):
            accuracy_0 = (tp_0 + tn_0)/(tp_0 + fp_0 + fn_0 + tn_0)
            accuracy_1 = (tp_1 + tn_1)/(tp_1 + fp_1 + fn_1 + tn_1)
            # accuracy_2 = (tp_2 + tn_2)/(tp_2 + fp_2 + fn_2 + tn_2)
            # accuracy_3 = (tp_3 + tn_3)/(tp_3 + fp_3 + fn_3 + tn_3)

            tf.summary.scalar('accuracy_0', accuracy_0)
            tf.summary.scalar('accuracy_1', accuracy_1)
            # tf.summary.scalar('accuracy_2', accuracy_2)
            # tf.summary.scalar('accuracy_3', accuracy_3)

        with tf.name_scope('weighted_log_loss_by_class'):
            weighted_log_loss_0 = tf.losses.log_loss(y_labels[0], y[0], weights=class_weights[0], epsilon=10e-15)
            weighted_log_loss_1 = tf.losses.log_loss(y_labels[1], y[1], weights=class_weights[1], epsilon=10e-15)
            # weighted_log_loss_2 = tf.losses.log_loss(y_labels[2], y[2], weights=class_weights[2], epsilon=10e-15)
            # weighted_log_loss_3 = tf.losses.log_loss(y_labels[3], y[3], weights=class_weights[3], epsilon=10e-15)

            tf.summary.scalar('weighted_log_loss_0', weighted_log_loss_0)
            tf.summary.scalar('weighted_log_loss_1', weighted_log_loss_1)
            # tf.summary.scalar('weighted_log_loss_2', weighted_log_loss_2)
            # tf.summary.scalar('weighted_log_loss_3', weighted_log_loss_3)

        with tf.name_scope('f1_score_by_class'):
            f1_score_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0)
            f1_score_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1)
            # f1_score_2 = 2 * (precision_2 * recall_2) / (precision_2 + recall_2)
            # f1_score_3 = 2 * (precision_3 * recall_3) / (precision_3 + recall_3)
            # #f1_score = (f1_score_0 * 40591.0/69920.0) + (f1_score_1 * 14624.0/69920.0) + (f1_score_2 * 10490.0/69920.0) + (f1_score_3 *4215.0/ 69920.0)
            tf.summary.scalar('f1_score_0', f1_score_0)
            tf.summary.scalar('f1_score_1', f1_score_1)
            # tf.summary.scalar('f1_score_2', f1_score_2)
            # tf.summary.scalar('f1_score_3', f1_score_3)

        #with tf.name_scope('train'):
        #    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, name='adam_optimizer').minimize(softmax_cross_entropy)

        merged = tf.summary.merge_all()
        saver = tf.train.Saver()

    # Setting up config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = FLAGS.allow_growth
    config.log_device_placement=FLAGS.log_device_placement
    config.allow_soft_placement=FLAGS.allow_soft_placement

    with tf.Session(graph=graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(MODEL_PATH + 'model.meta')
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))

        for patient_uid in patient_uids:
            print("Processing patient ", patient_uid)
            if os.path.isfile(DATA_PATH + PATIENT_SCANS + patient_uid + '.npy'):
                scans = np.load(DATA_PATH + PATIENT_SCANS + patient_uid + '.npy')
            elif os.path.isfile(DATA_PATH2 + PATIENT_SCANS + patient_uid + '.npy'):
                scans = np.load(DATA_PATH2 + PATIENT_SCANS + patient_uid + '.npy')
            else:
                with open("error.out", "a") as myfile:
                    myfile.write('Couldnt find scan for patient {}'.format(patient_uid))

            step_size = int(FLAGS.chunk_size * (1 - OVERLAP_PERCENTAGE))
            num_chunks_0 = int(scans.shape[0] / step_size) + 1
            num_chunks_1 = int(scans.shape[1] / step_size) + 1
            num_chunks_2 = int(scans.shape[2] / step_size) + 1
            total_num_chunks = num_chunks_0 * num_chunks_1 * num_chunks_2
            chunk_list = []

            predictions = np.ndarray([total_num_chunks, FLAGS.num_classes], dtype=np.float32)
            transfer_values = np.ndarray([total_num_chunks, 1, 1, 1, 256], dtype=np.float32)

            predictions_idx = 0
            for i in range(0, num_chunks_0):
                coordZ1 = i * step_size
                coordZ2 = coordZ1 + FLAGS.chunk_size

                for j in range(0, num_chunks_1):
                    coordY1 = j * step_size
                    coordY2 = coordY1 + FLAGS.chunk_size

                    for k in range(0, num_chunks_2):
                        coordX1 = k * step_size
                        coordX2 = coordX1 + FLAGS.chunk_size

                        coordZ2 = scans.shape[0] if  (coordZ2 > scans.shape[0]) else coordZ2
                        coordY2 = scans.shape[1] if  (coordY2 > scans.shape[1]) else coordY2
                        coordX2 = scans.shape[2] if  (coordX2 > scans.shape[2]) else coordX2

                        chunk = np.full((FLAGS.chunk_size, FLAGS.chunk_size, FLAGS.chunk_size), -1000.0)
                        chunk[0:coordZ2-coordZ1, 0:coordY2-coordY1, 0:coordX2-coordX1] = scans[coordZ1:coordZ2, coordY1:coordY2, coordX1:coordX2]
                        chunk_list.append(chunk)

                        if (len(chunk_list) == FLAGS.batch_size) or ((i == num_chunks_0 -1) and (j == num_chunks_1 -1) and (k == num_chunks_2 - 1)):
                            X = np.ndarray([len(chunk_list), FLAGS.chunk_size, FLAGS.chunk_size, FLAGS.chunk_size], dtype=np.float32)
                            for m in range(0, len(chunk_list)):
                                X[m, :, :, :] = chunk_list[m]

                            # Normalizing and Zero Centering and Adding extra channel
                            X = X.astype(np.float32, copy=False)
                            X = normalize(X)
                            X = zero_center(X)
                            X = img_to_rgb(X)

                            feed_dict = {x: X, y_labels: np.zeros([X.shape[0], FLAGS.num_classes], dtype=np.float32), keep_prob: 1.0}
                            pred, trans_val = sess.run([y, relu5_3_out], feed_dict=feed_dict)

                            batch_start = predictions_idx
                            batch_end = predictions_idx + X.shape[0]
                            predictions[batch_start: batch_end, :] = pred
                            transfer_values[batch_start: batch_end, :] = trans_val

                            predictions_idx = predictions_idx + X.shape[0]
                            chunk_list.clear()
                            del X

            np.save(OUTPUT_PATH + patient_uid + '_predictions.npy', predictions)
            np.save(OUTPUT_PATH + patient_uid + '_transfer_values.npy', transfer_values)
            del scans

    sess.close()

def predict_features():
    processed_patients = set()
    for patients in glob.glob(OUTPUT_PATH + '*_transfer_values.npy'):
        n = re.match('([a-f0-9].*)_transfer_values.npy', os.path.basename(patients))
        processed_patients.add(n.group(1))

    train_patient_ids = set()
    for folder in glob.glob(DATA_PATH + PATIENT_SCANS + '*'):
        m = re.match(PATIENT_SCANS +'([a-f0-9].*).npy', os.path.basename(folder))
        patient_uid = m.group(1)
        train_patient_ids.add(patient_uid)

    for folder in glob.glob(DATA_PATH2 + PATIENT_SCANS + '*'):
        m = re.match(PATIENT_SCANS +'([a-f0-9].*).npy', os.path.basename(folder))
        patient_uid = m.group(1)
        train_patient_ids.add(patient_uid)

    train_patient_ids = list(train_patient_ids.difference(processed_patients))

    processes = {}
    patient_batch_size = int(math.ceil(float(len(train_patient_ids)) / NUM_PROCESSES))

    for p_id in range(NUM_PROCESSES):
        print('Starting process ', p_id)
        start = p_id * patient_batch_size
        end = start + patient_batch_size
        end = len(train_patient_ids) if end > len(train_patient_ids) else end
        processes[p_id] = mp.Process(target=worker, args=(train_patient_ids[start:end],))
        processes[p_id].start()

    for key in processes.keys():
        processes[key].join()

if __name__ == '__main__':
    start_time = time.time()

    NUM_PROCESSES = mp.cpu_count()
    DATA_PATH = '/kaggle/dev/data-science-bowl-2017-data/stage1_processed/'
    DATA_PATH2 = '/kaggle_3/stage2_processed/'
    OUTPUT_PATH = '/kaggle_3/all_stage_features_segmented/'
    PATIENT_SCANS = 'scan_segmented_lungs_fill_'
    TENSORBOARD_SUMMARIES = '/kaggle/dev/data-science-bowl-2017-data/tensorboard_summaries/'
    MODEL_PATH = '/kaggle_2/luna/luna16/models/4ba9ca74-7994-42bf-9d9f-3a8dd682e623/'
    OVERLAP_PERCENTAGE = 0.7

    #globals initializing
    FLAGS = tf.app.flags.FLAGS

    ## Prediction problem specific
    tf.app.flags.DEFINE_integer('num_classes', 2,
                                """Number of classes to predict.""")
    tf.app.flags.DEFINE_integer('chunk_size', 48,
                                """Chunk size""")
    tf.app.flags.DEFINE_integer('batch_size', 192,
                                """Number of items in a batch.""")
    tf.app.flags.DEFINE_float('require_improvement_percentage', 0.20,
                                """Percent of max_iterations after which optimization will be halted if no improvement found""")
    tf.app.flags.DEFINE_float('iteration_analysis_percentage', 0.10,
                                """Percent of max_iterations after which analysis will be done""")
    tf.app.flags.DEFINE_float('reg_constant', 0.1, 'Regularization constant.')
    tf.app.flags.DEFINE_float('dropout', 0.5, 'Dropout')

    ## Tensorflow specific
    tf.app.flags.DEFINE_integer('num_gpus', 2,
                                """How many GPUs to use.""")
    tf.app.flags.DEFINE_boolean('log_device_placement', False,
                                """Whether to log device placement.""")
    tf.app.flags.DEFINE_boolean('allow_soft_placement', True,
                                """Whether to allow soft placement of calculations by tf.""")
    tf.app.flags.DEFINE_boolean('allow_growth', True,
                                """Whether to allow GPU growth by tf.""")


    predict_features()
    end_time = time.time()
    print("Total Time usage: " + str(timedelta(seconds=int(round(end_time - start_time)))))
