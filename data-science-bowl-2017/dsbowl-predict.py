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

# Fixes "SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame"
pd.options.mode.chained_assignment = None

import scipy.misc
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
    scans = np.load(DATA_PATH + PATIENT_SCANS + patient_id + '.npy')
    chunk_counter = 1
    step_size = int(CHUNK_SIZE * (1 - OVERLAP_PERCENTAGE))
    num_chunks_0 = int(scans.shape[0] / step_size) + 1
    num_chunks_1 = int(scans.shape[1] / step_size) + 1
    num_chunks_2 = int(scans.shape[2] / step_size) + 1
    chunk_list = []

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
    return X

def predict_features():
    # Graph construction
    graph = tf.Graph()
    with graph.as_default():
        model_name = 'convnet3D_v0.1'
        with tf.name_scope(model_name):
            x = tf.placeholder(tf.float32, shape=[None, 64, 64, 64, 1], name = 'x')
            y = tf.placeholder(tf.float32, shape=[None, FLAGS.num_classes], name = 'y')
            y_labels = tf.placeholder(tf.float32, shape=[None, FLAGS.num_classes], name ='y_labels')
            class_weights2 = tf.ones_like(y_labels)
            class_weights = tf.multiply(class_weights2 , [1000/40513.0, 1000/48.0, 1000/2876.0, 1000/1511.0, 1000/587.0, 1000/315.0, 1000/528.0])

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

            with tf.name_scope('softmax_cross_entropy'):
                softmax_cross_entropy = tf.losses.softmax_cross_entropy(y_labels, layer6_dense3d_out)
                tf.summary.scalar('softmax_cross_entropy', softmax_cross_entropy)

            #with tf.name_scope('sparse_softmax_cross_entropy'):
            #    sparse_softmax_cross_entropy = tf.losses.sparse_softmax_cross_entropy(y_labels,
            #                                                                          layer6_dense3d_out)
            #    tf.summary.scalar('sparse_softmax_cross_entropy', sparse_softmax_cross_entropy)

            #with tf.name_scope('weighted_sparse_softmax_cross_entropy'):
            #    weighted_sparse_softmax_cross_entropy = tf.losses.sparse_softmax_cross_entropy(y_labels,
            #                                                                          layer6_dense3d_out,
            #                                                                          weights=class_weights)
            #    tf.summary.scalar('weighted_sparse_softmax_cross_entropy', weighted_sparse_softmax_cross_entropy)

            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_labels, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                tf.summary.scalar('accuracy', accuracy)

            with tf.name_scope('weighted_log_loss'):
                weighted_log_loss = tf.losses.log_loss(y_labels, y, weights=class_weights, epsilon=10e-15)
                tf.summary.scalar('weighted_log_loss', weighted_log_loss)

            # Metrics calculations
            y_pred_class = tf.argmax(y, 1)
            y_labels_class = tf.argmax(y_labels, 1)

            confusion_matrix = tf.confusion_matrix(y_labels_class, y_pred_class, num_classes=FLAGS.num_classes)

            sum_row_0 = tf.reduce_sum(confusion_matrix[0, :])
            sum_row_1 = tf.reduce_sum(confusion_matrix[1, :])
            sum_row_2 = tf.reduce_sum(confusion_matrix[2, :])
            sum_row_3 = tf.reduce_sum(confusion_matrix[3, :])
            sum_row_4 = tf.reduce_sum(confusion_matrix[4, :])
            sum_row_5 = tf.reduce_sum(confusion_matrix[5, :])
            sum_row_6 = tf.reduce_sum(confusion_matrix[6, :])

            sum_col_0 = tf.reduce_sum(confusion_matrix[:, 0])
            sum_col_1 = tf.reduce_sum(confusion_matrix[:, 1])
            sum_col_2 = tf.reduce_sum(confusion_matrix[:, 2])
            sum_col_3 = tf.reduce_sum(confusion_matrix[:, 3])
            sum_col_4 = tf.reduce_sum(confusion_matrix[:, 4])
            sum_col_5 = tf.reduce_sum(confusion_matrix[:, 5])
            sum_col_6 = tf.reduce_sum(confusion_matrix[:, 6])

            sum_all = tf.reduce_sum(confusion_matrix[:, :])

            with tf.name_scope('precision'):
                precision_0 = confusion_matrix[0,0] / sum_col_0
                precision_1 = confusion_matrix[1,1] / sum_col_1
                precision_2 = confusion_matrix[2,2] / sum_col_2
                precision_3 = confusion_matrix[3,3] / sum_col_3
                precision_4 = confusion_matrix[4,4] / sum_col_4
                precision_5 = confusion_matrix[5,5] / sum_col_5
                precision_6 = confusion_matrix[6,6] / sum_col_6
                tf.summary.scalar('precision_0', precision_0)
                tf.summary.scalar('precision_1', precision_1)
                tf.summary.scalar('precision_2', precision_2)
                tf.summary.scalar('precision_3', precision_3)
                tf.summary.scalar('precision_4', precision_4)
                tf.summary.scalar('precision_5', precision_5)
                tf.summary.scalar('precision_6', precision_6)

            with tf.name_scope('recall'):
                recall_0 = confusion_matrix[0,0] / sum_row_0
                recall_1 = confusion_matrix[1,1] / sum_row_1
                recall_2 = confusion_matrix[2,2] / sum_row_2
                recall_3 = confusion_matrix[3,3] / sum_row_3
                recall_4 = confusion_matrix[4,4] / sum_row_4
                recall_5 = confusion_matrix[5,5] / sum_row_5
                recall_6 = confusion_matrix[6,6] / sum_row_6
                tf.summary.scalar('recall_0', recall_0)
                tf.summary.scalar('recall_1', recall_1)
                tf.summary.scalar('recall_2', recall_2)
                tf.summary.scalar('recall_3', recall_3)
                tf.summary.scalar('recall_4', recall_4)
                tf.summary.scalar('recall_5', recall_5)
                tf.summary.scalar('recall_6', recall_6)

            with tf.name_scope('specificity'):
                tn_0 = sum_all - (sum_row_0 + sum_col_0 - confusion_matrix[0,0])
                fp_0 = sum_col_0 - confusion_matrix[0,0]
                specificity_0 = tn_0 / (tn_0 + fp_0)

                tn_1 = sum_all - (sum_row_1 + sum_col_1 - confusion_matrix[1,1])
                fp_1 = sum_col_1 - confusion_matrix[1,1]
                specificity_1 = tn_1 / (tn_1 + fp_1)

                tn_2 = sum_all - (sum_row_2 + sum_col_2 - confusion_matrix[2,2])
                fp_2 = sum_col_2 - confusion_matrix[2,2]
                specificity_2 = tn_2 / (tn_2 + fp_2)

                tn_3 = sum_all - (sum_row_3 + sum_col_3 - confusion_matrix[3,3])
                fp_3 = sum_col_3 - confusion_matrix[3,3]
                specificity_3 = tn_3 / (tn_3 + fp_3)

                tn_4 = sum_all - (sum_row_4 + sum_col_4 - confusion_matrix[4,4])
                fp_4 = sum_col_4 - confusion_matrix[4,4]
                specificity_4 = tn_4 / (tn_4 + fp_4)

                tn_5 = sum_all - (sum_row_5 + sum_col_5 - confusion_matrix[5,5])
                fp_5 = sum_col_5 - confusion_matrix[5,5]
                specificity_5 = tn_5 / (tn_5 + fp_5)

                tn_6 = sum_all - (sum_row_6 + sum_col_6 - confusion_matrix[6,6])
                fp_6 = sum_col_6 - confusion_matrix[6,6]
                specificity_6 = tn_6 / (tn_6 + fp_6)

                tf.summary.scalar('specificity_0', specificity_0)
                tf.summary.scalar('specificity_1', specificity_1)
                tf.summary.scalar('specificity_2', specificity_2)
                tf.summary.scalar('specificity_3', specificity_3)
                tf.summary.scalar('specificity_4', specificity_4)
                tf.summary.scalar('specificity_5', specificity_5)
                tf.summary.scalar('specificity_6', specificity_6)

            #optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, name='adam_optimizer').minimize(weighted_log_loss)

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

        processed_patients = set()
        for patients in glob.glob(OUTPUT_PATH + '*_transfer_values.npy'):
            n = re.match('([a-f0-9].*)_transfer_values.npy', os.path.basename(patients))
            processed_patients.add(n.group(1))

        for folder in tqdm(glob.glob(DATA_PATH + PATIENT_SCANS + '*')):
            m = re.match(PATIENT_SCANS +'([a-f0-9].*).npy', os.path.basename(folder))
            patient_uid = m.group(1)

            if patient_uid in processed_patients:
                #print('Skipping already processed patient {}'.format(patient_uid))
                continue

            # print('Processing patient {}'.format(patient_uid))
            x_in = get_patient_data_chunks(patient_uid)
            # print('Got Data for patient {}'.format(patient_uid))
            X = np.ndarray([x_in.shape[0], 64, 64, 64, 1], dtype=np.float32)
            X[0: x_in.shape[0], :, :, :, :] = img_to_rgb(x_in)

            # print('X: {}'.format(X.shape))
            predictions = np.ndarray([X.shape[0], FLAGS.num_classes], dtype=np.float32)
            transfer_values = np.ndarray([X.shape[0], 512], dtype=np.float32)

            num_batches = int(math.ceil(X.shape[0] / FLAGS.batch_size))
            for i in range(0, num_batches):
                batch_start = i * FLAGS.batch_size
                batch_end = batch_start + FLAGS.batch_size
                batch_end = X.shape[0] if (batch_end > X.shape[0]) else batch_end
                feed_dict = {x: X[batch_start : batch_end],
                             y_labels: np.zeros([batch_end - batch_start, FLAGS.num_classes], dtype=np.float32)}

                # print('X[{}]: {}'.format(i, X[batch_start:batch_end].shape))
                pred, trans_val = sess.run([y, layer4_dense3d_out], feed_dict=feed_dict)
                predictions[batch_start: batch_end, :] = pred
                transfer_values[batch_start: batch_end, :] = trans_val
                #print('predictions: ' + str(predictions.shape))
                #print('transfer_values: ' + str(transfer_values.shape))

            np.save(OUTPUT_PATH + patient_uid + '_predictions.npy', predictions)
            np.save(OUTPUT_PATH + patient_uid + '_transfer_values.npy', transfer_values)

            del x_in, X

    sess.close()

if __name__ == '__main__':
    start_time = time.time()

    DATA_PATH = '/kaggle/dev/data-science-bowl-2017-data/stage1_processed/'
    OUTPUT_PATH = '/kaggle/dev/data-science-bowl-2017-data/stage1_features_v2/'
    PATIENT_SCANS = 'scan_segmented_lungs_fill_'
    TENSORBOARD_SUMMARIES = '/kaggle/dev/data-science-bowl-2017-data/tensorboard_summaries/'
    MODEL_PATH = '/kaggle_2/luna/luna16/models/3ac0a07d-9264-487a-b730-7c8a04557706/'
    CHUNK_SIZE = 64
    OVERLAP_PERCENTAGE = 0.7

    #globals initializing
    FLAGS = tf.app.flags.FLAGS

    ## Prediction problem specific
    tf.app.flags.DEFINE_integer('num_classes', 7,
                                """Number of classes to predict.""")
    tf.app.flags.DEFINE_integer('batch_size', 600,
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

    predict_features()
    end_time = time.time()
    print("Total Time usage: " + str(timedelta(seconds=int(round(end_time - start_time)))))
