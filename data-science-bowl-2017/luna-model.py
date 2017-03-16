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

def get_ids(PATH):
    ids = []
    for path in glob.glob(PATH + '[0-9\.]*_X.npy'):
        patient_id = re.match(r'([0-9\.]*)_X.npy', os.path.basename(path)).group(1)
        ids.append(patient_id)
    return ids

def get_data(patient_ids, PATH):
    num_chunks = 0

    for patient_id in patient_ids:
        x = np.load(PATH + patient_id + '_X.npy')
        num_chunks = num_chunks + x.shape[0]

    X = np.ndarray([num_chunks, 64, 64, 64, 1], dtype=np.float32)
    Y = np.ndarray([num_chunks, 7], dtype=np.float32)

    count = 0
    for patient_id in patient_ids:
        x = np.load(PATH + patient_id + '_X.npy').astype(np.float32, copy=False)
        y = np.load(PATH + patient_id + '_Y.npy').astype(np.float32, copy=False)

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



def train_3d_nn():

    #### Helper function ####
    def predict_prob_validation(validation_x, labels, write_to_tensorboard=False):
        num_images = len(validation_x)

        prob_pred = np.zeros(shape=[num_images, FLAGS.num_classes], dtype=np.float64)

        i = 0
        while i < num_images:
            j = min(i + FLAGS.batch_size, num_images)
            feed_dict = {x: validation_x[i:j],
                         y_labels: labels[i:j]}

            y_calc, step_summary = sess.run([y, merged], feed_dict=feed_dict)
            prob_pred[i:j] = y_calc

            if write_to_tensorboard:
                validation_writer.add_summary(step_summary, i)
            # Set the start-index for the next batch to the
            # end-index of the current batch.
            i = j
        return prob_pred


    def calc_validation_log_loss(write_to_tensorboard = False):
        prob_pred = predict_prob_validation(validation_x,
                                            labels = validation_y,
                                            write_to_tensorboard = write_to_tensorboard)
        p = np.maximum(np.minimum(prob_pred, 1-10e-15), 10e-15)
        l = np.transpose(validation_y + 0.0)
        n = validation_y.shape[0]
        temp = np.matmul(l, np.log(p)) + np.matmul((1 - l), np.log(1 - p))

        # Divide by 2 (magic number)
        validation_log_loss = -1.0 * (temp[0,0] + temp[1,1])/(2 * n)

        return validation_log_loss

    #### Helper function ####



    patient_ids = get_ids(DATA_PATH_PREPROCESS)
    X, Y = get_data(patient_ids[0:100], DATA_PATH_PREPROCESS) ## IMPORTANT: Remove bounds when traning on the whole
    #X = normalize(X)
    #X = zero_center(X)


    print('Splitting into train, validation sets')
    train_x, validation_x, train_y, validation_y = model_selection.train_test_split(X, Y, random_state=42, stratify=Y,
                                                                    test_size=0.20)

    print(len(X))
    print(len(Y))
    print('train_x: {}'.format(train_x.shape))
    print('validation_x: {}'.format(validation_x.shape))
    print('train_y: {}'.format(train_y.shape))
    print('validation_y: {}'.format(validation_y.shape))

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

    # timestamp used to identify the start of run
    start_timestamp = str(int(time.time()))

    # Name used to save all artifacts of run
    run_name = 'type=train_timestamp=' + start_timestamp + '_batch=' + str(FLAGS.batch_size) + '_iterations=' + str(FLAGS.max_iterations) + '_model=' + model_name + '_train=' + str(train_x.shape[0]) + '_validation=' + str(validation_x.shape[0])

    with tf.Session(graph=graph, config=config) as sess:
        train_writer = tf.summary.FileWriter(TENSORBOARD_SUMMARIES + run_name, sess.graph)
        sess.run(tf.global_variables_initializer())

        print('\nPre-train validation log loss: {0:.5}'.format(calc_validation_log_loss()))

        for i in tqdm(range(FLAGS.max_iterations)):
            x_batch, y_batch = get_batch(train_x, train_y, FLAGS.batch_size)
            _,step_summary, loss_val = sess.run([optimizer, merged, log_loss],
                                                feed_dict={x: x_batch, y_labels: y_batch})
            train_writer.add_summary(step_summary, i)
        print('\nPost-train validation log loss: {0:.5}'.format(calc_validation_log_loss()))

        # Saving model
        checkpoint_loc = os.path.join(MODELS, run_name)
        if not os.path.exists(checkpoint_loc):
            os.makedirs(checkpoint_loc)
        save_path = os.path.join(checkpoint_loc, 'saved-model_'+ run_name)
        saver.save(sess=sess, save_path=save_path)

        # Clossing session
        sess.close()
    print('all artifacts associated with this are will have the following run_name: {}'.format(run_name))

def post_process():
    print("in post_process")


if __name__ == '__main__':
    DATA_PATH_PREPROCESS = '/kaggle_2/luna/luna16/data/pre_processed_chunks/'
    DATA_PATH_POSTPROCESS = '/kaggle_2/luna/luna16/data/pre_processed_chunks_normalized_zerocentered/'
    TENSORBOARD_SUMMARIES = '/kaggle_2/luna/luna16/data/tensorboard_summaries/'
    MODELS = '/kaggle_2/luna/luna16/models/'

    #globals initializing
    FLAGS = tf.app.flags.FLAGS

    ## Prediction problem specific
    tf.app.flags.DEFINE_integer('num_classes', 7,
                                """Number of classes to predict.""")
    tf.app.flags.DEFINE_integer('batch_size', 32,
                                """Number of items in a batch.""")
    tf.app.flags.DEFINE_integer('max_iterations', 100, #100000
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

    #post_process()
    train_3d_nn()
    print('all processing done')
