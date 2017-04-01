import glob
import os
import csv
import re
import uuid
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
from sklearn import model_selection
import xgboost as xgb
import scipy as sp
from sklearn.decomposition import PCA
import sklearn.metrics

def img_to_rgb(im):
    x = im.shape[0]
    ret = np.empty((x, 1), dtype=np.float32)
    ret[:, 0] = im
    return ret

def get_patient_labels(patient_ids):
    labels = pd.read_csv(LABELS)
    input_labels = {}
    for patient_id in patient_ids:
        try:
            label = int(labels.loc[labels['id'] == patient_id, 'cancer'])
            input_labels[patient_id] = label
        except TypeError:
            print('ERROR: Couldnt find label for patient {}'.format(patient_id))
            continue
    return input_labels

def get_patient_features(patient_ids):
    input_features = {}
    num_patients = len(patient_ids)
    count = 0
    for patient_id in patient_ids:
        predictions = np.array(np.load(DATA_PATH + patient_id + '_predictions.npy'))
        transfer_values = np.array(np.load(DATA_PATH + patient_id + '_transfer_values.npy'))
        features_shape = (transfer_values.shape[0], transfer_values.shape[1] + FLAGS.num_classes_luna)
        features = np.zeros(shape=features_shape, dtype=np.float32)
        features[:, 0:transfer_values.shape[1]] = transfer_values
        features[:, transfer_values.shape[1]:transfer_values.shape[1] + FLAGS.num_classes_luna] = predictions
        input_features[patient_id] = features
        count = count + 1
        print('Loaded data for patient {}/{}'.format(count, num_patients))
    return input_features

def conv1d(inputs,             # The previous layer.
           filter_size,        # Width and height of each filter.
           num_filters,        # Number of filters.
           num_channels,       # 1
           strides,            # [1,1,1,1,1]
           layer_name):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            filters = tf.get_variable(layer_name + 'weights', shape = [filter_size, num_channels, num_filters],
                                      initializer = tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32),
                                      regularizer = tf.contrib.layers.l2_regularizer(FLAGS.reg_constant))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0, shape=[num_filters], dtype=tf.float32))
        with tf.name_scope('conv'):
            conv = tf.nn.conv1d(inputs, filters, strides, padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        return out, filters

# def max_pool_1d(inputs,
#                 filter_size,
#                 strides,
#                 layer_name):
#     with tf.name_scope(layer_name):
#         return tf.nn.max_pool(inputs,
#                                 ksize=filter_size,
#                                 strides=strides,
#                                 padding='SAME',
#                                 data_format='NHWC',
#                                 name='max_pool')

def dropout_1d(inputs,
               keep_prob,
               layer_name):
    with tf.name_scope(layer_name):
        return tf.nn.dropout(inputs, keep_prob, name='dropout')

def flatten_1d(layer, layer_name):
    with tf.name_scope(layer_name):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:3].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
        return layer_flat, num_features

def relu_1d(inputs,
            layer_name):
    with tf.name_scope(layer_name):
        return tf.nn.relu(inputs, name='relu')

def dense_1d(inputs,
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

def get_training_batch(train_x, train_y, batch_size):
    num_images = len(train_x)
    idx = np.random.choice(num_images,
                           size=batch_size,
                           replace=False)
    x_batch = train_x[idx]
    y_batch = train_y[idx]

    return x_batch, y_batch

def get_validation_batch(validation_x, validation_y, batch_number, batch_size):
    num_images = len(validation_x)

    start_index = batch_number * batch_size
    end_index = start_index + batch_size
    end_index = num_images if end_index > num_images else end_index

    return validation_x[start_index:end_index], validation_y[start_index:end_index]

def save_model(sess, model_id, saver):
    checkpoint_folder = os.path.join(MODELS, model_id)
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    save_path = os.path.join(checkpoint_folder, 'model')
    saver.save(sess=sess, save_path=save_path)

def train_nn():
    print('Loading data..')
    time0 = time.time()
    patient_ids = set()
    for file_path in glob.glob(DATA_PATH + "*_transfer_values.npy"):
        filename = os.path.basename(file_path)
        patient_id = re.match(r'([a-f0-9].*)_transfer_values.npy', filename).group(1)
        patient_ids.add(patient_id)

    sample_submission = pd.read_csv(STAGE1_SUBMISSION)
    test_patient_ids = set(sample_submission['id'].tolist())
    train_patient_ids = patient_ids.difference(test_patient_ids)

    #train_patient_ids = list(train_patient_ids)[0:20]

    train_inputs = get_patient_features(train_patient_ids)
    train_labels = get_patient_labels(train_patient_ids)

    X_list = []
    Y_list = []
    for key in train_inputs.keys():
        patient_features = train_inputs[key]
        patient_label = train_labels[key]
        for i in range(patient_features.shape[0]):
            X_list.append(img_to_rgb(patient_features[i]))
            Y_list.append(patient_label)

    num_chunks = len(X_list)

    print(num_chunks)
    X = np.asarray(X_list)
    Y = np.asarray(Y_list)

    print('X.shape: {}'.format(X.shape))
    print('Y.shape: {}'.format(Y.shape))
    print("Total time to load data: " + str(timedelta(seconds=int(round(time.time() - time0)))))

    print('\nSplitting data into train, validation')
    train_x, validation_x, train_y, validation_y = model_selection.train_test_split(X, Y, random_state=42, stratify=Y, test_size=0.20)

    del X
    del Y

    # One-hot encode
    train_y = (np.arange(FLAGS.num_classes) == train_y[:, None])+0
    validation_y = (np.arange(FLAGS.num_classes) == validation_y[:, None])+0

    print('train_x: {}'.format(train_x.shape))
    print('validation_x: {}'.format(validation_x.shape))
    print('train_y: {}'.format(train_y.shape))
    print('validation_y: {}'.format(validation_y.shape))

    # Seed numpy random to generate identical random numbers every time (used in batching)
    np.random.seed(42)

    def feed_dict(is_train, batch_number = 0):
        if is_train:
            x_batch, y_batch = get_training_batch(train_x, train_y, FLAGS.batch_size)
            k = FLAGS.dropout
        else:
            x_batch, y_batch = get_validation_batch(validation_x, validation_y, batch_number, FLAGS.batch_size)
            k = 1.0

        return {x: x_batch, y_labels: y_batch, keep_prob: k}

    # Graph construction
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, shape=[None, FLAGS.transfer_values_shape + FLAGS.num_classes_luna, 1], name = 'x')
        y = tf.placeholder(tf.float32, shape=[None, FLAGS.num_classes], name = 'y')
        y_labels = tf.placeholder(tf.float32, shape=[None, FLAGS.num_classes], name ='y_labels')
        keep_prob = tf.placeholder(tf.float32)

        # layer1
        conv1_out, conv1_weights = conv1d(inputs = x, filter_size = 3, num_filters = 16, num_channels = 1, strides = 3, layer_name ='conv1')

        relu1_out = relu_1d(inputs = conv1_out, layer_name='relu1')

        # layer2
        conv2_out, conv2_weights = conv1d(inputs = relu1_out, filter_size = 3, num_filters = 32, num_channels = 16, strides = 3, layer_name ='conv2')

        relu2_out = relu_1d(inputs = conv2_out, layer_name='relu2')

        # layer3
        conv3_out, conv3_weights = conv1d(inputs = relu2_out, filter_size = 3, num_filters = 64, num_channels = 32, strides = 3, layer_name ='conv3')

        relu3_out = relu_1d(inputs = conv3_out, layer_name='relu3')

        dropout3_out = dropout_1d(inputs = relu3_out, keep_prob = keep_prob, layer_name='drop3')

        flatten4_out, flatten4_features = flatten_1d(dropout3_out, layer_name='flatten4')

        # layer6
        dense5_out = dense_1d(inputs=flatten4_out, num_inputs=int(flatten4_out.shape[1]), num_outputs=512, layer_name ='fc5')

        relu5_out = relu_1d(inputs = dense5_out, layer_name='relu5')

        dropout5_out = dropout_1d(inputs = relu5_out, keep_prob = keep_prob, layer_name='drop5')

        # layer7
        dense6_out = dense_1d(inputs=dropout5_out, num_inputs=int(dropout5_out.shape[1]), num_outputs=128, layer_name ='fc6')

        relu6_out = relu_1d(inputs = dense6_out, layer_name='relu6')

        dropout6_out = dropout_1d(inputs = relu6_out, keep_prob = keep_prob, layer_name='drop6')

        # layer9
        final_layer_out = dense_1d(inputs=dropout6_out, num_inputs=int(dropout6_out.shape[1]), num_outputs=FLAGS.num_classes, layer_name ='fc7')

        # Final softmax
        y = tf.nn.softmax(final_layer_out)

        # Overall Metrics Calculations
        with tf.name_scope('log_loss'):
            log_loss = tf.losses.log_loss(y_labels, y, epsilon=10e-15) + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            tf.summary.scalar('log_loss', log_loss)

        with tf.name_scope('softmax_cross_entropy'):
            softmax_cross_entropy = tf.losses.softmax_cross_entropy(y_labels, final_layer_out)
            tf.summary.scalar('softmax_cross_entropy', softmax_cross_entropy)

        with tf.name_scope('sparse_softmax_cross_entropy'):
            y_labels_argmax_int = tf.to_int32(tf.argmax(y_labels, axis=1))
            sparse_softmax_cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_labels_argmax_int, logits=final_layer_out)
            tf.summary.scalar('sparse_softmax_cross_entropy', sparse_softmax_cross_entropy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            tf.summary.scalar('accuracy', accuracy)

        # with tf.name_scope('weighted_log_loss'):
        #     weighted_log_loss = tf.losses.log_loss(y_labels, y, weights=class_weights, epsilon=10e-15) + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        #     tf.summary.scalar('weighted_log_loss', weighted_log_loss)

        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, name='adam_optimizer').minimize(log_loss)

        merged = tf.summary.merge_all()
        saver = tf.train.Saver()

    # Setting up config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = FLAGS.allow_growth
    config.log_device_placement=FLAGS.log_device_placement
    config.allow_soft_placement=FLAGS.allow_soft_placement

    # timestamp used to identify the start of run
    start_timestamp = str(int(time.time()))
    model_id = str(uuid.uuid4())

    # Name used to save all artifacts of run
    run_name = 'runType={0:}_timestamp={1:}_batchSize={2:}_maxIterations={3:}_numTrain={4:}_numValidation={5:}_modelId={6:}'
    train_run_name = run_name.format('train', start_timestamp, FLAGS.batch_size, FLAGS.max_iterations,
                               train_x.shape[0], validation_x.shape[0], model_id)

    test_run_name = run_name.format('test', start_timestamp, FLAGS.batch_size, FLAGS.max_iterations,
                               train_x.shape[0], validation_x.shape[0], model_id)

    print('Run_name: {}'.format(train_run_name))

    k_count = 0
    with tf.Session(graph=graph, config=config) as sess:
        train_writer = tf.summary.FileWriter(TENSORBOARD_SUMMARIES + train_run_name, sess.graph)
        test_writer = tf.summary.FileWriter(TENSORBOARD_SUMMARIES + test_run_name, sess.graph)
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        for i in tqdm(range(FLAGS.max_iterations)):
            if (i % FLAGS.iteration_analysis == 0) or (i == (FLAGS.max_iterations - 1)):
                save_model(sess, model_id, saver)
                # Validation
                num_batches = int(math.ceil(float(len(validation_x)) / FLAGS.batch_size))
                for k in range(num_batches):
                    _, step_summary = sess.run([y, merged], feed_dict=feed_dict(False, k))
                    test_writer.add_summary(step_summary, k_count)
                    k_count = k_count + 1
            else:
                # Train
                _, step_summary = sess.run([optimizer, merged], feed_dict=feed_dict(True))
                train_writer.add_summary(step_summary, i)

        train_writer.close()
        test_writer.close()
        # Clossing session
        sess.close()

# def make_submission():
    # print('Loading data..')
    # time0 = time.time()
    # patient_ids = set()
    # for file_path in glob.glob(DATA_PATH + "*_transfer_values.npy"):
    #     filename = os.path.basename(file_path)
    #     patient_id = re.match(r'([a-f0-9].*)_transfer_values.npy', filename).group(1)
    #     patient_ids.add(patient_id)
    #
    # sample_submission = pd.read_csv(STAGE1_SUBMISSION)
    # #df = pd.merge(sample_submission, patient_ids_df, how='inner', on=['id'])
    # test_patient_ids = set(sample_submission['id'].tolist())
    # train_patient_ids = patient_ids.difference(test_patient_ids)
    # train_inputs = get_patient_features(train_patient_ids)
    # train_labels = get_patient_labels(train_patient_ids)
    #
    # num_patients = len(train_patient_ids)
    # X = np.ndarray(shape=(num_patients, FEATURES_SHAPE * FEATURES_SHAPE), dtype=np.float32)
    # Y = np.ndarray(shape=(num_patients), dtype=np.float32)
    #
    # count = 0
    # for key in train_inputs.keys():
    #     X[count] = train_inputs[key]
    #     Y[count] = train_labels[key]
    #     count = count + 1
    #
    # print('Loaded train data for {} patients'.format(count))
    # print("Total time to load data: " + str(timedelta(seconds=int(round(time.time() - time0)))))
    # print('\nSplitting data into train, validation')
    # train_x, validation_x, train_y, validation_y = model_selection.train_test_split(X, Y, random_state=42, stratify=Y, test_size=0.20)
    #
    # del X
    # del Y
    #
    # print('train_x: {}'.format(train_x.shape))
    # print('validation_x: {}'.format(validation_x.shape))
    # print('train_y: {}'.format(train_y.shape))
    # print('validation_y: {}'.format(validation_y.shape))
    #
    # print('\nTraining..')
    # clf = train_xgboost(train_x, validation_x, train_y, validation_y)
    #
    # del train_x, train_y, validation_x, validation_y
    #
    # print('\nPredicting on validation set')
    # validation_y_predicted = clf.predict(validation_x)
    # validation_log_loss = sklearn.metrics.log_loss(validation_y, validation_y_predicted, eps=1e-15)
    # print('Post-trian validation log loss: {}'.format(validation_log_loss))
    # #print(validation_y)
    # #print(validation_y_predicted)
    #
    # num_patients = len(test_patient_ids)
    # test_inputs = get_patient_features(test_patient_ids)
    # X = np.ndarray(shape=(num_patients, FEATURES_SHAPE * FEATURES_SHAPE), dtype=np.float32)
    #
    # timestamp = str(int(time.time()))
    # filename = OUTPUT_PATH + 'submission-' + timestamp + ".csv"
    #
    # with open(filename, 'w') as csvfile:
    #     submission_writer = csv.writer(csvfile, delimiter=',')
    #     submission_writer.writerow(['id', 'cancer'])
    #
    #     print('\nPredicting on test set')
    #     for key in test_inputs.keys():
    #         x = test_inputs[key]
    #         y = clf.predict([x])
    #         submission_writer.writerow([key, y[0]])
    #
    # print('Generated submission file: {}'.format(filename))

if __name__ == '__main__':
    start_time = time.time()
    OUTPUT_PATH = '/kaggle/dev/data-science-bowl-2017-data/submissions/'
    DATA_PATH = '/kaggle/dev/data-science-bowl-2017-data/stage1_features_v3/'
    LABELS = '/kaggle/dev/data-science-bowl-2017-data/stage1_labels.csv'
    STAGE1_SUBMISSION = '/kaggle/dev/data-science-bowl-2017-data/stage1_sample_submission.csv'
    TENSORBOARD_SUMMARIES = '/kaggle/dev/data-science-bowl-2017-data/tensorboard_summaries/'
    MODELS = '/kaggle/dev/data-science-bowl-2017-data/models/'

    #globals initializing
    FLAGS = tf.app.flags.FLAGS

    ## Prediction problem specific
    tf.app.flags.DEFINE_integer('iteration_analysis', 100,
                                """Number of steps after which analysis code is executed""")
    tf.app.flags.DEFINE_integer('num_classes', 2,
                                """Number of classes to predict.""")
    tf.app.flags.DEFINE_integer('num_classes_luna', 4,
                                """Number of classes predicted by LUNA model.""")
    tf.app.flags.DEFINE_integer('transfer_values_shape', 512,
                                'Size of transfer values')
    tf.app.flags.DEFINE_integer('batch_size', 128,
                                """Number of items in a batch.""")
    tf.app.flags.DEFINE_integer('max_iterations', 60000,
                                """Number of batches to run.""")
    tf.app.flags.DEFINE_float('reg_constant', 0.1, 'Regularization constant.')
    tf.app.flags.DEFINE_float('dropout', 0.5, 'Dropout')

    ## Tensorflow specific
    tf.app.flags.DEFINE_boolean('log_device_placement', False,
                                """Whether to log device placement.""")
    tf.app.flags.DEFINE_boolean('allow_soft_placement', True,
                                """Whether to allow soft placement of calculations by tf.""")
    tf.app.flags.DEFINE_boolean('allow_growth', True,
                                """Whether to allow GPU growth by tf.""")

    train_nn()
    end_time = time.time()
    print("Total Time usage: " + str(timedelta(seconds=int(round(end_time - start_time)))))
