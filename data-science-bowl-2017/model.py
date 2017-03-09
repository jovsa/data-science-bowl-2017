import helpers.helpers as helpers
import helpers.cache as cache
import helpers.download as download
import helpers.inception as inception
from helpers.inception import transfer_values_cache

import numpy as np
import pandas as pd
import re
import sys
import datetime
import matplotlib
# Force matplotlib to not use any Xwindows backend, so that you can output graphs
matplotlib.use('Agg')
import cv2
import dicom
import os
from sklearn import model_selection
import glob
from matplotlib import pyplot as plt
import math
from sklearn.decomposition import PCA
import time
from datetime import timedelta
import tensorflow as tf
import prettytensor as pt
from tqdm import tqdm

# Fixes "SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame"
pd.options.mode.chained_assignment = None

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
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
    def get_batch(x, y, batch_size):
        # Number of images (transfer-values) in the training-set.
        num_images = len(x)

        # Create a random index.
        idx = np.random.choice(num_images,
                               size=batch_size,
                               replace=False)

        # Use the random index to select random x and y-values.
        # We use the transfer-values instead of images as x-values.
        x_batch = x[idx]
        y_batch = y[idx]

        return x_batch, y_batch

    def predict_prob(transfer_values, **kwargs):
        # Number of images.
        num_images = len(transfer_values)
        if kwargs:
            labels = kwargs['labels']

        # Allocate an array for the predicted probs which
        # will be calculated in batches and filled into this array.
        prob_pred = np.zeros(shape=[num_images, num_classes], dtype=np.float64)

        # Now calculate the predicted probs for the batches.

        # The starting index for the next batch is denoted i.
        i = 0

        while i < num_images:
            # The ending index for the next batch is denoted j.
            if kwargs:
                j = min(i + batch_size, num_images)
            else:
                j = 1

            # Create a feed-dict with the images and labels
            # between index i and j.
            if kwargs:
                feed_dict = {x: transfer_values[i:j],
                             y_labels: labels[i:j]}
            else:
                feed_dict = {x: transfer_values[i:j],
                             y_labels: np.zeros([j-i, num_classes], dtype=np.float32)}

            # Calculate the predicted class using TensorFlow.
            y_calc, step_summary = sess.run([y, merged], feed_dict=feed_dict)
            prob_pred[i:j] = y_calc
            test_writer.add_summary(step_summary, i)

            # Set the start-index for the next batch to the
            # end-index of the current batch.
            i = j

        # Create a boolean array whether each image is correctly classified.
        # print('Predicted prob: ' + str(prob_pred.shape))

        return prob_pred

    def predict_prob_test():
        return predict_prob(transfer_values = validation_x,
                           labels = test_labels)

    def submission():
        ids = list()
        for s in glob.glob(stage1_features_inception + "*"):
            id = os.path.basename(s)
            id = re.match(r'inception_cifar10_([a-f0-9].*).pkl' , id).group(1)
            ids.append(id)
        ids = pd.DataFrame(ids,  columns=["id"])

        submission_sample = pd.read_csv(stage1_submission)
        df = pd.merge(submission_sample, ids, how='inner', on=['id'])
        x_test = np.array([np.mean(np.load(stage1_features_inception + "inception_cifar10_" + s + ".pkl"), axis=0) for s in df['id'].tolist()])

        for i in range(0, len(x_test)):
            pred = predict_prob(transfer_values = x_test[i].reshape(1,-1))
            df['cancer'][i] = pred[0,1]

        #Submission preparation
        submission = pd.merge(submission_sample, df, how='left', on=['id'])
        submission = submission.iloc[:,(0,2)]
        submission = submission.rename(index=str, columns={"cancer_y": "cancer"})

        # Outputting submission file
        timestamp = datetime.datetime.now().isoformat()
        filename = submissions + 'submission-' + str(timestamp) + ".csv"
        submission.to_csv(filename, index=False)

        # Submission file analysis
        print("\n---- Submission file analysis ----")
        patient_count = submission['id'].count()
        predecited = submission['cancer'].count()
        print("Total number of patients: " + str(patient_count))
        print("Number of predictions: " + str(predecited))
        print("\nSubmission file stored at: " + filename)

    def calc_validation_log_loss():
        # For all the images in the test-set,
        # calculate the predicted classes and whether they are correct.
        prob_pred = predict_prob_test()
        p = np.maximum(np.minimum(prob_pred, 1-10e-15), 10e-15)
        l = np.transpose(test_labels + 0.0)
        n = test_labels.shape[0]
        temp = np.matmul(l, np.log(p)) + np.matmul((1 - l), np.log(1 - p))

        # Divide by 2 (magic number)
        validation_log_loss = -1.0 * (temp[0,0] + temp[1,1])/(2 * n)

        return validation_log_loss

    num_classes = 2
    batch_size = 10
    ids = list()
    for s in glob.glob(stage1_features_inception + "*"):
        id = os.path.basename(s)
        id = re.match(r'inception_cifar10_([a-f0-9].*).pkl' , id).group(1)
        ids.append(id)
    ids = pd.DataFrame(ids,  columns=["id"])

    df = pd.read_csv(labels)
    df = pd.merge(df, ids, how='inner', on=['id'])

    X = np.array([np.mean(np.load(stage1_features_inception + "inception_cifar10_" + s + ".pkl"), axis=0) for s in df['id'].tolist()])
    Y = df['cancer'].as_matrix()

    train_x, validation_x, train_y, validation_y = model_selection.train_test_split(X, Y, random_state=42, stratify=Y,
                                                                    test_size=0.20)

    test_labels = (np.arange(num_classes) == validation_y[:, None])+0
    train_labels = (np.arange(num_classes) == train_y[:, None])+0

    graph = tf.Graph()
    with graph.as_default():
        # for i in range(FLAGS.num_gpus):
        #     with tf.device('/gpu:%d' % i):
        model = inception.Inception()
        transfer_len = model.transfer_len

        with tf.name_scope('layer1'):
            x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')
            y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
            y_labels = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_labels')
            with tf.name_scope('weights'):
                W = tf.Variable(tf.zeros([transfer_len, num_classes]))
                variable_summaries(W)
            with tf.name_scope('biases'):
                b = tf.Variable(tf.zeros([num_classes]))
                variable_summaries(b)
            with tf.name_scope('Wx_plus_b'):
                logits = tf.matmul(x, W) + b
                tf.summary.histogram('Wx_plus_b', logits)

            y = tf.nn.softmax(logits)
            tf.summary.histogram('y', y)

            with tf.name_scope('log_loss'):
                log_loss = tf.losses.log_loss(y_labels, y, epsilon=10e-15)
                tf.summary.scalar('log_loss', log_loss)

            with tf.name_scope('train'):
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(log_loss)

        merged = tf.summary.merge_all()

    timestamp = str(int(time.time()))

    # Setting up config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement=FLAGS.log_device_placement
    config.allow_soft_placement=FLAGS.allow_soft_placement
    with tf.Session(graph=graph, config=config) as sess:
        train_writer = tf.summary.FileWriter(tensorboard_summaries + '/train-' + timestamp, sess.graph)
        test_writer = tf.summary.FileWriter(tensorboard_summaries + '/test-' + timestamp)
        sess.run(tf.global_variables_initializer())

        print('\nPre-train validation log loss: {0:.5}'.format(calc_validation_log_loss()))
        for i in tqdm(range(FLAGS.max_steps)):
            x_batch, y_batch = get_batch(train_x, train_labels, batch_size)
            _, step_summary, loss_val = sess.run([optimizer, merged, log_loss], feed_dict={x: x_batch, y_labels: y_batch})
            train_writer.add_summary(step_summary, i)
            # print('Batch {0} Log_loss: {1:.5}'.format(i, loss_val))

        print('Post-train validation log loss: {0:.5}'.format(calc_validation_log_loss()))

        print('\nTensorboard runs: train-{} test-{}'. format(timestamp, timestamp))
        submission()
        sess.close() #clossing the session for good measure


def make_submission():
    clf = train_nn()

if __name__ == '__main__':
    start_time = time.time()
    data = '/kaggle/dev/data-science-bowl-2017-data/'
    stage1 = '/kaggle/dev/data-science-bowl-2017-data/stage1/'
    labels = '/kaggle/dev/data-science-bowl-2017-data/stage1_labels.csv'
    stage1_processed = '/kaggle/dev/data-science-bowl-2017-data/stage1_processed/'
    stage1_features_resnet = '/kaggle/dev/data-science-bowl-2017-data/stage1_features_mx/'
    stage1_submission = '/kaggle/dev/data-science-bowl-2017-data/stage1_sample_submission.csv'
    naive_submission = '/kaggle/dev/jovan/data-science-bowl-2017/data-science-bowl-2017/submissions/naive_submission.csv'
    stage1_processed_pca = '/kaggle/dev/data-science-bowl-2017-data/stage1_processed_pca/'
    stage1_features_inception = '/kaggle/dev/data-science-bowl-2017-data/CIFAR-10/cache/'
    submissions = '/kaggle/dev/data-science-bowl-2017-data/submissions/'
    tensorboard_summaries = '/kaggle/dev/data-science-bowl-2017-data/tensorboard_summaries'

    #globals initializing
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer('max_steps', 10000,
                                """Number of batches to run.""")
    tf.app.flags.DEFINE_integer('num_gpus', 2,
                                """How many GPUs to use.""")
    tf.app.flags.DEFINE_boolean('log_device_placement', False,
                                """Whether to log device placement.""")
    tf.app.flags.DEFINE_boolean('allow_soft_placement', True,
                                """Whether to allow soft placement of calculations by tf.""")

    make_submission()
    end_time = time.time()
    print("Total Time usage: " + str(timedelta(seconds=int(round(end_time - start_time)))))
