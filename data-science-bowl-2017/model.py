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
#import mxnet as mx
from sklearn import cross_validation
import glob
from matplotlib import pyplot as plt
import math
from sklearn.decomposition import PCA
import time
from datetime import timedelta
import tensorflow as tf
import prettytensor as pt

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

    def predict_prob(transfer_values, labels):
        # Number of images.
        num_images = len(transfer_values)

        # Allocate an array for the predicted probs which
        # will be calculated in batches and filled into this array.
        prob_pred = np.zeros(shape=[num_images, num_classes], dtype=np.float64)

        # Now calculate the predicted probs for the batches.

        # The starting index for the next batch is denoted i.
        i = 0

        while i < num_images:
            # The ending index for the next batch is denoted j.
            j = min(i + batch_size, num_images)

            # Create a feed-dict with the images and labels
            # between index i and j.
            feed_dict = {x: transfer_values[i:j],
                         y_: labels[i:j]}

            # Calculate the predicted class using TensorFlow.
            y_temp = sess.run(y, feed_dict=feed_dict)
            # print(np.argmax(y_temp, axis=1))
            prob_pred[i:j] = y_temp

            # Set the start-index for the next batch to the
            # end-index of the current batch.
            i = j

        # Create a boolean array whether each image is correctly classified.
        # print('Predicted prob: ' + str(prob_pred.shape))

        return prob_pred

    def predict_prob_test():
        return predict_prob(transfer_values = validation_x,
                           labels = test_labels)

    def print_validation_log_loss():

        # For all the images in the test-set,
        # calculate the predicted classes and whether they are correct.
        prob_pred = predict_prob_test()
        p = np.maximum(np.minimum(prob_pred, 1-10e-15), 10e-15)
        l = np.transpose(test_labels + 0.0)
        n = test_labels.shape[0]
        temp = np.matmul(l, np.log(p)) + np.matmul((1 - l), np.log(1 - p))

        # Divide by 2 (magic number)
        validation_log_loss = -1.0 * (temp[0,0] + temp[1,1])/(2 * n)

        print('Validation log loss: {0:.5}'.format(validation_log_loss))

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

    train_x, validation_x, train_y, validation_y = cross_validation.train_test_split(X, Y, random_state=42, stratify=Y,
                                                                    test_size=0.20)

    test_labels = (np.arange(num_classes) == validation_y[:, None])+0
    train_labels = (np.arange(num_classes) == train_y[:, None])+0

    model = inception.Inception()
    transfer_len = model.transfer_len

    x = tf.placeholder(tf.float32, shape=[batch_size, transfer_len], name='x')
    y_ = tf.placeholder(tf.float32, shape=[batch_size, num_classes], name='y')
    # y_class = tf.argmax(y_, dimension=1)

    W = tf.Variable(tf.zeros([transfer_len, num_classes]))
    b = tf.Variable(tf.zeros([batch_size, num_classes]))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        logits = tf.matmul(x, W) + b
        y = tf.nn.softmax(logits)
        log_loss = tf.losses.log_loss(y_, y, epsilon=10e-15)
        train_step = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(log_loss)

        print_validation_log_loss()
        for i in range(1000):
            batch = get_batch(train_x, train_labels, batch_size)
            _, loss_val = sess.run([train_step, log_loss], feed_dict={x: batch[0], y_: batch[1]})
            print('Batch {0} Log_loss: {1:.5}'.format(i, loss_val))

        print_validation_log_loss()

def make_submission():
    clf = train_nn()

if __name__ == '__main__':
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

    ## nn hyper-params
    num_classes = 2
    train_batch_size = 64

    make_submission()
    print("done")
