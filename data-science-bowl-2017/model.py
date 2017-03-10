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
        prob_pred = np.zeros(shape=[num_images, FLAGS.num_classes], dtype=np.float64)

        # Now calculate the predicted probs for the batches.

        # The starting index for the next batch is denoted i.
        i = 0

        while i < num_images:
            # The ending index for the next batch is denoted j.
            if kwargs:
                j = min(i + FLAGS.batch_size, num_images)
            else:
                j = 1

            # Create a feed-dict with the images and labels
            # between index i and j.
            if kwargs:
                feed_dict = {x: transfer_values[i:j],
                             y_labels: labels[i:j]}
            else:
                feed_dict = {x: transfer_values[i:j],
                             y_labels: np.zeros([j-i, FLAGS.num_classes], dtype=np.float32)}

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

    def submission(timestamp):
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
        filename = submissions + 'submission-' + str(timestamp) + ".csv"
        submission.to_csv(filename, index=False)

        patient_count = submission['id'].count()
        predicted = submission['cancer'].count()


        return patient_count, predicted, filename


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

    test_labels = (np.arange(FLAGS.num_classes) == validation_y[:, None])+0
    train_labels = (np.arange(FLAGS.num_classes) == train_y[:, None])+0

    # Best validation accuracy seen so far.
    best_validation_loss = 100.0

    # Iteration-number for last improvement to validation accuracy.
    last_improvement = 0

    require_improvement = int((FLAGS.require_improvement) * (FLAGS.max_iterations))
    iteration_analysis = int((FLAGS.iteration_analysis)*(FLAGS.max_iterations))


    # Graph construction
    graph = tf.Graph()
    with graph.as_default():

        model = inception.Inception()
        transfer_len = model.transfer_len

        # for i in range(FLAGS.num_gpus):
        #     with tf.device('/gpu:%d' % i):
        with tf.name_scope('layer1'):
            x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')
            y = tf.placeholder(tf.float32, shape=[None, FLAGS.num_classes], name='y')
            y_labels = tf.placeholder(tf.float32, shape=[None, FLAGS.num_classes], name='y_labels')
            with tf.name_scope('weights'):
                W = tf.Variable(tf.zeros([transfer_len, FLAGS.num_classes]))
                variable_summaries(W)
            with tf.name_scope('biases'):
                b = tf.Variable(tf.zeros([FLAGS.num_classes]))
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
        saver = tf.train.Saver()

    # Setting up config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = FLAGS.allow_growth
    config.log_device_placement=FLAGS.log_device_placement
    config.allow_soft_placement=FLAGS.allow_soft_placement

    all_timstamps = []
    # timestamp used to identify the start of run
    start_timestamp = str(int(time.time()))
    all_timstamps.append(start_timestamp)

    # Session construction
    with tf.Session(graph=graph, config=config) as sess:
        train_writer = tf.summary.FileWriter(tensorboard_summaries + '/train-' + start_timestamp, sess.graph)
        sess.run(tf.global_variables_initializer())

        print('\nPre-train validation log loss: {0:.5}'.format(calc_validation_log_loss()))

        for i in tqdm(range(FLAGS.max_iterations)):
            x_batch, y_batch = get_batch(train_x, train_labels, FLAGS.batch_size)
            _, step_summary, loss_val = sess.run([optimizer, merged, log_loss], feed_dict={x: x_batch, y_labels: y_batch})
            train_writer.add_summary(step_summary, i)

            # Iteration analysis after every iteration_analysis iterations and after last iteration
            if (i % iteration_analysis == 0) or (i == (i - 1)):
                training_loss = loss_val
                cv_loss = calc_validation_log_loss()

                if cv_loss < best_validation_loss:
                    best_validation_loss = cv_loss
                    last_improvement = i

                    # timestamp used to update
                    update_timestamp = str(int(time.time()))
                    all_timstamps.append(update_timestamp)

                    # Save model
                    checkpoint_loc = os.path.join(model_checkpoints, 'checkpoint-' + update_timestamp )
                    if not os.path.exists(checkpoint_loc):
                        os.makedirs(checkpoint_loc)
                    save_path = os.path.join(checkpoint_loc, 'best_validation-'+ update_timestamp)
                    saver.save(sess=sess, save_path=save_path)

                    # Add to Tensorboard
                    test_writer = tf.summary.FileWriter(tensorboard_summaries + '/test-' + update_timestamp)

                    # Create prediction
                    patient_count, predicted, filename = submission(update_timestamp)

                    # Output message
                    metrics_msg = "Improvement found on iteration:{0:>6}, Train-Batch Log Loss: {1:f}, Validation Log Loss: {2:f}"
                    print(metrics_msg.format(i + 1, training_loss, cv_loss))
                    output_msg = "Submission file: {0:}"
                    print(output_msg.format(filename))
                    print('\nTensorboard runs: train-{} test-{}'. format(start_timestamp, update_timestamp))


                # If no improvement found in the required number of iterations.
                if i - last_improvement > require_improvement:
                    print("No improvement found in a while, stopping optimization.")
                    break


        print('Post-train validation log loss: {0:.5}'.format(calc_validation_log_loss()))
        print(all_timstamps)

        # submission()
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
    model_checkpoints = '/kaggle/dev/data-science-bowl-2017-data/models/checkpoints/'


    #globals initializing
    FLAGS = tf.app.flags.FLAGS

    ## Prediction problem specific
    tf.app.flags.DEFINE_integer('num_classes', 2,
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

    make_submission()
    end_time = time.time()
    print("Total Time usage: " + str(timedelta(seconds=int(round(end_time - start_time)))))
