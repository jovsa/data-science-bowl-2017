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
import scipy

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

def train_nn():
    def img_to_rgb(im):
        w, h = im.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = im
        ret[:, :, 1] = im
        ret[:, :, 2] = im
        return ret

    def get_batch(x, y, batch_size):
        num_images = len(x)
        idx = np.random.choice(num_images,
                               size=batch_size,
                               replace=False)
        x_batch = x[idx]
        y_batch = y[idx]

        return x_batch, y_batch

    def predict_prob_validation(transfer_values, labels, write_to_tensorboard=False):
        num_images = len(transfer_values)
        prob_pred = np.zeros(shape=[num_images, FLAGS.num_classes], dtype=np.float64)

        i = 0
        while i < num_images:
            j = min(i + FLAGS.batch_size, num_images)
            feed_dict = {x: transfer_values[i:j],
                         y_labels: labels[i:j]}

            y_calc, step_summary = sess.run([y, merged], feed_dict=feed_dict)
            prob_pred[i:j] = y_calc

            if write_to_tensorboard:
                validation_writer.add_summary(step_summary, i)
            # Set the start-index for the next batch to the
            # end-index of the current batch.
            i = j
        return prob_pred

    def predict_prob_test(transfer_values):
        num_images = len(transfer_values)
        prob_pred = np.zeros(shape=[num_images, FLAGS.num_classes], dtype=np.float64)

        i = 0
        while i < num_images:
            j = 1
            feed_dict = {x: transfer_values[i:j],
                         y_labels: np.zeros([j, FLAGS.num_classes], dtype=np.float32)}

            y_calc, step_summary = sess.run([y, merged], feed_dict=feed_dict)
            prob_pred[i:j] = y_calc
            # Set the start-index for the next batch to the
            # end-index of the current batch.
            i = j
        return prob_pred

    def calc_validation_log_loss(write_to_tensorboard = False):
        prob_pred = predict_prob_validation(transfer_values = validation_x, labels = validation_labels, write_to_tensorboard = write_to_tensorboard)
        p = np.maximum(np.minimum(prob_pred, 1-10e-15), 10e-15)
        l = np.transpose(validation_labels + 0.0)
        n = validation_labels.shape[0]
        temp = np.matmul(l, np.log(p)) + np.matmul((1 - l), np.log(1 - p))

        # Divide by 2 (magic number)
        validation_log_loss = -1.0 * (temp[0,0] + temp[1,1])/(2 * n)

        return validation_log_loss

    def submission(timestamp):
        ids = list()
        for s in glob.glob(stage1_features_inception + "*"):
            id = os.path.basename(s)
            id = re.match(r'lungs_pca_([a-f0-9].*).npy' , id).group(1)
            ids.append(id)
        ids = pd.DataFrame(ids,  columns=["id"])

        submission_sample = pd.read_csv(stage1_submission)
        df = pd.merge(submission_sample, ids, how='inner', on=['id'])
        x_test = np.array([np.mean(np.load(stage1_features_inception + "lungs_pca_" + s + ".npy"), axis=0) for s in df['id'].tolist()])

        for i in range(0, len(x_test)):
            pred = predict_prob_test(transfer_values = x_test[i].reshape(1,-1))
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

    def new_weights(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def new_biases(length):
        return tf.Variable(tf.constant(0.05, shape=[length]))

    def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

        # Shape of the filter-weights for the convolution.
        # This format is determined by the TensorFlow API.
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        # Create new weights aka. filters with the given shape.
        weights = new_weights(shape=shape)

        # Create new biases, one for each filter.
        biases = new_biases(length=num_filters)

        # Create the TensorFlow operation for convolution.
        # Note the strides are set to 1 in all dimensions.
        # The first and last stride must always be 1,
        # because the first is for the image-number and
        # the last is for the input-channel.
        # But e.g. strides=[1, 2, 2, 1] would mean that the filter
        # is moved 2 pixels across the x- and y-axis of the image.
        # The padding is set to 'SAME' which means the input image
        # is padded with zeroes so the size of the output is the same.
        layer = tf.nn.conv2d(input=input,
                             filter=weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')

        # Add the biases to the results of the convolution.
        # A bias-value is added to each filter-channel.
        layer += biases

        # Use pooling to down-sample the image resolution?
        if use_pooling:
            # This is 2x2 max-pooling, which means that we
            # consider 2x2 windows and select the largest value
            # in each window. Then we move 2 pixels to the next window.
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')

        # Rectified Linear Unit (ReLU).
        # It calculates max(x, 0) for each input pixel x.
        # This adds some non-linearity to the formula and allows us
        # to learn more complicated functions.
        layer = tf.nn.relu(layer)

        # Note that ReLU is normally executed before the pooling,
        # but since relu(max_pool(x)) == max_pool(relu(x)) we can
        # save 75% of the relu-operations by max-pooling first.

        # We return both the resulting layer and the filter-weights
        # because we will plot the weights later.
        return layer, weights

    def flatten_layer(layer):
        # Get the shape of the input layer.
        layer_shape = layer.get_shape()

        # The shape of the input layer is assumed to be:
        # layer_shape == [num_images, img_height, img_width, num_channels]

        # The number of features is: img_height * img_width * num_channels
        # We can use a function from TensorFlow to calculate this.
        num_features = layer_shape[1:4].num_elements()

        # Reshape the layer to [num_images, num_features].
        # Note that we just set the size of the second dimension
        # to num_features and the size of the first dimension to -1
        # which means the size in that dimension is calculated
        # so the total size of the tensor is unchanged from the reshaping.
        layer_flat = tf.reshape(layer, [-1, num_features])

        # The shape of the flattened layer is now:
        # [num_images, img_height * img_width * num_channels]

        # Return both the flattened layer and the number of features.
        return layer_flat, num_features

    def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

        # Create new weights and biases.
        weights = new_weights(shape=[num_inputs, num_outputs])
        biases = new_biases(length=num_outputs)

        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        layer = tf.matmul(input, weights) + biases

        # Use ReLU?
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer



    img_height = 299
    img_width = 299

    # Convolutional Layer 1.
    num_channels1 = 3
    filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
    num_filters1 = 16         # There are 16 of these filters.

    # Convolutional Layer 2.
    filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
    num_filters2 = 36         # There are 36 of these filters.

    # Fully-connected layer.
    fc_size = 128             # Number of neurons in fully-connected layer.


    img_size_flat = img_height * img_width
    ids = list()
    for s in glob.glob(stage1_processed_pca + "*"):
        id = os.path.basename(s)
        id = re.match(r'lungs_pca_([a-f0-9].*).npy' , id).group(1)
        ids.append(id)
    ids = pd.DataFrame(ids,  columns=["id"])

    df = pd.read_csv(labels)
    df = pd.merge(df, ids, how='inner', on=['id'])

    X = np.array([np.load(stage1_processed_pca + "lungs_pca_" + s + ".npy") for s in df['id'].tolist()])

    # Adding channel to input data
    for i in range(X.shape[0]):
        X[i] = img_to_rgb(X[i])
        X[i] = scipy.misc.imresize(X[i], [img_height,img_width], interp='bilinear')

    Y = df['cancer'].as_matrix()
    train_x, validation_x, train_y, validation_y = model_selection.train_test_split(X, Y, random_state=42, stratify=Y,
                                                                    test_size=0.20)

    validation_labels = (np.arange(FLAGS.num_classes) == validation_y[:, None])+0
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

        # for i in range(FLAGS.num_gpus):
        #     with tf.device('/gpu:%d' % i):
        with tf.name_scope('base_cnn'):
            x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
            y = tf.placeholder(tf.float32, shape=[None, FLAGS.num_classes], name='y')
            y_labels = tf.placeholder(tf.float32, shape=[None, FLAGS.num_classes], name='y_labels')
            with tf.name_scope('cv_1'):
                layer_conv1, weights_conv1 = \
                    new_conv_layer(input=x,
                                   num_input_channels=num_channels1,
                                   filter_size=filter_size1,
                                   num_filters=num_filters1,
                                   use_pooling=True)
            with tf.name_scope('cv_2'):
                layer_conv2, weights_conv2 = \
                    new_conv_layer(input=layer_conv1,
                                   num_input_channels=num_filters1,
                                   filter_size=filter_size2,
                                   num_filters=num_filters2,
                                   use_pooling=True)

            with tf.name_scope('fc_1'):
                layer_flat, num_features = flatten_layer(layer_conv2)

                layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)

            with tf.name_scope('fc_2'):
                layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)


            y = tf.nn.softmax(layer_fc2)
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

    all_timstamps = ()
    # timestamp used to identify the start of run
    start_timestamp = str(int(time.time()))
    all_timstamps = all_timstamps + ({'train', start_timestamp},)


    # Session construction
    with tf.Session(graph=graph, config=config) as sess:
        train_writer = tf.summary.FileWriter(tensorboard_summaries + '/train-' + start_timestamp, sess.graph)
        sess.run(tf.global_variables_initializer())

        print("before Pre-train")
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
                    all_timstamps = all_timstamps + ({'validation', update_timestamp},)

                    # Save model
                    checkpoint_loc = os.path.join(model_checkpoints, 'checkpoint-' + update_timestamp )
                    if not os.path.exists(checkpoint_loc):
                        os.makedirs(checkpoint_loc)
                    save_path = os.path.join(checkpoint_loc, 'best_validation-'+ update_timestamp)
                    saver.save(sess=sess, save_path=save_path)

                    # Add to Tensorboard
                    validation_writer = tf.summary.FileWriter(tensorboard_summaries + '/validation-' + update_timestamp)
                    calc_validation_log_loss(write_to_tensorboard=True)

                    # Create prediction on test and output a submission
                    patient_count, predicted, filename = submission(update_timestamp)

                    # Output message
                    metrics_msg = "Improvement found on iteration:{0:>6}, Train-Batch Log Loss: {1:f}, Validation Log Loss: {2:f}"
                    print(metrics_msg.format(i + 1, training_loss, cv_loss))
                    output_msg = "Submission file: {0:}"
                    print(output_msg.format(filename))

                # If no improvement found in the required number of iterations.
                if i - last_improvement > require_improvement:
                    print("No improvement found in a while, you should consider stopping the optimization.")
                    # break

        print('Post-train validation log loss: {0:.5}'.format(calc_validation_log_loss()))
        print(all_timstamps)
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
