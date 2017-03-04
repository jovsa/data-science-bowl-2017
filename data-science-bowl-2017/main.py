# main

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
import xgboost as xgb
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

######################
def pre_process():
    # Pre-processing
    stage1_loc = helpers.verify_location(stage1)
    labels_loc = helpers.verify_location(labels)

    patient_data = helpers.folder_explorer(stage1_loc)
    patient_data = pd.DataFrame(list(patient_data.items()), columns=["id", "scans"])
    labels = pd.read_csv(labels_loc)

    data = pd.merge(patient_data, labels, how="inner", on=['id'])
    return

######################

def get_extractor():
    model = mx.model.FeedForward.load('models/resnet-50', 0, ctx=mx.cpu(), numpy_batch_size=1)
    fea_symbol = model.symbol.get_internals()["flatten0_output"]
    feature_extractor = mx.model.FeedForward(ctx=mx.cpu(), symbol=fea_symbol, numpy_batch_size=64,
                                             arg_params=model.arg_params, aux_params=model.aux_params,
                                             allow_extra_params=True)

    return feature_extractor


def get_3d_data(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return np.stack([s.pixel_array for s in slices])


def get_data_id(path):
    sample_image = np.load(path)
    sample_image[sample_image == -2000] = 0

    batch = []
    cnt = 0
    dx = int((40.0/512.0) * sample_image.shape[1])
    ds = sample_image.shape[1]
    for i in range(0, sample_image.shape[0] - 3, 3):
        tmp = []
        for j in range(3):
            img = sample_image[i + j]
            img = 255.0 / np.amax(img) * img
            img = cv2.equalizeHist(img.astype(np.uint8))
            img = img[dx: ds - dx, dx: ds - dx]
            img = cv2.resize(img, (224, 224))
            tmp.append(img)

        tmp = np.array(tmp)
        batch.append(np.array(tmp))

    batch = np.array(batch)
    return batch

def calc_features():
    net = get_extractor()
    count = 0
    for folder in glob.glob(stage1_processed + 'segment_lungs_fill_*'):
        p_id = re.match(r'segment_lungs_fill_([a-f0-9].*).npy', os.path.basename(folder)).group(1)
        print('Processing patient ' + str(count) + ' id: ' + p_id)
        batch = get_data_id(folder)
        feats = net.predict(batch)
        np.save(stage1_features + p_id, feats)
        count = count + 1

def normalize_scans(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def normalize_general(image):
    MIN_BOUND = np.min(image)
    MAX_BOUND = np.max(image)
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    return image

def zero_center(image):
    PIXEL_MEAN = 0.25
    image = image - PIXEL_MEAN
    return image

def train_xgboost():
    ids = list()
    for s in glob.glob(stage1_features_inception + "*"):
        id = os.path.basename(s)
        id = re.match(r'inception_cifar10_([a-f0-9].*).pkl' , id).group(1)
        ids.append(id)
    ids = pd.DataFrame(ids,  columns=["id"])

    df = pd.read_csv(labels)
    df = pd.merge(df, ids, how='inner', on=['id'])


    x = np.array([np.mean(np.load(stage1_features_inception + "inception_cifar10_" + s + ".pkl"), axis=0) for s in df['id'].tolist()])

    y = df['cancer'].as_matrix()
    trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=42, stratify=y,
                                                                    test_size=0.20)

    clf = xgb.XGBRegressor(max_depth=10,
                           gamma=0.5,
                           objective="binary:logistic",
                           n_estimators=1500,
                           min_child_weight=6,
                           learning_rate=0.005,
                           nthread=8,
                           subsample=0.80,
                           colsample_bytree=0.80,
                           seed=79,
                           max_delta_step=1,
                           reg_alpha=0.1,
                           reg_lambda=0.5)

    clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=50)
    return clf

def train_nn():
    ###########
    # Start of transfer learning code
    def random_batch():
        # Number of images (transfer-values) in the training-set.
        num_images = len(transfer_values_train)

        # Create a random index.
        idx = np.random.choice(num_images,
                               size=train_batch_size,
                               replace=False)

        # Use the random index to select random x and y-values.
        # We use the transfer-values instead of images as x-values.
        x_batch = transfer_values_train[idx]
        y_batch = labels_train[idx]

        return x_batch, y_batch

    def optimize(num_iterations):
        # Start-time used for printing time-usage below.
        start_time = time.time()

        for i in range(num_iterations):
            # Get a batch of training examples.
            # x_batch now holds a batch of images (transfer-values) and
            # y_true_batch are the true labels for those images.
            x_batch, y_true_batch = random_batch()

            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            feed_dict_train = {x: x_batch,
                               y_true: y_true_batch}

            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            # We also want to retrieve the global_step counter.
            i_global, _ = session.run([global_step, optimizer],
                                      feed_dict=feed_dict_train)

            # Print status to screen every 100 iterations (and last).
            if (i_global % 100 == 0) or (i == num_iterations - 1):
                # Calculate the accuracy on the training-batch.
                batch_acc = session.run(accuracy,
                                        feed_dict=feed_dict_train)

                # Print status.
                msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
                print(msg.format(i_global, batch_acc))

        # Ending time.
        end_time = time.time()

        # Difference between start and end-times.
        time_dif = end_time - start_time

        # Print the time-usage.
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

    def plot_example_errors(cls_pred, correct):
        # This function is called from print_test_accuracy() below.

        # cls_pred is an array of the predicted class-number for
        # all images in the test-set.

        # correct is a boolean array whether the predicted class
        # is equal to the true class for each image in the test-set.

        # Negate the boolean array.
        incorrect = (correct == False)

        # Get the images from the test-set that have been
        # incorrectly classified.
        images = images_test[incorrect]

        # Get the predicted classes for those images.
        cls_pred = cls_pred[incorrect]

        # Get the true classes for those images.
        cls_true = cls_test[incorrect]

        n = min(9, len(images))

        # Plot the first n images.
        plot_images(images=images[0:n],
                    cls_true=cls_true[0:n],
                    cls_pred=cls_pred[0:n])

    # Import a function from sklearn to calculate the confusion-matrix.
    from sklearn.metrics import confusion_matrix

    def plot_confusion_matrix(cls_pred):
        # This is called from print_test_accuracy() below.

        # cls_pred is an array of the predicted class-number for
        # all images in the test-set.

        # Get the confusion matrix using sklearn.
        cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                              y_pred=cls_pred)  # Predicted class.

        # Print the confusion matrix as text.
        for i in range(num_classes):
            # Append the class-name to each line.
            class_name = "({}) {}".format(i, class_names[i])
            print(cm[i, :], class_name)

        # Print the class-numbers for easy reference.
        class_numbers = [" ({0})".format(i) for i in range(num_classes)]
        print("".join(class_numbers))

    # Split the data-set in batches of this size to limit RAM usage.
    batch_size = 256

    def predict_cls(transfer_values, labels, cls_true):
        # Number of images.
        num_images = len(transfer_values)

        # Allocate an array for the predicted classes which
        # will be calculated in batches and filled into this array.
        cls_pred = np.zeros(shape=num_images, dtype=np.int)

        # Now calculate the predicted classes for the batches.
        # We will just iterate through all the batches.
        # There might be a more clever and Pythonic way of doing this.

        # The starting index for the next batch is denoted i.
        i = 0

        while i < num_images:
            # The ending index for the next batch is denoted j.
            j = min(i + batch_size, num_images)

            # Create a feed-dict with the images and labels
            # between index i and j.
            feed_dict = {x: transfer_values[i:j],
                         y_true: labels[i:j]}

            # Calculate the predicted class using TensorFlow.
            cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

            # Set the start-index for the next batch to the
            # end-index of the current batch.
            i = j

        # Create a boolean array whether each image is correctly classified.
        correct = (cls_true == cls_pred)

        return correct, cls_pred

    def predict_cls_test():
        return predict_cls(transfer_values = transfer_values_test,
                           labels = labels_test,
                           cls_true = cls_test)

    def classification_accuracy(correct):
        # When averaging a boolean array, False means 0 and True means 1.
        # So we are calculating: number of True / len(correct) which is
        # the same as the classification accuracy.

        # Return the classification accuracy
        # and the number of correct classifications.
        return correct.mean(), correct.sum()

    def print_test_accuracy(show_example_errors=False,
                            show_confusion_matrix=False):

        # For all the images in the test-set,
        # calculate the predicted classes and whether they are correct.
        correct, cls_pred = predict_cls_test()

        # Classification accuracy and the number of correct classifications.
        acc, num_correct = classification_accuracy(correct)

        # Number of images being classified.
        num_images = len(correct)

        # Print the accuracy.
        msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
        print(msg.format(acc, num_correct, num_images))

        # Plot some examples of mis-classifications, if desired.
        if show_example_errors:
            print("Example errors:")
            plot_example_errors(cls_pred=cls_pred, correct=correct)

        # Plot the confusion matrix, if desired.
        if show_confusion_matrix:
            print("Confusion Matrix:")
            plot_confusion_matrix(cls_pred=cls_pred)
    #############




    print('in train_nn')

    ids = list()
    for s in glob.glob(stage1_features_inception + "*"):
        id = os.path.basename(s)
        id = re.match(r'inception_cifar10_([a-f0-9].*).pkl' , id).group(1)
        ids.append(id)
    ids = pd.DataFrame(ids,  columns=["id"])

    df = pd.read_csv(labels)
    df = pd.merge(df, ids, how='inner', on=['id'])


    x = np.array([np.mean(np.load(stage1_features_inception + "inception_cifar10_" + s + ".pkl"), axis=0) for s in df['id'].tolist()])

    y = df['cancer'].as_matrix()
    trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=42, stratify=y,
                                                                    test_size=0.20)

    # global transfer_values_test
    # global transfer_values_train
    # global labels_test
    # global labels_train
    # global cls_train
    # global cls_test

    transfer_values_test = val_x
    transfer_values_train = trn_x
    cls_test = val_y
    cls_train = trn_y
    labels_test = (np.arange(num_classes) == val_y[:, None])+0
    labels_train = (np.arange(num_classes) == trn_y[:, None])+0

    print("transfer_values_test : " + str(transfer_values_test.shape))
    print("transfer_values_train : " + str(transfer_values_train.shape))
    print("labels_test : " + str(labels_test.shape))
    print("labels_train : " + str(labels_train.shape))
    print("cls_test : " + str(cls_test.shape))
    print("cls_train : " + str(cls_train.shape))

    print('start of transfer learning tutorial')
    model = inception.Inception()
    transfer_len = model.transfer_len

    x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)

    x_pretty = pt.wrap(x)

    with pt.defaults_scope(activation_fn=tf.nn.relu):
        y_pred, loss = x_pretty.\
            fully_connected(size=1024, name='layer_fc1').\
            softmax_classifier(num_classes=num_classes, labels=y_true)

    global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step)
    y_pred_cls = tf.argmax(y_pred, dimension=1)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    session = tf.Session()

    session.run(tf.global_variables_initializer())

    print_test_accuracy(show_example_errors=False,
                    show_confusion_matrix=False)
    optimize(num_iterations=10000)

    print('new score:')
    print_test_accuracy(show_example_errors=False,
                    show_confusion_matrix=False)



    # ## Submission
    # ids = list()
    # for s in glob.glob(stage1_features_inception + "*"):
    #     id = os.path.basename(s)
    #     id = re.match(r'inception_cifar10_([a-f0-9].*).pkl' , id).group(1)
    #     ids.append(id)
    # ids = pd.DataFrame(ids,  columns=["id"])

    # submission_sample = pd.read_csv(stage1_submission)
    # df = pd.merge(submission_sample, ids, how='inner', on=['id'])
    # x_test = np.array([np.mean(np.load(stage1_features_inception + "inception_cifar10_" + s + ".pkl"), axis=0) for s in df['id'].tolist()])

    # feed_dict_test = {}
    # for i in range(0, len(x_test)):
    #     feed_dict_test = {x:[x_test[i]]}
    # num_images = len(x_test)
    # cls_pred_test = np.zeros(shape=len(x_test), dtype=np.int)
    # y_labels = np.zeros(shape=len(x_test), dtype=np.int)
    # for i in range(0, len(x_test)):
    #     j = min(i + batch_size, num_images)
    #     feed_dict = {x: x_test[i:j],
    #                      y_true: y_labels[i:j] }
    #     cls_pred_test[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)



    # print(cls_pred_test)




    # pred = clf.predict(x)
    # df['cancer'] = pred

    # #Submission preparation
    # submission = pd.merge(submission_sample, df, how='left', on=['id'])
    # submission = submission.iloc[:,(0,2)]
    # submission = submission.rename(index=str, columns={"cancer_y": "cancer"})

    # # Outputting submission file
    # timestamp = datetime.datetime.now().isoformat()
    # filename = submissions + 'submission-' + str(timestamp) + ".csv"
    # submission.to_csv(filename, index=False)

    # # Submission file analysis
    # print("----submission file analysis----")
    # patient_count = submission['id'].count()
    # predecited = submission['cancer'].count()
    # print("Total number of patients: " + str(patient_count))
    # print("Number of predictions: " + str(predecited))
    # print("submission file stored at: " + filename)


    #### submission



    return

def make_submit():
    # clf = train_xgboost()
    clf = train_nn()

    # ids = list()
    # for s in glob.glob(stage1_features_inception + "*"):
    #     id = os.path.basename(s)
    #     id = re.match(r'inception_cifar10_([a-f0-9].*).pkl' , id).group(1)
    #     ids.append(id)
    # ids = pd.DataFrame(ids,  columns=["id"])

    # submission_sample = pd.read_csv(stage1_submission)
    # df = pd.merge(submission_sample, ids, how='inner', on=['id'])
    # x = np.array([np.mean(np.load(stage1_features_inception + "inception_cifar10_" + s + ".pkl"), axis=0) for s in df['id'].tolist()])

    # pred = clf.predict(x)
    # df['cancer'] = pred

    # #Submission preparation
    # submission = pd.merge(submission_sample, df, how='left', on=['id'])
    # submission = submission.iloc[:,(0,2)]
    # submission = submission.rename(index=str, columns={"cancer_y": "cancer"})

    # # Outputting submission file
    # timestamp = datetime.datetime.now().isoformat()
    # filename = submissions + 'submission-' + str(timestamp) + ".csv"
    # submission.to_csv(filename, index=False)

    # # Submission file analysis
    # print("----submission file analysis----")
    # patient_count = submission['id'].count()
    # predecited = submission['cancer'].count()
    # print("Total number of patients: " + str(patient_count))
    # print("Number of predictions: " + str(predecited))
    # print("submission file stored at: " + filename)

def file_exists(id):
    returnVal = True
    for folder in glob.glob(stage1_processed_pca + 'lungs_pca_*'):
        filename = re.match(r'lungs_pca_([a-f0-9].*).npy', os.path.basename(folder))
        file_id = filename.group(1)
        if(file_id == id):
            returnVal = False
    return returnVal

def PCA_transform(patient_data, components):
    if(components >= patient_data.shape[0]):
        n_components = patient_data.shape[0]
    else:
        n_components = components
    h = int(math.sqrt(patient_data.shape[1]))
    pca = PCA(n_components=n_components, svd_solver='randomized',
              whiten=True).fit(patient_data)
    patient_data_pca = pca.transform(patient_data)
    eigenvectors = pca.components_.reshape((n_components, h, h))
    explained_variance_ratio = pca.explained_variance_ratio_
    return patient_data_pca, eigenvectors, explained_variance_ratio

def process_pca():
    t0 = time()
    index = 1
    pca_n_components = 10000 # want to have n_componets == dim[0]
    for folder in glob.glob(stage1_processed + 'segment_lungs_fill_*'):
        t0 = time()
        filename = re.match(r'segment_lungs_fill_([a-f0-9].*).npy', os.path.basename(folder))
        p_id = filename.group(1)
        if(file_exists(p_id)):
            segment_lungs_fill_ = np.load(stage1_processed + filename.group(0))
            segment_lungs_ = np.load(stage1_processed + "segmented_lungs_" + str(filename.group(1)) + ".npy" )
            lungs = segment_lungs_fill_ -  segment_lungs_
            lungs = lungs.reshape(lungs.shape[0], lungs.shape[1]* lungs.shape[2])
            lungs_pca, eigenvectors, _ = PCA_transform(lungs, pca_n_components)
            np.save(stage1_processed_pca + "lungs_pca_" + p_id, lungs_pca)
            print("id: " + p_id + " -> (" + str(index) + "/1595)" + " done in %0.3fs" % (time() - t0))
        else:
            print("already exists, skipping: " + p_id)
        index += 1
    print("total PCA done in %0.3fs" % (time() - t0))


# Helper function for scans to
def img_to_rgb(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret

# Convert grayscale scans to rgb
# (num_scans, w, h) -> (num_scans, w, h, 3)
def scans_to_rgb(scans):
    num_scans, w, h = scans.shape
    reshaped_scans = np.empty((num_scans, w, h, 3), dtype=np.uint8)
    for scn in enumerate(scans):
        reshaped_scans[scn[0]] = img_to_rgb(scn[1])
    return reshaped_scans


def calc_features_inception():
    inception.maybe_download()
    download.maybe_download_and_extract(cifar10_url, cifar_data)
    model = inception.Inception()
    count = 0

    for folder in glob.glob(stage1_processed + 'scan_segmented_lungs_fill_*'):
        p_id = re.match(r'scan_segmented_lungs_fill_([a-f0-9].*).npy', os.path.basename(folder))
        print('Processing patient ' + str(count) + ' id: ' + p_id.group(1))
        data = np.load(stage1_processed + p_id.group(0))
        # print("original: " + str(data.shape))
        data = scans_to_rgb(data)
        data = normalize_scans(data)
        data = zero_center(data)
        data = normalize_general(data)
        # print("after: " + str(data.shape))



        # Scale images because Inception needs pixels to be between 0 and 255,
        data = data * 255.0
        filepath_cache = cifar_data + "cache/inception_cifar10_" + p_id.group(1) + ".pkl"
        # print(np.min(data))
        # print(np.max(data))
        # print("after scalling: " + str(data.shape))
        transfer_values_train = transfer_values_cache(cache_path=filepath_cache, images=data, model=model)
        count = count + 1




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

    cifar10_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    cifar_data = "/kaggle/dev/data-science-bowl-2017-data/CIFAR-10/"

    ## nn hyper-params
    num_classes = 2
    train_batch_size = 64

    # #globals initializing
    # transfer_values_test = np.empty([2, 2])
    # transfer_values_train = np.empty([2, 2])
    # labels_test = np.empty([2, 2])
    # labels_train = np.empty([2, 2])
    # cls_train = np.empty([2, 2])
    # cls_test = np.empty([2, 2])


    #process_pca()
    #calc_features()
    #calc_features_inception()
    #convnet_3D()
    make_submit()
    print("done")

# Model Building and Traning

# Predicting on CV

# Predicting on Test

# Post-test Analysis

# Submission
