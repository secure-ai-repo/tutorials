"""
For the purpose of upgrading the development of Tensorflow AI Vision Project
Edited by Dr. Hyeung-yun Kim


Train the model on dataset
"""
import tensorflow as tf
from SSD_settings import *
from SSD_model import SSDModel
from SSD_model import ModelHelper
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import math
import os
import time
import pickle
from PIL import Image


def next_batch(X, y_conf, y_loc, batch_size):

    """
    Next batch generator
    Arguments:
         X: List of image file names
         y_conf: List of ground-truth vectors for class labels
         y_loc: List of ground-truth vectors for localization
         batch_size: Batch size
    Yields:
        images: Batch numpy array representation of batch of images
        y_true_conf: Batch numpy array for ground-truth class labels
        y_true_loc: Batch numpy array for ground-truth localization
        conf_loss_mask: Loss mask for confidence loss, to set NEG_POS_RATIO
    """
    start_idx = 0
    while True:
        image_files = X[start_idx:start_idx+batch_size]
        y_true_conf = np.array(y_conf[start_idx:start_idx+batch_size])
        y_true_loc = np.array(y_loc[start_idx:start_idx+batch_size])

        images = []
        file_dir = "C:\\Users\\mpole\\Dataset\\Xray\\img_sample_w400_h260\\"
        for image_file in image_files:
            image = Image.open(file_dir+'resized_images_%sx%s/%s' % (IMG_W, IMG_H, image_file))
            image = np.array(image)
            images.append(image)
        images = np.array(images, dtype='float32')

        if NUM_CHANNELS == 1:
            images = np.expand_dims(images, axis=-1)
            # images = np.squeeze(images, axis=4)
        # Gray scale images have shape (H, W), but we want shape (W, H, 1)
        # Normalize pixel values (scale them btn -1(0) and 1(255))
        images = images/127.5 - 1.

        # For y_true_conf, calculate how many negative examples we need to satisfy NEG_POS_RATIO
        num_pos = np.where(y_true_conf > 0)[0].shape[0]
        num_neg = NEG_POS_RATIO * num_pos
        y_true_conf_size = np.sum(y_true_conf.shape)

        # CreATE Confidence loss mask to satisfy NEG_POS_RATIO
        if num_pos + num_neg < y_true_conf_size:
            conf_loss_mask = np.copy(y_true_conf)
            conf_loss_mask[np.where(conf_loss_mask > 0)] = 1.

            # Find all (i,j) tuples where y_true_conf[i][j]===0
            zero_indices = np.where(conf_loss_mask==0.) # ([i1. i2. ...], [j1, j2,]]]
            zero_indices = np.transpose(zero_indices)   # [[i1,j1], [i2,j2], ...]

            # Randomly choose num_neg rows from zero_indices, w/o replacement
            chosen_zero_indices = zero_indices[np.random.choice(zero_indices.shape[0], int(num_neg), False)]

            # "Enable" chosen negative examples, specified by chosen_zero_indices
            for zero_index in chosen_zero_indices:
                i, j = zero_index
                conf_loss_mask[i][j] = 1.
        else:
            # we have so many positive examples such that num_pos+num_neg >= y_true_conf_size
            # no need to prune negative data
            conf_loss_mask = np.ones_like(y_true_conf)

        yield (images, y_true_conf, y_true_loc, conf_loss_mask)

        # Update start index for the next batch
        start_idx += start_idx
        if start_idx >= X.shape[0]:
            start_idx = 0


def calculate_accuracy(data_gen, data_size, batch_size, accuracy, x, y, keep_prob, sess):
    """	Helper function to calculate accuracy on a particular dataset
        Arguments:
        * data_gen: Generator to generate batches of data
        * data_size: Total size of the data set, must be consistent with generator
        * batch_size: Batch size, must be consistent with generator
        * accuracy, x, y, keep_prob: Tensor objects in the neural network
        * sess: TensorFlow session object containing the neural network graph
    Returns:
        * Float representing accuracy on the data set
    """
    num_batches = math.ceil(data_size / batch_size)
    last_batch_size = data_size % batch_size

    accs = []  # accuracy for each batch
    for _ in range(num_batches):
        images, labels = next(data_gen)

        # Perform forward pass and calculate accuracy
        # Note we set keep_prob to 1.0, since we are performing inference
        acc = sess.run(accuracy, feed_dict={x: images, y: labels, keep_prob: 1.})

    accs.append(acc)

    # Calculate average accuracy of all full batches (the last batch is the only partial batch)
    acc_full = np.mean(accs[:-1])

    # Calculate weighted average of accuracy accross batches
    acc = (acc_full * (data_size - last_batch_size) + accs[-1] * last_batch_size) / data_size

    return acc


def run_training():
    """
    1. Load training and test data
    2. Training process
    3. Plot train/validation losses
    4. Report test loss
    5. Save model
    """
    # Load training and test data
    file_dir = "C:\\Users\\mpole\\Dataset\\Xray\\"
    with open(file_dir + 'xray_data_prep_%sx%s.p' % (IMG_W, IMG_H), mode='rb') as f:
        train = pickle.load(f)
    # Format data
    x_train = []
    y_train_conf = []
    y_train_loc = []
    for image_file in train.keys():
        x_train.append(image_file)
        y_train_conf.append(train[image_file]['y_true_conf'])
        y_train_loc.append(train[image_file]['y_true_loc'])

    x_train = np.array(x_train)
    y_train_conf = np.array(y_train_conf)
    y_train_loc = np.array(y_train_loc)

    # Load train and validation data split
    x_train, x_valid, y_train_conf, y_valid_conf, y_train_loc, y_valid_loc = train_test_split(
        x_train, y_train_conf, y_train_loc, test_size=VALIDATION_SIZE, random_state=1)

    # Launch the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        # Instantiate neural network, get relevant tensors
        model = SSDModel()
        x = model['x']
        y_true_conf = model['y_true_conf']
        y_true_loc = model['y_true_loc']
        conf_loss_mask = model['conf_loss_mask']
        is_training = model['is_training']
        optimizer = model['optimizer']
        reported_loss = model['loss']

        # accuracy = tf.reduce_mean(tf.cast(model['conf_loss'], tf.float32))

        # Training process
        # TF saver to save/restore trained model
        saver = tf.train.Saver()

        if RESUME:
            saver.restore(sess, MODEL_SAVE_PATH)
            with open('loss_history.p', 'rp') as f:
                loss_history = pickle.load(f)
        else:
            sess.run(tf.global_variables_initializer())
            loss_history = []

        # Run NUM_EPOCH epochs od training
        for epoch in range(NUM_EPOCH):
            train_gen = next_batch(x_train, y_train_conf, y_train_loc, BATCH_SIZE)
            num_batches_train = math.ceil(x_train.shape[0]/BATCH_SIZE)

            losses = [] # list of loss values for book-keeping
            # Run training on each batch
            for _ in range(num_batches_train):
                # Obtain the training data and labels from generator
                images, y_true_conf_gen, y_train_loc_gen, conf_loss_mask_gen = next(train_gen)
                # Perform gradient update (i.e. training step) on current batch
                #_, loss, local_loss_dbg, loc_loss_mask, local_loss = sess.run([optimizer, reported_loss, model['local_loss_dbg]], feed_dict={
                _, loss = sess.run([optimizer, reported_loss], feed_dict={
                    x: images,
                    y_true_conf: y_true_conf_gen,
                    y_true_loc: y_train_loc_gen,
                    conf_loss_mask: conf_loss_mask_gen,
                    is_training: True
                })
                losses.append(loss)

            train_loss = np.mean(losses)

            # calculate validation loss at the end of epoch
            valid_gen = next_batch(x_valid, y_valid_conf, y_valid_loc, BATCH_SIZE)
            losses = []
            num_batches_valid = math.ceil(x_valid.shape[0] / BATCH_SIZE)
            for _ in range(num_batches_valid):
                images, y_true_conf_gen, y_train_loc_gen, conf_loss_mask_gen = next(valid_gen)
                loss = sess.run(reported_loss, feed_dict={
                    x: images,
                    y_true_conf: y_true_conf_gen,
                    y_true_loc: y_train_loc_gen,
                    conf_loss_mask: conf_loss_mask_gen,
                    is_training: False
                })
                losses.append(loss)
            valid_loss = np.mean(losses)

            # Record and report train/validation/test losses for this epoch
            loss_history.append((train_loss, valid_loss))

        test_loss = 0.

        # After training is complete, evaluate accuracy on test set
        """
        testing_file = 'test.p'
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)
        X_test, y_test = test['features'], test['labels']
        test_gen = next_batch(X_test, y_test, BATCH_SIZE)
        test_size = X_test.shape[0]
        test_acc = calculate_accuracy(test_gen, test_size, BATCH_SIZE, accuracy, x, y, keep_prob, sess)
        """

        if SAVE_MODEL:
            save_path = saver.save(sess, MODEL_SAVE_PATH)
            print('Trained model saved at: %s' % save_path)

            with open('loss_history.p', 'wb') as f:
                pickle.dump(loss_history, f)

    return test_loss, loss_history


def run_visualiztion():
    FM_ONLY = False
    with tf.Graph().as_default(), tf.Session() as sess:
        if FM_ONLY:
            if MODEL == 'AlexNet':
                from SSD_model import AlexNet as MyModel
            else :
                raise NotImplementedError('Model %s not supported' % MODEL)
            _ = MyModel()

        tf.summary.merge_all()
        writer = tf.summary.FileWriter('./tensorboard_out', sess.graph)
        tf.global_variables_initializer().run()


if __name__ == '__main__':
    run_training()
    run_visualiztion()
