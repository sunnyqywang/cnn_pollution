"""
Utility functions of CNN models

@author: shenhao/qingyi
"""

import copy
import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf

from setup import *

def obtain_mini_batch(X, Y, weights, n_mini_batch):
    '''
    Return mini_batch
    '''
    N = X[0].shape[0]
    index = np.random.choice(N, size = n_mini_batch)
    Y_batch = Y[index, :]
    X_batch = []
    for a in X:
        X_batch.append(a[index, :])
    weights_batch = weights[index, :]
    return X_batch, Y_batch, weights_batch

def truncate_images(images, size):
    '''
    note: size <= 61
    images size: _, 61, 61, 8; (default)
    return: _, size, size, 8;
    '''
    n_images, full_size, full_size, n_channels = images.shape
    truncated_images = np.empty((n_images, size, size, n_channels))
    short_length = np.int((full_size - size)/2)
    long_length = np.int(full_size - short_length)
    for i in range(n_images):
        truncated_images[i, :, :, :] = images[i, short_length:long_length, short_length:long_length, :]
    return truncated_images

def augment_images(images, outputs, weights, scale_info, additional_image_size, epsilon_variance):

    # augment images, outputs, weights, X_mean, X_std...
    new_images = copy.deepcopy(images)
    new_outputs = copy.deepcopy(outputs)
    new_weights = copy.deepcopy(weights)
    new_X_scale = []
    for j in scale_info:
        new_X_scale.append(copy.deepcopy(j))

    n_input,image_height,image_width,n_channel = images.shape
    for i in range(n_input):
        print("\rAdding Images: %i"%i, end="")
        associated_old_img_idx = i%n_input
        old_img = images[associated_old_img_idx, :]
        old_output = outputs[associated_old_img_idx, :]
        old_weight = weights[associated_old_img_idx, :]
        old_X_scale = []
        for j in scale_info:
            old_X_scale.append(j[associated_old_img_idx, :])
        # add noise
        epsilon = np.random.normal(loc=0.0, scale=epsilon_variance, size = (image_height,image_width,n_channel))
        noise_image = old_img + epsilon
        # horizontal flip image
        h_flipped_image = old_img[::-1,:,:]
        # invert images
        v_flipped_image = old_img[:,::-1,:]
        # attach new images
        new_images = np.concatenate([new_images, noise_image[np.newaxis, :], h_flipped_image[np.newaxis, :], v_flipped_image[np.newaxis, :]], axis = 0)
        new_outputs = np.concatenate([new_outputs, old_output[np.newaxis, :], old_output[np.newaxis, :], old_output[np.newaxis, :]], axis = 0) # 3 times for three new observations added
        new_weights = np.concatenate([new_weights, old_weight[np.newaxis, :], old_weight[np.newaxis, :], old_weight[np.newaxis, :]], axis = 0) # 3 times for three new observations added
        for j in range(len(new_X_scale)):
            old = old_X_scale[j]
            new_X_scale[j] = np.concatenate([new_X_scale[j], old[np.newaxis, :], old[np.newaxis, :], old[np.newaxis, :]], axis = 0) # 3 times for three new observations added

    return new_images, new_outputs, new_weights, new_X_scale


def add_convolutional_layer(input_layer,
                            conv_filter_number, conv_kernel_size, conv_stride,
                            pool_size, pool_stride,
                            drop, dropout_rate,
                            bn,
                            pool):
    # drop, bn, and pool are bool values.
    output_layer = tf.layers.conv2d(input_layer, filters = conv_filter_number,
                                    kernel_size = (conv_kernel_size, conv_kernel_size),
                                    strides=(conv_stride,conv_stride),
                                    activation = tf.nn.relu,
                                    padding = 'same')
    if pool:
        # max pool layer
        output_layer = tf.layers.max_pooling2d(output_layer,
                                               pool_size = (pool_size,pool_size),
                                               strides = (pool_stride,pool_stride),
                                               padding = 'same')
    if bn:
        # batch normalization
        output_layer = tf.layers.batch_normalization(inputs=output_layer, axis=3)

    if drop:
        # dropout layer
        output_layer = tf.layers.dropout(inputs=output_layer, rate=dropout_rate)
    return output_layer


def add_fc_layer(input_layer, n_fc, init_mtx=None):
    # fully connected layer
    if init_mtx is None:
        input_layer = tf.layers.dense(input_layer, n_fc, activation=tf.nn.relu)
    else:
        input_layer = tf.layers.dense(input_layer, n_fc, activation=tf.nn.relu, kernel_initializer=init_mtx)
    return input_layer

def get_ensembled_prediction(output_folder, output_var, radius, cnn_top_models, train_size, run_suffix=''):


    output_cnn_train = None
    output_cnn_validation = None
    output_cnn_test = None

    for (hp_idx, linear_coef, model_idx) in cnn_top_models:
        if model_idx == -1:
            with open(output_dir + output_folder + '/results/results_' + ''.join([str(v) for v in output_var]) + '_' + \
                      str(radius) + '/model_output_hyper_searching_dic_' + str(linear_coef) + '_' + str(int(hp_idx)) + \
                      run_suffix + '.pickle', 'rb') as f:
                output_cnn = pkl.load(f)
        else:
            with open(output_dir + output_folder + '/results/results_' + ''.join([str(v) for v in output_var]) + '_' + \
                      str(radius) + '/model_output_hyper_searching_dic_' + str(linear_coef) + '_' + str(int(hp_idx)) + \
                      '_' + str(int(model_idx)) + run_suffix + '.pickle', 'rb') as f:
                output_cnn = pkl.load(f)

        augment = output_cnn[1][-4]

        if output_cnn_train is None:
            if augment:
                output_cnn_train = output_cnn[0]['output_train'][:train_size]
            else:
                output_cnn_train = output_cnn[0]['output_train']
            output_cnn_validation = output_cnn[0]['output_validation']
            output_cnn_test = output_cnn[0]['output_test']
        else:
            if augment:
                output_cnn_train += output_cnn[0]['output_train'][:train_size]
            else:
                output_cnn_train += output_cnn[0]['output_train']
            output_cnn_validation += output_cnn[0]['output_validation']
            output_cnn_test += output_cnn[0]['output_test']

    output_cnn_train /= len(cnn_top_models)
    output_cnn_validation /= len(cnn_top_models)
    output_cnn_test /= len(cnn_top_models)

    return output_cnn_train, output_cnn_validation, output_cnn_test
