"""
Run CNN models

@author: shenhao/qingyi
"""

import itertools
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import itertools
import os
from util_cnn import *
from setup import *
# setup_cnn: cnn hyperparameters
from setup_cnn import *

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
#tf.debugging.set_log_device_placement(True)

char_name = '_no_airport_no_gas_coal_combined_oil_combined'
variables_export = ['COAL', 'INDCT', 'INDOT', 'SVC', 'OIL', 'DEM', 'rain', 'TEM']
output_folder = '210908'
standard = 'const'
#output_vars = ['pm25', 'pm10', 'so2', 'no2', 'co', 'o3', 'aqi']
output_vars = [0,1,2,3,4,5]
image_radius = 30 # max 30
linear_coefs = [0,0.25,0.5,0.75,1]
import_hyperparameters = 121
multitask_learning = True

### 0. read data
with open(data_dir+"process/data_process_dic"+char_name+"_"+standard+".pickle", 'rb') as data_standard:
    data_full_package = pickle.load(data_standard)

def train_cnn(idx, train_images, train_y, train_weights, validation_images, validation_y, validation_weights, test_images, test_y, test_weights,
              scale_info, hyperparams, output_var, linear_coef, import_hyperparameters):

    (n_iterations, n_mini_batch, conv_layer_number, pool_list, conv_filter_number, conv_kernel_size, \
     conv_stride, pool_size, pool_stride, drop, dropout_rate, bn, fc_layer_number, n_fc, \
     augment, additional_image_size, epsilon_variance, standard) = hyperparams

    # augment images
    if standard == 'minmax':
        (train_mean_images, validation_mean_images, test_mean_images, train_max_images, validation_max_images, test_max_images) = scale_info
        # augment data
        if augment:
            train_images, train_y, train_weights, [train_mean_images, train_max_images] = \
                augment_images(train_images, train_y, train_weights, [train_mean_images, train_max_images], additional_image_size,
                               epsilon_variance)
    elif standard == 'norm':
        (train_mean_images, validation_mean_images, test_mean_images, train_std_images, validation_std_images, test_std_images) = scale_info
        # augment data
        if augment:
            train_images, train_y, train_weights, [train_mean_images, train_std_images] = \
                augment_images(train_images, train_y, train_weights, [train_mean_images, train_std_images],
                               additional_image_size, epsilon_variance)
    elif standard == 'const':
        (train_mean_images, validation_mean_images, test_mean_images) = scale_info
        # augment data
        if augment:
            train_images, train_y, train_weights, [train_mean_images] = \
                augment_images(train_images, train_y, train_weights, [train_mean_images], additional_image_size, epsilon_variance)

    ## build model here
    tf.reset_default_graph()
    _,image_height,image_width,n_channel = train_images.shape
    _,n_air_variables = train_y.shape
    training_loss_list = []
    validation_loss_list = []
    testing_loss_list = []

    with tf.name_scope("inputs"):
        X = tf.placeholder(tf.float32, shape=[None,image_height,image_width,n_channel], name = "X")
        y = tf.placeholder(tf.float32, shape=[None, n_air_variables], name = "y")
        X_mean = tf.placeholder(tf.float32, shape=[None, n_channel], name="X_mean")

        weights = tf.placeholder(tf.float32, shape=[None, 1], name = "weights")
        if standard == 'minmax':
            X_max = tf.placeholder(tf.float32, shape=[None,n_channel], name = "X_max")
        elif standard == 'norm':
            # change 191122 remove standard deviation; change 200315 reinstate standard deviation
            X_std = tf.placeholder(tf.float32, shape=[None,n_channel], name = "X_std")

    with tf.name_scope("hidden_layers"):
        # conv1
        input_layer = X
        for i in range(conv_layer_number):
            input_layer = add_convolutional_layer(input_layer,
                                                  conv_filter_number, conv_kernel_size, conv_stride,
                                                  pool_size, pool_stride,
                                                  drop, dropout_rate,
                                                  bn,
                                                  pool_list[i])

        # flatten
        input_layer = tf.contrib.layers.flatten(input_layer)
        # fully connected layers
        for i in range(fc_layer_number):
            input_layer = add_fc_layer(input_layer, n_fc)
        # if standard == 'minmax':
        #     input_layer=tf.concat([input_layer,X_max], axis = 1) # augment max values
        # elif standard == 'norm':
        #     input_layer=tf.concat([input_layer,X_mean], axis = 1) # augment mean values
        #     input_layer=tf.concat([input_layer,X_std], axis = 1) # augment std values

        input_layer = add_fc_layer(input_layer, n_fc) # change 191122: add nonlinearity
        input_layer = add_fc_layer(input_layer, n_fc)
        input_layer = add_fc_layer(input_layer, n_fc)

    with tf.name_scope("linear_part"):
        linear_part = add_fc_layer(X_mean, n_air_variables)

    with tf.name_scope("outputs"):
        # reduce to vector prediction
        # dim of output: N*n_air_variables
        bias = tf.Variable(tf.random_normal([n_air_variables]), name='bias')
        output = tf.layers.dense(input_layer, n_air_variables) + linear_part*linear_coef + bias
        output = tf.identity(output, name='output')

    with tf.name_scope("train"):
        # use L2 distance between two matrices as loss
        loss = tf.losses.mean_squared_error(y, output, weights = weights)

    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    ref1 = 0
    ref2 = 0

    # evaluation values
    if standard == 'minmax':
        feed_train = {X: train_images, y: train_y, X_mean: train_mean_images, X_max: train_max_images, weights: train_weights}
        feed_validation = {X: validation_images, y: validation_y, X_mean: validation_mean_images, X_max: validation_max_images, weights: validation_weights}
        feed_test = {X: test_images, y: test_y, X_mean: test_mean_images, X_max: test_max_images, weights: test_weights}
    elif standard == 'norm':
        feed_train = {X: train_images, y: train_y, X_mean: train_mean_images, X_std: train_std_images, weights: train_weights}
        feed_validation = {X: validation_images, y: validation_y, X_mean: validation_mean_images,
                           X_std: validation_std_images, weights: validation_weights}
        feed_test = {X: test_images, y: test_y, X_mean: test_mean_images, X_std: test_std_images, weights: test_weights}
    elif standard == 'const':
        feed_train = {X: train_images, y: train_y, X_mean: train_mean_images, weights: train_weights}
        feed_validation = {X: validation_images, y: validation_y, X_mean: validation_mean_images, weights: validation_weights}
        feed_test = {X: test_images, y: test_y, X_mean: test_mean_images, weights: test_weights}

    with tf.Session(config=config) as sess:
        init.run()
        for iteration in range(n_iterations):
            print("\rEpoch: %d, Iteration: %d" % ((iteration//n_mini_batch,iteration)), end = "")

            if standard == 'minmax':
                [X_batch, X_mean_batch, X_max_batch], y_batch, weights_batch = \
                    obtain_mini_batch([train_images, train_mean_images, train_max_images], train_y, train_weights, n_mini_batch)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch, X_mean: X_mean_batch, weights: weights_batch, X_max: X_max_batch})
            elif standard == 'norm':
                [X_batch, X_mean_batch, X_std_batch], y_batch, weights_batch = \
                    obtain_mini_batch([train_images, train_mean_images, train_std_images], train_y, train_weights, n_mini_batch)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch, weights: weights_batch, X_mean: X_mean_batch, X_std: X_std_batch})
            elif standard == 'const':
                [X_batch, X_mean_batch], y_batch, weights_batch = \
                    obtain_mini_batch([train_images, train_mean_images], train_y, train_weights, n_mini_batch)
                sess.run(training_op,
                         feed_dict={X: X_batch, y: y_batch, X_mean: X_mean_batch, weights: weights_batch})

            if iteration % (len(train_images)//n_mini_batch) == 0:
                loss_ = loss.eval(feed_dict = feed_train)
                training_loss_list.append(loss_)
                validation_loss_list.append(loss.eval(feed_dict = feed_validation))
                testing_loss_list.append(loss.eval(feed_dict = feed_test))

                if iteration > 100:
                    if (np.abs(loss_ - ref1) / ref1 < 0.005) & (np.abs(loss_ - ref2) / ref2 < 0.005):
                        print("Early stopping at iteration", iteration)
                        break
                    if (ref1 < loss_) & (ref1 < ref2):
                        print("Diverging. stop.")
                        break
                    if loss_ < best:
                        best = loss_
                        best_epoch = iteration
                else:
                    best = loss_
                    best_epoch = iteration

                ref2 = ref1
                ref1 = loss_

                ## save models
                if best_epoch == iteration:
                    saver.save(sess, output_dir+output_folder+"/models/models_"+str(output_var)+"_"+str(image_radius)+
                               "/model_"+str(linear_coef) +"_"+str(import_hyperparameters) + "_" +str(idx)+"_"+str(iteration)+".ckpt")
                    files = glob.glob(output_dir+output_folder+"/models/models_"+str(output_var)+"_"+str(image_radius)+
                                      "/model_"+str(linear_coef)+"_"+str(import_hyperparameters)+"_"+str(idx)+"_*.ckpt*")

        for f in files:
            e = int(f.split("_")[-1].split(".")[0])
            if e != best_epoch:
                os.remove(f)

        # training
        output_train = output.eval(feed_dict=feed_train)
        # validation
        output_validation = output.eval(feed_dict=feed_validation)
        # testing
        output_test = output.eval(feed_dict=feed_test)
        # store results
        model_output = {}
        model_output['output_train'] = output_train
        model_output['output_validation'] = output_validation
        model_output['output_test'] = output_test

        model_output['training_loss_list'] = training_loss_list
        model_output['validation_loss_list'] = validation_loss_list
        model_output['testing_loss_list'] = testing_loss_list


    return model_output


for output_var, linear_coef in itertools.product(output_vars, linear_coefs):
    lb = 30-image_radius
    ub = 30+image_radius+1
    # use standardized energy data and non-standardized air pollution data
    cnn_data_name = 'energy_'+standard+'_air'
    input_cnn_training = data_full_package[cnn_data_name]['input_training'][:, lb:ub, lb:ub, :]
    input_cnn_validation = data_full_package[cnn_data_name]['input_validation'][:, lb:ub, lb:ub, :]
    input_cnn_testing = data_full_package[cnn_data_name]['input_testing'][:, lb:ub, lb:ub, :]
    output_cnn_training = data_full_package[cnn_data_name]['output_training'][:, output_var][:, np.newaxis]
    output_cnn_validation = data_full_package[cnn_data_name]['output_validation'][:, output_var][:, np.newaxis]
    output_cnn_testing = data_full_package[cnn_data_name]['output_testing'][:, output_var][:, np.newaxis]
    output_cnn_all_vars_training = data_full_package[cnn_data_name]['output_training']
    output_cnn_all_vars_validation = data_full_package[cnn_data_name]['output_validation']
    output_cnn_all_vars_testing = data_full_package[cnn_data_name]['output_testing']
    index_cnn_training = data_full_package[cnn_data_name]['index_training']
    index_cnn_validation = data_full_package[cnn_data_name]['index_validation']
    index_cnn_testing = data_full_package[cnn_data_name]['index_testing']
    weights_cnn_training = data_full_package[cnn_data_name]['weight_training'].T/100
    weights_cnn_validation = data_full_package[cnn_data_name]['weight_validation'].T/100
    weights_cnn_testing = data_full_package[cnn_data_name]['weight_testing'].T/100

    mean_data_name = 'energy_mean_air'
    input_mean_training = data_full_package[mean_data_name]['input_training']
    input_mean_validation = data_full_package[mean_data_name]['input_validation']
    input_mean_testing = data_full_package[mean_data_name]['input_testing']
    train_mean_images = input_mean_training / 10000
    validation_mean_images = input_mean_validation / 10000
    test_mean_images = input_mean_testing / 10000

    if standard == 'norm':
        std_data_name = 'energy_std_air'
        input_std_training = data_full_package[std_data_name]['input_training']
        input_std_validation = data_full_package[std_data_name]['input_validation']
        input_std_testing = data_full_package[std_data_name]['input_testing']
        train_std_images = input_std_training / 10000
        validation_std_images = input_std_validation / 10000
        test_std_images = input_std_testing / 10000
    elif standard == 'minmax':
        min_data_name = 'energy_min_air'
        input_min_training = data_full_package[min_data_name]['input_training']
        input_min_validation = data_full_package[min_data_name]['input_validation']
        input_min_testing = data_full_package[min_data_name]['input_testing']
        max_data_name = 'energy_max_air'
        input_max_training = data_full_package[max_data_name]['input_training']
        input_max_validation = data_full_package[max_data_name]['input_validation']
        input_max_testing = data_full_package[max_data_name]['input_testing']
        train_max_images = input_max_training / 10000
        validation_max_images = input_max_validation  / 10000
        test_max_images = input_max_testing / 10000

    ### 1. Search hyperparameters for CNN
    model_output_hyper_searching = {}

    # read data
    train_images=input_cnn_training
    train_y=output_cnn_training
    train_weights = weights_cnn_training
    validation_images= input_cnn_validation
    validation_y =output_cnn_validation
    validation_weights = weights_cnn_validation
    test_images =input_cnn_testing
    test_y =output_cnn_testing
    test_weights = weights_cnn_testing

    hp = list(itertools.product(n_iterations_list,n_mini_batch_list,conv_layer_number_list,conv_filter_number_list,conv_kernel_size_list,\
                      conv_stride_list, pool_size_list, pool_stride_list, drop_list, dropout_rate_list, bn_list, \
                      fc_layer_number_list, n_fc_list, augment_list, additional_image_size_list, epsilon_variance_list))

    if import_hyperparameters is None:
        n_hyperparam_searching = -1 # -1 means all; >0 means randomly sampled combos
        if n_hyperparam_searching == -1:
            n_hyperparam_searching = len(hp)
        else:
            idx = np.random.choice(np.arange(len(hp)), n_hyperparam_searching)
            hp = [hp[a] for a in idx]
    else:
        # import linear coef = 0
        with open(output_dir+output_folder+'/results/results_'+str(output_var)+'_'+str(image_radius)+
                  '/model_output_hyper_searching_dic_0_' + str(import_hyperparameters)+'.pickle', 'rb') as f:
            hp = pkl.load(f)[1]
        # the number of times this set of hyperparameters will be repeated.
        n_hyperparam_searching = 20

    for idx in range(n_hyperparam_searching):
        print("The current index is: ", idx, " out of total ", n_hyperparam_searching)

        if hp is not None:
            if import_hyperparameters is None:
                hp_cur = hp[idx]
            else:
                hp_cur = hp
            (n_iterations, n_mini_batch, conv_layer_number, pool_list, conv_filter_number, conv_kernel_size, \
             conv_stride, pool_size, pool_stride, drop, dropout_rate, bn, fc_layer_number, n_fc, \
             augment, additional_image_size, epsilon_variance, standard) = hp_cur
        else:
            n_iterations = np.random.choice(n_iterations_list)
            n_mini_batch = np.random.choice(n_mini_batch_list)
            #
            conv_layer_number = np.random.choice(conv_layer_number_list)
            conv_filter_number = np.random.choice(conv_filter_number_list)
            conv_kernel_size = np.random.choice(conv_kernel_size_list)
            conv_stride = np.random.choice(conv_stride_list)
            #
            pool_size=np.random.choice(pool_size_list)
            pool_stride=np.random.choice(pool_stride_list)
            drop=np.random.choice(drop_list)
            dropout_rate=np.random.choice(dropout_rate_list)
            bn=np.random.choice(bn_list)
            fc_layer_number=np.random.choice(fc_layer_number_list)
            n_fc=np.random.choice(n_fc_list)
            #
            augment=np.random.choice(augment_list)
            additional_image_size=np.random.choice(additional_image_size_list)
            epsilon_variance=np.random.choice(epsilon_variance_list)
            #

        #
        pool_list=np.random.choice([True], size=conv_layer_number)

        # hyper
        hyperparams=(n_iterations, n_mini_batch, conv_layer_number, pool_list, conv_filter_number, conv_kernel_size, \
                     conv_stride, pool_size, pool_stride, drop, dropout_rate, bn, fc_layer_number, n_fc, \
                     augment, additional_image_size, epsilon_variance, standard)

        hyperparams_names=('n_iterations', 'n_mini_batch', 'conv_layer_number', 'pool_list', 'conv_filter_number', 'conv_kernel_size', \
                           'conv_stride', 'pool_size', 'pool_stride', 'drop', 'dropout_rate', 'bn', 'fc_layer_number', 'n_fc', \
                           'augment', 'additional_image_size', 'epsilon_variance', 'standard_method')

        # Train CNN
        if standard == 'minmax':
            scale_info = (train_mean_images, validation_mean_images, test_mean_images, train_max_images, validation_max_images, test_max_images)
        elif standard == 'norm':
            scale_info = (train_mean_images, validation_mean_images, test_mean_images, train_std_images, validation_std_images, test_std_images)
        elif standard == 'const':
            scale_info = (train_mean_images, validation_mean_images, test_mean_images)

        if not os.path.exists(output_dir+output_folder+"/models/models_"+str(output_var)+"_"+str(image_radius)+"/"):
            os.mkdir(output_dir+output_folder+"/models/models_"+str(output_var)+"_"+str(image_radius))

        model_output = train_cnn(idx, train_images, train_y, train_weights, validation_images, validation_y, validation_weights,
                                 test_images, test_y, test_weights, scale_info, hyperparams, output_var, linear_coef, import_hyperparameters)
        model_output_hyper_searching[str(idx)] = (model_output,hyperparams,hyperparams_names,linear_coef)

        if not os.path.exists(output_dir+output_folder+"/results/results_"+str(output_var)+"_"+str(image_radius)+"/"):
            os.mkdir(output_dir+output_folder+"/results/results_"+str(output_var)+"_"+str(image_radius))
        if import_hyperparameters is None:
            with open(output_dir+output_folder+'/results/results_'+str(output_var)+'_'+str(image_radius)+
                      "/model_output_hyper_searching_dic_"+str(linear_coef)+"_"+str(idx)+'.pickle', 'wb') as model_output_hyper_searching_dic:
                pickle.dump(model_output_hyper_searching[str(idx)], model_output_hyper_searching_dic, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(output_dir+output_folder+'/results/results_'+str(output_var)+'_'+str(image_radius)+
                      '/model_output_hyper_searching_dic_'+str(linear_coef)+"_"+str(import_hyperparameters)+"_"+str(idx)+'.pickle', 'wb') as model_output_hyper_searching_dic:
                pickle.dump(model_output_hyper_searching[str(idx)], model_output_hyper_searching_dic, protocol=pickle.HIGHEST_PROTOCOL)

