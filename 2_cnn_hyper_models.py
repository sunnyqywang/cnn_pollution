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

import util_data
from util_cnn import *
from setup import *
# setup_cnn: cnn hyperparameters
from setup_cnn import *

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
#tf.debugging.set_log_device_placement(True)

char_name = '_no_airport_no_gas_coal_combined_oil_combined'
# variables_export = ['COAL', 'INDCT', 'INDOT', 'SVC', 'OIL', 'DEM', 'rain', 'TEM']
output_folder = '211010'
standard = 'const'
#output_vars = ['pm25', 'pm10', 'so2', 'no2', 'co', 'o3', 'aqi']
output_vars = [[0,1,2,3,4,5]]
image_radius = 30 # max 30
linear_coefs = [0.2]#[0,0.1,0.2,0.3,0.5,0.7,0.9,0.95]
# linear_coefs = [0]
# if import_hyperparameters is not None, then import the hyperparameter # in the saved models (with the same radius and output variable(s))
import_hyperparameters = None
run_suffix = "_linearfix"

# if import_hyperparamters is None, then do a search of hyperparameters
# -1 means all; >0 means randomly sampled combos
# if import_hyperparameter is not None, this parameter represents the number of times that this set of hyperparameters will be repeated.
n_hyperparam_searching = -1
start_parameter = 0

### 0. read data
with open(data_dir+"process/data_process_dic"+char_name+"_"+standard+".pickle", 'rb') as data_standard:
    data_full_package = pickle.load(data_standard)

def train_cnn(idx, train_images, train_y, train_weights, validation_images, validation_y, validation_weights, test_images, test_y, test_weights,
              scale_info, control_var_training, control_var_validation, control_var_testing, hyperparams, output_var, linear_coef, import_hyperparameters):

    (n_iterations, n_mini_batch, conv_layer_number, pool_list, conv_filter_number, conv_kernel_size, \
     conv_stride, pool_size, pool_stride, drop, dropout_rate, bn, fc_layer_number, n_fc, \
     augment, additional_image_size, epsilon_variance, standard) = hyperparams

    # augment images
    if standard == 'minmax':
        (train_mean_images, validation_mean_images, test_mean_images, train_max_images, validation_max_images, test_max_images) = scale_info
        # augment data
        if augment:
            train_images, train_y, train_weights, [train_mean_images, train_max_images, control_var_training] = \
                augment_images(train_images, train_y, train_weights, [train_mean_images, train_max_images, control_var_training], additional_image_size,
                               epsilon_variance)
    elif standard == 'norm':
        (train_mean_images, validation_mean_images, test_mean_images, train_std_images, validation_std_images, test_std_images) = scale_info
        # augment data
        if augment:
            train_images, train_y, train_weights, [train_mean_images, train_std_images, control_var_training] = \
                augment_images(train_images, train_y, train_weights, [train_mean_images, train_std_images, control_var_training],
                               additional_image_size, epsilon_variance)
    elif standard == 'const':
        (train_mean_images, validation_mean_images, test_mean_images) = scale_info
        # augment data
        if augment:
            train_images, train_y, train_weights, [train_mean_images, control_var_training] = \
                augment_images(train_images, train_y, train_weights, [train_mean_images, control_var_training], additional_image_size, epsilon_variance)

    ## build model here
    tf.reset_default_graph()
    _,image_height,image_width,n_channel = train_images.shape
    _,n_air_variables = train_y.shape
    _,n_control_var = control_var_training.shape
    training_loss_list = []
    validation_loss_list = []
    testing_loss_list = []

    with tf.name_scope("inputs"):
        X = tf.placeholder(tf.float32, shape=[None,image_height,image_width,n_channel], name = "X")
        y = tf.placeholder(tf.float32, shape=[None, n_air_variables], name = "y")
        X_mean = tf.placeholder(tf.float32, shape=[None, n_channel+n_control_var], name="X_mean")

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

    with tf.name_scope("outputs"):
        # reduce to vector prediction
        # dim of output: N*n_air_variables
        bias = tf.Variable(tf.random_normal([n_air_variables]), name='bias')
        # init = tf.constant_initializer([0.00258922454, -1.25796299e-05, -0.000169064926, -0.00040304034, 0.00070013143,\
        #     -0.000477985298, -0.000117271471, -0.0982265421, 23.9178895, -18.9303794, 9.42604944, 0.45162213, 5.10710306])
        # linear_part = add_fc_layer(X_mean, n_air_variables, init_mtx=init)

        linear_fix_dic = {0:[2.52973782e-03, -1.35041764e-05, -1.28294696e-04, -3.86105665e-04, 7.18041745e-04,
                -1.11069719e-03, -2.44344984e-04, -9.15259169e-02, 1.50037549e+01, 8.78439986e+00, 4.69878389e-01],
                          1:[4.00649877e+01,  2.70000045e-01, -1.13559922e+00,
                             -1.39532146e+01,  5.74306777e+00,  4.64453235e+01, -1.39366689e+01,
                             -1.07186417e+03,  3.04264490e+01,  4.43978906e+00,  9.00812453e-01],
                          2:[8.19002462e+00,  7.09963810e-01,  3.37301044e+00,
                              8.71421165e+00, -6.26245408e+00,  2.53395976e+01,  7.29376113e+00,
                             -7.90922979e+02, -2.11138615e+00,  4.20828551e+00,  3.46035635e-01],
                          3:[8.2578667, -0.04405988,   1.36660061,  -1.9467246,
                               3.92231257,   4.65380866,  -6.18859082, -15.98711687,  -0.30667645,
                              -0.52432386,   0.03683353],
                          4:[0.53037012,  0.01096355,  0.07183032, -0.42391902, -0.14307672,
                              0.66418298,  0.1271171,  -1.39470095, -0.10631515, -0.02114975, -0.01165508],
                          5:[-1.21018167e+01,  1.51479958e-01,  4.67799162e+00,
                              1.13204531e+01, -9.64383562e-02,  2.78793501e+01, -1.30869541e+01,
                              8.42973625e+02,  1.94361059e+01,  1.32027164e+01, -4.70762088e-01]}
        linear_fix = tf.constant(np.array([linear_fix_dic[v] for v in output_var]).T, dtype=tf.float32)#, shape=(n_channel+n_control_var,len(output_var)))
        linear_part = tf.linalg.matmul(X_mean, linear_fix)
        linear_part = tf.identity(linear_part, name='linear_part')
        cnn_part = tf.layers.dense(input_layer, n_air_variables)
        cnn_part = tf.identity(cnn_part, name='cnn_part')
        output = cnn_part*(1-linear_coef) + linear_part*linear_coef + bias
        output = tf.identity(output, name='final_output')

    with tf.name_scope("train"):
        # use L2 distance between two matrices as loss
        loss = tf.losses.mean_squared_error(y, output, weights=weights)

    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    ref1 = 0
    ref2 = 0

    # evaluation values
    if standard == 'minmax':
        feed_train = {X: train_images, y: train_y, X_mean: np.hstack([train_mean_images,control_var_training]), X_max: train_max_images, weights: train_weights}
        feed_validation = {X: validation_images, y: validation_y, X_mean: np.hstack([validation_mean_images,control_var_validation]), X_max: validation_max_images, weights: validation_weights}
        feed_test = {X: test_images, y: test_y, X_mean: np.hstack([test_mean_images,control_var_testing]), X_max: test_max_images, weights: test_weights}
    elif standard == 'norm':
        feed_train = {X: train_images, y: train_y, X_mean: np.hstack([train_mean_images,control_var_training]), X_std: train_std_images, weights: train_weights}
        feed_validation = {X: validation_images, y: validation_y, X_mean: np.hstack([validation_mean_images,control_var_validation]),
                           X_std: validation_std_images, weights: validation_weights}
        feed_test = {X: test_images, y: test_y, X_mean: np.hstack([test_mean_images,control_var_testing]), X_std: test_std_images, weights: test_weights}
    elif standard == 'const':
        feed_train = {X: train_images, y: train_y, X_mean: np.hstack([train_mean_images, control_var_training]), weights: train_weights}
        feed_validation = {X: validation_images, y: validation_y, X_mean: np.hstack([validation_mean_images,control_var_validation]), weights: validation_weights}
        feed_test = {X: test_images, y: test_y, X_mean: np.hstack([test_mean_images,control_var_testing]), weights: test_weights}

    with tf.Session(config=config) as sess:
        init.run()
        for iteration in range(n_iterations):
            print("\rEpoch: %d, Iteration: %d" % ((iteration//n_mini_batch,iteration)), end = "")

            if standard == 'minmax':
                [X_batch, X_mean_batch, X_max_batch], y_batch, weights_batch = \
                    obtain_mini_batch([train_images, np.hstack([train_mean_images,control_var_training]), train_max_images], train_y, train_weights, n_mini_batch)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch, X_mean: X_mean_batch, weights: weights_batch, X_max: X_max_batch})
            elif standard == 'norm':
                [X_batch, X_mean_batch, X_std_batch], y_batch, weights_batch = \
                    obtain_mini_batch([train_images, np.hstack([train_mean_images,control_var_training]), train_std_images], train_y, train_weights, n_mini_batch)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch, weights: weights_batch, X_mean: X_mean_batch, X_std: X_std_batch})
            elif standard == 'const':
                [X_batch, X_mean_batch], y_batch, weights_batch = \
                    obtain_mini_batch([train_images, np.hstack([train_mean_images,control_var_training])], train_y, train_weights, n_mini_batch)
                sess.run(training_op,
                         feed_dict={X: X_batch, y: y_batch, X_mean: X_mean_batch, weights: weights_batch})

            if iteration % (len(train_images)//n_mini_batch) == 0:
                loss_ = loss.eval(feed_dict = feed_train)
                training_loss_list.append(loss_)
                validation_loss_list.append(loss.eval(feed_dict = feed_validation))
                testing_loss_list.append(loss.eval(feed_dict = feed_test))

                if iteration > 100:
                    if (np.abs(loss_ - ref1) / ref1 < 0.005) & (np.abs(loss_ - ref2) / ref2 < 0.005):
                        print("\nEarly stopping at iteration", iteration)
                        break
                    if (ref1 < loss_) & (ref1 < ref2):
                        print("\nDiverging. stop.")
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
                    if import_hyperparameters is not None:
                        saver.save(sess, output_dir+output_folder+"/models/models_"+"".join([str(ov) for ov in output_var])+"_"+str(image_radius)+
                                   "/model_"+str(linear_coef) +"_"+str(import_hyperparameters) + "_" +str(idx)+run_suffix+"_"+str(iteration)+".ckpt")
                        files = glob.glob(output_dir+output_folder+"/models/models_"+"".join([str(ov) for ov in output_var])+"_"+str(image_radius)+
                                          "/model_"+str(linear_coef)+"_"+str(import_hyperparameters)+"_"+str(idx)+run_suffix+"_*.ckpt*")
                    else:
                        saver.save(sess, output_dir + output_folder + "/models/models_" + "".join(
                            [str(ov) for ov in output_var]) + "_" + str(image_radius) +
                                   "/model_" + str(linear_coef) + "_" + str(idx) + "_hp_" + str(iteration) + ".ckpt")
                        files = glob.glob(output_dir + output_folder + "/models/models_" + "".join(
                            [str(ov) for ov in output_var]) + "_" + str(image_radius) +
                                   "/model_" + str(linear_coef) + "_"  + str(idx) + "_hp_*.ckpt*")

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

cnn_data_name = 'energy_'+standard+'_air'

control_var_training, control_var_validation, control_var_testing, control_scale = \
    util_data.get_control_variables(filename='agriculture_variables_station.xlsx',
                                train_index=data_full_package[cnn_data_name]['index_training'],
                                validation_index=data_full_package[cnn_data_name]['index_validation'],
                                test_index=data_full_package[cnn_data_name]['index_testing'])

### 1. Search hyperparameters for CNN
for output_var, linear_coef in itertools.product(output_vars, linear_coefs):
    model_output_hyper_searching = {}

    lb = 30-image_radius
    ub = 30+image_radius+1
    # use standardized energy data and non-standardized air pollution data
    input_cnn_training = data_full_package[cnn_data_name]['input_training'][:, lb:ub, lb:ub, :]
    input_cnn_validation = data_full_package[cnn_data_name]['input_validation'][:, lb:ub, lb:ub, :]
    input_cnn_testing = data_full_package[cnn_data_name]['input_testing'][:, lb:ub, lb:ub, :]
    output_cnn_training = data_full_package[cnn_data_name]['output_training'][:, output_var]
    output_cnn_validation = data_full_package[cnn_data_name]['output_validation'][:, output_var]
    output_cnn_testing = data_full_package[cnn_data_name]['output_testing'][:, output_var]
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
        if n_hyperparam_searching == -1:
            n_hyperparam_searching = len(hp)
        else:
            idx = np.random.choice(np.arange(len(hp)), n_hyperparam_searching)
            hp = [hp[a] for a in idx]
    else:
        # import linear coef = 0
        # default import from predicting output variable #0 (pm2.5) and radius 30
        with open(output_dir+output_folder+'/results/results_0_30/model_output_hyper_searching_dic_0_' +
                  str(import_hyperparameters)+'_hp.pickle', 'rb') as f:
            hp = pkl.load(f)[1]


    for idx in range(start_parameter, n_hyperparam_searching):
        print("The current index is: ", idx, " out of total ", n_hyperparam_searching)

        if hp is not None:
            if import_hyperparameters is None:
                hp_cur = hp[idx]
                (n_iterations, n_mini_batch, conv_layer_number, conv_filter_number, conv_kernel_size, conv_stride, \
                 pool_size, pool_stride, drop, dropout_rate, bn, fc_layer_number, n_fc, augment, additional_image_size, \
                 epsilon_variance) = hp_cur

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

        if not os.path.exists(output_dir+output_folder+"/models/models_"+"".join([str(ov) for ov in output_var])+"_"+str(image_radius)+"/"):
            os.mkdir(output_dir+output_folder+"/models/models_"+"".join([str(ov) for ov in output_var])+"_"+str(image_radius))

        model_output = train_cnn(idx, train_images, train_y, train_weights, validation_images, validation_y,
                                 validation_weights,
                                 test_images, test_y, test_weights, scale_info, control_var_training,
                                 control_var_validation, control_var_testing,
                                 hyperparams, output_var, linear_coef, import_hyperparameters)

        model_output_hyper_searching[str(idx)] = (model_output,hyperparams,hyperparams_names,linear_coef)

        if not os.path.exists(output_dir+output_folder+"/results/results_"+"".join([str(ov) for ov in output_var])+"_"+str(image_radius)+"/"):
            os.mkdir(output_dir+output_folder+"/results/results_"+"".join([str(ov) for ov in output_var])+"_"+str(image_radius))
        if import_hyperparameters is None:
            with open(output_dir+output_folder+'/results/results_'+"".join([str(ov) for ov in output_var])+'_'+str(image_radius)+
                      "/model_output_hyper_searching_dic_"+str(linear_coef)+"_"+str(idx)+'_hp.pickle', 'wb') as model_output_hyper_searching_dic:
                pickle.dump(model_output_hyper_searching[str(idx)], model_output_hyper_searching_dic, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(output_dir+output_folder+'/results/results_'+"".join([str(ov) for ov in output_var])+'_'+str(image_radius)+
                      '/model_output_hyper_searching_dic_'+str(linear_coef)+"_"+str(import_hyperparameters)+"_"+str(idx)+run_suffix+'.pickle', 'wb') as model_output_hyper_searching_dic:
                pickle.dump(model_output_hyper_searching[str(idx)], model_output_hyper_searching_dic, protocol=pickle.HIGHEST_PROTOCOL)

