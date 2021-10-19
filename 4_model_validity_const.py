import glob
import numpy as np
import os
import pandas as pd
import pickle
import sys
import tensorflow as tf

from setup import *
from setup_cnn import *
import util_data

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
tf.debugging.set_log_device_placement(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

output_folder = "211010"
output_var = [0]
output_var_name = output_var_names_all[output_var[0]]
if len(output_var) > 1:
    print("Multiple outputs not accomodated. Exiting...")
    sys.exit()

radius = 30
standard = 'const'
run_suffix = '_linearfix'
import_hyperparameters = 145

run_dir = output_dir+output_folder+"/models/models_" + "".join([str(ov) for ov in output_var]) + "_" + str(radius)
if import_hyperparameters is None:
    cnn_results = pd.read_csv(output_dir+output_folder+"/results/cnn_performance_table_"+"".join([str(ov) for ov in output_var])\
                          +"_"+str(radius)+run_suffix+".csv")
else:
    cnn_results = pd.read_csv(
        output_dir + output_folder + "/results/cnn_performance_table_" + "".join([str(ov) for ov in output_var]) \
        + "_" + str(radius) + "_" + str(import_hyperparameters) + run_suffix + ".csv")
cnn_results = cnn_results.sort_values(by='val_mse_'+output_var_name)
best_val = cnn_results['val_mse_'+output_var_name].iloc[0]

# indices of variables to modify in scenario testing
# industrial coal, transportation, residential coal
# INDICES NEED TO CHANGE IN THE UPDATED DATASET
'''
# 191206
scenario_variables = [3, 9, 10]
variables_export = ['AGC', 'AGN', 'AGO', 'INDCT', 'INDNT',
                    'INDOT', 'SVC', 'SVN', 'SVO', 'trans',
                    'RDC', 'RDN', 'DEM', 'rain', 'TEM']
# require gradients to be positive: AGO, INDCT, SVC, SVO, TRANS, RDC
major_variables = [2,3,6,8,9,10]
oil_coal_variables = [0,2,3,5,6,8,9,10]
gas_variables = [1,4,7,11]

# 191219
scenario_variables = [2,6,7]
variables_export = ['AGC', 'AGO', 'INDCT',
                    'INDOT', 'SVC', 'SVO', 'trans',
                    'RDC', 'DEM', 'rain', 'TEM']
# require gradients to be positive: AGO, INDCT, SVC, SVO, TRANS, RDC
major_variables = [1,2,4,5,6,7]

#200111
scenario_variables = [0, 2, 6]
variables_export = ['COAL', 'AGO', 'INDCT',
                    'INDOT', 'SVC', 'SVO', 'trans',
                    'DEM', 'rain', 'TEM']
# require gradients to be positive: AGO, INDCT, SVC, SVO, TRANS, COAL (RDC+AGC)
major_variables = [0,1,2,4,5,6]

#200222 and 200317 and 200417
scenario_variables = [0,2,5]
variables_export = ['COAL', 'AGO', 'INDCT',
                    'INDOT', 'SVC', 'SVOtrans',
                    'DEM', 'rain', 'TEM']
                    
#20060602
char_name = '_no_airport_no_gas_no_SVO'
scenario_variables = [0,3,6]
variables_export = ['RDC', 'AGC', 'AGO', 'INDCT',
                    'INDOT', 'SVC', 'SVOtrans',
                    'DEM', 'rain', 'TEM']
'''

#20060601
char_name = '_no_airport_no_gas_coal_combined_oil_combined'
scenario_variables = [0,1,4]
variables_export = ['COAL', 'INDCT',
                    'INDOT', 'SVC', 'OIL',
                    'DEM', 'rain', 'TEM']
n_channel = len(variables_export)

# require gradients to be positive: AGO, INDCT, SVC, SVO, TRANS, COAL (RDC+AGC)
major_variables = [0,1,2,3,4]
indct = 2
coal = 0

# Inputs
(train_images,train_y,train_weights,
    validation_images,validation_y,validation_weights,
    test_images,test_y,test_weights,
    train_mean_images,validation_mean_images,test_mean_images,
    index_cnn_training,index_cnn_validation,index_cnn_testing,
    sector_max) = util_data.read_data(char_name, standard, radius, output_var)

control_var_training, control_var_validation, control_var_testing, control_scale = \
    util_data.get_control_variables(filename='agriculture_variables_station.xlsx',
                                train_index=index_cnn_training,
                                validation_index=index_cnn_validation,
                                test_index=index_cnn_testing)

all_trials_scenario_output = {}
all_trials_tr_gradients = {}
all_trials_val_gradients = {}
all_trials_te_gradients = {}
all_trials_tr_gradients_mean = {}
all_trials_val_gradients_mean = {}
all_trials_te_gradients_mean = {}

model_validity = []

for hp_idx, model_idx, linear_coef, val, test in zip(cnn_results['hp_index'].astype(int), cnn_results['model_index'].astype(int),
    cnn_results['linear_coef'], cnn_results['val_mse_'+output_var_name], cnn_results['test_mse_'+output_var_name]):

    # if prediction error too large, then terminate analysis
    if val > best_val * 1.5:
        print('Results 50% worse than best model. Terminating...')
        break

    tf.reset_default_graph()

    sess = tf.Session(config=config)
    if linear_coef == 0:
        linear_coef = int(linear_coef)
    files = glob.glob(run_dir + "/model_"+str(linear_coef) +"_"+str(hp_idx) + "_" +str(model_idx)+run_suffix+"_*.ckpt.meta")
    if len(files) == 0:
        print("Model not found. Exiting...")
        print(run_dir + "/model_"+str(linear_coef) +"_"+str(hp_idx) + "_" +str(model_idx)+run_suffix+"_*.ckpt.meta")
        sys.exit()
    elif len(files) > 1:
        print('More than one model found. Exiting...')
        sys.exit()
    else:
        file = files[0]

    saver = tf.train.import_meta_graph(file)
    saver.restore(sess, '.'.join(file.split('.')[:-1]))
    graph = tf.get_default_graph()

    X = graph.get_tensor_by_name("inputs/X:0")
    y = graph.get_tensor_by_name("inputs/y:0")
    X_mean = graph.get_tensor_by_name("inputs/X_mean:0")
    weights = graph.get_tensor_by_name("inputs/weights:0")

    output = graph.get_tensor_by_name("outputs/final_output:0")
    bias = graph.get_tensor_by_name("outputs/bias:0")

    # GRADIENTS
    # gradients w.r.t. mean values
    # gradients_mean = tf.gradients(output, X_mean)
    #
    # tr_gradients_mean = sess.run(gradients_mean, feed_dict={X: train_images, X_mean: np.hstack([train_mean_images, control_var_training])})
    # val_gradients_mean = sess.run(gradients_mean, feed_dict={X: validation_images, X_mean: np.hstack([validation_mean_images, control_var_validation])})
    # te_gradients_mean = sess.run(gradients_mean, feed_dict={X: test_images, X_mean: np.hstack([test_mean_images, control_var_testing])})
    # for (i,k) in control_scale:
    #     tr_gradients_mean[i+n_channel] = tr_gradients_mean[i+n_channel] / np.power(10, k)
    #     val_gradients_mean[i+n_channel] = val_gradients_mean[i+n_channel] / np.power(10, k)
    #     te_gradients_mean[i+n_channel] = te_gradients_mean[i+n_channel] / np.power(10, k)

    # gradients w.r.t. raw image values (chain rule）
    gradients = tf.gradients(output, X)
    # [0] since only one variable is in sess.run
    # 10000* unit is dPM / d 10k tons of emissions
    # (radius*2+1) x (radius*2+1) x #sectors x #observations
    tr_gradients = sess.run(gradients, feed_dict={X: train_images, X_mean: np.hstack([train_mean_images, control_var_training])})
    tr_gradients = 10000*np.array([[[np.divide(tr_gradients[0][:, i, j, l], sector_max[k]) \
                                     for k,l in zip(variables_export, range(len(variables_export)))] \
                                    for i in range(radius*2+1)] for j in range(radius*2+1)])

    val_gradients = sess.run(gradients, feed_dict={X: validation_images, X_mean: np.hstack([validation_mean_images, control_var_validation])})
    val_gradients = 10000*np.array([[[np.divide(val_gradients[0][:, i, j, l], sector_max[k]) \
                                      for k,l in zip(variables_export, range(len(variables_export)))] \
                                     for i in range(radius*2+1)] for j in range(radius*2+1)])

    te_gradients = sess.run(gradients, feed_dict={X: test_images, X_mean: np.hstack([test_mean_images, control_var_testing])})
    te_gradients = 10000*np.array([[[np.divide(te_gradients[0][:, i, j, l], sector_max[k]) \
                                     for k,l in zip(variables_export, range(len(variables_export)))] \
                                    for i in range(radius*2+1)] for j in range(radius*2+1)])

    gradients_im_mean = np.sum(np.concatenate((tr_gradients, val_gradients, te_gradients), axis=3), axis=(0,1))

    # Filter models based on validity of gradients
    gradients_im_mean_mean = np.mean(gradients_im_mean, axis=1)
    gradients_mean_major = gradients_im_mean_mean[[major_variables]]
    notvalid = (gradients_im_mean_mean[coal] < gradients_im_mean_mean[indct])

    '''
    # gas variables taken out of the model 191219
    notvalid = 0
    for gas_var in gas_variables:
        notvalid += np.sum(te_gradients_mean_mean[gas_var] > te_gradients_mean_mean[[oil_coal_variables]])
    '''
    bias = sess.run(bias)[0]
    model_validity.append([hp_idx, linear_coef, model_idx, val, test, int(np.sum(gradients_mean_major < 0) != 0), int(notvalid), bias] \
                          + gradients_im_mean_mean.tolist())
    sess.close()
    print(hp_idx, linear_coef, model_idx, val)


model_validity = pd.DataFrame(model_validity, columns=['hp_index','linear_coef','model_index', 'validation', 'test',
                                                       'neg_gradients', 'coals', 'bias'] + \
                [v+"_cell_mean" for v in variables_export]).to_csv(output_dir+output_folder+\
                 "/results/model_validity_"+"".join([str(ov) for ov in output_var])+"_"+str(radius)+run_suffix+".csv",
                index=False, float_format='%.3f')
