# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 22:32:32 2019

@author: wangqi44
"""

import pickle as pkl
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import sys

output_folder = "200317"
top_models = [57,308,165]
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#tf.debugging.set_log_device_placement(True)

run_dir = "../output/" + output_folder + "/models"

# indices of variables to modify in scenario testing
# industrial coal, transportation, residential coal
# INDICES NEED TO CHANGE IN THE UPDATED DATASET
'''
# 191206
scenario_variables = [3, 9, 10]
variables_export = ['AGC', 'AGN', 'AGO', 'INDCT', 'INDNT',
                    'INDOT', 'SVC', 'SVN', 'SVO', 'trans',
                    'RDC', 'RDN', 'DEM', 'rain', 'TEM']
# 191219
scenario_variables = [2,6,7]
variables_export = ['AGC', 'AGO', 'INDCT',
                    'INDOT', 'SVC', 'SVO', 'trans',
                    'RDC', 'DEM', 'rain', 'TEM']
#200111
scenario_variables = [0, 2, 6]
variables_export = ['COAL', 'AGO', 'INDCT',
                    'INDOT', 'SVC', 'SVO', 'trans',
                    'DEM', 'rain', 'TEM']'''
#200222
scenario_variables = [0,2,5]
variables_export = ['COAL', 'AGO', 'INDCT',
                    'INDOT', 'SVC', 'SVOtrans',
                    'DEM', 'rain', 'TEM']
# require gradients to be positive: AGO, INDCT, SVC, SVO, TRANS, COAL (RDC+AGC)
major_variables = [0,1,2,4,5]

# Input
station_index = pd.read_excel('../data/project2_energy_1812/raw/stationID.xlsx')
with open('../data/project2_energy_1812/process/full_data_process_dic_no_airport_no_gas_coal_combined_no_SVO.pickle', 'rb') as data_standard:
    data_full_package = pickle.load(data_standard)

cnn_data_name = 'energy_norm_air_nonstand'
input_cnn_training = data_full_package[cnn_data_name]['input_training']
input_cnn_validation = data_full_package[cnn_data_name]['input_validation']
input_cnn_testing = data_full_package[cnn_data_name]['input_testing']
output_cnn_training = data_full_package[cnn_data_name]['output_training'][:, 0][:, np.newaxis]
output_cnn_validation = data_full_package[cnn_data_name]['output_validation'][:, 0][:, np.newaxis]
output_cnn_testing = data_full_package[cnn_data_name]['output_testing'][:, 0][:, np.newaxis]
output_cnn_all_vars_training = data_full_package[cnn_data_name]['output_training']
output_cnn_all_vars_validation = data_full_package[cnn_data_name]['output_validation']
output_cnn_all_vars_testing = data_full_package[cnn_data_name]['output_testing']
index_cnn_training = data_full_package[cnn_data_name]['index_training']
index_cnn_validation = data_full_package[cnn_data_name]['index_validation']
index_cnn_testing = data_full_package[cnn_data_name]['index_testing']
weights_cnn_training = data_full_package[cnn_data_name]['weight_training'].T / 100
weights_cnn_validation = data_full_package[cnn_data_name]['weight_validation'].T / 100
weights_cnn_testing = data_full_package[cnn_data_name]['weight_testing'].T / 100
mean_data_name = 'energy_mean_air_nonstand'
input_mean_training = data_full_package[mean_data_name]['input_training']
input_mean_validation = data_full_package[mean_data_name]['input_validation']
input_mean_testing = data_full_package[mean_data_name]['input_testing']
std_data_name = 'energy_std_air_nonstand'
input_std_training = data_full_package[std_data_name]['input_training']
input_std_validation = data_full_package[std_data_name]['input_validation']
input_std_testing = data_full_package[std_data_name]['input_testing']

train_images = input_cnn_training
train_y = output_cnn_training
train_weights = weights_cnn_training
validation_images = input_cnn_validation
validation_y = output_cnn_validation
validation_weights = weights_cnn_validation
test_images = input_cnn_testing
test_y = output_cnn_testing
test_weights = weights_cnn_testing

train_mean_images = input_mean_training / 10000
validation_mean_images = input_mean_validation / 10000
test_mean_images = input_mean_testing / 10000
train_std_images = input_std_training / 10000
validation_std_images = input_std_validation / 10000
test_std_images = input_std_testing / 10000

sys.exit()
with open('../data/project2_energy_1812/process/energy_data_dic_no_airport_no_gas_coal_combined_no_SVO.pickle', 'rb') as data_dic:
    data_raw = pickle.load(data_dic)

all_trials_scenario_output = {}
all_trials_tr_gradients = {}
all_trials_val_gradients = {}
all_trials_te_gradients = {}
all_trials_tr_gradients_mean = {}
all_trials_val_gradients_mean = {}
all_trials_te_gradients_mean = {}

# a list of station codes (string)
indices = np.concatenate((index_cnn_training, index_cnn_validation, index_cnn_testing))

batch_size = 100

id10 = []
mask = {}
for v in variables_export:
    mask[v] = []
for s in range(943):
    folder = station_index[station_index['station_code'] == indices[s]]['STID'].iloc[0]
    pixelID = pd.read_csv('../data/project2_energy_1812/raw/result/' + str(folder) + '/ID10.csv',
                          header=None).to_numpy()
    id10 += pixelID.flatten().tolist()
    for v in variables_export:
        mask[v] += (data_raw[indices[s]][v].values.flatten() != 0).tolist()

id10 = np.array(id10).reshape((-1, 1))
mask_df = id10
for v in variables_export:
    mask[v] = np.array(mask[v]).reshape((-1, 1))
    mask_df = np.append(mask_df, mask[v], axis=1)
mask_df = pd.DataFrame(mask_df, index=id10, columns = ['id10'] + variables_export)
mask_full_df = mask_df.copy()
mask_df = mask_df.groupby('id10', as_index=False).max()

print(np.sum(mask_df == 0) / len(mask_df)/ 7)

for index in top_models:
    print("Model: ", index)
    print('Calculating gradients...')

    tf.reset_default_graph()

    sess = tf.Session(config=config)
    saver = tf.train.import_meta_graph(run_dir + "/model_" + str(index) + ".ckpt.meta")
    saver.restore(sess, run_dir + '/model_' + str(index) + ".ckpt")
    graph = tf.get_default_graph()

    X = graph.get_tensor_by_name("inputs/X:0")
    X_mean = graph.get_tensor_by_name("inputs/X_mean:0")
    y = graph.get_tensor_by_name("inputs/y:0")
    output = graph.get_tensor_by_name("outputs/output:0")
    weights = graph.get_tensor_by_name("inputs/weights:0")

    # gradients w.r.t. images
    gradients = tf.gradients(output, X)
    gradients_mean = tf.gradients(output, X_mean)

    # replace 0 values with inf so that the gradients will be 0
    input_std_training[input_std_training == 0] = np.inf
    input_std_validation[input_std_validation == 0] = np.inf
    input_std_testing[input_std_testing == 0] = np.inf

    # gradients w.r.t. mean
    te_gradients_mean = sess.run(gradients_mean, feed_dict={X: test_images, X_mean: test_mean_images})
    te_gradients = sess.run(gradients, feed_dict={X: test_images, X_mean: test_mean_images})
    '''te_gradients = np.array([[np.divide(te_gradients[0][:, i, j, :], input_std_testing) * 10000 + np.array(
        te_gradients_mean) / (61 * 61) for i in range(61)] for j in range(61)])'''
    te_gradients = np.array([[np.divide(te_gradients[0][:, i, j, :], input_std_testing) * 10000  for i in range(61)] for j in range(61)])

    for k in np.arange(batch_size, len(train_images)+batch_size, batch_size):
        if k > len(train_images):
            tr_gradients_mean = np.concatenate((tr_gradients_mean, sess.run(gradients_mean, feed_dict={X: train_images[k - batch_size:,], X_mean: train_mean_images[k - batch_size:,]})[0]))
            tr_gradients = np.concatenate((tr_gradients, sess.run(gradients, feed_dict={X: train_images[k - batch_size:,], X_mean: train_mean_images[k - batch_size:,]})[0]))
        elif k == batch_size:
            tr_gradients_mean = sess.run(gradients_mean, feed_dict={X: train_images[k - batch_size:k,], X_mean: train_mean_images[k - batch_size:k,]})[0]
            tr_gradients = sess.run(gradients, feed_dict={X: train_images[k - batch_size:k,], X_mean: train_mean_images[k - batch_size:k,]})[0]
        else:
            tr_gradients_mean = np.concatenate((tr_gradients_mean, sess.run(gradients_mean, feed_dict={X: train_images[k - batch_size:k,], X_mean: train_mean_images[k - batch_size:k,]})[0]))
            tr_gradients = np.concatenate((tr_gradients, sess.run(gradients, feed_dict={X: train_images[k - batch_size:k,], X_mean: train_mean_images[k - batch_size:k,]})[0]))

    '''tr_gradients = np.array([[np.divide(tr_gradients[:, i, j, :], input_std_training) * 10000 + np.array(
        tr_gradients_mean) / (61 * 61) for i in range(61)] for j in range(61)])'''
    tr_gradients = np.array([[np.divide(tr_gradients[:, i, j, :], input_std_training) * 10000 for i in range(61)] for j in range(61)])

    val_gradients_mean = sess.run(gradients_mean, feed_dict={X: validation_images, X_mean: validation_mean_images})
    val_gradients = sess.run(gradients, feed_dict={X: validation_images, X_mean: validation_mean_images})
    '''val_gradients = np.array([[np.divide(val_gradients[0][:, i, j, :], input_std_validation) * 10000 + np.array(
        val_gradients_mean) / (61 * 61) for i in range(61)] for j in range(61)])'''
    val_gradients = np.array([[np.divide(val_gradients[0][:, i, j, :], input_std_validation) * 10000 for i in range(61)] for j in range(61)])

    # scenario testing
    print('Scenario testing...')
    scenario_output = []
    for variable in scenario_variables:
        temp = []
        for factor in np.arange(0, 0.22, 0.02).tolist():
            train_mean_update = np.copy(train_mean_images)
            train_mean_update[:, variable] = train_mean_update[:, variable] * (1 - factor)
            y_train_scenario = sess.run(output, feed_dict={X: train_images, X_mean: train_mean_update})
            val_mean_update = np.copy(validation_mean_images)
            val_mean_update[:, variable] = val_mean_update[:, variable] * (1 - factor)
            y_val_scenario = sess.run(output, feed_dict={X: validation_images, X_mean: val_mean_update})
            test_mean_update = np.copy(test_mean_images)
            test_mean_update[:, variable] = test_mean_update[:, variable] * (1 - factor)
            y_test_scenario = sess.run(output, feed_dict={X: test_images, X_mean: test_mean_update})
            temp.append(list(y_train_scenario) + list(y_val_scenario) + list(y_test_scenario))
        scenario_output.append(temp)

    all_trials_scenario_output[index] = scenario_output
    all_trials_tr_gradients[index] = tr_gradients
    all_trials_val_gradients[index] = val_gradients
    all_trials_te_gradients[index] = te_gradients
    all_trials_tr_gradients_mean[index] = tr_gradients_mean
    all_trials_val_gradients_mean[index] = val_gradients_mean
    all_trials_te_gradients_mean[index] = te_gradients_mean

    print('Fill in gradients...')
    # Fill in gradients in a matrix form
    gradients_all = np.concatenate((tr_gradients, val_gradients, te_gradients), axis=2)
    gradients = None
    for i in range(gradients_all.shape[2]):
        if gradients is None:
            gradients = np.reshape(gradients_all[:,:,i,:], (-1, gradients_all.shape))
        else:
            gradients = np.concatenate((gradients, np.reshape(gradients_all[:,:,i,:], (-1, gradients_all.shape[3]))), axis=0)

    gradients_full = pd.DataFrame(np.append(id10, gradients, axis=1), index=id10, columns = ['id10'] + variables_export)
    gradients = gradients_full.groupby('id10', as_index=False).max()

    gradients_masked = gradients * mask_df
    gradients_masked['id10'] = np.sqrt(gradients_masked['id10'])

    gradients_full_masked = gradients_full * mask_full_df
    gradients_full_masked['id10'] = np.sqrt(gradients_full_masked['id10'])

    with open("../output/"+output_folder+"/results_"+str(index)+".pkl", "wb") as f:
        pkl.dump(scenario_output, f)
        pkl.dump(gradients, f)
        pkl.dump(gradients_masked, f)
        pkl.dump(gradients_full, f)
        pkl.dump(mask_full_df, f)
        pkl.dump(id10, f)
        pkl.dump(indices, f)

    gradients.to_csv("../output/"+output_folder+"/gradients_"+str(index)+"_test.csv", index=False)
    gradients_masked.to_csv("../output/"+output_folder+"/gradients_masked_"+str(index)+"_test.csv", index=False)


'''
# testing the limits of the stationIDs
max_id = 0
min_id = 10000
for folder in listdir('../data/project2_energy_1812/raw/result'):
    if 'DS' not in folder and '._' not in folder: # address the DS_Store...
        for csv_file in listdir('../data/project2_energy_1812/raw/result/'+ folder):
            directory_path = '../data/project2_energy_1812/raw/result/'+folder+'/ID10.csv'
            pixelID = pd.read_csv(directory_path, header = None)
            pixelID = pixelID[((pixelID != 9999999.0) & (pixelID != 8888888.0))]
            if np.max(np.max(pixelID)) > max_id:
                max_id = np.max(np.max(pixelID))
                print(max_id)
            if np.min(np.min(pixelID)) < min_id:
                min_id = np.min(np.min(pixelID))
                print(min_id)
'''

