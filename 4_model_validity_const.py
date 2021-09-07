import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
tf.debugging.set_log_device_placement(True)

output_folder = "200905"
output_var = 0
radius = 20

run_dir = "../output/"+output_folder+"/models/models_" + str(output_var) + "_" + str(radius)
cnn_results = pd.read_csv("../output/"+output_folder+"/results/cnn_performance_table_"+str(output_var)+"_"+str(radius)+".csv")
best_val = cnn_results['validation_mse'].iloc[0]

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

# require gradients to be positive: AGO, INDCT, SVC, SVO, TRANS, COAL (RDC+AGC)
major_variables = [0,1,2,3,4]
indct = 2
coal = 0

# Inputs
with open('../data/project2_energy_1812/process/full_data_process_dic'+char_name+'.pickle', 'rb') as data_standard:
    data_full_package = pickle.load(data_standard)
cnn_data_name = 'energy_const_air_nonstand'
lb = 30 - radius
ub = 30 + radius + 1
input_cnn_training = data_full_package[cnn_data_name]['input_training'][:, lb:ub, lb:ub, :]
input_cnn_validation = data_full_package[cnn_data_name]['input_validation'][:, lb:ub, lb:ub, :]
input_cnn_testing = data_full_package[cnn_data_name]['input_testing'][:, lb:ub, lb:ub, :]
output_cnn_training = data_full_package[cnn_data_name]['output_training'][:, 0][:, np.newaxis]
output_cnn_validation = data_full_package[cnn_data_name]['output_validation'][:, 0][:, np.newaxis]
output_cnn_testing = data_full_package[cnn_data_name]['output_testing'][:, 0][:, np.newaxis]
output_cnn_all_vars_training = data_full_package[cnn_data_name]['output_training']
output_cnn_all_vars_validation = data_full_package[cnn_data_name]['output_validation']
output_cnn_all_vars_testing = data_full_package[cnn_data_name]['output_testing']
index_cnn_training = data_full_package[cnn_data_name]['index_training']
index_cnn_validation = data_full_package[cnn_data_name]['index_validation']
index_cnn_testing = data_full_package[cnn_data_name]['index_testing']
weights_cnn_training = data_full_package[cnn_data_name]['weight_training'].T/100
weights_cnn_validation = data_full_package[cnn_data_name]['weight_validation'].T/100
weights_cnn_testing = data_full_package[cnn_data_name]['weight_testing'].T/100
sector_max = data_full_package['energy_magnitude_air_nonstand']

train_images=input_cnn_training
train_y=output_cnn_training
train_weights = weights_cnn_training
validation_images= input_cnn_validation
validation_y =output_cnn_validation
validation_weights = weights_cnn_validation
test_images =input_cnn_testing
test_y =output_cnn_testing
test_weights = weights_cnn_testing

all_trials_scenario_output = {}
all_trials_tr_gradients = {}
all_trials_val_gradients = {}
all_trials_te_gradients = {}
all_trials_tr_gradients_mean = {}
all_trials_val_gradients_mean = {}
all_trials_te_gradients_mean = {}

model_validity = []

for index, val in zip(cnn_results['index'].astype(int), cnn_results['validation_mse']):

    # if prediction error too large, then terminate analysis
    if val > best_val * 1.5:
        print('Results 50% worse than best model. Terminating...')
        break

    tf.reset_default_graph()

    sess = tf.Session(config=config)
    saver = tf.train.import_meta_graph(run_dir + "/model_" + str(index) + ".ckpt.meta")
    saver.restore(sess, run_dir + '/model_' + str(index) + ".ckpt")
    graph = tf.get_default_graph()

    X = graph.get_tensor_by_name("inputs/X:0")
    y = graph.get_tensor_by_name("inputs/y:0")
    output = graph.get_tensor_by_name("outputs/output:0")
    weights = graph.get_tensor_by_name("inputs/weights:0")
    bias = graph.get_tensor_by_name("outputs/bias:0")

    # gradients w.r.t. images
    gradients = tf.gradients(output, X)

    # gradients w.r.t. raw image values (chain rule）
    tr_gradients = sess.run(gradients, feed_dict={X: train_images})
    # [0] since only one variable is in sess.run
    # 10000* unit is dPM / d 10k tons of emissions
    # (radius*2+1) x (radius*2+1) x #sectors x #observations
    tr_gradients = 10000*np.array([[[np.divide(tr_gradients[0][:, i, j, l], sector_max[k]) \
                                     for k,l in zip(variables_export, range(len(variables_export)))] \
                                    for i in range(radius*2+1)] for j in range(radius*2+1)])

    val_gradients = sess.run(gradients, feed_dict={X: validation_images})
    val_gradients = 10000*np.array([[[np.divide(val_gradients[0][:, i, j, l], sector_max[k]) \
                                      for k,l in zip(variables_export, range(len(variables_export)))] \
                                     for i in range(radius*2+1)] for j in range(radius*2+1)])

    te_gradients = sess.run(gradients, feed_dict={X: test_images})
    te_gradients = 10000*np.array([[[np.divide(te_gradients[0][:, i, j, l], sector_max[k]) \
                                     for k,l in zip(variables_export, range(len(variables_export)))] \
                                    for i in range(radius*2+1)] for j in range(radius*2+1)])

    gradients_mean = np.sum(np.concatenate((tr_gradients, val_gradients, te_gradients), axis=3), axis=(0,1))

    # Filter models based on validity of gradients
    gradients_mean_mean = np.mean(gradients_mean, axis=1)
    gradients_mean_major = gradients_mean_mean[[major_variables]]
    notvalid = (gradients_mean_mean[coal] < gradients_mean_mean[indct])

    '''
    # gas variables taken out of the model 191219
    notvalid = 0
    for gas_var in gas_variables:
        notvalid += np.sum(te_gradients_mean_mean[gas_var] > te_gradients_mean_mean[[oil_coal_variables]])
    '''
    bias = sess.run(bias)[0]
    model_validity.append([index, val, int(np.sum(gradients_mean_major < 0) != 0), int(notvalid), bias] \
                          + gradients_mean_mean.tolist())
    sess.close()
    print(index, val)


model_validity = pd.DataFrame(model_validity, columns=['index', 'validation', 'neg_gradients', 'coals', 'bias'] + \
                [v+"cell_mean" for v in variables_export]).to_csv("../output/"+output_folder+\
                 "/results/model_validity_"+str(output_var)+"_"+str(radius)+".csv", index=False, float_format='%.3f')
