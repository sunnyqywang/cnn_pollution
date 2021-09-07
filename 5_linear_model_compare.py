"""
Created on Wed Jan  2 15:03:42 2019

@author: qingyi
"""

import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy
import statsmodels.api as sm
import util_cnn
import util_performance
import sys

output_folder = '200905'
standard = 'const'
output_var = 0
# need further modification if radius other than 30 is used
radius = 30
# number of models to ensemble for cnn
num_models = 5

### 0 read data
cnn_data_name = 'energy_'+standard+'_air_nonstand'
char_name = '_no_airport_no_gas_coal_combined_oil_combined'
with open('../data/project2_energy_1812/process/full_data_process_dic'+char_name+'.pickle', 'rb') as data_standard:
    data_full_package = pickle.load(data_standard)

## use all input variables
# use standardized energy data and non-standardized air pollution data
mean_data_name = 'energy_mean_air_nonstand'
input_mean_training = data_full_package[mean_data_name]['input_training']
input_mean_validation = data_full_package[mean_data_name]['input_validation']
input_mean_testing = data_full_package[mean_data_name]['input_testing']
output_training = data_full_package[mean_data_name]['output_training'][:, output_var]
output_validation = data_full_package[mean_data_name]['output_validation'][:, output_var]
output_testing = data_full_package[mean_data_name]['output_testing'][:, output_var]
weights_training = data_full_package[mean_data_name]['weight_training'].flatten()/100
weights_validation = data_full_package[mean_data_name]['weight_validation'].flatten()/100
weights_testing = data_full_package[mean_data_name]['weight_testing'].flatten()/100

## add constant terms to input
input_mean_training = sm.add_constant(input_mean_training)
input_mean_validation = sm.add_constant(input_mean_validation)
input_mean_testing = sm.add_constant(input_mean_testing)

ntraining = len(input_mean_training)
nvalidation = len(input_mean_validation)
ntesting = len(input_mean_testing)

##########################################################################################
### station linear regression
#
linear_model = sm.WLS(output_training, input_mean_training, weights = weights_training).fit()
# 
#energy_vars = data_full_package['energy_vars']
#params_df = pd.DataFrame(linear_model.params, columns = ['params'], index=['const']+energy_vars)
#print("Coefficients should be mainly positive, however, it looks like: ")
#print(params_df)

# pred
output_linear_mean_train = linear_model.predict(input_mean_training)
output_linear_mean_validation = linear_model.predict(input_mean_validation)
output_linear_mean_test = linear_model.predict(input_mean_testing)

'''
model_output_linear_reg = {}
model_output_linear_reg['output_train'] = output_training_linear_pred
model_output_linear_reg['output_validation'] = output_validation_linear_pred 
model_output_linear_reg['output_test'] = output_testing_linear_pred


with open('../output/' + output_folder + '/model_output_linear_reg_dic.pickle', 'wb') as model_output_linear_reg_dic:
    pickle.dump(model_output_linear_reg, model_output_linear_reg_dic, protocol=pickle.HIGHEST_PROTOCOL)
'''

# individual cell value
cnn_data_name = 'energy_' + standard + '_air_nonstand'
input_cell_training = data_full_package[cnn_data_name]['input_training'].reshape((ntraining, -1))
input_cell_validation = data_full_package[cnn_data_name]['input_validation'].reshape((nvalidation, -1))
input_cell_testing = data_full_package[cnn_data_name]['input_testing'].reshape((ntesting, -1))

input_cell_training = sm.add_constant(input_cell_training)
input_cell_validation = sm.add_constant(input_cell_validation)
input_cell_testing = sm.add_constant(input_cell_testing)

##########################################################################################
### cell linear regression
#
linear_model = sm.WLS(output_training, input_cell_training, weights = weights_training).fit()

output_linear_cell_train = linear_model.predict(input_cell_training)
output_linear_cell_validation = linear_model.predict(input_cell_validation)
output_linear_cell_test = linear_model.predict(input_cell_testing)


##########################################################################################
## load cnn model results
output_cnn_train, output_cnn_validation, output_cnn_test = util_cnn.get_ensembled_prediction(output_folder, output_var,
                                                            radius, num_models, ntraining)
output_cnn_train = output_cnn_train.flatten()
output_cnn_validation = output_cnn_validation.flatten()
output_cnn_test = output_cnn_test.flatten()

##########################################################################################
### compare performance measure

performance_measure_list = [util_performance.w_pearson_coeff, util_performance.w_r2, util_performance.w_rmse,
                            util_performance.w_normalized_rmse,
                            util_performance.w_mean_bias, util_performance.w_normalized_mean_bias,
                            util_performance.w_mean_error, util_performance.w_mean_absolute_percentage_error,
                            util_performance.w_mfb, util_performance.w_mfe]
performance_measure_names=['R', 'R2', 'RMSE', 'NRMSE', 'MB', 'NMB', 'ME', 'MAPE', 'MFB', 'MFE']
datasets = ['train', 'validation', 'test']
index_names = list(itertools.product(performance_measure_names, datasets))
col_names = ['CNN','Linear Station','Linear Cell']

results_cnn = []
results_linear_mean = []
results_linear_cell = []

for performance_measure in performance_measure_list:

    # cnn
    p_train = performance_measure(output_cnn_train, output_training, weights_training)
    p_val = performance_measure(output_cnn_validation, output_validation, weights_validation)
    p_test = performance_measure(output_cnn_test, output_testing, weights_testing)
    results_cnn.append(p_train)
    results_cnn.append(p_val)
    results_cnn.append(p_test)

    # linear mean
    p_train = performance_measure(output_linear_mean_train, output_training, weights_training)
    p_val = performance_measure(output_linear_mean_validation, output_validation, weights_validation)
    p_test = performance_measure(output_linear_mean_test, output_testing, weights_testing)
    results_linear_mean.append(p_train)
    results_linear_mean.append(p_val)
    results_linear_mean.append(p_test)

    # linear cell
    p_train = performance_measure(output_linear_cell_train, output_training, weights_training)
    p_val = performance_measure(output_linear_cell_validation, output_validation, weights_validation)
    p_test = performance_measure(output_linear_cell_test, output_testing, weights_testing)
    results_linear_cell.append(p_train)
    results_linear_cell.append(p_val)
    results_linear_cell.append(p_test)

performance_table = pd.DataFrame(np.array([results_cnn, results_linear_mean, results_linear_cell]).T,
                                     index=index_names, columns=col_names).to_csv('../output/'+output_folder+\
                                     '/complete_performance_table_'+str(output_var)+'_'+str(radius)+'.csv', float_format='%.3f')
