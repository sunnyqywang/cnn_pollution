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

from setup import *
import util_data

output_folder = '210921'
standard = 'const'
output_var = [0]
# need further modification if radius other than 30 is used
radius = 30
# number of models to ensemble for cnn
num_models = 5

variables_export = ['COAL', 'INDCT',
                    'INDOT', 'SVC', 'OIL',
                    'DEM', 'rain', 'TEM']
control_vars = ["fertilizer_area", "N_fertilizer_area", "livestock_area", "poultry_area", "pcGDP"]

### 0 read data
cnn_data_name = 'energy_'+standard+'_air'
char_name = '_no_airport_no_gas_coal_combined_oil_combined'
with open(data_dir+"process/data_process_dic"+char_name+"_"+standard+".pickle", 'rb') as data_standard:
    data_full_package = pickle.load(data_standard)

## use all input variables
# use standardized energy data and non-standardized air pollution data
mean_data_name = 'energy_mean_air'
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

control_var_training, control_var_validation, control_var_testing, control_scale = \
    util_data.get_control_variables(filename='agriculture_variables_station.xlsx',
                                train_index=data_full_package[cnn_data_name]['index_training'],
                                validation_index=data_full_package[cnn_data_name]['index_validation'],
                                test_index=data_full_package[cnn_data_name]['index_testing'])


##########################################################################################
### station linear regression
#
# linear_model = sm.WLS(output_training, np.hstack((input_mean_training,control_var_training)), weights = weights_training).fit()
linear_model_2 = sm.WLS(output_training, input_mean_training, weights = weights_training).fit()

# params_df = pd.DataFrame(linear_model.params, columns = ['params'], index=['const']+variables_export+control_vars)
params_df_2 = pd.DataFrame(linear_model_2.params, columns = ['params'], index=['const']+variables_export)
print("Coefficients should be mainly positive, however, it looks like: ")
print(params_df)

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

# Remove saved models
