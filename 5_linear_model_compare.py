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

output_folder = '211010'
standard = 'const'
run_suffix = '_linearfix'
output_var = [0]
# need further modification if radius other than 30 is used
radius = 30
# models to ensemble for cnn
top_models = [(145, 0.3, 6), (145, 0.2, 2), (145, 0.1, 14)]
top_models_multi = [(88, 0.2, -1), (157, 0.2, -1), (81, 0.2, -1)]

variables_export = ['COAL', 'INDCT',
                    'INDOT', 'SVC', 'OIL',
                    'DEM', 'rain', 'TEM']
control_vars = ["fertilizer_area", "livestock_area", "poultry_area"]

### 0 read data
cnn_data_name = 'energy_'+standard+'_air'
char_name = '_no_airport_no_gas_coal_combined_oil_combined'
with open(data_dir+"process/data_process_dic"+char_name+"_"+standard+".pickle", 'rb') as data_standard:
    data_full_package = pickle.load(data_standard)


(train_images,train_y,train_weights,
    validation_images,validation_y,validation_weights,
    test_images,test_y,test_weights,
    train_mean_images,validation_mean_images,test_mean_images,
    index_cnn_training,index_cnn_validation,index_cnn_testing,
    sector_max) = util_data.read_data(char_name, standard, radius=30, output_var=output_var)

train_weights = train_weights.flatten()
validation_weights = validation_weights.flatten()
test_weights = test_weights.flatten()

## add constant terms to input
input_mean_training = sm.add_constant(train_mean_images)
input_mean_validation = sm.add_constant(validation_mean_images)
input_mean_testing = sm.add_constant(test_mean_images)

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
input_mean_training = np.hstack((input_mean_training,control_var_training))
input_mean_validation = np.hstack((input_mean_validation, control_var_validation))
input_mean_testing = np.hstack((input_mean_testing, control_var_testing))

linear_model = sm.WLS(train_y, input_mean_training, weights=train_weights).fit()
params_df = pd.DataFrame(linear_model.params, columns = ['params'], index=['const']+variables_export+control_vars)

print("Coefficients should be mainly positive, however, it looks like: ")
print(params_df['params'].to_numpy())

# pred
output_linear_mean_train = linear_model.predict(input_mean_training)
output_linear_mean_validation = linear_model.predict(input_mean_validation)
output_linear_mean_test = linear_model.predict(input_mean_testing)

output_linear_mean_train = output_linear_mean_train.reshape((-1, len(output_var)))
output_linear_mean_validation = output_linear_mean_validation.reshape((-1, len(output_var)))
output_linear_mean_test = output_linear_mean_test.reshape((-1, len(output_var)))

##########################################################################################
### multitask learning

output_multi_train, output_multi_validation, output_multi_test = \
    util_cnn.get_ensembled_prediction(output_folder, [0,1,2,3,4,5], 30, top_models_multi, len(train_y), run_suffix="_hp")

##########################################################################################
## cnn model results
output_cnn_train, output_cnn_validation, output_cnn_test = \
    util_cnn.get_ensembled_prediction(output_folder, output_var, 30, top_models, len(train_y), run_suffix=run_suffix)


##########################################################################################
### compare performance measure

performance_measure_list = [util_performance.w_pearson_coeff, util_performance.w_r2, util_performance.w_rmse,
                            util_performance.w_normalized_rmse,
                            util_performance.w_mean_bias, util_performance.w_normalized_mean_bias,
                            util_performance.w_mean_error, util_performance.w_mean_absolute_percentage_error,
                            util_performance.w_mfb, util_performance.w_mfe]
performance_measure_names=['R', 'R2', 'RMSE', 'NRMSE', 'MB', 'NMB', 'ME', 'MAPE', 'MFB', 'MFE']
datasets = ['train', 'validation', 'test']
index_names = np.array(list(itertools.product(performance_measure_names, datasets)))
col_names = ['Metric','Dataset','CNN','Linear Station','Multitask Learning']

results_cnn = []
results_linear_mean = []
results_multi = []

for v in output_var:

    for performance_measure in performance_measure_list:

        # cnn
        p_train = performance_measure(output_cnn_train[:,v], train_y[:,v], train_weights)
        p_val = performance_measure(output_cnn_validation[:,v], validation_y[:,v], validation_weights)
        p_test = performance_measure(output_cnn_test[:,v], test_y[:,v], test_weights)
        results_cnn.append(p_train)
        results_cnn.append(p_val)
        results_cnn.append(p_test)

        # linear mean
        p_train = performance_measure(output_linear_mean_train[:,v], train_y[:,v], train_weights)
        p_val = performance_measure(output_linear_mean_validation[:,v], validation_y[:,v], validation_weights)
        p_test = performance_measure(output_linear_mean_test[:,v], test_y[:,v], test_weights)
        results_linear_mean.append(p_train)
        results_linear_mean.append(p_val)
        results_linear_mean.append(p_test)

        # multi-task learning
        p_train = performance_measure(output_multi_train[:,v], train_y[:,v], train_weights)
        p_val = performance_measure(output_multi_validation[:,v], validation_y[:,v], validation_weights)
        p_test = performance_measure(output_multi_test[:,v], test_y[:,v], test_weights)
        results_multi.append(p_train)
        results_multi.append(p_val)
        results_multi.append(p_test)


performance_table = pd.DataFrame(np.hstack((np.array(index_names),np.array([results_cnn, results_linear_mean, results_multi]).T)),
                                     columns=col_names).to_csv(output_dir+output_folder+\
                                     '/complete_performance_table_'+''.join([str(v) for v in output_var])+'_'+str(radius)+'.csv',
                                     float_format='%.3f', index=False)
