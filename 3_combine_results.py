import glob
import pickle
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import mean_squared_error
import util_performance
import matplotlib.pyplot as plt
from setup import *
from setup_cnn import *

standard = 'const'
output_folder = '210930'
char_name = '_no_airport_no_gas_coal_combined_oil_combined'
output_var = [0]
output_var_names = [output_var_names_all[ov] for ov in output_var]
image_radius = 30
run_suffix = "_withinit"

import_hyperparameters = 153

cnn_data_name = 'energy_'+standard+'_air'
with open(data_dir+"process/data_process_dic"+char_name+"_"+standard+".pickle", "rb") as data_standard:
    data_full_package = pickle.load(data_standard)
output_cnn_train = data_full_package[cnn_data_name]['output_training'][:, output_var]
output_cnn_validation = data_full_package[cnn_data_name]['output_validation'][:, output_var]
output_cnn_testing = data_full_package[cnn_data_name]['output_testing'][:, output_var]
weights_cnn_train = data_full_package[cnn_data_name]['weight_training'].T/100
weights_cnn_validation = data_full_package[cnn_data_name]['weight_validation'].T/100
weights_cnn_testing = data_full_package[cnn_data_name]['weight_testing'].T/100

mean_data_name = 'energy_mean_air'
input_mean_training = data_full_package[mean_data_name]['input_training']
input_mean_validation = data_full_package[mean_data_name]['input_validation']
input_mean_testing = data_full_package[mean_data_name]['input_testing']
train_mean_images = input_mean_training / 10000
validation_mean_images = input_mean_validation / 10000
test_mean_images = input_mean_testing / 10000

# combined = False
# if combined:
#     with open(output_dir + output_folder + "/results/results_"+"".join([str(ov) for ov in output_var])+'_'+str(image_radius)+
#               "/model_output_hyper_searching_dic.pkl", "rb") as f:
#         model_output_hyper_searching = pickle.load(f)
# else:
#     model_output_hyper_searching = {}
if import_hyperparameters is None:
    files = glob.glob(output_dir+output_folder+'/results/results_'+"".join([str(ov) for ov in output_var])+'_'+str(image_radius)+
                  '/model_output_hyper_searching_dic_*'+run_suffix+'.pickle')
else:
    files = glob.glob(
        output_dir + output_folder + '/results/results_' + "".join([str(ov) for ov in output_var]) + '_' + str(
            image_radius) + '/model_output_hyper_searching_dic_*_' + str(import_hyperparameters) + '_*'+run_suffix+'.pickle')

cnn_train_loss_list = []
cnn_validation_loss_list = []
cnn_test_loss_list = []
cnn_train_r2_list = []
cnn_validation_r2_list = []
cnn_test_r2_list = []
hyperparam_list = []
linear_coef_list = []
hp_idx_list = []
idx_list = []

for f in files:
    # see how many parameters are in the file name
    # if 7 then it is part of the hyperparameter search phase, the hp combo is run only once
    # if 8 the additional parameter is the model number of hp re-run
    temp = f.split("/")[-1].split("_")
    if len(temp) == 8:
        idx_list.append("-1")
        hp_idx_list.append(temp[-2])
    else:
        idx_list.append(temp[-2])
        hp_idx_list.append(temp[-3])

    with open(f, "rb") as model_output_hyper_searching_dic:
        model_output_hyper_searching = pickle.load(model_output_hyper_searching_dic)

    augment = model_output_hyper_searching[1][-4]
    if augment:
        output_model_train = model_output_hyper_searching[0]['output_train'][:len(output_cnn_train)]
    else:
        output_model_train = model_output_hyper_searching[0]['output_train']

    for ov in range(len(output_var)):
        cnn_train_loss_list.append(mean_squared_error(output_cnn_train[:,ov], output_model_train[:,ov]))
        cnn_validation_loss_list.append(mean_squared_error(output_cnn_validation[:,ov], model_output_hyper_searching[0]['output_validation'][:,ov]))
        cnn_test_loss_list.append(mean_squared_error(output_cnn_testing[:,ov], model_output_hyper_searching[0]['output_test'][:,ov]))
        cnn_train_r2_list.append(util_performance.w_r2(output_model_train[:,ov],
                                         output_cnn_train[:,ov], weights_cnn_train))
        cnn_validation_r2_list.append(util_performance.w_r2(model_output_hyper_searching[0]['output_validation'][:,ov],
                                         output_cnn_validation[:,ov], weights_cnn_validation))
        cnn_test_r2_list.append(util_performance.w_r2(model_output_hyper_searching[0]['output_test'][:,ov],
                                         output_cnn_testing[:,ov], weights_cnn_testing))

    # overall, if more than one variable, for ranking of models
    if len(output_var) > 1:
        cnn_train_loss_list.append(mean_squared_error(output_cnn_train, output_model_train))
        cnn_validation_loss_list.append(mean_squared_error(output_cnn_validation, model_output_hyper_searching[0]['output_validation']))
        cnn_test_loss_list.append(mean_squared_error(output_cnn_testing, model_output_hyper_searching[0]['output_test']))
        cnn_train_r2_list.append(util_performance.w_r2(output_model_train,
                                         output_cnn_train, weights_cnn_train))
        cnn_validation_r2_list.append(util_performance.w_r2(model_output_hyper_searching[0]['output_validation'],
                                         output_cnn_validation, weights_cnn_validation))
        cnn_test_r2_list.append(util_performance.w_r2(model_output_hyper_searching[0]['output_test'],
                                         output_cnn_testing, weights_cnn_testing))

    hyperparam_list.append(model_output_hyper_searching[1])
    linear_coef_list.append(model_output_hyper_searching[-1])

hyperparam_names = model_output_hyper_searching[2]

if len(output_var) == 1:
    output_var_len = len(output_var)
else:
    output_var_len = len(output_var)+1

cnn_train_loss_list = np.array(cnn_train_loss_list).reshape(-1, output_var_len)
cnn_validation_loss_list = np.array(cnn_validation_loss_list).reshape(-1, output_var_len)
cnn_test_loss_list = np.array(cnn_test_loss_list).reshape(-1, output_var_len)
cnn_train_r2_list = np.array(cnn_train_r2_list).reshape(-1, output_var_len)
cnn_validation_r2_list = np.array(cnn_validation_r2_list).reshape(-1, output_var_len)
cnn_test_r2_list = np.array(cnn_test_r2_list).reshape(-1, output_var_len)
np_df = np.array([hp_idx_list, linear_coef_list, idx_list]).T
np_df = np.hstack([np_df, cnn_train_loss_list, \
    cnn_validation_loss_list, cnn_test_loss_list, cnn_train_r2_list, cnn_validation_r2_list, cnn_test_r2_list])

col_df = ['hp_index','linear_coef','model_index']
if len(output_var) > 1:
    output_var_names += ["all"]
    rank_var = 'val_mse_all'
else:
    rank_var = 'val_mse_'+output_var_names[0]

for m in ['train_mse','val_mse','test_mse','train_r2','val_r2','test_r2']:
    for v in output_var_names:
        col_df += [m+'_'+v]

if import_hyperparameters is None:
    cnn_performance_table = pd.DataFrame(np_df, columns = col_df)\
        .sort_values(by=['linear_coef', rank_var], ascending=True).to_csv(output_dir+output_folder+\
        '/results/cnn_performance_table_'+"".join([str(ov) for ov in output_var])+'_'+str(image_radius)+run_suffix+'.csv', index=False)
else:
    cnn_performance_table = pd.DataFrame(np_df, columns = col_df)\
        .sort_values(by=['linear_coef', rank_var], ascending=True).to_csv(output_dir+output_folder+\
        '/results/cnn_performance_table_'+"".join([str(ov) for ov in output_var])+'_'+str(image_radius)+'_'+
        str(import_hyperparameters)+run_suffix+'.csv', index=False)

# if not combined:
#     with open(output_dir + output_folder + "/results/results_"+"".join([str(ov) for ov in output_var])+'_'+str(image_radius)+
#               "/model_output_hyper_searching_dic_"+"".join([str(ov) for ov in output_var])+'_'+str(image_radius)+"_"+str(linear_coef)+".pkl", "wb") as f:
#         pickle.dump(model_output_hyper_searching, f)
#
#
# hyperparam_list = np.array(hyperparam_list)
# df = np.concatenate((hyperparam_list, np.reshape(cnn_validation_loss_list, (-1, 1)),\
#                      np.reshape(np.repeat(output_folder, n_hyperparam_searching), (-1, 1))), axis=1)
# hyperparam_table = pd.DataFrame(df, columns = hyperparam_names+('val','output_date')).to_csv(output_dir + output_folder +
#                     "/results/results_"+"".join([str(ov) for ov in output_var])+'_'+str(image_radius)+
#                     "/hyperparams_result_"+"".join([str(ov) for ov in output_var])+'_'+str(image_radius)+"_"+str(linear_coef)+".csv")
