import glob
import pickle
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import mean_squared_error
import util_performance
import matplotlib.pyplot as plt
from setup import *

standard = 'const'
output_folder = '210908'
char_name = '_no_airport_no_gas_coal_combined_oil_combined'
output_var = 0
image_radius = 30

cnn_data_name = 'energy_'+standard+'_air'
with open(data_dir+"process/data_process_dic"+char_name+"_"+standard+".pickle", "rb") as data_standard:
    data_full_package = pickle.load(data_standard)
output_cnn_train = data_full_package[cnn_data_name]['output_training'][:, output_var][:, np.newaxis]
output_cnn_validation = data_full_package[cnn_data_name]['output_validation'][:, output_var][:, np.newaxis]
output_cnn_testing = data_full_package[cnn_data_name]['output_testing'][:, output_var][:, np.newaxis]
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
#     with open(output_dir + output_folder + "/results/results_"+str(output_var)+'_'+str(image_radius)+
#               "/model_output_hyper_searching_dic.pkl", "rb") as f:
#         model_output_hyper_searching = pickle.load(f)
# else:
#     model_output_hyper_searching = {}
files = glob.glob(output_dir+output_folder+'/results/results_'+str(output_var)+'_'+str(image_radius)+
              '/model_output_hyper_searching_dic_*_*.pickle')

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
    temp = f.split("/")[-1].split("_")
    if len(temp) == 7:
        idx_list.append(temp[-1].split(".")[0])
        hp_idx_list.append(idx_list[-1])
    else:
        idx_list.append(temp[-1].split(".")[0])
        hp_idx_list.append(temp[-2])

    with open(f, "rb") as model_output_hyper_searching_dic:
        model_output_hyper_searching = pickle.load(model_output_hyper_searching_dic)

    augment = model_output_hyper_searching[1][-4]
    if augment:
        output_model_train = model_output_hyper_searching[0]['output_train'][:len(output_cnn_train)]
    else:
        output_model_train = model_output_hyper_searching[0]['output_train']

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

#print(np.array([[i for i in range(n_hyperparam_searching)], cnn_validation_loss_list, cnn_test_loss_list]).T)
#print(['index'] + list(model_output_hyper_searching.keys()))
cnn_performance_table = pd.DataFrame(np.array([hp_idx_list, linear_coef_list, idx_list, cnn_train_loss_list, \
    cnn_validation_loss_list, cnn_test_loss_list, cnn_train_r2_list, cnn_validation_r2_list, cnn_test_r2_list]).T,
    columns = ['hp_index','linear_coef','model_index','train_mse','validation_mse','test_mse','train_r2','validation_r2','test_r2'])\
    .sort_values(by=['linear_coef','validation_mse'], ascending=True).to_csv(output_dir+output_folder+\
    '/results/cnn_performance_table_'+str(output_var)+'_'+str(image_radius)+'.csv', index=False)

# if not combined:
#     with open(output_dir + output_folder + "/results/results_"+str(output_var)+'_'+str(image_radius)+
#               "/model_output_hyper_searching_dic_"+str(output_var)+'_'+str(image_radius)+"_"+str(linear_coef)+".pkl", "wb") as f:
#         pickle.dump(model_output_hyper_searching, f)
#
#
# hyperparam_list = np.array(hyperparam_list)
# df = np.concatenate((hyperparam_list, np.reshape(cnn_validation_loss_list, (-1, 1)),\
#                      np.reshape(np.repeat(output_folder, n_hyperparam_searching), (-1, 1))), axis=1)
# hyperparam_table = pd.DataFrame(df, columns = hyperparam_names+('val','output_date')).to_csv(output_dir + output_folder +
#                     "/results/results_"+str(output_var)+'_'+str(image_radius)+
#                     "/hyperparams_result_"+str(output_var)+'_'+str(image_radius)+"_"+str(linear_coef)+".csv")
