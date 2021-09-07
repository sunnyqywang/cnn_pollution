import pickle
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import mean_squared_error
import util_performance
import matplotlib.pyplot as plt

n_hyperparam_searching = 150
standard = 'const'
output_folder = '200905'
char_name = '_no_airport_no_gas_coal_combined_oil_combined'
output_var = 0
image_radius = 30

cnn_data_name = 'energy_'+standard+'_air_nonstand'
with open('../data/project2_energy_1812/process/full_data_process_dic'+char_name+'.pickle', 'rb') as data_standard:
    data_full_package = pickle.load(data_standard)
print(data_full_package[cnn_data_name]['input_training'].shape)
output_cnn_train = data_full_package[cnn_data_name]['output_training'][:, output_var][:, np.newaxis]
output_cnn_validation = data_full_package[cnn_data_name]['output_validation'][:, output_var][:, np.newaxis]
output_cnn_testing = data_full_package[cnn_data_name]['output_testing'][:, output_var][:, np.newaxis]
weights_cnn_train = data_full_package[cnn_data_name]['weight_training'].T/100
weights_cnn_validation = data_full_package[cnn_data_name]['weight_validation'].T/100
weights_cnn_testing = data_full_package[cnn_data_name]['weight_testing'].T/100

combined = False
if combined:
    with open("../output/" + output_folder + "/results/model_output_hyper_searching_dic.pkl", "rb") as f:
        model_output_hyper_searching = pickle.load(f)
else:  
    model_output_hyper_searching = {}
    ### Find the best hyperparameters
    for i in range(n_hyperparam_searching):
        with open('../output/'+output_folder+'/results/results_'+str(output_var)+'_'+str(image_radius)+'/model_output_hyper_searching_dic_'+str(i)+'.pickle', 'rb') as model_output_hyper_searching_dic:
            model_output_hyper_searching[str(i)] = pickle.load(model_output_hyper_searching_dic)

cnn_train_loss_list = []
cnn_validation_loss_list = []
cnn_test_loss_list = []
cnn_train_r2_list = []
cnn_validation_r2_list = []
cnn_test_r2_list = []

hyperparam_list = []
for i in range(n_hyperparam_searching):
    augment = model_output_hyper_searching[str(i)][1][-4]
    if augment:
        output_model_train = model_output_hyper_searching[str(i)][0]['output_train'][:len(output_cnn_train)]
    else:
        output_model_train = model_output_hyper_searching[str(i)][0]['output_train']

    cnn_train_loss_list.append(mean_squared_error(output_cnn_train, output_model_train))
    cnn_validation_loss_list.append(mean_squared_error(output_cnn_validation, model_output_hyper_searching[str(i)][0]['output_validation']))
    cnn_test_loss_list.append(mean_squared_error(output_cnn_testing, model_output_hyper_searching[str(i)][0]['output_test']))
    cnn_train_r2_list.append(util_performance.w_r2(output_model_train,
                                     output_cnn_train, weights_cnn_train))
    cnn_validation_r2_list.append(util_performance.w_r2(model_output_hyper_searching[str(i)][0]['output_validation'],
                                     output_cnn_validation, weights_cnn_validation))
    cnn_test_r2_list.append(util_performance.w_r2(model_output_hyper_searching[str(i)][0]['output_test'],
                                     output_cnn_testing, weights_cnn_testing))
    hyperparam_list.append(model_output_hyper_searching[str(i)][1])


hyperparam_names = model_output_hyper_searching['0'][2]
#print(np.array([[i for i in range(n_hyperparam_searching)], cnn_validation_loss_list, cnn_test_loss_list]).T)
#print(['index'] + list(model_output_hyper_searching.keys()))
cnn_performance_table = pd.DataFrame(np.array([[i for i in range(n_hyperparam_searching)], cnn_train_loss_list, \
    cnn_validation_loss_list, cnn_test_loss_list, cnn_train_r2_list, cnn_validation_r2_list, cnn_test_r2_list]).T,
    columns = ['index','train_mse','validation_mse','test_mse','train_r2','validation_r2','test_r2']).sort_values(by='validation_mse', \
    ascending=True).to_csv('../output/'+output_folder+'/results/cnn_performance_table_'+str(output_var)+'_'+str(image_radius)+'.csv', index=False)

if not combined:
    with open("../output/" + output_folder + "/results/model_output_hyper_searching_dic_"+str(output_var)+'_'+str(image_radius)+".pkl", "wb") as f:
        pickle.dump(model_output_hyper_searching, f)


hyperparam_list = np.array(hyperparam_list)
df = np.concatenate((hyperparam_list, np.reshape(cnn_validation_loss_list, (-1, 1)),\
                     np.reshape(np.repeat(output_folder, n_hyperparam_searching), (-1, 1))), axis=1)
hyperparam_table = pd.DataFrame(df, columns = hyperparam_names+('val','output_date')).to_csv("../output/" + output_folder + "/results/hyperparams_result_"+str(output_var)+'_'+str(image_radius)+".csv")
