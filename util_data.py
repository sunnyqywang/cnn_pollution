"""
utilities for data preprocessing

@author: qingyi
"""


import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
from setup import *

def read_data(char_name, standard, radius, output_var):
    with open(data_dir + "process/data_process_dic" + char_name + "_" + standard + ".pickle", 'rb') as data_standard:
        data_full_package = pickle.load(data_standard)
    cnn_data_name = 'energy_' + standard + '_air'
    lb = 30 - radius
    ub = 30 + radius + 1
    input_cnn_training = data_full_package[cnn_data_name]['input_training'][:, lb:ub, lb:ub, :]
    input_cnn_validation = data_full_package[cnn_data_name]['input_validation'][:, lb:ub, lb:ub, :]
    input_cnn_testing = data_full_package[cnn_data_name]['input_testing'][:, lb:ub, lb:ub, :]
    output_cnn_training = data_full_package[cnn_data_name]['output_training'][:, output_var]
    output_cnn_validation = data_full_package[cnn_data_name]['output_validation'][:, output_var]
    output_cnn_testing = data_full_package[cnn_data_name]['output_testing'][:, output_var]
    # output_cnn_all_vars_training = data_full_package[cnn_data_name]['output_training']
    # output_cnn_all_vars_validation = data_full_package[cnn_data_name]['output_validation']
    # output_cnn_all_vars_testing = data_full_package[cnn_data_name]['output_testing']
    index_cnn_training = data_full_package[cnn_data_name]['index_training']
    index_cnn_validation = data_full_package[cnn_data_name]['index_validation']
    index_cnn_testing = data_full_package[cnn_data_name]['index_testing']
    weights_cnn_training = data_full_package[cnn_data_name]['weight_training'].T / 100
    weights_cnn_validation = data_full_package[cnn_data_name]['weight_validation'].T / 100
    weights_cnn_testing = data_full_package[cnn_data_name]['weight_testing'].T / 100
    sector_max = data_full_package['energy_magnitude_air']
    mean_data_name = 'energy_mean_air'
    input_mean_training = data_full_package[mean_data_name]['input_training']
    input_mean_validation = data_full_package[mean_data_name]['input_validation']
    input_mean_testing = data_full_package[mean_data_name]['input_testing']

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

    return (train_images,train_y,train_weights,
            validation_images,validation_y,validation_weights,
            test_images,test_y,test_weights,
            train_mean_images,validation_mean_images,test_mean_images,
            index_cnn_training,index_cnn_validation,index_cnn_testing,
            sector_max)

def get_control_variables(filename, train_index, validation_index, test_index):
    control_var = pd.read_excel(data_dir + 'raw/' + filename)
    control_var = control_var[
        ["station_code", "fertilizer_area", "livestock_area", "poultry_area"]]
        # ["station_code", "fertilizer_area", "N_fertilizer_area", "livestock_area", "poultry_area", "pcGDP"]]

    # Control scale
    control_var["fertilizer_area"] /= 100
    # control_var["N_fertilizer_area"] /= 100
    control_var["livestock_area"] /= 1000
    control_var["poultry_area"] /= 1000
    # control_var["pcGDP"] /= 100000

    # control_scale = [2,2,3,3,5]
    control_scale = [2, 3, 3]
    control_var = control_var.set_index("station_code")
    control_var = control_var.fillna(0)

    control_var_training = control_var.loc[train_index].to_numpy()
    control_var_validation = control_var.loc[validation_index].to_numpy()
    control_var_testing = control_var.loc[test_index].to_numpy()

    return control_var_training, control_var_validation, control_var_testing, control_scale

def prepare_tensor_data(input_data_dic, output_data_df, pop_weights, all_variables, stratify=None):
    '''
    Prepare data for training. Turn dictionary to t
    Inputs include:
        input_data_dic, output_data_df, truncated_image_size
    Return:
        input_tensor_training
        input_tensor_testing
        output_tensor_training
        output_tensor_testing
        index_city_final_training
        index_city_final_testing
    '''
    n_station = len(input_data_dic.keys())
    image_height = 61
    image_width = 61
    n_channel = len(all_variables)

    # initialize input tensor (data_standard_dic or data_dic)
    input_tensor = np.zeros((n_station, n_channel) + list(list(input_data_dic.values())[0].values())[0].shape)
    # initialize output tensor
    output_tensor = np.zeros(output_data_df.shape)
    # initialize the order and station_name
    index_station_code_list = []
    # need to sort order of data_standard_dic to match air_pollution
    for i in range(len(output_data_df.index)):
        key = output_data_df.index[i]
        output_tensor[i, :] = output_data_df.loc[key, :]
        index_station_code_list.append(key)
        for j in range(len(all_variables)):
            var_ = all_variables[j]
            input_tensor[i, j] = input_data_dic[key][var_]
    # index_station_code_array
    index_station_code_array = np.array(index_station_code_list)
    # sort pop_weights according to index_station_code_array
    pop_weights_sorted = pop_weights[index_station_code_array]

    # Move channels to the last axis
    input_tensor = np.moveaxis(input_tensor, 1, -1)

    # split training and testing set; 4/5 for training,  1/5 for testing
    # index_city_final is the corresponding index of input tensors
    index_array = np.arange(n_station)
    if stratify is None:
        np.random.seed(10)  # for replication
        np.random.shuffle(index_array)
        index_training = index_array[:np.int(n_station * (3 / 5))]
        index_validation = index_array[np.int(n_station * (3 / 5)):np.int(n_station * (4 / 5))]
        index_testing = index_array[np.int(n_station * (4 / 5)):]
    else:
        skf = StratifiedKFold(n_splits=5)
        count = 0
        for _, test_index in skf.split(index_array, stratify):
            if count == 0:
                index_testing = test_index
                count += 1
            elif count == 1:
                index_validation = test_index
                count += 1
            else:
                break
        index_training = index_array[~(np.isin(index_array, index_testing) | np.isin(index_array, index_validation))]

    # use the index
    # input
    input_tensor_training = input_tensor[index_training]
    input_tensor_validation = input_tensor[index_validation]
    input_tensor_testing = input_tensor[index_testing]
    # output
    output_tensor_training = output_tensor[index_training]
    output_tensor_validation = output_tensor[index_validation]
    output_tensor_testing = output_tensor[index_testing]
    # names
    index_station_code_training = index_station_code_array[index_training]
    index_station_code_validation = index_station_code_array[index_validation]
    index_station_code_testing = index_station_code_array[index_testing]
    # weights
    weight_training = pop_weights_sorted.values[index_training][np.newaxis]
    weight_validation = pop_weights_sorted.values[index_validation][np.newaxis]
    weight_testing = pop_weights_sorted.values[index_testing][np.newaxis]
    #
    data_training_validation_testing = {}
    data_training_validation_testing['input_training'] = input_tensor_training
    data_training_validation_testing['input_validation'] = input_tensor_validation
    data_training_validation_testing['input_testing'] = input_tensor_testing
    data_training_validation_testing['output_training'] = output_tensor_training
    data_training_validation_testing['output_validation'] = output_tensor_validation
    data_training_validation_testing['output_testing'] = output_tensor_testing
    data_training_validation_testing['index_training'] = index_station_code_training
    data_training_validation_testing['index_validation'] = index_station_code_validation
    data_training_validation_testing['index_testing'] = index_station_code_testing
    data_training_validation_testing['weight_training'] = weight_training
    data_training_validation_testing['weight_validation'] = weight_validation
    data_training_validation_testing['weight_testing'] = weight_testing

    return data_training_validation_testing
