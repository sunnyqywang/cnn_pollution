"""
utilities for data preprocessing

@author: qingyi
"""


import numpy as np

def prepare_tensor_data(input_data_dic, output_data_df, pop_weights, all_variables):
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
    input_tensor = np.zeros((n_station, image_height, image_width, n_channel))
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
            input_tensor[i, :, :, j] = input_data_dic[key][var_]
    # index_station_code_array
    index_station_code_array = np.array(index_station_code_list)
    # sort pop_weights according to index_station_code_array
    pop_weights_sorted = pop_weights[index_station_code_array]

    # split training and testing set; 4/5 for training,  1/5 for testing
    # index_city_final is the corresponding index of input tensors
    index_array = np.arange(n_station)
    np.random.seed(10)  # for replication
    np.random.shuffle(index_array)
    index_training = index_array[:np.int(n_station * (3 / 5))]
    index_validation = index_array[np.int(n_station * (3 / 5)):np.int(n_station * (4 / 5))]
    index_testing = index_array[np.int(n_station * (4 / 5)):]
    # use the index
    # input
    input_tensor_training = input_tensor[index_training, :, :, :]
    input_tensor_validation = input_tensor[index_validation, :, :, :]
    input_tensor_testing = input_tensor[index_testing, :, :, :]
    # output
    output_tensor_training = output_tensor[index_training, :]
    output_tensor_validation = output_tensor[index_validation, :]
    output_tensor_testing = output_tensor[index_testing, :]
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


def prepare_mean_value_data(input_data_dic, output_data_df, pop_weights, all_variables):
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
    n_channel = len(all_variables)
    # initialize input tensor (data_standard_dic or data_dic)
    input_tensor = np.zeros((n_station, n_channel))
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
    # sort pop weights by index
    pop_weights_sorted = pop_weights[index_station_code_array]

    # split training and testing set; 4/5 for training,  1/5 for testing
    # index_city_final is the corresponding index of input tensors
    index_array = np.arange(n_station)
    np.random.seed(10)  # for replication
    np.random.shuffle(index_array)
    index_training = index_array[:np.int(n_station * (3 / 5))]
    index_validation = index_array[np.int(n_station * (3 / 5)):np.int(n_station * (4 / 5))]
    index_testing = index_array[np.int(n_station * (4 / 5)):]
    # use the index to training and testing sets
    # input
    input_tensor_training = input_tensor[index_training, :]
    input_tensor_validation = input_tensor[index_validation, :]
    input_tensor_testing = input_tensor[index_testing, :]
    # output
    output_tensor_training = output_tensor[index_training, :]
    output_tensor_validation = output_tensor[index_validation, :]
    output_tensor_testing = output_tensor[index_testing, :]
    # names
    index_station_code_training = index_station_code_array[index_training]
    index_station_code_validation = index_station_code_array[index_validation]
    index_station_code_testing = index_station_code_array[index_testing]
    # weights
    weight_training = pop_weights_sorted.values[index_training][:, np.newaxis]
    weight_validation = pop_weights_sorted.values[index_validation][:, np.newaxis]
    weight_testing = pop_weights_sorted.values[index_testing][:, np.newaxis]
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
