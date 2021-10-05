output_var_names_all = ['pm25', 'pm10', 'so2', 'no2', 'co', 'o3', 'aqi']

# hyperparameter space
n_iterations_list = [2000]
n_mini_batch_list = [64]

conv_layer_number_list = [1, 2]
conv_filter_number_list = [20, 50, 80]
conv_kernel_size_list = [2, 3]
conv_stride_list = [2]
pool_size_list = [2]
pool_stride_list = [2]
drop_list = [True, False]
dropout_rate_list = [0.1]
bn_list = [True, False]
fc_layer_number_list = [1, 2]
n_fc_list = [200]
# augment data
augment_list = [True, False]
additional_image_size_list = [1]
epsilon_variance_list = [0.05, 0.1]

