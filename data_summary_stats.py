import numpy as np
import util_data

char_name = '_no_airport_no_gas_coal_combined_oil_combined'
standard = 'const'
radius = 30
output_var = [0,1,2,3,4,5,6]

variables = ["Rural and Residential Coal", "Industrial Coal", "Industrial Oil", "Service Sector Coal",
             "Road and Transportation Oil", "Altitude (m)", "Temperature (0.1\degree C)", "Rain (0.1mm)",
             "Fertilizer Area ??",  "Livestock Area", "Poultry Area",
             "PM2.5", "PM10", "SO_2", "NO_2", "CO", "O^3"]

(train_images,train_y,train_weights,
    validation_images,validation_y,validation_weights,
    test_images,test_y,test_weights,
    train_mean_images,validation_mean_images,test_mean_images,
    index_cnn_training,index_cnn_validation,index_cnn_testing,
    sector_max) = util_data.read_data(char_name, standard, radius, output_var)

control_var_training, control_var_validation, control_var_testing, control_scale = \
    util_data.get_control_variables(filename='agriculture_variables_station.xlsx',
                                train_index=index_cnn_training,
                                validation_index=index_cnn_validation,
                                test_index=index_cnn_testing)

input_mean_train = np.mean(train_mean_images, axis=0)
input_mean_val = np.mean(validation_mean_images, axis=0)
input_mean_test = np.mean(test_mean_images, axis=0)

control_mean_train = np.mean(control_var_training, axis=0)
control_mean_val = np.mean(control_var_validation, axis=0)
control_mean_test = np.mean(control_var_testing, axis=0)

output_mean_train = np.mean(train_y, axis=0)
output_mean_val = np.mean(validation_y, axis=0)
output_mean_test = np.mean(test_y, axis=0)

mean_train = np.concatenate([input_mean_train, control_mean_train, output_mean_train])
mean_val = np.concatenate([input_mean_val, control_mean_val, output_mean_val])
mean_test = np.concatenate([input_mean_test, control_mean_test, output_mean_test])

input_std_train = np.std(train_mean_images, axis=0)
input_std_val = np.std(validation_mean_images, axis=0)
input_std_test = np.std(test_mean_images, axis=0)
input_std_all = np.std(np.concatenate([train_mean_images, validation_mean_images, test_mean_images]), axis=0)

control_std_train = np.std(control_var_training, axis=0)
control_std_val = np.std(control_var_validation, axis=0)
control_std_test = np.std(control_var_testing, axis=0)
control_std_all = np.std(np.concatenate([control_var_training, control_var_validation, control_var_testing]), axis=0)

output_std_train = np.std(train_y, axis=0)
output_std_val = np.std(validation_y, axis=0)
output_std_test = np.std(test_y, axis=0)
output_std_all = np.std(np.concatenate([train_y, validation_y, test_y]), axis=0)

std_train = np.concatenate([input_std_train, control_std_train, output_std_train])
std_val = np.concatenate([input_std_val, control_std_val, output_std_val])
std_test = np.concatenate([input_std_test, control_std_test, output_std_test])
std_all = np.concatenate([input_std_all, control_std_all, output_std_all])

for v,a,b,c,d,e,f,g in zip(variables, mean_train, mean_val, mean_test, std_all, std_train, std_val, std_test):
    print("%s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & \\\\" % (v,(a*3+b+c)/5,a,b,c,d,e,f,g))
    if (v == "Rain (0.1mm)") or (v == "GDP per capita"):
        print("\hline")