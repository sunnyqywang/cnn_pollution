import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.metrics import mean_squared_error
import util_cnn
import util_performance


### Graph. CNN prediction for all pollutants

output_folder = '200905'

output_vars = ['pm25', 'pm10', 'so2', 'no2', 'co', 'o3']
colors = ['skyblue','burlywood','yellowgreen','gold','lightseagreen','lightsalmon']
plot_vars = [0,1,2,3,4,5]
num_models = 5


char_name = '_no_airport_no_gas_coal_combined_oil_combined'
with open('../data/project2_energy_1812/process/full_data_process_dic'+char_name+'.pickle', 'rb') as data_standard:
    data_full_package = pkl.load(data_standard)
mean_data_name = 'energy_mean_air_nonstand'
weights_training = data_full_package[mean_data_name]['weight_training']/100
weights_validation = data_full_package[mean_data_name]['weight_validation']/100
weights_testing = data_full_package[mean_data_name]['weight_testing']/100

results = []
for air_var in plot_vars:
    # true output
    output_training = data_full_package[mean_data_name]['output_training'][:, air_var][:, np.newaxis]
    output_validation = data_full_package[mean_data_name]['output_validation'][:, air_var][:, np.newaxis]
    output_testing = data_full_package[mean_data_name]['output_testing'][:, air_var][:, np.newaxis]

    # model ensembling
    output_cnn_train, output_cnn_validation, output_cnn_test = util_cnn.get_ensembled_prediction(output_folder, air_var, 30, num_models, len(output_training))

    cnn_train_loss=mean_squared_error(output_training, output_cnn_train, sample_weight=weights_training)
    cnn_validation_loss=mean_squared_error(output_validation, output_cnn_validation, sample_weight=weights_validation)
    cnn_test_loss=mean_squared_error(output_testing, output_cnn_test, sample_weight=weights_testing)
    cnn_train_r2=util_performance.w_r2(output_cnn_train, output_training, weights_training)
    cnn_validation_r2=util_performance.w_r2(output_cnn_validation, output_validation, weights_validation)
    cnn_test_r2=util_performance.w_r2(output_cnn_test, output_testing, weights_testing)

    results.append([cnn_train_loss, cnn_validation_loss, cnn_test_loss, cnn_train_r2, cnn_validation_r2, cnn_test_r2])

results = pd.DataFrame(np.array(results), columns=['train_mse','validation_mse','test_mse','train_r2','validation_r2','test_r2'])

width = 0.7 / len(plot_vars)

fig, ax = plt.subplots()
for i in range(len(plot_vars)):
    ax.bar(np.arange(3)+(i-len(plot_vars)/2+0.5)*width, results.iloc[i][['train_r2', 'validation_r2','test_r2']], width, color=colors[i], label=output_vars[i])
ax.legend()
ax.set_xticks(np.arange(3))
ax.set_xticklabels(['Train','Validation','Test'])
ax.set_ylabel('Weighted R2')
fig.savefig("../output/"+output_folder+"/plots/all_pollutants_r2_"+str(num_models)+".png", bbox_inches='tight')

### Graph. CNN prediction for all image sizes
output_var = 0
image_radii = [10, 20, 30]

output_training = data_full_package[mean_data_name]['output_training'][:, output_var][:, np.newaxis]
output_validation = data_full_package[mean_data_name]['output_validation'][:, output_var][:, np.newaxis]
output_testing = data_full_package[mean_data_name]['output_testing'][:, output_var][:, np.newaxis]

results = []
for rad in image_radii:
    output_cnn_train, output_cnn_validation, output_cnn_test = util_cnn.get_ensembled_prediction(output_folder, output_var, rad, num_models, len(output_training))

    cnn_train_loss=mean_squared_error(output_training, output_cnn_train, sample_weight=weights_training)
    cnn_validation_loss=mean_squared_error(output_validation, output_cnn_validation, sample_weight=weights_validation)
    cnn_test_loss=mean_squared_error(output_testing, output_cnn_test, sample_weight=weights_testing)
    cnn_train_r2=util_performance.w_r2(output_cnn_train, output_training, weights_training)
    cnn_validation_r2=util_performance.w_r2(output_cnn_validation, output_validation, weights_validation)
    cnn_test_r2=util_performance.w_r2(output_cnn_test, output_testing, weights_testing)

    results.append([cnn_train_loss, cnn_validation_loss, cnn_test_loss, cnn_train_r2, cnn_validation_r2, cnn_test_r2])

results = pd.DataFrame(np.array(results), columns=['train_mse','validation_mse','test_mse','train_r2','validation_r2','test_r2'])

width = 0.7 / len(image_radii)

fig, ax = plt.subplots()
for i in range(len(image_radii)):
    ax.bar(np.arange(3)+(i-len(image_radii)/2+0.5)*width, results.iloc[i][['train_r2', 'validation_r2','test_r2']], width, color=colors[i], label='Size '+ str(2*image_radii[i]))
ax.legend()
ax.set_xticks(np.arange(3))
ax.set_xticklabels(['Train','Validation','Test'])
ax.set_ylabel('Weighted R2')
fig.savefig("../output/"+output_folder+"/plots/all_radii_r2_"+str(num_models)+".png", bbox_inches='tight')
