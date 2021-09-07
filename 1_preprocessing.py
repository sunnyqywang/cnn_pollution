"""
pre process data

@author: shenhao/qingyi
"""

import pandas as pd
import numpy as np
from os import listdir
import pickle
import sys
from util_data import *
import math

date = '20200606'
char_name = '_no_airport_no_gas_no_SVO'
# 2020060601
variables_export = ['COAL', 'INDCT',
                    'INDOT', 'SVC', 'OIL',
                    'DEM', 'rain', 'TEM']

# 2020060602
variables_export = ['RDC', 'AGC', 'AGO', 'INDCT',
                    'INDOT', 'SVC', 'SVOtrans',
                    'DEM', 'rain', 'TEM']

# airport ID10
airport = pd.read_csv('../data/project2_energy_1812/raw/airport.csv')
airport = np.array(airport['ID10'])

###############################################################################
# map stationID
station_index = pd.read_excel('../data/project2_energy_1812/raw/stationID.xlsx')
## note:
#    STID is used in the energy folder as index.
#    station_code is used in air pollution file as index.
station_index['STID'] = [str(v) for v in station_index['STID']]
station_index['station_code'] = [str(v) for v in station_index['station_code']]

###############################################################################
### 0 read energy data into dictionary

energy_variables = ['AGC', 'AGN', 'AGO', 'INDCT', 'INDNT', 'INDOT',
                    'RUC', 'RUN', 'SVC', 'SVN', 'SVO', 'trans', 'UBC', 'UBN', 'RDC', 'RDN']
weather_variables = ['DEM', 'rain', 'TEM']

#pop_variables = ['POP']

# store the raw data
data = {}
airport_cells = []
print("Read raw energy data...")
for folder in listdir('../data/project2_energy_1812/raw/result'):
    if 'DS' not in folder and '._' not in folder: # address the DS_Store...
        folder_station_code = station_index.loc[station_index['STID'] == folder, 'station_code'].values[0]
        data[folder_station_code] = {}
        airport_cell=False
        for csv_file in listdir('../data/project2_energy_1812/raw/result/'+ folder):
            directory_path = '../data/project2_energy_1812/raw/result/'+folder+'/'+csv_file
            csv_file_ = csv_file[:-4]
            if csv_file_ == 'ID10':
                id10 = pd.read_csv(directory_path, header = None)
                if np.sum(np.sum(id10.isin(airport))) > 0:
                    x,y = np.where(np.array(id10.isin(airport)) == True)
                    airport_cell=True
            if csv_file_ in energy_variables: # add 15 energy variables
                data[folder_station_code][csv_file_] = pd.read_csv(directory_path, header = None).fillna(0)

        # Combine rural (RU) and urban (UB) gas (N) and coal (C) into residential (RD)
        data[folder_station_code]['RDC'] = data[folder_station_code]['RUC'] + data[folder_station_code]['UBC']
        data[folder_station_code]['RDN'] = data[folder_station_code]['RUN'] + data[folder_station_code]['UBN']
        # Combine RDC and AGC (20200111)
        data[folder_station_code]['COAL'] = data[folder_station_code]['RDC'] + data[folder_station_code]['AGC']
        # Combine SVO and trans (20200222)
        data[folder_station_code]['SVOtrans'] = data[folder_station_code]['SVO'] + data[folder_station_code]['trans']
        # Combine SVO AGO and trans (2020060601)
        data[folder_station_code]['OIL'] = data[folder_station_code]['SVO'] + data[folder_station_code]['trans']\
                                                    + data[folder_station_code]['AGO']
        # Clear airport cell transportation input
        if airport_cell:
            for i,j in zip(x,y):
                airport_cells.append(id10[j][i])
                data[folder_station_code]['trans'].loc[i,j] = 0
print("Cleared transportation inputs from ", len(set(airport_cells)), " airports.")

print("Read raw weather data...")
for folder in listdir('../data/project2_energy_1812/raw/weather'):
    if 'DS' not in folder and 'Rhistory' not in folder and '._' not in folder: # address the DS_Store and Rhistory...
        folder_station_code = station_index.loc[station_index['STID'] == folder, 'station_code'].values[0]
        for csv_file in listdir('../data/project2_energy_1812/raw/weather/'+ folder):
            directory_path = '../data/project2_energy_1812/raw/weather/'+folder+'/'+csv_file
            csv_file_ = csv_file[:-4]
            if csv_file_ in weather_variables: # add 3 weather variables
                data[folder_station_code][csv_file_] = pd.read_csv(directory_path, header = None).fillna(0)

with open('../data/project2_energy_1812/process/energy_data_dic'+char_name+'.pickle', 'wb') as data_dic:
    pickle.dump(data, data_dic, protocol=pickle.HIGHEST_PROTOCOL)

###############################################################################

### 1. obtain scale info for normalization
mean_scale_dic = {}
sd_scale_dic = {}
min_dic = {}
max_dic = {}
sector_max = {v : 0 for v in variables_export}
for key in data.keys():
    mean_scale_dic[key] = {}
    sd_scale_dic[key] = {}
    min_dic[key] = {}
    max_dic[key] = {}
    for var_ in variables_export:
        mean_scale_dic[key][var_] = np.nanmean(data[key][var_].values)
        sd_scale_dic[key][var_] = np.sqrt(np.nanvar(data[key][var_].values))
        min_dic[key][var_] = np.nanmin(data[key][var_].values)
        max_dic[key][var_] = np.nanmax(data[key][var_].values)
        sector_max[var_] = np.max([sector_max[var_], max_dic[key][var_]])

for v in variables_export:
    sector_max[v] = np.power( 10, int(math.log10(sector_max[v])))

## standardize each image
# 1. Use scale information to transform images
# 2. Fill in NaN as zeros.
print("Normalize energy data...")
data_standard_norm = {}
data_standard_minmax = {}
data_standard_const = {}
for key in data.keys():
    data_standard_norm[key] = {}
    data_standard_minmax[key] = {}
    data_standard_const[key] = {}
    for var_ in variables_export:
        # whiten images (standard normalization)
        data_standard_norm[key][var_] = (data[key][var_] - mean_scale_dic[key][var_])/sd_scale_dic[key][var_]
        # min-max normalization
        data_standard_minmax[key][var_] = (data[key][var_] - min_dic[key][var_]) / (max_dic[key][var_] - min_dic[key][var_])
        # a fixed scale
        data_standard_const[key][var_] = data[key][var_] / sector_max[var_]
        # fill in zeros
        data_standard_norm[key][var_].fillna(0, inplace = True)
        data_standard_minmax[key][var_].fillna(0, inplace=True)
        data_standard_const[key][var_].fillna(0, inplace=True)

# turn data, mean_scale_dic, sd_scale_dic to tensors
data_standard = data_standard_minmax
n_station = len(data_standard.keys())
image_height = 61
image_width = 61
n_channel = len(variables_export)
# initialize three tensors
energy_data_tensor = np.zeros((n_station, image_height, image_width, n_channel))
mean_tensor = np.zeros((n_station, n_channel))
sd_tensor = np.zeros((n_station, n_channel))
for station_key_id in range(len(list(data.keys()))):
    station_key = list(data.keys())[station_key_id]
    for j in range(len(variables_export)):
        var_ = variables_export[j]
        energy_data_tensor[station_key_id, :, :, j]=data[station_key][var_]
        mean_tensor[station_key_id,j]=mean_scale_dic[station_key][var_]
        sd_tensor[station_key_id,j]=sd_scale_dic[station_key][var_]

## obtain only a mean list as an alternative comparison
# note: whiten images first, then get this mean list. 
data_mean_dic = {}
data_std_dic = {}
print("Create mean and std lists of energy data...")
for key in data.keys():
    data_mean_dic[key] = {}
    data_std_dic[key] = {}
    for var_ in variables_export:
        data_mean_dic[key][var_] = np.nanmean(data[key][var_])
        data_std_dic[key][var_] = np.nanstd(data[key][var_])

###############################################################################
## import air pollution data, clean and translate
air_p = pd.read_excel('../data/project2_energy_1812/raw/air_quality_annual.xls')
air_p.index = [str(v) for v in air_p['station_code']]
print("Number of stations in air_pollution dataset is: ", air_p.shape[0])
print("Number of stations in energy dataset is: ", len(data_standard.keys())) # only a subset: 943...
useful_vars = ['pm25', 'pm10', 'so2', 'no2', 'co', 'o3', 'aqi']
air_p_subset = air_p.loc[list(data_standard.keys()), useful_vars]
print("Fill in one NA value...")
air_p_subset.fillna(air_p_subset.mean(), inplace = True)
## normalize this air_p_subset
air_p_subset_standard = (air_p_subset - air_p_subset.mean())/np.sqrt(air_p_subset.var())
# obtain populaiton as weights
pop_weights = air_p.loc[list(data_standard.keys()),'station_pop']
# obtain northern city indicators
north_city_index = air_p.loc[list(data_standard.keys()),'city_clean_heating']
# export datasets
air_p.to_csv('../data/project2_energy_1812/process/air_pollution_raw.csv')
#air_p_subset.to_csv('../data/project2_energy_1812/process/air_pollution_processed.csv')
#air_p_subset_standard.to_csv('../data/project2_energy_1812/process/air_pollution_standard_processed.csv')

# split energy, energy mean, energy std, and air pollution datasets into three types: training, validation, and testing sets.
print("Split the datasets into training, validation, and testing...")

# split datasets
# minmax standardized data
data_minmax_training_validation_testing_y_nonstand = prepare_tensor_data(data_standard_minmax, air_p_subset, pop_weights, variables_export)
# standardized data
data_norm_training_validation_testing_y_nonstand = prepare_tensor_data(data_standard_norm, air_p_subset, pop_weights, variables_export)
# constant standardized data
data_const_training_validation_testing_y_nonstand = prepare_tensor_data(data_standard_const, air_p_subset, pop_weights, variables_export)
# scale information
data_mean_training_validation_testing_y_nonstand = prepare_mean_value_data(data_mean_dic, air_p_subset, pop_weights, variables_export)
data_std_training_validation_testing_y_nonstand = prepare_mean_value_data(data_std_dic, air_p_subset, pop_weights, variables_export)
data_min_training_validation_testing_y_nonstand = prepare_mean_value_data(min_dic, air_p_subset, pop_weights, variables_export)
data_max_training_validation_testing_y_nonstand = prepare_mean_value_data(max_dic, air_p_subset, pop_weights, variables_export)

full_data_process = {}
# change 20200315: two normalizations are included. in previous versions only energy_air_nonstand exists (standard normalization)
full_data_process['energy_minmax_air_nonstand'] = data_minmax_training_validation_testing_y_nonstand
full_data_process['energy_norm_air_nonstand'] = data_norm_training_validation_testing_y_nonstand
full_data_process['energy_const_air_nonstand'] = data_const_training_validation_testing_y_nonstand

full_data_process['energy_mean_air_nonstand'] = data_mean_training_validation_testing_y_nonstand
full_data_process['energy_std_air_nonstand'] = data_std_training_validation_testing_y_nonstand
full_data_process['energy_min_air_nonstand'] = data_min_training_validation_testing_y_nonstand
full_data_process['energy_max_air_nonstand'] = data_max_training_validation_testing_y_nonstand
full_data_process['energy_magnitude_air_nonstand'] = sector_max
full_data_process['energy_vars'] = variables_export
full_data_process['air_vars'] = ['pm25','pm10','so2','no2','co','o3','aqi']

full_data_process['energy_full_raw_data'] = energy_data_tensor
data_raw_training_validation_testing = prepare_tensor_data(data, air_p_subset, pop_weights, variables_export)
full_data_process['energy_raw_nonstand'] = data_raw_training_validation_testing
full_data_process['energy_full_mean_by_station_var'] = mean_tensor
full_data_process['energy_full_std_by_station_var'] = sd_tensor
full_data_process['energy_full_weights'] = pop_weights.values[np.newaxis, :]
full_data_process['north_city_index'] = north_city_index
                          
with open('../data/project2_energy_1812/process/full_data_process_dic'+char_name+'.pickle', 'wb') as full_data_process_dic:
    pickle.dump(full_data_process, full_data_process_dic, protocol=pickle.HIGHEST_PROTOCOL)
