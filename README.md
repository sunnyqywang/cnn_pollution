Code repository for the paper "Estimating air quality co-benefits of energy transition using machine learning".

## 00_plot_all_pollutants_radii.py

This script plots the model performance of the supplemental material (prediction of all pollutants and radii).

## 00_prune_hp_space.py

This script gathers information from all runs (no matter normalization method and variables) and plots a general trend of hyperparameters to narrow down the search space in 2_cnn_hyper_models.

## 1_preprocessing.py
This script preprocess the data from folder project2_energy_1812, containing the following modules:
1. Read stationIDs from stationID.xlsx
2. Read energy and weather data from raw/(result/weather) by station. All sectors except for INDP were taken in. RUC and UBC are combined to RDC and RUN and UBN are combined to RDN.
3. Data exported to ''/process/energy_data_dic_(version characteristics string).pickle'
 - data (dictionary): {station : {sector : 61x61 energy/weather information matrix}}
 - characteristics string indicates different versions of the data
4. Normalize matrix (image): Three normalization methods
 - norm: each entry is normalized to its z-score
 - minmax: normalized by the max value of each sector - all values are then between 0-1
 - const: normalized by a constant value - taken to be the magnitude of the maximum of each sector
5. Read pollution data from 'raw/air_quality_annual.xls'
 - get population weights from the column 'station_pop'
 - air_p has 1400+ stations
 - air_p_subset has 943 stations used for modelling
6. Split the dataset into train, validation, test (3:1:1)
 - stratified based on provincial region from 'raw/stationID_province_region.xlsx'
7. Export to '/process/data_process_dic_(version characteristics string)_(normalization method).pickle'
 - dictionary:

 |dataset names (dictionary keys)| dataset (dictionary values)|
 |--------|--------|
 |energy_(norm/const/minmax)_air|input standardized, output raw, train-test-split data dictionary|
 |energy_(mean/std/magnitude/min/max)_air|scale information for respective normalization methods|
 |energy_vars| a list of sector variables exported: ['AGC', 'AGN', 'AGO', 'INDCT', 'INDNT', 'INDOT', 'SVC', 'SVN', 'SVO', 'trans', 'RDC', 'RDN', 'DEM', 'rain', 'TEM']|
 |air_vars| a list of all air quality indicator names: ['pm25','pm10','so2','no2','co','o3','aqi']|

 - Train-test-split data dictionary represents a data dictionary of the following format:
 
 |Name|Description|
 |--------|--------|
 |input_(training/validation/testing)|tensor of the shape (station code, sector, 61, 61)|
 |output_(training/validation/testing)|output of the pollution metrics|
 |index_(training/validation/testing)|station codes corresponding to training sample|
 |weight_(training/validation/testing)|population corresponding to the stations|

## 2_cnn_hyper_models.py
This script builds the CNN and does hyperparameter searching.

Notes:
- The mean and standard deviation was scaled down by 10000 in order to make training easier. (since images are scaled to [0,1])
- Hyperparameter space (original, later shrinked see script for the shortened search space):
    - Number of iterations: n_iterations_list = [2000] (each iteration is a batch)
    - Number of mini batch: n_mini_batch_list = [20, 50, 100, 200]
    - Number of convolutional layers: conv_layer_number_list = [1,2,3,4,5]
    - Number of filters for convolutional layers: conv_filter_number_list=[20,50,80,100]
    - Size of convolution filters: conv_kernel_size_list=[2,3,4,5,6]
    - Stride of convolution filters: conv_stride_list=[1,2]
    - Pooling size: pool_size_list=[2,3,4,5,6]
    - Pooling stride: pool_stride_list=[1,2]
    - Whether to drop out: drop_list=[True,False]
    - Dropout rate: dropout_rate_list=[0.0,0.1,0.2,0.5]
    - Whether to do batch normalization: bn_list=[True,False]
    - Number of fully connected layers: fc_layer_number_list = [1,2,3]
    - Number of neurons in the fully connected layers: n_fc_list = [50,100,200]
    - Whether to augment data (horizontal flip, inversion, noise): augment_list = [True, False]
    - Number of images to augment: additional_image_size_list = [50,100,200]
    - Noise to add to the augmented image: epsilon_variance_list = [0.05,0.1,0.2]
- Output: For each model:
    - results_(output_var)\_(output_radius)\_(linear_coef)/model_output_hyper_searching_dic_(model_number).pickle
        - contains a tuple: (model_output, hyperparameters, hyperparameter names)
        - model_output is a dictionary
    
        |key|value|
        |---|---|
        |output_(training/validation/testing)|output of the model|
        |(training/validation/testing)\_loss_list|losses at each epoch (each time entire dataset is through)|

    - note some runs output_train will have more entries because of data augmentation, the first 565 is the original
    - file of the models: models/models_(output_var)\_(output_radius)\_(linear_coef)/model_(model_number)_(iteration_number).ckpt


## 3_combine_results.py
This script combines the results of all hyperparameter search runs in step 2 if the whole set is finished in multiple runs.

Output:
1. model_output_hyper_searching_dic_(output_var)\_(output_radius).pickle
 - a complete list results from all runs under the scope of the run
2. cnn_performance_table_(output_var)\_(output_radius).csv
 - validation and testing error for each run, sorted
3. hyperparameter_result_(output_var)\_(output_radius).csv
 - hyperparameters and validation results, used as input for 00_prune_hp_space when combined with results from other dates.

## 4_model_validity_(norm/minmax/const).py
This script evaluates the validity of the models by their gradients (unit: dPM / d 10k tons). Terminates when the model validation error is more than 50% worse than the best validation error.

Output: model_validity_(output_var)\_(output_radius).csv

For each model, the validation error, whether all gradients are positive, whether gradients of indct is smaller than other coals, mean of the gradients for each sector avg_{over all samples}(sum_{over 61x61 image}(dPM/dpixel))

## 5_linear_model_compare.py
This script runs linear regression on the dataset using the mean of each station, sector, as well as the flattened cell values. Tabulated metrics are: ['R', 'R2', 'RMSE', 'NRMSE', 'MB', 'NMB', 'ME', 'MAPE', 'MFB', 'MFE']

Output:
complete_performance_table_(output_var)\_(output_radius).csv:

Tabulated metrics for CNN and two linear regression models.

### There is no 6_ at the moment.

## 7_cell_gradients_scenarios_(norm/minmax/const).py
This script does scenario testing as well as calculating gradients (per pixel).

Outputs:
1. (model_number)_scenario_pm.xlsx

 - Three sheets (INDCT, COAL, OIL), from left to right, true PM, modelled PM, modelled PM with 2% - 20% (increments of 2%) reduction in emissions from all sectors.

2. (model_number)_marginal_avg_sensitivity.png
(supplemental) PM under different fractions of emissions

3. gradients_by_cell_(model_number).pkl
 - (tr/val/te)\_gradients: 61 x 61 x #sectors x #observations
 - (tr/val/te)\_gradients_mean: #sector x #observations
 - **unit: dPM/d(10k tons)**


4. results_(model_number).pkl
 - scenario_output: # scenario variables x # scenarios x # stations
 - gradients: gradients from the above pkl flattened, concatenated, and grouped by id10, taking maximum if one id10 belongs to multiple stations
 - gradients_masked: same with gradients except that the gradient is cleared to 0 if there was no emissions in the pixel in the raw dataset
 - gradients_full: concatenated gradients by the order of train, val, test, not grouped
 - mask_full_df: concatenated mask values (1 if pixel has emissions; 0 for 0 emisssions)
 - id10: id10 of gradients and mask_full_df
 - indices: list of statoin codes (string)
 - fitted_y_(train/validation/test)


5. gradients_(model_number).csv
Gradients of the individual pixels, whenever a pixel belongs to more than 1 station (so there will be multiple gradient values), the larger value is taken.

6. gradients_masked_(model_number).csv
Masked gradients of the individual pixels. Gradients of pixels where there are no emissions are cleared to 0.

## 8_premature_death.py

This script calculates the avoided premature death using functions in GEMM_function_20200328 and GEMM_function_mean_only_0328. The two functions for calculating premature death give the same unit as the population parameter (input). In this case, the unit is **10k people**.

Outputs:
1. (model_number)\_md_mean_full.csv
 - marginal damage measured in $, for each pixel, not grouped
 - marginal damage = avoided death (**10k people / 10k tons**) * 1.8 * 1e10 (**$/10k people**)
 - **unit: $/10k tons**
2. (model_number)\_md_mean.csv
 - marginal damage,
 - grouped by id10 (pixel) and took maximum (when a pixel contributes to more than one station, take the maximum of the marginal damage.)
3. (model_number)_scenario_ad_xlsx.xlsx
 - avoided death in different scenarios
 - **unit: 10k ppl**

4. (model_number)_scenario_ad_csv.csv
 - avoided death in different scenarios
 - columns = ['variable','scenario','index','p0','p1','pop','mean','p5','p95']

## 9_plot.py
1. Marginal damage histogram by sector
 - input: (model_number)\_md_mean.csv
 - process: divide by 10000 to get **$/ton**
 - output: (model_number)\_md_hist_(variables_name).png

2. Measured vs. Modelled scatterplot
 - input: (model_number)_results.pkl
 - output: (model_number)_modelled_measured.png
3. Scenario PM
 - output: (model_number)_scenarios_pm.png
4. Scenario avoided death
 - input: (model_number)_scenario_ad_csv.csv
 - output: (model_number)_scenarios_ad.png
5. Total damage by distance from source
 - input: (model_number)_md_mean_full.csv
 - process: divide marginal damage by 1e9 to get **$Billions/ 10k ton**, multiply by input emissions/10000 (**10k tons**) to get **$ Billions** in total damage
 - output: (model_number)_td_distance_from_source_by_sector.png
6. Total damage by sector
 - output: (model_number)_td_by_sector.png
7. Total damage by provincial regions
 - (model_number)_td_by_province.png

## util_cnn.py
CNN related functions

## util_performance.py
This script contains the code for all performance metrics.
