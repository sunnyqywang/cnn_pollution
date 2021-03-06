# -*- coding: utf-8 -*-
"""
Created on Tue Mar 3 22:32:32 2019

@author: qingyi
"""

import pickle as pkl
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import sys
from GEMM_function_mean_only_0328 import cal_mean
from setup import *
import util_data

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
output_folder = "211010"
top_models = [14]
hp_idx = 145
linear_coef = 0.1
run_suffix = '_linearfix'

plt.rcParams.update({'font.size': 20})

# indices of variables to modify in scenario testing
# industrial coal, transportation, residential coal
# INDICES NEED TO CHANGE IN THE UPDATED DATASET
#200222 and 200317
scenario_variables = [0,1,4]
scenario_name = ['res_col', 'ind_col', 'trn_oil']
colors = ['salmon','mediumseagreen','mediumslateblue']

variables_export = ['COAL', 'INDCT',
                    'INDOT', 'SVC', 'OIL',
                    'DEM', 'rain', 'TEM']

# require gradients to be positive: AGO, INDCT, SVC, SVO, TRANS, COAL (RDC+AGC)
major_variables = [0,1,2,3,4]
char_name = '_no_airport_no_gas_coal_combined_oil_combined'
standard = 'const'
# Input
station_prov = pd.read_excel(data_dir+'raw/stationID_province_region.xlsx')
(train_images,train_y,train_weights,
    validation_images,validation_y,validation_weights,
    test_images,test_y,test_weights,
    train_mean_images,validation_mean_images,test_mean_images,
    index_cnn_training,index_cnn_validation,index_cnn_testing,
    sector_max) = util_data.read_data(char_name, standard, radius=30, output_var=[0])

control_var_training, control_var_validation, control_var_testing, control_scale = \
    util_data.get_control_variables(filename='agriculture_variables_station.xlsx',
                                train_index=index_cnn_training,
                                validation_index=index_cnn_validation,
                                test_index=index_cnn_testing)
if standard == 'const':
    input_raw_training = train_images
    input_raw_validation = validation_images
    input_raw_testing = test_images

    for i,v in enumerate(variables_export):
        input_raw_training[:,:,:,i] = train_images[:,:,:,i]*sector_max[v]
        input_raw_validation[:,:,:,i] = validation_images[:,:,:,i]*sector_max[v]
        input_raw_testing[:,:,:,i] = test_images[:,:,:,i]*sector_max[v]

    input_raw_training = input_raw_training[:,:,:,np.array(major_variables)]
    input_raw_validation = input_raw_validation[:,:,:,np.array(major_variables)]
    input_raw_testing = input_raw_testing[:,:,:,np.array(major_variables)]
else:
    print("Go get raw values!")

# unit: tons
input_emissions =  np.squeeze(np.concatenate((input_raw_training, input_raw_validation, input_raw_testing)))

# a list of station codes (string)
indices = np.concatenate((index_cnn_training, index_cnn_validation, index_cnn_testing))
trainvalte = np.concatenate((np.zeros(len(index_cnn_training)), np.ones(len(index_cnn_validation)), np.zeros(len(index_cnn_testing))+2))
observed_y = np.squeeze(np.concatenate((train_y, validation_y, test_y)))
weights = np.squeeze(np.concatenate((train_weights, validation_weights, test_weights)))

for model_idx in top_models:
    print("Model: ", model_idx)

    with open(output_dir + output_folder + "/"+str(hp_idx)+"_"+str(linear_coef)+"_"+str(model_idx)+"_results"+run_suffix+".pkl", "rb") as f:
        scenario_output = np.array(pkl.load(f))[:, :, :, 0]
        pkl.load(f)  #gradients
        pkl.load(f) #gradients_masked
        pkl.load(f) # gradients_full
        mask_full_df = pkl.load(f)
        id10_full = pkl.load(f)
        indices = pkl.load(f)
        fitted_y_train = pkl.load(f)
        fitted_y_val = pkl.load(f)
        fitted_y_test = pkl.load(f)

    with open(output_dir + output_folder + "/"+str(hp_idx)+"_"+str(linear_coef)+"_"+str(model_idx)+"_gradients_by_cell"+run_suffix+".pkl", "rb") as f:
        tr_gradients = pkl.load(f)
        val_gradients = pkl.load(f)
        te_gradients = pkl.load(f)
        tr_gradients_mean = pkl.load(f)
        val_gradients_mean = pkl.load(f)
        te_gradients_mean = pkl.load(f)

    fitted_y = np.squeeze(np.concatenate((fitted_y_train, fitted_y_val, fitted_y_test)))


    ### ------------ Plot measured vs modelled ------------ ###

    fig, ax = plt.subplots()
    ax.scatter(train_y, fitted_y_train, color='black', label='Train', alpha = 0.5)
    ax.scatter(validation_y, fitted_y_val, color = 'blue', label='Validation', alpha = 0.5)
    ax.scatter(test_y, fitted_y_test, color = 'green', label='Test', alpha = 0.5)
    ax.set_xlabel('Observed PM2.5')
    ax.set_ylabel('Fitted PM2.5')
    ax.legend()
    fig.savefig(output_dir+output_folder+"/plots/"+str(hp_idx)+"_"+str(linear_coef)+"_"+str(model_idx)+"_modelled_measured.png", bbox_inches='tight')

    pd.DataFrame(np.array([indices,fitted_y,observed_y,trainvalte, fitted_y-observed_y]).T,
                 columns=['model_idx','modelled','measured','train/val/te','difference']).to_csv(output_dir+\
                output_folder+"/"+str(hp_idx)+"_"+str(linear_coef)+"_"+str(model_idx)+"_model_performance"+run_suffix+".csv",index=False)

    ### ------------ Plot scenario PM ------------ ###
    fig, ax = plt.subplots()
    pct = np.arange(0, 22, 2)
    print(scenario_output.shape)
    for s in range(np.shape(scenario_output)[0]):
        # starting from 1 because the first column is the actual value
        s_output = np.average(scenario_output[s,1:,:], axis=1, weights=weights)
        ax.plot(pct, s_output, label=scenario_name[s], color=colors[s])
    ax.set_xticks(pct)
    ax.set_xlabel('Pollution Reduction (%)')
    ax.set_ylabel('PM2.5')
    ax.legend()
    fig.savefig(output_dir+output_folder+"/plots/"+str(hp_idx)+"_"+str(linear_coef)+"_"+str(model_idx)+"_scenarios_pm.png", bbox_inches='tight')

    ### ------------ Plot scenario avoided death ------------ ###
    # columns: 'variable','scenario','station_index','p0','p1','pop','mean','p5','p95'
    fig, ax = plt.subplots()
    scenario_ad = pd.read_csv(output_dir + output_folder + '/'+str(hp_idx)+"_"+str(linear_coef)+"_"+str(model_idx)+'_scenario_ad_csv'+run_suffix+'.csv')
    scenario_ad = scenario_ad.groupby(['variable','scenario'], as_index=False).sum()[['variable','scenario','mean','p5','p95','pop']]
    scenario_ad = scenario_ad.sort_values(by=['variable','scenario'], ascending=True)
    for v in range(np.shape(scenario_output)[0]):
        ax.fill_between(pct, scenario_ad[scenario_ad['variable'] == v]['p5'], \
                        scenario_ad[scenario_ad['variable'] == v]['p95'], \
                        color=colors[v], alpha=0.5)
        ax.plot(pct, scenario_ad[scenario_ad['variable'] == v]['mean'], color = colors[v], label=scenario_name[v])
    ax.set_xticks(pct)
    ax.set_xlabel('Pollution Reduction (%)')
    ax.set_ylabel('Avoided Deaths (10k deaths)')
    ax.legend()
    fig.savefig(output_dir+output_folder+"/plots/"+str(hp_idx)+"_"+str(linear_coef)+"_"+str(model_idx)+"_scenarios_ad.png", bbox_inches='tight')


    # read marginal damage (full dataframe) (unit: $/10k ton)
    md_mean_full = pd.read_csv(output_dir+output_folder+"/"+str(hp_idx)+"_"+str(linear_coef)+"_"+str(model_idx)+"_md_mean_full"+run_suffix+".csv")
    # (summed by id10)
    md = pd.read_csv(output_dir+output_folder+"/"+str(hp_idx)+"_"+str(linear_coef)+"_"+str(model_idx)+"_md_mean"+run_suffix+".csv")
    md = md[((md['id10'] != 8888888)&(md['id10'] != 9999999))]

    # filter emissions from images to by id10 (unit: ton)
    emissions_by_cell = np.reshape(input_emissions, (-1, len(major_variables)))
    emissions_by_cell = pd.DataFrame(np.insert(emissions_by_cell, 0, md_mean_full['id10'].to_numpy(), axis=1),
                                     columns=['id10']+[variables_export[i] for i in major_variables])
    # already checked: mean and max and min are the same one id10 per sector has one emission value
    emissions_by_cell = emissions_by_cell.groupby('id10', as_index=False).mean()
    emissions_by_cell = emissions_by_cell[((emissions_by_cell['id10'] != 8888888)&(emissions_by_cell['id10'] != 9999999))]
    # it's also fine if these two cells are not taken out as they have 0 emissions for all sectors
    print(emissions_by_cell.sum())

    # weight marginal damage by emissions
    for i in major_variables:
        print("%s: %.2f" % (variables_export[i],
                            np.average(md[variables_export[i]]/10000, weights=emissions_by_cell[variables_export[i]])))

        '''
        # check the influence of extreme values
        # prints weighted average with 99% percentile of both emissions and marginal damage
        p = np.array((emissions_by_cell[variables_export[i]].to_numpy(), md[variables_export[i]].to_numpy()/10000)).T
        pct = np.percentile(p, 99, axis=0)
        print(np.average(p[((p[:,1]<pct[1])&(p[:,0]<pct[0])),1], weights = p[((p[:,1]<pct[1])&(p[:,0]<pct[0])),0]))
        '''
        '''
        # checking the relationship between emission magnitude and marginal damage magnitude
        # not correlated
        plt.figure()
        plt.scatter(p[((p[:,1]<pct[1])&(p[:,0]<pct[0])),0], p[((p[:,1]<pct[1])&(p[:,0]<pct[0])),1])
        plt.savefig('temp'+str(i)+'.png')
        '''
    print('ALL COAL: %.2f' % np.average(md[[variables_export[i] for i in [0,1,3]]].to_numpy().flatten()/10000,
                                 weights=emissions_by_cell[[variables_export[i] for i in [0,1,3]]].to_numpy().flatten()))
    print('ALL OIL: %.2f' % np.average(md[[variables_export[i] for i in [2,4]]].to_numpy().flatten()/10000,
                                 weights=emissions_by_cell[[variables_export[i] for i in [2,4]]].to_numpy().flatten()))
    print('ALL: %.2f' % np.average(md[[variables_export[i] for i in major_variables]].to_numpy().flatten()/10000,
                                 weights=emissions_by_cell[[variables_export[i] for i in major_variables]].to_numpy().flatten()))
    print('ALL STD: %.2f' % np.sqrt(np.cov(md[[variables_export[i] for i in major_variables]].to_numpy().flatten()/10000,
                    fweights=emissions_by_cell[[variables_export[i] for i in major_variables]].to_numpy(dtype=int).flatten())))

    ### ------------ Plot marginal damage histogram by sector ------------ ###
    for i in major_variables:
        fig, ax = plt.subplots(figsize=(10, 10))
        n, bins, patches = ax.hist(md[variables_export[i]]/10000, density=True)
        ax.set_xlabel('Marginal Damage ($/ton)')
        ax.set_ylabel('Density')
        fig.savefig(output_dir + output_folder + "/plots/" + str(hp_idx)+"_"+str(linear_coef)+"_"+str(model_idx) + "_md_hist_"+variables_export[i]+".png",
                    bbox_inches='tight')


    # take out id10 and change money unit
    # billion dollars / 10k ton emissions
    md_mean_full = md_mean_full.to_numpy()[:, 1:] / 1e9

    # calculate total damage
    # shape = (num stations, 61, 61, num major variables)
    # billion dollars
    td_total = np.multiply(np.reshape(md_mean_full, (-1,61,61,len(major_variables))), input_emissions/1e4)

    ### ------------ Plot damage by sector ------------ ###
    fig, ax = plt.subplots()
    td_sector_total = np.sum(td_total, axis=(0,1,2))
    ax.bar(np.arange(len(td_sector_total)), td_sector_total, tick_label = [variables_export[v] for v in major_variables])
    ax.set_xlabel('Sectors')
    ax.set_ylabel('Total Damage ($ Billions)')
    fig.savefig(output_dir+output_folder+"/plots/"+str(hp_idx)+"_"+str(linear_coef)+"_"+str(model_idx)+"_td_by_sector.png", bbox_inches='tight')

    ### ------------ Plot damage by provincial regions ------------ ###
    td_by_station = np.sum(td_total, axis=(1,2))
    td_total_by_station = np.sum(td_by_station, axis=1)
    td_by_station = pd.DataFrame(td_by_station, columns =[variables_export[v] for v in major_variables])
    td_by_station['Total'] = td_total_by_station
    td_by_station['station_code'] = indices
    td_by_station = pd.merge(td_by_station, station_prov, on='station_code')
    td_by_region = td_by_station.groupby('region', as_index=False).sum()

    region_code = {1:'JJJ', 2:'YRD', 3:'PRD', 4:'Other East', 5:'Other Central', 6:'West', 7:'Northeast'}
    fig, ax = plt.subplots(nrows=len(major_variables)+1, sharex=True, figsize=(10,15))
    plot_variables = ['Total'] + [variables_export[v] for v in major_variables]
    for i in range(len(plot_variables)):
        col = plot_variables[i]
        ax[i].bar(np.arange(len(td_by_region)), td_by_region[col], tick_label=[region_code[r] for r in td_by_region['region']])
        ax[i].set_ylabel(col + " ($ 1B)")
    plt.xticks(rotation = 30)
    fig.savefig(output_dir+output_folder+"/plots/"+str(hp_idx)+"_"+str(linear_coef)+"_"+str(model_idx)+"_td_by_province.png", bbox_inches='tight')

    ### ------------ Plot total damage by distance from source ------------ ###
    middle_pixel = 30
    pixel_step = 5
    cumulative_output = [] #np.zeros(len(np.arange(0, 31, 10))) + bias
    for dist in np.arange(0, 31, pixel_step):
        temp = []
        for sector in major_variables:
            lower_index = middle_pixel-dist
            upper_index = middle_pixel+dist+1

            temp.append(np.sum(td_total[:, lower_index:upper_index, lower_index:upper_index, sector]))
        cumulative_output.append(temp)
    cumulative_output = np.array(cumulative_output)
    cu = np.zeros(len(np.arange(0, 31, pixel_step)))

    fig, ax = plt.subplots()
    for sector in major_variables:
        ax.fill_between(np.arange(0, 31, pixel_step), cu, cu + cumulative_output[:, sector], label=variables_export[sector], alpha=0.75)
        cu = cu + cumulative_output[:, sector]
    ax.set_xlabel('Distance from station (km)')
    ax.set_ylabel('Total Damage ($ Billions)')
    ax.legend(bbox_to_anchor=(1,1))
    fig.savefig(output_dir+output_folder+"/plots/"+str(hp_idx)+"_"+str(linear_coef)+"_"+str(model_idx)+"_td_distance_from_source_by_sector.png", bbox_inches='tight')
