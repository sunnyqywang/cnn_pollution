# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 23:55:21 2020

@author: wangqi44
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

output_folders = ['191122','191206','191219','200111']
output_df = pd.DataFrame()
for f in output_folders:
    temp = pd.read_csv("../output/" + f + "/hyperparams_result.csv")
    output_df = pd.concat((output_df, temp))
    
mean = output_df.groupby('output_date', as_index=False).mean()[['output_date','val']]
mean.columns = ['output_date','mean']
output_df = pd.merge(output_df, mean, on='output_date')
output_df['val'] = output_df['val'] - output_df['mean']

hp_names = ['n_iterations', 'n_mini_batch', 'conv_layer_number',
       'conv_filter_number', 'conv_kernel_size', 'conv_stride',
       'pool_size', 'pool_stride', 'drop', 'dropout_rate', 'bn',
       'fc_layer_number', 'n_fc', 'augment', 'additional_image_size',
       'epsilon_variance']

for hp in hp_names:
    fig,ax = plt.subplots()
    ax.scatter(output_df[hp], output_df['val'])
    temp = output_df.groupby(hp, as_index=False).mean()
    ax.plot(temp[hp], temp['val'], linewidth=3)
    temp = output_df.groupby(hp, as_index=False).var()
    ax2 = ax.twinx()
    ax2.plot(temp[hp], temp['val'], color = 'orange', linewidth=3)
    ax.set_title(hp)
    