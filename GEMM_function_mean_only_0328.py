#!/usr/bin/env python
# coding: utf-8

# In[5]:


#get_ipython().run_line_magic('time', '')
import pandas as pd    
import numpy as np
 
## SN: sample times
sn = np.array(1000)

## Array
theta_db = np.zeros(2*12*sn).reshape(2,12,sn)
invRR = np.zeros(2*12*sn).reshape(2,12,sn)
dy = np.zeros(12*sn).reshape(12,sn)
dy_db = np.zeros(3).reshape(3)

## Data loading(GBDx, http://ghdx.healthdata.org, accessed Mar 20, 2020)
data = np.load("./data0328.npy",allow_pickle=True)

## Total population in China
pop_cn = 1369045857.5093808

## Baseline death for each age subgroup (y0 * pop_ag1 in the formal version)
## source: GBD
bsc_dth = data[:,0]

## GEMM paramaters NCD+LRI
## Data source：Burnett‘s paper SI
## [theta,SE,alpha,mu,nu][NCD+LRI(25-30，30-35，35-40，...,75-80,80+)]
prm = np.array([[0.1585 , 0.01477 , 1.6 , 15.5   , 36.8],
	            [0.1577 , 0.01470 , 1.6 , 15.5   , 36.8],
	            [0.1570 , 0.01463 , 1.6 , 15.5   , 36.8],
	            [0.1558 , 0.01450 , 1.6 , 15.5   , 36.8],
	            [0.1532 , 0.01425 , 1.6 , 15.5   , 36.8],
	            [0.1499 , 0.01394 , 1.6 , 15.5   , 36.8],
	            [0.1462 , 0.01361 , 1.6 , 15.5   , 36.8],
	            [0.1421 , 0.01325 , 1.6 , 15.5   , 36.8],
	            [0.1374 , 0.01284 , 1.6 , 15.5   , 36.8],
	            [0.1319 , 0.01234 , 1.6 , 15.5   , 36.8],
	            [0.1253 , 0.01174 , 1.6 , 15.5   , 36.8],
	            [0.1141 , 0.01071 , 1.6 , 15.5   , 36.8]
	            ])
theta = prm[0:12,0]
SE = prm[0:12,1]
alpha = prm[0:12,2]
mu = prm[0:12,3]
nu = prm[0:12,4]
pm_cf  = 2.4

const = bsc_dth / pop_cn


## Calculation of the aviodance of premature death lead by exposure to pm2.5
## Input: pm0: initial pm value, pm1: pm value under a policy, pop: population in this grid  
def cal_mean(pm0,pm1,pop):

## Calculation of the invRR(1/RR, inverse hazard ratio) for 2 scenarios, 12 age subgroups and 1,000 sets of theta
    pm0 = max(0,(pm0-pm_cf))
    pm1 = max(0,(pm1-pm_cf))
    para0 = (np.log(pm0/alpha+1)/(1+np.exp(-(pm0-mu)/nu))) * (-1)
    para1 = (np.log(pm1/alpha+1)/(1+np.exp(-(pm1-mu)/nu))) * (-1)
    
## Calculation of dy's mean(log-normal distribution’s analytical solution)   
    thpa0 = theta * para0 
    thpa1 = theta * para1
    SEpa0 = SE * para0
    SEpa1 = SE * para1
    dy_mean = np.sum(const*pop*(np.exp(thpa1+np.power(SEpa1,2)/2)-np.exp(thpa0+np.power(SEpa0,2)/2)))*-1
    
## Output: dy's mean
    dy_db = np.array([dy_mean])
    return dy_db

