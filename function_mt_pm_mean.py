import pandas as pd    
import numpy as np

## Data loading
data = np.load("./data.npy",allow_pickle=True)

## Totak population in China
pop_cn = 1369045857.5093808

## Population in each age group(2010,2030)
## Data source£ºLi¡®s code
pop_ag1 = data[:,0]

## Basic mortality rate [IHD(25-30£¬30-35£¬35-40£¬...,75-80,80+), CEV(25-30£¬30-35£¬35-40£¬...,75-80,80+), COPD(25+), LC(25+), ALRI(25+)]
y0 = data[:,1]

## GEMM paramaters
## Data source£ºB¡®s paper SI
## [theta,SE,alpha,mu,nu][IHD(25-30£¬30-35£¬35-40£¬...,75-80,80+), CEV(25-30£¬30-35£¬35-40£¬...,75-80,80+), COPD(25+), LC(25+), ALRI(25+)]
prm = np.array([[0.5070 , 0.02458 , 1.9 , 12   , 40.2],
             [0.4762 , 0.02309 , 1.9 , 12   , 40.2],
             [0.4455 , 0.02160 , 1.9 , 12   , 40.2],
             [0.4148 , 0.02011 , 1.9 , 12   , 40.2],
             [0.3841 , 0.01862 , 1.9 , 12   , 40.2],
             [0.3533 , 0.01713 , 1.9 , 12   , 40.2],
             [0.3226 , 0.01564 , 1.9 , 12   , 40.2],
             [0.2919 , 0.01415 , 1.9 , 12   , 40.2],
             [0.2612 , 0.01266 , 1.9 , 12   , 40.2],
             [0.2304 , 0.01117 , 1.9 , 12   , 40.2],
             [0.1997 , 0.00968 , 1.9 , 12   , 40.2],
             [0.1536 , 0.00745 , 1.9 , 12   , 40.2],
             [0.4513 , 0.11919 , 6.2 , 16.7 , 23.7],
             [0.4240 , 0.11197 , 6.2 , 16.7 , 23.7],
             [0.3966 , 0.10475 , 6.2 , 16.7 , 23.7],
             [0.3693 , 0.09752 , 6.2 , 16.7 , 23.7],
             [0.3419 , 0.09030 , 6.2 , 16.7 , 23.7],
             [0.3146 , 0.08307 , 6.2 , 16.7 , 23.7],
             [0.2872 , 0.07585 , 6.2 , 16.7 , 23.7],
             [0.2598 , 0.06863 , 6.2 , 16.7 , 23.7],
             [0.2325 , 0.06190 , 6.2 , 16.7 , 23.7],
             [0.2051 , 0.05418 , 6.2 , 16.7 , 23.7],
             [0.1778 , 0.04695 , 6.2 , 16.7 , 23.7],
             [0.1368 , 0.03611 , 6.2 , 16.7 , 23.7],
             [0.2510 , 0.06762 , 6.5 , 2.5  , 32  ],
             [0.2942 , 0.06147 , 6.2 , 9.3  , 29.8],
             [0.4468 , 0.11735 , 6.4 , 5.7  , 8.4 ]])
theta = prm[0:27,0]
SE = prm[0:27,1]
alpha = prm[0:27,2]
mu = prm[0:27,3]
nu = prm[0:27,4]
pm_cf  = 2.4

## Constant
const = y0 * pop_ag1 / pop_cn

## Calculation of the aviodance of premature death lead by exposure to pm2.5
## Input: pm0: innitial pm value, pm1: pm value under a policy, pop: population in this grid  
def cal_mean(pm0,pm1,pop):
    
## Calculation of dy's mean(log-normal distribution¡¯s analytical solution)   
    pm0 = max(0,(pm0-pm_cf))
    pm1 = max(0,(pm1-pm_cf))
    para0 = (np.log(pm0/alpha+1)/(1+np.exp(-(pm0-mu)/nu))) * (-1)
    para1 = (np.log(pm1/alpha+1)/(1+np.exp(-(pm1-mu)/nu))) * (-1)
    thpa0 = theta * para0 
    thpa1 = theta * para1
    SEpa0 = SE * para0
    SEpa1 = SE * para1
    dy_mean = np.sum(const*pop*(np.exp(thpa1+np.power(SEpa1,2)/2)-np.exp(thpa0+np.power(SEpa0,2)/2)))*-1000

## Output: dy's mean
    dy_mean = np.array([dy_mean])
    return dy_mean