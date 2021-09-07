import pandas as pd
import numpy as np

## Sn: sample number (based on the result of simulation, the result is almost convergent when setting the sn as 1,000)
sn = np.array(1000)

## Array
theta_db = np.zeros(2 * 27 * sn).reshape(2, 27, sn)
RR = np.zeros(2 * 27 * sn).reshape(2, 27, sn)
dy = np.zeros(27 * sn).reshape(27, sn)
dy_db = np.zeros(3).reshape(3)

## Data loading
data = np.load("./data.npy", allow_pickle=True)

## Total population in China
pop_cn = 1369045857.5093808

## Population in each age group(2010,2030)
## Data source：Li's code
pop_ag1 = data[:, 0]

## Basic mortality rate [IHD(25-30，30-35，35-40，...,75-80,80+), CEV(25-30，30-35，35-40，...,75-80,80+), COPD(25+), LC(25+), ALRI(25+)]
y0 = data[:, 1]

## distribution of parameter theta
## assume the distribution of theta is normal distribution, sample randomly from this distribution
## a set of "well selected" theta(2 senarios * 27 age & desease grouops * 100000 samples) is presented
## reading this set of theta is a little bit time consuming
theta_db[0] = data[:, 2:1002]
theta_db[1] = data[:, 1002:2002]

## GEMM paramaters
## Data source：B‘s paper SI
## [theta,SE,alpha,mu,nu][IHD(25-30，30-35，35-40，...,75-80,80+), CEV(25-30，30-35，35-40，...,75-80,80+), COPD(25+), LC(25+), ALRI(25+)]
prm = np.array([[0.5070, 0.02458, 1.9, 12, 40.2],
				[0.4762, 0.02309, 1.9, 12, 40.2],
				[0.4455, 0.02160, 1.9, 12, 40.2],
				[0.4148, 0.02011, 1.9, 12, 40.2],
				[0.3841, 0.01862, 1.9, 12, 40.2],
				[0.3533, 0.01713, 1.9, 12, 40.2],
				[0.3226, 0.01564, 1.9, 12, 40.2],
				[0.2919, 0.01415, 1.9, 12, 40.2],
				[0.2612, 0.01266, 1.9, 12, 40.2],
				[0.2304, 0.01117, 1.9, 12, 40.2],
				[0.1997, 0.00968, 1.9, 12, 40.2],
				[0.1536, 0.00745, 1.9, 12, 40.2],
				[0.4513, 0.11919, 6.2, 16.7, 23.7],
				[0.4240, 0.11197, 6.2, 16.7, 23.7],
				[0.3966, 0.10475, 6.2, 16.7, 23.7],
				[0.3693, 0.09752, 6.2, 16.7, 23.7],
				[0.3419, 0.09030, 6.2, 16.7, 23.7],
				[0.3146, 0.08307, 6.2, 16.7, 23.7],
				[0.2872, 0.07585, 6.2, 16.7, 23.7],
				[0.2598, 0.06863, 6.2, 16.7, 23.7],
				[0.2325, 0.06190, 6.2, 16.7, 23.7],
				[0.2051, 0.05418, 6.2, 16.7, 23.7],
				[0.1778, 0.04695, 6.2, 16.7, 23.7],
				[0.1368, 0.03611, 6.2, 16.7, 23.7],
				[0.2510, 0.06762, 6.5, 2.5, 32],
				[0.2942, 0.06147, 6.2, 9.3, 29.8],
				[0.4468, 0.11735, 6.4, 5.7, 8.4]])
theta = prm[0:27, 0]
SE = prm[0:27, 1]
alpha = prm[0:27, 2]
mu = prm[0:27, 3]
nu = prm[0:27, 4]
pm_cf = 2.4

## Constant
const = y0 * pop_ag1 / pop_cn


## Calculation of the change of premature death lead by exposure to pm2.5 (death at pm1- death at pm0)
## Input: pm0: innitial pm value, pm1: pm value under a policy, pop: population in this grid
def cal_premature_death(pm0, pm1, pop):
	## Calculation of the RR(hazard ratio) for 2 senarios, 27 age & desease groups and 100,000 sets of theta
	pm0 = max(0, (pm0 - pm_cf))
	pm1 = max(0, (pm1 - pm_cf))
	RR[0] = np.exp(theta_db[0] * ((np.log(pm0 / alpha + 1) / (1 + np.exp(-(pm0 - mu) / nu)))[:, None]))
	RR[1] = np.exp(theta_db[1] * ((np.log(pm1 / alpha + 1) / (1 + np.exp(-(pm1 - mu) / nu)))[:, None]))

	## Calculation of of the dy(Aviodance of premature death)
	dy = (const * pop)[:, None] * (1 / RR[1] - 1 / RR[0])

	##  Summation of the 27 age & desease groups
	dy = np.sum(dy, axis=0) * -1000

	## Calculation of dy's mean, 5% CI, 95 CI
	dy_mean = np.mean(dy)
	dy_5pct = np.percentile(dy, 5)
	dy_95pct = np.percentile(dy, 95)

	## Output: dy's mean, 5% CI, 95 CI
	dy_db = np.array([dy_mean, dy_5pct, dy_95pct])

	return dy_db


def cal_premature_death_vec(pm0,pm1,pop):

## Calculation of the RR(hazard ratio) for 2 scenarios, 27 age & disease groups and 100,000 sets of theta
	N = len(pm1)

	#RR0 = np.zeros(27 * sn).reshape(27, sn)
	#RR1 = np.zeros(N * 27 * sn).reshape(N, 27, sn)
	pm0 = np.maximum(0,(pm0-pm_cf))
	pm1 = np.maximum(0,(pm1-pm_cf))

	RR0 = 1/np.exp(theta_db[0]*((np.log(pm0/alpha+1)/(1+np.exp(-(pm0-mu)/nu)))[:,None]))
	RR1 = 1/np.exp(theta_db[1]*((np.log(pm1[:,None]/alpha+1)/(1+np.exp(-(pm1[:,None]-mu)/nu)))[:,:,None]))

	## Calculation of dy (Avoidance of premature death)
	dy = const[:, None] * (RR1-RR0)

	##	Summation of the 27 age & disease groups
	dy = -np.sum(dy, axis=1)

	## Calculation of dy's mean, 5% CI, 95 CI
	dy_mean = np.mean(dy,axis=1)
	dy_5pct = np.percentile(dy,5,axis=1)
	dy_95pct = np.percentile(dy,95,axis=1)

	## Output: dy's mean, 5% CI, 95 CI
	dy_db = np.array([dy_mean,dy_5pct,dy_95pct]) * pop * 1000
	return dy_db

if __name__ == "__main__":
	print(cal_premature_death(100, 200, 300))
	print(cal_premature_death(100, 143, 300))
	print(cal_premature_death(100, 166, 300))

	print(cal_premature_death_vec(100, np.array([200, 143,166]), 300))