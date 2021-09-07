"""
Created on Thu Jan  3 11:28:39 2019

performance metrics
    This script incorporates all the performance metrics for continuous variables

@author: shenhao
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

### 1. unweighted performance metrics
# 1. R square
def pearson_coeff(y_pred, y_true):
    # this correlation coefficient is the same as R square when vars have only one dim
    p_coeff = pearsonr(y_pred.flatten(), y_true.flatten())
    return p_coeff[0]
# 2. Root mean squared error
def rmse(y_pred, y_true):
    # compute root mean squared error
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return rmse
def normalized_rmse(y_pred, y_true):
    n_rmse = rmse(y_pred, y_true)/(np.mean(y_pred)*np.mean(y_true))
    return n_rmse
# 3. bias
def mean_bias(y_pred, y_true):
    mb = np.mean(y_true) - np.mean(y_pred)
    return mb
def normalized_mean_bias(y_pred, y_true):
    # compute mean bias:
    # \sum (y_pred - y_true) / \sum(y_true)
    nmb = (np.sum(y_pred) - np.sum(y_true))/(np.sum(y_true))
    return nmb
# 4. errors
def mean_error(y_pred, y_true):
    me = np.mean(np.abs(y_true - y_pred))
    return me
def mean_absolute_percentage_error(y_pred, y_true):
    # https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    # compute mean error
    # \sum |y_pred - y_true| / \sum(y_true)
    mape = np.mean(np.abs(y_pred - y_true)/np.abs(y_true))
    return mape

def mfb(y_pred, y_true):
    mfb = np.mean(2 * np.divide((y_true-y_pred), y_true + y_pred))
    return mfb

def mfe(y_pred, y_true):
    mfe = np.mean(2 * np.divide(np.abs(y_true-y_pred), y_true + y_pred))
    return mfe
## test
#mean_absolute_percentage_error(y_pred_array, y_true_array)


### 2. weighted performance measures
def w_cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - np.average(x, weights=w)) * (y - np.average(y, weights=w))) / np.sum(w)
def w_pearson_coeff(y_pred, y_true, w):
    """Weighted Correlation"""
    return w_cov(y_pred, y_true, w) / np.sqrt(w_cov(y_pred, y_pred, w) * w_cov(y_true, y_true, w))
def w_rmse(y_pred, y_true, w):
    # compute root mean squared error
    w_rmse = sqrt(mean_squared_error(y_true, y_pred, sample_weight = w))
    return w_rmse
def w_normalized_rmse(y_pred, y_true, w):
    w_n_rmse = w_rmse(y_pred, y_true, w)/np.average(y_true, weights=w)
    return w_n_rmse
# 3. bias
def w_mean_bias(y_pred, y_true, w):
    w_mb = np.average(y_true, weights=w) - np.average(y_pred, weights=w)
    return w_mb
def w_normalized_mean_bias(y_pred, y_true, w):
    # compute mean bias:
    w_nmb = (np.average(y_pred, weights=w) - np.average(y_true, weights=w))/(np.average(y_true, weights=w))
    return w_nmb
# 4. errors
def w_mean_error(y_pred, y_true, w):
    w_me = np.average(np.abs(y_true - y_pred), weights=w)
    return w_me
def w_mean_absolute_percentage_error(y_pred, y_true, w):
    # https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    # compute mean error
    # \sum |y_pred - y_true| / \sum(y_true)
    w_mape = np.average(np.abs(y_pred - y_true)/np.abs(y_true), weights=w)
    return w_mape

def w_mfb(y_pred,y_true, w):
    w_mfb = np.average(2 * np.divide((y_true-y_pred), y_true + y_pred), weights=w)
    return w_mfb

def w_mfe(y_pred, y_true, w):
    w_mfe = np.average(2 * np.divide(np.abs(y_true-y_pred), y_true + y_pred), weights=w)
    return w_mfe

def w_r2(y_pred, y_true, w):
    return r2_score(y_true, y_pred, sample_weight=w)