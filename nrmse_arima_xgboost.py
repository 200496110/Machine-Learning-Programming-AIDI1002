#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 12:41:53 2023

@author: rishu
"""

# finalize model and make a prediction for monthly births with xgboost
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from xgboost import XGBRegressor
import numpy as np
import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np



# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
 n_vars = 1 if type(data) is list else data.shape[1]
 df = DataFrame(data)
 cols = list()
 # input sequence (t-n, ... t-1)
 for i in range(n_in, 0, -1):
     cols.append(df.shift(i))
 # forecast sequence (t, t+1, ... t+n)
 for i in range(0, n_out):
     cols.append(df.shift(-i))
 # put it all together
 agg = concat(cols, axis=1)
 # drop rows with NaN values
 if dropnan:
     agg.dropna(inplace=True)
 return agg.values



def xgb_nrmse():
    series = read_csv('./stock.csv')
    X = series[['Open', 'High']]
    pred = X.tail(1)
    X = X.iloc[:-1,:]
    values = X.values
    # transform the time series data into supervised learning
    train = series_to_supervised(values, n_in=6)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(trainX, trainy)
    # construct an input for a new preduction
    row = values[-6:].flatten()
    row = np.append(row, pred['Open'])
    # make a one-step prediction
    yhat = model.predict(asarray([row]))
    #print('Input: %s, Predicted: %.3f\n\n' % (row, yhat[0]))
    #print('Actual: {}, Predicted: {}'.format(pred['High'].values[0], yhat[0]))
    result = pred['High'].values[0]
    mean_val = values[-6:, 1].mean()
    nrmse = abs(result-mean_val)/mean_val
    print("XGBOOST NRMSE:",nrmse)
    
    return nrmse




def arima_nrmse():
    series = read_csv('./stock.csv')
    y = series['High']
    act=y.values[-1]
    train = y.values[:-1]
    model = pm.auto_arima(train)
    pred_arima = model.predict(1)[0]
    mean_train_arima = train.mean()
    nrmse_arima = abs(pred_arima-mean_train_arima)/mean_train_arima
    print("ARIMA NRMSE: {}".format(nrmse_arima))
    return nrmse_arima








if __name__ == "__main__":
    nrmse_xgboost = xgb_nrmse()
    nrmse_arima = arima_nrmse()