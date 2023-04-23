#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 14:54:24 2023

@author: rishu
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This script is the demo of BHT-ARIMA algorithm
# References : "Block Hankel Tensor ARIMA for Multiple Short Time Series Forecasting"
# Here each row is time series, and this method specifically takes the advantage of the correlation in the 

# import libraries
import numpy as np

from BHT_ARIMA import BHTARIMA
from BHT_ARIMA.util.utility import get_index


def get_arima_nrmse():
   #prepare data
   # the data should be arranged as (ITEM, TIME) pattern
   # import traffic dataset
   ori_ts = np.load('./stock.npy').T
   #print("shape of data: {}".format(ori_ts.shape))
   #print("This dataset have {} series, and each serie have {} time step\n\n".format(
   #    ori_ts.shape[0], ori_ts.shape[1]
   #))

   # parameters setting
   ts = ori_ts[..., :-1] # training data, 
   label = ori_ts[..., -1] # label, take the last time step as label
   p = 8 # p-order
   d = 1 # d-order
   q = 1 # q-order
   taus = [5, 5] # MDT-rank
   Rs = [4, 5] # tucker decomposition ranks
   k =  10 # iterations
   tol = 0.001 # stop criterion
   Us_mode = 4 # orthogonality mode

   # Run program
   # result's shape: (ITEM, TIME+1) ** only one step forecasting **
   model = BHTARIMA(ts, p, d, q, taus, Rs, k, tol, verbose=0, Us_mode=Us_mode)
   result, _ = model.run()
   pred = result[..., -1]

   # print extracted forecasting result and evaluation indexes
   #print("forecast result(first 10 series):\n", pred[:10])

   #print("\nEvaluation index: \n{}".format(get_index(pred, label)))
   
   nrmse = get_index(pred, label)['nrmse']
   print("BHT_ARIMA NRMSE:",nrmse)
   return nrmse
   

if __name__ == "__main__":
    nrmse = get_arima_nrmse()