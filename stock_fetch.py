#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:07:41 2023

@author: rishu
"""

import yfinance as yf
import pandas as pd
import numpy as np


def stock_fetch(stock_ticker="MSFT"):
    #print("ok")
    tkr  = yf.Ticker(stock_ticker)
    
    hist = tkr.history(period="1mo", interval="1h")
    
    #print(hist)
    
    #print(tkr.history_metadata)
    
    df = pd.DataFrame(hist)
    
    
    ##
    # Dataframe pre processing module
    # 1. Remove splits by converting the price as if  not split has happened
    # 2. How to manage divideds. Problem: The devidends appear just a couple of time each year.
    ##
    
    
    df = df.reset_index(drop=True)
    df = df.drop(['Dividends', 'Stock Splits'], axis=1)
    #df = df.transpose()
    df = df.reset_index(drop=True)
    #print(df)
    df.to_csv("./stock.csv")
    
    npy = df.to_numpy()
    
    #print(npy)
    
    np.save('./stock.npy', npy)
    
    print("Saved '{}' data to stock.csv and stock.npy for last 1 month with 1 hour interval".format(stock_ticker))



if __name__=="__main__":
    stock_ticker = "TSLA" # Microsoft ticker 
