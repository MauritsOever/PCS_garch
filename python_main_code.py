# -*- coding: utf-8 -*-
""" 
Created on 10 jan 2019

Author: mauritsboy420

"""

#load in packages necessary for this garbage bullshit
import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pandas_datareader import DataReader
from arch.univariate import arch_model
#non-working pachages (why the fuck are you like this python)
#import pyflux as pf -- 

#first for the S&P, import that data xdd
os. chdir(r'C:\Users\Mauri\Desktop\VU\MINOR\Practical Case Study\python related')

#### def main function ####
def main(y, model, distr, SCALING):
    

    split_date = 1500
    

    if model == 'garch':    
        am = arch_model(y,mean='Zero', dist = distr)
        res = am.fit(last_obs=y[split_date])
        forecasts = res.forecast(horizon=1, start = split_date)
        
        std_error = res.std_err
        std_error[0] = std_error[0]/SCALING**2 
        
        params = res.params
        params[0] = params[0]/SCALING**2
        
        likelihood = -res.loglikelihood 
        aic_crit = 2*len(params) - 2*likelihood
        bic_crit = np.log(len(forecasts.variance)) * len(params) - 2*likelihood

        
    elif model == 'egarch':
        am = arch_model(y,mean='Zero', vol='EGARCH',p=1, o=1, q=1, dist = distr)
        res = am.fit(last_obs=y[split_date])
        forecasts = res.forecast(horizon=1, start = split_date)
        
        std_error = res.std_err
        std_error[0] = std_error[0]/SCALING**2     
        
        params = res.params
        params[0] = params[0]/SCALING**2
        
        likelihood = -res.loglikelihood 
        aic_crit = 2*len(params) - 2*likelihood
        bic_crit = np.log(len(forecasts.variance)) * len(params) - 2*likelihood
       
        
    elif model == 'gjr-garch':
        am = arch_model(y,mean='Zero', vol='GARCH', p=1, o=1, q=1, dist = distr)
        res = am.fit(last_obs=y[split_date])
        forecasts = res.forecast(horizon=1, start = split_date)
        
        std_error = res.std_err
        std_error[0] = std_error[0]/SCALING**2         
        
        params = res.params
        params[0] = params[0]/SCALING**2
        

        likelihood = -res.loglikelihood 
        aic_crit = 2*len(params) - 2*likelihood
        bic_crit = np.log(len(forecasts.variance)) * len(params) - 2*likelihood
        print(res.summary())




    if distr == 'gaussian':
        q = am.distribution.ppf([0.01, 0.05], None)

    elif distr == 't':
        q = am.distribution.ppf([0.01, 0.05], res.params[-1:])
        
    elif distr == 'skewt':
        q = am.distribution.ppf([0.01, 0.05], res.params[-2:])
        
        
        
    vf = forecasts.variance
    vf = np.square( (np.sqrt(forecasts.variance) / SCALING) )
    
    
    
    return vf, params, aic_crit, bic_crit, likelihood, q

# Volatility Models, naming shit
garch               = 'garch'
e                   = 'egarch'
gjr                 = 'gjr-garch'

# Distributions:
normal              = 'gaussian'
t                   = 't'
skewt               = 'skewt'
SCALING             = 1000

#dicking around with data to make it workable
dfsp = pd.read_csv("^GSPC.csv")
dfsp['Date'].astype(str) #make variable 'obs' strings
dfsp['Date'] = pd.to_datetime(dfsp['Date'])
dfsp = dfsp.rename(columns = {"Adj Close": "AdjClose"})
#change to ysp before running
dfsp['ysp'] = np.log(dfsp.AdjClose) - np.log(dfsp.AdjClose.shift(1))
del(dfsp['Open'], dfsp['High'], dfsp['Low'], dfsp['Close'], dfsp['AdjClose'], dfsp['Volume'])


#import data for EU as wel xdd
dfeu = pd.read_csv("^STOXX50E.csv")
dfeu['Date'].astype(str) #make variable 'obs' strings
dfeu['Date'] = pd.to_datetime(dfeu['Date'])
dfeu = dfeu.rename(columns = {"Adj Close": "AdjClose"})
dfeu['yeu'] = np.log(dfeu.AdjClose) - np.log(dfeu.AdjClose.shift(1))
del(dfeu['Open'], dfeu['High'], dfeu['Low'], dfeu['Close'], dfeu['AdjClose'], dfeu['Volume'])


#import volatility forecasts
vfsp = pd.read_csv("vf_sp.csv")
vfeu = pd.read_csv("vf_eu.csv")



