#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 14:22:22 2020

@author: sander
"""

# Import packages.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import arch

# Import and handle data.
sp = '^GSPC.csv'
eu = '^STOXX50E.csv'


file2use = sp
df = pd.read_csv(file2use, index_col = 0)
adjClose = df['Adj Close']
y = np.diff(np.log(adjClose))


# Define the forcast function.
def forecast(y, model, distribution):
    if(model == 'garch'):
        mdl = arch_model(y, mean='Zero', lags=0, vol = 'GARCH', p=1, o=0, q=1, dist = distribution, rescale = False)
        estMdl = mdl.fit(last_obs = 1500, disp='off')
        vf = estMdl.forecast(horizon=1, start=1501)
        return estMdl, vf
    
    if(model == 'egarch'):
        mdl = arch_model(y, mean='Zero', lags=0, vol = 'EGARCH', p=1, o=1, q=1, dist = distribution, rescale = False)
        estMdl = mdl.fit(last_obs = 1500, disp='off')
        vf = estMdl.forecast(horizon=1, start=1501)
        return estMdl, vf
    
    if(model == 'gjrgarch'):
        mdl = arch_model(y, mean='Zero', lags=0, vol = 'GARCH', p=1, o=1, q=1, dist = distribution, rescale = False)
        estMdl = mdl.fit(last_obs = 1500, disp='off')
        vf = estMdl.forecast(horizon=1, start=1501)
        return estMdl, vf


# Execute the forecasts and put the forcasted variances and model parameters in dataframe: vfdf, pardf.
vfdf = pd.DataFrame();
pardf = pd.DataFrame();

estMdlGN, vf = forecast(y, 'garch', 'normal')
vfdf['garch_norm'] = vf.variance['h.1']
pardf = pardf.append(estMdlGN.params)

estMdlGT, vf = forecast(y, 'garch', 't')
vfdf['garch_t'] = vf.variance['h.1']
pardf = pardf.append(estMdlGT.params)

estMdlGST, vf = forecast(y, 'garch', 'skewt')
vfdf['garch_skewt'] = vf.variance['h.1']
pardf = pardf.append(estMdlGST.params)

estMdlEGN, vf = forecast(y, 'egarch', 'normal')
vfdf['egarch_norm'] = vf.variance['h.1']
pardf = pardf.append(estMdlEGN.params)

estMdlEGT, vf = forecast(y, 'egarch', 't')
vfdf['egarch_t'] = vf.variance['h.1']
pardf = pardf.append(estMdlEGT.params)

estMdlEGST, vf = forecast(y, 'egarch', 'skewt')
vfdf['egarch_skewt'] = vf.variance['h.1']
pardf = pardf.append(estMdlEGST.params)

estMdlGJRGN, vf = forecast(y, 'gjrgarch', 'normal')
vfdf['gjr-garch_norm'] = vf.variance['h.1']
pardf = pardf.append(estMdlGJRGN.params)

estMdlGJRGT, vf = forecast(y, 'gjrgarch', 't')
vfdf['gjr-garch_t'] = vf.variance['h.1']
pardf = pardf.append(estMdlGJRGT.params)

estMdlGJRGST, vf = forecast(y, 'gjrgarch', 'skewt')
vfdf['gjr-garch_skewt'] = vf.variance['h.1']
pardf = pardf.append(estMdlGJRGST.params)

pardf.index = ['garch_norm','garch_t','garch_skewt','egarch_norm','egarch_t', \
                              'egarch_skewt','gjr-garch_norm','gjr-garch_t','gjr-garch_skewt'];

# Export vfdf and pardf
#vfdf.to_csv(r'vfdfsp500.csv')
#pardf.to_csv(r'pardfsp500.csv')

               
               
               
##---------------------------------------------------------------------------------------------------------------------
y = y/1000
ymean = np.mean(y)
eps = y
#scale all params to the right versions 

#%% make all log L vectors
G11vectorSPnormal = arch.univariate.Normal.loglikelihood('Normal', parameters=pardf['garch_norm'], resids=eps[1500], sigma2=vf.garch_norm[1500:], individual=True)
#G11vectorSPstudent = arch.univariate.StudentsT.loglikelihood('Standardized Student\'s t', parameters=np.array([garch_t_params['nu']]), resids=eps[1500], sigma2=vf.garch_t[1500:], individual=True)
#G11vectorSPskewt = arch.univariate.SkewStudent.loglikelihood(arch.univariate.SkewStudent(), parameters=np.array([garch_skewt_params['nu'],garch_skewt_params['lambda']]), resids=eps[1500], sigma2=vf.garch_skewt[1500:], individual=True)
#
##egarch
#GEvectorSPnormal = arch.univariate.Normal.loglikelihood('Normal', parameters=egarch_norm_params, resids=eps[1500], sigma2=vf.egarch_norm[1500:], individual=True)
#GEvectorSPstudent = arch.univariate.StudentsT.loglikelihood('Standardized Student\'s t', parameters=np.array([egarch_t_params['nu']]), resids=eps[1500], sigma2=vf.egarch_t[1500:], individual=True)
#GEvectorSPskewt = arch.univariate.SkewStudent.loglikelihood(arch.univariate.SkewStudent(), parameters=np.array([egarch_skewt_params['nu'],egarch_skewt_params['lambda']]), resids=eps[1500], sigma2=vf.egarch_skewt[1500:], individual=True)
#
#
###gjr
#GJRvectorSPnormal = arch.univariate.Normal.loglikelihood('Normal', parameters=gjr_norm_params, resids=eps[1500], sigma2=vf.gjr_norm[1500:], individual=True)
#GJRvectorSPstudent = arch.univariate.StudentsT.loglikelihood('Standardized Student\'s t', parameters=np.array([gjr_t_params['nu']]), resids=eps[1500], sigma2=vf.gjr_t[1500:], individual=True)
#GJRvectorSPskewt = arch.univariate.SkewStudent.loglikelihood(arch.univariate.SkewStudent(), parameters=np.array([gjr_skewt_params['nu'],gjr_skewt_params['lambda']]), resids=eps[1500], sigma2=vf.gjr_skewt[1500:], individual=True)