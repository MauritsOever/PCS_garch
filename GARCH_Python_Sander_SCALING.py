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

# Constants
SCALING = 100

sp          = '^GSPC.csv'
eu          = '^STOXX50E.csv'

### Constant Parameters and the data set you want: ###
FILE = eu

# Import and handle data.
filename = FILE
df = pd.read_csv(filename, index_col = 0)
adjClose = df['Adj Close']
y = SCALING * np.diff(np.log(adjClose))


# Define the forcast function.
def forecast(y, model, distribution):
    if(model == 'garch'):
        mdl = arch_model(y, mean='Zero', lags=0, vol = 'GARCH', p=1, o=0, q=1, dist = distribution, rescale = None)
        estMdl = mdl.fit(last_obs = 1500, disp='off')
        vf = estMdl.forecast(horizon=1, start=1501)
        return estMdl, vf
    
    if(model == 'egarch'):
        mdl = arch_model(y, mean='Zero', lags=0, vol = 'EGARCH', p=1, o=1, q=1, dist = distribution, rescale = None)
        estMdl = mdl.fit(last_obs = 1500, disp='off')
        vf = estMdl.forecast(horizon=1, start=1501)
        return estMdl, vf
    
    if(model == 'gjrgarch'):
        mdl = arch_model(y, mean='Zero', lags=0, vol = 'GARCH', p=1, o=1, q=1, dist = distribution, rescale = None)
        estMdl = mdl.fit(last_obs = 1500, disp='off')
        vf = estMdl.forecast(horizon=1, start=1501)
        return estMdl, vf


# Execute the forecasts and put the forcasted variances and model parameters in dataframe: vfdf, pardf.
vfdf = pd.DataFrame();
pardf = pd.DataFrame();

estMdlGN, vf = forecast(y, 'garch', 'normal')
vfdf['garch_norm'] = vf.variance['h.1'] / SCALING**2
pardf = pardf.append(estMdlGN.params)

estMdlGT, vf = forecast(y, 'garch', 't')
vfdf['garch_t'] = vf.variance['h.1'] / SCALING**2
pardf = pardf.append(estMdlGT.params)

estMdlGST, vf = forecast(y, 'garch', 'skewt')
vfdf['garch_skewt'] = vf.variance['h.1'] / SCALING**2
pardf = pardf.append(estMdlGST.params)

estMdlEGN, vf = forecast(y, 'egarch', 'normal')
vfdf['egarch_norm'] = vf.variance['h.1'] / SCALING**2
pardf = pardf.append(estMdlEGN.params)

estMdlEGT, vf = forecast(y, 'egarch', 't')
vfdf['egarch_t'] = vf.variance['h.1'] / SCALING**2
pardf = pardf.append(estMdlEGT.params)

estMdlEGST, vf = forecast(y, 'egarch', 'skewt')
vfdf['egarch_skewt'] = vf.variance['h.1'] / SCALING**2
pardf = pardf.append(estMdlEGST.params)

estMdlGJRGN, vf = forecast(y, 'gjrgarch', 'normal')
vfdf['gjr-garch_norm'] = vf.variance['h.1'] / SCALING**2
pardf = pardf.append(estMdlGJRGN.params)

estMdlGJRGT, vf = forecast(y, 'gjrgarch', 't')
vfdf['gjr-garch_t'] = vf.variance['h.1'] / SCALING**2
pardf = pardf.append(estMdlGJRGT.params)

estMdlGJRGST, vf = forecast(y, 'gjrgarch', 'skewt')
vfdf['gjr-garch_skewt'] = vf.variance['h.1'] / SCALING**2
pardf = pardf.append(estMdlGJRGST.params)

pardf.index = ['garch_norm','garch_t','garch_skewt','egarch_norm','egarch_t', \
                              'egarch_skewt','gjr-garch_norm','gjr-garch_t','gjr-garch_skewt'];

# Export vfdf and pardf
vfdf.to_csv(r'vfdfsp500.csv')
pardf.to_csv(r'pardfsp500.csv')


print(estMdlGN.summary())
print(estMdlGT.summary())
print(estMdlGST.summary())
























