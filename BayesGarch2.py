import arch
import scipy as sc
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import pandas_datareader.data as web

from arch import arch_model
from arch.univariate import arch_model

filename1 = '^STOXX50E.csv'
filename2 = '^GSPC.csv'

scaling = 100

df1 = pd.read_csv(filename1, index_col = 0) #stoxx
df2 = pd.read_csv(filename2, index_col = 0) #sp500

adjClose1 = df1['Adj Close'] #stoxx
adjClose2 = df2['Adj Close'] #sp500

y1 = np.diff(np.log(adjClose1))*scaling  #stoxx
y2 = np.diff(np.log(adjClose2))*scaling  #sp500


def forecast_stoxx(y1):
    mdl1 = arch_model(y1, vol = 'GARCH', p=1, q=1, dist = 't', rescale = False)
    estMdl1 = mdl1.fix([0, 0.07714/100**2, 0.10801, 0.85755, 6.7707])
    forecast1 = estMdl1.forecast(horizon=1, start=1500)
    vf_stoxx = np.square(((np.sqrt(forecast1.variance))/scaling))
    data1 = pd.DataFrame({'vf_stoxx':[vf_stoxx]})
    return data1, vf_stoxx
    
data1, vf_stoxx = forecast_stoxx(y1)

def forecast_sp500(y2):
    mdl2 = arch_model(y2, vol = 'GARCH', p=1, q=1, dist = 't', rescale = False)
    estMdl2 = mdl2.fix([0, 0.0488/100**2, 0.1556, 0.8030, 7.45828])
    forecast2 = estMdl2.forecast(horizon=1, start=1500)
    vf_sp500 = np.square(((np.sqrt(forecast2.variance))/scaling))
    data2 = pd.DataFrame({'vf_sp500':[vf_sp500]})
    return data2, vf_sp500
    
data2, vf_sp500 = forecast_sp500(y2)

#MAE Bayesian Estimation
Scaling = 1
y1_u = np.diff(np.log(adjClose1))*Scaling  #stoxx
y2_u = np.diff(np.log(adjClose2))*Scaling #sp500

y1_u_sq = y1_u ** 2
y2_u_sq = y2_u ** 2

def mae_stoxx(vf_stoxx):
    mae_stoxx = np.subtract(vf_stoxx['h.1'][1500:2508], y1_u_sq[1500:2508])
    mae_stoxx = np.abs(mae_stoxx)
    mae_stoxx = np.mean(mae_stoxx)
    return mae_stoxx

mae_stoxx = mae_stoxx(vf_stoxx)

def mae_sp500(vf_stoxx):
    mae_sp500= np.subtract(vf_sp500['h.1'][1500:2516], y2_u_sq[1500:2516])
    mae_sp500 = np.abs(mae_sp500)
    mae_sp500 = np.mean(mae_sp500)
    return mae_sp500

mae_sp500= mae_sp500(vf_sp500)
































#----------------------------------------------------------------------------------------------------------------------
#do some mf log likelyhoods
G11vectorSPstudentBaysian = arch.univariate.StudentsT.loglikelihood('Standardized Student\'s t', parameters=np.array([6.7707]), resids=y2[1500], sigma2=vf_sp500[1500:], individual=True)
G11vectorEUstudentBaysian = arch.univariate.StudentsT.loglikelihood('Standardized Student\'s t', parameters=np.array([7.45828]), resids=y1[1500], sigma2=vf_stoxx[1500:], individual=True)

#y1 = stoxx
#y2 = SP

#test if bayesian is better
dm = []




















