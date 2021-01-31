# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:24:03 2020

@author: Maurits
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch.univariate import arch_model
import arch
import scipy as sc
# Data Sets:
sp          = '^GSPC.csv'
eu          = '^STOXX50E.csv'

### Constant Parameters and the data set you want: ###
FILE = eu
SCALING = 100

#----------------------------------------------------------------------------------------------------------------------

def main(y, model, distr, SCALING):
    
    y = y[1500:] #just for checking leverage effect structural breaks, usually not there
    split_date = y.size #just for checking leverage effect structural breaks, used to be 1500
    #random comment

    if model == 'garch':    
        am = arch_model(y,mean='Zero', dist = distr)
        res = am.fit(last_obs=split_date)
        forecasts = res.forecast(horizon=1, start = split_date)
        
        std_error = res.std_err
        std_error[0] = std_error[0]/SCALING**2 
        
        params = res.params
        params[0] = params[0]/SCALING**2
        
        likelihood = res.loglikelihood 
        aic_crit = 2*len(params) - 2*likelihood
        bic_crit = np.log(len(forecasts.variance)) * len(params) - 2*likelihood

    #random comment 2
    elif model == 'egarch':
        am = arch_model(y,mean='Zero', vol='EGARCH',p=1, o=1, q=1, dist = distr)
        res = am.fit(last_obs=split_date)
        forecasts = res.forecast(horizon=1, start = split_date)
        
        std_error = res.std_err
        std_error[0] = std_error[0]/SCALING**2     
        
        params = res.params
        params[0] = params[0]/SCALING**2
        
        likelihood = res.loglikelihood 
        aic_crit = 2*len(params) - 2*likelihood
        bic_crit = np.log(len(forecasts.variance)) * len(params) - 2*likelihood
       
        
    elif model == 'gjr-garch':
        am = arch_model(y,mean='Zero', vol='GARCH', p=1, o=1, q=1, dist = distr)
        res = am.fit(last_obs=split_date)
        forecasts = res.forecast(horizon=1, start = split_date)
        
        std_error = res.std_err
        std_error[0] = std_error[0]/SCALING**2         
        
        params = res.params
        params[0] = params[0]/SCALING**2
        

        likelihood = res.loglikelihood 
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
    
    
    
    return vf, params, aic_crit, bic_crit, likelihood,q

#-------------------------------------------------------------------------------------------------------------------


def dmtest(e1, e2):
    d = e1 - e2
    n = len(d)
    dmean = np.mean(d)
    s2d = 0
    for i in range(0,n):
        s2d = s2d + (d[i]-dmean)**2
    s2d = s2d*(1/(n-1))
    sd = np.sqrt(s2d)
    teststat = np.sqrt(n)*((dmean-0)/sd);
    return teststat

#----------------------------------------------------------------------------------------------------------------------

def forecast_stoxx(y):
    mdl1 = arch_model(y, vol = 'GARCH', p=1, q=1, dist = 't', rescale = False)
    estMdl1 = mdl1.fix([0, 0.07714, 0.10801, 0.85755, 6.7707])  #ask walter how he changed it xdd
    forecast1 = estMdl1.forecast(horizon=1, start=1500)
    vf_stoxx = np.square(((np.sqrt(forecast1.variance))/SCALING))
    data1 = pd.DataFrame({'vf_stoxx':[vf_stoxx]})
    return data1, vf_stoxx

def forecast_sp500(y):
    mdl2 = arch_model(y, vol = 'GARCH', p=1, q=1, dist = 't', rescale = False)
    estMdl2 = mdl2.fix([0, 0.0488, 0.1556, 0.8030, 7.45828])     #ask walter how he changed it xdd
    forecast2 = estMdl2.forecast(horizon=1, start=1500)
    vf_sp500 = np.square(((np.sqrt(forecast2.variance))/SCALING))
    data2 = pd.DataFrame({'vf_sp500':[vf_sp500]})
    return data2, vf_sp500


# Possible models: ‘GARCH’ (default), ‘ARCH’, ‘EGARCH’, ‘FIARCH’ and ‘HARCH’

# Possible distributions: Normal: ‘normal’, ‘gaussian’ (default)
# Students’s t: ‘t’, ‘studentst’
# Skewed Student’s t: ‘skewstudent’, ‘skewt’
# Generalized Error Distribution: ‘ged’, ‘generalized error”
    
###### Main Parameters ######



# Create data frame with dates
df             = pd.read_csv(FILE)
df['y']        = np.log(df['Adj Close']).diff()

y = SCALING*df['y'].dropna().to_numpy()

# if scaling is off:
#vf = np.square((np.sqrt(forecasts.variance)/1000))


# Volatility Models
garch               = 'garch'
e                   = 'egarch'
gjr                 = 'gjr-garch'

# Distributions:
normal              = 'gaussian'
student_t           = 't'
skew_student_t      = 'skewt'
ged                 = 'ged'





### Main Function: create a df of all the vfs ###

# GARCH model:
vf, garch_norm_params,\
 garch_norm_aic_crit, \
 garch_norm_bic_crit, \
 garch_norm_likelihood,_ = main(y, garch, normal, SCALING)
vf.columns = ['garch_norm']

vf['garch_t'], \
garch_t_params, \
garch_t_aic_crit, \
garch_t_bic_crit, \
garch_t_likelihood,_ = main(y, garch, student_t, SCALING)

vf['garch_skewt'], \
garch_skewt_params, \
garch_skewt_aic_crit, \
garch_skewt_bic_crit, \
garch_skewt_likelihood,_ = main(y, garch, skew_student_t, SCALING)



# EGARCH model:
vf['egarch_norm'], \
egarch_norm_params, \
egarch_norm_aic_crit, \
egarch_norm_bic_crit, \
egarch_norm_likelihood,_ = main(y, e, normal, SCALING)

vf['egarch_t'], \
egarch_t_params, \
egarch_t_aic_crit, \
egarch_t_bic_crit, \
egarch_t_likelihood,_ = main(y, e, student_t, SCALING)

vf['egarch_skewt'], \
egarch_skewt_params, \
egarch_skewt_aic_crit, \
egarch_skewt_bic_crit, \
egarch_skewt_likelihood,_ = main(y, e, skew_student_t, SCALING)



# GJR-GARCH model:
vf['gjr_norm'], \
gjr_norm_params, \
gjr_norm_aic_crit, \
gjr_norm_bic_crit, \
gjr_norm_likelihood,_ = main(y, gjr, normal, SCALING)

vf['gjr_t'], \
gjr_t_params, \
gjr_t_aic_crit, \
gjr_t_bic_crit, \
gjr_t_likelihood,_ = main(y, gjr, student_t, SCALING)

vf['gjr_skewt'], \
gjr_skewt_params, \
gjr_skewt_aic_crit, \
gjr_skewt_bic_crit, \
gjr_skewt_likelihood,_ = main(y, gjr, skew_student_t, SCALING)





# Date
vf['Date'] = df['Date'].shift(-1)
vf['Date'] = pd.to_datetime(vf['Date'])
vf.set_index('Date', inplace=True)

#%%

#garch
y = y/1000
ymean = np.mean(y)
eps = y
#scale all params to the right versions 

#%% make all log L vectors
G11vectorSPnormal = arch.univariate.Normal.loglikelihood('Normal', parameters=garch_norm_params, resids=eps[1500], sigma2=vf.garch_norm[1500:], individual=True)
G11vectorSPstudent = arch.univariate.StudentsT.loglikelihood('Standardized Student\'s t', parameters=np.array([garch_t_params['nu']]), resids=eps[1500], sigma2=vf.garch_t[1500:], individual=True)
G11vectorSPskewt = arch.univariate.SkewStudent.loglikelihood(arch.univariate.SkewStudent(), parameters=np.array([garch_skewt_params['nu'],garch_skewt_params['lambda']]), resids=eps[1500], sigma2=vf.garch_skewt[1500:], individual=True)

#egarch
GEvectorSPnormal = arch.univariate.Normal.loglikelihood('Normal', parameters=egarch_norm_params, resids=eps[1500], sigma2=vf.egarch_norm[1500:], individual=True)
GEvectorSPstudent = arch.univariate.StudentsT.loglikelihood('Standardized Student\'s t', parameters=np.array([egarch_t_params['nu']]), resids=eps[1500], sigma2=vf.egarch_t[1500:], individual=True)
GEvectorSPskewt = arch.univariate.SkewStudent.loglikelihood(arch.univariate.SkewStudent(), parameters=np.array([egarch_skewt_params['nu'],egarch_skewt_params['lambda']]), resids=eps[1500], sigma2=vf.egarch_skewt[1500:], individual=True)


##gjr
GJRvectorSPnormal = arch.univariate.Normal.loglikelihood('Normal', parameters=gjr_norm_params, resids=eps[1500], sigma2=vf.gjr_norm[1500:], individual=True)
GJRvectorSPstudent = arch.univariate.StudentsT.loglikelihood('Standardized Student\'s t', parameters=np.array([gjr_t_params['nu']]), resids=eps[1500], sigma2=vf.gjr_t[1500:], individual=True)
GJRvectorSPskewt = arch.univariate.SkewStudent.loglikelihood(arch.univariate.SkewStudent(), parameters=np.array([gjr_skewt_params['nu'],gjr_skewt_params['lambda']]), resids=eps[1500], sigma2=vf.gjr_skewt[1500:], individual=True)


#which vector 2 use
if FILE == eu:
    data1, vf_stoxx = forecast_stoxx(y)
    Baysian_vector = arch.univariate.StudentsT.loglikelihood('Standardized Student\'s t', parameters=np.array([7.45828]), resids=y[1500], sigma2=vf_stoxx[1500:], individual=True)
    Baysian_vector['Date'] = df['Date'].shift(-1)
    Baysian_vector['Date'] = pd.to_datetime(Baysian_vector['Date'])
    Baysian_vector.set_index('Date', inplace=True)
else:
    data1, vf_sp500 = forecast_sp500(y)
    Baysian_vector = arch.univariate.StudentsT.loglikelihood('Standardized Student\'s t', parameters=np.array([6.7707]), resids=y[1500], sigma2=vf_sp500[1500:], individual=True)
    Baysian_vector['Date'] = df['Date'].shift(-1)
    Baysian_vector['Date'] = pd.to_datetime(Baysian_vector['Date'])
    Baysian_vector.set_index('Date', inplace=True)
    

#%%
## some logl dm testing-main
#dm = []
##row one
#dm = np.append(dm,dmtest(G11vectorSPnormal, G11vectorSPstudent))
#dm = np.append(dm,dmtest(G11vectorSPnormal, G11vectorSPskewt))
#dm = np.append(dm,dmtest(G11vectorSPnormal, GEvectorSPnormal))
#dm = np.append(dm,dmtest(G11vectorSPnormal, GEvectorSPstudent))
#dm = np.append(dm,dmtest(G11vectorSPnormal, GEvectorSPskewt))
#dm = np.append(dm,dmtest(G11vectorSPnormal, GJRvectorSPnormal))
#dm = np.append(dm,dmtest(G11vectorSPnormal, GJRvectorSPstudent))
#dm = np.append(dm,dmtest(G11vectorSPnormal, GJRvectorSPskewt))
#
##row two
#dm = np.append(dm,dmtest(G11vectorSPstudent, G11vectorSPskewt))
#dm = np.append(dm,dmtest(G11vectorSPstudent, GEvectorSPnormal))
#dm = np.append(dm,dmtest(G11vectorSPstudent, GEvectorSPstudent))
#dm = np.append(dm,dmtest(G11vectorSPstudent, GEvectorSPskewt))
#dm = np.append(dm,dmtest(G11vectorSPstudent, GJRvectorSPnormal))
#dm = np.append(dm,dmtest(G11vectorSPstudent, GJRvectorSPstudent))
#dm = np.append(dm,dmtest(G11vectorSPstudent, GJRvectorSPskewt))
##
###row three
#dm = np.append(dm,dmtest(G11vectorSPskewt, GEvectorSPnormal))
#dm = np.append(dm,dmtest(G11vectorSPskewt, GEvectorSPstudent))
#dm = np.append(dm,dmtest(G11vectorSPskewt, GEvectorSPskewt))
#dm = np.append(dm,dmtest(G11vectorSPskewt, GJRvectorSPnormal))
#dm = np.append(dm,dmtest(G11vectorSPskewt, GJRvectorSPstudent))
#dm = np.append(dm,dmtest(G11vectorSPskewt, GJRvectorSPskewt))
##
###row four
#dm = np.append(dm,dmtest(GEvectorSPnormal, GEvectorSPstudent))
#dm = np.append(dm,dmtest(GEvectorSPnormal, GEvectorSPskewt))
#dm = np.append(dm,dmtest(GEvectorSPnormal, GJRvectorSPnormal))
#dm = np.append(dm,dmtest(GEvectorSPnormal, GJRvectorSPstudent))
#dm = np.append(dm,dmtest(GEvectorSPnormal, GJRvectorSPskewt))
##
###row five
#dm = np.append(dm,dmtest(GEvectorSPstudent, GEvectorSPskewt))
#dm = np.append(dm,dmtest(GEvectorSPstudent, GJRvectorSPnormal))
#dm = np.append(dm,dmtest(GEvectorSPstudent, GJRvectorSPstudent))
#dm = np.append(dm,dmtest(GEvectorSPstudent, GJRvectorSPskewt))
##
###row six
#dm = np.append(dm,dmtest(GEvectorSPskewt, GJRvectorSPnormal))
#dm = np.append(dm,dmtest(GEvectorSPskewt, GJRvectorSPstudent))
#dm = np.append(dm,dmtest(GEvectorSPskewt, GJRvectorSPskewt))
##
###row seven
#dm = np.append(dm,dmtest(GJRvectorSPnormal, GJRvectorSPstudent))
#dm = np.append(dm,dmtest(GJRvectorSPnormal, GJRvectorSPskewt))
##
###row eight
#dm = np.append(dm,dmtest(GJRvectorSPstudent, GJRvectorSPskewt))
##
#
#yeet = len(dm)
#
#yeet2 = 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1


#------------------------------------------------------------------------------------------------------------------
#logl dm testing for bayes
dm = []
dm = np.append(dm,dmtest(Baysian_vector['h.1'], G11vectorSPnormal))
dm = np.append(dm,dmtest(Baysian_vector['h.1'], G11vectorSPstudent))
dm = np.append(dm,dmtest(Baysian_vector['h.1'], G11vectorSPskewt))
dm = np.append(dm,dmtest(Baysian_vector['h.1'], GEvectorSPnormal))
dm = np.append(dm,dmtest(Baysian_vector['h.1'], GEvectorSPstudent))
dm = np.append(dm,dmtest(Baysian_vector['h.1'], GEvectorSPskewt))
dm = np.append(dm,dmtest(Baysian_vector['h.1'], GJRvectorSPnormal))
dm = np.append(dm,dmtest(Baysian_vector['h.1'], GJRvectorSPstudent))
dm = np.append(dm,dmtest(Baysian_vector['h.1'], GJRvectorSPskewt))

















