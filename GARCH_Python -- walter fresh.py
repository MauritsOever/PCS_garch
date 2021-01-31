import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch.univariate import arch_model

# Data Sets:
sp          = '^GSPC.csv'
eu          = '^STOXX50E.csv'

### Constant Parameters and the data set you want: ###
FILE = eu
SCALING = 1000


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
    
    
    
    return vf, params, aic_crit, bic_crit, likelihood,q





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
vf['gjr-garch_norm'], \
gjr_norm_params, \
gjr_norm_aic_crit, \
gjr_norm_bic_crit, \
gjr_norm_likelihood,_ = main(y, gjr, normal, SCALING)

vf['gjr-garch_t'], \
gjr_t_params, \
gjr_t_aic_crit, \
gjr_t_bic_crit, \
gjr_t_likelihood,_ = main(y, gjr, student_t, SCALING)

vf['gjr-garch_skewt'], \
gjr_skewt_params, \
gjr_skewt_aic_crit, \
gjr_skewt_bic_crit, \
gjr_skewt_likelihood,_ = main(y, gjr, skew_student_t, SCALING)





# Date
vf['Date'] = df['Date'].shift(-1)
vf['Date'] = pd.to_datetime(vf['Date'])
vf.set_index('Date', inplace=True)


# Write to csv

# SP
#vf.to_csv('vf_sp.csv', index=True)
#
#garch_norm_params.to_csv('garch_norm_params_sp.csv',index=True)
#garch_t_params.to_csv('garch_t_params_sp.csv',index=True)
#garch_skewt_params.to_csv('garch_skewt_params_sp.csv',index=True)
#
#egarch_norm_params.to_csv('egarch_norm_params_sp.csv',index=True)
#egarch_t_params.to_csv('egarch_t_params_sp.csv',index=True)
#egarch_skewt_params.to_csv('egarch_skewt_params_sp.csv',index=True)
#
#gjr_norm_params.to_csv('gjr_norm_params_sp.csv',index=True)
#gjr_t_params.to_csv('gjr_t_params_sp.csv',index=True)
#gjr_skewt_params.to_csv('gjr_skewt_params_sp.csv',index=True)



# EU
#vf.to_csv('vf_eu.csv', index=True)
##
#garch_norm_params.to_csv('garch_norm_params_eu.csv',index=True)
#garch_t_params.to_csv('garch_t_params_eu.csv',index=True)
#garch_skewt_params.to_csv('garch_skewt_params_eu.csv',index=True)
#
#egarch_norm_params.to_csv('egarch_norm_params_eu.csv',index=True)
#egarch_t_params.to_csv('egarch_t_params_eu.csv',index=True)
#egarch_skewt_params.to_csv('egarch_skewt_params_eu.csv',index=True)
#
#gjr_norm_params.to_csv('gjr_norm_params_eu.csv',index=True)
#gjr_t_params.to_csv('gjr_t_params_eu.csv',index=True)
#gjr_skewt_params.to_csv('gjr_skewt_params_eu.csv',index=True)





### Calculate quantile per (skewed) student-t ###

# GARCH:
_,_,_,_,_, garch_q_t = main(y, garch, student_t, SCALING)
_,_,_,_,_, garch_q_skewt = main(y, garch, skew_student_t, SCALING)

# EGARCH:
_,_,_,_,_, egarch_q_t = main(y, e, student_t, SCALING)
_,_,_,_,_, egarch_q_skewt = main(y, e, skew_student_t, SCALING)

# GJR-GARCH:
_,_,_,_,_, gjr_q_t = main(y, gjr, student_t, SCALING)
_,_,_,_,_, gjr_q_skewt = main(y, gjr, skew_student_t, SCALING)


































