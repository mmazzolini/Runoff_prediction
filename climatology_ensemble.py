import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import os

import pdb


def monthly_climatology(daily_input):

    if isinstance(daily_input, str):
        daily_input = pd.read_csv(daily_input, index_col=0, parse_dates=True)

    monthly_mean_columns = [c for c in daily_input.columns if c[0] in ['Q', 'T']]
    monthly_mean = daily_input.loc[:, monthly_mean_columns].groupby(by=daily_input.index.month).mean()
    #remember to add ['E', 'P']
    monthly_sum_columns = [c for c in daily_input.columns if c[0] in ['P','E']]
    monthly_sum = daily_input.loc[:, monthly_sum_columns].groupby(by=daily_input.index.month).mean() * 30
    #pdb.set_trace()
    return pd.concat([monthly_mean, monthly_sum], axis=1)#[monthly_mean_columns,monthly_sum_columns]



def daily_climatology_p_et_ensemble(daily_input,radius=2/3):

    # This function takes as input the daily temperature, precipitation and runoff and generates the input-target matrix
    runoff = daily_input[['Q']]
    temp = daily_input[[c for c in daily_input.columns if c[0] == 'T']]
    prec = daily_input[[c for c in daily_input.columns if c[0] == 'P']]
    evap = daily_input[[c for c in daily_input.columns if c[0] == 'E']]


    # Compute the 30 days average runoff
    runoff_30 = runoff.rolling(30, min_periods=30).mean()

    
    # Compute the 30 days average temperature
    if not temp.empty:
        temp_30 = temp.rolling(30, min_periods=30).mean()


    # Compute the 30 days sum evapotranspiration
    if not evap.empty:
        evap_30 = evap.rolling(30, min_periods=30).sum()

        
    # Compute the 30 days sum precipitation
    if not prec.empty:
        prec_30 = prec.rolling(30, min_periods=30).sum()

        
    daily_30 = pd.concat([runoff_30 ,prec_30, temp_30, evap_30], axis=1)
    daily = daily_30.groupby(by=daily_30.index.day_of_year).mean()
        
    prec_mean=prec_30.groupby(by=prec_30.index.day_of_year).mean()
    prec_q75=prec_30.groupby(by=prec_30.index.day_of_year).quantile(q=0.75)
    prec_q75=prec_q75.add_suffix('_Q75')
    prec_q25=prec_30.groupby(by=prec_30.index.day_of_year).quantile(q=0.25)
    prec_q25=prec_q25.add_suffix('_Q25')

    
    temp_q75=pd.DataFrame(data=None, columns=temp_30.columns)
    evap_q75=pd.DataFrame(data=None, columns=evap_30.columns)

    temp_q25=pd.DataFrame(data=None, columns=temp_30.columns)
    evap_q25=pd.DataFrame(data=None, columns=evap_30.columns)

    #get the range as 2/3 of the mean difference between the quantile and the mean precipitation over the year.
    
    range75= radius*(prec_q75.mean().mean()-prec_30).mean().mean()
    range25= radius*(-prec_q25.mean().mean()+prec_30).mean().mean()

    for i in range(1,367):

        #get dates where the dayofyear is ==1 for the 3 variables
        day_i=(prec_30.index.day_of_year==i)
        prec_30_i=prec_30[day_i]
        temp_30_i=temp_30[day_i]
        evap_30_i=evap_30[day_i]    

        #compute the mean and interesting quantiles for the precipitation
        prec_30_i_mean=prec_30_i.mean(axis=1)

        #prec_30_i_mean=prec_30_i.drop(columns=prec_30.columns[-4:]).mean(axis=1)
        prec_q75_i=prec_q75[prec_q75.index==i].mean(axis=1)
        prec_q25_i=prec_q25[prec_q25.index==i].mean(axis=1)

        #select the situations where the precipitation is similar (closer than a range) to the quantile
        situations75=(np.abs(np.array(prec_30_i_mean)-np.array(prec_q75_i))<range75)
        situations25=(np.abs(np.array(prec_30_i_mean)-np.array(prec_q25_i))<range25)

        #average the variables values happened int the selected situations and append it
        s=temp_30_i[situations75].mean()
        s.name=i
        temp_q75=temp_q75.append(s)

        s=evap_30_i[situations75].mean()
        s.name=i
        evap_q75=evap_q75.append(s)

        s=temp_30_i[situations25].mean()
        s.name=i
        temp_q25=temp_q25.append(s)

        s=evap_30_i[situations25].mean()
        s.name=i
        evap_q25=evap_q25.append(s)

    temp_q75=temp_q75.add_suffix('_Q75')
    evap_q75=evap_q75.add_suffix('_Q75')

    temp_q25=temp_q25.add_suffix('_Q25')
    evap_q25=evap_q25.add_suffix('_Q25')

    #pdb.set_trace()
    
    # Create the input-target matrix
    return pd.concat([daily,
                     temp_q75, prec_q75, evap_q75,
                     temp_q25, prec_q25, evap_q25,], axis=1).dropna()





def daily_climatology_p_ensemble(daily_input):
    runoff = daily_input[['Q']]
    temp = daily_input[[c for c in daily_input.columns if c[0] == 'T']]
    prec = daily_input[[c for c in daily_input.columns if c[0] == 'P']]
    evap = daily_input[[c for c in daily_input.columns if c[0] == 'E']]


    # Compute the 30 days average runoff
    runoff_30 = runoff.rolling(30, min_periods=30).mean()

    
    # Compute the 30 days average temperature
    if not temp.empty:
        temp_30 = temp.rolling(30, min_periods=30).mean()
        #temp_30 = pd.concat([shift_series_30days(temp_30.loc[:, col], (-t_length + 1, 1)) for col in temp_30], axis=1)


    # Compute the 30 days sum evapotranspiration
    if not evap.empty:
        evap_30 = evap.rolling(30, min_periods=30).sum()
        #evap_30 = pd.concat([shift_series_30days(evap_30.loc[:, col], (-t_length + 1, 1)) for col in evap_30], axis=1)
        
        
    # Compute the 30 days sum precipitation
    if not prec.empty:
        prec_30 = prec.rolling(30, min_periods=30).sum()
        #prec_30 = pd.concat([shift_series_30days(prec_30.loc[:, col], (-t_length + 1, 1)) for col in prec_30], axis=1)
    
    daily_30 = pd.concat([runoff_30, temp_30, evap_30], axis=1)
    daily = daily_30.groupby(by=daily_30.index.day_of_year).mean()
    

    prec_mean=prec_30.groupby(by=prec_30.index.day_of_year).mean()
    prec_q75 =prec_30.groupby(by=prec_30.index.day_of_year).quantile(q=0.75)
    prec_q75=prec_q75.add_suffix('_Q75')
    prec_q25 =prec_30.groupby(by=prec_30.index.day_of_year).quantile(q=0.25)
    prec_q25=prec_q25.add_suffix('_Q25')
    daily   = pd.concat([daily,prec_mean,prec_q75,prec_q25], axis=1)


    return daily