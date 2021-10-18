import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import os

import pdb


def monthly_climatology(daily_input,t_unit):

    if isinstance(daily_input, str):
        daily_input = pd.read_csv(daily_input, index_col=0, parse_dates=True)

    monthly_mean_columns = [c for c in daily_input.columns if c[0] in ['Q', 'T']]
    monthly_mean = daily_input.loc[:, monthly_mean_columns].groupby(by=daily_input.index.month).mean()
    #remember to add ['E', 'P']
    monthly_sum_columns = [c for c in daily_input.columns if c[0] in ['P','E']]
    monthly_sum = daily_input.loc[:, monthly_sum_columns].groupby(by=daily_input.index.month).mean() * t_unit
    #pdb.set_trace()
    return pd.concat([monthly_mean, monthly_sum], axis=1)#[monthly_mean_columns,monthly_sum_columns]



def daily_climatology_p_et_ensemble(daily_input,t_unit,radius=2/3):

    # This function takes as input the daily temperature, precipitation and runoff and generates the input-target matrix
    runoff = daily_input[['Q']]
    temp = daily_input[[c for c in daily_input.columns if c[0] == 'T']]
    prec = daily_input[[c for c in daily_input.columns if c[0] == 'P']]
    evap = daily_input[[c for c in daily_input.columns if c[0] == 'E']]


    # Compute the t_unit days average runoff
    runoff_t_unit = runoff.rolling(t_unit, min_periods=t_unit).mean()

    
    # Compute the t_unit days average temperature
    if not temp.empty:
        temp_t_unit = temp.rolling(t_unit, min_periods=t_unit).mean()


    # Compute the t_unit days sum evapotranspiration
    if not evap.empty:
        evap_t_unit = evap.rolling(t_unit, min_periods=t_unit).sum()

        
    # Compute the t_unit days sum precipitation
    if not prec.empty:
        prec_t_unit = prec.rolling(t_unit, min_periods=t_unit).sum()

        
    daily_t_unit = pd.concat([runoff_t_unit ,prec_t_unit, temp_t_unit, evap_t_unit], axis=1)
    daily = daily_t_unit.groupby(by=daily_t_unit.index.day_of_year).mean()
        
    prec_mean=prec_t_unit.groupby(by=prec_t_unit.index.day_of_year).mean()
    prec_q75=prec_t_unit.groupby(by=prec_t_unit.index.day_of_year).quantile(q=0.75)
    prec_q75=prec_q75.add_suffix('_Q75')
    prec_q25=prec_t_unit.groupby(by=prec_t_unit.index.day_of_year).quantile(q=0.25)
    prec_q25=prec_q25.add_suffix('_Q25')

    
    temp_q75=pd.DataFrame(data=None, columns=temp_t_unit.columns)
    evap_q75=pd.DataFrame(data=None, columns=evap_t_unit.columns)

    temp_q25=pd.DataFrame(data=None, columns=temp_t_unit.columns)
    evap_q25=pd.DataFrame(data=None, columns=evap_t_unit.columns)

    #get the range as 2/3 of the mean difference between the quantile and the mean precipitation over the year.
    
    range75= radius*(prec_q75.mean().mean()-prec_t_unit).mean().mean()
    range25= radius*(-prec_q25.mean().mean()+prec_t_unit).mean().mean()

    for i in range(1,367):

        #get dates where the dayofyear is ==1 for the 3 variables
        day_i=(prec_t_unit.index.day_of_year==i)
        prec_t_unit_i=prec_t_unit[day_i]
        temp_t_unit_i=temp_t_unit[day_i]
        evap_t_unit_i=evap_t_unit[day_i]    

        #compute the mean and interesting quantiles for the precipitation
        prec_t_unit_i_mean=prec_t_unit_i.mean(axis=1)

        #prec_t_unit_i_mean=prec_t_unit_i.drop(columns=prec_t_unit.columns[-4:]).mean(axis=1)
        prec_q75_i=prec_q75[prec_q75.index==i].mean(axis=1)
        prec_q25_i=prec_q25[prec_q25.index==i].mean(axis=1)

        #select the situations where the precipitation is similar (closer than a range) to the quantile
        situations75=(np.abs(np.array(prec_t_unit_i_mean)-np.array(prec_q75_i))<range75)
        situations25=(np.abs(np.array(prec_t_unit_i_mean)-np.array(prec_q25_i))<range25)

        #average the variables values happened int the selected situations and append it
        s=temp_t_unit_i[situations75].mean()
        s.name=i
        temp_q75=temp_q75.append(s)

        s=evap_t_unit_i[situations75].mean()
        s.name=i
        evap_q75=evap_q75.append(s)

        s=temp_t_unit_i[situations25].mean()
        s.name=i
        temp_q25=temp_q25.append(s)

        s=evap_t_unit_i[situations25].mean()
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





def daily_climatology_p_ensemble(daily_input, t_unit):
    runoff = daily_input[['Q']]
    temp = daily_input[[c for c in daily_input.columns if c[0] == 'T']]
    prec = daily_input[[c for c in daily_input.columns if c[0] == 'P']]
    evap = daily_input[[c for c in daily_input.columns if c[0] == 'E']]

    # Compute the t_unit days average runoff
    runoff_t_unit = runoff.rolling(t_unit, min_periods=t_unit).mean()
   
    # Compute the t_unit days average temperature
    if not temp.empty:
        temp_t_unit = temp.rolling(t_unit, min_periods=t_unit).mean()
        #temp_t_unit = pd.concat([shift_series_t_unitdays(temp_t_unit.loc[:, col], (-t_length + 1, 1)) for col in temp_t_unit], axis=1)

    # Compute the t_unit days sum evapotranspiration
    if not evap.empty:
        evap_t_unit = evap.rolling(t_unit, min_periods=t_unit).sum()
        #evap_t_unit = pd.concat([shift_series_t_unitdays(evap_t_unit.loc[:, col], (-t_length + 1, 1)) for col in evap_t_unit], axis=1)
      
    # Compute the t_unit days sum precipitation
    if not prec.empty:
        prec_t_unit = prec.rolling(t_unit, min_periods=t_unit).sum()
        #prec_t_unit = pd.concat([shift_series_t_unitdays(prec_t_unit.loc[:, col], (-t_length + 1, 1)) for col in prec_t_unit], axis=1)
    
    daily_t_unit = pd.concat([runoff_t_unit, temp_t_unit, evap_t_unit], axis=1)
    daily = daily_t_unit.groupby(by=daily_t_unit.index.day_of_year).mean()
    

    prec_mean=prec_t_unit.groupby(by=prec_t_unit.index.day_of_year).mean()
    prec_q75 =prec_t_unit.groupby(by=prec_t_unit.index.day_of_year).quantile(q=0.75)
    prec_q75=prec_q75.add_suffix('_Q75')
    prec_q25 =prec_t_unit.groupby(by=prec_t_unit.index.day_of_year).quantile(q=0.25)
    prec_q25=prec_q25.add_suffix('_Q25')
    daily   = pd.concat([daily,prec_mean,prec_q75,prec_q25], axis=1)


    return daily