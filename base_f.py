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


#shift the temporal dimension of the daily_input matrix
def shift_series_(s, shift_range,t_unit):

    s_shifts = [s.shift(-t_unit * shift, freq='D').rename(f'{s.name}_{shift}') for shift in range(*shift_range)]
    return pd.concat(s_shifts, axis=1)


#define the input-target matrix for the model (necessary to chain the past year meteo+snow variables while the target is just the last 30 days discharge average.
def create_it_matrix(daily_input, t_length,t_unit):

    # This function takes as input the daily temperature, precipitation and runoff and generates the input-target matrix

    # Read the daily input and extract runoff, evaporation, temperature and precipitation dataframe
    if isinstance(daily_input, str):
        daily_input = pd.read_csv(daily_input, index_col=0, parse_dates=True)
    runoff = daily_input[['Q']]
    temp = daily_input[[c for c in daily_input.columns if c[0] == 'T']]
    prec = daily_input[[c for c in daily_input.columns if c[0] == 'P']]
    evap = daily_input[[c for c in daily_input.columns if c[0] == 'E']]
    snow = daily_input[[c for c in daily_input.columns if c[0] == 'S']]
    run  = daily_input[[c for c in daily_input.columns if c[0] == 'R']]


    output = []
    # Compute the t_unit days average runoff
    runoff_t_unit = runoff.rolling(30, min_periods=30).mean()
    output.append(runoff_t_unit)
    
    
    # Compute the t_unit days average temperature for the last year.
    if not temp.empty:
        temp_t_unit = temp.rolling(t_unit, min_periods=t_unit).mean()
        temp_t_unit = pd.concat([shift_series_(temp_t_unit.loc[:, col], (-t_length + 1, 1),t_unit) for col in temp_t_unit], axis=1)
        output.append(temp_t_unit)

    # Compute the t_unit days average snow water equivalent for the last year.
    if not snow.empty:
        snow_t_unit = snow.rolling(t_unit, min_periods=t_unit).mean()
        snow_t_unit = pd.concat([shift_series_(snow_t_unit.loc[:, col], (-t_length + 1, 1),t_unit) for col in snow_t_unit], axis=1)
        output.append(snow_t_unit)

    # Compute the t_unit days sum precipitation for the last year.
    if not prec.empty:
        prec_t_unit = prec.rolling(t_unit, min_periods=t_unit).sum()
        prec_t_unit = pd.concat([shift_series_(prec_t_unit.loc[:, col], (-t_length + 1, 1),t_unit) for col in prec_t_unit], axis=1)
        output.append(prec_t_unit)
        
    # Compute the t_unit days sum evapotranspiration for the last year.
    if not evap.empty:
        evap_t_unit = evap.rolling(t_unit, min_periods=t_unit).sum()
        evap_t_unit = pd.concat([shift_series_(evap_t_unit.loc[:, col], (-t_length + 1, 1),t_unit) for col in evap_t_unit], axis=1)
        output.append(evap_t_unit)
    #pdb.set_trace()
    
    # Create the input-target matrix
    return pd.concat(output, axis=1).dropna()





"""
def monthly_climatology(daily_input,t_unit):

    if isinstance(daily_input, str):
        daily_input = pd.read_csv(daily_input, index_col=0, parse_dates=True)

    monthly_mean_columns = [c for c in daily_input.columns if c[0] in ['Q', 'T']]
    monthly_mean = daily_input.loc[:, monthly_mean_columns].groupby(by=daily_input.index.month).mean()
    #remember to add ['E', 'P']
    monthly_sum_columns = [c for c in daily_input.columns if c[0] in ['P','E']]
    monthly_sum = daily_input.loc[:, monthly_sum_columns].groupby(by=daily_input.index.month).sum()
    #pdb.set_trace()
    return pd.concat([monthly_mean, monthly_sum], axis=1)#[monthly_mean_columns,monthly_sum_columns]
"""

# compute the climatology (averaged with a t_unit moving average) for each variable in the daily_input
def daily_climatology(daily_input,t_unit):
    
    runoff = daily_input[['Q']]
    temp = daily_input[[c for c in daily_input.columns if c[0] == 'T']]
    prec = daily_input[[c for c in daily_input.columns if c[0] == 'P']]
    evap = daily_input[[c for c in daily_input.columns if c[0] == 'E']]
    snow = daily_input[[c for c in daily_input.columns if c[0] == 'S']]


    # Compute the t_unit days average runoff
    runoff_t_unit = runoff.rolling(30, min_periods=30).mean()

    
    # Compute the t_unit days average temperature
    if not temp.empty:
        temp_t_unit = temp.rolling(t_unit, min_periods=t_unit).mean()
        #temp_t_unit = pd.concat([shift_series_t_unitdays(temp_t_unit.loc[:, col], (-t_length + 1, 1)) for col in temp_t_unit], axis=1)


    # Compute the t_unit days average snow water equivalent
    if not snow.empty:
        snow_t_unit = snow.rolling(t_unit, min_periods=t_unit).mean()
        #temp_t_unit = pd.concat([shift_series_t_unitdays(temp_t_unit.loc[:, col], (-t_length + 1, 1)) for col in temp_t_unit], axis=1)


    # Compute the t_unit days sum precipitation
    if not prec.empty:
        prec_t_unit = prec.rolling(t_unit, min_periods=t_unit).sum()
        #prec_t_unit = pd.concat([shift_series_t_unitdays(prec_t_unit.loc[:, col], (-t_length + 1, 1)) for col in prec_t_unit], axis=1)
    
    # Compute the t_unit days sum evapotranspiration
    if not evap.empty:
        evap_t_unit = evap.rolling(t_unit, min_periods=t_unit).sum()
        #evap_t_unit = pd.concat([shift_series_t_unitdays(evap_t_unit.loc[:, col], (-t_length + 1, 1)) for col in evap_t_unit], axis=1)
        

    daily_t_unit = pd.concat([runoff_t_unit, temp_t_unit, prec_t_unit, evap_t_unit, snow_t_unit], axis=1)
    daily_mean = daily_t_unit.groupby(by=daily_t_unit.index.dayofyear).mean()

    #pdb.set_trace()
    return daily_mean



#   for creating a gap between train and test set (in ordert to mantain independence
#   the gap is obtained reducing the train set.
def create_gap(train_index,test_index,gap):
    right=((train_index+1 == test_index[0]).sum()==1) and ((train_index-1 == test_index[-1]).sum()==0)
    centre=((train_index+1 == test_index[0]).sum()==1) and ((train_index-1 == test_index[-1]).sum()==1)
    left = ((train_index+1 == test_index[0]).sum()==0) and ((train_index-1 == test_index[-1]).sum()==1)
    if right:
        train_index=train_index[0:-gap]

    if left:
        train_index=train_index[gap:]

    if centre:
        pos = np.where(train_index+1 == test_index[0])[0][0]
        train_index=np.concatenate((train_index[:pos-gap],train_index[pos+gap:]),axis=0)
    return train_index


# -----------------------------------------------------------------------

# Read and plot results


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

"""
def monthly_rmse(fileName, plot=True):

    # Open the result file
    runoff = pd.read_csv(fileName, index_col=0, parse_dates=True)

    # Create the ensamble mean
    for lt in range(1, 8):
        columns = [c for c in runoff.columns if 'sfTP_m' in c and f'lt{lt}' in c]
        runoff[f'sfTP_em_lt{lt}'] = runoff.loc[:, columns].mean(axis=1)

    # Compute the RMSE for each month
    runoff_error = []
    for m in range(1, 12):
        runoff_m = runoff.loc[runoff.index.month == m, :]
        runoff_error.append(
            runoff_m.apply(lambda y_pred: root_mean_squared_error(runoff_m['true_runoff'], y_pred), axis=0)
        )

    runoff_error = pd.DataFrame(data=runoff_error, index=range(1, 12))

    if plot:
        plt.figure()
        for lt in range(1, 8):
            runoff_error.loc[:, f'sfTP_em_lt{lt}'].plot(marker='o', label=f'lead_time={lt}')
        runoff_error.loc[:, 'runoff_clim'].plot(marker='o', color='black', label='climatology', linewidth=3)
        runoff_error.loc[:, 'trueTP'].plot(marker='o', color='red', label='era5', linewidth=3)
        plt.legend()
        plt.ylabel('RMSE ($m^3/s$)')
        plt.xlabel('Month')


def plot_it_matrix(daily_input, var, common_ylim=True):

    # ## Plot each variable of the input-target matrix

    # Create the input-target matrix from the daily_input
    it_matrix = create_it_matrix(daily_input, 1).rename(columns=lambda c: c[:2])

    # Create the ylabel dictionary
    plt_ylabel = {'P': 'Total precipitation (m)', 'T': 'Mean temperature (K)', 'Q': 'Runoff ($m^3/s$)'}

    # Set the ylim
    if common_ylim:
        selected_vars = it_matrix.loc[:, [c for c in it_matrix.columns if c[0] == var[0]]].values
        b = (selected_vars.max() - selected_vars.min()) * 0.03
        plt_ylim = (selected_vars.min()-b, selected_vars.max()+b)

    # Plot each year with a different color using the day of the year as x axis
    plt.figure()
    for y in range(it_matrix.index.year.min(), it_matrix.index.year.max() + 1):
        curr = it_matrix[it_matrix.index.year == y]
        curr.set_index(curr.index.dayofyear, inplace=True)
        curr[var].plot(label='_nolegend_')

    # Plot the daily climatology
    clim = it_matrix[var].groupby(it_matrix.index.dayofyear).mean().loc[1:365]
    clim.plot(label='Mean', color='black', linewidth=5)

    # Set the figure properties
    if common_ylim:
        plt.ylim(plt_ylim)
    plt.xlabel('Day')
    plt.ylabel(plt_ylabel[var[0]])
    plt.legend()


def lead_time_rmse(fileName):

    # fileName = '/home/mcallegari@eurac.edu/SECLI-FIRM/Mattia/SF_runoff/Zoccolo/Results/Learning_curve/Runoff_forecast_26_trainingyears.csv'

    # Open the result file
    runoff = pd.read_csv(fileName, index_col=0, parse_dates=True)

    # Create the ensamble mean
    for lt in range(1, 8):
        columns = [c for c in runoff.columns if 'sfTP_m' in c and f'lt{lt}' in c]
        runoff[f'sfTP_em_lt{lt}'] = runoff.loc[:, columns].mean(axis=1)

    # Compute the RMSE
    runoff_error = runoff.apply(lambda y_pred: root_mean_squared_error(runoff['true_runoff'], y_pred), axis=0)

    plt.figure()
    plt.plot([1, 7], [runoff_error['runoff_clim']]*2, color='black', label='runoff climatology')
    plt.plot([1, 7], [runoff_error['trueTP']] * 2, color='red', label='era5')
    plt.plot(range(1, 8), runoff_error[[f'climTP_lt{lt}' for lt in range(1, 8)]], marker='o', label='era5 climatology')
    plt.plot(range(1, 8), runoff_error[[f'sfTP_em_lt{lt}' for lt in range(1, 8)]], marker='o', label='SEAS5 ensamble mean')
    plt.plot(range(1, 8), runoff_error[[f'sfTP_m1_lt{lt}' for lt in range(1, 8)]], color='C1', alpha=0.5, label='SEAS5 members')
    for m in range(2, 26):
        plt.plot(range(1, 8), runoff_error[[f'sfTP_m{m}_lt{lt}' for lt in range(1, 8)]], color='C1', alpha=0.5)
    plt.legend()
    plt.xlabel('Lead time (months)')
    plt.ylabel('RMSE ($m^3/s$)')
"""    

def spatial_avg_daily_input(daily_input):
    t_columns = [c for c in daily_input.columns if c[0] =='T']
    daily_input['T'] = daily_input[t_columns].mean(axis=1)
    daily_input=daily_input.drop(columns = t_columns)

    e_columns = [c for c in daily_input.columns if c[0] =='E']
    daily_input['E'] = daily_input[e_columns].mean(axis=1)
    daily_input=daily_input.drop(columns = e_columns)

    p_columns = [c for c in daily_input.columns if c[0] =='P']
    daily_input['P'] = daily_input[p_columns].mean(axis=1)
    daily_input=daily_input.drop(columns = p_columns)
    
    s_columns = [c for c in daily_input.columns if c[0] =='S']
    daily_input['S'] = daily_input[s_columns].mean(axis=1)
    daily_input=daily_input.drop(columns = s_columns)
    
    r_columns = [c for c in daily_input.columns if c[0] =='R']
    daily_input['R'] = daily_input[r_columns].mean(axis=1)
    daily_input=daily_input.drop(columns = r_columns)
    
    return daily_input;


def spatial_stats_daily_input(daily_input):
    
    new_daily_input=pd.DataFrame(daily_input.Q)
    
    t_columns = [c for c in daily_input.columns if c[0] =='T']
    t_vars=daily_input[t_columns]
    new_daily_input.loc[:,'T']  = t_vars.mean(axis=1)
    new_daily_input.loc[:,'T5'] =t_vars.quantile(q=0.05,axis=1)
    new_daily_input.loc[:,'T25']=t_vars.quantile(q=0.25,axis=1)
    new_daily_input.loc[:,'T75']=t_vars.quantile(q=0.75,axis=1)
    new_daily_input.insert(loc=5,column='T95',value=t_vars.quantile(q=0.95,axis=1))
    
    
    s_columns = [c for c in daily_input.columns if c[0] =='S']
    s_vars=daily_input[s_columns]
    new_daily_input.loc[:,'S']  = s_vars.mean(axis=1)
    new_daily_input.loc[:,'S5'] =s_vars.quantile(q=0.05,axis=1)
    new_daily_input.loc[:,'S25']=s_vars.quantile(q=0.25,axis=1)
    new_daily_input.loc[:,'S75']=s_vars.quantile(q=0.75,axis=1)
    new_daily_input.loc[:,'S95']=s_vars.quantile(q=0.95,axis=1)
    
    p_columns = [c for c in daily_input.columns if c[0] =='P']
    p_vars=daily_input[p_columns]
    new_daily_input.loc[:,'P']  = p_vars.mean(axis=1)
    new_daily_input.loc[:,'P5'] =p_vars.quantile(q=0.05,axis=1)
    new_daily_input.loc[:,'P25']=p_vars.quantile(q=0.25,axis=1)
    new_daily_input.loc[:,'P75']=p_vars.quantile(q=0.75,axis=1)
    new_daily_input.loc[:,'P95']=p_vars.quantile(q=0.95,axis=1)

    e_columns = [c for c in daily_input.columns if c[0] =='E']
    e_vars=daily_input[e_columns]
    new_daily_input.loc[:,'E']  = e_vars.mean(axis=1)
    new_daily_input.loc[:,'E5'] =e_vars.quantile(q=0.05,axis=1)
    new_daily_input.loc[:,'E25']=e_vars.quantile(q=0.25,axis=1)
    new_daily_input.loc[:,'E75']=e_vars.quantile(q=0.75,axis=1)
    new_daily_input.loc[:,'E95']=e_vars.quantile(q=0.95,axis=1)
    """
    r_columns = [c for c in daily_input.columns if c[0] =='R']
    r_vars=daily_input[r_columns]
    new_daily_input.loc[:,'R'] = r_vars.mean(axis=1)
    new_daily_input.loc[:,'R5']= r_vars.quantile(q=0.05,axis=1)
    new_daily_input.loc[:,'R25']=r_vars.quantile(q=0.25,axis=1)
    new_daily_input.loc[:,'R75']=r_vars.quantile(q=0.75,axis=1)
    new_daily_input.loc[:,'R95']=r_vars.quantile(q=0.95,axis=1)
    """
    return new_daily_input


def create_gap(train_index,test_index,gap):
    right=((train_index+1 == test_index[0]).sum()==1) and ((train_index-1 == test_index[-1]).sum()==0)
    centre=((train_index+1 == test_index[0]).sum()==1) and ((train_index-1 == test_index[-1]).sum()==1)
    left = ((train_index+1 == test_index[0]).sum()==0) and ((train_index-1 == test_index[-1]).sum()==1)
    if right:
        train_index=train_index[0:-gap]

    if left:
        train_index=train_index[gap:]

    if centre:
        pos = np.where(train_index+1 == test_index[0])[0][0]
        train_index=np.concatenate((train_index[:pos-gap],train_index[pos+gap:]),axis=0)
        
    return train_index;


def compute_anomalies(climatologies,pred):
    
    #compute real climatology
    anomalies=pd.DataFrame(pred.true_runoff-pred.runoff_clim,columns=['true_runoff'])
    
    #get the climatology of prediction on the wanted days
    clim_on_test_dates = pd.DataFrame(climatologies.loc[anomalies.index.dayofyear])
    
    #create an array with the proper shape
    repeated_clim=np.repeat((np.array(clim_on_test_dates.prediction)[...,np.newaxis]),
                    pred.shape[1]-2,
                    axis=1)

    #subtract clim to predictions
    anomalies_pred= pred.iloc[:,2:]- repeated_clim

    #put together real and predicted anomalies
    anomalies=pd.concat([anomalies,anomalies_pred],axis=1)

    return anomalies
    
    
    
    

def create_gap(train_index,test_index,gap):
    right=((train_index+1 == test_index[0]).sum()==1) and ((train_index-1 == test_index[-1]).sum()==0)
    centre=((train_index+1 == test_index[0]).sum()==1) and ((train_index-1 == test_index[-1]).sum()==1)
    left = ((train_index+1 == test_index[0]).sum()==0) and ((train_index-1 == test_index[-1]).sum()==1)
    if right:
        train_index=train_index[0:-gap]

    if left:
        train_index=train_index[gap:]

    if centre:
        pos = np.where(train_index+1 == test_index[0])[0][0]
        train_index=np.concatenate((train_index[:pos-gap],train_index[pos+gap:]),axis=0)
    return train_index
    