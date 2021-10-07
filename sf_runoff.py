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

def shift_series_30days(s, shift_range):

    s_shifts = [s.shift(-30 * shift, freq='D').rename(f'{s.name}_{shift}') for shift in range(*shift_range)]
    return pd.concat(s_shifts, axis=1)


def create_it_matrix(daily_input, t_length):

    # This function takes as input the daily temperature, precipitation and runoff and generates the input-target matrix

    # Read the daily input and extract runoff, evaporation, temperature and precipitation dataframe
    if isinstance(daily_input, str):
        daily_input = pd.read_csv(daily_input, index_col=0, parse_dates=True)
    runoff = daily_input[['Q']]
    temp = daily_input[[c for c in daily_input.columns if c[0] == 'T']]
    prec = daily_input[[c for c in daily_input.columns if c[0] == 'P']]
    evap = daily_input[[c for c in daily_input.columns if c[0] == 'E']]


    # Compute the 30 days average runoff
    runoff_30 = runoff.rolling(30, min_periods=30).mean()

    
    # Compute the 30 days average temperature
    if not temp.empty:
        temp_30 = temp.rolling(30, min_periods=30).mean()
        temp_30 = pd.concat([shift_series_30days(temp_30.loc[:, col], (-t_length + 1, 1)) for col in temp_30], axis=1)

    # Compute the 30 days sum precipitation
    if not prec.empty:
        prec_30 = prec.rolling(30, min_periods=30).sum()
        prec_30 = pd.concat([shift_series_30days(prec_30.loc[:, col], (-t_length + 1, 1)) for col in prec_30], axis=1)
    
    # Compute the 30 days sum evapotranspiration
    if not evap.empty:
        evap_30 = evap.rolling(30, min_periods=30).sum()
        evap_30 = pd.concat([shift_series_30days(evap_30.loc[:, col], (-t_length + 1, 1)) for col in evap_30], axis=1)


    # Create the input-target matrix
    return pd.concat([runoff_30, temp_30, prec_30, evap_30], axis=1).dropna()





def sf_input_matrix(input_matrix, seasonal_forecast, lead_time, member):

    X = input_matrix.copy()
    X = X.loc[X.index.is_month_end, :]

    for offset in range(0, -lead_time, -1):
        sf_dates = X.index + pd.offsets.MonthBegin(offset - 1)
        X_columns = [c for c in X.columns if int(c.split('_')[1]) == offset]
        sf_columns = [f'SF_{c.split("_")[0]}_lt{lead_time + offset}_m{member}' for c in X_columns]
        X.loc[:, X_columns] = seasonal_forecast.loc[sf_dates, sf_columns].values

    return X


def svr_gridSearch(daily_input, t_length, C_range=np.logspace(-3, 1, 5), epsilon_range=np.logspace(-5, 0, 5),
                   plot=False, n_splits=8):

    # svr_gridSearch run the grid search on C and epsilon svr parameters.
    
    # Read the input-target matrix 
    it_matrix = create_it_matrix(daily_input, t_length)
    X = it_matrix.drop(columns='Q')
    y = it_matrix['Q']
    
    # Set up the splits respecting of the time-series nature of the dataset

    tscv = TimeSeriesSplit(gap=30,n_splits=n_splits, test_size=365)
    tscv.split(X)
    
    #pdb.set_trace()
     
    # Set up the grid search parameters
    svr_estimator = SVR(kernel='rbf', gamma='scale', cache_size=1000)
    svr_estimator = make_pipeline(StandardScaler(),
                                  TransformedTargetRegressor(regressor=svr_estimator, transformer=StandardScaler()))
    parameters = {'transformedtargetregressor__regressor__C': C_range,
                  'transformedtargetregressor__regressor__epsilon': epsilon_range}
    svr_model = GridSearchCV(svr_estimator, cv=tscv, param_grid=parameters, n_jobs=-1, verbose=1, refit=True)


    # execute the grid search
    svr_model.fit(X, y)
    
    
    # Select the best C and epsilon
    best_C = svr_model.best_params_['transformedtargetregressor__regressor__C']
    best_epsilon = svr_model.best_params_['transformedtargetregressor__regressor__epsilon']
    print(f'For {t_length} months of data input: Best estimator: C={best_C}, epsilon={best_epsilon}')

    # Check if the best C (or epsion) is in the border of the grid
    if best_C == max(C_range) or best_C == min(C_range):
        print(f'Warning: best C found on the grid limit: C = {best_C}')
    if best_epsilon == max(epsilon_range) or best_epsilon == min(epsilon_range):
        print(f'Warning: best epsilon found on the grid limit: epsilon = {best_epsilon}')
    print()
    
    if plot:

        # Gridsearch plot
        plt.figure()
        scores = svr_model.cv_results_['mean_test_score'].reshape(
            len(svr_model.param_grid['transformedtargetregressor__regressor__C']),
            len(svr_model.param_grid['transformedtargetregressor__regressor__epsilon']))
        plt.imshow(scores)
        plt.colorbar()
        plt.xticks(np.arange(len(svr_model.param_grid['transformedtargetregressor__regressor__epsilon'])),
                   ['%.3f' % a for a in svr_model.param_grid['transformedtargetregressor__regressor__epsilon']], rotation=45)
        plt.yticks(np.arange(len(svr_model.param_grid['transformedtargetregressor__regressor__C'])),
                   ['%.3f' % a for a in svr_model.param_grid['transformedtargetregressor__regressor__C']])
        plt.xlabel('epsilon')
        plt.ylabel('C')
        plt.title(f'Heatmap for {t_length} months of input data')
        plt.tight_layout()

        # Scatterplot true vs. estimated training
        plt.figure()
        true_values = y
        est_values = svr_model.predict(X)
        xy = np.vstack([true_values, est_values])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = true_values[idx], est_values[idx], z[idx]
        plt.scatter(x, y, c=z)
        plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], color='black')
        plt.xlabel('True values')
        plt.ylabel('Estimated values')
        plt.title(f'Heatmap for {t_length} months of input data')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.draw()
    return {'C': best_C, 'epsilon': best_epsilon, 'score': svr_model.best_score_,
            'best_estimator': svr_model.best_estimator_}


def feature_sel(daily_input):

    # Find the best t_length and temperature/precipitation stations. Return score for each combination

    gridSearch_results = []
    for t_len in range(1, 12):
        gridSearch_results.append(svr_gridSearch(daily_input, 
                                                 t_len,
                                                 C_range=np.logspace(-2, 2, 6), 
                                                 epsilon_range=np.logspace(-3, 0, 6),
                                                 plot=True))

    return pd.DataFrame(data=gridSearch_results, index=range(1, 12))

    # if isinstance(daily_input, str):
    #     daily_input = pd.read_csv('/home/mcallegari@eurac.edu/SECLI-FIRM/Mattia/SF_runoff/Zoccolo/inputs/daily_input.csv', index_col=0, parse_dates=True)
    #
    # temp_col = [c for c in daily_input.columns if c[0] == 'T']
    # prec_col = [c for c in daily_input.columns if c[0] == 'P']
    #
    # for feat in itertools.product(temp_col, prec_col):
    #     for t_len in range(1, 12):
    #         svr_gridSearch(daily_input.loc[:, ('Q',) + feat], t_len, plot=False)


def training(daily_input, t_length=None, svr_C=None, svr_epsilon=None):

    # This function takes as input the daily temperature, precipitation and runoff and train a model to predict the
    # monthly mean runoff using a time series of temperature and precipitation of length t_length as input features

    # Execute the feature selection if t_length is None
    if t_length is None:

        print('Finding the best input time series length...')
        fs = feature_sel(daily_input)
        fs_best = fs.loc[fs['score'].idxmax()]
        svr_C, svr_epsilon = fs_best['C'], fs_best['epsilon']
        t_length = fs_best.name
        print(f'Best time series length: {t_length}. '
              f'Best C={svr_C}, epsilon={svr_epsilon} (score={fs_best["score"]})')
        svr_model = fs_best['best_estimator']

    else:

        it_matrix = create_it_matrix(daily_input, t_length)
        X = it_matrix.drop(columns='Q')
        y = it_matrix['Q']

        # Fit the SVR with standardized input and target
        svr_estimator = SVR(kernel='rbf', C=svr_C, epsilon=svr_epsilon, gamma='scale', cache_size=1000)
        svr_model = make_pipeline(StandardScaler(),
                                  TransformedTargetRegressor(regressor=svr_estimator, transformer=StandardScaler()))
        svr_model.fit(X, y)

    return svr_model


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

def daily_climatology(daily_input):
    
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

    # Compute the 30 days sum precipitation
    if not prec.empty:
        prec_30 = prec.rolling(30, min_periods=30).sum()
        #prec_30 = pd.concat([shift_series_30days(prec_30.loc[:, col], (-t_length + 1, 1)) for col in prec_30], axis=1)
    
    # Compute the 30 days sum evapotranspiration
    if not evap.empty:
        evap_30 = evap.rolling(30, min_periods=30).sum()
        #evap_30 = pd.concat([shift_series_30days(evap_30.loc[:, col], (-t_length + 1, 1)) for col in evap_30], axis=1)

    daily_30 = pd.concat([runoff_30, temp_30, prec_30, evap_30], axis=1)
    daily_mean = daily_30.groupby(by=daily_30.index.day_of_year).mean()

    #pdb.set_trace()
    return daily_mean




def loyo_cv_lc_nofor(daily_input,output_folder, t_length=None, svr_C=None, svr_epsilon=None,
               lead_time=range(1, 8)):

    # Compute the climatology for all the inputs
    if isinstance(daily_input, str):
        daily_input = pd.read_csv(daily_input, index_col=0, parse_dates=True)
    monthly_clim = monthly_climatology(daily_input)


    # Create the total input-target matrix and select the start and end year for the loo-cv
    it_matrix = create_it_matrix(daily_input, t_length)
    year_start = min(it_matrix.index[(it_matrix.index.month == 1) & (it_matrix.index.day == 31)].year)
    year_end = max(it_matrix.index[(it_matrix.index.month == 12) & (it_matrix.index.day == 31)].year)

    # For each training length and each year train a different model and test with different configurations
    for Nyears_training in range(1, year_end-year_start+1):
        prediction = []
        for year in range(year_start, year_end+1):
            print(f'Testing on year {year}; Training years: {Nyears_training} ...')

            # Drop the years that should not be considered in the training and train the model
            daily_input_loo = daily_input.copy()
            training_years = (list(range(year+1, year_end+1)) + list(range(year_start, year)))[-Nyears_training:]
            daily_input_loo.loc[[i for i in daily_input.index if i.year not in training_years], 'Q'] = np.nan
            # daily_input_loo.loc[daily_input.index.year == year, 'Q'] = np.nan
            svr_model = training(daily_input_loo, t_length, svr_C, svr_epsilon)

            # Select the dates on which to execute the prediction
            test_dates = daily_input.index[daily_input.index.is_month_end & (daily_input.index.year == year)]

            # Save the true runoff values
            y = {}
            y['true_runoff'] = daily_input.loc[daily_input.index.year == year, 'Q'].resample("M").mean().values

            # Compute runoff monthly climatology considering the subset used for training the svr
            y['runoff_clim'] = [daily_input_loo.loc[daily_input_loo.index.month == m, 'Q'].mean() for m in range(1, 13)]

            # Predict using true temperature and precipitation (no forecast)
            X_trueTP = it_matrix.loc[test_dates, :].drop(columns='Q')
            y['trueTP'] = svr_model.predict(X_trueTP)

             # Predict using temperature and precipitation climatology (no forecast) for all lead times
            X_climTP = X_trueTP.copy()
            for lt in lead_time:
                change_dest = [c for c in X_climTP.columns if c.split('_')[1] == str(-lt + 1)]
                change_source = [c.split('_')[0] for c in change_dest]
                X_climTP.loc[:, change_dest] = monthly_clim.loc[(test_dates - pd.offsets.MonthBegin(lt)).month, change_source].values
                y[f'climTP_lt{lt}'] = svr_model.predict(X_climTP)
                
            '''
            # Predict using seasonal forecast with different lead times
            for lt in lead_time:
                for m in range(1, 26):
                    X_sf = sf_input_matrix(X_trueTP, seasonal_forecast, lt, m)
                    y[f'sfTP_m{m}_lt{lt}'] = svr_model.predict(X_sf)
            '''
            # Store the results in a dataframe
            prediction.append(pd.DataFrame(data=y, index=test_dates-pd.offsets.MonthBegin()))
            
        pd.concat(prediction, axis=0).to_csv(
            os.path.join(output_folder, f'Runoff_forecast_{Nyears_training}_trainingyears.csv'))

def loyo_cv_lc(daily_input, seasonal_forecast, output_folder, t_length=None, svr_C=None, svr_epsilon=None,
               lead_time=range(1, 8)):

    # Compute the climatology for all the inputs
    if isinstance(daily_input, str):
        daily_input = pd.read_csv(daily_input, index_col=0, parse_dates=True)
    monthly_clim = monthly_climatology(daily_input)

    # Read the seasonal forecast of temperature and precipitation
    if isinstance(seasonal_forecast, str):
        seasonal_forecast = pd.read_csv(seasonal_forecast, index_col=0, parse_dates=True)

    # Create the total input-target matrix and select the start and end year for the loo-cv
    it_matrix = create_it_matrix(daily_input, t_length)
    year_start = min(it_matrix.index[(it_matrix.index.month == 1) & (it_matrix.index.day == 31)].year)
    year_end = max(it_matrix.index[(it_matrix.index.month == 12) & (it_matrix.index.day == 31)].year)

    # For each training length and each year train a different model and test with different configurations
    for Nyears_training in range(1, year_end-year_start+1):
        prediction = []
        for year in range(year_start, year_end+1):
            print(f'Testing on year {year}; Training years: {Nyears_training} ...')

            # Drop the years that should not be considered in the training and train the model
            daily_input_loo = daily_input.copy()
            training_years = (list(range(year+1, year_end+1)) + list(range(year_start, year)))[-Nyears_training:]
            daily_input_loo.loc[[i for i in daily_input.index if i.year not in training_years], 'Q'] = np.nan
            # daily_input_loo.loc[daily_input.index.year == year, 'Q'] = np.nan
            svr_model = training(daily_input_loo, t_length, svr_C, svr_epsilon)

            # Select the dates on which to execute the prediction
            test_dates = daily_input.index[daily_input.index.is_month_end & (daily_input.index.year == year)]

            # Save the true runoff values
            y = {}
            y['true_runoff'] = daily_input.loc[daily_input.index.year == year, 'Q'].resample("M").mean().values

            # Compute runoff monthly climatology considering the subset used for training the svr
            y['runoff_clim'] = [daily_input_loo.loc[daily_input_loo.index.month == m, 'Q'].mean() for m in range(1, 13)]

            # Predict using true temperature and precipitation (no forecast)
            X_trueTP = it_matrix.loc[test_dates, :].drop(columns='Q')
            y['trueTP'] = svr_model.predict(X_trueTP)

            # Predict using temperature and precipitation climatology (no forecast) for all lead times
            X_climTP = X_trueTP.copy()
            for lt in lead_time:
                change_dest = [c for c in X_climTP.columns if c.split('_')[1] == str(-lt + 1)]
                change_source = [c.split('_')[0] for c in change_dest]
                X_climTP.loc[:, change_dest] = monthly_clim.loc[(test_dates - pd.offsets.MonthBegin(lt)).month, change_source].values
                y[f'climTP_lt{lt}'] = svr_model.predict(X_climTP)
                

            # Predict using seasonal forecast with different lead times
            for lt in lead_time:
                for m in range(1, 26):
                    X_sf = sf_input_matrix(X_trueTP, seasonal_forecast, lt, m)
                    y[f'sfTP_m{m}_lt{lt}'] = svr_model.predict(X_sf)

            # Store the results in a dataframe
            prediction.append(pd.DataFrame(data=y, index=test_dates-pd.offsets.MonthBegin()))

        pd.concat(prediction, axis=0).to_csv(
            os.path.join(output_folder, f'Runoff_forecast_{Nyears_training}_trainingyears.csv'))


# -----------------------------------------------------------------------

# Read and plot results


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


def learning_curve_rmse(result_folder, forecast='sfTP_em', plot=True):

    # forecast = 'sfTP_mX': plot the member X of seasonal forecast
    # forecast = 'sfTP_em': plot the ensamble mean of seasonal forecast
    # forecast = 'climTP': plot the forecast using the climatology of temperature and precipitation

    # result_folder = '/home/mcallegari@eurac.edu/SECLI-FIRM/Mattia/SF_runoff/Zoccolo/Results/Learning_curve/'

    runoff_error = []
    for year in range(1, 27):

        # Open the result file
        fileName = os.path.join(result_folder, f'Runoff_forecast_{year}_trainingyears.csv')
        runoff = pd.read_csv(fileName, index_col=0, parse_dates=True)

        # Create the ensamble mean
        for lt in range(1, 8):
            columns = [c for c in runoff.columns if 'sfTP_m' in c and f'lt{lt}' in c]
            runoff[f'sfTP_em_lt{lt}'] = runoff.loc[:, columns].mean(axis=1)

        # Compute the RMSE
        runoff_error.append(
            runoff.apply(lambda y_pred: root_mean_squared_error(runoff['true_runoff'], y_pred), axis=0)
        )

    runoff_error = pd.DataFrame(data=runoff_error, index=range(1, 27))

    if plot:
        plt.figure()
        for lt in range(1, 7):
            runoff_error.loc[:, f'{forecast}_lt{lt}'].plot(label=f'leadtime={lt}')
        runoff_error.loc[:, 'runoff_clim'].plot(color='black', label='climatology', linewidth=3)
        runoff_error.loc[:, 'trueTP'].plot(color='red', label='era5', linewidth=3)
        plt.legend()
        plt.ylabel('RMSE ($m^3/s$)')
        plt.xlabel('Years of training')

    else:
        return runoff_error


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
    
    return daily_input;