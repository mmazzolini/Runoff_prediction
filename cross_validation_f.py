# import necessary functions.

# basic libraries
import numpy as np
import pandas as pd

# plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn necessary modules
from sklearn.svm import SVR, LinearSVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit, KFold
from sklearn.metrics import r2_score

# other modules imports
from base_f import create_it_matrix, create_gap
from climatology_ensemble import  daily_climatology_p_et_ensemble

# just for debugging
import pdb







########### classic_Cross_validation.##############



def classic_CV_SVR_predict(daily_input, C, eps,t_length,t_unit, n_splits, linear=False):#, radius_for_ensemble):

    
    #compute the daily climatology and the quantile analysis
    daily_clim = daily_climatology_p_et_ensemble(daily_input,0,t_unit)
    #get the input-target matrix
    it_matrix=create_it_matrix(daily_input,t_length,t_unit)
    #split in train-test sets
    tscv = KFold(n_splits=n_splits)
    sets = tscv.split(it_matrix.index)
    
    j=0;
    prediction=pd.DataFrame(data=None)
    
    for train_index, test_index in sets:
        
        #reduce train_set to have a gap with test index and ensure independence
        train_index=create_gap(train_index, test_index,t_unit)

        #set up training features
        X = it_matrix.drop(columns='Q').iloc[train_index]
        y = it_matrix['Q'].iloc[train_index]
        
        #set up the model according to the parameters
        if linear:
            svr_estimator = LinearSVR(tol=0.0001, C=C, epsilon=eps,random_state=0)
        else:
            svr_estimator = SVR(kernel='rbf', gamma='scale', C=C, epsilon=eps, cache_size=7000)
            
            
        svr_model_tuned = make_pipeline(StandardScaler(),
                                        TransformedTargetRegressor(regressor=svr_estimator, transformer=StandardScaler()))

        #fit the model
        svr_model_tuned.fit(X, y)

        # get the test dates (end of the month)
        test_dates = it_matrix.index[test_index]#[daily_input.index.is_month_end]
        # and their day of the year
        doy_test_dates = test_dates.dayofyear

        # Save the true runoff values (with t_unit days rolling average)
        target = {}
        target['true_runoff'] = daily_input.Q.rolling(30, min_periods=30).mean().loc[test_dates]

        # Compute runoff monthly climatology using the whole dataset
        runoff_daily_clim = daily_input.Q.rolling(30, min_periods=30).mean()
        target['runoff_clim'] = [runoff_daily_clim.loc[runoff_daily_clim.index.dayofyear == d].mean() for d in doy_test_dates]
        target['runoff_clim_25'] = [runoff_daily_clim.loc[runoff_daily_clim.index.dayofyear == d].quantile(q=0.25) for d in doy_test_dates]
        target['runoff_clim_75'] = [runoff_daily_clim.loc[runoff_daily_clim.index.dayofyear == d].quantile(q=0.75) for d in doy_test_dates]
        

        X_trueTP = it_matrix.loc[test_dates, :].drop(columns='Q')
        target['prediction'] = svr_model_tuned.predict(X_trueTP)

        """
        # Predict using temperature and precipitation climatology
        # predict also for 25th and 75th quantile situations.
        
        X_climTP = X_trueTP.copy()
        X_climTP_Q25=X_trueTP.copy()
        X_climTP_Q75=X_trueTP.copy()

        #predict till 6 months of advance
        lead_times = range(1,6)
        for lt in lead_times:
            
            # modify the X matrix by substituting the climatology to the real meteo vars for lt months.
            change_dest = [c for c in X_climTP.columns if c.split('_')[1] == str(-lt + 1)]
            change_source = [c.split('_')[0] for c in change_dest]
            X_climTP.loc[:, change_dest]=daily_clim.loc[(test_dates-np.timedelta64(t_unit*(lt-1),'D')).dayofyear][change_source].values
            
            #predict
            target[f'climTP_lt{lt}'] = svr_model_tuned.predict(X_climTP)

            # modify the X matrix by substituting the climatology to the extreme (25th and 75th quantiles) meteo vars for lt months.
            change_source_25 = []
            change_source_75 = []
            #modify the source, by taking the daily climathological data referred to the quantiles situations
            for i in change_source:
                    change_source_25.append(i+'_Q25')
                    change_source_75.append((i+'_Q75'))

            X_climTP_Q25.loc[:, change_dest]=daily_clim.loc[(test_dates-np.timedelta64(t_unit*(lt-1),'D')).dayofyear][change_source_25].values
            target[f'climTP_lt{lt}_Q25'] = svr_model_tuned.predict(X_climTP_Q25)

            X_climTP_Q75.loc[:, change_dest]=daily_clim.loc[(test_dates-np.timedelta64(t_unit*(lt-1),'D')).dayofyear][change_source_75].values
            target[f'climTP_lt{lt}_Q75'] = svr_model_tuned.predict(X_climTP_Q75)
        """
        
        target['split']= np.repeat(j,test_index.shape[0]) 
    
        #add this split prediction to the 
        #pdb.set_trace()

        prediction=prediction.append(pd.DataFrame(data=target, index=test_dates))
        j=j+1

    return prediction.drop(columns='split')



def classic_CV_PCA_SVR_predict(daily_input, C, eps, n, t_length,t_unit ,n_splits,linear=False): #radius_for_ensemble):
    
    #compute the daily climatology and the quantile analysis
    daily_clim = daily_climatology_p_et_ensemble(daily_input,0,t_unit)
    #get the input-target matrix
    it_matrix=create_it_matrix(daily_input,t_length,t_unit)
    #split in train-test sets
    tscv = KFold(n_splits=n_splits)
    sets = tscv.split(it_matrix.index)
    
    j=0;
    prediction=pd.DataFrame(data=None)
    
    for train_index, test_index in sets:
        
        #reduce train_set to have a gap with test index and ensure independence
        train_index=create_gap(train_index, test_index,t_unit)

        #set up training features
        X = it_matrix.drop(columns='Q').iloc[train_index]
        y = it_matrix['Q'].iloc[train_index]
        
        #set up the model according to the parameters
        if linear:
            svr_estimator = LinearSVR(tol=0.0001, C=C, epsilon=eps,random_state=0)
        else:
            svr_estimator = SVR(kernel='rbf', gamma='scale', C=C, epsilon=eps, cache_size=7000)

        svr_model_tuned = make_pipeline(StandardScaler(),
                                        PCA(n_components=n),
                                      TransformedTargetRegressor(regressor=svr_estimator, transformer=StandardScaler()))

        #fit the model
        svr_model_tuned.fit(X, y)

        # get the test dates (end of the month)
        test_dates = it_matrix.index[test_index]
        # and their day of the year
        doy_test_dates = test_dates.dayofyear

        # Save the true runoff values (with t_unit days rolling average)
        target = {}
        target['true_runoff'] = daily_input.Q.rolling(30, min_periods=30).mean().loc[test_dates]

        # Compute runoff monthly climatology using the whole dataset
        runoff_daily_clim = daily_input.Q.rolling(30, min_periods=30).mean()
        target['runoff_clim'] = [runoff_daily_clim.loc[runoff_daily_clim.index.dayofyear == d].mean() for d in doy_test_dates]

        X_trueTP = it_matrix.loc[test_dates, :].drop(columns='Q')
        target['prediction'] = svr_model_tuned.predict(X_trueTP)

        """
        # Predict using temperature and precipitation climatology
        # predict also for 25th and 75th quantile situations.
        
        X_climTP = X_trueTP.copy()
        X_climTP_Q25=X_trueTP.copy()
        X_climTP_Q75=X_trueTP.copy()

        #predict till 6 months of advance
        lead_times = range(1,6)
        for lt in lead_times:
            
            # modify the X matrix by substituting the climatology to the real meteo vars for lt months.
            change_dest = [c for c in X_climTP.columns if c.split('_')[1] == str(-lt + 1)]
            change_source = [c.split('_')[0] for c in change_dest]
            X_climTP.loc[:, change_dest]=daily_clim.loc[(test_dates-np.timedelta64(t_unit*(lt-1),'D')).dayofyear][change_source].values
            
            #predict
            target[f'climTP_lt{lt}'] = svr_model_tuned.predict(X_climTP)

            # modify the X matrix by substituting the climatology to the extreme (25th and 75th quantiles) meteo vars for lt months.
            change_source_25 = []
            change_source_75 = []
            #modify the source, by taking the daily climathological data referred to the quantiles situations
            for i in change_source:
                    change_source_25.append(i+'_Q25')
                    change_source_75.append((i+'_Q75'))

            X_climTP_Q25.loc[:, change_dest]=daily_clim.loc[(test_dates-np.timedelta64(t_unit*(lt-1),'D')).dayofyear][change_source_25].values
            target[f'climTP_lt{lt}_Q25'] = svr_model_tuned.predict(X_climTP_Q25)

            X_climTP_Q75.loc[:, change_dest]=daily_clim.loc[(test_dates-np.timedelta64(t_unit*(lt-1),'D')).dayofyear][change_source_75].values
            target[f'climTP_lt{lt}_Q75'] = svr_model_tuned.predict(X_climTP_Q75)
        """
        target['split']= np.repeat(j,test_index.shape[0])
            
        #pdb.set_trace()

        #add this split prediction to the 
        prediction=prediction.append(pd.DataFrame(data=target, index=test_dates))
        
        j=j+1

    return prediction.drop(columns='split')
    
    
    
    

########### nested_Cross_validation.##############




def SVR_nested_CV_gridsearch(daily_input, C_range, epsilon_range, t_range,t_unit,n_splits,test_siz, linear=False):


    for t_length in t_range:
        it_matrix=create_it_matrix(daily_input,t_length,t_unit).astype('float32')
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_siz,gap=t_unit)
        sets = tscv.split(it_matrix.index)
        all_models= []
        #pdb.set_trace()
        for train_index, test_index in sets:
            #validation set is the last 2 years of the "old_training"
            val_index   = train_index[-365*2:]

            #training set is reduced by one year and 1month"
            train_index_update = train_index[:-365*2-30]
            trainCvSplit = [(list(train_index_update),list(val_index))]
            
            X = it_matrix.drop(columns='Q')
            y = it_matrix['Q']
            
            
            if linear:
                svr_estimator = LinearSVR(tol=0.0001,random_state=0)
            else:
                svr_estimator = SVR(kernel='rbf', gamma='scale', cache_size=15000)


            svr_estimator = make_pipeline(StandardScaler(),
                                          TransformedTargetRegressor(regressor=svr_estimator, transformer=StandardScaler()))
            parameters = {'transformedtargetregressor__regressor__C': C_range,
                          'transformedtargetregressor__regressor__epsilon': epsilon_range}


            svr_model = GridSearchCV(svr_estimator, cv=trainCvSplit, param_grid=parameters, n_jobs=-1, verbose=1, refit=True, return_train_score=True)

            # execute the grid search
            svr_model.fit(X, y)
            all_models.append(pd.DataFrame(svr_model.cv_results_))
        
        #PUT ALL THE TRAINED MODELS AND RESULTS IN A DATAFRAME
        all_m = pd.DataFrame(data=None,columns=all_models[0].columns)
        for i in all_models :
            all_m=all_m.append(i);
        
        #GROUP BY PARAMETERS AND AVERAGE OVER THE DIFFERENT VALIDATION SETS
        par=(['param_transformedtargetregressor__regressor__C','param_transformedtargetregressor__regressor__epsilon'])
        avg_models = all_m.groupby(par).mean()
        #avg_models['train_test_diff']= avg_models.mean_train_score - avg_models.mean_test_score
        
        # SELECT THE SINGLE BEST MODEL OVERALL
        best_model_overall = avg_models.loc[[avg_models.mean_test_score.idxmax()]]
        
        best_C=best_model_overall.reset_index().param_transformedtargetregressor__regressor__C[0]
        best_epsilon = best_model_overall.reset_index().param_transformedtargetregressor__regressor__epsilon[0]

        #INVESTIGATE WITH HEATMAPS OVERFITTING
        # get the coordinates of the "best model"
        y=np.where(epsilon_range==best_epsilon)[0]+0.5
        x=np.where(C_range==best_C)[0]+0.5

        hm_test = avg_models.reset_index().pivot( columns='param_transformedtargetregressor__regressor__C', index='param_transformedtargetregressor__regressor__epsilon', values='mean_test_score')
        hm_train= avg_models.reset_index().pivot(columns='param_transformedtargetregressor__regressor__C',index='param_transformedtargetregressor__regressor__epsilon',values='mean_train_score')
        plt.figure(figsize=(15,7))
        plt.subplot(1,2,1)
        sns.heatmap(hm_test,vmin=hm_test.min().min(),vmax=hm_train.max().max())
        plt.title('VALIDATION')
        plt.plot(x,y,marker='o')

        plt.subplot(1,2,2)
        sns.heatmap(hm_train,vmin=hm_test.min().min(),vmax=hm_train.max().max())
        plt.title('TRAIN')
        plt.plot(x,y,marker='o')

        plt.tight_layout()
            

        # Check if the best C (or epsion) is in the border of the grid
        if best_C == max(C_range) or best_C == min(C_range):
            print(f'Warning: best C found on the grid limit: C = {best_C}')
        if best_epsilon == max(epsilon_range) or best_epsilon == min(epsilon_range):
            print(f'Warning: best epsilon found on the grid limit: epsilon = {best_epsilon}')
        print()
    
    return best_C, best_epsilon


def SVR_PCA_nested_CV_gridsearch(daily_input, C_range, epsilon_range, components_range, t_range,t_unit,n_splits,test_size, linear=False):
                         
    for t_length in t_range:
        it_matrix=create_it_matrix(daily_input,t_length,t_unit).astype('float32')
        tscv = TimeSeriesSplit(gap=t_unit ,n_splits=n_splits, test_size=test_size)
        sets = tscv.split(it_matrix.index)
        
        all_models= []
    
        for train_index, test_index in sets:
            #validation set is the last 2 years of the "old_training"
            val_index   = train_index[-365*2:]

            #training set is reduced by 2y and 1month"
            train_index_update = train_index[:-365*2-30]
            trainCvSplit = [(list(train_index_update),list(val_index))]
            
            X = it_matrix.drop(columns='Q')
            y = it_matrix['Q']

            if linear:
                svr_estimator = LinearSVR(tol=0.0001, random_state=0)
            else:
                svr_estimator = SVR(kernel='rbf', gamma='scale', cache_size=12000)
                
            svr_estimator = make_pipeline(StandardScaler(),
                                          PCA(),
                                          TransformedTargetRegressor(regressor=svr_estimator, transformer=StandardScaler()))
            parameters = {'pca__n_components': components_range,
                          'transformedtargetregressor__regressor__C': C_range,
                          'transformedtargetregressor__regressor__epsilon': epsilon_range
                         }

            svr_model = GridSearchCV(svr_estimator, cv=trainCvSplit, param_grid=parameters, n_jobs=-1, verbose=1, refit=True, return_train_score=True)

            # execute the grid search
            svr_model.fit(X, y)
            all_models.append(pd.DataFrame(svr_model.cv_results_))
        
        #PUT ALL THE TRAINED MODELS AND RESULTS IN A DATAFRAME
        all_m = pd.concat(all_models)
        
        #GROUP BY PARAMETERS AND AVERAGE OVER THE DIFFERENT VALIDATION SETS
        par=(['param_pca__n_components', 'param_transformedtargetregressor__regressor__C', 'param_transformedtargetregressor__regressor__epsilon'])
        
        
        #pdb.set_trace()
        avg_models = all_m.groupby(par).mean()
        #avg_models['train_test_diff']= avg_models.mean_train_score - avg_models.mean_test_score

        # SELECT THE SINGLE BEST MODEL OVERALL
        best_model_overall = avg_models.loc[[avg_models.mean_test_score.idxmax()]]
        best_C=best_model_overall.reset_index().param_transformedtargetregressor__regressor__C[0]
        best_epsilon = best_model_overall.reset_index().param_transformedtargetregressor__regressor__epsilon[0]
        best_n = best_model_overall.reset_index().param_pca__n_components[0]
        
        #INVESTIGATE WITH HEATMAPS THE FACT THAT WE'RE NOT OVERFITTING
        # get the coordinates of the "best model"
        y=np.where(epsilon_range==best_epsilon)[0]+0.5
        x=np.where(C_range==best_C)[0]+0.5

        #get the models with a certain number of components
        query=f'param_pca__n_components=={best_n}'
        nc=avg_models.query(query)

        hm_test  = nc.reset_index().pivot( columns='param_transformedtargetregressor__regressor__C', index='param_transformedtargetregressor__regressor__epsilon', values='mean_test_score')
        hm_train = nc.reset_index().pivot( columns='param_transformedtargetregressor__regressor__C', index='param_transformedtargetregressor__regressor__epsilon', values='mean_train_score')

        plt.figure(figsize=(15,7))
        plt.subplot(1,2,1)
        sns.heatmap(hm_test,vmin=hm_test.min().min(),vmax=hm_train.max().max())
        plt.title('TEST')
        plt.plot(x,y,marker='o')

        plt.subplot(1,2,2)
        sns.heatmap(hm_train,vmin=hm_test.min().min(),vmax=hm_train.max().max())
        plt.title('TRAIN')
        plt.plot(x,y,marker='o')

        plt.tight_layout()
        
               
        # Check if the best C, epsion or gamma is in the border of the grid
        if best_C == max(C_range) or best_C == min(C_range):
            print(f'Warning: best C found on the grid limit: C = {best_C}')
        if best_epsilon == max(epsilon_range) or best_epsilon == min(epsilon_range):
            print(f'Warning: best epsilon found on the grid limit: epsilon = {best_epsilon}')
        print()
        
    return best_C, best_epsilon, best_n




def nested_CV_SVR_predict(daily_input, C, eps, t_length, t_unit, n_splits, test_size, radius_for_ensemble, linear=False):
    
    #compute the daily climatology and the quantile analysis
    daily_clim = daily_climatology_p_et_ensemble(daily_input,t_unit,radius_for_ensemble)
    #get the input-target matrix
    it_matrix=create_it_matrix(daily_input,t_length,t_unit)
    #split in train-test sets
    tscv = TimeSeriesSplit(gap=t_unit ,n_splits=n_splits, test_size=test_size)
    sets = tscv.split(it_matrix.index)
    j=0;
    prediction=pd.DataFrame(data=None)
    #pdb.set_trace()
    for train_index, test_index in sets:
        #set up training features
        X = it_matrix.drop(columns='Q').iloc[train_index]
        y = it_matrix['Q'].iloc[train_index]
        
        #set up the model according to the parameters
        if linear:
            svr_estimator = LinearSVR(tol=0.0001, C=C, epsilon=eps,random_state=0)
        else:
            svr_estimator = SVR(kernel='rbf', gamma='scale', C=C, epsilon=eps, cache_size=7000)
        svr_model_tuned = make_pipeline(StandardScaler(),
                                        TransformedTargetRegressor(regressor=svr_estimator, transformer=StandardScaler()))

        #fit the model
        svr_model_tuned.fit(X, y)

        # get the test dates (end of the month)
        test_dates = it_matrix.index[test_index]#[daily_input.index.is_month_end]
        # and their day of the year
        doy_test_dates = test_dates.dayofyear

        # Save the true runoff values (with t_unit days rolling average)
        target = {}
        target['true_runoff'] = daily_input.Q.rolling(30, min_periods=30).mean().loc[test_dates]



        # Compute runoff monthly climatology using the whole dataset
        runoff_daily_clim = daily_input.Q.rolling(30, min_periods=30).mean()
        target['runoff_clim'] = [runoff_daily_clim.loc[runoff_daily_clim.index.dayofyear == d].mean() for d in doy_test_dates]
        target['runoff_clim_25'] = [runoff_daily_clim.loc[runoff_daily_clim.index.dayofyear == d].quantile(q=0.25) for d in doy_test_dates]
        target['runoff_clim_75'] = [runoff_daily_clim.loc[runoff_daily_clim.index.dayofyear == d].quantile(q=0.75) for d in doy_test_dates]
        
        
        X_trueTP = it_matrix.loc[test_dates, :].drop(columns='Q')
        target['trueTP'] = svr_model_tuned.predict(X_trueTP)


        # Predict using temperature and precipitation climatology
        # predict also for 25th and 75th quantile situations.
        
        X_climTP = X_trueTP.copy()
        X_climTP_Q25=X_trueTP.copy()
        X_climTP_Q75=X_trueTP.copy()

        #predict till 6 time_units of advance
        lead_times = range(1,6)
        for lt in lead_times:
            
            
            # modify the X matrix by substituting the climatology to the real meteo vars for lt.
            change_dest = [c for c in X_climTP.columns if c.split('_')[1] == str(-lt + 1)]
            change_source = [c.split('_')[0] for c in change_dest]
            #pdb.set_trace()
            X_climTP.loc[:, change_dest]=daily_clim.loc[(test_dates-np.timedelta64(t_unit*(lt-1),'D')).dayofyear][change_source].values
            
            #predict
            target[f'climTP_lt{lt}'] = svr_model_tuned.predict(X_climTP)

            # modify the X matrix by substituting the climatology to the extreme (25th and 75th quantiles) meteo vars for lt.
            change_source_25 = []
            change_source_75 = []
            #modify the source, by taking the daily climathological data referred to the quantiles situations
            for i in change_source:
                    change_source_25.append(i+'_Q25')
                    change_source_75.append((i+'_Q75'))

            X_climTP_Q25.loc[:, change_dest]=daily_clim.loc[(test_dates-np.timedelta64(t_unit*(lt-1),'D')).dayofyear][change_source_25].values
            target[f'climTP_lt{lt}_Q25'] = svr_model_tuned.predict(X_climTP_Q25)

            X_climTP_Q75.loc[:, change_dest]=daily_clim.loc[(test_dates-np.timedelta64(t_unit*(lt-1),'D')).dayofyear][change_source_75].values
            target[f'climTP_lt{lt}_Q75'] = svr_model_tuned.predict(X_climTP_Q75)
            #pdb.set_trace()

        target['split']= np.repeat(j,test_size)
            
        #add this split prediction to the 
        prediction=prediction.append(pd.DataFrame(data=target, index=test_dates))
        
        j=j+1

    return prediction



def nested_CV_PCA_SVR_predict(daily_input, C, eps, n, t_length, t_unit,  n_splits, test_size, radius_for_ensemble, linear=False):
    
    #compute the daily climatology and the quantile analysis
    daily_clim = daily_climatology_p_et_ensemble(daily_input,t_unit,radius_for_ensemble)
    #get the input-target matrix
    it_matrix=create_it_matrix(daily_input,t_length,t_unit)
    #split in train-test sets
    tscv = TimeSeriesSplit(gap=t_unit ,n_splits=n_splits, test_size=test_size)
    sets = tscv.split(it_matrix.index)
    j=0;
    prediction=pd.DataFrame(data=None)
    
    for train_index, test_index in sets:
        #set up training features
        X = it_matrix.drop(columns='Q').iloc[train_index]
        y = it_matrix['Q'].iloc[train_index]
        
        #set up the model according to the parameters
        if linear:
            svr_estimator = LinearSVR(tol=0.0001, C=C, epsilon=eps,random_state=0)
        else:
            svr_estimator = SVR(kernel='rbf', gamma='scale', C=C, epsilon=eps, cache_size=7000)
        svr_model_tuned = make_pipeline(StandardScaler(),
                                        PCA(n_components=n),
                                      TransformedTargetRegressor(regressor=svr_estimator, transformer=StandardScaler()))

        #fit the model
        svr_model_tuned.fit(X, y)

        # get the test dates (end of the month)
        test_dates = it_matrix.index[test_index]#[daily_input.index.is_month_end]
        # and their day of the year
        doy_test_dates = test_dates.dayofyear

        # Save the true runoff values (with t_unit days rolling average)
        target = {}
        target['true_runoff'] = daily_input.Q.rolling(30, min_periods=30).mean().loc[test_dates]

        # Compute runoff monthly climatology using the whole dataset
        runoff_daily_clim = daily_input.Q.rolling(30, min_periods=30).mean()
        target['runoff_clim'] = [runoff_daily_clim.loc[runoff_daily_clim.index.dayofyear == d].mean() for d in doy_test_dates]

        X_trueTP = it_matrix.loc[test_dates, :].drop(columns='Q')
        target['trueTP'] = svr_model_tuned.predict(X_trueTP)


        # Predict using temperature and precipitation climatology
        # predict also for 25th and 75th quantile situations.
        
        X_climTP = X_trueTP.copy()
        X_climTP_Q25=X_trueTP.copy()
        X_climTP_Q75=X_trueTP.copy()

        #predict till 6 periods of advance
        lead_times = range(1,6)
        for lt in lead_times:
            
            # modify the X matrix by substituting the climatology to the real meteo vars for lt months.
            change_dest = [c for c in X_climTP.columns if c.split('_')[1] == str(-lt + 1)]
            change_source = [c.split('_')[0] for c in change_dest]
            #pdb.set_trace()
            X_climTP.loc[:, change_dest]=daily_clim.loc[(test_dates-np.timedelta64(t_unit*(lt-1),'D')).dayofyear][change_source].values
            
            #predict
            target[f'climTP_lt{lt}'] = svr_model_tuned.predict(X_climTP)

            # modify the X matrix by substituting the climatology to the extreme (25th and 75th quantiles) meteo vars for lt months.
            change_source_25 = []
            change_source_75 = []
            #modify the source, by taking the daily climathological data referred to the quantiles situations
            for i in change_source:
                    change_source_25.append(i+'_Q25')
                    change_source_75.append((i+'_Q75'))

            X_climTP_Q25.loc[:, change_dest]=daily_clim.loc[(test_dates-np.timedelta64(t_unit*(lt-1),'D')).dayofyear][change_source_25].values
            target[f'climTP_lt{lt}_Q25'] = svr_model_tuned.predict(X_climTP_Q25)

            X_climTP_Q75.loc[:, change_dest]=daily_clim.loc[(test_dates-np.timedelta64(t_unit*(lt-1),'D')).dayofyear][change_source_75].values
            target[f'climTP_lt{lt}_Q75'] = svr_model_tuned.predict(X_climTP_Q75)
            
            #pdb.set_trace()
        target['split']= np.repeat(j,test_size)

            
        #add this split prediction to the 
        prediction = prediction.append(pd.DataFrame(data=target, index=test_dates))
        #pdb.set_trace()
        j=j+1

    return prediction


########## PLOTTING FUNCTIONS ############


def plot_prediction(prediction):

    splits=prediction['split'].max()
    for i in range(splits+1):
        query=f'split=={i}'
        #query='split==' + str(i)
        pred=prediction.query(query)
        pred.loc[:,'date']= pred.index

        ax,fig=plt.subplots(figsize=(20,10))
        
        #plot the real and modelled discharge
        sns.lineplot(y=("true_runoff"),x="date",data=pred,color='red',linewidth=1.3,legend='auto')
        sns.lineplot(y=("trueTP"),x="date",data=pred,color='green',linewidth=1.3,legend='auto')


        #plot the clim_distr
        sns.lineplot(y=("runoff_clim"),x="date",data=pred,color='yellow',linewidth=1.3,legend='auto')
        lt1=pred[["runoff_clim_25","runoff_clim_75"]]
        plt.fill_between(x=lt1.index, y1=lt1['runoff_clim_25'], y2=lt1['runoff_clim_75'], alpha=0.12,color='y')


        #plot the lead_time_
        lt1=pred[["climTP_lt1","climTP_lt1_Q25","climTP_lt1_Q75"]]
        #lt1.columns=np.repeat('climatologia_lt1_ensemple_prec',3)
        sns.lineplot(data=lt1["climTP_lt1"],legend='auto')
        plt.fill_between(x=lt1.index, y1=lt1['climTP_lt1_Q25'], y2=lt1['climTP_lt1_Q75'], alpha=0.2)

        """
        #plot the lead_time_
        lt4=pred[["climTP_lt4","climTP_lt4_Q25","climTP_lt4_Q75"]]
        #lt4.columns=np.repeat('climatologia_lt4_ensemple_prec',3)
        sns.lineplot(data=lt4["climTP_lt4"], palette=['green'],legend='auto')
        plt.fill_between(x=lt4.index, y1=lt4['climTP_lt4_Q25'], y2=lt4['climTP_lt4_Q75'], alpha=0.2)
        """

        plt.ylabel('30 days discharge average [m^3/sec]')

        plt.legend(['TRUE DISCHARGE', 'DISCHARGE CLIMATOLOGY', 'LEAD TIME = 0', 'LEAD TIME = 1'])    
        plt.title("Precipitation variability(Q= 25 and 75) mapped on the prediction discharge")
    return;



def plot_anomalies(prediction):

    splits=prediction['split'].max()
    for i in range(splits+1):
        query=f'split=={i}'
        #query='split==' + str(i)
        pred=prediction.query(query)
        pred.loc[:,'date']= pred.index

        ax,fig=plt.subplots(figsize=(20,10))
        #plot the real
        sns.lineplot(y=("true_runoff"),x="date",data=pred,color='red',linewidth=1.3,legend='auto')
        sns.lineplot(y=("trueTP"),x="date",data=pred,color='green',linewidth=1.3,legend='auto')


        #plot the lead_time_
        lt1=pred[["climTP_lt1","climTP_lt1_Q25","climTP_lt1_Q75"]]
        #lt1.columns=np.repeat('climatologia_lt1_ensemple_prec',3)
        sns.lineplot(data=lt1["climTP_lt1"],legend='auto')
        plt.fill_between(x=lt1.index, y1=lt1['climTP_lt1_Q25'], y2=lt1['climTP_lt1_Q75'], alpha=0.2)

        #plot the lead_time_
        lt4=pred[["climTP_lt4","climTP_lt4_Q25","climTP_lt4_Q75"]]
        #lt4.columns=np.repeat('climatologia_lt4_ensemple_prec',3)
        sns.lineplot(data=lt4["climTP_lt4"], palette=['green'],legend='auto')
        plt.fill_between(x=lt4.index, y1=lt4['climTP_lt4_Q25'], y2=lt4['climTP_lt4_Q75'], alpha=0.2)

        plt.axhline(0,ls='--')
        plt.ylabel('30 days averaged discharge anomaly [m^3/sec]')

        plt.legend(['TRUE DISCHARGE','LEAD TIME = 0','LEAD TIME = 1','LEAD TIME = 4'])    
        plt.title("Anomalies plotting with precipitation variability(Q= 25 and 75) mapped on the prediction discharge")
    return;






########## EVALUATION  FUNCTIONS ############




def evaluate_prediction(prediction):
    to_drop=[]
    for c in prediction.columns:
        if c[-3] == 'Q':
            to_drop.append(c)
    to_drop.append('split')
    runoff=prediction.drop(columns=to_drop)
    
    dictio={"true_runoff" : "measured runoff",
        "runoff_clim" : "runoff climatology",
        "trueTP"      : "model output",
        "climTP_lt1"  : "output 1 month lead time",
        "climTP_lt2"  : "output 2 month lead time",
        "climTP_lt3"  : "output 3 month lead time",
        "climTP_lt4"  : "output 4 month lead time",
        "climTP_lt5"  : "output 5 month lead time",
       }
       
    runoff = runoff.rename(columns=dictio)
    runoff_error_r2=runoff.apply(lambda y_pred: r2_score(runoff['measured runoff'], y_pred), axis=0)
       
    runoff_error_r2.plot.bar(ylabel="R^2 score [/]", rot=90)
    
    #plt.legend(['MEASURED DISCHARGE','RUNOFF CLIMATOLOGY','MODEL PREDICTION','PREDICTION WITH 1 t_UNIT LEAD TIME','','']
    
    return runoff_error_r2
    


def evaluate_class(prediction):

    tp=np.sum(np.logical_and(prediction.true_runoff<prediction.runoff_clim_25, prediction.trueTP<prediction.runoff_clim_25))
    fp=np.sum(np.logical_and(prediction.true_runoff>prediction.runoff_clim_25, prediction.trueTP<prediction.runoff_clim_25))
    p=np.sum(prediction.true_runoff<prediction.runoff_clim_25)
    tn=np.sum(np.logical_and(prediction.true_runoff>prediction.runoff_clim_25, prediction.trueTP>prediction.runoff_clim_25))
    fn=np.sum(np.logical_and(prediction.true_runoff<prediction.runoff_clim_25, prediction.trueTP>prediction.runoff_clim_25))
    n=np.sum(prediction.true_runoff>prediction.runoff_clim_25)

    #sensitivity
    sen=tp/p

    #specificity
    spe=tn/n

    #precision
    prec=tp/(tp+fp)
    
    result = pd.DataFrame(data=np.transpose(np.array([[sen],[spe],[prec],[p]])),columns=['sensitivity','specificity','precision','number'])
   
    
    return result
    
    
    
    
def evaluate_class_season(prediction):
    
    prediction['season'] = prediction.index.month%12 // 3

    results=pd.DataFrame(data=None)
    
    for s in range(4):

        r = evaluate_class(prediction[prediction.season==s])
        r['season']=s
        results=results.append(r)

    return results