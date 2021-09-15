# Import libraries
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.impute import KNNImputer
import datetime as dt
import sklearn
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from tensorflow.keras.layers import Dense, Input, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

from etl.datamarts.view_import_functions import get_df_simple, get_df_with_lags, get_df_with_lags_per_area
from analysis.data_prep import get_dnn_test_train, get_ml_test_train

# Multivariate

def multivariate_regressor(X_train, y_train, X_test, y_test):
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(X_train, y_train)
    
    lin_reg_model.score(X_train, y_train)
    lin_reg_model.score(X_test, y_test)
    
    y_pred = lin_reg_model.predict(X_test)
    y_pred_train = lin_reg_model.predict(X_train)
    
    mse_train = np.sqrt(mean_squared_error(y_pred_train, y_train))
    mae_train = mean_absolute_error(y_pred_train, y_train)
    
    mse_test = np.sqrt(mean_squared_error(y_pred, y_test))
    mae_test = mean_absolute_error(y_pred, y_test)

    y_mean = np.mean(y_test)*np.ones(y_test.shape)
    print('Dumb MSE:')
    print(np.sqrt(mean_squared_error(y_mean, y_test)))
    print('Dumb MAE:')
    print(mean_absolute_error(y_mean, y_test))
    
    print(f'''
    Test MSE = {mse_test}
    Train MSE = {mse_train}
    Test MAE = {mae_test}
    Train MAE = {mae_train}
          ''')
          

# K Nearest Neighbor
def kneighborsregressor(X_train, y_train, X_test, y_test):    
    knr_model = KNeighborsRegressor(n_neighbors=10)
    knr_model.fit(X_train, y_train)
    
    print('R Squared for training data:')
    print(knr_model.score(X_train, y_train))
    print('R Squared for test data:')
    print(knr_model.score(X_test, y_test))
    
    y_pred = knr_model.predict(X_test)
    y_pred_train = knr_model.predict(X_train)
    
    mse_train = np.sqrt(mean_squared_error(y_pred_train, y_train))
    mae_train = mean_absolute_error(y_pred_train, y_train)
    
    mse_test = np.sqrt(mean_squared_error(y_pred, y_test))
    mae_test = mean_absolute_error(y_pred, y_test)
    
    y_mean = np.mean(y_test)*np.ones(y_test.shape)
    print('Dumb MSE:')
    print(np.sqrt(mean_squared_error(y_mean, y_test)))
    print('Dumb MAE:')
    print(mean_absolute_error(y_mean, y_test))
    
    print(f'''
    Test MSE = {mse_test}
    Train MSE = {mse_train}
    Test MAE = {mae_test}
    Train MAE = {mae_train}
    Correlation matrix:
          ''')
          

# XGBoost
X_train, X_test, y_train, y_test = get_ml_test_train(value = 'pm2_5', basis = None, latitude_category = False, longitude_category = False)

def xgb_regressor(X_train, y_train, X_test, y_test):
    xgbr_model = xgb.XGBRegressor()
    fit_params = dict(
        eval_set=[(X_test, y_test)], 
        early_stopping_rounds=10
        )
    xgbr_model.fit(X_train, y_train, **fit_params)
    
    r_sqrt_train = xgbr_model.score(X_train, y_train)
    print(f'R Squared for training data: {r_sqrt_train}')
    r_sqrt_test = xgbr_model.score(X_test, y_test)
    print(f'R Squared for test data: {r_sqrt_train}')
    
    y_pred = xgbr_model.predict(X_test)
    y_pred_train = xgbr_model.predict(X_train)
    
    mse_train = np.sqrt(mean_squared_error(y_pred_train, y_train))
    mae_train = mean_absolute_error(y_pred_train, y_train)
    
    mse_test = np.sqrt(mean_squared_error(y_pred, y_test))
    mae_test = mean_absolute_error(y_pred, y_test)
    
    y_mean = np.mean(y_test)*np.ones(y_test.shape)
    print('Dumb MSE:')
    print(np.sqrt(mean_squared_error(y_mean, y_test)))
    print('Dumb MAE:')
    print(mean_absolute_error(y_mean, y_test))
    
    print(f'''
    Test MSE = {mse_test}
    Train MSE = {mse_train}
    Test MAE = {mae_test}
    Train MAE = {mae_train}
            ''')
    
    feature_importance = xgbr_model.get_booster().get_score(importance_type="gain")
    
    stats = {
        'r squared train': r_sqrt_train,
        'r squared test': r_sqrt_test,
        'mse_train': mse_train,
        'mae_train': mae_train,
        'mse_test': mse_test,
        'mae_test': mae_test,
    }
    
    return y_pred, stats, feature_importance

def xgb_predictor(X, y, X_pred, y_dates):
    xgbr_model = xgb.XGBRegressor(max_depth = 2, learning_rate=0.05, n_estimators = 800, verbosity = 0)
    xgbr_model.fit(X, y)
    
    print('R Squared for training data:')
    print(xgbr_model.score(X, y))

    
    y_pred = xgbr_model.predict(X_pred)
    # y_pred = y_pred.flatten()
    prediction = np.vstack([y_dates.flatten() ,y_pred]).transpose()
    
    return prediction