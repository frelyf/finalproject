import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from etl.datamarts.view_import_functions import get_df_prediction_test, get_df_simple, get_df_with_lags, get_df_with_lags_per_area

def get_dnn_test_train():
    df = get_df_with_lags()
    df = df.sort_values(by='dateid_serial', ignore_index = True)
    df = df.iloc[364:].reset_index()


    X_cols = [
        'traffic_volume', 'precipitation', 'wind_speed','air_temperature',
        'traffic_volume_lag_1', 'precipitation_lag_1', 'wind_speed_lag_1',
        'air_temperature_lag_1', 'pm2_5_lag_1', 'pm10_lag_1', 'nox_lag_1',
        'no2_lag_1', 'no_lag_1', 'traffic_volume_lag_2', 'precipitation_lag_2',
        'wind_speed_lag_2', 'air_temperature_lag_2', 'pm2_5_lag_2',
        'pm10_lag_2', 'nox_lag_2', 'no2_lag_2', 'no_lag_2',
        'traffic_volume_lag_3', 'precipitation_lag_3', 'wind_speed_lag_3',
        'air_temperature_lag_3', 'pm2_5_lag_3', 'pm10_lag_3', 'nox_lag_3',
        'no2_lag_3', 'no_lag_3', 'traffic_volume_lag_6', 'precipitation_lag_6',
        'wind_speed_lag_6', 'air_temperature_lag_6', 'pm2_5_lag_6',
        'pm10_lag_6', 'nox_lag_6', 'no2_lag_6', 'no_lag_6',
        'traffic_volume_lag_12', 'precipitation_lag_12', 'wind_speed_lag_12',
        'air_temperature_lag_12', 'pm2_5_lag_12', 'pm10_lag_12', 'nox_lag_12',
        'no2_lag_12', 'no_lag_12'
    ]

    y_col = ['pm10','pm2_5', 'no','no2', 'nox']

    # Imputing values
    imputer_X = KNNImputer(n_neighbors = 5)
    imputer_X.fit(df[X_cols])
    df[X_cols] = imputer_X.transform(df[X_cols])

    imputer_y = KNNImputer(n_neighbors = 5)
    imputer_y.fit(df[y_col])
    df[y_col] = imputer_y.transform(df[y_col])

    # StandardScaling for df_avg
    s_scaler = StandardScaler()
    s_scaler.fit(df[X_cols])
    df[X_cols] = s_scaler.transform(df[X_cols])

    # Splitting into train and test, X and y
    df_avg_test = df.iloc[:365]
    df_avg_train = df.iloc[365:]

    X_train = np.c_[df_avg_train[X_cols]]
    X_test = np.c_[df_avg_test[X_cols]]

    y_train = np.c_[df_avg_train[y_col]]
    y_test = np.c_[df_avg_test[y_col]]

    return X_train, X_test, y_train, y_test


def get_prediction_dnn():

    df = get_df_with_lags()
    df_pred = get_df_prediction_test()
    
    df_pred = df_pred.merge()
    
    df = df.sort_values(by='dateid_serial', ignore_index = True)
    df = df.iloc[364:].reset_index()
    
    
    X_cols = [
        'traffic_volume', 'precipitation', 'wind_speed','air_temperature',
        'traffic_volume_lag_1', 'precipitation_lag_1', 'wind_speed_lag_1',
        'air_temperature_lag_1', 'pm2_5_lag_1', 'pm10_lag_1', 'nox_lag_1',
        'no2_lag_1', 'no_lag_1', 'traffic_volume_lag_2', 'precipitation_lag_2',
        'wind_speed_lag_2', 'air_temperature_lag_2', 'pm2_5_lag_2',
        'pm10_lag_2', 'nox_lag_2', 'no2_lag_2', 'no_lag_2',
        'traffic_volume_lag_3', 'precipitation_lag_3', 'wind_speed_lag_3',
        'air_temperature_lag_3', 'pm2_5_lag_3', 'pm10_lag_3', 'nox_lag_3',
        'no2_lag_3', 'no_lag_3', 'traffic_volume_lag_6', 'precipitation_lag_6',
        'wind_speed_lag_6', 'air_temperature_lag_6', 'pm2_5_lag_6',
        'pm10_lag_6', 'nox_lag_6', 'no2_lag_6', 'no_lag_6',
        'traffic_volume_lag_12', 'precipitation_lag_12', 'wind_speed_lag_12',
        'air_temperature_lag_12', 'pm2_5_lag_12', 'pm10_lag_12', 'nox_lag_12',
        'no2_lag_12', 'no_lag_12'
    ]
    
    y_col = ['pm10','pm2_5', 'no','no2', 'nox']
    
    # Imputing values
    imputer_X = KNNImputer(n_neighbors = 5)
    imputer_X.fit(df[X_cols])
    df[X_cols] = imputer_X.transform(df[X_cols])
    
    imputer_y = KNNImputer(n_neighbors = 5)
    imputer_y.fit(df[y_col])
    df[y_col] = imputer_y.transform(df[y_col])
    
    # StandardScaling for df_avg
    s_scaler = StandardScaler()
    s_scaler.fit(df[X_cols])
    df[X_cols] = s_scaler.transform(df[X_cols])
    
    # Splitting X and y
    
    X = np.c_[df[X_cols]]
    y = np.c_[df[y_col]]

    return X, y, X_pred, y_pred