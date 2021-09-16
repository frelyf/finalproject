import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from etl.datamarts.view_import_functions import get_df_prediction_test, get_df_simple, get_df_with_lags, get_df_with_lags_per_area, get_dates


def get_dnn_test_train(basis = None, latitude_category = False, longitude_category = False):
    if latitude_category == False and longitude_category == False:
        df = get_df_with_lags()
    
    elif latitude_category == False and longitude_category == True:
        df = get_df_with_lags_per_area('longitude')
        
        geo_ohe = OneHotEncoder(categories = 'auto')
        geo_ohe.fit(df[['traffic_geo']])
        feature_arr = geo_ohe.transform(df[['traffic_geo']]).toarray()
        feature_labels = np.array(['west','center','east'])
        features = pd.DataFrame(feature_arr, columns=feature_labels)
        df = df.join(features)
        df = df.drop(['traffic_geo'], axis = 1)
    
    elif latitude_category == True and longitude_category == False:
        df = get_df_with_lags_per_area('latitude')
        
        geo_ohe = OneHotEncoder(categories = 'auto')
        geo_ohe.fit(df[['traffic_geo']])
        feature_arr = geo_ohe.transform(df[['traffic_geo']]).toarray()
        feature_labels = np.array(['north','south'])
        features = pd.DataFrame(feature_arr, columns=feature_labels)
        df = df.join(features)
        df = df.drop(['traffic_geo'], axis = 1)

    
    elif latitude_category == True and longitude_category == True:
        df = get_df_with_lags_per_area('latitude_and_longitude')
        
        geo_ohe = OneHotEncoder(categories = 'auto')
        geo_ohe.fit(df[['traffic_geo_lat','traffic_geo_lon']])
        feature_arr = geo_ohe.transform(df[['traffic_geo_lat','traffic_geo_lon']]).toarray()
        feature_labels = np.array(['north','south','west','center','east'])
        features = pd.DataFrame(feature_arr, columns=feature_labels)
        df = df.join(features)
        df = df.drop(['traffic_geo_lat','traffic_geo_lon'], axis = 1)
    
    else:
        print('Invalid input. Please only use boolean values')
        
    df = df.sort_values(by='dateid_serial', ignore_index = True)
    df = df.loc[df['dateid_serial'] >= 20180101]


    X_cols = [
        'traffic_volume_lag_1', 'precipitation_lag_1',
        'air_temperature_lag_1', 'pm2_5_lag_1', 'pm10_lag_1', 'nox_lag_1',
        'no2_lag_1', 'no_lag_1', 'traffic_volume_lag_2', 'precipitation_lag_2', 
        'air_temperature_lag_2', 'pm2_5_lag_2',
        'pm10_lag_2', 'nox_lag_2', 'no2_lag_2', 'no_lag_2',
        'traffic_volume_lag_3', 'precipitation_lag_3',
        'air_temperature_lag_3', 'pm2_5_lag_3', 'pm10_lag_3', 'nox_lag_3',
        'no2_lag_3', 'no_lag_3', 'traffic_volume_lag_6', 'precipitation_lag_6',
        'air_temperature_lag_6', 'pm2_5_lag_6',
        'pm10_lag_6', 'nox_lag_6', 'no2_lag_6', 'no_lag_6',
        'traffic_volume_lag_12', 'precipitation_lag_12',
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

    # Standard Scaling
    mm_scaler = MinMaxScaler()
    mm_scaler.fit(df[X_cols])
    df[X_cols] = mm_scaler.transform(df[X_cols])
    
    # Handling missing wind speed data
    wind_speed = [
        'wind_speed', 'wind_speed_lag_1', 'wind_speed_lag_2', 
        'wind_speed_lag_3', 'wind_speed_lag_6', 'wind_speed_lag_12']
    
    mm_wind = MinMaxScaler()
    mm_wind.fit(df[wind_speed])
    df[wind_speed] = mm_wind.transform(df[wind_speed])
    
    df[wind_speed] = df[wind_speed].replace(np.nan, -1)
    
    df_dates = get_dates()
    df = df.merge(df_dates, how = 'left', on = 'dateid_serial')
    
    # Setting X columns based on what to train on
    if basis == None:
        X_cols = df.columns.difference([
            'dateid_serial', 'traffic_volume', 'precipitation', 'wind_speed',
            'air_temperature', 'pm2_5', 'pm10', 'nox', 'no2', 'no'])
    
    if basis == 'traffic':
        X_cols = [
            'traffic_volume_lag_1', 'traffic_volume_lag_2','traffic_volume_lag_3',
            'traffic_volume_lag_3', 'traffic_volume_lag_6', 'traffic_volume_lag_12',
            'sin','cos','north','south','east','center','west']

    if basis == 'weather':
        X_cols = [
            'precipitation_lag_1', 'air_temperature_lag_1', 'wind_speed_lag_1',
            'precipitation_lag_2', 'air_temperature_lag_2', 'wind_speed_lag_2',
            'precipitation_lag_3', 'air_temperature_lag_3', 'wind_speed_lag_3',
            'precipitation_lag_6', 'air_temperature_lag_6', 'wind_speed_lag_6',
            'precipitation_lag_12', 'air_temperature_lag_12', 'wind_speed_lag_12',
            'sin','cos','north','south','east','center','west']
    if basis == 'weather and traffic':
        X_cols = [
            'traffic_volume_lag_1', 'traffic_volume_lag_2','traffic_volume_lag_3',
            'traffic_volume_lag_3', 'traffic_volume_lag_6', 'traffic_volume_lag_12',
            'precipitation_lag_1', 'air_temperature_lag_1', 'wind_speed_lag_1',
            'precipitation_lag_2', 'air_temperature_lag_2', 'wind_speed_lag_2',
            'precipitation_lag_3', 'air_temperature_lag_3', 'wind_speed_lag_3',
            'precipitation_lag_6', 'air_temperature_lag_6', 'wind_speed_lag_6',
            'precipitation_lag_12', 'air_temperature_lag_12', 'wind_speed_lag_12',
            'sin','cos','north','south','east','center','west']

    
    # Splitting into train and test, X and y
    df_avg_test = df.loc[df['dateid_serial'] >= 20210101]
    df_avg_train = df.loc[df['dateid_serial'] < 20210101]

    X_train = np.c_[df_avg_train[X_cols]]
    X_test = np.c_[df_avg_test[X_cols]]

    y_train = np.c_[df_avg_train[y_col]]
    y_test = np.c_[df_avg_test[y_col]]

    return X_train, X_test, y_train, y_test

 
def get_dnn_X_y_X_pred(basis = None, latitude_category = False, longitude_category = False):
    # Prepping X and y
    if latitude_category == False and longitude_category == False:
        df = get_df_with_lags()

    elif latitude_category == False and longitude_category == True:
        df = get_df_with_lags_per_area('longitude')
        
        geo_ohe = OneHotEncoder(categories = 'auto')
        geo_ohe.fit(df[['traffic_geo']])
        feature_arr = geo_ohe.transform(df[['traffic_geo']]).toarray()
        feature_labels = np.array(['west','center','east'])
        features = pd.DataFrame(feature_arr, columns=feature_labels)
        df = df.join(features)
        df = df.drop(['traffic_geo'], axis = 1)

    elif latitude_category == True and longitude_category == False:
        df = get_df_with_lags_per_area('latitude')
        
        geo_ohe = OneHotEncoder(categories = 'auto')
        geo_ohe.fit(df[['traffic_geo']])
        feature_arr = geo_ohe.transform(df[['traffic_geo']]).toarray()
        feature_labels = np.array(['north','south'])
        features = pd.DataFrame(feature_arr, columns=feature_labels)
        df = df.join(features)
        df = df.drop(['traffic_geo'], axis = 1)


    elif latitude_category == True and longitude_category == True:
        df = get_df_with_lags_per_area('latitude_and_longitude')
        
        
        geo_ohe = OneHotEncoder(categories = 'auto')
        geo_ohe.fit(df[['traffic_geo_lat','traffic_geo_lon']])
        feature_arr = geo_ohe.transform(df[['traffic_geo_lat','traffic_geo_lon']]).toarray()
        feature_labels = np.array(['north','south','west','center','east'])
        features = pd.DataFrame(feature_arr, columns=feature_labels)
        df = df.join(features)
        df = df.drop(['traffic_geo_lat','traffic_geo_lon'], axis = 1)
        
        df_pred = get_df_prediction_test('latitude_and_longitude')
        geo_ohe = OneHotEncoder(categories = 'auto')
        geo_ohe.fit(df_pred[['traffic_geo_lat','traffic_geo_lon']])
        feature_arr = geo_ohe.transform(df_pred[['traffic_geo_lat','traffic_geo_lon']]).toarray()
        feature_labels = np.array(['north','south','west','center','east'])
        features = pd.DataFrame(feature_arr, columns=feature_labels)
        df_pred = df_pred.join(features)
        df_pred = df_pred.drop(['traffic_geo_lat','traffic_geo_lon'], axis = 1)

    else:
        print('Invalid input. Please only use boolean values')
    
    df = df.iloc[364:].reset_index()

    
    X_cols = [
        'traffic_volume_lag_1', 'precipitation_lag_1',
        'air_temperature_lag_1', 'pm2_5_lag_1', 'pm10_lag_1', 'nox_lag_1',
        'no2_lag_1', 'no_lag_1', 'traffic_volume_lag_2', 'precipitation_lag_2', 
        'air_temperature_lag_2', 'pm2_5_lag_2',
        'pm10_lag_2', 'nox_lag_2', 'no2_lag_2', 'no_lag_2',
        'traffic_volume_lag_3', 'precipitation_lag_3',
        'air_temperature_lag_3', 'pm2_5_lag_3', 'pm10_lag_3', 'nox_lag_3',
        'no2_lag_3', 'no_lag_3', 'traffic_volume_lag_6', 'precipitation_lag_6',
        'air_temperature_lag_6', 'pm2_5_lag_6',
        'pm10_lag_6', 'nox_lag_6', 'no2_lag_6', 'no_lag_6',
        'traffic_volume_lag_12', 'precipitation_lag_12',
        'air_temperature_lag_12', 'pm2_5_lag_12', 'pm10_lag_12', 'nox_lag_12',
        'no2_lag_12', 'no_lag_12'
    ]

    y_col = ['pm10','pm2_5', 'no','no2', 'nox']
    
    # Imputing values
    imputer_X = KNNImputer(n_neighbors = 5)
    imputer_X.fit(df[X_cols])
    df[X_cols] = imputer_X.transform(df[X_cols])
    df_pred[X_cols] = imputer_X.transform(df_pred[X_cols])
    imputer_y = KNNImputer(n_neighbors = 5)
    imputer_y.fit(df[y_col])
    df[y_col] = imputer_y.transform(df[y_col])
    
    # StandardScaling for df_avg
    s_scaler = StandardScaler()
    s_scaler.fit(df[X_cols])
    df[X_cols] = s_scaler.transform(df[X_cols])
    df_pred[X_cols] = s_scaler.transform(df_pred[X_cols])
    
    # Handling missing wind speed data
    wind_speed = [
        'wind_speed', 'wind_speed_lag_1', 'wind_speed_lag_2', 
        'wind_speed_lag_3', 'wind_speed_lag_6', 'wind_speed_lag_12']
    
    # mm_wind = MinMaxScaler()
    # mm_wind.fit(df[wind_speed])
    # df[wind_speed] = mm_wind.transform(df[wind_speed])
    # df_pred[wind_speed] = mm_wind.transform(df_pred[wind_speed])
    
    df[wind_speed] = df[wind_speed].replace(np.nan, -1)
    wind_speed = [
        'wind_speed_lag_1', 'wind_speed_lag_2', 
        'wind_speed_lag_3', 'wind_speed_lag_6', 'wind_speed_lag_12']
    df_pred[wind_speed] = df_pred[wind_speed].replace(np.nan, -1)
    
    df_dates = get_dates()
    df = df.merge(df_dates, how = 'left', on = 'dateid_serial')
    df_pred = df_pred.merge(df_dates, how = 'left', on = 'dateid_serial')
    
    # Setting X columns based on what to train on
    if basis == None:
        X_cols = df.columns.difference([
            'index','dateid_serial', 'traffic_volume', 'precipitation', 'wind_speed',
            'air_temperature', 'pm2_5', 'pm10', 'nox', 'no2', 'no'])
    
    if basis == 'traffic':
        X_cols = [
            'traffic_volume_lag_1', 'traffic_volume_lag_2','traffic_volume_lag_3',
            'traffic_volume_lag_3', 'traffic_volume_lag_6', 'traffic_volume_lag_12',
            'sin','cos','north','south','east','center','west']

    if basis == 'weather':
        X_cols = [
            'precipitation_lag_1', 'air_temperature_lag_1', 'wind_speed_lag_1',
            'precipitation_lag_2', 'air_temperature_lag_2', 'wind_speed_lag_2',
            'precipitation_lag_3', 'air_temperature_lag_3', 'wind_speed_lag_3',
            'precipitation_lag_6', 'air_temperature_lag_6', 'wind_speed_lag_6',
            'precipitation_lag_12', 'air_temperature_lag_12', 'wind_speed_lag_12',
            'sin','cos','north','south','east','center','west']
    if basis == 'weather and traffic':
        X_cols = [
            'traffic_volume_lag_1', 'traffic_volume_lag_2','traffic_volume_lag_3',
            'traffic_volume_lag_3', 'traffic_volume_lag_6', 'traffic_volume_lag_12',
            'precipitation_lag_1', 'air_temperature_lag_1', 'wind_speed_lag_1',
            'precipitation_lag_2', 'air_temperature_lag_2', 'wind_speed_lag_2',
            'precipitation_lag_3', 'air_temperature_lag_3', 'wind_speed_lag_3',
            'precipitation_lag_6', 'air_temperature_lag_6', 'wind_speed_lag_6',
            'precipitation_lag_12', 'air_temperature_lag_12', 'wind_speed_lag_12',
            'sin','cos','north','south','east','center','west']
    
    # Splitting X and y
    X = np.c_[df[X_cols]]
    y = np.c_[df[y_col]]
    
    # Prepping X_pred and y_dates
    X_pred = np.c_[df_pred[X_cols]]
    y_dates = np.c_[df_pred['dateid_serial']]
    
    # Return prediction items
    return X, y, X_pred, y_dates

X, y, X_pred, y_dates = get_dnn_X_y_X_pred(basis = None, latitude_category = True, longitude_category = True)

def get_ml_test_train(value, basis = None, latitude_category = False, longitude_category = False):
    if latitude_category == False and longitude_category == False:
        df = get_df_with_lags()
    
    elif latitude_category == False and longitude_category == True:
        df = get_df_with_lags_per_area('longitude')
        
        geo_ohe = OneHotEncoder(categories = 'auto')
        geo_ohe.fit(df[['traffic_geo']])
        feature_arr = geo_ohe.transform(df[['traffic_geo']]).toarray()
        feature_labels = np.array(['west','center','east'])
        features = pd.DataFrame(feature_arr, columns=feature_labels)
        df = df.join(features)
        df = df.drop(['traffic_geo'], axis = 1)
        geo_features = ['west','center','east']
    
    elif latitude_category == True and longitude_category == False:
        df = get_df_with_lags_per_area('latitude')
        
        geo_ohe = OneHotEncoder(categories = 'auto')
        geo_ohe.fit(df[['traffic_geo']])
        feature_arr = geo_ohe.transform(df[['traffic_geo']]).toarray()
        feature_labels = np.array(['north','south'])
        features = pd.DataFrame(feature_arr, columns=feature_labels)
        df = df.join(features)
        df = df.drop(['traffic_geo'], axis = 1)
        geo_features = ['north','south']

    
    elif latitude_category == True and longitude_category == True:
        df = get_df_with_lags_per_area('latitude_and_longitude')
        
        geo_ohe = OneHotEncoder(categories = 'auto')
        geo_ohe.fit(df[['traffic_geo_lat','traffic_geo_lon']])
        feature_arr = geo_ohe.transform(df[['traffic_geo_lat','traffic_geo_lon']]).toarray()
        feature_labels = np.array(['north','south','west','center','east'])
        features = pd.DataFrame(feature_arr, columns=feature_labels)
        df = df.join(features)
        df = df.drop(['traffic_geo_lat','traffic_geo_lon'], axis = 1)
        geo_features = ['north','south','west','center','east']
    
    else:
        print('Invalid input. Please only use boolean values')
        
    df = df.sort_values(by='dateid_serial', ignore_index = True)
    df = df.loc[df['dateid_serial'] >= 20180101]


    X_cols = [
        'traffic_volume', 'precipitation', 'wind_speed',
        'air_temperature', 'traffic_volume_lag_1', 'precipitation_lag_1',
        'air_temperature_lag_1', 'pm2_5_lag_1', 'pm10_lag_1', 'nox_lag_1',
        'no2_lag_1', 'no_lag_1', 'traffic_volume_lag_2', 'precipitation_lag_2', 
        'air_temperature_lag_2', 'pm2_5_lag_2',
        'pm10_lag_2', 'nox_lag_2', 'no2_lag_2', 'no_lag_2',
        'traffic_volume_lag_3', 'precipitation_lag_3',
        'air_temperature_lag_3', 'pm2_5_lag_3', 'pm10_lag_3', 'nox_lag_3',
        'no2_lag_3', 'no_lag_3', 'traffic_volume_lag_6', 'precipitation_lag_6',
        'air_temperature_lag_6', 'pm2_5_lag_6',
        'pm10_lag_6', 'nox_lag_6', 'no2_lag_6', 'no_lag_6',
        'traffic_volume_lag_12', 'precipitation_lag_12',
        'air_temperature_lag_12', 'pm2_5_lag_12', 'pm10_lag_12', 'nox_lag_12',
        'no2_lag_12', 'no_lag_12'
    ]

    y_col = [value]

    # Imputing values
    imputer_X = KNNImputer(n_neighbors = 5)
    imputer_X.fit(df[X_cols])
    df[X_cols] = imputer_X.transform(df[X_cols])

    imputer_y = KNNImputer(n_neighbors = 5)
    imputer_y.fit(df[y_col])
    df[y_col] = imputer_y.transform(df[y_col])

    # Standard Scaling
    mm_scaler = MinMaxScaler()
    mm_scaler.fit(df[X_cols])
    df[X_cols] = mm_scaler.transform(df[X_cols])
    
    # Handling missing wind speed data
    wind_speed = [
        'wind_speed', 'wind_speed_lag_1', 'wind_speed_lag_2', 
        'wind_speed_lag_3', 'wind_speed_lag_6', 'wind_speed_lag_12']
    
    mm_wind = MinMaxScaler()
    mm_wind.fit(df[wind_speed])
    df[wind_speed] = mm_wind.transform(df[wind_speed])
    
    df[wind_speed] = df[wind_speed].replace(np.nan, -1)
    
    df_dates = get_dates()
    df = df.merge(df_dates, how = 'left', on = 'dateid_serial')
    
    # Setting X columns based on what to train on
    if basis == None:
        X_cols = df.columns.difference([
            'dateid_serial', 'pm2_5', 'pm10', 'nox', 'no2', 'no'])
    
    if basis == 'traffic':
        X_cols = [
            'traffic_volume',
            'traffic_volume_lag_1', 'traffic_volume_lag_2','traffic_volume_lag_3',
            'traffic_volume_lag_3', 'traffic_volume_lag_6', 'traffic_volume_lag_12',
            'sin','cos']

    if basis == 'weather':
        X_cols = [
            'precipitation', 'wind_speed','air_temperature',
            'precipitation_lag_1', 'air_temperature_lag_1', 'wind_speed_lag_1',
            'precipitation_lag_2', 'air_temperature_lag_2', 'wind_speed_lag_2',
            'precipitation_lag_3', 'air_temperature_lag_3', 'wind_speed_lag_3',
            'precipitation_lag_6', 'air_temperature_lag_6', 'wind_speed_lag_6',
            'precipitation_lag_12', 'air_temperature_lag_12', 'wind_speed_lag_12',
            'sin','cos']
    if basis == 'weather and traffic':
        X_cols = [
            'traffic_volume', 'precipitation', 'wind_speed', 'air_temperature',
            'traffic_volume_lag_1', 'traffic_volume_lag_2','traffic_volume_lag_3',
            'traffic_volume_lag_3', 'traffic_volume_lag_6', 'traffic_volume_lag_12',
            'precipitation_lag_1', 'air_temperature_lag_1', 'wind_speed_lag_1',
            'precipitation_lag_2', 'air_temperature_lag_2', 'wind_speed_lag_2',
            'precipitation_lag_3', 'air_temperature_lag_3', 'wind_speed_lag_3',
            'precipitation_lag_6', 'air_temperature_lag_6', 'wind_speed_lag_6',
            'precipitation_lag_12', 'air_temperature_lag_12', 'wind_speed_lag_12',
            'sin','cos']
    

    # Splitting into train and test, X and y
    df_avg_test = df.loc[df['dateid_serial'] >= 20210101]
    df_avg_train = df.loc[df['dateid_serial'] < 20210101]

    X_train = np.c_[df_avg_train[X_cols]]
    X_test = np.c_[df_avg_test[X_cols]]

    y_train = np.c_[df_avg_train[y_col]]
    y_test = np.c_[df_avg_test[y_col]]

    return X_train, X_test, y_train, y_test