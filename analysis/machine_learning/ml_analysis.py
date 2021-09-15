from analysis.machine_learning.ml_models import xgb_regressor
from analysis.data_prep import get_ml_test_train
from etl.datamarts.view_import_functions import get_df_with_lags
import json



df = get_df_with_lags()
each = df.columns.difference([
    'dateid_serial', 'pm2_5', 'pm10', 'nox', 'no2', 'no'])

weather_and_traffic = [
    'traffic_volume', 'precipitation', 'wind_speed', 'air_temperature',
    'traffic_volume_lag_1', 'traffic_volume_lag_2','traffic_volume_lag_3',
    'traffic_volume_lag_3', 'traffic_volume_lag_6', 'traffic_volume_lag_12',
    'precipitation_lag_1', 'air_temperature_lag_1', 'wind_speed_lag_1',
    'precipitation_lag_2', 'air_temperature_lag_2', 'wind_speed_lag_2',
    'precipitation_lag_3', 'air_temperature_lag_3', 'wind_speed_lag_3',
    'precipitation_lag_6', 'air_temperature_lag_6', 'wind_speed_lag_6',
    'precipitation_lag_12', 'air_temperature_lag_12', 'wind_speed_lag_12',
    'sin','cos'
]

weather = [
    'precipitation', 'wind_speed','air_temperature',
    'precipitation_lag_1', 'air_temperature_lag_1', 'wind_speed_lag_1',
    'precipitation_lag_2', 'air_temperature_lag_2', 'wind_speed_lag_2',
    'precipitation_lag_3', 'air_temperature_lag_3', 'wind_speed_lag_3',
    'precipitation_lag_6', 'air_temperature_lag_6', 'wind_speed_lag_6',
    'precipitation_lag_12', 'air_temperature_lag_12', 'wind_speed_lag_12',
    'sin','cos'
]

traffic = [
    'traffic_volume',
    'traffic_volume_lag_1', 'traffic_volume_lag_2','traffic_volume_lag_3',
    'traffic_volume_lag_3', 'traffic_volume_lag_6', 'traffic_volume_lag_12',
    'sin','cos'
]


values = ['pm2_5','pm10','no','no2','nox']
basis_list = [None, 'weather and traffic', 'weather', 'traffic']
feature_names_list = [each, weather_and_traffic, weather, traffic]

y_pred_basis_dict = {}
stats_basis_dict = {}
feature_importance_basis_dict = {}




for basis, feature_names in zip(basis_list, feature_names_list):
    y_pred_dict = {}
    stats_dict = {}
    feature_importance_dict = {}
    for value in values:
        X_train, X_test, y_train, y_test = get_ml_test_train(value = value, basis = basis, latitude_category = False, longitude_category = False)
        y_pred, stats, feature_importance = xgb_regressor(X_train, y_train, X_test, y_test)
        
        y_pred_dict[value] = y_pred.tolist()
        stats_dict[value] = stats
        
        feature_importance_named = {}
        for feature_name, importance in zip(feature_names, feature_importance.values()):
            feature_importance_named[feature_name] = importance
        
        feature_importance_dict[value] = feature_importance_named
    
    y_pred_basis_dict[basis] = y_pred_dict
    stats_basis_dict[basis] = stats_dict
    feature_importance_basis_dict[basis] = feature_importance_dict
json_dict = {
    'y_pred': y_pred_basis_dict,
    'stats': stats_basis_dict,
    'feature_importance': feature_importance_basis_dict}

with open(r'files\analysis.json', 'w') as outfile:
    json.dump(json_dict, outfile)