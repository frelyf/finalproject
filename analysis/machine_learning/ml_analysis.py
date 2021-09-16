from analysis.machine_learning.ml_models import xgb_regressor
from analysis.data_prep import get_ml_test_train
from etl.datamarts.view_import_functions import get_df_with_lags
from sklearn.preprocessing import StandardScaler
import json
import matplotlib.pyplot as plt
import pandas as pd

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

y_true_dict = {}    
for value in values:
    X_train, X_test, y_train, y_test = get_ml_test_train(value = value, basis = basis, latitude_category = False, longitude_category = False)
    y_true_dict[value] = y_test.flatten().tolist()
y_pred_basis_dict['true_values'] = y_true_dict
    
json_dict = {
    'y_pred': y_pred_basis_dict,
    'stats': stats_basis_dict,
    'feature_importance': feature_importance_basis_dict}

# with open(r'files\analysis.json', 'w') as outfile:
#     json.dump(json_dict, outfile)


# Creating dataframes with predictions and list of dataframes
df_pred_all = pd.DataFrame(y_pred_basis_dict[None])
df_pred_wt = pd.DataFrame(y_pred_basis_dict['weather and traffic']) 
df_pred_w= pd.DataFrame(y_pred_basis_dict['weather'])
df_true= pd.DataFrame(y_pred_basis_dict['true_values'])
df_pred_t= pd.DataFrame(y_pred_basis_dict['traffic'])

df_list = [df_pred_all, df_pred_wt, df_pred_w, df_pred_t]


# Plotting models against each other
def plot_all():
    for value in values:
        for df in df_list:
            plt.plot(df[value])
        
    plt.show()

def plot_true_and_pred():
    basis_list = ["All features", 'weather and traffic', 'weather', 'true','traffic']
    for basis, df in zip(basis_list,df_list):
        plt.cla()
        plt.scatter(df_true['nox'], df['nox'])
        plt.ylabel('True')
        plt.xlabel(basis)
        
        x = [i for i in range(int(df_true.max().max()))]
        y = [i for i in range(int(df_true.max().max()))]
    
        plt.plot(x,y, '--', color = '#1C2833')
        plt.savefig(f'files\plots\{basis}')
        
# Scaling predictions for comparison
df_scaling = get_df_with_lags()[['pm2_5', 'pm10', 'no', 'no2', 'nox']]
pred_scaler = StandardScaler()
pred_scaler.fit(df_scaling)

df_pred_all = pd.DataFrame(pred_scaler.transform(df_pred_all), columns = ['pm2_5', 'pm10', 'no', 'no2', 'nox'])
df_pred_wt = pd.DataFrame(pred_scaler.transform(df_pred_wt), columns = ['pm2_5', 'pm10', 'no', 'no2', 'nox'])
df_pred_w = pd.DataFrame(pred_scaler.transform(df_pred_w), columns = ['pm2_5', 'pm10', 'no', 'no2', 'nox'])
df_true = pd.DataFrame(pred_scaler.transform(df_true), columns = ['pm2_5', 'pm10', 'no', 'no2', 'nox'])
df_pred_t = pd.DataFrame(pred_scaler.transform(df_pred_t), columns = ['pm2_5', 'pm10', 'no', 'no2', 'nox'])

df_pred_all_error = df_true.subtract(df_pred_all).abs().sum(axis = 1).to_frame(name = 'all_features')
df_pred_wt_error = df_true.subtract(df_pred_wt).abs().sum(axis = 1).to_frame(name = 'weather_traffic')
df_pred_w_error = df_true.subtract(df_pred_w).abs().sum(axis = 1).to_frame(name = 'weather')
df_pred_t_error = df_true.subtract(df_pred_t).abs().sum(axis = 1).to_frame(name = 'traffic')

errors = pd.concat([df_pred_all_error, df_pred_wt_error, df_pred_w_error, df_pred_t_error], axis = 1)
errors.plot()

errors_smooth = errors.rolling(9).mean()
errors_smooth = errors_smooth.iloc[::9, :].reset_index(drop = True)

days = pd.date_range('2021-01-01', periods = 28, freq = '10d').to_frame(name = 'date').reset_index().drop(columns = ['index'])
errors_smooth = pd.concat([errors_smooth, days], axis = 1).set_index('date', drop = True)
errors_smooth.plot()

# Finding worst and best days for each model
m_min = errors.eq(errors.min(axis=1), 0)
min_error = m_min.dot(errors.columns + ',').str.rstrip(',').value_counts()
min_error = min_error.apply(lambda x: 100 * x / float(min_error.sum()))

m_max = errors.eq(errors.max(axis=1), 0)
max_error = m_max.dot(errors.columns + ',').str.rstrip(',').value_counts()
max_error = max_error.apply(lambda x: 100 * x / float(max_error.sum()))