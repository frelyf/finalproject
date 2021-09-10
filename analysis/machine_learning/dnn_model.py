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

df_avg = get_df_with_lags()
df_avg = df_avg.sort_values(by='dateid_serial', ignore_index = True)
df_avg = df_avg.iloc[364:].reset_index()


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
imputer_X.fit(df_avg[X_cols])
df_avg[X_cols] = imputer_X.transform(df_avg[X_cols])

imputer_y = KNNImputer(n_neighbors = 5)
imputer_y.fit(df_avg[y_col])
df_avg[y_col] = imputer_y.transform(df_avg[y_col])

# StandardScaling for df_avg
s_scaler = StandardScaler()
s_scaler.fit(df_avg[X_cols])
df_avg[X_cols] = s_scaler.transform(df_avg[X_cols])

# Splitting into train and test, X and y
df_avg_test = df_avg.iloc[:365]
df_avg_train = df_avg.iloc[365:]

X_train = np.c_[df_avg_train[X_cols]]
X_test = np.c_[df_avg_test[X_cols]]

y_train = np.c_[df_avg_train[y_col]]
y_test = np.c_[df_avg_test[y_col]]


# DNN

input_layer = Input(shape = (49,))
normlization_layer = BatchNormalization()(input_layer)
second_hidden_layer = Dense(128, activation = 'relu')(normlization_layer)
first_dropout_layer = Dropout(0.2)(second_hidden_layer)
third_hidden_layer = Dense(64, activation = 'relu')(first_dropout_layer)
first_regularization_layer = Dense(32)(third_hidden_layer)
second_dropout_layer = Dropout(0.2)(first_regularization_layer)
fourth_hidden_layer = Dense(16, activation = 'relu')(second_dropout_layer)
output_layer = Dense(5)(fourth_hidden_layer)

model = Model(inputs = input_layer, outputs = output_layer)
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
history = model.fit(
    X_train, 
    y_train, 
    batch_size=256, 
    epochs=2000,
    validation_data = (X_test, y_test))


def plot():
    plt.plot(np.sqrt(history.history['loss']))
    plt.plot(np.sqrt(history.history['val_loss']))
    plt.plot(history.history['mae'], label = 'mae')
    plt.plot(history.history['val_mae'], label = 'val_mae')
    plt.legend(['mae','val_mae', 'loss','val_loss'])
    plt.show()

plot()
