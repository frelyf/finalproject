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
from analysis.data_prep import get_dnn_test_train, get_dnn_X_y_X_pred

# DNN

X_train, X_test, y_train, y_test = get_dnn_test_train()

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

def results():
    y_mean = np.mean(y_test)*np.ones(y_train.shape)
    dumb_mse = np.sqrt(mean_squared_error(y_mean, y_train))
    print(f'Dumb MSE: {dumb_mse}')
    dumb_mae = mean_absolute_error(y_mean, y_train)
    print(f'Dumb MAE: {dumb_mae}')
    plt.plot(np.sqrt(history.history['loss']))
    plt.plot(np.sqrt(history.history['val_loss']))
    plt.plot(dumb_mse*np.ones(y_train.shape))
    plt.plot(history.history['mae'], label = 'mae')
    plt.plot(history.history['val_mae'], label = 'val_mae')
    plt.plot(dumb_mae*np.ones(y_train.shape))
    plt.legend(['mse','val_mse', 'dumb_mse', 'mae','val_mae', 'dumb_mae'])
    plt.show()

results()
