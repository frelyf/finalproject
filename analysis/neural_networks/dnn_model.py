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
second_hidden_layer = Dense(98, activation = 'relu')(input_layer)
third_hidden_layer = Dense(64, activation = 'relu')(second_hidden_layer)
first_regularization_layer = Dense(128, bias_regularizer=regularizers.l2(l2=120), #140 #120
                                   kernel_regularizer=regularizers.L1(l1=19), 
                                   activity_regularizer=regularizers.l1_l2(l1=190, l2=190))(third_hidden_layer)
fourth_hidden_layer = Dense(78, activation = 'relu')(first_regularization_layer)
fifht_hidden_layer = Dense(38, activation = 'relu')(fourth_hidden_layer)
sixth_hidden_layer = Dense(12, activation = 'relu')(fifht_hidden_layer)
output_layer = Dense(5,)(sixth_hidden_layer)

second_hidden_layer2 = Dense(300, activation='relu')(input_layer)
first_dropout_layer2 = Dropout(0.5)(second_hidden_layer2)
third_hidden_layer2 = Dense(200, activation = 'relu')(first_dropout_layer2)
second_dropout_layer2 = Dropout(0.4)(third_hidden_layer2)
fourth_hidden_layer2 = Dense(78, activation = 'relu')(second_dropout_layer2)
fifht_hidden_layer2 = Dense(38, activation = 'relu')(fourth_hidden_layer2)
output_layer2 = Dense(5,)(fifht_hidden_layer2)

output_layer3 = Concatenate(axis=-1)([output_layer, output_layer2])
#output_final = Dense(5)(output_layer3)

first_combined_layer = Dense(64, activation = 'relu')(output_layer3)
second_combined_layer = Dense(20, activation = 'relu')(first_combined_layer)
combined_output_layer = Dense(5,)(second_combined_layer)

model = Model(inputs = input_layer, outputs = combined_output_layer)
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
history = model.fit(
    X_train, 
    y_train, 
    batch_size=640, 
    epochs=10000,
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
