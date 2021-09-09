import psycopg2
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
import datetime as dt
import sklearn
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
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
from keras_tuner import RandomSearch, Hyperband
from keras import backend as K
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def get_connection():
    connection = psycopg2.connect(
                host="trafikkluft.postgres.database.azure.com",                
                port="5432",
                user="postgres@trafikkluft",                
                password="Awesome1337",                
                database="postgres",            
            )
    return connection

engine = get_connection()

max_query = """
    select * from max_values_per_date
"""

min_query = """
    select * from min_values_per_date
"""

avg_query = """
    select * from avg_values_per_date
"""

df_max = pd.read_sql_query(max_query, con = engine)
df_min = pd.read_sql_query(min_query, con = engine)
df_avg = pd.read_sql_query(avg_query, con = engine)

# Log scaling for df_avg
# df_avg['precipitation'] = np.log(df_avg['precipitation'])

# StandardScaling for df_avg
s_scaler = StandardScaler()
s_scaler.fit(df_avg[['traffic_volume','wind_speed','air_temperature', 'precipitation']])

df_avg[['traffic_volume','wind_speed','air_temperature', 'precipitation']] = s_scaler.transform(df_avg[['traffic_volume','wind_speed','air_temperature', 'precipitation']])
df_avg.hist(bins = 50)

# Splitting into train and test, X and y
df_avg_test = df_avg.iloc[:365]
df_avg_train = df_avg.iloc[365:]

X_train = np.c_[df_avg_train[['traffic_volume', 'precipitation','wind_speed','air_temperature']]]
X_test = np.c_[df_avg_test[['traffic_volume', 'precipitation','wind_speed','air_temperature']]]

pm2_5_train = np.c_[df_avg_train['pm2_5']]
pm2_5_test = np.c_[df_avg_test['pm2_5']]

pm10_train = np.c_[df_avg_train['pm10']]
pm10_test = np.c_[df_avg_test['pm10']]

nox_train = np.c_[df_avg_train['nox']]
nox_test = np.c_[df_avg_test['nox']]

no_train = np.c_[df_avg_train['no']]
no_test = np.c_[df_avg_test['no']]

no2_train = np.c_[df_avg_train['no2']]
no2_test = np.c_[df_avg_test['no2']]

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

    corr_matrix = df_avg_train.corr()
    
    print(f'''
    Test MSE = {mse_test}
    Train MSE = {mse_train}
    Test MAE = {mae_test}
    Train MAE = {mae_train}
    Correlation matrix:
    {corr_matrix}
          ''')
          
multivariate_regressor(X_train, nox_train, X_test, nox_test)

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
    
    corr_matrix = df_avg_train.corr()
    
    print(f'''
    Test MSE = {mse_test}
    Train MSE = {mse_train}
    Test MAE = {mae_test}
    Train MAE = {mae_train}
    Correlation matrix:
    {corr_matrix}
          ''')
          
kneighborsregressor(X_train, nox_train, X_test, nox_test)

# XGBoost

def xgb_regressor(X_train, y_train, X_test, y_test):
    xgbr_model = xgb.XGBRegressor()
    xgbr_model.fit(X_train, y_train, eval_set = [(X_test, y_test)], early_stopping_rounds = 100)
    
    print('R Squared for training data:')
    print(xgbr_model.score(X_train, y_train))
    print('R Squared for test data:')
    print(xgbr_model.score(X_test, y_test))
    
    y_pred = xgbr_model.predict(X_test)
    y_pred_train = xgbr_model.predict(X_train)
    
    mse_train = np.sqrt(mean_squared_error(y_pred_train, y_train))
    mae_train = mean_absolute_error(y_pred_train, y_train)
    
    mse_test = np.sqrt(mean_squared_error(y_pred, y_test))
    mae_test = mean_absolute_error(y_pred, y_test)
    
    print(f'''
    Test MSE = {mse_test}
    Train MSE = {mse_train}
    Test MAE = {mae_test}
    Train MAE = {mae_train}
            ''')

xgb_regressor(X_train, nox_train, X_test, nox_test)


# DNN

input_layer = Input(shape = (4,))
second_hidden_layer = Dense(32, activation = 'relu')(input_layer)
first_dropout_layer = Dropout(0.2)(second_hidden_layer)
third_hidden_layer = Dense(16, activation = 'relu')(first_dropout_layer)
first_regularization_layer = Dense(8, kernel_regularizer = regularizers.l2(0.01))(third_hidden_layer)
second_dropout_layer = Dropout(0.1)(first_regularization_layer)
fourth_hidden_layer = Dense(4, activation = 'relu')(second_dropout_layer)
output_layer = Dense(1)(fourth_hidden_layer)

model = Model(inputs = input_layer, outputs = output_layer)
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
history = model.fit(
    X_train, 
    nox_train, 
    batch_size=32, 
    epochs=750,
    validation_data = (X_test, nox_test))


def plot_mse():
    plt.plot(np.sqrt(history.history['loss']))
    plt.plot(np.sqrt(history.history['val_loss']))
    plt.legend(['loss','val_loss'])
    plt.show()

plot_mse()

def plot_mae():
    plt.plot(history.history['mae'], label = 'mae')
    plt.plot(history.history['val_mae'], label = 'val_mae')
    plt.legend(['mae','val_mae'])
    plt.show()

plot_mae()