import psycopg2
import pandas as pd
import datetime as dt
import math
import numpy as np
import datetime as dt
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
query = f"""select * from october_predicted"""
df = pd.read_sql_query(query, con = engine)

df = df.drop('index', axis = 1)

def date_converter(number):
    string = str(number)
    year = string[:4]
    month = string[4:6]
    day = string[6:8]
    return f'{year}-{month}-{day}'
    

df['dateid_serial'] = df['dateid_serial'].apply(date_converter)
df['dateid_serial'] = pd.to_datetime(df['dateid_serial'])

north_west = df.loc[(df['north'] == 1) & (df['west'] == 1)].drop(['north','south','east','center','west'], axis = 1).reset_index(drop = True)
north_center = df.loc[(df['north'] == 1) & (df['center'] == 1)].drop(['north','south','east','center','west'], axis = 1).reset_index(drop = True)
north_east = df.loc[(df['north'] == 1) & (df['east'] == 1)].drop(['north','south','east','center','west'], axis = 1).reset_index(drop = True)
south_west = df.loc[(df['south'] == 1) & (df['west'] == 1)].drop(['north','south','east','center','west'], axis = 1).reset_index(drop = True)
south_center = df.loc[(df['south'] == 1) & (df['center'] == 1)].drop(['north','south','east','center','west'], axis = 1).reset_index(drop = True)
south_east = df.loc[(df['south'] == 1) & (df['east'] == 1)].drop(['north','south','east','center','west'], axis = 1).reset_index(drop = True)

north_west = north_west[['dateid_serial','pm10', 'pm2_5', 'no', 'no2', 'nox']]
north_center = north_center[['dateid_serial','pm10', 'pm2_5', 'no', 'no2', 'nox']]
north_east = north_east[['dateid_serial','pm10', 'pm2_5', 'no', 'no2', 'nox']]
south_west = south_west[['dateid_serial','pm10', 'pm2_5', 'no', 'no2', 'nox']]
south_center = south_center[['dateid_serial','pm10', 'pm2_5', 'no', 'no2', 'nox']]
south_east = south_east[['dateid_serial','pm10', 'pm2_5', 'no', 'no2', 'nox']]

df_list = [north_west, north_center, north_east, south_west, south_center, south_east]

limits = {
    'pm2_5': [30,50, 150],
    'pm10':[60,120, 400],
    'no':[100,200, 400],
    'no2':[100,200, 400],
    'nox':[100,180, 240]    
}

def plot_prediction(value):   
    for df in df_list:
        plt.plot(df[value])
    plt.axhline(y = limits[value][0], color = 'g', linestyle = '--')
    plt.axhline(y = limits[value][1], color = 'y', linestyle = '--')
    # plt.axhline(y = limits[value][2], color = 'r', linestyle = '--')
    plt.legend(['north_west', 'north_center', 'north_east', 'south_west', 'south_center','south_east'])
    plt.title(value)
    plt.show()

plot_prediction('pm2_5')
plot_prediction('pm10')
plot_prediction('no')
plot_prediction('no2')
plot_prediction('nox')

df_all = pd.concat(df_list)

arrays = [
    ['north_west', 'north_center', 'north_east', 'south_west', 'south_center','south_east'],
    [x for x in range(30)]    
]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_product(arrays, names = ['place','day'])

df_all = df_all.set_index(index)
df_all_unstack = df_all.unstack(0)
df_pm10 = df_all_unstack.iloc[:,6:12].reset_index(drop = True)
df_pm10.columns = ['north_west', 'north_center', 'north_east', 'south_west', 'south_center','south_east']
df_pm2_5 = df_all_unstack.iloc[:,12:18]
df_pm2_5.columns = ['north_west', 'north_center', 'north_east', 'south_west', 'south_center','south_east']
df_no = df_all_unstack.iloc[:,18:24]
df_no.columns = ['north_west', 'north_center', 'north_east', 'south_west', 'south_center','south_east']
df_no2 = df_all_unstack.iloc[:,24:30]
df_no2.columns = ['north_west', 'north_center', 'north_east', 'south_west', 'south_center','south_east']
df_nox = df_all_unstack.iloc[:,30:]
df_nox.columns = ['north_west', 'north_center', 'north_east', 'south_west', 'south_center','south_east']
df_list_unstack = [df_pm10, df_pm2_5, df_no, df_no2, df_nox]

df_best = pd.DataFrame([], index = ['north_west', 'north_center', 'north_east', 'south_west', 'south_center','south_east'])
for df in df_list_unstack:
    m_min = df.eq(df.min(axis=1), 0)
    min_error = m_min.dot(df.columns + ',').str.rstrip(',').value_counts()
    min_error = min_error.apply(lambda x: 100 * x / float(min_error.sum()))
    df_best = pd.concat([df_best, min_error], axis = 1)
df_best.columns = ['pm10', 'pm2_5', 'no', 'no2', 'nox']
df_best = df_best.fillna(0)
df_best_avg = df_best.mean(axis = 1)

df_worst = pd.DataFrame([], index = ['north_west', 'north_center', 'north_east', 'south_west', 'south_center','south_east'])
for df in df_list_unstack:
    m_max = df.eq(df.max(axis=1), 0)
    max_error = m_max.dot(df.columns + ',').str.rstrip(',').value_counts()
    max_error = max_error.apply(lambda x: 100 * x / float(max_error.sum()))
    df_worst = pd.concat([df_worst, max_error], axis = 1)
df_worst.columns = ['pm10', 'pm2_5', 'no', 'no2', 'nox']
df_worst = df_worst.fillna(0)
df_worst_avg = df_worst.mean(axis = 1)

df_score = pd.concat([df_best_avg, df_worst_avg], axis = 1)
return df_score