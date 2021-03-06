import psycopg2
import pandas as pd
import datetime as dt
import math
import numpy as np

def get_connection():
    connection = psycopg2.connect(
                host="trafikkluft.postgres.database.azure.com",                
                port="5432",
                user="postgres@trafikkluft",                
                password="Awesome1337",                
                database="postgres",            
            )
    return connection

def get_df_simple(aggregator):
    engine = get_connection()
    query = f"""select * from {aggregator}_values_per_date"""
    df = pd.read_sql_query(query, con = engine)
    return df


def get_df_with_lags():
    engine = get_connection()
    
    #Queries
    avg_query = """select * from avg_values_per_date"""
    month_lag_query = """select * from avg_values_month_lag"""
    bi_month_lag_query = """select * from avg_values_bi_month_lag"""
    qrt_year_lag_query = """select * from avg_values_qrt_year_lag"""
    half_year_lag_query = """select * from avg_values_half_year_lag"""
    year_lag_query = """select * from avg_values_year_lag"""

    # Import dataframes
    df = pd.read_sql_query(avg_query, con = engine)
    df_month_lag = pd.read_sql_query(month_lag_query, con = engine)
    df_bi_month_lag = pd.read_sql_query(bi_month_lag_query, con = engine)
    df_qrt_year_lag = pd.read_sql_query(qrt_year_lag_query, con = engine)
    df_half_year_lag = pd.read_sql_query(half_year_lag_query, con = engine)
    df_yearlag = pd.read_sql_query(year_lag_query, con = engine)
    
    # Remove dateid_serial from lags and rename to dateid_serial for merging
    del df_month_lag['dateid_serial']
    del df_bi_month_lag['dateid_serial']
    del df_qrt_year_lag['dateid_serial']
    del df_half_year_lag['dateid_serial']
    del df_yearlag['dateid_serial']
    
    df_month_lag['dateid_serial'] = df_month_lag['month_lag']
    df_bi_month_lag['dateid_serial'] = df_bi_month_lag['bi_month_lag']
    df_qrt_year_lag['dateid_serial'] = df_qrt_year_lag['qrt_year_lag']
    df_half_year_lag['dateid_serial'] = df_half_year_lag['half_year_lag']
    df_yearlag['dateid_serial'] = df_yearlag['year_lag']
    
    del df_month_lag['month_lag']
    del df_bi_month_lag['bi_month_lag']
    del df_qrt_year_lag['qrt_year_lag']
    del df_half_year_lag['half_year_lag']
    del df_yearlag['year_lag']
    
    # Merge lags
    df = df.merge(df_month_lag, how = 'left', on = 'dateid_serial')
    df = df.merge(df_bi_month_lag, how = 'left', on = 'dateid_serial')
    df = df.merge(df_qrt_year_lag, how = 'left', on = 'dateid_serial')
    df = df.merge(df_half_year_lag, how = 'left', on = 'dateid_serial')
    df = df.merge(df_yearlag, how = 'left', on = 'dateid_serial')
    
    #Return df
    return df


def get_df_with_lags_per_area(coordinate):    
    engine = get_connection()

    # Queries
    month_lag_query = f"""select * from avg_values_month_lag_{coordinate}"""
    bi_month_lag_query = f"""select * from avg_values_bi_month_lag_{coordinate}"""
    qrt_year_lag_query = f"""select * from avg_values_qrt_year_lag_{coordinate}"""
    half_year_lag_query = f"""select * from avg_values_half_year_lag_{coordinate}"""
    year_lag_query = f"""select * from avg_values_year_lag_{coordinate}"""
    avg_query = f"""select * from avg_values_per_{coordinate}"""
    
    # Import dataframes
    df = pd.read_sql_query(avg_query, con = engine)
    df_month_lag = pd.read_sql_query(month_lag_query, con = engine)
    df_bi_month_lag = pd.read_sql_query(bi_month_lag_query, con = engine)
    df_qrt_year_lag = pd.read_sql_query(qrt_year_lag_query, con = engine)
    df_half_year_lag = pd.read_sql_query(half_year_lag_query, con = engine)
    df_yearlag = pd.read_sql_query(year_lag_query, con = engine)
    
    # Remove dateid_serial from lags and rename to dateid_serial for merging
    del df_month_lag['dateid_serial']
    del df_bi_month_lag['dateid_serial']
    del df_qrt_year_lag['dateid_serial']
    del df_half_year_lag['dateid_serial']
    del df_yearlag['dateid_serial']
    
    df_month_lag['dateid_serial'] = df_month_lag['month_lag']
    df_bi_month_lag['dateid_serial'] = df_bi_month_lag['month_lag']
    df_qrt_year_lag['dateid_serial'] = df_qrt_year_lag['month_lag']
    df_half_year_lag['dateid_serial'] = df_half_year_lag['month_lag']
    df_yearlag['dateid_serial'] = df_yearlag['month_lag']
    
    del df_month_lag['month_lag']
    del df_bi_month_lag['month_lag']
    del df_qrt_year_lag['month_lag']
    del df_half_year_lag['month_lag']
    del df_yearlag['month_lag']
    
    # Merge lags
    if 'traffic_geo' in df.columns:
        df = df.merge(df_month_lag, how = 'left', on=['dateid_serial', 'traffic_geo'])
        df = df.merge(df_bi_month_lag, how = 'left', on=['dateid_serial', 'traffic_geo'])
        df = df.merge(df_qrt_year_lag, how = 'left', on=['dateid_serial', 'traffic_geo'])
        df = df.merge(df_half_year_lag, how = 'left', on=['dateid_serial', 'traffic_geo'])
        df = df.merge(df_yearlag, how = 'left', on=['dateid_serial', 'traffic_geo'])
    else:
        df = df.merge(df_month_lag, how = 'left', on=['dateid_serial', 'traffic_geo_lat','traffic_geo_lon'])
        df = df.merge(df_bi_month_lag, how = 'left', on=['dateid_serial', 'traffic_geo_lat','traffic_geo_lon'])
        df = df.merge(df_qrt_year_lag, how = 'left', on=['dateid_serial', 'traffic_geo_lat','traffic_geo_lon'])
        df = df.merge(df_half_year_lag, how = 'left', on=['dateid_serial', 'traffic_geo_lat','traffic_geo_lon'])
        df = df.merge(df_yearlag, how = 'left', on=['dateid_serial', 'traffic_geo_lat','traffic_geo_lon'])

    #Return df
    return df


def get_df_prediction_test(coordinate):
    engine = get_connection()
    
    query = 'select * from prediction_test'
    month_lag_query = f"""select * from avg_values_month_lag_{coordinate}"""
    bi_month_lag_query = f"""select * from avg_values_bi_month_lag_{coordinate}"""
    qrt_year_lag_query = f"""select * from avg_values_qrt_year_lag_{coordinate}"""
    half_year_lag_query = f"""select * from avg_values_half_year_lag_{coordinate}"""
    year_lag_query = f"""select * from avg_values_year_lag_{coordinate}"""
    avg_query = f"""select * from avg_values_per_{coordinate}"""
    
    # Import dataframes
    df = pd.read_sql_query(query, con = engine)
    df_month_lag = pd.read_sql_query(month_lag_query, con = engine)
    df_bi_month_lag = pd.read_sql_query(bi_month_lag_query, con = engine)
    df_qrt_year_lag = pd.read_sql_query(qrt_year_lag_query, con = engine)
    df_half_year_lag = pd.read_sql_query(half_year_lag_query, con = engine)
    df_yearlag = pd.read_sql_query(year_lag_query, con = engine)
    
    # Remove dateid_serial from lags and rename to dateid_serial for merging
    del df_month_lag['dateid_serial']
    del df_bi_month_lag['dateid_serial']
    del df_qrt_year_lag['dateid_serial']
    del df_half_year_lag['dateid_serial']
    del df_yearlag['dateid_serial']
    
    df_month_lag['dateid_serial'] = df_month_lag['month_lag']
    df_bi_month_lag['dateid_serial'] = df_bi_month_lag['month_lag']
    df_qrt_year_lag['dateid_serial'] = df_qrt_year_lag['month_lag']
    df_half_year_lag['dateid_serial'] = df_half_year_lag['month_lag']
    df_yearlag['dateid_serial'] = df_yearlag['month_lag']
    
    del df_month_lag['month_lag']
    del df_bi_month_lag['month_lag']
    del df_qrt_year_lag['month_lag']
    del df_half_year_lag['month_lag']
    del df_yearlag['month_lag']
    
    # Merge lags
    if 'traffic_geo' in df.columns:
        df = df.merge(df_month_lag, how = 'left', on=['dateid_serial', 'traffic_geo'])
        df = df.merge(df_bi_month_lag, how = 'left', on=['dateid_serial', 'traffic_geo'])
        df = df.merge(df_qrt_year_lag, how = 'left', on=['dateid_serial', 'traffic_geo'])
        df = df.merge(df_half_year_lag, how = 'left', on=['dateid_serial', 'traffic_geo'])
        df = df.merge(df_yearlag, how = 'left', on=['dateid_serial', 'traffic_geo'])
    else:
        df = df.merge(df_month_lag, how = 'left', on=['dateid_serial'])
        df = df.merge(df_bi_month_lag, how = 'left', on=['dateid_serial', 'traffic_geo_lat','traffic_geo_lon'])
        df = df.merge(df_qrt_year_lag, how = 'left', on=['dateid_serial', 'traffic_geo_lat','traffic_geo_lon'])
        df = df.merge(df_half_year_lag, how = 'left', on=['dateid_serial', 'traffic_geo_lat','traffic_geo_lon'])
        df = df.merge(df_yearlag, how = 'left', on=['dateid_serial', 'traffic_geo_lat','traffic_geo_lon'])
    
    return df

df = get_df_prediction_test('latitude_and_longitude')

def get_dates():
    engine = get_connection()
    query = 'select dateid_serial, date_actual, year_actual from dim_date;'
    df = pd.read_sql_query(query, con = engine)
    
    def create_new_year(int):
        year = str(int)
        year = f'{year}-01-01'
        return year
        
    year = dt.timedelta(days=365)
    df['year_actual'] = df['year_actual'].apply(create_new_year)
    df['year_actual'] = pd.to_datetime(df['year_actual'])
    df['date_actual'] = pd.to_datetime(df['date_actual'])
    df['seconds'] = ((df['date_actual'] - df['year_actual'])/year)*2*math.pi
    
    df['sin'] = np.sin(df['seconds'])
    df['cos'] = np.cos(df['seconds'])
    df = df.drop(['date_actual','year_actual','seconds'], axis = 1)
    
    return df
