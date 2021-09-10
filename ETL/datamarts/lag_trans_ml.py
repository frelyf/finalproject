import psycopg2
import pandas as pd
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

def get_df_with_lags():
    engine = get_connection()
    
    month_lag_query = """select * from avg_values_month_lag"""
    bi_month_lag_query = """select * from avg_values_bi_month_lag"""
    qrt_year_lag_query = """select * from avg_values_qrt_year_lag"""
    half_year_lag_query = """select * from avg_values_half_year_lag"""
    year_lag_query = """select * from avg_values_year_lag"""
    avg_query = """select * from avg_values_per_date"""
    
    # Import dataframes
    df = pd.read_sql_query(avg_query, con = engine)
    df_month_lag = pd.read_sql_query(month_lag_query, con = engine)
    df_bi_month_lag = pd.read_sql_query(bi_month_lag_query, con = engine)
    df_qrt_year_lag = pd.read_sql_query(qrt_year_lag_query, con = engine)
    df_half_year_lag = pd.read_sql_query(half_year_lag_query, con = engine)
    df_yearlag = pd.read_sql_query(year_lag_query, con = engine)
    
    # Remove dateid_serial from lags
    del df_month_lag['dateid_serial']
    del df_bi_month_lag['dateid_serial']
    del df_qrt_year_lag['dateid_serial']
    del df_half_year_lag['dateid_serial']
    del df_yearlag['dateid_serial']
    
    # Merge lags
    df = df.merge(df_month_lag, how = 'left', left_on='dateid_serial', right_on='month_lag')
    df = df.merge(df_bi_month_lag, how = 'left', left_on='dateid_serial', right_on='bi_month_lag')
    df = df.merge(df_qrt_year_lag, how = 'left', left_on='dateid_serial', right_on='qrt_year_lag')
    df = df.merge(df_half_year_lag, how = 'left', left_on='dateid_serial', right_on='half_year_lag')
    df = df.merge(df_yearlag, how = 'left', left_on='dateid_serial', right_on='year_lag')
    
    del df['month_lag']
    del df['bi_month_lag']
    del df['qrt_year_lag']
    del df['half_year_lag']
    del df['year_lag']
    
    #Return df
    return df


def get_df_with_lags_per_area(coordinate):    
    engine = get_connection()

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
    
    # Remove dateid_serial from lags
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
    df = df.merge(df_month_lag, how = 'left', on=['dateid_serial', 'traffic_geo'])
    df = df.merge(df_bi_month_lag, how = 'left', on=['dateid_serial', 'traffic_geo'])
    df = df.merge(df_qrt_year_lag, how = 'left', on=['dateid_serial', 'traffic_geo'])
    df = df.merge(df_half_year_lag, how = 'left', on=['dateid_serial', 'traffic_geo'])
    df = df.merge(df_yearlag, how = 'left', on=['dateid_serial', 'traffic_geo'])

    #Return df
    return df

