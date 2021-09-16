# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 11:50:36 2021

@author: Inger Lise
"""
import psycopg2
import requests
import pandas as pd
import numpy as np
import datetime
from datetime import date


def get_connection():
    connection = psycopg2.connect(
                host="trafikkluft.postgres.database.azure.com",                
                port="5432",
                user="postgres@trafikkluft",                
                password="Awesome1337",                
                database="postgres",            
            )
    return connection

def get_weather_data(referencetime):
    result_list = []
    connection = get_connection()
    with connection.cursor() as weather_cursor:
        weather_cursor.execute('select source_id, sk_weather from dim_weather')
        station_info = weather_cursor.fetchall()
        for station_tuple in station_info:
            try:
                client_id = '7a4475dc-6ff0-47b7-8c97-b2df66a4b44b'
                endpoint = 'https://frost.met.no/observations/v0.jsonld'
                parameters = {
                    'sources': station_tuple[0],
                    'elements': 'mean(air_temperature P1D), mean(wind_speed P1D), sum(precipitation_amount P1D)',
                    'referencetime': referencetime,
                    }
                results = requests.get(endpoint, parameters, auth=(client_id,''))
                weather_json = results.json()
                for data in weather_json['data']:
                    sk_date = (data['referenceTime'][0:10]).replace('-','')
                    for obs in data['observations']:
                        out_dict = {}
                        out_dict['sk_date'] = sk_date
                        out_dict['source_id'] = station_tuple[0]
                        out_dict['sk_weather'] = station_tuple[1]
                        out_dict['component'] = obs['elementId']
                        out_dict['value'] = obs['value']
                        result_list.append(out_dict)
            except:
                print(station_tuple[0])
    weather_df = pd.DataFrame(result_list)
    weather_df = weather_df.pivot_table(index = ['source_id', 'sk_date','sk_weather'], columns = ['component'], values = ['value'], fill_value = np.nan).reset_index()
    weather_df.columns = [c[0] + "_" + c[1] for c in weather_df.columns]
    return weather_df       

def update_facts_table_weather():
    to_date = (date.today()-datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    connection = get_connection()
    
    with connection.cursor() as weather_cursor:
        weather_cursor.execute('select max(sk_date) from facts_weather')
        last_date_tuple = weather_cursor.fetchone()
        for last_date in last_date_tuple:
            from_date = pd.to_datetime(str(last_date),format='%Y%m%d')
        from_date_plus = (datetime.datetime.strptime(str(from_date), '%Y-%m-%d %H:%M:%S')+datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    
    df_update = get_weather_data(f'{from_date_plus}/{to_date}')
    
    with connection.cursor() as weather_cursor:
        for index, row in df_update.iterrows():
            weather_cursor.execute(
                """insert into facts_weather(sk_date, sk_weather, "mean(air_temperature P1D)", 
                "mean(wind_speed P1D)", "sum(precipitation_amount P1D)") 
                VALUES(%s, %s, %s, %s, %s)""",
                (row['sk_date_'], row['sk_weather_'], row['value_mean(air_temperature P1D)'], row['value_mean(wind_speed P1D)'], row['value_sum(precipitation_amount P1D)']))  
        connection.commit()

update_facts_table_weather()

