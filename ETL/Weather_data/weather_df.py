# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 10:23:27 2021

@author: Abder
"""
import json
import pandas as pd
import psycopg2


#-------------------------------------------locationname and coordinates----------------------------------------------------------
path = r"C:\Users\Abder\Desktop\json graduation\sources.json"

sources_list = []
names_list = []
coordinates_list_lat = []
coordinates_list_long = []

with open(path) as crude_sources:
    data = json.load(crude_sources)
    
    for i in range(0,40):
        sources_list.append(data['data'][i]['id'])
        names_list.append(data['data'][i]['name'])
        coordinates_list_lat.append(data['data'][i]['geometry']['coordinates'][0])
        coordinates_list_long.append(data['data'][i]['geometry']['coordinates'][1])
        
dict_list = zip(sources_list, names_list, coordinates_list_lat, coordinates_list_long)

location_df = pd.DataFrame(dict_list)
location_df.columns = ['source_id', 'station_name', 'long', 'lat']
#-------------------------------------------locationname and coordinates----------------------------------------------------------


station_location_df = r"C:\Users\Abder\Desktop\json graduation\sources_clean.csv"

savepath = r"C:\Users\Abder\Desktop\json graduation\df_wrangled.csv"
savepath2 = r"C:\Users\Abder\Desktop\json graduation\df_wrangled2021.csv"

path_2017 = r"C:\Users\Abder\Desktop\json graduation\all_stations2017.json"
path_2018 = r"C:\Users\Abder\Desktop\json graduation\all_stations2018.json"
path_2019 = r"C:\Users\Abder\Desktop\json graduation\all_stations2019.json"
path_2020 = r"C:\Users\Abder\Desktop\Graduation project\ETL\Weather_data\all_stations2020.json"
path_2021 = r"C:\Users\Abder\Desktop\Graduation project\ETL\Weather_data\all_stations2021includingJune.json"


def json_to_df (path, savepoint):
    
    reference_time = []
    sourceid = []
    observations_crude = []
    data_list = []

    with open(path) as filename:
        translated_json = json.load(filename)
        for i in range(0, len(translated_json['data'])):
            observations_crude.append(translated_json['data'][i]['observations'])
            reference_time.append(translated_json['data'][i]['referenceTime'])
            sourceid.append(translated_json['data'][i]['sourceId'])

    for observation_set, reference_time, sourceid in zip(observations_crude, reference_time, sourceid):
        for observation in observation_set:
            data_dict = {}
            data_dict['source_id'] = sourceid
            data_dict['date'] = reference_time
            data_dict['element_id'] = observation['elementId']
            data_dict['value'] = observation['value']
            data_list.append(data_dict)

    df = pd.DataFrame(data_list)
    df_pivot = df.pivot_table('value', ['date', 'source_id'], 'element_id').reset_index()
    df_pivot['date'] = pd.to_datetime(df_pivot['date'], errors = 'coerce')
    df_pivot['date'] = df_pivot['date'].dt.strftime('%Y%m%d').astype(int)
    
    df_pivot.to_csv(path_or_buf=savepoint)

#json_to_df(path_2020, year_2020, savepath)
#json_to_df(path_2021, savepath)

#-------------------------------------------2020----------------------------------------------------------
reference_time = []
sourceid = []
observations_crude = []
data_list = []



with open(path_2020) as year_2020:
    translated_json_2020 = json.load(year_2020)
    for i in range(0, len(translated_json_2020['data'])):
        observations_crude.append(translated_json_2020['data'][i]['observations'])
        reference_time.append(translated_json_2020['data'][i]['referenceTime'])
        unfinished = (translated_json_2020['data'][i]['sourceId'])
        partly_finished = unfinished.split(':')
        sourceid.append(partly_finished[0])
        

for observation_set, reference_time, sourceid in zip(observations_crude, reference_time, sourceid):
    for observation in observation_set:
        data_dict = {}
        data_dict['source_id'] = sourceid
        data_dict['date'] = reference_time
        data_dict['element_id'] = observation['elementId']
        data_dict['value'] = observation['value']
        data_list.append(data_dict)

df = pd.DataFrame(data_list)
df_pivot = df.pivot_table('value', ['date', 'source_id'], 'element_id').reset_index()
df_pivot['date'] = pd.to_datetime(df_pivot['date'], errors = 'coerce')
df_pivot['date'] = df_pivot['date'].dt.strftime('%Y%m%d').astype(int)

df_pivot_full = pd.merge(left=df_pivot, right=location_df, how='left', on=['source_id'])
#-------------------------------------------2020----------------------------------------------------------

#-------------------------------------------2021----------------------------------------------------------
reference_time2 = []
sourceid2 = []
observations_crude2 = []
data_list2 = []

with open(path_2021) as year_2021:
    translated_json_2021 = json.load(year_2021)
    for i in range(0, len(translated_json_2021['data'])):
        observations_crude2.append(translated_json_2021['data'][i]['observations'])
        reference_time2.append(translated_json_2021['data'][i]['referenceTime'])
        unfinished = (translated_json_2021['data'][i]['sourceId'])
        partly_finished = unfinished.split(':')
        sourceid2.append(partly_finished[0])
        
for observation_set, reference_time, sourceid in zip(observations_crude2, reference_time2, sourceid2):
    for observation in observation_set:
        data_dict = {}
        data_dict['source_id'] = sourceid
        data_dict['date'] = reference_time
        data_dict['element_id'] = observation['elementId']
        data_dict['value'] = observation['value']
        data_list2.append(data_dict)

df2 = pd.DataFrame(data_list2)
df_pivot2 = df2.pivot_table('value', ['date', 'source_id'], 'element_id').reset_index()
df_pivot2['date'] = pd.to_datetime(df_pivot2['date'], errors = 'coerce')
df_pivot2['date'] = df_pivot2['date'].dt.strftime('%Y%m%d').astype(int)

df_pivot_full2 = pd.merge(left=df_pivot2, right=location_df, how='left', on=['source_id'])
#-------------------------------------------2021----------------------------------------------------------


#-------------------------------------------2019----------------------------------------------------------
reference_time3 = []
sourceid3 = []
observations_crude3 = []
data_list3 = []

with open(path_2019) as year_2019:
    translated_json_2019 = json.load(year_2019)
    for i in range(0, len(translated_json_2019['data'])):
        observations_crude3.append(translated_json_2019['data'][i]['observations'])
        reference_time3.append(translated_json_2019['data'][i]['referenceTime'])
        unfinished = (translated_json_2019['data'][i]['sourceId'])
        partly_finished = unfinished.split(':')
        sourceid3.append(partly_finished[0])
        
for observation_set, reference_time, sourceid in zip(observations_crude3, reference_time3, sourceid3):
    for observation in observation_set:
        data_dict = {}
        data_dict['source_id'] = sourceid
        data_dict['date'] = reference_time
        data_dict['element_id'] = observation['elementId']
        data_dict['value'] = observation['value']
        data_list3.append(data_dict)

df3 = pd.DataFrame(data_list3)
df_pivot3 = df3.pivot_table('value', ['date', 'source_id'], 'element_id').reset_index()
df_pivot3['date'] = pd.to_datetime(df_pivot3['date'], errors = 'coerce')
df_pivot3['date'] = df_pivot3['date'].dt.strftime('%Y%m%d').astype(int)

df_pivot_full3 = pd.merge(left=df_pivot3, right=location_df, how='left', on=['source_id'])
#-------------------------------------------2019----------------------------------------------------------

#-------------------------------------------2018----------------------------------------------------------
reference_time4 = []
sourceid4 = []
observations_crude4 = []
data_list4 = []

with open(path_2018) as year_2018:
    translated_json_2018 = json.load(year_2018)
    for i in range(0, 13180):
        observations_crude4.append(translated_json_2018['data'][i]['observations'])
        reference_time4.append(translated_json_2018['data'][i]['referenceTime'])
        unfinished = (translated_json_2018['data'][i]['sourceId'])
        partly_finished = unfinished.split(':')
        sourceid4.append(partly_finished[0])
        
for observation_set, reference_time, sourceid in zip(observations_crude4, reference_time4, sourceid4):
    for observation in observation_set:
        data_dict = {}
        data_dict['source_id'] = sourceid
        data_dict['date'] = reference_time
        data_dict['element_id'] = observation['elementId']
        data_dict['value'] = observation['value']
        data_list4.append(data_dict)

df4 = pd.DataFrame(data_list4)
df_pivot4 = df4.pivot_table('value', ['date', 'source_id'], 'element_id').reset_index()
df_pivot4['date'] = pd.to_datetime(df_pivot4['date'], errors = 'coerce')
df_pivot4['date'] = df_pivot4['date'].dt.strftime('%Y%m%d').astype(int)

df_pivot_full4 = pd.merge(left=df_pivot4, right=location_df, how='left', on=['source_id'])
#-------------------------------------------2018----------------------------------------------------------


#-------------------------------------------2017----------------------------------------------------------
reference_time5 = []
sourceid5 = []
observations_crude5 = []
data_list5 = []

with open(path_2017) as year_2017:
    translated_json_2017 = json.load(year_2017)
    for i in range(0, 12837):
        observations_crude5.append(translated_json_2017['data'][i]['observations'])
        reference_time5.append(translated_json_2017['data'][i]['referenceTime'])
        unfinished = (translated_json_2017['data'][i]['sourceId'])
        partly_finished = unfinished.split(':')
        sourceid5.append(partly_finished[0])
        
for observation_set, reference_time, sourceid in zip(observations_crude5, reference_time5, sourceid5):
    for observation in observation_set:
        data_dict = {}
        data_dict['source_id'] = sourceid
        data_dict['date'] = reference_time
        data_dict['element_id'] = observation['elementId']
        data_dict['value'] = observation['value']
        data_list5.append(data_dict)

df5 = pd.DataFrame(data_list5)
df_pivot5 = df5.pivot_table('value', ['date', 'source_id'], 'element_id').reset_index()
df_pivot5['date'] = pd.to_datetime(df_pivot5['date'], errors = 'coerce')
df_pivot5['date'] = df_pivot5['date'].dt.strftime('%Y%m%d').astype(int)

df_pivot_full5 = pd.merge(left=df_pivot5, right=location_df, how='left', on=['source_id'])
#-------------------------------------------2017----------------------------------------------------------


#-------------------------------------------an attempt was made----------------------------------------------------------
def opensesame():
        connection = psycopg2.connect(
                user="postgres@trafikkluft",
                password="Awesome1337",
                host="trafikkluft.postgres.database.azure.com",
                port="5432",
                database="postgres",
        )
        return connection


df_full_range = pd.concat([df_pivot, df_pivot2, df_pivot3, df_pivot4, df_pivot5],ignore_index=True)

location_df = location_df.reset_index()

def surrogate_key(sourceid):
    sk = location_df.loc[location_df['source_id'] == sourceid, 'index']
    return sk.iloc[0]

surrogate_key('SN18269')

df_full_range['source_id'] = df_full_range['source_id'].apply(surrogate_key)

def database_writer(data):
    with opensesame() as connection:
        cursor = connection.cursor()
        cursor.executemany(f"""INSERT INTO dim_weather(weather_station_measurement_sk, weather_station_id, location_name, long, lat) VALUES(%s, %s, %s, %s, %s)""",data)
        return
#-------------------------------------------an attempt was made----------------------------------------------------------

#-------------------------------------------this will create a whole new table based on given dataframe-------------------------------------
from sqlalchemy import create_engine
engine = create_engine('postgresql://postgres@trafikkluft:Awesome1337@trafikkluft.postgres.database.azure.com:5432/postgres')
df_full_range.to_sql('facts_weather', engine)

database_writer(location_df)
#-------------------------------------------this will create a whole new table based on given dataframe-------------------------------------

