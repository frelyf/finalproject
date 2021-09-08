# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 10:23:27 2021

@author: Abder
"""
import json
import pprint
import pandas as pd

path_2020 = r"C:\Users\Fredrik Lyford\Documents\GitHub\finalproject\ETL\Weather_data\all_stations2020.json"
path_2021 = r"C:\Users\Fredrik Lyford\Documents\GitHub\finalproject\ETL\Weather_data\all_stations2021includingJune.json"

reference_time = []
sourceid = []
observations_crude = []
data_list = []


with open(path_2020) as year_2020:
    translated_json_2020 = json.load(year_2020)
    for i in range(0,13269):
        observations_crude.append(translated_json_2020['data'][i]['observations'])
        reference_time.append(translated_json_2020['data'][i]['referenceTime'])
        sourceid.append(translated_json_2020['data'][i]['sourceId'])
    
    
    
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

