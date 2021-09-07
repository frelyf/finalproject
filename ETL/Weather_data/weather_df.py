# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 10:23:27 2021

@author: Abder
"""
import json
import pprint
import pandas as pd

path_2020 = r"C:\Users\Abder\Desktop\json graduation\all_stations2020.json-1631002853467.json"
path_2021 = r"C:\Users\Abder\Desktop\json graduation\all_stations2021includingJune.json-1631002941125.json"

reference_time = []
sourceid = []
observations = []
height_above_ground = []
with open(path_2020) as year_2020:
    translated_json_2020 = json.load(year_2020)
    for i in range(0,13267):
        reference_time.append(translated_json_2020['data'][i]['referenceTime'])
        sourceid.append(translated_json_2020['data'][i]['sourceId'])
        observations.append(translated_json_2020['data'][i]['observations'][0])
        
    
elementID = []
value = []
for i in observations:
    elementID.append(i['elementId'])
    value.append(i['value'])
    print(i['elementId'], i['value'])


weather_df = pd.DataFrame(reference_time)
#renaming columns?
weather_df['sourceID'] = pd.DataFrame(sourceid)
weather_df['observationtype'] = pd.DataFrame(elementID)
weather_df['value'] = pd.DataFrame(value)
weather_df.columns = ['datetime', 'sourceID', 'observationtype', 'value']

pprint.pprint(translated_json_2020['data'][0]['observations'])

