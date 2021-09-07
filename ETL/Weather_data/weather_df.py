# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 10:23:27 2021

@author: Abder
"""
import json
import pprint
import pandas as pd

path_2020 = r"C:\Users\Abder\Desktop\json graduation\all_stations2020.json"
path_2021 = r"C:\Users\Abder\Desktop\json graduation\all_stations2021includingJune.json"

reference_time = []
sourceid = []
observations_crude = []
observations_clean_elementID_temper = []
observations_clean_values_temper = []
observations_clean_elementID_rain = []
observations_clean_values_rain = []


with open(path_2020) as year_2020:
    translated_json_2020 = json.load(year_2020)
    for i in range(0,13269):
        observations_crude.append(translated_json_2020['data'][i]['observations'])
        reference_time.append(translated_json_2020['data'][i]['referenceTime'])
        sourceid.append(translated_json_2020['data'][i]['sourceId'])
    for dictionaries in observations_crude:
        for dictionary in dictionaries:
            if ((dictionary['elementId'] == 'mean(air_temperature P1D)') and (dictionary['timeOffset'] == 'PT0H')):
                observations_clean_elementID_temper.append(dictionary['elementId'])
                observations_clean_values_temper.append(dictionary['value'])
            elif ((dictionary['elementId'] == 'sum(precipitation_amount P1D)') and (dictionary['timeOffset'] == 'PT6H')):
                observations_clean_elementID_rain.append(dictionary['elementId'])
                observations_clean_values_rain.append(dictionary['value'])
            
        



weather_df = pd.DataFrame(reference_time)
#renaming columns?
weather_df['sourceID'] = pd.DataFrame(sourceid)
# weather_df['observationtype'] = pd.DataFrame(elementID)
# weather_df['value'] = pd.DataFrame(value)
weather_df.columns = ['datetime', 'sourceID', 'observationtype', 'value']


weather_df_pivot = weather_df.pivot_table('value', ['datetime', 'sourceID'], 'observationtype')

pprint.pprint(translated_json_2020['data'][0]['observations'])

