
import requests
import csv
import pandas as pd

def nilf (search_terms):
    url = f'https://api.nilu.no/{search_terms}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return 'Nah'

#'lookup/areas'
#'lookup/stations'
#'aq/historical/2019-01-01/2020-01-01/Hjortnes' (from date, to date, station)
#'/obs/historical/{fromtime}/{totime}/{latitude}/{longitude}/{radius}'
    
oslo_stations_call = nilf('lookup/stations')

oslo_stations = []
for i in test1:
    if i['municipality'] == 'Oslo':
        oslo_stations.append(i)

oslo_stations_info = []
for i in oslo_stations:
    if i['lastMeasurment'] > '2020-01-01':
        oslo_stations_info_temp = {}
        oslo_stations_info_temp['id'] = i['id']
        oslo_stations_info_temp['area'] = i['area']
        oslo_stations_info_temp['station'] = i['station']
        oslo_stations_info_temp['lastMeasurement'] =  i['lastMeasurment']
        oslo_stations_info_temp['components'] = i['components']
        oslo_stations_info_temp['latitude'] = i['latitude']
        oslo_stations_info_temp['longitude'] = i['longitude']
        oslo_stations_info.append(oslo_stations_info_temp)
print(oslo_stations_info)

df_oslo_stations = pd.DataFrame(oslo_stations_info)
df_oslo_stations.head(5)

oslo_station_names = []
for i in oslo_stations:
    oslo_station_names.append(i['station'])
print(oslo_station_names)


def write_to_csv (search_terms):
    #data = nilf(search_terms)
    data = search_terms
    with open('station_7_obs.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(data[0].keys())
        for i in data:
            writer.writerow(i.values())

#write_to_csv(oslo_stations_info)
#write_to_csv('/obs/historical/2021-08-01/2021-08-07/Alnabru')


def obs_per_day (from_date, to_date, station_name):
    return nilf(f'/stats/day/{from_date}/{to_date}/{station_name}')

obs_test = obs_per_day('2021-08-01','2021-08-07', 'Alnabru')
print(obs_test)

def get_all_obs (from_date, to_date):
    all_obs_list = []
    for station_name in oslo_station_names:
        all_obs_list.append(nilf(f'/stats/day/{from_date}/{to_date}/{station_name}'))
    return all_obs_list

all_obs_test = get_all_obs('2021-08-01','2021-08-07')
print(all_obs_test)


